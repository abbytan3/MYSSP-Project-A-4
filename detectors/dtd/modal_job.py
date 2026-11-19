"""DTD Modal job (T14). Follows detectors/adcd/modal_job.py. Run from this laptop:

  0. Environment check (CPU only, no GPU, no pages; builds the image, loads the weights):
       modal run detectors/dtd/modal_job.py --stage env
  1. DTD (one T4 GPU, one size at a time):
       modal run detectors/dtd/modal_job.py --stage dtd --size shrink512 --pages page_0001,page_0003,page_0009
       modal run detectors/dtd/modal_job.py --stage dtd --size full      --pages page_0001,page_0003,page_0009
       modal run detectors/dtd/modal_job.py --stage dtd --size shrink512      (all pages)
       modal run detectors/dtd/modal_job.py --stage dtd --size full           (all pages)
  2. Check a finished size (CPU):
       modal run detectors/dtd/modal_job.py --stage check_dtd --size shrink512

Reads pages from the private Volume `a4-data` (pages/<page_id>.<ext>) and writes
  out/dtd/<size>/<page_id>.npz        compact output (shared/output_format.py)
  out/dtd/<size>/run_info.json, run_log.csv
Weights must already be on the Volume under weights/dtd/ (see detectors/dtd/README.md).

Nothing about a page is ever printed: only anonymous IDs and numbers.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------- settings
# Name of the private Volume that holds pages, weights and outputs.
VOLUME_NAME = "a4-data"
# Where the Volume appears inside every container.
DATA = "/data"
# Weights on the Volume: the released DocTamper weights and the two ImageNet backbone files the
# authors' model code loads by name when it is built (they are overwritten by the DocTamper weights).
WEIGHTS_DIR = DATA + "/weights/dtd"
CKPT_NAME = "dtd_doctamper.pth"
BACKBONE_NAMES = ("vph_imagenet.pt", "swin_imagenet.pt")
CKPT = WEIGHTS_DIR + "/" + CKPT_NAME
# Pages and outputs on the Volume.
PAGES_DIR = DATA + "/pages"
OUT_ROOT = DATA + "/out"
# The authors' code, pinned to the commit used for this project.
REPO_URL = "https://github.com/qcf-568/DocTamper"
REPO_COMMIT = "2d53389932f72941575adbb595377955d366881c"
REPO_DIR = "/opt/dtd_repo"
REPO_MODELS = REPO_DIR + "/models"
# DTD's input size (authors' evaluation and inference code use 512x512), and the seed (docs/PLAN.md).
PIECE = 512
SEED = 42
# JPEG quality used when a page has to be re-saved as JPEG (docs/PLAN.md section 3).
RESAVE_QUALITY = 100
# Largest quantisation-table value DTD can take (its table embedding has 64 entries).
MAX_TABLE_VALUE = 63
# Tiles per forward pass on the GPU.
TILE_BATCH = 4
# Save the Volume to disk after this many pages.
COMMIT_EVERY = 25

# Folder of this file, and the shared folder (both are copied into the image).
HERE = Path(__file__).resolve().parent
# Inside the Modal container this file sits at /root, which has no grandparent folder; the
# paths are only used on the laptop when the image is built, so any value is fine there.
SHARED = HERE.parents[1] / "shared" if len(HERE.parents) > 1 else HERE

app = modal.App("a4-dtd")
volume = modal.Volume.from_name(VOLUME_NAME)

# ---------------------------------------------------------------- image
# One image for every stage (its own Python and packages, never shared with another detector).
# Versions follow the authors' models/requirements.txt where they install on Python 3.10
# (see detectors/dtd/requirements.txt); the versions really installed are logged in run_info.json.
dtd_image = (
    modal.Image.debian_slim(python_version="3.10")
    # System JPEG headers and a compiler, needed to build jpegio.
    .apt_install("git", "build-essential", "libjpeg-dev", "libgl1", "libglib2.0-0")
    # torch 1.11.0 for CUDA 11.3, from PyTorch's own download site.
    .pip_install(
        "torch==1.11.0+cu113", "torchvision==0.12.0+cu113", "numpy==1.23.5",
        extra_index_url="https://download.pytorch.org/whl/cu113",
    )
    # The packages the authors' model code imports.
    .pip_install(
        "numpy==1.23.5", "opencv-python-headless==4.5.5.62", "Pillow==9.5.0", "albumentations==1.3.0",
        "scipy==1.9.1", "timm==0.4.12", "efficientnet_pytorch==0.7.1", "six==1.15.0", "tqdm==4.64.0",
        "lmdb", "pretrainedmodels", "munch",
    )
    # segmentation_models_pytorch 0.2.1 pins efficientnet-pytorch 0.6.3, which clashes with the 0.7.1 the
    # authors document, so it is installed without its own dependencies (they are installed above).
    .pip_install("segmentation_models_pytorch==0.2.1", extra_options="--no-deps")
    # jpegio has an old setup.py: it needs cython and no build isolation.
    .pip_install("setuptools<70", "wheel", "cython", "numpy==1.23.5")
    .run_commands("pip install --no-build-isolation jpegio==0.2.8")
    # Full clone, then the exact commit.
    .run_commands(
        f"git clone -q {REPO_URL} {REPO_DIR}",
        f"git -C {REPO_DIR} checkout -q {REPO_COMMIT}",
    )
    # Our own helpers, copied in last so editing them does not rebuild the image.
    .add_local_file(str(HERE / "tiling.py"), "/app/tiling.py")
    .add_local_file(str(SHARED / "output_format.py"), "/app/output_format.py")
)


def _list_pages(limit=0, names=""):
    # All page files on the Volume whose name is an anonymous ID, sorted.
    sys.path.insert(0, "/app")
    from output_format import PAGE_ID_PATTERN
    pages = sorted(p for p in Path(PAGES_DIR).iterdir() if PAGE_ID_PATTERN.match(p.stem))
    # Named pages (e.g. "page_0001,page_0003,page_0009"): exactly these, in the order given.
    if names:
        wanted = [n.strip() for n in names.split(",") if n.strip()]
        by_id = {p.stem: p for p in pages}
        unknown = [n for n in wanted if n not in by_id]
        # Say which IDs are wrong (anonymous IDs only), never guess.
        if unknown:
            raise ValueError("pages not found on the Volume: " + ", ".join(unknown))
        return [by_id[n] for n in wanted]
    # Test mode: only the first `limit` pages.
    return pages[:limit] if limit > 0 else pages


def _version(name):
    # Installed version of a package, or "unknown".
    import importlib.metadata as md
    try:
        return md.version(name)
    except Exception:  # noqa: BLE001 - a missing name must not stop the run
        return "unknown"


def package_versions():
    # The versions really installed in the container (logged in run_info.json).
    names = ["torch", "torchvision", "numpy", "opencv-python-headless", "Pillow", "albumentations",
             "scipy", "timm", "segmentation_models_pytorch", "efficientnet_pytorch", "jpegio"]
    return {name: _version(name) for name in names}


def build_run_info(of, size, gpu, versions, n_missing, n_unexpected, seconds, n_ok, n_bad, limit, started, counts):
    """Return the content of run_info.json for one size.

    The first seven keys are the ones shared/score.py requires, spelled exactly
    as in `of.REQUIRED_RUN_INFO_FIELDS`; tile_size is None (null) for shrink512.
    """
    info = {
        # --- required by shared/score.py ---
        "detector": "dtd",
        "weights": [CKPT_NAME] + list(BACKBONE_NAMES),
        "size": size,
        "tile_size": PIECE if size == "full" else None,
        "versions": versions,
        # Median seconds per page over every ok page in run_log.csv (test runs included).
        "seconds_per_page": round(statistics.median(seconds), 3) if seconds else None,
        "seed": SEED,
        # --- extra detail ---
        "shrink_longest_side": PIECE if size == "shrink512" else None,
        "threshold_for_flagging": of.FLAG_THRESHOLD,
        "value_scale": "softmax probability of tampered, float32 (not rounded to 8 bit)",
        "gpu": gpu,
        "repo_url": REPO_URL,
        "repo_commit": REPO_COMMIT,
        "model_load": {"missing_keys": n_missing, "unexpected_keys": n_unexpected},
        "pages_ok": n_ok,
        "pages_not_ok": n_bad,
        "mean_seconds_per_page": round(statistics.mean(seconds), 3) if seconds else None,
        "limit_used_in_last_call": limit,
        "last_call_started_utc": started,
        # How the pages of the last call were read (docs/PLAN.md: "re-save PNG pages at quality 100, and log it").
        "last_call_pages_resaved_as_jpeg_q100": counts["resaved"],
        "last_call_pages_jpeg_used_as_is": counts["as_is"],
        "last_call_jpeg_pages_resaved_because_table_value_above_63": counts["coarse_table"],
        "notes": "Input as in the authors' evaluation: greyscale page turned into RGB, luma DCT numbers clipped to 0..20, "
                 "8x8 quantisation table. PNG pages are re-saved as JPEG quality 100 just before reading (after padding for "
                 "full); JPEG pages keep their own DCT numbers and table unless the table has a value above 63 (then they are "
                 "re-saved too, PLAN CHANGE); shrink512 pages are always re-saved. "
                 "full = 512x512 tiles from (0,0), no overlap, white padding right/bottom, joined back, padding cropped. "
                 "shrink512 = longest side 512 (cv2 INTER_AREA), padded white to 512x512, map resized back with bilinear interpolation.",
    }
    # Fail now, not in scoring, if a required name is missing.
    missing = [k for k in of.REQUIRED_RUN_INFO_FIELDS if k not in info]
    if missing:
        raise ValueError("run_info is missing required fields: " + ", ".join(missing))
    return info


# ---------------------------------------------------------------- model and page helpers
def load_model(device):
    """Build DTD, load the DocTamper weights and return (model, n_missing, n_unexpected).

    The authors' DTD class loads vph_imagenet.pt and swin_imagenet.pt from the current folder when it is
    built, so we work from a temporary folder holding links to them. Stops if the weights do not match.
    """
    import os
    import tempfile
    import torch

    work = Path(tempfile.mkdtemp(prefix="dtd_cwd_"))
    for name in BACKBONE_NAMES:
        (work / name).symlink_to(Path(WEIGHTS_DIR) / name)
    os.chdir(work)
    sys.path.insert(0, REPO_MODELS)
    import dtd
    import fph
    import swins
    # The two backbone files are whole saved objects; if they were saved from a script, their classes are looked
    # up in __main__, so make the authors' classes visible there too.
    import __main__
    for module in (dtd, swins, fph):
        for key, value in vars(module).items():
            if not key.startswith("__") and not hasattr(__main__, key):
                setattr(__main__, key, value)

    model = dtd.seg_dtd("", 2)
    # The released file was saved from a DataParallel model, so its names start with "module.".
    state = torch.load(CKPT, map_location="cpu")["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # A wrong weight file must stop the run, not give silent nonsense.
    if missing or unexpected:
        raise RuntimeError("weights do not match the model (missing %d, unexpected %d)" % (len(missing), len(unexpected)))
    model.to(device).eval()
    return model, len(missing), len(unexpected)


def predict_tiles(model, tiles, device, batch=TILE_BATCH):
    """Run DTD on a list of (grey 512x512 uint8, DCT numbers 512x512, quantisation table 8x8).

    Returns one 512x512 float32 array per tile: the softmax probability of "tampered".
    """
    import numpy as np
    import torch

    # Same normalisation as the authors' loader (their mean list has 0.455 as its middle value).
    mean = torch.tensor([0.485, 0.455, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    maps = []
    with torch.no_grad():
        for start in range(0, len(tiles), batch):
            chunk = tiles[start:start + batch]
            # Grey page as 3 identical channels, scaled to 0..1, then normalised.
            grey = torch.from_numpy(np.stack([t[0] for t in chunk])).float().div(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
            data = ((grey - mean) / std).to(device)
            # DCT numbers: absolute value clipped to 0..20 (the embedding has 21 entries).
            dct = torch.from_numpy(np.stack([np.clip(np.abs(t[1]), 0, 20) for t in chunk])).long().to(device)
            qtb = torch.from_numpy(np.stack([np.asarray(t[2]).reshape(8, 8) for t in chunk])).long().unsqueeze(1).to(device)
            logits = model(data, dct, qtb)
            # Probability of class 1 ("tampered"), back to float32 on the CPU.
            prob = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            maps.extend(prob[i].astype(np.float32) for i in range(prob.shape[0]))
    return maps


def read_page(path):
    """Return (grey uint8 page, None) for a PNG-like page, or (grey page, (dct, table)) for a JPEG page.

    The grey page uses PIL's own greyscale conversion, as the authors' loader does. For a JPEG file the
    luma DCT numbers and the 8x8 quantisation table are read from the file itself (cropped to whole blocks).
    """
    import jpegio
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        is_jpeg = image.format == "JPEG"
        grey = np.array(image.convert("L"))
    if not is_jpeg:
        return grey, None
    jpg = jpegio.read(str(path))
    height8, width8 = -(-grey.shape[0] // 8) * 8, -(-grey.shape[1] // 8) * 8
    dct = jpg.coef_arrays[0]
    if dct.shape[0] < height8 or dct.shape[1] < width8:
        raise RuntimeError("DCT array smaller than the page")
    return grey, (dct[:height8, :width8].copy(), np.asarray(jpg.quant_tables[0], dtype=np.int64).copy())


def resave_as_jpeg(grey, work):
    """Save a grey page as JPEG at quality 100 and read back (DCT numbers, quantisation table).

    The grey array must already be a whole number of 8x8 blocks (it is, after padding to 512).
    """
    import jpegio
    import numpy as np
    from PIL import Image

    tmp = work / "resave.jpg"
    Image.fromarray(grey).save(str(tmp), "JPEG", quality=RESAVE_QUALITY)
    jpg = jpegio.read(str(tmp))
    dct = jpg.coef_arrays[0][: grey.shape[0], : grey.shape[1]].copy()
    return dct, np.asarray(jpg.quant_tables[0], dtype=np.int64).copy()


# ---------------------------------------------------------------- stage 0: environment check (CPU)
@app.function(image=dtd_image, volumes={DATA: volume}, timeout=1800)
def env_check():
    """Check the image, the weight files and the model on the CPU. Uses one random tile, never a page."""
    import numpy as np
    import torch

    sys.path.insert(0, "/app")
    import tiling

    volume.reload()
    res = {"versions": package_versions(), "cuda_available": torch.cuda.is_available(), "repo_commit": REPO_COMMIT}
    # The three weight files must be on the Volume.
    found = {}
    for name in (CKPT_NAME,) + BACKBONE_NAMES:
        path = Path(WEIGHTS_DIR) / name
        found[name] = path.stat().st_size if path.exists() else None
    res["weight_file_sizes"] = found
    if any(v is None for v in found.values()):
        res["passed"] = False
        res["problem"] = "weight file missing on the Volume"
        return res
    # The authors' tool for the DCT numbers must work in this image.
    import jpegio  # noqa: F401
    model, n_missing, n_unexpected = load_model("cpu")
    res["model_load"] = {"missing_keys": n_missing, "unexpected_keys": n_unexpected}
    # One random tile (not a page): output must be 512x512, finite, between 0 and 1.
    rng = np.random.default_rng(SEED)
    tile = (rng.integers(0, 256, (PIECE, PIECE), dtype=np.uint8), rng.integers(-30, 30, (PIECE, PIECE)), np.ones((8, 8), dtype=np.int64))
    out = predict_tiles(model, [tile], "cpu")[0]
    res["random_tile_output"] = {"shape": list(out.shape), "min": float(out.min()), "max": float(out.max()), "finite": bool(np.isfinite(out).all())}
    res["tiling_piece"] = tiling.PIECE
    res["passed"] = (n_missing == 0 and n_unexpected == 0 and out.shape == (PIECE, PIECE)
                     and bool(np.isfinite(out).all()) and 0.0 <= float(out.min()) and float(out.max()) <= 1.0)
    return res


# ---------------------------------------------------------------- stage 0b: JPEG table count (CPU)
@app.function(image=dtd_image, volumes={DATA: volume}, timeout=3600)
def qtable_count():
    """Count pages by file format and look at the quantisation tables of the JPEG pages. Numbers only.

    DTD's quantisation-table embedding has 64 entries, so a table value above 63 would make that page fail.
    Also checks that each JPEG page's DCT array covers the page (whole 8x8 blocks). Anonymous IDs are listed
    only for pages with a problem.
    """
    import collections
    import jpegio
    from PIL import Image

    volume.reload()
    pages = _list_pages()
    formats = collections.Counter()
    n_jpeg, over_63, too_small, unreadable = 0, [], [], []
    table_max = []
    for page in pages:
        try:
            with Image.open(page) as image:
                formats[image.format] += 1
                is_jpeg = image.format == "JPEG"
                width, height = image.size
            if not is_jpeg:
                continue
            n_jpeg += 1
            jpg = jpegio.read(str(page))
            biggest = int(jpg.quant_tables[0].max())
            table_max.append(biggest)
            if biggest > 63:
                over_63.append(page.stem)
            dct = jpg.coef_arrays[0]
            if dct.shape[0] < -(-height // 8) * 8 or dct.shape[1] < -(-width // 8) * 8:
                too_small.append(page.stem)
        except Exception as exc:  # noqa: BLE001 - count it, do not stop
            unreadable.append(page.stem + ":" + type(exc).__name__)
    cap = 50
    return {
        "n_pages": len(pages), "formats": dict(formats), "n_jpeg": n_jpeg,
        "table_max_min": min(table_max) if table_max else None,
        "table_max_max": max(table_max) if table_max else None,
        "n_jpeg_with_table_value_above_63": len(over_63), "ids_above_63": over_63[:cap],
        "n_dct_array_smaller_than_page": len(too_small), "ids_dct_small": too_small[:cap],
        "n_unreadable": len(unreadable), "unreadable": unreadable[:cap],
    }


# ---------------------------------------------------------------- stage 1: DTD
@app.function(image=dtd_image, gpu="T4", volumes={DATA: volume}, timeout=6 * 3600)
def run_dtd(size, limit=0, names=""):
    """Run DTD on every page at one size and save the compact outputs."""
    import csv
    import tempfile
    import time
    import traceback
    from datetime import datetime, timezone

    sys.path.insert(0, "/app")
    import cv2
    import numpy as np
    import output_format as of
    import tiling
    import torch

    # Only the two agreed sizes are allowed.
    if size not in of.VALID_SIZES:
        raise ValueError("size must be one of %s" % (of.VALID_SIZES,))

    # Seeds (evaluation has no randomness, but this is our rule).
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, n_missing, n_unexpected = load_model("cuda")
    print("model ready | missing keys: %d | unexpected keys: %d" % (n_missing, n_unexpected), flush=True)
    work = Path(tempfile.mkdtemp(prefix="dtd_work_"))
    # How many pages were re-saved as JPEG or used as JPEG in this call (logged in run_info.json).
    counts = {"resaved": 0, "as_is": 0, "coarse_table": 0}

    def heatmap_for(page_path):
        # Page-sized heat map (0 to 1, float32) for one page at the chosen size.
        grey, jpeg = read_page(page_path)
        height, width = grey.shape
        if size == "shrink512":
            # Shrink so the longest side is 512, keeping the shape (area averaging when shrinking).
            scale = PIECE / max(height, width)
            new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            small = cv2.resize(grey, (new_w, new_h), interpolation=interp)
            # DTD needs exactly 512x512: pad white on the right and bottom.
            padded = tiling.pad_to(small, PIECE, PIECE, 255)
            # A shrunk page has no JPEG grid of its own, so it is always re-saved at quality 100.
            dct, qtb = resave_as_jpeg(padded, work)
            counts["resaved"] += 1
            prob = predict_tiles(model, [(padded, dct, qtb)], "cuda")[0]
            # Crop the padding, then back to the page size with bilinear interpolation (width, height).
            prob = prob[:new_h, :new_w]
            if prob.shape != (height, width):
                prob = cv2.resize(prob.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            return np.clip(prob.astype(np.float32), 0.0, 1.0)
        # full: pad to a whole number of 512 tiles (white), cut, predict, join.
        pad_h, pad_w = tiling.ceil_div(height, PIECE) * PIECE, tiling.ceil_div(width, PIECE) * PIECE
        grey_p = tiling.pad_to(grey, pad_h, pad_w, 255)
        # DTD's table embedding has 64 entries: a JPEG page whose table has a value above 63 (quality below about
        # 70) would fail, so it is re-saved at quality 100 like a PNG page (PLAN CHANGE, docs/decisions.md).
        if jpeg is not None and int(np.max(jpeg[1])) > MAX_TABLE_VALUE:
            jpeg = None
            counts["coarse_table"] += 1
        if jpeg is None:
            # PNG page: re-save the padded page at quality 100 just before reading.
            dct, qtb = resave_as_jpeg(grey_p, work)
            counts["resaved"] += 1
        else:
            # JPEG page: its own DCT numbers and table; new blocks are those of a white page.
            dct, qtb = jpeg[0], jpeg[1]
            dct = tiling.pad_dct(dct, pad_h, pad_w, qtb)
            counts["as_is"] += 1
        grey_tiles = tiling.cut_padded(grey_p)
        dct_tiles = tiling.cut_padded(dct)
        keys = sorted(grey_tiles)
        maps = predict_tiles(model, [(grey_tiles[k], dct_tiles[k], qtb) for k in keys], "cuda")
        heat = tiling.join_tiles(dict(zip(keys, maps)), height, width)
        return np.clip(heat, 0.0, 1.0)

    out_dir = of.output_dir(OUT_ROOT, "dtd", size)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pages = _list_pages(limit, names)
    done = 0
    for page in pages:
        page_id = page.stem
        # Resumable: skip pages that already have a result.
        if (out_dir / (page_id + ".npz")).exists():
            continue
        t0 = time.time()
        try:
            heat = heatmap_for(page)
            of.save_page_output(out_dir, page_id, heat)
            torch.cuda.synchronize()
            of.append_run_log(out_dir, page_id, size, time.time() - t0, float(heat.max()), "ok")
        except Exception as exc:  # noqa: BLE001 - one bad page must not stop the run
            # Only the error type is logged to the file; the console also gets the code line (never the message,
            # which could hold a path or page numbers).
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            print("page %s failed: %s in %s line %d" % (page_id, type(exc).__name__, frame.name, frame.lineno), flush=True)
            of.append_run_log(out_dir, page_id, size, time.time() - t0, 0.0, "error:" + type(exc).__name__)
        done += 1
        if done % COMMIT_EVERY == 0:
            volume.commit()
            print("%d pages done at %s" % (done, size), flush=True)

    # Seconds per page and counts, from the run log (covers earlier test runs too).
    log_path = out_dir / "run_log.csv"
    seconds, n_ok, n_bad = [], 0, 0
    if log_path.exists():
        with open(log_path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["status"] == "ok":
                    n_ok += 1
                    seconds.append(float(row["seconds"]))
                else:
                    n_bad += 1
    # Build run_info.json (the seven names shared/score.py requires are checked inside) and write it.
    info = build_run_info(
        of, size, torch.cuda.get_device_name(0), package_versions(),
        n_missing, n_unexpected, seconds, n_ok, n_bad, limit, started, counts)
    of.write_run_info(out_dir, info)
    # Read it back with the shared reader, so a missing key fails the run here and not in scoring.
    of.read_run_info(out_dir / "run_info.json")
    volume.commit()
    return {"size": size, "processed_this_call": done, "ok_in_log": n_ok, "not_ok_in_log": n_bad, **counts}


# ---------------------------------------------------------------- stage check
@app.function(image=dtd_image, volumes={DATA: volume}, timeout=3600)
def check_stage(size="shrink512"):
    """Check one finished size on the Volume. Returns numbers and anonymous IDs only.

    A result for every page, no failed page, every .npz opens with top in 0..1 and the page's shape,
    run_info.json has the required keys, and at least 90% of pages have a top score above 0.
    """
    import cv2
    sys.path.insert(0, "/app")
    import output_format as of
    # Make sure this container sees everything the run committed.
    volume.reload()
    pages = _list_pages()
    res = {"kind": "dtd", "size": size, "n_pages": len(pages)}
    cap = 50
    out_dir = of.output_dir(OUT_ROOT, "dtd", size)
    absent, bad, tops = [], [], []
    for page in pages:
        npz = out_dir / (page.stem + ".npz")
        if not npz.exists():
            absent.append(page.stem)
            continue
        try:
            out = of.load_page_output(npz)
            img = cv2.imread(str(page), cv2.IMREAD_COLOR)
            if not (0.0 <= out["top"] <= 1.0) or img is None or out["shape"] != img.shape[:2]:
                bad.append(page.stem)
            else:
                tops.append(out["top"])
        except Exception:  # noqa: BLE001 - an unreadable file counts as bad
            bad.append(page.stem)
    # Pages whose latest run_log row is not ok (a failed page has no .npz, so it shows in `absent` too).
    failed = []
    log_path = out_dir / "run_log.csv"
    if log_path.exists():
        last = {}
        with open(log_path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                last[row["page_id"]] = row["status"]
        failed = sorted(k for k, v in last.items() if v != "ok")
    info_ok, info_missing = False, list(of.REQUIRED_RUN_INFO_FIELDS)
    if (out_dir / "run_info.json").exists():
        with open(out_dir / "run_info.json", encoding="utf-8") as handle:
            info = json.load(handle)
        info_missing = [k for k in of.REQUIRED_RUN_INFO_FIELDS if k not in info]
        info_ok = not info_missing
    n_pos = sum(1 for t in tops if t > 0.0)
    share_pos = n_pos / max(len(pages), 1)
    res.update({
        "results_found": len(pages) - len(absent),
        "absent_ids": absent[:cap], "n_absent": len(absent),
        "failed_ids": failed[:cap], "n_failed": len(failed),
        "bad_ids": bad[:cap], "n_bad": len(bad),
        "run_info_ok": info_ok, "run_info_missing_keys": info_missing,
        "pages_top_above_0": n_pos, "share_top_above_0_pct": round(100 * share_pos, 2),
        "top_min": round(min(tops), 4) if tops else None, "top_max": round(max(tops), 4) if tops else None,
    })
    res["passed"] = (not absent) and (not failed) and (not bad) and info_ok and share_pos >= 0.9
    return res


# ---------------------------------------------------------------- entry point
@app.local_entrypoint()
def main(stage: str, size: str = "shrink512", limit: int = 0, pages: str = ""):
    # stage: "env", "dtd" or "check_dtd". limit: 0 = all pages. pages: comma-separated anonymous IDs (the agreed test pages).
    if stage == "env":
        result = env_check.remote()
        # One machine-readable line (numbers and file names of weights only).
        print("ENV " + json.dumps(result))
        if not result["passed"]:
            raise SystemExit(1)
    elif stage == "qtcount":
        print("QTCOUNT " + json.dumps(qtable_count.remote()))
    elif stage == "dtd":
        print(run_dtd.remote(size, limit, pages))
    elif stage == "check_dtd":
        result = check_stage.remote(size)
        print("CHECK " + json.dumps(result))
        if not result["passed"]:
            raise SystemExit(1)
    else:
        raise SystemExit("stage must be 'env', 'dtd' or 'check_dtd'")
