"""DAF inference script (T15). Runs INSIDE the Python 3.9 environment of the Modal image, started by
detectors/daf/modal_job.py (the Modal job itself runs on a newer Python and only launches this file).

Three modes:
  env    load both weight files, print the missing/unexpected key counts, run one random-noise image
  run    run one weight at one size on the pages and save the compact outputs
  check  check a finished weight and size on the Volume (CPU only)

Method follows the team's Week 7 notebook and the authors' README (github.com/qcf-568/OSTF):
the config's test step is a resize to fit 1024 x 1536 keeping the shape; DAF returns boxes (in the
pixels of the image it was given) with [real, tampered] probabilities; the tampered score is column 1.
Each box is filled with its score into a page-sized map (shared/output_format.py boxes_to_heatmap).

Nothing about a page is printed except its anonymous ID and numbers.
"""

import argparse
import csv
import json
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# The Volume as seen inside the container, and where the weights, pages and outputs sit on it.
DATA = Path("/data")
WEIGHTS_DIR = DATA / "weights" / "daf"
PAGES_DIR = DATA / "pages"
OUT_ROOT = DATA / "out"
# The authors' code (cloned, pinned and unpacked when the image is built).
REPO = "/opt/OSTF"
# The two released weight files and the name of each detector folder under out/.
WEIGHT_FILES = {"anytext": "anytext.pth", "diffste": "diffste.pth"}
SEED = 42
# The config's test-time resize (fit within 1024 x 1536, keep the shape), as in the authors' config.
TEST_RESCALE = (1024, 1536)
# Most boxes kept per image (authors' README, inference step).
MAX_PER_IMG = 1000
# Tell the parent to save the Volume after this many pages.
COMMIT_EVERY = 25
# Folder of this file: tiling.py and output_format.py are copied next to it in the image.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import output_format as of  # noqa: E402
import tiling  # noqa: E402


def detector_name(weight):
    # Folder name under out/ and the "detector" value in run_info.json: daf_anytext or daf_diffste.
    return "daf_" + weight


def package_versions():
    # The versions really installed in this environment (saved in run_info.json).
    import importlib.metadata as md
    versions = {"python": sys.version.split()[0]}
    for name in ["torch", "torchvision", "numpy", "opencv-python", "opencv-python-headless", "Pillow",
                 "mmcv", "mmengine", "timm", "scipy", "numba"]:
        try:
            versions[name] = md.version(name)
        except Exception:  # noqa: BLE001 - a missing name must not stop the run
            versions[name] = "unknown"
    import mmdet
    import torch
    versions["mmdet"] = mmdet.__version__
    versions["torch_cuda"] = torch.version.cuda
    return versions


def list_pages(limit=0, names=""):
    # All page files on the Volume whose name is an anonymous ID, sorted.
    pages = sorted(p for p in PAGES_DIR.iterdir() if of.PAGE_ID_PATTERN.match(p.stem))
    # Named pages (e.g. "page_0001,page_0003,page_0009"): exactly these, in the order given.
    if names:
        wanted = [n.strip() for n in names.split(",") if n.strip()]
        by_id = {p.stem: p for p in pages}
        unknown = [n for n in wanted if n not in by_id]
        if unknown:
            raise ValueError("pages not found on the Volume: " + ", ".join(unknown))
        return [by_id[n] for n in wanted]
    # Test mode: only the first `limit` pages.
    return pages[:limit] if limit > 0 else pages


# ---------------------------------------------------------------- model
def load_model(weight):
    """Build DAF (Faster R-CNN) from the config of this weight and load the weight file.

    Returns (model on the GPU in eval mode, number of missing keys, number of unexpected keys).
    The caller decides what to do with non-zero counts.
    """
    import torch
    # The repo root holds suploss.py, which mmdet imports.
    sys.path.insert(0, REPO)
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    import mmdet.models  # noqa: F401 - registers every DAF part

    cfg = Config.fromfile(REPO + "/configs/fasterrcnn_x" + weight + ".py")
    # Authors' README: inference uses max_per_img = 1000.
    cfg.model.test_cfg.rcnn.max_per_img = MAX_PER_IMG
    init_default_scope("mmdet")
    model = MODELS.build(cfg.model)
    checkpoint = torch.load(str(WEIGHTS_DIR / WEIGHT_FILES[weight]), map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    # Drop a "module." prefix left by multi-GPU training.
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    model = model.cuda().eval()
    return model, len(result.missing_keys), len(result.unexpected_keys)


def run_one(model, img):
    """Run DAF on one BGR image (numpy, H x W x 3). Returns a list of (x1, y1, x2, y2, tampered_score).

    The boxes are in the pixels of the image given. An image with no box gives an empty list.
    """
    import mmcv
    import numpy as np
    import torch
    from mmdet.structures import DetDataSample

    height, width = img.shape[:2]
    # The config's test step: resize to fit 1024 x 1536, keep the shape.
    resized = mmcv.imrescale(img, TEST_RESCALE, interpolation="bilinear", backend="cv2")
    new_h, new_w = resized.shape[:2]
    tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1)))
    # scale_factor is (width scale, height scale), as mmdet uses it to bring boxes back to this image.
    sample = DetDataSample(metainfo=dict(img_id=0, img_path="page", ori_shape=(height, width),
                                         img_shape=(new_h, new_w), scale_factor=(new_w / width, new_h / height)))
    with torch.no_grad():
        boxes, probs = model.test_step(dict(inputs=[tensor], data_samples=[sample]))[0]
    if hasattr(boxes, "tensor"):
        boxes = boxes.tensor
    # No box: the model returns an empty container instead of a tensor.
    if (not torch.is_tensor(probs)) or probs.numel() == 0:
        return []
    boxes = boxes.float().cpu().numpy()
    # Column 1 of the softmax is "tampered".
    tampered = probs.float().cpu().numpy()[:, 1]
    return [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(min(max(s, 0.0), 1.0)))
            for b, s in zip(boxes, tampered)]


def page_boxes(model, path, size):
    """Return (height, width, boxes in page pixels) for one page at one size."""
    import cv2
    # Raw pixels, no EXIF turning (same as the masks and boxes).
    img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError("page could not be read")
    height, width = img.shape[:2]
    boxes = []
    if size == "shrink512":
        # Shrink so the long side is 512 (area averaging); DAF then rescales it itself.
        new_w, new_h = tiling.shrunk_size(height, width)
        interp = cv2.INTER_AREA if max(height, width) > tiling.SHRINK_LONG else cv2.INTER_LINEAR
        small = cv2.resize(img, (new_w, new_h), interpolation=interp)
        # Boxes come back in the shrunk copy's pixels: scale them up to the page.
        boxes = tiling.scale_boxes(run_one(model, small), width / new_w, height / new_h)
    else:
        # full: DAF-sized pieces, no overlap; each piece's boxes are moved to page pixels.
        for y0, x0, piece_w, piece_h in tiling.piece_grid(height, width):
            piece = tiling.cut_piece(img, y0, x0, piece_w, piece_h)
            boxes.extend(tiling.shift_boxes(run_one(model, piece), x0, y0))
    return height, width, boxes


# ---------------------------------------------------------------- mode: env
def mode_env():
    """Both weights: key counts must be 0 and 0; one random-noise image (not a page) must run."""
    import numpy as np
    import torch

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    result = {"versions": package_versions(), "cuda_available": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "weights": {}}
    passed = torch.cuda.is_available()
    # A fixed random image of 1024 x 1536 grey noise.
    noise = np.random.default_rng(SEED).integers(0, 256, (1024, 1536, 3), dtype=np.uint8)
    for weight in WEIGHT_FILES:
        path = WEIGHTS_DIR / WEIGHT_FILES[weight]
        entry = {"file_size": path.stat().st_size if path.exists() else None}
        if entry["file_size"] is None:
            entry["problem"] = "weight file missing on the Volume"
            passed = False
        else:
            model, n_missing, n_unexpected = load_model(weight)
            entry["missing_keys"], entry["unexpected_keys"] = n_missing, n_unexpected
            boxes = run_one(model, noise)
            entry["noise_boxes"] = len(boxes)
            entry["noise_max_score"] = round(max([b[4] for b in boxes], default=0.0), 4)
            # Zero missing and zero unexpected keys, or the weight does not match its config.
            passed = passed and n_missing == 0 and n_unexpected == 0
            del model
            torch.cuda.empty_cache()
        result["weights"][weight] = entry
    result["passed"] = bool(passed)
    print("RESULT " + json.dumps(result), flush=True)
    return 0 if passed else 1


# ---------------------------------------------------------------- mode: run
def build_run_info(args, size, versions, gpu, n_missing, n_unexpected, seconds, n_ok, n_bad, started, extra):
    """Content of run_info.json. The first seven keys are the ones shared/score.py requires."""
    info = {
        # --- required by shared/score.py ---
        "detector": detector_name(args.weight),
        "weights": WEIGHT_FILES[args.weight],
        "size": size,
        # [length along the page's long side, along its short side]; null for shrink512.
        "tile_size": [tiling.PIECE_LONG, tiling.PIECE_SHORT] if size == "full" else None,
        "versions": versions,
        # Median seconds per page over every ok page in run_log.csv (test runs included).
        "seconds_per_page": round(statistics.median(seconds), 3) if seconds else None,
        "seed": SEED,
        # --- extra detail ---
        "shrink_longest_side": tiling.SHRINK_LONG if size == "shrink512" else None,
        "threshold_for_flagging": of.FLAG_THRESHOLD,
        "value_scale": "softmax probability of tampered (column 1) per box, float32; box filled with its score, maximum where boxes overlap, 0 elsewhere",
        "config": "configs/fasterrcnn_x%s.py" % args.weight,
        "max_per_img": MAX_PER_IMG,
        "test_rescale": list(TEST_RESCALE),
        "repo_url": "https://github.com/qcf-568/OSTF",
        "repo_commit": args.repo_commit,
        "gpu": gpu,
        "model_load": {"missing_keys": n_missing, "unexpected_keys": n_unexpected},
        "pages_ok": n_ok,
        "pages_not_ok": n_bad,
        "mean_seconds_per_page": round(statistics.mean(seconds), 3) if seconds else None,
        "limit_used_in_last_call": args.limit,
        "last_call_started_utc": started,
        "last_call_pages_without_boxes": extra["no_boxes"],
        "last_call_boxes_total": extra["boxes"],
        "notes": "Input as in the authors' README and config: page read as BGR, resized by DAF to fit 1024x1536 keeping the shape, "
                 "max_per_img=1000. full = 1536 (long side) x 1024 (short side) pieces from (0,0), no overlap, white padding "
                 "right/bottom, boxes moved to page pixels and clipped to the page. shrink512 = long side 512 (cv2 INTER_AREA), "
                 "boxes scaled back to the page. A box that crosses a piece border is split in two.",
    }
    missing = [k for k in of.REQUIRED_RUN_INFO_FIELDS if k not in info]
    if missing:
        raise ValueError("run_info is missing required fields: " + ", ".join(missing))
    return info


def mode_run(args):
    """Run one weight at one size on the pages and save the compact outputs."""
    import numpy as np
    import torch

    size = args.size
    if size not in of.VALID_SIZES:
        raise ValueError("size must be one of %s" % (of.VALID_SIZES,))
    if args.weight not in WEIGHT_FILES:
        raise ValueError("weight must be one of %s" % (sorted(WEIGHT_FILES),))
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, n_missing, n_unexpected = load_model(args.weight)
    print("model ready | weight %s | missing keys: %d | unexpected keys: %d" % (args.weight, n_missing, n_unexpected), flush=True)
    # A weight that does not match its config would give nonsense: stop at once.
    if n_missing or n_unexpected:
        raise RuntimeError("weights do not match the model (missing %d, unexpected %d)" % (n_missing, n_unexpected))
    versions = package_versions()
    gpu = torch.cuda.get_device_name(0)

    out_dir = of.output_dir(OUT_ROOT, detector_name(args.weight), size)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pages = list_pages(args.limit, args.pages)
    done = 0
    extra = {"no_boxes": 0, "boxes": 0}
    for page in pages:
        page_id = page.stem
        # Resumable: skip pages that already have a result.
        if (out_dir / (page_id + ".npz")).exists():
            continue
        t0 = time.time()
        try:
            height, width, boxes = page_boxes(model, page, size)
            # Fill each box with its score; maximum where boxes overlap; 0 elsewhere.
            heat = of.boxes_to_heatmap((height, width), boxes)
            of.save_page_output(out_dir, page_id, heat)
            torch.cuda.synchronize()
            extra["boxes"] += len(boxes)
            extra["no_boxes"] += 0 if boxes else 1
            of.append_run_log(out_dir, page_id, size, time.time() - t0, float(heat.max()), "ok")
        except Exception as exc:  # noqa: BLE001 - one bad page must not stop the run
            # Only the error type goes to the file; the console also gets the code line, never the message.
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            print("page %s failed: %s in %s line %d" % (page_id, type(exc).__name__, frame.name, frame.lineno), flush=True)
            of.append_run_log(out_dir, page_id, size, time.time() - t0, 0.0, "error:" + type(exc).__name__)
            # Free GPU memory after a failure (for example out of memory) so the next page can run.
            torch.cuda.empty_cache()
        done += 1
        if done % COMMIT_EVERY == 0:
            # The Modal job saves the Volume when it sees this line.
            print("COMMIT", flush=True)
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
    info = build_run_info(args, size, versions, gpu, n_missing, n_unexpected, seconds, n_ok, n_bad, started, extra)
    of.write_run_info(out_dir, info)
    # Read it back with the shared reader, so a missing key fails the run here and not in scoring.
    of.read_run_info(out_dir / "run_info.json")
    print("COMMIT", flush=True)
    print("RESULT " + json.dumps({"detector": detector_name(args.weight), "size": size, "processed_this_call": done,
                                  "ok_in_log": n_ok, "not_ok_in_log": n_bad, **extra}), flush=True)
    return 0


# ---------------------------------------------------------------- mode: check
def mode_check(args):
    """Check one finished weight and size on the Volume (CPU). Prints numbers and anonymous IDs only.

    A result for every page, no failed page, every .npz opens with top in 0..1 and the page's shape,
    and run_info.json has the required keys. A DAF page with no box legitimately has top 0, so there is
    no minimum share of pages above 0 (unlike DTD); the share is printed instead.
    """
    from PIL import Image

    size = args.size
    pages = list_pages()
    out_dir = of.output_dir(OUT_ROOT, detector_name(args.weight), size)
    cap = 50
    absent, bad, tops = [], [], []
    for page in pages:
        npz = out_dir / (page.stem + ".npz")
        if not npz.exists():
            absent.append(page.stem)
            continue
        try:
            out = of.load_page_output(npz)
            # PIL reads the size without turning the page by its EXIF tag, like our reading code.
            with Image.open(page) as image:
                page_shape = (image.size[1], image.size[0])
            if not (0.0 <= out["top"] <= 1.0) or out["shape"] != page_shape:
                bad.append(page.stem)
            else:
                tops.append(out["top"])
        except Exception:  # noqa: BLE001 - an unreadable file counts as bad
            bad.append(page.stem)
    # Pages whose latest run_log row is not ok.
    failed = []
    log_path = out_dir / "run_log.csv"
    if log_path.exists():
        last = {}
        with open(log_path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                last[row["page_id"]] = row["status"]
        failed = sorted(k for k, v in last.items() if v != "ok")
    info_missing = list(of.REQUIRED_RUN_INFO_FIELDS)
    if (out_dir / "run_info.json").exists():
        with open(out_dir / "run_info.json", encoding="utf-8") as handle:
            info = json.load(handle)
        info_missing = [k for k in of.REQUIRED_RUN_INFO_FIELDS if k not in info]
    n_pos = sum(1 for t in tops if t > 0.0)
    n_flag = sum(1 for t in tops if t >= of.FLAG_THRESHOLD)
    result = {
        "kind": detector_name(args.weight), "size": size, "n_pages": len(pages),
        "results_found": len(pages) - len(absent),
        "absent_ids": absent[:cap], "n_absent": len(absent),
        "failed_ids": failed[:cap], "n_failed": len(failed),
        "bad_ids": bad[:cap], "n_bad": len(bad),
        "run_info_ok": not info_missing, "run_info_missing_keys": info_missing,
        "pages_top_above_0": n_pos, "pages_top_at_least_0.5": n_flag,
        "top_min": round(min(tops), 4) if tops else None, "top_max": round(max(tops), 4) if tops else None,
    }
    result["passed"] = (not absent) and (not failed) and (not bad) and (not info_missing)
    print("RESULT " + json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["env", "run", "check"])
    parser.add_argument("--weight", default="anytext")
    parser.add_argument("--size", default="shrink512")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pages", default="")
    parser.add_argument("--repo-commit", default="unknown")
    args = parser.parse_args()
    if args.mode == "env":
        return mode_env()
    if args.mode == "run":
        return mode_run(args)
    return mode_check(args)


if __name__ == "__main__":
    sys.exit(main())
