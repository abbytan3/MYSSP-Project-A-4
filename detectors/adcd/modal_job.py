"""ADCD-Net Modal job (T13). Two stages, both run from this laptop:

  1. OCR masks (CPU, many containers):
       modal run detectors/adcd/modal_job.py --stage ocr [--limit 3]
  2. ADCD-Net (one T4 GPU, one size at a time):
       modal run detectors/adcd/modal_job.py --stage adcd --size shrink512 [--limit 3]
       modal run detectors/adcd/modal_job.py --stage adcd --size full      [--limit 3]

Reads pages from the private Volume `a4-data` (pages/<page_id>.<ext>) and writes
  adcd_work/ocr_masks/<page_id>.png          page-sized text mask (Volume only)
  out/adcd/<size>/<page_id>.npz              compact output (shared/output_format.py)
  out/adcd/<size>/run_info.json, run_log.csv
Weights must already be on the Volume (see detectors/adcd/README.md).

Nothing about a page is ever printed: only anonymous IDs and numbers.
"""

import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------- settings
# Name of the private Volume that holds pages, weights and outputs.
VOLUME_NAME = "a4-data"
# Where the Volume appears inside every container.
DATA = "/data"
# Weights and quantisation table on the Volume.
WEIGHTS_DIR = DATA + "/weights/adcd"
CKPT = WEIGHTS_DIR + "/ADCDNet.pth"
DOCRES = WEIGHTS_DIR + "/docres.pkl"
QT = WEIGHTS_DIR + "/qt_table.pk"
# Pages, OCR masks and outputs on the Volume.
PAGES_DIR = DATA + "/pages"
OCR_DIR = DATA + "/adcd_work/ocr_masks"
OCR_LOG_DIR = DATA + "/adcd_work/ocr_logs"
OUT_ROOT = DATA + "/out"
# The authors' code, pinned to the commit used in Week 5/7.
REPO_URL = "https://github.com/KAHIMWONG/ACDC-Net"
REPO_COMMIT = "169d04d"
REPO_DIR = "/opt/adcd_repo"
# ADCD-Net's own input size, and the seed (docs/PLAN.md).
PIECE = 512
SEED = 42
# Pages per OCR container call, and how many OCR containers run at once.
OCR_CHUNK = 20
OCR_MAX_CONTAINERS = 8
# Save the Volume to disk after this many pages.
COMMIT_EVERY = 25

# Folder of this file, and the shared folder (both are copied into the images).
HERE = Path(__file__).resolve().parent
SHARED = HERE.parents[1] / "shared"

app = modal.App("a4-adcd")
volume = modal.Volume.from_name(VOLUME_NAME)

# ---------------------------------------------------------------- images
# GPU image: the package versions that worked in Week 5/7 (env_versions_2A.txt).
adcd_image = (
    modal.Image.debian_slim(python_version="3.10")
    # System JPEG headers and a compiler, needed to build jpeg2dct.
    .apt_install("git", "build-essential", "libjpeg-dev", "libgl1", "libglib2.0-0")
    # torch 2.3.0 for CUDA 12.1, from PyTorch's own download site.
    .pip_install(
        "torch==2.3.0+cu121", "torchvision==0.18.0+cu121", "numpy==1.26.4",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # The other packages the repo imports (numpy pinned so nothing pulls numpy 2).
    .pip_install(
        "numpy==1.26.4", "opencv-python-headless==4.11.0.86", "albumentations==2.0.8",
        "lmdb", "six", "pillow", "tqdm", "einops", "efficientnet_pytorch", "tensorboard",
    )
    # jpeg2dct has an old setup.py: it needs setuptools<70 and no build isolation.
    .pip_install("setuptools<70", "wheel", "numpy==1.26.4")
    .run_commands("pip install --no-build-isolation jpeg2dct")
    # Full clone, then the exact commit (a shallow clone may not hold it).
    .run_commands(
        f"git clone -q {REPO_URL} {REPO_DIR}",
        f"git -C {REPO_DIR} checkout -q {REPO_COMMIT}",
    )
    # Our own helpers, copied in last so editing them does not rebuild the image.
    .add_local_file(str(HERE / "tiling.py"), "/app/tiling.py")
    .add_local_file(str(SHARED / "output_format.py"), "/app/output_format.py")
)

# CPU image for PaddleOCR (its own environment, as in Week 5/7). Versions were not
# pinned in Week 7, so they are not pinned here either; they are logged at run time.
ocr_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgomp1", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "paddlepaddle", "paddleocr", "opencv-python-headless")
    .add_local_file(str(SHARED / "output_format.py"), "/app/output_format.py")
)


def _list_pages(limit=0):
    # All page files on the Volume whose name is an anonymous ID, sorted.
    sys.path.insert(0, "/app")
    from output_format import PAGE_ID_PATTERN
    pages = sorted(p for p in Path(PAGES_DIR).iterdir() if PAGE_ID_PATTERN.match(p.stem))
    # Test mode: only the first `limit` pages.
    return pages[:limit] if limit > 0 else pages


# ---------------------------------------------------------------- stage 1: OCR
@app.function(image=ocr_image, volumes={DATA: volume}, timeout=600)
def pages_without_ocr(limit=0):
    # IDs of pages that still have no OCR mask (this makes the stage resumable).
    return [p.name for p in _list_pages(limit) if not (Path(OCR_DIR) / (p.stem + ".png")).exists()]


@app.function(image=ocr_image, volumes={DATA: volume}, timeout=3600, max_containers=OCR_MAX_CONTAINERS)
def ocr_chunk(names):
    """Make page-sized PaddleOCR text masks for a few pages (same logic as Week 7 ocr_masks.py)."""
    import csv
    import os
    import time
    # Stop PaddleOCR from checking model-hosting sites at start-up.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    import cv2
    import numpy as np
    import paddle
    import paddleocr
    from paddleocr import TextDetection

    Path(OCR_DIR).mkdir(parents=True, exist_ok=True)
    Path(OCR_LOG_DIR).mkdir(parents=True, exist_ok=True)
    # Same detector as the authors' seg_char.py; oneDNN off because it crashed on CPU in Week 5.
    model = TextDetection(model_name="PP-OCRv5_server_det", enable_mkldnn=False)
    rows = []
    for name in names:
        t0 = time.time()
        page_path = str(Path(PAGES_DIR) / name)
        # Detect text boxes (batch of 1).
        res = model.predict(input=page_path, batch_size=1)[0]
        polys, scores = res["dt_polys"], res["dt_scores"]
        # Page height and width.
        h, w = cv2.imread(page_path).shape[:2]
        # All-black page-sized mask; fill every polygon with score > 0.5 (repo rule).
        mask = np.zeros((h, w), dtype=np.uint8)
        kept = 0
        for poly, score in zip(polys, scores):
            if score > 0.5:
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                kept += 1
        # Save as 0/255 PNG, via a temporary name so a crash leaves no half file.
        out = Path(OCR_DIR) / (Path(name).stem + ".png")
        ok, buf = cv2.imencode(".png", mask * 255)
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(buf.tobytes())
        os.replace(tmp, out)
        rows.append([Path(name).stem, round(time.time() - t0, 2), kept])
    # Per-chunk log (anonymous IDs only) with the versions used.
    log = Path(OCR_LOG_DIR) / ("chunk_%s.csv" % Path(names[0]).stem)
    with open(log, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["page_id", "seconds", "n_text_regions", "paddle=" + paddle.__version__, "paddleocr=" + paddleocr.__version__])
        writer.writerows(rows)
    volume.commit()
    return len(rows)


# ---------------------------------------------------------------- stage 2: ADCD-Net
@app.function(image=adcd_image, gpu="T4", volumes={DATA: volume}, timeout=6 * 3600)
def run_adcd(size, limit=0):
    """Run ADCD-Net on every page at one size and save the compact outputs."""
    import csv
    import importlib.metadata as md
    import json
    import os
    import pickle
    import statistics
    import tempfile
    import time
    from datetime import datetime, timezone

    sys.path.insert(0, "/app")
    import cv2
    import numpy as np
    import output_format as of
    import tiling

    # Only the two agreed sizes are allowed.
    if size not in of.VALID_SIZES:
        raise ValueError("size must be one of %s" % (of.VALID_SIZES,))

    # Work from inside the repo so its own imports (cfg, ds, model) work.
    os.chdir(REPO_DIR)
    sys.path.insert(0, REPO_DIR)
    # Point the repo's settings at our files in memory (cfg.py is not edited), as Week 7 infer.py did.
    pkl_dir = tempfile.mkdtemp(prefix="adcd_pkl_")
    import cfg
    cfg.ckpt = CKPT
    cfg.docres_ckpt_path = DOCRES
    cfg.pkl_dir = pkl_dir
    cfg.qt_path = QT
    cfg.val_max_size = PIECE
    # The repo's loader, padding function and model.
    import torch
    from ds import GeneralValDs, pad_collate
    from model.model import ADCDNet
    from torch.amp import autocast
    from torch.utils.data import DataLoader

    # Seeds (evaluation has no randomness, but this is our rule).
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Build the model (loads docres.pkl) and load ADCDNet.pth the way main.py does.
    model = ADCDNet().cuda()
    state = torch.load(CKPT, map_location="cpu", weights_only=True)["model"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    print("model ready | missing keys: %d | unexpected keys: %d" % (len(missing), len(unexpected)), flush=True)
    # A wrong weight file must stop the run, not give silent nonsense.
    if missing or unexpected:
        raise RuntimeError("weights do not match the model (missing %d, unexpected %d)" % (len(missing), len(unexpected)))

    # One all-zero 512x512 answer-key mask for tiles (the key is not used for prediction).
    work = Path(tempfile.mkdtemp(prefix="adcd_work_"))
    zero_tile_mask = work / "zero512.png"
    cv2.imwrite(str(zero_tile_mask), np.zeros((PIECE, PIECE), np.uint8))

    def predict_items(items):
        # items: list of (image path, mask path, ocr path). Returns one probability map per item.
        with open(os.path.join(pkl_dir, "page.pkl"), "wb") as handle:
            pickle.dump(items, handle)
        dataset = GeneralValDs("page")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=pad_collate)
        maps = []
        with torch.no_grad():
            for batch in loader:
                img, dct, qt, mask, ocr_mask = [x.cuda() for x in batch[:5]]
                h, w = batch[6][0].tolist()
                # float16 mode, exactly as main.py uses in evaluation.
                with autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(img, dct, qt, mask, ocr_mask, is_train=False)[0]
                # Crop the padding, back to float32, probability of class 1 ("tampered").
                logit = logits[0, :, :h, :w].float()
                maps.append(torch.softmax(logit, dim=0)[1].cpu().numpy())
        return maps

    def heatmap_for(page_path, ocr_path):
        # Page-sized heat map (0 to 1, float32) for one page at the chosen size.
        img = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
        height, width = img.shape[:2]
        ocr = cv2.imread(str(ocr_path), cv2.IMREAD_GRAYSCALE)
        if ocr is None or ocr.shape != (height, width):
            raise RuntimeError("OCR mask missing or wrong size")
        if size == "shrink512":
            # The loader shrinks the page to a longest side of 512 itself; give it a page-sized zero mask.
            zero = work / "zero_page.png"
            cv2.imwrite(str(zero), np.zeros((height, width), np.uint8))
            prob = predict_items([(str(page_path), str(zero), str(ocr_path))])[0]
            # Resize back to the page size with bilinear interpolation (width, height).
            if prob.shape != (height, width):
                prob = cv2.resize(prob.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            return np.clip(prob.astype(np.float32), 0.0, 1.0)
        # full: 512x512 tiles on the page grid, white padding for the image, 0 for the text mask.
        img_tiles = tiling.cut_tiles(img, 255)
        ocr_tiles = tiling.cut_tiles(ocr, 0)
        keys = sorted(img_tiles)
        items = []
        for (r, c) in keys:
            # Names carry no dots (the loader names things by the text before the first dot).
            img_file = work / ("t_r%d_c%d.png" % (r, c))
            ocr_file = work / ("o_r%d_c%d.png" % (r, c))
            cv2.imwrite(str(img_file), img_tiles[(r, c)])
            cv2.imwrite(str(ocr_file), ocr_tiles[(r, c)])
            items.append((str(img_file), str(zero_tile_mask), str(ocr_file)))
        maps = predict_items(items)
        heat = tiling.join_tiles(dict(zip(keys, maps)), height, width)
        # Remove this page's tile files.
        for (r, c) in keys:
            (work / ("t_r%d_c%d.png" % (r, c))).unlink()
            (work / ("o_r%d_c%d.png" % (r, c))).unlink()
        return np.clip(heat, 0.0, 1.0)

    out_dir = of.output_dir(OUT_ROOT, "adcd", size)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pages = _list_pages(limit)
    done = 0
    for page in pages:
        page_id = page.stem
        # Resumable: skip pages that already have a result.
        if (out_dir / (page_id + ".npz")).exists():
            continue
        t0 = time.time()
        try:
            heat = heatmap_for(page, Path(OCR_DIR) / (page_id + ".png"))
            of.save_page_output(out_dir, page_id, heat)
            torch.cuda.synchronize()
            of.append_run_log(out_dir, page_id, size, time.time() - t0, float(heat.max()), "ok")
        except Exception as exc:  # noqa: BLE001 - one bad page must not stop the run
            # Only the error type is logged, never the message (it could hold a path).
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
    of.write_run_info(out_dir, {
        "detector": "adcd",
        "size": size,
        "weights_files": ["ADCDNet.pth", "docres.pkl", "qt_table.pk"],
        "tile_size": PIECE if size == "full" else None,
        "shrink_longest_side": PIECE if size == "shrink512" else None,
        "threshold_for_flagging": of.FLAG_THRESHOLD,
        "value_scale": "softmax probability of tampered, float32 (not rounded to 8 bit)",
        "seed": SEED,
        "gpu": torch.cuda.get_device_name(0),
        "repo_url": REPO_URL,
        "repo_commit": REPO_COMMIT,
        "versions": {name: md.version(name) for name in
                     ["torch", "torchvision", "numpy", "opencv-python-headless", "albumentations", "pillow"]},
        "model_load": {"missing_keys": len(missing), "unexpected_keys": len(unexpected)},
        "pages_ok": n_ok,
        "pages_not_ok": n_bad,
        "median_seconds_per_page": round(statistics.median(seconds), 3) if seconds else None,
        "mean_seconds_per_page": round(statistics.mean(seconds), 3) if seconds else None,
        "limit_used_in_last_call": limit,
        "last_call_started_utc": started,
        "notes": "Official loader: greyscale, re-saved as JPEG quality 100 inside the loader, DCT numbers read from it. "
                 "full = 512x512 tiles from (0,0), white padding right/bottom, joined back, padding cropped. "
                 "shrink512 maps resized back to page size with bilinear interpolation.",
    })
    volume.commit()
    return {"size": size, "processed_this_call": done, "ok_in_log": n_ok, "not_ok_in_log": n_bad}


# ---------------------------------------------------------------- entry point
@app.local_entrypoint()
def main(stage: str, size: str = "shrink512", limit: int = 0):
    # stage: "ocr" or "adcd". limit: 0 = all pages, 3 = the first 3 (the test run).
    if stage == "ocr":
        todo = pages_without_ocr.remote(limit)
        print("pages still needing an OCR mask:", len(todo))
        chunks = [todo[i:i + OCR_CHUNK] for i in range(0, len(todo), OCR_CHUNK)]
        made = sum(ocr_chunk.map(chunks))
        print("OCR masks made:", made)
    elif stage == "adcd":
        print(run_adcd.remote(size, limit))
    else:
        raise SystemExit("stage must be 'ocr' or 'adcd'")
