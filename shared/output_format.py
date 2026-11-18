"""Shared detector output format (docs/PLAN.md section 3).

Every detector job (ADCD-Net, DTD, DAF) saves its result for one page with
`save_page_output`, and the scoring script reads it back with
`load_page_output`. No full heat maps are ever written to disk.

Layout on the Modal Volume `a4-data` (and anywhere else the jobs write):

    out/<detector>/<size>/<page_id>.npz   one file per page
    out/<detector>/<size>/run_info.json   settings of the run
    out/<detector>/<size>/run_log.csv     one row per page

Each .npz holds three arrays:
    top    float32 scalar    the page's highest heat-map value
    shape  int32, 2 values   (height, width) of the original page
    flag   uint8, 1-D        pixels >= 0.5, packed with np.packbits
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

# The one fixed threshold: a pixel is "flagged" when its value is >= 0.5.
FLAG_THRESHOLD = 0.5

# The two input sizes every detector runs at.
VALID_SIZES = ("shrink512", "full")

# Only anonymous IDs such as page_0001 are allowed, never real file names.
PAGE_ID_PATTERN = re.compile(r"^page_\d{4,}$")

# Column order of run_log.csv.
RUN_LOG_COLUMNS = ["page_id", "size", "seconds", "max_value", "status"]


def _check_page_id(page_id):
    # Refuse anything that is not an anonymous ID, so real names cannot leak.
    if not PAGE_ID_PATTERN.match(page_id):
        raise ValueError(f"page_id must look like page_0001, got {page_id!r}")


def _check_size(size):
    # Only the two agreed input sizes are allowed.
    if size not in VALID_SIZES:
        raise ValueError(f"size must be one of {VALID_SIZES}, got {size!r}")


def output_dir(root, detector, size):
    """Folder that holds one detector's output at one size: root/<detector>/<size>."""
    # Build the path with pathlib so it works on Mac and Windows.
    _check_size(size)
    return Path(root) / detector / size


def boxes_to_heatmap(shape, boxes):
    """Turn DAF-style boxes into a page-sized heat map.

    shape: (height, width) of the page.
    boxes: list of (x1, y1, x2, y2, score) in page pixels.
    Each box is filled with its score; where boxes overlap the maximum is
    kept; everything else is 0. Box edges are rounded outwards (floor for
    x1/y1, ceil for x2/y2), the end is exclusive, and boxes are clipped to
    the page.
    """
    height, width = int(shape[0]), int(shape[1])
    # Start with an all-zero map (no box = no evidence).
    heat = np.zeros((height, width), dtype=np.float32)
    for x1, y1, x2, y2, score in boxes:
        # Round the box outwards and clip it to the page.
        left = max(int(np.floor(x1)), 0)
        top = max(int(np.floor(y1)), 0)
        right = min(int(np.ceil(x2)), width)
        bottom = min(int(np.ceil(y2)), height)
        if right <= left or bottom <= top:
            continue
        # Keep the larger score where boxes overlap.
        region = heat[top:bottom, left:right]
        np.maximum(region, np.float32(score), out=region)
    return heat


def save_page_output(out_dir, page_id, heatmap):
    """Save one page's compact output and return the file path.

    heatmap: 2-D array, same height and width as the original page,
    values between 0 and 1. It is converted to float32 first, and both
    `top` and `flag` are computed from that float32 copy.
    """
    _check_page_id(page_id)
    # Convert once so `top` and `flag` always agree with each other.
    heat = np.asarray(heatmap, dtype=np.float32)
    if heat.ndim != 2 or heat.size == 0:
        raise ValueError(f"heatmap must be a non-empty 2-D array, got shape {heat.shape}")
    # A NaN would silently count as "not flagged", so refuse it.
    if not np.isfinite(heat).all():
        raise ValueError("heatmap contains NaN or infinity")
    if heat.min() < 0.0 or heat.max() > 1.0:
        raise ValueError("heatmap values must be between 0 and 1")

    # The page's highest score (used for rule B and AUC).
    top = np.float32(heat.max())
    # Page height and width (used to unpack the flags).
    shape = np.array(heat.shape, dtype=np.int32)
    # Pixels at or above 0.5, flattened row by row and packed 8 per byte.
    flag = np.packbits((heat >= FLAG_THRESHOLD).ravel())

    # Make the folder if needed, then write the compressed file.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{page_id}.npz"
    np.savez_compressed(path, top=top, shape=shape, flag=flag)
    return path


def load_page_output(path):
    """Read one .npz and return {'top': float, 'shape': (h, w), 'flag': bool array (h, w)}."""
    with np.load(Path(path)) as data:
        top = float(data["top"])
        height, width = (int(v) for v in data["shape"])
        packed = data["flag"]
    # Unpack the bits, drop the padding bits at the end, restore the 2-D shape.
    bits = np.unpackbits(packed)[: height * width]
    flag = bits.reshape(height, width).astype(bool)
    return {"top": top, "shape": (height, width), "flag": flag}


def write_run_info(out_dir, info):
    """Write run_info.json (weights, sizes, tile size, versions, seconds per page, seed)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_info.json"
    # UTF-8, sorted keys, so the file is easy to compare between runs.
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def append_run_log(out_dir, page_id, size, seconds, max_value, status):
    """Add one row to run_log.csv (header is written the first time)."""
    _check_page_id(page_id)
    _check_size(size)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_log.csv"
    # Write the header only if the file is new.
    is_new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(RUN_LOG_COLUMNS)
        writer.writerow([page_id, size, f"{seconds:.3f}", f"{max_value:.6f}", status])
    return path
