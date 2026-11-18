"""Real-page check for one detector's copied outputs (run before scoring).

For each size it checks, on every page of sample.csv:
    - the .npz exists and holds top (float32 scalar), shape (int32, 2 values), flag (uint8, 1-D)
    - flag unpacks to the stored shape, and top is between 0 and 1
    - rule A agrees with the score: a page has a flagged pixel exactly when top >= 0.5
    - every edit box fits inside the stored page shape (catches swapped height and width)
    - run_log.csv has one "ok" row per page and its max_value matches top
and that run_info.json holds the required fields. Prints counts and page IDs only.
Exit code 0 = everything passed, 1 = a problem was found.

Run:
    python3 shared/check_outputs.py --detector adcd --sample <sample.csv> --out-root <private outputs folder>
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Make output_format importable whether run directly or through pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import output_format as of  # noqa: E402

# Largest allowed gap between run_log.csv's max_value (6 decimals) and the file's top.
LOG_TOLERANCE = 1e-4


def check_size(detector, size, sample_rows, out_root):
    """Check one size. Returns (problems: {name: [page_ids]}, number of pages checked)."""
    folder = of.output_dir(out_root, detector, size)
    problems = {}

    def flag(name, page_id=""):
        # Record one problem (page_id may be empty for whole-folder problems).
        problems.setdefault(name, []).append(page_id)

    # run_info.json must hold every required field.
    try:
        of.read_run_info(folder / "run_info.json")
    except (OSError, ValueError) as error:
        flag(f"run_info: {type(error).__name__}")
    # run_log.csv: one row per page, all "ok".
    log = {}
    try:
        with open(folder / "run_log.csv", encoding="utf-8", newline="") as handle:
            log = {r["page_id"]: r for r in csv.DictReader(handle)}
    except OSError:
        flag("run_log.csv missing")
    for page_id, row in log.items():
        if row["status"] != "ok":
            flag("run_log status not ok", page_id)
    checked = 0
    for row in sample_rows:
        page_id = row["page_id"]
        path = folder / f"{page_id}.npz"
        if not path.exists():
            flag("npz missing", page_id)
            continue
        with np.load(path) as data:
            if sorted(data.files) != ["flag", "shape", "top"]:
                flag("wrong arrays in npz", page_id)
                continue
            good_types = (data["top"].dtype == np.float32 and data["top"].shape == ()
                          and data["shape"].dtype == np.int32 and data["shape"].shape == (2,)
                          and data["flag"].dtype == np.uint8 and data["flag"].ndim == 1)
        if not good_types:
            flag("wrong dtype or shape of arrays", page_id)
            continue
        out = of.load_page_output(path)
        height, width = out["shape"]
        checked += 1
        if out["flag"].shape != (height, width):
            flag("flag does not unpack to shape", page_id)
        if not 0.0 <= out["top"] <= 1.0:
            flag("top outside 0 to 1", page_id)
        # Rule A (any flagged pixel) must equal top >= 0.5.
        if bool(out["flag"].any()) != (out["top"] >= of.FLAG_THRESHOLD):
            flag("rule A differs from top >= 0.5", page_id)
        # Every edit box must fit inside (width, height).
        for x1, y1, x2, y2 in json.loads(row["boxes"]):
            if x2 > width or y2 > height:
                flag("edit box outside page shape", page_id)
                break
        if page_id in log and abs(float(log[page_id]["max_value"]) - out["top"]) > LOG_TOLERANCE:
            flag("run_log max_value differs from top", page_id)
    if len(log) != len(sample_rows):
        flag(f"run_log has {len(log)} rows, sample has {len(sample_rows)}")
    return problems, checked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--sizes", nargs="+", default=list(of.VALID_SIZES))
    args = parser.parse_args()
    with open(args.sample, encoding="utf-8", newline="") as handle:
        sample_rows = list(csv.DictReader(handle))
    failed = False
    for size in args.sizes:
        problems, checked = check_size(args.detector, size, sample_rows, args.out_root)
        ok = not problems
        failed |= not ok
        print(f"{args.detector}/{size}: pages in sample {len(sample_rows)}, checked {checked}, "
              f"problems {sum(len(v) for v in problems.values())} -> {'PASS' if ok else 'FAIL'}")
        for name, ids in problems.items():
            print(f"  {name}: {len(ids)} (first: {[i for i in ids[:5] if i]})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
