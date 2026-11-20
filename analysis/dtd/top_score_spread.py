"""Spread of a detector's page-level top scores (T17, DTD).

Reads only the compact `.npz` outputs (the `top` value of each page) and the group and
file-format columns of `sample.csv`, both from a private folder outside the repo. It never
opens a page and prints numbers only, no page IDs.

Run (the private folder holds `sample.csv` and `outputs/<detector>/<size>/<page_id>.npz`):

    python -I analysis/dtd/top_score_spread.py --private-dir <private folder>
    python -I analysis/dtd/top_score_spread.py --private-dir <private folder> --limit 3   (test mode)
"""

import argparse
import collections
import csv
from pathlib import Path

import numpy as np

# The two input sizes every detector is run at (docs/PLAN.md section 3).
SIZES = ("shrink512", "full")


def quartiles(values):
    # One line: count, minimum, quartiles, maximum.
    a = np.asarray(values, dtype=np.float32)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return "n=%d min %.4f q1 %.4f med %.4f q3 %.4f max %.4f" % (len(a), a.min(), q1, med, q3, a.max())


def side(row):
    # "untouched" for authentic pages, "edited" for every tampered page (AI or handcrafted).
    return "untouched" if row["group"] == "authentic" else "edited"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-dir", required=True, type=Path, help="private folder with sample.csv and outputs/")
    parser.add_argument("--detector", default="dtd", help="folder name under outputs/ (default: dtd)")
    parser.add_argument("--limit", type=int, default=0, help="test mode: only the first N pages (0 = all)")
    args = parser.parse_args()

    # Page list with group and file format (UTF-8, as the repo rules say).
    with open(args.private_dir / "sample.csv", encoding="utf-8", newline="") as handle:
        meta = {row["page_id"]: row for row in csv.DictReader(handle)}
    page_ids = sorted(meta)[: args.limit] if args.limit > 0 else sorted(meta)
    print("pages:", len(page_ids), "| groups:", dict(collections.Counter(meta[p]["group"] for p in page_ids)),
          "| formats:", dict(collections.Counter(meta[p]["ext"] for p in page_ids)))

    for size in SIZES:
        folder = args.private_dir / "outputs" / args.detector / size
        # Each page's highest score, read from its compact .npz.
        tops = {}
        for page_id in page_ids:
            with np.load(folder / (page_id + ".npz")) as npz:
                tops[page_id] = float(npz["top"])
        print("\n==", size)

        # Distribution of the top score, edited pages against untouched pages.
        for label in ("edited", "untouched"):
            print(" ", label, quartiles([tops[p] for p in page_ids if side(meta[p]) == label]))

        # How many pages sit at the single highest value, and how many different values there are.
        every = np.array(list(tops.values()), dtype=np.float32)
        top_value = every.max()
        print("  max value %.6f | pages exactly at max %d | pages within 0.001 of max %d | distinct values %d" % (
            top_value, (every == top_value).sum(), (every >= top_value - 0.001).sum(), len(np.unique(every))))
        for label in ("edited", "untouched"):
            v = np.array([tops[p] for p in page_ids if side(meta[p]) == label], dtype=np.float32)
            print("   %s: at max %d/%d; >=0.90 %d; 0.5 to 0.90 %d; below 0.5 %d" % (
                label, (v == top_value).sum(), len(v), (v >= 0.90).sum(), ((v >= 0.5) & (v < 0.90)).sum(), (v < 0.5).sum()))
        # Values rounded to 4 decimals as plain Python numbers, so the print has no float noise.
        common = collections.Counter(round(float(x), 4) for x in every).most_common(5)
        print("  most common top values (4 decimals):", common)

        # Same distribution split by the format of the stored page (png or jpg).
        for ext in sorted({meta[p]["ext"] for p in page_ids}):
            for label in ("edited", "untouched"):
                v = [tops[p] for p in page_ids if meta[p]["ext"] == ext and side(meta[p]) == label]
                if v:
                    print("  format %-4s %-9s %s" % (ext, label, quartiles(v)))


if __name__ == "__main__":
    main()
