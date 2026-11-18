"""Does a page's top score rise with its number of 512 tiles? (T13 follow-up)

Reads only existing ADCD-Net outputs (out/adcd/<size>/<page_id>.npz, already copied to a
local folder) and, optionally, meta/sample.csv for the document ID. Needs only numpy.

For every page: tiles = ceil(height / 512) * ceil(width / 512), taken from the stored shape.
Prints, per size folder: Pearson and Spearman correlation of top score with tile count, and the
mean top score by tile-count group (1-4, 5-12, 13 or more), each with a 95% range from 1,000
bootstrap resamples of documents (pages of one document stay together; seed 42). Without
--sample-csv the resampling is by page and says so. Only anonymous page IDs and numbers are
printed. Nothing is written to disk.

Usage:
  python analysis/tile_check/tile_count_vs_top.py --folder <dir with .npz files> [--folder ...]
         [--sample-csv <meta/sample.csv>] [--limit 3]
"""

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np

# The ADCD-Net tile size used at full size, and the seed of every random choice.
PIECE = 512
SEED = 42
N_BOOT = 1000
# Tile-count groups: (label, lowest, highest).
GROUPS = [("1-4 tiles", 1, 4), ("5-12 tiles", 5, 12), ("13+ tiles", 13, 10**9)]
PAGE_ID = re.compile(r"^page_\d{4,}$")


def load_tops_and_tiles(folder, limit):
    """Return (page_ids, top scores, tile counts) from every .npz under `folder`."""
    files = sorted(p for p in Path(folder).rglob("*.npz") if PAGE_ID.match(p.stem))
    if limit:
        files = files[:limit]
    ids, tops, tiles = [], [], []
    for path in files:
        with np.load(path) as data:
            # Only `top` and `shape` are read; the packed flags are left alone.
            top = float(data["top"])
            height, width = (int(v) for v in data["shape"])
        ids.append(path.stem)
        tops.append(top)
        tiles.append(math.ceil(height / PIECE) * math.ceil(width / PIECE))
    return ids, np.array(tops), np.array(tiles)


def doc_ids_for(ids, sample_csv):
    """Document ID per page (used only to keep pages of one document together; never printed)."""
    if not sample_csv:
        # No metadata: every page is its own cluster.
        return list(ids), False
    with open(sample_csv, encoding="utf-8", newline="") as handle:
        by_page = {row["page_id"]: row["doc_id"] for row in csv.DictReader(handle)}
    return [by_page.get(i, i) for i in ids], True


def average_ranks(x):
    # Ranks 1..n with ties given the average rank (needed for Spearman).
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    # Each distinct value occupies positions (end - count + 1) .. end; its rank is their midpoint.
    ends = np.cumsum(counts)
    return ((ends - counts + 1 + ends) / 2.0)[inverse]


def pearson(a, b):
    # Plain Pearson correlation; nan if one side does not vary.
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def statistics(tops, tiles):
    # Everything reported, as one dict (used for the full data and for each resample).
    out = {"pearson": pearson(tops, tiles), "spearman": pearson(average_ranks(tops), average_ranks(tiles))}
    for label, low, high in GROUPS:
        keep = (tiles >= low) & (tiles <= high)
        out[label] = float(tops[keep].mean()) if keep.any() else float("nan")
    return out


def bootstrap(tops, tiles, docs):
    # 95% ranges: resample documents with replacement, keep each document's pages together.
    rng = np.random.default_rng(SEED)
    clusters = {}
    for index, doc in enumerate(docs):
        clusters.setdefault(doc, []).append(index)
    members = list(clusters.values())
    draws = {}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(members), size=len(members))
        index = np.concatenate([members[k] for k in pick])
        for name, value in statistics(tops[index], tiles[index]).items():
            draws.setdefault(name, []).append(value)
    # nan-aware percentiles (a group can be empty in a resample).
    return {name: tuple(np.nanpercentile(vals, [2.5, 97.5])) for name, vals in draws.items()}


def report(folder, sample_csv, limit):
    ids, tops, tiles = load_tops_and_tiles(folder, limit)
    docs, by_document = doc_ids_for(ids, sample_csv)
    stats = statistics(tops, tiles)
    ranges = bootstrap(tops, tiles, docs)
    print(f"--- {len(ids)} pages | top score {tops.min():.4f} to {tops.max():.4f} | "
          f"tiles per page {tiles.min()} to {tiles.max()} | bootstrap by {'document' if by_document else 'PAGE (no sample.csv)'}")
    for name in ("pearson", "spearman"):
        print(f"{name:>9} correlation of top with tile count: {stats[name]:.3f} (95% range {ranges[name][0]:.3f} to {ranges[name][1]:.3f})")
    for label, low, high in GROUPS:
        n = int(((tiles >= low) & (tiles <= high)).sum())
        print(f"  mean top, {label:<10}: {stats[label]:.3f} (95% range {ranges[label][0]:.3f} to {ranges[label][1]:.3f}), n = {n} pages")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", action="append", required=True, help="folder with <page_id>.npz files (repeatable)")
    parser.add_argument("--sample-csv", default="", help="meta/sample.csv (for document IDs)")
    parser.add_argument("--limit", type=int, default=0, help="test mode: only the first N pages (use 3)")
    args = parser.parse_args()
    for folder in args.folder:
        print(f"[{Path(folder).name}]")
        report(folder, args.sample_csv, args.limit)


if __name__ == "__main__":
    main()
