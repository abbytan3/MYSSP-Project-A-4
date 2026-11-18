"""Is page size a shortcut? Tile count against edited / untouched (T13 follow-up).

Numbers only; nothing is written to disk and no page content is read. Uses:
  - sample.csv (anonymous; group and document of every page)
  - ADCD-Net full-size outputs (.npz): the stored page shape and the top score
and the AUC and resampling functions of shared/score.py (imported, not changed).
Needs numpy and scipy (shared/requirements.txt).

Tile count per page = ceil(height / 512) * ceil(width / 512), from the stored shape.
Prints:
  1. min / median / max tile count for edited vs untouched pages, and AI vs handcrafted.
  2. AUC of tile count alone (higher count = "edited") and of page area alone, for edited vs
     untouched pages, with a 95% range from 1,000 resamples of documents (seed 42; pages of
     one document stay together). Below 0.5 means untouched pages are the larger ones.
  3. AUC of the ADCD-Net full-size top score inside tile-count groups, next to the overall
     AUC (which is checked against results/adcd/scores.csv when --scores is given).
Grouping rule (fixed before looking at AUCs): walk up through the distinct tile counts and close
a group as soon as it holds at least MIN_PER_CLASS edited AND MIN_PER_CLASS untouched pages; a
short tail is merged into the last group. Groups with fewer than SMALL pages in either class are flagged.

Usage:
  python3 analysis/tile_check/tile_count_vs_label.py --sample <sample.csv> --folder <adcd/full folder> \
      [--scores results/adcd/scores.csv]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Use the shared scoring code (read-only): AUC, document resampling, sample reader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import score as sc  # noqa: E402

PIECE = 512
SEED = 42
N_BOOT = 1000
# A group closes once it has this many edited and this many untouched pages.
MIN_PER_CLASS = 30
# Groups with fewer pages than this in either class are flagged as small.
SMALL = 50


def read_shapes_and_tops(folder, page_ids):
    """Page height, width and top score for every page, from the .npz files."""
    heights, widths, tops = [], [], []
    for page_id in page_ids:
        # np.load reads only the arrays asked for, so the packed flags are not loaded.
        with np.load(Path(folder) / f"{page_id}.npz") as data:
            height, width = (int(v) for v in data["shape"])
            tops.append(float(data["top"]))
        heights.append(height)
        widths.append(width)
    return np.array(heights), np.array(widths), np.array(tops)


def make_groups(tiles, edited):
    """Group distinct tile counts into ranges by the fixed rule. Returns [(low, high), ...]."""
    values = sorted(set(tiles.tolist()))
    groups, start, ed, un = [], None, 0, 0
    for v in values:
        start = v if start is None else start
        ed += int(((tiles == v) & edited).sum())
        un += int(((tiles == v) & ~edited).sum())
        if ed >= MIN_PER_CLASS and un >= MIN_PER_CLASS:
            groups.append((start, v))
            start, ed, un = None, 0, 0
    # A short tail joins the last group.
    if start is not None:
        groups[-1] = (groups[-1][0], values[-1])
    return groups


def summary(x):
    # min / median / max of an array, as text.
    return f"{x.min():g} / {np.median(x):g} / {x.max():g}"


def interval(draws):
    # 95% range: 2.5th and 97.5th percentile of the usable draws.
    ok = draws[~np.isnan(draws)]
    return (np.nan, np.nan) if ok.size == 0 else tuple(np.percentile(ok, [2.5, 97.5]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--folder", required=True, type=Path, help="ADCD-Net full-size output folder with the .npz files")
    parser.add_argument("--scores", type=Path, default=None, help="results/adcd/scores.csv, to cross-check the overall AUC")
    args = parser.parse_args()

    sample = sc.load_sample(args.sample)
    height, width, top = read_shapes_and_tops(args.folder, sample["page_id"])
    tiles = (-(-height // PIECE)) * (-(-width // PIECE))
    area = height * width
    edited, group = sample["tampered"], sample["group"]

    # 1. Tile count: min / median / max.
    print("1. Tile count per page (min / median / max), pages:")
    for name, mask in (("edited", edited), ("untouched", ~edited), ("AI-edited", group == "ai"), ("handcrafted", group == "handcrafted")):
        print(f"   {name:<12} {summary(tiles[mask]):<14} n = {int(mask.sum())}")

    # Resampling set-up: the same draws as shared/score.py (RandomState(42), same call order).
    n_docs = int(sample["doc"].max()) + 1
    pages_of_doc = [np.flatnonzero(sample["doc"] == d) for d in range(n_docs)]
    groups = make_groups(tiles, edited)
    in_group = [(tiles >= lo) & (tiles <= hi) for lo, hi in groups]
    # Each row: (label, scores, positive mask, negative mask).
    tasks = [
        ("tile count, edited vs untouched", tiles.astype(float), edited, ~edited),
        ("tile count, AI vs untouched", tiles.astype(float), group == "ai", ~edited),
        ("tile count, handcrafted vs untouched", tiles.astype(float), group == "handcrafted", ~edited),
        ("page area, edited vs untouched", area.astype(float), edited, ~edited),
        ("page area, AI vs untouched", area.astype(float), group == "ai", ~edited),
        ("page area, handcrafted vs untouched", area.astype(float), group == "handcrafted", ~edited),
        ("ADCD full top score, all pages", top, edited, ~edited),
    ]
    for (lo, hi), mask in zip(groups, in_group):
        label = f"ADCD full top score, tiles {lo}-{hi}" if lo != hi else f"ADCD full top score, tiles {lo}"
        tasks.append((label, top, edited & mask, ~edited & mask))

    point = [sc.auc(x[p], x[n]) for _, x, p, n in tasks]
    draws = np.full((len(tasks), N_BOOT), np.nan)
    rng = np.random.RandomState(SEED)
    for b in range(N_BOOT):
        idx = sc.resample_indices(pages_of_doc, rng)
        for t, (_, x, p, n) in enumerate(tasks):
            draws[t, b] = sc.auc(x[idx][p[idx]], x[idx][n[idx]])

    print(f"\n2. AUC of page size alone (no detector), 95% range from {N_BOOT} document resamples, seed {SEED}:")
    for t, (label, _, p, n) in enumerate(tasks[:6]):
        lo, hi = interval(draws[t])
        print(f"   {label:<38} {point[t]:.3f} [{lo:.3f}, {hi:.3f}]  (n = {int(p.sum())} vs {int(n.sum())})")

    print(f"\n3. ADCD-Net full-size top score, AUC inside tile-count groups (group closes at >= {MIN_PER_CLASS} edited and >= {MIN_PER_CLASS} untouched):")
    lo, hi = interval(draws[6])
    print(f"   {'all pages':<14} {point[6]:.3f} [{lo:.3f}, {hi:.3f}]  (n = {int(tasks[6][2].sum())} vs {int(tasks[6][3].sum())})")
    for t in range(7, len(tasks)):
        label, _, p, n = tasks[t]
        lo, hi = interval(draws[t])
        flag = "  SMALL" if min(int(p.sum()), int(n.sum())) < SMALL else ""
        print(f"   {label.split(', ')[1]:<14} {point[t]:.3f} [{lo:.3f}, {hi:.3f}]  (n = {int(p.sum())} edited vs {int(n.sum())} untouched){flag}")
    print(f"   (SMALL = fewer than {SMALL} pages in either class; its range is wide.)")

    # Cross-check the overall AUC against the results file, if given.
    if args.scores:
        with open(args.scores, encoding="utf-8", newline="") as handle:
            row = next(r for r in csv.DictReader(handle)
                       if (r["size"], r["breakdown"], r["subset"], r["metric"]) == ("full", "all", "all", "auc"))
        print(f"\n   cross-check: overall AUC here {point[6]:.6f} [{interval(draws[6])[0]:.6f}, {interval(draws[6])[1]:.6f}]"
              f" vs results file {float(row['value']):.6f} [{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}]")


if __name__ == "__main__":
    main()
