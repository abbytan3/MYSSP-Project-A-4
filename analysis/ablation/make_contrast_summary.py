"""T20: paired contrasts for the Week 7 ablation (TextCtrl set only).

Three contrasts, each as a difference in AUC and in "edit found":
  A: AI edit minus each recreation (generation, copy_move, splicing), per size and format
  B: full size minus shrink512 (DAF: its `default` size is its own row), per version and format
  C: PNG minus JPG, per version and size
Paired: one bootstrap draw of documents (page ID = document) is used for both sides of a
difference. 1,000 resamples, seed 42 (from shared/score.py), 95% range; "clear" = range excludes 0.
AUC comes from shared/score.py; the file loader and name checks come from score_ablation.py.

Usage: python3 analysis/ablation/make_contrast_summary.py PER_PAGE_CSV [PER_PAGE_CSV ...] [--out FILE] [--test]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Reuse the ablation's loader and the shared scoring functions (read only)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import score as sc  # noqa: E402
import score_ablation as ab  # noqa: E402

# Only the TextCtrl set is used (the recreations are made on the same pages)
SET = "textctrl"
RECREATIONS = ["generation", "copy_move", "splicing"]
VERSIONS = ["ai_edit"] + RECREATIONS


def main():
    argv = sys.argv[1:]
    test = "--test" in argv
    out = HERE / "results" / "contrast_summary.csv"
    if "--out" in argv:
        i = argv.index("--out")
        out = Path(argv[i + 1])
        del argv[i:i + 2]
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        sys.exit(__doc__)
    # Same loader as the ablation: UTF-8, anonymous-ID check, SRNet skipped
    rows = [r for r in ab.load(paths, test) if r["set"] == SET]

    # Page ID = document; number the documents 0..n-1
    ids = sorted({r["page"] for r in rows})
    doc_of = {p: i for i, p in enumerate(ids)}
    n_docs = len(ids)

    # One entry per cell (detector, weights, size, format, version): its pages plus the
    # untouched pages of the same detector, weights, size and format
    group = defaultdict(list)
    for r in rows:
        group[(r["detector"], r["weights"], r["size"], r["format"], r["version"])].append(r)
    cells = {}
    for key, tamp in group.items():
        if key[4] == "untouched":
            continue
        pg = tamp + group[key[:4] + ("untouched",)]
        pos = np.array([r["version"] != "untouched" for r in pg])
        doc = np.array([doc_of[r["page"]] for r in pg])
        cells[key] = {
            "top": np.array([float(r["max_value"]) for r in pg]),
            "found": np.array([r["detected"] == "1.0" for r in pg]),
            "pos": pos,
            "by_doc": [np.flatnonzero(doc == d) for d in range(n_docs)],
            "n": (int(pos.sum()), int((~pos).sum())),
        }

    def stats(c, idx):
        # AUC of the page's top score and the share of edited pages with the edit found
        p = c["pos"][idx]
        top = c["top"][idx]
        auc = sc.auc(top[p], top[~p])
        found = c["found"][idx][p].mean() if p.any() else np.nan
        return auc, found

    # Point estimates on all pages, and one shared set of document draws for every cell
    point = {k: stats(c, np.arange(len(c["top"]))) for k, c in cells.items()}
    rng = np.random.RandomState(sc.SEED)
    draws = {k: np.full((sc.N_BOOT, 2), np.nan) for k in cells}
    for b in range(sc.N_BOOT):
        pick = rng.randint(0, n_docs, size=n_docs)
        for k, c in cells.items():
            idx = np.concatenate([c["by_doc"][d] for d in pick])
            draws[k][b] = stats(c, idx)

    # Build the contrast list as (contrast, side a cell, side b cell, label)
    pairs = []
    for (det, w, size, fmt, ver) in sorted(cells):
        # A: AI edit minus each recreation, same detector, weights, size and format
        if ver == "ai_edit":
            for rec in RECREATIONS:
                if (det, w, size, fmt, rec) in cells:
                    pairs.append(("A_ai_minus_recreation", (det, w, size, fmt, ver), (det, w, size, fmt, rec)))
        # B: full minus shrink512, and (DAF) default minus shrink512
        if size in ("full", "default") and (det, w, "shrink512", fmt, ver) in cells:
            label = "B_full_minus_512" if size == "full" else "B_default_minus_512"
            pairs.append((label, (det, w, size, fmt, ver), (det, w, "shrink512", fmt, ver)))
        # C: PNG minus JPG
        if fmt == "png" and (det, w, size, "jpg", ver) in cells:
            pairs.append(("C_png_minus_jpg", (det, w, size, fmt, ver), (det, w, size, "jpg", ver)))

    header = ["contrast", "detector", "weights", "version_a", "version_b", "size_a", "size_b",
              "format_a", "format_b", "n_tampered_a", "n_untouched_a", "n_tampered_b", "n_untouched_b"]
    for m in ("auc", "found"):
        header += [f"{m}_a", f"{m}_b", f"{m}_diff", f"{m}_diff_lo", f"{m}_diff_hi", f"{m}_clear"]
    table = []
    for name, a, b in pairs:
        line = [name, a[0], a[1], a[4], b[4], a[2], b[2], a[3], b[3], *cells[a]["n"], *cells[b]["n"]]
        for j in (0, 1):
            diff_draws = draws[a][:, j] - draws[b][:, j]
            lo, hi = sc.interval(diff_draws)
            d = point[a][j] - point[b][j]
            clear = "yes" if (lo > 0 or hi < 0) else "no"
            line += [round(point[a][j], 4), round(point[b][j], 4), round(d, 4), round(lo, 4), round(hi, 4), clear]
        table.append(line)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    print(f"wrote {out} ({len(table)} rows, {n_docs} documents, seed {sc.SEED}, {sc.N_BOOT} resamples)")


if __name__ == "__main__":
    main()
