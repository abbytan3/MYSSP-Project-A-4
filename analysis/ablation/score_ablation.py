"""T20 Week 7 ablation: rescore the safe per_page.csv files at document level and "edit found".

Rule A, rule B (2-fold cross-fitted cut-off), AUC and the bootstrap all come from
shared/score.py (T11); nothing is copied here. False-region counts cannot be
recomputed (heat maps were not kept), so location precision and F1 are left out.

Usage: python3 analysis/ablation/score_ablation.py PER_PAGE_CSV [PER_PAGE_CSV ...] [--out FILE] [--test]
--test keeps only 12 pages per set (3 are too few for two halves). DAF SRNet rows are skipped (decision by Abby).
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Import the shared scoring functions (read only: shared/ is not changed here)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import score as sc  # noqa: E402

# Anonymous page IDs only: page_0001 or img01 style
SAFE_ID = re.compile(r"^(page_\d{4}|img\d{2})$")
# Anything that looks like a real file name or doc ID must stop the run
UNSAFE = re.compile(r"_v\d+__[0-9a-f]{10}|\.(png|jpe?g|pdf|tiff?)\b", re.I)
# Metrics reported (shared names); location precision and F1 need false regions, so they are left out
KEEP = ["precision_A", "recall_A", "f1_A", "false_alarm_A",
        "precision_B", "recall_B", "f1_B", "false_alarm_B", "auc", "loc_recall"]


def load(paths, test=False):
    # Read every file as UTF-8, and refuse it if any cell looks like a real name
    rows = []
    for path in paths:
        with open(path, encoding="utf-8", newline="") as f:
            rows += list(csv.DictReader(f))
    # SRNet weight is not used in the main study (Abby's decision)
    rows = [r for r in rows if r["weights"] != "srnet"]
    for r in rows:
        if not SAFE_ID.match(r["page"]):
            sys.exit("Unsafe page id in input: stop and check the file.")
        if any(UNSAFE.search(v) for v in r.values()):
            sys.exit("A cell looks like a real file name: stop and check the file.")
    if test:
        keep = defaultdict(list)
        for r in rows:
            if r["page"] not in keep[r["set"]] and len(keep[r["set"]]) < 12:
                keep[r["set"]].append(r["page"])
        rows = [r for r in rows if r["page"] in keep[r["set"]]]
    return rows


def main():
    argv = sys.argv[1:]
    test = "--test" in argv
    out = Path("analysis/ablation/results/ablation_scores.csv")
    if "--out" in argv:
        i = argv.index("--out")
        out = Path(argv[i + 1])
        del argv[i:i + 2]
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        sys.exit(__doc__)
    rows = load(paths, test)

    # Source page = document: the page ID is shared by every version and format, and
    # AnyText and DiffSTE share their imgNN pages, so one ID is one document
    ids = sorted({r["page"] for r in rows})
    doc_of = {p: i for i, p in enumerate(ids)}
    # One random 2-fold split of documents, the same for every cell (seed 42, from shared/score.py)
    fold_of_doc = sc.document_folds(np.arange(len(ids)))
    pages_of_doc_all = None  # built per cell below

    cells = defaultdict(list)
    for r in rows:
        cells[(r["detector"], r["weights"], r["size"], r["format"], r["set"])].append(r)

    header = ["detector", "weights", "size", "format", "set", "version", "n_tampered", "n_untouched"]
    for m in KEEP:
        header += [m, m + "_lo", m + "_hi"]
    header += ["cutoff_fold0", "cutoff_fold1"]
    out_rows = []
    for key in sorted(cells):
        crow = cells[key]
        clean = [r for r in crow if r["version"] == "untouched"]
        for ver in sorted({r["version"] for r in crow} - {"untouched"}):
            tamp = [r for r in crow if r["version"] == ver]
            pg = tamp + clean
            n = len(pg)
            tampered = np.array([r["version"] != "untouched" for r in pg])
            doc = np.array([doc_of[r["page"]] for r in pg])
            v = {"top": np.array([float(r["max_value"]) for r in pg]),
                 "predA": np.array([r["flagged_any"] == "1.0" for r in pg]),
                 # found = a region overlaps the edit box (`detected`); one edit per page here
                 "found": np.array([r["detected"] == "1.0" for r in pg], dtype=int) * tampered,
                 "edits": tampered.astype(int),
                 "false": np.zeros(n, dtype=int)}
            # Rule B: cut-off chosen on the other half only (shared/score.py)
            fold = fold_of_doc[doc]
            v["predB"], cut = sc.crossfit_rule_b(v["top"], tampered, np.ones(n, bool), fold)
            pos, neg = tampered, ~tampered
            point = sc.compute_metrics(v, np.arange(n), pos, neg)
            # Bootstrap documents (pages with the same ID stay together), seed 42
            local_docs = sorted(set(doc))
            by_doc = [np.flatnonzero(doc == d) for d in local_docs]
            rng = np.random.RandomState(sc.SEED)
            draws = {m: np.full(sc.N_BOOT, np.nan) for m in KEEP}
            for b in range(sc.N_BOOT):
                m_b = sc.compute_metrics(v, sc.resample_indices(by_doc, rng), pos, neg)
                for m in KEEP:
                    draws[m][b] = m_b[m]
            line = list(key[:5]) + [ver, int(pos.sum()), int(neg.sum())]
            for m in KEEP:
                lo, hi = sc.interval(draws[m])
                line += [round(float(point[m]), 4), round(lo, 4), round(hi, 4)]
            line += [round(c, 6) for c in cut]
            out_rows.append(line)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out_rows)
    print(f"wrote {out} ({len(out_rows)} rows, {len(ids)} documents, seed {sc.SEED}, {sc.N_BOOT} resamples)")


if __name__ == "__main__":
    main()
