"""T20 Week 7 ablation: rescore the safe per_page.csv with rule A and "edit found".

Rule B (2-fold cross-fitted cut-off) is NOT here yet: it waits for the shared
scoring functions from T11. False-region counts cannot be recomputed (heat maps
were not kept), so they are left out on purpose.

Usage: python3 analysis/ablation/score_ablation.py PER_PAGE_CSV [PER_PAGE_CSV ...] [--out FILE]
Add --test to run on only 3 pages per set. DAF SRNet rows are skipped (PLAN.md section 3).
"""
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# Fixed seed and number of bootstrap resamples (PLAN.md section 4)
SEED = 42
N_BOOT = 1000
# Anonymous page IDs only: page_0001 or img01 style
SAFE_ID = re.compile(r"^(page_\d{4}|img\d{2})$")
# Anything that looks like a real file name or doc ID must stop the run
UNSAFE = re.compile(r"_v\d+__[0-9a-f]{10}|\.(png|jpe?g|pdf|tiff?)\b", re.I)
# Columns that identify a row (grouping keys)
GROUP = ["detector", "weights", "size", "set", "version", "format"]


def load(paths, test=False):
    # Read every file as UTF-8, and refuse it if any cell looks like a real name
    rows = []
    for path in paths:
        with open(path, encoding="utf-8", newline="") as f:
            rows += list(csv.DictReader(f))
    # SRNet weight is not used in the project (PLAN.md section 3)
    rows = [r for r in rows if r["weights"] != "srnet"]
    for r in rows:
        if not SAFE_ID.match(r["page"]):
            sys.exit("Unsafe page id in input: stop and check the file.")
        if any(UNSAFE.search(v) for v in r.values()):
            sys.exit("A cell looks like a real file name: stop and check the file.")
    if test:
        keep = defaultdict(list)
        for r in rows:
            if r["page"] not in keep[r["set"]] and len(keep[r["set"]]) < 3:
                keep[r["set"]].append(r["page"])
        rows = [r for r in rows if r["page"] in keep[r["set"]]]
    return rows


def metrics(tp, fn, fp, tn):
    # Precision, recall, F1 and false-alarm rate from the four counts (None if undefined)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    far = fp / (fp + tn) if fp + tn else None
    return prec, rec, f1, far


def score_cell(tampered, clean):
    # tampered / clean are lists of rows. Rule A: tampered if any pixel flagged.
    tp = sum(r["flagged_any"] == "1.0" for r in tampered)
    fp = sum(r["flagged_any"] == "1.0" for r in clean)
    prec, rec, f1, far = metrics(tp, len(tampered) - tp, fp, len(clean) - fp)
    # Edit found: a region overlaps the edit box (the file's `detected` column)
    found = sum(r["detected"] == "1.0" for r in tampered) / len(tampered) if tampered else None
    return {"A_precision": prec, "A_recall": rec, "A_f1": f1,
            "A_false_alarm": far, "edit_found": found}


def table(rows):
    # Group rows into (size, format, set, version) cells; untouched pages of the
    # same size, format and set are the clean pages for that cell
    tamp = defaultdict(list)
    clean = defaultdict(list)
    for r in rows:
        key = (r["detector"], r["weights"], r["size"], r["format"], r["set"])
        (clean if r["version"] == "untouched" else tamp)[key + (r["version"],)].append(r)
    out = {}
    for k, t in tamp.items():
        # clean pages for this cell share its detector, weights, size, format and set
        out[k] = score_cell(t, clean.get(k[:5] + ("untouched",), []))
    return out


def main():
    argv = sys.argv[1:]
    test = "--test" in argv
    out = Path("analysis/ablation/results/ablation_rule_a_found.csv")
    if "--out" in argv:
        i = argv.index("--out")
        out = Path(argv[i + 1])
        del argv[i:i + 2]
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    rows = load(args, test)
    point = table(rows)

    # Bootstrap: resample pages (one page = one document here, since the safe
    # file has no document ID); all rows of a page stay together
    by_page = defaultdict(list)
    for r in rows:
        by_page[(r["set"], r["page"])].append(r)
    keys = sorted(by_page)
    rng = random.Random(SEED)
    draws = defaultdict(lambda: defaultdict(list))
    for _ in range(N_BOOT):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in by_page[k]]
        for cell, m in table(sample).items():
            for name, v in m.items():
                if v is not None:
                    draws[cell][name].append(v)

    def ci(cell, name):
        # 95% range = 2.5th and 97.5th percentile of the resamples
        d = sorted(draws[cell][name])
        if not d:
            return "", ""
        return d[int(0.025 * (len(d) - 1))], d[int(0.975 * (len(d) - 1))]

    out.parent.mkdir(parents=True, exist_ok=True)
    names = ["A_precision", "A_recall", "A_f1", "A_false_alarm", "edit_found"]
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "weights", "size", "format", "set", "version", "n_tampered", "n_untouched"]
                   + [c for n in names for c in (n, n + "_lo", n + "_hi")])
        for cell in sorted(point):
            det, wts, size, fmt, st, ver = cell
            same = [r for r in rows if (r["detector"], r["weights"], r["size"], r["format"], r["set"]) == (det, wts, size, fmt, st)]
            nt = sum(r["version"] == ver for r in same)
            nu = sum(r["version"] == "untouched" for r in same)
            vals = []
            for n in names:
                v = point[cell][n]
                lo, hi = ci(cell, n)
                vals += ["" if v is None else round(v, 4), lo if lo == "" else round(lo, 4), hi if hi == "" else round(hi, 4)]
            w.writerow([det, wts, size, fmt, st, ver, nt, nu] + vals)
    print(f"wrote {out} ({len(point)} cells, {len(keys)} pages, seed {SEED}, {N_BOOT} resamples)")


if __name__ == "__main__":
    main()
