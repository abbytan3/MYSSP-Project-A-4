"""T11: shared scoring script (docs/PLAN.md section 4). One script for every detector.

Reads, for one detector:
    <out-root>/<detector>/<size>/<page_id>.npz   compact outputs (shared/output_format.py)
    <out-root>/<detector>/<size>/run_info.json   must hold the required fields
    sample.csv                                    anonymous sample (shared/sample.py)
Writes into --results-dir (numbers and anonymous IDs only):
    scores.csv        every metric with its 95% range and counts
    differences.csv   AI minus handcrafted, and shrink512 minus full, with ranges
    score_info.json   seed, bootstrap size, cut-offs, missing pages, run_info copies

Run:
    python3 shared/score.py --detector adcd --sample <sample.csv> \
        --out-root <folder holding out/...> --results-dir results/adcd
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

# Make output_format importable whether run directly or through pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import output_format as of  # noqa: E402

# Seed for the document split and the bootstrap (PLAN.md section 4).
SEED = 42
# Number of bootstrap resamples of documents (PLAN.md section 4).
N_BOOT = 1000
# Touching flagged pixels, including diagonals, form one region (8-connectivity).
EIGHT_CONNECTED = np.ones((3, 3), dtype=int)

# Metrics that get a 95% range, in output order.
METRICS = [
    "precision_A", "recall_A", "f1_A", "false_alarm_A",
    "precision_B", "recall_B", "f1_B", "false_alarm_B",
    "auc", "loc_precision", "loc_recall", "loc_f1",
]
# Metrics compared in differences.csv.
DIFF_METRICS = ["recall_A", "recall_B", "auc", "loc_recall"]


# ---------------------------------------------------------------- reading

def load_sample(path):
    """Read sample.csv into arrays, one entry per page, in file order."""
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Turn the anonymous document IDs into numbers 0..D-1.
    doc_numbers = {d: i for i, d in enumerate(sorted({r["doc_id"] for r in rows}))}
    boxes = [json.loads(r["boxes"]) for r in rows]
    group = np.array([r["group"] for r in rows])
    # Sanity: untouched pages have no boxes, edited pages have at least one.
    for g, b, r in zip(group, boxes, rows):
        if g not in ("ai", "handcrafted", "authentic"):
            raise ValueError(f"unknown group {g!r} for {r['page_id']}")
        if (g == "authentic") != (len(b) == 0):
            raise ValueError(f"{r['page_id']}: group {g} but {len(b)} boxes")
    return {
        "page_id": [r["page_id"] for r in rows],
        "doc": np.array([doc_numbers[r["doc_id"]] for r in rows]),
        "tampered": group != "authentic",
        "group": group,
        "detail": np.array([r["editor_or_method"] for r in rows]),
        "capture": np.array([r["capture"] for r in rows]),
        "boxes": boxes,
    }


def analyse_page(flag, boxes):
    """Return (edits found, false regions) for one page.

    flag: 2-D bool array of flagged pixels. boxes: [[x1, y1, x2, y2], ...] in page pixels.
    An edit is found if at least one region overlaps its box. A false region is a
    region that overlaps no box. Box edges are rounded outwards, end exclusive,
    clipped to the page (same rule as output_format.boxes_to_heatmap).
    """
    # Label the regions with 8-connectivity.
    labels, n_regions = ndimage.label(flag, structure=EIGHT_CONNECTED)
    if n_regions == 0:
        return 0, 0
    height, width = flag.shape
    touched, found = set(), 0
    for x1, y1, x2, y2 in boxes:
        left, top = max(int(np.floor(x1)), 0), max(int(np.floor(y1)), 0)
        right, bottom = min(int(np.ceil(x2)), width), min(int(np.ceil(y2)), height)
        if right <= left or bottom <= top:
            continue
        # Region numbers that fall inside this box (0 is background).
        ids = np.unique(labels[top:bottom, left:right])
        ids = ids[ids > 0]
        if ids.size:
            found += 1
            touched.update(ids.tolist())
    return found, n_regions - len(touched)


def load_variant(out_root, detector, size, sample):
    """Read every page's output for one detector at one size.

    Returns (arrays aligned with sample.csv, run_info). Pages with no .npz are
    marked present=False and left out of every metric (they are counted in score_info.json).
    """
    folder = of.output_dir(out_root, detector, size)
    # Fail early if run_info.json is missing a required field.
    info = of.read_run_info(folder / "run_info.json")
    n = len(sample["page_id"])
    v = {
        "present": np.zeros(n, dtype=bool), "top": np.zeros(n), "predA": np.zeros(n, dtype=bool),
        "found": np.zeros(n, dtype=int), "false": np.zeros(n, dtype=int),
        "edits": np.array([len(b) for b in sample["boxes"]]),
    }
    for i, page_id in enumerate(sample["page_id"]):
        path = folder / f"{page_id}.npz"
        if not path.exists():
            continue
        out = of.load_page_output(path)
        v["present"][i] = True
        v["top"][i] = out["top"]
        # Rule A: tampered if any pixel is flagged.
        v["predA"][i] = bool(out["flag"].any())
        v["found"][i], v["false"][i] = analyse_page(out["flag"], sample["boxes"][i])
    return v, info


# ---------------------------------------------------------------- rule B (cross-fitting)

def best_cutoff(top, pos):
    """Cut-off (predict tampered when top >= cut-off) that maximises F1 on these pages.

    Candidates are the distinct top scores; ties go to the lowest cut-off.
    """
    cands = np.unique(top)
    above = top[None, :] >= cands[:, None]
    tp = (above & pos[None, :]).sum(axis=1)
    fp = (above & ~pos[None, :]).sum(axis=1)
    fn = pos.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    # argmax returns the first (lowest) cut-off among equal F1 values.
    return float(cands[int(np.argmax(f1))])


def document_folds(doc, seed=SEED):
    """Split documents at random into two halves; return the half (0 or 1) of every page."""
    n_docs = int(doc.max()) + 1
    order = np.random.RandomState(seed).permutation(n_docs)
    half_of_doc = np.zeros(n_docs, dtype=int)
    half_of_doc[order[n_docs // 2:]] = 1
    return half_of_doc[doc]


def crossfit_rule_b(top, tampered, present, fold):
    """Rule B with 2-fold cross-fitting. Returns (predictions, cut-offs).

    The cut-off applied to fold k is chosen on the other fold only, so test pages
    never choose their own cut-off. cutoffs[k] is the cut-off used on fold k.
    """
    pred = np.zeros(len(top), dtype=bool)
    cutoffs = [np.nan, np.nan]
    for k in (0, 1):
        train = present & (fold != k)
        test = present & (fold == k)
        if tampered[train].sum() == 0 or (~tampered[train]).sum() == 0:
            raise ValueError("a half has no tampered or no untouched pages; cannot choose a cut-off")
        cutoffs[k] = best_cutoff(top[train], tampered[train])
        pred[test] = top[test] >= cutoffs[k]
    return pred, cutoffs


# ---------------------------------------------------------------- metrics

def _div(a, b):
    # Division that gives NaN (not an error) when the denominator is 0.
    return a / b if b > 0 else np.nan


def auc(pos_scores, neg_scores):
    """Area under the ROC curve from the pages' top scores (ties count half)."""
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(np.concatenate([pos_scores, neg_scores]))
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_metrics(v, idx, pos, neg):
    """All metrics for the pages in idx (repeats allowed, for the bootstrap).

    pos / neg: bool arrays over all pages marking this subset's edited and untouched pages.
    """
    p, n = pos[idx], neg[idx]
    out = {}
    for rule, pred in (("A", v["predA"][idx]), ("B", v["predB"][idx])):
        tp, fp, fn = (pred & p).sum(), (pred & n).sum(), (~pred & p).sum()
        out[f"precision_{rule}"] = _div(tp, tp + fp)
        out[f"recall_{rule}"] = _div(tp, tp + fn)
        out[f"f1_{rule}"] = _div(2 * tp, 2 * tp + fp + fn)
        out[f"false_alarm_{rule}"] = _div(fp, n.sum())
    top = v["top"][idx]
    out["auc"] = auc(top[p], top[n])
    # Location level: found edits on edited pages, false regions on any page.
    found, edits = v["found"][idx][p].sum(), v["edits"][idx][p].sum()
    false = v["false"][idx][p | n].sum()
    out["loc_precision"] = _div(found, found + false)
    out["loc_recall"] = _div(found, edits)
    out["loc_f1"] = _div(2 * found, found + false + edits)
    # Counts (no range): reported next to every row.
    out["_counts"] = (int(p.sum()), int(n.sum()), int(edits), int(found), int(false))
    return out


def subset_masks(sample, present):
    """Subsets scored: (breakdown, name) -> (edited-page mask, untouched-page mask)."""
    t = sample["tampered"] & present
    u = ~sample["tampered"] & present
    subsets = {("all", "all"): (t, u)}
    # AI vs handcrafted: edited pages of the group against all untouched pages.
    for g in ("ai", "handcrafted"):
        subsets[("group", g)] = (t & (sample["group"] == g), u)
    # Editor / method: edited pages of that editor or method against all untouched pages.
    for d in sorted(set(sample["detail"][sample["tampered"]])):
        subsets[("editor_or_method", d)] = (t & (sample["detail"] == d), u)
    # Capture condition: both edited and untouched pages with that condition.
    for c in sorted(set(sample["capture"])):
        subsets[("capture", c)] = (t & (sample["capture"] == c), u & (sample["capture"] == c))
    return subsets


def resample_indices(pages_of_doc, rng):
    """Draw documents with replacement and return the indices of all their pages.

    Pages of the same document always come together (PLAN.md section 4).
    """
    draw = rng.randint(0, len(pages_of_doc), size=len(pages_of_doc))
    return np.concatenate([pages_of_doc[d] for d in draw])


def bootstrap(sample, variants, n_boot=N_BOOT, seed=SEED):
    """Point estimates and bootstrap draws for every variant, subset and metric.

    Documents are resampled with replacement; all pages of a drawn document come
    with it. The same draws are used for every variant and subset, so differences
    between them have honest paired ranges.
    Returns {(size, breakdown, name): {metric: (point, draws array, counts)}}.
    """
    n_pages = len(sample["page_id"])
    n_docs = int(sample["doc"].max()) + 1
    pages_of_doc = [np.flatnonzero(sample["doc"] == d) for d in range(n_docs)]
    subsets = {s: subset_masks(sample, v["present"]) for s, v in variants.items()}
    result = {}
    # Point estimates on the real sample.
    everything = np.arange(n_pages)
    for size, v in variants.items():
        for key, (pos, neg) in subsets[size].items():
            m = compute_metrics(v, everything, pos, neg)
            result[(size, *key)] = {"point": m, "draws": {k: np.full(n_boot, np.nan) for k in METRICS}}
    # Bootstrap resamples.
    rng = np.random.RandomState(seed)
    for b in range(n_boot):
        idx = resample_indices(pages_of_doc, rng)
        for size, v in variants.items():
            for key, (pos, neg) in subsets[size].items():
                m = compute_metrics(v, idx, pos, neg)
                for k in METRICS:
                    result[(size, *key)]["draws"][k][b] = m[k]
    return result


def interval(draws):
    """95% range: 2.5th and 97.5th percentile of the bootstrap draws (NaN if none usable)."""
    ok = draws[~np.isnan(draws)]
    if ok.size == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(ok, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------- output tables

def score_rows(detector, result):
    """One row per (size, breakdown, subset, metric) with value, range and counts."""
    rows = []
    for (size, breakdown, name), entry in result.items():
        n_pos, n_neg, edits, found, false = entry["point"]["_counts"]
        for metric in METRICS:
            lo, hi = interval(entry["draws"][metric])
            rows.append({
                "detector": detector, "size": size, "breakdown": breakdown, "subset": name,
                "metric": metric, "value": entry["point"][metric], "ci_low": lo, "ci_high": hi,
                "n_tampered": n_pos, "n_untouched": n_neg, "n_edits": edits,
                "n_found": found, "n_false_regions": false,
            })
    return rows


def difference_rows(detector, result, sizes):
    """AI minus handcrafted (per size), and shrink512 minus full (per subset), with paired ranges."""
    rows = []

    def add(comparison, size, subset, a, b):
        for metric in DIFF_METRICS:
            diff = a["point"][metric] - b["point"][metric]
            lo, hi = interval(a["draws"][metric] - b["draws"][metric])
            # A difference is "clear" only if its range excludes 0.
            clear = bool(not np.isnan(lo) and (lo > 0 or hi < 0))
            rows.append({"detector": detector, "comparison": comparison, "size": size, "subset": subset,
                         "metric": metric, "difference": diff, "ci_low": lo, "ci_high": hi, "clear": clear})

    for size in sizes:
        add("ai minus handcrafted", size, "group", result[(size, "group", "ai")], result[(size, "group", "handcrafted")])
    if "shrink512" in sizes and "full" in sizes:
        for (size, breakdown, name) in result:
            if size == "shrink512":
                add("shrink512 minus full", "both", f"{breakdown}:{name}", result[(size, breakdown, name)], result[("full", breakdown, name)])
    return rows


def write_table(path, rows):
    # Write a list of dicts as a UTF-8 CSV; NaN becomes an empty cell, floats keep 6 digits.
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if isinstance(x, float) and np.isnan(x) else (f"{x:.6f}" if isinstance(x, float) else x))
                             for k, x in row.items()})


# ---------------------------------------------------------------- main

def score_detector(detector, sample_path, out_root, results_dir, sizes=("shrink512", "full"), n_boot=N_BOOT, seed=SEED):
    """Score one detector end to end and write the three result files. Returns the score rows."""
    sample = load_sample(sample_path)
    fold = document_folds(sample["doc"], seed)
    variants, infos, cutoffs, missing = {}, {}, {}, {}
    for size in sizes:
        folder = of.output_dir(out_root, detector, size)
        if not folder.exists():
            print(f"{detector}/{size}: no output folder, skipped")
            continue
        v, info = load_variant(out_root, detector, size, sample)
        # Rule B: cut-off chosen by 2-fold cross-fitting over documents.
        v["predB"], cutoffs[size] = crossfit_rule_b(v["top"], sample["tampered"], v["present"], fold)
        variants[size], infos[size] = v, info
        missing[size] = [p for p, ok in zip(sample["page_id"], v["present"]) if not ok]
    if not variants:
        raise SystemExit("no sizes found to score")
    result = bootstrap(sample, variants, n_boot, seed)
    rows = score_rows(detector, result)
    diffs = difference_rows(detector, result, list(variants))
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    write_table(results_dir / "scores.csv", rows)
    write_table(results_dir / "differences.csv", diffs)
    info = {
        "detector": detector, "seed": seed, "n_bootstrap": n_boot, "flag_threshold": of.FLAG_THRESHOLD,
        "pages_in_sample": len(sample["page_id"]), "documents": int(sample["doc"].max()) + 1,
        "sizes_scored": list(variants), "rule_b_cutoffs_by_fold": cutoffs,
        "pages_missing": {s: {"count": len(m), "page_ids": m} for s, m in missing.items()},
        "run_info": infos,
    }
    with open(results_dir / "score_info.json", "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--sizes", nargs="+", default=["shrink512", "full"])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    rows = score_detector(args.detector, args.sample, args.out_root, args.results_dir, tuple(args.sizes), args.n_boot)
    # Print the headline rows: all edited pages, rule B and location level.
    for r in rows:
        if r["breakdown"] == "all" and r["metric"] in ("recall_B", "false_alarm_B", "auc", "loc_recall", "loc_precision"):
            print(f"{r['size']:>9} {r['metric']:<14} {r['value']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]  "
                  f"edited={r['n_tampered']} untouched={r['n_untouched']} edits={r['n_edits']}")


if __name__ == "__main__":
    main()
