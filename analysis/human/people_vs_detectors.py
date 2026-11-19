"""T49: people vs detectors on the 500 real stage 2 pages (docs/PLAN.md section 5).

Compares what two raters (Abby, Nazhan) thought of each page with what each detector did
on the same page. Document level only: ranking (AUC from the page's top score), the
rule B decision next to its false-alarm rate, and the raters' own "edit found".
No pixel-level or location-level scores, no rater-agreement statistic.

Reads (paths are asked for when the script runs, never stored in code):
    the private joined ratings file (made by join_ratings.py, outside the repo)
    sample.csv (anonymous sample, 1,650 pages)
    the detector outputs (<out-root>/<detector>/<size>/<page_id>.npz)
Writes numbers only (no page IDs) into results/human/:
    rating_summary.csv, auc.csv, auc_differences.csv, detector_auc_by_people.csv,
    flag_rates.csv, edit_found_rates.csv, analysis_info.json

Another detector is added by naming it when the script asks "Detectors"; its outputs are read
from <out-root>/<detector>/ and checked against results/<detector>/ (no code change).

Direction: people score 5 = looks genuine, 1 = clearly edited, so people "edited-ness" is
6 - score (6 - mean for the combined score); the detector top score is already high = edited.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

# Repo root, found from this file's place (no laptop paths).
REPO = Path(__file__).resolve().parents[2]
# Reuse the shared scoring code. It is imported, not copied, and never edited here.
sys.path.insert(0, str(REPO / "shared"))
import output_format as of  # noqa: E402
import score as sc  # noqa: E402

# The two raters used (PLAN CHANGE: two raters, not three).
RATERS = ("abby", "nazhan")
# Page groups, as written in sample.csv.
GROUPS = ("ai", "handcrafted", "authentic")
# Sizes a detector run can have.
SIZES = ("shrink512", "full")
# A range from fewer pages than this is shaky; say so (fixed before any result).
SMALL_N = 30
# A rater "flags" a page with score 2 or lower; the mean of two raters flags at 2.5 or lower.
FLAG_RATER_MAX = 2
FLAG_MEAN_MAX = 2.5
# Tolerance when comparing with results/<detector>/scores.csv (that file keeps 6 decimals).
REPRO_TOL = 5e-7


# ---------------------------------------------------------------- reading

def read_joined(path):
    """Read the private joined ratings file into rows (dicts)."""
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_people(rows, sample):
    """Turn the joined rows into arrays over the real stage 2 pages, sorted by page_id.

    Checks: each rater rates each page once, both raters cover the same pages, scores are
    1 to 5, and the group agrees with sample.csv. Returns a dict of arrays plus the position
    of each page inside the 1,650-page sample.
    """
    by_rater = {r: {} for r in RATERS}
    for row in rows:
        who = row["rater"]
        if who not in by_rater:
            raise ValueError(f"unexpected rater {who!r}")
        if row["page_id"] in by_rater[who]:
            raise ValueError(f"{who} has two rows for {row['page_id']}")
        by_rater[who][row["page_id"]] = row
    page_ids = sorted(by_rater[RATERS[0]])
    if set(page_ids) != set(by_rater[RATERS[1]]):
        raise ValueError("the two raters did not rate the same pages")
    position = {p: i for i, p in enumerate(sample["page_id"])}
    missing = [p for p in page_ids if p not in position]
    if missing:
        raise ValueError(f"{len(missing)} rated pages are not in sample.csv")
    at = np.array([position[p] for p in page_ids])
    group = np.array([by_rater[RATERS[0]][p]["group"] for p in page_ids])
    # The group in the joined file must equal the group in sample.csv.
    if not np.array_equal(group, sample["group"][at]):
        raise ValueError("group in the joined file differs from sample.csv")
    for who in RATERS:
        if any(by_rater[who][p]["group"] != g for p, g in zip(page_ids, group)):
            raise ValueError("the two raters disagree on a page's group")
    people = {"page_ids": page_ids, "at": at, "group": group}
    for who in RATERS:
        score = np.array([int(by_rater[who][p]["score"]) for p in page_ids])
        if score.min() < 1 or score.max() > 5:
            raise ValueError(f"{who} has a score outside 1 to 5")
        people[who] = score
        # "True"/"False" text from the CSV becomes a real boolean.
        people[f"found_{who}"] = np.array([by_rater[who][p]["edit_found"] == "True" for p in page_ids])
    # Documents renumbered 0..D-1 among these pages, so pages of one document stay together.
    docs = sample["doc"][at]
    people["doc"] = np.unique(docs, return_inverse=True)[1]
    return people


# ---------------------------------------------------------------- definitions fixed before any result

def edited_ness(score):
    """People score to "higher = more edited": 6 - score (works for a mean of scores too)."""
    return 6.0 - np.asarray(score, dtype=float)


def combined_score(people):
    """A page's human score: the mean of the two raters."""
    return (people[RATERS[0]] + people[RATERS[1]]) / 2.0


def people_flagged(mean_score):
    """People flagged a page: mean score of 2.5 or lower. A 3 (unsure) does not count."""
    return np.asarray(mean_score) <= FLAG_MEAN_MAX


def people_missed(score_a, score_b):
    """People missed an edit: both raters gave a 5."""
    return (np.asarray(score_a) == 5) & (np.asarray(score_b) == 5)


# ---------------------------------------------------------------- bootstrap by document

def make_resamples(doc, n_boot, seed):
    """Pre-draw every bootstrap resample (page indices, whole documents, with replacement).

    The same draws are used for every number, so differences between two numbers get paired ranges.
    """
    pages_of_doc = [np.flatnonzero(doc == d) for d in range(int(doc.max()) + 1)]
    rng = np.random.RandomState(seed)
    return [sc.resample_indices(pages_of_doc, rng) for _ in range(n_boot)]


def evaluate(fn, n_pages, resamples):
    """Run one statistic on the real pages and on every resample. Returns (point, draws)."""
    point = fn(np.arange(n_pages))
    draws = np.array([fn(idx) for idx in resamples], dtype=float)
    return point, draws


def auc_fn(values, pos, neg):
    """Statistic: AUC of values for pos pages above neg pages (ties count half)."""
    def fn(idx):
        v, p, n = values[idx], pos[idx], neg[idx]
        return sc.auc(v[p], v[n])
    return fn


def mean_fn(values, mask):
    """Statistic: mean of values over the masked pages."""
    def fn(idx):
        m = mask[idx]
        return float(values[idx][m].mean()) if m.any() else np.nan
    return fn


def share_fn(cond, denom):
    """Statistic: share of the denominator pages that meet cond."""
    def fn(idx):
        d = denom[idx]
        return float((cond[idx] & d).sum() / d.sum()) if d.any() else np.nan
    return fn


def with_range(point, draws):
    """(value, low, high): the point estimate and its 95% range (2.5th to 97.5th percentile)."""
    lo, hi = sc.interval(draws)
    return point, lo, hi


# ---------------------------------------------------------------- the tables

def rating_summary(people, resamples):
    """Output 1: per group and per rater, n, mean score and share given a 5 (500 real pages only)."""
    n_pages = len(people["group"])
    mean = combined_score(people)
    # What each "rater" row measures. For the two raters together: mean of the two, and "both gave 5".
    who_scores = {RATERS[0]: people[RATERS[0]], RATERS[1]: people[RATERS[1]], "mean_of_two": mean}
    gave5 = {RATERS[0]: people[RATERS[0]] == 5, RATERS[1]: people[RATERS[1]] == 5,
             "mean_of_two": people_missed(people[RATERS[0]], people[RATERS[1]])}
    rows = []
    for group in GROUPS:
        mask = people["group"] == group
        for who in who_scores:
            m = with_range(*evaluate(mean_fn(who_scores[who], mask), n_pages, resamples))
            s = with_range(*evaluate(share_fn(gave5[who], mask), n_pages, resamples))
            rows.append({"group": group, "rater": who, "n_pages": int(mask.sum()),
                         "mean_score": m[0], "mean_low": m[1], "mean_high": m[2],
                         "share_5": s[0], "share_5_low": s[1], "share_5_high": s[2]})
    return rows


def comparisons(people):
    """The three edited-vs-authentic comparisons: name -> (edited mask, authentic mask)."""
    g = people["group"]
    auth = g == "authentic"
    return {"edited_vs_authentic": (g != "authentic", auth),
            "ai_vs_authentic": (g == "ai", auth),
            "handcrafted_vs_authentic": (g == "handcrafted", auth)}


def auc_tables(people, dets, resamples):
    """Output 2: AUC (and people-minus-detector differences) with 95% ranges and n.

    dets: {(detector, size): {"top": array aligned to the 500 pages, ...}}.
    Returns (auc rows, difference rows, draws kept for the differences).
    """
    n_pages = len(people["group"])
    scorers = [("people_mean", "people", "", "", edited_ness(combined_score(people))),
               ("people_abby", "people", "", "", edited_ness(people["abby"])),
               ("people_nazhan", "people", "", "", edited_ness(people["nazhan"]))]
    for (det, size), d in dets.items():
        scorers.append((f"{det}_{size}", "detector", det, size, d["top"]))
    rows, keep = [], {}
    for comp, (pos, neg) in comparisons(people).items():
        for name, source, det, size, values in scorers:
            point, draws = evaluate(auc_fn(values, pos, neg), n_pages, resamples)
            keep[(comp, name)] = (point, draws)
            v, lo, hi = with_range(point, draws)
            rows.append({"comparison": comp, "scorer": name, "source": source, "detector": det, "size": size,
                         "auc": v, "ci_low": lo, "ci_high": hi,
                         "n_edited": int(pos.sum()), "n_authentic": int(neg.sum()),
                         "small_n": bool(min(pos.sum(), neg.sum()) < SMALL_N)})
    # People (mean) minus each detector, using the same resamples. Clear only if the range excludes 0.
    diffs = []
    for comp in comparisons(people):
        for (det, size) in dets:
            a, b = keep[(comp, "people_mean")], keep[(comp, f"{det}_{size}")]
            lo, hi = sc.interval(a[1] - b[1])
            diffs.append({"comparison": comp, "detector": det, "size": size,
                          "people_mean_auc_minus_detector_auc": a[0] - b[0], "ci_low": lo, "ci_high": hi,
                          "clear": bool(not np.isnan(lo) and (lo > 0 or hi < 0))})
    return rows, diffs


def people_verdict_masks(people):
    """Edited pages split by what people did: flagged (mean <= 2.5), missed (both gave 5), in between."""
    edited = people["group"] != "authentic"
    mean = combined_score(people)
    flagged = edited & people_flagged(mean)
    missed = edited & people_missed(people["abby"], people["nazhan"])
    between = edited & ~flagged & ~missed
    return {"people_flagged": flagged, "people_missed": missed, "in_between": between}


def detector_auc_by_people(people, dets, resamples):
    """Output 3: each detector's AUC on the edits people missed / flagged / neither, vs all authentic pages."""
    n_pages = len(people["group"])
    auth = people["group"] == "authentic"
    verdicts = people_verdict_masks(people)
    scopes = {"edited": np.ones(n_pages, dtype=bool), "ai": people["group"] == "ai",
              "handcrafted": people["group"] == "handcrafted"}
    rows = []
    for (det, size), d in dets.items():
        for verdict, vmask in verdicts.items():
            for scope, smask in scopes.items():
                pos = vmask & smask
                v, lo, hi = with_range(*evaluate(auc_fn(d["top"], pos, auth), n_pages, resamples))
                rows.append({"detector": det, "size": size, "people": verdict, "edited_pages": scope,
                             "auc": v, "ci_low": lo, "ci_high": hi,
                             "n_edited": int(pos.sum()), "n_authentic": int(auth.sum()),
                             "small_n": bool(pos.sum() < SMALL_N)})
    return rows


def flag_rates(people, dets, resamples):
    """Share of each group's pages that people / rule B flagged. On the authentic group this is the false-alarm rate."""
    n_pages = len(people["group"])
    mean = combined_score(people)
    measures = {
        "people_flagged_mean_2.5_or_lower": people_flagged(mean),
        "people_flagged_abby_2_or_lower": people["abby"] <= FLAG_RATER_MAX,
        "people_flagged_nazhan_2_or_lower": people["nazhan"] <= FLAG_RATER_MAX,
        "people_both_gave_5": people_missed(people["abby"], people["nazhan"]),
    }
    for (det, size), d in dets.items():
        measures[f"rule_b_{det}_{size}"] = d["predB"]
    rows = []
    for name, cond in measures.items():
        for group in GROUPS:
            denom = people["group"] == group
            v, lo, hi = with_range(*evaluate(share_fn(cond, denom), n_pages, resamples))
            rows.append({"measure": name, "group": group, "share": v, "ci_low": lo, "ci_high": hi,
                         "n_pages": int(denom.sum()), "n_meeting": int((cond & denom).sum())})
    return rows


def edit_found_rates(people, resamples):
    """Output 4 (descriptive): of the edited pages a rater flagged (score 2 or lower), how often their click hit the edit."""
    n_pages = len(people["group"])
    scopes = {"edited": people["group"] != "authentic", "ai": people["group"] == "ai",
              "handcrafted": people["group"] == "handcrafted"}
    rows = []
    for who in RATERS:
        for scope, smask in scopes.items():
            flagged = smask & (people[who] <= FLAG_RATER_MAX)
            hit = people[f"found_{who}"]
            v, lo, hi = with_range(*evaluate(share_fn(hit, flagged), n_pages, resamples))
            rows.append({"rater": who, "edited_pages": scope, "edit_found_rate": v, "ci_low": lo, "ci_high": hi,
                         "n_flagged": int(flagged.sum()), "n_edit_found": int((hit & flagged).sum()),
                         "small_n": bool(flagged.sum() < SMALL_N)})
    return rows


# ---------------------------------------------------------------- detectors and the reproduction check

def read_reference(results_dir):
    """Read results/<detector>/scores.csv rows for all pages: {(size, metric): row}."""
    with open(Path(results_dir) / "scores.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(r["size"], r["metric"]): r for r in rows if r["breakdown"] == "all" and r["subset"] == "all"}


def check_reproduces(size, variant, sample, reference, ref_cutoffs, cutoffs):
    """Rule B (and every other point value) on all pages must equal the committed scores.csv.

    Returns a dict with the largest difference. Raises SystemExit with a clear message if it does not match.
    """
    n = len(sample["page_id"])
    pos = sample["tampered"] & variant["present"]
    neg = ~sample["tampered"] & variant["present"]
    got = sc.compute_metrics(variant, np.arange(n), pos, neg)
    worst = 0.0
    for metric in sc.METRICS:
        ref_text = reference[(size, metric)]["value"]
        ref_value = float(ref_text) if ref_text != "" else np.nan
        both_nan = np.isnan(ref_value) and np.isnan(got[metric])
        diff = 0.0 if both_nan else abs(got[metric] - ref_value)
        if np.isnan(diff) or diff > REPRO_TOL:
            raise SystemExit(f"STOP: {size} {metric} is {got[metric]:.6f}, results file says {ref_value:.6f}. "
                             "Rule B does not reproduce; tell Abby.")
        worst = max(worst, diff)
    # Page counts must match too (edited pages, untouched pages).
    first = reference[(size, "recall_B")]
    if (int(first["n_tampered"]), int(first["n_untouched"])) != (int(pos.sum()), int(neg.sum())):
        raise SystemExit(f"STOP: {size} page counts differ from the results file; tell Abby.")
    # The two fold cut-offs must be the same numbers as in score_info.json.
    if not np.allclose(cutoffs, ref_cutoffs, rtol=0, atol=1e-12):
        raise SystemExit(f"STOP: {size} rule B cut-offs {cutoffs} differ from score_info.json {ref_cutoffs}; tell Abby.")
    return {"largest_difference": worst, "metrics_compared": len(sc.METRICS),
            "recall_B": got["recall_B"], "false_alarm_B": got["false_alarm_B"], "cutoffs": list(cutoffs)}


def load_detector(detector, out_root, sample, fold, results_dir, people):
    """Load one detector at every size it has; rebuild rule B on all 1,650 pages; check it matches scores.csv.

    Returns {(detector, size): {"top", "predB"}} aligned to the 500 pages, plus a log for analysis_info.json.
    """
    reference = read_reference(results_dir)
    with open(Path(results_dir) / "score_info.json", encoding="utf-8") as handle:
        ref_info = json.load(handle)
    dets, log = {}, {}
    for size in SIZES:
        if not of.output_dir(out_root, detector, size).exists():
            print(f"{detector}/{size}: no output folder, skipped")
            continue
        variant, info = sc.load_variant(out_root, detector, size, sample)
        # Same cross-fitting and same document halves as shared/score.py: the cut-offs come from all 1,650 pages.
        variant["predB"], cutoffs = sc.crossfit_rule_b(variant["top"], sample["tampered"], variant["present"], fold)
        log[f"{detector}/{size}"] = check_reproduces(size, variant, sample, reference,
                                                     ref_info["rule_b_cutoffs_by_fold"][size], cutoffs)
        at = people["at"]
        if not variant["present"][at].all():
            raise SystemExit(f"STOP: {detector}/{size} has no output for some of the 500 pages; tell Abby.")
        dets[(detector, size)] = {"top": variant["top"][at], "predB": variant["predB"][at]}
    return dets, log


# ---------------------------------------------------------------- main

def run_all(people, dets, n_boot=sc.N_BOOT, seed=sc.SEED):
    """Build every table from the people arrays and detector arrays. Returns {file name: rows}."""
    resamples = make_resamples(people["doc"], n_boot, seed)
    auc_rows, diff_rows = auc_tables(people, dets, resamples)
    return {
        "rating_summary.csv": rating_summary(people, resamples),
        "auc.csv": auc_rows,
        "auc_differences.csv": diff_rows,
        "detector_auc_by_people.csv": detector_auc_by_people(people, dets, resamples),
        "flag_rates.csv": flag_rates(people, dets, resamples),
        "edit_found_rates.csv": edit_found_rates(people, resamples),
    }


def ask(prompt):
    """Ask for a path or value at run time (nothing about this laptop is written in code)."""
    return input(prompt).strip().strip("'\"")


def main():
    joined_path = Path(ask("Private joined ratings file (t49_joined_ratings.csv): "))
    sample_path = Path(ask("Path to the anonymous sample.csv: "))
    out_root = Path(ask("Folder holding one sub-folder per detector (e.g. a4_outputs): "))
    detectors = [d for d in ask("Detectors, comma separated [adcd]: ").replace(",", " ").split()] or ["adcd"]
    results_dir = REPO / "results" / "human"

    sample = sc.load_sample(sample_path)
    people = build_people(read_joined(joined_path), sample)
    counts = {g: int((people["group"] == g).sum()) for g in GROUPS}
    print(f"pages {len(people['page_ids'])}; by group {counts}; documents {int(people['doc'].max()) + 1}")

    # The same two document halves as shared/score.py, so rule B decisions are the main run's.
    fold = sc.document_folds(sample["doc"], sc.SEED)
    dets, repro = {}, {}
    for det in detectors:
        d, log = load_detector(det, out_root, sample, fold, REPO / "results" / det, people)
        dets.update(d)
        repro.update(log)
    for name, r in repro.items():
        print(f"{name}: matches results/{name.split('/')[0]}/scores.csv "
              f"(largest difference {r['largest_difference']:.1e}; rule B recall {r['recall_B']:.3f}, "
              f"false-alarm rate {r['false_alarm_B']:.3f})")

    tables = run_all(people, dets)
    results_dir.mkdir(parents=True, exist_ok=True)
    for fname, rows in tables.items():
        sc.write_table(results_dir / fname, rows)
    info = {"task": "T49", "raters": list(RATERS), "n_pages": len(people["page_ids"]), "pages_by_group": counts,
            "documents": int(people["doc"].max()) + 1, "seed": sc.SEED, "n_bootstrap": sc.N_BOOT,
            "small_n_below": SMALL_N, "people_flagged": f"mean score <= {FLAG_MEAN_MAX}; one rater <= {FLAG_RATER_MAX}",
            "people_missed": "both raters gave 5", "direction": "people edited-ness = 6 - score; detector top score unchanged",
            "detectors": sorted({f"{d}/{s}" for d, s in dets}), "rule_b_reproduction_check": repro,
            "numpy": np.__version__}
    with open(results_dir / "analysis_info.json", "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, sort_keys=True)
        handle.write("\n")

    # Print the headline ranking rows.
    print("\nAUC, edited vs authentic (95% range, n edited / n authentic):")
    for r in tables["auc.csv"]:
        if r["comparison"] == "edited_vs_authentic":
            print(f"  {r['scorer']:<22} {r['auc']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]  n={r['n_edited']}/{r['n_authentic']}")
    print(f"\nwrote {len(tables)} tables and analysis_info.json to results/human/")


if __name__ == "__main__":
    main()
