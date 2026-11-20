"""Cross-detector results tables, copied by script from the score files (AGENTS.md "Honest results").

Reads only:
    results/<detector>/scores.csv and differences.csv   (adcd, dtd, daf_anytext, daf_diffste)
    results/human/*.csv                                   (T49 tables)
Never opens a page, a sample map or a rating file. Every number is copied as the text the source
file holds (no rounding, no arithmetic). A value that a source file does not hold is written as
"missing"; the script counts these and the CSVs can be searched for the word.

Writes to results/combined/tables/:
    detector_table.csv         one row per detector and size (AUC overall / AI / handcrafted, rule B
                               false-alarm rate, location recall, false regions)
    ai_minus_handcrafted.csv   one row per detector and size, every metric in differences.csv
    size_difference.csv        one row per detector, direction "shrink512 minus full" as stored
    people_vs_detectors.csv    AUC of people and detectors vs authentic pages, plus the stored
                               people-minus-detector differences
    human_grade_breakdown.csv  results/human tables on pages by human grade, with a "shaky" flag

Run from the repo root:
    python3 analysis/combined/make_tables.py
"""

import argparse
import csv
from pathlib import Path

# Repo root, found from this file's place: <root>/analysis/combined/make_tables.py.
REPO = Path(__file__).resolve().parents[2]

# Detectors and sizes in the order the tables list them.
DETECTORS = ["adcd", "dtd", "daf_anytext", "daf_diffste"]
SIZES = ["shrink512", "full"]
# Metrics held in differences.csv, in table order.
DIFF_METRICS = ["auc", "recall_A", "recall_B", "loc_recall"]
# A group with fewer pages than this is flagged "shaky" (same cut-off as results/human small_n).
SHAKY_BELOW = 30
MISSING = "missing"


def read_csv(path):
    """Read a CSV as text (UTF-8); a missing file gives an empty list."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, header, rows):
    """Write rows (lists) under a header, UTF-8, return the number of data rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return len(rows)


class Tally:
    """Counts the cells written as "missing" and remembers which they are."""

    def __init__(self):
        self.items = []

    def get(self, row, column, label):
        """Return row[column] as text, or "missing" (and remember it) if absent or empty."""
        if row is None or row.get(column, "") == "":
            self.items.append(label)
            return MISSING
        return row[column]


def load_scores(root):
    """{(detector, size, breakdown, subset, metric): row} from every results/<det>/scores.csv."""
    out = {}
    for det in DETECTORS:
        for r in read_csv(Path(root) / det / "scores.csv"):
            out[(det, r["size"], r["breakdown"], r["subset"], r["metric"])] = r
    return out


def load_diffs(root):
    """{(detector, comparison, size, subset, metric): row} from every results/<det>/differences.csv."""
    out = {}
    for det in DETECTORS:
        for r in read_csv(Path(root) / det / "differences.csv"):
            out[(det, r["comparison"], r["size"], r["subset"], r["metric"])] = r
    return out


def detector_table(root, tally):
    """(a) one row per detector and size from scores.csv (rows breakdown/subset as named)."""
    scores = load_scores(root)
    # (column prefix, breakdown, subset, metric)
    wanted = [
        ("auc_overall", "all", "all", "auc"),
        ("auc_ai", "group", "ai", "auc"),
        ("auc_handcrafted", "group", "handcrafted", "auc"),
        ("rule_B_false_alarm_rate", "all", "all", "false_alarm_B"),
        ("location_recall", "all", "all", "loc_recall"),
    ]
    header = ["detector", "size"]
    for prefix, *_ in wanted:
        header += [prefix, prefix + "_ci_low", prefix + "_ci_high"]
    header += ["false_regions_count", "n_tampered_pages", "n_untouched_pages"]
    rows = []
    for det in DETECTORS:
        for size in SIZES:
            row = [det, size]
            for prefix, breakdown, subset, metric in wanted:
                r = scores.get((det, size, breakdown, subset, metric))
                label = f"{det} {size} {prefix}"
                row += [tally.get(r, "value", label), tally.get(r, "ci_low", label + " low"),
                        tally.get(r, "ci_high", label + " high")]
            # False regions, page counts: taken from the "all pages, location recall" row.
            r = scores.get((det, size, "all", "all", "loc_recall"))
            row += [tally.get(r, "n_false_regions", f"{det} {size} false regions"),
                    tally.get(r, "n_tampered", f"{det} {size} n_tampered"),
                    tally.get(r, "n_untouched", f"{det} {size} n_untouched")]
            rows.append(row)
    return header, rows


def diff_table(root, tally, comparison, sizes, subset, direction):
    """Shared builder for (b) and (c): the stored differences with ranges and the file's clear flag."""
    diffs = load_diffs(root)
    header = ["detector", "size", "comparison"]
    for metric in DIFF_METRICS:
        name = f"{direction}__{metric}"
        header += [name, name + "_ci_low", name + "_ci_high", name + "_range_excludes_0_per_file"]
    rows = []
    for det in DETECTORS:
        for size in sizes:
            row = [det, size, comparison]
            for metric in DIFF_METRICS:
                r = diffs.get((det, comparison, size, subset, metric))
                label = f"{det} {size} {comparison} {metric}"
                row += [tally.get(r, "difference", label), tally.get(r, "ci_low", label + " low"),
                        tally.get(r, "ci_high", label + " high"), tally.get(r, "clear", label + " clear")]
            rows.append(row)
    return header, rows


def ai_minus_handcrafted(root, tally):
    """(b) AI minus handcrafted, one row per detector and size (differences.csv subset "group")."""
    return diff_table(root, tally, "ai minus handcrafted", SIZES, "group", "ai_minus_handcrafted")


def size_difference(root, tally):
    """(c) shrink512 minus full, one row per detector (the file stores it with size "both")."""
    return diff_table(root, tally, "shrink512 minus full", ["both"], "all:all", "shrink512_minus_full")


def people_vs_detectors(human):
    """(d) auc.csv (people and detectors vs authentic) plus auc_differences.csv, as stored."""
    header = ["row_type", "comparison", "scorer", "source", "detector", "size", "measure", "estimate",
              "ci_low", "ci_high", "range_excludes_0_per_file", "n_edited_group", "n_authentic", "small_n_per_file"]
    rows = []
    for r in read_csv(Path(human) / "auc.csv"):
        rows.append(["auc", r["comparison"], r["scorer"], r["source"], r["detector"], r["size"], "auc",
                     r["auc"], r["ci_low"], r["ci_high"], "", r["n_edited"], r["n_authentic"], r["small_n"]])
    for r in read_csv(Path(human) / "auc_differences.csv"):
        rows.append(["difference", r["comparison"], "people_mean minus detector", "detector", r["detector"],
                     r["size"], "people_mean_auc_minus_detector_auc",
                     r["people_mean_auc_minus_detector_auc"], r["ci_low"], r["ci_high"], r["clear"], "", "", ""])
    return header, rows


def shaky(n):
    """"yes" when the group has fewer than SHAKY_BELOW pages (or its size is unknown), else "no"."""
    try:
        return "yes" if int(n) < SHAKY_BELOW else "no"
    except (TypeError, ValueError):
        return "yes"


def split_rule_b_measure(measure):
    """"rule_b_daf_anytext_full" -> ("daf_anytext", "full"); anything else -> ("", "")."""
    if measure.startswith("rule_b_"):
        rest = measure[len("rule_b_"):]
        for det in DETECTORS:
            for size in SIZES:
                if rest == f"{det}_{size}":
                    return det, size
    return "", ""


def human_grade_breakdown(human, tally):
    """(e) everything in results/human that is split by the people's grade, long format.

    Group size n_group is the number of pages the estimate is over:
      detector AUC by people verdict: n_edited (pages in that verdict group)
      edit-found rate: n_flagged (pages the rater flagged)
      flag rates / ratings: n_pages (pages in that group)
    shaky = yes when n_group < 30. "small_n_per_file" repeats the human files' own flag.
    """
    header = ["source_file", "measure", "people_group", "page_group", "detector", "size", "estimate",
              "ci_low", "ci_high", "n_group", "n_authentic", "n_count", "shaky", "small_n_per_file"]
    rows = []
    # Detector AUC split by what the people said (flagged / missed / in between).
    for r in read_csv(Path(human) / "detector_auc_by_people.csv"):
        rows.append(["detector_auc_by_people.csv", "detector_auc_vs_authentic", r["people"], r["edited_pages"],
                     r["detector"], r["size"], r["auc"], r["ci_low"], r["ci_high"], r["n_edited"],
                     r["n_authentic"], "", shaky(r["n_edited"]), r["small_n"]])
    # How often each rater's flagged pages had the edit found.
    for r in read_csv(Path(human) / "edit_found_rates.csv"):
        rows.append(["edit_found_rates.csv", "edit_found_rate", "rater_" + r["rater"] + "_flagged",
                     r["edited_pages"], "", "", r["edit_found_rate"], r["ci_low"], r["ci_high"], r["n_flagged"],
                     "", r["n_edit_found"], shaky(r["n_flagged"]), r["small_n"]])
    # Share of pages meeting each people / detector flag rule.
    for r in read_csv(Path(human) / "flag_rates.csv"):
        det, size = split_rule_b_measure(r["measure"])
        rows.append(["flag_rates.csv", r["measure"], "", r["group"], det, size, r["share"], r["ci_low"],
                     r["ci_high"], r["n_pages"], "", r["n_meeting"], shaky(r["n_pages"]), ""])
    # Mean rating and share of 5s per group and rater.
    for r in read_csv(Path(human) / "rating_summary.csv"):
        rows.append(["rating_summary.csv", "mean_score", "rater_" + r["rater"], r["group"], "", "", r["mean_score"],
                     r["mean_low"], r["mean_high"], r["n_pages"], "", "", shaky(r["n_pages"]), ""])
        rows.append(["rating_summary.csv", "share_gave_5", "rater_" + r["rater"], r["group"], "", "", r["share_5"],
                     r["share_5_low"], r["share_5_high"], r["n_pages"], "", "", shaky(r["n_pages"]), ""])
    # A detector with no rows at all in the AUC-by-people file is listed as missing, not filled in.
    present = {(r[4], r[5]) for r in rows if r[0] == "detector_auc_by_people.csv"}
    for det in DETECTORS:
        for size in SIZES:
            if (det, size) not in present:
                tally.items.append(f"human_grade_breakdown {det} {size} (no rows in detector_auc_by_people.csv)")
                rows.append(["detector_auc_by_people.csv", "detector_auc_vs_authentic", MISSING, MISSING, det, size,
                             MISSING, MISSING, MISSING, MISSING, MISSING, "", "yes", ""])
    return header, rows


def build_all(results_root, out_dir):
    """Write all five CSVs; returns ({file name: row count}, list of missing items)."""
    results_root, out_dir = Path(results_root), Path(out_dir)
    human = results_root / "human"
    tally = Tally()
    counts = {}
    for name, (header, rows) in {
        "detector_table.csv": detector_table(results_root, tally),
        "ai_minus_handcrafted.csv": ai_minus_handcrafted(results_root, tally),
        "size_difference.csv": size_difference(results_root, tally),
        "people_vs_detectors.csv": people_vs_detectors(human),
        "human_grade_breakdown.csv": human_grade_breakdown(human, tally),
    }.items():
        counts[name] = write_csv(out_dir / name, header, rows)
    return counts, tally.items


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=REPO / "results")
    parser.add_argument("--out-dir", type=Path, default=REPO / "results" / "combined" / "tables")
    args = parser.parse_args()
    counts, missing = build_all(args.results_root, args.out_dir)
    # Row counts only (no numbers), plus how many cells were missing.
    for name, n in counts.items():
        print(f"{name}: {n} rows")
    print(f"missing cells: {len(missing)}")


if __name__ == "__main__":
    main()
