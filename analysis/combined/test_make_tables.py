"""Tests for analysis/combined/make_tables.py. Uses small made-up score files, never real pages.

Run from the repo root:  python3 analysis/combined/test_make_tables.py
(also works with pytest)
"""

import csv
import sys
import tempfile
from pathlib import Path

# Make make_tables importable whether run directly or through pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_tables as mt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MISSING = mt.MISSING


def write(path, header, rows):
    """Write a small CSV for the fake results folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(header)
        w.writerows(rows)


def fake_results(root, skip_detector=None):
    """Made-up results folders for all four detectors (one can be left out) and a results/human."""
    score_head = ["detector", "size", "breakdown", "subset", "metric", "value", "ci_low", "ci_high",
                  "n_tampered", "n_untouched", "n_edits", "n_found", "n_false_regions"]
    diff_head = ["detector", "comparison", "size", "subset", "metric", "difference", "ci_low", "ci_high", "clear"]
    for det in mt.DETECTORS:
        if det == skip_detector:
            continue
        scores, diffs = [], []
        for size in mt.SIZES:
            for breakdown, subset, metric in [("all", "all", "auc"), ("group", "ai", "auc"),
                                              ("group", "handcrafted", "auc"), ("all", "all", "false_alarm_B"),
                                              ("all", "all", "loc_recall")]:
                scores.append([det, size, breakdown, subset, metric, "0.5", "0.4", "0.6", "10", "5", "12", "6", "7"])
            for metric in mt.DIFF_METRICS:
                diffs.append([det, "ai minus handcrafted", size, "group", metric, "0.1", "0.05", "0.2", "True"])
        for metric in mt.DIFF_METRICS:
            diffs.append([det, "shrink512 minus full", "both", "all:all", metric, "-0.1", "-0.2", "-0.05", "True"])
        write(root / det / "scores.csv", score_head, scores)
        write(root / det / "differences.csv", diff_head, diffs)
    human = root / "human"
    write(human / "auc.csv", ["comparison", "scorer", "source", "detector", "size", "auc", "ci_low", "ci_high",
                              "n_edited", "n_authentic", "small_n"],
          [["edited_vs_authentic", "people_mean", "people", "", "", "0.7", "0.6", "0.8", "320", "180", "False"],
           ["edited_vs_authentic", "adcd_full", "detector", "adcd", "full", "0.5", "0.4", "0.6", "320", "180", "False"]])
    write(human / "auc_differences.csv", ["comparison", "detector", "size", "people_mean_auc_minus_detector_auc",
                                          "ci_low", "ci_high", "clear"],
          [["edited_vs_authentic", "adcd", "full", "0.2", "0.1", "0.3", "True"]])
    people_rows = []
    for det in mt.DETECTORS:
        for size in mt.SIZES:
            people_rows.append([det, size, "people_missed", "ai", "0.6", "0.4", "0.8", "10", "180", "True"])
            people_rows.append([det, size, "people_flagged", "ai", "0.6", "0.5", "0.7", "93", "180", "False"])
    write(human / "detector_auc_by_people.csv", ["detector", "size", "people", "edited_pages", "auc", "ci_low",
                                                 "ci_high", "n_edited", "n_authentic", "small_n"], people_rows)
    write(human / "edit_found_rates.csv", ["rater", "edited_pages", "edit_found_rate", "ci_low", "ci_high",
                                           "n_flagged", "n_edit_found", "small_n"],
          [["abby", "handcrafted", "0.7", "0.4", "0.9", "16", "11", "True"]])
    write(human / "flag_rates.csv", ["measure", "group", "share", "ci_low", "ci_high", "n_pages", "n_meeting"],
          [["rule_b_daf_anytext_full", "ai", "0.9", "0.8", "1.0", "160", "159"],
           ["people_both_gave_5", "ai", "0.06", "0.02", "0.1", "160", "10"]])
    write(human / "rating_summary.csv", ["group", "rater", "n_pages", "mean_score", "mean_low", "mean_high",
                                         "share_5", "share_5_low", "share_5_high"],
          [["ai", "abby", "160", "2.6", "2.3", "2.9", "0.36", "0.28", "0.43"]])


def read(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_row_counts_and_values_are_copied_as_text():
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp) / "results", Path(tmp) / "tables"
        fake_results(root)
        counts, missing = mt.build_all(root, out)
        # 4 detectors x 2 sizes; 4 detectors for the size table.
        assert counts["detector_table.csv"] == 8
        assert counts["ai_minus_handcrafted.csv"] == 8
        assert counts["size_difference.csv"] == 4
        assert counts["people_vs_detectors.csv"] == 3
        assert counts["human_grade_breakdown.csv"] == 16 + 1 + 2 + 2
        assert missing == []
        # A value keeps the exact text of the source (no reformatting).
        assert read(out / "detector_table.csv")[0]["auc_overall"] == "0.5"
        # The size table keeps the file's direction in its column names.
        assert "shrink512_minus_full__auc" in read(out / "size_difference.csv")[0]


def test_ranges_are_ordered_low_estimate_high():
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp) / "results", Path(tmp) / "tables"
        fake_results(root)
        mt.build_all(root, out)
        for name in ["detector_table.csv", "ai_minus_handcrafted.csv", "size_difference.csv"]:
            for row in read(out / name):
                for col in [c for c in row if c.endswith("_ci_low")]:
                    est = float(row[col[:-len("_ci_low")]])
                    assert float(row[col]) <= est <= float(row[col[:-len("_ci_low")] + "_ci_high"])
        for name in ["people_vs_detectors.csv", "human_grade_breakdown.csv"]:
            for row in read(out / name):
                assert float(row["ci_low"]) <= float(row["estimate"]) <= float(row["ci_high"])


def test_shaky_flag_below_30_pages():
    assert mt.shaky("10") == "yes" and mt.shaky("29") == "yes"
    assert mt.shaky("30") == "no" and mt.shaky("160") == "no"
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp) / "results", Path(tmp) / "tables"
        fake_results(root)
        mt.build_all(root, out)
        rows = read(out / "human_grade_breakdown.csv")
        cell = [r for r in rows if r["detector"] == "daf_anytext" and r["size"] == "full"
                and r["people_group"] == "people_missed" and r["page_group"] == "ai"][0]
        assert cell["n_group"] == "10" and cell["shaky"] == "yes"
        flagged = [r for r in rows if r["people_group"] == "people_flagged"][0]
        assert flagged["shaky"] == "no"
        # The rule B row is split into its detector and size.
        rule = [r for r in rows if r["measure"] == "rule_b_daf_anytext_full"][0]
        assert (rule["detector"], rule["size"]) == ("daf_anytext", "full")


def test_missing_detector_is_written_as_missing_not_filled():
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp) / "results", Path(tmp) / "tables"
        fake_results(root, skip_detector="dtd")
        counts, missing = mt.build_all(root, out)
        assert counts["detector_table.csv"] == 8
        assert missing
        dtd = [r for r in read(out / "detector_table.csv") if r["detector"] == "dtd"]
        assert all(r["auc_overall"] == "missing" for r in dtd)


def test_real_results_row_counts_and_range_order():
    # Smoke check on the real score files (numbers only); skipped if they are not here.
    if not (REPO / "results" / "adcd" / "scores.csv").exists():
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "tables"
        counts, missing = mt.build_all(REPO / "results", out)
        assert counts["detector_table.csv"] == 8
        assert counts["ai_minus_handcrafted.csv"] == 8
        assert counts["size_difference.csv"] == 4
        assert counts["people_vs_detectors.csv"] == 33 + 24
        assert counts["human_grade_breakdown.csv"] == 72 + 6 + 36 + 18
        # Range order: low <= estimate <= high; list the rows where the source file itself differs.
        odd = []
        for name, est_col in [("people_vs_detectors.csv", "estimate"), ("human_grade_breakdown.csv", "estimate")]:
            for row in read(out / name):
                if row[est_col] == "missing":
                    continue
                if not float(row["ci_low"]) <= float(row[est_col]) <= float(row["ci_high"]):
                    odd.append((name, row.get("scorer") or row.get("measure"), row["detector"], row["size"]))
        assert not odd, f"estimate outside its own range in the source file: {odd}"
        # Same check for the three tables built from scores.csv and differences.csv.
        for name in ["detector_table.csv", "ai_minus_handcrafted.csv", "size_difference.csv"]:
            for row in read(out / name):
                for col in [c for c in row if c.endswith("_ci_low")]:
                    base = col[:-len("_ci_low")]
                    if MISSING in (row[base], row[col], row[base + "_ci_high"]):
                        continue
                    assert float(row[col]) <= float(row[base]) <= float(row[base + "_ci_high"]), (name, row["detector"], base)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print("ok", fn.__name__)
