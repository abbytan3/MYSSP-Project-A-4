"""Tests for analysis/human. Every input is a small made-up array or table (no pages, no real names).

Run from the repo root:  python3 -m pytest analysis/human -q
"""

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make the scripts importable whether run directly or through pytest.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import join_ratings as jr  # noqa: E402
import people_vs_detectors as pvd  # noqa: E402


# ---------------------------------------------------------------- made-up data

def fake_people(seed=0):
    """42 made-up pages: 12 AI, 12 handcrafted, 18 authentic. Authentic pages share documents with edited ones."""
    rng = np.random.RandomState(seed)
    group = np.array(["ai"] * 12 + ["handcrafted"] * 12 + ["authentic"] * 18)
    n = len(group)
    # Edited pages: people lean low; authentic pages: people lean high.
    low = rng.choice([1, 2, 3, 5], size=n, p=[0.4, 0.3, 0.2, 0.1])
    high = rng.choice([3, 4, 5], size=n, p=[0.2, 0.2, 0.6])
    abby = np.where(group == "authentic", high, low)
    nazhan = np.clip(abby + rng.choice([-1, 0, 1], size=n), 1, 5)
    return {
        "page_ids": [f"page_{i:04d}" for i in range(n)], "at": np.arange(n), "group": group,
        "doc": np.arange(n) % 24, "abby": abby, "nazhan": nazhan,
        "found_abby": rng.rand(n) < 0.5, "found_nazhan": rng.rand(n) < 0.5,
    }


def fake_detector(people, seed=1):
    """A made-up detector: edited pages score a little higher; rule B flags the top scorers."""
    rng = np.random.RandomState(seed)
    top = np.clip(rng.rand(len(people["group"])) * 0.6 + (people["group"] != "authentic") * 0.3, 0, 1)
    return {"top": top, "predB": top >= 0.5}


# ---------------------------------------------------------------- definitions

def test_edited_ness_flips_the_people_score():
    # 5 (genuine) becomes 1, 1 (clearly edited) becomes 5, a mean of 2.5 becomes 3.5.
    assert list(pvd.edited_ness([1, 3, 5])) == [5.0, 3.0, 1.0]
    assert pvd.edited_ness(2.5) == 3.5


def test_flagged_and_missed_definitions():
    # A 3 is "unsure": not flagged. Mean 2.5 is flagged, 2.6 is not (scores are whole numbers, means halves).
    assert list(pvd.people_flagged([1.0, 2.0, 2.5, 3.0, 5.0])) == [True, True, True, False, False]
    # Missed only when BOTH raters gave a 5.
    assert list(pvd.people_missed([5, 5, 4, 1], [5, 4, 5, 5])) == [True, False, False, False]


def test_verdict_groups_do_not_overlap_and_only_cover_edited_pages():
    people = fake_people()
    v = pvd.people_verdict_masks(people)
    edited = people["group"] != "authentic"
    assert not (v["people_flagged"] & v["people_missed"]).any()
    assert (v["people_flagged"] | v["people_missed"] | v["in_between"]).sum() == edited.sum()
    assert not (v["people_flagged"] & ~edited).any()


# ---------------------------------------------------------------- ranking direction

def test_auc_is_one_when_people_rank_edited_above_authentic_after_the_flip():
    # People: edited pages score 1, authentic score 5. After 6 - score, edited are higher: AUC 1.
    people = fake_people()
    edited = people["group"] != "authentic"
    people["abby"] = np.where(edited, 1, 5)
    people["nazhan"] = np.where(edited, 2, 5)
    pos, neg = pvd.comparisons(people)["edited_vs_authentic"]
    auc = pvd.auc_fn(pvd.edited_ness(pvd.combined_score(people)), pos, neg)(np.arange(len(edited)))
    assert auc == 1.0
    # Without the flip the same ratings would give AUC 0 (the mistake the flip prevents).
    wrong = pvd.auc_fn(pvd.combined_score(people), pos, neg)(np.arange(len(edited)))
    assert wrong == 0.0


def test_auc_with_all_ties_is_one_half():
    # Everyone gave a 3: all pages tie, mid-ranks give AUC 0.5.
    people = fake_people()
    pos, neg = pvd.comparisons(people)["edited_vs_authentic"]
    auc = pvd.auc_fn(np.full(len(pos), 3.0), pos, neg)(np.arange(len(pos)))
    assert auc == 0.5


# ---------------------------------------------------------------- bootstrap

def test_resamples_keep_whole_documents_and_are_repeatable():
    doc = np.array([0, 0, 1, 1, 2, 3, 3])
    draws = pvd.make_resamples(doc, 50, 42)
    for idx in draws:
        counts = np.bincount(idx, minlength=len(doc))
        # Pages of the same document appear equally often: they come and go together.
        assert counts[0] == counts[1] and counts[2] == counts[3] and counts[5] == counts[6]
    again = pvd.make_resamples(doc, 50, 42)
    assert all(np.array_equal(a, b) for a, b in zip(draws, again))


def test_share_and_mean_statistics():
    cond = np.array([True, False, True, False])
    denom = np.array([True, True, True, False])
    assert pvd.share_fn(cond, denom)(np.arange(4)) == pytest.approx(2 / 3)
    assert pvd.mean_fn(np.array([1.0, 2.0, 3.0, 9.0]), denom)(np.arange(4)) == pytest.approx(2.0)
    # No pages in the denominator gives NaN, not an error.
    assert np.isnan(pvd.share_fn(cond, np.zeros(4, dtype=bool))(np.arange(4)))


# ---------------------------------------------------------------- tables

def test_rating_summary_numbers_for_a_tiny_case():
    people = {"group": np.array(["ai", "ai", "authentic", "authentic"]), "doc": np.arange(4),
              "abby": np.array([5, 1, 5, 5]), "nazhan": np.array([5, 2, 4, 5])}
    rows = pvd.rating_summary(people, pvd.make_resamples(people["doc"], 20, 42))
    pick = {(r["group"], r["rater"]): r for r in rows}
    assert pick[("ai", "abby")]["n_pages"] == 2 and pick[("ai", "abby")]["mean_score"] == 3.0
    assert pick[("ai", "abby")]["share_5"] == 0.5
    # mean_of_two: mean of the two raters' scores; share_5 means BOTH gave a 5.
    assert pick[("authentic", "mean_of_two")]["mean_score"] == pytest.approx(4.75)
    assert pick[("authentic", "mean_of_two")]["share_5"] == 0.5


def test_run_all_builds_every_table_with_counts_and_ranges():
    people = fake_people()
    dets = {("fake", "shrink512"): fake_detector(people), ("fake", "full"): fake_detector(people, seed=2)}
    tables = pvd.run_all(people, dets, n_boot=30, seed=42)
    assert set(tables) == {"rating_summary.csv", "auc.csv", "auc_differences.csv",
                           "detector_auc_by_people.csv", "flag_rates.csv", "edit_found_rates.csv"}
    # 3 comparisons x (3 people scorers + 2 detector sizes).
    assert len(tables["auc.csv"]) == 15
    for row in tables["auc.csv"]:
        assert 0.0 <= row["auc"] <= 1.0 and row["ci_low"] <= row["ci_high"]
        assert row["n_edited"] > 0 and row["n_authentic"] == 18
        # Fewer than 30 pages is marked shaky (all made-up groups are small).
        assert row["small_n"] is True
    # Rule B appears together with the authentic row (its false-alarm rate).
    rule_b = [r for r in tables["flag_rates.csv"] if r["measure"] == "rule_b_fake_full"]
    assert {r["group"] for r in rule_b} == {"ai", "handcrafted", "authentic"}
    # Verdict rows: flagged, missed and in-between pages add up to all 24 edited pages.
    by = {}
    for r in tables["detector_auc_by_people.csv"]:
        if r["size"] == "full" and r["edited_pages"] == "edited":
            by[r["people"]] = r["n_edited"]
    assert sum(by.values()) == 24


def test_edit_found_rate_uses_only_flagged_pages():
    people = {"group": np.array(["ai", "ai", "ai", "authentic"]), "doc": np.arange(4),
              "abby": np.array([1, 2, 5, 1]), "nazhan": np.array([5, 5, 5, 5]),
              "found_abby": np.array([True, False, True, False]), "found_nazhan": np.zeros(4, dtype=bool)}
    rows = pvd.edit_found_rates(people, pvd.make_resamples(people["doc"], 10, 42))
    abby_ai = [r for r in rows if r["rater"] == "abby" and r["edited_pages"] == "ai"][0]
    # Abby flagged 2 of the 3 AI pages (scores 1 and 2); her click hit on 1 of them.
    assert abby_ai["n_flagged"] == 2 and abby_ai["n_edit_found"] == 1 and abby_ai["edit_found_rate"] == 0.5


# ---------------------------------------------------------------- rule B reproduction check

def tiny_variant():
    """A made-up detector variant with every field compute_metrics reads."""
    top = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1])
    tampered = np.array([True, True, True, False, False, False])
    n = len(top)
    variant = {"present": np.ones(n, dtype=bool), "top": top, "predA": top >= 0.5, "predB": top >= 0.65,
               "found": np.array([1, 0, 1, 0, 0, 0]), "false": np.array([0, 1, 0, 0, 2, 0]),
               "edits": np.array([1, 1, 1, 0, 0, 0])}
    return variant, {"page_id": [f"p{i}" for i in range(n)], "tampered": tampered}


def reference_from(variant, sample, size="full"):
    """Build a scores.csv-style reference from the variant's own metrics (6 decimals, like the file)."""
    n = len(variant["top"])
    pos, neg = sample["tampered"], ~sample["tampered"]
    m = pvd.sc.compute_metrics(variant, np.arange(n), pos, neg)
    return {(size, k): {"value": "" if np.isnan(m[k]) else f"{m[k]:.6f}", "n_tampered": "3", "n_untouched": "3"}
            for k in pvd.sc.METRICS}


def test_reproduction_check_passes_when_numbers_match_and_stops_when_they_do_not():
    variant, sample = tiny_variant()
    ref = reference_from(variant, sample)
    out = pvd.check_reproduces("full", variant, sample, ref, [0.5, 0.6], [0.5, 0.6])
    assert out["largest_difference"] < 1e-6
    # A different rule B decision changes recall_B, so the check stops.
    changed = dict(variant, predB=variant["top"] >= 0.95)
    with pytest.raises(SystemExit):
        pvd.check_reproduces("full", changed, sample, ref, [0.5, 0.6], [0.5, 0.6])
    # Different cut-offs also stop it.
    with pytest.raises(SystemExit):
        pvd.check_reproduces("full", variant, sample, ref, [0.5, 0.6], [0.5, 0.61])


# ---------------------------------------------------------------- building the people arrays

def joined_rows(people, group_override=None):
    """Write made-up people into the joined-file row format."""
    rows = []
    for who in pvd.RATERS:
        for i, page in enumerate(people["page_ids"]):
            rows.append({"page_id": page, "group": (group_override or people["group"])[i], "rater": who,
                         "score": str(people[who][i]), "no_traces": "False",
                         "edit_found": str(bool(people[f"found_{who}"][i]))})
    return rows


def test_build_people_checks_groups_against_the_sample():
    people = fake_people()
    n = len(people["group"])
    sample = {"page_id": people["page_ids"], "group": people["group"], "doc": people["doc"]}
    built = pvd.build_people(joined_rows(people), sample)
    assert built["group"].tolist() == people["group"].tolist()
    assert built["abby"].tolist() == people["abby"].tolist()
    # A group that disagrees with sample.csv must stop the analysis.
    wrong = ["ai"] * n
    with pytest.raises(ValueError):
        pvd.build_people(joined_rows(people, group_override=wrong), sample)
    # A rater missing a page must stop it too.
    with pytest.raises(ValueError):
        pvd.build_people(joined_rows(people)[:-1], sample)


# ---------------------------------------------------------------- join_ratings

def test_latest_rating_wins_for_a_repeated_page():
    records = [{"item_id": "set1/a", "position": 3, "score": 1},
               {"item_id": "set1/a", "position": 40, "score": 5},
               {"item_id": "set1/b", "position": 7, "score": 2}]
    latest, repeats = jr.latest_per_item(records)
    assert latest["set1/a"]["score"] == 5 and repeats == 1 and len(latest) == 2


def test_join_skips_synthetic_pages_and_counts_unmatched():
    rec = {"label": "tampered", "score": 2, "no_traces": False, "edit_found": True, "false_alarm": False}
    latest = {"set1/a": rec, "set3/zzz": rec, "set2/missing": rec}
    key_to_page = {"set1/a": {"page_id": "page_0001", "group": "handcrafted", "stage2": "1"}}
    joined, unmatched = jr.join_one_rater(latest, key_to_page)
    assert [j["page_id"] for j in joined] == ["page_0001"] and unmatched == 1


def test_joined_file_has_only_the_allowed_columns_and_never_goes_inside_the_repo():
    rows = [{"page_id": "page_0001", "group": "ai", "stage2": "1", "label": "tampered", "score": 2,
             "no_traces": False, "edit_found": True, "false_alarm": False}]
    with tempfile.TemporaryDirectory() as tmp:
        out_file, n_rows = jr.write_joined({"abby": rows}, Path(tmp) / "private")
        with open(out_file, encoding="utf-8", newline="") as fh:
            header = next(csv.reader(fh))
    assert header == ["page_id", "group", "rater", "score", "no_traces", "edit_found", "false_alarm"]
    assert n_rows == 1
    # A folder inside this repo is refused before anything is written.
    with pytest.raises(SystemExit):
        jr.write_joined({"abby": rows}, HERE / "should_not_exist")
    assert not (HERE / "should_not_exist").exists()
