"""Tests for shared/score.py. All inputs are tiny arrays made in memory (no pages).

Run from the repo root:  python3 shared/test_score.py   (also works with pytest)
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make `score` and `output_format` importable whether run directly or through pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import output_format as of  # noqa: E402
import score  # noqa: E402


def test_regions_use_eight_connectivity():
    # Two diagonal pixels are ONE region; a far pixel is a second region.
    flag = np.zeros((10, 10), dtype=bool)
    flag[2, 2] = flag[3, 3] = True
    flag[8, 8] = True
    # Box over the diagonal pair: edit found, the far pixel is a false region.
    assert score.analyse_page(flag, [[1, 1, 5, 5]]) == (1, 1)
    # Box over nothing: edit missed, both regions are false.
    assert score.analyse_page(flag, [[6, 0, 7, 1]]) == (0, 2)
    # No flagged pixels at all.
    assert score.analyse_page(np.zeros((5, 5), dtype=bool), [[0, 0, 3, 3]]) == (0, 0)
    # One region touching two boxes finds both edits and is not false.
    wide = np.zeros((10, 10), dtype=bool)
    wide[5, 1:9] = True
    assert score.analyse_page(wide, [[0, 4, 3, 7], [6, 4, 10, 7]]) == (2, 0)


def test_auc_known_values():
    # Perfect ranking = 1, everything tied = 0.5, a hand-counted example = 0.75.
    assert score.auc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    assert score.auc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.5
    assert score.auc(np.array([0.9, 0.4]), np.array([0.5, 0.1])) == 0.75
    assert np.isnan(score.auc(np.array([]), np.array([0.1])))


def test_best_cutoff_separable():
    # With a clean gap between the classes the cut-off is the lowest edited-page score.
    top = np.array([0.1, 0.2, 0.8, 0.9])
    pos = np.array([False, False, True, True])
    assert score.best_cutoff(top, pos) == 0.8


def test_document_folds_keep_documents_together():
    # Pages of one document land in one half; both halves are used; same seed, same split.
    doc = np.repeat(np.arange(10), 3)
    fold = score.document_folds(doc, seed=42)
    for d in range(10):
        assert len(set(fold[doc == d])) == 1
    assert set(fold) == {0, 1}
    assert np.array_equal(fold, score.document_folds(doc, seed=42))


def test_crossfit_cutoff_never_uses_its_own_half():
    # The cut-off applied to half 0 is chosen on half 1 only: changing half 0 must not move it.
    rng = np.random.RandomState(0)
    n = 60
    fold = np.tile([0, 0, 1, 1], n // 4)
    tampered = np.tile([True, False, True, False], n // 4)
    top = np.where(tampered, rng.uniform(0.4, 1.0, n), rng.uniform(0.0, 0.6, n))
    present = np.ones(n, dtype=bool)
    pred, cutoffs = score.crossfit_rule_b(top, tampered, present, fold)
    top2 = top.copy()
    top2[fold == 0] = rng.uniform(0.0, 1.0, (fold == 0).sum())
    _, cutoffs2 = score.crossfit_rule_b(top2, tampered, present, fold)
    assert cutoffs[0] == cutoffs2[0]
    # Predictions on half 0 use that cut-off.
    assert np.array_equal(pred[fold == 0], top[fold == 0] >= cutoffs[0])


def test_metrics_hand_values():
    # 3 edited pages (0-2), 3 untouched (3-5), numbers worked out by hand.
    v = {
        "predA": np.array([1, 1, 0, 1, 0, 0], dtype=bool), "predB": np.array([1, 1, 1, 0, 0, 0], dtype=bool),
        "top": np.array([0.9, 0.8, 0.2, 0.6, 0.1, 0.0]),
        "edits": np.array([1, 1, 2, 0, 0, 0]), "found": np.array([1, 0, 1, 0, 0, 0]),
        "false": np.array([0, 1, 0, 2, 0, 0]),
    }
    pos = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    neg = ~pos
    m = score.compute_metrics(v, np.arange(6), pos, neg)
    # Rule A: tp 2, fp 1, fn 1.
    assert np.isclose(m["precision_A"], 2 / 3) and np.isclose(m["recall_A"], 2 / 3)
    assert np.isclose(m["f1_A"], 2 / 3) and np.isclose(m["false_alarm_A"], 1 / 3)
    # Rule B: tp 3, fp 0, fn 0.
    assert m["precision_B"] == 1 and m["recall_B"] == 1 and m["f1_B"] == 1 and m["false_alarm_B"] == 0
    # AUC: 8 of 9 pairs ranked correctly (0.2 loses to 0.6).
    assert np.isclose(m["auc"], 8 / 9)
    # Location: found 2 of 4 edits, 3 false regions: precision 2/5, recall 1/2, F1 4/9.
    assert np.isclose(m["loc_precision"], 2 / 5) and np.isclose(m["loc_recall"], 1 / 2)
    assert np.isclose(m["loc_f1"], 4 / 9)
    assert m["_counts"] == (3, 3, 4, 2, 3)


def test_resampling_keeps_documents_together():
    # Every page of a drawn document appears the same number of times as the document itself.
    doc = np.repeat(np.arange(8), 3)
    pages_of_doc = [np.flatnonzero(doc == d) for d in range(8)]
    rng = np.random.RandomState(42)
    for _ in range(20):
        idx = score.resample_indices(pages_of_doc, rng)
        counts = np.bincount(idx, minlength=len(doc))
        for d in range(8):
            assert len(set(counts[doc == d])) == 1
        assert len(idx) == len(doc)


def _make_world(root, detector, sizes=("shrink512", "full")):
    """12 documents, each with one edited page and one untouched page; returns the sample.csv path."""
    sample_rows, heats = [], {}
    for d in range(12):
        group = "ai" if d < 6 else "handcrafted"
        detail = ("AnyText" if d < 3 else "TextCtrl") if d < 6 else "retypeset"
        edited, clean = f"page_{2 * d + 1:04d}", f"page_{2 * d + 2:04d}"
        sample_rows.append({"page_id": edited, "doc_id": f"doc_{d + 1:04d}", "group": group, "editor_or_method": detail,
                            "capture": "native", "n_edits": 1, "boxes": json.dumps([[5, 5, 10, 10]]), "ext": "png"})
        sample_rows.append({"page_id": clean, "doc_id": f"doc_{d + 1:04d}", "group": "authentic", "editor_or_method": "",
                            "capture": "native", "n_edits": 0, "boxes": "[]", "ext": "png"})
        # Docs 0-7: detector flags the edit (0.9). Docs 8-11: it misses (max 0.1).
        heat = np.full((30, 30), 0.1 if d >= 8 else 0.0, dtype=np.float32)
        if d < 8:
            heat[5:10, 5:10] = 0.9
        # Doc 0: one extra false region on the edited page.
        if d == 0:
            heat[20:25, 20:25] = 0.9
        heats[edited] = heat
        # Docs 0 and 1: the untouched page gets a false alarm (0.6).
        clean_heat = np.zeros((30, 30), dtype=np.float32)
        if d < 2:
            clean_heat[12:15, 12:15] = 0.6
        heats[clean] = clean_heat
    sample_path = Path(root) / "sample.csv"
    with open(sample_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    for size in sizes:
        folder = of.output_dir(root, detector, size)
        of.write_run_info(folder, {name: None for name in of.REQUIRED_RUN_INFO_FIELDS})
        for page_id, heat in heats.items():
            of.save_page_output(folder, page_id, heat)
    return sample_path


def test_end_to_end_known_numbers():
    # Hand-counted: 12 edits, 8 found, 3 false regions, rule A tp 8 fp 2 fn 4, AUC 136/144.
    with tempfile.TemporaryDirectory() as tmp:
        sample_path = _make_world(tmp, "toy")
        out_dir = Path(tmp) / "results"
        score.score_detector("toy", sample_path, tmp, out_dir, n_boot=50)
        with open(out_dir / "scores.csv", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        get = lambda size, breakdown, subset, metric: next(
            r for r in rows if (r["size"], r["breakdown"], r["subset"], r["metric"]) == (size, breakdown, subset, metric))
        r = get("full", "all", "all", "loc_precision")
        assert (r["n_tampered"], r["n_untouched"], r["n_edits"], r["n_found"], r["n_false_regions"]) == ("12", "12", "12", "8", "3")
        assert np.isclose(float(r["value"]), 8 / 11, atol=1e-5)
        assert np.isclose(float(get("full", "all", "all", "loc_recall")["value"]), 8 / 12, atol=1e-5)
        assert np.isclose(float(get("full", "all", "all", "precision_A")["value"]), 0.8)
        assert np.isclose(float(get("full", "all", "all", "false_alarm_A")["value"]), 2 / 12, atol=1e-5)
        assert np.isclose(float(get("full", "all", "all", "auc")["value"]), 136 / 144, atol=1e-5)
        # Every breakdown is present, and every row has a range around its value.
        assert {r["breakdown"] for r in rows} == {"all", "group", "editor_or_method", "capture"}
        for r in rows:
            if r["value"] and r["ci_low"]:
                assert float(r["ci_low"]) <= float(r["value"]) + 1e-9 <= float(r["ci_high"]) + 2e-9
        # The two sizes hold identical data, so their difference is 0 and not "clear".
        with open(out_dir / "differences.csv", encoding="utf-8", newline="") as handle:
            diffs = list(csv.DictReader(handle))
        size_diffs = [d for d in diffs if d["comparison"] == "shrink512 minus full"]
        assert size_diffs and all(float(d["difference"]) == 0 and d["clear"] == "False" for d in size_diffs)
        assert any(d["comparison"] == "ai minus handcrafted" for d in diffs)
        info = json.loads((out_dir / "score_info.json").read_text(encoding="utf-8"))
        assert info["seed"] == 42 and info["pages_missing"]["full"]["count"] == 0


def test_missing_page_is_counted_not_hidden():
    # A page with no .npz is left out of the metrics and listed in score_info.json.
    with tempfile.TemporaryDirectory() as tmp:
        sample_path = _make_world(tmp, "toy", sizes=("full",))
        (of.output_dir(tmp, "toy", "full") / "page_0001.npz").unlink()
        out_dir = Path(tmp) / "results"
        score.score_detector("toy", sample_path, tmp, out_dir, sizes=("full",), n_boot=20)
        info = json.loads((out_dir / "score_info.json").read_text(encoding="utf-8"))
        assert info["pages_missing"]["full"] == {"count": 1, "page_ids": ["page_0001"]}
        with open(out_dir / "scores.csv", encoding="utf-8", newline="") as handle:
            row = next(r for r in csv.DictReader(handle) if r["metric"] == "auc" and r["breakdown"] == "all")
        assert row["n_tampered"] == "11" and row["n_untouched"] == "12"


def test_incomplete_run_info_is_refused():
    # A run_info.json without the required fields stops the scoring with a clear message.
    with tempfile.TemporaryDirectory() as tmp:
        sample_path = _make_world(tmp, "toy", sizes=("full",))
        of.write_run_info(of.output_dir(tmp, "toy", "full"), {"detector": "toy"})
        try:
            score.score_detector("toy", sample_path, tmp, Path(tmp) / "results", sizes=("full",), n_boot=5)
        except ValueError as error:
            assert "missing required fields" in str(error)
            return
    raise AssertionError("expected ValueError")


def test_repeatable():
    # Two runs with the same seed give byte-identical score tables.
    with tempfile.TemporaryDirectory() as tmp:
        sample_path = _make_world(tmp, "toy", sizes=("full",))
        for name in ("a", "b"):
            score.score_detector("toy", sample_path, tmp, Path(tmp) / name, sizes=("full",), n_boot=30)
        assert (Path(tmp) / "a" / "scores.csv").read_bytes() == (Path(tmp) / "b" / "scores.csv").read_bytes()


if __name__ == "__main__":
    # Run every test_ function and print one line each.
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print(f"ok  {name}")
    print(f"{len(names)} tests passed")
