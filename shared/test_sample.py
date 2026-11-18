"""Tests for shared/sample.py selection rules, on tiny made-up metadata (no pages).

Run from the repo root:  python3 shared/test_sample.py   (also works with pytest)
"""

import csv
import sys
import tempfile
from pathlib import Path

# Make `sample` importable whether run directly or through pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample  # noqa: E402

# Smaller targets so the made-up world is small: retypeset 3, copymove 2, pdf_edit 1.
sample.HANDCRAFTED_TARGETS = {"retypeset": 3, "copymove": 2, "pdf_edit": 1}


def make_world():
    """20 made-up documents. Docs 1-5 have AI edits (doc 5 rejected), docs 6-20 handcrafted."""
    records, decisions = {}, {}

    def add(key, label, doc, **extra):
        records[key] = {"key": key, "set": key.split("/")[0], "file_name": key.split("/")[1],
                        "path": "unused", "label": label, "model": None, "method": None,
                        "capture": "native", "n_edits": 0 if label == "authentic" else 1,
                        "boxes": [] if label == "authentic" else [[1, 2, 3, 4]], "doc_id": doc}
        records[key].update(extra)

    methods = ["m3_retypeset", "m1_copymove", "m7_pdf_edit"]
    for n in range(1, 21):
        doc = f"DOC{n:02d}"
        # Every document has the same authentic page in both sets, as in the real data.
        add(f"set1/{doc}_c0.jpg", "authentic", doc)
        add(f"set2/{doc}_c0.jpg", "authentic", doc)
        if n <= 5:
            key = f"set2/{doc}_t0_c0.jpg"
            add(key, "tampered", doc, model="AnyText v1.1.3 (ICLR 2024)")
            decisions[key] = "reject" if n == 5 else "keep"
        else:
            key = f"set1/{doc}_t0_c0.jpg"
            add(key, "tampered", doc, method=methods[n % 3])
            decisions[key] = "keep"
    return records, decisions


def run(seed=42, stage2=None):
    records, decisions = make_world()
    stage2 = stage2 if stage2 is not None else {"set1/DOC09_t0_c0.jpg", "set1/DOC19_c0.jpg"}
    return sample.build_sample(records, decisions, stage2, seed=seed)


def test_group_counts():
    # AI = kept set2 tampered (4); handcrafted = 3+2+1; originals = one per document behind them.
    rows, report = run()
    assert report["counts"]["ai"] == 4
    assert report["counts"]["handcrafted"] == 6
    assert report["counts"]["authentic"] == len({r["real_doc_id"] for r in rows})
    assert not report["warnings"]


def test_rejected_page_left_out():
    # The rejected AI page must not be sampled.
    rows, _ = run()
    assert not any(r["key"] == "set2/DOC05_t0_c0.jpg" for r in rows)


def test_stage2_pages_included():
    # A stage 2 handcrafted page is always in; a stage 2 authentic page is in even if no edit uses its document.
    rows, _ = run(stage2={"set1/DOC09_t0_c0.jpg", "set1/DOC05_c0.jpg"})
    keys = {r["key"] for r in rows}
    assert "set1/DOC09_t0_c0.jpg" in keys
    assert "set1/DOC05_c0.jpg" in keys


def test_one_original_per_document():
    # The set1 and set2 copies are the same page, so each document appears once as an original.
    rows, _ = run()
    originals = [r["real_doc_id"] for r in rows if r["group"] == "authentic"]
    assert len(originals) == len(set(originals))
    assert all(r["set"] == "set1" for r in rows if r["group"] == "authentic")


def test_repeatable_and_seeded():
    # Same seed gives the same sample; a different seed gives a different one.
    a = [(r["page_id"], r["key"]) for r in run(seed=42)[0]]
    b = [(r["page_id"], r["key"]) for r in run(seed=42)[0]]
    c = [(r["page_id"], r["key"]) for r in run(seed=7)[0]]
    assert a == b and a != c


def test_anonymous_ids_and_sample_csv_has_no_real_names():
    # Page IDs run page_0001..; documents share doc_NNNN; sample.csv holds no real name or document ID.
    rows, _ = run()
    assert [r["page_id"] for r in sorted(rows, key=lambda r: r["page_id"])][:2] == ["page_0001", "page_0002"]
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r["real_doc_id"], set()).add(r["doc_id"])
    assert all(len(v) == 1 for v in by_doc.values())
    with tempfile.TemporaryDirectory() as tmp:
        sample.write_outputs(rows, tmp)
        text = (Path(tmp) / "sample.csv").read_text(encoding="utf-8")
        header = next(csv.reader(text.splitlines()))
        map_text = (Path(tmp) / "sample_map.csv").read_text(encoding="utf-8")
    assert header == sample.SAMPLE_COLUMNS
    assert "DOC" not in text and "_c0" not in text and "unused" not in text
    assert "DOC" in map_text  # the private map does keep the real names


def test_target_not_reachable_is_reported():
    # If the pool cannot reach a target, the script warns instead of silently shrinking.
    sample.HANDCRAFTED_TARGETS["pdf_edit"] = 50
    try:
        _, report = run()
    finally:
        sample.HANDCRAFTED_TARGETS["pdf_edit"] = 1
    assert report["warnings"]


if __name__ == "__main__":
    # Run every test_ function and print one line each.
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print(f"ok  {name}")
    print(f"{len(names)} tests passed")
