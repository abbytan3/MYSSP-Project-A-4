"""Tests for shared/check_outputs.py on tiny in-memory arrays (no pages).

Run from the repo root:  python3 shared/test_check_outputs.py   (also works with pytest)
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make `check_outputs` and `output_format` importable whether run directly or through pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_outputs as co  # noqa: E402
import output_format as of  # noqa: E402


def _world(root, top_of_second=0.9):
    """Two pages (one edited, one untouched) written as a detector job would write them."""
    folder = of.output_dir(root, "toy", "full")
    of.write_run_info(folder, {name: None for name in of.REQUIRED_RUN_INFO_FIELDS})
    rows = [{"page_id": "page_0001", "boxes": json.dumps([[2, 2, 6, 6]])}, {"page_id": "page_0002", "boxes": "[]"}]
    heats = {"page_0001": np.zeros((10, 12), dtype=np.float32), "page_0002": np.zeros((10, 12), dtype=np.float32)}
    heats["page_0001"][2:6, 2:6] = 0.9
    heats["page_0002"][0, 0] = top_of_second
    for row in rows:
        of.save_page_output(folder, row["page_id"], heats[row["page_id"]])
        of.append_run_log(folder, row["page_id"], "full", 0.5, float(heats[row["page_id"]].max()), "ok")
    return rows


def test_good_outputs_pass():
    # A correctly written folder has no problems.
    with tempfile.TemporaryDirectory() as tmp:
        rows = _world(tmp)
        problems, checked = co.check_size("toy", "full", rows, tmp)
    assert problems == {} and checked == 2


def test_problems_are_found():
    # A missing file, a missing run_info field, a log mismatch and a box outside the page are all reported.
    with tempfile.TemporaryDirectory() as tmp:
        rows = _world(tmp)
        folder = of.output_dir(tmp, "toy", "full")
        (folder / "page_0002.npz").unlink()
        of.write_run_info(folder, {"detector": "toy"})
        rows[0]["boxes"] = json.dumps([[2, 2, 99, 6]])
        problems, checked = co.check_size("toy", "full", rows, tmp)
    assert checked == 1
    assert problems["npz missing"] == ["page_0002"]
    assert problems["edit box outside page shape"] == ["page_0001"]
    assert any(name.startswith("run_info") for name in problems)


if __name__ == "__main__":
    # Run every test_ function and print one line each.
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print(f"ok  {name}")
    print(f"{len(names)} tests passed")
