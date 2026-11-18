"""Tests for shared/output_format.py. Uses arrays made in memory, never real pages.

Run from the repo root:  python3 shared/test_output_format.py
(also works with pytest)
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make `output_format` importable whether run directly or through pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import output_format as of  # noqa: E402

# Seed for the random test heat maps (recorded here as required).
SEED = 42


def test_round_trip_odd_size():
    # A page whose pixel count is not a multiple of 8 checks the padding bits.
    rng = np.random.default_rng(SEED)
    heat = rng.random((37, 53), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = of.save_page_output(Path(tmp), "page_0001", heat)
        back = of.load_page_output(path)
    assert back["shape"] == (37, 53)
    assert back["top"] == float(heat.max())
    # Flags must match the 0.5 threshold exactly, pixel by pixel.
    assert np.array_equal(back["flag"], heat >= 0.5)


def test_threshold_is_inclusive():
    # A pixel at exactly 0.5 is flagged; just below is not.
    heat = np.array([[0.5, 0.49999], [0.0, 1.0]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        back = of.load_page_output(of.save_page_output(Path(tmp), "page_0002", heat))
    assert back["flag"].tolist() == [[True, False], [False, True]]


def test_dtypes_stored():
    # Check the dtypes written to disk match what PLAN.md section 3 says.
    heat = np.full((10, 12), 0.7, dtype=np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        path = of.save_page_output(Path(tmp), "page_0003", heat)
        with np.load(path) as data:
            assert sorted(data.files) == ["flag", "shape", "top"]
            assert data["top"].dtype == np.float32 and data["top"].shape == ()
            assert data["shape"].dtype == np.int32 and data["shape"].tolist() == [10, 12]
            assert data["flag"].dtype == np.uint8 and data["flag"].ndim == 1


def test_all_zero_page():
    # A page with nothing flagged has top 0 and no flagged pixels.
    with tempfile.TemporaryDirectory() as tmp:
        back = of.load_page_output(of.save_page_output(Path(tmp), "page_0004", np.zeros((5, 7))))
    assert back["top"] == 0.0 and not back["flag"].any()


def test_bad_inputs_refused():
    # Real-looking names, NaN, out-of-range values and 3-D arrays must all fail.
    with tempfile.TemporaryDirectory() as tmp:
        for page_id, heat in [
            ("ABC_v1__0123456789", np.zeros((4, 4))),
            ("page_0005", np.full((4, 4), np.nan)),
            ("page_0005", np.full((4, 4), 1.5)),
            ("page_0005", np.zeros((4, 4, 3))),
        ]:
            try:
                of.save_page_output(Path(tmp), page_id, heat)
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for {page_id!r}")


def test_boxes_to_heatmap():
    # Overlapping boxes keep the maximum; the rest of the page stays 0.
    heat = of.boxes_to_heatmap((20, 30), [(2, 2, 10, 10, 0.4), (5, 5, 15, 15, 0.9), (-5, -5, 1, 1, 0.6)])
    assert heat.shape == (20, 30) and heat.dtype == np.float32
    assert heat[3, 3] == np.float32(0.4)
    assert heat[7, 7] == np.float32(0.9)
    assert heat[14, 14] == np.float32(0.9)
    assert heat[0, 0] == np.float32(0.6)  # box clipped at the page edge
    assert heat[19, 29] == 0.0
    # A box fully outside the page adds nothing.
    assert of.boxes_to_heatmap((5, 5), [(10, 10, 20, 20, 1.0)]).max() == 0.0


def test_run_files():
    # run_info.json and run_log.csv go in the same folder as the .npz files.
    with tempfile.TemporaryDirectory() as tmp:
        folder = of.output_dir(tmp, "adcd", "shrink512")
        of.write_run_info(folder, {"detector": "adcd", "seed": 42})
        of.append_run_log(folder, "page_0001", "shrink512", 1.25, 0.91, "ok")
        of.append_run_log(folder, "page_0002", "shrink512", 0.5, 0.1, "ok")
        info = json.loads((folder / "run_info.json").read_text(encoding="utf-8"))
        rows = list(csv.reader((folder / "run_log.csv").open(encoding="utf-8", newline="")))
    assert info == {"detector": "adcd", "seed": 42}
    assert rows[0] == of.RUN_LOG_COLUMNS and len(rows) == 3
    assert rows[1] == ["page_0001", "shrink512", "1.250", "0.910000", "ok"]


def test_bad_size_refused():
    # Only shrink512 and full are valid sizes.
    try:
        of.output_dir("x", "adcd", "half")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    # Run every test_ function and print one line each.
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print(f"ok  {name}")
    print(f"{len(names)} tests passed")
