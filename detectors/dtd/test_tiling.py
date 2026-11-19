"""Local check of detectors/dtd/tiling.py (T14). Random arrays only, no pages.

Run:  python detectors/dtd/test_tiling.py      (or: pytest detectors/dtd)
"""

import sys
from pathlib import Path

import numpy as np

# Import tiling.py from this folder, whatever the current folder is.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiling  # noqa: E402

# A fixed seed, so the test is repeatable.
RNG = np.random.default_rng(42)


def test_round_trip_is_exact():
    # Cutting a map into tiles and joining them back must give the same map, for odd sizes too.
    for height, width in [(512, 512), (600, 1030), (1754, 1240), (100, 700), (513, 511)]:
        page = RNG.random((height, width)).astype(np.float32)
        tiles = tiling.cut_tiles(page, 0.0)
        assert all(t.shape == (512, 512) for t in tiles.values())
        assert len(tiles) == tiling.ceil_div(height, 512) * tiling.ceil_div(width, 512)
        assert np.array_equal(tiling.join_tiles(tiles, height, width), page)


def test_padding_is_white_right_and_bottom():
    # A grey page of 700x600 is padded with 255 on the right and bottom only.
    page = RNG.integers(0, 255, (700, 600), dtype=np.uint8)
    tiles = tiling.cut_tiles(page, 255)
    assert np.array_equal(tiles[(0, 0)], page[:512, :512])
    assert (tiles[(1, 1)][700 - 512:, :] == 255).all()
    assert (tiles[(1, 1)][:, 600 - 512:] == 255).all()


def test_missing_tile_is_refused():
    page = RNG.random((600, 600)).astype(np.float32)
    tiles = tiling.cut_tiles(page, 0.0)
    tiles.pop((1, 1))
    try:
        tiling.join_tiles(tiles, 600, 600)
    except ValueError:
        return
    raise AssertionError("a missing tile should be refused")


def test_white_dc_values():
    # With quantisation table all 1s (JPEG quality 100) a white block has DC = 8 * 127 = 1016.
    assert tiling.white_dc(np.ones((8, 8), dtype=np.int64)) == 1016
    # A table starting with 8 gives 127.
    table = np.ones((8, 8), dtype=np.int64)
    table[0, 0] = 8
    assert tiling.white_dc(table) == 127


def test_pad_dct_keeps_original_and_fills_white_blocks():
    dct = RNG.integers(-5, 6, (16, 24), dtype=np.int32)
    qt = np.ones((8, 8), dtype=np.int64)
    out = tiling.pad_dct(dct, 32, 40, qt)
    assert out.shape == (32, 40)
    # The original numbers are untouched.
    assert np.array_equal(out[:16, :24], dct)
    # Every new block has DC 1016 and zeros elsewhere.
    new_rows = np.zeros((32, 40), dtype=bool)
    new_rows[16:, :] = True
    new_rows[:, 24:] = True
    block_is_new = new_rows[::8, ::8]
    assert (out[::8, ::8][block_is_new] == 1016).all()
    ac = out.copy()
    ac[::8, ::8] = 0
    assert (ac[new_rows] == 0).all()


def test_pad_dct_rejects_sizes_not_multiple_of_8():
    qt = np.ones((8, 8), dtype=np.int64)
    try:
        tiling.pad_dct(np.zeros((10, 16), dtype=np.int32), 16, 16, qt)
    except ValueError:
        return
    raise AssertionError("sizes that are not multiples of 8 should be refused")


if __name__ == "__main__":
    # Run every test function in this file and report.
    count = 0
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            count += 1
            print("ok", name)
    print("all %d tests passed" % count)
