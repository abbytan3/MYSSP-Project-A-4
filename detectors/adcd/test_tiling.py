"""Checks for tiling.py. Run: python detectors/adcd/test_tiling.py (needs numpy).

Uses random arrays only, never a page.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiling  # noqa: E402


def check_roundtrip(height, width, channels):
    # Seeded random "page" (colour or grey), values 0..254 so white (255) is distinguishable.
    rng = np.random.default_rng(42)
    shape = (height, width) + ((channels,) if channels else ())
    page = rng.integers(0, 255, size=shape, dtype=np.uint8)
    tiles = tiling.cut_tiles(page, 255)
    n_rows, n_cols = tiling.grid_shape(height, width)
    assert len(tiles) == n_rows * n_cols
    assert all(t.shape[:2] == (512, 512) for t in tiles.values())
    # Padding must be white; stitching the pixel values back must give the page exactly.
    canvas = np.zeros((n_rows * 512, n_cols * 512) + page.shape[2:], dtype=np.uint8)
    for (r, c), t in tiles.items():
        canvas[r * 512:(r + 1) * 512, c * 512:(c + 1) * 512] = t
    assert (canvas[:height, :width] == page).all()
    assert (canvas[height:, :] == 255).all() and (canvas[:, width:] == 255).all()
    # Joining per-tile maps must give the page size and keep each value in place.
    maps = {k: np.full((512, 512), k[0] * 10 + k[1], dtype=np.float32) for k in tiles}
    joined = tiling.join_tiles(maps, height, width)
    assert joined.shape == (height, width) and joined.dtype == np.float32
    assert joined[0, 0] == 0 and joined[height - 1, width - 1] == (n_rows - 1) * 10 + (n_cols - 1)


def main():
    # Exact multiple, one pixel over, smaller than one tile, and a tall page.
    for h, w, ch in [(1024, 1024, 3), (1025, 513, 3), (300, 200, 3), (2339, 1654, 0), (512, 512, 0)]:
        check_roundtrip(h, w, ch)
    # A missing tile must be refused.
    maps = {(0, 0): np.zeros((512, 512), np.float32)}
    try:
        tiling.join_tiles(maps, 1024, 512)
    except ValueError:
        pass
    else:
        raise AssertionError("missing tile was not refused")
    print("tiling checks passed")


if __name__ == "__main__":
    main()
