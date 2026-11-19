"""Local check of detectors/daf/tiling.py (T15). Random arrays only, no pages.

Run:  python detectors/daf/test_tiling.py      (or: pytest detectors/daf)
"""

import sys
from pathlib import Path

import numpy as np

# Import tiling.py from this folder and output_format.py from shared/, whatever the current folder is.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import output_format as of  # noqa: E402
import tiling  # noqa: E402

# A fixed seed, so the test is repeatable.
RNG = np.random.default_rng(42)


def test_piece_shape_follows_page_orientation():
    # Landscape (and square) pages get 1536 wide x 1024 high pieces, portrait pages 1024 x 1536.
    assert tiling.piece_dims(1000, 1500) == (1536, 1024)
    assert tiling.piece_dims(1000, 1000) == (1536, 1024)
    assert tiling.piece_dims(1754, 1240) == (1024, 1536)


def test_pieces_cover_the_page_without_overlap():
    # Every pixel of the page is in exactly one piece, for odd sizes too.
    for height, width in [(1754, 1240), (1024, 1536), (1025, 1537), (100, 700), (3000, 2100)]:
        count = np.zeros((height, width), dtype=np.int32)
        for y0, x0, piece_w, piece_h in tiling.piece_grid(height, width):
            count[y0:y0 + piece_h, x0:x0 + piece_w] += 1
        assert count.min() == 1 and count.max() == 1


def test_a_page_that_fits_is_one_piece():
    # A page no bigger than one piece is not cut.
    assert len(tiling.piece_grid(1024, 1536)) == 1
    assert len(tiling.piece_grid(1025, 1536)) == 2


def test_cut_piece_pads_white_right_and_bottom():
    # A 700 x 600 colour page: the piece is full size, holds the page top-left, and 255 elsewhere.
    page = RNG.integers(0, 255, (700, 600, 3), dtype=np.uint8)
    (y0, x0, piece_w, piece_h), = tiling.piece_grid(700, 600)
    piece = tiling.cut_piece(page, y0, x0, piece_w, piece_h)
    assert piece.shape == (1536, 1024, 3)
    assert np.array_equal(piece[:700, :600], page)
    assert (piece[700:] == 255).all() and (piece[:, 600:] == 255).all()


def test_shrunk_size_long_side_is_512():
    # The long side becomes 512, the other side keeps the shape (at least 1 px).
    assert tiling.shrunk_size(1754, 1240) == (362, 512)
    assert tiling.shrunk_size(1240, 1754) == (512, 362)
    assert tiling.shrunk_size(5000, 1) == (1, 512)


def test_moved_boxes_land_where_the_pixels_are():
    # A box found in the second piece of a landscape page, shifted, covers the same page pixels.
    height, width = 1000, 2000
    grid = tiling.piece_grid(height, width)
    assert len(grid) == 2
    y0, x0, _, _ = grid[1]
    # In the piece the box is at x 10..60, y 20..40; on the page it is at x 1546..1596.
    moved = tiling.shift_boxes([(10, 20, 60, 40, 0.9)], x0, y0)
    assert moved == [(1546, 20, 1596, 40, 0.9)]
    heat = of.boxes_to_heatmap((height, width), moved)
    assert heat[20:40, 1546:1596].min() == np.float32(0.9)
    assert heat.sum() == np.float32(0.9) * 50 * 20


def test_box_in_the_padding_is_cut_off():
    # A box that lies partly in the white padding is clipped to the page by the shared helper.
    heat = of.boxes_to_heatmap((100, 200), tiling.shift_boxes([(150, 50, 400, 300, 0.7)], 0, 0))
    assert heat.shape == (100, 200)
    assert heat[50:, 150:].min() == np.float32(0.7) and heat[:50].max() == 0.0


def test_scale_back_from_the_shrunk_copy():
    # A box on a 362 x 512 copy of a 1240 x 1754 page is scaled by the page-to-copy ratio.
    fx, fy = 1240 / 362, 1754 / 512
    (x1, y1, x2, y2, score), = tiling.scale_boxes([(100, 200, 150, 250, 0.5)], fx, fy)
    assert abs(x1 - 100 * fx) < 1e-9 and abs(y2 - 250 * fy) < 1e-9 and score == 0.5


def test_overlapping_boxes_keep_the_maximum():
    # Where two boxes overlap, the larger score stays; elsewhere 0 (docs/PLAN.md section 3).
    heat = of.boxes_to_heatmap((50, 50), [(0, 0, 20, 20, 0.4), (10, 10, 30, 30, 0.8)])
    assert heat[15, 15] == np.float32(0.8) and heat[5, 5] == np.float32(0.4) and heat[40, 40] == 0.0


if __name__ == "__main__":
    # Run every test_ function when started as a plain script.
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print("ok", name)
