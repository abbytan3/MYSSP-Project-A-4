"""Full-size tiling for ADCD-Net (T13).

Rebuilt as plain functions from cell 25 of the Week 7 notebook 2A, which had
no function form (it used notebook globals). The rules are unchanged:
  * the page is cut into PIECE x PIECE pieces (512 = ADCD-Net's own input size)
    on a grid that starts at the top-left corner (0, 0), with no overlap;
  * the right and bottom edges are padded up to a multiple of PIECE
    (colour image: white = 255; OCR text mask: 0 = no text);
  * the per-piece probability maps are put back on one canvas and the
    padding is cropped off, so the result has exactly the page's size.

Only numpy is needed, so this file can be tested on a laptop.
"""

import numpy as np

# ADCD-Net's own input size (the "tile size" in docs/PLAN.md).
PIECE = 512


def ceil_div(a, b):
    # Integer ceiling, e.g. ceil_div(1025, 512) = 3.
    return -(-a // b)


def grid_shape(height, width, piece=PIECE):
    # Number of piece rows and piece columns needed to cover the page.
    return ceil_div(height, piece), ceil_div(width, piece)


def pad_to(arr, height_pad, width_pad, value):
    # New array of the padded size, filled with `value`, same channels and type.
    out = np.full((height_pad, width_pad) + arr.shape[2:], value, dtype=arr.dtype)
    # Copy the original into the top-left corner (padding stays right and bottom).
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def cut_tiles(arr, fill, piece=PIECE):
    """Pad `arr` on the right/bottom with `fill`, then cut it into piece x piece tiles.

    Returns {(row, col): tile}, every tile exactly piece x piece.
    """
    height, width = arr.shape[:2]
    n_rows, n_cols = grid_shape(height, width, piece)
    padded = pad_to(arr, n_rows * piece, n_cols * piece, fill)
    tiles = {}
    for r in range(n_rows):
        for c in range(n_cols):
            # Copy so each tile is its own array (not a view of the big one).
            tiles[(r, c)] = padded[r * piece:(r + 1) * piece, c * piece:(c + 1) * piece].copy()
    return tiles


def join_tiles(tiles, height, width, piece=PIECE):
    """Put per-tile 2-D maps back together and crop to (height, width).

    tiles: {(row, col): 2-D array of shape (piece, piece)}; every tile of the
    grid must be present. Returns a float32 map of exactly (height, width).
    """
    n_rows, n_cols = grid_shape(height, width, piece)
    expected = {(r, c) for r in range(n_rows) for c in range(n_cols)}
    # A missing or extra tile would silently leave a hole, so refuse it.
    if set(tiles) != expected:
        raise ValueError("tiles do not match the grid for a %dx%d page" % (height, width))
    canvas = np.zeros((n_rows * piece, n_cols * piece), dtype=np.float32)
    for (r, c), tile in tiles.items():
        if tile.shape != (piece, piece):
            raise ValueError("tile (%d, %d) has shape %s, expected %s" % (r, c, tile.shape, (piece, piece)))
        canvas[r * piece:(r + 1) * piece, c * piece:(c + 1) * piece] = tile
    # Crop the padding away.
    return canvas[:height, :width]
