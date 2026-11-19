"""Full-size tiling for DTD (T14).

Same rules as detectors/adcd/tiling.py (docs/PLAN.md: "each detector's own tile
size, no overlap, joined back"), plus one helper for DTD's extra input, the JPEG
DCT numbers:
  * the page is cut into PIECE x PIECE pieces (512 = DTD's input size) on a grid
    that starts at the top-left corner (0, 0), with no overlap;
  * the right and bottom edges are padded up to a multiple of PIECE
    (grey image: white = 255; DCT numbers: the numbers a white block would have);
  * the per-piece probability maps are put back on one canvas and the padding is
    cropped off, so the result has exactly the page's size.

Only numpy is needed, so this file can be tested on a laptop.
"""

import numpy as np

# DTD's input size (the "tile size" in docs/PLAN.md); read from the authors' code.
PIECE = 512


def ceil_div(a, b):
    # Integer ceiling, e.g. ceil_div(1025, 512) = 3.
    return -(-a // b)


def grid_shape(height, width, piece=PIECE):
    # Number of piece rows and piece columns needed to cover the page.
    return ceil_div(height, piece), ceil_div(width, piece)


def pad_to(arr, height_pad, width_pad, value):
    # New array of the padded size, filled with `value`, same type.
    out = np.full((height_pad, width_pad) + arr.shape[2:], value, dtype=arr.dtype)
    # Copy the original into the top-left corner (padding stays right and bottom).
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def white_dc(qtable):
    """DC number of an all-white 8x8 block for a JPEG with this 8x8 quantisation table.

    JPEG subtracts 128 from every pixel (255 - 128 = 127), the 8x8 DCT of a flat block
    puts 8 * 127 into the DC cell, and the number stored is that divided by the table's
    first value and rounded.
    """
    return int(round(8 * 127 / float(np.asarray(qtable).reshape(-1)[0])))


def pad_dct(dct, height_pad, width_pad, qtable):
    """Pad a page's DCT-number array (blocks laid out in place, as jpegio gives them).

    `dct` must cover whole 8x8 blocks (height and width multiples of 8). New blocks are
    those of a white page: DC = white_dc(qtable), every other number 0.
    """
    height, width = dct.shape
    if height % 8 or width % 8 or height_pad % 8 or width_pad % 8:
        raise ValueError("DCT arrays and padded sizes must be multiples of 8")
    out = np.zeros((height_pad, width_pad), dtype=dct.dtype)
    out[:height, :width] = dct
    # The DC cell of a block sits at its top-left corner. Fill it for every new block.
    dc = np.full((height_pad // 8, width_pad // 8), white_dc(qtable), dtype=dct.dtype)
    dc[: height // 8, : width // 8] = dct[::8, ::8]
    out[::8, ::8] = dc
    return out


def cut_tiles(arr, fill, piece=PIECE):
    """Pad `arr` on the right/bottom with `fill`, then cut it into piece x piece tiles.

    Returns {(row, col): tile}, every tile exactly piece x piece.
    """
    height, width = arr.shape[:2]
    n_rows, n_cols = grid_shape(height, width, piece)
    padded = pad_to(arr, n_rows * piece, n_cols * piece, fill)
    return cut_padded(padded, piece)


def cut_padded(padded, piece=PIECE):
    # Cut an array whose size is already a multiple of `piece` into {(row, col): tile}.
    n_rows, n_cols = padded.shape[0] // piece, padded.shape[1] // piece
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
