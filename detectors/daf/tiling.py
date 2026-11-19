"""Full-size cutting and box moving for DAF (T15).

docs/PLAN.md: "full size, cut into each detector's own tile size, no overlap, joined back".
DAF's own test size is the config's resize to fit 1024 x 1536 (keeping the shape), so a DAF
tile is 1536 px along the page's long side and 1024 px along its short side:
  * a landscape page (width >= height) gets 1536 wide x 1024 high pieces, a portrait page
    gets 1024 wide x 1536 high pieces;
  * pieces sit on a grid that starts at the top-left corner (0, 0), with no overlap;
  * the right and bottom edges are padded with white (255) up to a whole piece;
  * DAF returns boxes, not maps, so "joining back" means moving each box by its piece's
    position and clipping it to the page (the clip removes the padding).

Only numpy is needed, so this file can be tested on a laptop. It must also run on Python 3.9
(the DAF environment), so it uses no newer syntax.
"""

import numpy as np

# Piece length along the page's long side and along its short side (DAF's 1536 x 1024 test size).
PIECE_LONG = 1536
PIECE_SHORT = 1024
# Length of the page's long side after shrink512.
SHRINK_LONG = 512


def piece_dims(height, width):
    """Return (piece_width, piece_height): landscape pages get landscape pieces, portrait pages portrait."""
    if width >= height:
        return PIECE_LONG, PIECE_SHORT
    return PIECE_SHORT, PIECE_LONG


def piece_grid(height, width):
    """Return the list of (y0, x0, piece_width, piece_height), rows from the top, columns from the left."""
    piece_w, piece_h = piece_dims(height, width)
    return [(y0, x0, piece_w, piece_h)
            for y0 in range(0, height, piece_h)
            for x0 in range(0, width, piece_w)]


def cut_piece(img, y0, x0, piece_w, piece_h):
    """Cut one piece out of a page (H x W x 3) and pad it with white to exactly piece_h x piece_w."""
    # White canvas of the full piece size: whatever the page does not cover stays white (the padding).
    piece = np.full((piece_h, piece_w) + img.shape[2:], 255, dtype=img.dtype)
    crop = img[y0:y0 + piece_h, x0:x0 + piece_w]
    # Paste the part of the page at the top-left of the piece.
    piece[:crop.shape[0], :crop.shape[1]] = crop
    return piece


def shrunk_size(height, width):
    """Return (new_width, new_height) when the long side is shrunk to SHRINK_LONG (at least 1 px)."""
    scale = SHRINK_LONG / max(height, width)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def shift_boxes(boxes, x0, y0):
    """Move boxes (x1, y1, x2, y2, score) from a piece's own pixels to page pixels."""
    return [(x1 + x0, y1 + y0, x2 + x0, y2 + y0, s) for x1, y1, x2, y2, s in boxes]


def scale_boxes(boxes, fx, fy):
    """Scale boxes (x1, y1, x2, y2, score) by fx in x and fy in y (shrunk copy back to page size)."""
    return [(x1 * fx, y1 * fy, x2 * fx, y2 * fy, s) for x1, y1, x2, y2, s in boxes]
