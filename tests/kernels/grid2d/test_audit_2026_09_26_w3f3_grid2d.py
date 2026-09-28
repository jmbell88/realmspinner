"""Closing plotter-tiles-01 and plotter-tiles-02 from the 2026-09-26 audit.

One test per finding, named for the claim it proves. Both failed against the
unfixed code before their matching fix landed -- proven by loading the pre-fix
source from ``git show HEAD:<path>`` as a throwaway package copy in the
fixer's scratchpad and running it there, never by checking the old text out
over the tree (``dev/INVARIANTS.md``'s rule). See the fixer's own return for
the pasted failing output.
"""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.grid2d import picking
from realmspinner.kernels.grid2d.tileset import TileRect, Tileset


def _pixels(w: int, h: int) -> np.ndarray:
    array = np.zeros((h, w, 4), dtype=np.uint8)
    array[..., 3] = 255
    return array


# --- plotter-tiles-01: a partial final column is dropped, not refused ------


def test_tileset_docstring_no_longer_claims_a_partial_final_column_is_refused():
    """The module docstring said a spacing that leaves a partial final column
    is refused at construction; :attr:`Tileset.columns`/``.rows`` floor-divide
    and simply drop it instead -- ``test_a_partial_final_column_is_not_a_tile``
    (this module's neighbour, ``test_tileset.py``) already pins the drop, so
    only the doc text was wrong."""
    import realmspinner.kernels.grid2d.tileset as tileset_module

    assert tileset_module.__doc__ is not None
    flat = " ".join(tileset_module.__doc__.split())
    assert (
        "a margin larger than the image, a spacing that leaves a "
        "partial final column -- is a file or a form the user got wrong"
    ) not in flat


def test_a_tileset_whose_spacing_leaves_a_partial_column_is_still_constructed():
    """The behaviour the corrected docstring now describes: a partial final
    column costs nothing but its own width, it is dropped, and the tileset is
    still built rather than refused."""
    # 3 tiles of 16px plus 4px of spacing after each needs 3*16 + 3*4 = 60px to
    # fit a 4th; giving it 68px leaves an 8px sliver that is not a whole tile.
    ts = Tileset(name="t", pixels=_pixels(68, 16), tile_w=16, tile_h=16, spacing=4)
    assert ts.columns == 3


# --- plotter-tiles-02: resized() re-clamps after the minimum-side push -----


def test_resized_does_not_push_a_subpixel_shape_outside_the_tile():
    """A shape pinned within ``minimum`` of an edge -- a sub-pixel ``w`` -- used
    to have its moving edge pushed past 0 by the minimum-side enforcement, with
    nothing re-clamping the result to the tile the shape is drawn on."""
    shape = TileRect(x=0.0, y=0.0, w=0.3, h=0.3)
    out = picking.resized(shape, "w", (0.9, 0.0), 16, 16)
    x, y, w, h = picking.bounds(out)
    assert x >= 0.0
    assert x + w <= 16.0


def test_resized_does_not_push_a_subpixel_shape_past_the_far_edge():
    """The mirror case: a shape pinned within ``minimum`` of the tile's far
    edge, dragged with the opposite handle."""
    shape = TileRect(x=15.7, y=0.0, w=0.3, h=0.3)
    out = picking.resized(shape, "e", (15.1, 0.0), 16, 16)
    x, y, w, h = picking.bounds(out)
    assert x >= 0.0
    assert x + w <= 16.0


def test_resized_still_clamps_top_and_bottom_the_same_way():
    shape = TileRect(x=0.0, y=0.0, w=0.3, h=0.3)
    out = picking.resized(shape, "n", (0.0, 0.9), 16, 16)
    x, y, w, h = picking.bounds(out)
    assert y >= 0.0
    assert y + h <= 16.0
