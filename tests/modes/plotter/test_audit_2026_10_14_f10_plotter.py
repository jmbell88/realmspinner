"""The 2026-10-14 audit, fixer F10: a Wang brush paints only accounted-for gids."""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.grid2d import gid as gidlib
from realmspinner.kernels.grid2d.tileset import Tileset, TilesetRef
from realmspinner.kernels.grid2d.wang import WangColour, WangSet
from realmspinner.studio.modes.plotter.engine import terrain


def _pixels(size: int = 32) -> np.ndarray:
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., 3] = 255
    return out


def test_a_wang_set_naming_a_tile_the_tileset_does_not_have_never_paints_an_unaccounted_gid() -> (
    None
):
    # Four tiles (local 0..3); the set's only full-colour tile is "500".
    wangset = WangSet(
        name="W",
        kind="corner",
        colours=(WangColour("a", "#00ff00"),),
        tiles={500: (0, 1, 0, 1, 0, 1, 0, 1), -2: (0, 1, 0, 1, 0, 1, 0, 1)},
    )
    tileset = Tileset(
        name="t", pixels=_pixels(), tile_w=16, tile_h=16, wangsets=(wangset,)
    )
    ref = TilesetRef(firstgid=1, tileset=tileset)
    data = gidlib.empty_layer(5, 5)

    region = terrain.paint_wang(data, 2, 2, 1, ref, tileset.wangsets[0])

    if region is not None:
        _, _, block = region
        for value in np.asarray(block).ravel():
            gid = int(value) & gidlib.GID_MASK
            assert gid == 0 or ref.holds(gid), f"painted gid {gid} nobody accounts for"


def test_the_brush_refuses_a_wang_set_that_was_never_trimmed_by_the_tileset() -> None:
    """``paint_wang`` takes its set as an argument, so it must not trust that the
    tileset already trimmed it."""
    wangset = WangSet(
        name="W",
        kind="corner",
        colours=(WangColour("a", "#00ff00"),),
        tiles={500: (0, 1, 0, 1, 0, 1, 0, 1)},
    )
    ref = TilesetRef(
        firstgid=1, tileset=Tileset(name="t", pixels=_pixels(), tile_w=16, tile_h=16)
    )
    data = gidlib.empty_layer(5, 5)

    assert terrain.paint_wang(data, 2, 2, 1, ref, wangset) is None
    assert not np.asarray(data).any()
