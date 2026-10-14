"""The palette half of ``pixelsheet``, headless: one palette across the atlas.

The band-restyle helpers this file once covered (banding, crop-back, remask,
downscale, the pixel sidecar) were deleted with the sheet pipeline that drove
them: the 2026-10-14 audit, finding inker-80. What stays is the property that
makes eight directions agree on a colour: one palette, chosen once across the
whole atlas, whether the median cut or an authored file supplies it.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from realmspinner.pipelines import pixelsheet


def _meta(frame_size=128, columns=8, rows=1, cells=None):
    return {
        "id": "abcdef123456",
        "name": "sheet",
        "source_job": "job1",
        "frame_size": frame_size,
        "columns": columns,
        "rows": rows,
        "width": frame_size * columns,
        "height": frame_size * rows,
        "elevation": 30.0,
        "lighting": "flat",
        "yaws": [i * 45.0 for i in range(columns)],
        "poses": [{"id": None, "name": "rest"}],
        "cells": cells if cells is not None else [],
    }


def _atlas(meta, colour=(200, 60, 60)):
    """A source render: an opaque blob in each cell on transparency."""
    atlas = Image.new("RGBA", (meta["width"], meta["height"]), (0, 0, 0, 0))
    frame = meta["frame_size"]
    for row in range(meta["rows"]):
        for column in range(meta["columns"]):
            block = Image.new("RGBA", (frame // 2, frame // 2), (*colour, 255))
            atlas.paste(block, (column * frame + frame // 4, row * frame + frame // 4))
    return atlas


# --- the shared palette --------------------------------------------------------


def test_one_palette_is_chosen_across_the_whole_atlas():
    """Per-cell quantization is how the same shirt comes out two shades in two
    directions, which is what reads as flicker when a character turns."""
    meta = _meta(columns=4)
    atlas = Image.new("RGBA", (meta["width"], meta["height"]), (0, 0, 0, 0))
    frame = meta["frame_size"]
    # Four cells, each a *slightly* different red: a per-cell quantize would
    # give each its own entry, a shared one collapses them.
    for column in range(4):
        block = Image.new("RGBA", (frame, frame), (200 + column, 60, 60, 255))
        atlas.paste(block, (column * frame, 0))

    out, palette = pixelsheet.quantize_shared(atlas, 2)

    assert len(palette) <= 2
    arr = np.asarray(out)
    opaque = arr[:, :, 3] > 0
    assert len({tuple(p) for p in arr[:, :, :3][opaque]}) <= 2
    assert all(entry.startswith("#") and len(entry) == 7 for entry in palette)


def test_quantized_alpha_is_binary():
    meta = _meta(columns=2)
    out, _palette = pixelsheet.quantize_shared(_atlas(meta), 8)
    assert set(np.unique(np.asarray(out)[:, :, 3])) <= {0, 255}


def test_an_empty_atlas_says_so_rather_than_quantizing_nothing():
    with pytest.raises(ValueError, match="empty"):
        pixelsheet.quantize_shared(Image.new("RGBA", (32, 32), (0, 0, 0, 0)), 8)


def test_quantize_shared_refuses_a_non_positive_colour_count():
    """The 2026-09-18 audit (inker-05): an unchecked ``colors`` reached
    ``Image.quantize`` verbatim, so a bad value surfaced as Pillow's own
    error rather than a refusal this app names."""
    atlas = Image.new("RGBA", (8, 8), (10, 20, 30, 255))
    for bad in (0, -1, 257):
        with pytest.raises(ValueError, match="1 to 256"):
            pixelsheet.quantize_shared(atlas, bad)


# --- where a palette comes from -------------------------------------------------


RAMP = ((16, 16, 24), (72, 56, 48), (168, 120, 88), (232, 216, 184))


def test_a_designed_palette_is_returned_verbatim_and_never_requantised():
    """The whole point of naming a palette: those colours, unchanged.

    A median cut run over the atlas "as well" would silently substitute the
    render's own muddy colours for the ones the artist chose, which is the
    failure an authored ramp exists to fix.
    """
    entries, source = pixelsheet.resolve_palette(
        _atlas(_meta(columns=2)), colors=8, entries=RAMP
    )
    assert source == "designed"
    assert entries == RAMP


def test_a_designed_palette_is_taken_without_looking_at_the_atlas():
    """No pixels are read on that branch, so an atlas ``quantize_shared`` would
    refuse outright is still fine to map with a palette."""
    empty = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    entries, source = pixelsheet.resolve_palette(empty, colors=8, entries=RAMP)
    assert (entries, source) == (RAMP, "designed")


def test_with_no_palette_the_median_cut_becomes_the_palette_source():
    meta = _meta(columns=2)
    entries, source = pixelsheet.resolve_palette(_atlas(meta), colors=4, entries=None)
    assert source == "derived"
    assert 0 < len(entries) <= 4
    assert all(len(e) == 3 and all(0 <= c <= 255 for c in e) for e in entries)


def test_the_derived_branch_is_exactly_the_hex_list_parsed_back():
    """The collapse the retired Troupe worker was refactored onto, pinned: the
    triples are ``quantize_shared``'s own answer and nothing else."""
    atlas = _atlas(_meta(columns=2))
    _mapped, hexes = pixelsheet.quantize_shared(atlas, 8)
    old = tuple(
        (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in hexes
    )
    entries, source = pixelsheet.resolve_palette(atlas, colors=8, entries=None)
    assert (entries, source) == (old, "derived")


def test_an_empty_tuple_is_a_palette_of_no_colours_and_not_an_absent_one():
    """``None`` is how a caller says "I have no palette". ``()`` is a palette
    file somebody emptied, and passing it through is what lets
    ``map_palette``'s "an empty palette maps nothing" be the refusal the user
    sees rather than a silent median cut."""
    entries, source = pixelsheet.resolve_palette(
        _atlas(_meta(columns=2)), colors=8, entries=()
    )
    assert (entries, source) == ((), "designed")


def test_the_shared_property_survives_a_designed_palette():
    """The insight ``resolve_palette``'s docstring carries, as a test.

    What made ``quantize_shared`` shared was never the median cut -- it is that
    one entry set is applied over the whole atlas in one pass. So four cells
    that differ by a hair still collapse onto the same entries when the palette
    is designed, exactly as they do when it is derived.
    """
    from realmspinner.pipelines import pixel

    meta = _meta(columns=4)
    atlas = Image.new("RGBA", (meta["width"], meta["height"]), (0, 0, 0, 0))
    frame = meta["frame_size"]
    for column in range(4):
        block = Image.new("RGBA", (frame, frame), (200 + column, 60, 60, 255))
        atlas.paste(block, (column * frame, 0))

    entries, _ = pixelsheet.resolve_palette(atlas, colors=8, entries=RAMP)
    mapped = np.asarray(pixel.map_palette(atlas, entries))
    cells = [
        {tuple(p) for p in mapped[:, column * frame : (column + 1) * frame, :3]
         .reshape(-1, 3).tolist()}
        for column in range(4)
    ]
    assert cells[0] == cells[1] == cells[2] == cells[3]
