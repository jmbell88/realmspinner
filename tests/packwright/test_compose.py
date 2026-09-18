"""Painting the atlas, and the gutter.

Extrude is the interesting half: it exists so a filtered texture sampling just
past a sprite's edge finds that sprite's own colour, and it is wrong in a way
that only shows on a GPU at some zoom levels -- which is exactly why it is
asserted pixel by pixel here.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.modes.packwright.engine import compose
from warlock.studio.modes.packwright.engine.layout import Frame, Layout, PackSettings, layout
from warlock.studio.modes.packwright.engine.sources import Sprite

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _solid(key: str, w: int, h: int, colour) -> Sprite:
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    pixels[...] = colour
    return Sprite(key=key, name=key, pixels=pixels)


def test_each_sprite_lands_exactly_where_its_frame_says():
    sprites = [_solid("a", 6, 4, RED), _solid("b", 6, 4, BLUE)]
    result = layout(sprites, PackSettings(power_of_two=False, padding=2))
    atlas = compose.compose(sprites, result)
    assert atlas.shape == (result.height, result.width, 4)
    for frame, colour in ((result.frame("a"), RED), (result.frame("b"), BLUE)):
        block = atlas[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w]
        assert (block == np.array(colour, np.uint8)).all()


def test_only_the_trimmed_region_is_pasted():
    """MaxRects, because a *grid* never trims: a tile moved to its own bounding
    box no longer sits where the arithmetic that slices the tileset says."""
    pixels = np.zeros((10, 10, 4), dtype=np.uint8)
    pixels[4:6, 3:7] = RED
    sprites = [Sprite(key="a", name="a", pixels=pixels)]
    result = layout(sprites, PackSettings(mode="maxrects", power_of_two=False))
    atlas = compose.compose(sprites, result)
    frame = result.frames[0]
    assert (frame.w, frame.h) == (4, 2)
    assert (atlas[frame.y : frame.y + 2, frame.x : frame.x + 4] == np.array(RED, np.uint8)).all()
    assert int(atlas[..., 3].sum()) == 4 * 2 * 255


def test_a_grid_cell_keeps_a_tile_where_the_artist_drew_it():
    """The whole point of the mode. Trimming used to move the content to the
    cell's top-left, so a 10px tile whose art sat three pixels in came out three
    pixels up and left of every neighbour -- and a ``.tsx`` slices by
    arithmetic and cannot know (the 2026-09-02 review, section 7)."""
    pixels = np.zeros((10, 10, 4), dtype=np.uint8)
    pixels[4:6, 3:7] = RED
    sprites = [Sprite(key="a", name="a", pixels=pixels)]
    result = layout(sprites, PackSettings(mode="grid", trim=True, power_of_two=False))
    atlas = compose.compose(sprites, result)
    frame = result.frames[0]

    assert (frame.w, frame.h) == (10, 10), "the cell is the whole tile"
    assert frame.trim == (0, 0, 10, 10)
    block = atlas[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w]
    assert (block[4:6, 3:7] == np.array(RED, np.uint8)).all()
    assert int(block[..., 3].sum()) == 4 * 2 * 255


def test_the_gutter_is_transparent_without_extrude():
    sprites = [_solid("a", 4, 4, RED), _solid("b", 4, 4, BLUE)]
    result = layout(sprites, PackSettings(power_of_two=False, padding=2, extrude=0))
    atlas = compose.compose(sprites, result)
    assert int(atlas[..., 3].sum()) == 2 * 4 * 4 * 255


def test_extrude_replicates_the_border_outward_including_the_corners():
    """The corners come free by extruding vertically *after* the side columns
    are written, which is why the order in ``_extrude`` matters."""
    sprites = [_solid("a", 4, 4, RED)]
    result = layout(sprites, PackSettings(power_of_two=False, padding=4, extrude=2))
    atlas = compose.compose(sprites, result)
    frame = result.frames[0]
    grown = atlas[frame.y - 2 : frame.y + frame.h + 2, frame.x - 2 : frame.x + frame.w + 2]
    assert grown.shape == (8, 8, 4)
    assert (grown == np.array(RED, np.uint8)).all()


def test_extrude_never_crosses_into_a_neighbour():
    """The whole point of ``padding >= extrude * 2``: two sprites extruding
    into one gutter must not meet."""
    sprites = [_solid(f"s{i}", 8, 8, RED if i % 2 else BLUE) for i in range(9)]
    result = layout(sprites, PackSettings(power_of_two=False, padding=4, extrude=2))
    atlas = compose.compose(sprites, result)
    for frame in result.frames:
        colour = np.array(RED if int(frame.key[1]) % 2 else BLUE, np.uint8)
        grown = atlas[frame.y - 2 : frame.y + frame.h + 2, frame.x - 2 : frame.x + frame.w + 2]
        assert (grown == colour).all(), f"{frame.key} was reached by a neighbour"


def test_extrude_stays_inside_the_atlas_in_both_modes():
    """A sprite against the edge of the atlas has to have room too, which is
    the case an un-offset MaxRects pack gets wrong -- and getting it wrong here
    is an out-of-bounds write rather than a cosmetic artefact."""
    sprites = [_solid(f"s{i:02d}", 7, 5, RED) for i in range(12)]
    for mode in ("grid", "maxrects"):
        settings = PackSettings(mode=mode, padding=4, extrude=2, power_of_two=False)
        result = layout(sprites, settings)
        atlas = compose.compose(sprites, result)  # must not raise
        assert atlas.shape[:2] == (result.height, result.width)


def test_a_blank_sprite_composites_as_nothing():
    blank = Sprite(key="b", name="b", pixels=np.zeros((6, 6, 4), np.uint8))
    sprites = [_solid("a", 4, 4, RED), blank]
    result = layout(sprites, PackSettings(power_of_two=False))
    atlas = compose.compose(sprites, result)
    assert int(atlas[..., 3].sum()) == 4 * 4 * 255


def test_a_frame_naming_a_missing_sprite_raises():
    """An atlas with an invisible gap looks like an art problem and sends the
    user looking in the wrong place -- the rule ``sheet.pack`` already follows
    for a missing rendered frame."""
    sprites = [_solid("a", 4, 4, RED)]
    result = layout(sprites, PackSettings(power_of_two=False))
    with pytest.raises(ValueError, match="no sprite for packed frame"):
        compose.compose([], result)


def test_png_bytes_round_trips_the_atlas():
    from PIL import Image

    sprites = [_solid("a", 4, 4, RED)]
    result = layout(sprites, PackSettings(power_of_two=False))
    atlas = compose.compose(sprites, result)
    import io

    decoded = np.asarray(Image.open(io.BytesIO(compose.png_bytes(atlas))).convert("RGBA"))
    assert np.array_equal(decoded, atlas)


def test_two_composites_of_one_layout_are_identical():
    sprites = [_solid(f"s{i}", 5, 5, RED) for i in range(6)]
    result = layout(sprites, PackSettings())
    assert np.array_equal(compose.compose(sprites, result), compose.compose(sprites, result))


# --- packwright-03 (2026-09-11 audit): the margin guard must survive -O -----


def test_extrude_refuses_a_margin_violating_frame_even_under_dash_o():
    """The only defence against a ``Frame`` whose x/y sit closer to the atlas
    edge than its own extrude margin used to be a bare ``assert``, which
    ``python -O``/``PYTHONOPTIMIZE`` compiles out -- silently removing the one
    check standing between a margin-violating frame and a negative-index
    wraparound that overwrites real pixels outside the frame's own footprint.
    ``PackSettings`` and both packers already guarantee the room, so the only
    way to reach this is a hand-built ``Layout``, which is what this does.

    Asserts the *type*: a bare ``assert`` also raises here (``AssertionError``,
    not compiled out in an ordinary test run), so a test that only checked
    "raises something" would pass against the unfixed code too."""
    sprites = [_solid("a", 4, 4, RED)]
    frame = Frame(
        key="a", name="a", x=1, y=1, w=4, h=4, trim=(0, 0, 4, 4), source_w=4, source_h=4
    )
    bad_layout = Layout(width=8, height=8, mode="maxrects", padding=4, extrude=2, frames=(frame,))
    with pytest.raises(ValueError, match="no room for its"):
        compose.compose(sprites, bad_layout)


def test_studio_packwright_states_its_invariants_without_assert():
    """``assert`` is compiled out under ``python -O``, so a guard written that
    way vanishes and the failure it was catching becomes silent corruption
    instead of a loud one. The durable half of packwright-03's fix: the
    behavioural test above only proves *this build's* guard raises
    ``ValueError``; it says nothing about whether the *next* one that reaches
    for a quick invariant check reaches for ``assert`` again. Mirrors
    ``tests/inker/test_inker_document.py::test_the_engine_states_its_invariants_without_assert``,
    scoped to Packwright's engine -- that scan is scoped to Inker's and does
    not reach this package.

    The root is derived from the package and the glob must find files: a
    literal ``studio/packwright`` path went on globbing an empty directory
    after restructure P6 moved the engine, and passed."""
    import pathlib

    root = pathlib.Path(compose.__file__).parent
    paths = sorted(root.glob("*.py"))
    assert len(paths) >= 8, f"only {len(paths)} engine modules under {root}"
    offenders = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("assert "):
                offenders.append(f"{path.name}:{number}")
    assert offenders == [], offenders
