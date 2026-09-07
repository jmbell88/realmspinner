"""``inker/nineslice.py``: inferring a centre, and turning one into pixels.

Neighbours in shape: ``test_slice_overlay.py`` (numbers checked against a
hand-built page, not against the helper that produced them) and
``test_sheet_slices.py`` (plain arithmetic, no ``Document`` where a bare array
says the same thing). Nothing here needs a ``Document`` at all -- :func:`fit`,
:func:`stretch` and :func:`ninepatch` take a plane and a rectangle, the same
inputs :meth:`~warlock.studio.inker.document.Document.flatten` and a slice's
own ``bounds``/``center`` already are.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path

import numpy as np
import pytest

from warlock.studio.inker import nineslice

CORNER = (10, 20, 30, 255)
EDGE_H = (40, 50, 60, 255)  # the top/bottom border -- repeats along x
EDGE_V = (70, 80, 90, 255)  # the left/right border -- repeats along y
MIDDLE = (100, 110, 120, 255)


def _panel(w: int = 6, h: int = 6, border: int = 2) -> np.ndarray:
    """A hand-built panel frame: a ``border``-px ring, flat inside it.

    Built from four constants rather than random noise, so every assertion
    below can name the exact pixel it expects -- "the corner colour", not
    "whatever ``panel[0, 0]`` happens to hold".
    """
    panel = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        top, bottom = y < border, y >= h - border
        for x in range(w):
            left, right = x < border, x >= w - border
            if (top or bottom) and (left or right):
                panel[y, x] = CORNER
            elif top or bottom:
                panel[y, x] = EDGE_H
            elif left or right:
                panel[y, x] = EDGE_V
            else:
                panel[y, x] = MIDDLE
    return panel


def _gradient(w: int = 6, h: int = 6) -> np.ndarray:
    """No two adjacent columns match and no two adjacent rows match.

    ``x`` and ``y`` each ride a different channel at a step no rounding could
    collapse, so this is a plane with *no* constant run on either axis -- the
    refusal case, not a coincidence of one particular size.
    """
    panel = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            panel[y, x] = ((x * 40) % 256, (y * 40) % 256, ((x + y) * 20) % 256, 255)
    return panel


# --- fit ------------------------------------------------------------------


def test_fit_finds_the_known_centre_of_a_hand_built_panel_frame():
    panel = _panel()
    assert nineslice.fit(panel, (0, 0, 6, 6)) == (2, 2, 4, 4)


def test_fit_returns_none_on_a_gradient_rather_than_guessing():
    """No constant run on either axis, so a guessed rectangle would be a lie
    about art that has no stretchable middle at all."""
    panel = _gradient()
    assert nineslice.fit(panel, (0, 0, 6, 6)) is None


def test_fit_reads_only_the_bounds_it_is_given():
    """A slice's own rectangle inside a bigger canvas -- ``fit`` must not see,
    and must not be thrown by, whatever is drawn outside it."""
    canvas = _gradient(12, 12)
    canvas[3:9, 3:9] = _panel()
    assert nineslice.fit(canvas, (3, 3, 9, 9)) == (2, 2, 4, 4)


# --- stretch ----------------------------------------------------------------


def test_stretch_leaves_the_four_corners_bit_identical():
    panel = _panel()
    center = (2, 2, 4, 4)
    result = nineslice.stretch(panel, (0, 0, 6, 6), center, 12, 10)
    assert np.array_equal(result[0:2, 0:2], panel[0:2, 0:2])  # top-left
    assert np.array_equal(result[0:2, -2:], panel[0:2, -2:])  # top-right
    assert np.array_equal(result[-2:, 0:2], panel[-2:, 0:2])  # bottom-left
    assert np.array_equal(result[-2:, -2:], panel[-2:, -2:])  # bottom-right


def test_stretch_returns_exactly_the_requested_size():
    panel = _panel()
    result = nineslice.stretch(panel, (0, 0, 6, 6), (2, 2, 4, 4), 17, 23)
    assert result.shape[:2] == (23, 17)


def test_stretch_tiles_the_middle_rather_than_blending_it():
    """Every pixel in the stretched middle is one of the source middle's own
    colours -- never a value in between, which is what a resample would have
    produced instead of a repeat."""
    panel = _panel()
    result = nineslice.stretch(panel, (0, 0, 6, 6), (2, 2, 4, 4), 20, 20)
    middle = result[2:-2, 2:-2]
    assert np.all(middle == np.array(MIDDLE, dtype=np.uint8))


def test_stretch_refuses_a_target_smaller_than_the_fixed_corners():
    """The corners alone are 2px + 2px = 4px wide; asking for 3 has no honest
    answer, so this must raise rather than silently clip a corner."""
    panel = _panel()
    with pytest.raises(ValueError, match="corners"):
        nineslice.stretch(panel, (0, 0, 6, 6), (2, 2, 4, 4), 3, 10)


def test_stretch_at_the_source_size_is_the_identity():
    """The 1x case a live preview always draws first: asking for exactly the
    slice's own size must hand back exactly its own pixels."""
    panel = _panel()
    result = nineslice.stretch(panel, (0, 0, 6, 6), (2, 2, 4, 4), 6, 6)
    assert np.array_equal(result, panel)


# --- ninepatch ----------------------------------------------------------------


def test_ninepatch_round_trip_through_a_real_png():
    """Write the guide format, encode it, decode it back, and read off exactly
    what Android's own tool would: a 1px transparent ring, the source pixels
    inside it, and black marks over the stretch region on the top and left
    edges alone."""
    from PIL import Image

    panel = _panel()
    center = (2, 2, 4, 4)
    built = nineslice.ninepatch(panel, (0, 0, 6, 6), center)
    assert built.shape == (8, 8, 4)

    buffer = io.BytesIO()
    Image.fromarray(built, "RGBA").save(buffer, "PNG")
    buffer.seek(0)
    round_tripped = np.array(Image.open(buffer).convert("RGBA"))

    assert np.array_equal(round_tripped, built)
    # The source pixels, ringed by one transparent pixel.
    assert np.array_equal(round_tripped[1:-1, 1:-1], panel)
    assert tuple(round_tripped[0, 0]) == (0, 0, 0, 0)
    assert tuple(round_tripped[-1, -1]) == (0, 0, 0, 0)
    # The stretch marks: opaque black over columns/rows [2, 4), transparent
    # everywhere else on the guide edges.
    top_row = round_tripped[0, 1:-1, 3]
    assert list(top_row) == [0, 0, 255, 255, 0, 0]
    left_col = round_tripped[1:-1, 0, 3]
    assert list(left_col) == [0, 0, 255, 255, 0, 0]
    # No padding guide is invented on the bottom/right edges.
    assert np.all(round_tripped[-1, :, 3] == 0)
    assert np.all(round_tripped[:, -1, 3] == 0)


# --- the import pin -----------------------------------------------------------


def test_the_module_reaches_for_nothing_outside_numpy_and_its_own_package():
    """``tests/inker/test_inker_imports.py`` pins the whole package's outward
    imports and already globs this file in -- it does not need a second
    listing here to keep passing. What is checked here is the *reason* it
    keeps passing: this module's own outward imports are numpy and nothing
    else, so it has nothing to add to that pin's exact-match set.
    """
    tree = ast.parse(Path(nineslice.__file__).read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.add((node.module or "").split(".")[0])
    assert roots == {"numpy", "__future__"}
