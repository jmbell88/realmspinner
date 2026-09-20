"""``inker/nineslice.py``: inferring a centre, and turning one into pixels.

Neighbours in shape: ``test_slice_overlay.py`` (numbers checked against a
hand-built page, not against the helper that produced them) and
``test_sheet_slices.py`` (plain arithmetic, no ``Document`` where a bare array
says the same thing). Nothing here needs a ``Document`` at all -- :func:`fit`,
:func:`stretch` and :func:`ninepatch` take a plane and a rectangle, the same
inputs :meth:`~realmspinner.kernels.pixel.document.Document.flatten` and a slice's
own ``bounds``/``center`` already are.
"""

from __future__ import annotations

import ast
import io
import time
from pathlib import Path

import numpy as np
import pytest

from realmspinner.kernels.pixel import nineslice

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


# --- _matching_run parity and perf, batch 11 --------------------------------
#
# The 2026-09-17 native-kernel review (batch 11) replaced the per-column/
# per-row Python loop (one ``np.array_equal`` call per adjacent pair) with a
# single vectorised reduction plus a run-length decode over its boolean
# result. These tests hold a verbatim copy of the old loop for parity, on
# cases chosen to exercise its tie-break and edge-exclusion rules directly.


def _old_matching_run(panel: np.ndarray, *, axis: int) -> tuple[int, int] | None:
    """``nineslice._matching_run`` exactly as it stood before batch 11."""
    size = panel.shape[1] if axis == 1 else panel.shape[0]
    if size < 4:
        return None
    best: tuple[int, int] | None = None
    run_start: int | None = None
    for i in range(1, size - 2):
        line = panel[:, i] if axis == 1 else panel[i, :]
        neighbour = panel[:, i + 1] if axis == 1 else panel[i + 1, :]
        if np.array_equal(line, neighbour):
            if run_start is None:
                run_start = i
            end = i + 2
            if best is None or (end - run_start) > (best[1] - best[0]):
                best = (run_start, end)
        else:
            run_start = None
    return best


def _row(*values: int, channels: int = 4) -> list[int]:
    return list(values[:channels]) + [255] * max(0, channels - len(values))


def test_matching_run_matches_old_loop_on_noise_with_no_run():
    rng = np.random.default_rng(1)
    panel = rng.integers(0, 255, size=(6, 8, 4)).astype(np.uint8)
    assert nineslice._matching_run(panel, axis=1) == _old_matching_run(panel, axis=1) is None


def test_matching_run_matches_old_loop_when_every_column_is_identical():
    panel = np.tile(np.array([1, 2, 3, 4], dtype=np.uint8), (6, 8, 1))
    assert nineslice._matching_run(panel, axis=1) == _old_matching_run(panel, axis=1)


def test_matching_run_matches_old_loop_on_one_interior_run():
    # 8 columns: 0 and 7 are the corners; columns 2,3,4 share one colour.
    panel = np.zeros((6, 8, 4), dtype=np.uint8)
    rng = np.random.default_rng(2)
    panel[:] = rng.integers(0, 255, size=(6, 1, 4))
    panel[:, 2:5] = [10, 20, 30, 255]
    assert nineslice._matching_run(panel, axis=1) == _old_matching_run(panel, axis=1) == (2, 5)


def test_matching_run_matches_old_loop_on_two_runs_of_different_lengths():
    # Columns [1,2] match (length-2 run) and [4,5,6] match (length-3 run) --
    # the wider one must win on both implementations.
    panel = np.zeros((6, 9, 4), dtype=np.uint8)
    rng = np.random.default_rng(3)
    for i in range(9):
        panel[:, i] = rng.integers(0, 255, size=4)
    panel[:, 1] = panel[:, 2] = [5, 5, 5, 255]
    panel[:, 4] = panel[:, 5] = panel[:, 6] = [9, 9, 9, 255]
    old = _old_matching_run(panel, axis=1)
    new = nineslice._matching_run(panel, axis=1)
    assert old == new == (4, 7)


def test_matching_run_matches_old_loop_on_a_tie_the_leftmost_run_wins():
    # Two equal-length runs: [1,2] and [5,6]. The old loop's strict ``>``
    # keeps whichever it saw first -- the leftmost.
    panel = np.zeros((6, 9, 4), dtype=np.uint8)
    rng = np.random.default_rng(4)
    for i in range(9):
        panel[:, i] = rng.integers(0, 255, size=4)
    panel[:, 1] = panel[:, 2] = [7, 7, 7, 255]
    panel[:, 5] = panel[:, 6] = [8, 8, 8, 255]
    old = _old_matching_run(panel, axis=1)
    new = nineslice._matching_run(panel, axis=1)
    assert old == new == (1, 3)


def test_matching_run_matches_old_loop_when_the_run_touches_the_edge():
    # Columns [0,1] match, reaching the very first column -- interior-only
    # scanning must not count it, on both implementations.
    panel = np.zeros((6, 8, 4), dtype=np.uint8)
    rng = np.random.default_rng(5)
    for i in range(8):
        panel[:, i] = rng.integers(0, 255, size=4)
    panel[:, 0] = panel[:, 1] = [3, 3, 3, 255]
    old = _old_matching_run(panel, axis=1)
    new = nineslice._matching_run(panel, axis=1)
    assert old == new is None


def test_matching_run_matches_old_loop_on_a_single_channel_panel():
    rng = np.random.default_rng(6)
    panel = rng.integers(0, 255, size=(6, 9)).astype(np.uint8)
    panel[:, 3] = panel[:, 4] = panel[:, 5] = 42
    old = _old_matching_run(panel, axis=1)
    new = nineslice._matching_run(panel, axis=1)
    assert old == new == (3, 6)
    # And along rows too, for the same panel.
    old_r = _old_matching_run(panel, axis=0)
    new_r = nineslice._matching_run(panel, axis=0)
    assert old_r == new_r


def test_matching_run_matches_old_loop_on_a_non_multiple_of_4_width():
    rng = np.random.default_rng(7)
    panel = rng.integers(0, 255, size=(6, 11, 4)).astype(np.uint8)
    panel[:, 4] = panel[:, 5] = panel[:, 6] = [1, 1, 1, 255]
    old = _old_matching_run(panel, axis=1)
    new = nineslice._matching_run(panel, axis=1)
    assert old == new == (4, 7)


def test_matching_run_matches_old_loop_on_random_panels():
    """Broad coverage beyond the hand-picked cases above."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        h = int(rng.integers(1, 10))
        w = int(rng.integers(1, 10))
        c = int(rng.choice([1, 2, 4]))
        shape = (h, w, c) if rng.random() < 0.8 else (h, w)
        panel = rng.integers(0, 3, size=shape).astype(np.uint8)
        for axis in (0, 1):
            old = _old_matching_run(panel, axis=axis)
            new = nineslice._matching_run(panel, axis=axis)
            assert old == new, f"axis={axis} shape={shape}\n{panel}"


def test_matching_run_on_a_wide_panel_finishes_well_under_the_old_time():
    """The measured case: a wide, short panel forces many adjacent-pair
    comparisons along the guide-drag axis, each touching very little data --
    exactly the shape where the old per-pair ``np.array_equal`` loop paid
    call overhead thousands of times over. The 2026-09-17 native-kernel
    review (batch 11) measured the old loop at ~15 ms here against ~0.4 ms
    for the vectorised run-length pass -- a ~37x gap. 6 ms fails the unfixed
    loop by more than 2x and leaves the fixed pass 15x of room, because a
    sub-millisecond bound is what a loaded xdist worker trips on."""
    h, w = 6, 8192
    rng = np.random.default_rng(0)
    panel = rng.integers(0, 255, size=(h, w, 4)).astype(np.uint8)
    border = 2
    # Every row alike across the whole interior band -- a genuine full-height
    # matching run from ``border`` to ``w - border``, exercising the actual
    # "widest run" search rather than an immediate refusal.
    panel[:, border:-border] = [11, 22, 33, 255]

    start = time.perf_counter()
    result = nineslice._matching_run(panel, axis=1)
    elapsed = time.perf_counter() - start

    assert result is not None
    assert elapsed < 0.006, f"took {elapsed:.4f}s -- still the old per-column loop?"


def test_the_module_reaches_for_nothing_outside_numpy_and_its_own_package():
    """``tests/modes/inker/test_inker_imports.py`` pins the whole package's outward
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
