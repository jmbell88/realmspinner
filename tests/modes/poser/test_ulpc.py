"""The example sheets as free regression oracles.

Every number here was measured off ``examples/*.png`` first and written down
second. They are validation material only -- no ULPC art ships, and these files
stay out of the package and out of any training set (CC-BY-SA/GPL).

If one of these fails, either the reader's layout table drifted or the example
files were replaced with something that is not a ULPC "full" sheet; both are
worth a loud stop, because everything else in this suite that leans on the
sheets is silently decoding the wrong cells from that point on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from warlock.studio.modes.poser.engine import ulpc

# Only the one regression test below needs it, to run this module inside a
# module of its own with the example sheets made to look uninstalled.
pytest_plugins = ["pytester"]

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
SHEETS = ("male_base_spritesheet.png", "female_base_spritesheet.png")

_SHEETS_MISSING = not all((EXAMPLES / name).exists() for name in SHEETS)


@pytest.fixture(scope="module")
def arrays():
    # The 2026-09-15 audit, finding troupe-01: a module-level ``pytestmark``
    # skip used to gate every test below, including the three that read
    # nothing but ``ulpc``'s own layout table -- so ``ulpc.py`` had zero
    # executed coverage on a checkout without the unshipped (CC-BY-SA/GPL)
    # example PNGs. The skip now lives on the one fixture that actually reads
    # them, which only the tests parametrized or dependent on ``arrays`` pull
    # in; the pure layout-table tests never request it and always run.
    if _SHEETS_MISSING:
        pytest.skip("the ULPC reference sheets are not checked out")

    from PIL import Image

    out = {}
    for name in SHEETS:
        with Image.open(EXAMPLES / name) as im:
            out[name] = np.asarray(im.convert("RGBA"))
    return out


def _occupancy(arr) -> list[int]:
    rows, cols = arr.shape[0] // ulpc.CELL, arr.shape[1] // ulpc.CELL
    return [
        sum(
            1
            for c in range(cols)
            if arr[
                r * ulpc.CELL : (r + 1) * ulpc.CELL,
                c * ulpc.CELL : (c + 1) * ulpc.CELL,
                3,
            ].any()
        )
        for r in range(rows)
    ]


def test_the_layout_table_totals_three_hundred_and_fifty_two_cells():
    assert sum(ulpc.row_frame_counts()) == 352
    assert len(ulpc.cells()) == 352


def test_the_table_is_the_published_full_layout():
    """``(7,8,9,6,13) x 4`` + ``6 + 6`` + ``(2,5,3,3,8,2,13,6) x 4``."""
    assert ulpc.row_frame_counts() == (
        7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 6, 6, 6, 6, 13, 13, 13, 13,
        6, 6,
        2, 2, 2, 2, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8,
        2, 2, 2, 2, 13, 13, 13, 13, 6, 6, 6, 6,
    )


@pytest.mark.parametrize("name", SHEETS)
def test_both_sheets_are_the_measured_size_and_grid(arrays, name):
    arr = arrays[name]
    assert (arr.shape[1], arr.shape[0]) == ulpc.SHEET_SIZE
    assert (arr.shape[1] // ulpc.CELL, arr.shape[0] // ulpc.CELL) == (13, 54)


@pytest.mark.parametrize("name", SHEETS)
def test_the_sheet_occupancy_matches_the_layout_table_row_for_row(arrays, name):
    assert tuple(_occupancy(arrays[name])) == ulpc.row_frame_counts()


@pytest.mark.parametrize("name", SHEETS)
def test_there_is_no_antialiasing_anywhere(arrays, name):
    """Every pixel is alpha 0 or 255, which is what makes the sheets losslessly
    indexable -- and is the bar Troupe's own pixeliser has to clear."""
    assert set(np.unique(arrays[name][:, :, 3]).tolist()) == {0, 255}


@pytest.mark.parametrize("name", SHEETS)
def test_sixteen_opaque_colours(arrays, name):
    arr = arrays[name]
    opaque = arr[arr[:, :, 3] == 255][:, :3]
    assert len({tuple(p) for p in opaque.tolist()}) == 16


def _walk_mirror_diffs(arr) -> list[tuple[int, int, int, int, int]]:
    """Per walk frame: ``(count, y_min, y_max, x_min, x_max)`` of the pixels a
    west/east mirror leaves behind."""
    out = []
    for c in range(9):
        west = arr[9 * ulpc.CELL : 10 * ulpc.CELL, c * ulpc.CELL : (c + 1) * ulpc.CELL]
        east = arr[11 * ulpc.CELL : 12 * ulpc.CELL, c * ulpc.CELL : (c + 1) * ulpc.CELL]
        diff = (west != east[:, ::-1]).any(axis=-1)
        ys, xs = np.nonzero(diff)
        out.append(
            (int(diff.sum()), int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
        )
    return out


@pytest.mark.parametrize("name", SHEETS)
def test_west_and_east_walk_frames_are_near_exact_mirrors(arrays, name):
    """The property Phase 6's mirror-assisted cleanup rests on. Row 9 is west
    and row 11 is east; mirroring one onto the other leaves only the face --
    a few dozen pixels out of a 64x64 cell, in one horizontal band."""
    for count, y0, _y1, x0, x1 in _walk_mirror_diffs(arrays[name]):
        assert count <= 50, count
        assert y0 >= 17           # never the shoulders or below
        assert x0 >= 24 and x1 <= 41   # the face, and only the face


def test_the_male_sheet_mirrors_to_the_exact_measured_count(arrays):
    """Written down to the pixel because it is the tightest form of the claim
    and the one the plan quotes; the female sheet's face carries a little more
    asymmetry (37-46) and is covered by the looser assertion above."""
    counts = [d[0] for d in _walk_mirror_diffs(arrays[SHEETS[0]])]
    assert set(counts) == {36, 37}


@pytest.mark.parametrize("name", SHEETS)
def test_no_other_shift_explains_the_difference(arrays, name):
    """The mirror is real facial asymmetry, not a centring offset: every
    non-zero horizontal shift is an order of magnitude worse."""
    arr = arrays[name]
    west = arr[9 * ulpc.CELL : 10 * ulpc.CELL, : ulpc.CELL]
    east = arr[11 * ulpc.CELL : 12 * ulpc.CELL, : ulpc.CELL][:, ::-1]
    at_zero = int((west != east).any(axis=-1).sum())
    for shift in (-1, 1):
        rolled = np.roll(east, shift, axis=1)
        assert int((west != rolled).any(axis=-1).sum()) > at_zero * 4


def test_reading_a_sheet_yields_every_named_cell(arrays):
    from PIL import Image

    with Image.open(EXAMPLES / SHEETS[0]) as im:
        decoded = ulpc.read(im)
    assert len(decoded) == 352
    assert ("walk", "west", 0) in decoded
    assert ("hurt", "hurt", 5) in decoded
    assert all(cell.size == (ulpc.CELL, ulpc.CELL) for cell in decoded.values())


def test_a_sheet_that_is_not_the_full_layout_is_refused():
    from PIL import Image

    with (
        Image.new("RGBA", (64, 64)) as im,
        pytest.raises(ValueError, match="full sheet is 832x3456"),
    ):
        ulpc.read(im)


def test_male_and_female_differ_in_almost_every_cell(arrays):
    male, female = arrays[SHEETS[0]], arrays[SHEETS[1]]
    differing = sum(
        1
        for c in ulpc.cells()
        if (
            male[c.box[1] : c.box[3], c.box[0] : c.box[2]]
            != female[c.box[1] : c.box[3], c.box[0] : c.box[2]]
        ).any()
    )
    assert differing >= 300


def test_the_pure_layout_table_tests_run_without_the_example_sheets_checked_out(
    pytester,
):
    """The 2026-09-15 audit, finding troupe-01: a module-level ``pytestmark``
    skip used to gate all 18 tests in this file, including the three above
    that touch nothing but ``ulpc``'s own layout table -- so ``ulpc.py`` had
    zero executed coverage on a checkout without the unshipped example PNGs.

    Run for real, in a sub-pytest, rather than by calling the test functions
    as plain Python: a raw call would skip nothing either way, fixed or not,
    because ``pytestmark``/``pytest.skip`` only take effect inside pytest's
    own collection and execution machinery. Pointing ``EXAMPLES`` at a
    directory that holds neither sheet reproduces "not checked out" without
    touching the real ``examples/`` tree.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    # Sliced above this very test's own ``def`` -- copying it whole would
    # duplicate the ``EXAMPLES = ...`` needle inside this test's own source
    # (quoted below as a plain string) and corrupt *that* copy instead.
    marker = (
        "\n\ndef "
        "test_the_pure_layout_table_tests_run_without_the_example_sheets_checked_out("
    )
    assert source.count(marker) == 1
    body = source[: source.index(marker)]

    missing = pytester.path / "no-such-examples"
    needle = 'EXAMPLES = Path(__file__).resolve().parents[3] / "examples"'
    assert body.count(needle) == 1
    patched = body.replace(needle, f"EXAMPLES = Path({str(missing)!r})")
    pytester.makepyfile(test_ulpc_probe=patched)

    result = pytester.runpytest("-v")

    result.assert_outcomes(passed=3, skipped=15)
    result.stdout.fnmatch_lines(
        [
            "*test_the_layout_table_totals_three_hundred_and_fifty_two_cells PASSED*",
            "*test_the_table_is_the_published_full_layout PASSED*",
            "*test_a_sheet_that_is_not_the_full_layout_is_refused PASSED*",
        ]
    )
