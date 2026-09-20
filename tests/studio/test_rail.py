"""Direct unit tests for ``rail.fitted_height`` and ``rail.expanded_fits``.

shell-08 (the 2026-09-20 audit): both functions' own docstrings claim to be
testable arithmetic ("Pure, so the arithmetic is testable at every window
size" -- ``fitted_height``), but the only test in the suite that named
either (``test_rail_comment_row_count_matches_rail_groups``, in
``tests/test_comment_counts_2026_09_13.py``) only scans the *comment* beside
the call site for a stale digit; it never calls either function. The
item-compression ladder these two implement could regress silently -- an off
day in the clamp order and a mode simply falls off the bottom of the rail --
with a fully green suite.
"""

from __future__ import annotations

from types import SimpleNamespace

from realmspinner.studio import layout as layout_mod
from realmspinner.studio import rail


def test_fitted_height_draws_full_height_rows_when_they_comfortably_fit():
    # room = (400 - 40) / 5 = 72, well past ITEM_H, so the ceiling wins.
    assert rail.fitted_height(5, gaps=40.0, avail=400.0) == rail.sp(rail.ITEM_H)


def test_fitted_height_compresses_between_the_two_floors_when_tight():
    # room = (300 - 0) / 10 = 30, strictly between MIN_ITEM_H (24) and
    # ITEM_H (44), so neither clamp fires and the raw division is returned.
    assert rail.fitted_height(10, gaps=0.0, avail=300.0) == rail.sp(30.0)


def test_fitted_height_never_drops_below_the_min_item_h_floor():
    # room = 10 / 20 = 0.5, far under MIN_ITEM_H: the floor must win even
    # though it means the column overflows and has to scroll instead.
    assert rail.fitted_height(20, gaps=0.0, avail=10.0) == rail.sp(rail.MIN_ITEM_H)


def test_fitted_height_returns_full_item_height_for_zero_rows():
    # The guard clause: an empty rail (no groups drawn) must not divide by
    # zero, and has no reason to compress anything either.
    assert rail.fitted_height(0, gaps=0.0, avail=0.0) == rail.sp(rail.ITEM_H)


class _FakeStyle:
    def __init__(self) -> None:
        self.item_spacing = SimpleNamespace(x=0.0)
        self.window_padding = SimpleNamespace(x=0.0)


class _FakeViewport:
    def __init__(self, width: float) -> None:
        self.work_size = (width, 0.0)


class _FakeImgui:
    """Enough of ``imgui`` for :func:`rail.expanded_fits`'s own arithmetic,
    without a real GL context -- the function's early-out already covers "no
    context", and the rest is a pure comparison against ``get_style`` and
    ``get_main_viewport``, both trivially fakeable.
    """

    def __init__(self, *, context: object | None, viewport_width: float = 0.0) -> None:
        self._context = context
        self._viewport = _FakeViewport(viewport_width)

    def get_current_context(self) -> object | None:
        return self._context

    def get_style(self) -> _FakeStyle:
        return _FakeStyle()

    def get_main_viewport(self) -> _FakeViewport:
        return self._viewport


# The exact threshold ``expanded_fits`` computes, with spacing/padding zeroed
# by ``_FakeStyle`` above so the fake viewport width can be set precisely on
# either side of it.
_NEED = (
    rail.sp(rail.RAIL_EXPANDED_W)
    + rail.sp(layout_mod.PANEL_MIN) * 2
    + rail.sp(layout_mod.CENTRE_MIN)
)


def test_expanded_fits_defaults_true_with_no_imgui_context():
    # A headless caller (no window yet) must not be told the rail cannot
    # expand -- the preference itself is what decides that until a real
    # viewport says otherwise.
    rail.imgui = _FakeImgui(context=None)
    try:
        assert rail.expanded_fits() is True
    finally:
        from imgui_bundle import imgui as real_imgui

        rail.imgui = real_imgui


def test_expanded_fits_true_when_the_viewport_is_wide_enough():
    rail.imgui = _FakeImgui(context=object(), viewport_width=_NEED + 1.0)
    try:
        assert rail.expanded_fits() is True
    finally:
        from imgui_bundle import imgui as real_imgui

        rail.imgui = real_imgui


def test_expanded_fits_false_when_the_viewport_is_too_narrow():
    rail.imgui = _FakeImgui(context=object(), viewport_width=_NEED - 1.0)
    try:
        assert rail.expanded_fits() is False
    finally:
        from imgui_bundle import imgui as real_imgui

        rail.imgui = real_imgui
