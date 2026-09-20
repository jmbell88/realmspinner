"""The tour overlay's pure arithmetic: where the hole sits, where the card sits.

The 2026-09-08 audit, finding tour-02: ``panes/tour.py``'s three drawing
helpers -- ``_hole``, ``_veil`` and ``_card_pos`` -- had no test anywhere in
the tree, unlike every other pane's "decidable half" (``sizeguard``,
``_view_cache``, ``_view_overlay``, ``viewer/camera.py`` and more, all covered
in ``tests/test_findings_blind_spots.py``). ``_veil`` reaches for a real
imgui foreground draw list and is out of scope for a headless test; ``_hole``
and ``_card_pos`` are plain arithmetic over a ``SimpleNamespace`` viewport and
are exactly the kind of thing that pattern already covers elsewhere.

``_card_pos`` in particular is a real branch: the card sits bottom-right by
default, and swaps to bottom-left when the ringed control's centre has passed
the viewport's horizontal midpoint (so the card does not sit on top of what it
is pointing at). A future refactor could silently invert or drop that branch,
and nothing but a screenshot regression would notice. The ``_hole``/``_card_pos``
half of this file was a test-only addition -- ``panes/tour.py`` was not touched.

``_veil`` itself stayed untested even after that pass, because it draws
straight to a real imgui foreground draw list. The 2026-09-11 audit's tour-02
split its band/span decomposition out as ``_veil_spans`` -- pure geometry
returning the scrim's paint rectangles instead of drawing them -- which is
what the tests below cover; ``_veil`` is now just that function plus one draw
call per rectangle.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from realmspinner.studio import tokens
from realmspinner.studio.panes import tour as tour_pane
from realmspinner.studio.tokens import sp


def _viewport(x=0.0, y=0.0, w=1000.0, h=800.0):
    pos = SimpleNamespace(x=x, y=y)
    size = SimpleNamespace(x=w, y=h)
    # tour._card_pos reads work_pos/work_size, not pos/size -- the work area
    # excludes the setup banner, same as App._transition_overlay.
    return SimpleNamespace(pos=pos, size=size, work_pos=pos, work_size=size)


# --- _card_pos ------------------------------------------------------------


def test_card_pos_defaults_to_bottom_right_with_no_hole():
    viewport = _viewport()
    x, y = tour_pane._card_pos(viewport, None)
    assert x == viewport.work_pos.x + viewport.work_size.x - sp(tokens.SP_4)
    assert y == viewport.work_pos.y + viewport.work_size.y - sp(tokens.SP_4)


def test_card_pos_keeps_bottom_right_when_the_hole_is_left_of_centre():
    viewport = _viewport(w=1000.0)
    # A hole whose centre (150) sits left of the viewport's midpoint (500).
    hole = (100.0, 200.0, 100.0, 40.0)
    x, _y = tour_pane._card_pos(viewport, hole)
    assert x == viewport.work_pos.x + viewport.work_size.x - sp(tokens.SP_4)


def test_card_pos_swaps_sides_when_the_hole_is_past_centre():
    viewport = _viewport(w=1000.0)
    # A hole whose centre (850) sits right of the viewport's midpoint (500).
    hole = (800.0, 200.0, 100.0, 40.0)
    x, _y = tour_pane._card_pos(viewport, hole)
    assert x == viewport.work_pos.x + sp(tokens.SP_4) + sp(tour_pane.CARD_W)
    # And it must actually have moved off the default right-anchored spot.
    default_x, _ = tour_pane._card_pos(viewport, None)
    assert x != default_x


def test_card_pos_only_swaps_horizontally_never_vertically():
    """A card that also chased the hole vertically would jump the length of
    the window between two steps pointing at the top and bottom of the same
    pane -- the docstring's own stated reason for the one-axis swap."""
    viewport = _viewport()
    no_hole_y = tour_pane._card_pos(viewport, None)[1]
    left_hole_y = tour_pane._card_pos(viewport, (100.0, 10.0, 50.0, 20.0))[1]
    right_hole_y = tour_pane._card_pos(viewport, (900.0, 700.0, 50.0, 20.0))[1]
    assert no_hole_y == left_hole_y == right_hole_y


def test_card_pos_bottom_offset_only_shifts_the_card_vertically():
    """tour-01 (the 2026-09-20 audit): ``_card_pos`` reused one ``margin`` for
    both the vertical lift that clears ``panes.bottom_pane`` and the
    horizontal inset from the viewport's edge, so opening the Familiar dock
    (a nonzero ``bottom_offset``) shifted every card horizontally too, by the
    pane's full height -- reproduced with a 200 px offset moving ``x`` by
    exactly 200. The docstring says only the vertical position should move.
    """
    viewport = _viewport()
    x0, y0 = tour_pane._card_pos(viewport, None, 0.0)
    x1, y1 = tour_pane._card_pos(viewport, None, 200.0)
    assert x1 == x0
    assert y1 == y0 - 200.0

    # The swapped-side branch reads the same inset, and must not move either.
    hole = (800.0, 200.0, 100.0, 40.0)  # past centre, as above
    hx0, hy0 = tour_pane._card_pos(viewport, hole, 0.0)
    hx1, hy1 = tour_pane._card_pos(viewport, hole, 200.0)
    assert hx1 == hx0
    assert hy1 == hy0 - 200.0


# --- _hole ------------------------------------------------------------------


def test_hole_pads_the_anchor_rect_by_hole_pad(monkeypatch):
    monkeypatch.setattr(tour_pane.anchors, "rect", lambda key: (50.0, 60.0, 20.0, 10.0))
    step = SimpleNamespace(anchor="some/control")
    hole = tour_pane._hole(SimpleNamespace(), step)
    pad = sp(tour_pane.HOLE_PAD)
    assert hole == (50.0 - pad, 60.0 - pad, 20.0 + pad * 2, 10.0 + pad * 2)


def test_hole_is_none_when_the_step_has_no_anchor():
    step = SimpleNamespace(anchor=None)
    assert tour_pane._hole(SimpleNamespace(), step) is None


def test_hole_is_none_when_the_anchor_did_not_draw_this_frame(monkeypatch):
    """A control inside a collapsed section or behind another tab simply did
    not draw -- an ordinary state, not a failure."""
    monkeypatch.setattr(tour_pane.anchors, "rect", lambda key: None)
    step = SimpleNamespace(anchor="hidden/control")
    assert tour_pane._hole(SimpleNamespace(), step) is None


# --- _veil_spans --------------------------------------------------------------
#
# The 2026-09-11 audit, finding tour-02: ``_veil``'s band-decomposition
# arithmetic (which y-bands and x-spans of the scrim get filled around one or
# two holes) had no test anywhere, unlike ``_hole`` and ``_card_pos`` above,
# which the 2026-09-08 audit already pinned. It is now split out as
# ``_veil_spans``, a pure function of the viewport rect and the hole list that
# returns the scrim's paint rectangles instead of drawing them.


def _rect_area(rect: tuple[float, float, float, float]) -> float:
    left, top, right, bottom = rect
    return max(0.0, right - left) * max(0.0, bottom - top)


def _clip(
    rect: tuple[float, float, float, float], bounds: tuple[float, float, float, float]
) -> tuple[float, float, float, float] | None:
    """``rect`` cut down to ``bounds``, or ``None`` if nothing is left."""
    left, top, right, bottom = rect
    bx0, by0, bx1, by1 = bounds
    left, top = max(left, bx0), max(top, by0)
    right, bottom = min(right, bx1), min(bottom, by1)
    if left >= right or top >= bottom:
        return None
    return (left, top, right, bottom)


def _overlap_area(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    al, at, ar, ab = a
    bl, bt, br, bb = b
    width = max(0.0, min(ar, br) - max(al, bl))
    height = max(0.0, min(ab, bb) - max(at, bt))
    return width * height


@pytest.mark.parametrize(
    "holes",
    [
        [],
        [(100.0, 100.0, 200.0, 50.0)],
        # Two holes in the same horizontal band -- the sorted-spans loop's own
        # reason for existing.
        [(100.0, 100.0, 100.0, 50.0), (400.0, 110.0, 100.0, 40.0)],
        # Two holes in different bands, side by side vertically -- one ringed
        # control near the top, one card near the bottom, the real shape
        # ``_veil``'s own docstring describes.
        [(100.0, 50.0, 150.0, 60.0), (600.0, 600.0, 250.0, 120.0)],
        # A hole that runs off the edge of the viewport -- only the clipped
        # portion should ever be treated as "the hole".
        [(-50.0, 100.0, 150.0, 50.0)],
        # Overlapping holes, which a real frame never produces (a step has one
        # anchor and the card is placed clear of it) but the arithmetic must
        # not double-subtract if it ever happened.
        [(100.0, 100.0, 150.0, 100.0), (150.0, 150.0, 150.0, 100.0)],
    ],
    ids=["none", "one", "same-band", "different-bands", "off-edge", "overlapping"],
)
def test_veil_spans_cover_every_hole_and_nothing_else(holes):
    viewport = (0.0, 0.0, 1000.0, 800.0)
    x0, y0, x1, y1 = viewport
    rects = tour_pane._veil_spans(x0, y0, x1, y1, holes)

    # 1. No painted rectangle overlaps any hole (clipped to the viewport) --
    #    the scrim must never dim through, or paint over, what a hole exists
    #    to keep visible.
    clipped_holes = []
    for hx, hy, hw, hh in holes:
        clipped = _clip((hx, hy, hx + hw, hy + hh), viewport)
        if clipped is not None:
            clipped_holes.append(clipped)
    for rect in rects:
        for hole in clipped_holes:
            assert _overlap_area(rect, hole) == 0.0, (rect, hole)

    # 2. No two painted rectangles overlap each other -- the "darker where two
    #    rectangles happen to meet" artefact the docstring names.
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            assert _overlap_area(a, b) == 0.0, (a, b)

    # 3. Together, the painted area and the (viewport-clipped, deduplicated by
    #    total coverage) hole area account for the whole viewport -- nothing
    #    is left unpainted and un-holed. Overlapping holes are covered by
    #    summing over a fine sample grid rather than by area arithmetic, since
    #    two overlapping holes' areas are not simply additive.
    painted_area = sum(_rect_area(r) for r in rects)
    step = 10.0
    samples = 0
    covered = 0
    x = x0 + step / 2
    while x < x1:
        y = y0 + step / 2
        while y < y1:
            samples += 1
            in_hole = any(hx <= x < hx + hw and hy <= y < hy + hh for hx, hy, hw, hh in holes)
            in_paint = any(
                left <= x < right and top <= y < bottom for left, top, right, bottom in rects
            )
            assert in_hole != in_paint, (x, y, in_hole, in_paint)
            if in_hole:
                covered += 1
            y += step
        x += step
    assert samples > 0
    # Sanity on the area bookkeeping for the non-overlapping cases: painted
    # area plus sampled hole area should be close to the full viewport area
    # (the sample grid is an approximation at the pixel-fraction level only
    # for the deliberately-overlapping case, so this is not asserted exactly).
    assert painted_area <= _rect_area((x0, y0, x1, y1))
