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
and nothing but a screenshot regression would notice. This is a test-only
addition -- ``panes/tour.py`` is not touched.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.studio import tokens
from warlock.studio.panes import tour as tour_pane
from warlock.studio.tokens import sp


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
