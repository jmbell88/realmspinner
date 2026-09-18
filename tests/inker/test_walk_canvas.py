"""``modes/inker/ui/panes/walk_canvas.py``'s decidable half: the mouse-ownership gate,
the zoom-to-image-space radius, and the two read-only pass-throughs.

The 2026-09-11 audit, finding inker-10: this module's name appeared only in
``tests/manual/test_coverage.py``'s chapter sweep -- never imported or driven
by a functional test -- while its sibling setup panel, ``modes/inker/ui/panes/walk.py``,
is driven live by ``tests/inker/test_walk_session.py``. Nothing here found a
defect; this file is coverage for logic that was already correct, in
``test_walk_session.py``'s own shape (a real ``inker_walk`` session, no GL) but
aimed at the canvas half: ``owns_mouse``/``handle`` (would a walk-setup click
reach the paint tools underneath it, or only the session?), ``_radius`` (the
zoom scaling), and ``clipping_warning``/``refusal`` (the pass-through).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.kernels import pixel as inker
from warlock.kernels.pixel.walk import rig as R
from warlock.studio.modes.inker import state as inker_state
from warlock.studio.modes.inker import walk as inker_walk
from warlock.studio.modes.inker.ui.panes import walk_canvas as pane

SIZE = (32, 32)


class _Ctx:
    def __init__(self, state: inker_state.InkerState) -> None:
        self.state = SimpleNamespace(inker=state)
        self.viewer = None  # inker_walk.cancel's _release_textures no-ops on this
        self.toasts: list = []

    def toast(self, text, level="info", **_):
        self.toasts.append((text, level))


def _scene():
    state = inker_state.InkerState()
    ctx = _Ctx(state)
    doc = inker.Document.blank(*SIZE)
    tab = inker_state.InkerDoc(doc=doc)
    state.docs.append(tab)
    state.active_uid = tab.uid
    return ctx, state, tab


# -- owns_mouse / handle: the gate between a walk session and the paint tools --


def test_walk_canvas_owns_the_mouse_only_while_a_session_is_open(monkeypatch):
    ctx, state, tab = _scene()
    assert pane.owns_mouse(state, tab) is False

    assert inker_walk.open_session(ctx, tab)
    assert pane.owns_mouse(state, tab) is True

    inker_walk.cancel(ctx, tab)
    assert pane.owns_mouse(state, tab) is False


def test_handle_routes_a_click_to_inker_walk_press_only_while_a_session_is_open(monkeypatch):
    """A regression here would silently let a walk-setup click either paint on
    the drawing (a session open but ``handle`` not consuming the press) or
    reach nothing at all (a session open but ``handle`` refusing a click it
    should take) -- both of which the orchestrator's finding names directly.
    """
    ctx, state, tab = _scene()
    monkeypatch.setattr(pane.imgui, "is_mouse_clicked", lambda button: True)
    monkeypatch.setattr(pane.imgui, "is_mouse_down", lambda button: False)
    pressed: list[tuple] = []
    monkeypatch.setattr(
        inker_walk, "press", lambda ctx, tab, point, radius: pressed.append(point) or True
    )

    # No session: a click must not reach inker_walk.press. This is the "paint
    # tools underneath" half of owns_mouse -- with no session, they are the
    # only thing a click can mean.
    pane.handle(ctx, state, tab, (5.0, 6.0), active=True)
    assert pressed == []

    # A session open: the same click now belongs to it.
    assert inker_walk.open_session(ctx, tab)
    pane.handle(ctx, state, tab, (5.0, 6.0), active=True)
    assert pressed == [(5.0, 6.0)]

    # Closed again: back to falling through to nothing (the paint tools, from
    # this module's point of view).
    inker_walk.cancel(ctx, tab)
    pressed.clear()
    pane.handle(ctx, state, tab, (5.0, 6.0), active=True)
    assert pressed == []


def test_handle_drags_and_releases_only_while_the_session_holds_a_grab(monkeypatch):
    ctx, state, tab = _scene()
    assert inker_walk.open_session(ctx, tab)
    session = inker_walk.session(state, tab)

    dragged: list[tuple] = []
    released: list[bool] = []
    monkeypatch.setattr(
        inker_walk, "drag", lambda ctx, tab, point: dragged.append(point) or True
    )
    monkeypatch.setattr(inker_walk, "release", lambda ctx, tab: released.append(True) or True)
    monkeypatch.setattr(pane.imgui, "is_mouse_clicked", lambda button: False)

    # Nothing grabbed: a mouse-down frame is not a drag.
    monkeypatch.setattr(pane.imgui, "is_mouse_down", lambda button: True)
    pane.handle(ctx, state, tab, (1.0, 2.0), active=False)
    assert dragged == []
    assert released == []

    # Grabbed and still down: a drag.
    session.grab = "near_hip"
    pane.handle(ctx, state, tab, (3.0, 4.0), active=False)
    assert dragged == [(3.0, 4.0)]

    # Grabbed and released: the mouse went up.
    monkeypatch.setattr(pane.imgui, "is_mouse_down", lambda button: False)
    pane.handle(ctx, state, tab, (3.0, 4.0), active=False)
    assert released == [True]


# -- _radius: zoom-to-image-space -------------------------------------------------------


def test_radius_shrinks_as_the_view_zooms_in():
    """``plotter_canvas._handle_at``'s rule, restated in this module's own
    docstring: a grab target fixed in screen pixels and converted back to
    document pixels must shrink in document space as the view zooms in, or a
    handle that is comfortable at 100% becomes an unusably fat target at 800%.
    """
    _, _, tab = _scene()
    tab.view.zoom = 1.0
    at_1x = pane._radius(tab)
    tab.view.zoom = 4.0
    at_4x = pane._radius(tab)
    assert at_4x == pytest.approx(at_1x / 4.0)
    assert at_1x > 0.0


def test_radius_never_reaches_zero_or_goes_negative_at_extreme_or_bad_zoom():
    """``_radius`` floors both the zoom it divides by (``max(1e-6, ...)``) and
    its own result (``max(1.0, ...)``); a stray zero or negative zoom must not
    reach a division by zero or a negative grab radius."""
    _, _, tab = _scene()
    tab.view.zoom = 0.0
    assert pane._radius(tab) >= 1.0
    tab.view.zoom = -3.0
    assert pane._radius(tab) >= 1.0
    tab.view.zoom = 1000.0
    assert pane._radius(tab) >= 1.0


# -- clipping_warning / refusal: the pass-throughs --------------------------------------


def test_clipping_warning_and_refusal_are_empty_with_nothing_open():
    _, state, tab = _scene()
    assert pane.clipping_warning(state, tab) == ""
    assert pane.refusal(state, tab) == ""


def test_refusal_mirrors_the_open_rigs_own_refusal_message():
    ctx, state, tab = _scene()
    assert inker_walk.open_session(ctx, tab)
    session = inker_walk.session(state, tab)
    # A fresh rig has nothing assigned yet, so walk.rig.refusal has something
    # to say -- and inker_walk_canvas.refusal must say exactly that, not its
    # own paraphrase of it.
    assert pane.refusal(state, tab) == R.refusal(session.rig)
    assert pane.refusal(state, tab) != ""


def test_clipping_warning_stays_empty_until_the_rig_is_ready_to_render():
    ctx, state, tab = _scene()
    assert inker_walk.open_session(ctx, tab)
    session = inker_walk.session(state, tab)
    assert not inker_walk.ready(session)
    assert pane.clipping_warning(state, tab) == ""
