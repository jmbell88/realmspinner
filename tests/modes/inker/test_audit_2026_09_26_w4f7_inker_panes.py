"""Regression tests for the 2026-09-26 audit's w4f7 fixer slice.

Six findings, all in the Inker's panes (``ui/panes/`` and the rest of ``ui/``):
a masked regenerate reaching an SDXL round trip before a tilemap-layer refusal
that used to only fire bare past the landing door (inker-panes-03,
``ui/panes/bridge.py``); the non-modal filter popup drawing against whichever
tab is in front rather than the one that opened it (inker-panes-04, same
file); a walk-cycle session's row drawing under a higher-precedence bar
instead of yielding to it (inker-panes-05, ``ui/panes/canvas.py``); the
Generation pane's New button skipping the New dialog (inker-panes-06,
``ui/panes/generate.py``); onion skinning keeping the wrong tint on a wrapped
clip (inker-panes-07, ``ui/panes/canvas.py``); the shortcut editor accepting
un-answerable chords (inker-panes-08, ``ui/panes/menu.py``); the transport and
the frame-header menu greying with no reason, or the wrong one
(inker-panes-09/10, ``ui/panes/timeline.py``); and the symmetry axis field
showing a placeholder instead of the real canvas centre (inker-panes-12,
``ui/panes/context.py``).

inker-panes-11 (``dev/INVARIANTS.md`` citing a deleted ``timeline_open``/
``toggle`` pair) is documentation-only and owns no test here; the corrected
paragraph is returned to the orchestrator instead.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.pixel.tiles import strip
from realmspinner.studio import controls as controls_mod
from realmspinner.studio import probe, widgets
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker import state as inker_state
from realmspinner.studio.modes.inker import walk as inker_walk
from realmspinner.studio.modes.inker.ui.panes import bridge as inker_bridge
from realmspinner.studio.modes.inker.ui.panes import canvas as inker_canvas
from realmspinner.studio.modes.inker.ui.panes import context as inker_context
from realmspinner.studio.modes.inker.ui.panes import generate as inker_generate
from realmspinner.studio.modes.inker.ui.panes import menu as inker_menu
from realmspinner.studio.modes.inker.ui.panes import timeline as inker_timeline
from realmspinner.studio.modes.inker.ui.panes import walk_canvas as inker_walk_canvas


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` for why it is not a
    conftest fixture."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui


class _Ctx:
    """Enough of ``Ctx`` for the bridge doors under test."""

    def __init__(self, state: inker_state.InkerState) -> None:
        self.state = SimpleNamespace(inker=state, preview={})
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))

    def submit(self, key: str, fn: Any, *a: Any, **k: Any) -> bool:
        return True


def _tab(doc: Any = None) -> inker_state.InkerDoc:
    doc = inker.Document.blank(16, 16) if doc is None else doc
    return inker_state.InkerDoc(doc=doc, title="t", saved_head=doc.history.head)


def _tilemap_doc() -> Any:
    doc = inker.Document.blank(16, 16)
    tile = np.zeros((16, 16, 4), dtype=np.uint8)
    tile[..., 3] = 255
    stack = np.stack([np.zeros((16, 16, 4), dtype=np.uint8), tile], axis=0)
    slot = doc.add_tileset(strip(stack))
    doc.add_tilemap_layer(slot.uid)
    return doc


# --- inker-panes-03: Regenerate selection on a tilemap layer -----------------


def test_regenerate_refuses_a_tilemap_layer_before_queuing_a_job():
    """The 2026-09-26 audit, finding inker-panes-03: "Regenerate selection..."
    had no gate of its own -- it passed every check ``_open_inpaint`` made
    (a selection exists, nothing already pending) on a tilemap layer exactly
    as readily as on a raster one, opened the popup, and only
    ``land_inpaint``'s ``apply_pixels`` found out later, after an SDXL round
    trip. Refused here now, before ``imgui.open_popup`` is ever reached.
    """
    doc = _tilemap_doc()
    doc.select_all()
    tab = _tab(doc)
    state = inker_state.InkerState()
    state.add(tab)
    ctx = _Ctx(state)

    inker_bridge._open_inpaint(ctx, tab)

    assert ctx.toasts, "a tilemap layer must be refused, not silently ignored"
    assert ctx.toasts[-1][1] == "warn"
    assert "tilemap" in ctx.toasts[-1][0].lower()


def test_regenerate_still_opens_on_an_ordinary_raster_layer(ui):
    """The fix must not cost the ordinary case: a plain selection still opens
    the popup, or the fix for inker-panes-03 would just refuse everything."""
    doc = inker.Document.blank(16, 16)
    doc.select_all()
    tab = _tab(doc)
    state = inker_state.InkerState()
    state.add(tab)
    ctx = _Ctx(state)

    ui.new_frame()
    ui.begin("##host")
    inker_bridge._open_inpaint(ctx, tab)
    ui.end()
    ui.end_frame()

    assert not ctx.toasts, ctx.toasts


def test_land_inpaint_says_why_instead_of_raising_on_a_tilemap_layer():
    """The landing door's own belt: even bypassing ``_open_inpaint`` (a stale
    pending regeneration whose layer was converted to a tilemap layer while
    the job was in flight), ``apply_pixels``'s ``ValueError`` must not unwind
    whichever frame this lands on.
    """
    doc = _tilemap_doc()
    layer_uid = doc.stack.active.uid
    tab = _tab(doc)
    state = inker_state.InkerState()
    state.add(tab)
    ctx = _Ctx(state)
    pending = {
        "tab_uid": tab.uid,
        "layer_uid": layer_uid,
        "box": (0, 0, 4, 4),
        "weight": None,
    }
    pixels = np.full((4, 4, 4), 255, dtype=np.uint8)

    ok = inker_bridge.land_inpaint(ctx, pending, pixels)

    assert ok is False
    assert ctx.toasts and ctx.toasts[-1][1] == "warn"
    assert "tilemap" in ctx.toasts[-1][0].lower()


# --- inker-panes-04: the filter popup's owner across a tab switch -----------


def test_a_filter_popup_left_open_across_a_tab_switch_cancels_the_owners_preview(ui):
    """The 2026-09-26 audit, finding inker-panes-04: the non-modal filter
    popup is drawn against ``state.active`` -- the *front* tab this frame --
    rather than the tab ``_open_filter`` actually began the session on.
    Ctrl+Tab to another document while it was still up ran the live preview
    against a document with no open session (silently doing nothing) and
    routed Cancel there too: the popup closed and ``filter_uid`` was cleared
    without the real owner's session ever being cancelled, leaving its preview
    pixels live in the layer with nothing on the undo stack -- exactly the
    state a save then serialises as if it had been approved.
    """
    doc_a = inker.Document.blank(8, 8)
    doc_a.stack.active.pixels[:] = (255, 0, 0, 255)
    tab_a = _tab(doc_a)
    doc_b = inker.Document.blank(8, 8)
    tab_b = _tab(doc_b)
    state = inker_state.InkerState()
    state.add(tab_a)
    state.add(tab_b)
    state.active_uid = tab_a.uid
    ctx = _Ctx(state)

    before = doc_a.stack.active.pixels.copy()

    def _frame(fn):
        ui.new_frame()
        ui.begin("##host")
        try:
            fn()
        finally:
            ui.end()
            ui.end_frame()

    # Open the filter on A while A is the active tab.
    _frame(lambda: inker_bridge._open_filter(ctx, tab_a))
    assert state.filter_uid == tab_a.uid
    assert doc_a._filter is not None, "sanity: a session opened on A"

    # Ctrl+Tab: B comes to the front while the popup is still up. Spy on each
    # document's own ``preview_filter`` -- the sharpest signal there is, since
    # it is what a frame with the popup up calls *every* frame it stays open.
    a_calls: list[Any] = []
    b_calls: list[Any] = []
    doc_a.preview_filter = lambda *a, **k: (a_calls.append(1), True)[1]  # type: ignore[method-assign]
    doc_b.preview_filter = lambda *a, **k: (b_calls.append(1), True)[1]  # type: ignore[method-assign]

    state.activate(tab_b.uid)
    _frame(lambda: inker_bridge.popups(ctx))

    assert b_calls == [], "B never opened a filter and must never be previewed"
    assert a_calls == [1], "the live preview must keep running against its real owner, A"

    # B never opened a filter and must stay untouched.
    assert doc_b._filter is None, "B must never gain a filter session"
    # A's session must still be the live one -- not silently orphaned.
    assert doc_a._filter is not None, "A's own session must survive the tab switch"
    assert state.filter_uid == tab_a.uid

    # Cancel is a plain call to ``cancel_filter`` on the popup's owner
    # (``_filter_popup``'s Cancel button, line for line); what this test
    # guards is that the *owner* -- resolved by ``popups`` above -- is still A
    # and not whatever tab happened to be in front when the button was drawn.
    doc_a.cancel_filter()
    state.filter_uid = ""

    assert doc_a._filter is None
    assert np.array_equal(doc_a.stack.active.pixels, before)


# --- inker-panes-05: one bar above the canvas, even with a walk session open -


def test_the_walk_row_does_not_draw_while_transform_outranks_it(monkeypatch):
    """The 2026-09-26 audit, finding inker-panes-05: neither Paste nor
    Transform refuses while a walk-cycle session is open, and
    ``inker_walk_canvas.row`` was called unconditionally from
    ``inker_canvas.draw`` -- so starting either drew that row *underneath*
    whichever bar ``which_bar``'s own precedence actually gave the frame to,
    breaking "one bar above the canvas". The row is now gated on the same
    precedence ``inker_context.draw`` already uses.
    """
    doc = inker.Document.blank(16, 16)
    tab = _tab(doc)
    state = inker_state.InkerState()
    state.add(tab)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))

    assert inker_walk.open_session(ctx, tab) is True
    state.transforming = True  # a free transform now outranks the walk row

    for name in ("_tab_bar", "_transform_row", "_canvas", "new_popup", "_note_size_dialogs"):
        monkeypatch.setattr(inker_canvas, name, lambda *a, **k: None)
    monkeypatch.setattr(inker_canvas.inker_menu, "draw_popups", lambda *a, **k: None)
    monkeypatch.setattr(inker_canvas.inker_bridge, "popups", lambda *a, **k: None)
    calls: list[Any] = []
    monkeypatch.setattr(inker_walk_canvas, "row", lambda *a, **k: calls.append(1))

    inker_canvas.draw(ctx)

    assert calls == [], "the walk row must not draw while a transform holds the bar"


def test_the_walk_row_draws_when_nothing_outranks_it(monkeypatch):
    """The fix must not cost the ordinary case: with no transform, no floating
    buffer and no open gesture, the walk row is still the one thing drawn."""
    doc = inker.Document.blank(16, 16)
    tab = _tab(doc)
    state = inker_state.InkerState()
    state.add(tab)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))
    assert inker_walk.open_session(ctx, tab) is True

    for name in ("_tab_bar", "_transform_row", "_canvas", "new_popup", "_note_size_dialogs"):
        monkeypatch.setattr(inker_canvas, name, lambda *a, **k: None)
    monkeypatch.setattr(inker_canvas.inker_menu, "draw_popups", lambda *a, **k: None)
    monkeypatch.setattr(inker_canvas.inker_bridge, "popups", lambda *a, **k: None)
    calls: list[Any] = []
    monkeypatch.setattr(inker_walk_canvas, "row", lambda *a, **k: calls.append(1))

    inker_canvas.draw(ctx)

    assert calls == [1]


# --- inker-panes-06: Generation pane's New button goes through the dialog ---


def test_the_generation_panes_new_button_opens_the_new_canvas_dialog(ui, monkeypatch):
    """The 2026-09-26 audit, finding inker-panes-06: this button used to call
    ``inker_mode.new_document(ctx, 1024, 1024)`` straight from the click,
    contradicting manual 28's "New opens a dialog, 64x64 by default". It must
    now open the same shared "new-canvas" popup every other New already uses,
    and must never call ``new_document`` itself.
    """
    doc = inker.Document.blank(8, 8)
    tab = _tab(doc)
    state = inker_state.InkerState()
    state.add(tab)
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=state, preview={}), toasts=[]
    )
    ctx.toast = lambda *a, **k: ctx.toasts.append(a)

    made: list[Any] = []
    monkeypatch.setattr(inker_mode, "new_document", lambda *a, **k: made.append(a))

    click = {"now": False}

    def fake_button(label: str, *a: Any, **k: Any) -> bool:
        return bool(click["now"]) and label == widgets.DOCUMENT_ACTION_LABELS["New"]

    monkeypatch.setattr(controls_mod, "button", fake_button)

    def frame() -> tuple[list[Any], bool]:
        probe.begin_frame()
        ui.new_frame()
        ui.set_next_window_size((520.0, 900.0))
        ui.set_next_window_pos((0.0, 0.0))
        ui.begin("##host")
        inker_generate.draw(ctx)
        opened = ui.is_popup_open("new-canvas")
        ui.end()
        ui.end_frame()
        return list(probe.FRAME_CONTROLS), opened

    _, opened_before = frame()
    assert opened_before is False

    click["now"] = True
    _, opened_after_click = frame()
    click["now"] = False

    assert made == [], "New must not call new_document directly any more"
    assert opened_after_click is True, "New must open the shared New dialog"

    seen, opened_still = frame()
    assert opened_still is True, "the dialog must stay open for a frame to answer"
    # The New dialog's own body drew -- its two size fields, unique to it.
    labels = {c.text for c in seen}
    assert {"W", "H"} <= labels, labels


# --- inker-panes-07: onion skinning on a wrapping clip ----------------------


def test_onion_wrap_duplicate_keeps_the_nearest_ghosts_tint(ui, monkeypatch):
    """The 2026-09-26 audit, finding inker-panes-07: ``_onion`` walked offsets
    furthest-first and deduped in the same pass, so on a clip short enough to
    wrap, the *furthest* offset claimed a frame first and the nearer, stronger
    ghost was then skipped as "already drawn" -- the opposite of the comment's
    own stated rule that the closest ghost is the one worth keeping.

    Three frames, current = 0, ``onion_before=2``/``onion_after=1``: offset 1
    forward and offset 2 backward both resolve to frame 1 by wrapping, and the
    nearer one (offset 1, the forward/green tint) must be the one drawn.
    """
    doc = inker.Document.blank(8, 8)
    doc.add_frame()
    doc.add_frame()
    doc.set_current_frame(0)
    tab = _tab(doc)
    state = inker_state.InkerState()
    state.add(tab)
    state.onion_before = 2
    state.onion_after = 1
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))

    calls: list[tuple[Any, int]] = []
    monkeypatch.setattr(
        inker_canvas.inker_textures,
        "frame_texture",
        lambda ctx, tab, uid, track_uid=None: uid,
    )
    monkeypatch.setattr(
        inker_canvas,
        "_blit",
        lambda draw_list, texture, view, origin, x0, y0, x1, y1, colour: calls.append(
            (texture, colour)
        ),
    )

    ui.new_frame()
    ui.begin("##host")
    inker_canvas._onion(ctx, state, tab, draw_list=None, view=None, origin=None, size=(16, 16))
    ui.end()
    ui.end_frame()

    frame1_uid = doc.anim.frames[1].uid
    hits = [c for c in calls if c[0] == frame1_uid]
    assert len(hits) == 1, "a wrapped frame must be ghosted once, not stacked twice"
    expected = inker_canvas._u32(
        state.onion_tint_forward, state.onion_alpha / (1.0**state.onion_falloff)
    )
    assert hits[0][1] == expected, (
        "the wrap must keep the nearest offset's tint (forward/green), not the furthest's"
    )


# --- inker-panes-08: the shortcut editor and un-answerable chords ----------


@pytest.mark.parametrize("bad", ["Ctrl+Banana", "+", "Ctrl+", "   "])
def test_recognised_chord_refuses_text_no_keyboard_can_answer(bad):
    """The 2026-09-26 audit, finding inker-panes-08: the shortcut editor's
    Apply button passed whatever text was typed straight to
    ``inker_ops.set_shortcuts``. ``Binding.__post_init__`` checks the *raw*
    string is non-empty before ``canonical_chord`` collapses it -- so
    ``"Ctrl+"`` (canonicalises to the modifier alone) and ``"+"``
    (canonicalises to the empty string) both slipped through, and
    ``"Ctrl+Banana"`` was never checked against any real key at all.
    """
    assert inker_menu._recognised_chord(bad) is False


@pytest.mark.parametrize("good", ["Ctrl+S", "F5", "Delete", "Ctrl+Shift+Z", "["])
def test_recognised_chord_accepts_ordinary_chords(good):
    """The fix must not cost the ordinary case: every chord already bound in
    this app still passes."""
    assert inker_menu._recognised_chord(good) is True


def test_apply_refuses_a_bogus_draft_without_calling_set_shortcuts(ui, monkeypatch):
    """The Apply button itself, driven through the real popup: a bogus chord
    in the draft must not reach ``inker_ops.set_shortcuts`` at all, and must
    say why instead of silently unbinding the command."""
    from realmspinner.studio.modes.inker import ops as inker_ops

    state = inker_state.InkerState()
    state.shortcut_target = inker_ops.binding_target("command", "new")
    state.shortcut_draft = "Ctrl+Banana"
    state.shortcut_context = ""
    state.shortcut_trigger = "press"
    state.pending_dialog = inker_menu.SHORTCUTS_POPUP
    said: list[str] = []
    state.say = lambda text: said.append(text)  # type: ignore[method-assign]
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))

    called: list[Any] = []
    monkeypatch.setattr(inker_ops, "set_shortcuts", lambda *a, **k: called.append(a))
    monkeypatch.setattr(inker_menu.inker_mode, "persist", lambda *a, **k: None)
    # The only ``widgets.primary_button`` in this popup is Apply.
    monkeypatch.setattr(inker_menu.widgets, "primary_button", lambda *a, **k: True)

    ui.new_frame()
    ui.begin("host")
    inker_menu._shortcuts_popup(ctx, state)
    ui.end()
    ui.end_frame()

    assert called == [], "set_shortcuts must never be called for a bogus draft"
    assert said and "Ctrl+Banana" in said[0]


def test_apply_still_calls_set_shortcuts_for_an_ordinary_chord(ui, monkeypatch):
    """The fix must not cost the ordinary case: a real chord still applies."""
    from realmspinner.studio.modes.inker import ops as inker_ops

    state = inker_state.InkerState()
    state.shortcut_target = inker_ops.binding_target("command", "new")
    state.shortcut_draft = "Ctrl+Alt+N"
    state.shortcut_context = ""
    state.shortcut_trigger = "press"
    state.pending_dialog = inker_menu.SHORTCUTS_POPUP
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))

    called: list[Any] = []
    monkeypatch.setattr(
        inker_ops, "set_shortcuts", lambda *a, **k: called.append(a) or {}
    )
    monkeypatch.setattr(inker_menu.inker_mode, "persist", lambda *a, **k: None)
    monkeypatch.setattr(inker_menu.widgets, "primary_button", lambda *a, **k: True)

    ui.new_frame()
    ui.begin("host")
    inker_menu._shortcuts_popup(ctx, state)
    ui.end()
    ui.end_frame()

    assert len(called) == 1


# --- inker-panes-09: the transport's busy reasons, and Delete frame's own --


def test_delete_frame_shows_the_busy_reason_while_the_tab_is_busy(ui):
    """The 2026-09-26 audit, finding inker-panes-09: "Delete frame" carried a
    reason, but only for its own "one frame left" gate. Greyed instead because
    the tab is busy (saving), it went on saying "A clip needs at least one
    frame" -- true of the document, false of why the button would not run.
    """
    doc = inker.Document.blank(8, 8)
    doc.add_frame()  # two frames: the "one frame left" gate would read True
    tab = _tab(doc)
    tab.saving = True
    state = inker_state.InkerState()
    state.add(tab)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))

    captured: dict[str, Any] = {}

    def fake_toolbar(bar_id: str, items: Any, on_click: Any, **kwargs: Any) -> None:
        captured["items"] = items

    import realmspinner.studio.toolbar as toolbar_mod

    original = toolbar_mod.toolbar
    ui.new_frame()
    ui.begin("##host")
    try:
        toolbar_mod.toolbar = fake_toolbar  # type: ignore[assignment]
        inker_timeline._transport(ctx, tab)
    finally:
        toolbar_mod.toolbar = original
        ui.end()
        ui.end_frame()

    remove_item = next(i for i in captured["items"] if i.key == "remove")
    assert remove_item.enabled is False
    assert remove_item.reason == widgets.DOCUMENT_SAVING_WHY
    assert remove_item.reason != "A clip needs at least one frame."

    add_item = next(i for i in captured["items"] if i.key == "add")
    assert add_item.enabled is False
    assert add_item.reason == widgets.DOCUMENT_SAVING_WHY


def test_delete_frame_keeps_its_own_reason_when_not_busy(ui):
    """The other half: on a one-frame clip with nothing busy, the row is
    still greyed, and for its own, original reason."""
    doc = inker.Document.blank(8, 8)  # never animated: no add_frame call
    doc.add_frame()
    doc.remove_frame(1)  # back down to one frame
    tab = _tab(doc)
    state = inker_state.InkerState()
    state.add(tab)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))

    captured: dict[str, Any] = {}

    def fake_toolbar(bar_id: str, items: Any, on_click: Any, **kwargs: Any) -> None:
        captured["items"] = items

    import realmspinner.studio.toolbar as toolbar_mod

    original = toolbar_mod.toolbar
    ui.new_frame()
    ui.begin("##host")
    try:
        toolbar_mod.toolbar = fake_toolbar  # type: ignore[assignment]
        inker_timeline._transport(ctx, tab)
    finally:
        toolbar_mod.toolbar = original
        ui.end()
        ui.end_frame()

    remove_item = next(i for i in captured["items"] if i.key == "remove")
    assert remove_item.enabled is False
    assert remove_item.reason == "A clip needs at least one frame."


# --- inker-panes-10: the frame-header menu's own Delete row -----------------


def test_the_frame_menus_delete_row_is_disabled_on_a_one_frame_clip(ui):
    """The 2026-09-26 audit, finding inker-panes-10: this row was enabled on a
    one-frame clip, where ``remove_frame`` already refuses (returns ``False``)
    rather than leaving none -- so the click did nothing, with no reason."""
    doc = inker.Document.blank(8, 8)
    doc.add_frame()
    doc.remove_frame(1)  # one frame left
    tab = _tab(doc)

    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    ui.button("anchor")
    ui.open_popup("framemenu0")
    inker_timeline._frame_menu(tab, 0)
    ui.end()
    ui.end_frame()

    found = [c for c in probe.FRAME_CONTROLS if c.text == "Delete"]
    assert found, [c.text for c in probe.FRAME_CONTROLS]
    assert found[0].enabled is False
    assert found[0].reason == "A clip needs at least one frame."


def test_the_frame_menus_delete_row_stays_enabled_with_more_than_one_frame(ui):
    """The fix must not cost the ordinary case: with more than one frame, the
    row is still enabled."""
    doc = inker.Document.blank(8, 8)
    doc.add_frame()
    tab = _tab(doc)

    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    ui.button("anchor")
    ui.open_popup("framemenu0")
    inker_timeline._frame_menu(tab, 0)
    ui.end()
    ui.end_frame()

    found = [c for c in probe.FRAME_CONTROLS if c.text == "Delete"]
    assert found and found[0].enabled is True


# --- inker-panes-12: the symmetry axis field and the real canvas centre ----


def test_the_symmetry_axis_field_shows_and_preserves_the_real_centre(ui, monkeypatch):
    """The 2026-09-26 audit, finding inker-panes-12: with the axis unset this
    field showed ``(0, 0)`` regardless of the document's size, so a document
    wider or taller than a couple of pixels showed "0 0" for a mirror line
    that was actually sitting at the middle of the canvas. Typing a new X then
    wrote ``(x, 0.0)`` -- the field's own wrong placeholder, not the true
    centre -- jumping the Y mirror line to the top edge.
    """
    doc = inker.Document.blank(64, 32)
    tab = _tab(doc)
    state = inker_state.InkerState()
    state.add(tab)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))

    calls: list[list[float]] = []

    def fake_input_float2(label: str, values: Any, fmt: str = "%.0f") -> tuple[bool, list[float]]:
        calls.append(list(values))
        # The user edits only the X box; imgui hands the untouched sibling
        # back exactly as it was shown.
        return True, [99.0, values[1]]

    monkeypatch.setattr(inker_context.controls, "input_float2", fake_input_float2)

    ui.new_frame()
    ui.begin("##host")
    inker_context._symmetry_axis(ctx, state, tab)
    ui.end()
    ui.end_frame()

    assert calls == [[31.5, 15.5]], "the field must show the real canvas centre, not (0, 0)"
    assert state.symmetry_axis == (99.0, 15.5), (
        "editing X must not jump the untouched Y mirror to 0"
    )
