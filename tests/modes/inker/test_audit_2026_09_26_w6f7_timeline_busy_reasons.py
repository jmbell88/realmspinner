"""the 2026-09-26 audit, finding inker-panes-09: while playback or saving is
in progress, every frame/layer/cel/tag/group menu row in the timeline was
wrapped in a bare ``imgui.begin_disabled(tab.busy)`` with no reason of its
own. That wrap dims the row, but the hover tooltip and the probe census both
read a control's *own* ``enabled``/``reason`` pair -- which a row that passed
``enabled=True`` for some unrelated reason (e.g. a range being selected)
still carried, so it looked live to both while being unclickable. The
transport row and the frame-header menu's "Delete" row already carry the
busy reason (a prior fix pass); this covers the remaining rows via
``_busy_gate`` -- ``_frame_menu``'s other rows, ``_row_menu``, and
``_range_menu`` (reached through ``_cell_menu``), where busy must win over
a row's own, unrelated reason.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from _ui_context import imgui_context

from realmspinner.kernels import pixel as inker
from realmspinner.studio import probe, widgets
from realmspinner.studio.modes.inker import state as inker_state
from realmspinner.studio.modes.inker.ui.panes import timeline as inker_timeline


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _tab(doc: Any = None) -> inker_state.InkerDoc:
    doc = inker.Document.blank(16, 16) if doc is None else doc
    return inker_state.InkerDoc(doc=doc, title="t", saved_head=doc.history.head)


def _ctx(state: inker_state.InkerState) -> Any:
    return SimpleNamespace(state=SimpleNamespace(inker=state), toast=lambda *a, **k: None)


# --- the shared gate itself ---------------------------------------------------


def test_busy_gate_wins_over_a_rows_own_true_reason():
    tab = _tab()
    tab.saving = True
    enabled, reason = inker_timeline._busy_gate(tab, ok=True, reason="")
    assert enabled is False
    assert reason == widgets.DOCUMENT_SAVING_WHY


def test_busy_gate_keeps_the_rows_own_reason_when_not_busy():
    tab = _tab()
    enabled, reason = inker_timeline._busy_gate(tab, ok=False, reason="not this frame")
    assert enabled is False
    assert reason == "not this frame"


def test_busy_gate_is_fully_enabled_when_neither_applies():
    tab = _tab()
    enabled, reason = inker_timeline._busy_gate(tab)
    assert enabled is True
    assert reason == ""


# --- _frame_menu: rows besides the already-fixed "Delete" --------------------


def test_frame_menus_insert_before_shows_the_busy_reason_while_saving(ui):
    doc = inker.Document.blank(8, 8)
    doc.add_frame()
    tab = _tab(doc)
    tab.saving = True

    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    ui.button("anchor")
    ui.open_popup("framemenu0")
    inker_timeline._frame_menu(tab, 0)
    ui.end()
    ui.end_frame()

    found = [c for c in probe.FRAME_CONTROLS if c.text == "Insert before"]
    assert found, [c.text for c in probe.FRAME_CONTROLS]
    assert found[0].enabled is False
    assert found[0].reason == widgets.DOCUMENT_SAVING_WHY


def test_frame_menus_insert_before_is_enabled_when_not_busy(ui):
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

    found = [c for c in probe.FRAME_CONTROLS if c.text == "Insert before"]
    assert found and found[0].enabled is True and found[0].reason == ""


# --- _row_menu: the layer verbs ----------------------------------------------


def test_row_menus_rename_shows_the_busy_reason_while_playing(ui):
    doc = inker.Document.blank(8, 8)
    tab = _tab(doc)
    tab.playing = True  # busy is "saving or playing"
    state = inker_state.InkerState()
    state.add(tab)
    ctx = _ctx(state)

    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    ui.button("anchor")
    ui.open_popup("layer-menu")
    inker_timeline._row_menu(ctx, tab, doc, 0)
    ui.end()
    ui.end_frame()

    found = [c for c in probe.FRAME_CONTROLS if c.text == "Rename"]
    assert found, [c.text for c in probe.FRAME_CONTROLS]
    assert found[0].enabled is False
    assert found[0].reason == widgets.DOCUMENT_SAVING_WHY


# --- _range_menu (through _cell_menu): busy outranks "has a range" ----------


def test_range_menus_copy_cels_shows_the_busy_reason_even_with_a_range_selected(ui):
    """Before the fix, ``_range_menu`` passed ``enabled=has_range`` straight
    through with no regard for ``tab.busy``, leaving the caller's bare
    ``begin_disabled(tab.busy)`` wrap to grey the row -- which shows no
    tooltip at all, since that machinery reads the row's own ``enabled``.
    With a range selected (``has_range`` true) and the tab busy, the row must
    now say *why it really is* ungettable: the busy reason, not "".
    """
    doc = inker.Document.blank(8, 8)
    doc.add_frame()
    tab = _tab(doc)
    tab.range_sel = (0, 0, 0, 1)  # a real range: "Copy cels" would else be live
    tab.saving = True
    state = inker_state.InkerState()
    state.add(tab)
    ctx = _ctx(state)

    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    ui.button("anchor")
    ui.open_popup("celmenu")
    inker_timeline._cell_menu(ctx, tab, 0, 0, True, False)
    ui.end()
    ui.end_frame()

    found = [c for c in probe.FRAME_CONTROLS if c.text == "Copy cels"]
    assert found, [c.text for c in probe.FRAME_CONTROLS]
    assert found[0].enabled is False
    assert found[0].reason == widgets.DOCUMENT_SAVING_WHY


def test_range_menus_copy_cels_still_greys_for_no_range_when_not_busy(ui):
    """The fix must not cost the ordinary case: with no range and nothing
    busy, the row keeps its own original reason."""
    doc = inker.Document.blank(8, 8)
    doc.add_frame()
    tab = _tab(doc)
    state = inker_state.InkerState()
    state.add(tab)
    ctx = _ctx(state)

    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    ui.button("anchor")
    ui.open_popup("celmenu")
    inker_timeline._cell_menu(ctx, tab, 0, 0, True, False)
    ui.end()
    ui.end_frame()

    found = [c for c in probe.FRAME_CONTROLS if c.text == "Copy cels"]
    assert found and found[0].enabled is False
    assert found[0].reason == "Select a block of cels on the timeline first."
