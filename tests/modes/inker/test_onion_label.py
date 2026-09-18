"""The onion row's "Current layer only" checkbox label, read from a real frame.

Manual chapters 06 and 29 bold-quote this checkbox's label verbatim, and the
2026-09-18 audit (finding docs-05) found the two sides mismatched: the manual
said "Current layer only" while the control drew "current layer only". The
row's other two controls ("Ahead", "Fade") are sentence case, and a bold
quote opening a sentence in lowercase reads as a typo -- so the checkbox
moved to match the manual and its sentence-case neighbours, not the other
way. This file pins the code side of that agreement -- what the checkbox
actually renders -- without reading the manual file itself.
"""

from __future__ import annotations

import pytest
from _ui_context import imgui_context

from warlock.studio import probe
from warlock.studio.modes.inker import state as inker_state

#: The label the manual bold-quotes (chapters 06 and 29). Kept as a
#: constant, not inlined, so the one place this test would need editing if
#: the label ever changes on purpose is visible at a glance.
ONION_CURRENT_LAYER_LABEL = "Current layer only"


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` for why it is not a
    conftest fixture."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _frame(imgui, build):
    io = imgui.get_io()
    io.add_mouse_pos_event(0.0, 0.0)
    io.add_mouse_button_event(0, False)
    probe.begin_frame()
    imgui.new_frame()
    imgui.set_next_window_size((520.0, 300.0))
    imgui.set_next_window_pos((0.0, 0.0))
    imgui.begin("##host")
    build()
    imgui.end()
    imgui.end_frame()
    return list(probe.FRAME_CONTROLS)


def test_onion_current_layer_checkbox_label_matches_manual_citation(ui):
    from warlock.studio.modes.inker.ui.panes import timeline as inker_timeline

    state = inker_state.InkerState()
    state.onion = True

    controls = _frame(ui, lambda: inker_timeline._onion_controls(state))
    found = [c for c in controls if c.kind == "checkbox" and c.label == ONION_CURRENT_LAYER_LABEL]
    assert found, [c.label for c in controls if c.kind == "checkbox"]
