"""Review's label loop, drawn: whether a disabled control explains itself.

``tests/test_review_mode.py`` covers ``review_mode`` (the headless half);
this file is for ``review_panes.py``'s own drawing defects, which only show up
once the buttons are actually submitted to imgui and the control census reads
them back.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _ui_context import imgui_context

from warlock.studio import probe, review_mode, review_panes


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` for why it is not a
    conftest fixture."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _draw(imgui, ctx, state):
    probe.begin_frame()
    imgui.new_frame()
    imgui.set_next_window_size((520.0, 400.0))
    imgui.set_next_window_pos((0.0, 0.0))
    imgui.begin("##host")
    review_panes.ReviewPanes()._review_label_panel(ctx, state, review_mode)
    imgui.end()
    imgui.end_frame()
    return list(probe.FRAME_CONTROLS)


def test_the_good_button_explains_its_gate_like_its_two_neighbours(ui):
    """Shell-09, the 2026-09-07 audit: "Good (A)" shares its gate with "Bad
    (R)" and "Skip (S)" beside it -- the surrounding comment says so -- but
    drew with no ``reason``, so it greyed out with no explanation while its
    two neighbours, disabled for the identical cause, said why.
    """
    ctx = SimpleNamespace(textures=None)
    # No rows at all: ``current_label`` answers None, which is the one gate
    # all three buttons share.
    state = SimpleNamespace(labels=review_mode.LabelPass(stage="mesh", rows=[]))

    controls = {c.text: c for c in _draw(ui, ctx, state) if c.kind == "button"}
    good, bad, skip = controls["Good (A)"], controls["Bad (R)"], controls["Skip (S)"]

    assert not good.enabled and not bad.enabled and not skip.enabled
    assert bad.reason == "There is nothing left to label in this pass."
    assert good.reason == bad.reason == skip.reason
