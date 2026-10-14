"""The 2026-10-14 audit, finding plotter-09: a tab change made outside the bar
(Ctrl+Tab's ``cycle``) was undone by the very next frame, because the bar copied
imgui's own selected tab back into the state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from imgui_bundle import imgui

from realmspinner.studio import docmodes, theme


@dataclass
class _Tab:
    uid: str
    label: str
    dirty: bool = False


@pytest.fixture
def frames():
    previous = imgui.get_current_context()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)

    def draw(state: Any) -> None:
        imgui.new_frame()
        imgui.set_next_window_size((900, 300))
        imgui.begin("bar")
        docmodes.tab_bar(None, state, "test-tabs", lambda tab: state.close(tab.uid))
        imgui.end()
        imgui.end_frame()
        imgui.render()

    yield draw
    imgui.destroy_context(ctx)
    if previous is not None:
        imgui.set_current_context(previous)


def _state() -> Any:
    state = docmodes.DocTabs()
    for uid in ("a", "b", "c"):
        state.add(_Tab(uid, f"Doc {uid}"))
    return state


def test_cycling_tabs_survives_the_next_tab_bar_frame(frames):
    state = _state()
    for _ in range(3):
        frames(state)
    assert state.active_uid == "c"
    state.cycle(1)
    assert state.active_uid == "a"
    for _ in range(3):
        frames(state)
    assert state.active_uid == "a", "the bar handed the old tab back"
    state.cycle(-1)
    for _ in range(3):
        frames(state)
    assert state.active_uid == "c"


def test_a_state_restored_on_a_later_tab_is_not_reset_to_the_first_by_the_bar(frames):
    state = _state()
    state.activate("b")
    for _ in range(3):
        frames(state)
    assert state.active_uid == "b"
