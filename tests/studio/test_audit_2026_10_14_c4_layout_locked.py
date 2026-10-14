"""The 2026-10-14 audit, finding shell-documents-12: with a newer build's layout
active, a splitter drag reverted silently because the library discards every
write. The handle is now inert, with the geometry it reserved kept."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from imgui_bundle import imgui

from realmspinner.studio import layout as layout_mod
from realmspinner.studio import theme


@pytest.fixture
def frames(monkeypatch):
    # bind_workspace writes the flag; a failing test must not leak it.
    monkeypatch.setattr(layout_mod, "_SHARES_LOCKED", False)
    previous = imgui.get_current_context()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)
    yield io
    imgui.destroy_context(ctx)
    if previous is not None:
        imgui.set_current_context(previous)


class _Settings:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


def _library(readable: bool) -> Any:
    return SimpleNamespace(current=lambda: SimpleNamespace(readable=readable))


def _drag_the_splitter(io: Any, lay: Any) -> float:
    """Press on the handle, move 30 px down over three frames; -> the largest
    delta ``splitter`` reported."""
    seen = {"pos": None, "delta": 0.0}

    def frame(mouse: tuple[float, float], down: bool) -> None:
        io.add_mouse_pos_event(*mouse)
        io.add_mouse_button_event(0, down)
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((600, 400))
        imgui.begin("smoke")
        if seen["pos"] is None:
            seen["pos"] = imgui.get_cursor_screen_pos()
        seen["delta"] = max(
            seen["delta"], layout_mod.splitter("t-share", vertical=False, length=200.0)
        )
        imgui.end()
        imgui.end_frame()
        imgui.render()

    frame((900, 900), False)
    pos = seen["pos"]
    here = (pos.x + 50.0, pos.y + 2.0)
    frame(here, False)
    frame(here, True)
    for step in (10, 20, 30):
        frame((here[0], here[1] + step), True)
    frame((here[0], here[1] + 30), False)
    return seen["delta"]


def test_a_live_splitter_reports_a_drag(frames):
    lay = layout_mod.Layout(_Settings())
    lay.bind_workspace(_library(readable=True), "clay")
    assert _drag_the_splitter(frames, lay) > 0.0


def test_a_splitter_is_inert_while_the_active_layout_is_a_newer_version(frames):
    lay = layout_mod.Layout(_Settings())
    lay.bind_workspace(_library(readable=False), "clay")
    assert _drag_the_splitter(frames, lay) == 0.0
    # Settings choosing a readable layout frees it on the next frame.
    lay.bind_workspace(_library(readable=True), "clay")
    assert _drag_the_splitter(frames, lay) > 0.0
