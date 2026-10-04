"""The 2026-10-03 audit's Medium findings on the shell's widgets (shell-31, shell-39)."""

from __future__ import annotations

import time

import pytest
from _ui_context import imgui_context

from realmspinner.studio import controls, probe, tokens, widgets


def _frame(imgui, draw):
    probe.begin_frame()
    imgui.new_frame()
    imgui.set_next_window_size((400.0, 200.0))
    imgui.set_next_window_pos((0.0, 0.0))
    imgui.begin("##host")
    draw()
    imgui.end()
    imgui.end_frame()


def _switch_alphas(monkeypatch, enabled):
    """The style alpha in force at every colour the switch asked imgui for."""
    with imgui_context(monkeypatch) as imgui:
        seen: list[float] = []
        real = imgui.get_color_u32

        def recording(*args, **kwargs):
            seen.append(float(imgui.get_style().alpha))
            return real(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(imgui, "get_color_u32", recording)
            _frame(
                imgui,
                lambda: controls.switch(
                    "Preview diff", False, enabled=enabled, reason="No mirror here."
                ),
            )
        return seen


def test_a_disabled_switch_is_drawn_at_the_disabled_alpha(monkeypatch):
    live = _switch_alphas(monkeypatch, True)
    off = _switch_alphas(monkeypatch, False)

    assert live and live == [1.0] * len(live)
    # Track, knob and label: every colour the switch paints is dimmed.
    assert len(off) == len(live)
    assert off == [pytest.approx(tokens.DISABLED_ALPHA)] * len(live)


def _spinner_and_marquee_geometry(monkeypatch, clock_value):
    """Every vertex the spinner and the marquee added to the window this frame."""
    with imgui_context(monkeypatch) as imgui:
        found: dict[str, list[tuple[float, float]]] = {}

        def body():
            draw = imgui.get_window_draw_list()
            before = len(draw.vtx_buffer)
            widgets.spinner()
            widgets.progress_bar(0.0, width=200.0)
            found["verts"] = [
                (round(v.pos.x, 3), round(v.pos.y, 3))
                for v in (draw.vtx_buffer[i] for i in range(before, len(draw.vtx_buffer)))
            ]

        with monkeypatch.context() as patch:
            patch.setattr(time, "monotonic", lambda: clock_value)
            _frame(imgui, body)
        return found["verts"]


def test_the_spinner_and_the_marquee_hold_still_under_reduce_motion(monkeypatch):
    from realmspinner.studio import motion

    monkeypatch.setattr(motion, "REDUCED", True)
    first = _spinner_and_marquee_geometry(monkeypatch, 10.0)
    later = _spinner_and_marquee_geometry(monkeypatch, 10.37)

    assert first, "nothing was drawn"
    assert first == later


def test_the_spinner_and_the_marquee_still_move_when_motion_is_on(monkeypatch):
    from realmspinner.studio import motion

    monkeypatch.setattr(motion, "REDUCED", False)
    first = _spinner_and_marquee_geometry(monkeypatch, 10.0)
    later = _spinner_and_marquee_geometry(monkeypatch, 10.37)

    assert first != later
