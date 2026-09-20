"""The tour card's keyboard verbs (``panes/tour.py:_card_body``).

Driven against a stub imgui, ``dialogs.py``'s own reason: what is worth
pinning is which key presses advance the card, not the renderer. Written for
the 2026-09-13 audit's tour-02: ``App._shortcut`` (main.py) swallows
``K_KP_ENTER`` whenever the tour has focus, so a reader who confirms with the
numpad Enter key gets nothing -- every other Enter-confirms site in the app
reads both ``imgui.Key.enter`` and ``imgui.Key.keypad_enter``, and the tour
card did not.
"""

from __future__ import annotations

from typing import Any

from realmspinner.studio.panes import tour as tour_mod


class _Key:
    left_arrow = "left_arrow"
    right_arrow = "right_arrow"
    enter = "enter"
    keypad_enter = "keypad_enter"


class _FakeImgui:
    """Enough imgui to run ``_card_body`` once with one key "pressed"."""

    Key = _Key

    def __init__(self, pressed: str | None) -> None:
        self.pressed = pressed

    def same_line(self, *_a: Any, **_k: Any) -> None: ...
    def get_frame_height(self) -> float:
        return 0.0

    def set_cursor_pos_x(self, *_a: Any) -> None: ...
    def get_cursor_pos_x(self) -> float:
        return 0.0

    def get_content_region_avail(self) -> Any:
        return type("R", (), {"x": 0.0})()

    def text_wrapped(self, *_a: Any) -> None: ...
    def dummy(self, *_a: Any) -> None: ...
    def is_key_pressed(self, key: Any) -> bool:
        return key == self.pressed


class _Step:
    title = "Step"
    body = "Body."
    chapter = None

    class done:
        name = "manual"


class _Tour:
    title = "Tour"

    def __len__(self) -> int:
        return 2


class _State:
    def __init__(self) -> None:
        self.index = 0
        self.satisfied = False


def _run(monkeypatch: Any, pressed: str | None, *, focused: bool = True) -> list[int]:
    """Runs ``_card_body`` and returns the deltas passed to ``advance``."""

    calls: list[int] = []
    monkeypatch.setattr(tour_mod, "imgui", _FakeImgui(pressed))
    monkeypatch.setattr(tour_mod.widgets, "secondary", lambda *a, **k: None)
    monkeypatch.setattr(tour_mod.widgets, "pane_title", lambda *a, **k: None)
    monkeypatch.setattr(tour_mod.widgets, "icon_button", lambda *a, **k: False)
    monkeypatch.setattr(tour_mod.widgets, "primary_button", lambda *a, **k: False)
    monkeypatch.setattr(tour_mod.controls, "button", lambda *a, **k: False)
    monkeypatch.setattr(tour_mod, "advance", lambda ctx, delta=1: calls.append(delta))
    tour_mod._card_body(object(), _Tour(), _Step(), _State(), focused)
    return calls


def test_tour_card_advances_on_keypad_enter_when_focused(monkeypatch: Any) -> None:
    assert _run(monkeypatch, "keypad_enter") == [1]


def test_tour_card_advances_on_plain_enter_when_focused(monkeypatch: Any) -> None:
    assert _run(monkeypatch, "enter") == [1]


def test_tour_card_ignores_keypad_enter_when_not_focused(monkeypatch: Any) -> None:
    assert _run(monkeypatch, "keypad_enter", focused=False) == []
