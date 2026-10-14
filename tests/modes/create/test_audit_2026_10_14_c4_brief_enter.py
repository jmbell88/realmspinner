"""The 2026-10-14 audit, finding muse-07 (Create's twin): an Enter typed into the
prompt while the Tab ring rests on Generate pressed Generate as well."""

from __future__ import annotations

from realmspinner.studio.modes.create.ui import brief


class _Key:
    enter = "enter"
    keypad_enter = "keypad_enter"


class _Imgui:
    Key = _Key

    def __init__(self, *, typing: bool, pressed: set[str]) -> None:
        self._typing = typing
        self._pressed = pressed

    def get_io(self):
        typing = self._typing

        class _Io:
            want_text_input = typing

        return _Io()

    def is_key_pressed(self, key):
        return key in self._pressed


def test_enter_typed_into_the_prompt_is_not_a_press_of_the_focused_control(monkeypatch):
    monkeypatch.setattr(brief, "imgui", _Imgui(typing=True, pressed={"enter"}))
    assert brief._enter_pressed() is False
    monkeypatch.setattr(brief, "imgui", _Imgui(typing=True, pressed={"keypad_enter"}))
    assert brief._enter_pressed() is False


def test_enter_with_no_caret_still_presses_the_focused_control(monkeypatch):
    monkeypatch.setattr(brief, "imgui", _Imgui(typing=False, pressed={"enter"}))
    assert brief._enter_pressed() is True
    monkeypatch.setattr(brief, "imgui", _Imgui(typing=False, pressed={"keypad_enter"}))
    assert brief._enter_pressed() is True
    monkeypatch.setattr(brief, "imgui", _Imgui(typing=False, pressed=set()))
    assert brief._enter_pressed() is False
