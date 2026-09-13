"""``controls.input_vec``: X/Y/Z(/W) letters over a vector field's boxes.

Added for the 2026-09-12 consistency pass: Clay's position/scale/rotation
fields were three (or four) identical-looking boxes with one label to their
left, so a user had to count boxes rather than read them to tell a vector
field's components apart. ``input_vec`` draws one axis letter over each
component -- centred using the same per-component width imgui's own
``InputScalarN`` splits an N-wide field into -- then dispatches to the
existing ``input_float{n}`` looked up by name, so a caller (or a test) that
monkeypatches ``controls.input_float3``/``input_float2`` still intercepts it.
"""

from __future__ import annotations

import pytest
from _ui_context import imgui_context

from warlock.studio import controls


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _spy_draw_list(monkeypatch, imgui):
    """Swap the *first* ``get_window_draw_list()`` call for a spy, real after.

    ``input_vec`` is the only caller that wants to be watched; whatever runs
    after it (the dispatched ``input_float{n}`` -> ``_finish_item`` -> a focus
    or selection ring) wants the real draw list back, since the spy below
    only knows ``add_text``.
    """
    real = imgui.get_window_draw_list
    texts: list[str] = []

    class _Spy:
        def add_text(self, pos, col, text):  # noqa: ANN001 - imgui's own shape
            texts.append(text)

    state = {"n": 0}

    def spy():
        state["n"] += 1
        return _Spy() if state["n"] == 1 else real()

    monkeypatch.setattr(imgui, "get_window_draw_list", spy)
    return texts


def test_input_vec_draws_one_letter_per_axis(ui, monkeypatch) -> None:
    """Fails against a bare ``input_float3`` call: nothing draws a letter at
    all, so there is nothing here to spy on."""
    texts = _spy_draw_list(monkeypatch, ui)
    ui.new_frame()
    ui.begin("##host")
    try:
        controls.input_vec("v##t1", [1.0, 2.0, 3.0], ("X", "Y", "Z"))
    finally:
        ui.end()
        ui.end_frame()
    assert texts == ["X", "Y", "Z"]


def test_input_vec_draws_four_letters_for_a_quaternion(ui, monkeypatch) -> None:
    texts = _spy_draw_list(monkeypatch, ui)
    ui.new_frame()
    ui.begin("##host")
    try:
        controls.input_vec("q##t2", [0.0, 0.0, 0.0, 1.0], ("X", "Y", "Z", "W"))
    finally:
        ui.end()
        ui.end_frame()
    assert texts == ["X", "Y", "Z", "W"]


def test_input_vec_returns_the_field_it_dispatches_to_unchanged(ui) -> None:
    """No edit made this frame: the same ``(changed, values)`` shape a bare
    ``input_float3`` call would answer with, since ``input_vec`` must not
    alter what it returns."""
    ui.new_frame()
    ui.begin("##host")
    try:
        changed, out = controls.input_vec("v##t3", [1.0, 2.0, 3.0], ("X", "Y", "Z"))
    finally:
        ui.end()
        ui.end_frame()
    assert changed is False
    assert list(out) == [1.0, 2.0, 3.0]


def test_input_vec_dispatches_by_module_attribute_lookup(ui, monkeypatch) -> None:
    """A caller that monkeypatches ``controls.input_float3`` -- exactly what
    ``tests/clay/test_clay_props_widget.py`` already does -- must still
    intercept a call made through ``input_vec``. Fails if ``input_vec`` were
    written to call ``imgui.input_float3`` (or a captured reference) directly
    instead of looking the name up on this module at call time."""
    monkeypatch.setattr(
        controls, "input_float3", lambda label, values: (True, (9.0, 8.0, 7.0))
    )
    ui.new_frame()
    ui.begin("##host")
    try:
        changed, out = controls.input_vec("v##t4", [1.0, 2.0, 3.0], ("X", "Y", "Z"))
    finally:
        ui.end()
        ui.end_frame()
    assert changed is True
    assert out == (9.0, 8.0, 7.0)


def test_input_vec_rejects_a_mismatched_axis_and_value_count() -> None:
    with pytest.raises(ValueError):
        controls.input_vec("v##t5", [1.0, 2.0], ("X", "Y", "Z"))


def test_input_vec_rejects_an_axis_count_outside_two_to_four() -> None:
    with pytest.raises(ValueError):
        controls.input_vec("v##t6", [1.0], ("X",))
