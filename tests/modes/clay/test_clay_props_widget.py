"""``clay_props._widget`` picks a control from the *default's* type, and must
only claim a shape it can actually draw.

Found while wiring up the array-valued generators a lathe profile needs
(``list[list[float]]``): the tuple/list branch assumed every sequence default
was a flat 2- or 3-tuple of numbers, which is true of ``box``'s and
``plane``'s ``size`` but not of everything a future generator's defaults
dict can hold. Three ways that assumption breaks, each pinned below:

* a flat sequence of 4 or more numbers was silently cut down to 3 and written
  back that short -- the object's stored params then disagreed with what a
  five-number generator was actually called with, with nothing on screen to
  say so;
* a *nested* default such as ``[[0.5, -0.5], [0.5, 0.5]]`` (what a lathe
  profile looks like) matched ``len(default) == 2`` and handed two Python
  lists to ``imgui.input_float2``, which has no try/except above it on the
  frame thread and takes the whole app down;
* a saved document whose value for the key is a bare scalar where the
  default is a sequence made ``list(value)`` raise ``TypeError`` outright.

The fix routes all three through the existing read-only fallback --
``widgets.secondary(f"{key}: {value!r}")`` -- the same one a parameter type
nobody has built a widget for yet already uses, rather than growing a special
case for each.
"""

from __future__ import annotations

import pytest
from _ui_context import imgui_context

from realmspinner.studio.modes.clay.ui.panes import props as clay_props


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` for why this is not a
    conftest fixture (a second ``conftest.py`` here would shadow the root
    one's importable name for the tests that do ``from conftest import``)."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _drawn(ui, key, value, default):
    """Run ``_widget`` inside one real (headless) imgui frame."""
    ui.new_frame()
    ui.begin("##host")
    try:
        out, changed = clay_props._widget(key, value, default)
    finally:
        ui.end()
        ui.end_frame()
    return out, changed


def test_a_five_number_parameter_is_shown_not_cut_down_to_three(ui) -> None:
    """Fails today: the tuple/list branch takes the ``len(default) != 2``
    path to ``input_float3`` regardless of the real length, so this comes
    back as a 3-tuple and two of the five numbers are gone with no warning."""
    value = default = (1.0, 2.0, 3.0, 4.0, 5.0)
    out, changed = _drawn(ui, "five", value, default)
    assert changed is False
    assert tuple(out) == value


def test_a_profile_shaped_parameter_does_not_take_the_panel_down(ui) -> None:
    """Fails today: a nested default of length 2 matches the
    ``len(default) == 2`` branch and hands ``input_float2`` two Python lists
    instead of two floats -- this call raises rather than returning."""
    value = default = [[0.5, -0.5], [0.5, 0.5]]
    out, changed = _drawn(ui, "profile", value, default)
    assert changed is False
    assert out == value


def test_a_scalar_value_under_a_sequence_default_does_not_raise(ui) -> None:
    """Fails today: ``list(value)`` on a bare scalar raises ``TypeError``
    before the widget is even chosen -- a saved document whose value for this
    key disagrees in shape with the current default must not take the panel
    down for it."""
    out, changed = _drawn(ui, "size", 1.0, (1.0, 1.0, 1.0))
    assert changed is False
    assert out == 1.0


def test_a_box_size_still_draws_and_edits(ui, monkeypatch) -> None:
    """The case this branch exists for must keep working: a flat 3-tuple
    default with a same-shaped value is still handed to ``input_float3``."""
    from realmspinner.studio import controls

    monkeypatch.setattr(
        controls, "input_float3", lambda label, values: (True, (2.0, 1.0, 1.0))
    )
    out, changed = _drawn(ui, "size", (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert changed is True
    assert out == (2.0, 1.0, 1.0)


def test_a_plane_size_still_draws_and_edits(ui, monkeypatch) -> None:
    """The other case this branch exists for: a flat 2-tuple default, as
    ``plane`` and ``grid``'s ``size`` are."""
    from realmspinner.studio import controls

    monkeypatch.setattr(
        controls, "input_float2", lambda label, values: (True, (2.0, 1.0))
    )
    out, changed = _drawn(ui, "size", (1.0, 1.0), (1.0, 1.0))
    assert changed is True
    assert out == (2.0, 1.0)


def test_removing_the_last_palette_material_greys_out_with_a_stated_reason() -> None:
    """The 2026-09-12 audit's clay-05: with exactly one material, the old
    inline ``removable = users == 0 and len(doc.materials) > 1`` was always
    False, but the explanatory line right below it was gated on that same
    ``len(doc.materials) > 1``, so it never fired either -- Remove greyed out
    with nothing on screen saying a document must keep at least one material,
    unlike the "still in use" case one line below it in the same file.

    ``_palette_remove_reason`` is the pure function this decision now lives
    in, so it is testable without a live imgui frame the way
    ``clay_ops.reason_for`` and ``plotter_menu._layer_reason`` already are.
    """
    # A single material and no users at all: the one case the old code left
    # silently disabled.
    reason = clay_props._palette_remove_reason(material_count=1, users=0)
    assert reason, "Remove is greyed with no stated reason"
    assert "at least one material" in reason

    # The pre-existing "still in use" case must keep working unchanged.
    reason = clay_props._palette_remove_reason(material_count=2, users=3)
    assert "3 face(s) use this slot" in reason

    # And the control is enabled -- no reason -- only when it truly can act.
    assert clay_props._palette_remove_reason(material_count=2, users=0) == ""
