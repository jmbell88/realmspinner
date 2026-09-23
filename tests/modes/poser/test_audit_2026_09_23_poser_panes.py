"""Regression tests for the poser-panes fix-brief of the 2026-09-23 audit.

One module for four findings with no existing test file they sit beside more
naturally than this one: poser-03 (``new_key`` reachable while scrubbing),
poser-04 (the pixel-art branch of ``_pixel_report_lines`` never checked
``exact_stride``), poser-07 (the nearest-neighbour hint shown only for a
custom size, never a 24/48/96 preset) and poser-08 (two stale citations of a
deleted Troupe test file, docstring-only, no behaviour and so no test here).
"""

from __future__ import annotations

import pytest
from _ui_context import imgui_context

from modes.poser.test_poser_mode import _clip_ctx, _turned
from realmspinner.studio.modes.poser import mode as poser_mode
from realmspinner.studio.modes.poser.ui.panes import sheet as poser_sheet
from realmspinner.studio.modes.poser.ui.panes.clips import _new_key_reason


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` for why it is not a
    fixture there. ``ui/panes/send.py`` and ``ui/panes/sheet.py`` both call
    real imgui/``widgets``/``controls`` functions that need a live frame."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui


# --- poser-03: "New key from pose..." while scrubbing ----------------------


def test_new_key_from_pose_refuses_while_scrubbing_an_in_between_frame():
    """``new_key`` used to check nothing but its own arguments, so scrubbing
    to an in-between frame and then naming a new key authored the
    *interpolated* pose as a brand-new key pose -- ``capture_key`` ("Update
    key") already refused the identical state by name.
    """
    ctx, state = _clip_ctx()
    poser_mode.scrub(ctx, 1)
    assert state.frame >= 0, "the scrub must have landed on an in-between frame"

    before_keys = list(state.open_clip()["keys"])
    poser_mode.new_key(ctx, "A crouch")

    assert state.open_clip()["keys"] == before_keys, (
        "an in-between frame must not become an authored key"
    )
    assert state.key_pose("A crouch") is None
    assert any("in-between frame" in msg for msg, _kind in ctx.toasts)


def test_new_key_from_pose_still_works_on_a_selected_key():
    """The refusal is scoped to scrubbing, not to every call -- the existing
    happy path (``test_a_new_key_is_authored_from_the_armature_and_inserted``
    in ``test_poser_mode.py``) must still pass undisturbed."""
    ctx, state = _clip_ctx()
    editor = ctx.poser_viewer.editor
    poser_mode.select_key(ctx, 0)
    editor.apply({"head": _turned()}, dirty=True)
    assert state.frame < 0

    poser_mode.new_key(ctx, "A crouch")

    assert state.key_pose("A crouch") is not None
    assert state.open_clip()["keys"] == ["A", "A crouch", "B", "C"]


def test_new_key_reason_names_the_scrubbed_frame_when_posing():
    """The pane's reason function, checked in isolation (no imgui frame
    needed) -- ``_update_key_reason``'s own justification for being pulled
    out as a pure function applies here too."""
    assert _new_key_reason(True, 3) == (
        "That is an in-between frame, not a key. Pick a key first."
    )
    assert _new_key_reason(True, -1) == ""


def test_new_key_reason_default_frame_keeps_the_old_two_argument_callers_working():
    """``frame`` was added with a default so ``test_poser_panes_smoke.py``'s
    existing ``_new_key_reason(False, ...)`` calls -- written before this
    finding -- still mean "not posing yet" rather than raising a
    ``TypeError``."""
    assert _new_key_reason(False) == "The skeleton preview is still loading."


# --- poser-04: pixel-art sheets never reported an inexact stride -----------


def test_pixel_report_lines_reports_an_inexact_stride_for_pixel_art_sheets_too():
    """Only the HD branch of ``_pixel_report_lines`` read ``exact_stride``;
    a 24/48/96 px pixel-art sheet whose render size does not divide evenly
    was NEAREST-resized with no line on screen saying so."""
    report = {
        "style": "pixel_art",
        "colors": 16,
        "palette_name": "nes",
        "exact_stride": False,
    }
    lines = poser_sheet._pixel_report_lines(report)
    assert "frame size is not an exact stride of the render" in lines


def test_pixel_report_lines_says_nothing_extra_when_the_stride_is_exact():
    report = {
        "style": "pixel_art",
        "colors": 16,
        "palette_name": "nes",
        "exact_stride": True,
    }
    lines = poser_sheet._pixel_report_lines(report)
    assert not any("stride" in line for line in lines)


# --- poser-07: the nearest-neighbour hint for preset sizes ------------------


def test_size_combo_warns_about_nearest_neighbour_for_a_non_dividing_preset_too(ui, monkeypatch):
    """``ui/panes/send.py``'s ``_size`` gated the "don't divide 512" hint
    inside the ``if state.custom_size:`` branch, so picking 24, 48 or 96 px
    off the ladder itself -- not typing a custom number -- said nothing, even
    though 512 % 24, 512 % 48 and 512 % 96 are all nonzero and the sheet is
    NEAREST-resized exactly as a bad custom size would be.

    A real imgui frame (``ui``), because ``_size`` calls ``widgets.
    labeled_combo`` and ``controls.input_int``, which need one; the combo is
    never opened, so ``controls.combo`` returns the value already selected
    (``picked = current``) rather than requiring a simulated click.
    """
    from realmspinner.kernels import charsheet
    from realmspinner.studio.modes.poser.ui.panes import send as poser_send

    for preset in (24, 48, 96):
        assert charsheet.RENDER_SIZE % preset != 0, (
            f"{preset} px must not divide {charsheet.RENDER_SIZE} for this "
            "case to exercise the bug; if the render size ever changes, so "
            "must this fixture"
        )

    notes: list[str] = []
    monkeypatch.setattr(poser_send.widgets, "muted_wrapped", lambda text: notes.append(text))
    state = poser_send.PoserSend(logical_size=24, custom_size=False)
    options = {"logical_sizes": (24, 32, 48, 64, 96), "logical_size_range": (8, 256)}

    ui.new_frame()
    ui.begin("host")
    try:
        poser_send._size(state, options)
    finally:
        ui.end()
        ui.end_frame()

    assert any("don't divide 512" in note for note in notes)


def test_sheet_pane_size_combo_warns_about_a_non_dividing_preset_too(ui, monkeypatch):
    """The same finding (poser-07), in ``ui/panes/sheet.py``'s own copy of
    the size row (``_logical_size``, drawn in the Sheet pane's send form
    rather than the Send-to-Poser dialog)."""
    from realmspinner.studio import forms
    from realmspinner.studio.modes.poser.ui.panes import sheet as poser_sheet_mod

    notes: list[str] = []
    monkeypatch.setattr(
        poser_sheet_mod.widgets, "muted_wrapped", lambda text: notes.append(text)
    )
    form = {"logical_size": 48}
    options = {"logical_sizes": (24, 32, 48, 64, 96), "logical_size_range": (8, 256)}

    ui.new_frame()
    ui.begin("host")
    try:
        with forms.Form("poser-panes-audit") as form_ui:
            poser_sheet_mod._logical_size(form, form_ui, options)
    finally:
        ui.end()
        ui.end_frame()

    assert any("don't divide 512" in note for note in notes)
