"""The 2026-10-03 audit's poser-12 follow-up.

``score_sheet(pixel_art=False)`` leaves palette flicker out and says so in
``SheetScore.skipped``; the heatmap used to ignore that, so an HD sheet's card
read "No frame flagged" with no hint that one of the scores was never taken.
Driven through the real ``_scorecard`` with the muted-text door recorded, the
way ``test_scorecard_draw`` drives it for the ring.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _ui_context import imgui_context

from realmspinner.studio import probe
from realmspinner.studio.modes.poser import mode as poser_mode
from realmspinner.studio.modes.poser.engine import qa
from realmspinner.studio.modes.poser.ui.panes import sheet as poser_sheet


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _score(skipped: tuple[str, ...]) -> qa.SheetScore:
    cell = qa.CellScore(
        cell=0, animation="walk", direction="south", frame=0, metrics={}, flags=()
    )
    return qa.SheetScore(cells=(cell,), worst=None, flagged=0, skipped=skipped)


def _muted_lines(ui, monkeypatch, skipped: tuple[str, ...]) -> list[str]:
    monkeypatch.setattr(poser_mode, "scores", lambda ctx: _score(skipped))
    monkeypatch.setattr(
        poser_mode,
        "preview_movement",
        lambda ctx: {"key": "walk", "frames": 1, "directions": [{"key": "south"}]},
    )
    lines: list[str] = []
    monkeypatch.setattr(poser_sheet.widgets, "muted", lambda text, *a, **k: lines.append(text))
    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    try:
        poser_sheet._scorecard(
            SimpleNamespace(), SimpleNamespace(sheet_direction="south", sheet_frame=0)
        )
    finally:
        ui.end()
        ui.end_frame()
    return lines


def test_the_heatmap_says_palette_flicker_is_not_scored_on_an_hd_sheet(ui, monkeypatch):
    lines = _muted_lines(ui, monkeypatch, ("palette_flicker",))
    assert "No frame flagged." in lines
    assert any("Palette flicker is not scored on an HD sheet" in line for line in lines), lines


def test_the_heatmap_has_no_skipped_note_on_a_pixel_art_sheet(ui, monkeypatch):
    lines = _muted_lines(ui, monkeypatch, ())
    assert not any("not scored" in line for line in lines), lines
