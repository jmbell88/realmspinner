"""The 2026-10-03 audit's inker-panes-04 and inker-panes-05: chapter 28 told
the reader to look for layer controls the timeline does not draw, and said
brush strokes are allowed during a save when ``canvas._input`` refuses them
whenever ``tab.busy``."""

from pathlib import Path

MANUAL = Path(__file__).resolve().parents[2] / "docs" / "manual"


def _text() -> str:
    return (MANUAL / "28-inker.md").read_text(encoding="utf-8")


def test_manual_layers_section_names_only_controls_the_timeline_draws():
    text = _text()
    assert "Above the grid are the **Blend** mode" not in text
    assert "a thumbnail and the layer's name" not in text
    assert "Under the list is the action strip" not in text
    assert "Layer properties..." in text


def test_manual_saving_section_matches_the_busy_gate_on_strokes():
    text = _text()
    assert "Brush strokes are still allowed" not in text
    assert "Brush strokes are refused" in text
