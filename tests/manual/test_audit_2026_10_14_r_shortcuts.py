"""The 2026-10-14 audit's residue round (fixer R), finding plotter-36.

Chapter 35's Plotter row listed the object-tool letters without ``C`` (capsule). The
in-app Ctrl+/ sheet carries no object-letter row of its own, so the chapter is the one
shortcut table that has to name every letter ``OBJECT_TOOLS`` binds.
"""

from __future__ import annotations

from pathlib import Path

from realmspinner.studio.modes.plotter.state import OBJECT_TOOLS

CHAPTER = Path(__file__).resolve().parents[2] / "docs" / "manual" / "35-shortcuts.md"


def test_the_shortcut_table_lists_every_object_tool_letter():
    text = CHAPTER.read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if "insert a shape" in line)
    for tool, _label, letter in OBJECT_TOOLS:
        if tool == "object":  # ``S`` is the Objects tool the row itself is keyed on
            continue
        assert f"`{letter}`" in row, f"{tool}: `{letter}` is missing from chapter 35's object row"
