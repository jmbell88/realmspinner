"""Chapter labels against the controls that actually draw them.

A sibling of ``test_manual_labels.py``'s idea, in its own module because this
audit's finding (docs-03) is the first one filed here rather than against the
2026-09-06 batch that file documents.
"""

from __future__ import annotations

from pathlib import Path

MANUAL = Path(__file__).resolve().parents[2] / "docs" / "manual"


def _read(key: str) -> str:
    return (MANUAL / f"{key}.md").read_text(encoding="utf-8")


def test_manual_sprite_sheet_frame_label_matches_pane():
    """The 2026-09-18 audit, finding docs-03: chapter 27 called the control
    "**Frame**"; ``sheet_panel.py`` draws it as "Frame size"
    (``form_ui.combo("frame_size", "Frame size", ...)``)."""
    text = _read("27-sprite-sheets")
    assert "**Frame size**" in text, (
        "docs/manual/27-sprite-sheets.md never names the control by its real "
        "label, 'Frame size' (sheet_panel.py:274)"
    )
    assert "**Frame** is" not in text
