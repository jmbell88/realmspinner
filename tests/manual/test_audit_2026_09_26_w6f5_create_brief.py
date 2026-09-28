"""The 2026-09-26 audit, finding create-brief-02.

Chapters 02 and 22 named "**Asset type**"/"**Description**" as if they were
visible captions on Create's P5 command bar, but the bar's type combo
(``##generation-type``) and prompt field (``##brief-prompt``) -- both drawn in
``studio/modes/create/ui/brief.py`` -- are unlabelled imgui widgets: neither
string is drawn anywhere on the bar. Chapter 22 also put the whole brief in
"the left sidebar", which was true when the settings pane was a single column
but no longer matches the P5 split: the command bar is now the top row and the
recipe column is the sidebar underneath it.

Reworded to describe the controls by position instead of a label that
doesn't exist, rather than inventing a second, competing label scheme.
"""

from __future__ import annotations

from pathlib import Path

MANUAL = Path(__file__).resolve().parents[2] / "docs" / "manual"
BRIEF = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "realmspinner"
    / "studio"
    / "modes"
    / "create"
    / "ui"
    / "brief.py"
)


def _read(key: str) -> str:
    return (MANUAL / f"{key}.md").read_text(encoding="utf-8")


def test_manual_names_no_control_label_the_create_bar_does_not_draw():
    """Neither ``brief.py`` (the command bar's own drawing code) nor the two
    chapters that describe it may claim a "Asset type"/"Description" caption:
    the code never draws one, so a chapter naming it is describing a control
    that isn't there.
    """
    source = BRIEF.read_text(encoding="utf-8")
    for stale_label in ("Asset type", "Description"):
        assert stale_label not in source, (
            f"brief.py now draws a {stale_label!r} string; if the bar grew a "
            "real caption, name it directly in the manual instead of by "
            "position"
        )

    for chapter in ("02-your-first-asset", "22-generating-references"):
        text = _read(chapter)
        assert "**Asset type" not in text, (
            f"docs/manual/{chapter}.md still bolds a stale 'Asset type' label; "
            "the command bar's type combo (##generation-type) draws no label "
            "at all -- describe it by position instead"
        )
        assert "**Description" not in text, (
            f"docs/manual/{chapter}.md still bolds a stale 'Description' "
            "label; the command bar's prompt field (##brief-prompt) draws no "
            "label at all -- describe it by position instead"
        )


def test_manual_22_places_the_brief_on_the_top_bar_not_the_sidebar():
    """Chapter 22's opening paragraph used to put "everything in this
    chapter" in "the left sidebar" -- true of the old single-column settings
    pane, but the P5 split puts the command bar (type combo, prompt, count,
    Generate, Reset) across the top and only the recipe column in the
    sidebar."""
    text = _read("22-generating-references")
    assert "command bar across the top" in text, (
        "docs/manual/22-generating-references.md's opening paragraph should "
        "place the command bar across the top, matching the P5 layout "
        "described later in the same chapter's 'screen at a glance' section"
    )
    assert "the 2D reference mode's settings pane, in the left sidebar" not in text, (
        "docs/manual/22-generating-references.md still describes the whole "
        "brief as living in the left sidebar, which is now only true of the "
        "recipe column"
    )
