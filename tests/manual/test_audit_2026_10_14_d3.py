"""Manual and COMPAT drift from the 2026-10-14 audit, pinned against the code it describes.

Each test here was written against the text as it stood and failed on it. Where the claim is a
count,
a label or a set (the registered recovery kinds, the doors Packwright offers, the headings the Inker
bridge pane draws) the fact is read from the code, so the next change to either side fails here and
not in a reader's hands.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "docs" / "manual"
COMPAT = ROOT / "docs" / "COMPAT.md"
SRC = ROOT / "src" / "realmspinner"

#: Every chapter this pass owned; the retired-product sweep reads all of them.
CHAPTERS = (
    "05-drawing.md",
    "06-animating.md",
    "10-packing-an-atlas.md",
    "14-casting-a-spell.md",
    "25-inker.md",
    "26-inker-animation.md",
    "30-packwright.md",
)


def _chapter(name: str) -> str:
    return (MANUAL / name).read_text(encoding="utf-8").replace("\r\n", "\n")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _paragraph_with(text: str, needle: str) -> str:
    for block in re.split(r"\n\s*\n", text):
        if needle in _flat(block):
            return _flat(block)
    raise AssertionError(f"no paragraph contains {needle!r}")


def _compat() -> str:
    return COMPAT.read_text(encoding="utf-8").replace("\r\n", "\n")


def _compat_row(start: str) -> str:
    rows = [line for line in _compat().splitlines() if line.startswith(start)]
    assert len(rows) == 1, f"{start!r}: {len(rows)} rows"
    return rows[0]


# --- shell-documents-13: the recovery chapter -------------------------------


def test_the_recovery_chapter_lists_exactly_the_registered_journal_kinds() -> None:
    from realmspinner.studio import journal

    journal.ensure_providers()
    registered = {provider.ext for provider in journal.providers()}
    paragraph = _paragraph_with(_chapter("25-inker.md"), "This is no longer only Inker's")
    named = set(re.findall(r"`(\.[a-z]+)`", paragraph))
    assert named == registered, (named, registered)
    # The pose provider is gone; the chapter must not promise one or describe its refusal.
    assert ".pose.json" not in _chapter("25-inker.md")
    assert "a pose you are authoring" not in _chapter("25-inker.md")
    assert "A pose is the one that can decline" not in _chapter("25-inker.md")


# --- inker-sheets-03 / docs-F2-13: retired sheet products -------------------


def test_the_manual_names_no_retired_sheet_product_as_a_source() -> None:
    for name in CHAPTERS:
        text = _flat(_chapter(name))
        for needle in (
            "character sheet",
            "3D sprite sheet",
            "baked from a 3D",
            "rendered sheet",
            "Character sheet",
        ):
            assert needle not in text, f"{name} still says {needle!r}"


def test_manual_has_no_3d_sprite_sheet_baking_claims() -> None:
    assert "sprite-sheet sidecar format" in _flat(_chapter("25-inker.md"))
    assert "from Inker is a regular grid" in _flat(_chapter("30-packwright.md"))


# --- docs-F1-03: Packwright's doors -----------------------------------------


def test_packing_chapter_door_count_matches_the_door_list() -> None:
    text = _chapter("10-packing-an-atlas.md")
    section = text.split("## Getting sprites in", 1)[1].split("\n## ", 1)[0]
    words = {"Five": 5, "Six": 6, "Four": 4, "Seven": 7}
    opener = re.match(r"\s*(\w+) doors", section)
    assert opener is not None and opener.group(1) in words
    bullets = re.findall(r"^- \*\*", section, flags=re.MULTILINE)
    assert words[opener.group(1)] == len(bullets)
    assert "From Poser" not in section


# --- inker-flourish-17 / docs-F1-11: the casting chapter --------------------


def test_the_casting_a_spell_chapter_does_not_point_at_a_models_row_for_the_text_model() -> None:
    from realmspinner.studio.modes.inker import flourish as inker_flourish

    paragraph = _paragraph_with(_chapter("14-casting-a-spell.md"), "text model")
    assert "Settings → Models" not in paragraph
    assert inker_flourish.TEXT_MODEL_DIR in paragraph


def test_flourish_chapter_names_the_facing_options_by_their_labels() -> None:
    from realmspinner.studio.modes.inker.ui.panes import flourish as pane

    labels = [label for _, label in pane.DIRECTION_OPTIONS]
    text = _flat(_chapter("14-casting-a-spell.md"))
    for label in labels[1:]:
        assert f"**{label}**" in text, label


# --- docs-F2-12: the Inker bridge pane's own headings -----------------------


def test_inker_manual_export_png_and_pipeline_section_match_the_panes() -> None:
    from realmspinner.studio import widgets

    chapter = _flat(_chapter("25-inker.md"))
    pane = (SRC / "studio" / "modes" / "inker" / "ui" / "panes" / "generate.py").read_text(
        encoding="utf-8"
    )
    # The pane's file block has no Export PNG; it is a File-menu op.
    assert 'widgets.section("Drawing file")' in pane
    assert "**Save as** and **Export PNG** are in the" not in chapter
    assert "**Export PNG** is in the **File** menu" in chapter
    # The heading the verbs sit under is the widget's, not "pipeline".
    assert "**pipeline** section" not in chapter
    assert f"**{widgets.EXITS_HEADING}**" in chapter


# --- the fixer hand-backs ----------------------------------------------------


def test_the_manual_names_the_onion_depth_and_the_fps_box() -> None:
    from realmspinner.studio.modes.inker.ui.panes import timeline

    limit = f"0 to {timeline.MAX_ONION}"
    for name in ("06-animating.md", "26-inker-animation.md"):
        text = _flat(_chapter(name))
        assert limit in text, name
        assert "**FPS**" in text, name


def test_the_gif_open_paragraph_says_identical_frames_merge() -> None:
    assert "identical consecutive frames as one frame" in _flat(_chapter("26-inker-animation.md"))


def test_the_clipboard_is_documented_as_shared_between_tabs() -> None:
    assert "shared between tabs" in _flat(_chapter("25-inker.md"))


def test_the_fit_remedy_names_only_options_that_apply() -> None:
    text = _flat(_chapter("30-packwright.md"))
    assert "unless it is already at the ceiling" in text
    assert "MaxRects only" in text


# --- inker-codecs-13 / docs-D-06..08: COMPAT.md's Aseprite ledger ------------


def test_compat_reference_layer_rows_agree_on_visibility() -> None:
    written = _compat_row("| a reference layer |")
    read = _compat_row("| A reference layer |")
    assert "opened hidden" not in written
    assert "VISIBLE bit" in written and "VISIBLE bit" in read


def test_compat_refusal_row_names_the_real_tilemap_and_ceiling_refusals() -> None:
    row = _compat_row("| Everything refused outright |")
    assert "bit-mask offset" not in row
    assert "positioned off its tile grid" in row
    assert "three canvases" in row
    assert "layer budget" in row
    assert "raster cel on a tilemap layer" in row


def test_compat_colour_profile_row_states_what_is_warned() -> None:
    row = _compat_row("| Colour profile | warned")
    assert "ICC" in row and "fixed gamma" in row and "silent" in row


def test_compat_lists_the_writers_refusals() -> None:
    row = _compat_row("| Everything the writer refuses outright |")
    for needle in ("65,535", "MAX_DECODE_PIXELS", "not grey", "tilemap layer", "256 colours"):
        assert needle in row, needle
    grey = _compat_row("| Grayscale storage |")
    assert "refused" in grey


def test_compat_has_a_row_for_the_xml_control_character_refusal() -> None:
    assert "| `a control character in a name or property` | refused |" in _compat()
