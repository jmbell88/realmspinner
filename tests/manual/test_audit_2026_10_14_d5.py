"""The 2026-10-14 audit, docs fixer D5: Mason, Plotter, Sirens and Muse chapters.

``tests/manual/`` gates a chapter's structure, not its claims, so each of these
reads one sentence of a chapter against the code it describes.
"""

from __future__ import annotations

import re
from pathlib import Path

_DOCS = Path(__file__).resolve().parents[2] / "docs" / "manual"


def _chapter(name: str) -> str:
    # Joined across the hard wraps, so a phrase split over two lines still matches.
    text = (_DOCS / name).read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", text)


def test_tutorial_chapter_names_the_export_files_the_exporter_writes() -> None:
    chapter = _chapter("16-dressing-a-scene.md")
    # The exporter names the GLB after the file the user picks and writes the
    # manifest as ``{stem}.json`` beside it (mode.py), so no fixed names.
    assert "`scene.glb`" not in chapter
    assert "`scene.json`" not in chapter
    assert "manifest named to match it" in chapter


def test_manual_places_the_prefabs_panel_in_the_column_the_skeleton_does() -> None:
    from realmspinner.studio import skeletons

    columns = skeletons.mason(None)
    in_right = [slot.id for slot in columns["right"].slots]
    in_left = [slot.id for slot in columns["left"].slots]
    assert "mason-prefabs" in in_right
    assert "mason-prefabs" not in in_left
    for name in ("16-dressing-a-scene.md", "28-mason.md"):
        chapter = _chapter(name)
        assert "Prefabs** panel appears in the right column" in chapter or (
            "Prefabs panel then appears in the right column" in chapter
        ), name
        assert "appears in the left column" not in chapter, name


def test_the_plotter_chapter_names_every_object_tool_letter() -> None:
    from realmspinner.studio.modes.plotter.state import OBJECT_TOOLS

    chapter = _chapter("29-plotter.md")
    line = chapter.split("Standing on an object layer changes the whole tool pill", 1)[1][:600]
    for key, _label, letter in OBJECT_TOOLS:
        assert f"`{letter}`" in line, (key, letter)


def test_the_plotter_map_panel_is_not_said_to_offer_the_empty_canvas_buttons() -> None:
    assert "offers the same two buttons" not in _chapter("29-plotter.md")


def test_chapter_09_does_not_say_new_map_asks_for_only_two_things() -> None:
    chapter = _chapter("09-building-a-map.md")
    assert "asks for two things" not in chapter
    for answer in ("Infinite", "Then"):
        assert answer in chapter.split("## Tilesets", 1)[0], answer


def test_the_manual_names_every_button_an_order_row_draws() -> None:
    chapter = _chapter("31-sirens.md")
    source = (
        Path(__file__).resolve().parents[2]
        / "src/realmspinner/studio/modes/sirens/ui/panes/orders.py"
    ).read_text(encoding="utf-8")
    drawn = len(re.findall(r"###sirens-order-(?:up|down|drop)-", source)) + 1  # + retarget
    assert drawn == 4
    assert "carries its own four buttons" in chapter
    assert "three buttons" not in chapter
    assert "the middle one" not in chapter


def test_the_manual_claim_about_correcting_the_low_nibble_matches_the_keys_it_takes() -> None:
    chapter = _chapter("31-sirens.md")
    # ``state.digit`` is 1 only after ``write_hex`` itself wrote the high digit
    # and any caret move resets it, so the low nibble needs the high one first.
    assert "is one keystroke on the second character" not in chapter
    assert "two keystrokes" in chapter


def test_the_muse_chapter_says_loop_memory_is_session_only_and_capped() -> None:
    from realmspinner.studio.modes.muse import state as muse_state

    chapter = _chapter("32-muse.md")
    assert f"last {muse_state.LOOP_MEMORY} takes you auditioned" in chapter
    assert "this session" in chapter
