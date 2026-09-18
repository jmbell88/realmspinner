"""Documentation findings from the 2026-09-13 audit, pinned against the tree.

Each test reads its truth from the module or chapter that makes the claim
true or false rather than repeating a second hand-written copy, the same
shape as ``test_manual_promises.py`` and ``test_manual_prose_drift_2026_09_12.py``,
kept separate because those belong to different fixers' file lists.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "docs" / "manual"
STUDIO = ROOT / "src" / "warlock" / "studio"


def _chapter(name: str) -> str:
    return (MANUAL / name).read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The prose under one ``##`` heading, up to the next ``##`` (or EOF)."""
    pattern = rf"^## {re.escape(heading)}\s*$"
    match = re.search(pattern, text, re.MULTILINE)
    assert match, f"no '## {heading}' heading found"
    rest = text[match.end():]
    nxt = re.search(r"^## ", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def _flat(text: str) -> str:
    """Prose with hand-wrapped line breaks collapsed to single spaces, so a
    phrase that happens to straddle two lines in the manual's own wrapping
    still matches a plain substring check."""
    return re.sub(r"\s+", " ", text)


# --- shell-01: chapter 20's Issues claim -----------------------------------


def test_manual_status_bar_issues_claim_matches_the_app():
    """Chapter 20 must not claim the amber count is clickable or that there
    is an ``Issues`` palette command -- ``status_bar.py`` never wires
    ``is_item_clicked`` on it and ``palette.py`` defines no such command.

    T0 of the Familiar programme moved the health figure (and the rest of the
    per-item status readouts) from a status bar at the window's foot into a
    right-aligned group in the menu bar -- "The status group", chapter 20's
    new heading for it -- so this now reads that section instead.
    """
    status_bar = (STUDIO / "status_bar.py").read_text(encoding="utf-8")
    palette = (STUDIO / "palette.py").read_text(encoding="utf-8")
    assert "is_item_clicked" not in status_bar
    assert '"Issues"' not in palette

    text = _section(_chapter("20-overview.md"), "The status group")
    assert "Clicking that last one opens the Issues list" not in text
    assert "it is **Issues** in the command\npalette" not in text
    assert "Settings → Health" in text or "Settings → Health" in text


# --- shell-09: workspace layouts undocumented -------------------------------


def test_app_settings_manual_documents_workspace_layouts_section():
    panes = (STUDIO / "panes" / "app_settings.py").read_text(encoding="utf-8")
    assert "def _layouts(" in panes
    for label in ("Duplicate", "Rename...", "Reset", "Delete this layout"):
        assert label in panes

    text = _chapter("42-app-settings.md")
    assert "Workspace layouts" in text
    for label in ("Duplicate", "Rename...", "Reset", "Delete this layout"):
        assert label in text


# --- shell-10: Storage has four buttons ------------------------------------


def test_app_settings_manual_storage_section_names_all_four_buttons():
    panes = (STUDIO / "panes" / "app_settings.py").read_text(encoding="utf-8")
    assert 'controls.button("Check library")' in panes
    assert 'controls.button("Back up the index")' in panes

    text = _section(_chapter("42-app-settings.md"), "Storage")
    assert "four buttons" in text
    assert "Two buttons" not in text and "two buttons" not in text
    assert "Check library" in text
    assert "Back up the index" in text


# --- shell-11: first run has three buttons ---------------------------------


def test_before_you_begin_manual_names_the_show_me_around_button():
    first_run = (STUDIO / "panes" / "first_run.py").read_text(encoding="utf-8")
    assert '"Show me around first"' in first_run

    text = _chapter("01-before-you-begin.md")
    assert "three buttons" in text
    assert "Show me around first" in text


# --- create-06: the Count row lives in the command bar ---------------------


def test_manual_chapter_12_places_the_count_control_in_the_command_bar():
    # Derived from the module rather than spelled as a path: the file moved
    # once already (studio/create_brief.py -> modes/create/ui/brief.py).
    from warlock.studio.modes.create.ui import brief as create_brief

    brief = Path(create_brief.__file__).read_text(encoding="utf-8")
    assert "def _count(" in brief

    text = _chapter("12-tuning-what-you-get.md")
    assert "beside the seed field" not in text
    assert "command bar" in text


# --- packwright-02: Add to Packwright parks a pending tileset import -------


def test_manual_add_to_packwright_from_troupe_describes_the_tileset_popup_step():
    packwright_mode = (STUDIO / "packwright_mode.py").read_text(encoding="utf-8")
    assert "def add_rendered_sheet(" in packwright_mode
    assert "tileset_import" in packwright_mode

    for chapter, forbidden in (
        ("33-packwright.md", "contributes one sprite per cell to whatever atlas is open"),
        ("10-packing-an-atlas.md", "a rendered character sheet contributes one sprite per cell"),
        ("34-troupe.md", "contributes one sprite per cell to an open atlas"),
    ):
        text = _flat(_chapter(chapter))
        assert forbidden not in text, f"{chapter} still claims a direct contribution"
        assert "Import" in text


# --- tour-01: chapter 21 names every tour in TOURS -------------------------


def test_home_chapter_names_every_tour_in_TOURS():
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from warlock.studio.tour.scripts import TOURS

    text = _flat(_section(_chapter("21-home.md"), "New here?"))
    assert "five tours" in text
    for tour in TOURS:
        assert tour.title in text, f"tour {tour.title!r} not named in chapter 21"


# --- mason-02: chapter 31's Locked paragraph --------------------------------


def test_manual_lock_paragraph_matches_pick_not_being_gated_by_locked():
    view = (STUDIO / "mason_view.py").read_text(encoding="utf-8")
    # The engine still selects a locked node like any other -- the claim this
    # test pins is that the manual no longer says otherwise.
    text = _flat(_chapter("31-mason.md"))
    assert "keeps a node from being picked in the viewport" not in text
    assert "stops a drag in the viewport" in text
    assert view  # source read for the record; behaviour itself is untouched


# --- docs-01: CONTRIBUTING and CLAUDE.md agree on the suite size ----------
#
# CLAUDE.md moved out of the public checkout on 2026-09-16 (untracked,
# gitignored at the root), so this pairing test moved whole to
# ``dev/tests/manual/test_manual_prose_drift_2026_09_13.py`` -- it compares
# two documents and only one of them stayed public.


# --- docs-02: COMPAT.md row 314 matches the partial retirement -------------


def test_compat_md_aseprite_user_data_row_matches_partial_retirement():
    # P3 of the restructure (dev/RESTRUCTURE.md) moved studio/inker/ to
    # warlock/kernels/pixel/, aseout.py included.
    aseout = (ROOT / "src" / "warlock" / "kernels" / "pixel" / "aseout.py").read_text(
        encoding="utf-8"
    )
    assert "_user_data_chunks" in aseout

    compat = (ROOT / "docs" / "COMPAT.md").read_text(encoding="utf-8")
    assert "User data (layer/cel/tileset/tile) | dropped | #14 |" not in compat
    assert "User data (tileset/tile) | dropped | #14" in compat


# --- docs-04: chapter 17's Radius label ------------------------------------


def test_chapter_17_radius_label_matches_the_sculpt_pane():
    palette = (STUDIO / "panes" / "mason_palette.py").read_text(encoding="utf-8")
    assert 'widgets.field_label("radius (cells)")' in palette

    text = _flat(_chapter("17-dressing-a-scene.md"))
    assert "radius (cells)" in text
    assert "set **Radius** to" not in text


# --- docs-06: chapter 28's Shift+C claim ------------------------------------


def test_manual_shift_c_sentence_does_not_claim_slice_has_a_paired_tool():
    inker_ops = (STUDIO / "modes" / "inker" / "ops.py").read_text(encoding="utf-8")
    assert 'Binding("slice", "Shift+C", "tool", priority=10)' in inker_ops
    assert 'Binding("slice", "C", "tool")' in inker_ops

    text = _chapter("28-inker.md")
    assert "Shift+B/G/L/U/D/M/C" not in text
    assert "Shift+B/G/L/U/D/M" in text


# --- docs-07: chapter 08's Save as label ------------------------------------


def test_manual_poser_save_as_label_matches_button():
    controls = (STUDIO / "modes/poser/ui/panes/controls.py").read_text(encoding="utf-8")
    assert '"Save as reusable pose..."' in controls

    text = _chapter("08-rigging-and-posing.md")
    assert "Save as reusable pose..." in text
    assert re.search(r"\*\*Save as\*\*(?! reusable)", text) is None


# --- docs-08: chapter 08's Revert label --------------------------------------


def test_manual_poser_revert_label_matches_button():
    clips = (STUDIO / "modes/poser/ui/panes/clips.py").read_text(encoding="utf-8")
    assert '"Revert to shipped clips"' in clips

    text = _flat(_chapter("08-rigging-and-posing.md"))
    assert "Revert to shipped clips" in text
    assert re.search(r"\*\*Revert\*\*(?! to)", text) is None


# --- docs-09: chapter 38's Start from current settings label ----------------


def test_manual_review_start_from_current_settings_label():
    review_panes = (STUDIO / "modes/review/ui/workspace.py").read_text(encoding="utf-8")
    assert '"Start from current 2D/3D settings"' in review_panes

    text = _chapter("38-review.md")
    assert "Start from current 2D/3D settings" in text
    assert re.search(r"\*\*Start from current settings\*\*", text) is None


# --- docs-10: chapter 43's Home row claim ------------------------------------


def test_manual_home_setup_row_label_matches_landing_pane():
    landing = (STUDIO / "panes" / "landing.py").read_text(encoding="utf-8")
    assert "Generation is not set up yet" in landing

    text = _chapter("43-troubleshooting.md")
    assert "Issues / Set up models" not in text
    assert "Generation is not set up yet" in text
