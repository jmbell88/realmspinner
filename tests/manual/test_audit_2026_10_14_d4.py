"""The 2026-10-14 audit's manual findings for chapters 01, 03, 12, 20-22, 33-35 and 40-46.

``tests/manual/`` gates the manual's structure; these read a chapter's *claims*
against the code they describe, so the next removal (Poser and the
character-sheet pipeline left in eed32549 and took half a dozen sentences with
it) fails here instead of drifting.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "docs" / "manual"


def _chapter(name: str) -> str:
    return (MANUAL / f"{name}.md").read_text(encoding="utf-8").replace("\r\n", "\n")


def _flat(text: str) -> str:
    """One line, so a sentence the file wraps is still one string."""
    return re.sub(r"\s+", " ", text)


def _paragraph(text: str, lead: str) -> str:
    """The blank-line-delimited paragraph containing ``lead``."""
    for block in text.split("\n\n"):
        if lead in block:
            return _flat(block)
    raise AssertionError(f"no paragraph contains {lead!r}")


def _section(text: str, heading: str, level: str = "##") -> str:
    start = text.index(f"{level} {heading}")
    rest = text[start + len(level) + 1 + len(heading) :]
    end = re.search(rf"^{re.escape(level)} ", rest, re.M)
    return rest[: end.start()] if end else rest


# --- shell-review-settings-02 -------------------------------------------------


def test_manual_42_agent_paragraph_names_no_character_library_or_sprite_sheet_tools():
    paragraph = _paragraph(_chapter("42-app-settings"), "**AI agents.**").lower()
    for gone in (
        "sprite sheet",
        "species name",
        "can only add to it",
        "character library",
        "rig or",
    ):
        assert gone not in paragraph, gone
    assert "clay" in paragraph


# --- shell-chrome-08 ----------------------------------------------------------


def test_overview_inker_menu_roots_match_the_ops_and_the_workspace_menu_count_matches_the_modes():
    from realmspinner.studio import modes
    from realmspinner.studio.modes.inker import ops as inker_ops

    roots = {op.menu.split("/")[0] for op in inker_ops.OPS if op.menu} - {"File", "Edit", "View"}
    text = _flat(_chapter("20-overview"))
    sentence = text.split("Inker is the one exception:", 1)[1].split("and it also adds rows", 1)[0]
    named = set(re.findall(r"\*\*([A-Z][a-z]+)\*\*", sentence))
    assert named == roots, (named, roots)
    numbers = {
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
    }
    count = next(
        n for word, n in numbers.items() if f"contributes {word} mode-specific" in sentence
    )
    assert count == len(roots)
    held = re.search(r"it holds all (\w+) modes", text)
    assert held and numbers[held.group(1)] == len(modes.MODES)


# --- shell-home-library-14, docs-F2-14, docs-F1-06 -------------------------------


def test_the_home_chapter_names_exactly_the_new_menu_entries():
    from realmspinner.studio.modes.home.ui.panes import landing

    labels = [label.removeprefix("New ") for _key, label, _icon, _action in landing.NEW_ITEMS]
    text = _flat(_chapter("21-home"))
    sentence = text.split("a menu behind it with the ", 1)[1].split(
        " things this app can begin from nothing: ", 1
    )
    words = {"eight": 8, "nine": 9, "seven": 7, "ten": 10}
    assert words[sentence[0]] == len(labels)
    listed = sentence[1].split(". It used to", 1)[0]
    for label in labels:
        assert re.search(rf"\ban? {re.escape(label)}\b", listed), label
    assert "character" not in listed


def test_home_manual_resume_routing_matches_asset_open_route():
    from realmspinner import retired

    text = _flat(_chapter("21-home"))
    resume = text.split("the stage or workspace that holds the product for a", 1)[1].split(
        "Work made in a Create workspace", 1
    )[0]
    assert "character sheet" in resume and "Library" in resume  # said to open the Library
    assert "rig, a sprite sheet or a track" in resume
    assert "sheet" in retired.RETIRED_KINDS


def test_shortcut_chapter_describes_home_resume_as_the_grid_it_is():
    text = _flat(_chapter("35-shortcuts"))
    assert "Resume list is one column" not in text
    assert "Resume is a grid of thumbnails" in text


# --- create-brief-03 / create-workspace-04 / create-workspace-08 -----------------


def test_manual_22_does_not_promise_character_exported_sheets():
    text = _flat(_chapter("22-generating-references"))
    assert "Character meshes can switch" not in text
    assert "exported sheets once those sheets exist" not in text


def test_the_manual_names_the_ordering_figure_as_the_ui_does():
    source = (
        ROOT / "src" / "realmspinner" / "studio" / "modes" / "create" / "ui" / "workspace.py"
    ).read_text(encoding="utf-8")
    assert "Order hint" in source
    section = _flat(
        _section(_chapter("22-generating-references"), "Which candidate to look at first", "###")
    )
    assert "**order hint**" in section
    assert "labelled *rank*" not in section and "carries a **rank**" not in section


# --- clay-agent-tools-06, agents-protocol-07 ---------------------------------------


def test_manual_46_does_not_deny_the_booleans_clay_op_runs():
    from realmspinner.studio.modes.clay import ops

    names = {op.name for op in ops.OPS}
    assert {"union", "difference", "intersection"} <= names
    text = _flat(_chapter("46-extending"))
    denial = text.split("There is no figure,", 1)[1].split("because Clay has none", 1)[0]
    assert "boolean," not in denial
    assert "through `clay_op`" in text


def test_manual_names_no_agent_tool_the_catalogue_does_not_offer():
    from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

    offered = {tool.name for tool in agent_clay.tools()}
    for chapter in ("12-putting-it-in-a-game", "42-app-settings", "46-extending"):
        text = _chapter(chapter)
        assert "character_export" not in text, chapter
        for name in set(re.findall(r"`(clay_[a-z_]+)`", text)):
            if name.endswith("_"):  # a family prefix such as `clay_reference_`
                continue
            assert name in offered, (chapter, name)


def test_manual_12_names_no_export_that_was_removed():
    text = _chapter("12-putting-it-in-a-game")
    for gone in ("Knight-frames", "Knight.png", "Knight.json", "an export asked to replace"):
        assert gone not in text, gone
    assert "Knight-godot-2" in text


# --- poser-poses-06 -----------------------------------------------------------------


def test_the_clip_map_docs_say_the_importer_is_a_deferred_back_end():
    extending = _flat(_chapter("46-extending"))
    pipelines = _flat(_chapter("45-pipelines"))
    assert "deferred back end" in extending
    assert "no control" in extending and "Import Animation" in extending
    assert "no control in front of it" in pipelines and "Import Animation" in pipelines
    assert '"Import clip" adds' not in pipelines


# --- docs-F1-02 -----------------------------------------------------------------------


def test_extending_chapter_names_the_three_partitioning_mode_sets():
    from realmspinner.studio import main, modes

    # The sets the gate in tests/studio/test_studio_state.py really partitions.
    assert set(main._SINGLE_PANE_MODES) | set(modes.VIEWPORT_MODES) | set(
        modes.WORKSPACE_MODES
    ) == set(modes.KEYS)
    assert modes.WORKSPACE_MODES < modes.WORK_MODES
    text = _flat(_chapter("46-extending"))
    assert (
        "`VIEWPORT_MODES`, `WORKSPACE_MODES` and `main._SINGLE_PANE_MODES` must partition" in text
    )
    assert "`WORK_MODES`, `WORKSPACE_MODES` and `main._SINGLE_PANE_MODES`" not in text


# --- service-kinds-04 / docs-F2-04 / docs-F2-05 / docs-F1-10 ----------------------------


def test_the_library_chapter_names_no_retired_kind_among_the_open_offering_followups():
    text = _flat(_chapter("33-library-and-jobs"))
    bullet = text.split("- A finished follow-up (", 1)[1].split(")", 1)[0]
    assert "character sheet" not in bullet


def test_library_manual_kind_list_matches_kind_options():
    from realmspinner.studio.modes.library.ui.panes import library

    text = _flat(_chapter("33-library-and-jobs"))
    bullet = text.split("- **Kind**: ", 1)[1].split(". Note that", 1)[0]
    for _key, label in library.KIND_OPTIONS:
        assert label in bullet, label
    assert " sheets," not in bullet.replace("sprite sheets,", "").replace("tile sheets,", "")


def test_library_manual_job_directory_list_names_sprites_not_sheets():
    from realmspinner.kernels.rig import store

    assert store.SPRITE_DIR_NAME == "sprites"
    for name in ("33-library-and-jobs", "41-configuration"):
        text = _chapter(name)
        assert "`sheets/`" not in text and "sheets/<sheet_id>" not in text, name
    assert "`sprites/`" in _chapter("33-library-and-jobs")
    assert "sprites/<draft_id>.json" in _chapter("41-configuration")


def test_clean_library_chapters_do_not_name_a_pose_library_the_app_no_longer_has():
    for name in ("03-finding-your-work", "33-library-and-jobs", "42-app-settings"):
        assert "pose library" not in _flat(_chapter(name)), name


# --- service-kinds-05 / service-queue-03 / service-queue-08 -------------------------------


def test_every_service_module_the_architecture_chapter_names_exists():
    text = _flat(_chapter("44-architecture"))
    sentence = text.split("The split into modules is by subject — ", 1)[1].split(
        " — and folding", 1
    )[0]
    # Re-split on the first sentence's own stop so only the module list is read.
    names = re.findall(r"`(\w+)`", sentence)
    assert names
    for name in names:
        assert (ROOT / "src" / "realmspinner" / "service" / f"{name}.py").exists(), name


def test_the_manual_does_not_claim_a_migration_entry_is_one_transaction():
    text = _flat(_chapter("44-architecture"))
    assert "each applied in one transaction" not in text
    assert "autocommit" in text and "ADD COLUMN" in text


def test_the_configuration_chapter_says_the_image_checkpoint_is_released_at_the_end_of_each_stage():
    text = _chapter("41-configuration")
    coexist = _paragraph(text, "**Coexist.**")
    assert "unloads its checkpoint when it ends" in coexist
    assert "both are evicted after the idle timeout" not in coexist
    assert "a text job does not pay to restart the engine" in coexist  # still true of the engine


# --- docs-F1-01 / docs-F1-07 / the nav-key reservation -----------------------------------


def test_shortcut_chapter_inker_f_and_s_rows_say_what_a_selection_changes():
    from realmspinner.studio.modes.inker import ops as inker_ops

    selection_keys = {
        b.target: b.chord for b in inker_ops.BINDINGS if getattr(b, "context", "") == "Selection"
    }
    assert selection_keys["fill_selection"] == "F" and selection_keys["stroke_selection"] == "S"
    inker = _section(_chapter("35-shortcuts"), "Inker")
    rows = {
        line.split("|")[1].strip(): line for line in inker.splitlines() if line.startswith("| `")
    }
    assert "Fill selection" in rows["`F` or `Shift+L`"]
    assert "Stroke selection" in rows["`S` or `Shift+M`"]


def test_shortcut_chapter_lists_the_inspector_page_as_clay_escs_first_stage():
    clay = _section(_chapter("35-shortcuts"), "Clay")
    row = next(line for line in clay.splitlines() if line.startswith("| Esc |"))
    assert row.index("Inspector page") < row.index("the picks") < row.index("element mode")


# --- docs-F1-08 / config ranges -------------------------------------------------------------


def test_every_config_env_var_in_effective_table_is_documented_in_the_configuration_chapter():
    from realmspinner import config

    text = _chapter("41-configuration")
    table = text.split("| Variable | Default | Effect |", 1)[1]
    for _field, env in (*config.SETTINGS, *config.SWITCHES):
        assert f"`{env}`" in table or f"`{env}`" in text, env
    for env in (
        "REALMSPINNER_FAMILIAR_RUNTIME",
        "REALMSPINNER_FAMILIAR_MODELS",
        "REALMSPINNER_FAMILIAR_PORT",
        "REALMSPINNER_FAMILIAR_EMBED_PORT",
        "REALMSPINNER_FAMILIAR_IDLE",
    ):
        assert f"| `{env}` |" in table, env


def test_the_configuration_chapter_states_the_ranges_the_config_enforces():
    from realmspinner import config

    text = _flat(_chapter("41-configuration"))
    assert (config._ENV_PORT[0], config._ENV_PORT[1]) == (1, 65535)
    assert "A whole number from 1 to 65535" in text
    assert config._ENV_RETRIES_MAX == 10 and "From 0 to 10" in text
    assert "Must be a known template key" in text and "Must be a known base model key" in text
    assert "sprite-sheet render" not in text  # the sheet timeout bounds the frame-set renders now


# --- shell-review-settings: review chapter ---------------------------------------------------


def test_review_manual_names_the_label_pass_titles_verbatim():
    from realmspinner.studio import main

    text = _chapter("34-review")
    for title in main._LABEL_TITLES.values():
        assert f"**{title}**" in text, title


# --- docs-A-B-08 and friends ---------------------------------------------------------------------


def test_manual_no_longer_promises_character_sheets():
    # Chapter 33's retired-rows note and chapter 21's one parenthetical are the two places the
    # words are meant to survive: a reader holding an old row needs to be told what it is.
    for name in (
        "01-before-you-begin",
        "03-finding-your-work",
        "04-judging-what-you-made",
        "12-putting-it-in-a-game",
        "20-overview",
        "22-generating-references",
        "34-review",
        "35-shortcuts",
        "40-installation",
        "41-configuration",
        "42-app-settings",
        "44-architecture",
        "45-pipelines",
        "46-extending",
    ):
        text = _flat(_chapter(name)).lower()
        assert "character sheet" not in text and "character-sheet" not in text, name
    assert "character sheet" in _flat(_chapter("33-library-and-jobs"))


def test_the_first_chapter_names_no_removed_poser_or_character_sheet_feature():
    text = _flat(_chapter("01-before-you-begin"))
    assert "posing and" not in text
    assert "rigging, posing" not in text
    assert "has not yet been seen on a computer" not in text
    assert "**Continue**" in text  # the primary button's third state (shell-review-settings-07)


def test_no_install_chapter_names_a_removed_poser_or_character_sheet_feature():
    from realmspinner import packs

    text = _flat(_chapter("40-installation"))
    assert "Rigging, posing and sprite sheets" not in text
    assert "posing" not in text.split("| `rig` |", 1)[1].split("|", 3)[2]
    assert "sheets/<sheet_id>" not in _chapter("41-configuration")
    assert "skinning" in (packs.find("rig").summary)


# --- docs-F2-02 / familiar-host-02 / agents-host-07 ---


def test_overview_character_plan_sentence_names_only_what_the_plan_card_draws():
    text = _flat(_chapter("20-overview"))
    assert "whichever movements" not in text and "direction count or name" not in text
    assert "it shows a plan: the species, its look (the theme) and a time estimate" in text


def test_manual_20_names_the_settings_section_the_agent_switch_is_in():
    text = _flat(_chapter("20-overview"))
    assert "Settings → Agent" not in text
    assert "Settings → Advanced" in text.split("Agent connected", 1)[1][:200]
