"""The 2026-10-03 audit's Low docs findings docs-74 .. docs-91 (fixer docs4): manual
wording that drifted from the control, key or module it names, and two comments that
described behaviour the neighbouring code no longer has."""

import ast
import inspect
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "docs" / "manual"


def _text(name: str) -> str:
    return (MANUAL / name).read_text(encoding="utf-8")


def test_manual_03_sort_names_match_state_SORTS():
    from realmspinner.studio.state import SORTS

    text = _text("03-finding-your-work.md")
    labels = [label for _key, label in SORTS]
    flat = " ".join(text.split())
    assert "Sorting** offers " + ", ".join(labels[:-1]) + ", and " + labels[-1] in flat
    assert "newest, name, kind, duration" not in flat


def test_manual_10_trim_paragraph_says_grid_ignores_it():
    text = " ".join(_text("10-packing-an-atlas.md").split())
    assert "**Trim transparent edges**" in text
    assert "a grid pack never trims" in text
    from realmspinner.studio.modes.packwright.ui.panes import settings

    assert "Trim transparent edges" in inspect.getsource(settings)


def test_manual_09_object_letters_cover_every_object_tool():
    from realmspinner.studio.modes.plotter.state import OBJECT_TOOLS

    text = _text("09-building-a-map.md")
    sentence = text[text.index("On an object layer `R`") :].split("\n\n")[0]
    for tool, label, letter in OBJECT_TOOLS:
        if tool == "object":  # Select objects: S, the pointer, is the Objects tool row above
            continue
        shape = label.replace("Insert ", "")
        assert f"`{letter}`" in sentence, f"{label}: letter {letter} missing"
        assert shape in sentence.replace("\r\n", " "), f"{label}: shape {shape} missing"


def test_manual_46_names_only_modules_that_exist():
    text = _text("46-extending.md")
    # tests/modes/clay/test_agent_clay.py is a real file; the bare module is not.
    stripped = text.replace("test_agent_clay", "")
    assert "agent_clay" not in stripped
    agent = ROOT / "src" / "realmspinner" / "studio" / "modes" / "clay" / "agent"
    for rel in ("studio/modes/clay/agent/dispatch.py", "studio/modes/clay/agent/validate.py"):
        assert rel in text
        assert (ROOT / "src" / "realmspinner" / rel).is_file()
    assert re.search(r"^def fail\(", (agent / "validate.py").read_text(encoding="utf-8"), re.M)
    assert re.search(r"^def call\(", (agent / "dispatch.py").read_text(encoding="utf-8"), re.M)


def test_manual_01_and_40_openings_do_not_overclaim_offline():
    for name in ("01-before-you-begin.md", "40-installation.md"):
        opening = _text(name)[:1200]
        assert "never touches the network again" not in opening, name
        assert "never goes online on its own" in " ".join(opening.split()), name


def test_manual_28_tool_letters_and_menu_rows_match_the_bindings():
    from realmspinner.studio.modes.inker import ops
    from realmspinner.studio.modes.inker.state import TOOLS

    letters = {key: letter for key, _label, letter in TOOLS}
    text = " ".join(_text("28-inker.md").split())
    assert f"**polyline** (`{letters['polyline']}`)" in text
    assert f"**polygon** (`{letters['polygon']}`)" in text
    assert "**Edit > Fill** " not in text and "**Edit > Stroke** " not in text
    assert "**Edit > Fill selection**" in text
    assert "**Edit > Stroke selection...**" in text
    src = inspect.getsource(ops)
    assert '"Fill selection"' in src and '"Stroke selection..."' in src


def test_manual_26_clip_counts_match_the_shipped_libraries():
    import json

    clips = ROOT / "src" / "realmspinner" / "templates" / "clips"
    libs = sorted(p.stem for p in clips.glob("*.json"))
    counts = set()
    for n in libs:
        data = json.loads((clips / f"{n}.json").read_text(encoding="utf-8"))
        counts.add(len(data.get("clips", data)))
    assert counts == {10} and len(libs) == 4
    text = " ".join(_text("26-poser.md").split())
    assert "every skeleton ships ten" not in text
    assert "each of the four skeletons with a clip library ships ten" in text
    assert "each carries all five movements" not in text
    assert "each ships all ten clips" in text


def test_manual_24_lists_every_viewport_toolbar_control():
    from realmspinner.studio.panes import overlay

    text = " ".join(_text("24-the-3d-viewport.md").split())
    assert f"**Tiled {overlay.TILE_REPEAT}x{overlay.TILE_REPEAT}**" in text
    assert "**Open in Inker**" in text
    source = inspect.getsource(overlay.toolbar)
    assert "Open in Inker" in source and "Tiled" in source


def test_manual_32_find_box_threshold_matches_list_filter():
    from realmspinner.studio import widgets

    minimum = inspect.signature(widgets.list_filter).parameters["minimum"].default
    assert minimum == 8
    text = " ".join(_text("32-plotter.md").split())
    assert "Past eight of them a **Find** box" not in text
    assert "From eight of them a **Find** box" in text


def test_sprite_panel_delete_comment_matches_sheet_panel_behaviour():
    from realmspinner.studio.panes import sheet_panel, sprite_panel

    assert "dialogs.ask_delete" in inspect.getsource(sheet_panel._ask_delete)
    source = inspect.getsource(sprite_panel)
    assert "exactly as deleting a rendered sheet has none" not in source


def test_settings_2d_negative_has_no_unreachable_inert_branch():
    from realmspinner.studio.modes.create.ui.panes import settings_2d

    func = ast.parse(inspect.getsource(settings_2d._negative)).body[0]
    names = {n.id for n in ast.walk(func) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(func) if isinstance(n, ast.Attribute)}
    assert "negative_prompt_note" not in attrs and "begin_disabled" not in attrs
    assert "inert" not in names
    # The gate that makes the branch unreachable is the one call site's own.
    assert "negative_supported" in inspect.getsource(settings_2d)
    text = " ".join(_text("22-generating-references.md").split())
    assert "The box is hidden, not greyed" in text
