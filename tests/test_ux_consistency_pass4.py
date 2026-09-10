"""The fourth consistency pass (2026-09-08): labels above their controls.

The user's rule was blunt -- "labels need to be above inputs/selections/drop
downs when feasible" -- and an inventory found fourteen panes still drawing a
control's name *beside* it (imgui's default) directly under a correctly
labelled ``field_label``/``labeled_combo`` for the same section, so the two
idioms sat one line apart. Each fix moves the name onto its own small-caps
line through :func:`widgets.field_label` and hides the control's own copy by
folding it behind ``##`` -- the house rule this pass leans on throughout:
a previously visible ``"Foo"`` becomes ``"##Foo"`` (or, where the control
already carried a hidden id suffix, ``"Foo##suffix"`` becomes
``"##Foo##suffix"``), never a new id, so nothing keyed on the old one moves.

This is ``test_ux_consistency_pass3.py``'s ``_LABELLED_RAW`` idea (a source
scan for a raw ``controls.*`` call whose first argument is a *visible* label,
i.e. does not start with ``##``), extended from the one file pass 3 covered to
the fourteen this pass does, each with its own explicit allow-list of the
controls a fix deliberately left beside their box -- a sentence-reading
checkbox, a coordinate-row's own short letter, or a repeated list row with no
spare line. An allow-list entry that stops matching the source (because the
line moved or was itself fixed) fails the second half of the test, so the list
cannot go stale silently.
"""

from __future__ import annotations

import re
from pathlib import Path

# Only ``controls.*`` calls: ``form_ui.combo``/``form_ui.field`` pass a field
# *key* as their first argument, not a label (Form's own idiom, pass 2's
# territory), and would otherwise false-positive here.
_RAW_LABELLED = re.compile(
    r"controls\.(slider_float|slider_int|checkbox|input_int|input_float"
    r"|input_float2|input_float3|drag_int|drag_float|color_edit4)"
    r"\(\s*\n?\s*\"(?!##)([^\"]+)\"",
)

#: Every deviation the inventory found, per file, and what replaced it.
#: ``None`` in place of a set means "every offender in this file was fixed";
#: kept as explicit files (not a glob) so a fifteenth pane silently added to
#: the ownership list is not assumed clean.
_OWNED_FILES = (
    "settings_2d.py",
    "settings_3d.py",
    "clay_props.py",
    "clay_header.py",
    "clay_menu.py",
    "plotter_canvas.py",
    "plotter_tools.py",
    "plotter_tileset.py",
    "plotter_tileset_editor.py",
    "poser_controls.py",
    "poser_clips.py",
    "sheet_panel.py",
    "packwright_sources.py",
    "library.py",
    "sirens_envelopes.py",
)

#: What each file is still allowed to draw with a visible label beside a
#: ``controls.*`` call, and why. Everything else the inventory named must have
#: moved its name onto a ``field_label`` line above.
_ALLOW: dict[str, set[str]] = {
    "settings_2d.py": {
        # Checkboxes whose label reads as a sentence ("this list keeps one
        # style", "this pass erases the seam", "start from this image") are
        # switches, not name+value fields -- the judgement call this pass's
        # own brief carves out. Untouched by the inventory; only the five
        # ``Strength``/``Until`` sliders beneath these sections were named.
        "Keep one style across the list",
        "Erase the seam",
        "Start from this image (img2img)",
    },
    "settings_3d.py": {
        "Rig when the mesh lands",  # sentence checkbox
    },
    "clay_props.py": {
        # A single combined imgui control (one label, three boxes inline) --
        # not the "several separate boxes" shape this pass's coordinate-row
        # rule is about, and not named by the inventory (only the generic
        # per-param widget and the material sliders were).
        "position##bt",
        "scale##bs",
    },
    "clay_header.py": set(),
    "clay_menu.py": set(),
    "plotter_canvas.py": {
        # The coordinate-row idiom itself: a field_label above the pair, a
        # short letter *beside* each box ("W##map-w"/"H##map-h" for the map
        # size, "W##tile-w"/"H##tile-h" for the tile size -- pre-existing and
        # the very precedent the Go-to popup's "X##goto-x"/"Y##goto-y" was
        # brought into line with).
        "X##goto-x",
        "Y##goto-y",
        "W##map-w",
        "H##map-h",
        "W##tile-w",
        "H##tile-h",
        "Infinite",  # sentence checkbox, own tooltip
    },
    "plotter_tools.py": {
        # A single combined imgui control (one label, several boxes inline),
        # not the "several separate boxes" shape this pass's coordinate-row
        # rule is about -- and not named by the inventory.
        "Parallax origin",
        "Skew",
        "Wrap around the edges",  # sentence checkbox
        # The tile-pixel-size coordinate row, matching plotter_canvas's own
        # W/H idiom -- pre-existing, not part of the "Hex side" fix.
        "W##tile-px-w",
        "H##tile-px-h",
    },
    "plotter_tileset.py": set(),
    "plotter_tileset_editor.py": {
        # A per-frame duration in a repeated animation-frame row (name is
        # the unit, "ms", not a field name) -- not named by the inventory,
        # which only flagged the two "Probability" copies.
        "ms",
        # The wang-colour list's own "Probability": unlike the per-tile copy
        # in _tiles_tab (fixed), this one is a cell in a repeated row --
        # name, swatch, probability, one row per colour -- where a label
        # line would double every row's height.
        "Probability",
    },
    "poser_controls.py": {
        "Move root",  # sentence checkbox
    },
    "poser_clips.py": {
        "Onion skin",  # a toggle's short name, not named by the inventory
        "Loops",  # sentence checkbox
    },
    "sheet_panel.py": {
        "Lock silhouettes",  # sentence checkbox
    },
    "packwright_sources.py": {
        "Drop duplicate tiles",  # sentence checkbox
        "match flipped / rotated",  # sentence checkbox, sub-option
        "Anchor",  # a toggle's short name, not named by the inventory
    },
    "library.py": set(),
    "sirens_envelopes.py": set(),
}


def _panes_root() -> Path:
    from warlock.studio import panes

    return Path(panes.__file__).parent


def test_every_owned_pane_puts_the_label_above_the_control():
    """The inventory's own claim, file by file: no ``controls.*`` call left in
    any owned pane draws a visible label beside itself unless that exact
    string is in the file's allow-list above.

    Fails against the pre-fix source for every file this pass touched --
    "Strength##ip", "Triangles", "base colour##bm", "grid (m)##snapt",
    "{param.label}##{op.name}-{param.name}", "Column##goto-x", "Hex side",
    "Probability" (the per-tile copy), "Rotate X", "Frames after this key",
    "Strength" (sheet_panel), "Keep the newest" all matched this regex before
    their fixes and match nothing after, because each fix's id starts with
    ``##``.
    """
    root = _panes_root()
    for name in _OWNED_FILES:
        source = (root / name).read_text(encoding="utf-8")
        found = {m.group(2) for m in _RAW_LABELLED.finditer(source)}
        allowed = _ALLOW.get(name, set())
        unexpected = found - allowed
        assert not unexpected, (name, sorted(unexpected))
        # The allow-list is not a place to park a claim once and forget it:
        # an entry that stops matching (the line moved, or got fixed after
        # all) must be dropped from the list in the same change.
        stale = allowed - found
        assert not stale, (name, sorted(stale))


def test_settings_2d_sub_fields_each_get_their_own_name_line():
    """The five ``Strength``/``Until`` sliders, named: each is a sub-field of
    the section's own field_label above it (appearance/start image/structure/
    Style LoRA), so each gets one small-caps name line of its own rather than
    an indented half-line -- the same shape ``field_label`` already draws for
    every full field in this pane."""
    from warlock.studio.panes import settings_2d

    source = Path(settings_2d.__file__).read_text(encoding="utf-8")
    for ident in ("##Strength##ip", "##Strength##init", "##Strength##cn", "##Until##cn"):
        assert f'"{ident}"' in source, ident
    assert 'controls.slider_float("##Strength", form["lora_weight"]' in source
    # Each hidden slider has a field_label immediately above it in the
    # source, in the order the four appear -- not just somewhere in the
    # function, and not the same one every time.
    cursor = 0
    for label, hidden in (
        ("Strength", "##Strength##ip"),
        ("Strength", "##Strength##init"),
        ("Strength", "##Strength##cn"),
        ("Until", "##Until##cn"),
    ):
        marker = f'widgets.field_label("{label}")'
        pos = source.find(marker, cursor)
        assert pos != -1, label
        window = source[pos : pos + 700]
        assert hidden in window, (label, hidden)
        cursor = pos + len(marker)


def test_poser_xyz_triples_are_one_label_above_short_letters():
    """Rotate X/Y/Z and Offset X/Y/Z are coordinate rows, not three fields --
    one ``field_label`` above, ``X``/``Y``/``Z`` beside each box, the
    ``plotter_canvas._setup_body`` precedent this pass's brief pointed at."""
    from warlock.studio.panes import poser_controls

    source = Path(poser_controls.__file__).read_text(encoding="utf-8")
    assert 'widgets.field_label("Rotate")' in source
    assert 'widgets.field_label("Offset")' in source
    # The old three-spelled-out-labels tuple is gone as code (it survives
    # only in this pass's own explanatory comments, which name the old form
    # on purpose).
    assert 'enumerate(("Rotate X", "Rotate Y", "Rotate Z"))' not in source
    assert 'enumerate(("Offset X", "Offset Y", "Offset Z"))' not in source
    assert 'enumerate(("X", "Y", "Z"))' in source
    # Ids kept stable: the load-bearing suffix survives under the new letters.
    assert '##poser-joint-rot-{axis}' in source
    assert '##poser-root-offset-{axis}' in source


def test_plotter_goto_popup_matches_its_own_files_precedent():
    """The Go-to popup used "Column"/"Row" beside two boxes while this same
    file's ``_setup_body`` -- three lines away in the same module -- already
    drew a coordinate row as one field_label above short letters. Now they
    agree."""
    from warlock.studio.panes import plotter_canvas

    source = Path(plotter_canvas.__file__).read_text(encoding="utf-8")
    assert '"Column##goto-x"' not in source
    assert '"Row##goto-y"' not in source
    assert '"X##goto-x"' in source
    assert '"Y##goto-y"' in source
    goto_start = source.index("def goto_popup")
    goto_end = source.index("\ndef ", goto_start + 1)
    goto_body = source[goto_start:goto_end]
    assert 'widgets.field_label("Coordinate")' in goto_body


def test_clay_props_generator_params_are_each_named():
    """The generic ``_widget`` loop used to draw one field_label for the whole
    generator and none per param, so every checkbox/input in it read as a bare
    box. Each param now gets its own name line, and ``_widget``'s id keeps the
    old ``##gen{key}`` suffix that undo/patch call sites are keyed on."""
    from warlock.studio.panes import clay_props

    source = Path(clay_props.__file__).read_text(encoding="utf-8")
    assert 'widgets.field_label(key.replace("_", " "))' in source
    assert 'label = f"##{key.replace(\'_\', \' \')}##gen{key}"' in source
    for hidden in ("##base colour##bm", "##metallic##bm", "##roughness##bm"):
        assert f'"{hidden}"' in source, hidden


def _clay_props_scratch_reverted() -> str:
    """The generator-loop and material-slot fixes, undone in memory.

    Used only to prove the regression test above fails against the code as it
    stood before this pass -- never written to disk, and git is never touched
    to produce it (the working tree has four other agents editing it right
    now).
    """
    from warlock.studio.panes import clay_props

    source = Path(clay_props.__file__).read_text(encoding="utf-8")
    source = source.replace(
        '        widgets.field_label(key.replace("_", " "))\n', ""
    )
    source = source.replace(
        "label = f\"##{key.replace('_', ' ')}##gen{key}\"",
        "label = f\"{key.replace('_', ' ')}##gen{key}\"",
    )
    for label in ("base colour", "metallic", "roughness"):
        source = source.replace(f'widgets.field_label("{label}")\n', "")
        source = source.replace(f'"##{label}##bm"', f'"{label}##bm"')
    return source


def test_the_generator_param_regression_test_fails_against_the_unfixed_code():
    """Proof, not assertion: feed the pre-fix source (reconstructed in memory,
    not via git) through the same scan the pass-4 test above uses and confirm
    it flags exactly the params this pass fixed."""
    reverted = _clay_props_scratch_reverted()
    found = {m.group(2) for m in _RAW_LABELLED.finditer(reverted)}
    assert {"base colour##bm", "metallic##bm", "roughness##bm"} <= found


def test_settings_3d_size_keeps_its_unit_and_gets_a_label():
    """"Size" moved to a field_label; the unit stays in the drag's own printf
    format (K96's reason for a drag over a slider), not duplicated into the
    label."""
    from warlock.studio.panes import settings_3d

    source = Path(settings_3d.__file__).read_text(encoding="utf-8")
    assert 'widgets.field_label("Size")' in source
    assert '"##Size"' in source
    assert 'fmt = "unset - keeps the reference\'s" if value <= 0.0 else "%.2f m"' in source
    assert 'widgets.field_label("Triangles")' in source
    assert '"##Triangles"' in source


def test_clay_header_popover_units_stay_in_the_label():
    """"grid (m)"/"angle (deg)"/"radius (m)" keep their units, moved above."""
    from warlock.studio.panes import clay_header

    source = Path(clay_header.__file__).read_text(encoding="utf-8")
    for label, hidden in (
        ("grid (m)", "##grid (m)##snapt"),
        ("angle (deg)", "##angle (deg)##snapr"),
        ("radius (m)", "##radius (m)##propr"),
    ):
        assert f'widgets.field_label("{label}")' in source
        assert f'"{hidden}"' in source


def test_clay_menu_op_params_are_each_labelled():
    """The generic op-param popup loop names each field above its box, one
    ``field_label`` per iteration -- so a five-param op reads as five named
    fields, not five bare boxes under the op's own title."""
    from warlock.studio.panes import clay_menu

    source = Path(clay_menu.__file__).read_text(encoding="utf-8")
    body = source[source.index("def params_popup") :]
    assert "widgets.field_label(param.label)" in body
    assert 'label = f"##{param.label}##{op.name}-{param.name}"' in body


def test_packwright_cell_pair_matches_inker_bridges_fixed_shape():
    """``_cell_pair`` drew "tile size" as muted text after both boxes on the
    same line; moved above, to agree with ``inker_bridge._pair``'s fix landing
    the same day (that file is owned by another agent -- not asserted here,
    only that this file's own half of the agreement is done)."""
    from warlock.studio.panes import packwright_sources

    source = Path(packwright_sources.__file__).read_text(encoding="utf-8")
    body = source[source.index("def _cell_pair") : source.index("def _tileset_popup")]
    assert 'widgets.field_label("tile size")' in body
    assert 'widgets.muted("tile size")' not in body


def test_plotter_tileset_probability_fields_are_each_labelled():
    """The two near-duplicate per-tile ``Probability`` fields (this file's own
    ``_tile_form``, and ``plotter_tileset_editor._tiles_tab``), fixed
    identically, named -- the file-level allow-list above cannot tell this
    fixed copy apart from the wang-colour list row's deliberately-left one
    (same literal string, different function), so this test scopes to the
    function each copy actually lives in."""
    from warlock.studio.panes import plotter_tileset, plotter_tileset_editor

    tileset_source = Path(plotter_tileset.__file__).read_text(encoding="utf-8")
    form_start = tileset_source.index("def _tile_form")
    form_end = tileset_source.index("\ndef ", form_start + 1)
    form_body = tileset_source[form_start:form_end]
    assert 'widgets.field_label("Probability")' in form_body
    assert '"##Probability"' in form_body

    editor_source = Path(plotter_tileset_editor.__file__).read_text(encoding="utf-8")
    tiles_start = editor_source.index("def _tiles_tab")
    tiles_end = editor_source.index("\ndef ", tiles_start + 1)
    tiles_body = editor_source[tiles_start:tiles_end]
    assert 'widgets.field_label("Probability")' in tiles_body
    assert '"##Probability"' in tiles_body

    # The wang-colour list row keeps the old, deliberately-left shape --
    # proof this test is not merely restating the file-wide allow-list.
    colours_start = editor_source.index("def _wang_colours")
    colours_end = editor_source.index("\ndef ", colours_start + 1)
    colours_body = editor_source[colours_start:colours_end]
    assert '"Probability"' in colours_body
    assert 'widgets.field_label("Probability")' not in colours_body


def test_library_prune_dialog_keeps_the_newest_field_labelled():
    from warlock.studio.panes import library

    source = Path(library.__file__).read_text(encoding="utf-8")
    assert 'widgets.field_label("Keep the newest")' in source
    assert '"##Keep the newest"' in source
