"""The third consistency pass (2026-09-05): one vocabulary per shared surface.

Where pass 2 found divergent *behaviour* (the wheel, the tab bar, the save
gate), this pass finds surfaces that say the same thing in more than one
register. Each test here is one of those stated as a claim.

Sectioned by item, because later waves of the same pass add to this file.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

# --- item 2: one empty-state system, with actions ---------------------------
#
# Two tables say what an empty screen says and there is no third:
#   * ``widgets.nothing_open``  -- no document open at all (the workspace
#     screen: a hint, a primary, ghosts, recents).
#   * ``overlay.PLACEHOLDERS`` + ``overlay.centred_empty`` -- a document *is*
#     open and its viewport has nothing in it.
# Packwright's preview drew a third: one muted sentence in the top-left.


def _pane_sources() -> dict[str, str]:
    from _panes import pane_files

    return {name: path.read_text(encoding="utf-8") for name, path in pane_files().items()}


def test_no_pane_answers_an_empty_viewport_with_a_muted_sentence():
    """Packwright's preview said "Add a sprite to see the atlas." in muted
    body text where the other nine viewports drew the icon-title-hint form."""
    from warlock.studio.panes import overlay, packwright_preview

    source = inspect.getsource(packwright_preview)
    assert "Add a sprite to see the atlas." not in source
    assert "overlay.centred_empty(" in source
    assert 'overlay.PLACEHOLDERS["packwright"]' in source

    # And no pane re-draws one of the table's own sentences in the muted
    # register: that is how the third spelling got in the first time.
    spellings = {text for entry in overlay.PLACEHOLDERS.values() for text in entry[1:]}
    for name, text in _pane_sources().items():
        for sentence in spellings:
            assert f'widgets.muted("{sentence}")' not in text, (name, sentence)


#: Hints whose first word is one of these are telling the reader to *do*
#: something, so the app has to offer the doing.
_IMPERATIVES = ("Describe", "Add", "Rig", "Start", "Choose", "Draw", "Write")

#: The documented exemption: a hint beginning "Pick ..." points at a list the
#: user works in another pane, and "Describe the music you want above" points
#: at a control already on screen above it. A button repeating a pointer is a
#: second way to do one thing, which is the divergence this pass closes.
_POINTERS = {"create/mesh", "create/rig", "create/export", "poser", "review", "troupe", "muse"}


def test_every_imperative_placeholder_offers_the_thing_it_asks_for():
    from warlock.studio.panes import overlay

    for key, (_icon, _title, hint) in overlay.PLACEHOLDERS.items():
        if key in _POINTERS:
            continue
        first = hint.split()[0].rstrip(",")
        if first not in _IMPERATIVES:
            continue
        assert key in overlay.ACTIONS, f"{key} tells the reader to {first.lower()}, with no button"


def test_every_pointer_hint_really_is_a_pointer():
    """The exemption list is not a place to park work: an exempt entry must
    actually be pointing somewhere else."""
    from warlock.studio.panes import overlay

    for key in _POINTERS:
        hint = overlay.PLACEHOLDERS[key][2]
        assert hint.startswith("Pick ") or "above" in hint or "on the left" in hint, (key, hint)


def test_the_placeholder_table_stays_data():
    """The action is resolved at draw time. A callable in the table would drag
    every mode module in behind an import of the sentences."""
    from warlock.studio.panes import overlay

    for key, entry in overlay.PLACEHOLDERS.items():
        assert len(entry) == 3, key
        assert all(isinstance(part, str) for part in entry), key


def test_centred_empty_forwards_its_action_to_the_one_empty_state():
    from warlock.studio import widgets
    from warlock.studio.panes import overlay

    assert "action" in inspect.signature(overlay.centred_empty).parameters
    assert "action=action" in inspect.getsource(overlay.centred_empty)
    assert "action" in inspect.signature(widgets.empty_state).parameters


def test_action_for_binds_the_ctx_and_is_none_where_there_is_nothing_to_do():
    from warlock.studio.panes import overlay

    ctx = SimpleNamespace(state=SimpleNamespace(focus_key={}, focus_moved=False))
    label, run = overlay.action_for(ctx, "create/reference")
    assert label == "Write a brief"
    run()
    assert ctx.state.focus_key["brief"] == "prompt"
    assert ctx.state.focus_moved is True
    assert overlay.action_for(ctx, "review") is None


def test_the_packwright_button_opens_the_picker_that_exists():
    """``ask_add_image`` does not exist; ``ask_add_sources`` is the picker."""
    from warlock.studio import packwright_mode
    from warlock.studio.panes import overlay

    assert callable(packwright_mode.ask_add_sources)
    assert "packwright_mode.ask_add_sources(ctx)" in inspect.getsource(overlay._packwright_add)


def test_the_clay_button_never_names_a_generator():
    """The registry is data (``clay_props``' rule), and the button that adds a
    primitive lives under the same rule."""
    from warlock.kernels.mesh import primitives as bp
    from warlock.studio.panes import overlay

    source = inspect.getsource(overlay._clay_box)
    for name in bp.GENERATORS:
        assert f'"{name}"' not in source, f"{name} is hardcoded in overlay"


# --- item 2: Clay's properties pane tells the truth about a multi-selection --


def _clay_props_ctx(monkeypatch, selection: set[str]):
    """``clay_props._body`` with everything but the empty-state stubbed out.

    Cheaper than a real imgui frame, and the claim is about *which sentence*
    is drawn rather than about the frame surviving -- the smoke suite owns
    that half.
    """
    from warlock.studio.modes.clay.ui.panes import props as clay_props

    drawn: list[tuple[str, str]] = []
    doc = SimpleNamespace(
        selection=selection,
        element_mode="object",
        by_uid=lambda uid: None,
    )
    tab = SimpleNamespace(doc=doc, saving=False)
    state = SimpleNamespace(active=tab)
    monkeypatch.setattr(clay_props.clay_mode, "ensure", lambda ctx: state)
    monkeypatch.setattr(clay_props.widgets, "section", lambda *a, **k: None)
    monkeypatch.setattr(clay_props.manual_render, "help_button", lambda *a, **k: None)
    monkeypatch.setattr(
        clay_props.widgets,
        "empty_state",
        lambda icon, title, hint="", **k: drawn.append((title, hint)),
    )
    clay_props._body(SimpleNamespace())
    return drawn


def test_clay_says_how_many_objects_are_selected_instead_of_nothing_selected(monkeypatch):
    """With sixteen objects lit up the pane said "Nothing selected", which the
    viewport plainly contradicts."""
    drawn = _clay_props_ctx(monkeypatch, {"a", "b"})
    assert drawn == [("2 objects selected", "Select one to edit it.")]


def test_clay_still_says_nothing_selected_when_nothing_is(monkeypatch):
    drawn = _clay_props_ctx(monkeypatch, set())
    assert drawn == [("Nothing selected", "Click an object in the viewport.")]


def test_the_multi_selection_refusal_itself_is_unchanged(monkeypatch):
    """Only the sentence changed: ``_selected`` still refuses to edit one of
    many, which is the whole reason the branch exists."""
    from warlock.studio.modes.clay.ui.panes import props as clay_props

    doc = SimpleNamespace(selection={"a", "b"}, by_uid=lambda uid: "an object")
    assert clay_props._selected(doc) is None
    one = SimpleNamespace(selection={"a"}, by_uid=lambda uid: "an object")
    assert clay_props._selected(one) == "an object"


@pytest.mark.parametrize("count", [3, 16])
def test_the_count_is_the_real_one(monkeypatch, count):
    drawn = _clay_props_ctx(monkeypatch, {str(index) for index in range(count)})
    assert drawn[0][0] == f"{count} objects selected"
    assert re.fullmatch(r"\d+ objects selected", drawn[0][0])


# --- item 6: Settings speaks in one register --------------------------------
#
# ``panes/app_settings.py`` was drawing three registers on one page: a
# ``forms.Form`` field (small caps through ``widgets.field_label``), a raw
# ``controls.*`` call carrying its own trailing sentence-case label, and a
# bare ``imgui.text`` used as a name column. The decision of 2026-09-05 is
# that ``field_label`` is the one field face, so the raw labelled controls
# move onto the form and the bare text moves onto a widgets wrapper.
#
# This is pass 2's ``test_a_form_field_is_labelled_in_the_one_field_face``
# (tests/test_ux_consistency_pass2.py) extended to the call sites: that test
# pins the face inside ``Form._label``, and these pin that Settings actually
# goes through it.


def _app_settings_source() -> str:
    return _pane_sources()["app_settings.py"]


#: Raw ``controls.*`` calls take a *label* as their first argument, and imgui
#: draws it beside the control. A quoted first argument that is not an
#: ``##``-hidden id is therefore a second field face on the page.
_LABELLED_RAW = re.compile(
    r"controls\.(slider_float|slider_int|checkbox|input_text|input_float|combo|drag_int)"
    r"\(\s*\n?\s*\"(?!##)([^\"]+)\"",
)


def test_settings_labels_no_field_outside_the_form():
    """``controls.slider_float("UI scale", ...)`` and
    ``controls.checkbox("Licensed for commercial use", ...)`` drew sentence
    case in the body face two rungs from a small-caps ``field_label``."""
    offenders = [match.group(0) for match in _LABELLED_RAW.finditer(_app_settings_source())]
    assert offenders == [], offenders


def test_no_sidebar_sentence_is_drawn_with_the_helper_that_cannot_wrap():
    """``muted`` does not wrap, and a sidebar is 300 dp wide.

    That pairing is the defect the refreshed screenshots found in Settings'
    Maintenance group -- four sentences cut mid-word by the child, no ellipsis,
    no scrollbar to reach the rest. It is not special to that pane: ``muted``
    is the *right* helper for a short status line ("0 jobs - 0 B", "Measuring
    the trash...") and the wrong one for a sentence explaining what a button
    does, and nothing separated the two except the author's memory.

    Sixty characters is the line: at the body face a 300 dp column holds
    roughly that, so a longer literal is one a sidebar cannot show. The scan is
    over string *literals* only -- an f-string's source is far longer than what
    it draws, and judging those by source length is how a short status line
    ("{n} jobs - {size}") gets flagged as prose.
    """
    import ast

    from warlock import studio

    root = Path(inspect.getfile(studio)).resolve().parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "muted"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and len(node.args[0].value) > 60
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], offenders


def test_the_maintenance_help_wraps_and_both_bulk_deletes_look_alike():
    """Two defects the refreshed screenshot corpus found in one block.

    Settings' body is capped at ``CONTENT_W`` (640 dp) so that a line of prose
    is a readable measure rather than the width of the monitor. ``widgets.muted``
    does not wrap -- deliberately, because most of its callers are short status
    lines -- so every explanatory sentence in the Maintenance group ran off that
    cap and was clipped mid-word by the child: "Changes nothi", "regenerated
    from t", "are ke", "is meas". ``muted_wrapped`` exists for exactly this and
    the four sentences now use it.

    And "Prune..." was a plain button sitting three lines above a red "Clean
    library...". Both delete assets off the disk and no undo reaches either, so
    painting one as an ordinary action was the Maintenance group telling the
    user the two are different kinds of thing. They are drawn alike now; they
    stay on separate rows, which is what that block's own comment was actually
    arguing for.
    """
    source = _app_settings_source()
    # The Maintenance group only. The figures above it -- "0 jobs - 0 B",
    # "Measuring the trash..." -- are short status lines and stay unwrapped,
    # which is the case ``muted`` exists for; every line below the heading is
    # a sentence explaining what a button does.
    body = source[source.index('widgets.section("Maintenance")') : source.index("# --- models")]
    assert "widgets.muted(" not in body, "a Maintenance sentence still cannot wrap"
    assert body.count("widgets.muted_wrapped(") >= 4
    assert 'destructive_button("Prune..."' in body
    assert 'destructive_button("Clean library..."' in body


def test_settings_ui_scale_and_the_licence_box_are_form_fields():
    """The two migrated call sites, named, so a revert is a failure rather
    than a silently absent assertion."""
    source = _app_settings_source()
    assert 'form_ui.combo(\n        "ui_scale",\n        "UI scale",' in source
    assert 'form_ui.switch(\n            "commercial", "Licensed for commercial use"' in source


def test_the_licence_box_is_a_switch_because_the_form_has_no_checkbox():
    """A Boolean in the field grid is submitted as ``##field``; an unlabelled
    imgui checkbox is ELEV_1 on ELEV_1 and vanishes when off. ``forms.Form``
    records that where it declines to grow a ``checkbox`` method, and this is
    the claim that the record still holds."""
    from warlock.studio import forms

    assert not hasattr(forms.Form, "checkbox")
    assert callable(forms.Form.switch) and callable(forms.Form.slider)


def test_settings_draws_no_bare_imgui_text_as_a_name_column():
    """The health list ran ``imgui.text(row.name)`` under a coloured glyph,
    beside a ``muted_wrapped`` detail -- body face in a pane whose every other
    line goes through ``widgets``. The wrapped *prose* at the two
    ``imgui.text_wrapped`` sites is not a field and stays."""
    source = _app_settings_source()
    assert "imgui.text(row.name)" not in source
    assert "widgets.muted(row.name)" in source
    # Only ``##``-hidden ids may reach raw ``imgui.text``-family label calls.
    bare = re.findall(r"^\s*imgui\.text\((?!_wrapped)", source, re.M)
    assert bare == [], bare


# --- Item 7: bound the modals -------------------------------------------
#
# At 1280x800 with UI scale 2.0 Plotter's "New map" ran off the bottom of the
# viewport: the popup is ``always_auto_resize`` with no constraints, so it grew
# past the screen, imgui centred the overflow, and Create and Cancel were
# unreachable with nothing to scroll (2026-09-05).


def _modal_sources() -> dict[str, str]:
    from warlock.studio import dialogs
    from warlock.studio.modes.create.ui.panes import settings_3d
    from warlock.studio.panes import plotter_canvas

    return {
        name: inspect.getsource(module)
        for name, module in (
            ("dialogs", dialogs),
            ("plotter_canvas", plotter_canvas),
            ("settings_3d", settings_3d),
        )
    }


def test_modal_max_height_is_a_fraction_of_the_viewport():
    from warlock.studio import widgets

    assert widgets.MODAL_MAX_FRACTION == 0.85
    assert widgets.modal_max_height(800.0) == 680.0
    assert widgets.modal_max_height(1369.0) == pytest.approx(1163.65)
    # Never zero or negative, whatever a degenerate viewport reports.
    assert widgets.modal_max_height(0.0) > 0.0
    assert widgets.modal_max_height(-100.0) > 0.0


def test_modal_body_height_reserves_the_action_row():
    from warlock.studio import widgets

    # 800 px viewport -> 680 for the modal, less a 40 px button row and 60 px
    # of title bar and padding.
    assert widgets.modal_body_height(800.0, 40.0, 60.0) == 580.0
    # A viewport too short for the reservation floors rather than going
    # negative: imgui reads a negative child height as "fill", which is exactly
    # the overflow this helper exists to prevent.
    assert widgets.modal_body_height(100.0, 200.0, 60.0) == widgets.MODAL_MIN_BODY


def test_every_auto_resize_modal_is_bounded():
    """The four ``always_auto_resize`` popups each ask for bounds first."""
    found = 0
    for name, source in _modal_sources().items():
        for match in re.finditer(r"begin_popup_modal\(", source):
            head = source[: match.start()]
            assert "widgets.modal_bounds(" in head[-600:], name
            found += 1
    assert found == 4, found


def test_every_bounded_modal_scrolls_its_body():
    for name, source in _modal_sources().items():
        assert "widgets.modal_body(" in source, name


def test_plotter_presets_do_not_all_share_one_row():
    """Five presets at ``grid_width(5)`` truncated every label."""
    from warlock.studio import plotter_setup
    from warlock.studio.panes import plotter_canvas

    assert len(plotter_setup.PRESETS) == 5
    assert len(plotter_setup.PRESETS) > plotter_canvas.PRESET_COLUMNS
    source = _modal_sources()["plotter_canvas"]
    assert "grid_width(len(plotter_setup.PRESETS))" not in source
    assert "grid_width(PRESET_COLUMNS)" in source
    # The stale comment named three presets.
    assert "The three presets" not in source


def test_no_modal_centres_only_on_the_appearing_frame():
    """Capping the height was not enough on its own. Each site centred once,
    on ``Cond_.appearing``, with a ``(0.5, 0.5)`` pivot -- applied to the size
    an auto-resize window has before it has measured anything, so the window
    grew downwards from a top edge chosen for a stub and "New map" still hung
    off the bottom at 1280x800, scale 2.0. ``modal_bounds`` centres every
    frame; no site may go back to doing it itself."""
    for name, source in _modal_sources().items():
        assert "set_next_window_pos(centre" not in source, name


def test_modal_bounds_centres_every_frame():
    from warlock.studio import widgets

    source = inspect.getsource(widgets.modal_bounds)
    assert "set_next_window_pos(" in source
    assert "Cond_.always" in source


# --- item 8: labels above inputs/selections/drop downs, when feasible -------
#
# The user's own wording. ``widgets.field_label`` plus ``labeled_combo`` /
# ``labeled_slider_int`` / ``labeled_slider_float`` / ``labeled_drag_int`` are
# the house idiom (``widgets.py`` lines 2920-3110); this item is nine Inker
# pane files an inventory found still drawing a raw ``controls.*`` call with
# its name beside the control, imgui's own placement, instead. It is
# ``_LABELLED_RAW`` (item 6, above) read onto a different set of files, with
# its own kind list and its own allow-list -- the two are independent tables
# because the exemptions differ: Settings had none, and this pass has the
# W/H-pair precedent and two fixed-height rows.

_INKER_LABEL_ABOVE_FILES = (
    "inker_flourish.py",
    "inker_bridge.py",
    "inker_timeline.py",
    "inker_generate.py",
    "inker_canvas.py",
    "inker_tiles.py",
    "inker_context.py",
    "inker_tools.py",
    "inker_menu.py",
)

#: A raw ``controls.*`` data-entry call whose first argument -- literal or an
#: f-string -- is captured whole (quotes and any ``f`` prefix included), so
#: ``_inker_field_is_hidden`` can tell a control that draws nothing beside
#: itself (the id starts ``##`` right after the quote) from one that still
#: does. ``checkbox`` and ``color_edit4`` are left out of the kind list on
#: purpose: a checkbox already reads its own name as the control (the "a
#: checkbox stays" exemption every file in this pass was held to), and no
#: colour swatch in these nine files carries a beside-label to begin with.
_INKER_LABELLED_RAW = re.compile(
    r'controls\.(slider_float|slider_int|input_int|input_float|combo|drag_int'
    r'|input_text|input_text_multiline|input_text_with_hint)'
    r'\(\s*\n?\s*(f?"[^"]*")',
)


def _inker_field_is_hidden(literal: str) -> bool:
    body = literal[1:] if literal[0] == "f" else literal
    return body[1:].startswith("##")


#: Every deliberate exception the pass found, as ``(filename, exact literal)``
#: -- the longer reason for each is the comment at its own call site.
_INKER_LABEL_ABOVE_ALLOWED = {
    # ``_wh_row``'s ``W``/``H`` pair: the house precedent
    # (``plotter_canvas.setup_popup``, which cites this function by name) is
    # one ``field_label`` above the pair with a short *visible* letter beside
    # each box, not a hidden field with nothing on the row to read at all.
    ("inker_bridge.py", 'f"{label}##{prefix}{tag}"'),
    # ``inker_canvas``'s New canvas popup and ``inker_tiles``'s Convert-to-
    # tilemap popup: the same ``W``/``H`` precedent, one caption per pair.
    ("inker_canvas.py", '"W##newcanvas"'),
    ("inker_canvas.py", '"H##newcanvas"'),
    ("inker_tiles.py", '"W"'),
    ("inker_tiles.py", '"H"'),
    # ``inker_canvas._transform_row``: a single pinned toolbar line -- its own
    # docstring measures its width as exactly one row's worth of five fields
    # plus a Link checkbox -- with no room for a stacked label under any of
    # them.
    ("inker_canvas.py", '"Angle"'),
    ("inker_canvas.py", '"X##inkscalex"'),
    ("inker_canvas.py", '"Y##inkscaley"'),
    ("inker_canvas.py", '"H##inkshearx"'),
    ("inker_canvas.py", '"V##inksheary"'),
    # ``inker_timeline._frame_trailing``'s duration box: the transport row's
    # own trailing measurement, on the row its docstring calls "the worst
    # same_line chain in the app" -- no room for a second text line.
    ("inker_timeline.py", '"ms"'),
}


def test_inker_fields_are_labelled_above_not_beside():
    """``labels need to be above inputs/selections/drop downs when
    feasible`` -- the user's own instruction for the 2026-09-08 pass.

    Every remaining raw ``controls.*`` call in the nine files the inventory
    named must either hide its own id (``field_label`` draws the name above
    it instead) or be one of the deliberate, reasoned exceptions in
    ``_INKER_LABEL_ABOVE_ALLOWED``. An id that is merely hidden and not also
    given a ``field_label`` line would still pass this scan -- it is the
    beside-label anti-pattern this test rules out, the same scope
    ``_LABELLED_RAW`` has for Settings.
    """
    sources = _pane_sources()
    offenders = []
    for name in _INKER_LABEL_ABOVE_FILES:
        source = sources[name]
        for match in _INKER_LABELLED_RAW.finditer(source):
            literal = match.group(2)
            if _inker_field_is_hidden(literal):
                continue
            if (name, literal) in _INKER_LABEL_ABOVE_ALLOWED:
                continue
            offenders.append((name, literal))
    assert offenders == []


def test_every_inker_label_above_exception_is_still_in_its_file():
    """The allow-list is not a place to park a stale entry: every string in
    it must still be the exact literal at some call site, or the exception it
    documents has drifted from the code it was written against."""
    sources = _pane_sources()
    for name, literal in _INKER_LABEL_ABOVE_ALLOWED:
        assert literal in sources[name], (name, literal)


def test_flourish_generic_param_widget_labels_above_its_field():
    """The one call the scan above cannot reach: ``_param_control`` and
    ``_asset_control`` build their hidden id into a variable
    (``control_id``) rather than passing a literal straight to
    ``controls.*``, so ``_INKER_LABELLED_RAW`` never sees the call at all --
    pinned by name instead.
    """
    from warlock.studio.modes.inker.ui.panes import flourish as inker_flourish

    param_source = inspect.getsource(inker_flourish._param_control)
    assert 'widgets.field_label(name, tip or None)' in param_source
    assert 'control_id = f"##{name}##fl-p-{name}"' in param_source
    # The label is drawn *before* the control it labels, or a caller reading
    # top to bottom would see the control before its name.
    assert param_source.index("field_label(") < param_source.index("controls.slider_float(")

    asset_source = inspect.getsource(inker_flourish._asset_control)
    assert 'widgets.field_label(name)' in asset_source
    assert 'f"##{name}##fl-p-{name}"' in asset_source
    assert asset_source.index("field_label(") < asset_source.index("controls.combo(")


def test_the_wh_precedent_still_reads_field_label_above_wh_below():
    """``_wh_row``, the New canvas popup and the Convert-to-tilemap popup all
    draw one ``field_label`` before their ``W``/``H`` pair -- the shape
    ``plotter_canvas.setup_popup`` cites ``inker_canvas`` as the precedent
    for. A caption that moved *after* the pair, or vanished, would still pass
    the id-hiding scan above (``W``/``H`` were never hidden) so this pins the
    caption's position directly instead.
    """
    sources = _pane_sources()
    bridge = sources["inker_bridge.py"]
    wh_row = bridge[bridge.index("def _wh_row(") : bridge.index("def _scale_dialog(")]
    assert wh_row.index("widgets.field_label(caption)") < wh_row.index('controls.input_int(')

    canvas = sources["inker_canvas.py"]
    new_canvas = canvas[canvas.index('"New canvas")') : canvas.index('"W##newcanvas"')]
    assert 'widgets.field_label("Canvas size, in pixels")' in new_canvas

    tiles = sources["inker_tiles.py"]
    tile_popup = tiles[tiles.index("def _tile_size_popup(") : tiles.index('"W", int(tile_w)')]
    assert 'widgets.field_label("Tile size, in pixels")' in tile_popup
