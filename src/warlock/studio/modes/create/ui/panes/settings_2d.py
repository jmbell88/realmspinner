"""The 2D pane: drawing, and the orchestration of a press.

This pane owns the prompt and every field that reaches the text encoder; the
3D pane owns nothing that does. Since the 2026-08-17 taxonomy retirement the
form is flat -- no folds, no guidance groups -- and every section draws as a
full-width tinted block, matching Plotter's tools pane: the block scope is
opened *inside* the ``2d-form`` child so the fills land on the child's own
draw list rather than under its opaque background.

**What a recipe means** -- the plan, the validation, the kwargs, the option
lists, the notes explaining a disabled control -- lives in
``modes/create/engine/recipe.py`` now (2026-09-18 restructure, P5): this
module is what draws, and what a press does once it is accepted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from ...... import generation, vectors
from ...... import models as modelslib
from ......bench import findings as findings_lib
from ......pipelines import tileatlas as tileatlaslib
from ......service import findings as svc_findings
from ......service import jobs as svc_jobs
from ......service import palettes as svc_palettes
from ......service import sprites as svc_sprites
from ......service import tilesheets as svc_tilesheets
from ......service.errors import Invalid
from ......service.validation import MAX_PROMPT, MAX_UPLOAD_BYTES, random_seed
from ..... import controls, dialogs, focus, forms, theme, tokens, widgets
from ..... import problems as problem_types
from .....formvalues import coerce_form_value
from .....manual import render as manual_render
from .....tokens import sp
from .....widgets import field_options as _options
from ...engine import assets as create_assets
from ...engine import character as character_engine
from ...engine import recipe as create_recipe
from .. import workspace as generation_workspace
from . import settings_character

# This pane's key in the focus ring (UX.md Phase 3). The controls on the common
# path take a place in it: the ring exists so a first job can be composed and
# submitted without the mouse.
FOCUS_PANE = "2d"


FOCUS_PANE = "2d"

_submit_px = [96.0]

def draw(ctx: Any) -> None:
    state = ctx.state
    form = state.form_2d
    findings_doc = findings_lib.load(Path(ctx.svc.config.bench_dir) / "findings.json")
    # Compatibility for live callers from the pre-registry UI that still set
    # ``output`` directly. Settings loaded from disk have already migrated and
    # been synchronised, so only an in-memory non-reference override reaches
    # this bridge.
    if "asset_type" not in form:
        form["asset_type"] = create_assets.legacy_asset_type(form)
    create_assets.sync_legacy_fields(form)
    create_recipe.verify_reference_path(ctx, form)
    # Form.errors now places the rings and copy beneath the owning controls;
    # these are the routes it replaces and keeps wired by the same field keys:
    # field_error(ctx.state, "prompt")
    # field_error(ctx.state, "base_model")
    # field_error(ctx.state, "style_lora")
    # field_error(ctx.state, "count")
    # Before the form is built, because ``forms.Form`` snapshots the error map
    # at construction: a refusal recorded under a *recipe* field name has to be
    # re-filed under the control that answers to it, or the ring lands nowhere.
    if create_recipe.is_character(form):
        character_engine.mirror_errors(ctx)
    with forms.Form("create-2d", errors=ctx.state.field_errors) as form_ui:
        # The plan block is pinned and does not scroll (K92): the statement of
        # what a press will cost must not be at the bottom of a scrolled column
        # when the press itself is in the bar above.
        focus.pump(state, FOCUS_PANE)
        focus.begin(state, FOCUS_PANE)
        if imgui.begin_child("2d-form", (0, -sp(_submit_px[0]))):
            # The block scope opens *inside* the child: section() fills go to
            # the current window's draw list, and a scope opened outside would
            # paint onto the parent pane's list, where this child's opaque
            # PANEL background covers all but the 8dp left overhang. The
            # ``with`` closes before end_child -- an unbalanced splitter
            # corrupts the next frame (widgets.py, _BlockScope).
            with widgets.section_blocks():
                # **This column is "how"; the bar above is "what".** The type,
                # the prompt, the count and Generate moved to
                # ``create_brief``; what is left is the recipe, whatever the
                # chosen type needs, and the conditioning -- and it is flat,
                # because "Advanced controls" was one disclosure holding six
                # sections, which is a second navigation inside a sidebar.
                intent = create_assets.selected(form).intent
                if intent == "character":
                    # **The whole column, and none of the rest of this one.** A
                    # character runs no text encoder: there is no checkpoint to
                    # pick, no LoRA fitted to it, no negative branch to weight,
                    # no conditioning image and no prompt history worth reusing
                    # -- so every section in the ``else`` would be a control
                    # whose only outcome is that it does nothing. ``_plan_footer``
                    # stays shared, because it is about the *form* rather than
                    # about SDXL (Reset moved to ``create_brief`` with the rest
                    # of the bar, 2026-09-07 -- it is about the form too, but
                    # this pane no longer draws it). An ``if/else`` rather than
                    # an early return: the block scope and the child both have
                    # to close in order, and this file has already shipped the
                    # frame-corrupting version of that once.
                    settings_character.draw_block(ctx, form, form_ui)
                else:
                    widgets.section("Recipe")
                    manual_render.help_button(ctx, "settings-2d")
                    if intent == "tileset":
                        _locked_sheet_recipe(ctx, "Tile-set recipe", part="model")
                        _locked_sheet_recipe(
                            ctx, "Locked for coherent pixel tiles", part="lora"
                        )
                    else:
                        _model(ctx, form, findings_doc)
                        _lora(ctx, form, show_strength=False, findings_doc=findings_doc)
                    if intent == "sprite":
                        _locked_sheet_recipe(ctx, "Final sheet recipe", sprite=True)
                    _seed_row(ctx, form, form_ui)
                    if form.get("style_lora") and intent != "tileset":
                        widgets.section("Style strength")
                        _lora_strength(ctx, form, findings_doc)
                    if create_recipe.negative_supported(ctx, form):
                        widgets.section("Negative prompt / Avoid")
                        _negative(ctx, form)
                    _history(ctx, form)
                    # One contextual section, for the one type that needs it.
                    # Image and 3D Model draw none at all, which is the whole
                    # point: a control that cannot apply is not shown greyed, it
                    # is not shown.
                    if create_recipe.is_tile_arm(form):
                        widgets.section("Tileset")
                        manual_render.help_button(ctx, "settings-sheet")
                        _tile_layout(ctx, form, form_ui)
                        _tile_size(ctx, form, form_ui)
                        _target_cell(ctx, form, form_ui)
                        _pixel_look(ctx, form, form_ui, sprite=False)
                    elif form.get("output") == "sheet":
                        widgets.section("Sprite sheet")
                        manual_render.help_button(ctx, "settings-sheet")
                        _sprite_layout(ctx, form, form_ui)
                        _sprite_size(ctx, form, form_ui)
                        _target_cell(ctx, form, form_ui)
                        _pixel_look(ctx, form, form_ui, sprite=True)
                    # The one disclosure left, and it holds one thing.
                    # Collapsed by default because most runs attach no image at
                    # all; the tail says when one is attached, so a closed
                    # section never hides a setting that is doing something.
                    opened = controls.collapsing_header(
                        f"Conditioning{create_recipe.conditioning_tail(form)}##create"
                    )
                    if opened:
                        _references(ctx, form)
        imgui.end_child()
        top = imgui.get_cursor_pos_y()
        _plan_footer(ctx, form)
        height = imgui.get_cursor_pos_y() - top
        if height > 0:
            _submit_px[0] = height / max(tokens.SCALE, 0.01)

def _locked_sheet_recipe(
    ctx: Any, note: str, *, part: str = "both", sprite: bool = False
) -> None:
    """Say what the pinned sheet stage really loads; never draw fake pickers."""
    base_key = (
        svc_sprites.SPRITE_BASE_MODEL if sprite else svc_tilesheets.TILE_SHEET_BASE_MODEL
    )
    lora_key = modelslib.PIXEL_SHEET_LORA
    if part in ("model", "both"):
        if part == "both":
            widgets.field_label("Image model")
        imgui.text_wrapped(modelslib.BASE_MODELS[base_key].label)
    if part in ("lora", "both"):
        if part == "both":
            widgets.field_label("Style LoRA")
        imgui.text_wrapped(modelslib.STYLE_LORAS[lora_key].label)
    widgets.muted_wrapped(note)

def _tile_size(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """Only the editable dimension of a tileset asset type."""
    sizes = create_recipe.tile_sizes_for(form)
    changed, picked = form_ui.segmented_choice(
        "tile_size", "Tile size", str(form.get("tile_size", "32")),
        tuple((str(size), str(size)) for size in sizes),
        help_text="How many pixels across one tile is.",
        # Why the menu is shorter here than it is for the grid layout. Said
        # rather than left as an absence: 48 px is offered for a grid sheet and
        # is missing from this row, and an unexplained gap reads as a bug.
        helper=(
            f"A seamless material is drawn at {tileatlaslib.MATERIAL_PX} px and "
            f"reduced, so its tile size has to divide that exactly."
            if create_recipe.is_seamless(form)
            else ""
        ),
        compact=True,
    )
    if changed:
        form["tile_size"] = picked
        ctx.state.clear_field_error("tile_size")

TILE_MODE_CLEARED_KEY = "tile_mode_cleared"

def _tile_layout(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """What this sheet is a sheet *of*, and therefore which request it compiles.

    Three layouts, and the pane never holds a second opinion about any of their
    ceilings: the list of them, their labels, the material and cell limits and
    the two geometry menus all come from ``svc_tilesheets.tile_sheet_options``,
    which is the door that enforces them.
    """
    options = create_recipe.tile_options()
    before = create_recipe.tile_mode_of(form)
    changed, picked = form_ui.combo(
        "mode",
        "Layout",
        before,
        tuple((key, options["mode_labels"].get(key, key)) for key in options["modes"]),
        help_text=(
            "Materials and Terrain set draw each surface on its own, seamlessly, "
            "and lay the results out. Grid paints one frame through a guide and "
            "cuts it into sixty-four cells."
        ),
    )
    if changed and picked != before:
        form["tile_mode"] = picked
        # The geometry a seamless layout cannot keep, dropped with a sentence --
        # and the preview recomposed, because the words that will be sent are a
        # different set of words now.
        ctx.state.preview[TILE_MODE_CLEARED_KEY] = create_recipe.clear_for_layout(form)
        for field in _TILE_FIELDS:
            ctx.state.clear_field_error(field)
    for note in ctx.state.preview.get(TILE_MODE_CLEARED_KEY) or ():
        widgets.muted_wrapped(note)
    mode = create_recipe.tile_mode_of(form)
    if mode == svc_tilesheets.MODE_MATERIALS:
        _tile_materials(ctx, form, form_ui, options)
    elif mode == svc_tilesheets.MODE_TERRAIN:
        _tile_terrain(ctx, form, form_ui)
    else:
        _tile_grid(ctx, form, form_ui, options)

_TILE_FIELDS = (
    "mode",
    "prompt_items",
    "variants",
    "inner_terrain",
    "outer_terrain",
    "boundary",
    "tile_size",
    "projection",
)

def _tile_materials(
    ctx: Any, form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]
) -> None:
    """The list of surfaces, and how many draws of each.

    One generation per cell, which is why the count is said out loud beside the
    field rather than left to be discovered when the queue takes four minutes.
    """
    lines = create_recipe.material_lines(form)
    variants = create_recipe.safe_int(form.get("variants"), 1)
    cells = len(lines) * max(variants, 1)
    before = str(form.get("materials") or "")
    changed, text = form_ui.multiline_text(
        "prompt_items",
        "Materials",
        before,
        height=90,
        max_length=MAX_PROMPT * int(options["max_materials"]),
        help_text=(
            "One surface per line. Each line is generated on its own as a "
            "seamless tile, so this list is where the variety comes from."
        ),
        helper=(
            f"{len(lines)}/{options['max_materials']} materials - "
            f"{len(lines)} x {variants} = {cells} cells, "
            f"{options['max_cells']} at most"
        ),
    )
    if changed:
        form["materials"] = text
        ctx.state.clear_field_error("prompt_items")
    changed, picked = form_ui.segmented_choice(
        "variants",
        "Draws of each",
        str(variants),
        tuple((str(count), str(count)) for count in range(1, int(options["max_variants"]) + 1)),
        help_text=(
            "How many times each line is drawn. Every draw is its own full "
            "generation, on its own seed."
        ),
        compact=True,
    )
    if changed:
        form["variants"] = picked
        ctx.state.clear_field_error("variants")
    # Reachable at last: the service and the worker have carried ``style_lock``
    # since the materials mode landed, and no pane set it. The cost is stated
    # beside it because it is the one thing the checkbox changes about the
    # budget -- the first material becomes the IP-Adapter reference for every
    # one after it, which loads the encoder.
    changed, locked = controls.checkbox(
        "Keep one style across the list", bool(form.get("style_lock"))
    )
    widgets.help_marker(
        "The first material is generated on its own, then used as the appearance "
        "reference for every material after it, so the list reads as one artist's "
        "set. Loads the IP-Adapter (about 1.2 GB more on the card) and makes "
        "materials 2..N depend on the first one's roll."
    )
    if changed:
        form["style_lock"] = locked
    changed, erase = controls.checkbox("Erase the seam", bool(form.get("seam_erase")))
    widgets.help_marker(
        "After each material is drawn, roll it so the wrap seam runs through the "
        "middle and redraw a band around it in place. One more pass per material; "
        "use it when the wrap preview shows a join."
    )
    if changed:
        form["seam_erase"] = erase
    _tile_description_note()

def _tile_terrain(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """Two surfaces and the world they share.

    The forty-seven cases are composited from the pair by a computed coverage
    field, so neither field here describes an edge -- see :func:`_tile_terrain`'s
    boundary helper and ``pipelines.tilemask``.
    """
    for key, label, help_text in (
        (
            "inner_terrain",
            "Inside",
            "The surface the forty-seven cases are pictures of: the islands, "
            "coastlines and peninsulas a stroke paints.",
        ),
        (
            "outer_terrain",
            "Outside",
            "What surrounds it. Generated too, so it is described too.",
        ),
    ):
        changed, text = form_ui.text(
            key, label, str(form.get(key) or ""), help_text=help_text, max_length=MAX_PROMPT
        )
        if changed:
            form[key] = text
            ctx.state.clear_field_error(key)
    changed, text = form_ui.text(
        "boundary",
        "Shared setting",
        str(form.get("boundary") or ""),
        help_text=(
            "Words added to both surfaces so two separate generations come back "
            "sharing a world and a palette -- 'a temperate coastline'. Optional."
        ),
        # Named for the *place*, and the helper says why: the boundary itself is
        # a computed field, and a drawn edge inside a tile is the one defect this
        # layout exists to make impossible.
        helper=(
            "Not a description of the join. The join is computed, and a drawn "
            "edge would be cut across by it."
        ),
        max_length=MAX_PROMPT,
    )
    if changed:
        form["boundary"] = text
        ctx.state.clear_field_error("boundary")
    _tile_description_note()

def _tile_grid(
    ctx: Any, form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]
) -> None:
    """The original layout: one frame, one guide, sixty-four cells.

    Kept reachable and described honestly rather than hidden. It is the only
    layout that draws a 3/4 or an isometric tile, which is why the view lives
    here -- the other two accept one view and would draw a picker with nothing
    to pick.
    """
    before = create_recipe.view_of(form)
    changed, picked = form_ui.combo(
        "projection",
        "View",
        before,
        tuple((key, options["view_labels"].get(key, key)) for key in options["views"]),
        help_text="Where the camera is. Only this layout draws the other two.",
    )
    if changed and picked != before:
        form["projection"] = picked
        ctx.state.clear_field_error("projection")
    widgets.muted_wrapped(
        "One 1024 px frame is painted through a grid guide and cut into "
        f"{options['tiles']} cells. Every cell of the guide is identical, so the "
        "cells tend to come back as one scene cut up or as one tile repeated "
        "(measured 2026-08-18-tile-sheet-grid). Materials and Terrain "
        "set were built to replace it; it stays for 3/4 and isometric, and for "
        "rerunning a sheet made under it."
    )

def _tile_description_note() -> None:
    """What the Description above actually does in the two seamless layouts.

    It names the sheet and is recorded with it; the words that reach the model
    are the ones typed in this section. Said out loud because the alternative is
    a form with two prompt-shaped fields, one of which silently does nothing to
    the picture -- which is the failure a preview of the wrong template would
    also produce.
    """
    widgets.muted_wrapped(
        "The Description above names this sheet in the library. What each tile is "
        "painted from is what you type here."
    )

def _sprite_layout(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """What the sheet depicts: an action, and how many ways it is drawn.

    Above the model and not under Advanced, for the reason the tile arm's layout
    is: it decides what the sheet is a sheet *of*, and it is the choice that
    decides how long the press will take -- eight directions is eight
    generations, which is a fact a user is owed before pressing rather than
    after.
    """
    options = create_recipe.sprite_options()
    layout = str(form.get("sheet_layout") or "turnaround")
    _mode, action, directions = generation.sprite_from_layout(layout)
    current = create_recipe.sprite_action_key(layout)
    changed, picked = form_ui.combo(
        "sprite_action",
        "Action",
        current,
        create_recipe.sprite_action_options(options, current),
        help_text=(
            "What the character is doing. Only the actions this install has a "
            "pose guide for are offered -- the guide is what puts the limbs "
            "where they belong."
        ),
    )
    if changed:
        form["sheet_layout"] = create_recipe.sprite_layout_for(options, picked, directions)
        ctx.state.clear_field_error("sheet_type")
        layout = str(form["sheet_layout"])
        _mode, action, directions = generation.sprite_from_layout(layout)
    entry = create_recipe.sprite_action_entry(options, action)
    if entry is not None and layout not in generation.SPRITE_LEGACY_MODES:
        counts = [row["count"] for row in entry["directions"]]
        help_text = (
            "How many ways the character is drawn facing. One direction is "
            "one generation, so eight of them is eight."
        )
        # A segmented control with one segment is a control that cannot be
        # operated: it reads as a choice and answers every click with the
        # answer it already had. ``DIRECTION_COUNTS`` is (4, 8), but the
        # discovery behind ``counts`` keeps only the kinds with a guide file on
        # disk and this install ships ``*8.json`` alone -- so today every
        # action offers exactly one count. State it as a fact instead, and let
        # the selector come back on its own the day a 4-way guide ships.
        if len(counts) < 2:
            form_ui.readonly(
                "sprite_directions",
                "Directions",
                f"{counts[0]} ways",
                help_text=help_text,
            )
        else:
            changed, count = form_ui.segmented_choice(
                "sprite_directions",
                "Directions",
                str(directions),
                tuple((str(c), f"{c} ways") for c in counts),
                help_text=help_text,
                compact=True,
            )
            if changed:
                form["sheet_layout"] = create_recipe.sprite_layout_for(options, action, int(count))
    widgets.muted_wrapped(create_recipe.sprite_cost(create_recipe.sprite_plan(form)))

def _sprite_size(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """Only the editable dimension of a sprite asset type.

    **The ladder is gated on the action**, which is the difference between a
    refusal the user can act on and one they meet after pressing. One direction
    of an eight-frame walk is eight cells of ``PX_PER_ART_PIXEL`` times the
    logical size, and above 32px that band is past one SDXL frame -- so
    ``spritesynth.plan_sheet`` refuses it, naming both numbers, and both service
    doors re-raise that sentence. A picker still offering 48 and 64 there would
    be three sizes of which two are a refusal.
    """
    options = create_recipe.sprite_options()
    plan = create_recipe.sprite_plan(form)
    sizes = plan["sizes"] or tuple(options["logical_sizes"])
    current = str(plan["logical_size"])
    if str(form.get("cell_size", "")) != current:
        # Written back so the control shows what the submit will send. The clamp
        # itself is the recipe engine's own clamp, above -- a picker that was the only
        # thing holding the line would not hold it for a user who never opened
        # this section.
        form["cell_size"] = current
    changed, picked = form_ui.segmented_choice(
        "cell_size", "Cell size", current,
        tuple((str(size), str(size)) for size in sizes),
        help_text="How many pixels across one frame is.", compact=True,
    )
    if changed:
        form["cell_size"] = picked
    if len(sizes) < len(options["logical_sizes"]):
        # "An 8-frame", "A 4-frame". The frame counts a plan can carry are 4, 6,
        # 8 and 16, and 8 is the only one spoken with a leading vowel -- so this
        # is the whole rule rather than a general article function, which would
        # be a paragraph of English for three numbers that never change.
        article = "An" if plan["frames"] == 8 else "A"
        widgets.muted_wrapped(
            f"{article} {plan['frames']}-frame {plan['action']} is drawn one "
            f"whole direction at a time, and only {max(sizes)}px and below fit "
            "one generation."
        )

def _pixel_look(
    ctx: Any, form: dict[str, Any], form_ui: forms.Form, *, sprite: bool
) -> None:
    """An authored palette, dithering, and -- on the sprite arm only -- outlines.

    The three settings both sheet doors have taken since they started sharing
    ``service.pixelopts`` and that no pane offered, which made the whole
    capability unreachable: an authored ramp is the single highest-leverage art
    input in the program (``pipelines.pixelize``' own words) and it could not be
    named from the one form that composes a sheet.

    **No outline control on the tile arm, and that is not an omission.**
    ``pixelize._edge_mask`` pads with ``constant_values=False``, so on a cell
    that is opaque edge to edge -- which every tile is -- every border pixel has
    a "transparent" neighbour and ``inner`` returns the outer ring of *each*
    cell: a grid line around every tile rather than an outline of anything in
    one. ``create_tile_sheet`` refuses it by name; a form offering it would be a
    control whose only outcome is that refusal.

    **The dither box is not hidden behind the palette**, which is the opposite
    of what ``inspector`` does two panes over, and the difference is in the
    pipelines rather than in taste. ``asset2d`` applies dither *inside*
    ``map_palette`` and takes its own quantize branch otherwise, so there it
    genuinely does nothing without a palette -- and that file records
    ``bool(opts.dither and opts.palette)`` for exactly that reason. Both sheet
    paths route through ``map_palette`` either way:
    ``tilesheet.quantize_tiles`` branches on ``not entries and not dither``, and
    ``queue`` hands the sprite atlas to ``pixelize.pixelize_atlas`` with
    whatever ``resolve_palette`` returned. So on this form a dither with no
    palette dithers against the derived table, which is a real and different
    picture -- and hiding the box would make *that* the unreachable capability.
    Troupe's pane, the one that shipped these controls first, draws it
    unconditionally for the same reason.

    The palette list comes from the arm's own door and never from
    ``tile_sheet_options`` / ``create_recipe.sprite_options``: those are pure functions of
    module constants and this pane caches them for the life of the process, so
    a directory listing inside one would mean a palette dropped in five minutes
    ago never appears. ``inspector.palette_names`` is the one stat-per-frame
    guard over that listing and is shared rather than copied.
    """
    from .....panes import inspector

    door = svc_sprites.sprite_palettes if sprite else svc_tilesheets.tile_sheet_palettes
    installed = inspector.palette_names(ctx, door)
    chosen = str(form.get("palette") or "")
    if installed or chosen:
        # Only when there is something to pick. A combo whose one entry is
        # "derive one" is a picker with nothing in it, and palettes are opt-in
        # -- the honest rendering of "none installed" is no control, which is
        # ``palettes.available``'s own stated rule and ``inspector``'s. A form
        # that *names* one is the exception: see :func:`create_recipe.palette_options`.
        changed, picked = form_ui.combo(
            "palette",
            "Palette",
            chosen,
            create_recipe.palette_options(installed, chosen),
            help_text=(
                "Map every pixel to the nearest colour of a palette you "
                "authored, instead of to the colours this render happened to "
                "contain."
            ),
            # From the loader's own tuple, never restated. This line named
            # ``.pal`` and ``.txt`` until 2026-08-29 while ``palettes.SUFFIXES``
            # carried neither, so a user who dropped one in the folder was told
            # it would work and then watched it not appear -- no error, no row,
            # nothing to see. Both formats gained readers on 2026-08-30 and the
            # tuple now carries them; being derived is what kept the sentence
            # true through both the removal and the addition.
            helper=svc_palettes.SUFFIX_HELP,
        )
        if changed:
            form["palette"] = picked
            ctx.state.clear_field_error("palette")
    changed, dithered = form_ui.switch(
        "dither",
        "Dither",
        bool(form.get("dither")),
        help_text=(
            "Add an ordered 4x4 offset before each pixel picks its colour, so "
            "a gradient reads as a texture rather than as a band."
        ),
        helper=(
            "Against the chosen palette."
            if form.get("palette")
            else "Against the palette this sheet derives for itself."
        ),
    )
    if changed:
        form["dither"] = dithered
    if not sprite:
        return
    options = create_recipe.sprite_options()
    changed, picked = form_ui.segmented_choice(
        "outline",
        "Outline",
        str(form.get("outline") or options["defaults"]["outline"]),
        tuple((mode, create_recipe.OUTLINE_LABELS.get(mode, mode)) for mode in options["outlines"]),
        help_text=(
            "Darken the edge of each frame. Inside recolours the character's "
            "own edge pixels; Around grows the silhouette by one pixel, which "
            "a frame already touching its cell edge will have clipped."
        ),
        compact=True,
    )
    if changed:
        form["outline"] = picked
        ctx.state.clear_field_error("outline")

def _target_cell(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """Optional final reduction; blank means keep the high-resolution cell."""
    values = [("", "Keep working resolution")]
    values.extend((str(size), f"{size}px") for size in generation.TARGET_CELL_PRESETS)
    values.append(("custom", "Custom (8–256px)"))
    current = str(form.get("target_cell_px") or "")
    known = current if current in {x[0] for x in values} else "custom"
    selected = widgets.combo("##target_cell_px", known, values)
    if selected != "custom":
        form["target_cell_px"] = selected
        return
    raw = form.get("target_cell_px")
    try:
        number = int(raw)
    except (TypeError, ValueError):
        number = generation.TARGET_CELL_PRESETS[-1]
    changed, number = form_ui.number("target_cell_px_custom", "Custom cell size", number)
    if changed:
        form["target_cell_px"] = str(number)
    widgets.muted_wrapped("Blank preserves the 256px/512px working cell; reduction never upscales.")

def _hint(
    ctx: Any,
    form: dict[str, Any],
    param: str,
    value: Any,
    findings_doc: Any = create_recipe.LOAD_FINDINGS,
) -> None:
    """Draw the findings hint for the control just drawn, plus the offer to
    jump straight to what the evidence favours -- ``settings_3d._hint``'s
    shape, in the pane that owns the prompt these hints are scoped by.

    The 2026-09-07 review's ask, "findings become actionable at the control":
    the hint says what the *current* value scored, and until now that was
    where it stopped -- a user agreeing had to go find the winning value and
    dial it in by hand. ``_best_value_offer`` is the click.
    """
    hint = create_recipe.findings_hint(ctx, param, value, findings_doc)
    if hint is not None:
        widgets.hint_text(hint)
    _best_value_offer(ctx, form, param, value, findings_doc)

def _best_value_offer(
    ctx: Any,
    form: dict[str, Any],
    param: str,
    value: Any,
    findings_doc: Any = create_recipe.LOAD_FINDINGS,
) -> None:
    """"7/8 usable (47%+) · avg +2.9 · this subject" with a button, when the
    evidence favours a value other than the one already set.

    **Offered, never applied** -- ``settings_3d._size_suggestion``'s shape: a
    button that silently rewrote a slider the moment a sweep tipped the
    ranking would be indistinguishable from the app deciding the setting for
    the user, which every findings surface in this app deliberately refuses
    to do. Silent when the current value already leads
    (``bench.findings.best_value`` answers None then), because a button
    offering to set what is already set is not an offer, it is clutter.
    """
    doc = findings_doc
    if doc is create_recipe.LOAD_FINDINGS:
        doc = findings_lib.load(Path(ctx.svc.config.bench_dir) / "findings.json")
    found = findings_lib.best_value(
        doc,
        param,
        value,
        min_n=svc_findings.PRESET_MIN_N,
        prompt_hash=vectors.prompt_hash(ctx.state.form_2d.get("prompt")),
    )
    if found is None:
        return
    value_str, entry, scope = found
    widgets.muted(findings_lib.best_value_line(entry, scope))
    imgui.same_line()
    if controls.button(f"Use {value_str}##best-{param}"):
        form[param] = coerce_form_value(form[param], value_str)

def _reset(ctx: Any) -> None:
    """The 2D form back to first-launch defaults.

    A fresh ``default_form_2d`` rather than a field-by-field clear, so a field
    added later is reset by having been added rather than by somebody
    remembering this function -- and so the seed is *rerolled* rather than
    zeroed, which is what that default does and why it is a function.
    """
    from .....state import default_form_2d

    ctx.state.form_2d = default_form_2d()
    ctx.state.preview = {}
    ctx.toast("The image settings are back to their defaults.")

def _history(ctx: Any, form: dict[str, Any]) -> None:
    """Reuse a prompt from this session.

    No leading ``same_line``: it used to sit beside the prompt field, and with
    that field in the command bar the continuation landed against whatever
    happened to be drawn before it -- in practice the style-strength slider,
    which is how a button ends up orphaned in the middle of another section.
    """
    if not ctx.state.history:
        return
    if controls.button("Recent prompts..."):
        imgui.open_popup("prompt-history")
    if imgui.begin_popup("prompt-history"):
        widgets.popup_chrome(_imgui=imgui)
        for entry in ctx.state.history:
            label = entry if len(entry) <= 60 else entry[:57] + "..."
            if controls.menu_item(f"{label}##{hash(entry)}", "", False)[0]:
                form["prompt"] = entry
        imgui.end_popup()

def _references(ctx: Any, form: dict[str, Any]) -> None:
    """Conditioning: an image to steer appearance and/or structure.

    Every control below the picker is hidden until there is a reference, and
    the Structure group is hidden again unless the chosen base can run a
    ControlNet. That is this pane's existing rule -- the same one that hides
    the LoRA strength slider without a LoRA: a control with nothing to act on
    is a control that cannot do anything.
    """
    # The block is grouped so a dropped file can outline exactly what it landed
    # in (H70).
    imgui.begin_group()
    origin = imgui.get_cursor_screen_pos()
    try:
        _reference_body(ctx, form)
    finally:
        imgui.end_group()
        widgets.ring(
            origin,
            imgui.get_item_rect_max(),
            theme.ACCENT,
            widgets.drop_flash(ctx.state, "2d-ref"),
        )

def _reference_body(ctx: Any, form: dict[str, Any]) -> None:
    path = form["ref_path"]
    if path:
        imgui.text_wrapped(Path(path).name)
        if controls.button("Clear##ref"):
            form["ref_path"] = ""
            # The selections go with it: they cannot be submitted without an
            # image, and leaving them set would disable Generate with a
            # message about a picker the user just emptied.
            form["ip_adapter"] = ""
            form["control"] = ""
            form["init_image"] = False
            return
        imgui.same_line()
    busy = ctx.busy("ref-upload")
    if widgets.disabled_button(
        "Choose an image..." if not path else "Replace...",
        not busy,
        reason="A file picker is already open.",
    ):
        ctx.submit(
            "ref-upload", dialogs.open_file, "Choose a reference image", dialogs.IMAGE_FILTER
        )
    if not path:
        widgets.muted("...or drop an image on the window.")
        return

    # Field labels rather than nested section() calls: inside the block scope
    # a section would open a new block, and these are two halves of the one
    # References block.
    widgets.field_label("appearance")
    form["ip_adapter"] = widgets.combo(
        "##ip_adapter", form["ip_adapter"], _options(ctx, "ip_adapter")
    )
    if form["ip_adapter"]:
        # A sub-field of "appearance" above, not a section of its own: one
        # small-caps name line, id kept stable (2026-09-08 consistency pass,
        # "Strength##ip" visible -> "##Strength##ip" hidden).
        widgets.field_label("Strength")
        changed, value = controls.slider_float(
            "##Strength##ip", float(form["ip_scale"]), *_range(ctx, "ip_scale_range", 0.0, 1.5)
        )
        if changed:
            form["ip_scale"] = value
            ctx.state.clear_field_error("ip_scale")
        # The 2026-09-16 audit, finding create-panes-01: guidance.normalize's
        # `_number` refuses an out-of-range `ip_scale` by name, but nothing on
        # this pane rang the control -- a persisted form carrying a stale value
        # reached the queue door with the Conditioning section still collapsed
        # and no ring anywhere to say which slider was at fault.
        widgets.field_error(ctx.state, "ip_scale")
        _hint(ctx, form, "ip_scale", form["ip_scale"])

    widgets.field_label("start image")
    # The 2026-09-05 audit, finding create-04: this checkbox used to be drawn
    # unconditionally, so a non-SDXL base under Advanced showed it live, took
    # the tick, and only refused at the queue door (guidance.normalize). It is
    # disabled with a reason rather than hidden, matching ``_negative``'s
    # pattern for the same shape of problem: a value restored from a prior
    # SDXL run must stay visible, not vanish silently, until the user acts.
    inert = create_recipe.img2img_note(ctx, form)
    if inert is not None:
        imgui.begin_disabled()
    changed, on = controls.checkbox(
        "Start from this image (img2img)", bool(form.get("init_image"))
    )
    widgets.help_marker(
        "The reference is the picture the drawing starts from rather than only "
        "what it looks at. Low strength keeps its layout and repaints the surface; "
        "high strength keeps only the gist."
    )
    if changed:
        form["init_image"] = on
    if form.get("init_image"):
        widgets.field_label("Strength")
        changed, value = controls.slider_float(
            "##Strength##init",
            float(form.get("init_strength") or 0.45),
            *_range(ctx, "init_strength_range", 0.3, 0.65),
        )
        if changed:
            form["init_strength"] = value
            ctx.state.clear_field_error("init_strength")
        # The 2026-09-16 audit, finding create-panes-01: same gap as ip_scale
        # above -- guidance.normalize refuses a stale init_strength by name and
        # this slider never rang.
        widgets.field_error(ctx.state, "init_strength")
        _hint(ctx, form, "init_strength", float(form.get("init_strength") or 0.45))
    if inert is not None:
        imgui.end_disabled()
        widgets.muted_wrapped(inert)

    widgets.field_label("structure")
    note = create_recipe.recipe_structure_note(ctx, form) or create_recipe.structure_note(ctx, form)
    if note is not None:
        widgets.muted_wrapped(note)
        return
    form["control"] = widgets.combo("##control", form["control"], _options(ctx, "control"))
    if form["control"]:
        widgets.field_label("Strength")
        changed, value = controls.slider_float(
            "##Strength##cn",
            float(form["control_scale"]),
            *_range(ctx, "control_scale_range", 0.0, 2.0),
        )
        if changed:
            form["control_scale"] = value
            ctx.state.clear_field_error("control_scale")
        # The 2026-09-16 audit, finding create-panes-01: same gap as ip_scale
        # above -- guidance.normalize refuses a stale control_scale by name and
        # this slider never rang.
        widgets.field_error(ctx.state, "control_scale")
        _hint(ctx, form, "control_scale", form["control_scale"])
        widgets.field_label("Until")
        changed, value = controls.slider_float(
            "##Until##cn", float(form["control_end"]), *_range(ctx, "control_end_range", 0.0, 1.0)
        )
        if changed:
            form["control_end"] = value
            ctx.state.clear_field_error("control_end")
        # Same gap, control_end's own name.
        widgets.field_error(ctx.state, "control_end")
        _hint(ctx, form, "control_end", form["control_end"])
        widgets.help_marker(
            "How far into the drawing the structure keeps acting. Ending early "
            "lets the last steps add detail the reference never had; 1.0 holds "
            "the shape to the end and tends to look traced."
        )

def _range(ctx: Any, key: str, low: float, high: float) -> tuple[float, float]:
    """The bounds the service will actually enforce, so a slider can never
    produce a value the submit rejects."""
    bounds = ctx.guidance.get(key)
    if isinstance(bounds, list) and len(bounds) == 2:
        return (float(bounds[0]), float(bounds[1]))
    return (low, high)

CLEARED_KEY = "base_model_cleared"

def _model(ctx: Any, form: dict[str, Any], findings_doc: Any = create_recipe.LOAD_FINDINGS) -> None:
    auto = str(form.get("model_mode") or "auto") == "auto"
    before = "" if auto else str(form.get("base_model") or "")
    # The 2026-09-07 Create review, item 5.5.3: drawn as a bare ``##model``
    # widget with no visible name, unlike ``_locked_sheet_recipe``'s "Image
    # model" label at the same spot in the tileset/sprite arms' pinned
    # display. "Recipe" above names the section, not this control -- a
    # section heading is not a field label, and the two neighbouring
    # controls in it (Seed, Style LoRA) both have their own.
    widgets.field_label("Image model")
    picked = widgets.combo("##model", before, create_recipe.model_options(ctx))
    if picked != before:
        if picked:
            form["model_mode"] = "advanced"
            form["base_model"] = picked
            form["model_override"] = picked
            ctx.state.preview[CLEARED_KEY] = create_recipe.clear_unusable(ctx, form)
        else:
            form["model_mode"] = "auto"
            form["model_override"] = ""
        ctx.state.clear_field_error("base_model")
    # The refusal this most often carries is ``check_weights``' -- a model that
    # is selected and not downloaded, with the ``hf download`` line in it.
    widgets.field_error(ctx.state, "base_model")
    if form.get("model_mode") == "auto":
        # The 2026-09-13 audit, finding create-03: this called
        # ``generation.resolve_recipe`` directly instead of going through
        # the ``create_recipe.resolved_recipe`` memo, repeating
        # ``provenance._dir_fingerprint``'s ``rglob`` over every installed
        # checkpoint directory every frame -- the five sibling call sites
        # were moved onto the memo by the 2026-09-08 create-06 fix, but the
        # Model combo's own Automatic branch was missed.
        resolved = create_recipe.resolved_recipe(ctx, form)
        if resolved is None:
            widgets.muted_wrapped(
                "No compatible installed recipe is available. "
                "Install a model in Settings, or pick one above."
            )
        else:
            # What Automatic actually resolved to, every frame. A control whose
            # value is "Automatic" and says nothing else is a control that
            # refuses to tell you what it did.
            widgets.muted_wrapped(f"{resolved.recipe.label} - {resolved.base_model}")
            # What the resolved recipe trades, if it trades anything. This was
            # ``_recipe_note`` under the retired Fast/Quality combo: a tier
            # that is honestly worse and says so is a choice somebody can
            # make, and the silence it replaced had both tiers naming
            # ``sdxl_cfg`` while the control changed nothing at all.
            if resolved.recipe.note:
                widgets.muted_wrapped(resolved.recipe.note)
            if resolved.warning:
                widgets.wrapped(theme.WARN, resolved.warning)
        return
    for note in ctx.state.preview.get(CLEARED_KEY) or ():
        widgets.muted_wrapped(note)
    _hint(ctx, form, "base_model", form["base_model"], findings_doc)
    _licence_note(form["base_model"])

def _licence_note(key: str) -> None:
    """What this checkpoint's weights permit, under the picker that chose them.

    **The users of a game-asset generator will sell the output**, and this app
    shipped two checkpoints that restrict exactly that while saying so nowhere
    -- not here, not at download time, not in ``docs/MODELS.md``. Telling them
    nothing is the posture most likely to hurt somebody who trusted the tool.

    Drawn only when there is something to say: a permissive licence gets one
    muted line, and eleven identical "commercially permitted" rows would train
    the eye to skip the one row that is not.
    """
    spec = modelslib.BASE_MODELS.get(key or "")
    if spec is None or not spec.license:
        return
    if not spec.commercial:
        # ``wrapped`` in WARN rather than ``muted_wrapped``: this is the one
        # line on the pane that can cost the user money, and a muted sentence
        # among muted sentences is a sentence nobody reads.
        widgets.wrapped(
            theme.WARN,
            f"Licence: {spec.license}. Output from this model may NOT be used "
            f"commercially. {spec.license_note}".strip(),
        )
        return
    if spec.license_note:
        widgets.hint_text(f"Licence: {spec.license}. {spec.license_note}")
        return
    widgets.muted_wrapped(f"Licence: {spec.license} — commercial use permitted.")

def _lora(
    ctx: Any,
    form: dict[str, Any],
    *,
    show_strength: bool = True,
    findings_doc: Any = create_recipe.LOAD_FINDINGS,
) -> None:
    # The 2026-09-07 Create review, item 5.5.3: drawn as a bare ``##style_lora``
    # widget with no visible name, unlike ``_locked_sheet_recipe``'s "Style
    # LoRA" label at the same spot in the tileset/sprite arms' pinned display.
    # Drawn before the disabled block below, not inside it: the name of a
    # disabled control is exactly the thing a disabled control must not hide.
    widgets.field_label("Style LoRA")
    no_lora = create_recipe.lora_note(ctx, form)
    if no_lora is not None:
        # Disabled rather than hidden, this pane's stated rule: the form holds
        # a style the user picked under another base, and hiding the control
        # would make that selection vanish with no explanation of why the
        # submit is now refused.
        imgui.begin_disabled()
    was_lora = form["style_lora"]
    form["style_lora"] = widgets.combo(
        "##style_lora", form["style_lora"], create_recipe.lora_options(ctx, form)
    )
    widgets.field_error(ctx.state, "style_lora")
    if form["style_lora"] != was_lora:
        ctx.state.clear_field_error("style_lora")
    create_recipe.reseed_lora_weight(form, was_lora)
    _hint(ctx, form, "style_lora", form["style_lora"], findings_doc)
    if form["style_lora"] and show_strength:
        _lora_strength(ctx, form, findings_doc)
    if no_lora is not None:
        imgui.end_disabled()
        widgets.muted_wrapped(no_lora)
    else:
        # One sentence at a time: create_recipe.lora_note explains a control that cannot act,
        # create_recipe.lora_filter_note one acting on less than the whole list, and both
        # under a disabled combo would be one control saying two things.
        narrowed = create_recipe.lora_filter_note(ctx, form)
        if narrowed is not None:
            widgets.muted_wrapped(narrowed)

def _lora_strength(
    ctx: Any, form: dict[str, Any], findings_doc: Any = create_recipe.LOAD_FINDINGS
) -> None:
    """The advanced half of the style choice."""
    if not form.get("style_lora"):
        return
    # A sub-field of "Style LoRA" above (2026-09-08 consistency pass): the
    # combo already carries the field label, so this slider gets its own
    # name line rather than repeating the sentence-case label beside it.
    widgets.field_label("Strength")
    # The 2026-09-16 audit, finding create-panes-02: this hardcoded the
    # literal 0.0, 1.5 instead of going through ``_range``, the pattern every
    # sibling numeric control in this file follows -- two copies of one bound
    # with nothing to keep them equal if ``lora_weight_range`` ever changes.
    changed, value = controls.slider_float(
        "##Strength", form["lora_weight"], *_range(ctx, "lora_weight_range", 0.0, 1.5)
    )
    if changed:
        form["lora_weight"] = value
    default = create_recipe.lora_default_weight(form["style_lora"])
    widgets.muted_wrapped(f"tuned default: {default:g}")
    _hint(ctx, form, "lora_weight", form["lora_weight"], findings_doc)

def _negative(ctx: Any, form: dict[str, Any]) -> None:
    inert = create_recipe.negative_prompt_note(ctx, form)
    if inert is not None:
        # Disabled rather than hidden, and with the reason underneath: the
        # field holds text the user typed under another base, and hiding it
        # would make that text vanish without saying why.
        imgui.begin_disabled()
    # The section heading above is the label; imgui would draw a multiline's
    # own label to the *right* of a -1-wide field, clipped off the panel.
    form["negative_prompt"] = widgets.multiline(
        "##negative", form["negative_prompt"], 54, MAX_PROMPT
    )
    widgets.char_count(form["negative_prompt"], MAX_PROMPT)
    if inert is not None:
        imgui.end_disabled()
        widgets.muted_wrapped(inert)

def _seed_row(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """The seed, and the two controls that act on it.

    One function since the count went to the bar: this used to be the tail of
    ``_run_controls``, which existed to draw the count first and branch on the
    sheet arm that has none. With the count gone there is one path.
    """
    with focus.item(ctx.state, FOCUS_PANE, "seed"):
        changed, seed = form_ui.number("seed", "Seed", int(form["seed"]))
    if changed:
        form["seed"] = max(0, seed)
        ctx.state.clear_field_error("seed")
    # Rung, unlike this comment used to claim. ``service.validation.check_seed``
    # raises ``Invalid(..., field="seed")`` for a seed outside 0..MAX_SEED or
    # not an int, and ``create_job`` calls it as ``check_seed("seed", seed)`` --
    # a refusal this control can in fact be named in. What actually keeps it
    # unreachable through this widget is incidental, not structural: Dear
    # ImGui's plain InputInt stores into a C int32 whose range happens to
    # coincide with ``MAX_SEED = 2**31-1``. A seed can still arrive out of range
    # from a hand-edited settings.json -- Python ints on load are unbounded --
    # in which case the submit was refused with no control on this pane ringing
    # to say why (the 2026-09-11 audit, finding create-07). Called immediately
    # after the control, before Reroll and Lock draw over its rect.
    widgets.field_error(ctx.state, "seed")
    # Wrapped rather than clipped: at 1.5 scale the seed field, its label,
    # Reroll, Lock and the help marker come to more than the sidebar's content
    # region, and ``same_line`` past the edge draws a control nowhere -- the
    # bug that once hid seven of them. Found by the 1.5-scale half of the
    # screenshot pass, which is the half that keeps finding these.
    if controls.button("Reroll", role=controls.ButtonRole.GHOST):
        form["seed"] = random_seed()
    changed, locked = form_ui.switch(
        "seed_locked",
        "Lock seed",
        bool(form["seed_locked"]),
        help_text="Reuse this seed when the form is unchanged.",
        helper="Unlocked, every submit rerolls it.",
    )
    if changed:
        form["seed_locked"] = locked

def _advisory_fix(ctx: Any, form: dict[str, Any], advisory: problem_types.Advisory) -> None:
    """The one-press repair for an advisory, where there is a safe one.

    ``_preflight_fix``'s shape and its rule: only repairs that need no second
    decision. Appending a clause is reversible and visible in the box the user
    is looking at; rewriting their sentence would not be.
    """
    if getattr(advisory, "field", "") != "prompt":
        return
    prompt = str(form.get("prompt") or "")
    if create_recipe.CLOSED_FORM_CLAUSE in prompt:
        return
    if controls.button(
        "Ask for a closed form##advisory-open-form", role=controls.ButtonRole.GHOST
    ):
        form["prompt"] = f"{prompt.rstrip().rstrip(',')}, {create_recipe.CLOSED_FORM_CLAUSE}"
        ctx.state.clear_field_error("prompt")

def _plan_footer(ctx: Any, form: dict[str, Any]) -> None:
    """What a press will cost, and what is stopping it. Pinned, never scrolled.

    The button this used to carry is in ``create_brief`` now. The *statement*
    stays here, because it is a paragraph with one-click repairs in it and a
    one-row bar has no place to put either -- the bar's disabled Generate wears
    the first problem as its tooltip and this is the list.
    """
    imgui.dummy((0, sp(8)))
    widgets.divider()
    _generation_plan(
        ctx, form, create_recipe.problems_for(ctx, form), create_recipe.advisories_for(ctx, form)
    )

def _generation_plan(
    ctx: Any,
    form: dict[str, Any],
    problems: list[problem_types.Problem],
    advisories: list[problem_types.Advisory] | None = None,
) -> None:
    """The persistent, actionable statement of what Generate will do.

    Validation still belongs to :func:`create_recipe.validate` and the service.  This is the
    in-place account of their answer, kept immediately beside the commitment
    rather than in a footer whose errors explain nothing about the run.
    """
    widgets.secondary("Generation plan")
    resolved = create_recipe.resolved_recipe(ctx, form)
    plan = generation_workspace.plan_for(form, resolved)
    imgui.text_wrapped(plan.stages)
    if plan.generations > 0:
        widgets.muted(
            f"{plan.candidates} candidate{'s' if plan.candidates != 1 else ''} · "
            f"{plan.generations} image generation"
            f"{'s' if plan.generations != 1 else ''} · "
            f"{plan.duration}"
        )
    else:
        # A character draws no images at all, and "1 candidate · 0 image
        # generations" is a line that reads as a bug rather than as a fact.
        # The duration still matters -- it is the whole cost of the press.
        widgets.muted(plan.duration)
    widgets.muted(f"Recipe: {plan.recipe}")
    active = getattr(ctx.cache, "active", None)
    if active is not None:
        position = generation_workspace.queue_position(ctx, str(active.get("id") or ""))
        if active.get("status") == "queued":
            widgets.muted(f"Queue: position {position}" if position else "Queue: waiting")
        else:
            widgets.muted("Queue: one local generation is running")
    else:
        widgets.muted("Queue: ready")
    refusal = str(getattr(ctx.state.create, "submit_refusal", "") or "")
    advisories = advisories or []
    if not problems and not refusal:
        # "Ready to generate" is still true with an advisory standing -- that
        # is the whole difference between the two lists -- so it is said, and
        # then the advisory is drawn under it rather than instead of it.
        widgets.muted("Ready to generate.")
        _advisories_block(ctx, form, advisories)
        return
    if refusal:
        # Above the form problems: the form is fine -- this is the *door*
        # saying no, and it is the reason the last press did nothing. It stays
        # until a press is accepted, because a fading toast is what this
        # sentence was already tried as.
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
        imgui.text_wrapped(f"Refused: {refusal}")
        imgui.pop_style_color()
    for problem in problems:
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
        imgui.text_wrapped(f"Needs attention: {problem}")
        imgui.pop_style_color()
        _preflight_fix(ctx, form, problem)
    _advisories_block(ctx, form, advisories)

def _advisories_block(
    ctx: Any, form: dict[str, Any], advisories: list[problem_types.Advisory]
) -> None:
    """The advisories, under the problems, in the warning colour.

    Under, and in a different colour, because the reading order is the order
    they matter in: a problem is why the button is off, and an advisory is
    something to think about while pressing it. "Worth knowing" rather than
    "Needs attention" for the same reason -- nothing here needs anything.
    """
    for advisory in advisories:
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.WARN)))
        imgui.text_wrapped(f"Worth knowing: {advisory}")
        imgui.pop_style_color()
        _advisory_fix(ctx, form, advisory)

def _preflight_fix(ctx: Any, form: dict[str, Any], problem: problem_types.Problem) -> None:
    """Offer the safe, direct repairs which do not need another decision."""
    field = getattr(problem, "field", "")
    message = str(problem)
    # First, because the character arm's refusals are about a species and this
    # function's other arms are about a checkpoint: "guidance 0" appears in no
    # character sentence, but "not downloaded" could one day, and a pane that
    # offered "Open model setup" under "Warlock has no manticore yet" would be
    # pointing at a download that changes nothing.
    if create_recipe.is_character(form) and settings_character.preflight_fix(ctx, form, problem):
        return
    if field == "ref_path":
        if controls.button(
            "Choose a reference##preflight-reference", role=controls.ButtonRole.GHOST
        ):
            ctx.submit(
                "ref-upload", dialogs.open_file, "Choose a reference image", dialogs.IMAGE_FILTER
            )
        return
    # The 2026-09-18 audit, finding create-04: this used to match on
    # ``message`` alone, with a ``"guidance 0" in message`` alternative that
    # matches no reachable ``Problem`` -- the one refusal that ever said
    # "guidance 0" (``generation.validate_request``'s CompatibilityIssue, née
    # the create-05 finding) reaches the pane through ``refuse``'s
    # ``note_field_error``, never through this function's ``problem``
    # argument. The live ``Problem`` this repairs is
    # ``recipe.py``'s own -- field ``base_model``, message naming
    # "full-CFG" -- so the field is now part of the match rather than
    # trusting prose alone to say which control the repair changes.
    if field == "base_model" and "full-CFG" in message:
        if controls.button(
            "Switch to Automatic##preflight-quality", role=controls.ButtonRole.GHOST
        ):
            # The 2026-09-07 Create review, item 5.5.2, the create-03 finding's
            # shape repeated: this used to write ``form["quality"]``, a key no
            # control sets any more since the Fast/Quality tier folded into the
            # Model combo (``create_recipe.model_options``) -- so the write changed nothing
            # ``resolve_recipe`` reads once ``model_mode`` is "auto". What the
            # combo's own Automatic entry actually writes is these two fields
            # (``_model``'s ``else`` branch), which is what genuinely decides
            # whether the recipe that resolves next can run a ControlNet.
            form["model_mode"] = "auto"
            form["model_override"] = ""
            create_recipe.clear_for_tier(ctx, form)
            ctx.state.clear_field_error("base_model")
        return
    if "not downloaded" in message and controls.button(
        "Open model setup##preflight-models", role=controls.ButtonRole.GHOST
    ):
        from .....state import set_mode

        set_mode(ctx.state, "settings")

def submit_job(ctx: Any, run: Any) -> bool:
    """Queue ``run`` under the shared ``"submit"`` key. -> whether it was taken.

    ``TaskRunner.submit`` refuses a key that is already in flight and says so
    only by its return, and four callers ignored it: a second Ctrl+Enter while
    the first was still at the door queued nothing and said nothing.
    ``submit_promotion`` had the check; this is it, once, for every door.
    """
    if ctx.submit("submit", run):
        return True
    ctx.toast("Still submitting the last one - try again in a moment.")
    return False

def _enter_pressed() -> bool:
    return imgui.is_key_pressed(imgui.Key.enter) or imgui.is_key_pressed(imgui.Key.keypad_enter)

def _generate_tile_sheet(ctx: Any, form: dict[str, Any]) -> bool:
    """Submit the tile set, on the shared ``submit`` key.

    The same key every other output uses, deliberately: it is one form and one
    Generate button, so two submits in flight from it is the thing the key
    exists to prevent -- and the busy state the button reads is keyed on that
    name.

    The form values are read here, on the frame thread, because they are UI
    state; the reference *file* is read in the task, because a large one would
    freeze the window for as long as the disk took. ``generate``'s own split,
    kept.
    """
    kwargs = create_recipe.tile_sheet_kwargs(form)
    ref_path = form.get("ref_path") or ""

    def run():
        reference = None
        if ref_path:
            try:
                with Path(ref_path).open("rb") as fh:
                    reference = fh.read(MAX_UPLOAD_BYTES + 1)
            except OSError as exc:
                # ``field=`` so the ring lands on the file control rather than
                # the refusal arriving as a toast with no subject.
                raise Invalid(
                    f"could not read {Path(ref_path).name}: {exc}", field="ref_path"
                ) from exc
        return svc_tilesheets.create_tile_sheet(ctx.svc, reference=reference, **kwargs)

    return submit_job(ctx, run)

def refuse(ctx: Any, problems: list[problem_types.Problem]) -> None:
    """Say no where the user can see it, whichever door they came through.

    Shared with ``settings_3d.promote`` because the two refusals are the same
    refusal: a form that would not submit, reached from a button that is
    disabled *and* from two keyboard doors where nothing is disabled at all.
    The button path shows the whole list as red text above itself and needs
    nothing from here; the keyboard paths get the first problem as a toast --
    first rather than all of them, because a stack of four toasts is a wall,
    and the ring below marks the rest.
    """
    ctx.state.clear_field_errors()
    for problem in problems:
        ctx.state.note_field_error(getattr(problem, "field", ""), str(problem))
    if problems:
        ctx.toast(str(problems[0]), "warn")

def generate(ctx: Any, form: dict[str, Any]) -> None:
    character = create_recipe.is_character(form)
    if character:
        # **The keyboard door.** Ctrl+Enter and the palette call straight in
        # here and never draw the Character block, so a form filled only from
        # that draw would submit the species of the *previous* prompt for
        # anyone who typed and pressed in one motion.
        character_engine.sync_from_prompt(form)
    problems = create_recipe.validate(form, ctx)
    if character:
        problems = [*problems, *character_engine.problems(ctx, form)]
    # The install-shaped refusal, folded in behind the form-shaped ones: it is
    # the same "this will not submit" from the user's side, and putting it here
    # rather than at the service door turns a two-minute queue-and-fail into an
    # immediate sentence naming the control. The service still refuses at the
    # door; this only means the user rarely reaches it.
    weights = create_recipe.weights_problem(ctx, form)
    if weights is not None and not problems:
        problems = [weights]
    if problems:
        # Said out loud, because this is not only the button's path. Ctrl+Enter
        # and the palette's Generate call straight in here, and the only
        # feedback a refusal had was the red block above the button -- which
        # the keyboard user is by definition not looking at, and which the
        # palette covers. So the first problem becomes a toast and every
        # problem that names a control gets its ring, which is exactly what the
        # button path shows without being asked.
        refuse(ctx, problems)
        return
    # A new submit is judged on its own: the rings from the last one describe a
    # request that no longer exists, and leaving them up would have the app
    # pointing at a control while it works on the value in it.
    ctx.state.clear_field_errors()
    # The seed rolls and the prompt joins history only once the submit is
    # *accepted*: ``ctx.submit`` refuses a key still in flight, and a refused
    # Ctrl+Enter used to reroll the seed and grow the history while queuing
    # nothing and saying nothing.
    seed_before = form["seed"]
    if not form["seed_locked"]:
        # Generation is deterministic in the seed, so an unchanged form would
        # otherwise produce the identical image twice and read as a no-op.
        form["seed"] = random_seed()
    if character:
        # The second output that does not go through ``create_job``, and the
        # only one that generates no image at all: ``create_character`` builds
        # a mesh from the recipe, mints its row finished and queues the rig
        # that will mint the sheet. ``create_job``'s own allowlist refuses
        # ``asset_type="character"`` by name, which is what keeps this branch
        # from being merely a convention.
        if settings_character.submit(ctx, form):
            ctx.state.remember_prompt(form["prompt"])
        else:
            form["seed"] = seed_before
        return
    if create_recipe.is_tile_arm(form):
        # The one output that does not go through ``create_job``: a tile set is
        # its own job kind, with its own door and its own admission. The sprite
        # arm deliberately *does* go through it -- see ``create_recipe.submit_kwargs`` -- so
        # this is the only branch here.
        if _generate_tile_sheet(ctx, form):
            ctx.state.remember_prompt(form["prompt"])
        else:
            form["seed"] = seed_before
        return
    resolved = None
    if create_assets.selected(form).key in {"image", "3d_model", "seamless_material"}:
        request = generation.request_from_legacy(form)
        resolved = generation.resolve_recipe(request, ctx.svc.config)
        recipe_issues = generation.validate_request(request, resolved)
        if recipe_issues:
            refuse(ctx, [problem_types.Problem(item.message, item.field) for item in recipe_issues])
            return
    kwargs = create_recipe.submit_kwargs(form)
    if resolved is not None:
        # Automatic routing is resolved at submit time. The selected recipe is
        # copied after the legacy door accepts the request so reruns retain the
        # exact model/checksum even if the registry changes later.
        kwargs["guidance_fields"]["base_model"] = resolved.base_model
        # The request document preserves the user's Avoid text, but an inert
        # negative branch must not be sent through to a distilled worker.
        kwargs["negative_prompt"] = generation.effective_negative_prompt(request, resolved) or None
        # Handed to the door rather than merged onto the row afterwards. The
        # merge lost a race with the worker twice over -- ``next_queued`` can
        # claim the row first, and ``_q_generate`` writes its claim-time
        # snapshot of ``params`` back whole, deleting anything added since.
        # See ``_jobs_create.create_job``'s ``extra_params``.
        kwargs["extra_params"] = {
            "generation_request": request.to_dict(),
            "resolved_recipe": {
                "version": generation.RECIPE_REGISTRY_VERSION,
                **resolved.to_dict(),
            },
        }
    ref_path = form.get("ref_path")

    # The form values are read here, on the frame thread, because they are UI
    # state; the *file* is read in the task, because a large one would freeze
    # the window for as long as the disk took. Copied from settings_3d.upload,
    # including the MAX_UPLOAD_BYTES + 1 read that is create_job's contract.
    def run():
        if ref_path:
            try:
                with Path(ref_path).open("rb") as fh:
                    kwargs["reference"] = fh.read(MAX_UPLOAD_BYTES + 1)
            except OSError as exc:
                # ``field=`` so the ring lands on the file control rather
                # than the refusal arriving as a toast with no subject: the
                # caller cannot know which of the form's inputs was at fault,
                # and this one does.
                raise Invalid(
                    f"could not read {Path(ref_path).name}: {exc}", field="ref_path"
                ) from exc
        return svc_jobs.create_job(ctx.svc, **kwargs)

    if submit_job(ctx, run):
        ctx.state.remember_prompt(form["prompt"])
    else:
        form["seed"] = seed_before
