"""Poser's character-sheet stage: Troupe folded in (P9, 2026-09-18).

Troupe was a third door onto one sheet pipeline -- rig, then clips, then a
sheet -- and this module is that third door's drawing, ported into Poser's
own workspace rather than a workspace of its own. Four jobs, each a straight
port of one of Troupe's panes, now that all four read the *bound* asset
(``PoserState.job_id``) instead of a selection Troupe kept independently:

* :func:`draw_sheets_section` -- Troupe's cast/sheets list, minus the
  cross-character cast (dropped, decision 1 of the folding brief: binding an
  asset through the library's own picker is the one way in now) -- drawn from
  ``ui/panes/library.py``.
* :func:`draw_new_character` -- Troupe's "New character" form (the reference
  that starts the chain), drawn from the same pane, always available: a fresh
  character has nothing bound yet to gate it on.
* :func:`draw_preview` -- Troupe's centre pane, the sprite at an integer
  scale -- drawn from ``ui/viewport.py`` in place of the pose viewport while
  ``PoserState.sheet_view`` is set.
* :func:`draw_info` -- Troupe's two right panes (what the sheet is, and the
  ways out) merged into one, drawn from ``ui/panes/controls.py`` in place of
  the pose controls under the same condition.

Every control here calls into :mod:`..mode` (``poser_mode``); this module
only draws.
"""

from __future__ import annotations

from typing import Any

from ......kernels import charsheet
from ......kernels.rig import store as rig_store
from ..... import controls, forms, icons, probe, theme, tokens, toolbar, verbs, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import mode as poser_mode
from ...engine import qa

#: The "Start a new character" section's own ``widgets.request_open`` key --
#: Home's "New character" tile (``panes.landing.start_poser``) asks for it
#: open on arrival, the way a dropped file opens the 2D pane's Reference
#: block. Named here rather than at the one call site that opens it, so a
#: header whose key changes cannot drift from the request that names it.
NEW_CHARACTER_SECTION = "poser-new-character"

#: The playback multipliers the transport offers. A short ladder rather than a
#: slider: the useful speeds for reading a run cycle are a quarter, a half and
#: full, and a continuous control invites 0.87x, which is not a thing anyone
#: wants.
_SPEEDS: tuple[tuple[str, str], ...] = (
    ("0.25", "0.25x"),
    ("0.5", "0.5x"),
    ("1.0", "1x"),
    ("2.0", "2x"),
    ("4.0", "4x"),
)


def _speed_key(value: float) -> str:
    """The ladder rung ``value`` sits on -- nearest, never a refusal."""
    return min((key for key, _ in _SPEEDS), key=lambda key: abs(float(key) - value))


# --- the left column: which sheet, and starting a new one -------------------


def draw_sheets_section(ctx: Any, state: Any) -> None:
    """The bound asset's character sheets, and the door to render another.

    Drawn only when an asset is bound -- ``ui/panes/library.py``'s own call
    site gates it, the same way it gates the skeleton block above it.
    """
    from imgui_bundle import imgui

    widgets.section("Character sheets")
    manual_render.help_button(ctx, "poser-sheets")
    records = poser_mode.sheets(ctx, state.job_id)
    if not records:
        widgets.empty_state(
            icons.PERSON_STANDING,
            "No character sheets yet",
            "Build one below -- it renders from the rig already bound.",
        )
    else:
        for record in records:
            size = int(record.get("frame_size") or 0)
            name = str(record.get("name") or "").strip()
            label = f"{name or 'sheet'} - {size}px"
            selected = state.sheet_view and record["id"] == state.sheet_id
            if controls.selectable_row(f"poser-sheet-{record['id']}", label, selected=selected):
                poser_mode.select_sheet(ctx, record["id"])
            if poser_mode.needs_repair(record):
                # On the row rather than only in the info pane, so a user
                # comparing 32px against 64px can see which came back broken
                # without selecting each in turn. Structural, never the QA
                # scores, which rank and gate nothing.
                imgui.same_line()
                widgets.pill(f"{icons.TRIANGLE_ALERT} needs repair", theme.WARN)
        if state.sheet_view and controls.button("Back to posing", (-1, 0)):
            state.sheet_view = False
    imgui.dummy((0, sp(tokens.SP_1)))
    _build_another(ctx, state)


def _build_another(ctx: Any, state: Any) -> None:
    """Another sheet of the bound character, at the form's current options.

    Collapsed: the settings underneath are the same ones
    :func:`draw_new_character` draws, and a column already carrying the pose
    library, the rigged-asset picker and this sheet list has no room for them
    standing open by default.
    """
    from imgui_bundle import imgui

    if not widgets.header("Build a new sheet", default_open=False):
        return
    options = poser_mode.sheet_options(ctx)
    form = poser_mode.sheet_form(ctx)
    with forms.Form(
        "poser-sheet-build",
        errors=ctx.state.field_errors,
        on_edit=ctx.state.clear_field_error,
    ) as form_ui:
        _layout(form, form_ui, options)
        _style(form, form_ui)
        _frame_rate(form, form_ui, options)
        _size(form, form_ui, options)
        _palette(ctx, form, form_ui, options)
    imgui.dummy((0, sp(tokens.SP_1)))
    key = f"troupe-sheet:{state.job_id}"
    count = poser_mode.cell_count(form)
    valid = 0 < count <= charsheet.MAX_CELLS
    if widgets.disabled_button(
        "Build another sheet",
        not ctx.busy(key) and valid,
        (-1, 0),
        reason=(
            "A sheet is already being queued for this character."
            if ctx.busy(key)
            else f"Select a layout of at most {charsheet.MAX_CELLS} cells."
        ),
    ):
        poser_mode.build_sheet(ctx, state.job_id, form)
    widgets.cost_note(
        f"{count} rendered cells from the rig that already exists -- minutes of "
        "CPU, no GPU. The settings above are what it uses."
    )
    if count > charsheet.WARN_CELLS:
        widgets.muted(
            f"Large sheet: over {charsheet.WARN_CELLS} cells can take substantially longer."
        )


def draw_new_character(ctx: Any) -> None:
    """The form that starts a character from nothing -- Troupe's own "New
    character" pane, ported whole.

    It submits the *first* link of the chain and nothing else: one cheap pose
    reference. The gate is the whole point of the shape -- the user approves
    the drawing in Create, and only then is the reconstruction spent -- so
    this section deliberately has no "and then build everything" button.
    Always offered, bound asset or not: a fresh character has nothing to bind
    to yet.
    """
    from imgui_bundle import imgui

    if not widgets.header(
        "Start a new character", default_open=False, persist_key=NEW_CHARACTER_SECTION
    ):
        return
    manual_render.help_button(ctx, "poser-new-character")

    options = poser_mode.sheet_options(ctx)
    form = poser_mode.sheet_form(ctx)
    with forms.Form(
        "poser-new-character",
        errors=ctx.state.field_errors,
        on_edit=ctx.state.clear_field_error,
    ) as form_ui:
        _changed, form["prompt"] = form_ui.text("prompt", "Describe them", form["prompt"])
        _changed, form["variant"] = form_ui.combo(
            "variant",
            "Build",
            form["variant"],
            [(v, v) for v in options.get("variants") or ()],
        )
        _changed, form["pose"] = form_ui.combo(
            "pose",
            "Reference pose",
            form["pose"],
            [
                (name, poser_mode.POSE_LABELS.get(name, name))
                for name in options.get("poses") or ()
            ],
            help_text=(
                "The stick figure the first drawing is conditioned on. "
                "A-pose matches the rig template, so the joints are fitted "
                "straight to it. T-pose separates the limbs further, which is "
                "what the reconstruction has the least trouble with -- pick it "
                "if the arms come back fused to the body."
            ),
        )
        _layout(form, form_ui, options)
        _style(form, form_ui)
        _frame_rate(form, form_ui, options)
        _size(form, form_ui, options)
        _palette(ctx, form, form_ui, options)
    imgui.dummy((0, sp(tokens.SP_1)))
    _submit_new_character(ctx, form)


def _layout(form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]) -> None:
    """Per-movement frame and direction controls; the total is always derived.

    Every row comes from ``clip_vocabulary[<the rig's template>]`` rather than
    a second, hand-written list -- see
    ``dev/measurements/2026-09-12-troupe-open-clip-vocabulary.md``.
    """
    layout = form["layout"]
    form_ui.note("layout")
    template = str(layout.get("template") or (options.get("defaults") or {}).get("template") or "")
    vocabulary = {
        str(row.get("name")): row
        for row in (options.get("clip_vocabulary") or {}).get(template) or ()
    }
    presets = [int(n) for n in options.get("direction_presets") or (1, 4, 8, 16)]
    for movement in layout.get("movements") or ():
        key = str(movement.get("key") or "")
        label = key.replace("_", " ").title()
        clip = vocabulary.get(key) or {}
        provisional = bool(clip.get("provisional"))
        _changed, movement["enabled"] = form_ui.switch(
            f"movement_{key}",
            label,
            bool(movement.get("enabled", True)),
            help_text=(
                "Placeholder keyframes; an animator's pass is still owed"
                if provisional
                else ""
            ),
            helper="Provisional" if provisional else "",
        )
        if not movement["enabled"]:
            continue
        _changed, frames = form_ui.number(
            f"frames_{key}",
            f"{label} frames",
            int(movement.get("frames") or clip.get("frames") or 1),
            helper=f"1-{charsheet.MAX_FRAMES} frames",
        )
        movement["frames"] = max(1, min(int(frames), charsheet.MAX_FRAMES))
        _changed, directions = form_ui.combo(
            f"directions_{key}",
            f"{label} directions",
            str(movement.get("directions") or 8),
            [(str(n), f"{n}-direction") for n in presets],
        )
        movement["directions"] = int(directions)


def _style(form: dict[str, Any], form_ui: forms.Form) -> None:
    """Pixel art or HD -- what render this sheet's cells come out as."""
    _changed, style = form_ui.combo(
        "style",
        "Style",
        poser_mode._style_choice(form),
        [
            (poser_mode.STYLE_PIXEL_ART, "Pixel art"),
            (poser_mode.STYLE_HD, "HD"),
        ],
        help_text=(
            "Pixel art reduces the render to a logical size, a colour budget "
            "and an outline pass. HD keeps the render as painted, with no "
            "colour budget."
        ),
    )
    form["style"] = style


def _frame_rate(form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]) -> None:
    """A layout-wide rate, or every clip's own recorded speed."""
    choices = [("", "Authored")] + [
        (str(n), f"{n} fps") for n in options.get("fps_choices") or ()
    ]
    current = "" if form.get("fps") in (None, "") else str(form["fps"])
    _changed, choice = form_ui.combo(
        "fps",
        "Frame rate",
        current,
        choices,
        help_text=(
            "Authored keeps every included movement at its own recorded "
            "speed. A rate here overrides all of them to play at once."
        ),
    )
    form["fps"] = int(choice) if choice else None


def _size(form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]) -> None:
    presets = options.get("camera_presets") or {}
    _changed, camera = form_ui.combo(
        "camera",
        "Camera",
        str(form.get("camera") or ""),
        [(key, str(entry.get("label") or key)) for key, entry in presets.items()],
        helper=_camera_helper(presets, str(form.get("camera") or "")),
    )
    form["camera"] = camera
    _logical_size(form, form_ui, options)
    hd = poser_mode._style_choice(form) == poser_mode.STYLE_HD
    _changed, outline = form_ui.combo(
        "outline",
        "Outline",
        form["outline"],
        [(m, m) for m in options.get("outline_modes") or ()],
        enabled=not hd,
        reason="Style is HD, so there is no outline pass." if hd else "",
    )
    form["outline"] = outline
    _changed, reduce_mode = form_ui.combo(
        "reduce_mode",
        "Reduction",
        form["reduce_mode"],
        [(m, m) for m in options.get("reduce_modes") or ()],
    )
    form["reduce_mode"] = reduce_mode


#: The combo's sentinel for "type your own number" -- kept identical to
#: ``poser_send._CUSTOM``, though the two modules do not share the constant:
#: neither reads the other's combo state.
_CUSTOM = "custom"


def _logical_size(form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]) -> None:
    """The sprite size row: the ladder, plus a custom box."""
    sizes = options.get("logical_sizes") or ()
    current = int(form["logical_size"])
    if "logical_size_custom" not in form:
        form["logical_size_custom"] = current not in sizes
    choices = [(str(s), f"{s} px") for s in sizes]
    choices.append((_CUSTOM, "Custom..."))
    combo_value = _CUSTOM if form["logical_size_custom"] else str(current)
    _changed, picked = form_ui.combo("logical_size", "Sprite size", combo_value, choices)
    if picked == _CUSTOM:
        form["logical_size_custom"] = True
    else:
        form["logical_size_custom"] = False
        form["logical_size"] = int(picked)
    if form["logical_size_custom"]:
        lo, hi = options.get("logical_size_range") or (8, 256)
        _changed, value = form_ui.number(
            "logical_size", "Custom size (px)", int(form["logical_size"])
        )
        form["logical_size"] = max(int(lo), min(int(hi), int(value)))
        if form["logical_size"] and charsheet.RENDER_SIZE % form["logical_size"] != 0:
            widgets.muted_wrapped(
                "Sizes that don't divide 512 are resized with nearest-neighbour."
            )


def _camera_helper(presets: dict[str, Any], key: str) -> str:
    entry = presets.get(key) or {}
    if "elevation" not in entry:
        return ""
    return f"{float(entry['elevation']):g} degrees above the horizon"


def _palette(ctx: Any, form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]) -> None:
    """A designed palette if one is installed, a colour budget otherwise."""
    hd = poser_mode._style_choice(form) == poser_mode.STYLE_HD
    installed = list(options.get("palettes") or ())
    choices = [("", "Derived from the render")] + [(name, name) for name in installed]
    _changed, palette = form_ui.combo(
        "palette",
        "Palette",
        form["palette"],
        choices,
        enabled=not hd,
        reason="Style is HD, so there is no palette to choose." if hd else "",
    )
    form["palette"] = palette
    if not palette:
        _changed, colors = form_ui.combo(
            "colors",
            "Colours",
            str(form["colors"]),
            [(str(n), f"{n} colours") for n in options.get("colors") or ()],
            enabled=not hd,
            reason="Style is HD, so there is no colour budget." if hd else "",
        )
        form["colors"] = int(colors)
    _changed, form["dither"] = form_ui.switch(
        "dither",
        "Dither",
        bool(form["dither"]),
        enabled=not hd,
        reason="Style is HD, so there is no dithering to turn on." if hd else "",
    )
    _changed, form["name"] = form_ui.text(
        "name",
        "Name this sheet",
        str(form.get("name") or ""),
        hint="optional",
        max_length=rig_store.MAX_SHEET_NAME,
        helper="Shown in the sheet chooser. The size and cell count are added for you.",
    )


def _submit_new_character(ctx: Any, form: dict[str, Any]) -> None:
    """The section's own commit action -- ``widgets.primary_button``, the one
    accented call to action per pane (the 2026-09-05 consistency pass):
    Troupe's own copy of this door drew a plain ``disabled_button``, and the
    only primary-styled control in the whole mode was the (now-dropped, per
    decision 1 of the P9 folding brief) "existing mesh" picker's Send button.
    """
    busy = ctx.busy("troupe-start")
    count = poser_mode.cell_count(form)
    ready = bool(form["prompt"].strip()) and 0 < count <= charsheet.MAX_CELLS
    if busy:
        widgets.busy("Drawing the reference")
    if widgets.primary_button(
        "Draw the reference",
        (-1, 0),
        enabled=not busy and ready,
        reason=(
            f"Describe the character and select a layout of at most "
            f"{charsheet.MAX_CELLS} cells."
        )
        if not ready
        else "A reference is already being queued.",
    ):
        poser_mode.start_character(ctx, form)
    widgets.cost_note(
        f"One image, and then it stops. After approval, the mesh, rig, and "
        f"{count} rendered cells follow."
    )
    if count > charsheet.WARN_CELLS:
        widgets.muted(
            f"Large sheet: over {charsheet.WARN_CELLS} cells can take substantially longer."
        )


# --- the centre pane: the sprite ---------------------------------------------


def draw_preview(ctx: Any) -> None:
    """Troupe's centre pane, ported whole: one sprite, at an integer scale.

    Also the sheet section's heartbeat -- there is no per-mode update hook, so
    the pane that draws is what pumps the clock (``poser_mode.sheet_advance``).
    """
    from imgui_bundle import imgui

    state = poser_mode.ensure(ctx)
    poser_mode.sheet_advance(ctx, imgui.get_io().delta_time)

    _transport(ctx, state)
    imgui.dummy((0, sp(tokens.SP_2)))

    texture = poser_mode.atlas_texture(ctx)
    record = poser_mode.active_sheet(ctx)
    if texture is None or record is None:
        widgets.empty_state(
            icons.PERSON_STANDING,
            "No character on screen",
            "Pick a sheet on the left, or build one.",
        )
        return
    _scorecard(ctx, state)
    _sprite(ctx, state, texture, record)


#: The heatmap's square, in design pixels.
QA_CELL = 14.0
QA_GAP = 2.0


def _scorecard(ctx: Any, state: Any) -> None:
    """The animation scores as a heatmap over the frame x direction matrix."""
    from imgui_bundle import imgui

    score = poser_mode.scores(ctx)
    movement = poser_mode.preview_movement(ctx)
    if movement is None:
        return
    if score is None:
        widgets.muted(
            "Could not score this sheet." if poser_mode.scores_failed(ctx) else "scoring..."
        )
        return
    if score.worst is None:
        widgets.muted("No frame flagged.")
    else:
        metric, value, cell = score.worst
        where = next((c for c in score.cells if c.cell == cell), None)
        place = (
            f" ({where.animation}/{where.direction}, frame {where.frame + 1})" if where else ""
        )
        widgets.muted(
            f"worst: {metric} {value:.2f}{place} - "
            f"{score.flagged} of {len(score.cells)} cells flagged"
        )

    by = score.lookup()
    animation = str(movement.get("key") or "")
    frames = int(movement.get("frames") or 1)
    directions = [str(d.get("key") or "") for d in movement.get("directions") or ()]
    side = sp(QA_CELL)
    gap = sp(QA_GAP)
    draw = imgui.get_window_draw_list()
    fills = {
        qa.LEVEL_OK: imgui.get_color_u32(theme.rgba(theme.ELEV_2)),
        qa.LEVEL_WARN: imgui.get_color_u32(theme.rgba(theme.WARN)),
        qa.LEVEL_BAD: imgui.get_color_u32(theme.rgba(theme.ERR)),
    }
    ring = imgui.get_color_u32(theme.rgba(theme.ACCENT))
    ink = imgui.get_color_u32(theme.rgba(theme.TEXT))
    label_w = max((imgui.calc_text_size(d).x for d in directions), default=0.0) + sp(8)
    for direction in directions:
        origin = imgui.get_cursor_screen_pos()
        draw.add_text(
            (origin.x, origin.y + (side - imgui.get_text_line_height()) * 0.5), ink, direction
        )
        imgui.set_cursor_screen_pos((origin.x + label_w, origin.y))
        for frame in range(frames):
            cell = by.get((animation, direction, frame))
            level = qa.level(cell) if cell is not None else qa.LEVEL_OK
            current = direction == state.sheet_direction and frame == state.sheet_frame
            a = imgui.get_cursor_screen_pos()
            b = (a.x + side, a.y + side)
            clicked = imgui.invisible_button(f"##poser-sheet-qa/{direction}/{frame}", (side, side))
            tip = _tooltip(cell)
            probe.record(
                label=f"{animation} {direction} frame {frame + 1}",
                kind="heatmap_cell",
                selected=current,
                tooltip=tip,
            )
            draw.add_rect_filled((a.x, a.y), b, fills[level], sp(2))
            if level == qa.LEVEL_BAD:
                inset = side * 0.28
                draw.add_line((a.x + inset, a.y + inset), (b[0] - inset, b[1] - inset), ink, 1.5)
                draw.add_line((b[0] - inset, a.y + inset), (a.x + inset, b[1] - inset), ink, 1.5)
            if current:
                draw.add_rect((a.x - 1, a.y - 1), (b[0] + 1, b[1] + 1), ring, sp(2), 2.0)
            if imgui.is_item_hovered() and tip:
                imgui.set_tooltip(tip)
            if clicked:
                poser_mode.sheet_goto(ctx, direction, frame)
            if frame + 1 < frames:
                imgui.same_line(0.0, gap)
    imgui.dummy((0, sp(4)))


def _tooltip(cell: Any) -> str:
    if cell is None:
        return ""
    lines = [", ".join(cell.flags) if cell.flags else "ok"]
    for name in qa.METRICS:
        value = cell.metrics.get(name)
        if value is not None:
            lines.append(f"{name}: {value:.2f}")
    return "\n".join(lines)


def _transport(ctx: Any, state: Any) -> None:
    """Play/pause, the two clip selectors, and the frame step."""
    from imgui_bundle import imgui

    layout = poser_mode.preview_layout(ctx)
    movements = list(layout.get("movements") or ())
    movement = poser_mode.preview_movement(ctx)
    manual_render.help_button(ctx, "poser-sheet-preview")

    label, glyph, tip = widgets.transport_label(state.sheet_playing)
    frames = int((movement or {}).get("frames") or 1)

    def _trailing() -> None:
        widgets.frame_counter(state.sheet_frame, frames)
        imgui.same_line()
        imgui.set_next_item_width(sp(110))
        changed, zoom = controls.input_int("##poser-sheet-zoom", int(state.sheet_zoom), 1, 2)
        if changed:
            state.sheet_zoom = max(1, min(int(zoom), 32))
        imgui.same_line()
        imgui.set_next_item_width(sp(90))
        picked, name = controls.combo(
            "##poser-sheet-speed", _speed_key(state.sheet_speed), _SPEEDS
        )
        if picked:
            state.sheet_speed = float(name)

    row = [
        toolbar.Item(key="play", label=label, icon=glyph, tooltip=tip, pinned=True),
        toolbar.Item(key="back", label="Previous frame", icon=icons.ARROW_LEFT, pinned=True),
        toolbar.Item(key="fwd", label="Next frame", icon=icons.CHEVRON_RIGHT, pinned=True),
        toolbar.Item(
            key="checker",
            label="Checkerboard",
            icon=icons.GRID,
            tooltip="C -- a checkerboard behind the sprite, so transparent "
            "pixels are transparent rather than the panel's colour.",
            selected=state.sheet_checker,
            priority=1,
        ),
        toolbar.Item(
            key="pivot",
            label="Pivot",
            icon=icons.CROSSHAIR,
            tooltip="P -- where the engine will place this sprite, from the "
            "sidecar. Nothing is drawn on a sheet that records no pivot.",
            selected=state.sheet_show_pivot,
            priority=1,
        ),
    ]
    clicked = toolbar.toolbar("poser-sheet-transport", row, trailing=(sp(340), _trailing))
    if clicked == "play":
        state.sheet_playing = not state.sheet_playing
    elif clicked == "back":
        poser_mode.sheet_step(ctx, -1)
    elif clicked == "fwd":
        poser_mode.sheet_step(ctx, 1)
    elif clicked == "checker":
        state.sheet_checker = not state.sheet_checker
    elif clicked == "pivot":
        state.sheet_show_pivot = not state.sheet_show_pivot

    _choices(
        [str(row.get("key") or "") for row in movements],
        state.sheet_animation,
        "anim",
        lambda name: poser_mode.set_sheet_animation(ctx, name),
    )
    _choices(
        [str(row.get("key") or "") for row in (movement or {}).get("directions") or ()],
        state.sheet_direction,
        "dir",
        lambda name: poser_mode.set_sheet_direction(ctx, name),
    )


def _choices(names: list[str], current: str, suffix: str, choose: Any) -> None:
    """One wrapping row of radio buttons."""
    from imgui_bundle import imgui

    for name in names:
        if controls.radio_button(f"{name}##poser-sheet-{suffix}", current == name):
            choose(name)
        widgets.same_line_or_wrap(
            widgets.button_width(name)
            + imgui.get_frame_height()
            + imgui.get_style().item_inner_spacing.x
        )
    imgui.new_line()


def _sprite(ctx: Any, state: Any, texture: Any, record: dict[str, Any]) -> None:
    """One cell of the atlas, drawn as a sub-rectangle at an integer scale."""
    from imgui_bundle import imgui

    index = poser_mode.cell_index(ctx)
    if index is None:
        widgets.muted("That animation and direction are not on this sheet.")
        return
    columns = int(record.get("columns") or 8)
    size = int(record.get("frame_size") or 0)
    if size < 1:
        size = int(record.get("frame_w") or 0)
    if size < 1:
        widgets.muted("That sheet does not describe its cell size.")
        return

    width, height = texture.size
    column, row = index % columns, index // columns
    uv0 = (column * size / width, row * size / height)
    uv1 = ((column + 1) * size / width, (row + 1) * size / height)

    drawn = size * state.sheet_zoom
    avail = imgui.get_content_region_avail()
    imgui.dummy((0, max((avail.y - drawn) * 0.5, 0)))
    imgui.dummy((max((avail.x - drawn) * 0.5, 0), 0))
    imgui.same_line()
    origin = imgui.get_cursor_screen_pos()
    low = (origin.x, origin.y)
    high = (origin.x + drawn, origin.y + drawn)
    draw_list = imgui.get_window_draw_list()
    if state.sheet_checker:
        widgets.checkerboard(draw_list, low, high)
    imgui.image(widgets.texture_ref(texture), (drawn, drawn), uv0, uv1)
    if state.sheet_show_pivot:
        _pivot_mark(draw_list, low, poser_mode.pivot_of(record, index), state.sheet_zoom)

    io = imgui.get_io()
    if imgui.is_item_hovered() and io.mouse_wheel:
        state.sheet_zoom = max(1, min(int(state.sheet_zoom + io.mouse_wheel), 32))


PIVOT_ARM = 3.0


def _pivot_mark(draw_list: Any, low, pivot, zoom: int) -> None:
    """A cross and a ring where the engine will place this sprite's origin."""
    from imgui_bundle import imgui

    if pivot is None:
        return
    at = (low[0] + pivot[0] * zoom, low[1] + pivot[1] * zoom)
    colour = imgui.get_color_u32(theme.rgba(theme.ACCENT))
    arm = max(PIVOT_ARM * zoom, sp(4))
    draw_list.add_line((at[0] - arm, at[1]), (at[0] + arm, at[1]), colour, sp(1))
    draw_list.add_line((at[0], at[1] - arm), (at[0], at[1] + arm), colour, sp(1))
    draw_list.add_circle(at, arm * 0.6, colour, 12)


# --- the right sidebar: what the sheet is, and the ways out ------------------


def draw_info(ctx: Any) -> None:
    """The selected sheet's own facts and the way to render another,
    followed by the ways out. Troupe's ``sheets.py`` and ``bridge.py``
    panes, merged into one column now that they share a sidebar with the
    pose controls rather than each other.
    """
    from imgui_bundle import imgui

    state = poser_mode.ensure(ctx)
    widgets.section("Sheet")
    manual_render.help_button(ctx, "poser-sheet-info")

    record = poser_mode.active_sheet(ctx)
    if record is None:
        widgets.muted_wrapped("Pick a character sheet on the left.")
        return

    columns = int(record.get("columns") or 0)
    rows = int(record.get("rows") or 0)
    size = int(record.get("frame_size") or 0)
    widgets.muted(f"{columns} x {rows} cells at {size} px")
    tags = (record.get("animation") or {}).get("tags") or []
    widgets.muted(f"{len(tags)} tagged runs")
    line = camera_line(record)
    if line:
        widgets.muted(line)
    _repair(ctx, record)
    _character(ctx, record)

    for line in _pixel_report_lines(_pixel_report(ctx, state)):
        widgets.muted(line)

    imgui.dummy((0, sp(tokens.SP_2)))
    _rerender(ctx, state)

    imgui.dummy((0, sp(tokens.SP_2)))
    _bridge(ctx, state)


def camera_line(record: dict[str, Any]) -> str:
    """What this sheet was framed from, in one line. Empty when it does not
    say."""
    camera = record.get("camera")
    if not isinstance(camera, dict):
        elevation = record.get("elevation")
        return "" if elevation is None else f"framed at {float(elevation):g} degrees"
    labels = {key: label for key, label, _angle in charsheet.CAMERA_PRESETS}
    preset = str(camera.get("preset") or "")
    name = labels.get(preset, preset) or "custom"
    parts = [f"{name}, {float(camera.get('elevation') or 0.0):g} degrees"]
    if "front_yaw" in camera:
        parts.append(f"front at {float(camera['front_yaw']):g} degrees")
    projection = str(camera.get("projection") or "")
    if projection:
        parts.append(projection)
    pixel = camera.get("pixel_size")
    if pixel:
        parts.append(f"{int(pixel)} px sprite")
    return " -- ".join(parts)


def _repair(ctx: Any, record: dict[str, Any]) -> None:
    """The structural verdict, and it is **not** the QA heatmap."""
    if not poser_mode.needs_repair(record):
        return
    widgets.pill(f"{icons.TRIANGLE_ALERT} Needs repair", theme.WARN)
    widgets.muted_wrapped(
        "Structural check: cells are missing, empty or cut off at the frame "
        "edge. This is not the QA heatmap, which ranks the drawing and never "
        "refuses anything -- re-render the runs below to fix these."
    )
    for note in poser_mode.repair_notes(record):
        widgets.muted_wrapped(f"- {note}")


def _character(ctx: Any, record: dict[str, Any]) -> None:
    """Who this is, and the way to make another one like them."""
    from imgui_bundle import imgui

    block = record.get("character")
    if not isinstance(block, dict):
        return
    recipe = poser_mode.recipe_of(record)
    family = str(block.get("family") or recipe.get("family") or "")
    if not family:
        return
    version = block.get("family_version")
    line = family if version is None else f"{family} v{int(version)}"
    theme_name = str(recipe.get("theme") or "")
    if theme_name:
        line += f", {theme_name}"
    widgets.muted(line)
    seed = recipe.get("seed")
    if seed is not None:
        widgets.muted(f"seed {int(seed)}")
    if not recipe:
        return
    imgui.dummy((0, sp(tokens.SP_1)))
    if widgets.disabled_button(
        "Vary in Create",
        True,
        (-1, 0),
        tooltip="Loads this character's recipe into Create as your own "
        "settings, so a new seed, a wider palette or a longer horn makes the "
        "next one. Editing the prompt will not undo them.",
    ):
        poser_mode.vary_in_create(ctx, record)


def _pixel_report_lines(report: dict[str, Any]) -> list[str]:
    """The muted lines a pixel-art report earns, worded for what it measured."""
    if not report:
        return []
    if report.get("style") == "hd":
        lines = ["HD -- full colour"]
        if report.get("exact_stride") is False:
            lines.append("frame size is not an exact stride of the render")
        return lines
    palette = report.get("palette_name") or report.get("palette") or ""
    lines = [f"{report.get('colors', '?')} colours ({palette})"]
    if report.get("orphans"):
        lines.append(f"{report['orphans']} stray pixels cleaned")
    return lines


def _pixel_report(ctx: Any, state: Any) -> dict[str, Any]:
    """What the pixel-art pass measured about *this* atlas."""
    import time

    now = time.monotonic()
    if (
        state.pixel_report_cache is not None
        and state.pixel_report_key == state.sheet_id
        and now < state.pixel_report_next
    ):
        return state.pixel_report_cache
    found: dict[str, Any] = {}
    for row in ctx.svc.store.list(limit=poser_mode.SCAN_LIMIT, kind="charsheet"):
        params = row.get("params") or {}
        if params.get("sheet_id") == state.sheet_id:
            found = dict(params.get("pixel_report") or {})
            break
    state.pixel_report_cache = found
    state.pixel_report_key = state.sheet_id
    state.pixel_report_next = now + poser_mode.SHEETS_REFRESH
    return found


def _rerender(ctx: Any, state: Any) -> None:
    """Re-render some of this sheet's runs, keeping the rest."""
    from imgui_bundle import imgui

    runs = poser_mode.sheet_runs(ctx)
    if not runs:
        return
    if not widgets.header("Re-render some runs", default_open=False):
        return
    chosen = ctx.state.preview.setdefault(poser_mode.RERENDER_SLOT, set())
    by_animation: dict[str, list[dict[str, str]]] = {}
    for run in runs:
        by_animation.setdefault(run["animation"], []).append(run)

    for animation, entries in by_animation.items():
        if not imgui.tree_node(f"{animation}##poser-rerender-{animation}"):
            continue
        for run in entries:
            token = f"{run['animation']}/{run['direction']}"
            _clicked, on = controls.checkbox(
                f"{run['direction']}##poser-rr-{token}", token in chosen
            )
            if on:
                chosen.add(token)
            else:
                chosen.discard(token)
        imgui.tree_pop()

    subset = [
        {"animation": token.split("/", 1)[0], "direction": token.split("/", 1)[1]}
        for token in sorted(chosen)
    ]
    key = f"troupe-sheet:{state.job_id}"
    busy = ctx.busy(key)
    everything = len(subset) == len(runs)
    if widgets.disabled_button(
        f"Re-render {len(subset)} run(s)",
        bool(subset) and not everything and not busy,
        (-1, 0),
        reason=(
            "A sheet is already being queued for this character."
            if busy
            else "That is every run -- use Build another sheet instead."
            if everything
            else "Tick the runs to re-render."
        ),
    ) and poser_mode.rerender_runs(ctx, subset):
        chosen.clear()
    widgets.cost_note(
        "Renders only the ticked runs and copies the rest from this sheet, at "
        "this sheet's own settings. The result is a new sheet -- open it in "
        "Inker with Merge re-render to keep hand edits."
    )


def _bridge(ctx: Any, state: Any) -> None:
    """The ways out -- Troupe's own ``bridge.py``, ported whole."""
    from imgui_bundle import imgui

    widgets.section("Take it somewhere")
    manual_render.help_button(ctx, "poser-sheet-bridge")

    ready = bool(state.job_id and state.sheet_id)
    if widgets.disabled_button(
        verbs.open_in("inker"),
        ready,
        (-1, 0),
        reason="Pick a character sheet first.",
        tooltip="Opens the sheet sliced on its own grid, with one tag per "
        "animation and direction. It opens unlinked: the first Ctrl+S is a "
        "Save As, so cleaning up frames cannot overwrite the render they came "
        "from.",
    ):
        poser_mode.open_in_inker(ctx)
    imgui.dummy((0, sp(tokens.SP_1)))
    if widgets.disabled_button(
        verbs.add_to("packwright"),
        ready,
        (-1, 0),
        reason="Pick a character sheet first.",
        tooltip="One sprite per cell, packed beside everything else in the "
        "atlas.",
    ):
        poser_mode.add_to_packwright(ctx)
    imgui.dummy((0, sp(tokens.SP_1)))
    busy = ctx.busy(poser_mode.export_key(state.job_id, state.sheet_id))
    if widgets.disabled_button(
        "Export package...",
        ready and not busy,
        (-1, 0),
        reason=(
            "That sheet is already being exported."
            if busy
            else "Pick a character sheet first."
        ),
        tooltip="Copies the PNG and its JSON sidecar together -- the pair an "
        "engine imports. Asks where to put them unless an export folder is "
        "configured.",
    ):
        poser_mode.export_package(ctx)

    imgui.dummy((0, sp(tokens.SP_1)))
    frames_busy = ctx.busy(poser_mode.frames_key(state.job_id, state.sheet_id))
    if widgets.disabled_button(
        "Export frames...",
        ready and not frames_busy,
        (-1, 0),
        reason=(
            "That sheet is already being exported."
            if frames_busy
            else "Pick a character sheet first."
        ),
        tooltip="Writes one PNG per frame into clip and compass-direction "
        "folders (N, NE, E...) with a manifest.json of frame rates and "
        "loops. Asks where to put them unless an export folder is "
        "configured.",
    ):
        poser_mode.export_frames(ctx)

    imgui.dummy((0, sp(tokens.SP_2)))
    widgets.muted_wrapped(
        "The sheet and its sidecar are already on disk beside the rig. The "
        "Library's export list is where the files themselves are."
    )
