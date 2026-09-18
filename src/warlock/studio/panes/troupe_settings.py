"""Troupe's left-bottom pane: the form that starts a character.

It submits the *first* link of the chain and nothing else: one cheap pose
reference. The gate is the whole point of the shape -- the user approves the
drawing in Create, and only then is the reconstruction spent -- so this pane
deliberately has no "and then build everything" button. What it does have is the
sheet's options, because they ride along on the reference and are validated at
its door: a bad palette found an hour later, on a row the worker minted, would
be a refusal the user never submitted.
"""

from __future__ import annotations

from typing import Any

from ...kernels import charsheet
from ...kernels.rig import store
from .. import forms, tokens, troupe_mode, verbs, widgets
from ..manual import render as manual_render
from ..tokens import sp


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    widgets.section("New character")
    manual_render.help_button(ctx, "troupe-settings")

    options = troupe_mode.options(ctx)
    form = troupe_mode.form(ctx)
    # ``errors``/``on_edit``: the chain this form starts refuses by name --
    # ``prompt``, ``pose``, ``variant``, ``palette``, ``outline``,
    # ``reduce_mode`` -- and each is a field here, so the refusal had an
    # address and nothing at the other end of it. See ``main._collect_tasks``.
    with forms.Form(
        "troupe-settings",
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
        # Beside the build, because the two together are the guide the
        # reference is drawn against and neither means much alone.
        _changed, form["pose"] = form_ui.combo(
            "pose",
            "Reference pose",
            form["pose"],
            [
                (name, troupe_mode.POSE_LABELS.get(name, name))
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
    _submit(ctx, form)
    imgui.dummy((0, sp(tokens.SP_2)))
    _existing_mesh(ctx, form)


#: Where the picker's current choice lives. On ``state.preview`` and not on
#: ``TroupeState``: the mode holds a *selection* (which character and which
#: sheet are on screen) and this is neither -- see :func:`_existing_mesh`.
_PICK_SLOT = "troupe_send_mesh"


def _existing_mesh(ctx: Any, form: dict[str, Any]) -> None:
    """Send a mesh you already have, using the settings above.

    Collapsed and *below* the form, sharing it rather than repeating it: the
    layout, size and palette controls above are visibly the settings this will
    use, which is how ``_rebuild`` already words the same relationship. Two
    competing forms in one 300 px column would be the pane asking the same
    questions twice.

    **It never calls** ``troupe_mode.select``, and that is the trap it is
    written around: ``select`` accepts any job id, and ``sheets()`` returns []
    for a bare mesh -- so pointing the mode at one lands on the blank arrival
    ``open_sheet``'s False return exists to prevent. The picker holds a local
    choice and the button calls the same ``send_to_troupe`` the library item
    calls, so nothing points the mode at anything and "Troupe holds a
    selection" stays intact.
    """
    from imgui_bundle import imgui

    if not widgets.header("Or use a mesh you already have", default_open=False):
        return
    meshes = troupe_mode.sendable_meshes(ctx)
    if not meshes:
        widgets.muted_wrapped("No finished meshes yet. Anything with a mesh can come in here.")
        return
    current = str(ctx.state.preview.get(_PICK_SLOT) or "")
    if current not in {mesh["id"] for mesh in meshes}:
        current = meshes[0]["id"]
    options = [(mesh["id"], _mesh_label(mesh)) for mesh in meshes]
    picked = widgets.labeled_combo("Mesh", current, options)
    if picked != current:
        ctx.state.preview[_PICK_SLOT] = picked
        current = picked
    chosen = next((mesh for mesh in meshes if mesh["id"] == current), None)
    busy = ctx.busy(f"troupe-send:{current}")
    if busy:
        widgets.busy("Sending")
    # The pane's commit verb, so it is the primary -- the rank the same verb
    # has in Clay's and Packwright's bridges.
    if widgets.primary_button(verbs.send_to("troupe"), (-1, 0), enabled=not busy):
        troupe_mode.send_to_troupe(ctx, chosen, form)
    widgets.cost_note(
        "A mesh that is not rigged is rigged first, as a humanoid. Then the "
        f"{cell_count(form)} cells above are rendered."
    )
    imgui.dummy((0, sp(tokens.SP_1)))


def _mesh_label(mesh: dict[str, Any]) -> str:
    """A name a person can pick from, never an empty row."""
    text = str(mesh.get("prompt") or "").strip()
    if not text:
        return str(mesh["id"])[:8]
    return text if len(text) <= 48 else f"{text[:47]}..."


# ``_options`` and ``_form`` moved to ``troupe_mode`` on 2026-09-05, when they
# stopped having one caller: Create's Character arm offers "Draw it in Troupe"
# as the escape route from a species the registry does not model, and that
# route has to put the brief into *this* form -- so the construction of the
# request had to live where both callers can reach it. They are
# ``troupe_mode.options`` and ``troupe_mode.form``, unchanged line for line.


def _layout(form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]) -> None:
    """Per-movement frame and direction controls; the total is always derived.

    Every row comes from ``clip_vocabulary[<the rig's template>]`` -- the
    rig's whole clip library, open past the closed legacy five -- rather than
    a second, hand-written list: see
    ``dev/measurements/2026-09-12-troupe-open-clip-vocabulary.md``. A row
    whose clip is ``provisional`` says so, in both the short muted note under
    its switch and the tooltip beside its name, because a row offered with no
    such mark reads as an animator's finished pass.

    **The vocabulary is timed to the rig this table was actually built for**,
    read off ``layout["template"]`` rather than the door's own default: a
    quadruped, bird or blob bound in Troupe used to have its rows read against
    the default (usually humanoid) vocabulary regardless of its own clip
    library, which offers clips the rig lacks and hides ones it has.
    ``troupe_mode._default_layout`` is what stamps ``template`` on the table
    in the first place and rebuilds it when the bound rig changes; a layout
    that predates that stamp (an old session's saved form) falls back to the
    door's default here exactly as it always answered.
    """

    layout = form["layout"]
    # ``check_troupe`` refuses the composed sheet with ``field="layout"``, and
    # the layout is not one control -- it is this whole table. So the message
    # goes above it rather than being rung onto an arbitrary row: the reader
    # needs to be pointed at the *set* of switches and counts that add up to
    # the refusal. ``widgets.field_error`` is the same helper the single-control
    # case uses, which keeps the wording and the colour identical.
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
    """Pixel art or HD -- what render this sheet's cells come out as.

    The two ``pixel_art`` states ``service.troupe._check_options`` already
    validates (D5), given a name and a control: HD disables rather than hides
    the four controls a pixel-art render has and an HD one does not, so a
    control that is off says why rather than simply not being there.
    """
    _changed, style = form_ui.combo(
        "style",
        "Style",
        troupe_mode._style_choice(form),
        [
            (troupe_mode.STYLE_PIXEL_ART, "Pixel art"),
            (troupe_mode.STYLE_HD, "HD"),
        ],
        help_text=(
            "Pixel art reduces the render to a logical size, a colour budget "
            "and an outline pass. HD keeps the render as painted, with no "
            "colour budget."
        ),
    )
    form["style"] = style


def _frame_rate(form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]) -> None:
    """A layout-wide rate, or every clip's own recorded speed.

    "Authored" sends no ``fps`` at all, which is what keeps a form that never
    touches this control byte-identical to one built before it existed. A
    chosen rate rides the request's ``layout`` block, not a field of its own
    -- ``charsheet.resolve_layout`` reads it from there, and only on a v3
    payload, which is why choosing one is what moves ``troupe_mode
    ._layout_request``'s ``"version"`` from 2 to 3.
    """
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


def cell_count(form: dict[str, Any]) -> int:
    return sum(
        int(row.get("frames") or 0) * int(row.get("directions") or 0)
        for row in (form.get("layout") or {}).get("movements") or ()
        if row.get("enabled", True)
    )


def _size(form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]) -> None:
    # **Above the sprite size, and that order is the argument for it.** The
    # camera is a statement about the *render* -- where the eye is while the
    # 512px frames are made -- and the size is a statement about the sprite
    # those frames become. Reading the render's questions before the sprite's
    # is the order the pipeline actually runs in.
    #
    # The presets come from ``troupe_options``, which reads
    # ``charsheet.CAMERA_PRESETS``: the form must never carry a second copy of
    # an angle, or it offers a framing the renderer does not use.
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
    hd = troupe_mode._style_choice(form) == troupe_mode.STYLE_HD
    _changed, outline = form_ui.combo(
        "outline",
        "Outline",
        form["outline"],
        [(m, m) for m in options.get("outline_modes") or ()],
        enabled=not hd,
        reason="Style is HD, so there is no outline pass." if hd else "",
    )
    form["outline"] = outline
    # Beside the size, because it is a statement about the same act: how the
    # 512px render becomes a sprite of that size. Both modes were validated and
    # tested from the day the mode shipped and neither was ever askable, so
    # ``point`` -- the crisp, every-Nth-sample answer -- was a real code path
    # reachable only by editing a job row.
    _changed, reduce_mode = form_ui.combo(
        "reduce_mode",
        "Reduction",
        form["reduce_mode"],
        [(m, m) for m in options.get("reduce_modes") or ()],
    )
    form["reduce_mode"] = reduce_mode


#: The combo's sentinel for "type your own number" -- ``troupe_send._CUSTOM``'s
#: value, kept identical though the two files do not share the constant,
#: because neither reads the other's combo state and there is nothing to keep
#: in sync by importing it.
_CUSTOM = "custom"


def _logical_size(form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]) -> None:
    """The sprite size row: the ladder, plus a custom box (task G).

    ``form["logical_size_custom"]`` is UI-only state -- ``troupe_mode.start``
    builds the submitted ``troupe`` block field by field and never reads it --
    kept on the form rather than as a local so it survives the pane closing
    and reopening the way every other field here does.
    """
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
        # Same field name as the combo above, so a refusal naming
        # ``logical_size`` (``check_troupe``'s field) rings whichever of the
        # two controls is actually on screen -- the two are never drawn on
        # the same frame, so there is no id collision to worry about.
        _changed, value = form_ui.number(
            "logical_size", "Custom size (px)", int(form["logical_size"])
        )
        form["logical_size"] = max(int(lo), min(int(hi), int(value)))
        if form["logical_size"] and charsheet.RENDER_SIZE % form["logical_size"] != 0:
            widgets.muted_wrapped(
                "Sizes that don't divide 512 are resized with nearest-neighbour."
            )


def _camera_helper(presets: dict[str, Any], key: str) -> str:
    """The chosen preset's angle, in words. Empty when there is nothing to say.

    The number rather than a description, because the number is the thing that
    transfers: a user matching Troupe sprites to a Plotter map already knows
    what elevation that map is drawn at, and a preset's name does not answer
    that question while its angle in degrees does.
    """
    entry = presets.get(key) or {}
    if "elevation" not in entry:
        return ""
    return f"{float(entry['elevation']):g} degrees above the horizon"


def _palette(
    ctx: Any, form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]
) -> None:
    """A designed palette if one is installed, a colour budget otherwise.

    Two controls rather than one because they are two different answers: a
    named palette is the artist's decision and the budget is the machine's --
    a median cut over the atlas, which is the fallback and says so.
    """
    hd = troupe_mode._style_choice(form) == troupe_mode.STYLE_HD
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
    # **Last, and optional.** A sheet has always been able to carry a name --
    # the door validates it, the worker writes it into the sidecar and the
    # chooser reads it back -- and there was no field, so every sheet a
    # character had was "sheet - 32px". Two builds at one size were two
    # identical rows in a list that only appears once there are two.
    _changed, form["name"] = form_ui.text(
        "name",
        "Name this sheet",
        str(form.get("name") or ""),
        hint="optional",
        max_length=store.MAX_SHEET_NAME,
        helper="Shown in the sheet chooser. The size and cell count are added for you.",
    )


def _submit(ctx: Any, form: dict[str, Any]) -> None:

    busy = ctx.busy("troupe-start")
    count = cell_count(form)
    # The 2026-09-07 audit (troupe-05) found these ladders restated as bare
    # numbers -- ``charsheet.MAX_CELLS``/``WARN_CELLS`` are the door's own
    # limits, and a pane that quotes them from memory is a pane that goes
    # stale the day the door's number moves.
    ready = bool(form["prompt"].strip()) and 0 < count <= charsheet.MAX_CELLS
    if busy:
        widgets.busy("Drawing the reference")
    if widgets.disabled_button(
        "Draw the reference",
        not busy and ready,
        (-1, 0),
        reason=(
            f"Describe the character and select a layout of at most "
            f"{charsheet.MAX_CELLS} cells."
        )
        if not ready
        else "A reference is already being queued.",
    ):
        troupe_mode.start_character(ctx, form)
    widgets.cost_note(
        f"One image, and then it stops. After approval, the mesh, rig, and "
        f"{count} rendered cells follow."
    )
    if count > charsheet.WARN_CELLS:
        widgets.muted(
            f"Large sheet: over {charsheet.WARN_CELLS} cells can take substantially longer."
        )
