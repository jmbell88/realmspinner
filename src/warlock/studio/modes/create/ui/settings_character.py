"""The Character type's own column in Create. **Not part of ``settings_2d``.**

``settings_2d`` is "everything that composes the SDXL prompt, and Generate", and
a character composes none of it: no checkpoint, no LoRA, no negative prompt, no
conditioning image, no history of prompts that were sent to a text encoder. Put
here rather than as a sixth branch inside that 2600-line module because the two
have nothing in common except the column they are drawn in and the button they
are refused by -- ``_plan_footer`` stays shared, and ``settings_2d.draw`` calls
:func:`draw_block` instead of the Recipe section. ``_reset_row`` was the other
shared half until 2026-09-07, when Reset moved onto Create's command bar; the
bar draws it for both arms, so a character form still has its way back and this
column simply no longer owns it.

**What a recipe means** -- the plan, the validation, the kwargs, the option
lists -- lives in ``modes/create/engine/character.py`` now (2026-09-18
restructure, P5): this module is the drawing and the orchestration of a press
over that engine.

**The prompt fills the form; the form is never the prompt's prisoner.** Typing
in the command bar re-resolves the brief and writes every field the user has not
touched (``character_engine.sync_from_prompt``). Touching a control records it
in ``character_overrides``, and from then on that control is the user's -- a
prompt edit stops writing it. "Reset to prompt" empties the list. Without the
override list the two directions fight: either the prompt cannot fill anything
after the first frame, or every keystroke silently undoes a species the user
just picked.

**Nothing here ever substitutes a species.** ``characters.resolve`` refuses to,
by construction (``Resolution.family`` is ``None`` for a creature we do not
make), and this pane keeps that promise at the other end: ``character_family``
stays ``""``, Generate is refused in ``resolve.offer_sentence``'s exact words,
and the substitution is offered as a *button the user presses* -- see
:func:`preflight_fix`. A form that quietly filled in "wyvern" for the word
"dragon" would produce a character nobody asked for, which is the whole failure
that sentence exists to prevent.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .... import controls, forms, problems, widgets
from ....manual import render as manual_render
from ..engine import character as character_engine

#: Where the toast for an accepted press waits until the door answers.
#: Composed at submit time rather than at landing time because that is where
#: the form and the registry are both to hand -- ``create_character`` returns
#: two ids and a kind, which is all a door should have to know about a sentence.
TOAST_SLOT = "character_toast"


# --- the block ----------------------------------------------------------------


def draw_block(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """The whole Character column, inside ``settings_2d``'s form and child."""
    from . import settings_2d

    character_engine.sync_from_prompt(form)
    opts = character_engine.options(ctx)
    widgets.section("Character")
    manual_render.help_button(ctx, "settings-character")
    _family(ctx, form, form_ui, opts)
    _theme(ctx, form, form_ui, opts)
    _camera(ctx, form, form_ui, opts)
    _actions(ctx, form, form_ui)
    _pixels(ctx, form, form_ui, opts)
    _appearance(ctx, form, form_ui, opts)
    settings_2d._seed_row(ctx, form, form_ui)
    _name(ctx, form, form_ui)
    _unrecognised(form)
    _footer(ctx, form)


def _family(ctx: Any, form: dict[str, Any], form_ui: forms.Form, opts: dict[str, Any]) -> None:
    current = str(form.get("character_family") or "")
    changed, picked = form_ui.combo(
        "character_family",
        "Species",
        current,
        character_engine.family_options(opts, current),
        help_text=(
            "What to build. Grouped by body plan, because the plan decides the "
            "skeleton, the clips and which sliders this character has."
        ),
    )
    if changed and picked != current:
        form["character_family"] = picked
        character_engine.touched(form, "character_family")
        # The sliders and the look belong to the species that had them.
        form["character_body"] = "{}"
        if not character_engine.theme_offered(
            opts, picked, str(form.get("character_theme") or "")
        ):
            form["character_theme"] = character_engine.THEME_UNSET
        ctx.state.clear_field_error("character_family")


def _theme(ctx: Any, form: dict[str, Any], form_ui: forms.Form, opts: dict[str, Any]) -> None:
    family = str(form.get("character_family") or "")
    choices = character_engine.theme_options(opts, family)
    if len(choices) < 2:
        # No species chosen, or one that paints a single look: a combo with one
        # entry is a control that answers every click with the answer it had.
        return
    current = str(form.get("character_theme") or character_engine.THEME_UNSET)
    changed, picked = form_ui.combo(
        "character_theme",
        "Look",
        current,
        choices,
        help_text="The palette this species is painted in.",
    )
    if changed:
        form["character_theme"] = picked
        character_engine.touched(form, "character_theme")
        ctx.state.clear_field_error("character_theme")


def _camera(ctx: Any, form: dict[str, Any], form_ui: forms.Form, opts: dict[str, Any]) -> None:
    current = character_engine.camera_of(form, opts)
    changed, picked = form_ui.combo(
        "character_camera",
        "Camera",
        current,
        character_engine.camera_options(opts),
        help_text="Where the eye is while the frames are rendered.",
        helper=character_engine.camera_helper(opts, current),
    )
    if changed:
        form["character_camera"] = picked
        character_engine.touched(form, "character_camera")
        ctx.state.clear_field_error("character_camera")


def _actions(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """One switch per movement, each carrying the frames it costs."""
    form_ui.note("character_actions")
    live = set(character_engine.actions_of(form))
    directions = character_engine.DIRECTIONS
    for name, frames in character_engine.MOVEMENTS:
        changed, on = form_ui.switch(
            f"character_action_{name}",
            name.title(),
            name in live,
            help_text=f"{frames} frames, drawn {directions} ways.",
            helper=f"{frames} frames x {directions} = {frames * directions} cells",
        )
        if changed:
            if on:
                live.add(name)
            else:
                live.discard(name)
            form["character_actions"] = ",".join(
                key for key, _f in character_engine.MOVEMENTS if key in live
            )
            character_engine.touched(form, "character_actions")
            ctx.state.clear_field_error("character_actions")
            live = set(character_engine.actions_of(form))
    widgets.muted(f"{character_engine.cell_count(form)} cells")


def _pixels(ctx: Any, form: dict[str, Any], form_ui: forms.Form, opts: dict[str, Any]) -> None:
    """The two ladders the door enforces, read from the door.

    Two calls rather than a loop over a table of two, deliberately: the field id
    is the *address* a refusal is rung on, and
    ``tests/test_field_error_wiring.py`` reads those ids out of this file's
    source. An id built from a loop variable is an address no scan can see, and
    an address no scan can see is how ``retarget_panel`` came to draw a control
    a refusal could never reach.
    """
    changed, picked = form_ui.segmented_choice(
        "character_pixel",
        "Sprite size",
        str(form.get("character_pixel") or ""),
        tuple((str(v), f"{v} px") for v in opts["troupe"]["logical_sizes"]),
        compact=True,
        help_text="How many pixels across one rendered frame is.",
    )
    if changed:
        form["character_pixel"] = picked
        character_engine.touched(form, "character_pixel")
        ctx.state.clear_field_error("character_pixel")
    changed, picked = form_ui.segmented_choice(
        "character_colors",
        "Colours",
        str(form.get("character_colors") or ""),
        tuple((str(v), str(v)) for v in opts["troupe"]["colors"]),
        compact=True,
        help_text="How many colours the finished sheet is reduced to.",
    )
    if changed:
        form["character_colors"] = picked
        character_engine.touched(form, "character_colors")
        ctx.state.clear_field_error("character_colors")


def _appearance(
    ctx: Any, form: dict[str, Any], form_ui: forms.Form, opts: dict[str, Any]
) -> None:
    """One slider per channel the *species' archetype* declares.

    Never a fixed column of sliders: the channel set belongs to the body plan,
    so a wolf has none of an ogre's and a form that drew a fixed group would
    offer four controls of which three are refusals.
    """
    channels = character_engine.channels_of(form, opts)
    if not channels:
        return
    # Above the sliders rather than rung onto one of them: the door refuses the
    # ``appearance`` *block*, and a ring on an arbitrary channel would point at
    # the wrong control. ``troupe_settings``' handling of ``layout``, which is
    # the same shape of address.
    form_ui.note("character_body")
    body = character_engine.body_of(form, opts)
    for channel in channels:
        key = str(channel["key"])
        changed, value = form_ui.slider(
            f"character_body_{key}",
            str(channel["label"]),
            float(body.get(key, channel["default"])),
            float(channel["lo"]),
            float(channel["hi"]),
            fmt="%.2f",
        )
        if changed:
            character_engine.set_channel(form, key, value)
            character_engine.touched(form, "character_body")
            ctx.state.clear_field_error("character_body")


def _name(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    from .....characters import recipe as recipe_mod

    changed, text = form_ui.text(
        "character_name",
        "Name",
        str(form.get("character_name") or ""),
        hint="optional",
        max_length=recipe_mod.MAX_NAME,
        helper="Shown in the library. The species' name is used when this is blank.",
    )
    if changed:
        form["character_name"] = text
        character_engine.touched(form, "character_name")
        ctx.state.clear_field_error("character_name")


def _unrecognised(form: dict[str, Any]) -> None:
    """What the brief said that this form did nothing with.

    Said out loud rather than dropped: a user who typed "a fire ogre with a
    greataxe" is owed the fact that the axe was not understood, or they will
    look for it on the sheet.
    """
    words = character_engine.resolution_of(form).unrecognised
    if words:
        widgets.muted_wrapped("Not interpreted: " + ", ".join(words))


def _footer(ctx: Any, form: dict[str, Any]) -> None:
    """The two ghost buttons under the block, on one row when both are drawn.

    ``same_line`` is placed by whoever drew first rather than at the top of the
    second control, because a bare continuation on a frame where the first was
    skipped attaches the button to whatever the block happened to end with --
    the bug that once orphaned "Recent prompts" against a strength slider.
    """
    if character_engine.overrides_of(form) and controls.button(
        "Reset to prompt##character-reset", role=controls.ButtonRole.GHOST
    ):
        character_engine.reset_to_prompt(form)
        ctx.state.clear_field_errors()
    # Read *after* the press, not before: Reset empties the list, so the row it
    # was on has one button left on the frame it is pressed and the
    # continuation must not reach for a control that is no longer there.
    drew = bool(character_engine.overrides_of(form))
    if not str(form.get("character_family") or ""):
        return
    if drew:
        imgui.same_line()
    # **Never gated on the form's problems**, and it has its own task key: a
    # preview is a look at the body, which is exactly what a user with a
    # refused brief wants while deciding what to change -- and a preview that
    # shared ``"submit"`` would let a dragged slider swallow a press.
    busy = ctx.busy(character_engine.PREVIEW_KEY)
    if widgets.disabled_button(
        "Preview character##character-preview",
        not busy,
        reason="Still building the last preview.",
    ):
        character_engine.preview(ctx, form)


def submit(ctx: Any, form: dict[str, Any]) -> bool:
    """Build the character. -> whether the press was taken.

    On the shared ``"submit"`` key, the same one every other Create output
    uses: it is one form and one Generate button, so two submits in flight from
    it is exactly what that key exists to prevent.
    """
    from . import settings_2d

    opts = character_engine.options(ctx)
    kwargs = character_engine.recipe_kwargs(form, opts)
    prompt = str(form.get("prompt") or "")
    resolution = character_engine.resolution_of(form).to_dict()
    name = str(form.get("character_name") or "").strip() or None

    def run():
        from .....service import characters as svc_characters

        return svc_characters.create_character(
            ctx.svc, kwargs, name=name, prompt=prompt, resolution=resolution
        )

    taken = settings_2d.submit_job(ctx, run)
    if taken:
        ctx.state.preview[TOAST_SLOT] = character_engine.toast_for(form, opts)
    return taken


# --- the repairs offered under a refusal --------------------------------------


def preflight_fix(ctx: Any, form: dict[str, Any], problem: problems.Problem) -> bool:
    """The one-click ways out of a character refusal. -> whether it drew any.

    Called first from ``settings_2d._preflight_fix``, which owns the SDXL arm's
    repairs and knows nothing about a species.
    """
    field = getattr(problem, "field", "")
    message = str(problem)
    if field == "prompt" and message.startswith("Warlock has no "):
        _offer_fixes(ctx, form)
        return True
    if "needs Blender" in message:
        if controls.button(
            "Open dependency packs##character-blender", role=controls.ButtonRole.GHOST
        ):
            from ....panes import app_settings
            from ....state import set_mode

            ctx.state.preview[app_settings.CATEGORY_SLOT] = "packs"
            set_mode(ctx.state, "settings")
        return True
    return False


def _offer_fixes(ctx: Any, form: dict[str, Any]) -> None:
    """Three real ways forward from "we do not make that", and no fourth.

    **The substitution is one of them, and it is a press.** Applying the offer
    is the *only* place in this program where a species the user did not name
    becomes the species that is built, and it happens because somebody read the
    sentence and clicked the button that repeats it. Nothing on any automatic
    path may do this -- see the module docstring.

    The other two keep the brief and change the deliverable, which is why both
    leave ``form["prompt"]`` alone: a user who came here for a manticore still
    wants a manticore, and the question is only which surface can draw one.
    """
    opts = character_engine.options(ctx)
    resolution = character_engine.resolution_of(form)
    offer = resolution.offer[0] if resolution.offer else ""
    row = next((f for f in opts["families"] if f["key"] == offer), None)
    if row is not None and controls.button(
        f"Make it a {row['label'].lower()}##character-offer",
        role=controls.ButtonRole.GHOST,
    ):
        character_engine.apply_offer(form, opts)
        ctx.state.clear_field_error("prompt")
    if controls.button(
        "Sprite sheet (experimental)##character-sprite", role=controls.ButtonRole.GHOST
    ):
        character_engine.switch_to_sprite_sheet(form)
        ctx.state.clear_field_errors()
    if controls.button("Draw it in Troupe##character-troupe", role=controls.ButtonRole.GHOST):
        hand_to_troupe(ctx, form)


def hand_to_troupe(ctx: Any, form: dict[str, Any]) -> None:
    """The third route: a generated reference and a reconstruction.

    The brief goes into ``troupe_mode.form`` -- *the* form that mode's pane
    draws, not a copy -- and then the mode opens. The prompt here is left
    exactly as it was, ``switch_to_sprite_sheet``'s rule and its reason.

    Not in the engine: it imports ``troupe_mode``, a sibling mode's UI module,
    which an engine module may never do.
    """
    from .... import troupe_mode
    from ....state import set_mode

    troupe_mode.form(ctx)["prompt"] = str(form.get("prompt") or "")
    set_mode(ctx.state, "troupe")
