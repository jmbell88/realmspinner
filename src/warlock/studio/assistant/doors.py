"""Familiar T8's two doors, acted out: where a message can send the user, and
how a message becomes a brief sitting in Create.

**Why this lives at studio level, not inside ``studio/familiar/``** --
exactly ``studio/assistant/ui.py``'s own reason (see that module's
docstring): :func:`destinations`/:func:`navigate` reach ``palette``
(``ctx.state``-reading command list), ``app_settings`` (the Settings
category slot) and ``state.set_mode``, and :func:`draft_in_create` reaches
``create_assets``/``create_stages``/``settings_character`` -- all studio
machinery ``studio/familiar/`` is pinned never to import even lazily
(``tests/familiar/test_familiar_imports.py``). The pure prompt/schema/parse
logic these two skills need is :mod:`.familiar.doors` instead; this module
is the other half, the one that actually does something once the model has
answered.

**Both doors act only through the app's own registries, never around
them.** :func:`navigate` never runs a palette command the palette itself
would refuse (``command.enabled(ctx)`` is checked here exactly as
``panes/palette.py``'s own ``_run`` checks it) and never opens a Settings
category through anything but the same ``state.preview``/``set_mode`` pair
``model_gate.request_install`` already uses. :func:`draft_in_create` never
fills the brief without asking ``model_gate.mode_gate`` first, the same
gate the rail greys Create with, and it never calls ``settings_2d.generate``
or anything else that would submit -- it fills the form and moves the
stage, exactly what a person clicking through Create's own controls would
have done, and stops one press short of Generate. A caller cannot reach
anywhere through these two functions that a person could not already click
their own way to, and cannot bypass a gate a person would hit doing it by
hand.

T7's character-plan card reuses :func:`draft_in_create` (``character_fields``
exists for exactly that caller) rather than writing its own copy of "fill
the brief, sync the legacy fields, enter Reference, never submit".
"""

from __future__ import annotations

from typing import Any

from ...familiar.doors import Destination

#: Palette command keys :func:`destinations` treats as places to send
#: someone, beyond the derived ``go:<mode>``/``tour:<key>`` rows. Every
#: other palette command (Save, Undo, reroll, wireframe, ...) is an action on
#: whatever is already in front of the user, not a destination -- offering
#: it here would answer "take me there" with a command that has nowhere to
#: take anyone.
_NAV_COMMAND_KEYS = frozenset({"manual", "shortcuts", "workspace-layout", "show-trash"})


def destinations(ctx: Any) -> list[Destination]:
    """Every place ``navigate`` may offer the model, read off the app's own
    registries -- ``palette.commands(ctx)`` (already computed against
    *this* session's gates: a gated mode's ``go:<key>`` row is still listed,
    carrying its own ``why``, the same "listed but greyed" contract the
    palette itself keeps) filtered to the navigation-shaped keys, plus one
    row per ``app_settings.CATEGORIES`` entry.

    Frame-thread only, like :func:`~.familiar_ui._capture_scene`:
    ``palette.commands`` reads ``ctx.state`` (the mode gate, the active
    document, the viewport) to decide which rows exist and how they read
    *this* frame, so the list handed to the model has to be built before the
    request leaves for the worker thread, not reconstructed once the reply
    comes back.
    """
    from .. import palette
    from ..modes.settings.ui.panes import app_settings

    out: list[Destination] = []
    for command in palette.commands(ctx):
        if (
            command.key.startswith("go:")
            or command.key.startswith("tour:")
            or command.key in _NAV_COMMAND_KEYS
        ):
            out.append(Destination(key=command.key, label=command.label))
    for key, label in app_settings.CATEGORIES:
        title = label.split(" ", 1)[-1]
        out.append(Destination(key=f"settings:{key}", label=f"Settings → {title}"))
    return out


def navigate(ctx: Any, key: str) -> str:
    """Act *key* out, on the frame thread. -> the sentence the transcript
    should show.

    *key* is expected to be one :func:`destinations` itself just offered the
    router; an unrecognised one (the document changed, a download finished
    and ungated a mode, the list this call sees is not the list the router
    saw a moment ago) is answered honestly rather than guessed at. A
    ``settings:<category>`` key opens that category exactly the way
    ``model_gate.request_install`` already does; any other key is a palette
    command, run only when the palette itself would run it -- ``why`` (a
    gated mode's own ``model_gate.mode_reason``, or whatever else the
    command carries) is returned unrun otherwise, never bypassed.
    """
    from .. import palette
    from ..modes.settings.ui.panes import app_settings
    from ..state import set_mode

    by_key = {d.key: d.label for d in destinations(ctx)}
    if key not in by_key:
        return "I'm not sure where that is."

    if key.startswith("settings:"):
        category = key.split(":", 1)[1]
        ctx.state.preview[app_settings.CATEGORY_SLOT] = category
        set_mode(ctx.state, "settings")
        return f"Opened {by_key[key]}."

    command = next(c for c in palette.commands(ctx) if c.key == key)
    if not command.enabled(ctx):
        return command.why
    command.run(ctx)
    return f"Opened {command.label}."


def draft_in_create(
    ctx: Any,
    asset_type: str,
    prompt: str,
    *,
    character_fields: dict[str, Any] | None = None,
) -> str:
    """Fill Create's brief with *asset_type*/*prompt* and stand at the
    Reference stage. -> the sentence the transcript should show. Never
    submits -- the door stops exactly where a person's own typing would
    still need a press of Generate.

    Refused, with the form left untouched, exactly when Create's own door
    would refuse: ``model_gate.mode_gate`` is the same gate the rail greys
    Create with and the palette's own "Go to Create" row checks
    (``palette._mode_commands``), so a Familiar-drafted request never opens
    a form the click path already knows cannot generate anything yet.

    *character_fields* exists for T7's character-plan card: applied the way
    ``troupe_mode.vary_in_create`` applies a recipe's own fields -- written,
    then marked ``character_engine.touched`` so the next prompt edit
    (``sync_from_prompt``) leaves them alone rather than silently
    overwriting what the card just proposed.
    """
    from ..modes.create.engine import assets as create_assets
    from ..modes.create.engine import character as character_engine
    from ..modes.create.ui import stages as create_stages
    from ..panes import model_gate

    where, _blocked = model_gate.mode_gate(ctx, "create")
    if where:
        return model_gate.mode_reason(ctx, "create")

    form = ctx.state.form_2d
    form["asset_type"] = asset_type
    form["generation_type"] = asset_type
    form["prompt"] = prompt
    if asset_type == "character":
        character_engine.sync_from_prompt(form)
        for field_key, value in (character_fields or {}).items():
            form[field_key] = value
            character_engine.touched(form, field_key)
    create_assets.sync_legacy_fields(form)
    create_stages.go(ctx, "reference")
    return "Drafted in Create -- check the brief and press Generate."
