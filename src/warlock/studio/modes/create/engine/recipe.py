"""What a 2D recipe means: the plan, the validation, the kwargs, the notes.

Split out of ``modes/create/ui/settings_2d.py`` (2026-09-18 restructure, P5)
-- the imgui-free half of that ~3,200-line module. Familiar
(``familiar_ui.py``, ``familiar_doors.py``), Review (``review_panes.py``) and
Troupe read this vocabulary today by reaching into a *pane*, through comments
that say they may not import it outright (``service/sprites.py``,
``service/tilesheets.py``); they import this module directly now, since it
draws nothing. What stays in ``modes/create/ui/settings_2d.py`` is the
drawing and the orchestration of a press (``draw``, the field callbacks,
``_preflight_fix``'s buttons, and ``generate``, which toasts and rings
fields).

Renamed on the way in, because a name a sibling module now calls may not keep
a leading underscore: ``_resolved_recipe`` -> :func:`resolved_recipe`,
``_is_character`` -> :func:`is_character`, ``_is_tile_arm`` ->
:func:`is_tile_arm`, ``_view_of`` -> :func:`view_of`, ``_tile_options`` ->
:func:`tile_options`, ``_sprite_options`` -> :func:`sprite_options`,
``_verify_reference_path`` -> :func:`verify_reference_path`,
``_conditioning_tail`` -> :func:`conditioning_tail`, ``_safe_int`` ->
:func:`safe_int`, ``_negative_supported`` -> :func:`negative_supported`,
``_sprite_cost`` -> :func:`sprite_cost`, ``_findings_hint`` ->
:func:`findings_hint`, and the ``_LOAD_FINDINGS`` sentinel -> :data:`LOAD_FINDINGS`.
Helpers with no caller outside this module (``_base_labels``, ``_lora_labels``,
``_lora_base``, ``_layout_problems``, ``_sprite_logical``, ``_sheet_options``,
``_WEIGHT_FIELDS``) keep their underscore.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..... import generation, vectors
from ..... import guidance as guidancelib
from ..... import models as modelslib
from .....bench import findings as findings_lib
from .....pipelines import tileatlas as tileatlaslib
from .....service import sprites as svc_sprites
from .....service import tilesheets as svc_tilesheets
from .....service.validation import MAX_PROMPT, MAX_REFERENCE_COUNT
from .... import problems as problem_types
from . import assets as create_assets
from . import character as character_engine

#: The sentinel default for a ``findings_doc`` parameter, so a caller that
#: passes nothing gets the *current* ``findings.load()`` read rather than a
#: value frozen at import time -- and so :func:`findings_hint` and the UI's
#: own findings-aware controls can tell "the caller supplied a doc" from "the
#: caller supplied nothing" without a second parameter. Shared identity is
#: the point: ``modes/create/ui/settings_2d.py`` compares against this same
#: object, never a second ``object()`` of its own.
LOAD_FINDINGS = object()


FIELD_LABELS: dict[str, str] = {
    "lora_weight": "Style strength",
    "negative_prompt": "Negative prompt",
    "ip_scale": "IP-Adapter scale",
    "control_scale": "ControlNet scale",
    "control_end": "ControlNet end",
    "reference_prep": "Reference prep",
    "bg_removal": "Background removal",
    "resolution": "Resolution",
    "profile": "Profile",
    "custom_triangles": "Custom triangles",
    "size_m": "Size in metres",
}

def field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " "))

def resolved_recipe(ctx: Any, form: dict[str, Any]) -> Any:
    """What automatic routing would load for this form, or None.

    Wrapped because it runs on the frame thread from three note helpers: a
    partially restored form must make the pane say nothing rather than raise
    inside the draw, which is ``negative_supported``'s standing rule here.

    Memoised on the *request*, not on frame or form identity (2026-09-08
    audit, finding create-06, and the orchestrator's ruling on it: a
    fingerprint feeding a provenance record may not be memoised on wall
    time, only on content or a generation counter). ``request_from_legacy``
    is cheap and pure, so it is rebuilt every call; only
    ``generation.resolve_recipe`` -- which fingerprints every installed
    checkpoint directory (``provenance.file_fingerprint`` via ``_checksum``)
    -- is worth skipping, and only while the request compares equal to the
    one the cached answer was resolved from and ``ctx.svc.config`` is the
    same object. The result stays correct across many unchanged frames and
    resolves again the moment either input actually changes, which a
    frame-keyed cache could not promise. Set with ``setattr`` rather than a
    declared ``AppState`` field: this module does not own ``state.py``, and a
    plain dataclass instance takes an extra attribute without one. A caller
    with no ``ctx.state`` at all (several note helpers are exercised
    headlessly against a bare ``SimpleNamespace``) gets the pre-fix
    behaviour instead of an ``AttributeError``: resolve every call, memoise
    nothing.
    """
    try:
        request = generation.request_from_legacy(form)
        config = ctx.svc.config
    except Exception:
        # Silent on purpose, and the same choice ``negative_supported`` makes
        # for the same reason: this runs sixty times a second inside the draw,
        # so a partially restored form must make the pane say *nothing* rather
        # than log a line per frame or raise through the frame loop. The
        # service remains the final compatibility gate, and it is not silent.
        # ``ctx.svc`` is read here, inside the same guard, for the same
        # reason: some note helpers are exercised headlessly against a
        # ``SimpleNamespace`` that carries no ``svc`` at all.
        return None

    # ``getattr`` rather than ``ctx.state``: the frame thread's ``ctx`` always
    # carries an ``AppState`` to hang the memo on, but several note helpers
    # (``recipe_structure_note`` and friends, exercised headlessly by
    # tests/test_settings_2d_notes.py and tests/test_generation_tiers.py) call
    # this with a bare ``SimpleNamespace(svc=..., guidance=...)`` that has no
    # ``.state`` at all. A headless caller with nothing to memoise onto just
    # gets the pre-fix behaviour -- resolve every call -- rather than an
    # ``AttributeError``.
    state = getattr(ctx, "state", None)
    cache = getattr(state, "_resolved_recipe_cache", None) if state is not None else None
    if cache is not None:
        cached_request, cached_config_id, cached_resolved = cache
        if cached_config_id == id(config) and cached_request == request:
            return cached_resolved

    try:
        resolved = generation.resolve_recipe(request, config)
    except Exception:
        # A form that resolves to a broken request must say nothing once,
        # not raise every frame -- caching the ``None`` here is what makes
        # that once rather than sixty times a second.
        resolved = None
    if state is not None:
        state._resolved_recipe_cache = (request, id(config), resolved)
    return resolved

def clear_for_tier(ctx: Any, form: dict[str, Any]) -> list[str]:
    """Drop the selections the newly chosen tier cannot run.

    -> one sentence per selection cleared, for the pane to show.

    :func:`clear_unusable`'s argument applied to the other end of the same
    routing. Under automatic routing the *tier* picks the checkpoint, so
    switching to Fast strands a ControlNet and an Avoid text exactly the way
    switching the base model under Advanced does -- and both of those controls
    are hidden rather than merely disabled once the tier cannot use them, which
    would leave Generate refusing on a field that is off screen. Clearing is
    the same choice ``clear_unusable`` makes and for the same reason: a
    refusal the user cannot act on is a dead end.

    Called only on a change of tier, never per frame -- see ``clear_unusable``.
    """
    cleared: list[str] = []
    if str(form.get("model_mode") or "auto") == "advanced":
        # The tier does not choose the checkpoint here, so it cannot strand
        # anything; ``clear_unusable`` owns that half.
        return cleared
    resolved = resolved_recipe(ctx, form)
    if resolved is None:
        # Either the form does not compile or this host qualifies no recipe;
        # the Recipe combo already says so, and clearing selections on the
        # strength of an answer nobody has would be the silent rewrite
        # ``clear_unusable`` refuses to do.
        return cleared
    # Cannot raise: ``resolved_recipe`` just built the same request and got a
    # recipe out of it.
    caps = generation.capability_controls(generation.request_from_legacy(form), resolved)
    if form.get("control") and not caps["controlnet"]:
        form["control"] = ""
        cleared.append(
            "The structure control was cleared: this recipe runs at guidance 0 "
            "and cannot run a ControlNet."
        )
    if str(form.get("negative_prompt") or "").strip() and not caps["negative_prompt"]:
        form["negative_prompt"] = ""
        cleared.append(
            "The Avoid text was cleared: this recipe runs at guidance 0, where "
            "a negative prompt has no effect."
        )
    return cleared

def clear_for_layout(form: dict[str, Any]) -> list[str]:
    """Drop the geometry the newly chosen layout cannot draw.

    -> one sentence per value moved, for the pane to show.

    :func:`clear_unusable`'s rule applied to the tile arm, and for its reason: a
    control that ``validate`` refuses while offering only legal values is a dead
    end unless the illegal value is cleared, and the two things a seamless layout
    cannot keep -- a 48 px tile and a view that does not wrap -- are both
    persisted, so both survive a switch of layout.

    Called only when the layout changes, never per frame. A form *restored* with
    a size the layout refuses keeps it: ``validate`` names it above Generate, the
    control offers the sizes that work, and rewriting a stored value on the way
    in would change a request nobody touched.
    """
    if not is_seamless(form):
        return []
    options = tile_options()
    cleared: list[str] = []
    sizes = tile_sizes_for(form)
    if str(form.get("tile_size") or "") not in {str(size) for size in sizes}:
        form["tile_size"] = str(options["defaults"]["tile_size"])
        cleared.append(
            f"The tile size moved to {form['tile_size']} px: a seamless material "
            f"is reduced from one {tileatlaslib.MATERIAL_PX} px frame, and only "
            f"{sizes} divide it exactly."
        )
    views = views_for(form)
    if view_of(form) not in views:
        form["projection"] = views[0]
        label = options["view_labels"].get(views[0], views[0])
        cleared.append(
            f"The view moved to {label}: a seamless material wraps a square, and "
            f"neither an isometric diamond nor a 3/4 tile's visible front face is "
            f"one."
        )
    return cleared

LEGACY_KEY_PREFIX = "legacy:"

LEGACY_LAYOUT_LABELS: dict[str, str] = {
    "turnaround": "Turnaround (still views)",
    "walk": "Walk (legacy, 4 frames)",
}

def sprite_action_key(layout: str) -> str:
    """The Action combo's key for a stored ``sheet_layout``. See
    :data:`LEGACY_KEY_PREFIX`."""
    mode, action, _directions = generation.sprite_from_layout(layout)
    if mode in generation.SPRITE_LEGACY_MODES:
        return f"{LEGACY_KEY_PREFIX}{mode}"
    if layout not in generation.SPRITE_SHEET_KINDS:
        # A kind from some other build: named as itself, so the combo can show
        # what the form is actually set to rather than moving it.
        return layout
    return action

def sprite_action_options(
    options: dict[str, Any], current: str
) -> tuple[tuple[str, str], ...]:
    """The Action combo's entries: the turnaround, then what has a guide.

    An action is offered **only if its pose guide is on this disk**, which is
    ``sprite_options``' own filter and the whole reason that key exists: the
    guide is what decides where the limbs go, so an action offered without one
    is a control whose result is eight bands of an unposed character -- or, once
    the doors refuse it, a control whose only outcome is that refusal.

    A stored layout the menu does not carry is appended rather than dropped,
    which is :func:`palette_options`' rule and its reason: silently moving a
    form off the thing it says it is set to is how a user comes to submit
    something they did not choose. That covers both the legacy ``walk`` -- a
    real sheet this build still draws -- and a kind from some other build, which
    is not.

    ``current`` is a :func:`sprite_action_key`, not a layout.
    """
    out = [(f"{LEGACY_KEY_PREFIX}turnaround", LEGACY_LAYOUT_LABELS["turnaround"])]
    out.extend((entry["key"], entry["label"]) for entry in options["actions"])
    if current not in {key for key, _label in out}:
        bare = current.removeprefix(LEGACY_KEY_PREFIX)
        out.append((current, LEGACY_LAYOUT_LABELS.get(bare, f"{bare} (unavailable)")))
    return tuple(out)

def sprite_action_entry(options: dict[str, Any], action: str) -> dict[str, Any] | None:
    """``sprite_options()['actions']``' row for ``action``, or None for a
    turnaround or a legacy kind -- neither of which is one."""
    for entry in options["actions"]:
        if entry["key"] == action:
            return entry
    return None

def sprite_layout_for(
    options: dict[str, Any], action: str, directions: int
) -> str:
    """The ``sheet_layout`` an Action/Directions pair names.

    Takes a :func:`sprite_action_key`, so a prefixed legacy entry resolves to the
    kind it names and the Directions control has nothing to say about it.

    Falls back to the action's *first available* direction count rather than to
    the asked-for one, because the two controls move independently: picking an
    action that has no eight-direction guide while the Directions control still
    says eight must land on a sheet that exists.
    """
    if action.startswith(LEGACY_KEY_PREFIX):
        return action.removeprefix(LEGACY_KEY_PREFIX)
    entry = sprite_action_entry(options, action)
    if entry is None:
        return action
    counts = [row["count"] for row in entry["directions"]]
    if not counts:
        return action
    return f"{action}{directions if directions in counts else counts[0]}"

def _sprite_logical(form: dict[str, Any], sizes: tuple[int, ...]) -> int:
    """The cell size this form will actually be submitted at.

    Clamped **here** rather than only in the picker, and that is the difference
    between a gate and a decoration: the Action control is always on screen and
    the size picker is inside Advanced, so a user who picks an eight-frame walk
    without ever opening Advanced would otherwise compile a 64px request that
    the door refuses -- a press that does nothing, decided by a section they
    never looked at.
    """
    if not sizes:
        return safe_int(form.get("cell_size"), 64)
    asked = safe_int(form.get("cell_size"), max(sizes))
    return asked if asked in sizes else max(sizes)

def sprite_plan(form: dict[str, Any]) -> dict[str, Any]:
    """What this form's sprite arm will actually draw, arithmetic included.

    One function, read by the Dimensions section's summary line, by the size
    picker's ladder and by :func:`sprite_sheet_kwargs`, so what the user is told
    and what is submitted are the same numbers rather than two calculations of
    them. Every one of them comes from ``sprite_options()`` -- the door's own --
    for the reason the tile arm's do: a pane that recomputes a cell count is a
    label that goes stale the first time a frame count moves.
    """
    options = sprite_options()
    layout = str(form.get("sheet_layout") or "turnaround")
    _mode, action, directions = generation.sprite_from_layout(layout)
    entry = sprite_action_entry(options, action)
    row = None
    if entry is not None and layout not in generation.SPRITE_LEGACY_MODES:
        row = next(
            (r for r in entry["directions"] if r["count"] == directions), None
        )
    if row is None:
        # A turnaround or a legacy walk: one generation of one fixed atlas, and
        # its grid is the ``sheet_types`` table's rather than an action's.
        fixed = next(
            (t for t in options["sheet_types"] if t["key"] == layout),
            options["sheet_types"][0],
        )
        sizes = tuple(fixed["logical_sizes"])
        return {
            "layout": layout,
            "action": "",
            "directions": len(fixed["directions"]),
            "frames": int(fixed["frames_per_direction"]),
            "cells": int(fixed["cells"]),
            "bands": 1,
            "candidates": 2,
            "generations": 2,
            "sizes": sizes,
            "logical_size": _sprite_logical(form, sizes),
        }
    candidates = int(row["candidates"])
    sizes = tuple(entry["logical_sizes"])
    return {
        "layout": layout,
        "action": action,
        "directions": int(row["count"]),
        "frames": int(entry["frames"]),
        "cells": int(row["cells"]),
        "bands": int(row["bands"]),
        "candidates": candidates,
        "generations": int(row["bands"]) * candidates,
        "sizes": sizes,
        "logical_size": _sprite_logical(form, sizes),
    }

def sprite_cost(plan: dict[str, Any]) -> str:
    """The one sentence under the sprite controls, from :func:`sprite_plan`."""
    # The wait comes from the door, not from a second multiplication of
    # ``seconds_per_generation`` here: the sprite panel draws the same sentence
    # about the same press, and two copies of the arithmetic is two promises.
    when = svc_sprites.generation_time_phrase(plan["generations"])
    draft = "one draft" if plan["candidates"] == 1 else f"{plan['candidates']} drafts"
    return (
        f"{plan['directions']} directions x {plan['frames']} frames = "
        f"{plan['cells']} cells, {plan['generations']} generations for "
        f"{draft}, {when}."
    )

def palette_options(installed: list[str], chosen: str) -> tuple[tuple[str, str], ...]:
    """The palette combo's entries: "derive one", what is installed, and a
    selection that is not.

    ``lora_options``' rule, and for its reason. A palette is a file, so a stem
    the form holds can stop existing between two launches -- an external drive,
    a folder tidied -- and the door refuses the submit by that name. Dropping it
    from the list would leave the combo showing its bare stem with no
    explanation, or, in any control that falls back to entry zero, silently
    rewrite the user's choice to "derive one" and change what the sheet looks
    like without saying so. Listed and marked, the thing keeping Generate off is
    the one thing on screen.

    It was shared with the profile editor, which drew the same picker over the
    same directory against a draft rather than the live form; that editor went
    with Profiles, and the helper stayed because the pane is its real caller.
    """
    options = [("", "Derived from the render"), *((name, name) for name in installed)]
    if chosen and chosen not in installed:
        options.append((chosen, f"{chosen} - not in the palette folder"))
    return tuple(options)

OUTLINE_LABELS = {"none": "None", "inner": "Inside", "outer": "Around"}

def findings_hint(
    ctx: Any,
    param: str,
    value: Any,
    doc: Any = LOAD_FINDINGS,
) -> str | None:
    """The sweep's own verdict on this field's current value, or None.

    Read fresh every frame -- ``findings.load`` is mtime-cached, so the common
    case (no bench dir, or an unchanged file) costs one ``stat()`` and never
    blocks the frame loop.

    Scoped to what the user is currently asking for. This pane owns the prompt,
    so it always knows its subject: the hash of the prompt in the form is what
    ``vectors.prompt_hash`` recorded on every verdict and observation, so
    ``hint`` can prefer the evidence about *this* subject and say when it fell
    back to the pooled corpus. Hashing a short string once per control per
    frame is a sha1 over a few dozen bytes, which is nothing beside the
    ``stat()`` above it.
    """
    if doc is LOAD_FINDINGS:
        doc = findings_lib.load(Path(ctx.svc.config.bench_dir) / "findings.json")
    return findings_lib.hint(
        doc,
        param,
        value,
        prompt_hash=vectors.prompt_hash(ctx.state.form_2d.get("prompt")),
    )

SHEET_TYPES: tuple[tuple[str, str], ...] = (
    ("tile", "Tile grid"),
    ("sprite", "Sprite sheet"),
)

_sheet_options: list[Any] = [None, None]

def tile_options() -> dict[str, Any]:
    if _sheet_options[0] is None:
        _sheet_options[0] = svc_tilesheets.tile_sheet_options()
    return _sheet_options[0]

def sprite_options() -> dict[str, Any]:
    if _sheet_options[1] is None:
        _sheet_options[1] = svc_sprites.sprite_options()
    return _sheet_options[1]

def is_tile_arm(form: dict[str, Any]) -> bool:
    """Whether this form is the Sheet output's *tile* arm.

    The expression six places in this file were spelling out, given a name once
    the tile arm grew three layouts of its own: "not a sprite sheet" and "the
    grid layout" stopped being the same sentence, and a local called ``grid``
    that meant the first is exactly how the submit came to compile a materials
    request with no materials in it.
    """
    return form.get("output") == "sheet" and form.get("sheet_type") != "sprite"

def is_character(form: dict[str, Any]) -> bool:
    """Whether this form is the Character type. **The registry, not a field.**

    Asked through ``create_assets.selected`` rather than by testing
    ``output == "character"``, for the reason ``selected`` itself gives: a form
    that one writer touched and another did not can carry a stale ``output``,
    and the *type* is the field the user set.
    """
    return create_assets.selected(form).intent == "character"

def sheet_rows(form: dict[str, Any]) -> tuple[str, ...]:
    """Which registry rows the Sheet output currently needs.

    A function of the form rather than a constant, because the two arms load
    different things and the tile arm's own list depends on whether a reference
    is attached: its IP-Adapter is optional, and a gate that demanded one would
    tell a user with everything the common request uses that they are missing a
    download. Shared with :func:`weights_problem` so the note above the button
    and the gate inside the section cannot disagree.

    **And on the layout**, since the tile arm grew three of them: the grid guide
    *is* a ControlNet and the two seamless layouts never open one, so asking for
    the grid's rows under a materials sheet told a user with everything that
    request uses to download canny weights it will never load. The per-mode maps
    are ``tile_sheet_options``' own, which is where :func:`rows_needed` publishes
    them.
    """
    if form.get("sheet_type") == "sprite":
        return svc_sprites.SPRITE_ROWS
    # ``style_lock`` counts as a reference. The checkbox makes the first
    # material the IP-Adapter reference for every material after it
    # (``tilesheets._check_weights`` folds it into ``rows_needed`` the same
    # way), so a locked sheet loads the adapter with no file attached -- and a
    # gate that only looked at ``ref_path`` let that press reach the door and
    # be refused there for a download this note had said nothing about.
    needs_adapter = bool(form.get("ref_path")) or bool(form.get("style_lock"))
    key = "mode_reference_rows_needed" if needs_adapter else "mode_rows_needed"
    return tuple(tile_options()[key][tile_mode_of(form)])

def tile_mode_of(form: dict[str, Any]) -> str:
    """The tile layout this form is asking for, in the service's own spelling.

    An unrecognised stored value reads as the default rather than as a refusal,
    which is this pane's standing rule for a settings-file value: the field is
    persisted, the menu can change between releases, and a form that resolved to
    nothing would disable Generate over a control whose value the user cannot
    see.

    **The default is not ``grid``.** The door refuses the grid layout unless the
    request explicitly asks for it -- see ``create_tile_sheet``'s ``allow_grid``
    -- precisely so that a default nobody chose can never land on the one layout
    a measurement says does not work.
    """
    stored = str(form.get("tile_mode") or svc_tilesheets.DEFAULT_MODE)
    return stored if stored in svc_tilesheets.TILE_MODES else svc_tilesheets.DEFAULT_MODE

def is_seamless(form: dict[str, Any]) -> bool:
    """Whether this form draws each tile as its own seamless material.

    The two layouts ``pipelines.tileatlas`` builds, asked as one question,
    because everything that differs between them and the grid -- the tile sizes
    that divide a 1024px material, the one view that wraps, what the preview
    composes -- differs the same way for both.
    """
    return tile_mode_of(form) != svc_tilesheets.MODE_GRID

def tile_sizes_for(form: dict[str, Any]) -> list[int]:
    """Which tile sizes this layout can publish.

    Sourced from the service rather than filtered here: a seamless material is
    reduced from one 1024px frame on an exact partition, so 48 is on the grid's
    menu and not on this one -- and the day that frame size changes, this list
    changes with it because ``tile_sheet_options`` derives it by asking
    ``pipelines.tileatlas``.
    """
    options = tile_options()
    return list(options["seamless_tile_sizes" if is_seamless(form) else "tile_sizes"])

def views_for(form: dict[str, Any]) -> list[str]:
    """Which views this layout can draw. One, for the seamless pair.

    ``tileatlas``' own list, for :func:`tile_sizes_for`'s reason: an isometric
    tile is a diamond and a 3/4 tile has a visible front face, so neither wraps,
    and the sentences explaining that live in the pipeline that refuses them.
    """
    options = tile_options()
    return list(options["seamless_views" if is_seamless(form) else "views"])

def material_lines(form: dict[str, Any]) -> tuple[str, ...]:
    """The materials field as the door takes it: one surface per non-blank line.

    Blank lines are dropped rather than counted, which is what makes a trailing
    newline harmless -- the door drops them too, and a form that counted them
    would report a cell total the request will not produce.
    """
    return tuple(
        line
        for line in (raw.strip() for raw in str(form.get("materials") or "").splitlines())
        if line
    )

def view_of(form: dict[str, Any]) -> str:
    """The form's view, in today's spelling.

    The form field is still ``projection`` -- it is a persisted key and a
    control the user has a name for -- and a form saved before the
    vocabulary widened carries ``"orthogonal"``. Read through the service's
    alias table rather than by comparing strings here, so the pane never holds
    a second opinion about what an old value means.
    """
    stored = str(form.get("projection") or svc_tilesheets.DEFAULT_VIEW)
    return svc_tilesheets.LEGACY_VIEWS.get(stored, stored)

def seamless_subject(form: dict[str, Any]) -> str | None:
    """The subject the *first* cell of a seamless layout will be generated from.

    ``None`` when the request does not describe one yet, which is a real answer
    rather than a failure: a materials sheet with no lines and a terrain set
    with no inner surface have no first material, and the honest preview of a
    request that names nothing is no preview at all.

    Composed by ``pipelines.tileatlas`` rather than here -- the style clause both
    layouts append and the context a terrain set shares between its two halves
    are that module's, and a second copy of either would be a preview of a
    sentence nothing sends.
    """
    mode = tile_mode_of(form)
    try:
        if mode == svc_tilesheets.MODE_TERRAIN:
            return tileatlaslib.terrain_subjects(
                str(form.get("inner_terrain") or ""),
                str(form.get("outer_terrain") or ""),
                str(form.get("boundary") or ""),
            )[0]
        lines = material_lines(form)
        return tileatlaslib.material_subject(lines[0], index=0, total=len(lines))
    except (IndexError, ValueError):
        return None

def verify_reference_path(ctx: Any, form: dict[str, Any]) -> None:
    """Clear a restored reference path that no longer names a file.

    The 2026-09-07 Create review, item 5.3a: ``ref_path`` used to be the one
    conditioning field ``settings.VOLATILE`` dropped on every restart, while
    its neighbours ``ip_adapter`` and ``control`` survived -- so a session
    that had conditioned a job reopened with the *conditioning* selections
    back and no reference to apply them to, and Generate refused for a reason
    that named a control the user had not touched this session. ``ref_path``
    now persists like the other two, which trades that defect for a new one a
    plain restore would have: a path that has since moved or been deleted
    would come back as a live-looking value that only fails at the far end of
    a submit, or worse, silently reaches the worker as "no reference" once
    ``generation.request_from_legacy`` starts guarding on existence too.
    Checked once per session, against the filesystem, rather than trusted
    because it round-tripped through JSON.

    A toast rather than a silent drop: the Conditioning header already claims
    a reference is attached until this clears it (``conditioning_tail``), so
    saying nothing here would make a control disappear with no visible cause.
    Once per session rather than every frame it is missing, so relaunching
    with the drive that held it still unmounted does not toast on every visit
    to this pane.
    """
    if ctx.state.create.reference_path_checked:
        return
    ctx.state.create.reference_path_checked = True
    path = str(form.get("ref_path") or "")
    if not path or Path(path).is_file():
        return
    form["ref_path"] = ""
    ctx.toast(f"The reference image is missing and was cleared: {path}", "warn")

def conditioning_tail(form: dict[str, Any]) -> str:
    """" - 2 attached" and the like, on the collapsed Conditioning header.

    A closed disclosure must never hide a setting that is doing something. The
    header says how many of its controls are live, so a reference image left
    attached from a previous run is visible without opening it.
    """
    live = sum(
        1
        for key in ("ref_path", "ip_adapter", "control")
        if str(form.get(key) or "")
    )
    live += 1 if form.get("init_image") else 0
    if not live:
        return ""
    return f"  ({live} on)"

def _base_labels(ctx: Any, keys: list[str]) -> str:
    """The picker's own labels for a set of base-model keys.

    Labels rather than keys: "sdxl_cfg" is not what the combo shows, and a
    message naming something the user cannot find in the list is worse than no
    message at all.
    """
    labels = [label for key, label in (ctx.base_models or []) if key in keys]
    return ", ".join(labels or keys)

def negative_prompt_note(ctx: Any, form: dict[str, Any]) -> str | None:
    """Why the negative prompt is inert here, or None when it is live.

    A distilled base runs at guidance 0, and text2image encodes the negative
    branch only above 1.0 -- so on turbo the field accepted text, stored it in
    params and changed nothing about the image. That silence is the bug; this
    is the sentence that ends it.

    The 2026-09-06 audit, finding create2-04: this note used to test only
    ``form["base_model"]`` directly, while the section's own visibility gate,
    ``negative_supported``, read the *resolved* recipe -- so under Automatic
    routing a stale distilled ``base_model`` left over from a prior Advanced
    selection (``_model``'s ``else`` branch clears ``model_override``, never
    ``base_model``) could disagree with a resolution that now lands on a
    full-CFG tier: the section opened because ``negative_supported`` and
    ``generate()`` both correctly consult the resolved recipe, but this note
    still reported the field dead. Resolve the recipe here too, exactly as
    ``img2img_note`` does for the same shape of staleness, and fall back to
    the raw base check whenever there is nothing to resolve -- a picked base
    with missing weights under Advanced, or a caller (this file's own note
    tests included) that supplies only a bare ``base_model`` with no
    ``ctx.svc`` to resolve against at all -- so the note and the gate can
    never say different things about the same field once a recipe *does*
    resolve.
    """
    resolved = resolved_recipe(ctx, form)
    if resolved is not None:
        request = generation.request_from_legacy(form)
        supported = generation.capability_controls(request, resolved)["negative_prompt"]
    else:
        # No recipe to resolve: fall back to the raw base check, the pre-fix
        # rule. ``ctx.guidance`` is read only down this path -- a caller that
        # can resolve a recipe need not carry a catalog at all, and several of
        # this file's own tests don't.
        bases = ctx.guidance.get("cfg_bases") or []
        supported = (form.get("base_model") or "") in bases
    if supported:
        return None
    bases = ctx.guidance.get("cfg_bases") or []
    return (
        "This model runs at guidance 0, so the negative prompt has no effect. "
        f"It does on: {_base_labels(ctx, bases)}."
    )

def _lora_labels(ctx: Any, keys: list[str]) -> str:
    """The picker's own labels for a set of style-LoRA keys.

    _base_labels' argument applied to the other combo: a message naming
    "pixelklein" points at something the user cannot find in the list.
    """
    labels = [label for key, label in (ctx.style_loras or []) if key in keys]
    return ", ".join(labels or keys)

def _lora_base(ctx: Any, form: dict[str, Any]) -> str:
    """The base model the Style LoRA picker should judge fit against.

    The 2026-09-14 audit, finding create-03: ``lora_note``/``lora_options``
    (and ``lora_filter_note``) used to read the raw, possibly stale
    ``form["base_model"]`` directly -- under Automatic routing that field is
    not what actually runs (``_model``'s Automatic branch never writes it),
    so the picker could label a selection fitted for a base that
    ``generation.validate_request`` then refuses at submit with a
    ``CompatibilityIssue(field='style_lora')``. Resolve the recipe first,
    exactly as ``img2img_note`` and ``negative_prompt_note`` (2026-09-05 and
    2026-09-06 audits) already do for the same shape of staleness, and fall
    back to the raw base only when nothing resolves -- a picked base with
    missing weights under Advanced, or a caller (this file's own note tests
    included) that supplies only a bare ``base_model`` with no ``ctx.svc`` to
    resolve against at all.
    """
    resolved = resolved_recipe(ctx, form)
    if resolved is not None:
        return resolved.base_model
    return form.get("base_model") or ""

def lora_note(ctx: Any, form: dict[str, Any]) -> str | None:
    """Why the style LoRA picker is inert here, or None when it is live.

    The narrow question -- whether *any* adapter in the registry is fitted to
    this architecture -- which is the only case where the control has nothing
    at all to do. An adapter names one architecture's modules, so a mismatch is
    not a weak effect but a refusal: the service rejects the submit outright
    rather than generating without it.
    """
    bases = ctx.guidance.get("lora_bases") or []
    if _lora_base(ctx, form) in bases:
        return None
    return (
        "No style LoRA in the registry is fitted to this model's architecture. "
        f"These models can use one: {_base_labels(ctx, bases)}."
    )

def lora_options(ctx: Any, form: dict[str, Any]) -> list[tuple[str, str]]:
    """The style-LoRA combo's entries for the chosen base.

    Those fitted to it, plus whatever the form already holds, marked. Keeping a
    stale selection listed is load-bearing rather than tidy: widgets.combo
    falls back to index 0 for a value it cannot find, so dropping it would draw
    "no style LoRA" over a selection ``validate`` is refusing -- making the
    value that keeps Generate off the one control the user cannot see. That is
    exactly the dead end ``clear_unusable`` exists to prevent, arriving by
    another door. The marking mirrors what main.py puts on a base whose weights
    are missing.
    """
    fitting = (ctx.guidance.get("loras_by_base") or {}).get(_lora_base(ctx, form)) or []
    options: list[tuple[str, str]] = []
    for key, label in ctx.style_loras or []:
        if key in fitting:
            options.append((key, label))
        elif key and key == (form.get("style_lora") or ""):
            options.append((key, f"{label} - not fitted to this model"))
    return options

def lora_filter_note(ctx: Any, form: dict[str, Any]) -> str | None:
    """Why the picker lists fewer styles than the registry holds, or None.

    A second function rather than a branch inside ``lora_note`` for
    ``tile_bases``' reason: that one explains a control that cannot act at all,
    this one a control acting on less than the whole list. Folded together,
    one sentence comes to say both things under a disabled combo.
    """
    by_base = ctx.guidance.get("loras_by_base") or {}
    fitting = by_base.get(_lora_base(ctx, form)) or []
    if not fitting:
        # lora_note owns this case; saying it twice is the fold above.
        return None
    everything = [key for key, _ in (ctx.style_loras or [])]
    if len(fitting) >= len(everything):
        return None
    return (
        "A style LoRA is fitted to one architecture, so this model is offered "
        f"only: {_lora_labels(ctx, fitting)}."
    )

def recipe_structure_note(ctx: Any, form: dict[str, Any]) -> str | None:
    """Why *automatic* routing's recipe cannot run a ControlNet, or None.

    :func:`structure_note`'s sibling for the other half of the Recipe control.
    That one answers for the checkpoint the user picked under Advanced; this
    one answers for the checkpoint the tier picks on their behalf, which is not
    in ``form["base_model"]`` at all. Without it the Structure picker was drawn
    under Fast, the selection was submitted, and the refusal came back from
    ``guidance.normalize`` naming ``base_model`` -- a combo automatic routing
    does not display.
    """
    if str(form.get("model_mode") or "auto") == "advanced":
        return None
    resolved = resolved_recipe(ctx, form)
    if resolved is None:
        # The pane already says "no compatible installed recipe" under the
        # Recipe combo; saying it again here names the wrong subject.
        return None
    spec = modelslib.BASE_MODELS.get(resolved.base_model)
    if spec is None or spec.controlnet:
        return None
    # The 2026-09-07 Create review, item 5.5.1: this used to say "Switch the
    # Recipe to Quality", a control that has not existed since the Fast/Quality
    # tier was folded into the Model combo (``model_options``'s own comment).
    # The only remedy left is the same one ``structure_note`` gives for the
    # advanced case -- pick a full-CFG checkpoint from that combo -- so this
    # says that instead of naming a control nobody can find.
    return (
        f"{resolved.recipe.label} runs at guidance 0 and cannot run a "
        "ControlNet. Pick a full-CFG model above to run one."
    )

def structure_note(ctx: Any, form: dict[str, Any]) -> str | None:
    """Which bases could run the ControlNet this one cannot, or None."""
    bases = ctx.guidance.get("controlnet_bases") or []
    if (form.get("base_model") or "") in bases:
        return None
    # The 2026-09-07 Create review, item 5.5.1: "under Advanced" named a
    # disclosure the 2026-08-17 taxonomy retirement flattened away -- the
    # Model combo this points at is drawn earlier in this same column, not
    # behind a fold, which is the wording ``model_options``'s own "no
    # compatible installed recipe" note already uses ("pick one above").
    return (
        "Structure control needs a full-CFG model -- pick one of "
        f"{_base_labels(ctx, bases)} above."
    )

def img2img_note(ctx: Any, form: dict[str, Any]) -> str | None:
    """Why "Start from this image" is inert here, or None when it is live.

    The 2026-09-05 audit, finding create-04: only the SDXL family's img2img
    path accepts a start image -- the same fact ``guidance.normalize``
    refuses on, late, at the queue door. Read through the resolved recipe
    rather than ``form["base_model"]`` directly, the reason
    ``recipe_structure_note`` exists beside ``structure_note``: under
    automatic routing a tier can resolve to FLUX.2 Klein (``image_flux2``)
    with the checkbox still ticked from an earlier SDXL run, and
    ``form["base_model"]`` is not what actually ran. One function rather than
    that pair's split, because the message here does not change with the
    routing mode -- there is no ControlNet-style "under automatic, name the
    tier instead" branch, only "pick an SDXL model" either way.
    """
    resolved = resolved_recipe(ctx, form)
    if resolved is None:
        # The pane already says "no compatible installed recipe" elsewhere;
        # naming that here too would be the wrong subject.
        return None
    request = generation.request_from_legacy(form)
    if generation.capability_controls(request, resolved)["img2img"]:
        return None
    return "This model cannot start from an image; pick an SDXL model to use img2img."

def clear_unusable(ctx: Any, form: dict[str, Any]) -> list[str]:
    """Drop the selections the newly chosen base cannot run.

    -> one sentence per selection cleared, for the pane to show.

    Called *only* when the base model changes, never per frame: a form restored
    with a style picked under another base must keep it until the user changes
    the base, or opening the pane silently rewrites a selection nobody touched
    and does it before the note explaining it can be read.

    Clearing rather than only disabling, because a disabled control that
    ``validate`` refuses is a dead end -- the value keeping Generate off is the
    one thing the user cannot reach, and the only recovery was to guess which
    earlier choice to undo. It applies to exactly the two gates ``validate``
    refuses: the style LoRA, whose picker goes disabled, and the structure
    control, whose whole group ``structure_note`` hides. The negative prompt
    stays in the brief: a distilled recipe cannot consume it, but that is not a
    reason to reject the generation. ``generation.effective_negative_prompt``
    removes it from the worker payload without erasing the authored text.
    """
    cleared: list[str] = []
    base = form.get("base_model") or ""
    # The *pair*, not the base. Asking "is this base in lora_bases()" was right
    # only while one architecture had adapters and the others had none: with
    # both families covered that test is never true, and the clear would
    # silently stop happening. An unknown stored base resolves to [] and
    # therefore clears, the pane's standing rule for a settings-file value.
    fitting = (ctx.guidance.get("loras_by_base") or {}).get(base) or []
    if form.get("style_lora") and form["style_lora"] not in fitting:
        form["style_lora"] = ""
        # The weight goes back to the default with it: it scales a selection
        # that no longer exists, and a strength left at 0.2 would silently
        # apply to whatever style is picked next.
        form["lora_weight"] = modelslib.DEFAULT_LORA_WEIGHT
        cleared.append(
            "The style LoRA was cleared: it is not fitted to this model's "
            "architecture."
            if fitting
            else "The style LoRA was cleared: this model cannot use one."
        )
    if form.get("control") and base not in (ctx.guidance.get("controlnet_bases") or []):
        # Only the selection, exactly as the Clear-reference button does: the
        # strengths are hidden with it and never submitted without it.
        form["control"] = ""
        cleared.append(
            "The structure control was cleared: this model cannot run a ControlNet."
        )
    # The 2026-09-05 audit, finding create-04: the third gate ``validate``
    # refuses, added beside the two above. Checked directly against the
    # spec's family rather than a models.py bases list -- see
    # ``generation._takes_img2img`` for why ``tile_bases()`` is the wrong
    # reuse here even though it answers the same question today. Left
    # disabled instead of cleared before this fix, "Start from this image"
    # stayed ticked with no explanation across a base change that
    # ``guidance.normalize`` would refuse outright.
    base_spec = modelslib.BASE_MODELS.get(base)
    if (
        form.get("init_image")
        and base_spec is not None
        and base_spec.family != modelslib.FAMILY_SDXL
    ):
        form["init_image"] = False
        form["init_strength"] = None
        cleared.append(
            "The start image was cleared: this model cannot start from an image."
        )
    return cleared

def model_options(ctx: Any) -> list[tuple[str, str]]:
    """The Model combo's entries: Automatic, then every installed checkpoint.

    **One control where there were three.** The pane used to draw a Fast/Quality
    tier, an Automatic/Advanced switch and a checkpoint combo -- three controls
    for one decision, of which the first two only ever chose *which checkpoint*.
    Nothing is lost by folding them: the sole ``fast`` recipe resolves to the
    ``sdxl`` checkpoint, which is in this list, so picking Fast and picking
    ``sdxl`` were the same act said two ways.

    ``""`` is Automatic, which is ``model_mode="auto"``; any other key is
    ``model_mode="advanced"`` with that key as the override. The two form
    fields are unchanged, because the door and the recipe registry read them.
    """
    return [("", "Automatic")] + list(ctx.base_models)

def lora_default_weight(key: str) -> float:
    """The measured strength for one style LoRA, or the flat default.

    ``ctx.style_loras`` is pinned to 2-tuples by the smoke tests and
    ``guidance.catalog()`` does not carry the weight, so the pane reads the
    registry it already imports.
    """
    spec = modelslib.STYLE_LORAS.get(key or "")
    return spec.default_weight if spec is not None else modelslib.DEFAULT_LORA_WEIGHT

def reseed_lora_weight(form: dict[str, Any], was_lora: str) -> None:
    """Put the newly-picked adapter's tuned strength into ``form``.

    One copy of the rule, which is the point of the function.
    Each adapter carries its own measured strength -- pixel-art-klein restores
    an rslora scale of 16, so the flat ``DEFAULT_LORA_WEIGHT`` is ~14x its
    usable band and returns black frames -- and ``guidance.normalize`` only
    applies ``default_weight`` when the caller *omits* the field. Both of these
    forms always send a number, so the seed has to happen at the widget. It
    lives here because the deleted profile editor already had this bug once by
    holding a second copy of the rule, and a third copy would find it again.
    """
    if form["style_lora"] != was_lora:
        form["lora_weight"] = lora_default_weight(form["style_lora"])

def negative_supported(ctx: Any, form: dict[str, Any]) -> bool:
    """Show Avoid only when the resolved recipe will actually consume it.

    The 2026-09-06 audit, finding create2-04: this used to resolve the recipe
    independently of ``negative_prompt_note``, which read only the raw
    ``form["base_model"]`` -- so the two could disagree on a stale field and
    the section would open with a note claiming the opposite. Deriving this
    gate from the same note that draws under it makes that impossible: there
    is exactly one place left that decides whether the negative prompt is
    live.
    """
    try:
        return negative_prompt_note(ctx, form) is None
    except Exception:
        # The service remains the final compatibility gate.  During a partially
        # restored form, hiding an unresolved control is safer than presenting
        # an active field whose text would be silently discarded.
        return False

def problems_for(ctx: Any, form: dict[str, Any]) -> list[problem_types.Problem]:
    """Everything stopping a press, form problems first. Once per frame.

    Cached as ``(frame, id(form)) -> problems`` on ``ctx.state`` rather than
    at module scope: the Reference stage asks the same question twice on every
    frame -- the command bar, to know whether Generate is live, and this
    footer, to list what is wrong -- and both answers have to agree, which one
    evaluation guarantees and two only tend to. Per-ctx because ``id(form)``
    can be reused after GC; a module global keyed on it alone would let a
    second ctx's form read the first ctx's stale verdict.
    """

    key = (int(getattr(ctx.state, "frame_index", 0)), id(form))
    cache = ctx.state.create.problems_cache
    if cache is not None and cache[0] == key:
        return cache[1]
    problems = validate(form, ctx)
    if is_character(form):
        # Appended here rather than inside ``validate`` because they need a
        # ``ctx``: whether Blender exists is a fact about this install, and the
        # species registry is read through the service door. Appended *at all*
        # so the ring, this footer, the disabled Generate's tooltip and the
        # Ctrl+Enter toast are one sentence -- which is the property this
        # function's cache exists to guarantee.
        problems = [*problems, *character_engine.problems(ctx, form)]
    weight = weights_problem(ctx, form)
    if weight is not None:
        problems = [*problems, weight]
    ctx.state.create.problems_cache = (key, problems)
    return problems

OPEN_FORM_WORDS = (
    "awning", "basket", "bellows", "birdcage", "bow", "branch", "branches",
    "bridge", "cage", "chain", "chains", "fence", "gate", "grate", "grating",
    "harp", "lattice", "ladder", "leg", "legs", "mesh", "net", "netting",
    "pane", "panes", "post", "posts", "railing", "rigging", "rope", "sail",
    "scaffold", "spoke", "spokes", "stairs", "string", "strings", "trellis",
    "web", "wheel", "wicker", "wire", "wires",
)

CLOSED_FORM_CLAUSE = "solid closed form, filled-in gaps, no see-through openings"

def open_form_words(prompt: str) -> tuple[str, ...]:
    """The open-form words in ``prompt``, in the order they appear.

    Whole words, lowercased, de-duplicated -- pure, so the wording of the
    advisory and the test that pins it read the same function.
    """
    import re

    seen: list[str] = []
    for word in re.findall(r"[a-z]+", str(prompt or "").lower()):
        if word in OPEN_FORM_WORDS and word not in seen:
            seen.append(word)
    return tuple(seen)

def advisories_for(ctx: Any, form: dict[str, Any]) -> list[problem_types.Advisory]:
    """Everything worth knowing that is **not** stopping the press.

    Deliberately not folded into :func:`problems_for`: that list is documented
    as "everything stopping a press" and every member of it disables Generate.
    An advisory disables nothing, and the separation is what makes it safe to
    say something uncertain.

    Cheap enough to run per frame without ``problems_for``'s cache -- a regex
    over one prompt -- and it takes ``ctx`` anyway so that the next tenant can
    look at the corpus without changing every call site.
    """
    del ctx
    out: list[problem_types.Advisory] = []
    if create_assets.selected(form).key == "3d_model":
        words = open_form_words(str(form.get("prompt") or ""))
        if words:
            named = ", ".join(words[:3])
            out.append(
                problem_types.Advisory(
                    f"\"{named}\" tends to draw an open form -- gaps, slats or thin "
                    "members. Open forms reconstruct usable about 2 times in 5; "
                    "the graded corpus overall runs about 1 in 2. Worth a press "
                    "either way -- this is a risk, not a verdict.",
                    field="prompt",
                )
            )
    return out

def validate(form: dict[str, Any], ctx: Any = None) -> list[problem_types.Problem]:
    """What would be refused, said before the button is pressed.

    A summary rather than a refusal on submit: the API checks all of this too,
    but a disabled button with a reason beats a toast after a round trip.

    Each entry carries the control it is about (:class:`problem_types.Problem`, a
    ``str`` subclass, so the aggregate block and every existing comparison are
    unchanged). The field is what lets the *keyboard* doors -- Ctrl+Enter and
    the palette, which call :func:`generate` directly and never draw that block
    -- put the ring on the control the button path would have pointed at.

    ``ctx`` is optional and new (the 2026-09-15 audit, finding create-03): the
    ControlNet/img2img/style-LoRA checks below compare against
    ``form["base_model"]``, which is correct under Advanced but stale under
    Automatic -- ``_model()``'s switch-to-Automatic branch moves the *notes*
    to the resolved recipe and leaves ``base_model`` holding whatever was
    last picked under Advanced. Without this, a mismatch with what Automatic
    actually loads passed validation here and only surfaced as a toast
    refusal after the round trip through the queue door. Every caller with a
    ``ctx`` (``problems_for``, ``generate``) now passes it; callers that
    cannot (tests exercising the form in isolation) keep the pre-fix
    raw-``base_model`` reading, which is exactly right under Advanced and the
    same approximation as before under Automatic.
    """
    problems: list[problem_types.Problem] = []
    asset_key = form.get("asset_type")
    if asset_key is not None and asset_key not in create_assets.ASSET_TYPES:
        problems.append(problem_types.Problem("Choose a recognised asset type.", "asset_type"))
    prompt = form.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        problems.append(problem_types.Problem("A prompt is required.", "prompt"))
    if isinstance(prompt, str) and len(prompt) > MAX_PROMPT:
        problems.append(
            problem_types.Problem(f"The prompt is over {MAX_PROMPT} characters.", "prompt")
        )
    count = safe_int(form.get("count"), 0)
    if not 1 <= count <= MAX_REFERENCE_COUNT:
        problems.append(
            problem_types.Problem(
                f"References must be between 1 and {MAX_REFERENCE_COUNT}.", "count"
            )
        )
    # The *tile arm*, which is not the same question as the *grid layout* -- see
    # :func:`is_tile_arm`. Every check below that was written when the two were
    # one thing is about the arm, because what makes a tile set exempt from them
    # is that its door pins its own recipe, which all three layouts do.
    tileset = is_tile_arm(form)
    # **The tileset precedent, applied to the second type with its own door.**
    # A character request reaches ``service.characters.create_character``, which
    # reads no checkpoint, no LoRA, no ControlNet and no reference -- so every
    # check below guarded by this flag would be a refusal about somebody else's
    # job, and reachable rather than theoretical: ``control`` is persisted and
    # so is ``ref_path``, but ``verify_reference_path`` clears a ``ref_path``
    # that has since moved or been deleted (the 2026-09-07 Create review, item
    # 5.3a), so a session that once conditioned an Object can still reopen
    # with the pair split.
    pinned = tileset or is_character(form)
    base = form.get("base_model")
    if ctx is not None and str(form.get("model_mode") or "auto") == "auto":
        # See the docstring: under Automatic, the resolved recipe's base is
        # what will actually load, not the stale ``form["base_model"]`` an
        # earlier Advanced pick left behind.
        resolved = resolved_recipe(ctx, form)
        if resolved is not None:
            base = resolved.base_model
    style = form.get("style_lora")
    # A tile set's fixed recipe does not read either selection. It validates
    # its pinned pair at its own service door.
    if not pinned and (not isinstance(base, str) or base not in modelslib.BASE_MODELS):
        problems.append(problem_types.Problem("Choose a recognised image model.", "base_model"))
    if not pinned and (
        not isinstance(style, str) or (style and style not in modelslib.STYLE_LORAS)
    ):
        problems.append(problem_types.Problem("Choose a recognised style LoRA.", "style_lora"))
    if not pinned and style:
        try:
            weight = float(form.get("lora_weight"))
        except (TypeError, ValueError, OverflowError):
            weight = float("nan")
        if not modelslib.LORA_WEIGHT_MIN <= weight <= modelslib.LORA_WEIGHT_MAX:
            problems.append(
                problem_types.Problem(
                    f"Style strength must be between {modelslib.LORA_WEIGHT_MIN:g} "
                    f"and {modelslib.LORA_WEIGHT_MAX:g}.",
                    "style_lora",
                )
            )
    # The tile arm is the one output that does not go through
    # ``create_job``: ``create_tile_sheet`` pins its own base, its own LoRA and
    # its own ControlNet and reads none of the four fields below. So the three
    # checks after this are skipped for it -- not as a tolerance, but because a
    # disabled Generate reading "Conditioning needs a reference image" over a
    # ``control`` the run will never load is a refusal about somebody else's
    # job. It is reachable rather than theoretical: ``control`` and ``ref_path``
    # both persist, but ``verify_reference_path`` clears a ``ref_path`` that
    # has since moved or been deleted (the 2026-09-07 Create review, item
    # 5.3a), so a session that once conditioned an Object can reopen with the
    # pair split. The sprite arm is deliberately *not* exempt -- its first
    # step is an ordinary reference job and reads all four. Both reachable
    # from a restored form rather than from this frame's controls, which is
    # why they are checked here and not only where the widgets are drawn: a
    # persisted ``control``/``ip_adapter`` can outlive the ``ref_path`` that
    # justified it, and the base model can be changed under Advanced after a
    # control was picked.
    if (
        not pinned
        and not form.get("ref_path")
        and (form.get("ip_adapter") or form.get("control"))
    ):
        problems.append(
            problem_types.Problem("Conditioning needs a reference image.", "ref_path")
        )
    # The 2026-09-16 audit, finding create-panes-01: guidance.normalize's
    # ``_number`` refuses ip_scale/control_scale/control_end/init_strength by
    # name (the same shape of check as lora_weight above), but this function
    # never range-checked any of the four before Generate is enabled -- a
    # persisted out-of-range value reached the queue door with the
    # Conditioning section still collapsed and no ring anywhere on this pane
    # to land on. Gated the same way ``submit_kwargs`` gates what it sends:
    # a slider whose selection is unset never reaches params as a live
    # setting, so it is not checked here either.
    if not pinned and form.get("ip_adapter"):
        try:
            ip_scale = float(form.get("ip_scale"))
        except (TypeError, ValueError, OverflowError):
            ip_scale = float("nan")
        if not modelslib.IP_SCALE_MIN <= ip_scale <= modelslib.IP_SCALE_MAX:
            problems.append(
                problem_types.Problem(
                    f"Reference strength must be between {modelslib.IP_SCALE_MIN:g} "
                    f"and {modelslib.IP_SCALE_MAX:g}.",
                    "ip_scale",
                )
            )
    if (
        not pinned
        and form.get("control")
        and base not in modelslib.controlnet_bases()
    ):
        problems.append(
            problem_types.Problem("Structure control needs a full-CFG model.", "base_model")
        )
    if not pinned and form.get("control"):
        try:
            control_scale = float(form.get("control_scale"))
        except (TypeError, ValueError, OverflowError):
            control_scale = float("nan")
        if not modelslib.CONTROL_SCALE_MIN <= control_scale <= modelslib.CONTROL_SCALE_MAX:
            problems.append(
                problem_types.Problem(
                    f"Structure strength must be between "
                    f"{modelslib.CONTROL_SCALE_MIN:g} and {modelslib.CONTROL_SCALE_MAX:g}.",
                    "control_scale",
                )
            )
        try:
            control_end = float(form.get("control_end"))
        except (TypeError, ValueError, OverflowError):
            control_end = float("nan")
        if not modelslib.CONTROL_END_MIN <= control_end <= modelslib.CONTROL_END_MAX:
            problems.append(
                problem_types.Problem(
                    f"Structure 'until' must be between "
                    f"{modelslib.CONTROL_END_MIN:g} and {modelslib.CONTROL_END_MAX:g}.",
                    "control_end",
                )
            )
    # The 2026-09-05 audit, finding create-04: this pane's docstring promises
    # "what would be refused, said before the button is pressed", but nothing
    # here checked img2img against the base's family, so the refusal only
    # arrived from the queue door (``guidance.normalize``) after a round
    # trip. Checked against the spec's family directly rather than a
    # models.py bases list -- see ``generation._takes_img2img``.
    base_spec = modelslib.BASE_MODELS.get(base)
    if (
        not pinned
        and form.get("init_image")
        and base_spec is not None
        and base_spec.family != modelslib.FAMILY_SDXL
    ):
        problems.append(
            problem_types.Problem(
                "This model cannot start from an image; pick an SDXL model.",
                "init_image",
            )
        )
    if not pinned and form.get("init_image") and form.get("ref_path"):
        try:
            init_strength = float(form.get("init_strength") or 0.45)
        except (TypeError, ValueError, OverflowError):
            init_strength = float("nan")
        if not (
            modelslib.IMG2IMG_STRENGTH_MIN <= init_strength <= modelslib.IMG2IMG_STRENGTH_MAX
        ):
            problems.append(
                problem_types.Problem(
                    f"Start strength must be between "
                    f"{modelslib.IMG2IMG_STRENGTH_MIN:g} and "
                    f"{modelslib.IMG2IMG_STRENGTH_MAX:g}.",
                    "init_strength",
                )
            )
    # Reachable the same way: a style picked under one base survives a change
    # of base under Advanced, and the service refuses the submit outright
    # rather than generating without it.
    if not pinned and form.get("style_lora") and form["style_lora"] not in (
        modelslib.loras_by_base().get(base) or []
    ):
        problems.append(
            problem_types.Problem(
                "The style LoRA is not fitted to this model's architecture.",
                "style_lora",
            )
        )
    if form.get("output") == "tile" and base not in modelslib.tile_bases():
        problems.append(
            problem_types.Problem("Seamless tiles need an SDXL model.", "base_model")
        )
    if form.get("output") == "sheet":
        # The tile arm's own two fields, and nothing about the model: what a
        # sheet is short of on this host is a different question, and
        # ``weights_problem`` asks it against the rows a sheet actually loads.
        size = str(form.get("tile_size") or "")
        # Both menus are asked for *this layout*: a seamless material is reduced
        # from one 1024px frame on an exact partition and wraps a square, so 48
        # px and two of the three views are on the grid's menu and not on
        # theirs. The lists come from the service so the pane holds no second
        # opinion about either ceiling.
        sizes = tile_sizes_for(form)
        views = views_for(form)
        if tileset and size not in {str(s) for s in sizes}:
            # Reachable from a restored form rather than from this frame's
            # control: the value is persisted, and the menu it came from can
            # change between releases -- or between layouts.
            problems.append(
                problem_types.Problem(f"Tile size must be one of {sizes}.", "tile_size")
            )
        if tileset and view_of(form) not in views:
            # Interpolated rather than spelled out. The sentence used to name
            # its two values, so the day a third arrived the form would have
            # refused it with a list that did not contain it.
            problems.append(
                problem_types.Problem(f"View must be one of {views}.", "projection")
            )
        if tileset:
            problems.extend(_layout_problems(form))
        if tileset or form.get("sheet_type") == "sprite":
            raw_target = form.get("target_cell_px") or ""
            target = None if raw_target == "" else safe_int(raw_target, -1)
            for issue in generation.validate_target_cell(
                target, isometric=tileset and view_of(form) == "isometric"
            ):
                problems.append(problem_types.Problem(issue.message, issue.field))
    return problems

def _layout_problems(form: dict[str, Any]) -> list[problem_types.Problem]:
    """What the chosen tile layout is still short of.

    The door's own refusals, asked before the request exists. Each one names the
    control it is about, and each ceiling is read from
    ``tile_sheet_options`` rather than written here -- the door enforces them
    against ``asset_workflows.collection_cells``, and a second set of numbers in
    a pane is a form that accepts what the door then refuses.

    The grid layout contributes nothing: everything it needs is the prompt and
    the geometry, both checked above.
    """
    options = tile_options()
    mode = tile_mode_of(form)
    problems: list[problem_types.Problem] = []
    if mode == svc_tilesheets.MODE_MATERIALS:
        lines = material_lines(form)
        variants = safe_int(form.get("variants"), 0)
        if not lines:
            problems.append(
                problem_types.Problem(
                    "A materials sheet is the list of surfaces you type; describe "
                    "at least one, one per line.",
                    "prompt_items",
                )
            )
        if len(lines) > int(options["max_materials"]):
            problems.append(
                problem_types.Problem(
                    f"{len(lines)} materials is past the {options['max_materials']} "
                    f"one sheet can name.",
                    "prompt_items",
                )
            )
        if any(len(line) > MAX_PROMPT for line in lines):
            problems.append(
                problem_types.Problem(
                    f"One material is over {MAX_PROMPT} characters.", "prompt_items"
                )
            )
        if not 1 <= variants <= int(options["max_variants"]):
            problems.append(
                problem_types.Problem(
                    f"Draws of each material must be between 1 and "
                    f"{options['max_variants']}.",
                    "variants",
                )
            )
        elif len(lines) * variants > int(options["max_cells"]):
            problems.append(
                problem_types.Problem(
                    f"{len(lines)} materials by {variants} draws is "
                    f"{len(lines) * variants} cells, past the "
                    f"{options['max_cells']} one sheet can hold; each cell is its "
                    f"own full generation.",
                    "variants",
                )
            )
    elif mode == svc_tilesheets.MODE_TERRAIN:
        # One sentence per empty field, and each names its own control.
        #
        # This loop used to append one *identical* sentence per empty field, so
        # a fresh terrain form -- where both are empty -- stacked the same
        # words twice above Generate, and neither copy said which of the two it
        # was about. A refusal here is meant to be a sentence the user can act
        # on, and "one of the two fields" is not one. The list stays per-field
        # rather than collapsing to a single line because ``refuse`` rings the
        # control each problem names: a merged line could ring only one of them
        # and would leave the other looking accepted.
        for field, name in (("inner_terrain", "Inside"), ("outer_terrain", "Outside")):
            if not str(form.get(field) or "").strip():
                problems.append(
                    problem_types.Problem(
                        f"{name} is empty. A terrain set is two surfaces and both "
                        f"are generated, so both have to be described.",
                        field,
                    )
                )
        for field in ("inner_terrain", "outer_terrain", "boundary"):
            if len(str(form.get(field) or "")) > MAX_PROMPT:
                problems.append(
                    problem_types.Problem(f"That is over {MAX_PROMPT} characters.", field)
                )
    return problems

def safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback

def submit_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The 2D form as create_job takes it.

    ``output`` is this pane's own switch and never "model": this pane is the
    first stage of a two-stage pipeline made visible, and going straight to a
    mesh from here would spend two minutes of GPU on an image nobody has
    approved. A tile has no second stage at all.
    """
    tile = form.get("output") == "tile"
    # Every surviving field is machinery (model identity, conditioning) rather
    # than subject taxonomy, so a tile submits the same set an object does.
    known = set(guidancelib.form_fields())
    fields = {k: v for k, v in form.items() if k in known and v not in ("", None)}
    sprite_sheet = (
        sprite_sheet_kwargs(form)
        if form.get("output") == "sheet" and form.get("sheet_type") == "sprite"
        else None
    )
    return {
        "kind": "text",
        "prompt": form["prompt"].strip(),
        "output": "tile" if tile else "reference",
        "count": safe_int(form.get("count"), 1),
        **({"sprite_sheet": sprite_sheet} if sprite_sheet is not None else {}),
        "seed": int(form["seed"]),
        # Verbatim, never ``or None``: an emptied box is the user asking for no
        # negative prompt, which is a different request from the default.
        "negative_prompt": str(form["negative_prompt"]),
        "lora_weight": float(form["lora_weight"]) if form.get("style_lora") else None,
        # Mirroring lora_weight: sent only alongside the selection it scales,
        # so an unused slider never reaches params as a live setting.
        "ip_scale": float(form["ip_scale"]) if form.get("ip_adapter") else None,
        "init_image": bool(form.get("init_image")) and bool(form.get("ref_path")),
        "init_strength": (
            float(form.get("init_strength") or 0.45)
            if form.get("init_image") and form.get("ref_path")
            else None
        ),
        "control_scale": float(form["control_scale"]) if form.get("control") else None,
        "control_end": float(form["control_end"]) if form.get("control") else None,
        "guidance_fields": fields,
        **create_assets.persisted_intent(form),
    }

def sprite_sheet_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The 2D form as ``create_job``'s ``sprite_sheet=`` block takes it.

    :func:`tile_sheet_kwargs`' opposite number on the other arm, and split out
    of :func:`submit_kwargs` for that function's reason: the sprite arm is an
    ordinary reference job *carrying a follow-up request* -- the rig checkbox's
    shape, so the character is a row in its own right and the sheet is queued
    against it once it lands -- and the compilation of that follow-up is the one
    part of the press no test could name while it lived inside a literal.

    ``_jobs_create._check_sprite_sheet`` validates every key here at the
    *reference* door rather than when the follow-up is minted, so a palette
    that has been deleted since the form listed it costs the request instead of
    an SDXL generation and an hour.

    ``candidates`` is sent rather than left to the door's own default, and that
    is not belt-and-braces: the Dimensions section has already told the user how
    many generations this press costs, and a block that let the worker decide
    the number separately is how a form comes to promise eight and spend
    sixteen. It is :func:`sprite_plan`'s number, which is the line's number.
    """
    plan = sprite_plan(form)
    return {
        "sheet_type": plan["layout"],
        "candidates": plan["candidates"],
        "logical_size": plan["logical_size"],
        "colors": svc_sprites.DEFAULT_SPRITE_COLORS,
        "target_cell_px": (
            None if form.get("target_cell_px") in (None, "")
            else safe_int(form.get("target_cell_px"), 0)
        ),
        # The three the form draws under Dimensions. Sent always rather than
        # only when set: the door's own defaults are these values, and a block
        # that omitted them would make "no palette" and "the form was never
        # asked" the same request -- which is how a setting comes to be recorded
        # as something nobody chose.
        "palette": str(form.get("palette") or ""),
        "dither": bool(form.get("dither")),
        "outline": str(form.get("outline") or svc_sprites.DEFAULT_SPRITE_OUTLINE),
    }

def tile_sheet_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The 2D form as ``create_tile_sheet`` takes it, minus the reference bytes.

    :func:`submit_kwargs`' opposite number, and it exists for the reason that
    one does: the tile arm is the only output that does not go through
    ``create_job``, so the compilation of its request had no name and lived
    inside a closure -- where no test could reach it. The submit then went on
    sending a request with no layout in it, against a door whose default layout
    is ``materials``, and every press was refused at ``field="prompt_items"``
    with nothing in the form saying why.

    **``allow_grid`` is sent when, and only when, the user picked the grid.**
    That flag is the door's escape hatch on a refusal about a measurement rather
    than about an impossibility, so an explicit choice is exactly what it is for
    -- and a default that carried it would put every unconsidered press back on
    the layout the measurement is about.
    """
    mode = tile_mode_of(form)
    kwargs: dict[str, Any] = {
        "prompt": str(form.get("prompt") or "").strip(),
        "tile_size": safe_int(form.get("tile_size"), 32),
        "view": view_of(form),
        "seed": int(form["seed"]),
        "negative_prompt": form.get("negative_prompt"),
        "mode": mode,
        # The pixel look, from the two controls under Dimensions. No ``outline``
        # key at all -- and the absence is load-bearing rather than tidy: the
        # door refuses one by name (a tile is opaque edge to edge, so an outline
        # is a grid line around every cell), and a form that sent even
        # ``"none"`` here would be naming a setting this kind does not have.
        "palette": str(form.get("palette") or ""),
        "dither": bool(form.get("dither")),
        # The two checkboxes under Materials. They were drawn, they wrote to
        # the form, and the form was never read: the request left without them
        # and ``default_form_2d`` declared neither, so the values did not
        # survive a restart either. Both halves below them were already live --
        # ``service.jobs`` passes them to the worker and
        # ``tilesheets._check_weights`` already widens the weight gate on
        # ``style_lock`` -- so this line is the whole of what was missing.
        "style_lock": bool(form.get("style_lock")),
        "seam_erase": bool(form.get("seam_erase")),
        **create_assets.persisted_intent(form),
    }
    if mode == svc_tilesheets.MODE_MATERIALS:
        kwargs["prompt_items"] = list(material_lines(form))
        kwargs["variants"] = safe_int(form.get("variants"), 1)
    elif mode == svc_tilesheets.MODE_TERRAIN:
        kwargs["inner_terrain"] = str(form.get("inner_terrain") or "").strip()
        kwargs["outer_terrain"] = str(form.get("outer_terrain") or "").strip()
        kwargs["boundary"] = str(form.get("boundary") or "").strip()
    else:
        kwargs["allow_grid"] = True
    return kwargs

_WEIGHT_FIELDS = (
    ("base_model", "base", "The image model"),
    ("style_lora", "lora", "The style LoRA"),
    ("ip_adapter", "adapter", "The reference adapter"),
    ("control", "control", "The structure control"),
)

def weights_problem(ctx: Any, form: dict[str, Any]) -> problem_types.Problem | None:
    """The first selected model this host has not downloaded, or None.

    Beside :func:`validate` rather than inside it, and the split is deliberate.
    ``validate`` is about the *form* -- a prompt that is empty, a count out of
    range -- and is true on any machine; this is about this **install**, and
    two forms identical in every field can disagree about it. Keeping them
    apart is also what stops the aggregate block above Generate from listing a
    download as a mistake the user made.

    ``model_gate``'s doctrine throughout: an empty ``model_rows`` says nothing
    rather than everything-is-missing (a headless ctx, or the first frame
    before the answers land, must not lock a fully-installed host), and a row
    the snapshot has never heard of is skipped. A stale snapshot costs a
    missing warning, never a wrong outcome -- ``service.validation.check_weights``
    is still the authority and still refuses at the door.
    """
    if is_character(form):
        # A character downloads nothing. Its body comes off the baked assets
        # this build ships and its cells are rendered by Blender, so walking
        # ``_WEIGHT_FIELDS`` here would point a "not downloaded" refusal at a
        # checkpoint the run never opens. What a character *is* short of --
        # Blender -- is ``character_engine.problems``'s sentence, in the Rig
        # segment's exact words.
        return None
    by_key = {str(row.get("row_key")): row for row in (getattr(ctx, "model_rows", None) or [])}
    if not by_key:
        return None
    if is_tile_arm(form):
        # A tile set's door pins its own base and LoRA and ignores the form's,
        # so walking ``_WEIGHT_FIELDS`` here would point at a selection the run
        # never reads. Sprite sheets are different: their preliminary
        # reference does use those selected fields, then a pinned final recipe.
        for row_key in sheet_rows(form):
            row = by_key.get(row_key)
            if row is None or row.get("present"):
                continue
            label = row.get("label") or row_key
            return problem_types.Problem(
                f"A sheet needs {label!r}, which is not downloaded. "
                f"Install it in Settings.",
                "output",
            )
        return None
    for field, kind, noun in _WEIGHT_FIELDS:
        chosen = str(form.get(field) or "")
        if not chosen:
            continue
        row = by_key.get(f"{kind}:{chosen}")
        if row is None or row.get("present"):
            continue
        label = row.get("label") or chosen
        return problem_types.Problem(
            f"{noun} {label!r} is selected but not downloaded. "
            f"Install it in Settings, or pick another.",
            field,
        )
    if form.get("output") == "sheet" and form.get("sheet_type") == "sprite":
        # Only after the visible preliminary recipe has passed: both stages
        # are real requirements, and the first problem should point at the
        # editable control before naming the locked follow-up recipe.
        for row_key in sheet_rows(form):
            row = by_key.get(row_key)
            if row is None or row.get("present"):
                continue
            label = row.get("label") or row_key
            return problem_types.Problem(
                f"The final sheet needs {label!r}, which is not downloaded. "
                f"Install it in Settings.",
                "output",
            )
    return None
