"""Clay's agent tool surface, the modifier-stack handler family:
``clay_modifier_add``, ``clay_modifier_set``, ``clay_modifier_remove``,
``clay_modifier_move`` and ``clay_modifier_apply``.

Tranche 2 (``dev/CLAY-PLAN.md``): the modifier stack (:mod:`.modifiers`)
landed in the kernel and in ``document.py`` first -- ``Obj.modifiers``, the
document's ``set_modifiers``/``apply_modifiers``/``evaluated``/``evaluation``
doors -- and this file is the agent surface over that stack, the same shape
the human surface (the Properties pane's own "Modifiers" section) reaches
through ``document.py`` directly. A new family file rather than folding into
``tools.py`` (which already carries the ten object-level tools) or
``tools_ops.py`` (the selection/ops/inspection ten): five tools that share
one new vocabulary -- a stack entry addressed by ``(uid, modifier id)``
rather than by uid alone -- are exactly the shape every existing split in
this fold already draws a line at (see ``studio/modes/clay/agent/tools.py``'s
own docstring for why the original split landed where it did).

See ``studio/modes/clay/agent/validate.py``'s own module docstring for why
every handler here reaches ``fail``/``ok``/``_json``/``Session``/``_tab``/
``_scene_row``/the shared validators through that module rather than through
``studio/modes/clay/agent/dispatch.py`` directly: this file has no import of
``dispatch.py`` at all, because none of these five handlers ever needs
anything that lives only there.

**Every value ``modifiers.make``/``with_params`` will ever see has already
been checked here first.** Both trust their own ``params`` dict to already
be coerced and clamped (:mod:`.ops_modifiers`'s own module docstring) --
which is exactly true of a call built through the Properties pane's own
number fields, and exactly *not* true of whatever JSON an agent sends. Every
handler below runs :func:`~.validate._modifier_params_type_refusal` (the
modifier-stack twin of :func:`~.validate._op_params_type_refusal`, which
``clay_op``'s own docstring names the three crashes it closed) before ever
calling into :mod:`.modifiers`, so an unknown parameter name or a non-numeric
value is refused by name rather than reaching a bare ``float()`` call two
frames deep. Clamping itself -- an out-of-range number silently pulled back
into ``[low, high]`` rather than refused -- is left to ``modifiers.make``/
``with_params`` themselves, the identical trade ``clay_op``'s own ``run``
already makes for its declared parameters (see that function's own
docstring): a subdivide asked for ``levels=99`` is not a refusal a caller
should have to write for itself.

**A ``target`` parameter is checked for real, not only for shape.** The wire
schema and :func:`~.validate._modifier_params_type_refusal` both see a
``boolean`` modifier's ``target`` as *a number* -- there is no ``"integer,
must name an existing object"`` constraint JSON Schema can express for one
key of an open-ended ``params`` object -- so :func:`_target_refusal` walks
the modifier's own declared parameters for the one (or, for a future kind,
more than one) marked ``target=True`` (:class:`~.modifiers.ModParam`'s own
field) and checks the *coerced* value against the live document: ``0`` is
the kind's own "none chosen" sentinel and is always let through (a boolean
modifier is allowed to exist with no target yet, refusing only once
something tries to evaluate it -- see :mod:`.modifiers`'s "skipped, not
fatal" rule), a nonzero value naming the object's own uid is refused as a
guaranteed self-cycle before ever reaching the document-wide cycle check,
and a nonzero value naming no live object is refused eagerly rather than
left to surface later as an evaluation-time "Target object N no longer
exists." error. ``ClayDoc.set_modifiers`` still runs its own
``would_cycle`` check after this -- an *indirect* cycle through some other
object's own boolean target is not this function's job to find, only the
two cheap, local mistakes an agent is likeliest to make.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .....kernels.mesh import modifiers as clay_modifiers
from .validate import (
    Session,
    _json,
    _modifier_params_type_refusal,
    _resolve_modifier,
    _resolve_uid,
    _scene_row,
    _tab,
    fail,
)


def _target_refusal(doc: Any, obj: Any, kind_def: Any, mod: Any) -> dict | None:
    """See this module's own docstring's ``target`` paragraph. ``mod`` is
    already a built :class:`~.modifiers.Modifier` (post ``make``/
    ``with_params``), so every declared parameter -- ``target`` included --
    already carries its coerced, clamped value."""
    for p in kind_def.params:
        if not p.target:
            continue
        value = int(mod.get(p.name, 0))
        if value == 0:
            continue
        if value == obj.uid:
            return fail(
                "target must not name the object's own uid -- that is a "
                "guaranteed cycle.",
                field="params",
            )
        if not any(o.uid == value for o in doc.objects):
            return fail(
                f"no object with uid {value}.",
                field="params",
                recovery="read_scene",
                uids=[value],
            )
    return None


def _kind_or_refusal(kind: Any) -> tuple[Any, dict | None]:
    kind_def = clay_modifiers.MODIFIERS.get(kind)
    if kind_def is None:
        return None, fail(
            f"kind must be one of {', '.join(sorted(clay_modifiers.MODIFIERS))}.", field="kind"
        )
    return kind_def, None


def _h_modifier_add(ctx: Any, session: Session, args: dict) -> dict:
    """Append (or insert) one new modifier onto an object's stack, as one
    undo step. See this module's own docstring for the validation order:
    ``kind`` against the registry, every ``params`` value's own shape, the
    target check, *then* ``index`` -- nothing below this line can refuse, so
    a refused call never touches the document (``document.set_modifiers``
    itself still runs the document-wide cycle check, the one thing this
    handler cannot answer locally).
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure

    kind_def, failure = _kind_or_refusal(args.get("kind"))
    if failure:
        return failure

    params_arg = args.get("params")
    if params_arg is not None and not isinstance(params_arg, dict):
        return fail("params must be an object.", field="params")
    params_arg = params_arg or {}
    failure = _modifier_params_type_refusal(kind_def, params_arg)
    if failure:
        return failure

    new_mod = clay_modifiers.make(
        kind_def.name, params_arg, id=clay_modifiers.next_id(obj.modifiers)
    )
    failure = _target_refusal(doc, obj, kind_def, new_mod)
    if failure:
        return failure

    stack = list(obj.modifiers)
    index_arg = args.get("index")
    if index_arg is None:
        index = len(stack)
    else:
        try:
            index = int(index_arg)
        except (TypeError, ValueError):
            return fail("index must be an integer.", field="index")
        if not (0 <= index <= len(stack)):
            return fail(f"index must be between 0 and {len(stack)}.", field="index")
    stack.insert(index, new_mod)

    doc.set_modifiers(obj.uid, tuple(stack))
    return _json({"modifier": new_mod.id, **_scene_row(doc, obj)})


def _h_modifier_set(ctx: Any, session: Session, args: dict) -> dict:
    """Change an existing modifier's params and/or its ``enabled`` flag, as
    one undo step. Neither given is refused outright -- the identical "give
    at least one" rule ``clay_transform`` already holds its own three
    optional vectors to."""
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure
    modifier_id, failure = _resolve_modifier(obj, args)
    if failure:
        return failure
    existing = next(m for m in obj.modifiers if m.id == modifier_id)

    kind_def = clay_modifiers.MODIFIERS.get(existing.kind)
    if kind_def is None:
        # Reachable only from a hand-edited or partially-loaded document
        # naming a kind this build no longer registers -- ``serialize.py``'s
        # own read-time refusal is the front door for that, this is the
        # backstop for whatever slipped past it.
        return fail(f"Unknown modifier kind {existing.kind!r}.", field="modifier")

    params_arg = args.get("params")
    enabled_arg = args.get("enabled")
    if params_arg is None and enabled_arg is None:
        return fail("give params and/or enabled.")
    if params_arg is not None and not isinstance(params_arg, dict):
        return fail("params must be an object.", field="params")
    if params_arg:
        failure = _modifier_params_type_refusal(kind_def, params_arg)
        if failure:
            return failure

    new_mod = existing
    if params_arg:
        new_mod = clay_modifiers.with_params(new_mod, params_arg)
        failure = _target_refusal(doc, obj, kind_def, new_mod)
        if failure:
            return failure
    if enabled_arg is not None:
        if not isinstance(enabled_arg, bool):
            return fail("enabled must be a boolean.", field="enabled")
        new_mod = replace(new_mod, enabled=enabled_arg)

    stack = tuple(new_mod if m.id == modifier_id else m for m in obj.modifiers)
    changed = doc.set_modifiers(obj.uid, stack)
    return _json({"modifier": modifier_id, "changed": changed, **_scene_row(doc, obj)})


def _h_modifier_remove(ctx: Any, session: Session, args: dict) -> dict:
    """Drop one modifier from an object's stack, as one undo step."""
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure
    modifier_id, failure = _resolve_modifier(obj, args)
    if failure:
        return failure

    stack = tuple(m for m in obj.modifiers if m.id != modifier_id)
    doc.set_modifiers(obj.uid, stack)
    return _json({"removed": modifier_id, **_scene_row(doc, obj)})


def _h_modifier_move(ctx: Any, session: Session, args: dict) -> dict:
    """Reorder one modifier within its object's stack, as one undo step.
    Order matters -- a mirror before a solidify shells the mirrored pair, a
    solidify before a mirror mirrors the shell -- so this is the one door
    that changes only where a modifier sits, never its params or its
    ``enabled`` flag."""
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure
    modifier_id, failure = _resolve_modifier(obj, args)
    if failure:
        return failure

    index_arg = args.get("index")
    if index_arg is None:
        return fail("give a value for 'index'.", field="index")
    try:
        index = int(index_arg)
    except (TypeError, ValueError):
        return fail("index must be an integer.", field="index")
    stack = list(obj.modifiers)
    if not (0 <= index < len(stack)):
        return fail(f"index must be between 0 and {len(stack) - 1}.", field="index")

    target = next(m for m in stack if m.id == modifier_id)
    stack.remove(target)
    stack.insert(index, target)

    changed = doc.set_modifiers(obj.uid, tuple(stack))
    return _json({"modifier": modifier_id, "changed": changed, **_scene_row(doc, obj)})


def _h_modifier_apply(ctx: Any, session: Session, args: dict) -> dict:
    """Bake a stack's prefix into the base mesh, as one undo step --
    ``document.apply_modifiers``'s own agent door. ``modifier`` omitted bakes
    the whole stack; given, it bakes everything through that id (inclusive)
    and leaves the rest of the stack live. A disabled modifier inside the
    prefix is dropped without being applied; a modifier inside the prefix
    that currently refuses refuses the whole apply, with its own message --
    see ``document.apply_modifiers``'s own docstring for why: never bake a
    half-result the caller never saw. An object with no modifiers at all
    (or, given a ``modifier``, one it does not have) is refused rather than
    silently answering ``changed: false`` -- unlike a no-op transform, there
    is no reading of "apply the stack" that is satisfied by there being none
    to apply.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure

    through_id: int | None = None
    if args.get("modifier") is not None:
        modifier_id, failure = _resolve_modifier(obj, args)
        if failure:
            return failure
        through_id = modifier_id
    elif not obj.modifiers:
        return fail("This object has no modifiers to apply.")

    changed = doc.apply_modifiers(obj.uid, through_id=through_id)
    return _json(
        {"applied_through": through_id, "changed": changed, **_scene_row(doc, obj)}
    )
