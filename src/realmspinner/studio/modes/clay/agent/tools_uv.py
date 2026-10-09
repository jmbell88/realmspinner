"""Clay's agent tool surface, the UV handler family: ``clay_uv`` alone.

One tool with three actions behind it (:data:`~.schema.UV_ACTIONS`) -- unwrap,
pack and transform -- rather than three separate tools, because the catalogue
is already large. See ``studio/modes/clay/agent/validate.py``'s own module
docstring for why this handler reaches ``fail``/``_json``/``Session``/``_tab``/
the shared validators through that module rather than through
``studio/modes/clay/agent/dispatch.py`` directly: this file has no import of
``dispatch.py`` at all.

**Every action is one document door, so every action is one undo step**
(``ClayDoc.set_mesh``) and keeps the generator (``keep_generator=True``, the
way the box unwrap op already does: a UV layout is not geometry).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .....kernels.mesh import uv as uv_mod
from .....kernels.mesh import uvtools
from .....kernels.mesh.elements import OpError
from .schema import UV_ACTION_ARGS, UV_ACTIONS
from .validate import (
    UV_MAX,
    Session,
    _as_float,
    _json,
    _resolve_uid,
    _scene_row,
    _tab,
    _validate_range,
    _whole_number,
    fail,
)


def _set_uv_mesh(doc: Any, obj: Any, mesh: Any, action: str, field: str = "uid") -> dict:
    try:
        changed = doc.set_mesh(obj.uid, mesh, keep_generator=True)
    except OpError as error:
        # *field* defaults to the uid (unwrap and pack take no number that could
        # have caused a refusal); transform names the number it was handed --
        # the 2026-10-09 second audit's clay-133 found a float32 overflow from
        # a ``translate`` blamed on ``uid``.
        return fail(str(error), field=field)
    return _json({"action": action, "changed": changed, **_scene_row(doc, obj)})


def _needs_uv(obj: Any) -> dict | None:
    if obj.mesh.uv is None:
        return fail(
            f"{obj.name!r} needs texture coordinates -- clay_uv action=unwrap first.",
            field="uid",
        )
    return None


def _uv_unwrap(doc: Any, obj: Any, args: dict) -> dict:
    del args
    return _set_uv_mesh(doc, obj, uv_mod.box_unwrap(obj.mesh), "unwrap")


def _uv_pack(doc: Any, obj: Any, args: dict) -> dict:
    failure = _needs_uv(obj)
    if failure:
        return failure
    margin, failure = _validate_range(args.get("margin", 0.005), "margin", 0.0, 0.5)
    if failure:
        return failure
    rotate_arg = args.get("rotate", False)
    if not isinstance(rotate_arg, bool):
        return fail("rotate must be a boolean.", field="rotate")
    mesh = uvtools.pack_islands(obj.mesh, margin=margin, rotate=rotate_arg)
    return _set_uv_mesh(doc, obj, mesh, "pack")


def _uv_transform(doc: Any, obj: Any, args: dict) -> dict:
    failure = _needs_uv(obj)
    if failure:
        return failure
    ids = uvtools.islands(obj.mesh)
    islands_arg = args.get("islands")
    if islands_arg is None:
        wanted = np.unique(ids).tolist()
    else:
        if not isinstance(islands_arg, list) or not islands_arg:
            return fail("islands must be a non-empty list of island ids.", field="islands")
        try:
            # ``_whole_number``: the 2026-10-07 audit's clay-35 -- ``int(1.9)``
            # transformed island 1.
            wanted = [_whole_number(i) for i in islands_arg]
        except (TypeError, ValueError, OverflowError):
            return fail("islands must be a list of integers.", field="islands")
        known = set(np.unique(ids).tolist())
        unknown = [i for i in wanted if i not in known]
        if unknown:
            return fail(
                f"no uv island {unknown[0]} on {obj.name!r} (it has {len(known)}).",
                field="islands",
            )

    translate = args.get("translate", [0.0, 0.0])
    if not isinstance(translate, list) or len(translate) != 2:
        return fail("translate must be [u, v].", field="translate")
    try:
        du, dv = _as_float(translate[0]), _as_float(translate[1])
    except (TypeError, ValueError, OverflowError):
        return fail("translate must be [u, v].", field="translate")
    if not (math.isfinite(du) and math.isfinite(dv)):
        return fail("translate must be finite numbers.", field="translate")
    # The 2026-10-09 second audit's clay-133: finiteness was the only bound, so a
    # 1e30 shift was stored and every later ``clay_scene`` row ran its overlap
    # bucketing over it. ``UV_MAX`` is the bound ``clay_add_mesh`` already holds
    # a uv to; a second call could still walk past it, which is the kernel's
    # float32 check below.
    if abs(du) > UV_MAX or abs(dv) > UV_MAX:
        return fail(
            f"translate must be no larger than {UV_MAX:g} in magnitude.", field="translate"
        )
    rotate_deg, failure = _validate_range(args.get("rotate_deg", 0.0), "rotate_deg", -360.0, 360.0)
    if failure:
        return failure
    scale, failure = _validate_range(args.get("scale", 1.0), "scale", 1e-3, 1e3)
    if failure:
        return failure

    # What a kernel refusal is about: too many islands is the ``islands``
    # argument; anything else is a float32 overflow, which ``translate`` is
    # bounded against above, so only a ``scale`` (or, at scale 1, a run of
    # translates) can have caused it. Named for that argument, never the uid.
    overflow_field = "scale" if scale != 1.0 else "translate"
    try:
        mesh = uvtools.transform_islands(
            obj.mesh,
            wanted,
            translate=(du, dv),
            rotate_deg=rotate_deg,
            scale=scale,
            ids=ids,
        )
    except OpError as error:
        too_many = len(wanted) > uvtools.MAX_UV_ISLANDS
        return fail(str(error), field="islands" if too_many else overflow_field)
    return _set_uv_mesh(doc, obj, mesh, "transform", field=overflow_field)


_UV_ACTION_HANDLERS: dict[str, Any] = {
    "unwrap": _uv_unwrap,
    "pack": _uv_pack,
    "transform": _uv_transform,
}
"""Every :data:`~.schema.UV_ACTIONS` name mapped to its own handler function
-- gated both ways by ``tests/modes/clay/test_agent_clay_uv.py`` against
``UV_ACTIONS`` itself, the same bidirectional shape every derived enum in
this fold already gets."""


def _h_uv(ctx: Any, session: Session, args: dict) -> dict:
    """Run one uv action against one object. See :data:`~.schema.UV_ACTIONS`
    for the three actions and their own params.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure
    action = args.get("action")
    # isinstance checked first: the 2026-09-26 audit's clay-agent-tools-06 --
    # ``x not in a_dict`` hashes ``x``, and a list or object ``action``
    # raised a bare, unhashable ``TypeError`` that only ``call()``'s generic
    # "failed unexpectedly" backstop caught, instead of this refusal.
    if not isinstance(action, str) or action not in UV_ACTIONS:
        return fail(f"action must be one of {', '.join(sorted(UV_ACTIONS))}.", field="action")
    # The 2026-10-09 second audit's clay-134: ``translate`` with ``action:
    # "pack"`` was read and dropped, so an agent that meant a transform was told
    # it had succeeded. Refused against ``UV_ACTION_ARGS``, the action's own set.
    allowed = {"uid", "action", *UV_ACTION_ARGS[action]}
    stray = next((k for k in args if k not in allowed), None)
    if stray is not None:
        return fail(
            f"{stray!r} is not an argument of the {action!r} action "
            f"(it takes {', '.join(UV_ACTION_ARGS[action]) or 'no further arguments'}).",
            field=stray,
        )
    return _UV_ACTION_HANDLERS[action](doc, obj, args)
