"""Clay's agent tool surface, the UV handler family (tranche 6,
``dev/CLAY-PLAN.md``): ``clay_uv`` alone.

A new family file, the same shape ``studio/modes/clay/agent/tools_structure.py``
landed in as tranche 3's own family -- one tool, but with five actions behind
it (:data:`~.schema.UV_ACTIONS`), because the brief's own steer is one tool
with an action enum rather than five separate ones ("the catalogue is
already too big"). See ``studio/modes/clay/agent/validate.py``'s own module docstring for why
this handler reaches ``fail``/``_json``/``Session``/``_tab``/the shared
validators through that module rather than through ``studio/modes/clay/agent/dispatch.py``
directly: this file has no import of ``dispatch.py`` at all.

**Every action is one document door, so every action is one undo step**
(``ClayDoc.set_mesh``/``set_seams``, both already atomic) -- ``pack``,
``density`` and ``unwrap_seams`` keep the generator (``keep_generator=True``,
the way the existing box/planar unwrap already does: an unwrap is not
geometry), and ``mark_seam``/``clear_seam`` touch ``Obj.seams`` alone, never
the mesh. All five refuse a locked object by name -- ``set_mesh``/
``set_seams`` both already do (``document.py``'s own locking paragraph); a
seam or a UV layout is authoring intent about the object's own geometry, the
identical footing a mesh edit already stands on.

**``mark_seam``/``clear_seam`` read ``edges`` if given, or the object's
current edge selection otherwise** -- the same shape the tranche 6
integration spec gives the *interactive* Mark Seam/Clear Seam buttons ("OPS
rows in edge mode"), reused here rather than forcing an agent to switch
element mode and select edges first just to name two vertices it already
knows.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .....kernels.mesh import uvtools, uvunwrap
from .....kernels.mesh.adjacency import adjacency
from .....kernels.mesh.elements import OpError
from .schema import UV_ACTIONS
from .validate import (
    Session,
    _json,
    _resolve_uid,
    _scene_row,
    _tab,
    _validate_range,
    fail,
)


def _validate_edges_arg(mesh: Any, edges_arg: Any) -> tuple[list[list[int]] | None, dict | None]:
    """*edges_arg* as ``[[v, v], ...]``, every pair a real edge of *mesh* --
    the identical check ``agent_clay_tools_ops._h_select_elements`` already
    runs for its own ``edges`` argument, reused rather than re-derived."""
    try:
        pairs = [[int(a), int(b)] for a, b in edges_arg]
    except (TypeError, ValueError):
        return None, fail("edges must be a list of [vertex, vertex] pairs.", field="edges")
    if not pairs:
        return None, fail("edges must not be empty.", field="edges")
    ids = adjacency(mesh).edge_ids(np.asarray(pairs, dtype="i4"))
    bad_at = next((i for i, e in enumerate(ids) if e < 0), None)
    if bad_at is not None:
        return None, fail(f"{pairs[bad_at]} is not an edge of this mesh.", field="edges")
    return pairs, None


def _seam_edges(obj: Any, doc: Any, args: dict) -> tuple[list[list[int]] | None, dict | None]:
    """``args["edges"]``, validated -- or, omitted, the object's current edge
    selection. Shared by ``mark_seam`` and ``clear_seam``; see this module's
    own docstring for why the fallback exists."""
    edges_arg = args.get("edges")
    if edges_arg is not None:
        return _validate_edges_arg(obj.mesh, edges_arg)
    sel_edges = doc.element_sel_of(obj.uid).edges
    if len(sel_edges) == 0:
        return None, fail(
            "give edges, or select some edges first (clay_select_elements or "
            "clay_select_by, in edge mode).",
            field="edges",
        )
    return [[int(a), int(b)] for a, b in sel_edges], None


def _uv_pack(doc: Any, obj: Any, args: dict) -> dict:
    if obj.mesh.uv is None:
        return fail(
            f"{obj.name!r} needs texture coordinates -- unwrap it first.", field="uid"
        )
    margin_arg = args.get("margin", 0.005)
    margin, failure = _validate_range(margin_arg, "margin", 0.0, 0.5)
    if failure:
        return failure
    rotate_arg = args.get("rotate", False)
    if not isinstance(rotate_arg, bool):
        return fail("rotate must be a boolean.", field="rotate")
    mesh = uvtools.pack_islands(obj.mesh, margin=margin, rotate=rotate_arg)
    try:
        changed = doc.set_mesh(obj.uid, mesh, keep_generator=True)
    except OpError as error:
        return fail(str(error), field="uid")
    return _json({"action": "pack", "changed": changed, **_scene_row(doc, obj)})


def _uv_density(doc: Any, obj: Any, args: dict) -> dict:
    if obj.mesh.uv is None:
        return fail(
            f"{obj.name!r} needs texture coordinates -- unwrap it first.", field="uid"
        )
    if args.get("target") is None:
        return fail("give a value for 'target'.", field="target")
    try:
        target = float(args["target"])
    except (TypeError, ValueError):
        return fail("target must be a number.", field="target")
    if not (target > 0.0):
        return fail("target must be a positive number.", field="target")
    texture_px_arg = args.get("texture_px", 1024)
    try:
        texture_px = int(texture_px_arg)
    except (TypeError, ValueError):
        return fail("texture_px must be an integer.", field="texture_px")
    if texture_px < 1:
        return fail("texture_px must be at least 1.", field="texture_px")
    mesh = uvtools.normalize_density(obj.mesh, target, texture_px=texture_px)
    try:
        changed = doc.set_mesh(obj.uid, mesh, keep_generator=True)
    except OpError as error:
        return fail(str(error), field="uid")
    return _json({"action": "density", "changed": changed, **_scene_row(doc, obj)})


def _uv_unwrap_seams(doc: Any, obj: Any, args: dict) -> dict:
    del args
    if not obj.seams:
        return fail(
            f"{obj.name!r} has no seams marked -- clay_uv action=mark_seam first.",
            field="uid",
        )
    try:
        mesh = uvunwrap.unwrap_lscm(obj.mesh, np.asarray(obj.seams, dtype="i4"))
        changed = doc.set_mesh(obj.uid, mesh, keep_generator=True)
    except OpError as error:
        return fail(str(error), field="uid")
    return _json({"action": "unwrap_seams", "changed": changed, **_scene_row(doc, obj)})


def _uv_mark_seam(doc: Any, obj: Any, args: dict) -> dict:
    pairs, failure = _seam_edges(obj, doc, args)
    if failure:
        return failure
    merged = uvtools.edge_keys(list(obj.seams) + pairs)
    try:
        changed = doc.set_seams(obj.uid, merged.tolist())
    except OpError as error:
        return fail(str(error), field="uid")
    return _json(
        {"action": "mark_seam", "changed": changed, "uid": obj.uid, "seam_count": len(obj.seams)}
    )


def _uv_clear_seam(doc: Any, obj: Any, args: dict) -> dict:
    pairs, failure = _seam_edges(obj, doc, args)
    if failure:
        return failure
    drop = {uvtools.edge_key(a, b) for a, b in pairs}
    remaining = [e for e in obj.seams if e not in drop]
    try:
        changed = doc.set_seams(obj.uid, remaining)
    except OpError as error:
        return fail(str(error), field="uid")
    return _json(
        {"action": "clear_seam", "changed": changed, "uid": obj.uid, "seam_count": len(obj.seams)}
    )


_UV_ACTION_HANDLERS: dict[str, Any] = {
    "pack": _uv_pack,
    "density": _uv_density,
    "unwrap_seams": _uv_unwrap_seams,
    "mark_seam": _uv_mark_seam,
    "clear_seam": _uv_clear_seam,
}
"""Every :data:`~.schema.UV_ACTIONS` name mapped to its own handler function
-- gated both ways by ``tests/modes/clay/test_agent_clay_uv.py`` against
``UV_ACTIONS`` itself, the same bidirectional shape every derived enum in
this fold already gets."""


def _h_uv(ctx: Any, session: Session, args: dict) -> dict:
    """Run one uv action against one object. See :data:`~.schema.UV_ACTIONS`
    for the five actions and their own params, and this module's own
    docstring for the undo/locking shape all five share.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure
    action = args.get("action")
    if action not in UV_ACTIONS:
        return fail(f"action must be one of {', '.join(sorted(UV_ACTIONS))}.", field="action")
    return _UV_ACTION_HANDLERS[action](doc, obj, args)
