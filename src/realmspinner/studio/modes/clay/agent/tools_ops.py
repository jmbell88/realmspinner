"""Clay's agent tool surface, the selection/ops/inspection handler family:
``clay_select``, ``clay_element_mode``, ``clay_select_elements``,
``clay_select_by``, ``clay_elements``, ``clay_op``, ``clay_render``,
``clay_diagnose``, ``clay_analyze``, ``clay_validate`` and ``clay_export``.

Split out of ``studio/modes/clay/agent/dispatch.py`` in the P4 restructure (``dev/RESTRUCTURE.md``);
see ``studio/modes/clay/agent/tools.py``'s own module docstring for where these handlers
actually lived in the pre-split file (the "# --- dispatch" banner, not "# ---
the tools" as the brief guessed from banner names alone) and why this file
exists as a second handler module beside it: ten object-level handlers plus
these ten selection/inspection/render ones would have made one file well
over the split's own ~2,000-line target, so the brief's suggested
scene/selection/ops families became two files -- this one folding selection,
element ops, render and the read-only inspectors (diagnose, analyze, and now
validate) together, since all of them act on a selection or read the
document rather than create or delete a whole object outright, and
``clay_export`` rides along as the one remaining handler with no better home.

Like ``studio/modes/clay/agent/tools.py``, this file reaches ``fail``/``ok``/``_json``/
``Session``/``_tab``/the shared validators through ``agent_clay_validate``
and its own ceilings through ``agent_clay_schema``, never through
``studio/modes/clay/agent/dispatch.py`` directly -- with one exception. ``_h_render`` needs
``agent_clay._view_for``, which stays in ``studio/modes/clay/agent/dispatch.py`` itself because two
tests monkeypatch it there by name (``tests/modes/clay/test_agent_clay.py``,
``tests/mcp/test_rpc_studio.py``); see :func:`_core` below for why that
reach is a function-scope import rather than a module-scope one -- the
identical shape ``studio/modes/clay/agent/dispatch.py``'s own ``_protocol()`` already uses, for a
different pair of reasons that both apply here too: ``studio/modes/clay/agent/dispatch.py`` imports
this module to build its own ``_HANDLERS`` table, so a module-scope import
back would be a real cycle, and ``tests/test_layering.py`` classifies
``studio/modes/clay/agent/dispatch.py`` as Clay's own mode file (layer 5) and this one as
unclassified shell (layer 4) -- a module-scope edge from here to there would
be a fresh, undocumented layering violation that a function-scope one never
becomes, because that walk is deliberately module-scope only (see that
test's own module docstring).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .....kernels.mesh import analyze as clay_analyze
from .....kernels.mesh import diagnose as clay_diagnose
from .....kernels.mesh import elements as el
from .....kernels.mesh import engines, readiness
from .....kernels.mesh import mesh as bm
from .....kernels.mesh import ops as clay_geom_ops
from .....kernels.mesh import select as bsel
from .....kernels.mesh.adjacency import adjacency
from ....viewer.camera import Camera
from .. import mode as clay_mode
from .. import ops as clay_ops
from .schema import (
    _QUERY_OPTIONAL_ARGS,
    ELEMENT_PAGE_DEFAULT,
    ELEMENT_PAGE_MAX,
    RENDER_FRAME_RESERVE,
    RENDER_PIXEL_BUDGET,
    RENDER_SHADINGS,
)
from .validate import (
    _OBJECT_SELECTION_DERIVED_REFUSAL,
    Session,
    _json,
    _op_params_type_refusal,
    _over_frame_budget,
    _protocol,
    _resolve_uid,
    _resolve_uids,
    _round,
    _sel_counts,
    _tab,
    _validate_query_arg,
    _validate_range,
    _validate_unit,
    fail,
    image_png,
    ok,
    text,
)

log = logging.getLogger(__name__)


def _core() -> Any:
    """``studio/modes/clay/agent/dispatch.py`` itself, imported lazily. See this module's own
    docstring for why: a module-scope import here would both cycle back
    through the module that imports this one to build ``_HANDLERS``, and
    register as a fresh layering violation ``tests/test_layering.py`` would
    have no entry for."""
    from . import dispatch as agent_clay

    return agent_clay


def _h_select(ctx: Any, session: Session, args: dict) -> dict:
    """Replace the *object* selection. Refused in an element mode -- see
    :data:`_OBJECT_SELECTION_DERIVED_REFUSAL` and :func:`_h_boolean`'s own
    comment (in ``studio/modes/clay/agent/tools.py``), which this shares the exact reason
    and the exact wording with.

    Tranche 3: ``tag``, given, adds every object carrying that tag to the
    result -- a union with ``uids``, never a replacement for it, which is
    what keeps ``uids``' own "an empty list clears the selection" meaning
    intact for a call that gives ``uids: []`` with no ``tag`` at all. Added
    here rather than as a ``clay_select_by`` query: ``select.QUERIES`` (a
    kernel registry this fold does not own -- see
    ``tools_structure``'s own module docstring) has no ``tag`` row, and a tag
    is whole-object metadata, not an element-mode query answerable from one
    mesh's own vertices/edges/faces the way every real ``QUERIES`` entry is.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    if doc.element_mode != "object":
        return fail(_OBJECT_SELECTION_DERIVED_REFUSAL, recovery="switch_mode")
    tag_arg = args.get("tag")
    if tag_arg is not None and not isinstance(tag_arg, str):
        return fail("tag must be a string.", field="tag")
    # The schema declares ``uids`` required, and ``_resolve_uids`` alone does
    # not enforce that: it treats a missing value the same as an explicit
    # empty list (``values or []``), because an empty list is this tool's own
    # "clear the selection" -- see that function's own docstring. Checked
    # for here, once, ahead of it, so *omitting* the argument entirely is
    # refused rather than silently read as the identical clearing call.
    if "uids" not in args:
        return fail(
            "give uids -- an empty list clears the selection.", field="uids"
        )
    uids, failure = _resolve_uids(doc, args.get("uids"), field="uids")
    if failure:
        return failure
    selected = set(uids)
    if tag_arg:
        needle = tag_arg.strip().lower()
        selected |= {obj.uid for obj in doc.objects if needle in obj.tags}
    doc.select(selected)
    return _json({"selection": sorted(doc.selection)})


def _h_element_mode(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    mode = args.get("mode")
    if mode not in el.MODES:
        return fail(f"mode must be one of {', '.join(el.MODES)}.", field="mode")
    doc.set_element_mode(mode)
    objects = [
        {"uid": uid, "stamp": doc.mesh_stamp(uid), "selected": _sel_counts(sel)}
        for uid, sel in doc.element_sel.items()
    ]
    return _json({"mode": doc.element_mode, "objects": objects})


def _h_select_elements(ctx: Any, session: Session, args: dict) -> dict:
    """Select vertices, edges or faces of one object by explicit index. See
    the tool's own description in ``agent_clay.tools`` for the full contract
    -- everything is validated against *this object's own mesh* before
    anything is switched or written, the same "validate everything before
    the first mutation" rule every other handler in this fold follows.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure

    mode = args.get("mode")
    if mode is not None and mode not in el.MODES:
        return fail(f"mode must be one of {', '.join(el.MODES)}.", field="mode")

    how = args.get("how", "replace")
    if how not in ("replace", "add", "subtract"):
        return fail("how must be 'replace', 'add' or 'subtract'.", field="how")

    expect_stamp = args.get("expect_stamp")
    if expect_stamp is not None:
        _, failure = _check_expect_stamp(doc, obj.uid, expect_stamp)
        if failure:
            return failure

    n_verts = len(obj.mesh.positions)
    n_faces = bm.face_count(obj.mesh)

    verts_arg = args.get("verts")
    vert_arr: list[int] | None = None
    if verts_arg is not None:
        try:
            vert_arr = [int(v) for v in verts_arg]
        except (TypeError, ValueError):
            return fail("verts must be a list of integers.", field="verts")
        bad = [v for v in vert_arr if not (0 <= v < n_verts)]
        if bad:
            return fail(
                f"vertex index {bad[0]} is out of range for this mesh "
                f"(0..{n_verts - 1}).",
                field="verts",
            )

    faces_arg = args.get("faces")
    face_arr: list[int] | None = None
    if faces_arg is not None:
        try:
            face_arr = [int(f) for f in faces_arg]
        except (TypeError, ValueError):
            return fail("faces must be a list of integers.", field="faces")
        bad = [f for f in face_arr if not (0 <= f < n_faces)]
        if bad:
            return fail(
                f"face index {bad[0]} is out of range for this mesh (0..{n_faces - 1}).",
                field="faces",
            )

    edges_arg = args.get("edges")
    edge_arr: list[list[int]] | None = None
    if edges_arg is not None:
        try:
            pairs = [[int(a), int(b)] for a, b in edges_arg]
        except (TypeError, ValueError):
            return fail("edges must be a list of [vertex, vertex] pairs.", field="edges")
        if pairs:
            # ``ElementSel`` accepts any vertex pair with no complaint -- it
            # is only an overlay index buffer once it reaches the viewport --
            # so an unchecked pair would draw a line between two vertices
            # with nothing between them, and a face index past the mesh's
            # count would take the overlay build down rather than refuse
            # cleanly. Mapped through the mesh's own adjacency and refused by
            # naming the first pair that is not really an edge here instead.
            ids = adjacency(obj.mesh).edge_ids(np.asarray(pairs, dtype="i4"))
            bad_at = next((i for i, e in enumerate(ids) if e < 0), None)
            if bad_at is not None:
                return fail(f"{pairs[bad_at]} is not an edge of this mesh.", field="edges")
        edge_arr = pairs

    # Mode switches *first*, converting whatever was already selected -- and
    # only after every index above has been checked against the unchanged
    # mesh, so a refused call has touched neither the mode nor the selection.
    # The order matters for how="add": switching first is what puts the
    # prior selection into the new mode's own currency before the union
    # below runs, rather than unioning arrays that describe two different
    # element kinds.
    if mode is not None:
        doc.set_element_mode(mode)

    requested = el.ElementSel(verts=vert_arr, edges=edge_arr, faces=face_arr)
    current = doc.element_sel_of(obj.uid)
    doc.set_element_sel(obj.uid, el.combine(current, requested, how))

    return _json(
        {
            "uid": obj.uid,
            "mode": doc.element_mode,
            "stamp": doc.mesh_stamp(obj.uid),
            "selected": _sel_counts(doc.element_sel_of(obj.uid)),
        }
    )


def _check_expect_stamp(
    doc: Any, uid: int, expect_stamp: Any
) -> tuple[int | None, dict | None]:
    """*expect_stamp* as an int, or a refusal naming ``field="expect_stamp"``
    if it does not match ``doc.mesh_stamp(uid)`` right now. Shared by
    :func:`_h_select_elements` and :func:`_h_select_by`, both of which refuse
    a stale stamp before touching the mode or the selection."""
    try:
        expect_stamp = int(expect_stamp)
    except (TypeError, ValueError):
        return None, fail("expect_stamp must be an integer.", field="expect_stamp")
    current = doc.mesh_stamp(uid)
    if expect_stamp != current:
        return None, fail(
            f"expect_stamp {expect_stamp} does not match this object's "
            f"current stamp {current} -- the mesh changed since that stamp "
            "was read; call clay_elements or clay_scene to see what it is "
            "now.",
            field="expect_stamp",
            # Not the ``"fix_arguments"`` a named field defaults to: the
            # stamp the client sent was the right shape and was true when it
            # read it. What is stale is its picture of the mesh, which is
            # exactly what this message already tells it to go and re-read.
            recovery="read_scene",
        )
    return expect_stamp, None


def _h_select_by(ctx: Any, session: Session, args: dict) -> dict:
    """Select elements by a seed or a parameter, through :data:`select.
    QUERIES`. See the tool's own description in ``agent_clay.tools``, and the
    module-level ``agent_clay_schema._QUERY_ARG_SCHEMAS``/
    ``agent_clay_validate._validate_query_arg`` for the one mapping from a
    query argument's name to what it is checked against -- ``select.py``
    itself holds no JSON-schema knowledge, by that module's own design.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure

    name = args.get("query")
    query = bsel.QUERIES.get(name)
    if query is None:
        return fail(
            f"query must be one of {', '.join(sorted(bsel.QUERIES))}.", field="query"
        )
    if doc.element_mode not in query.modes:
        # ``_in_mode_reason``'s own wording, not a paraphrase of it -- an
        # agent reading this refusal and a person reading the same query's
        # greyed-out menu row must never be told two different sentences for
        # the same gate.
        return fail(clay_ops._in_mode_reason(*query.modes)(doc), field="query")

    how = args.get("how", "replace")
    if how not in ("replace", "add", "subtract"):
        return fail("how must be 'replace', 'add' or 'subtract'.", field="how")

    expect_stamp = args.get("expect_stamp")
    if expect_stamp is not None:
        _, failure = _check_expect_stamp(doc, obj.uid, expect_stamp)
        if failure:
            return failure

    values: dict[str, Any] = {}
    for arg_name in query.args:
        if arg_name == "space":
            continue  # resolved below, never passed to a pure query function
        if arg_name not in args:
            if arg_name in _QUERY_OPTIONAL_ARGS:
                continue
            return fail(f"give a value for {arg_name!r}.", field=arg_name)
        value, failure = _validate_query_arg(arg_name, args[arg_name])
        if failure:
            return failure
        values[arg_name] = value

    if "direction" in values:
        # An agent reads translation/rotation/scale off clay_scene in world
        # space, so a direction it names ("up", "the way this object is
        # facing") is in that same frame -- and a face *normal* transforms by
        # the inverse-transpose of the object's matrix, not by its rotation
        # alone the moment the object carries a non-uniform scale. See
        # ``clay_geom_ops.local_direction``'s own docstring for the one that
        # is easy to get wrong.
        values["direction"] = clay_geom_ops.local_direction(
            obj, values["direction"], world=doc.world_matrix(obj.uid)
        )

    if name == "bounds":
        space = args.get("space", "world")
        if space not in ("world", "local"):
            return fail("space must be 'world' or 'local'.", field="space")
        lo = np.asarray(values["min"], dtype="f8")
        hi = np.asarray(values["max"], dtype="f8")
        # ``_q_bounds`` (this query's own ``run``) always measures against
        # the mesh's own local positions -- it has no ``positions=`` hook to
        # ask it for anything else -- so "world" is resolved here instead,
        # the way the module docstring's ``clay_select_by`` paragraph says:
        # passing ``clay_geom_ops.world_positions(obj)`` in for the mesh's
        # own local ``positions`` before the same box test ``_q_bounds`` and
        # ``select.faces_in_bounds`` already run.
        positions = (
            clay_geom_ops.world_positions(obj, world=doc.world_matrix(obj.uid))
            if space == "world"
            else None
        )
        pts = np.asarray(obj.mesh.positions if positions is None else positions, dtype="f8")
        if len(pts) == 0:
            sel = el.empty()
        else:
            inside = np.all((pts >= lo) & (pts <= hi), axis=1)
            sel = el.ElementSel(
                verts=np.flatnonzero(inside).astype("i4"),
                faces=bsel.faces_in_bounds(obj.mesh, lo, hi, positions=positions),
            )
    else:
        sel = query.run(obj.mesh, **values)

    current = doc.element_sel_of(obj.uid)
    doc.set_element_sel(obj.uid, el.combine(current, sel, how))

    return _json(
        {
            "uid": obj.uid,
            "query": name,
            "mode": doc.element_mode,
            "stamp": doc.mesh_stamp(obj.uid),
            "selected": _sel_counts(doc.element_sel_of(obj.uid)),
        }
    )


def _h_elements(ctx: Any, session: Session, args: dict) -> dict:
    """The read side of the element-selection tools -- paged raw indices, for
    the rarer moment an agent has to reason about which ones rather than how
    many. See :data:`agent_clay_schema.ELEMENT_PAGE_MAX`."""
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc

    kind = args.get("kind")
    if kind is not None and kind not in ("vertex", "edge", "face"):
        return fail("kind must be 'vertex', 'edge' or 'face'.", field="kind")

    offset = args.get("offset", 0)
    try:
        offset = int(offset)
    except (TypeError, ValueError):
        return fail("offset must be an integer.", field="offset")
    if offset < 0:
        return fail("offset must not be negative.", field="offset")

    limit = args.get("limit", ELEMENT_PAGE_DEFAULT)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return fail("limit must be an integer.", field="limit")
    if not (1 <= limit <= ELEMENT_PAGE_MAX):
        return fail(f"limit must be between 1 and {ELEMENT_PAGE_MAX}.", field="limit")

    if args.get("uid") is not None:
        obj, failure = _resolve_uid(doc, args)
        if failure:
            return failure
        targets = [obj]
    else:
        targets = [doc.by_uid(uid) for uid in doc.element_sel]

    field_name = {"vertex": "verts", "edge": "edges", "face": "faces"}.get(kind)
    rows = []
    for obj in targets:
        sel = doc.element_sel_of(obj.uid)
        row: dict[str, Any] = {
            "uid": obj.uid,
            "stamp": doc.mesh_stamp(obj.uid),
            "counts": _sel_counts(sel),
        }
        if field_name is not None:
            arr = getattr(sel, field_name)
            row["kind"] = kind
            row["total"] = len(arr)
            row["offset"] = offset
            row["indices"] = arr[offset : offset + limit].tolist()
        rows.append(row)

    return _json({"mode": doc.element_mode, "objects": rows})


@dataclass
class _OpCtx:
    """A sandboxed stand-in for the real app ``ctx``, handed to ``clay_ops.run``.

    ``clay_ops`` reaches ``ctx`` in five places now, not the four this
    docstring once counted: ``toast`` -- a module-level helper every refusal
    goes through -- ``getattr(ctx, "clay_view", None)`` in ``_frame`` (Frame
    Selection), ``getattr(ctx, "state", None)`` in ``_forget_manifold`` (the
    clay-08 manifold-cache pop), ``getattr(ctx, "inline", False)``/
    ``getattr(ctx, "gltfpack_exe", None)`` in decimate's own ``run``
    (``studio/modes/clay/ops.py``), and, since tranche 4's Blender ops
    (retopo/smart-unwrap/bake-detail) landed, ``getattr(ctx, "blender_
    timeout", None)`` in ``_blender_timeout``. The absent ``clay_view`` is
    what makes Frame Selection the no-op it should always have been for an
    agent with no viewport of its own; ``state`` is passed through for real
    because the manifold-cache pop is real work that still has to happen.
    See ``studio/modes/clay/agent/dispatch.py``'s own module docstring's ``clay_op``
    paragraph for why the real ``ctx`` used to be handed over unsandboxed.

    **``inline`` is a deliberate fourth departure, not a widening of the
    sandbox's own rule.** Interactively, ``decimate``'s gltfpack child process
    runs on a task thread -- the shape every blocking op in this codebase
    takes (``CLAUDE.md``'s "heavy work is always a child process" rule) -- so
    the button press returns immediately and the simplified mesh lands a
    frame or two later. An MCP ``clay_op`` call has no such later frame to
    land in: it returns once, from this call's own ``call()``, the identical
    problem ``studio/modes/clay/agent/dispatch.py``'s own module docstring
    already names for ``clay_export`` -- "there is no id to return from this
    call if this fold goes through it as written" -- and the same fix applies
    here. ``inline`` defaults ``True`` (unlike the interactive path, which
    never sets it and so reads ``False`` through ``getattr``) so
    ``decimate.run`` calls ``optimize.simplify_bytes`` synchronously, inside
    ``clay_ops.run``, before this handler returns -- keeping an agent's
    ``clay_op`` call to the one undo step every other op already is, at the
    cost of a real subprocess run on the frame thread for this one call: a
    deliberate one-shot cost, the same trade ``clay_export`` already makes
    for its own disk and database work, never the per-frame stall the
    task-thread split exists to prevent elsewhere. Retopologize/Smart
    Unwrap/Bake Detail take the identical ``inline`` branch in their own
    ``run`` functions, for the identical reason, against Blender rather than
    gltfpack.

    ``gltfpack_exe`` and ``blender_timeout`` are the two pieces of config
    those subprocesses need that the real app ``ctx`` carries under
    ``ctx.svc.config`` rather than as an attribute of its own (``ctx.svc.
    config.gltfpack_exe``, ``ctx.svc.config.rig_timeout`` -- see
    ``clay_ops._blender_timeout``'s own docstring for why the remesh job's
    timeout is the number reused rather than a new Clay-only field).
    :func:`_h_op` reads both through that chain, guarded against a test
    double with no ``svc`` at all, and hands them in here rather than leave
    each op's ``run`` to reach for ``ctx.svc`` itself the way no other op in
    the registry ever has to.
    """

    state: Any = None
    messages: list[str] = field(default_factory=list)
    inline: bool = True
    gltfpack_exe: Path | None = None
    blender_timeout: float | None = None

    def toast(self, message: str, level: str = "info") -> None:
        del level
        self.messages.append(message)


def _h_op(ctx: Any, session: Session, args: dict) -> dict:
    """Run one op by name. See ``studio/modes/clay/agent/dispatch.py``'s own module docstring's
    ``clay_op`` paragraph for the sandboxed proxy, and the "counts, never raw
    indices" rule this result follows: an agent does not need a
    200k-element array back from a ``select-all``, it needs to know that
    something changed and by how much.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    name = args.get("name")
    try:
        op = clay_ops.get(name)
    except KeyError:
        return fail(f"no op named {name!r}.", field="name", recovery="fix_arguments", op=name)
    # ``run`` itself returns False, silently, for a disabled op -- exactly the
    # answer that is useless to an agent with no menu to look at and read the
    # greyed row's tooltip from. Checked here, once, so the refusal names the
    # gate (``reason_for`` is only ever consulted once ``enabled`` has already
    # said no, matching ``clay_ops``'s own rule for the two never disagreeing).
    if not op.enabled(doc):
        # Not always ``recovery="switch_mode"``: ``op.reason`` covers several
        # unrelated gates (``_has_objects_reason``, ``_selection_reason``,
        # ``_has_two_visible_reason`` and the rest, see ``studio/modes/clay/ops.py``'s own
        # "reasons" section), and only some of them -- the ones built from
        # ``_in_mode_reason`` -- are about element mode at all. This handler
        # has no way to tell which gate fired from the string alone, so it
        # names the op rather than guessing a recovery that would be wrong
        # for "Select an object first."
        return fail(clay_ops.reason_for(op, doc), op=op.name)
    params = args.get("params")
    if params is None:
        params = {}
    elif not isinstance(params, dict):
        # The schema declares ``params`` an object; before this a non-dict
        # (a bare number, a list) reached ``clay_ops.run(proxy, doc, op,
        # **params)`` and failed there on ``**`` unpacking a non-mapping --
        # a real refusal, but the generic backstop in ``call()``'s own
        # ``except Exception``, logged as an unhandled failure rather than
        # named cleanly the way every other bad-shaped argument in this file
        # already is.
        return fail("params must be an object.", field="params")
    # Every value held to what ``op.params`` itself declares, before it ever
    # reaches ``clay_ops.run`` -- see ``agent_clay_validate._op_params_type_
    # refusal`` for the three crashes (2026-09-14 audit, docs-06) this closes.
    failure = _op_params_type_refusal(op, params)
    if failure:
        return failure
    # ``svc`` is absent on the test doubles several handler tests hand this
    # function -- guarded rather than a bare ``ctx.svc.config.gltfpack_exe``,
    # which would turn every one of those into the generic "failed
    # unexpectedly" backstop the moment an op that reads it (``decimate``) is
    # actually run against one. See ``_OpCtx``'s own docstring for why this
    # is filled here rather than left to the op itself to reach for.
    svc = getattr(ctx, "svc", None)
    gltfpack_exe = getattr(getattr(svc, "config", None), "gltfpack_exe", None)
    blender_timeout = getattr(getattr(svc, "config", None), "rig_timeout", None)
    proxy = _OpCtx(
        state=getattr(ctx, "state", None),
        gltfpack_exe=gltfpack_exe,
        blender_timeout=blender_timeout,
    )
    # Snapshotted by identity, before the op runs -- ``Mesh`` is ``eq=False``
    # and every op is ``Mesh -> Mesh`` (``document.py``'s own rule, the same
    # one ``set_mesh`` and ``mesh_stamp`` both rely on identity for), so
    # ``obj.mesh is before.get(obj.uid)`` after the call is a read of what the
    # op actually touched, not a second bookkeeping mechanism running beside
    # it. An object absent from ``before`` (an op like Duplicate makes one) is
    # "changed" too: there is no prior mesh for it to equal.
    before = {obj.uid: obj.mesh for obj in doc.objects}
    head = doc.history.head
    ran = clay_ops.run(proxy, doc, op, **params)
    changed = [
        {
            "uid": obj.uid,
            "stamp": doc.mesh_stamp(obj.uid),
            "faces": bm.face_count(obj.mesh),
            "verts": len(obj.mesh.positions),
            "selected": _sel_counts(doc.element_sel_of(obj.uid)),
        }
        for obj in doc.objects
        if before.get(obj.uid) is not obj.mesh
    ]
    return _json(
        {
            "op": op.name,
            "ran": ran,
            "pushed": doc.history.head != head,
            "element_mode": doc.element_mode,
            "messages": proxy.messages,
            "changed": changed,
        }
    )


def _parse_view_entry(entry: Any, valid_views: set[str]) -> tuple[str, dict, dict | None]:
    """One ``views[]`` entry -> ``(label, render_png kwargs, failure)``.

    A named axis/``three_quarter`` view maps straight to ``render_png``'s own
    ``view`` keyword. A free ``{yaw, pitch}`` pair maps to ``angles=(theta,
    phi)`` in radians: ``theta = radians(yaw)``, ``phi = radians(90 - pitch)``.
    That makes yaw 0 the front (``Camera.AXIS_VIEWS["front"]`` is theta=0) and
    +pitch look *down* from above (``AXIS_VIEWS["top"]`` is phi ~= 0, i.e.
    pitch +90) -- the sign that is wrong half the time and invisible until a
    picture is looked at, so it is written down here rather than trusted to
    memory.
    """
    if isinstance(entry, str):
        if entry not in valid_views:
            return (
                "",
                {},
                fail(
                    f"each view must be one of {', '.join(sorted(valid_views))} or "
                    "an object with yaw and pitch.",
                    field="views",
                ),
            )
        return entry, {"view": entry}, None
    if isinstance(entry, dict) and set(entry) == {"yaw", "pitch"}:
        try:
            yaw = float(entry["yaw"])
            pitch = float(entry["pitch"])
        except (TypeError, ValueError):
            return "", {}, fail("yaw and pitch must be numbers.", field="views")
        if not math.isfinite(yaw) or not (-89.0 <= pitch <= 89.0):
            return "", {}, fail("pitch must be between -89 and 89 degrees.", field="views")
        angles = (math.radians(yaw), math.radians(90.0 - pitch))
        return f"yaw={yaw:g},pitch={pitch:g}", {"angles": angles}, None
    return "", {}, fail("each view must be a name or an object with yaw and pitch.", field="views")


def _h_render(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc

    size_arg = args.get("size")
    if size_arg is None:
        size = 1024
    else:
        try:
            size = int(size_arg)
        except (TypeError, ValueError):
            return fail("size must be an integer.", field="size")
        # Refused, not clamped: the schema declares ``minimum: 64, maximum:
        # 2048``, and silently rounding a caller's own number into range
        # answered a request that was never made with no way to tell the
        # schema had lied about the ceiling it claimed to enforce.
        if not (64 <= size <= 2048):
            return fail("size must be between 64 and 2048.", field="size")

    view = args.get("view")
    views_arg = args.get("views")
    if view is not None and views_arg is not None:
        return fail("give either view or views, not both.", field="views")

    valid_views = set(Camera.AXIS_VIEWS) | {"three_quarter"}
    if views_arg is not None:
        if not isinstance(views_arg, list) or not views_arg:
            return fail("views must be a non-empty list.", field="views")
        entries = views_arg
    elif view is not None:
        if view not in valid_views:
            return fail(f"view must be one of {', '.join(sorted(valid_views))}.", field="view")
        entries = [view]
    else:
        entries = ["three_quarter"]

    parsed: list[tuple[str, dict]] = []
    for entry in entries:
        label, kwargs, failure = _parse_view_entry(entry, valid_views)
        if failure:
            return failure
        parsed.append((label, kwargs))

    grid = args.get("grid", False)
    # The schema declares this a boolean; a bare ``bool(grid)`` coercion
    # used to accept anything (a non-empty string, say) with no refusal at
    # all -- ``bool("off")`` is ``True``, which drew the grid an agent's own
    # value looked like it was asking not to see.
    if not isinstance(grid, bool):
        return fail("grid must be a boolean.", field="grid")

    shading = args.get("shading", RENDER_SHADINGS[0])
    if shading not in RENDER_SHADINGS:
        return fail(
            f"shading must be one of {', '.join(RENDER_SHADINGS)}.", field="shading"
        )
    if shading == "object_id" and grid:
        # The id pass (``ClayView.render_ids``) never draws a grid at all --
        # a grid line would be false colour with no uid behind it, corrupting
        # the very pixel counts this shading exists to produce. Refused here,
        # named at the field a caller can actually drop, rather than the grid
        # silently doing nothing or the id map silently going wrong.
        return fail("grid cannot be combined with shading 'object_id'.", field="grid")

    focus = args.get("focus")
    bounds = None
    if focus is not None:
        uids, failure = _resolve_uids(doc, focus, field="focus")
        if failure:
            return failure
        boxes = [
            box
            for box in (
                clay_geom_ops.world_box(doc.by_uid(u), world=doc.world_matrix(u)) for u in uids
            )
            if box is not None
        ]
        if boxes:
            lo = np.min([b[0] for b in boxes], axis=0)
            hi = np.max([b[1] for b in boxes], axis=0)
            bounds = (lo, hi)

    compare = args.get("compare")
    reference = None
    compare_mode = args.get("compare_mode", "beside")
    alpha = args.get("alpha", 0.5)
    if compare is not None:
        if shading == "object_id":
            # A compare reply is a picture-vs-picture comparison
            # (agent_refs.beside/overlay) with no room in its header for the
            # uid/colour/pixel table object_id exists to answer with, and a
            # reference was captured as an ordinary render in the first
            # place -- comparing it against flat id colours is not a
            # coherent question. Refused rather than silently dropping the
            # 'ids' table a caller would otherwise expect.
            return fail("shading 'object_id' cannot be combined with compare.", field="shading")
        reference = session.references.get(compare)
        if reference is None:
            return fail(f"no reference named {compare!r}.", field="compare")
        if compare_mode not in ("beside", "overlay"):
            return fail("compare_mode must be 'beside' or 'overlay'.", field="compare_mode")
        if compare_mode == "overlay":
            # The schema declares ``alpha`` a number, 0..1 -- a bare
            # ``float(alpha)`` only ever checked it converted, so a NaN, an
            # infinity, or a value past either end of the schema's own
            # declared range reached the blend with nothing having refused
            # it, the same unvalidated-number hole every other 0..1 knob in
            # this file (``clay_material``'s colour and metallic/roughness)
            # already closed with this same helper.
            alpha, failure = _validate_unit(alpha, "alpha")
            if failure:
                return failure
        if len(parsed) > 1:
            return fail("compare renders exactly one view.", field="views")
        if views_arg is None and view is None:
            # No view was asked for: default to the reference's own angle, so
            # the comparison is framed the way the picture being matched was.
            label = reference.view if reference.view in valid_views else "three_quarter"
            parsed = [(label, {"view": label})]
        # Refused, not clamped -- the same rule ``size`` above already
        # follows: this used to silently answer a smaller picture than the
        # one asked for, with nothing telling a caller the ceiling it named
        # was never the one actually enforced.
        if size > 1024:
            return fail("size must be 1024 or less when comparing to a reference.", field="size")
    else:
        total_pixels = len(parsed) * size * size
        if total_pixels > RENDER_PIXEL_BUDGET:
            return fail(
                f"{len(parsed)} view(s) at {size}x{size} would be "
                f"{total_pixels:,} pixels, over the {RENDER_PIXEL_BUDGET:,}-pixel "
                "render budget for one call -- ask for fewer or smaller views."
            )

    try:
        view_obj = _core()._view_for(ctx)
    except Exception:
        log.exception("agent render of a Clay document failed")
        return fail("That document could not be rendered; see the log.")

    ids_by_uid: dict[int, tuple[str, int]] = {}
    try:
        if shading == "object_id":
            pngs = []
            for _label, kwargs in parsed:
                png, rows = view_obj.render_ids(doc, size=size, bounds=bounds, **kwargs)
                pngs.append(png)
                # Summed across views rather than kept apart: the header has
                # one row per uid, not one per view, and "share one map"
                # (this tool's own description) is a promise about the
                # colour, not about collapsing a multi-view answer down to
                # whichever view happened to see the most of an object.
                for uid, hexcolor, px in rows:
                    _prev_hex, prev_px = ids_by_uid.get(uid, (hexcolor, 0))
                    ids_by_uid[uid] = (hexcolor, prev_px + px)
        else:
            pngs = [
                view_obj.render_png(
                    doc, size=size, grid=grid, bounds=bounds, shading=shading, **kwargs
                )
                for _label, kwargs in parsed
            ]
    except Exception:
        log.exception("agent render of a Clay document failed")
        return fail("That document could not be rendered; see the log.")

    # base64 costs 4 bytes for every 3 of input, rounded up: the frame budget
    # is checked against what actually crosses the wire, not the raw PNG
    # size. Applied identically to the compare path below and to the
    # ordinary multi-view path further down -- it used to run only on the
    # latter, so a beside/overlay sheet built from two large enough
    # references could reach `send_bytes` and fail there, past the point a
    # refusal could explain itself, exactly the failure mode this check
    # exists to head off.
    def _over_frame_budget(payload_pngs: list[bytes]) -> dict | None:
        b64_total = sum(((len(png) + 2) // 3) * 4 for png in payload_pngs)
        if b64_total > _protocol().MAX_FRAME - RENDER_FRAME_RESERVE:
            return fail(
                "This render is too large to send back in one reply frame; "
                "ask for fewer or smaller views."
            )
        return None

    if compare is not None:
        from . import refs as agent_refs

        if compare_mode == "beside":
            sheet = agent_refs.beside(reference.png, pngs[0], compare, "render", size=size)
        else:
            sheet = agent_refs.overlay(reference.png, pngs[0], alpha, size=size)

        over_budget = _over_frame_budget([sheet])
        if over_budget is not None:
            return over_budget

        import io

        from PIL import Image

        with Image.open(io.BytesIO(sheet)) as im:
            width, height = im.width, im.height

        # A second, private render: pngs[0] carries whatever shading the
        # caller asked for (lighting, wireframe...), and the IoU below wants
        # the flat, guaranteed-non-white object-id picture instead --
        # 'object_id' shading draws exactly that through render_ids, but is
        # refused combined with 'compare' above, so this takes that picture
        # for itself rather than the caller's. Never a refusal: the sheet
        # above is already worth returning whatever this finds, so any
        # failure here (including a moderngl one) reads as a null 'reason'
        # rather than losing the picture.
        try:
            from .....bench import metrics as bench_metrics

            ids_png, _ids_rows = view_obj.render_ids(
                doc, size=size, bounds=bounds, **parsed[0][1]
            )
            render_mask = bench_metrics.render_ids_mask(ids_png)
            silhouette = bench_metrics.compare_silhouette(reference.png, render_mask)
        except Exception:
            log.exception("agent render silhouette compare failed")
            silhouette = {"iou": None, "reason": "silhouette could not be measured; see the log."}

        session.last_render_png = sheet
        return ok(
            text(
                json.dumps(
                    {
                        "view": parsed[0][0],
                        "reference": compare,
                        "mode": compare_mode,
                        "width": width,
                        "height": height,
                        "silhouette": silhouette,
                    }
                )
            ),
            image_png(sheet),
        )

    over_budget = _over_frame_budget(pngs)
    if over_budget is not None:
        return over_budget

    session.last_render_png = pngs[0]
    header_obj: dict[str, Any] = {
        "views": [label for label, _ in parsed], "size": size, "grid": grid,
    }
    if shading == "object_id":
        header_obj["ids"] = [
            [uid, hexcolor, px] for uid, (hexcolor, px) in sorted(ids_by_uid.items())
        ]
    header = text(json.dumps(header_obj))
    # Deliberately not `_json` -- an image block has no JSON to duplicate,
    # and this header is already checked twice against `protocol.MAX_FRAME`
    # above (`RENDER_PIXEL_BUDGET`, `RENDER_FRAME_RESERVE`) before it leaves,
    # so a second copy in `structuredContent` would spend frame budget on
    # bytes nothing reads. See `_json`'s own docstring for the same claim.
    return ok(header, *(image_png(png) for png in pngs))


def _h_diagnose(ctx: Any, session: Session, args: dict) -> dict:
    """Report what is wrong with one or every visible mesh, and -- given
    ``select`` -- act on one finding the way the properties pane's own click
    handler does.

    ``clay_diagnose.Finding`` already carries the ``ElementSel`` that fixes
    each defect; before this, that was thrown away the moment it was turned
    into a JSON row, and an agent could describe a hole but never point at
    one. ``select`` closes that loop with the same three-call template
    ``studio/modes/clay/ui/props.py``'s ``_select_finding`` uses, for the same reason
    named there: the object selection must not be set by hand, because in an
    element mode it is *derived*, and the clear is what stops this finding's
    selection landing beside a stale one on another object.

    **Always reads the base mesh, never the evaluated one -- deliberately,
    unlike ``clay_render``.** A finding is a defect in the mesh's own
    topology (a hole, a non-manifold edge, a duplicate face) and its
    ``select`` selects that mesh's own vertices/edges/faces, both of which
    only make sense against the mesh an element edit would actually act on
    -- the base, exactly as ``document.py``'s own module docstring states.
    Running this against an evaluated mesh instead would report a hole a
    modifier stack has already closed (or invent one it opened), and a
    ``select`` naming indices into a mesh nothing in this document owns.
    ``clay_scene``'s own ``evaluated`` field is where the stack's own result
    is measured; this tool never touches it.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    uid = args.get("uid")
    if uid is None:
        targets = [obj for obj in doc.objects if obj.visible]
    else:
        obj, failure = _resolve_uid(doc, args, "uid")
        if failure:
            return failure
        targets = [obj]

    reports: dict[int, list] = {}
    report = []
    for obj in targets:
        # The 2026-09-19 audit's clay-39: ``findings`` past
        # ``ops_clean.MAX_CLEAN_CORNERS`` now raises ``OpError`` rather than
        # stalling. A whole-document call (``uid`` omitted) walks every
        # visible object in one loop, so letting that propagate would refuse
        # the *entire* call -- and every object's findings with it -- over
        # one oversized mesh among many legally-sized ones. Caught per object
        # and reported as a named skip instead, reusing the same row shape
        # every other finding already gets (``diagnose.too_large_finding``),
        # so the rest of the document is still answered for.
        try:
            rows = clay_diagnose.findings(obj.mesh)
        except el.OpError as error:
            rows = [clay_diagnose.too_large_finding(str(error))]
        reports[obj.uid] = rows
        report.append(
            {
                "uid": obj.uid,
                "name": obj.name,
                "clean": not rows,
                "findings": [
                    {"kind": row.kind, "label": row.label, "count": row.count, "mode": row.mode}
                    for row in rows
                ],
            }
        )

    select_arg = args.get("select")
    selected = None
    if select_arg is not None:
        # The schema declares this sub-object ``additionalProperties: False``
        # -- only ``uid`` and ``kind`` -- which nothing here checked before:
        # an extra key rode along unnoticed rather than being refused the way
        # the schema promises a client it will be.
        if not isinstance(select_arg, dict) or set(select_arg) - {"uid", "kind"}:
            return fail("select must be an object with only uid and kind.", field="select")
        sel_obj, failure = _resolve_uid(doc, select_arg, "uid")
        if failure:
            return failure
        kind = select_arg.get("kind")
        rows = reports.get(sel_obj.uid)
        if rows is None:
            # The object this call was asked to select in was not among this
            # call's own targets (a narrower ``uid`` was given, or it is
            # hidden) -- measured fresh rather than refused for a technicality
            # this call could answer on its own. Unlike the loop above, an
            # ``OpError`` here really is a refusal: a ``select`` this call
            # cannot compute has nothing to fall back to, so it takes the
            # ordinary refusal shape (``fail``) every other named-field
            # refusal on this surface already uses, rather than a new one.
            try:
                rows = clay_diagnose.findings(sel_obj.mesh)
            except el.OpError as error:
                return fail(str(error), field="select")
        row = next((r for r in rows if r.kind == kind), None)
        if row is None:
            available = sorted({r.kind for r in rows})
            return fail(
                f"{sel_obj.name!r} has no {kind!r} finding right now"
                + (f" -- it has {available}." if available else " -- it is clean."),
                field="select",
            )
        doc.set_element_mode(row.mode)
        doc.clear_element_sel()
        doc.set_element_sel(sel_obj.uid, row.sel)
        selected = {
            "uid": sel_obj.uid,
            "kind": row.kind,
            "mode": doc.element_mode,
            "stamp": doc.mesh_stamp(sel_obj.uid),
            "selected": _sel_counts(row.sel),
        }

    payload: dict[str, Any] = {"objects": report}
    # Document-level findings only on a whole-document call: they are about
    # how objects relate to each other, so asking them of a single named uid
    # would answer about objects the caller did not ask about.
    if uid is None:
        scene = clay_diagnose.scene_findings(list(doc.objects))
        if scene:
            payload["scene"] = [
                {"kind": row.kind, "label": row.label, "uids": list(row.uids)} for row in scene
            ]
    if selected is not None:
        payload["selected"] = selected
    # The 2026-09-18 audit's agents-03: a whole-document call (no uid) reports
    # findings for every visible object, so its reply grows with the
    # document's own size the same way clay_scene's does, and had the same
    # missing check. See validate._over_frame_budget's own docstring.
    over_budget = _over_frame_budget(payload)
    if over_budget is not None:
        return over_budget
    return _json(payload)


def _h_analyze(ctx: Any, session: Session, args: dict) -> dict:
    """Bounds, mass properties, ground contact, symmetry and pairwise
    distance/contact/overlap -- read-only, and selects nothing.

    ``uids`` given restricts both which objects are reported on and which
    pairs are computed among them, and switches ``floating`` off entirely --
    see :func:`~.analyze.analyze`'s own docstring for why a scoped call
    cannot answer that question. Omitted, every visible object takes part
    and ``floating`` is always present in the reply, even when empty.

    Passes ``doc=doc`` through to :func:`~.analyze.analyze`, which swaps
    every target's mesh for its evaluated one before measuring anything --
    bounds, area, volume, ground contact, symmetry, and pairwise distance/
    contact/overlap all then read what a mirror or an array modifier
    actually built, the same rule ``clay_scene``'s own ``bbox`` already
    follows.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc

    uids_arg = args.get("uids")
    if uids_arg is None:
        targets = [obj for obj in doc.objects if obj.visible]
        pairs_among = None
    else:
        uids, failure = _resolve_uids(doc, uids_arg, field="uids")
        if failure:
            return failure
        if not uids:
            return fail("uids must name at least one object.", field="uids")
        by_uid = {obj.uid: obj for obj in doc.objects}
        targets = [by_uid[uid] for uid in uids]
        pairs_among = uids

    contact_tol, failure = _validate_range(
        args.get("contact_tol", 0.001), "contact_tol", 0.0, 1.0
    )
    if failure:
        return failure
    near, failure = _validate_range(args.get("near", 0.05), "near", 0.0, 10.0)
    if failure:
        return failure
    symmetry_tol, failure = _validate_range(
        args.get("symmetry_tol", 0.002), "symmetry_tol", 0.0, 1.0
    )
    if failure:
        return failure

    result = clay_analyze.analyze(
        targets,
        doc=doc,
        pairs_among=pairs_among,
        contact_tol=contact_tol,
        near=near,
        symmetry_tol=symmetry_tol,
    )

    objects_out = [
        {
            "uid": row.uid,
            "name": row.name,
            "bounds": None
            if row.bounds is None
            else {"min": _round(row.bounds[0]), "max": _round(row.bounds[1])},
            "area": _round(row.area),
            "volume": None if row.volume is None else _round(row.volume),
            "closed": row.closed,
            "components": row.components,
            "ground": None
            if row.ground is None
            else {
                "min_y": _round(row.ground.min_y),
                "contact": row.ground.contact,
                "penetration": _round(row.ground.penetration),
            },
            "symmetry": _round(list(row.symmetry)),
        }
        for row in result.objects
    ]

    pairs_out = [
        {
            "uids": list(pair.uids),
            "distance": None if pair.distance is None else _round(pair.distance),
            "intersects": pair.intersects,
            "contact": pair.contact,
            "overlap": None
            if pair.overlap is None
            else {"volume": _round(pair.overlap.volume), "depth": _round(pair.overlap.depth)},
            "exact": pair.exact,
        }
        for pair in result.pairs
    ]

    payload: dict[str, Any] = {
        "objects": objects_out,
        "pairs": pairs_out,
        "tolerances": {"contact_tol": contact_tol, "near": near, "symmetry_tol": symmetry_tol},
    }
    if result.floating is not None:
        payload["floating"] = list(result.floating)
    if result.truncated:
        payload["truncated"] = True
    return _json(payload)


def _h_validate(ctx: Any, session: Session, args: dict) -> dict:
    """Advisory readiness checks against one :data:`readiness.PROFILES`
    entry -- see the tool's own description in ``agent_clay.tools`` for the
    full contract.

    A third read-only inspector beside ``clay_diagnose`` (mesh defects) and
    ``clay_analyze`` (placement facts): this one measures against a target's
    own import rules instead -- a triangle ceiling, a texture size, a
    material count and the rest, none of which is a defect in the mesh
    itself or a fact about where it sits. Pushes no undo step and selects
    nothing, the identical shape those two already hold to, so it needs no
    new paragraph in the undo enumeration.

    An empty (or, with ``visible_only``, all-hidden) document is not a
    refusal -- ``readiness.validate`` is handed a document with nothing to
    check and answers a ``"fail"`` status the same way it would for any
    other document that fails every check, because having nothing to check
    *is* the finding an agent asked for, not a malformed call.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc

    profile = args.get("profile", readiness.DEFAULT_PROFILE)
    if profile not in readiness.PROFILES:
        return fail(
            f"profile must be one of {', '.join(sorted(readiness.PROFILES))}.",
            field="profile",
        )

    visible_only = args.get("visible_only", True)
    # The schema declares this a boolean; checked the same way ``clay_render``'s
    # own ``grid`` and ``clay_batch``'s own ``rollback_on_error`` already are,
    # rather than a bare ``bool(...)`` coercion that would accept any truthy
    # value with no refusal at all.
    if not isinstance(visible_only, bool):
        return fail("visible_only must be a boolean.", field="visible_only")

    report = readiness.validate(doc, profile, visible_only=visible_only)
    return _json(
        {
            "profile": report.profile,
            "status": report.status,
            "checks": [
                {
                    "key": c.key,
                    "label": c.label,
                    "status": c.status,
                    "message": c.message,
                    "measured": c.measured,
                    "limit": c.limit,
                    "fix": c.fix,
                    "uids": list(c.uids or ()),
                }
                for c in report.checks
            ],
        }
    )


def _h_export(ctx: Any, session: Session, args: dict) -> dict:
    """Mint a finished model row from the document. See ``studio/modes/clay/agent/dispatch.py``'s
    own module docstring for the two departures from ``clay_mode.save_to``/
    ``export_asset`` this fold takes and why.

    **``engine``, tranche 7 (``dev/CLAY-PLAN.md``), names the export profile
    this row is written for** -- the collider naming an engine recognises,
    and the axis/scale convention an OBJ needs. It is checked against
    :data:`~.engines.ENGINES` so a bad value is refused by name rather than
    silently ignored, passed to ``clay_mode.build_asset`` (which applies it
    to the written file and never to the document's own names), and echoed
    back in the reply so a caller is not left guessing whether it took.
    Omitted, the person's own Settings choice stands: an agent that does not
    care about engines does not have to learn what they are.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    if tab.saving:
        return fail("A save for this document is already in progress.")
    if not any(obj.visible for obj in doc.objects):
        return fail("There is nothing visible to export.")

    engine_arg = args.get("engine")
    if engine_arg is not None and engine_arg not in engines.ENGINES:
        return fail(
            f"engine must be one of {', '.join(sorted(engines.ENGINES))}.", field="engine"
        )

    # ``clay_mode.camera_of`` reads *whatever tab the interactive viewport is
    # currently showing*, which is never this one -- an agent's document is
    # never on screen by definition. Passing ``tab.view`` straight through
    # avoids stamping the agent's document with the user's current camera.
    #
    # The chain itself -- ``to_model``, the GLB write, ``import_mesh``, and the
    # ``.rblk`` source sidecar (``save_clay_source``) -- is
    # ``clay_mode.build_asset`` now; see its docstring for why both this call
    # and ``export_asset``'s go through it, and ``studio/modes/clay/agent/dispatch.py``'s own
    # module docstring's first departure from ``save_to`` for why that
    # sidecar is the only copy this fold ever needs -- there is no second one
    # to keep of its own. There is no frame boundary an MCP call can hand the
    # encode across the way ``export_asset`` hands it to a task thread, so it
    # all runs right here, synchronously, before this handler returns -- a
    # deliberate one-shot cost, not the per-frame stall the task-thread split
    # exists to prevent.
    job_id = clay_mode.build_asset(
        ctx.svc, doc, title=tab.title, view=tab.view, engine=engine_arg
    )

    tab.job_id = job_id
    ctx.cache.invalidate()
    payload: dict[str, Any] = {"job_id": job_id}
    if engine_arg is not None:
        payload["engine"] = engine_arg
    return _json(payload)
