"""Clay's agent tool surface, the scene-structure handler family (tranche 3,
``dev/CLAY-PLAN.md``): ``clay_parent``, ``clay_group``, ``clay_ungroup``,
``clay_lock``, ``clay_tag``, ``clay_separate``, ``clay_set_origin``,
``clay_measure``, ``clay_checkpoint`` and ``clay_restore``.

A new family file, the same shape ``studio/modes/clay/agent/tools_modifiers.py`` landed in as
tranche 2's own family: one new vocabulary -- parenting, groups, locking,
tags, splitting an object apart, moving its origin, measuring it, and naming
a history position -- that shares no argument shape with the ten object-level
tools (``tools.py``) or the ten selection/ops/inspection ones (``tools_ops.py``).
See ``studio/modes/clay/agent/validate.py``'s own module docstring for why
every handler here reaches ``fail``/``ok``/``_json``/``Session``/``_tab``/the
shared validators through that module rather than through
``studio/modes/clay/agent/dispatch.py`` directly: this file has no import of
``dispatch.py`` at all, because none of these ten handlers ever needs
anything that lives only there.

**Three of the ten are exempt from the "one call, one undo step" rule, and
say so rather than pretend otherwise.** ``clay_checkpoint`` pushes no step at
all -- ``ClayDoc.set_checkpoint`` only writes a name into an in-memory dict,
never the undo stack -- the same shape ``clay_reference_add`` and the
selection tools already hold to (``dispatch.py``'s own "two families that
push none at all" paragraph). ``clay_restore`` **moves** the history head,
exactly the way ``clay_undo``/``clay_redo`` do (``ClayDoc.restore_checkpoint``
is ``step_history`` under the hood), which is why it joins them in
``schema.BATCH_EXCLUDED`` for the identical reason theirs gives: moving the
history head from inside a run that is about to fold into one step is
incoherent. Every other tool here is a genuine one-undo-step door, batchable,
the same as every tool in the fold before it.

**A checkpoint is session-only, on purpose.** ``ClayDoc.checkpoints`` is
never serialized -- the undo history itself is not saved either, so a name
recorded against a history position could not outlive the session that made
it even if the dict were written to disk. ``checkpoint_status``'s three real
answers (``"current"``, ``"reachable"``, ``"gone"``) are what
``clay_restore`` reports before it ever tries to move anything, and a
divergent push after a checkpoint was set -- a redo branch a later edit
discarded -- is exactly what turns ``"reachable"`` into ``"gone"``: not a
bug, the serial the name pointed at is simply no longer on any branch
``step_history`` can reach.

**Locking is not this file's own door.** ``ClayDoc.set_transform``/
``remove_object`` already refuse a locked object with ``OpError("<name> is
locked.")`` -- see ``document.py``'s own locking paragraph -- and
``tools.py``'s own ``_h_transform``/``_h_delete`` are what map that refusal
onto a field-named one. Of this file's two, only ``clay_separate`` reaches a
locking door (``ClayDoc.separate`` refuses a locked source the same way);
``clay_group`` does not, because ``ClayDoc.group`` reparents every member
with ``keep_world=True``, and ``set_parent``'s own docstring is explicit
that a ``keep_world`` reparent is exempt from ``_refuse_if_locked`` (it moves
nothing on screen) -- so a locked object can be grouped. Both handlers
(:func:`_h_group`, :func:`_h_separate`) catch the ``OpError`` and return it
through ``fail()``, the same as every other handler in this file, rather
than letting it through to ``call()``'s generic handling unwrapped.

(The 2026-09-23 audit, second run, finding clay-19: this paragraph
previously claimed the opposite of both -- that ``clay_group`` reaches a
locking door and that the ``OpError`` from either is let through unwrapped.
Neither held; corrected here.)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .....kernels.mesh import elements as el
from .....kernels.mesh import measure as clay_measure
from .....kernels.mesh import mesh as bm
from .....kernels.mesh import ops as clay_geom_ops
from .....kernels.mesh import separate as clay_separate
from .....kernels.mesh.elements import OpError
from .schema import MEASURE_KINDS, ORIGIN_MODES, SEPARATE_MODES
from .validate import (
    Session,
    _json,
    _label_top,
    _resolve_uid,
    _resolve_uids,
    _round,
    _scene_row,
    _tab,
    _validate_vec3,
    fail,
)

# --- clay_parent --------------------------------------------------------------


def _h_parent(ctx: Any, session: Session, args: dict) -> dict:
    """Reparent one object onto another (or make it a root), as one undo
    step -- ``ClayDoc.set_parent``'s own agent door. See that method's
    docstring for ``keep_world`` (default ``True``: the object's local TRS is
    recomputed so nothing on screen moves) and for the cycle refusal.

    ``parent`` is required to be *present*, not merely truthy: an absent key
    and an explicit ``null`` mean two different things (leave the parent
    argument off entirely versus deliberately clear it), so the schema
    declares no type for it at all -- ``set_parent`` itself does the real
    checking, this handler only tells an absent key apart from a given one.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure

    if "parent" not in args:
        return fail(
            "give a value for 'parent' -- pass null to make it a root.",
            field="parent",
        )
    parent_arg = args["parent"]
    parent_uid: int | None = None
    if parent_arg is not None:
        try:
            parent_uid = int(parent_arg)
        except (TypeError, ValueError):
            return fail("parent must be an integer uid or null.", field="parent")
        try:
            doc.by_uid(parent_uid)
        except KeyError:
            return fail(
                f"no object with uid {parent_uid}.",
                field="parent",
                recovery="read_scene",
                uids=[parent_uid],
            )

    keep_world = args.get("keep_world", True)
    if not isinstance(keep_world, bool):
        return fail("keep_world must be a boolean.", field="keep_world")

    try:
        changed = doc.set_parent(obj.uid, parent_uid, keep_world=keep_world)
    except OpError as error:
        return fail(str(error), field="parent")
    return _json({"changed": changed, **_scene_row(doc, obj)})


# --- clay_group / clay_ungroup -------------------------------------------------


def _h_group(ctx: Any, session: Session, args: dict) -> dict:
    """Parent every named object onto a new, mesh-less empty at their
    combined world bounds centre, as one undo step -- ``ClayDoc.group``'s own
    agent door. See the module docstring's "a group is parenting to an empty
    object" decision (``dev/CLAY-PLAN.md``): there is no second collection
    concept, so ungrouping is :func:`_h_ungroup` below, not a paired delete.

    A ``name`` collision is disambiguated automatically by ``ClayDoc.group``
    itself (``next_name``, the same suffixing every generator-placed object
    already gets) rather than refused the way ``clay_add_primitive``'s own
    ``name`` is -- this handler does not second-guess that; the row returned
    names whatever the group's empty actually ended up called.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    uids, failure = _resolve_uids(doc, args.get("uids"), field="uids")
    if failure:
        return failure
    if not uids:
        return fail("give at least one uid.", field="uids")

    name_arg = args.get("name")
    if name_arg is not None and (not isinstance(name_arg, str) or not name_arg.strip()):
        return fail("name must not be empty.", field="name")

    try:
        empty_obj = doc.group(uids, name=name_arg)
    except OpError as error:
        return fail(str(error), field="uids")
    return _json({"uid": empty_obj.uid, "members": uids, **_scene_row(doc, empty_obj)})


def _h_ungroup(ctx: Any, session: Session, args: dict) -> dict:
    """Remove a group's own empty, releasing its children onto its own
    parent -- world placement kept, exactly as ``ClayDoc.remove_object``
    already does for any removed parent. Refuses (naming ``field="uid"``,
    nothing pushed) an object with no children (nothing to ungroup) or one
    that carries geometry of its own (this door only ever removes an empty --
    an object that draws something is ``clay_delete``'s job, which does not
    pretend its children survived unaffected the way this tool's own name
    promises).
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure

    children = doc.children_of(obj.uid)
    if not children:
        return fail(f"{obj.name!r} has no children to ungroup.", field="uid")
    if bm.face_count(obj.mesh) != 0:
        return fail(
            f"{obj.name!r} has geometry of its own -- clay_ungroup only "
            "removes an empty group object. Use clay_delete for an object "
            "with a mesh.",
            field="uid",
        )

    try:
        doc.remove_object(obj.uid)
    except OpError as error:
        return fail(str(error), field="uid")
    return _json({"ungrouped": obj.uid, "released": children})


# --- clay_lock / clay_tag ------------------------------------------------------


def _h_lock(ctx: Any, session: Session, args: dict) -> dict:
    """Lock or unlock every named object, as one undo step. Not a locking
    door itself -- ``ClayDoc.set_props`` deliberately allows this even on an
    already-locked object (``document.py``'s own locking paragraph: a
    mistake made while locked has to be undoable by something).
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    uids, failure = _resolve_uids(doc, args.get("uids"), field="uids")
    if failure:
        return failure
    if not uids:
        return fail("give at least one uid.", field="uids")

    locked_arg = args.get("locked")
    if not isinstance(locked_arg, bool):
        return fail("locked must be a boolean.", field="locked")

    mark = doc.history.mark()
    changed_uids = [uid for uid in uids if doc.set_props(uid, locked=locked_arg)]
    doc.history.collapse_since(mark)
    _label_top(doc, mark, "Lock" if locked_arg else "Unlock")

    return _json({"locked": locked_arg, "uids": uids, "changed": changed_uids})


def _normalize_tag_list(value: Any) -> set[str]:
    """The same lower/strip/drop-empty rule ``document._normalize_tags``
    applies on the way into ``Obj.tags`` -- duplicated here, in two lines,
    rather than reached into across the kernel/studio boundary: what this
    computes is only a *set to union or subtract*, and ``set_props`` below
    normalizes the merged result the same way regardless, so a mismatch here
    could only ever under- or over-match a tag already spelled inconsistently
    -- never store anything this normalization did not also approve."""
    return {str(t).strip().lower() for t in value if str(t).strip()}


def _h_tag(ctx: Any, session: Session, args: dict) -> dict:
    """Add and/or remove tags on every named object, as one undo step. Not a
    locking door, same as :func:`_h_lock` and for the same reason.

    ``add``/``remove`` apply the same to every named object -- there is no
    per-object tag list in this call, the same "one params dict, several
    uids" shape ``clay_set_params`` already gives an agent for a repeated
    edit. Tags are normalized on the way in (see :func:`_normalize_tag_list`
    and ``document._normalize_tags``), so ``add=["Prop", "prop"]`` neither
    double-adds nor fights a tag already spelled ``prop``.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    uids, failure = _resolve_uids(doc, args.get("uids"), field="uids")
    if failure:
        return failure
    if not uids:
        return fail("give at least one uid.", field="uids")

    add_arg = args.get("add")
    remove_arg = args.get("remove")
    if add_arg is None and remove_arg is None:
        return fail("give add and/or remove.", field="add")

    def _string_list(value: Any, field: str) -> tuple[list[str] | None, dict | None]:
        if value is None:
            return [], None
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            return None, fail(f"{field} must be an array of strings.", field=field)
        return value, None

    add_list, failure = _string_list(add_arg, "add")
    if failure:
        return failure
    remove_list, failure = _string_list(remove_arg, "remove")
    if failure:
        return failure
    add_set = _normalize_tag_list(add_list or [])
    remove_set = _normalize_tag_list(remove_list or [])

    mark = doc.history.mark()
    rows: list[dict[str, Any]] = []
    for uid in uids:
        obj = doc.by_uid(uid)
        wanted = (set(obj.tags) | add_set) - remove_set
        doc.set_props(uid, tags=sorted(wanted))
        # A list of {uid, tags} rows, not a dict keyed by uid: JSON has no
        # integer keys, so a dict here would round-trip every uid through
        # ``json.dumps`` as a string ("1" rather than 1) -- exactly the trap
        # ``clay_scene``'s own ``objects`` list, and every other per-object
        # reply in this fold, is built as a list to avoid.
        rows.append({"uid": uid, "tags": list(doc.by_uid(uid).tags)})
    doc.history.collapse_since(mark)
    _label_top(doc, mark, "Tag")

    return _json({"objects": rows})


# --- clay_separate --------------------------------------------------------------

_SEPARATE_FUNCS = {
    "loose_parts": clay_separate.by_loose_parts,
    "material": clay_separate.by_material,
}


def _h_separate(ctx: Any, session: Session, args: dict) -> dict:
    """Split one object into several along ``by``, as one undo step --
    ``ClayDoc.separate``'s own agent door. ``by`` is one of
    :data:`schema.SEPARATE_MODES` -- ``loose_parts``, ``material`` (both take
    no further argument, they read the object's own mesh) or ``selection``
    (splits at the object's *current* face/vertex/edge selection, converted
    to faces the way every other selection-consuming tool in this fold
    already converts one -- see ``kernels.mesh.elements.convert``).

    Every piece keeps the source's parent, transform and modifier stack (the
    stack copied onto each, unevaluated -- see ``ClayDoc.separate``'s own
    docstring), and the source's generator is frozen, the same as every mesh
    edit's own freeze rule.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure

    by = args.get("by")
    if by not in SEPARATE_MODES:
        return fail(f"by must be one of {', '.join(SEPARATE_MODES)}.", field="by")

    try:
        if by == "selection":
            sel = doc.element_sel_of(obj.uid)
            if el.is_empty(sel):
                return fail(
                    f"{obj.name!r} has nothing selected to separate by -- "
                    "switch to an element mode and select some faces first.",
                    field="uid",
                )
            pieces = clay_separate.by_selection(obj.mesh, sel)
        else:
            pieces = _SEPARATE_FUNCS[by](obj.mesh)
        new_objs = doc.separate(obj.uid, pieces)
    except OpError as error:
        return fail(str(error))

    return _json(
        {"uids": [o.uid for o in new_objs], "objects": [_scene_row(doc, o) for o in new_objs]}
    )


# --- clay_set_origin -------------------------------------------------------------


def _origin_point(doc: Any, obj: Any, mode: str) -> np.ndarray | None:
    """The world point :func:`_h_set_origin` moves *obj*'s origin to for one
    of :data:`schema.ORIGIN_MODES` -- ``None`` when *mode* has nothing to
    measure (no geometry for ``bounds``/``base``, nothing selected for
    ``selection``).

    Measured off *obj*'s own base mesh, never the evaluated one: ``ClayDoc.
    set_origin`` shifts ``obj.mesh`` itself, so a bounds measured from a
    mirror or an array modifier's doubled geometry would move the pivot to a
    point the base mesh it actually bakes into does not agree describes it.
    """
    world = doc.world_matrix(obj.uid)
    if mode == "world":
        return np.zeros(3, dtype="f8")
    if mode in ("bounds", "base"):
        box = clay_geom_ops.world_box(obj, obj.mesh, world=world)
        if box is None:
            return None
        lo, hi = box
        center = (lo + hi) * 0.5
        if mode == "base":
            return np.array([center[0], lo[1], center[2]], dtype="f8")
        return center
    # mode == "selection"
    sel = doc.element_sel_of(obj.uid)
    if el.is_empty(sel):
        return None
    vsel = el.convert(obj.mesh, sel, "vertex")
    if len(vsel.verts) == 0:
        return None
    positions = clay_geom_ops.world_positions(obj, world=world)
    return positions[np.asarray(vsel.verts, dtype="i8")].mean(axis=0)


def _h_set_origin(ctx: Any, session: Session, args: dict) -> dict:
    """Move one object's origin (its local ``(0, 0, 0)``) to a world point,
    keeping the mesh and every child exactly where they are on screen --
    ``ClayDoc.set_origin``'s own agent door. Give exactly one of ``mode``
    (:data:`schema.ORIGIN_MODES` -- ``bounds``/``base``/``selection``/
    ``world``, computed by :func:`_origin_point`) or an explicit ``point``.

    Not a locking door, the same exemption ``ClayDoc.set_origin``'s own
    docstring states for itself: the mesh and every child stay exactly where
    they were on screen, only the pivot moves, so it costs nothing to allow
    even on a locked object.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure

    mode_arg = args.get("mode")
    point_arg = args.get("point")
    if (mode_arg is None) == (point_arg is None):
        return fail("give exactly one of mode or point.", field="mode")

    if point_arg is not None:
        point, failure = _validate_vec3(point_arg, "point")
        if failure:
            return failure
        point = np.asarray(point, dtype="f8")
    else:
        if mode_arg not in ORIGIN_MODES:
            return fail(f"mode must be one of {', '.join(ORIGIN_MODES)}.", field="mode")
        point = _origin_point(doc, obj, mode_arg)
        if point is None:
            return fail(
                f"{obj.name!r} has nothing to measure {mode_arg!r} from.", field="mode"
            )

    try:
        changed = doc.set_origin(obj.uid, point)
    except OpError as error:
        return fail(str(error), field="uid")
    return _json({"changed": changed, "point": _round(point), **_scene_row(doc, obj)})


# --- clay_measure ----------------------------------------------------------------


def _resolve_point(doc: Any, value: Any, field: str) -> tuple[np.ndarray | None, dict | None]:
    """*value* as a world-space point: ``[x, y, z]``, taken as already world
    space (however an agent got it there -- another object's own reported
    translation, an earlier measurement), or ``{"uid": <uid>}`` (that
    object's own world translation) / ``{"uid": <uid>, "vertex": <index>}``
    (one vertex of that object's *base* mesh, in world space -- the same
    mesh ``clay_select_elements``' own ``verts`` indexes, so a vertex an
    agent just selected can be measured from directly).
    """
    if isinstance(value, list):
        return _validate_vec3(value, field)
    if isinstance(value, dict):
        uid_val = value.get("uid")
        if uid_val is None:
            return None, fail(f"{field}.uid is required.", field=field)
        try:
            uid = int(uid_val)
            obj = doc.by_uid(uid)
        except (KeyError, TypeError, ValueError):
            return None, fail(
                f"{field} names no object with uid {uid_val!r}.",
                field=field,
                recovery="read_scene",
            )
        world = doc.world_matrix(obj.uid)
        if "vertex" in value:
            try:
                vertex = int(value["vertex"])
            except (TypeError, ValueError):
                return None, fail(f"{field}.vertex must be an integer.", field=field)
            n = len(obj.mesh.positions)
            if not (0 <= vertex < n):
                return None, fail(
                    f"{field}.vertex {vertex} is out of range for {obj.name!r} "
                    f"(0..{n - 1}).",
                    field=field,
                )
            positions = clay_geom_ops.world_positions(obj, world=world)
            return positions[vertex], None
        return np.asarray(world[:3, 3], dtype="f8"), None
    return None, fail(
        f"{field} must be [x, y, z] or an object naming uid (and, optionally, vertex).",
        field=field,
    )


def _h_measure(ctx: Any, session: Session, args: dict) -> dict:
    """Distance, angle, area or volume -- pure numbers, never a selection or
    an edit -- ``kernels.mesh.measure``'s own agent door.

    ``kind='distance'`` takes ``a``/``b`` (a :func:`_resolve_point` each);
    ``kind='angle'`` takes ``a``/``b``/``c`` (the angle at ``b``, between the
    rays ``b -> a`` and ``b -> c``). ``kind='area'`` takes ``uid`` and,
    optionally, ``faces`` (indices into the object's own base mesh -- the
    same ones ``clay_select_elements``/``clay_elements`` already use);
    omitted, it reads the object's *current* face selection, converted up
    from vertex/edge the way every selection-consuming tool already converts
    one. ``kind='volume'`` takes only ``uid``.

    Reads the *base* mesh throughout, like ``clay_diagnose`` and unlike
    ``clay_scene``/``clay_analyze`` -- deliberately: a vertex or face index
    an agent names (via ``a``/``b``/``faces``) is only meaningful against the
    mesh those indices actually come from, and every index-bearing tool in
    this fold (``clay_select_elements``, ``clay_elements``) already means the
    base. Measuring the evaluated mesh instead would silently disagree with
    whatever index an agent had just read.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc

    kind = args.get("kind")
    if kind not in MEASURE_KINDS:
        return fail(f"kind must be one of {', '.join(MEASURE_KINDS)}.", field="kind")

    if kind == "distance":
        a, failure = _resolve_point(doc, args.get("a"), "a")
        if failure:
            return failure
        b, failure = _resolve_point(doc, args.get("b"), "b")
        if failure:
            return failure
        value = clay_measure.distance(a, b)
        return _json({"kind": kind, "value": _round(value)})

    if kind == "angle":
        a, failure = _resolve_point(doc, args.get("a"), "a")
        if failure:
            return failure
        b, failure = _resolve_point(doc, args.get("b"), "b")
        if failure:
            return failure
        c, failure = _resolve_point(doc, args.get("c"), "c")
        if failure:
            return failure
        value = clay_measure.angle(a, b, c)
        return _json({"kind": kind, "value": _round(value)})

    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure
    world = doc.world_matrix(obj.uid)

    if kind == "volume":
        value = clay_measure.volume(obj.mesh, world=world)
        return _json({"kind": kind, "uid": obj.uid, "value": _round(value)})

    # kind == "area"
    faces_arg = args.get("faces")
    if faces_arg is not None:
        try:
            faces = [int(f) for f in faces_arg]
        except (TypeError, ValueError):
            return fail("faces must be a list of integers.", field="faces")
        n_faces = bm.face_count(obj.mesh)
        bad = [f for f in faces if not (0 <= f < n_faces)]
        if bad:
            return fail(
                f"face index {bad[0]} is out of range for this mesh (0..{n_faces - 1}).",
                field="faces",
            )
    else:
        sel = doc.element_sel_of(obj.uid)
        face_sel = el.empty() if el.is_empty(sel) else el.convert(obj.mesh, sel, "face")
        faces = face_sel.faces.tolist()
        if not faces:
            return fail(
                f"{obj.name!r} has no faces selected; give faces or select "
                "some first.",
                field="faces",
            )
    value = clay_measure.face_area(obj.mesh, faces, world=world)
    return _json({"kind": kind, "uid": obj.uid, "faces": len(faces), "value": _round(value)})


# --- clay_checkpoint / clay_restore ----------------------------------------------


def _h_checkpoint(ctx: Any, session: Session, args: dict) -> dict:
    """Name the current history position -- ``ClayDoc.set_checkpoint``'s own
    agent door. Pushes no undo step: see this module's own docstring for why.
    Re-setting an existing name overwrites it; there is only ever one
    position per name.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        return fail("name must not be empty.", field="name")
    doc.set_checkpoint(name)
    return _json({"name": name})


def _h_restore(ctx: Any, session: Session, args: dict) -> dict:
    """Move the history to a named checkpoint -- moving the head rather than
    pushing a step of its own, the same documented exception ``clay_undo``/
    ``clay_redo`` already are (see this module's own docstring).

    Refused (naming ``field="name"``, nothing moved) for a name this session
    never set, or one that is ``"gone"`` -- unreachable because a divergent
    edit since it was set evicted it from the done list or discarded it from
    the redo branch. A checkpoint already ``"current"`` moves nothing and is
    not a refusal: ``moved`` says so.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        return fail("name must not be empty.", field="name")

    status = doc.checkpoint_status(name)
    if status == "unknown":
        return fail(f"no checkpoint named {name!r}.", field="name", recovery="read_scene")
    if status == "gone":
        return fail(
            f"checkpoint {name!r} is no longer reachable -- a divergent edit "
            "since it was set has evicted or overwritten that history "
            "position.",
            field="name",
        )

    moved = doc.restore_checkpoint(name)
    return _json({"name": name, "status": doc.checkpoint_status(name), "moved": moved})
