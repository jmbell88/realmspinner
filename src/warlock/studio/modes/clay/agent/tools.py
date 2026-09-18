"""Clay's agent tool surface, the scene/creation handler family: the tools
that place, move, reshape, paint or remove a whole object -- ``clay_scene``,
``clay_add_primitive``, ``clay_add_figure``, ``clay_add_mesh``,
``clay_transform``, ``clay_set_params``, ``clay_material``, ``clay_boolean``,
``clay_delete`` and ``clay_rename``.

Split out of ``studio/modes/clay/agent/dispatch.py`` in the P4 restructure (``dev/RESTRUCTURE.md``).
The brief that started this split expected these handlers to sit under that
module's old "# --- the tools" banner; reading the file end to end found that
banner actually opens the *schema* builders (``tools()``/``instructions()``,
now ``studio/modes/clay/agent/schema.py``), while every handler -- this family included --
lived under the file's final "# --- dispatch" banner instead, alongside
``call`` itself. ``studio/modes/clay/agent/dispatch.py`` keeps ``call`` and the ``_HANDLERS`` table
that dispatches into this module; this family is what runs once that table
picks one of these ten names.

See ``studio/modes/clay/agent/validate.py``'s own module docstring for why every one of
these handlers reaches ``fail``/``ok``/``_json``/``Session``/``_tab`` and the
shared validators through that module rather than through ``studio/modes/clay/agent/dispatch.py``
directly: this file has no import of ``studio/modes/clay/agent/dispatch.py`` at all, because none
of these ten handlers ever needs anything that lives only there.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .....kernels.geom3d import gltf
from .....kernels.geom3d import math3d as m3
from .....kernels.mesh import diagnose as clay_diagnose
from .....kernels.mesh import document as bd
from .....kernels.mesh import mesh as bm
from .....kernels.mesh import ops as clay_geom_ops
from .....kernels.mesh import ops_boolean, presets, regen, shading
from .....kernels.mesh import primitives as bp
from .. import ops as clay_ops
from ..ui.panes import tools as pane_clay_tools
from .schema import MAX_MESH_FACES, MAX_MESH_VERTICES
from .validate import (
    _OBJECT_SELECTION_DERIVED_REFUSAL,
    Session,
    _json,
    _label_top,
    _params_shape_refusal,
    _quat_from_euler_xyz,
    _repaint,
    _resolve_uid,
    _resolve_uids,
    _round,
    _scene_row,
    _tab,
    _validate_params_values,
    _validate_unit,
    _validate_vec3,
    fail,
)


def _h_scene(ctx: Any, session: Session, args: dict) -> dict:
    del args
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    objects = [_scene_row(doc, obj) for obj in doc.objects]

    boxes = [
        clay_geom_ops.world_box(obj)
        for obj in doc.objects
        if obj.visible and clay_geom_ops.world_box(obj) is not None
    ]
    bounds = None
    if boxes:
        lo = np.min([b[0] for b in boxes], axis=0)
        hi = np.max([b[1] for b in boxes], axis=0)
        bounds = {
            "min": _round(lo),
            "max": _round(hi),
            "size": _round(hi - lo),
            "center": _round((lo + hi) * 0.5),
        }

    materials = [
        {
            "index": i,
            "name": m.name,
            "color": _round(list(m.base_color_factor)),
            "metallic": _round(m.metallic_factor),
            "roughness": _round(m.roughness_factor),
        }
        for i, m in enumerate(doc.materials)
    ]

    return _json(
        {
            "objects": objects,
            "selection": sorted(doc.selection),
            "element_mode": doc.element_mode,
            "dirty": doc.dirty,
            "object_count": len(doc.objects),
            "bounds": bounds,
            "materials": materials,
        }
    )


def _h_add_primitive(ctx: Any, session: Session, args: dict) -> dict:
    """Place one primitive, with everything validated before the first
    mutation so a refused call places nothing -- see the tool's own
    description in ``agent_clay.tools`` for the full argument list. Order:
    ``generator`` in the registry; ``params`` keys legal for it; the three
    TRS vectors well-formed; *then* the tab is resolved (minting one if the
    session owns none); *then* the object name (non-empty, not already
    taken) and the material index (in range) -- both of which need the
    document to answer.
    """
    generator = args.get("generator")
    if generator not in bp.GENERATORS:
        return fail(
            f"generator must be one of {', '.join(sorted(bp.GENERATORS))}.",
            field="generator",
        )

    params = args.get("params")
    if params is not None:
        if not isinstance(params, dict):
            return fail("params must be an object.", field="params")
        defaults = bp.GENERATORS[generator][0]
        unknown = sorted(set(params) - set(defaults))
        if unknown:
            return fail(
                f"unknown params {unknown} for {generator!r}; legal keys are "
                f"{sorted(defaults)}.",
                field="params",
            )
        # The schema declares each value ``number | array-of-numbers`` --
        # ``clay_set_params`` already checks every value against that shape
        # (``_validate_number_or_vec``) before it touches anything; this
        # tool never did, so a NaN or a string reached the generator
        # function directly and either poisoned a mesh's positions or, for
        # a non-numeric string, raised a bare ``TypeError`` that only
        # ``call()``'s generic backstop caught -- a logged "failed
        # unexpectedly" instead of a clean, field-named refusal. Run through
        # ``_validate_params_values`` rather than a bare loop over
        # ``_validate_number_or_vec`` so a bad value's refusal names *which*
        # key it was, not just "params" -- see that function's own docstring.
        failure = _validate_params_values(params, "params")
        if failure:
            return failure
        # ...and then each value against the *shape* that generator's own
        # default declares, which the wire schema cannot say -- see
        # ``_params_shape_refusal`` for the pyramid that crashed on a list.
        failure = _params_shape_refusal(params, defaults, "params", repr(generator))
        if failure:
            return failure

    translation = rotation_deg = scale = None
    if args.get("translation") is not None:
        translation, failure = _validate_vec3(args["translation"], "translation")
        if failure:
            return failure
    if args.get("rotation") is not None:
        rotation_deg, failure = _validate_vec3(args["rotation"], "rotation")
        if failure:
            return failure
    if args.get("scale") is not None:
        scale, failure = _validate_vec3(args["scale"], "scale")
        if failure:
            return failure

    # A boundary case surfaced by the ``changed`` audit, not missed: when this
    # session owns no document yet, ``_tab(..., create=True)`` below mints an
    # empty one and adopts it -- and the two refusals right after this can
    # still fire on that brand-new, empty document (an out-of-range
    # ``material`` needs no other object in the document to trigger). That
    # mint is deliberately not treated as ``changed=True`` here: it pushes no
    # undo step, adds no object and paints no face, so the three witnesses
    # this fold's own tests use to mean "the document moved" (history length,
    # ``dirty``, object count) read identically to a document that was never
    # minted at all -- the same reasoning ``fail()``'s own docstring gives
    # for why minting a Library row is not "the document" either.
    tab, failure = _tab(ctx, session, create=True)
    if failure:
        return failure
    doc = tab.doc

    obj_name = args.get("name")
    if obj_name is not None:
        # The schema declares ``name`` a string; a bare ``str(obj_name)``
        # coercion used to accept anything stringifiable with no refusal at
        # all -- the same unchecked-type hole ``clay_rename`` never had (it
        # already checks ``isinstance(name, str)`` for the identical field).
        if not isinstance(obj_name, str) or not obj_name.strip():
            return fail("name must not be empty.", field="name")
        if any(o.name == obj_name for o in doc.objects):
            return fail(f"an object is already named {obj_name!r}.", field="name")

    material_index = args.get("material")
    if material_index is not None:
        try:
            material_index = int(material_index)
        except (TypeError, ValueError):
            return fail("material must be a palette index.", field="material")
        if not (0 <= material_index < len(doc.materials)):
            return fail(
                f"material must be an index into the palette (0..{len(doc.materials) - 1}).",
                field="material",
            )

    mark = doc.history.mark()
    obj = pane_clay_tools.add_primitive(ctx, doc, generator)
    if params:
        merged = bp.clamp_params(generator, {**obj.params, **params})
        mesh = shading.auto_smooth(bp.GENERATORS[generator][1](**merged))
        was = {"params": dict(obj.params)}
        doc.set_generator_params(obj.uid, merged, mesh, was=was)
    if translation is not None or rotation_deg is not None or scale is not None:
        doc.set_transform(
            obj.uid,
            translation=translation,
            rotation=None if rotation_deg is None else _quat_from_euler_xyz(rotation_deg),
            scale=scale,
        )
    if obj_name is not None:
        doc.set_props(obj.uid, name=obj_name)
    if material_index is not None:
        _repaint(doc, [obj.uid], material_index)
    doc.history.collapse_since(mark)
    _label_top(doc, mark, f"Add {obj.name}")
    return _json(_scene_row(doc, obj))


def _h_add_figure(ctx: Any, session: Session, args: dict) -> dict:
    """Place a figure preset as one group, one undo step. See
    ``agent_clay.tools``'s description for ``translation``/``yaw``/``scale``/
    ``name_prefix``.

    Per part, with ``T`` the translation, ``s`` the uniform scale and
    ``q_y`` the yaw quaternion: ``t' = R_y(yaw) . (s . t) + T``,
    ``q' = q_y (x) q`` and ``s' = s . s_part``. Yaw and scale are applied
    to every part's *offset from the group origin*, not to each part in
    its own local frame -- a yawed figure turns where its limbs sit, it
    does not spin each limb about its own centre.
    """
    key = args.get("key")
    if key not in presets.ASSEMBLIES:
        return fail(f"key must be one of {', '.join(sorted(presets.ASSEMBLIES))}.", field="key")

    translation = None
    if args.get("translation") is not None:
        translation, failure = _validate_vec3(args["translation"], "translation")
        if failure:
            return failure

    yaw_deg = args.get("yaw")
    if yaw_deg is not None:
        try:
            yaw_deg = float(yaw_deg)
        except (TypeError, ValueError):
            return fail("yaw must be a number.", field="yaw")
        if not math.isfinite(yaw_deg):
            return fail("yaw must be finite.", field="yaw")

    scale = args.get("scale")
    if scale is not None:
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            return fail("scale must be a number.", field="scale")
        if not (math.isfinite(scale) and scale > 0):
            return fail("scale must be a positive, finite number.", field="scale")

    name_prefix = args.get("name_prefix")
    # Same unchecked-type hole as ``clay_add_primitive``'s own ``name``, fixed
    # the same way: the schema declares a string, so a non-string is refused
    # rather than silently coerced.
    if name_prefix is not None and not isinstance(name_prefix, str):
        return fail("name_prefix must be a string.", field="name_prefix")

    tab, failure = _tab(ctx, session, create=True)
    if failure:
        return failure
    doc = tab.doc

    mark = doc.history.mark()
    objs = pane_clay_tools.add_assembly(ctx, doc, key)

    if name_prefix:
        # Checked *after* placement, against the names ``add_assembly`` chose
        # (already run through ``pane_clay_tools._unique_name`` for whatever
        # this document already held) rather than predicted beforehand
        # against ``presets.build``'s raw part names -- re-deriving that
        # de-duplication here to guess its answer would be a second copy of
        # it, free to drift the day it changes. A collision is undone rather
        # than left half-renamed, so a refused prefix still places nothing.
        placed = {o.uid for o in objs}
        existing = {o.name for o in doc.objects if o.uid not in placed}
        prefixed = [f"{name_prefix}{o.name}" for o in objs]
        if len(set(prefixed)) != len(prefixed) or existing & set(prefixed):
            # A mutate-then-refuse path, audited rather than missed: the
            # figure's parts were already placed by ``add_assembly`` above,
            # so this refusal fires *after* a real mutation. ``doc.undo()``
            # on the line below is what keeps ``changed`` honestly ``False``
            # here (the wrapper's default, left unoverridden) rather than a
            # gap in the audit -- it reverses the very compound step
            # ``collapse_since`` just folded, so the object count, the undo
            # history's own length and ``doc.dirty`` all read exactly as they
            # did before this call started. See ``document.py``'s ``undo()``
            # and ``UndoStack.undo()`` for why that revert is exact rather
            # than approximate: the compound edit's own ``undo`` puts back
            # the very objects it added, by uid.
            doc.history.collapse_since(mark)
            doc.undo()
            return fail(
                f"{name_prefix!r} would collide with an existing object name.",
                field="name_prefix",
            )
        for obj, new_name in zip(objs, prefixed, strict=True):
            doc.set_props(obj.uid, name=new_name)

    if translation is not None or yaw_deg is not None or scale is not None:
        yaw_quat = m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), math.radians(yaw_deg or 0.0))
        s = 1.0 if scale is None else scale
        t = m3.vec3(*translation) if translation is not None else m3.vec3()
        for obj in objs:
            new_t = m3.quat_rotate(yaw_quat, obj.translation * s) + t
            new_q = m3.quat_mul(yaw_quat, obj.rotation)
            new_s = obj.scale * s
            doc.set_transform(obj.uid, translation=new_t, rotation=new_q, scale=new_s)

    doc.history.collapse_since(mark)
    label, _builder = presets.ASSEMBLIES[key]
    _label_top(doc, mark, f"Add {label}")
    return _json({"uids": [o.uid for o in objs], "objects": [_scene_row(doc, o) for o in objs]})


def _h_add_mesh(ctx: Any, session: Session, args: dict) -> dict:
    """Place one hand-built mesh, selected, as one undo step -- the door for
    geometry an agent computed itself rather than named by recipe. See
    ``agent_clay.tools``'s description for the exact shape of ``positions``/
    ``faces``/``uv``.

    Order, the same template :func:`_h_add_primitive` sets: ``positions`` and
    ``faces`` well-formed and within ``agent_clay_schema.MAX_MESH_VERTICES``/
    ``MAX_MESH_FACES``; ``uv`` (if given) nested exactly like ``faces``;
    the three TRS vectors well-formed; the mesh actually built and run
    through ``mesh.validate`` as a backstop -- *then* the tab is resolved
    (minting one if the session owns none), *then* the object name and
    material index, both of which need the document to answer. Nothing above
    that line needs a document, so nothing above it should wait for one, and
    a call that was always going to be refused should never have minted an
    empty tab just to be refused against -- see the module docstring's mint
    paragraph and :func:`_h_add_primitive`'s own comment on the identical
    boundary.

    **This object has no generator, from birth.** Every primitive keeps its
    generator's name and params until an element edit freezes them
    (``document.set_mesh``'s own docstring); a mesh handed over as raw
    coordinates has no recipe for ``clay_set_params`` to re-run, so it starts
    in exactly the state that freeze leaves an edited primitive in, rather
    than passing through it.
    """
    positions_arg = args.get("positions")
    if not isinstance(positions_arg, list) or not positions_arg:
        return fail("positions must be a non-empty array of [x, y, z].", field="positions")
    if len(positions_arg) > MAX_MESH_VERTICES:
        return fail(
            f"positions has {len(positions_arg)} entries, past the "
            f"{MAX_MESH_VERTICES:,} this tool accepts in one call.",
            field="positions",
        )
    positions: list[list[float]] = []
    for row in positions_arg:
        vec, failure = _validate_vec3(row, "positions")
        if failure:
            return failure
        positions.append(vec)
    n_positions = len(positions)

    faces_arg = args.get("faces")
    if not isinstance(faces_arg, list) or not faces_arg:
        return fail("faces must be a non-empty array of vertex-index loops.", field="faces")
    if len(faces_arg) > MAX_MESH_FACES:
        return fail(
            f"faces has {len(faces_arg)} entries, past the {MAX_MESH_FACES:,} "
            "this tool accepts in one call.",
            field="faces",
        )
    faces: list[list[int]] = []
    for fi, loop in enumerate(faces_arg):
        if not isinstance(loop, list):
            return fail(f"face {fi} must be an array of vertex indices.", field="faces")
        if len(loop) < 3:
            return fail(
                f"face {fi} has {len(loop)} corners; a face needs at least 3.", field="faces"
            )
        corners: list[int] = []
        for ci, idx in enumerate(loop):
            try:
                vi = int(idx)
            except (TypeError, ValueError):
                return fail(
                    f"face {fi} corner {ci} is {idx!r}, not a vertex index.", field="faces"
                )
            # Named down to the corner, not just the face: an agent that
            # miscounted one index in a thousand-face mesh cannot fix what
            # "a loop index is out of range" (mesh.validate's own wording)
            # does not say which of them it was.
            if not (0 <= vi < n_positions):
                return fail(
                    f"face {fi} corner {ci} indexes vertex {vi}, but positions "
                    f"has {n_positions} entries.",
                    field="faces",
                )
            corners.append(vi)
        faces.append(corners)

    uv_arg = args.get("uv")
    uv: list[list[list[float]]] | None = None
    if uv_arg is not None:
        if not isinstance(uv_arg, list):
            return fail("uv must be an array, nested exactly like faces.", field="uv")
        if len(uv_arg) != len(faces):
            return fail(
                f"uv has {len(uv_arg)} faces, but faces has {len(faces)}.", field="uv"
            )
        uv = []
        for fi, (loop, uv_loop) in enumerate(zip(faces, uv_arg, strict=True)):
            if not isinstance(uv_loop, list):
                return fail(f"uv[{fi}] must be an array of (u, v) pairs.", field="uv")
            # The single easiest mistake to make with this argument, so the
            # refusal says which face disagrees and by how much rather than
            # a bare "uv is the wrong shape".
            if len(uv_loop) != len(loop):
                return fail(
                    f"uv[{fi}] has {len(uv_loop)} corners, but face {fi} has "
                    f"{len(loop)}.",
                    field="uv",
                )
            corners_uv: list[list[float]] = []
            for ci, corner in enumerate(uv_loop):
                if not isinstance(corner, list) or len(corner) != 2:
                    return fail(f"uv[{fi}][{ci}] must be an array of 2 numbers.", field="uv")
                try:
                    u, v = float(corner[0]), float(corner[1])
                except (TypeError, ValueError):
                    return fail(
                        f"uv[{fi}][{ci}] must be an array of 2 numbers.", field="uv"
                    )
                if not (math.isfinite(u) and math.isfinite(v)):
                    return fail(f"uv[{fi}][{ci}] must be finite numbers.", field="uv")
                corners_uv.append([u, v])
            uv.append(corners_uv)

    translation = rotation_deg = scale = None
    if args.get("translation") is not None:
        translation, failure = _validate_vec3(args["translation"], "translation")
        if failure:
            return failure
    if args.get("rotation") is not None:
        rotation_deg, failure = _validate_vec3(args["rotation"], "rotation")
        if failure:
            return failure
    if args.get("scale") is not None:
        scale, failure = _validate_vec3(args["scale"], "scale")
        if failure:
            return failure

    mesh = bm.from_faces(positions, faces, uv)
    try:
        bm.validate(mesh)
    except ValueError as error:
        # Defence in depth, not the primary refusal path -- every rule
        # ``validate`` checks beyond what this handler already validated
        # above is about the CSR structure ``faces`` describes (``starts``
        # bracketing ``loops``), which is why this backstop names ``faces``
        # rather than leaving the ``ValueError`` to escape into ``call``'s
        # generic, field-blind "failed unexpectedly".
        return fail(f"not a valid mesh: {error}", field="faces")

    tab, failure = _tab(ctx, session, create=True)
    if failure:
        return failure
    doc = tab.doc

    obj_name = args.get("name")
    if obj_name is not None:
        if not isinstance(obj_name, str) or not obj_name.strip():
            return fail("name must not be empty.", field="name")
        if any(o.name == obj_name for o in doc.objects):
            return fail(f"an object is already named {obj_name!r}.", field="name")

    material_index = args.get("material")
    if material_index is not None:
        try:
            material_index = int(material_index)
        except (TypeError, ValueError):
            return fail("material must be a palette index.", field="material")
        if not (0 <= material_index < len(doc.materials)):
            return fail(
                f"material must be an index into the palette (0..{len(doc.materials) - 1}).",
                field="material",
            )

    if obj_name is None:
        # No generator name to base a default on, unlike ``add_primitive``'s
        # own ``pane_clay_tools.add_primitive`` -- "Mesh" is this door's own
        # base, counted up the same way ``ops.next_name`` already counts up
        # a duplicate.
        taken = {o.name for o in doc.objects}
        obj_name = "Mesh" if "Mesh" not in taken else clay_geom_ops.next_name("Mesh", taken)

    mark = doc.history.mark()
    obj = bd.Obj(uid=bd.new_uid(), name=obj_name, mesh=mesh, generator=None, params={})
    doc.add_object(obj)
    # Object mode only (the 2026-09-13 audit's clay-03) -- see
    # ``ClayDoc.add_objects``'s own comment for why selecting unconditionally
    # leaves the Properties panel naming one object while editing another.
    if doc.element_mode == "object":
        doc.select([obj.uid])
    if translation is not None or rotation_deg is not None or scale is not None:
        doc.set_transform(
            obj.uid,
            translation=translation,
            rotation=None if rotation_deg is None else _quat_from_euler_xyz(rotation_deg),
            scale=scale,
        )
    if material_index is not None:
        _repaint(doc, [obj.uid], material_index)
    doc.history.collapse_since(mark)
    _label_top(doc, mark, f"Add {obj.name}")

    # ``clay_diagnose.findings`` measures a mesh, not a live object, so it is
    # reused directly rather than routed back through ``_h_diagnose`` (which
    # resolves a uid, a tab and an optional ``select`` this call has no use
    # for). "Closed" is narrower than "clean": a flipped edge, a duplicate
    # face or an unused vertex is a real defect ``findings`` still reports,
    # but none of them is what stops ``clay_boolean`` -- only an open
    # boundary or a non-manifold edge does (``ops_boolean``'s own "needs
    # every selected object to be a closed solid" refusal), so those are the
    # two kinds this boolean is read from.
    rows = clay_diagnose.findings(obj.mesh)
    row = _scene_row(doc, obj)
    row["closed"] = not any(r.kind in ("hole", "nonmanifold") for r in rows)
    row["findings"] = [
        {"kind": r.kind, "label": r.label, "count": r.count, "mode": r.mode} for r in rows
    ]
    return _json(row)


def _h_transform(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure
    translation = args.get("translation")
    rotation_deg = args.get("rotation")
    scale = args.get("scale")
    if translation is None and rotation_deg is None and scale is None:
        return fail("give at least one of translation, rotation or scale.")
    # Every vector given validated before the single mutation below -- the
    # incident this closes: an unvalidated two-element ``translation`` once
    # reached ``set_transform`` and committed, and every later ``clay_scene``
    # raised trying to broadcast it into a 3x3 matrix, bricking introspection
    # for the whole document with no recovery but a blind undo. ``rotation``
    # only looked safe by accident -- ``_quat_from_euler_xyz``'s unpack
    # raises on the wrong length -- but let a NaN straight through
    # ``math.radians`` and out the other side as a poisoned quaternion; this
    # is the same "validate everything before the first mutation" rule
    # ``_h_add_primitive`` and ``_h_add_figure`` already follow.
    if translation is not None:
        translation, failure = _validate_vec3(translation, "translation")
        if failure:
            return failure
    if rotation_deg is not None:
        rotation_deg, failure = _validate_vec3(rotation_deg, "rotation")
        if failure:
            return failure
    if scale is not None:
        scale, failure = _validate_vec3(scale, "scale")
        if failure:
            return failure
    changed = doc.set_transform(
        obj.uid,
        translation=translation,
        rotation=None if rotation_deg is None else _quat_from_euler_xyz(rotation_deg),
        scale=scale,
    )
    return _json({"uid": obj.uid, "changed": changed})


def _h_set_params(ctx: Any, session: Session, args: dict) -> dict:
    """Set one object's generator params, or -- tranche 5's plural form --
    the same params on several at once: "make the wheels larger" is one call
    against every wheel's uid, not one call per wheel and one undo step per
    wheel. A persistent "these six objects are a wheel set" object was
    argued down in design review as more machinery than the ask needed; this
    is the cheap alternative -- a caller (or a script inside one) already
    holds the uids from ``clay_scene``, so paying params once per call is
    enough.

    ``uid`` and ``uids`` are both declared as plain optional properties in
    the schema (mirroring ``clay_reference_add``'s ``job_id``/``png_base64``
    pair) with only ``params`` required -- the exactly-one rule lives here,
    not in the schema, so it can run *after* ``uids`` has already been
    checked shape-and-membership sound. That ordering is deliberate, not
    incidental: ``tests/test_agent_schemas.py`` synthesises a call for every
    declared constraint on ``uids`` (wrong type, a non-integer element, an
    empty list) by taking a valid plural baseline and violating exactly one
    of those -- which leaves ``uid`` absent in every one of those cases -- so
    checking ``uids``' own shape before asking whether both were given is
    what makes those cases refuse naming ``field="uids"`` rather than the
    exactly-one rule's ``"uid"`` swallowing a more specific refusal.

    All-or-nothing: every named object is checked -- exists, still has a
    generator (not frozen by a topology edit), and accepts every key in
    ``params`` for *its own* generator -- before any of them is rebuilt. A
    per-object try/refuse loop would have let four cylinders retune and left
    a fifth, illegal box call half-applied; a caller with no way to inspect
    the document mid-call has no use for "some of these changed".
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc

    params = args.get("params")
    if not isinstance(params, dict) or not params:
        return fail("give at least one param to change.", field="params")

    uid_arg = args.get("uid")
    uids_arg = args.get("uids")
    uids: list[int] | None = None
    if uids_arg is not None:
        # Resolved (and, on ``[]``, refused) before the exactly-one check
        # below even looks at ``uid`` -- see this function's own docstring
        # on why that order is what gets the schema-walk's generated cases
        # naming ``field="uids"`` for free.
        uids, failure = _resolve_uids(doc, uids_arg, field="uids")
        if failure:
            return failure
        if not uids:
            return fail("give at least one uid.", field="uids")
    if (uid_arg is None) == (uids is None):
        return fail("give exactly one of uid or uids.", field="uid")

    if uids is None:
        obj, failure = _resolve_uid(doc, args)
        if failure:
            return failure
        uids = [obj.uid]

    objects = [doc.by_uid(u) for u in uids]
    # Whichever of the two the caller actually used, so a refusal below
    # points at an argument that is really in the call -- ``_resolve_uid``'s
    # own docstring holds itself to the same rule, and a plural call told to
    # fix its ``uid`` would be told to fix an argument it never sent.
    uid_field = "uids" if uids_arg is not None else "uid"

    # Pass 1: every object's own legality, checked in full before pass 2
    # rebuilds anything -- see the docstring's all-or-nothing paragraph.
    for obj in objects:
        if obj.generator is None:
            return fail(
                f"uid {obj.uid}: this object's topology has been edited, so "
                "it is no longer a generated shape -- there are no "
                "generator params left to set (see document.set_mesh's "
                "freeze).",
                field=uid_field,
                uids=[obj.uid],
            )
        defaults = bp.GENERATORS[obj.generator][0]
        unknown = sorted(set(params) - set(defaults))
        if unknown:
            return fail(
                f"unknown params {unknown} for {obj.generator!r} (uid "
                f"{obj.uid}); legal keys are {sorted(defaults)}.",
                field="params",
                uids=[obj.uid],
            )
        # Shape is per-generator, so unlike the finiteness sweep below this
        # cannot be hoisted out of the loop: one params dict may be aimed at
        # two objects whose generators want different shapes for the same
        # key name.
        failure = _params_shape_refusal(
            params, defaults, "params", f"{obj.generator!r} (uid {obj.uid})"
        )
        if failure:
            return failure
    # Every value validated before pass 2 touches anything -- ``bp.clamp_params``
    # only clamps the handful of keys it knows a floor or a relational limit
    # for, so a NaN or an infinity in a key it does not (or does, past the
    # clamp -- inf clamped against a finite ceiling is still inf) used to
    # sail straight through into the generator function and out the other
    # side as vertex positions, with nothing downstream ever checking a mesh
    # is made of finite numbers. Run through ``_validate_params_values``
    # rather than a bare loop over ``_validate_number_or_vec`` so the
    # refusal names *which* key was bad -- see that function's own
    # docstring. One check for every object: the params dict is the same
    # for all of them, and a value's own finiteness does not depend on which
    # generator reads it.
    failure = _validate_params_values(params, "params")
    if failure:
        return failure

    # Pass 2: nothing above can refuse anymore, so every object is rebuilt.
    # Folded into one undo step only when more than one object is
    # addressed. A single object's own ``set_generator_params`` call already
    # pushes exactly one step (it folds its own params-edit/mesh-edit pair
    # into one ``push`` -- see that method's docstring), so there is nothing
    # to fold, and wrapping it in ``mark()``/``collapse_since()`` unconditionally
    # the way ``_h_material`` does would relabel that already-one step "Set
    # Params", changing what the human's undo panel says for a call whose
    # behaviour never changed. ``_h_material`` can get away with an
    # unconditional label because it always pushes the same
    # ``add_material``/``_repaint`` pair regardless of how many uids it
    # paints; a single-uid ``clay_set_params`` has no such pair to fold.
    mark = doc.history.mark() if len(objects) > 1 else None
    rows = []
    for obj in objects:
        # Captured before anything below mutates ``obj.params`` -- the merge
        # two lines down edits it in place via a fresh dict, but
        # ``set_generator_params`` itself reassigns ``obj.params`` to the
        # very dict it is handed, so reading "before" off the object once
        # this call has run would compare a value against itself. See that
        # method's own docstring on why ``was`` is mandatory for this caller.
        was = {"params": dict(obj.params)}
        merged = bp.clamp_params(obj.generator, {**obj.params, **params})
        # ``regen.carry_over`` rather than a bare rebuild-and-``auto_smooth``:
        # this handler used to call ``shading.auto_smooth`` directly on
        # every rebuild, which re-derives shading from scratch and never
        # touched ``material`` at all -- so an object painted through
        # ``clay_material`` or given a hand-picked Shade Smooth by a prior
        # tool call came back grey and flat the moment its numbers changed
        # here. ``studio/modes/clay/ui/props.py``'s own rebuild carried shading (never
        # material) through the same two-case rule this fold now shares
        # rather than reimplements, which is exactly how the two doors
        # built two different meshes for the same edit before this.
        mesh = regen.carry_over(
            obj.mesh, bp.GENERATORS[obj.generator][1](**merged), material=obj.material
        )
        changed = doc.set_generator_params(obj.uid, merged, mesh, was=was)
        # Reported back rather than echoed: a caller that asked for
        # segments=2 learns here that clamp_params raised it to the
        # generator's own floor.
        rows.append(
            {"uid": obj.uid, "generator": obj.generator, "params": merged, "changed": changed}
        )
    if mark is not None:
        doc.history.collapse_since(mark)
        _label_top(doc, mark, "Set Params")

    payload = {"objects": rows, "changed": any(r["changed"] for r in rows)}
    # The single-uid shape (``uid``/``generator``/``params`` at the top
    # level, no ``objects`` list) predates the plural form, and
    # ``tests/test_agent_clay.py`` -- among them
    # ``test_set_params_clamps_and_reports_the_clamped_value_back`` and
    # ``test_a_profile_param_survives_the_whole_agent_door`` -- reads
    # ``payload["params"]``/``payload["uid"]`` directly, as does whatever
    # client is already out there driving today's tool. Rather than break
    # that shape, the one-object case mirrors its row at the top level too,
    # alongside the new ``objects`` list every caller can grow into.
    if len(rows) == 1:
        payload.update(rows[0])
    return _json(payload)


def _h_material(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    uids, failure = _resolve_uids(doc, args.get("uids"), field="uids")
    if failure:
        return failure
    if not uids:
        return fail("give at least one uid.", field="uids")
    color = args.get("color")
    if not isinstance(color, list) or len(color) not in (3, 4):
        return fail("color must be an array of 3 or 4 numbers, 0..1.", field="color")
    # Per component through ``_validate_unit`` rather than the old bare
    # ``isinstance(c, int | float)`` -- that check let ``float("nan")``
    # through (NaN *is* a float) straight into the palette, and
    # ``metallic``/``roughness`` had no check at all beyond the bare
    # ``float()`` conversion below. The same unvalidated-number hole
    # ``clay_transform`` had for its translation, one tool over.
    rgba = []
    for c in color:
        value, failure = _validate_unit(c, "color")
        if failure:
            return failure
        rgba.append(value)
    rgba = tuple(rgba)
    if len(rgba) == 3:
        rgba = (*rgba, 1.0)
    metallic, failure = _validate_unit(args.get("metallic", 0.0), "metallic")
    if failure:
        return failure
    roughness, failure = _validate_unit(args.get("roughness", 0.6), "roughness")
    if failure:
        return failure
    name_arg = args.get("name")
    # The schema declares ``name`` a string; a bare ``str(name_arg or "")``
    # coercion used to accept anything stringifiable with no refusal at all
    # -- the same hole ``clay_add_primitive``'s own ``name`` had, fixed the
    # same way ``clay_rename`` already checks its identical field.
    if name_arg is not None and not isinstance(name_arg, str):
        return fail("name must be a string.", field="name")
    material = gltf.Material(
        name=name_arg or "",
        base_color_factor=rgba,
        metallic_factor=metallic,
        roughness_factor=roughness,
    )

    # One material for the whole call -- never one per object -- folded into
    # one undo step the way ``add_material_and_assign`` folds its own pair,
    # so one tool call is one Ctrl+Z.
    mark = doc.history.mark()
    index = doc.add_material(material)
    _repaint(doc, uids, index)
    doc.history.collapse_since(mark)
    _label_top(doc, mark, "Set Material")
    return _json(
        {
            "index": index,
            "uids": uids,
            "color": list(rgba),
            "metallic": material.metallic_factor,
            "roughness": material.roughness_factor,
        }
    )


def _h_boolean(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    # This op writes object uids straight into ``doc.selection`` a few lines
    # down -- harmless before this change, because an agent could never leave
    # object mode at all. The moment element mode is reachable that write can
    # manufacture "selected with nothing selected inside it", the state
    # ``document.py``'s module docstring says the derived-selection invariant
    # forbids in an element mode. Refused rather than auto-switched: silently
    # changing the document's mode under a call that did not ask for it is
    # the hidden state change this codebase refuses instead of guessing at.
    if doc.element_mode != "object":
        return fail(_OBJECT_SELECTION_DERIVED_REFUSAL, recovery="switch_mode")
    kind = args.get("kind")
    if kind not in ops_boolean.KINDS:
        return fail(f"kind must be one of {', '.join(ops_boolean.KINDS)}.", field="kind")
    try:
        wanted = [int(u) for u in args.get("uids") or []]
    except (TypeError, ValueError):
        return fail("uids must be a list of integers.", field="uids")
    # ``_union``'s own shape, generalised over the three kinds: the targets
    # are read in the document's own object order, so "first" means the
    # target's place in that order -- never the order this list happened to
    # name them in. See ``ops_boolean.KINDS``' own docstring for why that is
    # the rule for a difference, where the order changes the answer.
    #
    # Derived by walking ``doc.objects`` against *wanted* rather than by
    # writing ``doc.select(wanted)`` first and re-reading it: the write was
    # only ever a way to get that ordering, and doing it here put a mutation
    # ahead of the count check below -- so a boolean refused for naming too
    # few visible objects left the person's own selection overwritten by a
    # call that changed nothing else. Walking the list gives the identical
    # answer (a uid naming no object simply never matches) with nothing
    # written, which is what lets the refusal below be honest that the
    # document did not move. The selection this op does mean to leave behind
    # is set once, at the end, to the survivor.
    keep = {int(u) for u in wanted}
    targets = [obj.uid for obj in doc.objects if obj.uid in keep and obj.visible]
    if len(targets) < 2:
        return fail(
            "Select at least two visible objects.",
            field="uids",
            uids=targets,
        )
    mesh = ops_boolean.boolean([doc.by_uid(u) for u in targets], kind)
    doc.join_objects(targets[0], mesh, targets[1:])
    # clay-08 (2026-09-08 audit), the same pop ``clay_ops._join``/``_union``
    # make: the objects a boolean absorbs must not leave their manifold-check
    # cache entries pinned alive under a uid nothing owns any more.
    clay_ops._forget_manifold(ctx, targets[1:])
    doc.select([targets[0]])
    return _json({"uid": targets[0], "kind": kind})


def _h_delete(ctx: Any, session: Session, args: dict) -> dict:
    """Remove objects by uid, directly -- never through ``clay_op``'s own
    Delete row.

    ``clay_op`` ``delete`` acts on the document's *selection*, and in a face
    or edge mode ``selection.delete_selected`` never removes an object at
    all -- so an agent that had switched element mode (perhaps from an
    earlier ``clay_op`` call) would see a silent no-op where it asked for a
    deletion. Working from the uids given, in whatever element mode the
    document happens to be in, is what keeps "delete these objects" meaning
    that regardless.

    **Deliberately not given the same element-mode refusal as ``clay_select``
    and ``clay_boolean``.** Those two *write* object uids straight into
    ``doc.selection``, which in an element mode can manufacture "selected
    with nothing selected inside it" -- the state the derived-selection
    invariant forbids. This handler never does: :meth:`~.document.ClayDoc.
    remove_object` only ever *removes* a uid from ``selection`` (and from
    ``element_sel``, on the same line), and removing an entry from a set
    cannot put it into the forbidden state that only a write can create.
    Refusing here would be refusing a call that was never capable of the
    defect the refusal exists to prevent.
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

    mark = doc.history.mark()
    for uid in uids:
        doc.remove_object(uid)
    doc.history.collapse_since(mark)
    _label_top(doc, mark, "Delete")
    # clay-08 (2026-09-08 audit): an object that leaves ``doc.objects`` must
    # not leave its manifold-check cache entry pinning a whole ``Mesh`` alive
    # under a uid nobody owns -- the same pop ``clay_ops._join``/``_union``
    # make when they absorb objects, here for the direct-delete path
    # ``clay_op``'s own Delete row does not take.
    clay_ops._forget_manifold(ctx, uids)
    return _json({"deleted": uids})


def _h_rename(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure
    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        return fail("name must not be empty.", field="name")
    if any(other.uid != obj.uid and other.name == name for other in doc.objects):
        return fail(f"an object is already named {name!r}.", field="name")
    doc.set_props(obj.uid, name=name)
    return _json({"uid": obj.uid, "name": name})
