"""Clay's agent tool surface, the scene/creation handler family: the tools
that place, move, reshape, paint or remove a whole object -- ``clay_scene``,
``clay_add_primitive``, ``clay_add_mesh``, ``clay_transform``,
``clay_set_params``, ``clay_material``, ``clay_delete`` and ``clay_rename``.

Split out of ``studio/modes/clay/agent/dispatch.py`` in the P4 restructure (``dev/RESTRUCTURE.md``).
The brief that started this split expected these handlers to sit under that
module's old "# --- the tools" banner; reading the file end to end found that
banner actually opens the *schema* builders (``tools()``/``instructions()``,
now ``studio/modes/clay/agent/schema.py``), while every handler -- this family included --
lived under the file's final "# --- dispatch" banner instead, alongside
``call`` itself. ``studio/modes/clay/agent/dispatch.py`` keeps ``call`` and the ``_HANDLERS`` table
that dispatches into this module; this family is what runs once that table
picks one of these eight names.

See ``studio/modes/clay/agent/validate.py``'s own module docstring for why every one of
these handlers reaches ``fail``/``ok``/``_json``/``Session``/``_tab`` and the
shared validators through that module rather than through ``studio/modes/clay/agent/dispatch.py``
directly: this file has no import of ``studio/modes/clay/agent/dispatch.py`` at all, because none
of these eight handlers ever needs anything that lives only there.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from .....kernels.mesh import document as bd
from .....kernels.mesh import mesh as bm
from .....kernels.mesh import ops as clay_geom_ops
from .....kernels.mesh import primitives as bp
from .....kernels.mesh import regen, shading
from ..ui.panes import tools as pane_clay_tools
from .schema import MAX_MESH_FACES, MAX_MESH_VERTICES, MAX_NAME_LENGTH, MAX_PALETTE
from .validate import (
    Session,
    _json,
    _label_top,
    _name_length_refusal,
    _over_frame_budget,
    _params_shape_refusal,
    _quat_from_euler_xyz,
    _repaint,
    _resolve_uid,
    _resolve_uids,
    _round,
    _scene_row,
    _tab,
    _validate_params_values,
    _validate_scale,
    _validate_translation,
    _validate_unit,
    _validate_vec3,
    _whole_number,
    fail,
)


def _h_scene(ctx: Any, session: Session, args: dict) -> dict:
    # The 2026-10-03 audit's clay-22: this took no arguments, so a document
    # whose whole-scene reply passed the frame budget (about 7,300 objects, one
    # very long name, a big ``clay_separate``) answered only "narrow the
    # request (fewer objects, or a single uid)" -- a narrowing the tool did not
    # have, and the document's one read refused forever. ``offset``/``limit``
    # page the object rows; everything else in the reply stays the whole
    # document's own (``object_count`` is the total, ``bounds`` spans every
    # visible object), so a page reads like the whole answer with fewer rows.
    offset = args.get("offset", 0)
    try:
        offset = int(offset)
    except (TypeError, ValueError, OverflowError):
        return fail("offset must be an integer.", field="offset")
    if offset < 0:
        return fail("offset must not be negative.", field="offset")
    limit = args.get("limit")
    if limit is not None:
        try:
            limit = int(limit)
        except (TypeError, ValueError, OverflowError):
            return fail("limit must be an integer.", field="limit")
        if limit < 1:
            return fail("limit must be at least 1.", field="limit")
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    page = doc.objects[offset:] if limit is None else doc.objects[offset : offset + limit]
    objects = [_scene_row(doc, obj) for obj in page]

    boxes = [
        box
        for box in (
            clay_geom_ops.world_box(obj, obj.mesh, world=doc.world_matrix(obj.uid))
            for obj in doc.objects
            if obj.visible
        )
        if box is not None
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
            # The 2026-10-07 audit's clay-86: a slot carrying a base-colour
            # texture (painted in Inker and pulled back) reads here as its
            # white factor alone, so an agent reused it as a colour or
            # painted a plain slot over a texture it could not see. "nearest"
            # is the pixel-art sampler flag the texture travels with.
            "textured": m.base_color is not None,
            "nearest": bool(m.nearest),
        }
        for i, m in enumerate(doc.materials)
    ]

    payload = {
        "objects": objects,
        "selection": sorted(doc.selection),
        "element_mode": doc.element_mode,
        "dirty": doc.dirty,
        "object_count": len(doc.objects),
        "bounds": bounds,
        "materials": materials,
    }
    # The 2026-09-18 audit's agents-03: unlike clay_render, whose own payload
    # is checked against protocol.MAX_FRAME before it leaves, clay_scene's
    # reply grows with the document's own object count and had no ceiling at
    # all -- a document of ~22,000 primitives encodes past MAX_FRAME and used
    # to reach send_bytes and fail there, rather than being refused with an
    # explanation. See _over_frame_budget's own docstring.
    over_budget = _over_frame_budget(
        payload,
        hint=(
            "read the objects in pages with 'offset' and 'limit' "
            "(object_count says how many there are)"
        ),
    )
    if over_budget is not None:
        return over_budget
    return _json(payload)


def _h_add_primitive(ctx: Any, session: Session, args: dict) -> dict:
    """Place one primitive, with everything validated before the first
    mutation so a refused call places nothing -- see the tool's own
    description in ``agent_clay.tools`` for the full argument list. Order:
    ``generator`` in the registry; ``params`` keys legal for it; whether
    ``params`` actually *builds* (the 2026-09-26 audit's clay-document-02 --
    see the comment at that check for why this has to run before the tab is
    even resolved); the three TRS vectors well-formed; *then* the tab is
    resolved (minting one if the session owns none); *then* the object name
    (non-empty, not already taken) and the material index (in range) -- both
    of which need the document to answer.
    """
    generator = args.get("generator")
    # ``isinstance`` checked first: the 2026-09-26 audit's clay-agent-tools-06
    # -- ``x not in a_dict`` hashes ``x``, and a list or object argument
    # raises a bare, unhashable ``TypeError`` that only ``call()``'s generic
    # "failed unexpectedly" backstop caught, instead of this refusal naming
    # ``field="generator"`` the way an unknown *string* already does.
    if not isinstance(generator, str) or generator not in bp.CLAY_GENERATOR_NAMES:
        return fail(
            f"generator must be one of {', '.join(sorted(bp.CLAY_GENERATOR_NAMES))}.",
            field="generator",
        )

    # The 2026-10-07 audit's clay-33: the length ceiling is judged here, before
    # anything is built and before the tab is resolved, so a name that was
    # always going to be refused neither mints an empty document nor lands and
    # pushes every later ``clay_scene`` page past the frame.
    failure = _name_length_refusal(args.get("name"))
    if failure:
        return failure

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
        # ``_params_shape_refusal`` for the generator that crashed on a list.
        failure = _params_shape_refusal(params, defaults, "params", repr(generator))
        if failure:
            return failure

    # The 2026-09-26 audit's clay-document-02 (agent-door half; flooring the
    # extents themselves inside ``stairs``/``doorway`` is a
    # kernels/mesh fix, not this one): every check above is about a value's
    # *type* and *shape*, none of them about whether the generator can
    # actually build it -- ``stairs``/``doorway`` degenerate to a face with
    # fewer than 3 corners, which ``clamp_params`` does not floor.
    # Built here, before the tab is resolved or the default object is
    # placed, so a generator's own refusal is exactly as clean as a shape
    # refusal above it: this used to run *after* ``mark()`` had opened the
    # undo gesture and the default-params object had already been added,
    # so a raise here left that object standing in the document and the
    # gesture's own ``_open_gestures`` counter permanently incremented --
    # nothing downstream ever reached ``collapse_since`` to close it.
    merged = mesh = None
    if params:
        merged = bp.clamp_params(generator, {**defaults, **params})
        try:
            mesh = shading.auto_smooth(bp.GENERATORS[generator][1](**merged))
        except (ValueError, ArithmeticError) as error:
            return fail(str(error), field="params")

    translation = rotation_deg = scale = None
    if args.get("translation") is not None:
        translation, failure = _validate_translation(args["translation"], "translation")
        if failure:
            return failure
    if args.get("rotation") is not None:
        rotation_deg, failure = _validate_vec3(args["rotation"], "rotation")
        if failure:
            return failure
    if args.get("scale") is not None:
        scale, failure = _validate_scale(args["scale"], "scale")
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
            # OverflowError: the 2026-09-26 audit's clay-agent-tools-09 --
            # ``int(float("inf"))`` raises it, and this tuple did not name
            # it, so an infinite ``material`` reached ``call()``'s generic
            # backstop instead of this refusal.
            material_index = int(material_index)
        except (TypeError, ValueError, OverflowError):
            return fail("material must be a palette index.", field="material")
        if not (0 <= material_index < len(doc.materials)):
            return fail(
                f"material must be an index into the palette (0..{len(doc.materials) - 1}).",
                field="material",
            )

    mark = doc.history.mark()
    obj = pane_clay_tools.add_primitive(ctx, doc, generator)
    if params:
        # ``merged``/``mesh`` were already built (and any refusal already
        # returned) above, before this gesture opened -- see this
        # function's own clay-document-02 comment for why.
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
    # The 2026-10-07 audit's clay-33: see ``_h_add_primitive``'s identical check.
    failure = _name_length_refusal(args.get("name"))
    if failure:
        return failure

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
                # OverflowError: the 2026-09-26 audit's clay-agent-tools-09 --
                # ``int(float("inf"))`` raises it, uncaught here before this
                # fix, past this refusal into ``call()``'s generic backstop.
                vi = int(idx)
            except (TypeError, ValueError, OverflowError):
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
                except (TypeError, ValueError, OverflowError):
                    return fail(
                        f"uv[{fi}][{ci}] must be an array of 2 numbers.", field="uv"
                    )
                if not (math.isfinite(u) and math.isfinite(v)):
                    return fail(f"uv[{fi}][{ci}] must be finite numbers.", field="uv")
                corners_uv.append([u, v])
            uv.append(corners_uv)

    translation = rotation_deg = scale = None
    if args.get("translation") is not None:
        translation, failure = _validate_translation(args["translation"], "translation")
        if failure:
            return failure
    if args.get("rotation") is not None:
        rotation_deg, failure = _validate_vec3(args["rotation"], "rotation")
        if failure:
            return failure
    if args.get("scale") is not None:
        scale, failure = _validate_scale(args["scale"], "scale")
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
            # OverflowError: the 2026-09-26 audit's clay-agent-tools-09, the
            # same fix as ``_h_add_primitive``'s identical ``material``
            # conversion above in this file -- ``int(float("inf"))`` raises
            # it, uncaught here before this fix.
            material_index = int(material_index)
        except (TypeError, ValueError, OverflowError):
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

    return _json(_scene_row(doc, obj))


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
    # ``_h_add_primitive`` already follows.
    if translation is not None:
        translation, failure = _validate_translation(translation, "translation")
        if failure:
            return failure
    if rotation_deg is not None:
        rotation_deg, failure = _validate_vec3(rotation_deg, "rotation")
        if failure:
            return failure
    if scale is not None:
        scale, failure = _validate_scale(scale, "scale")
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
        if obj.generator not in bp.CLAY_GENERATOR_NAMES:
            # A document opened from an older file may still carry a shape
            # Clay no longer offers (a pyramid, a lathe); it keeps drawing
            # and exporting, but its recipe is not an agent door any more.
            return fail(
                f"uid {obj.uid}: {obj.generator!r} is not a shape Clay offers any "
                "more, so its params cannot be set.",
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

    # The 2026-09-26 audit's clay-document-02 (agent-door half; flooring the
    # extents themselves is a kernels/mesh fix owned elsewhere in this pass):
    # every check above is about a value's *type* and *shape*, none of them
    # about whether the generator can actually build it -- a generator can still
    # divide by zero on numbers ``clamp_params`` does not floor. This used to be built inside pass 2
    # below, so a middle uid's generator raising left every *earlier* uid's
    # rebuild already pushed onto history with the multi-uid gesture never
    # collapsed, and the refusal itself reached ``call()``'s generic
    # backstop rather than naming ``field="params"``. Built once per object
    # here instead, before pass 2 touches anything -- the same
    # all-or-nothing rule this handler's own docstring already holds every
    # other check in this function to.
    built: list[tuple[Any, dict, Any]] = []
    for obj in objects:
        merged = bp.clamp_params(obj.generator, {**obj.params, **params})
        try:
            new_mesh = bp.GENERATORS[obj.generator][1](**merged)
        except (ValueError, ArithmeticError) as error:
            return fail(str(error), field="params", uids=[obj.uid])
        built.append((obj, merged, new_mesh))

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
    for obj, merged, new_mesh in built:
        # Captured before anything below mutates ``obj.params`` -- the merge
        # above edits it in place via a fresh dict, but
        # ``set_generator_params`` itself reassigns ``obj.params`` to the
        # very dict it is handed, so reading "before" off the object once
        # this call has run would compare a value against itself. See that
        # method's own docstring on why ``was`` is mandatory for this caller.
        was = {"params": dict(obj.params)}
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
        #
        # ``changed_keys=params`` -- the 2026-09-22 audit's clay-06: this is
        # the one caller that can change more than one of a generator's own
        # parameters in a single rebuild (``params`` is whatever the agent
        # passed, not the whole merged dict), which is exactly the shape
        # that can trigger ``carry_over``'s face-reordering risk (torus's
        # ``segments``/``sides`` swapped, same face count, every face
        # transposed). ``props.py``'s own call changes one field per
        # keystroke and does not need to pass this.
        mesh = regen.carry_over(
            obj.mesh,
            new_mesh,
            material=obj.material,
            changed_keys=params,
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
    # ``tests/modes/clay/test_agent_clay.py`` -- among them
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

    # ``faces`` and ``index`` are checked for their own shape before anything
    # that relates them to each other, so a malformed one names itself rather
    # than whichever combination rule happens to trip first. A bool is refused
    # as a number: JSON ``true`` is an ``int`` to Python and would paint face 1.
    faces_arg = args.get("faces")
    face_ids: list[int] | None = None
    if faces_arg is not None:
        if not isinstance(faces_arg, (list, tuple)) or not faces_arg:
            return fail(
                "faces must be a non-empty list of integers.",
                field="faces",
                recovery="fix_arguments",
            )
        try:
            face_ids = list(
                dict.fromkeys(_whole_number(f) for f in faces_arg)  # first-seen order
            )
        except (TypeError, ValueError, OverflowError):
            return fail(
                "faces must be a non-empty list of integers.",
                field="faces",
                recovery="fix_arguments",
            )
    slot = args.get("index")
    if slot is not None:
        try:
            slot = _whole_number(slot)
        except (TypeError, ValueError, OverflowError):
            return fail(
                "index must be a palette slot (a non-negative integer).",
                field="index",
                recovery="fix_arguments",
            )
        if slot < 0:
            return fail(
                "index must be a palette slot (a non-negative integer).",
                field="index",
                recovery="fix_arguments",
            )
        if slot >= len(doc.materials):
            return fail(
                f"there is no palette entry {slot}; the palette has "
                f"{len(doc.materials)} (0 to {len(doc.materials) - 1}). Leave "
                "index out and give a color to make a new one.",
                field="index",
                recovery="fix_arguments",
            )
    if face_ids is not None:
        # Faces are numbered per object, so they can only mean one object's.
        if len(uids) != 1:
            return fail(
                "faces needs exactly one uid in uids -- face numbers belong to "
                f"one object's mesh, and {len(uids)} were given.",
                field="faces",
                recovery="fix_arguments",
            )
        count = bm.face_count(doc.by_uid(uids[0]).mesh)
        bad = [f for f in face_ids if not 0 <= f < count]
        if bad:
            return fail(
                f"face index {bad[0]} is out of range for this mesh (0..{count - 1}).",
                field="faces",
                recovery="fix_arguments",
            )

    if slot is not None:
        # An existing slot: nothing to make, so a colour or a name beside it
        # would be silently dropped -- refuse rather than guess which was meant.
        if args.get("color") is not None or args.get("name") is not None:
            return fail(
                "give either index (use an existing palette entry) or color/name "
                "(make a new one), not both.",
                field="index",
                recovery="fix_arguments",
            )
        rgba = tuple(doc.materials[slot].base_color_factor)
        material = None
    else:
        color = args.get("color")
        if not isinstance(color, list) or len(color) not in (3, 4):
            return fail("color must be an array of 3 or 4 numbers, 0..1.", field="color")
        # Per component through ``_validate_unit`` rather than the old bare
        # ``isinstance(c, int | float)`` -- that check let ``float("nan")``
        # through (NaN *is* a float) straight into the palette. The same
        # unvalidated-number hole ``clay_transform`` had for its translation, one
        # tool over.
        rgba = []
        for c in color:
            value, failure = _validate_unit(c, "color")
            if failure:
                return failure
            rgba.append(value)
        rgba = tuple(rgba)
        if len(rgba) == 3:
            rgba = (*rgba, 1.0)
        name_arg = args.get("name")
        # The schema declares ``name`` a string; a bare ``str(name_arg or "")``
        # coercion used to accept anything stringifiable with no refusal at all
        # -- the same hole ``clay_add_primitive``'s own ``name`` had, fixed the
        # same way ``clay_rename`` already checks its identical field.
        if name_arg is not None and not isinstance(name_arg, str):
            return fail("name must be a string.", field="name")
        failure = _name_length_refusal(name_arg)
        if failure:
            return failure
        # The 2026-10-07 audit's clay-80: a new slot past ``MAX_PALETTE`` is
        # refused, because ``clay_scene`` returns the whole palette and one
        # grown past the frame could never be read again. ``index`` still
        # reuses an existing slot at the ceiling.
        if len(doc.materials) >= MAX_PALETTE:
            return fail(
                f"the palette is full ({MAX_PALETTE} entries); give 'index' to "
                "reuse one of the existing entries (clay_scene's 'materials') "
                "instead of a color.",
                field="color",
            )
        material = replace(bd.default_material(name_arg or ""), base_color_factor=rgba)

    # One material for the whole call -- never one per object -- folded into
    # one undo step the way ``add_material_and_assign`` folds its own pair,
    # so one tool call is one Ctrl+Z.
    mark = doc.history.mark()
    try:
        index = slot if material is None else doc.add_material(material)
        if face_ids is not None:
            # Only those faces; the object's default slot stays what it was.
            doc.paint_faces(uids[0], face_ids, index)
        else:
            _repaint(doc, uids, index)
    finally:
        doc.history.collapse_since(mark)
    _label_top(doc, mark, "Set Material")
    payload: dict[str, Any] = {"index": index, "uids": uids}
    if face_ids is not None:
        payload["faces"] = len(face_ids)
    payload["color"] = list(rgba)
    return _json(payload)


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

    **Deliberately not given the same element-mode refusal as ``clay_select``.**
    That tool *writes* object uids straight into ``doc.selection``, which in
    an element mode can manufacture "selected with nothing selected inside
    it" -- the state the derived-selection invariant forbids. This handler never does:
    :meth:`~.document.ClayDoc.remove_object` only ever *removes* a uid from ``selection`` (and from
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
    if len(name) > MAX_NAME_LENGTH:
        return fail(f"name must be at most {MAX_NAME_LENGTH} characters.", field="name")
    if any(other.uid != obj.uid and other.name == name for other in doc.objects):
        return fail(f"an object is already named {name!r}.", field="name")
    doc.set_props(obj.uid, name=name)
    return _json({"uid": obj.uid, "name": name})
