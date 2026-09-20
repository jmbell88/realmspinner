"""Clay tranche 2's modifier stack: the kernel, kind by kind, and evaluate().

Split from ``test_modifier_document.py`` (undo/document plumbing) and
``test_rblk_modifiers.py`` (the file format) the way this whole package is
split -- geometry here, document bookkeeping there. Every kind's own geometry
is tested against :mod:`.ops_modifiers` directly, over a bare ``Mesh``, with
no document in the loop at all; :func:`~.modifiers.evaluate` is tested
separately, over a :class:`~.document.ClayDoc`, for the semantics the module
docstring states -- the fast path, skip-and-record, the cache and cycles.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import modifiers as mod
from realmspinner.kernels.mesh import ops_modifiers as opm
from realmspinner.kernels.mesh import primitives as bp

from .topo_asserts import assert_closed, assert_consistently_oriented, assert_wound_outward


def _obj(name: str, mesh: bm.Mesh | None = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box() if mesh is None else mesh, **kwargs)


def _doc_with(mesh: bm.Mesh, **kwargs: object) -> tuple[bd.ClayDoc, bd.Obj]:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("Obj", mesh, **kwargs))
    return doc, obj


def _shifted(mesh: bm.Mesh, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> bm.Mesh:
    """*mesh* translated in its own local frame -- every primitive here is
    centred on its own origin, so an un-shifted box is its own mirror image
    and its own rotation image about that origin, which would leave a mirror
    or a radial-array test unable to tell a working transform from a no-op."""
    p = np.array(mesh.positions, dtype="f8")
    p += (dx, dy, dz)
    return replace(mesh, positions=p)


# --- ModParam / make / with_params -----------------------------------------


def test_make_fills_every_declared_default() -> None:
    m = mod.make("mirror", id=1)
    assert m.id == 1
    assert m.enabled is True
    assert m.get("axis") == 0  # "X", the first choice
    assert m.get("weld") == pytest.approx(0.0001)


def test_make_clamps_a_number_past_its_declared_high() -> None:
    m = mod.make("array", {"count": 999999}, id=1)
    assert m.get("count") == 200


def test_make_clamps_a_number_below_its_declared_low() -> None:
    m = mod.make("array", {"count": -5}, id=1)
    assert m.get("count") == 2


def test_make_coerces_a_choice_string_to_its_index() -> None:
    assert mod.make("mirror", {"axis": "Z"}, id=1).get("axis") == 2


def test_make_coerces_the_booleans_operation_choice() -> None:
    assert mod.make("boolean", {"operation": "intersection"}, id=1).get("operation") == 2


def test_modifier_param_coerce_refuses_an_out_of_range_choice_index() -> None:
    """The 2026-09-19 audit's clay-27: an unrecognised *string* choice is
    refused by name, but an out-of-range *numeric* index used to be silently
    clamped to the nearest legal choice instead -- ``axis=5`` on ``mirror``'s
    3-choice axis quietly became ``axis=2`` ("Z"), a different, unrequested
    choice, rather than a refusal an agent or a restored value could see."""
    with pytest.raises(el.OpError, match="Axis"):
        mod.make("mirror", {"axis": 5}, id=1)
    with pytest.raises(el.OpError, match="Axis"):
        mod.make("mirror", {"axis": -1}, id=1)


def test_make_refuses_an_unknown_kind() -> None:
    with pytest.raises(el.OpError, match="Unknown modifier kind"):
        mod.make("not-a-kind", id=1)


def test_make_refuses_an_unknown_param_name() -> None:
    with pytest.raises(el.OpError, match="no parameter named"):
        mod.make("mirror", {"bogus": 1}, id=1)


def test_with_params_only_touches_the_named_parameters() -> None:
    m = mod.make("array", {"count": 5}, id=3)
    m2 = mod.with_params(m, {"offset_x": 4.0})
    assert m2.id == 3
    assert m2.get("count") == 5
    assert m2.get("offset_x") == 4.0


def test_with_params_preserves_enabled() -> None:
    m = replace(mod.make("weld", id=1), enabled=False)
    m2 = mod.with_params(m, {"distance": 0.01})
    assert m2.enabled is False


def test_with_params_refuses_an_unknown_param() -> None:
    with pytest.raises(el.OpError, match="no parameter named"):
        mod.with_params(mod.make("weld", id=1), {"bogus": 1})


def test_next_id_is_one_past_the_highest_existing() -> None:
    stack = (mod.make("weld", id=1), mod.make("weld", id=5))
    assert mod.next_id(stack) == 6


def test_next_id_on_an_empty_stack_is_one() -> None:
    assert mod.next_id(()) == 1


def test_targets_reads_the_boolean_target_regardless_of_enabled() -> None:
    a = mod.make("boolean", {"target": 7}, id=1)
    b = replace(mod.make("boolean", {"target": 9}, id=2), enabled=False)
    assert mod.targets((a, b)) == {7, 9}


def test_targets_ignores_a_zero_target() -> None:
    assert mod.targets((mod.make("boolean", {"target": 0}, id=1),)) == set()


# --- would_cycle -------------------------------------------------------


def test_would_cycle_detects_a_two_object_loop() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    b.modifiers = (mod.make("boolean", {"target": a.uid}, id=1),)
    proposed = (mod.make("boolean", {"target": b.uid}, id=1),)
    assert mod.would_cycle(doc, a.uid, proposed) is True


def test_would_cycle_is_false_for_a_dag() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    c = doc.add_object(_obj("C"))
    b.modifiers = (mod.make("boolean", {"target": c.uid}, id=1),)
    proposed = (mod.make("boolean", {"target": b.uid}, id=1),)
    assert mod.would_cycle(doc, a.uid, proposed) is False


def test_would_cycle_detects_a_self_target() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    proposed = (mod.make("boolean", {"target": a.uid}, id=1),)
    assert mod.would_cycle(doc, a.uid, proposed) is True


def test_would_cycle_sees_the_whole_document_not_only_uids_own_edges() -> None:
    """A cycle the proposed stack does not itself close is still a cycle --
    the graph is built for the whole document, exactly what the document
    docstring's "including a self-target" line implies without saying."""
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    c = doc.add_object(_obj("C"))
    b.modifiers = (mod.make("boolean", {"target": c.uid}, id=1),)
    c.modifiers = (mod.make("boolean", {"target": b.uid}, id=1),)
    assert mod.would_cycle(doc, a.uid, ()) is True


def test_would_cycle_ignores_a_target_absent_from_the_document() -> None:
    """A missing uid is not a cycle by itself -- evaluate() reports that
    separately, as a missing-target error on the modifier that names it."""
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    proposed = (mod.make("boolean", {"target": 424242}, id=1),)
    assert mod.would_cycle(doc, a.uid, proposed) is False


# --- mirror --------------------------------------------------------------


def test_mirror_doubles_the_face_count() -> None:
    mesh = _shifted(bp.box(), dx=5.0)  # nowhere near the mirror plane
    out = opm.mirror(mesh, {"axis": 0, "weld": 0.0})
    assert bm.face_count(out) == 2 * bm.face_count(mesh)


def test_mirror_welds_a_seam_that_touches_the_plane_by_default() -> None:
    """A box whose left face sits exactly on the local x=0 plane: its mirror
    image's matching vertices land at the very same positions, a real seam
    rather than two disjoint copies with nothing to weld. (The touching face
    itself doubles -- a face lying exactly in the mirror plane maps onto
    itself, which is why a modelling package's own mirror tool warns about
    faces left sitting on the axis; that is a fact about this test's input,
    not something the weld parameter is meant to undo, so this does not also
    assert the result is a closed manifold.)
    """
    mesh = _shifted(bp.box(), dx=0.5)
    unwelded = opm.mirror(mesh, {"axis": 0, "weld": 0.0})
    welded = opm.mirror(mesh, {"axis": 0, "weld": 0.0001})
    assert len(welded.positions) < len(unwelded.positions)


def test_mirror_disjoint_copies_are_not_welded_by_a_zero_weld() -> None:
    mesh = _shifted(bp.box(), dx=5.0)
    out = opm.mirror(mesh, {"axis": 0, "weld": 0.0})
    assert len(out.positions) == 2 * len(mesh.positions)


# --- array -----------------------------------------------------------------


def test_array_makes_count_copies_translated_by_k_offset() -> None:
    mesh = bp.box()
    out = opm.array_linear(
        mesh, {"count": 4, "offset_x": 2.0, "offset_y": 0.0, "offset_z": 0.0, "weld": 0.0}
    )
    assert bm.face_count(out) == 4 * bm.face_count(mesh)
    _lo, hi = bm.bounds(out)
    assert hi[0] == pytest.approx(0.5 + 2.0 * 3)


def test_array_weld_merges_touching_copies() -> None:
    mesh = bp.box()  # size 1, spans [-0.5, 0.5]
    params = {"count": 3, "offset_x": 1.0, "offset_y": 0.0, "offset_z": 0.0}
    unwelded = opm.array_linear(mesh, {**params, "weld": 0.0})
    welded = opm.array_linear(mesh, {**params, "weld": 0.001})
    assert len(welded.positions) < len(unwelded.positions)


def test_array_refuses_before_allocating_past_the_triangle_ceiling(monkeypatch) -> None:
    import realmspinner.kernels.mesh.glbimport as glbimport

    monkeypatch.setattr(glbimport, "MAX_TRIANGLES", 10)
    with pytest.raises(el.OpError, match="triangles"):
        opm.array_linear(
            bp.box(), {"count": 5, "offset_x": 1.0, "offset_y": 0.0, "offset_z": 0.0, "weld": 0.0}
        )


# --- radial-array ------------------------------------------------------


def _group_centroid(mesh: bm.Mesh, per_face: int, k: int) -> np.ndarray:
    lo, hi = mesh.starts[k * per_face], mesh.starts[(k + 1) * per_face]
    verts = np.unique(mesh.loops[lo:hi])
    return mesh.positions[verts].astype("f8").mean(axis=0)


def test_radial_array_makes_count_copies() -> None:
    mesh = _shifted(bp.box(), dx=2.0)
    out = opm.array_radial(mesh, {"count": 5, "angle": 360.0, "axis": 1})
    assert bm.face_count(out) == 5 * bm.face_count(mesh)


def test_radial_array_closed_ring_divides_by_count() -> None:
    """A full 360-degree sweep spaces every copy, the original included,
    evenly around the whole turn."""
    from realmspinner.kernels.geom3d import math3d as m3

    mesh = _shifted(bp.box(), dx=2.0)
    out = opm.array_radial(mesh, {"count": 4, "angle": 360.0, "axis": 1})
    per_face = bm.face_count(mesh)
    original_centroid = mesh.positions.astype("f8").mean(axis=0)
    for k, degrees in enumerate((0.0, 90.0, 180.0, 270.0)):
        quat = m3.quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.radians(degrees))
        expected = m3.quat_rotate(quat, original_centroid)
        assert np.allclose(_group_centroid(out, per_face, k), expected, atol=1e-4)


def test_radial_array_open_arc_divides_by_count_minus_one() -> None:
    """An open arc reaches its far end exactly: three copies over 180 degrees
    land at 0, 90 and 180 -- dividing by count - 1, not by count."""
    from realmspinner.kernels.geom3d import math3d as m3

    mesh = _shifted(bp.box(), dx=2.0)
    out = opm.array_radial(mesh, {"count": 3, "angle": 180.0, "axis": 1})
    per_face = bm.face_count(mesh)
    original_centroid = mesh.positions.astype("f8").mean(axis=0)
    for k, degrees in enumerate((0.0, 90.0, 180.0)):
        quat = m3.quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.radians(degrees))
        expected = m3.quat_rotate(quat, original_centroid)
        assert np.allclose(_group_centroid(out, per_face, k), expected, atol=1e-4)


def test_radial_array_refuses_before_allocating_past_the_triangle_ceiling(monkeypatch) -> None:
    import realmspinner.kernels.mesh.glbimport as glbimport

    monkeypatch.setattr(glbimport, "MAX_TRIANGLES", 10)
    with pytest.raises(el.OpError, match="triangles"):
        opm.array_radial(bp.box(), {"count": 5, "angle": 360.0, "axis": 1})


# --- solidify ----------------------------------------------------------


def test_solidify_a_single_quad_becomes_a_closed_box() -> None:
    out = opm.solidify(bp.plane(), {"thickness": 1.0, "offset": 0.0})
    assert bm.face_count(out) == 6
    assert_closed(out)
    assert_consistently_oriented(out)
    assert_wound_outward(out)


def test_solidify_offset_minus_one_keeps_the_original_surface_on_one_side() -> None:
    plane = bp.plane()  # flat along y=0
    out = opm.solidify(plane, {"thickness": 1.0, "offset": -1.0})
    lo, hi = bm.bounds(out)
    assert abs(lo[1] - 0.0) < 1e-6 or abs(hi[1] - 0.0) < 1e-6


def test_solidify_on_a_closed_mesh_adds_no_rim() -> None:
    box = bp.box()
    out = opm.solidify(box, {"thickness": 0.1, "offset": 0.0})
    # Two independent closed shells (outer and inner), no rim connecting them.
    assert bm.face_count(out) == 2 * bm.face_count(box)
    assert_closed(out)


def test_solidify_refuses_a_boundary_past_its_own_ceiling_before_building_the_rim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-09-20 audit's clay-18: the rim-building Python loop had no
    ceiling of its own; the only check, `_refuse_growth`, runs *after*
    `apply()` has already built the whole result (both shells, the rim, and
    the triangle count it refuses on) -- too late for a cost that lives in
    the loop itself. `solidify` must refuse from the boundary corner count
    alone, before either shell is built.
    """
    monkeypatch.setattr(opm, "MAX_SOLIDIFY_RIM_CORNERS", 2)
    with pytest.raises(el.OpError, match="past the"):
        opm.solidify(bp.plane(), {"thickness": 0.1, "offset": -1.0})  # 4 boundary corners


def test_solidify_stays_reachable_under_its_own_ceiling() -> None:
    """The ceiling must not have crept down onto ordinary use."""
    assert opm.MAX_SOLIDIFY_RIM_CORNERS > 4
    out = opm.solidify(bp.plane(), {"thickness": 0.1, "offset": -1.0})
    assert bm.face_count(out) == 6


def test_solidify_on_an_empty_mesh_is_a_no_op() -> None:
    empty = replace(
        bp.box(),
        positions=np.zeros((0, 3), dtype="f4"),
        loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"),
        material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype=bool),
    )
    assert opm.solidify(empty, {"thickness": 1.0, "offset": 0.0}) is empty


# --- bevel (by angle) ----------------------------------------------------


def test_bevel_by_angle_bevels_a_cubes_ninety_degree_corners() -> None:
    box = bp.box()
    out = opm.bevel_by_angle(box, {"width": 0.1, "angle": 10.0})
    assert bm.face_count(out) > bm.face_count(box)
    bm.validate(out)


def test_bevel_by_angle_is_a_no_op_above_the_cubes_own_dihedral_angle() -> None:
    box = bp.box()  # every edge is a 90-degree dihedral
    out = opm.bevel_by_angle(box, {"width": 0.1, "angle": 95.0})
    assert out is box


# --- subdivide -----------------------------------------------------------


def test_subdivide_quads_every_face_per_level() -> None:
    box = bp.box()
    out = opm.subdivide(box, {"levels": 1})
    assert bm.face_count(out) == 4 * bm.face_count(box)
    out2 = opm.subdivide(box, {"levels": 2})
    assert bm.face_count(out2) == 16 * bm.face_count(box)


# --- weld ------------------------------------------------------------------


def test_weld_merges_coincident_vertices() -> None:
    box = bp.box()
    doubled = opm.array_linear(
        box, {"count": 2, "offset_x": 0.0, "offset_y": 0.0, "offset_z": 0.0, "weld": 0.0}
    )
    out = opm.weld(doubled, {"distance": 0.001})
    assert len(out.positions) < len(doubled.positions)


def test_weld_at_zero_distance_is_a_no_op_returning_the_same_object() -> None:
    box = bp.box()
    assert opm.weld(box, {"distance": 0.0}) is box


# --- triangulate -----------------------------------------------------------


def test_triangulate_every_face_becomes_a_triangle() -> None:
    box = bp.box()
    out = opm.triangulate(box, {})
    assert bm.face_count(out) == 2 * bm.face_count(box)
    assert np.all(np.diff(out.starts) == 3)


def test_triangulate_carries_material_smooth_and_per_corner_uv() -> None:
    box = bp.box()
    n_faces = bm.face_count(box)
    n_corners = len(box.loops)
    textured = replace(
        box,
        material=np.arange(n_faces, dtype="i4"),
        smooth=np.array([bool(i % 2) for i in range(n_faces)]),
        uv=np.stack(
            [np.arange(n_corners, dtype="f4") / n_corners, np.zeros(n_corners, dtype="f4")], axis=1
        ),
    )
    out = opm.triangulate(textured, {})
    tris_per_face = np.diff(textured.starts) - 2
    face_of_tri = np.repeat(np.arange(n_faces), tris_per_face)
    assert np.array_equal(out.material, textured.material[face_of_tri])
    assert np.array_equal(out.smooth, textured.smooth[face_of_tri])
    assert out.uv is not None and out.uv.shape == (len(out.loops), 2)


# --- smooth (Laplacian) -----------------------------------------------


def test_laplacian_smooth_moves_interior_vertices_on_a_closed_mesh() -> None:
    box = bp.box()
    out = opm.laplacian_smooth(box, {"factor": 0.5, "iterations": 3})
    assert not np.allclose(out.positions, box.positions)


def test_laplacian_smooth_holds_every_vertex_fixed_on_a_lone_open_face() -> None:
    plane = bp.plane()  # a single face: every vertex touches a boundary edge
    out = opm.laplacian_smooth(plane, {"factor": 0.9, "iterations": 5})
    assert out is plane


def test_laplacian_smooth_at_zero_factor_is_a_no_op() -> None:
    box = bp.box()
    assert opm.laplacian_smooth(box, {"factor": 0.0, "iterations": 3}) is box


# --- boolean (needs the optional CSG backend) -------------------------


def test_boolean_union_via_evaluate_uses_the_targets_evaluated_mesh() -> None:
    pytest.importorskip("manifold3d")
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    # B's own base mesh is shifted off its local origin before it is mirrored,
    # so B's *evaluated* result (two disjoint boxes) is a proper closed shape
    # rather than the doubled-face degenerate a symmetric box would mirror
    # into -- see the mirror seam test above for why that would refuse here.
    b = doc.add_object(_obj("B", mesh=_shifted(bp.box(), dx=3.0), translation=(0.5, 0.0, 0.0)))
    doc.set_modifiers(b.uid, (mod.make("mirror", {"axis": 0, "weld": 0.0001}, id=1),))
    doc.set_modifiers(a.uid, (mod.make("boolean", {"target": b.uid, "operation": "union"}, id=1),))
    ev = doc.evaluation(a.uid)
    assert ev.errors == ()
    assert_closed(ev.mesh)


# --- evaluate(): the fast path, skip-and-record, the cache, cycles ---------


def test_no_enabled_modifiers_evaluates_to_the_base_mesh_object_itself() -> None:
    doc, obj = _doc_with(bp.box())
    ev = doc.evaluation(obj.uid)
    assert ev.mesh is obj.mesh
    assert ev.errors == ()


def test_a_disabled_only_stack_also_takes_the_fast_path() -> None:
    doc, obj = _doc_with(bp.box())
    obj.modifiers = (replace(mod.make("weld", id=1), enabled=False),)
    ev = doc.evaluation(obj.uid)
    assert ev.mesh is obj.mesh


def test_evaluate_runs_enabled_modifiers_in_stack_order() -> None:
    doc, obj = _doc_with(bp.box())
    doc.set_modifiers(
        obj.uid,
        (
            mod.make(
                "array",
                {"count": 2, "offset_x": 5.0, "offset_y": 0.0, "offset_z": 0.0, "weld": 0.0},
                id=1,
            ),
            mod.make("triangulate", {}, id=2),
        ),
    )
    ev = doc.evaluation(obj.uid)
    assert np.all(np.diff(ev.mesh.starts) == 3)  # triangulate ran last
    assert bm.face_count(ev.mesh) == 2 * 2 * 6  # array doubled, triangulate doubled again


def test_a_refusing_modifier_is_skipped_and_recorded_not_fatal() -> None:
    doc, obj = _doc_with(bp.box())
    doc.set_modifiers(
        obj.uid,
        (
            mod.make("boolean", {"target": 0}, id=1),  # "Choose a target object."
            mod.make("triangulate", {}, id=2),
        ),
    )
    ev = doc.evaluation(obj.uid)
    assert [i for i, _ in ev.errors] == [1]
    assert "target" in ev.errors[0][1].lower()
    # triangulate still ran, over the mesh as it stood before the refused one.
    assert np.all(np.diff(ev.mesh.starts) == 3)


def test_a_disabled_modifier_is_skipped_silently() -> None:
    doc, obj = _doc_with(bp.box())
    disabled = replace(mod.make("weld", {"distance": 0.5}, id=1), enabled=False)
    real = mod.make("triangulate", {}, id=2)
    doc.set_modifiers(obj.uid, (disabled, real))
    ev = doc.evaluation(obj.uid)
    assert ev.errors == ()
    assert np.all(np.diff(ev.mesh.starts) == 3)


def test_a_missing_boolean_target_is_an_error_not_a_crash() -> None:
    doc, obj = _doc_with(bp.box())
    missing = 424242
    obj.modifiers = (mod.make("boolean", {"target": missing}, id=1),)
    ev = doc.evaluation(obj.uid)
    assert ev.mesh is obj.mesh  # the only modifier refused; base mesh stands
    assert ev.errors == ((1, f"Target object {missing} no longer exists."),)


def test_a_hand_edited_cycle_is_refused_on_the_closing_modifier_not_recursion() -> None:
    """``set_modifiers`` refuses a cycle going forward; a hand-edited stack
    (or one loaded from a ``.rblk`` -- see the ``.rblk`` reader's own tests)
    is the only way one reaches :func:`evaluate` at all, and it must not
    recurse until the stack overflows.

    Which object's *own* evaluation shows the error depends on which one is
    asked about first -- the one that starts the chain recurses into the
    other, and it is that *other* one's attempt to recurse back into the
    first that closes the loop. Evaluating ``a`` first: ``a`` recurses into
    ``b``, which tries to recurse back into ``a`` -- already in ``a``'s own
    evaluation, and therefore in the in-progress set ``b`` is handed. The
    refusal lands on *that* modifier (``b``'s), not on the one that started
    the chain: ``a``'s own boolean sees ``b``'s evaluated mesh, unaffected by
    ``b``'s own refused modifier, and its union of two plain boxes succeeds
    with no error of its own. Either way, nothing recurses without bound.
    """
    pytest.importorskip("manifold3d")
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B", translation=(2.0, 0.0, 0.0)))
    a.modifiers = (mod.make("boolean", {"target": b.uid}, id=1),)
    b.modifiers = (mod.make("boolean", {"target": a.uid}, id=1),)

    ev_a = doc.evaluation(a.uid)
    assert ev_a.errors == ()

    ev_b = doc.evaluation(b.uid)
    assert ev_b.errors and "cycle" in ev_b.errors[0][1].lower()


def test_repeated_evaluation_returns_the_same_mesh_object() -> None:
    doc, obj = _doc_with(bp.box())
    doc.set_modifiers(obj.uid, (mod.make("triangulate", {}, id=1),))
    first = doc.evaluated(obj.uid)
    second = doc.evaluated(obj.uid)
    assert first is second


def test_changing_a_param_invalidates_the_cache() -> None:
    doc, obj = _doc_with(bp.box())
    doc.set_modifiers(obj.uid, (mod.make("bevel", {"width": 0.02}, id=1),))
    first = doc.evaluated(obj.uid)
    doc.set_modifiers(obj.uid, (mod.with_params(obj.modifiers[0], {"width": 0.2}),))
    second = doc.evaluated(obj.uid)
    assert first is not second


def test_editing_the_base_mesh_invalidates_the_cache_and_keeps_the_stack() -> None:
    doc, obj = _doc_with(bp.box())
    doc.set_modifiers(obj.uid, (mod.make("triangulate", {}, id=1),))
    doc.evaluated(obj.uid)
    doc.set_mesh(obj.uid, bp.cone())
    assert obj.modifiers  # kept, per set_mesh's own rule
    ev = doc.evaluation(obj.uid)
    assert np.all(np.diff(ev.mesh.starts) == 3)


def test_a_booleans_cache_entry_is_invalidated_when_the_target_moves() -> None:
    pytest.importorskip("manifold3d")
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B", translation=(0.5, 0.0, 0.0)))
    doc.set_modifiers(a.uid, (mod.make("boolean", {"target": b.uid, "operation": "union"}, id=1),))
    first = doc.evaluated(a.uid)
    doc.set_transform(b.uid, translation=(0.9, 0.0, 0.0))
    second = doc.evaluated(a.uid)
    assert first is not second
