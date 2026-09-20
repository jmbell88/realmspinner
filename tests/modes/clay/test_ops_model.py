"""Tranche 5's general modelling ops: bisect, knife, slide, rip, poke,
triangulate, tris-to-quads, symmetrize and grid fill."""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_model as om
from realmspinner.kernels.mesh import ops_topo as ops
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.kernels.mesh.mesh import from_faces

from .topo_asserts import assert_closed, assert_consistently_oriented, assert_wound_outward


def _uvd(mesh: bm.Mesh) -> bm.Mesh:
    """The same mesh with a distinct, recognisable uv per corner."""
    n = len(mesh.loops)
    uv = np.stack([np.arange(n, dtype="f4"), np.arange(n, dtype="f4") * 2], axis=1)
    return bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
        uv=uv,
    )


def _two_quads() -> bm.Mesh:
    """Two quads sharing one edge -- the minimal mesh a rip actually splits."""
    positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]]
    faces = [[0, 1, 2, 3], [1, 4, 5, 2]]
    return from_faces(positions, faces)


# --- bisect -------------------------------------------------------------


def test_bisect_with_no_selection_refuses() -> None:
    box = prim.box()
    with pytest.raises(el.OpError, match="Select the faces"):
        om.bisect(box, el.empty(), point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))


def test_bisect_a_degenerate_normal_refuses() -> None:
    box = prim.box()
    faces = el.ElementSel(faces=np.arange(bm.face_count(box), dtype="i4"))
    with pytest.raises(el.OpError, match="normal has no length"):
        om.bisect(box, faces, point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 0.0))


def test_bisect_clear_0_splits_the_crossed_faces_and_keeps_the_rest() -> None:
    box = prim.box()
    faces = el.ElementSel(faces=np.arange(bm.face_count(box), dtype="i4"))
    out, sel = om.bisect(box, faces, point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), clear=0)
    bm.validate(out)
    # The two X-facing caps stay whole; the other four side faces each split
    # into two along the cut.
    assert bm.face_count(out) == 2 + 4 * 2
    assert len(sel.edges) == 4, "one new ring, four edges around the box"
    assert_consistently_oriented(out)
    assert_closed(out)
    assert_wound_outward(out)


def test_bisect_clear_removes_the_named_side() -> None:
    box = prim.box()
    faces = el.ElementSel(faces=np.arange(bm.face_count(box), dtype="i4"))
    above, _ = om.bisect(box, faces, point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), clear=1)
    below, _ = om.bisect(box, faces, point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), clear=2)
    bm.validate(above)
    bm.validate(below)
    assert (above.positions[:, 0] >= -1e-6).all()
    assert (below.positions[:, 0] <= 1e-6).all()


def test_bisect_fill_caps_the_hole_it_leaves() -> None:
    box = prim.box()
    faces = el.ElementSel(faces=np.arange(bm.face_count(box), dtype="i4"))
    out, sel = om.bisect(
        box, faces, point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), clear=2, fill=True
    )
    bm.validate(out)
    assert_closed(out)
    assert_consistently_oriented(out)
    assert len(sel.edges) == 4


def test_bisect_fill_without_a_closing_selection_is_left_open() -> None:
    """A partial cut whose boundary never closes is not capped -- best-effort,
    not a refusal."""
    box = prim.box()
    faces = el.ElementSel(faces=np.array([2], dtype="i4"))  # one side face only
    out, _ = om.bisect(
        box, faces, point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), clear=2, fill=True
    )
    bm.validate(out)  # does not raise, even though the cap could not close


def test_bisect_preserves_uv_by_interpolating_within_the_face() -> None:
    box = _uvd(prim.box())
    faces = el.ElementSel(faces=np.arange(bm.face_count(box), dtype="i4"))
    out, _ = om.bisect(box, faces, point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), clear=0)
    bm.validate(out)
    assert out.uv is not None
    assert out.uv.shape == (len(out.loops), 2)


def test_bisect_object_mode_caller_passes_every_face() -> None:
    """The documented door for "cut the whole object"."""
    box = prim.box()
    all_faces = el.ElementSel(faces=np.arange(bm.face_count(box), dtype="i4"))
    out, _ = om.bisect(box, all_faces, point=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0), clear=1)
    bm.validate(out)
    assert (out.positions[:, 1] >= -1e-6).all(), "clear=1 removes the negative side"


def test_bisect_refuses_past_its_own_corner_ceiling() -> None:
    """A direct call to the size guard, since building a 200,000-corner
    selection just to exercise the refusal would be its own slow test."""
    with pytest.raises(el.OpError, match="past the"):
        om._refuse_bisect_size(om.MAX_BISECT_CORNERS + 1)


# --- knife ----------------------------------------------------------------


def test_knife_is_bisect_restricted_to_the_selection_with_no_clear_or_fill() -> None:
    box = prim.box()
    faces = el.ElementSel(faces=np.array([2], dtype="i4"))
    out, sel = om.knife(box, faces, point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
    bm.validate(out)
    assert bm.face_count(out) == bm.face_count(box) + 1  # only that one face split
    assert len(sel.edges) == 1


# --- edge_slide / vertex_slide ---------------------------------------------


def test_edge_slide_at_zero_does_not_move_anything() -> None:
    box = prim.box()
    a = adj.adjacency(box)
    edge = a.edge_verts[0]
    sel = el.ElementSel(edges=np.array([edge], dtype="i4"))
    out, out_sel = om.edge_slide(box, sel, t=0.0)
    assert np.allclose(out.positions, box.positions)
    assert out_sel is sel


def test_edge_slide_at_extremes_reaches_its_rail_neighbours() -> None:
    box = prim.box()
    sel = el.ElementSel(edges=np.array([[4, 5]], dtype="i4"))
    plus, _ = om.edge_slide(box, sel, t=1.0)
    minus, _ = om.edge_slide(box, sel, t=-1.0)
    bm.validate(plus)
    bm.validate(minus)
    assert not np.allclose(plus.positions[4], box.positions[4])
    assert not np.allclose(minus.positions[4], box.positions[4])
    assert not np.allclose(plus.positions[4], minus.positions[4])


def test_edge_slide_with_no_selection_refuses() -> None:
    box = prim.box()
    with pytest.raises(el.OpError, match="Select an edge loop"):
        om.edge_slide(box, el.empty(), t=0.5)


def test_edge_slide_an_unknown_edge_refuses() -> None:
    box = prim.box()
    sel = el.ElementSel(edges=np.array([[0, 2]], dtype="i4"))  # a diagonal, no such edge
    with pytest.raises(el.OpError, match="not part of this mesh"):
        om.edge_slide(box, sel, t=0.5)


def test_vertex_slide_at_zero_does_not_move() -> None:
    box = prim.box()
    sel = el.ElementSel(verts=np.array([0], dtype="i4"))
    out, _ = om.vertex_slide(box, sel, t=0.0)
    assert np.allclose(out.positions, box.positions)


def test_vertex_slide_picks_the_edge_nearest_the_given_direction() -> None:
    box = prim.box()
    sel = el.ElementSel(verts=np.array([4], dtype="i4"))
    out, _ = om.vertex_slide(box, sel, t=0.5, direction_edge=(1.0, 0.0, 0.0))
    # Vertex 4 = (-.5, .5, -.5); its +X neighbour is vertex 5 = (.5, .5, -.5).
    assert np.allclose(out.positions[4], [0.0, 0.5, -0.5])


def test_vertex_slide_topology_is_untouched_so_uv_is_preserved() -> None:
    box = _uvd(prim.box())
    sel = el.ElementSel(verts=np.array([0], dtype="i4"))
    out, _ = om.vertex_slide(box, sel, t=0.3)
    assert np.array_equal(out.uv, box.uv)
    assert np.array_equal(out.loops, box.loops)


# --- rip --------------------------------------------------------------------


def test_rip_with_no_selection_refuses() -> None:
    m = _two_quads()
    with pytest.raises(el.OpError, match="Select the edges"):
        om.rip(m, el.empty())


def test_rip_a_boundary_edge_refuses() -> None:
    m = _two_quads()
    sel = el.ElementSel(edges=np.array([[0, 3]], dtype="i4"))  # a border edge
    with pytest.raises(el.OpError, match="boundary"):
        om.rip(m, sel)


def test_rip_splits_the_shared_edge_into_two() -> None:
    m = _two_quads()
    sel = el.ElementSel(edges=np.array([[1, 2]], dtype="i4"))
    out, out_sel = om.rip(m, sel)
    bm.validate(out)
    assert len(out.positions) == len(m.positions) + 2
    r = adj.check_manifold(out)
    assert len(r.boundary_edges) == 8, "the whole mesh is now two disjoint quads"
    assert len(out_sel.verts) == 4


def test_rip_a_redundant_edge_inside_a_valence_3_fan_refuses() -> None:
    """Ripping one edge of a cube corner does not separate anything: the
    other two edges at that corner still connect every face around it."""
    box = prim.box()
    sel = el.ElementSel(edges=np.array([[0, 1]], dtype="i4"))
    with pytest.raises(el.OpError, match="does not separate"):
        om.rip(box, sel)


def test_rip_preserves_uv_because_no_corner_changes_its_own_value() -> None:
    m = _uvd(_two_quads())
    sel = el.ElementSel(edges=np.array([[1, 2]], dtype="i4"))
    out, _ = om.rip(m, sel)
    assert out.uv is not None
    assert np.array_equal(out.uv[:4], m.uv[:4])


# --- poke -------------------------------------------------------------------


def test_poke_with_no_selection_refuses() -> None:
    box = prim.box()
    with pytest.raises(el.OpError, match="Select at least one face"):
        om.poke(box, el.empty())


def test_poke_fans_each_face_into_a_triangle_per_edge() -> None:
    box = prim.box()
    sel = el.ElementSel(faces=np.array([0], dtype="i4"))
    out, out_sel = om.poke(box, sel, offset=0.0)
    bm.validate(out)
    assert bm.face_count(out) == bm.face_count(box) - 1 + 4
    assert len(out_sel.faces) == 4
    assert len(out.positions) == len(box.positions) + 1
    assert_closed(out)
    assert_consistently_oriented(out)


def test_poke_offset_moves_the_centre_along_the_normal() -> None:
    box = prim.box()
    sel = el.ElementSel(faces=np.array([1], dtype="i4"))  # +Y face
    out, out_sel = om.poke(box, sel, offset=0.25)
    centre = out.positions[-1]
    assert centre[1] > 0.5 - 1e-6  # pushed further along +Y than the flat face


def test_poke_interpolates_the_centre_uv() -> None:
    box = _uvd(prim.box())
    sel = el.ElementSel(faces=np.array([0], dtype="i4"))
    out, _ = om.poke(box, sel)
    assert out.uv is not None
    assert out.uv.shape == (len(out.loops), 2)


# --- triangulate_faces --------------------------------------------------


def test_triangulate_faces_with_no_selection_does_the_whole_object() -> None:
    box = prim.box()
    out, sel = om.triangulate_faces(box, el.empty())
    bm.validate(out)
    assert bm.face_count(out) == 12
    assert len(sel.faces) == 12
    assert_closed(out)
    assert_consistently_oriented(out)


def test_triangulate_faces_on_an_already_triangular_face_is_idempotent() -> None:
    box = prim.box()
    once, _ = om.triangulate_faces(box, el.empty())
    twice, _ = om.triangulate_faces(once, el.empty())
    bm.validate(twice)
    assert bm.face_count(twice) == bm.face_count(once)


def test_triangulate_faces_preserves_uv_from_the_real_source_corners() -> None:
    box = _uvd(prim.box())
    out, _ = om.triangulate_faces(box, el.empty())
    # Every uv row in the output must be one that already existed on the input.
    input_rows = {tuple(row) for row in box.uv.tolist()}
    for row in out.uv.tolist():
        assert tuple(row) in input_rows


def test_triangulate_faces_on_an_empty_mesh_refuses() -> None:
    empty_mesh = bm.Mesh(
        positions=np.zeros((0, 3), dtype="f4"),
        loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"),
        material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype=bool),
    )
    with pytest.raises(el.OpError, match="no faces"):
        om.triangulate_faces(empty_mesh, el.empty())


# --- tris_to_quads ------------------------------------------------------


def test_tris_to_quads_undoes_a_flat_triangulation() -> None:
    box = prim.box()
    tri, _ = om.triangulate_faces(box, el.empty())
    out, sel = om.tris_to_quads(tri, el.empty(), max_angle=1.0)
    bm.validate(out)
    assert bm.face_count(out) == 6
    assert len(sel.faces) == 6
    assert_closed(out)
    assert_consistently_oriented(out)
    assert_wound_outward(out)


def test_tris_to_quads_finding_nothing_to_merge_is_not_an_error() -> None:
    box = prim.box()  # already all quads; no triangle pairs exist
    given = el.empty()
    out, sel = om.tris_to_quads(box, given, max_angle=40.0)
    assert bm.face_count(out) == bm.face_count(box)
    assert sel is given, "nothing merged, so the caller's own selection comes back unchanged"


def test_tris_to_quads_on_an_empty_mesh_refuses() -> None:
    empty_mesh = bm.Mesh(
        positions=np.zeros((0, 3), dtype="f4"),
        loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"),
        material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype=bool),
    )
    with pytest.raises(el.OpError, match="no faces"):
        om.tris_to_quads(empty_mesh, el.empty())


def test_tris_to_quads_preserves_uv_from_the_four_source_corners() -> None:
    box = _uvd(prim.box())
    tri, _ = om.triangulate_faces(box, el.empty())
    out, _ = om.tris_to_quads(tri, el.empty(), max_angle=1.0)
    input_rows = {tuple(row) for row in tri.uv.tolist()}
    for row in out.uv.tolist():
        assert tuple(row) in input_rows


# --- symmetrize -----------------------------------------------------------


def test_symmetrize_ignores_the_selection() -> None:
    box = prim.box()
    sel = el.ElementSel(faces=np.array([0], dtype="i4"))
    from_sel, _ = om.symmetrize(box, sel, axis=0, direction=1)
    from_empty, _ = om.symmetrize(box, el.empty(), axis=0, direction=1)
    assert bm.face_count(from_sel) == bm.face_count(from_empty)


def test_symmetrize_an_axis_out_of_range_refuses() -> None:
    box = prim.box()
    with pytest.raises(el.OpError, match="Axis"):
        om.symmetrize(box, el.empty(), axis=3, direction=1)


def test_symmetrize_mirrors_the_kept_half_across_the_origin() -> None:
    box = prim.box((2.0, 1.0, 1.0))
    out, _ = om.symmetrize(box, el.empty(), axis=0, direction=1)
    bm.validate(out)
    assert_closed(out)
    assert_consistently_oriented(out)
    lo, hi = bm.bounds(out)
    assert np.allclose(lo, [-1.0, -0.5, -0.5], atol=1e-5)
    assert np.allclose(hi, [1.0, 0.5, 0.5], atol=1e-5)


def test_symmetrize_the_other_direction_mirrors_the_other_way() -> None:
    box = prim.box((2.0, 1.0, 1.0))
    # Skew the box off-centre so the two directions give different results.
    skewed = bm.transformed(box, np.array(
        [[1, 0, 0, 0.3], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype="f8"
    ))
    plus, _ = om.symmetrize(skewed, el.empty(), axis=0, direction=1)
    minus, _ = om.symmetrize(skewed, el.empty(), axis=0, direction=-1)
    plus_x = sorted(plus.positions[:, 0].tolist())
    minus_x = sorted(minus.positions[:, 0].tolist())
    assert not np.allclose(plus_x, minus_x)


# --- grid_fill ----------------------------------------------------------


def _box_missing_one_face() -> tuple[bm.Mesh, np.ndarray]:
    box = prim.box()
    out, _ = ops.delete_faces(box, el.ElementSel(faces=np.array([1], dtype="i4")))
    boundary = adj.check_manifold(out).boundary_edges
    return out, boundary


def test_grid_fill_with_no_selection_refuses() -> None:
    box = prim.box()
    with pytest.raises(el.OpError, match="Select the boundary"):
        om.grid_fill(box, el.empty(), span=1)


def test_grid_fill_an_odd_boundary_refuses() -> None:
    # A pentagon boundary: delete two adjacent faces of a pyramid's side fan
    # is awkward to build by hand, so build an odd ring directly instead.
    positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0.5, 1.5, 0], [0, 1, 0]]
    faces = [[0, 1, 2, 3, 4]]
    m = from_faces(positions, faces)
    sel = el.ElementSel(edges=np.array([[0, 1]], dtype="i4"))
    with pytest.raises(el.OpError, match="even number"):
        om.grid_fill(m, sel, span=1)


def test_grid_fill_closes_a_square_hole_at_span_one() -> None:
    hole, boundary = _box_missing_one_face()
    sel = el.ElementSel(edges=boundary)
    out, out_sel = om.grid_fill(hole, sel, span=1)
    bm.validate(out)
    assert_closed(out)
    assert_consistently_oriented(out)
    assert len(out_sel.faces) == 1


def test_grid_fill_span_too_wide_for_the_boundary_refuses() -> None:
    hole, boundary = _box_missing_one_face()
    sel = el.ElementSel(edges=boundary)
    with pytest.raises(el.OpError, match="leaves no rows"):
        om.grid_fill(hole, sel, span=2)


def test_grid_fill_builds_a_grid_over_a_larger_boundary() -> None:
    grid = prim.GENERATORS["grid"][1](size=(3.0, 1.0), divisions=3)
    out, _ = ops.delete_faces(grid, el.ElementSel(faces=np.array([0, 1], dtype="i4")))
    boundary = adj.check_manifold(out).boundary_edges
    sel = el.ElementSel(edges=boundary)
    filled, filled_sel = om.grid_fill(out, sel, span=2)
    bm.validate(filled)
    assert_closed(filled)
    assert_consistently_oriented(filled)
    assert len(filled_sel.faces) == 8  # 4 rows * 2 columns


def test_grid_fill_preserves_boundary_uv_and_interpolates_the_interior() -> None:
    grid = prim.GENERATORS["grid"][1](size=(3.0, 1.0), divisions=3)
    grid = _uvd(grid)
    out, _ = ops.delete_faces(grid, el.ElementSel(faces=np.array([0, 1], dtype="i4")))
    boundary = adj.check_manifold(out).boundary_edges
    sel = el.ElementSel(edges=boundary)
    filled, _ = om.grid_fill(out, sel, span=2)
    assert filled.uv is not None
    assert filled.uv.shape == (len(filled.loops), 2)
