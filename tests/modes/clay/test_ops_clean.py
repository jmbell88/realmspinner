"""``ops_clean``'s survey and repair ops, pinned against small hand-built meshes.

Every mesh below is built by hand from a :func:`primitives.box` (the one
closed, UV-unwrapped, manifold reference every other test in this file starts
from) rather than from a general-purpose fixture factory, because the point of
each test is *exactly one* defect against an otherwise clean cube -- so a
survey before an op runs is the specification of what that op owes, and a
survey (or exact array comparison) after is the proof it paid only that debt
and nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_clean as oc
from realmspinner.kernels.mesh import ops_topo
from realmspinner.kernels.mesh import primitives as prim

# --- mesh builders -----------------------------------------------------------


def _box() -> bm.Mesh:
    return prim.box()


def _with_reversed_face(mesh: bm.Mesh, face: int = 0) -> bm.Mesh:
    """*mesh* with one face's winding reversed -- a single flipped face."""
    out, _sel = ops_topo.flip_normals(mesh, el.ElementSel(faces=np.array([face], dtype="i4")))
    return out


def _fully_inverted(mesh: bm.Mesh) -> bm.Mesh:
    """*mesh* with every face reversed -- an inside-out shell, but internally
    consistent (every pair of neighbours still agrees with each other)."""
    out, _sel = ops_topo.flip_normals(mesh, el.ElementSel())
    return out


def _with_duplicate_face(mesh: bm.Mesh, face: int = 0) -> bm.Mesh:
    """*mesh* plus one extra face over the same corners as *face*."""
    corners = np.arange(int(mesh.starts[face]), int(mesh.starts[face + 1]))
    loops = np.concatenate([mesh.loops.astype("i8"), mesh.loops.astype("i8")[corners]])
    starts = np.concatenate([mesh.starts.astype("i8"), [int(mesh.starts[-1]) + len(corners)]])
    material = np.concatenate([mesh.material, mesh.material[[face]]])
    smooth = np.concatenate([mesh.smooth, mesh.smooth[[face]]])
    uv = None if mesh.uv is None else np.concatenate([mesh.uv, mesh.uv[corners]])
    return bm.Mesh(
        positions=mesh.positions,
        loops=loops,
        starts=starts,
        material=material,
        smooth=smooth,
        uv=uv,
    )


def _with_loose_vertex(mesh: bm.Mesh, position=(5.0, 5.0, 5.0)) -> bm.Mesh:
    """*mesh* plus one vertex no face references."""
    positions = np.vstack([mesh.positions, np.array([position], dtype="f4")])
    return bm.Mesh(
        positions=positions,
        loops=mesh.loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
        uv=mesh.uv,
    )


def _with_degenerate_face(mesh: bm.Mesh) -> bm.Mesh:
    """*mesh* plus one extra triangle that reuses one vertex twice -- zero
    area, and fewer than three distinct vertices either way."""
    extra = np.array([0, 1, 0], dtype="i8")
    loops = np.concatenate([mesh.loops.astype("i8"), extra])
    starts = np.concatenate([mesh.starts.astype("i8"), [int(mesh.starts[-1]) + 3]])
    material = np.concatenate([mesh.material, [0]])
    smooth = np.concatenate([mesh.smooth, [False]])
    uv = None if mesh.uv is None else np.concatenate([mesh.uv, np.zeros((3, 2), dtype="f4")])
    return bm.Mesh(
        positions=mesh.positions,
        loops=loops,
        starts=starts,
        material=material,
        smooth=smooth,
        uv=uv,
    )


def _two_cubes_offset(mesh: bm.Mesh, eps: float = 1e-7) -> bm.Mesh:
    """Two copies of *mesh*, the second nudged by *eps* -- two shells whose
    corresponding vertices are coincident within any distance past *eps*."""
    b = bm.Mesh(
        positions=mesh.positions + eps,
        loops=mesh.loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
        uv=mesh.uv,
    )
    n_a = len(mesh.positions)
    positions = np.vstack([mesh.positions, b.positions])
    loops = np.concatenate([mesh.loops.astype("i8"), b.loops.astype("i8") + n_a]).astype("i4")
    starts = np.concatenate(
        [mesh.starts.astype("i8")[:-1], mesh.starts.astype("i8") + len(mesh.loops)]
    ).astype("i4")
    material = np.concatenate([mesh.material, b.material])
    smooth = np.concatenate([mesh.smooth, b.smooth])
    uv = None if mesh.uv is None else np.concatenate([mesh.uv, b.uv])
    return bm.Mesh(
        positions=positions,
        loops=loops,
        starts=starts,
        material=material,
        smooth=smooth,
        uv=uv,
    )


def _open_box(mesh: bm.Mesh, face: int = 0) -> bm.Mesh:
    """*mesh* with one face deleted -- an open box, one hole."""
    out, _sel = ops_topo.delete_faces(mesh, el.ElementSel(faces=np.array([face], dtype="i4")))
    return out


# --- survey: each defect counted exactly -------------------------------------


def test_survey_of_a_clean_box_is_all_zero() -> None:
    assert oc.survey(_box()) == oc.Survey(0, 0, 0, 0, 0, 0, 0)


def test_survey_counts_a_degenerate_face_exactly() -> None:
    s = oc.survey(_with_degenerate_face(_box()))
    assert s.degenerate_faces == 1
    assert s.duplicate_faces == 0
    assert s.flipped_faces == 0
    assert s.inside_out_shells == 0


def test_survey_counts_a_duplicate_face_exactly() -> None:
    s = oc.survey(_with_duplicate_face(_box()))
    # box's own faces already give every edge its twin, so the duplicate's
    # edges are used a *third* time and fall out of twin/flipped-pair
    # detection entirely (adjacency.py: only a 2-use edge is scored either
    # way) -- this mesh's only defect really is the duplicate face.
    assert s == oc.Survey(0, 1, 0, 0, 0, 0, 0)


def test_survey_counts_a_loose_vertex_exactly() -> None:
    s = oc.survey(_with_loose_vertex(_box()))
    assert s == oc.Survey(0, 0, 1, 0, 0, 0, 0)


def test_survey_counts_coincident_vertices_exactly() -> None:
    mesh = _two_cubes_offset(_box(), eps=1e-7)
    assert oc.survey(mesh, distance=1e-5).coincident_vertices == 8
    # Past the two shells' own separation, nothing is coincident.
    assert oc.survey(mesh, distance=1e-9).coincident_vertices == 0


def test_survey_counts_one_flipped_face_exactly() -> None:
    s = oc.survey(_with_reversed_face(_box(), face=0))
    assert s.flipped_faces == 1
    assert s.inside_out_shells == 0, "a single flipped face is a minority, not a net inversion"


def test_survey_counts_an_inside_out_shell_exactly() -> None:
    s = oc.survey(_fully_inverted(_box()))
    # Every face agrees with every neighbour -- there is no minority to flag,
    # only the shell's own volume sign says anything is wrong.
    assert s.flipped_faces == 0
    assert s.inside_out_shells == 1


def test_survey_counts_open_edges_exactly() -> None:
    assert oc.survey(_open_box(_box())).open_edges == 4
    assert oc.survey(_box()).open_edges == 0


# --- each op fixes only its own defect ---------------------------------------


def test_remove_degenerate_removes_only_the_degenerate_face() -> None:
    box = _box()
    out = oc.remove_degenerate(_with_degenerate_face(box))
    assert np.array_equal(out.positions, box.positions)
    assert np.array_equal(out.loops, box.loops)
    assert oc.survey(out) == oc.Survey(0, 0, 0, 0, 0, 0, 0)


def test_remove_duplicate_faces_removes_only_the_later_duplicate() -> None:
    box = _box()
    out = oc.remove_duplicate_faces(_with_duplicate_face(box))
    assert np.array_equal(out.loops, box.loops)
    assert np.array_equal(out.material, box.material)
    assert np.array_equal(out.smooth, box.smooth)
    assert oc.survey(out) == oc.Survey(0, 0, 0, 0, 0, 0, 0)


def test_remove_loose_vertices_removes_only_the_unreferenced_vertex() -> None:
    box = _box()
    out = oc.remove_loose_vertices(_with_loose_vertex(box))
    assert np.array_equal(out.positions, box.positions)
    assert np.array_equal(out.loops, box.loops)
    assert oc.survey(out) == oc.Survey(0, 0, 0, 0, 0, 0, 0)


def test_merge_by_distance_merges_only_the_coincident_vertices() -> None:
    box = _box()
    mesh = _two_cubes_offset(box, eps=1e-7)
    out = oc.merge_by_distance(mesh, 1e-5)
    assert len(out.positions) == len(box.positions)
    assert bm.face_count(out) == 2 * bm.face_count(box)
    assert oc.survey(out, distance=1e-5).coincident_vertices == 0


def test_fill_all_holes_fills_only_the_open_boundary() -> None:
    opened = _open_box(_box())
    out = oc.fill_all_holes(opened)
    assert bm.face_count(out) == bm.face_count(_box())
    s = oc.survey(out)
    assert s.open_edges == 0
    # Filling does not itself claim to restore the exact original winding or
    # UVs of the face it replaces -- only that the hole is gone.
    assert s.degenerate_faces == 0
    assert s.duplicate_faces == 0


def test_recalc_outside_fixes_only_the_winding() -> None:
    box = _box()
    out = oc.recalc_outside(_fully_inverted(box))
    assert np.array_equal(out.positions, box.positions)
    assert np.array_equal(out.loops, box.loops)
    assert np.array_equal(out.uv, box.uv)
    assert oc.survey(out) == oc.Survey(0, 0, 0, 0, 0, 0, 0)


# --- identity: each op is a no-op when nothing needs fixing ------------------


@pytest.mark.parametrize(
    "op",
    [
        oc.remove_degenerate,
        oc.remove_duplicate_faces,
        oc.remove_loose_vertices,
        oc.recalc_outside,
        oc.fill_all_holes,
    ],
)
def test_an_op_returns_the_identical_object_when_nothing_is_wrong(op) -> None:
    box = _box()
    assert op(box) is box


def test_merge_by_distance_returns_the_identical_object_when_nothing_merges() -> None:
    box = _box()
    assert oc.merge_by_distance(box, 1e-5) is box


def test_clean_returns_the_identical_object_when_nothing_is_wrong() -> None:
    box = _box()
    out, report = oc.clean(box)
    assert out is box
    assert report == oc.CleanReport(0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("name", sorted(prim.GENERATORS))
def test_clean_on_a_clean_primitive_returns_the_same_object(name: str) -> None:
    defaults, builder = prim.GENERATORS[name]
    mesh = builder(**defaults)
    out, _report = oc.clean(mesh)
    assert out is mesh


# --- recalc_outside: the winding claims themselves ---------------------------


def test_recalc_outside_leaves_every_closed_shell_with_positive_volume() -> None:
    inverted = _fully_inverted(_box())
    assert _signed_volume(inverted) < 0.0
    fixed = oc.recalc_outside(inverted)
    assert _signed_volume(fixed) > 0.0
    assert _signed_volume(_box()) == pytest.approx(_signed_volume(fixed))


def test_recalc_outside_leaves_every_manifold_edge_traversed_oppositely() -> None:
    inverted = _fully_inverted(_box())
    fixed = oc.recalc_outside(inverted)
    a = adj.adjacency(fixed)
    two_use = a.edge_uses == 2
    assert two_use.all(), "a cube has no boundary or non-manifold edge"
    assert len(a.flipped_pairs) == 0
    assert (a.twin >= 0).all()


def _signed_volume(mesh: bm.Mesh) -> float:
    # Public API surfaces this only as a sign (Survey.inside_out_shells), so
    # the module-private helper is read directly here to pin the geometric
    # claim itself rather than just the yes/no count derived from it.
    return float(oc._face_fan_volume(mesh).sum()) / 6.0


def test_flipping_keeps_uv_rows_attached_to_their_corners() -> None:
    """A face that gets flipped and then flipped back by ``recalc_outside``
    must come back with *exactly* the uv rows it started with -- the round
    trip only reproduces the original array if each corner's own uv followed
    it through both reversals rather than being dropped or left in place."""
    box = _box()
    assert box.uv is not None, "box() is UV-unwrapped; this claim needs real uvs"
    one_flipped = _with_reversed_face(box, face=0)
    assert not np.array_equal(one_flipped.uv, box.uv), "the flip must have moved something"
    fixed = oc.recalc_outside(one_flipped)
    assert np.array_equal(fixed.uv, box.uv)
    assert np.array_equal(fixed.loops, box.loops)


# --- the size ceiling ---------------------------------------------------------


def test_clean_refuses_past_the_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oc, "MAX_CLEAN_CORNERS", 4)
    with pytest.raises(el.OpError):
        oc.clean(_box())


def test_clean_runs_under_the_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oc, "MAX_CLEAN_CORNERS", 10_000)
    box = _box()
    out, report = oc.clean(box)
    assert out is box
    assert report == oc.CleanReport(0, 0, 0, 0, 0, 0)


# --- every op's output validates ---------------------------------------------


@pytest.mark.parametrize(
    "mesh_factory",
    [
        lambda: _with_degenerate_face(_box()),
        lambda: _with_duplicate_face(_box()),
        lambda: _with_loose_vertex(_box()),
        lambda: _two_cubes_offset(_box()),
        lambda: _fully_inverted(_box()),
        lambda: _open_box(_box()),
    ],
)
def test_clean_output_always_validates(mesh_factory) -> None:
    mesh = mesh_factory()
    out, _report = oc.clean(mesh, fill_holes=True, recalc=True)
    bm.validate(out)


@pytest.mark.parametrize("name", sorted(prim.GENERATORS))
def test_individual_op_outputs_validate_on_every_primitive(name: str) -> None:
    defaults, builder = prim.GENERATORS[name]
    mesh = builder(**defaults)
    bm.validate(oc.remove_degenerate(mesh))
    bm.validate(oc.remove_duplicate_faces(mesh))
    bm.validate(oc.remove_loose_vertices(mesh))
    bm.validate(oc.merge_by_distance(mesh, 1e-5))
    bm.validate(oc.fill_all_holes(mesh))
    bm.validate(oc.recalc_outside(mesh))
