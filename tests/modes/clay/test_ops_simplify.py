"""Simplify: a flat merge that is free, then edge collapse down to a budget."""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_simplify as simp
from realmspinner.kernels.mesh import ops_topo, topo
from realmspinner.kernels.mesh import primitives as prim

from .topo_asserts import (
    assert_closed,
    assert_consistently_oriented,
    assert_wound_outward,
    edge_use_counts,
)

TOP = 3  # the +Y side: sides run -X, +X, -Y, +Y, -Z, +Z


def _box(n: int, *, uv: bool = True, tag_sides: bool = False) -> bm.Mesh:
    """A box of ``n`` x ``n`` quads per side, vertices shared across the surface.

    Faces come side by side (``side * n * n`` onward), each side its own UV
    island so every box edge is a seam, which is how a real unwrap looks. With
    ``tag_sides`` the material is the side index and the smooth flag is set on
    the even sides, so a merge that takes either from the wrong face shows.
    """
    verts: dict[tuple[int, int, int], int] = {}
    points: list[tuple[float, float, float]] = []

    def vid(p: tuple[float, float, float]) -> int:
        key = tuple(int(round(c * n * 2)) for c in p)
        if key not in verts:
            verts[key] = len(points)  # type: ignore[index]
            points.append(p)
        return verts[key]  # type: ignore[index]

    faces: list[list[int]] = []
    uvs: list[list[tuple[float, float]]] = []
    material: list[int] = []
    smooth: list[bool] = []
    for axis, (i, j, k) in enumerate(((0, 1, 2), (1, 2, 0), (2, 0, 1))):
        for sign in (-1, 1):
            side = axis * 2 + (sign > 0)
            for a in range(n):
                for b in range(n):
                    quad = [(0, 0), (1, 0), (1, 1), (0, 1)]
                    if sign < 0:
                        quad = quad[::-1]  # j x k points along +i, so flip the minus side
                    corner = []
                    for da, db in quad:
                        c = [0.0, 0.0, 0.0]
                        c[i] = sign * 0.5
                        c[j] = -0.5 + (a + da) / n
                        c[k] = -0.5 + (b + db) / n
                        corner.append(vid((c[0], c[1], c[2])))
                    faces.append(corner)
                    u0, v0 = side % 3 * 0.33, side // 3 * 0.5
                    uvs.append(
                        [(u0 + (a + da) / n * 0.3, v0 + (b + db) / n * 0.3) for da, db in quad]
                    )
                    material.append(side if tag_sides else 0)
                    smooth.append(bool(tag_sides and side % 2 == 0))
    base = bm.from_faces(np.array(points, dtype="f4"), faces, uvs if uv else None)
    return bm.Mesh(
        positions=base.positions,
        loops=base.loops,
        starts=base.starts,
        material=np.array(material, dtype="i4"),
        smooth=np.array(smooth, dtype=bool),
        uv=base.uv,
    )


def _centres(mesh: bm.Mesh) -> np.ndarray:
    counts = np.diff(mesh.starts.astype("i8"))
    return (
        np.add.reduceat(mesh.positions[mesh.loops].astype("f8"), mesh.starts[:-1], axis=0)
        / (counts[:, None])
    )


def _sides(mesh: bm.Mesh) -> np.ndarray:
    """Which box side each face lies on, from its centre (-X +X -Y +Y -Z +Z)."""
    centres = _centres(mesh)
    axis = np.argmax(np.abs(centres), axis=1)
    return axis * 2 + (centres[np.arange(len(centres)), axis] > 0)


def _replace(mesh: bm.Mesh, **fields: object) -> bm.Mesh:
    keep = {
        "positions": mesh.positions,
        "loops": mesh.loops,
        "starts": mesh.starts,
        "material": mesh.material,
        "smooth": mesh.smooth,
        "uv": mesh.uv,
    }
    keep.update(fields)
    return bm.Mesh(**keep)  # type: ignore[arg-type]


def _grid(nx: int, nz: int) -> bm.Mesh:
    """An ``nx`` by ``nz`` sheet of quads in the XZ plane, facing +Y."""
    xs, zs = np.arange(nx + 1, dtype="f4"), np.arange(nz + 1, dtype="f4")
    positions = np.stack(
        [np.repeat(xs, nz + 1), np.zeros((nx + 1) * (nz + 1), dtype="f4"), np.tile(zs, nx + 1)],
        axis=1,
    )

    def v(i: int, j: int) -> int:
        return i * (nz + 1) + j

    faces = [
        [v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)] for i in range(nx) for j in range(nz)
    ]
    uv = [
        [
            (i / nx, j / nz),
            ((i + 1) / nx, j / nz),
            ((i + 1) / nx, (j + 1) / nz),
            (i / nx, (j + 1) / nz),
        ]
        for i in range(nx)
        for j in range(nz)
    ]
    return bm.from_faces(positions, faces, uv)


def _stripped(mesh: bm.Mesh) -> bm.Mesh:
    return _replace(mesh, uv=None)


def _assert_sound(out: bm.Mesh) -> None:
    bm.validate(out)
    assert_closed(out)
    assert_consistently_oriented(out)
    assert_wound_outward(out)


# --- the flat merge: a subdivided box is six quads ---------------------------


def test_a_ten_by_ten_by_ten_subdivided_box_simplifies_to_six_faces_and_eight_vertices() -> None:
    m = _box(10)
    assert bm.face_count(m) == 600, "the input really is a thousand-odd-face block"
    out = simp.simplify(m, 100)
    assert bm.face_count(out) == 6
    assert len(out.positions) == 8
    assert set(np.diff(out.starts).tolist()) == {4}, "six quads, not triangles"


def test_a_subdivided_box_without_uvs_also_comes_down_to_six_quads() -> None:
    out = simp.simplify(_stripped(_box(10)), 100)
    assert bm.face_count(out) == 6 and len(out.positions) == 8
    assert out.uv is None


def test_the_simplified_box_is_valid_closed_consistently_oriented_and_wound_outward() -> None:
    _assert_sound(simp.simplify(_box(10), 100))


def test_the_simplified_box_keeps_its_extent() -> None:
    m = _box(10)
    out = simp.simplify(m, 100)
    lo, hi = bm.bounds(out)
    assert lo.tolist() == bm.bounds(m)[0].tolist()
    assert hi.tolist() == bm.bounds(m)[1].tolist()


def test_the_flat_merge_alone_reaches_a_budget_without_collapsing_any_edge(monkeypatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("an edge collapse ran although the flat merge met the budget")

    monkeypatch.setattr(simp, "_collapse", boom)
    assert bm.face_count(simp.simplify(_box(6), 6)) == 6


def test_a_budget_the_flat_merge_cannot_reach_is_still_met_by_collapsing() -> None:
    sphere = prim.uv_sphere(0.5, 32, 16)
    assert bm.face_count(sphere) == 512
    out = simp.simplify(sphere, 200)
    assert 4 <= bm.face_count(out) <= 200


# --- what a flat merge must never cross -------------------------------------


def test_materials_and_smooth_flags_survive_the_merge_on_the_side_they_came_from() -> None:
    m = _box(8, tag_sides=True)
    out = simp.simplify(m, 100)
    assert bm.face_count(out) == 6
    sides = _sides(out)
    assert sorted(sides.tolist()) == [0, 1, 2, 3, 4, 5]
    for face, side in enumerate(sides.tolist()):
        assert int(out.material[face]) == side, "a merged face took another side's material"
        assert bool(out.smooth[face]) == (side % 2 == 0), "a merged face took another side's flag"


def test_faces_of_two_materials_on_one_plane_are_never_merged() -> None:
    m = _box(4)
    centres = _centres(m)
    on_top = np.arange(bm.face_count(m)) // 16 == TOP
    material = np.where(on_top & (centres[:, 0] > 0), 2, np.where(on_top, 1, 0)).astype("i4")
    out = simp.simplify(_replace(m, material=material), 7)
    assert bm.face_count(out) == 7, "five plain sides and two top halves"
    assert sorted(set(out.material.tolist())) == [0, 1, 2]
    assert out.material.tolist().count(1) == 1 and out.material.tolist().count(2) == 1
    _assert_sound(out)


def test_faces_that_differ_only_in_their_smooth_flag_are_never_merged() -> None:
    m = _box(4)
    centres = _centres(m)
    on_top = np.arange(bm.face_count(m)) // 16 == TOP
    smooth = on_top & (centres[:, 0] > 0)
    out = simp.simplify(_replace(m, smooth=smooth), 7)
    assert bm.face_count(out) == 7, "one shading mode per face"
    assert out.smooth.tolist().count(True) == 1


def test_a_uv_seam_is_never_merged_across() -> None:
    m = _box(4)
    centres = _centres(m)
    shift = np.zeros_like(m.uv)
    on_top = np.arange(bm.face_count(m)) // 16 == TOP
    # Shift the +X half of the top by half the map: the two halves are one flat
    # surface with a seam down the middle, and a merge across it would pull the
    # texture of one half onto the other.
    shifted_faces = np.flatnonzero(on_top & (centres[:, 0] > 0))
    for f in shifted_faces.tolist():
        shift[m.starts[f] : m.starts[f + 1], 0] = 0.4
    out = simp.simplify(_replace(m, uv=m.uv + shift), 7)
    assert bm.face_count(out) == 7, "the seam holds the top in two faces"
    _assert_sound(out)


def test_without_the_seam_the_same_top_merges_into_one_face() -> None:
    m = _box(4)
    assert bm.face_count(simp.simplify(m, 7)) == 6


def test_a_flat_face_whose_uvs_were_warped_is_not_straightened() -> None:
    plain = _grid(4, 4)
    assert bm.face_count(simp.simplify(plain, 4)) == 1, "the unwarped sheet is one quad"
    uv = plain.uv.copy()
    # Push every corner at the sheet's centre vertex sideways: still continuous
    # across every edge, but no longer one affine map over the face.
    centre = np.flatnonzero(np.all(plain.positions[plain.loops][:, [0, 2]] == 2.0, axis=1))
    uv[centre, 0] += 0.07
    warped = simp.simplify(_replace(plain, uv=uv), 4)
    assert bm.face_count(warped) > 1, "merging would have thrown the warp away"


def test_a_vertex_straight_in_one_face_but_a_corner_in_another_is_kept() -> None:
    m = _box(4)
    centres = _centres(m)
    on_top = np.arange(bm.face_count(m)) // 16 == TOP
    material = np.where(on_top & (centres[:, 0] > 0), 2, 0).astype("i4")
    out = simp.simplify(_replace(m, material=material), 7)
    assert len(out.positions) == 10, "eight corners and the two ends of the material split"
    # Each end of the split sits on a side face's straight edge; the side keeps
    # it as a fifth corner rather than leaving the top's corner on its edge.
    assert sorted(np.diff(out.starts).tolist()) == [4, 4, 4, 4, 4, 5, 5]
    uses = edge_use_counts(out)
    assert set(uses.values()) == {2}, "a T-junction would leave an edge with one use"


def test_a_gently_bent_strip_is_not_chained_into_one_face() -> None:
    # A hundred and twenty quads, each 0.3 degrees from the last: every
    # neighbouring pair is inside the one-degree angle and the coplanarity
    # tolerance, so a merge that only compared neighbours would fold the whole
    # thirty-six degree arc into one polygon.
    steps, radius = 120, 10.0
    angle = np.radians(0.3) * np.arange(steps + 1)
    ring = np.stack([radius * np.sin(angle), radius * np.cos(angle)], axis=1)
    positions = np.concatenate(
        [
            np.stack([ring[:, 0], ring[:, 1], np.zeros(steps + 1)], axis=1),
            np.stack([ring[:, 0], ring[:, 1], np.ones(steps + 1)], axis=1),
        ]
    ).astype("f4")
    faces = [[i, i + 1, steps + 2 + i, steps + 1 + i] for i in range(steps)]
    out = simp.simplify(bm.from_faces(positions, faces), 4)
    bm.validate(out)
    # Within a degree of the seed is three or four quads, but the corners of the
    # fourth sit past the flatness tolerance from the seed's plane: pairs merge
    # (60 faces), runs of four (30) would mean the distance test went missing.
    assert steps // 3 < bm.face_count(out) < steps, "merged in short runs, never chained"
    # Whatever it did merge sits within the flatness tolerance of its own plane.
    extent = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    normals = bm.face_normals(out)
    unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    for f in range(bm.face_count(out)):
        pts = out.positions[bm.face(out, f)].astype("f8")
        assert np.abs((pts - pts.mean(axis=0)) @ unit[f]).max() <= 1e-4 * extent


def test_an_awkward_group_is_left_alone_and_the_rest_still_merges() -> None:
    # Left: a 3x3 block with a different-material middle, so the eight faces
    # around it are an annulus Merge Faces refuses. Right: a clean 3x3 block in
    # a material of its own. The refusal must cost only the annulus.
    left, right = _grid(3, 3), _grid(3, 3)
    shifted = right.positions + np.array([3.0, 0.0, 0.0], dtype="f4")
    positions = np.concatenate([left.positions, shifted])
    loops = np.concatenate([left.loops, right.loops + len(left.positions)])
    material = np.concatenate([np.zeros(9, dtype="i4"), np.full(9, 2, dtype="i4")])
    material[4] = 1
    sheet = bm.Mesh(
        positions=positions,
        loops=loops,
        starts=topo.starts_from_counts([4] * 18),
        material=material,
        smooth=np.zeros(18, dtype=bool),
        uv=None,
    )
    out = simp.simplify(sheet, 5)
    bm.validate(out)
    assert bm.face_count(out) == 10, "eight ring faces, the middle one and the merged block"


# --- edge collapse ----------------------------------------------------------


def test_a_uv_sphere_above_the_target_is_reduced_to_it_and_stays_a_sound_solid() -> None:
    m = prim.uv_sphere(0.5, 32, 16)
    out = simp.simplify(m, 200)
    assert bm.face_count(out) <= 200
    assert bm.face_count(out) >= 4
    _assert_sound(out)
    assert out.uv is not None and out.uv.shape == (len(out.loops), 2)


def test_a_sphere_without_uvs_reaches_the_target_exactly() -> None:
    out = simp.simplify(_stripped(prim.uv_sphere(0.5, 32, 16)), 150)
    assert bm.face_count(out) == 150
    _assert_sound(out)


def _border_positions(mesh: bm.Mesh, *, materials: bool) -> set[tuple[float, ...]]:
    """Positions of the vertices on an open boundary, or on a material change."""
    a = adj.adjacency(mesh)
    corner = np.arange(len(mesh.loops))
    across = np.where(a.twin >= 0, a.twin, corner)
    if materials:
        bad = (a.twin >= 0) & (mesh.material[a.corner_face] != mesh.material[a.corner_face[across]])
    else:
        bad = a.edge_uses[a.corner_edge] == 1
    verts = np.unique(np.concatenate([mesh.loops[bad], mesh.loops[a.next_corner[bad]]]))
    return {tuple(np.round(mesh.positions[v].astype("f8"), 5)) for v in verts}


def test_collapsing_keeps_every_face_material_and_the_border_between_them() -> None:
    m = prim.icosphere(0.5, 3)
    material = (_centres(m)[:, 1] > 0).astype("i4") * 3
    split = _replace(m, material=material, uv=None)
    out = simp.simplify(split, 150)
    _assert_sound(out)
    assert bm.face_count(out) <= 150
    assert set(out.material.tolist()) == {0, 3}
    kept = {tuple(np.round(p.astype("f8"), 5)) for p in out.positions}
    assert _border_positions(split, materials=True) <= kept, "a material border vertex was lost"
    assert _border_positions(out, materials=True) == _border_positions(split, materials=True)


def test_a_collapse_never_moves_a_vertex_on_an_open_boundary() -> None:
    sheet = _stripped(_grid(8, 8))
    # Give it relief so the interior has something to collapse and the quadrics
    # are not all zero.
    pos = sheet.positions.copy()
    pos[:, 1] = 0.3 * np.sin(pos[:, 0] * 0.9) * np.cos(pos[:, 2] * 0.7)
    relief = _replace(sheet, positions=pos)
    out = simp.simplify(relief, 20)
    bm.validate(out)
    assert bm.face_count(out) < 64, "the interior had to thin for this to prove anything"
    assert _border_positions(out, materials=False) == _border_positions(relief, materials=False)


def test_collapsing_a_textured_surface_rederives_uvs_instead_of_leaving_stale_ones() -> None:
    # u = x and v = z on a heightfield, so the right UV of any vertex is its own
    # position's x and z. A collapse that kept the removed vertex's UV (or the
    # kept one's) on a vertex that moved along its edge would be off by a
    # fraction of an edge.
    n = 12
    xs = np.linspace(0.0, 1.0, n + 1)
    gx, gz = np.meshgrid(xs, xs, indexing="ij")
    gy = 0.2 * np.sin(3.0 * gx) * np.cos(2.0 * gz)
    positions = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype("f4")

    def v(i: int, j: int) -> int:
        return i * (n + 1) + j

    cells = [(i, j) for i in range(n) for j in range(n)]
    faces = [[v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)] for i, j in cells]
    uv = [
        [(xs[a], xs[b]) for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))]
        for i, j in cells
    ]
    sheet = bm.from_faces(positions, faces, uv)
    out = simp.simplify(sheet, 40)
    bm.validate(out)
    assert bm.face_count(out) < 144, "something had to collapse for this to prove anything"
    at = out.positions[out.loops]
    assert np.abs(out.uv - at[:, [0, 2]]).max() < 1e-5


def _uv_areas(mesh: bm.Mesh) -> np.ndarray:
    t = mesh.uv.reshape(-1, 3, 2).astype("f8")
    return (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (
        t[:, 2, 0] - t[:, 0, 0]
    )


def test_a_collapse_never_reverses_the_uv_orientation_of_a_face() -> None:
    m = prim.uv_sphere(0.5, 32, 16)
    tris, _ = ops_topo.triangulate_faces(m, el.empty())
    before = _uv_areas(tris)
    assert (before > 0).all() or (before < 0).all(), "the sphere unwraps one way round"
    out = simp.simplify(m, 150)
    assert set(np.diff(out.starts).tolist()) == {3}
    after = _uv_areas(out)
    assert np.sign(after[0]) == np.sign(before[0])
    assert (np.sign(after) == np.sign(before[0])).all(), "a face's texture was flipped"


def test_a_vertex_where_two_shells_touch_is_never_moved_or_removed() -> None:
    # Two icospheres sharing one vertex: its faces fall into two separate fans,
    # and collapsing an edge there would glue the shells together (or drag the
    # second one with the first).
    ball = prim.icosphere(0.5, 1)
    apex = ball.positions[0].astype("f8")
    mirrored = 2.0 * apex - ball.positions.astype("f8")
    remap = np.concatenate([[0], np.arange(len(ball.positions) - 1) + len(ball.positions)])
    loops_b = remap[ball.loops].reshape(-1, 3)[:, ::-1].reshape(-1)  # a point mirror flips winding
    pair = bm.Mesh(
        positions=np.concatenate([ball.positions, mirrored[1:].astype("f4")]),
        loops=np.concatenate([ball.loops, loops_b]),
        starts=topo.starts_from_counts([3] * (2 * bm.face_count(ball))),
        material=np.zeros(2 * bm.face_count(ball), dtype="i4"),
        smooth=np.zeros(2 * bm.face_count(ball), dtype=bool),
        uv=None,
    )
    bm.validate(pair)
    out = simp.simplify(pair, 40)
    bm.validate(out)
    assert bm.face_count(out) < 160
    assert any(np.allclose(p, apex, atol=1e-6) for p in out.positions), "the shared vertex moved"
    a = adj.adjacency(out)
    pinch = int(np.flatnonzero(np.all(np.isclose(out.positions, apex, atol=1e-6), axis=1))[0])
    assert len(a.vertex_corners(pinch)) >= 6, "the two shells lost their hold on the shared vertex"


def test_the_result_never_drops_below_four_faces() -> None:
    out = simp.simplify(_stripped(prim.icosphere(0.5, 1)), 4)
    assert bm.face_count(out) >= 4
    _assert_sound(out)


def test_a_collapse_that_would_pinch_two_faces_onto_each_other_is_refused() -> None:
    # A triangular bipyramid whose equator has one very short edge A-B. It is the
    # cheapest collapse by far, but A and B also share the equator's third vertex
    # C without (A, B, C) being a face, so merging them would lay the two halves'
    # faces over each other and give edge B-C four uses.
    angles = np.radians([0.0, 3.0, 200.0])
    equator = [(np.cos(a), 0.0, np.sin(a)) for a in angles]
    points = np.array([(0, 1, 0), (0, -1, 0), *equator], dtype="f4")
    top, bottom, a, b, c = range(5)
    faces = [[top, a, b], [top, b, c], [top, c, a], [bottom, b, a], [bottom, c, b], [bottom, a, c]]
    ring = bm.from_faces(points, faces)
    if bm.face_normals(ring)[0][1] < 0:
        ring = bm.from_faces(points, [f[::-1] for f in faces])
    out = simp.simplify(ring, 4)
    _assert_sound(out)
    assert bm.face_count(out) == 4


def test_collapsing_a_bumpy_height_field_never_folds_a_face_over() -> None:
    rng = np.random.default_rng(3)
    n = 14
    xs = np.linspace(0.0, 1.0, n + 1)
    gx, gz = np.meshgrid(xs, xs, indexing="ij")
    height = rng.random(gx.shape) * 0.02
    positions = np.stack([gx.ravel(), height.ravel(), gz.ravel()], axis=1).astype("f4")

    def v(i: int, j: int) -> int:
        return i * (n + 1) + j

    cells = [(i, j) for i in range(n) for j in range(n)]
    faces = [[v(i, j), v(i, j + 1), v(i + 1, j + 1), v(i + 1, j)] for i, j in cells]
    sheet = bm.from_faces(positions, faces)
    out = simp.simplify(sheet, 100)
    bm.validate(out)
    assert bm.face_count(out) <= 100
    normals = bm.face_normals(out)
    up = normals[:, 1] / np.linalg.norm(normals, axis=1)
    # Every face of a single-valued surface faces up; a collapse that let one
    # turn a corner past the limit leaves a near-vertical sliver on the outline.
    assert up.min() > 0.5


# --- the uv of a flat merge --------------------------------------------------


def test_the_uvs_of_a_textured_box_survive_the_flat_merge() -> None:
    m = _box(10)
    out = simp.simplify(m, 100)
    assert out.uv is not None and bm.face_count(out) == 6

    m_sides, out_sides = _sides(m), _sides(out)
    original: dict[tuple[int, tuple[float, ...]], np.ndarray] = {}
    for f in range(bm.face_count(m)):
        for c in range(m.starts[f], m.starts[f + 1]):
            key = (int(m_sides[f]), tuple(np.round(m.positions[m.loops[c]].astype("f8"), 5)))
            original[key] = m.uv[c]
    for f in range(bm.face_count(out)):
        for c in range(out.starts[f], out.starts[f + 1]):
            key = (int(out_sides[f]), tuple(np.round(out.positions[out.loops[c]].astype("f8"), 5)))
            assert np.allclose(out.uv[c], original[key], atol=1e-6), (
                "a border corner lost the uv it had at that position"
            )


# --- no-ops and refusals ----------------------------------------------------


def test_a_target_at_or_above_the_face_count_returns_the_identical_object() -> None:
    m = _box(3)
    assert simp.simplify(m, bm.face_count(m)) is m
    assert simp.simplify(m, bm.face_count(m) + 100) is m


def test_a_mesh_with_nothing_legal_to_remove_comes_back_as_the_same_object() -> None:
    box = prim.box()  # six quads, every box edge a uv seam: nothing merges or collapses
    assert simp.simplify(box, 4) is box


def test_a_target_below_four_raises_a_clear_error() -> None:
    for bad in (3, 1, 0, -5):
        with pytest.raises(el.OpError, match="at least 4 faces"):
            simp.simplify(_box(3), bad)


def test_an_over_ceiling_mesh_is_refused_before_anything_is_derived(monkeypatch) -> None:
    monkeypatch.setattr(simp, "MAX_SIMPLIFY_CORNERS", 400)

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("work began before the size was checked")

    monkeypatch.setattr(simp, "adjacency", boom)
    monkeypatch.setattr(simp, "face_normals", boom)
    monkeypatch.setattr(simp, "_flat_merge", boom)
    big = _stripped(_grid(11, 11))  # 484 corners
    with pytest.raises(el.OpError, match="Simplify"):
        simp.simplify(big, 10)


def test_the_ceiling_is_halved_for_a_mesh_that_carries_uvs(monkeypatch) -> None:
    monkeypatch.setattr(simp, "MAX_SIMPLIFY_CORNERS", 400)
    textured = _grid(9, 9)  # 324 corners: fine bare, past half of 400 with uvs
    simp.simplify(_stripped(textured), 10)
    with pytest.raises(el.OpError, match="Simplify"):
        simp.simplify(textured, 10)


def test_a_mesh_of_big_polygons_is_refused_on_its_triangle_count_too(monkeypatch) -> None:
    # A hundred-sided prism has 600 corners but splits into 396 triangles; at the
    # quad-calibrated half-a-triangle-per-corner rate that is past a 700 ceiling
    # that its corner count alone passes.
    monkeypatch.setattr(simp, "MAX_SIMPLIFY_CORNERS", 700)
    prism = _stripped(prim.cylinder(0.5, 1.0, 100))
    assert len(prism.loops) <= 700
    with pytest.raises(el.OpError, match="triangles"):
        simp.simplify(prism, 10)


def test_the_real_ceiling_is_far_above_an_ordinary_model() -> None:
    assert simp.MAX_SIMPLIFY_CORNERS >= 20_000
    assert len(_box(10).loops) < simp.MAX_SIMPLIFY_CORNERS // 2
