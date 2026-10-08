"""The 2026-10-07 audit's mesh-operation findings clay-30 to clay-34, clay-65
and clay-67 (Mirror Copy under a sheared world basis, region inset's UVs,
Triangulate Faces triangulating the whole mesh, the join / extrude / inset
ceilings' ``np.unique`` cost, a merge of nothing, and a docstring that promised
a refusal dissolve lacks).

Each test's name is the claim, and each one fails against the code it was
written for. Timing claims are pinned structurally (what is called, what comes
back) and never by a wall clock, which a loaded CI machine would turn into a
coin toss.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops, ops_dissolve, ops_topo
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.kernels.mesh.document import Obj
from realmspinner.kernels.mesh.elements import ElementSel

# --- clay-30: Mirror Copy under a sheared world basis -------------------------


def _box() -> bm.Mesh:
    generator, build = prim.GENERATORS["box"]
    return build(**generator)


def _world_positions(mesh: bm.Mesh, matrix: np.ndarray) -> np.ndarray:
    pts = np.hstack([mesh.positions.astype("f8"), np.ones((len(mesh.positions), 1))])
    return (matrix @ pts.T).T[:, :3]


def test_mirror_world_refuses_or_matches_a_sheared_world_matrix() -> None:
    box = _box()
    parent = Obj(uid=1, name="parent", mesh=box, scale=np.array([1.0, 3.0, 1.0]))
    child = Obj(
        uid=2,
        name="child",
        mesh=box,
        rotation=m3.quat_from_axis_angle([0, 0, 1], np.radians(45)),
        parent=1,
    )
    world = m3.compose(parent.translation, parent.rotation, parent.scale) @ m3.compose(
        child.translation, child.rotation, child.scale
    )
    basis = world[:3, :3]
    assert abs(float(basis[:, 0] @ basis[:, 1])) > 0.1  # the chain really shears

    offset = 5.0
    for axis in (0, 1, 2):
        copy = ops.duplicate(child, 9)
        try:
            mirrored = ops.mirror_world(copy, axis, offset, world=world)
        except el.OpError:
            continue  # a named refusal is the other legal answer
        placed = m3.compose(mirrored.translation, mirrored.rotation, mirrored.scale)
        got = _world_positions(mirrored.mesh, placed)
        want = _world_positions(box, world)
        want[:, axis] = 2.0 * offset - want[:, axis]
        assert np.abs(got - want).max() < 1e-5, axis


def test_mirror_world_leaves_an_unsheared_chain_to_the_closed_form() -> None:
    """The bake only runs for the residual; a clean chain keeps the plain mirror mesh."""
    box = _box()
    obj = Obj(
        uid=2,
        name="o",
        mesh=box,
        translation=np.array([1.0, 2.0, 3.0]),
        rotation=m3.quat_from_axis_angle([0, 1, 0], np.radians(30)),
        scale=np.array([2.0, 1.0, 0.5]),
    )
    world = m3.compose(obj.translation, obj.rotation, obj.scale)
    mirrored = ops.mirror_world(ops.duplicate(obj, 9), 0, 1.5, world=world)
    plain = ops.mirror(ops.duplicate(obj, 9), 0)
    assert np.array_equal(mirrored.mesh.positions, plain.mesh.positions)


# --- clay-31: region inset UVs -------------------------------------------------


def _uv_area(mesh: bm.Mesh, faces: list[int]) -> float:
    total = 0.0
    for f in faces:
        s, e = mesh.starts[f], mesh.starts[f + 1]
        u = mesh.uv[s:e].astype("f8")
        total += abs(0.5 * np.sum(u[:, 0] * np.roll(u[:, 1], -1) - np.roll(u[:, 0], -1) * u[:, 1]))
    return total


def test_region_inset_gives_the_rim_nonzero_uv_area_and_shrinks_the_cap_uvs() -> None:
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], "f4")
    mesh = bm.from_faces(pos, [[0, 1, 2, 3]], uv=[[[0, 0], [1, 0], [1, 1], [0, 1]]])
    per_face, _ = ops_topo.inset_faces(mesh, ElementSel(faces=[0]), thickness=0.25)
    region, sel = ops_topo.inset_faces(mesh, ElementSel(faces=[0]), thickness=0.25, region=True)
    cap = int(sel.faces[0])
    rim = [f for f in range(bm.face_count(region)) if f != cap]

    assert _uv_area(region, rim) > 0.0
    s, e = region.starts[cap], region.starts[cap + 1]
    cap_uv = region.uv[s:e]
    assert cap_uv.max() < 1.0 - 1e-6 and cap_uv.min() > 1e-6  # shrank off the full 0..1 face
    # On a flat region the two modes are the same inset, so the textures agree.
    per_cap = per_face.uv[per_face.starts[0] : per_face.starts[1]]
    assert np.allclose(np.sort(cap_uv, axis=0), np.sort(per_cap, axis=0), atol=1e-6)
    assert _uv_area(region, rim) == pytest.approx(_uv_area(per_face, list(range(1, 5))), rel=1e-4)


# --- clay-32: Triangulate Faces ------------------------------------------------


def _quad_row(k: int) -> bm.Mesh:
    base = np.arange(k)
    pos = np.concatenate(
        [
            np.stack([base * 2.0 + dx, np.zeros(k), np.zeros(k) + dz], 1)
            for dx, dz in ((0, 0), (1, 0), (1, 1), (0, 1))
        ]
    ).astype("f4")
    # vertex j of quad i sits at j * k + i
    loops = np.stack([base, base + k, base + 2 * k, base + 3 * k], 1).reshape(-1)
    return bm.Mesh(
        positions=pos,
        loops=loops.astype("i4"),
        starts=(np.arange(k + 1) * 4).astype("i4"),
        material=np.zeros(k, "i4"),
        smooth=np.zeros(k, bool),
        uv=None,
    )


def test_triangulating_one_face_does_not_triangulate_the_rest_of_the_mesh(monkeypatch) -> None:
    from realmspinner.kernels.mesh import earclip

    handed: list[int] = []
    real = earclip.corner_triangles

    def spy(positions, loops, starts, normals):
        handed.append(len(loops))
        return real(positions, loops, starts, normals)

    monkeypatch.setattr(earclip, "corner_triangles", spy)
    mesh = _quad_row(500)
    out, sel = ops_topo.triangulate_faces(mesh, ElementSel(faces=[3]))
    assert handed == [4]
    assert len(sel.faces) == 2
    assert bm.face_count(out) == 499 + 2


def test_triangulate_faces_still_earclips_a_concave_face_in_the_selected_subset() -> None:
    # An L-shaped hexagon (concave) and a quad: the subset path must still earclip.
    pos = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [2, 0, 1],
            [1, 0, 1],
            [1, 0, 2],
            [0, 0, 2],
            [5, 0, 0],
            [6, 0, 0],
            [6, 0, 1],
            [5, 0, 1],
        ],
        "f4",
    )
    mesh = bm.from_faces(pos, [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9]])
    out, sel = ops_topo.triangulate_faces(mesh, ElementSel(faces=[0]))
    assert bm.face_count(out) == 1 + 4
    assert set(sel.faces.tolist()) == {1, 2, 3, 4}
    tri_area = 0.0
    for f in sel.faces.tolist():
        p = out.positions[out.loops[out.starts[f] : out.starts[f + 1]]].astype("f8")
        tri_area += 0.5 * abs(np.cross(p[1] - p[0], p[2] - p[0])[1])
    assert tri_area == pytest.approx(3.0)  # the L's area; a fan from corner 0 would overshoot


def test_triangulate_all_faces_refuses_past_its_ceiling_before_building_anything(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ops_topo, "MAX_TRIANGULATED_CORNERS", 39)
    mesh = _quad_row(10)  # 40 corners
    with pytest.raises(el.OpError, match="Triangulating"):
        ops_topo.triangulate_faces(mesh, ElementSel())
    out, _ = ops_topo.triangulate_faces(mesh, ElementSel(faces=[0, 1, 2]))  # 12 corners: fine
    assert bm.face_count(out) == 7 + 6


# --- clay-33 / clay-34: the ceilings' np.unique cost ----------------------------


def test_sort_based_unique_helpers_match_np_unique_on_random_inputs() -> None:
    rng = np.random.default_rng(2026)
    cases = [
        np.zeros(0, "i8"),
        np.array([5], "i8"),
        np.array([3, 3, 3], "i8"),
        *(rng.integers(-50, 50, size=n).astype("i8") for n in (2, 10, 1000, 50_000)),
        rng.integers(0, 10**12, size=20_000).astype("i8"),
    ]
    for values in cases:
        assert np.array_equal(ops_topo._unique_sorted(values), np.unique(values))
        _, inverse = np.unique(values, return_inverse=True)
        assert np.array_equal(ops_topo._unique_inverse(values), inverse.reshape(-1))
    for n, width in ((0, 3), (1, 3), (7, 3), (5000, 3), (60_000, 3), (5000, 2**40)):
        cells = rng.integers(-width, width + 1, size=(n, 3)).astype(
            "i8"
        )  # 2**40: past the packed key
        want = (
            np.unique(cells, axis=0, return_inverse=True)[1].reshape(-1) if n else np.zeros(0, "i8")
        )
        assert np.array_equal(ops_topo._unique_rows_inverse(cells), want)


class _NoNumpyUnique:
    """``ops_topo``'s ``np`` with ``unique`` taken away: the hash path it must not reach."""

    def __getattr__(self, name):
        if name == "unique":
            raise AssertionError("np.unique on the hot path")
        return getattr(np, name)


def _grid(n: int, uv: bool = False) -> bm.Mesh:
    xs = np.arange(n + 1, dtype="f4")
    X, Z = np.meshgrid(xs, xs)
    pos = np.stack([X.ravel(), np.zeros(X.size, "f4"), Z.ravel()], 1)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    return bm.Mesh(
        positions=pos,
        loops=q.reshape(-1).astype("i4"),
        starts=(np.arange(len(q) + 1) * 4).astype("i4"),
        material=np.zeros(len(q), "i4"),
        smooth=np.zeros(len(q), bool),
        uv=(np.random.default_rng(0).random((q.size, 2)).astype("f4") if uv else None),
    )


def test_join_weld_never_reaches_np_unique(monkeypatch) -> None:
    monkeypatch.setattr(ops_topo, "np", _NoNumpyUnique())
    # 2 x (40 x 40 quads) = 3,362 vertices (exact arm); 2 x (100 x 100) = 20,402 (gridded arm).
    for n in (40, 100):
        objs = [
            Obj(uid=1, name="a", mesh=_grid(n)),
            Obj(uid=2, name="b", mesh=_grid(n), translation=np.array([float(n), 0.0, 0.0])),
        ]
        out = ops.join(objs, eps=1e-4)
        assert bm.face_count(out) == 2 * n * n


def test_extrude_with_offset_at_its_ceiling_finishes_under_a_second(monkeypatch) -> None:
    """Structural: extrude with an offset and region inset skip np.unique's hash path."""
    monkeypatch.setattr(ops_topo, "np", _NoNumpyUnique())
    mesh = _grid(30)
    sel = ElementSel(faces=np.arange(bm.face_count(mesh)))
    out, _ = ops_topo.extrude_faces(mesh, sel, offset=0.1)
    assert bm.face_count(out) > bm.face_count(mesh)
    out, _ = ops_topo.inset_faces(mesh, sel, thickness=0.1, depth=0.05, region=True)
    assert bm.face_count(out) > bm.face_count(mesh)


# --- clay-65: a merge of nothing -----------------------------------------------


def test_join_of_two_empty_objects_refuses_in_merge_words_not_weld_words() -> None:
    box = _box()
    empty_mesh, _ = ops_topo.delete_faces(box, ElementSel(faces=np.arange(6)))
    assert len(empty_mesh.positions) == 0
    for eps in (1e-4, 0.0):
        with pytest.raises(el.OpError) as raised:
            ops.join(
                [Obj(uid=1, name="a", mesh=empty_mesh), Obj(uid=2, name="b", mesh=empty_mesh)],
                eps=eps,
            )
        assert "weld" not in str(raised.value).lower()
        assert "merge" in str(raised.value).lower()
    # One empty and one real object is still a merge.
    out = ops.join([Obj(uid=1, name="a", mesh=empty_mesh), Obj(uid=2, name="b", mesh=box)])
    assert bm.face_count(out) == 6


# --- clay-67: the docstrings ---------------------------------------------------


def test_dissolve_docstring_does_not_promise_a_non_manifold_refusal_it_lacks() -> None:
    # dissolve really does not refuse one (the result is not corrupt), so the
    # prose must not say it does.
    assert "non-manifold edge" not in (ops_dissolve.__doc__ or "")
    assert "non-manifold edge" not in (ops_topo.__doc__ or "")
    # ...and the claim is true of the code: three faces on one edge dissolve.
    # Quads A and B side by side; A's far edge (0-5) is also used by two extra
    # triangles, so it is non-manifold -- and it is on the outline being walked.
    pos = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [2, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
            [0, 1, 0.5],
            [0, -1, 0.5],
        ],
        "f4",
    )
    mesh = bm.from_faces(pos, [[0, 1, 4, 5], [1, 2, 3, 4], [5, 0, 6], [0, 5, 7]])
    out, _ = ops_dissolve.dissolve_faces(mesh, ElementSel(faces=[0, 1]))
    assert bm.face_count(out) == 3  # the merged quad pair and the two stray triangles
