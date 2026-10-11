"""Regressions for the 2026-10-10 audit's clay-10, clay-11, clay-34 and clay-35:
region connectivity in Merge Faces / extrude offset / region inset, the cavity
pass of Recalculate Normals, Linear Subdivide's whole-mesh ceiling and the
pair-walk of Collapse (``ops_dissolve``, ``ops_topo``, ``ops_clean``,
``ops_subdiv``). One file because the findings are small and share no state;
each test names the finding it closes.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_clean, ops_dissolve, ops_subdiv, ops_topo
from realmspinner.kernels.mesh.elements import OpError

# --- clay-10: a flipped neighbour is part of the same region ----------------


def _strip(flipped: tuple[int, ...] = ()) -> bm.Mesh:
    """Three unit quads in a row on z = 0, each facing +z except the *flipped* ones.

    A flipped quad shares its edges with its neighbours in the *same* direction
    rather than the opposite one, which is what an import's inside-out face
    looks like to ``adjacency`` (``twin == -1``, ``edge_uses == 2``).
    """
    pos = [[i, j, 0] for i in range(4) for j in range(2)]
    faces = []
    for i in range(3):
        quad = [2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1]
        faces.append(quad[::-1] if i in flipped else quad)
    return bm.from_faces(pos, faces)


def test_merge_faces_and_extrude_offset_treat_a_flipped_neighbour_as_part_of_the_same_region() -> (
    None
):
    """The 2026-10-10 audit's clay-10: region connectivity was built from
    ``a.twin >= 0``, and ``twin`` is -1 on a flipped pair, so two selected faces
    sharing an edge but wound against each other were "unconnected" here while
    ``topo.region_boundary_corners`` treated the same edge as shared. Extrude
    with an offset gave a flipped middle quad its own region (caps at
    ``[1, 0, 0, -1]``-style z values, the shared vertices averaging two
    opposite directions to nothing) where the one connected block moves as one.
    """
    m = _strip(flipped=(1,))
    out, _ = ops_topo.extrude_faces(m, el.ElementSel(faces=[0, 1, 2]), offset=1.0)
    caps = out.positions[len(m.positions) :, 2]
    # One region: the mean normal of (+z, -z, +z) is +z, and every cap vertex
    # travels it.
    assert np.allclose(caps, 1.0), caps.tolist()


def test_merge_faces_on_a_flipped_pair_is_not_told_the_faces_are_not_touching() -> None:
    """The same finding at ``dissolve_faces``: two faces over one shared edge
    are touching whichever way they are wound. The message used to say "needs at
    least two of them touching" about a pair that shares an edge; the pair still
    cannot be merged (its outline visits a vertex twice) but the refusal now
    says so rather than denying the contact."""
    pos = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    flipped = bm.from_faces(pos, [[0, 1, 2], [3, 1, 2]])
    with pytest.raises(OpError) as caught:
        ops_dissolve.dissolve_faces(flipped, el.ElementSel(faces=[0, 1]))
    assert "touching" not in str(caught.value)


def test_region_inset_depth_treats_a_flipped_neighbour_as_part_of_the_same_region() -> None:
    """``inset_faces(region=True, depth=...)`` takes its depth direction from the
    same ``_region_offsets`` grouping, so the flipped middle quad of a strip used
    to be pushed the other way and tear the cap along its shared edges."""
    m = _strip(flipped=(1,))
    out, _ = ops_topo.inset_faces(
        m, el.ElementSel(faces=[0, 1, 2]), thickness=0.0, depth=1.0, region=True
    )
    caps = out.positions[len(m.positions) :, 2]
    assert np.allclose(caps, 1.0), caps.tolist()


def test_a_consistent_strip_still_extrudes_as_one_region() -> None:
    """The unflipped baseline the three tests above are measured against."""
    m = _strip()
    out, _ = ops_topo.extrude_faces(m, el.ElementSel(faces=[0, 1, 2]), offset=1.0)
    assert np.allclose(out.positions[len(m.positions) :, 2], 1.0)


# --- clay-11: Recalculate Normals' cavity pass -----------------------------

_TETRA_V = np.array([(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10)], dtype="f8")
_TETRA_F = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]  # outward
_CUBE_V = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype="f8")
_CUBE_F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]


def _nested_boxes(n_cand: int, n_cont: int) -> bm.Mesh:
    """*n_cont* tetrahedra and *n_cand* inside-out cubes whose bounding boxes
    all sit inside every tetrahedron's but whose volumes do not: each cube is a
    cavity candidate that every container's box holds and none encloses."""
    pos: list[list[float]] = []
    faces: list[list[int]] = []
    for _ in range(n_cont):
        base = len(pos)
        pos.extend(_TETRA_V.tolist())
        faces.extend([[base + c for c in f] for f in _TETRA_F])
    for _ in range(n_cand):
        base = len(pos)
        pos.extend((_CUBE_V * 0.5 + 8.0).tolist())
        faces.extend([[base + c for c in reversed(f)] for f in _CUBE_F])
    return bm.from_faces(np.array(pos), faces)


def test_recalculate_normals_with_nested_bounding_boxes_stays_under_a_second_at_the_cavity_pair_ceiling(  # noqa: E501
    monkeypatch,
) -> None:
    """The 2026-10-10 audit's clay-11: ``_cavity_shells`` was bounded only by
    ``MAX_CAVITY_PAIRS``, but every pair whose container box holds the
    candidate costs a ray test (~120-140 us); 80 inside-out cubes beside 800
    tetrahedra took 9.9 s on the frame thread, far under every other ceiling.
    The number of ray tests is now bounded too, and past the bound the plain
    "volume decides" rule applies, as it already does past ``MAX_CAVITY_PAIRS``.
    """
    mesh = _nested_boxes(40, 400)  # 17,600 pairs whose box holds the cube
    calls = []
    real = ops_clean._inside

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ops_clean, "_inside", counting)
    start = time.perf_counter()
    ops_clean.recalc_outside(mesh)
    elapsed = time.perf_counter() - start
    assert len(calls) <= 4_000, f"{len(calls)} ray tests on one press"
    assert elapsed < 1.0, f"Recalculate Normals took {elapsed:.2f}s"


def test_a_mesh_just_under_the_cavity_work_budget_still_casts_its_rays(monkeypatch) -> None:
    """Below ``MAX_CAVITY_WORK`` the pass is unchanged: every held pair is tested."""
    mesh = _nested_boxes(10, 230)  # 2,390 tests of 404 charged each
    calls = []
    real = ops_clean._inside
    monkeypatch.setattr(ops_clean, "_inside", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    ops_clean.recalc_outside(mesh)
    assert len(calls) == 2_390


def test_a_cavity_under_the_ray_test_budget_is_still_found() -> None:
    """Below the budget nothing changes: a cube wound inward inside a cube wound
    outward is a cavity and keeps its inward winding, a stray inside-out cube is
    turned out."""
    pos, faces = [], []
    for scale, shift, inward in ((10.0, 0.0, False), (2.0, 4.0, True), (1.0, 40.0, True)):
        base = len(pos)
        pos.extend((_CUBE_V * scale + shift).tolist())
        for f in _CUBE_F:
            faces.append([base + c for c in (reversed(f) if inward else f)])
    mesh = bm.from_faces(np.array(pos), faces)
    out = ops_clean.recalc_outside(mesh)
    # The inner cube (faces 6..11) is the cavity wall: untouched. The stray one
    # (12..17) is flipped outward.
    assert out.loops[24:48].tolist() == mesh.loops[24:48].tolist()
    assert out.loops[48:72].tolist() != mesh.loops[48:72].tolist()


# --- clay-34: Linear Subdivide of one face on a huge mesh ------------------


def _grid_mesh(n: int) -> bm.Mesh:
    xs = np.arange(n + 1, dtype="f4")
    x, y = np.meshgrid(xs, xs, indexing="ij")
    pos = np.stack([x.ravel(), y.ravel(), np.zeros((n + 1) ** 2, "f4")], 1)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v00 = (i * (n + 1) + j).ravel()
    v10, v11, v01 = v00 + (n + 1), v00 + (n + 2), v00 + 1
    loops = np.stack([v00, v10, v11, v01], 1).reshape(-1).astype("i4")
    f = n * n
    return bm.Mesh(
        positions=pos,
        loops=loops,
        starts=(np.arange(f + 1) * 4).astype("i4"),
        material=np.zeros(f, "i4"),
        smooth=np.zeros(f, bool),
    )


def test_a_refused_partial_subdivide_on_a_mesh_past_the_ceiling_does_not_tell_the_user_to_subdivide_a_part() -> (  # noqa: E501
    None
):
    """The 2026-10-10 audit's clay-34: one selected face on a mesh already past
    ``MAX_SUBDIVIDED_FACES`` was refused for the whole mesh's size and told to
    "Subdivide a part of it" -- the thing just tried. A selection cannot be made
    smaller than one face; the message now says the object is too big and names
    Separate."""
    side = int(ops_subdiv.MAX_SUBDIVIDED_FACES**0.5) + 2
    mesh = _grid_mesh(side)
    assert bm.face_count(mesh) > ops_subdiv.MAX_SUBDIVIDED_FACES
    with pytest.raises(OpError) as caught:
        ops_subdiv.subdivide(mesh, el.ElementSel(faces=[1000]))
    message = str(caught.value)
    assert "Subdivide a part of it" not in message
    assert "Separate" in message
    assert f"{bm.face_count(mesh):,}" in message


def test_a_selection_whose_growth_crosses_the_ceiling_is_still_told_to_subdivide_a_part(
    monkeypatch,
) -> None:
    """The remedy is right when the *selection* is what is too big."""
    mesh = _grid_mesh(8)  # 64 faces
    monkeypatch.setattr(ops_subdiv, "MAX_SUBDIVIDED_FACES", 100)
    with pytest.raises(OpError, match="Subdivide a part of it"):
        ops_subdiv.subdivide(mesh, el.ElementSel(faces=np.arange(20)))


# --- clay-35: Collapse's pair walk ----------------------------------------


def _disjoint_quads(n: int) -> bm.Mesh:
    k = np.arange(n)
    base = np.stack([(k % 700) * 2.0, (k // 700) * 2.0, np.zeros(n)], 1)
    pos = np.concatenate([base, base + [1, 0, 0], base + [1, 1, 0], base + [0, 1, 0]]).astype("f4")
    loops = np.stack([k, k + n, k + 2 * n, k + 3 * n], 1).reshape(-1).astype("i4")
    return bm.Mesh(
        positions=pos,
        loops=loops,
        starts=(np.arange(n + 1) * 4).astype("i4"),
        material=np.zeros(n, "i4"),
        smooth=np.zeros(n, bool),
    )


def test_collapse_builds_its_edge_pairs_without_a_numpy_call_per_selected_face(
    monkeypatch,
) -> None:
    """The 2026-10-10 audit's clay-35: ``collapse`` built each selected face's
    pairs with its own ``np.stack``/``np.roll``, ~10 us of numpy overhead a face
    that the pairs-only ceiling never counted -- 1.15 s at the ceiling on
    disjoint quads, where the constant had been measured on a shared-vertex
    grid. The pairs now come from one gather."""
    mesh = _disjoint_quads(2_000)
    calls = []
    real_stack, real_roll = np.stack, np.roll
    monkeypatch.setattr(np, "stack", lambda *a, **k: (calls.append(1), real_stack(*a, **k))[1])
    monkeypatch.setattr(np, "roll", lambda *a, **k: (calls.append(1), real_roll(*a, **k))[1])
    out, _ = ops_topo.collapse(mesh, el.ElementSel(faces=np.arange(2_000)))
    assert len(calls) < 100, f"{len(calls)} numpy calls for 2,000 faces"
    assert bm.face_count(out) == 0  # every collapsed quad is degenerate and gone
    # And the result is what the per-face loop gave: collapsing half leaves the
    # other half untouched.
    out, _ = ops_topo.collapse(mesh, el.ElementSel(faces=np.arange(1_000)))
    assert bm.face_count(out) == 1_000


@pytest.mark.perf
def test_collapse_at_its_pair_ceiling_on_disjoint_quads_stays_under_a_second() -> None:
    n = ops_topo.MAX_COLLAPSED_PAIRS // 4
    mesh = _disjoint_quads(n)
    sel = el.ElementSel(faces=np.arange(n))
    start = time.perf_counter()
    ops_topo.collapse(mesh, sel)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.9, f"collapse at the ceiling took {elapsed:.2f}s"
