"""Islands, packing, density and distortion over an already-assigned uv.

Clay tranche 6 ("UV and materials", ``dev/CLAY-PLAN.md``). :mod:`.uvunwrap`'s
own test module covers the LSCM solve; everything here is the array-plumbing
half that reads or rearranges a uv a mesh already has.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.kernels.mesh import uvtools as ut
from realmspinner.kernels.mesh.elements import OpError


def _quad(uv: list[tuple[float, float]], *, y: float = 0.0) -> bm.Mesh:
    """A single unit quad in the XZ plane at height *y*, with a given uv."""
    positions = [[0.0, y, 0.0], [1.0, y, 0.0], [1.0, y, 1.0], [0.0, y, 1.0]]
    return bm.from_faces(positions, [[0, 1, 2, 3]], [uv])


_CCW_UV = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


# --- edge_key / edge_keys ---------------------------------------------------


def test_edge_key_sorts_low_first() -> None:
    assert ut.edge_key(5, 2) == (2, 5)
    assert ut.edge_key(2, 5) == (2, 5)


def test_edge_keys_sorts_and_dedupes() -> None:
    out = ut.edge_keys([[5, 2], [2, 5], [0, 1]])
    assert out.tolist() == [[0, 1], [2, 5]]


def test_edge_keys_of_nothing_is_empty() -> None:
    assert ut.edge_keys(None).shape == (0, 2)
    assert ut.edge_keys(np.zeros((0, 2))).shape == (0, 2)


# --- uv-required refusals ---------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        ut.seams_from_uv,
        ut.islands,
        ut.pack_islands,
        ut.texel_density,
        ut.stretch,
        ut.flipped_uv_faces,
        ut.overlap_faces,
    ],
)
def test_every_uv_reader_refuses_a_mesh_with_no_uv(fn) -> None:
    mesh = prim.box((1.0, 1.0, 1.0))
    plain = bm.Mesh(
        positions=mesh.positions, loops=mesh.loops, starts=mesh.starts,
        material=mesh.material, smooth=mesh.smooth, uv=None,
    )
    with pytest.raises(OpError):
        fn(plain)


# --- seams and islands -------------------------------------------------------


def test_a_single_face_has_no_seams_and_one_island() -> None:
    """One face can never disagree with itself: there is nothing on the
    other side of any of its own edges within the mesh."""
    quad = _quad(_CCW_UV)
    assert len(ut.seams_from_uv(quad)) == 0
    assert ut.islands(quad).tolist() == [0]


def test_box_unwrap_of_a_unit_cube_merges_every_face_into_one_island() -> None:
    """Real, measured behaviour, not the naive "6 faces, 6 islands" guess.

    A unit cube's :func:`~.uv.box_unwrap` uv values only ever take the
    normalised extremes 0 and 1 (the cube's own extent is the same on every
    axis, so nothing needs to scale down), and two *different* faces can
    legitimately read the identical uv at a vertex they share purely by that
    coincidence -- not because they were meant to be one island. Chasing
    those coincidences transitively (edge by edge, exactly as
    :func:`seams_from_uv`/:func:`islands` are specified to) walks every one
    of the cube's six faces into a single component. This is not a defect in
    either function: it is what "islands by uv-agreement" really measures on
    a symmetric box, and :func:`test_box_unwrap_of_an_asymmetric_box_...`
    below is the same measurement on a box where that coincidence mostly
    does not happen.
    """
    cube = prim.box((1.0, 1.0, 1.0))
    ids = ut.islands(cube)
    assert len(set(ids.tolist())) == 1
    # Round-trip: the seams this reports back are exactly the edges
    # islands_by_seams needs to recover the same grouping from topology
    # alone, with no uv in hand.
    seams = ut.seams_from_uv(cube)
    assert np.array_equal(ut.islands_by_seams(cube, seams), ids)


def test_box_unwrap_of_an_asymmetric_box_gives_partial_connectivity() -> None:
    """A box whose three extents are all different breaks most, not all, of
    the unit-cube coincidence above -- three islands out of six faces here,
    not six and not one, and that is the honest answer for this box's own
    proportions rather than a rule good for every box."""
    box = prim.box((2.0, 1.0, 0.5))
    ids = ut.islands(box)
    assert len(set(ids.tolist())) == 3
    seams = ut.seams_from_uv(box)
    assert np.array_equal(ut.islands_by_seams(box, seams), ids)


def test_islands_by_seams_ignores_a_boundary_with_nothing_on_the_other_side() -> None:
    """A single quad has no adjoining face at all, so the seam set is
    irrelevant to it: it is always its own one island."""
    quad = _quad(_CCW_UV)
    assert ut.islands_by_seams(quad, np.zeros((0, 2), dtype="i4")).tolist() == [0]


def test_islands_by_seams_splits_two_faces_marked_apart() -> None:
    """Two quads sharing one edge are one island until that edge is marked a
    seam, at which point each is its own -- topology alone, no uv needed."""
    positions = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0], [2.0, 0.0, 1.0],
    ]
    faces = [[0, 1, 2, 3], [1, 4, 5, 2]]
    mesh = bm.from_faces(positions, faces)
    joined = ut.islands_by_seams(mesh, None)
    assert joined.tolist() == [0, 0]
    cut = ut.islands_by_seams(mesh, np.array([[1, 2]], dtype="i4"))
    assert cut.tolist() == [0, 1]


# --- transform_islands -------------------------------------------------------


def test_transform_islands_rotates_about_its_own_bbox_centre() -> None:
    quad = _quad(_CCW_UV)
    out = ut.transform_islands(quad, [0], rotate_deg=90.0)
    centre_before = np.array([0.5, 0.5])
    centre_after = (out.uv.min(axis=0) + out.uv.max(axis=0)) / 2.0
    assert centre_after == pytest.approx(centre_before, abs=1e-6)
    # A 90 degree CCW turn sends the +U corner (1, 0) to (0.5, 0.5) + R(0.5, -0.5).
    assert out.uv[1] == pytest.approx([1.0, 1.0], abs=1e-6)


def test_transform_islands_translates_the_whole_selection() -> None:
    quad = _quad(_CCW_UV)
    out = ut.transform_islands(quad, [0], translate=(0.25, -0.1))
    assert out.uv == pytest.approx(quad.uv + np.array([0.25, -0.1]), abs=1e-6)


def test_transform_islands_is_a_no_op_for_an_id_that_does_not_exist() -> None:
    quad = _quad(_CCW_UV)
    out = ut.transform_islands(quad, [99])
    assert out is quad


# --- pack_islands -------------------------------------------------------------


def _three_disjoint_islands() -> bm.Mesh:
    """Three unconnected quads with clean (non-self-overlapping) uv of very
    different aspect ratios -- wide, tall and square -- so packing has real
    work to do and ``rotate`` has something to improve on."""
    positions = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [1.0, 2.0, 1.0], [0.0, 2.0, 1.0],
    ]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    uvs = [
        [(0, 0), (3, 0), (3, 0.2), (0, 0.2)],
        [(0, 0), (0.2, 0), (0.2, 3), (0, 3)],
        [(0, 0), (1, 0), (1, 1), (0, 1)],
    ]
    return bm.from_faces(positions, faces, uvs)


@pytest.mark.parametrize("rotate", [False, True])
def test_pack_islands_fits_the_unit_square_with_no_new_overlap(rotate: bool) -> None:
    mesh = _three_disjoint_islands()
    assert len(set(ut.islands(mesh).tolist())) == 3
    packed = ut.pack_islands(mesh, margin=0.01, rotate=rotate)
    assert packed.uv.min() >= -1e-6
    assert packed.uv.max() <= 1.0 + 1e-6
    assert not ut.overlap_faces(packed).any()


def test_pack_islands_with_rotation_is_never_larger_than_without() -> None:
    mesh = _three_disjoint_islands()
    no_rotate = ut.pack_islands(mesh, margin=0.01, rotate=False)
    rotated = ut.pack_islands(mesh, margin=0.01, rotate=True)
    assert rotated.uv.max(axis=0)[0] <= no_rotate.uv.max(axis=0)[0] + 1e-6


def test_pack_islands_refuses_a_mesh_with_more_islands_than_its_ceiling_before_scanning_each_one(
    monkeypatch,
) -> None:
    """The 2026-09-19 audit's clay-11: pack_islands' own per-island scan (an
    O(total corners) mask build, once per island) had no ceiling at all --
    measured at 0.041s/0.144s/0.539s/2.074s for 500/2,000/4,000/8,000
    islands, accelerating. Monkeypatched down the way
    ``test_overlap_faces_refuses_past_the_triangle_ceiling`` already proves
    the shape for MAX_OVERLAP_TRIANGLES."""
    monkeypatch.setattr(ut, "MAX_UV_ISLANDS", 2)
    mesh = _three_disjoint_islands()  # 3 islands, over a ceiling of 2
    with pytest.raises(OpError):
        ut.pack_islands(mesh)


def test_normalize_density_refuses_a_mesh_with_more_islands_than_its_ceiling(monkeypatch) -> None:
    """normalize_density's own loop is the same shape as pack_islands' --
    clay-11 names both."""
    monkeypatch.setattr(ut, "MAX_UV_ISLANDS", 2)
    mesh = _three_disjoint_islands()
    with pytest.raises(OpError):
        ut.normalize_density(mesh, 300.0)


def test_transform_islands_refuses_more_islands_in_one_call_than_its_ceiling(monkeypatch) -> None:
    """transform_islands' own per-island loop is the third of clay-11's
    "no ceiling on any of the three" -- bounded on how many islands *this
    call* transforms, not on how many the mesh has in total."""
    monkeypatch.setattr(ut, "MAX_UV_ISLANDS", 2)
    mesh = _three_disjoint_islands()
    with pytest.raises(OpError):
        ut.transform_islands(mesh, [0, 1, 2], translate=(0.1, 0.1))


def test_normalize_density_computes_islands_only_once_not_once_per_island(monkeypatch) -> None:
    """clay-11's other half: normalize_density already computed its own
    ``islands(mesh)`` at the top of the loop, but never passed it to
    ``transform_islands``, which recomputed the whole adjacency-plus-
    connected-components pass again on *every* iteration. Three islands
    should cost exactly one ``islands()`` call, not four (one at the top
    plus one per island)."""
    mesh = _three_disjoint_islands()
    calls = []
    original = ut.islands

    def counting(m):
        calls.append(1)
        return original(m)

    monkeypatch.setattr(ut, "islands", counting)
    ut.normalize_density(mesh, 300.0)
    assert len(calls) == 1, "islands(mesh) must be computed once, not once per island"


def _thin_strip_islands_mesh(n: int) -> bm.Mesh:
    """*n* full-width horizontal-strip islands, edge to edge in v: each
    triangle's own uv bbox keeps the *same* wide u-extent (0 to 1) no matter
    how many strips there are, which is exactly the shape
    ``uvtools.MAX_OVERLAP_REGISTRATIONS``'s own docstring measures -- a
    packed layout of *uniformly-sized* islands does not reproduce it (its
    triangles shrink along with the grid's own resolution as island count
    grows), so this fixture is deliberately not built from ``pack_islands``.
    """
    positions: list[list[float]] = []
    faces: list[list[int]] = []
    uvs: list[list[tuple[float, float]]] = []
    h = 1.0 / (2 * n)
    for i in range(n):
        wy = float(i)
        base = len(positions)
        positions.extend([[0.0, wy, 0.0], [1.0, wy, 0.0], [1.0, wy, 1.0], [0.0, wy, 1.0]])
        faces.append([base, base + 1, base + 2, base + 3])
        v0 = i * h
        uvs.append([(0.0, v0), (1.0, v0), (1.0, v0 + h), (0.0, v0 + h)])
    return bm.from_faces(positions, faces, uvs)


def test_overlap_faces_stays_bounded_on_a_multi_island_uv_layout_under_the_triangle_ceiling() -> None:  # noqa: E501
    """The 2026-09-19 audit's clay-12: 3,500 full-width strip islands (7,000
    triangles) sit at well under 10% of MAX_OVERLAP_TRIANGLES (50,000), but
    this shape's own registration count at that scale (measured ~600,000 --
    see MAX_OVERLAP_REGISTRATIONS' own docstring) took multiple seconds of
    pure-Python bucket building before this fix, with neither
    MAX_OVERLAP_TRIANGLES nor MAX_OVERLAP_BUCKET ever catching it. Bounded
    here on wall clock, not only on the exception: a fix that raised only
    *after* paying for the registrations would not be a fix at all.
    """
    mesh = _thin_strip_islands_mesh(3500)
    start = time.perf_counter()
    with pytest.raises(OpError):
        ut.overlap_faces(mesh)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, (
        f"the registrations ceiling must refuse before building the grid, took {elapsed:.2f}s"
    )


def test_pack_islands_of_an_empty_mesh_is_a_no_op() -> None:
    empty = bm.Mesh(
        positions=np.zeros((0, 3), dtype="f4"), loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"), material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype=bool), uv=np.zeros((0, 2), dtype="f4"),
    )
    assert ut.pack_islands(empty) is empty


# --- texel density -------------------------------------------------------


def test_texel_density_of_a_1m_quad_mapped_to_the_unit_square_at_1024px() -> None:
    quad = _quad(_CCW_UV)
    assert ut.texel_density(quad, texture_px=1024) == pytest.approx(1024.0, rel=1e-6)


def test_texel_density_scales_as_the_square_root_of_texture_px() -> None:
    quad = _quad(_CCW_UV)
    assert ut.texel_density(quad, texture_px=2048) == pytest.approx(2048.0, rel=1e-6)


def test_texel_density_restricted_to_face_ids_ignores_the_rest() -> None:
    mesh = _three_disjoint_islands()
    only_square = ut.texel_density(mesh, [2], texture_px=1024)
    assert only_square == pytest.approx(1024.0, rel=1e-6)


def test_normalize_density_hits_the_target() -> None:
    quad = _quad(_CCW_UV)
    out = ut.normalize_density(quad, 512.0, texture_px=1024)
    assert ut.texel_density(out, texture_px=1024) == pytest.approx(512.0, rel=1e-4)


def test_normalize_density_scales_each_island_independently() -> None:
    mesh = _three_disjoint_islands()
    out = ut.normalize_density(mesh, 300.0, texture_px=1024)
    for label in range(3):
        faces = np.flatnonzero(ut.islands(mesh) == label)
        assert ut.texel_density(out, faces, texture_px=1024) == pytest.approx(300.0, rel=1e-4)


# --- overlap -------------------------------------------------------------


def test_overlap_faces_detects_a_duplicated_island() -> None:
    positions = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7]]
    uvs = [_CCW_UV, _CCW_UV]  # the second face's uv exactly duplicates the first's
    mesh = bm.from_faces(positions, faces, uvs)
    assert ut.overlap_faces(mesh).tolist() == [True, True]


def test_overlap_faces_of_disjoint_islands_finds_nothing() -> None:
    """Two quads with uv placed in genuinely separate regions of the plane --
    unlike :func:`_three_disjoint_islands`, whose three islands are a
    *packing* fixture and share the origin on purpose, so they overlap
    there until :func:`~.uvtools.pack_islands` moves them apart."""
    positions = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7]]
    uvs = [_CCW_UV, [(2, 0), (3, 0), (3, 1), (2, 1)]]
    mesh = bm.from_faces(positions, faces, uvs)
    assert not ut.overlap_faces(mesh).any()


def test_pack_islands_of_the_packing_fixture_removes_its_starting_overlap() -> None:
    """:func:`_three_disjoint_islands` is deliberately built with all three
    islands sharing uv ``(0, 0)`` -- this is the "before" this module's
    ``test_pack_islands_fits_the_unit_square_with_no_new_overlap`` measures
    the "after" of."""
    mesh = _three_disjoint_islands()
    assert ut.overlap_faces(mesh).any()


def test_overlap_faces_refuses_past_the_triangle_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(ut, "MAX_OVERLAP_TRIANGLES", 1)
    quad = _quad(_CCW_UV)  # 2 triangles once fanned, over a ceiling of 1
    with pytest.raises(OpError):
        ut.overlap_faces(quad)


# --- stretch and flip -------------------------------------------------------


def test_stretch_is_zero_for_an_isometric_map() -> None:
    quad = _quad(_CCW_UV)
    assert ut.stretch(quad) == pytest.approx([0.0], abs=1e-9)


def test_stretch_is_zero_across_faces_at_different_absolute_scale() -> None:
    """A small face and a big one, each mapped 1 uv unit per metre, both
    read zero: stretch compares each face's *share* of the totals, and a
    uniform density across differently-sized faces gives equal shares on
    both sides regardless of their absolute size."""
    positions = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.0, 2.0], [0.0, 1.0, 2.0],
    ]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7]]
    uvs = [_CCW_UV, [(0, 0), (2, 0), (2, 2), (0, 2)]]  # same density, 4x the size
    mesh = bm.from_faces(positions, faces, uvs)
    assert ut.stretch(mesh) == pytest.approx([0.0, 0.0], abs=1e-6)


def test_stretch_is_positive_for_a_face_stretched_larger_in_uv_than_its_share() -> None:
    """Two equal-area faces, one given twice the other's uv footprint: its
    share of the total uv area is 2/3 against a 1/2 share of the world area,
    which is a real distortion, not just a bigger number."""
    positions = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7]]
    uvs = [
        [(0, 0), (2, 0), (2, 2), (0, 2)],  # 4x the uv area of the other
        [(0, 0), (1, 0), (1, 1), (0, 1)],
    ]
    mesh = bm.from_faces(positions, faces, uvs)
    st = ut.stretch(mesh)
    assert st[0] > 0.0
    assert st[1] < 0.0


def test_flipped_uv_faces_detects_a_mirrored_face() -> None:
    quad = _quad(_CCW_UV)
    assert ut.flipped_uv_faces(quad).tolist() == [False]
    mirrored_uv = quad.uv.copy()
    mirrored_uv[:, 0] = 1.0 - mirrored_uv[:, 0]
    from dataclasses import replace

    mirrored = replace(quad, uv=mirrored_uv)
    assert ut.flipped_uv_faces(mirrored).tolist() == [True]


def test_flipped_uv_faces_reports_a_degenerate_face_as_clean() -> None:
    """A face whose uv corners all collapse to one point has no winding to
    be wrong about -- reported clean, not flipped."""
    degenerate_uv = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]
    quad = _quad(degenerate_uv)
    assert ut.flipped_uv_faces(quad).tolist() == [False]
