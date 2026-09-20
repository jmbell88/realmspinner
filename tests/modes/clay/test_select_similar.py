"""Select-similar: area, normal, material, side count, length and valence,
each matched against a seed set rather than a single seed element."""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import select
from realmspinner.kernels.mesh.mesh import from_faces


def _mixed_box():
    """A 2x1x1 box: two faces of area 2 (the long sides) and four of area 1."""
    return bp.box((2.0, 1.0, 1.0))


# --- similar_area -----------------------------------------------------------


def test_similar_area_matches_only_the_equal_area_faces() -> None:
    m = _mixed_box()
    areas = 0.5 * np.linalg.norm(select.bm.face_normals(m), axis=1)
    big = np.flatnonzero(np.isclose(areas, 2.0))
    result = select.similar_area(m, [int(big[0])], tolerance=0.01)
    assert set(result.tolist()) == set(big.tolist())


def test_similar_area_tolerance_widens_the_match() -> None:
    m = _mixed_box()  # areas are 1 and 2 -- a 100% tolerance covers both
    result = select.similar_area(m, [0], tolerance=1.5)
    assert len(result) == 6


def test_similar_area_with_no_seed_returns_nothing() -> None:
    m = bp.box()
    assert len(select.similar_area(m, [])) == 0


# --- similar_normal -----------------------------------------------------


def test_similar_normal_matches_only_the_same_facing() -> None:
    m = bp.box()
    result = select.similar_normal(m, [0], tolerance=1.0)
    assert result.tolist() == [0]


def test_similar_normal_a_wide_tolerance_still_excludes_the_opposite_face() -> None:
    m = bp.box()
    result = select.similar_normal(m, [0], tolerance=89.0)
    assert 1 not in result.tolist(), "face 1 (+Y) points the opposite way from face 0 (-Y)"


def test_similar_normal_seed_normals_that_cancel_match_nothing() -> None:
    m = bp.box()  # faces 0 and 1 point opposite ways; their mean is ~zero
    result = select.similar_normal(m, [0, 1], tolerance=45.0)
    assert len(result) == 0


# --- similar_material ---------------------------------------------------


def test_similar_material_matches_the_seeds_own_slot() -> None:
    m = bp.box()
    material = m.material.copy()
    material[2] = 1
    from dataclasses import replace

    m = replace(m, material=material)
    assert select.similar_material(m, [0]).tolist() == [0, 1, 3, 4, 5]
    assert select.similar_material(m, [2]).tolist() == [2]


def test_similar_material_a_seed_spanning_two_slots_matches_both() -> None:
    m = bp.box()
    material = m.material.copy()
    material[2] = 1
    from dataclasses import replace

    m = replace(m, material=material)
    result = select.similar_material(m, [0, 2])
    assert result.tolist() == [0, 1, 2, 3, 4, 5]


# --- similar_sides --------------------------------------------------------


def test_similar_sides_matches_only_the_same_arity() -> None:
    positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]]
    faces = [[0, 1, 2, 3], [1, 4, 2]]  # a quad and a triangle
    m = from_faces(positions, faces)
    assert select.similar_sides(m, [0], tolerance=0).tolist() == [0]
    assert select.similar_sides(m, [1], tolerance=0).tolist() == [1]


def test_similar_sides_tolerance_widens_the_arity_band() -> None:
    positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]]
    faces = [[0, 1, 2, 3], [1, 4, 2]]
    m = from_faces(positions, faces)
    result = select.similar_sides(m, [1], tolerance=1)  # triangle +/- 1 side reaches the quad
    assert set(result.tolist()) == {0, 1}


# --- similar_length -------------------------------------------------------


def test_similar_length_matches_only_the_equal_length_edges() -> None:
    m = _mixed_box()  # X-direction edges are length 2, the rest length 1
    long_edges = np.array([[0, 1]], dtype="i4")  # a length-2 edge
    result = select.similar_length(m, long_edges, tolerance=0.01)
    assert len(result) == 4
    lengths = np.linalg.norm(
        m.positions[result[:, 0]].astype("f8") - m.positions[result[:, 1]].astype("f8"), axis=1
    )
    assert np.allclose(lengths, 2.0)


def test_similar_length_with_no_seed_edges_returns_nothing() -> None:
    m = bp.box()
    assert len(select.similar_length(m, np.zeros((0, 2), dtype="i4"))) == 0


# --- similar_valence ------------------------------------------------------


def test_similar_valence_on_a_cube_matches_every_corner() -> None:
    m = bp.box()  # every corner has valence 3
    result = select.similar_valence(m, [0], tolerance=0)
    assert len(result) == 8


def test_similar_valence_distinguishes_a_pole_from_the_rest() -> None:
    m = bp.GENERATORS["cone"][1](segments=8)
    # Vertex count = ring (8) + apex (1); the apex has valence 8, the ring
    # vertices have valence 4 (two ring edges, one side edge, one base edge
    # -- wait, the base cap gives them 3, plus the two ring neighbours and
    # the apex spoke). Whatever the true numbers are, apex and ring differ.
    a = select.adjacency(m)
    valence = np.bincount(a.edge_verts.reshape(-1), minlength=len(m.positions))
    apex = int(np.argmax(valence))
    ring_vertex = int(np.argmin(valence))
    result = select.similar_valence(m, [apex], tolerance=0)
    assert apex in result.tolist()
    assert ring_vertex not in result.tolist()


def test_similar_valence_tolerance_widens_the_band() -> None:
    m = bp.GENERATORS["cone"][1](segments=8)
    a = select.adjacency(m)
    valence = np.bincount(a.edge_verts.reshape(-1), minlength=len(m.positions))
    apex = int(np.argmax(valence))
    wide = select.similar_valence(m, [apex], tolerance=int(valence.max()))
    assert len(wide) == len(m.positions)


# --- registry: QUERIES rows --------------------------------------------------


def test_every_similar_query_is_registered() -> None:
    names = {
        "similar_area",
        "similar_normal",
        "similar_material",
        "similar_sides",
        "similar_length",
        "similar_valence",
    }
    assert names <= set(select.QUERIES)
    for name in names:
        query = select.QUERIES[name]
        assert query.name == name
        assert "faces" in query.args or "edges" in query.args or "verts" in query.args
