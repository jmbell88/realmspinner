"""``mason/terrain.py``: the height field, its brushes, and the mesh they feed.

Each brush test reaches straight for the module function -- no ``Terrain``
needed to ask "does this rect land where I expect" -- and the array-ownership
and mesh-memo tests go through :class:`~warlock.studio.mason.terrain.Terrain`
because those are claims about the instance, not the arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.mason import terrain as T
from warlock.studio.viewer.gltf import Material


def _flat(side: int, height: float = 0.0) -> np.ndarray:
    return np.full((side + 1, side + 1), height, dtype=np.float32)


def _terrain(heights: np.ndarray, size: float = 8.0) -> T.Terrain:
    return T.Terrain(heights=heights, size_x=size, size_z=size, material=Material())


# --- brush clipping ----------------------------------------------------------


def test_a_brush_wholly_outside_the_array_returns_none_and_touches_nothing():
    heights = _flat(8)
    before = heights.copy()
    result = T.raise_lower(heights, -100.0, -100.0, 4.0, 1.0)
    assert result is None
    np.testing.assert_array_equal(heights, before)


def test_a_brush_clipped_by_an_edge_lands_only_its_visible_part():
    heights = _flat(8)
    rect, _ = T.raise_lower(heights, 0.0, 0.0, 3.0, 1.0)
    x0, y0, x1, y1 = rect
    assert x0 == 0 and y0 == 0
    assert x1 <= heights.shape[1] and y1 <= heights.shape[0]
    # The centre sits on the corner vertex, so the clipped rect must not
    # reach past it in the direction that has no array left.
    assert x1 < heights.shape[1] or y1 < heights.shape[0]


def test_a_brush_changes_nothing_outside_its_returned_rect():
    heights = _flat(16)
    before = heights.copy()
    rect, sub = T.raise_lower(heights, 8.0, 8.0, 3.0, 2.0)
    x0, y0, x1, y1 = rect
    patched = before.copy()
    patched[y0:y1, x0:x1] = sub
    # Outside the rect, the patched array must be byte-identical to the
    # original -- a brush that nudged one cell past its own rect would pass
    # every other test here and still corrupt a neighbour's undo record.
    mask = np.ones_like(before, dtype=bool)
    mask[y0:y1, x0:x1] = False
    np.testing.assert_array_equal(patched[mask], before[mask])


@pytest.mark.parametrize(
    "call",
    [
        lambda h: T.raise_lower(h, 4.0, 4.0, 2.0, 0.0),
        lambda h: T.flatten(h, 4.0, 4.0, 2.0, 0.0, 1.0),
    ],
)
def test_a_brush_computing_the_identity_returns_none_rather_than_an_empty_write(call):
    heights = _flat(8, height=0.0)
    assert call(heights) is None


def test_smooth_on_an_already_flat_terrain_returns_none():
    heights = _flat(8, height=3.0)
    assert T.smooth(heights, 4.0, 4.0, 2.0, 1.0) is None


def test_the_returned_sub_array_is_not_a_view_into_heights():
    heights = _flat(8)
    _, sub = T.raise_lower(heights, 4.0, 4.0, 2.0, 5.0)
    before = heights.copy()
    sub[0, 0] = -999.0
    np.testing.assert_array_equal(heights, before)


def test_noise_with_the_same_seed_twice_is_byte_identical():
    heights = _flat(16)
    _, a = T.noise(heights, 8.0, 8.0, 4.0, 1.0, seed=123)
    _, b = T.noise(heights, 8.0, 8.0, 4.0, 1.0, seed=123)
    np.testing.assert_array_equal(a, b)


def test_noise_with_different_seeds_is_not_identical():
    heights = _flat(16)
    _, a = T.noise(heights, 8.0, 8.0, 4.0, 1.0, seed=123)
    _, b = T.noise(heights, 8.0, 8.0, 4.0, 1.0, seed=456)
    assert not np.array_equal(a, b)


def test_an_unknown_falloff_is_refused_naming_what_it_got():
    heights = _flat(8)
    with pytest.raises(ValueError, match="bogus"):
        T.raise_lower(heights, 4.0, 4.0, 2.0, 1.0, falloff="bogus")


# --- the zero-radius / non-finite guard -----------------------------------


def test_a_brush_of_zero_radius_is_refused_rather_than_dividing_by_it():
    """Radius 0 is ``0 / 0`` inside ``_falloff_weight`` -- NaN, which
    ``np.clip`` passes straight through. Every brush must refuse before that
    division rather than let the NaN ride the falloff into the array."""
    heights = _flat(8)
    calls = [
        lambda h: T.raise_lower(h, 4.0, 4.0, 0.0, 1.0),
        lambda h: T.smooth(h, 4.0, 4.0, 0.0, 1.0),
        lambda h: T.flatten(h, 4.0, 4.0, 0.0, 1.0, 1.0),
        lambda h: T.noise(h, 4.0, 4.0, 0.0, 1.0, seed=1),
    ]
    for call in calls:
        result = call(heights)
        assert result is None
        # Belt and braces: even if a future change stopped refusing outright,
        # nothing it hands back may be able to plant a NaN in the ground.
        patched = heights.copy()
        if result is not None:
            (x0, y0, x1, y1), sub = result
            patched[y0:y1, x0:x1] = sub
        assert np.isfinite(patched).all()


def test_a_non_finite_brush_amount_is_refused():
    heights = _flat(8)
    assert T.raise_lower(heights, 4.0, 4.0, 2.0, float("nan")) is None
    assert T.smooth(heights, 4.0, 4.0, 2.0, float("nan")) is None
    assert T.flatten(heights, 4.0, 4.0, 2.0, float("inf"), 1.0) is None
    assert T.flatten(heights, 4.0, 4.0, 2.0, 1.0, float("nan")) is None
    assert T.noise(heights, 4.0, 4.0, 2.0, float("nan"), seed=1) is None
    assert T.set_height(heights, (0, 0, 2, 2), float("inf")) is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_brush_centred_on_a_non_finite_point_is_refused_rather_than_raising(bad):
    """A brush centre is usually a mouse ray hitting the ground, and
    ``viewer/picking.ray_plane`` divides by a dot product that is exactly
    zero when the ray is parallel to the plane -- an eye level with a flat
    horizon, not a contrived input. Unguarded, ``_clip_circle``'s
    ``int(np.floor(cx))`` turns that into a ``ValueError`` (NaN) or an
    ``OverflowError`` (infinity) out of a mouse-move handler; every brush
    must come back ``None`` instead, the same as a zero radius."""
    heights = _flat(8)
    assert T.raise_lower(heights, bad, 4.0, 2.0, 1.0) is None
    assert T.raise_lower(heights, 4.0, bad, 2.0, 1.0) is None
    assert T.smooth(heights, bad, 4.0, 2.0, 1.0) is None
    assert T.smooth(heights, 4.0, bad, 2.0, 1.0) is None
    assert T.flatten(heights, bad, 4.0, 2.0, 1.0, 1.0) is None
    assert T.flatten(heights, 4.0, bad, 2.0, 1.0, 1.0) is None
    assert T.noise(heights, bad, 4.0, 2.0, 1.0, seed=1) is None
    assert T.noise(heights, 4.0, bad, 2.0, 1.0, seed=1) is None
    assert T.set_height(heights, (bad, 0, 2, 2), 1.0) is None
    assert T.set_height(heights, (0, bad, 2, 2), 1.0) is None


def test_set_height_replaces_an_explicit_rect_with_one_level():
    heights = _flat(8)
    rect, sub = T.set_height(heights, (2, 2, 5, 6), 7.5)
    assert rect == (2, 2, 5, 6)
    assert np.all(sub == 7.5)
    assert sub.shape == (4, 3)


def test_set_height_onto_the_same_level_returns_none():
    heights = _flat(8, height=1.0)
    assert T.set_height(heights, (1, 1, 4, 4), 1.0) is None


# --- Terrain construction ------------------------------------------------


def test_terrain_refuses_a_non_square_array():
    with pytest.raises(ValueError, match="square"):
        _terrain(np.zeros((5, 6), dtype=np.float32))


def test_terrain_refuses_a_side_over_max_terrain_side():
    side = T.MAX_TERRAIN_SIDE + 1
    with pytest.raises(ValueError, match="MAX_TERRAIN_SIDE"):
        _terrain(np.zeros((side + 1, side + 1), dtype=np.float32))


def test_terrain_refuses_a_side_under_the_floor():
    with pytest.raises(ValueError, match="at least"):
        _terrain(np.zeros((0, 0), dtype=np.float32))


def test_terrain_refuses_a_non_positive_size():
    with pytest.raises(ValueError, match="size_x"):
        T.Terrain(heights=_flat(4), size_x=0.0, size_z=8.0, material=Material())


def test_terrain_refuses_a_non_finite_height():
    heights = _flat(4)
    heights[1, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        _terrain(heights)


def test_each_construction_refusal_names_a_different_thing():
    """Four distinct causes must not collapse onto one shared message -- a UI
    that shows the exception text needs it to say which control is wrong."""
    messages = set()
    for make in (
        lambda: _terrain(np.zeros((5, 6), dtype=np.float32)),
        lambda: _terrain(
            np.zeros((T.MAX_TERRAIN_SIDE + 2, T.MAX_TERRAIN_SIDE + 2), dtype=np.float32)
        ),
        lambda: T.Terrain(heights=_flat(4), size_x=-1.0, size_z=8.0, material=Material()),
        lambda: _terrain(np.array([[0.0, np.nan], [0.0, 0.0]], dtype=np.float32)),
    ):
        try:
            make()
        except ValueError as exc:
            messages.add(str(exc))
    assert len(messages) == 4


def test_terrain_owns_its_array_a_caller_mutation_does_not_reach_it():
    source = _flat(4)
    terrain = _terrain(source)
    source[0, 0] = 999.0
    assert terrain.heights[0, 0] == 0.0


def test_terrain_side_is_cells_per_edge_not_vertex_count():
    terrain = _terrain(_flat(6))
    assert terrain.side == 6
    assert terrain.heights.shape == (7, 7)


# --- terrain_mesh --------------------------------------------------------


def test_terrain_mesh_produces_the_expected_vertex_and_index_counts():
    n = 5
    terrain = _terrain(_flat(n))
    prim = T.terrain_mesh(terrain)
    assert prim.positions.shape == ((n + 1) ** 2, 3)
    assert prim.indices.shape == (2 * n * n * 3,)


def test_terrain_mesh_normals_are_all_unit_length():
    heights = _flat(6)
    rng = np.random.default_rng(0)
    heights += rng.uniform(-1, 1, size=heights.shape).astype(np.float32)
    terrain = _terrain(heights)
    prim = T.terrain_mesh(terrain)
    lengths = np.linalg.norm(prim.normals, axis=1)
    np.testing.assert_allclose(lengths, 1.0, rtol=1e-5, atol=1e-6)


def test_a_flat_terrains_normals_all_point_plus_y():
    terrain = _terrain(_flat(4, height=2.0))
    prim = T.terrain_mesh(terrain)
    expected = np.broadcast_to(np.array([0.0, 1.0, 0.0]), prim.normals.shape)
    np.testing.assert_allclose(prim.normals, expected, atol=1e-6)


def test_every_triangle_of_a_flat_terrain_winds_upward():
    """Pins ``terrain_mesh``'s winding claim: this is what culls the wrong way
    the day two indices in a cell's triangle get swapped."""
    terrain = _terrain(_flat(3, height=1.0))
    prim = T.terrain_mesh(terrain)
    idx = prim.indices.reshape(-1, 3)
    a = prim.positions[idx[:, 0]]
    b = prim.positions[idx[:, 1]]
    c = prim.positions[idx[:, 2]]
    face_normal = np.cross(b - a, c - a)
    assert np.all(face_normal[:, 1] > 0)


def test_the_mesh_memo_is_returned_again_for_the_same_array():
    terrain = _terrain(_flat(4))
    assert T.terrain_mesh(terrain) is T.terrain_mesh(terrain)


def test_the_mesh_memo_is_rebuilt_after_a_rebind():
    terrain = _terrain(_flat(4))
    first = T.terrain_mesh(terrain)
    terrain.heights = _flat(4, height=5.0)
    second = T.terrain_mesh(terrain)
    assert first is not second


# --- height_at -------------------------------------------------------------


def test_height_at_is_exact_on_a_vertex():
    heights = _flat(4)
    heights[1, 2] = 3.5
    terrain = _terrain(heights, size=8.0)
    n = terrain.side
    # Vertex (i=2, j=1) in local space: x=(2/4-0.5)*8=0, z=(1/4-0.5)*8=-2.
    x = (2 / n - 0.5) * terrain.size_x
    z = (1 / n - 0.5) * terrain.size_z
    assert T.height_at(terrain, x, z) == pytest.approx(3.5)


def test_height_at_a_cell_centre_is_the_mean_of_its_four_corners():
    heights = _flat(2, height=0.0)
    heights[0, 0], heights[0, 1] = 0.0, 2.0
    heights[1, 0], heights[1, 1] = 4.0, 6.0
    terrain = _terrain(heights, size=2.0)
    n = terrain.side
    x = ((0.5) / n - 0.5) * terrain.size_x
    z = ((0.5) / n - 0.5) * terrain.size_z
    expected = (0.0 + 2.0 + 4.0 + 6.0) / 4.0
    assert T.height_at(terrain, x, z) == pytest.approx(expected)


def test_height_at_clamps_past_the_edge_instead_of_raising():
    terrain = _terrain(_flat(4, height=1.5))
    far_beyond = terrain.size_x * 10
    assert T.height_at(terrain, far_beyond, far_beyond) == pytest.approx(1.5)
    assert T.height_at(terrain, -far_beyond, -far_beyond) == pytest.approx(1.5)
