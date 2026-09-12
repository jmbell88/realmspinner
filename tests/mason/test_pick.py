"""``mason/pick.py``: ray versus scene.

Every test name is a claim, written to fail against the easy wrong version of
the pick it covers: a BVH rebuilt per instance instead of per ref, a lock
that quietly makes an item unpickable, a terrain march that never terminates
or that reports a hit through solid ground.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.mason import nodes as nd
from warlock.studio.mason import pick, refs
from warlock.studio.mason import terrain as T
from warlock.studio.mason.scene import Placed
from warlock.studio.viewer import math3d as m3
from warlock.studio.viewer.gltf import Material, Primitive

IDENTITY = m3.identity()


def _cube_primitive() -> Primitive:
    """An axis-aligned unit cube, [-0.5, 0.5] on every side, two-sided
    winding not required (``ray_triangles`` is two-sided)."""
    positions = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype="f4",
    )
    faces = [
        (0, 1, 2), (0, 2, 3),  # -Z
        (4, 6, 5), (4, 7, 6),  # +Z
        (0, 3, 7), (0, 7, 4),  # -X
        (1, 5, 6), (1, 6, 2),  # +X
        (0, 4, 5), (0, 5, 1),  # -Y
        (3, 2, 6), (3, 6, 7),  # +Y
    ]
    indices = np.array(faces, dtype="u4").reshape(-1)
    return Primitive(positions=positions, indices=indices, material=Material())


def _world(translation) -> np.ndarray:
    return m3.compose(np.asarray(translation, dtype="f8"), m3.quat_identity(), m3.vec3(1, 1, 1))


def _placed(
    owner: int,
    world: np.ndarray,
    ref,
    *,
    node: nd.Node | None = None,
    visible: bool = True,
    locked: bool = False,
) -> Placed:
    real_node = node if node is not None else nd.MeshNode(uid=owner, ref=ref)
    return Placed(
        node=real_node,
        path=(owner,),
        owner=owner,
        world=world,
        visible=visible,
        locked=locked,
        static=False,
        ref=ref,
        material=None,
        prefab="",
        dangling=False,
    )


class _CountingSource:
    """A ``GeometrySource`` over one fixed cube mesh, counting resolutions.

    Hands back the *same* primitives list object on every call -- what a real
    host does once it has resolved a ref, and what makes it possible to tell
    "resolved again" apart from "rebuilt again": this module's own cache is
    supposed to notice the list did not change and skip the rebuild.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.rev = 0
        self._primitives = [_cube_primitive()]

    def primitives(self, ref):
        self.calls += 1
        return self._primitives

    def box(self, ref):
        return np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5])


def _grid_primitive(cx: float, cz: float, y: float, half: float = 0.5, n: int = 24) -> Primitive:
    """A flat, finely subdivided ``2*half`` square centred on ``(cx, cz)`` at
    height ``y`` -- ``2 * n * n`` triangles, comfortably over
    ``viewer.picking.BVH_MIN_TRIS`` so a real tree gets built and cached, not
    skipped for being too small to bother with.
    """
    steps = (np.arange(n + 1) / n - 0.5) * 2.0 * half
    xs, zs = np.meshgrid(cx + steps, cz + steps, indexing="ij")
    positions = np.stack([xs, np.full_like(xs, y), zs], axis=-1).reshape(-1, 3).astype("f4")
    tris = []
    for r in range(n):
        for c in range(n):
            v00, v10 = r * (n + 1) + c, (r + 1) * (n + 1) + c
            v01, v11 = r * (n + 1) + c + 1, (r + 1) * (n + 1) + c + 1
            tris.append((v00, v10, v01))
            tris.append((v10, v11, v01))
    indices = np.array(tris, dtype="u4").reshape(-1)
    return Primitive(positions=positions, indices=indices, material=Material())


class _ReparseableSource:
    """A ``GeometrySource`` whose one ref's geometry can be swapped out from
    under it -- standing in for a relink, or the user re-exporting the same
    asset from Clay: the job id (the ref) never changes, only what it now
    resolves to.
    """

    def __init__(self, primitives: list[Primitive]) -> None:
        self._primitives = primitives
        self.calls = 0
        self.rev = 0

    def primitives(self, ref):
        self.calls += 1
        return self._primitives

    def box(self, ref):
        positions = self._primitives[0].positions
        return positions.min(axis=0).astype("f8"), positions.max(axis=0).astype("f8")

    def reparse(self, primitives: list[Primitive]) -> None:
        """A *new* list object -- same ref, different geometry."""
        self._primitives = primitives


def _box_ref(**params) -> refs.PrimitiveRef:
    return refs.primitive_ref("box", params)


DOWN = np.array([0.0, -1.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])


# --- basic hits ---------------------------------------------------------


def test_a_ray_down_the_axis_hits_one_of_six_instances_owner_and_no_other():
    source = _CountingSource()
    ref = _box_ref()
    placed = [_placed(100 + i, _world((3.0 * i, 0.0, 0.0)), ref) for i in range(6)]

    hit = pick.ray_scene(placed, source, origin=(9.0, 10.0, 0.0), direction=DOWN)

    assert hit is not None
    assert hit.owner == 103  # instance index 3 sits at x=9


def test_the_hit_point_lies_on_the_world_ray_at_distance():
    source = _CountingSource()
    ref = _box_ref()
    placed = [_placed(1, _world((0.0, 0.0, 0.0)), ref)]

    origin = np.array([0.0, 10.0, 0.0])
    hit = pick.ray_scene(placed, source, origin=origin, direction=DOWN)

    assert hit is not None
    np.testing.assert_allclose(hit.point, origin + DOWN * hit.distance)
    # The box's top face is at world y=0.5.
    assert hit.point[1] == pytest.approx(0.5)


def test_the_nearest_of_two_overlapping_items_wins():
    source = _CountingSource()
    ref = _box_ref()
    near = _placed(1, _world((0.0, 5.0, 0.0)), ref)
    far = _placed(2, _world((0.0, 0.0, 0.0)), ref)

    hit = pick.ray_scene([far, near], source, origin=(0.0, 20.0, 0.0), direction=DOWN)

    assert hit is not None
    assert hit.owner == 1


def test_a_ray_that_misses_everything_is_none():
    source = _CountingSource()
    ref = _box_ref()
    placed = [_placed(1, _world((0.0, 0.0, 0.0)), ref)]

    hit = pick.ray_scene(placed, source, origin=(100.0, 10.0, 0.0), direction=DOWN)

    assert hit is None


def test_a_hidden_item_is_not_hit():
    source = _CountingSource()
    ref = _box_ref()
    placed = [_placed(1, _world((0.0, 0.0, 0.0)), ref, visible=False)]

    hit = pick.ray_scene(placed, source, origin=(0.0, 10.0, 0.0), direction=DOWN)

    assert hit is None


def test_a_locked_item_is_hit_the_lock_is_reported_not_enforced():
    source = _CountingSource()
    ref = _box_ref()
    placed = [_placed(1, _world((0.0, 0.0, 0.0)), ref, locked=True)]

    hit = pick.ray_scene(placed, source, origin=(0.0, 10.0, 0.0), direction=DOWN)

    assert hit is not None
    assert hit.owner == 1
    assert hit.placed.locked is True  # reported...
    # ...and this module never used it to refuse the ray in the first place.


# --- one BVH per ref, not per instance ---------------------------------


def test_the_bvh_is_built_once_per_ref_rather_than_once_per_instance():
    source = _CountingSource()
    # Five *distinct* PrimitiveRef objects that all compare equal -- exactly
    # what five placements of "a box" in a real document each hold, since
    # every MeshNode gets its own ref object out of primitive_ref().
    placed = [_placed(200 + i, _world((3.0 * i, 0.0, 0.0)), _box_ref()) for i in range(5)]

    hit = pick.ray_scene(placed, source, origin=(6.0, 10.0, 0.0), direction=DOWN)

    assert hit is not None
    # One resolution for the whole call, not five -- ray_scene's own local
    # memo (keyed on ref_key, not on which instance asked) is what this
    # would catch failing.
    assert source.calls == 1


def test_the_bvh_is_built_once_per_ref_even_across_two_different_but_equal_refs():
    """Two placements built through separate ``primitive_ref()`` calls with
    identical parameters are two different Python objects and one ``ref_key``
    -- this is the case that fails if the cache were keyed by ``id(ref)``
    instead of by value."""
    source = _CountingSource()
    ref_a = _box_ref(size=(1.0, 1.0, 1.0))
    ref_b = _box_ref(size=(1.0, 1.0, 1.0))
    assert ref_a is not ref_b
    assert ref_a == ref_b

    placed = [
        _placed(1, _world((0.0, 0.0, 0.0)), ref_a),
        _placed(2, _world((3.0, 0.0, 0.0)), ref_b),
    ]
    pick.ray_scene(placed, source, origin=(0.0, 10.0, 0.0), direction=DOWN)

    assert source.calls == 1


def test_ray_scene_does_not_rebuild_the_bvh_across_two_calls_with_unchanged_primitives(
    monkeypatch,
):
    """Two frames' worth of picks over a document nothing has touched must
    resolve the ref again each time (a real host's own cache is cheap to ask,
    and it is what notices a reparse) but never redo the concatenation-and-
    build work twice for geometry that has not changed -- the second half of
    what the ref-keyed cache used to get backwards, alongside picking the
    wrong shape after a reparse (see the test below).
    """
    build_calls = 0
    real_build_bvh = pick.picking.build_bvh

    def counting_build_bvh(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return real_build_bvh(*args, **kwargs)

    monkeypatch.setattr(pick.picking, "build_bvh", counting_build_bvh)

    source = _CountingSource()
    ref = _box_ref()
    placed = [_placed(1, _world((0.0, 0.0, 0.0)), ref)]

    pick.ray_scene(placed, source, origin=(0.0, 10.0, 0.0), direction=DOWN)
    pick.ray_scene(placed, source, origin=(0.0, 10.0, 0.0), direction=DOWN)

    assert source.calls == 2  # resolved once per call...
    assert build_calls == 1  # ...but only the first call had anything to build.


def test_a_reparsed_asset_does_not_keep_picking_against_its_old_geometry():
    """A relink, or the user re-exporting the same asset from Clay -- the
    plan names both as routine -- hands back a *new* primitives list for the
    same ref. The old geometry sat far from where this ray ever points; the
    new geometry sits directly under it. A cache keyed on the ref alone
    cannot tell the two apart and answers with the shape that is no longer
    there; this must answer with the one that is.
    """
    ref = _box_ref()
    far_away = [_grid_primitive(cx=100.0, cz=0.0, y=0.0)]
    here = [_grid_primitive(cx=0.0, cz=0.0, y=1.0)]
    source = _ReparseableSource(far_away)
    placed = [_placed(1, IDENTITY, ref)]

    miss = pick.ray_scene(placed, source, origin=(0.0, 10.0, 0.0), direction=DOWN)
    assert miss is None  # nothing is where the ray points -- yet

    source.reparse(here)  # same ref, a brand new list -- the geometry moved
    hit = pick.ray_scene(placed, source, origin=(0.0, 10.0, 0.0), direction=DOWN)

    assert hit is not None
    assert hit.point[1] == pytest.approx(1.0, abs=1e-3)
    assert hit.distance == pytest.approx(9.0, abs=1e-3)


# --- terrain: through ray_scene ------------------------------------------


def _flat_terrain(side: int = 4, size: float = 8.0, bump: float = 0.0) -> T.Terrain:
    heights = np.zeros((side + 1, side + 1), dtype="f4")
    if bump:
        heights[side // 2, side // 2] = bump
    return T.Terrain(heights=heights, size_x=size, size_z=size, material=Material())


def test_ray_scene_finds_the_terrain_when_nothing_else_is_in_the_way():
    source = _CountingSource()
    terrain = _flat_terrain(bump=2.0)
    terrain_placed = _placed(999, IDENTITY, None, node=nd.TerrainNode(uid=999))

    hit = pick.ray_scene(
        [terrain_placed], source, origin=(0.0, 10.0, 0.0), direction=DOWN,
        terrain=terrain, terrain_world=IDENTITY,
    )

    assert hit is not None
    assert hit.owner == 999
    assert hit.point[1] == pytest.approx(2.0, abs=1e-3)


# --- ray_terrain ----------------------------------------------------------


def test_a_terrain_march_hits_a_sculpted_hill_at_the_right_height():
    terrain = _flat_terrain(bump=3.0)
    found = pick.ray_terrain(terrain, IDENTITY, origin=(0.0, 20.0, 0.0), direction=DOWN)

    assert found is not None
    distance, point = found
    assert point[1] == pytest.approx(3.0, abs=1e-3)
    assert distance == pytest.approx(20.0 - 3.0, abs=1e-3)


def test_a_ray_parallel_to_and_above_a_terrain_terminates_and_returns_none():
    terrain = _flat_terrain(bump=3.0)
    # Well above the highest point on the field, running horizontally: it
    # never crosses, and the march must still come back rather than loop.
    found = pick.ray_terrain(
        terrain, IDENTITY, origin=(-100.0, 50.0, 0.0), direction=np.array([1.0, 0.0, 0.0])
    )
    assert found is None


def test_a_ray_from_under_the_terrain_does_not_report_a_spurious_hit():
    terrain = _flat_terrain(bump=3.0)
    # Straight up from well below the field, at a point where the field's
    # own height never comes back down to meet it -- an above-to-below
    # crossing is the only kind ray_terrain reports, and a ray that starts
    # underneath only ever produces a below-to-above one on its way out.
    found = pick.ray_terrain(terrain, IDENTITY, origin=(0.0, -50.0, 0.0), direction=UP)
    assert found is None


def test_ray_terrain_returns_none_for_a_ray_that_misses_the_terrains_own_extent():
    terrain = _flat_terrain()
    found = pick.ray_terrain(
        terrain, IDENTITY, origin=(1000.0, 10.0, 0.0), direction=DOWN
    )
    assert found is None


def test_ray_terrain_honours_a_translated_world_matrix():
    terrain = _flat_terrain(bump=3.0)
    world = _world((5.0, 0.0, 0.0))  # the terrain's own node sits at x=5
    # In world space the bump is now at x=5 (local x=0); a ray straight down
    # through world x=7 (local x=2, still on the terrain but off the single
    # raised cell) must land at the flat height instead.
    off_bump = pick.ray_terrain(terrain, world, origin=(7.0, 20.0, 0.0), direction=DOWN)
    on_bump = pick.ray_terrain(terrain, world, origin=(5.0, 20.0, 0.0), direction=DOWN)

    assert on_bump is not None
    assert on_bump[1][1] == pytest.approx(3.0, abs=1e-3)
    assert off_bump is not None
    assert off_bump[1][1] == pytest.approx(0.0, abs=1e-3)


# --- markers: a light and a camera have no geometry at all --------------------


def _marker_placed(owner: int, world: np.ndarray, node: nd.Node) -> Placed:
    """A placement with **no ref**, which is what a light or a camera resolves
    to -- and therefore what nothing in this module could hit before
    :func:`pick.ray_marker` existed."""
    return Placed(
        node=node,
        path=(owner,),
        owner=owner,
        world=world,
        visible=True,
        locked=False,
        static=False,
        ref=None,
        material=None,
        prefab="",
        dangling=False,
    )


def test_a_light_is_picked_as_a_sphere_at_its_own_position():
    """**The regression.** A ``LightNode`` resolves to a placement with no ref,
    and the ray loop skipped every one of those outright -- so a placed light
    could be selected from the outliner and never from the viewport, which made
    the plan's own sentence ("pickable and gizmo-draggable like any node") false
    of two of the six node kinds.
    """
    source = _CountingSource()
    light = nd.LightNode(uid=7, kind="point")
    placed = [_marker_placed(7, _world((0.0, 0.0, 0.0)), light)]

    hit = pick.ray_scene(placed, source, origin=(0.0, 10.0, 0.0), direction=DOWN)

    assert hit is not None
    assert hit.owner == 7
    # The near surface of the symbol, not its centre: ``distance`` is a true
    # world distance for every kind of hit, which is what makes one comparable
    # against a mesh's.
    assert hit.distance == pytest.approx(10.0 - pick.MARK_SIZE)


def test_a_camera_is_picked_the_same_way():
    source = _CountingSource()
    placed = [_marker_placed(9, _world((2.0, 0.0, -3.0)), nd.CameraNode(uid=9))]
    hit = pick.ray_scene(placed, source, origin=(2.0, 10.0, -3.0), direction=DOWN)
    assert hit is not None and hit.owner == 9


def test_a_group_with_no_ref_is_still_not_pickable():
    """The marker pass must be about the two kinds that *draw* a symbol, not
    about "anything with no ref": a group is a transform and a terrain node is
    picked by its own height march, and a sphere around either would put an
    invisible click target in the middle of the scene."""
    source = _CountingSource()
    for node in (nd.GroupNode(uid=3), nd.TerrainNode(uid=4)):
        placed = [_marker_placed(node.uid, _world((0.0, 0.0, 0.0)), node)]
        assert pick.ray_scene(placed, source, origin=(0.0, 10.0, 0.0), direction=DOWN) is None


def test_a_hidden_light_is_not_picked():
    source = _CountingSource()
    light = _marker_placed(7, _world((0.0, 0.0, 0.0)), nd.LightNode(uid=7, kind="point"))
    hidden = Placed(**{**light.__dict__, "visible": False})
    assert pick.ray_scene([hidden], source, origin=(0.0, 10.0, 0.0), direction=DOWN) is None


def test_a_light_behind_a_prop_loses_to_the_prop():
    """Folded into the same nearest-wins loop rather than tested in a pass of its
    own, so what a click selects agrees with what the depth-tested marker overlay
    shows: a light behind a wall is behind the wall."""
    source = _CountingSource()
    ref = _box_ref()
    wall = _placed(1, _world((0.0, 5.0, 0.0)), ref)
    light = _marker_placed(2, _world((0.0, 0.0, 0.0)), nd.LightNode(uid=2, kind="point"))

    hit = pick.ray_scene([light, wall], source, origin=(0.0, 20.0, 0.0), direction=DOWN)
    assert hit is not None and hit.owner == 1


def test_a_marker_is_the_same_size_however_the_node_is_scaled():
    """``mason_marks`` draws a marker at a fixed world size because a light
    scaled to five is not a bigger light; the hit radius has to agree, or the
    symbol's clickable area is a lie."""
    source = _CountingSource()
    world = _world((0.0, 0.0, 0.0)) @ m3.scaling((5.0, 5.0, 5.0))
    placed = [_marker_placed(7, world, nd.LightNode(uid=7, kind="point"))]
    hit = pick.ray_scene(placed, source, origin=(0.0, 10.0, 0.0), direction=DOWN)
    assert hit is not None
    assert hit.distance == pytest.approx(10.0 - pick.MARK_SIZE)
    # Just outside the fixed radius is a miss, scale or no scale.
    missed = pick.ray_scene(
        placed, source, origin=(pick.MARK_SIZE * 2.0, 10.0, 0.0), direction=DOWN
    )
    assert missed is None


def test_a_ray_starting_inside_a_marker_reports_zero_rather_than_the_far_side():
    """So a click that began inside a light's symbol selects it instead of
    reaching through to whatever the exit point would have been nearest to."""
    assert pick.ray_marker((0.0, 0.0, 0.0), DOWN, IDENTITY) == pytest.approx(0.0)


def test_a_marker_behind_the_ray_is_not_hit():
    """Both roots negative: the sphere is behind the camera, not in front of it."""
    assert pick.ray_marker((0.0, 10.0, 0.0), np.array([0.0, 1.0, 0.0]), IDENTITY) is None
