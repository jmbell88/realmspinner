"""Collision proxies: box/sphere/capsule fits contain every input point, a
convex hull is closed and correctly wound, and a compound collider groups
loose parts. Tranche 7's pure-kernel half -- see ``colliders.py``'s own
docstring for the design choices these tests hold it to.

**What "contains every input vertex" means here for the three analytic
shapes.** :func:`colliders.fit_box`, :func:`colliders.fit_sphere` and
:func:`colliders.fit_capsule` each return an *exact* analytic fit in
``Collider.params`` and a *rendered approximation* of it in ``Collider.mesh``
-- a low-poly sphere or capsule mesh is necessarily inscribed in its own
analytic shape (its flat faces cut inside the curved surface between
vertices), so a literal "is this point inside the polygon" test on the mesh
would be false at any tessellation cheap enough to ship as a collider, for
reasons that have nothing to do with whether the *fit* is correct. These
tests therefore check containment against ``params`` -- the actual
mathematical claim the fitting algorithm makes and the one an exporter with a
native analytic collider (a real target's ``SphereCollider``/``SphereShape3D``)
would use -- and check the *mesh*'s structural validity (closed, manifold,
consistently wound) separately, via ``ops_clean.survey``/
``adjacency.check_manifold``, per this tranche's own brief. A box has no such
gap (its faces are flat, so the mesh perfectly represents the box), and that
one case is checked against the actual mesh geometry as well.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import colliders as cl
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_clean
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.elements import OpError

from .topo_asserts import assert_closed, assert_consistently_oriented, assert_wound_outward


def _assert_valid_shell(mesh: bm.Mesh) -> None:
    """The shared claim every collider mesh in this file makes: a closed,
    manifold, consistently and outward wound shell -- checked the way this
    tranche's brief asks, through ``ops_clean.survey`` and
    ``adjacency.check_manifold``, not just the cheaper ``topo_asserts`` pair,
    so a defect either would catch (a degenerate or duplicate face,
    specifically) is covered too.
    """
    report = adj.check_manifold(mesh)
    assert report.clean, report
    survey = ops_clean.survey(mesh)
    assert survey.degenerate_faces == 0
    assert survey.duplicate_faces == 0
    assert survey.loose_vertices == 0
    assert_closed(mesh)
    assert_consistently_oriented(mesh)
    assert_wound_outward(mesh)


def _translated(mesh: bm.Mesh, offset: tuple[float, float, float]) -> bm.Mesh:
    m = np.eye(4)
    m[:3, 3] = offset
    return bm.transformed(mesh, m)


def _rotation_y(degrees: float) -> np.ndarray:
    t = np.radians(degrees)
    return np.array(
        [[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]]
    )


# --- fit_box ------------------------------------------------------------------


def test_axis_aligned_box_recovers_the_exact_extents_and_centre():
    mesh = _translated(bp.box((2.0, 1.0, 4.0)), (3.0, -1.0, 0.5))
    col = cl.fit_box(mesh)
    assert col.kind == "box"
    assert col.params["size"] == pytest.approx((2.0, 1.0, 4.0), abs=1e-5)
    assert col.params["center"] == pytest.approx((3.0, -1.0, 0.5), abs=1e-5)
    _assert_valid_shell(col.mesh)


def test_box_contains_every_input_vertex():
    mesh = bp.torus(segments=10, sides=8)  # a shape whose AABB is not itself a box
    col = cl.fit_box(mesh)
    center = np.array(col.params["center"])
    rotation = np.array(col.params["rotation"])
    half = np.array(col.params["size"]) / 2.0 + 1e-6
    local = (mesh.positions.astype("f8") - center) @ rotation
    assert np.all(np.abs(local) <= half)


def test_oriented_box_is_tighter_than_the_axis_aligned_one_on_a_rotated_box():
    """The AABB of a box rotated off-axis is inflated by the rotation; a PCA
    OBB should recover close to the box's own (unrotated) extents -- the
    whole reason ``oriented=True`` exists rather than always using the AABB.
    """
    rotated = bm.transformed(bp.box((2.0, 1.0, 1.0)), _pad(_rotation_y(37.0)))
    aabb = cl.fit_box(rotated, oriented=False)
    obb = cl.fit_box(rotated, oriented=True)
    assert max(aabb.params["size"]) > max(obb.params["size"]) + 0.05
    assert sorted(obb.params["size"]) == pytest.approx([1.0, 1.0, 2.0], abs=1e-4)
    _assert_valid_shell(obb.mesh)


def _pad(rotation3: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rotation3
    return m


def test_oriented_box_falls_back_rather_than_refuses_on_a_flat_mesh():
    """A ``plane`` has no volume, so its own convex hull is coplanar and
    :func:`colliders._quickhull_core` refuses it -- ``fit_box`` must not
    propagate that refusal; a flat object still has a perfectly good (if
    paper-thin) oriented box.
    """
    col = cl.fit_box(bp.plane((2.0, 3.0)), oriented=True)
    # ``size`` is in the fitted *local* frame, so the near-zero extent is
    # whichever axis PCA assigned it to, not a fixed world index.
    assert min(col.params["size"]) < 1e-3
    _assert_valid_shell(col.mesh)


def test_fit_box_refuses_an_empty_mesh():
    empty = bm.Mesh(
        positions=np.zeros((0, 3)),
        loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"),
        material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype=bool),
    )
    with pytest.raises(OpError):
        cl.fit_box(empty)


# --- fit_sphere -----------------------------------------------------------


def test_sphere_contains_every_input_vertex():
    mesh = bp.uv_sphere(radius=1.0, segments=24, rings=12)
    col = cl.fit_sphere(mesh)
    dist = np.linalg.norm(mesh.positions.astype("f8") - np.array(col.params["center"]), axis=1)
    assert np.all(dist <= col.params["radius"] + 1e-6)
    _assert_valid_shell(col.mesh)


def test_sphere_fit_on_an_off_centre_cluster_finds_a_small_radius():
    """A tight cluster far from the origin should not fit a sphere whose
    radius is dominated by its distance from (0, 0, 0) -- a regression the
    naive "sphere at the origin covering everything" reading would have."""
    rng = np.random.default_rng(0)
    cluster = rng.normal(loc=[50.0, 0.0, 0.0], scale=0.05, size=(200, 3))
    mesh = bm.from_faces(cluster, [[0, 1, 2]])
    col = cl.fit_sphere(mesh)
    assert col.params["radius"] < 1.0
    dist = np.linalg.norm(cluster - np.array(col.params["center"]), axis=1)
    assert np.all(dist <= col.params["radius"] + 1e-6)


def test_fit_sphere_refuses_an_empty_mesh():
    empty = bm.Mesh(
        positions=np.zeros((0, 3)),
        loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"),
        material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype=bool),
    )
    with pytest.raises(OpError):
        cl.fit_sphere(empty)


# --- fit_capsule ------------------------------------------------------------


def test_capsule_contains_every_input_vertex():
    mesh = bp.capsule(radius=0.3, height=1.0, segments=24, rings=6)
    col = cl.fit_capsule(mesh)
    center = np.array(col.params["center"])
    axis = np.array(col.params["axis"])
    radius = col.params["radius"]
    half = col.params["half_height"]

    rel = mesh.positions.astype("f8") - center
    t = rel @ axis
    perp = rel - np.outer(t, axis)
    perp_dist = np.linalg.norm(perp, axis=1)
    t_clamped = np.clip(t, -half, half)
    seg_dist = np.sqrt((t - t_clamped) ** 2 + perp_dist**2)
    assert np.all(seg_dist <= radius + 1e-6)
    _assert_valid_shell(col.mesh)


def test_capsule_recovers_the_input_radius_and_half_height():
    mesh = bp.capsule(radius=0.3, height=1.0, segments=24, rings=6)
    col = cl.fit_capsule(mesh)
    assert col.params["radius"] == pytest.approx(0.3, abs=1e-3)
    assert col.params["half_height"] == pytest.approx(0.5, abs=1e-3)
    axis = np.array(col.params["axis"])
    assert abs(abs(axis[1]) - 1.0) < 1e-3  # axis is +-Y for a Y-axis input capsule


def test_fit_capsule_refuses_an_empty_mesh():
    empty = bm.Mesh(
        positions=np.zeros((0, 3)),
        loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"),
        material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype=bool),
    )
    with pytest.raises(OpError):
        cl.fit_capsule(empty)


# --- convex_hull --------------------------------------------------------------


def test_hull_of_a_cube_has_eight_vertices_and_twelve_triangles():
    col = cl.convex_hull(bp.box((1.0, 1.0, 1.0)))
    assert col.kind == "convex"
    assert col.params["vertex_count"] == 8
    assert col.params["face_count"] == 12
    assert col.params["reduced"] is False
    _assert_valid_shell(col.mesh)


def test_hull_of_a_tetrahedron_is_itself():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="f8")
    mesh = bm.from_faces(positions, [[0, 1, 2]])  # topology is irrelevant; only positions matter
    col = cl.convex_hull(mesh)
    assert col.params["vertex_count"] == 4
    assert col.params["face_count"] == 4
    _assert_valid_shell(col.mesh)


@pytest.mark.parametrize(
    "points",
    [
        pytest.param(np.zeros((6, 3)), id="coincident"),
        pytest.param(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            id="collinear",
        ),
        pytest.param(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
            id="coplanar",
        ),
        pytest.param(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), id="only-three-points"
        ),
    ],
)
def test_convex_hull_refuses_degenerate_input(points: np.ndarray):
    mesh = bm.Mesh(
        positions=points,
        loops=np.array([0, 1, 2], dtype="i4"),
        starts=np.array([0, 3], dtype="i4"),
        material=np.zeros(1, dtype="i4"),
        smooth=np.zeros(1, dtype=bool),
    )
    with pytest.raises(OpError):
        cl.convex_hull(mesh)


def test_hull_respects_max_faces_on_a_dense_sphere_and_stays_close():
    """A 32x16 UV sphere is already convex, so its *true* hull is (close to)
    every one of its own vertices -- ``max_faces=64`` forces a real
    reduction, not a no-op.

    **Tolerance, stated honestly.** The reduced hull is built from a
    farthest-point-sampled *subset* of the sphere's own surface points (see
    ``colliders.py``'s own module docstring for why FPS rather than a
    per-point volume-cost removal), so a point *not* in that subset sits
    outside the smaller, flatter-faced polyhedron by roughly that face's own
    sagitta -- this is not measurement noise, it is the geometric price of
    trading precision for a triangle budget. Measured on this exact input
    (34 vertices / 64 triangles from a 482-vertex sphere), the worst
    per-point outside distance was ``0.1169`` (11.7% of the radius); the
    tolerance below is ``0.15`` (15%), a deliberate margin above that
    measurement rather than a number backed into after the fact, and at that
    tolerance every point -- not merely 99% of them -- is covered on this
    input. The ``>= 99%`` bar the brief asks for is checked as stated, not
    tightened to 100%, because a different point distribution is not
    guaranteed the same margin.
    """
    mesh = bp.uv_sphere(radius=1.0, segments=32, rings=16)
    col = cl.convex_hull(mesh, max_faces=64)
    assert col.params["face_count"] <= 64
    assert col.params["reduced"] is True
    _assert_valid_shell(col.mesh)

    hull = col.mesh
    tris = [bm.face(hull, i)[:3] for i in range(bm.face_count(hull))]
    normals = []
    anchors = []
    for tri in tris:
        p = hull.positions[tri].astype("f8")
        n = np.cross(p[1] - p[0], p[2] - p[0])
        normals.append(n / np.linalg.norm(n))
        anchors.append(p[0])
    normals = np.array(normals)
    anchors = np.array(anchors)

    points = mesh.positions.astype("f8")
    outside = np.full(len(points), -np.inf)
    for n, a in zip(normals, anchors, strict=True):
        outside = np.maximum(outside, (points - a) @ n)

    tolerance = 0.15  # fraction of the sphere's radius (1.0 here) -- see docstring above
    within = (outside <= tolerance).mean()
    assert within >= 0.99, f"only {within:.1%} of points within {tolerance} of the reduced hull"


def test_convex_hull_is_deterministic():
    mesh = bp.uv_sphere(radius=1.0, segments=16, rings=8)
    a = cl.convex_hull(mesh, max_faces=64)
    b = cl.convex_hull(mesh, max_faces=64)
    assert np.array_equal(a.mesh.positions, b.mesh.positions)
    assert np.array_equal(a.mesh.loops, b.mesh.loops)
    assert np.array_equal(a.mesh.starts, b.mesh.starts)


def test_convex_hull_refuses_a_mesh_past_a_point_ceiling_before_running_quickhull(
    monkeypatch,
):
    """The 2026-09-19 audit's clay-10: :func:`cl._quickhull_core` had no
    ceiling of its own -- ``guard_limit = 20 * n + 64`` at line 513 is a
    convergence valve against a numerically stuck loop, not a size refusal
    -- so Convex Hull ran the whole pure-Python incremental hull over every
    deduplicated point with no bound at all: 2.68 s at 2,000 points, 11.16 s
    at 20,000 (see :data:`cl.MAX_HULL_POINTS`'s own docstring for the
    re-measurement). The point-count ceiling must fire before
    ``_quickhull_core`` ever runs, on the deduplicated point set -- proved
    here by making ``_quickhull_core`` itself an assertion failure.
    """

    def _boom(points, eps):
        raise AssertionError("_quickhull_core ran past the point ceiling")

    monkeypatch.setattr(cl, "_quickhull_core", _boom)

    rng = np.random.default_rng(0)
    n = cl.MAX_HULL_POINTS + 10
    positions = rng.normal(size=(n, 3))
    mesh = bm.from_faces(positions, [[0, 1, 2]])  # topology is irrelevant here

    with pytest.raises(OpError):
        cl.convex_hull(mesh)


def test_compound_refuses_a_part_past_the_point_ceiling_before_running_quickhull(
    monkeypatch,
):
    """The same clay-10 ceiling, exercised through :func:`cl.compound` --
    the audit names Compound (once per part) as one of the three routes
    into the unbounded quickhull, alongside Convex Hull and Box (Oriented).
    """

    def _boom(points, eps):
        raise AssertionError("_quickhull_core ran past the point ceiling")

    monkeypatch.setattr(cl, "_quickhull_core", _boom)

    rng = np.random.default_rng(0)
    n = cl.MAX_HULL_POINTS + 30
    n_tris = n // 3
    positions = rng.normal(size=(n_tris * 3, 3))
    faces = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(n_tris)]
    mesh = bm.from_faces(positions, faces)

    with pytest.raises(OpError):
        cl.compound(mesh, face_groups=[list(range(n_tris))])


def test_oriented_box_fit_falls_back_rather_than_refuses_past_the_hull_point_ceiling():
    """``fit_box``'s own contract (module docstring: "refuse nothing" for
    every fit but ``convex_hull``) must survive :data:`cl.MAX_HULL_POINTS`:
    ``_hull_points_for_fit`` already treats any :class:`OpError` out of
    ``_quickhull_core`` as "fall back to every vertex for the PCA" (the
    same path a degenerate/coplanar input already takes), so the new
    ceiling must route through that fallback rather than propagate past
    ``fit_box``.
    """
    mesh = bp.uv_sphere(radius=1.0, segments=120, rings=90)
    assert len(np.unique(mesh.positions, axis=0)) > cl.MAX_HULL_POINTS

    col = cl.fit_box(mesh, oriented=True)
    assert col.kind == "box"
    _assert_valid_shell(col.mesh)


# --- compound -----------------------------------------------------------------


def test_compound_refuses_a_mesh_with_more_loose_parts_than_the_part_ceiling_before_hulling_any(
    monkeypatch,
):
    """The 2026-09-22 audit's clay-12: `compound` already bounds each part's
    own point count (`MAX_HULL_POINTS`), but nothing bounded how many *parts*
    it would hull -- a mesh with thousands of loose pieces ran `_quickhull_core`
    once per shell, in a Python loop, on the frame thread. Reproduced (audit's
    own probe): 0.89s at 1,000 parts, 5.3s at 6,000; re-measured at merge on a
    distinct-cube-per-part mesh: 0.67s at 700, 0.77s at 800, 0.86s at 900.

    Driven here with the ceiling lowered and `_quickhull_core` replaced with
    an assertion failure, on a handful of trivial one-face `face_groups`
    (cheap to build, never valid enough to actually hull) -- proving the
    refusal fires before any group reaches the hull step at all, rather than
    building thousands of real loose parts.
    """

    def _boom(points, eps):
        raise AssertionError("_quickhull_core ran past the part ceiling")

    monkeypatch.setattr(cl, "_quickhull_core", _boom)
    monkeypatch.setattr(cl, "MAX_COMPOUND_PARTS", 3)

    mesh = bp.box((1.0, 1.0, 1.0))
    n_faces = bm.face_count(mesh)
    face_groups = [[i % n_faces] for i in range(4)]  # 4 groups, past the lowered ceiling
    with pytest.raises(OpError, match="loose parts, past the"):
        cl.compound(mesh, face_groups=face_groups)


def test_compound_part_ceiling_has_not_crept_down_onto_ordinary_use():
    assert cl.MAX_COMPOUND_PARTS >= 500


def test_compound_on_two_separated_boxes_gives_two_hulls():
    a = bp.box((1.0, 1.0, 1.0))
    b = _translated(bp.box((1.0, 1.0, 1.0)), (5.0, 0.0, 0.0))
    merged = cl._concat_meshes([a, b])

    col = cl.compound(merged)
    assert col.kind == "compound"
    assert col.params["count"] == 2
    for part in col.params["parts"]:
        assert part.kind == "convex"
        assert part.params["vertex_count"] == 8
        assert part.params["face_count"] == 12
        _assert_valid_shell(part.mesh)
    _assert_valid_shell(col.mesh)  # two disjoint but individually clean shells


def test_compound_with_explicit_face_groups_matches_auto_detection():
    a = bp.box((1.0, 1.0, 1.0))
    b = _translated(bp.box((1.0, 1.0, 1.0)), (5.0, 0.0, 0.0))
    merged = cl._concat_meshes([a, b])

    auto = cl.compound(merged)
    n_faces_a = bm.face_count(a)
    n_faces_total = bm.face_count(merged)
    explicit = cl.compound(
        merged,
        face_groups=[np.arange(n_faces_a), np.arange(n_faces_a, n_faces_total)],
    )
    assert auto.params["count"] == explicit.params["count"] == 2


def test_compound_refuses_an_empty_face_group_list():
    mesh = bp.box((1.0, 1.0, 1.0))
    with pytest.raises(OpError):
        cl.compound(mesh, face_groups=[])


def test_compound_with_an_out_of_range_face_index_raises_op_error_not_index_error():
    """The 2026-09-20 audit, finding clay-14: an out-of-range face id in a
    hand-supplied ``face_groups`` used to reach ``_corners_of_faces``'s
    ``mesh.starts[f]`` indexing as a bare ``IndexError``, not this module's
    own named :class:`OpError` refusal every other malformed input raises.
    """
    mesh = bp.box((1.0, 1.0, 1.0))
    n_faces = bm.face_count(mesh)
    with pytest.raises(OpError):
        cl.compound(mesh, face_groups=[[0, n_faces]])


def test_compound_with_a_negative_face_index_raises_op_error_not_silently_wrapping():
    """Same finding, the other half: a negative face id did not raise at
    all -- ``_corners_of_faces`` indexed ``mesh.starts`` (one longer than the
    face count) directly with it, so ``-2`` silently returned the *last*
    face's corners instead of the second-to-last one a caller would expect,
    with no error anywhere to say the id was never checked.
    """
    mesh = bp.box((1.0, 1.0, 1.0))
    with pytest.raises(OpError):
        cl.compound(mesh, face_groups=[[0, -2]])


# --- registry -----------------------------------------------------------------


def test_collider_kinds_registry_round_trips_through_its_own_fit_function():
    mesh = bp.box((1.0, 1.0, 1.0))
    for kind, (label, fit, extra) in cl.COLLIDER_KINDS.items():
        assert isinstance(label, str) and label
        col = fit(mesh, **extra)
        assert col.kind == kind
