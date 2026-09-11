"""The object-level operations: mirror, snap, bake, duplicate.

The load-bearing one is mirror. A mirrored object could be expressed as a
negative node scale in a single line, and the reason it is not -- glTF readers
disagree about whether a negative scale flips winding -- is invisible until an
asset reaches an engine that decided the other way. So the test here does not
check that a mesh was mirrored; it checks that the mirrored mesh is still
consistently and *outwardly* wound, which is the property the negative-scale
version cannot promise.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from warlock.studio.clay import document as bd
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import ops
from warlock.studio.clay import primitives as bp
from warlock.studio.viewer import math3d as m3


def _obj(name: str = "A", mesh: bm.Mesh | None = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=bp.box() if mesh is None else mesh,
        **kwargs,  # type: ignore[arg-type]
    )


def _face_normal(mesh: bm.Mesh, i: int) -> np.ndarray:
    """Newell normal for one face, by the textbook formula rather than by
    importing the one the renderer uses -- see ``test_primitives`` for why."""
    p = mesh.positions[bm.face(mesh, i)].astype("f8")
    q = np.roll(p, -1, axis=0)
    return np.stack(
        [
            ((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])).sum(),
            ((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])).sum(),
            ((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])).sum(),
        ]
    )


def _volume_sum(mesh: bm.Mesh) -> float:
    """Six times the enclosed volume; positive iff the shell faces outward."""
    lo, hi = bm.bounds(mesh)
    centre = (lo + hi) * 0.5
    return sum(
        float(
            (mesh.positions[bm.face(mesh, i)].astype("f8").mean(axis=0) - centre)
            @ _face_normal(mesh, i)
        )
        for i in range(bm.face_count(mesh))
    )


def _max_directed_edge_use(mesh: bm.Mesh) -> int:
    """1 on a consistently oriented shell; 2 where two faces traverse an edge
    the same way, which is what a single flipped face produces."""
    counts: Counter[tuple[int, int]] = Counter()
    for i in range(bm.face_count(mesh)):
        loop = [int(v) for v in bm.face(mesh, i)]
        for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
            counts[(a, b)] += 1
    return max(counts.values()) if counts else 0


def _world_positions(obj: bd.Obj) -> np.ndarray:
    matrix = m3.compose(obj.translation, obj.rotation, obj.scale)
    homo = np.hstack([obj.mesh.positions.astype("f8"), np.ones((len(obj.mesh.positions), 1))])
    return (matrix @ homo.T).T[:, :3]


CLOSED = ["box", "cylinder", "cone", "uv_sphere", "torus"]


# --- mirror ------------------------------------------------------------------


@pytest.mark.parametrize("name", CLOSED)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_mirrored_object_is_still_wound_outward(name: str, axis: int) -> None:
    defaults, build = bp.GENERATORS[name]
    obj = _obj(name, build(**defaults))
    out = ops.mirror(obj, axis)

    assert _max_directed_edge_use(out.mesh) == 1
    assert _volume_sum(out.mesh) > 0.0


def test_mirror_bakes_into_the_mesh_rather_than_negating_the_scale() -> None:
    obj = _obj(translation=(1.0, 2.0, 3.0), scale=(2.0, 2.0, 2.0))
    out = ops.mirror(obj, 0)

    assert np.allclose(out.scale, [2.0, 2.0, 2.0])
    assert (out.scale > 0.0).all()
    assert np.allclose(out.translation, obj.translation)
    assert out.mesh is not obj.mesh
    assert np.allclose(out.mesh.positions[:, 0], -obj.mesh.positions[:, 0])
    # The other two axes are untouched, so this really is a reflection and not
    # a rotation that happens to land on one.
    assert np.allclose(out.mesh.positions[:, 1:], obj.mesh.positions[:, 1:])


def test_mirroring_twice_returns_the_original_geometry() -> None:
    obj = _obj("A", bp.cone())
    twice = ops.mirror(ops.mirror(obj, 2), 2)
    assert np.allclose(twice.mesh.positions, obj.mesh.positions)
    assert np.array_equal(twice.mesh.loops, obj.mesh.loops)


def test_mirror_keeps_the_uid_because_it_edits_one_object() -> None:
    obj = _obj()
    assert ops.mirror(obj, 1).uid == obj.uid


def test_mirror_rejects_an_axis_that_is_not_one_of_three() -> None:
    with pytest.raises(ValueError):
        ops.mirror(_obj(), 3)


# --- array placement (translated, rotated_about_origin, mirror_world) -------
#
# The per-copy steps behind ``clay_ops.array-linear``, ``array-radial`` and
# ``mirror-copy``. ``translated`` and ``rotated_about_origin`` are checked
# against the same "does this match a direct transform of the world-space
# vertices" oracle the mirror tests above use, because a sign error in either
# is exactly as invisible in the viewport as a sign error in ``mirror`` is.


def test_translated_moves_the_translation_and_shares_the_mesh() -> None:
    obj = _obj(
        "Box",
        translation=(1.0, 2.0, 3.0),
        rotation=m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), math.radians(20.0)),
        scale=(2.0, 1.0, 0.5),
    )
    out = ops.translated(obj, (0.5, -1.0, 2.0))

    assert np.allclose(out.translation, [1.5, 1.0, 5.0])
    assert np.allclose(out.rotation, obj.rotation)
    assert np.allclose(out.scale, obj.scale)
    assert out.mesh is obj.mesh, "a translation never touches the mesh"


def test_rotated_about_origin_matches_rotating_the_objects_world_space_vertices() -> None:
    """The whole point of turning about the *world* origin: picking the object
    up and spinning it about a point through the scene's centre must move
    every point on it exactly as a straight rotation of its world-space
    vertices would -- both where it sits and which way it faces, together,
    with no decomposition anywhere.
    """
    obj = _obj(
        "A",
        bp.uv_sphere(segments=8, rings=4),
        translation=(3.0, 1.0, -2.0),
        rotation=m3.quat_from_axis_angle(
            m3.vec3(1.0, 0.0, 1.0) / math.sqrt(2.0), math.radians(40.0)
        ),
        scale=(1.5, 0.5, 2.0),
    )
    before = _world_positions(obj)
    axis, degrees = 2, 65.0

    out = ops.rotated_about_origin(obj, axis, degrees)

    spin = m3.quat_from_axis_angle(m3.vec3(0.0, 0.0, 1.0), math.radians(degrees))
    expected = (m3.quat_to_mat4(spin)[:3, :3] @ before.T).T
    assert np.allclose(_world_positions(out), expected, atol=1e-6)
    assert out.mesh is obj.mesh, "a rotation about the origin never touches the mesh"
    assert np.allclose(out.scale, obj.scale)


def test_rotated_about_origin_at_zero_degrees_is_the_identity() -> None:
    obj = _obj("A", translation=(4.0, 0.0, 0.0))
    out = ops.rotated_about_origin(obj, 1, 0.0)
    assert np.allclose(out.translation, obj.translation)
    assert np.allclose(out.rotation, obj.rotation)


def test_rotated_about_origin_rejects_an_axis_that_is_not_one_of_three() -> None:
    with pytest.raises(ValueError):
        ops.rotated_about_origin(_obj(), 3, 90.0)


def test_mirror_world_matches_reflecting_the_objects_world_space_vertices_directly() -> None:
    """The derivation in ``mirror_world``'s own docstring, checked rather than
    trusted -- built from a rotated, non-uniformly-scaled object precisely
    because that is the fixture on which a wrong sign would still look right
    on anything unscaled or axis-aligned. Every part of this is invisible in
    the viewport when it is subtly wrong, which is what makes this comparison
    the regression this op most needs.
    """
    obj = _obj(
        "A",
        bp.box(),
        translation=(2.0, -1.0, 4.0),
        rotation=m3.quat_from_axis_angle(
            m3.vec3(1.0, 2.0, 3.0) / math.sqrt(14.0), math.radians(50.0)
        ),
        scale=(2.0, 0.5, 3.0),
    )
    before = _world_positions(obj)
    axis, offset = 1, 2.5

    out = ops.mirror_world(obj, axis, offset)

    expected = before.copy()
    expected[:, axis] = 2.0 * offset - expected[:, axis]
    assert np.allclose(_world_positions(out), expected, atol=1e-6)
    assert np.allclose(out.scale, obj.scale)
    assert out.mesh is not obj.mesh


@pytest.mark.parametrize("name", CLOSED)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_world_mirrored_object_is_still_wound_outward(name: str, axis: int) -> None:
    """``mirror_world`` routes through :func:`mirror` for its mesh, which is
    what obeys the module docstring's negative-scale rule; this is that
    routing checked the same way the plain mirror is."""
    defaults, build = bp.GENERATORS[name]
    obj = _obj(name, build(**defaults), translation=(1.0, 2.0, 3.0))
    out = ops.mirror_world(obj, axis, 0.5)

    assert _max_directed_edge_use(out.mesh) == 1
    assert _volume_sum(out.mesh) > 0.0


def test_mirror_world_rejects_an_axis_that_is_not_one_of_three() -> None:
    with pytest.raises(ValueError):
        ops.mirror_world(_obj(), 3, 0.0)


# --- align_y (promoted out of presets._align_y) and place_between -----------
#
# ``align_y`` used to live in ``presets.py`` as a private, and the eight-
# assembly digest comparison that proved the promotion faithful lives in the
# landing report rather than here (it needs the rig templates ``presets.py``
# itself is checked against). What belongs here is the ingredient's own
# claims and ``place_between``'s, which has no other caller to exercise it.


def test_align_y_is_the_identity_for_a_parallel_direction() -> None:
    assert ops.align_y((0.0, 5.0, 0.0)) == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_align_y_is_a_half_turn_about_x_for_an_antiparallel_direction() -> None:
    assert ops.align_y((0.0, -3.0, 0.0)) == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_align_y_rotates_plus_y_onto_the_given_direction() -> None:
    """The property that matters, checked directly rather than trusted from the
    two degenerate cases above: whatever direction comes in, rotating the
    canonical ``+Y`` by the returned quaternion lands on its unit vector."""
    direction = np.array([1.0, 2.0, -3.0])
    q = ops.align_y(direction)
    rotated = m3.quat_rotate(np.asarray(q), m3.vec3(0.0, 1.0, 0.0))
    assert np.allclose(rotated, direction / np.linalg.norm(direction), atol=1e-9)


def test_align_y_returns_the_identity_for_a_zero_length_direction() -> None:
    """No direction to align to; the identity is the only answer that invents
    nothing, and the one :func:`place_between` relies on for coincident
    anchors."""
    assert ops.align_y((0.0, 0.0, 0.0)) == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_place_between_sits_at_the_midpoint_of_the_two_anchors() -> None:
    obj = _obj("Strut", translation=(9.0, 9.0, 9.0))
    out = ops.place_between(obj, (0.0, 0.0, 0.0), (2.0, 4.0, 6.0), fit=False)
    assert np.allclose(out.translation, [1.0, 2.0, 3.0])


def test_place_betweens_local_y_axis_ends_up_along_the_line() -> None:
    """The claim, not the quaternion's own components: rotate the object's
    local +Y by the result and it must point where the line points, whatever
    axis convention produced that rotation internally."""
    obj = _obj("Strut")
    a, b = np.array([1.0, -2.0, 0.5]), np.array([4.0, 3.0, -1.5])
    out = ops.place_between(obj, a, b, fit=False)
    rotated_y = m3.quat_rotate(np.asarray(out.rotation), m3.vec3(0.0, 1.0, 0.0))
    expected = (b - a) / np.linalg.norm(b - a)
    assert np.allclose(rotated_y, expected, atol=1e-9)


def test_place_between_replaces_rotation_rather_than_composing_with_it() -> None:
    """"Aim this along that line" must not depend on which way the object
    already happened to be facing -- two objects starting at different
    rotations placed on the same segment must end up identically oriented."""
    a, b = (0.0, 0.0, 0.0), (0.0, 0.0, 5.0)
    facing_one_way = _obj(rotation=m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), 0.7))
    facing_another = _obj(rotation=m3.quat_from_axis_angle(m3.vec3(1.0, 0.0, 0.0), 2.1))
    out_a = ops.place_between(facing_one_way, a, b, fit=False)
    out_b = ops.place_between(facing_another, a, b, fit=False)
    assert np.allclose(out_a.rotation, out_b.rotation)


def test_place_between_with_fit_stretches_local_y_to_span_the_gap() -> None:
    """``box()``'s local Y extent is exactly 1, so fitting a gap of length
    *L* must set ``scale[1]`` to exactly *L*."""
    obj = _obj("Strut", bp.box(), scale=(3.0, 3.0, 3.0))
    out = ops.place_between(obj, (0.0, 0.0, 0.0), (0.0, 7.0, 0.0), fit=True)
    assert out.scale[1] == pytest.approx(7.0)
    # The other two axes are the caller's business, not the fit's.
    assert out.scale[0] == pytest.approx(3.0)
    assert out.scale[2] == pytest.approx(3.0)


def test_place_between_without_fit_leaves_the_scale_alone() -> None:
    obj = _obj("Strut", scale=(2.0, 5.0, 2.0))
    out = ops.place_between(obj, (0.0, 0.0, 0.0), (0.0, 9.0, 0.0), fit=False)
    assert np.allclose(out.scale, [2.0, 5.0, 2.0])


def test_place_between_on_coincident_anchors_does_not_produce_a_nan() -> None:
    """The zero-length-segment case ``align_y`` already names as its own
    identity branch, exercised through the op that actually divides by the
    segment's length when ``fit`` is asked for."""
    obj = _obj("Strut")
    out = ops.place_between(obj, (2.0, 2.0, 2.0), (2.0, 2.0, 2.0), fit=True)
    assert np.allclose(out.translation, [2.0, 2.0, 2.0])
    assert not np.isnan(out.rotation).any()
    assert not np.isnan(out.scale).any()
    assert np.allclose(out.rotation, [0.0, 0.0, 0.0, 1.0])


def test_place_between_skips_the_fit_on_a_mesh_with_no_y_extent() -> None:
    """A ``plane`` is authored flat in XZ, so its local Y span is exactly
    zero and dividing the gap length by it is undefined. The decision: skip
    the fit rather than refuse the whole placement, because the translation
    and rotation this call is for are still meaningful even when the shape
    has no length along Y to stretch -- see ``place_between``'s own
    docstring. The object still moves and turns; only the scale is left as
    it was."""
    obj = _obj("Sheet", bp.plane(), scale=(1.0, 1.0, 1.0))
    out = ops.place_between(obj, (0.0, 0.0, 0.0), (0.0, 5.0, 0.0), fit=True)
    assert np.allclose(out.scale, [1.0, 1.0, 1.0])
    assert np.allclose(out.translation, [0.0, 2.5, 0.0])


# --- snapping ----------------------------------------------------------------


def test_snapping_at_step_zero_is_the_identity() -> None:
    """Snapping *off* must not quantise everything to zero."""
    assert ops.snap_value(0.3456, 0.0) == pytest.approx(0.3456)
    assert np.allclose(ops.snap_translation((0.3, -1.7, 9.9), 0.0), [0.3, -1.7, 9.9])
    q = m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), math.radians(37.0))
    assert np.allclose(ops.snap_rotation(q, 0.0), q)


def test_snapping_is_symmetric_around_zero() -> None:
    assert ops.snap_value(0.34, 0.25) == pytest.approx(0.25)
    assert ops.snap_value(-0.34, 0.25) == pytest.approx(-0.25)


def test_a_value_exactly_between_two_steps_rounds_away_from_zero() -> None:
    """Half-away-from-zero, not Python's half-to-even: the grid must look the
    same on both sides of the origin, and ``round`` breaks that tie by parity."""
    assert ops.snap_value(0.125, 0.25) == pytest.approx(0.25)
    assert ops.snap_value(-0.125, 0.25) == pytest.approx(-0.25)


def test_snap_translation_snaps_each_component() -> None:
    out = ops.snap_translation((0.34, -0.9, 1.51), 0.5)
    assert np.allclose(out, [0.5, -1.0, 1.5])


def test_snap_rotation_quantises_the_angle_and_keeps_the_axis() -> None:
    axis = m3.vec3(0.0, 0.0, 1.0)
    q = ops.snap_rotation(m3.quat_from_axis_angle(axis, math.radians(37.0)), 15.0)
    expected = m3.quat_from_axis_angle(axis, math.radians(30.0))
    assert np.allclose(q, expected)


def test_snap_rotation_leaves_an_identity_rotation_alone() -> None:
    """The axis of a zero rotation is undefined, so there is nothing to keep."""
    assert np.allclose(ops.snap_rotation(m3.quat_identity(), 15.0), [0.0, 0.0, 0.0, 1.0])


def test_a_snapped_rotation_is_still_a_unit_quaternion() -> None:
    q = m3.quat_from_axis_angle(m3.vec3(1.0, 2.0, 3.0), math.radians(100.0))
    assert np.linalg.norm(ops.snap_rotation(q, 15.0)) == pytest.approx(1.0)


# --- baking ------------------------------------------------------------------


def test_bake_transform_folds_the_trs_in_and_resets_it() -> None:
    obj = _obj(
        translation=(1.0, 2.0, 3.0),
        rotation=m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), math.radians(30.0)),
        scale=(2.0, 1.0, 0.5),
    )
    before = _world_positions(obj)
    out = ops.bake_transform(obj)

    assert np.allclose(out.translation, [0.0, 0.0, 0.0])
    assert np.allclose(out.rotation, [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(out.scale, [1.0, 1.0, 1.0])
    assert np.allclose(_world_positions(out), before, atol=1e-5)


def test_baking_leaves_the_world_bounds_where_they_were() -> None:
    obj = _obj(
        translation=(0.0, 5.0, 0.0),
        rotation=m3.quat_from_axis_angle(m3.vec3(1.0, 0.0, 0.0), math.radians(45.0)),
        scale=(3.0, 3.0, 3.0),
    )
    before = _world_positions(obj)
    lo, hi = bm.bounds(ops.bake_transform(obj).mesh)
    assert np.allclose(lo, before.min(axis=0), atol=1e-5)
    assert np.allclose(hi, before.max(axis=0), atol=1e-5)


def test_baking_a_negative_scale_keeps_the_winding_outward() -> None:
    """A negative scale is exactly what mirror refuses to leave on the node, so
    the one function that can be handed one has to deal with it too."""
    obj = _obj("A", bp.uv_sphere(), scale=(-1.0, 1.0, 1.0))
    out = ops.bake_transform(obj)
    assert _max_directed_edge_use(out.mesh) == 1
    assert _volume_sum(out.mesh) > 0.0


def test_baking_an_untransformed_object_changes_nothing() -> None:
    obj = _obj()
    out = ops.bake_transform(obj)
    assert np.allclose(out.mesh.positions, obj.mesh.positions)


# --- world <-> local direction and position (Clay25, agent bounds/normal queries) --
#
# An agent reads "upward-facing" or "inside this box" out of a *world*-space
# scene report, and a mesh's own positions and normals are local -- so turning
# one into the other correctly is what makes ``select.faces_by_normal`` and
# ``select.faces_in_bounds`` answer the question that was actually asked
# rather than one that happens to agree with it on an unscaled primitive.


def test_local_direction_accounts_for_a_non_uniform_scale_not_just_the_rotation() -> None:
    """The derivation in ``local_direction``'s own docstring, made concrete.

    A naive inverse -- ``quat_rotate(quat_conjugate(obj.rotation), world_dir)``,
    the one line that looks like it must be the whole answer -- is exactly the
    inverse of how a *position* transforms, not of how a *normal* does. The two
    agree whenever ``obj.scale`` is uniform, which is why the mistake is
    invisible on every un-stretched primitive and needs a scaled, rotated
    fixture to show up at all: rotated 65 degrees about (1, 1, 1) with a
    ``[1, 4, 1]`` scale, this box's *local* +Y face is genuinely the one
    facing world "up" -- :func:`~warlock.studio.clay.select.faces_by_normal`
    finds it, at a generous 20-degree tolerance, from what ``local_direction``
    hands back. The naive, scale-blind inverse points somewhere else on this
    mesh entirely and finds no face at all at the same tolerance -- this was
    verified by writing exactly that naive line in place of the real one and
    watching the assertion below fail before ``local_direction`` multiplied by
    ``obj.scale`` at all.
    """
    from warlock.studio.clay import select

    axis = m3.vec3(1.0, 1.0, 1.0) / math.sqrt(3.0)
    obj = _obj(rotation=m3.quat_from_axis_angle(axis, math.radians(65.0)), scale=(1.0, 4.0, 1.0))
    world_up = (0.0, 1.0, 0.0)

    # The naive version: rotation only, no scale. Kept inline, not as a second
    # implementation in ``ops.py``, because its only job is to prove the real
    # one is not doing the same thing.
    naive = m3.quat_rotate(m3.quat_conjugate(obj.rotation), np.asarray(world_up, dtype="f8"))
    naive = naive / np.linalg.norm(naive)
    assert select.faces_by_normal(obj.mesh, naive, max_angle=20.0).tolist() == [], (
        "the naive, rotation-only inverse must not be the one that finds the top "
        "face here -- if it is, this fixture no longer demonstrates the mistake"
    )

    correct = ops.local_direction(obj, world_up)
    assert np.linalg.norm(correct) == pytest.approx(1.0)
    assert select.faces_by_normal(obj.mesh, correct, max_angle=20.0).tolist() == [1], (
        "face index 1 is +Y in primitives.box()'s own face table -- the top"
    )


def test_world_positions_moves_the_points_rather_than_the_box() -> None:
    """``world_box`` is a *conservative* upper bound under rotation, by its own
    docstring's admission: it moves the local mesh's eight box corners, not
    the geometry, so a sphere -- whose vertices never reach those corners at
    all -- reports a box wider than its true rotated footprint.
    ``faces_in_bounds`` needs the true footprint: a selection built from the
    looser box would select faces standing outside the box an agent actually
    asked for. So this moves every vertex instead, matching a direct
    ``compose`` oracle exactly, and reporting a strictly tighter bound than
    ``world_box`` gives the same object on the two axes its rotation actually
    moves.
    """
    obj = _obj(
        "A",
        bp.uv_sphere(segments=12, rings=6),
        translation=(1.0, 2.0, 3.0),
        rotation=m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), math.radians(37.0)),
        scale=(2.0, 1.0, 0.5),
    )
    got = ops.world_positions(obj)

    assert got.shape == (len(obj.mesh.positions), 3), "one row per vertex, not eight"
    assert np.allclose(got, _world_positions(obj), atol=1e-6)

    # Rotation is about Y, so X and Z are the axes it actually narrows; Y is
    # untouched by a Y-axis rotation and the two bounds agree on it exactly.
    box_lo, box_hi = ops.world_box(obj)
    for axis in (0, 2):
        assert got.min(axis=0)[axis] > box_lo[axis]
        assert got.max(axis=0)[axis] < box_hi[axis]


# --- duplicate ---------------------------------------------------------------


def test_duplicate_gets_a_new_uid_the_same_geometry_and_a_new_name() -> None:
    obj = _obj("Box", translation=(1.0, 0.0, 0.0))
    copy = ops.duplicate(obj, bd.new_uid())

    assert copy.uid != obj.uid
    assert copy.name != obj.name
    assert np.allclose(copy.mesh.positions, obj.mesh.positions)
    assert np.allclose(copy.translation, obj.translation)


def test_a_duplicate_shares_the_mesh_because_a_mesh_is_immutable() -> None:
    obj = _obj("Box")
    assert ops.duplicate(obj, bd.new_uid()).mesh is obj.mesh


def test_duplicate_avoids_the_names_it_is_told_are_taken() -> None:
    obj = _obj("Box")
    copy = ops.duplicate(obj, bd.new_uid(), taken=["Box", "Box.001", "Box.002"])
    assert copy.name == "Box.003"


def test_duplicating_a_duplicate_counts_up_rather_than_nesting() -> None:
    obj = _obj("Box.007")
    assert ops.duplicate(obj, bd.new_uid()).name == "Box.008"


def test_a_duplicates_transform_arrays_are_its_own() -> None:
    obj = _obj("Box", translation=(1.0, 0.0, 0.0))
    copy = ops.duplicate(obj, bd.new_uid())
    copy.translation[0] = 9.0
    assert obj.translation[0] == 1.0


def test_a_duplicates_params_are_its_own() -> None:
    """Sharing the dict would make editing the copy's radius edit the
    original's, which is the one thing a duplicate must never do."""
    obj = _obj("Box", generator="cylinder", params={"radius": 0.5})
    copy = ops.duplicate(obj, bd.new_uid())
    copy.params["radius"] = 1.0
    assert obj.params == {"radius": 0.5}
    assert copy.generator == "cylinder"


# --- join -------------------------------------------------------------------
#
# The property that matters is not "the arrays got longer" -- it is that the
# result is a *valid* CSR mesh and that every piece is where it was drawn.
# Concatenating CSR arrays is exactly the operation whose off-by-one lands in
# ``starts``, where it does not raise: ``validate`` is what catches it.


def test_join_concatenates_into_one_valid_mesh() -> None:
    a, b = _obj("A"), _obj("B", translation=(5.0, 0.0, 0.0))
    merged = ops.join([a, b], eps=0.0)
    bm.validate(merged)
    assert bm.face_count(merged) == bm.face_count(a.mesh) + bm.face_count(b.mesh)
    assert len(merged.positions) == len(a.mesh.positions) + len(b.mesh.positions)
    assert len(merged.loops) == len(a.mesh.loops) + len(b.mesh.loops)


def test_join_lands_every_piece_where_it_was_drawn() -> None:
    """The merged mesh is in the *target's* frame, so the second object's
    geometry has to arrive carrying its own transform relative to the first."""
    a = _obj("A", translation=(1.0, 2.0, 3.0))
    b = _obj("B", translation=(6.0, 2.0, 3.0))
    merged = ops.join([a, b], eps=0.0)
    lo, hi = bm.bounds(merged)
    # a is a unit box at the origin of its own frame; b sits five along +X.
    assert lo == pytest.approx([-0.5, -0.5, -0.5])
    assert hi == pytest.approx([5.5, 0.5, 0.5])


def test_join_leaves_the_targets_own_vertices_untouched() -> None:
    """Not "close to" -- exactly. The target keeps its transform, so nothing
    about it moves, and ``inv(M) @ M`` is only identity to a rounding error."""
    a = _obj("A", translation=(1.0, 2.0, 3.0), scale=(0.3, 7.0, 0.1))
    b = _obj("B", translation=(6.0, 0.0, 0.0))
    merged = ops.join([a, b], eps=0.0)
    n = len(a.mesh.positions)
    assert np.array_equal(merged.positions[:n], a.mesh.positions)


def test_join_welds_coincident_vertices_into_one_surface() -> None:
    """Two boxes in the same place are one box afterwards, which is what a
    user means by merging rather than by grouping."""
    a, b = _obj("A"), _obj("B")
    assert len(ops.join([a, b], eps=0.0).positions) == 2 * len(a.mesh.positions)
    welded = ops.join([a, b], eps=1e-4)
    bm.validate(welded)
    assert len(welded.positions) == len(a.mesh.positions)


def _bare(mesh: bm.Mesh) -> bm.Mesh:
    """The same mesh with no texture coordinates.

    Every generator produces UVs now, so a mesh without them is an *imported*
    one -- which is exactly the case the absent-uv branches exist for, and why
    the fixtures below strip them explicitly rather than relying on a primitive
    to have none.
    """
    from dataclasses import replace

    return replace(mesh, uv=None)


def test_join_keeps_uvs_when_only_one_side_has_them() -> None:
    """Dropping them would lose coordinates one half already had; the side
    without gets zeros, which is what "this mesh has no UVs" already means."""
    a = _obj("A", mesh=_bare(bp.box()))
    textured = bm.Mesh(
        positions=a.mesh.positions,
        loops=a.mesh.loops,
        starts=a.mesh.starts,
        material=a.mesh.material,
        smooth=a.mesh.smooth,
        uv=np.zeros((len(a.mesh.loops), 2), dtype="f4") + 0.25,
    )
    merged = ops.join([_obj("A", mesh=textured), a], eps=0.0)
    bm.validate(merged)
    assert merged.uv is not None
    assert merged.uv.shape == (len(merged.loops), 2)
    assert merged.uv[: len(textured.loops)] == pytest.approx(0.25)
    assert merged.uv[len(textured.loops) :] == pytest.approx(0.0)


def test_join_keeps_no_uvs_when_neither_side_has_them() -> None:
    a = _obj("A", mesh=_bare(bp.box()))
    b = _obj("B", mesh=_bare(bp.box()))
    assert ops.join([a, b], eps=0.0).uv is None


def test_join_keeps_the_uvs_both_sides_brought() -> None:
    """The ordinary case now that generators produce them: two primitives
    merged keep one corner's coordinates each."""
    merged = ops.join([_obj("A"), _obj("B")], eps=0.0)
    assert merged.uv is not None
    assert merged.uv.shape == (len(merged.loops), 2)


def test_join_carries_per_face_materials_through_unchanged() -> None:
    """The palette is the document's, so an index means the same thing in
    every object in it and needs no remapping."""
    a = _obj("A")
    other = bm.Mesh(
        positions=a.mesh.positions,
        loops=a.mesh.loops,
        starts=a.mesh.starts,
        material=np.full(bm.face_count(a.mesh), 3, dtype="i4"),
        smooth=a.mesh.smooth,
    )
    merged = ops.join([a, _obj("B", mesh=other)], eps=0.0)
    assert set(merged.material[: bm.face_count(a.mesh)]) == {0}
    assert set(merged.material[bm.face_count(a.mesh) :]) == {3}


def test_join_refuses_fewer_than_two_objects() -> None:
    from warlock.studio.clay.elements import OpError

    with pytest.raises(OpError):
        ops.join([_obj("A")])


def test_join_refuses_a_target_with_a_zero_scale() -> None:
    """np.linalg.inv would raise LinAlgError out of the frame loop; a refusal
    is a toast."""
    from warlock.studio.clay.elements import OpError

    with pytest.raises(OpError):
        ops.join([_obj("A", scale=(0.0, 1.0, 1.0)), _obj("B")], eps=0.0)


# --- the weld distance's units ----------------------------------------------
#
# The dialog says "weld distance (m)", and the weld runs in the *target's local
# frame* -- so handing the number straight through made a target at scale 2 weld
# at twice what was asked and one at 0.01 at a hundredth, with the field still
# saying metres. The property below is the one that makes the label true: the
# same world-space gap decides the same way whatever the target is scaled to.


def _faces_apart(gap: float, scale: tuple[float, float, float]) -> bm.Mesh:
    """Two equally-sized boxes whose facing sides are ``gap`` metres apart in
    *world*, both carrying ``scale`` so their corners genuinely coincide."""
    a = _obj("A", scale=scale)
    b = _obj("B", translation=(scale[0] + gap, 0.0, 0.0), scale=scale)
    return ops.join([a, b], eps=0.002)


def test_the_weld_distance_is_world_metres_whatever_the_target_is_scaled_to():
    verts = len(bp.box().positions)
    for s in (0.05, 1.0, 20.0):
        scale = (s, s, s)
        # A gap well inside the 2 mm asked for: the touching faces fuse, so
        # four of the eight vertices on each side become shared.
        assert len(_faces_apart(0.0005, scale).positions) == 2 * verts - 4, s
        # And well outside it: nothing fuses. Both answers at every scale, which
        # is the whole claim -- handed straight through, the small scale welds
        # nothing and the large one welds everything.
        assert len(_faces_apart(0.05, scale).positions) == 2 * verts, s


def test_a_non_uniform_scale_never_welds_further_than_it_was_asked_to():
    """``max`` of the scale components, not a mean: the bound has to hold on the
    axis that stretches local space the most, or that axis welds wider than the
    number on the dialog. 5 mm apart, asked for 2 -- ``max`` leaves it alone,
    ``mean`` (3.4 here) would fuse it, and no conversion at all fuses it twice
    over."""
    merged = _faces_apart(0.005, (10.0, 0.1, 0.1))
    assert len(merged.positions) == 2 * len(bp.box().positions)


def test_the_weld_still_closes_a_seam_under_an_unscaled_target():
    """The behaviour the units fix must not cost: two coincident boxes are one
    box afterwards, which is the whole point of a non-zero weld."""
    merged = ops.join([_obj("A"), _obj("B")], eps=1e-4)
    bm.validate(merged)
    assert len(merged.positions) == len(bp.box().positions)
