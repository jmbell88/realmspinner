"""Facts :func:`~.analyze.analyze` reports, and the caps that keep it cheap.

Diagnose's own tests build a defect and check the finding; these build a
small scene and check a measurement. Nothing here asserts on a selection --
this module never makes one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from warlock.studio.clay import analyze
from warlock.studio.clay import document as bd
from warlock.studio.clay import ops as clay_ops
from warlock.studio.clay import primitives as bp
from warlock.studio.clay.elements import OpError
from warlock.studio.viewer import math3d as m3


def _obj(mesh, *, translation=(0.0, 0.0, 0.0), rotation=None, name: str = "obj") -> bd.Obj:
    return bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=mesh,
        translation=list(translation),
        rotation=m3.quat_identity() if rotation is None else rotation,
    )


# --- per-object facts ---------------------------------------------------------


def test_a_unit_box_gives_volume_1_and_area_6() -> None:
    obj = _obj(bp.box((1.0, 1.0, 1.0)))
    result = analyze.analyze([obj])
    row = result.objects[0]
    assert row.closed is True
    assert row.volume == pytest.approx(1.0, abs=1e-6)
    assert row.area == pytest.approx(6.0, abs=1e-6)


def test_an_open_sheet_gives_null_volume() -> None:
    obj = _obj(bp.plane((1.0, 1.0)))
    result = analyze.analyze([obj])
    row = result.objects[0]
    assert row.closed is False
    assert row.volume is None
    # A single quad still has a real, measurable area.
    assert row.area == pytest.approx(1.0, abs=1e-6)


def test_a_boxs_symmetry_is_1_1_1() -> None:
    obj = _obj(bp.box((1.0, 2.0, 3.0)))
    result = analyze.analyze([obj], symmetry_tol=0.002)
    assert result.objects[0].symmetry == pytest.approx((1.0, 1.0, 1.0), abs=1e-9)


def test_a_rotated_objects_exact_bounds_are_tighter_than_world_boxs() -> None:
    """``world_box`` transforms the *local bounding box's* eight corners,
    which for a shape that does not itself fill its own bounding box (a
    cylinder, unlike a primitive box whose vertices *are* its box corners)
    over-reports under rotation -- see ``ops.world_box``'s own docstring. A
    single-axis tilt of an axis-aligned cylinder does not show this (each
    ring realises its own extreme x and z independently of the other, at
    some vertex, so the two conservative corners are actually reachable); a
    compound rotation about a diagonal axis mixes all three local axes at
    once, which is what actually separates the two answers. Exact,
    per-vertex bounds do not pay the conservative cost either way.
    """
    axis = np.array([1.0, 1.0, 1.0])
    axis /= np.linalg.norm(axis)
    quat = m3.quat_from_axis_angle(axis, math.radians(37.0))
    obj = _obj(bp.cylinder(radius=0.3, height=1.0, segments=16), rotation=quat)

    result = analyze.analyze([obj])
    lo, hi = result.objects[0].bounds
    exact_size = hi - lo

    world_lo, world_hi = clay_ops.world_box(obj)
    world_size = world_hi - world_lo

    assert np.all(exact_size <= world_size + 1e-9)
    assert np.any(exact_size < world_size - 1e-6)


# --- ground and floating -------------------------------------------------------


def test_a_grounded_box_is_not_floating_and_an_unsupported_one_is() -> None:
    grounded = _obj(bp.box((1.0, 1.0, 1.0)), translation=(0.0, 0.5, 0.0), name="grounded")
    raised = _obj(bp.box((1.0, 1.0, 1.0)), translation=(10.0, 10.5, 0.0), name="raised")

    result = analyze.analyze([grounded, raised])

    grounded_row = next(r for r in result.objects if r.uid == grounded.uid)
    raised_row = next(r for r in result.objects if r.uid == raised.uid)
    assert grounded_row.ground.contact is True
    assert raised_row.ground.contact is False

    assert result.floating is not None
    assert raised.uid in result.floating
    assert grounded.uid not in result.floating


# --- pairs ---------------------------------------------------------------------


def test_two_touching_boxes_are_in_contact_with_distance_near_zero() -> None:
    a = _obj(bp.box((1.0, 1.0, 1.0)), translation=(0.0, 0.5, 0.0), name="a")
    b = _obj(bp.box((1.0, 1.0, 1.0)), translation=(1.0, 0.5, 0.0), name="b")

    result = analyze.analyze([a, b])
    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.contact is True
    assert pair.distance == pytest.approx(0.0, abs=1e-6)


def test_a_cylinder_sunk_into_a_slab_gives_overlap_depth_near_0_02() -> None:
    pytest.importorskip("manifold3d")
    pytest.importorskip("trimesh")

    slab = _obj(bp.box((2.0, 0.5, 2.0)), translation=(0.0, 0.25, 0.0), name="slab")
    # The slab's top face sits at y = 0.5; the cylinder's own bottom face
    # sits 0.02 below it, so the two overlap in a 0.02-thick slice.
    cylinder = _obj(
        bp.cylinder(radius=0.3, height=1.0, segments=24),
        translation=(0.0, 0.98, 0.0),
        name="cylinder",
    )

    result = analyze.analyze([slab, cylinder])
    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.intersects is True
    assert pair.overlap is not None
    assert pair.overlap.depth == pytest.approx(0.02, abs=2e-3)


# --- caps ------------------------------------------------------------------


def test_the_object_cap_refuses() -> None:
    objs = [
        _obj(bp.box(), translation=(float(i) * 3.0, 0.5, 0.0))
        for i in range(analyze.MAX_ANALYZE_OBJECTS + 1)
    ]
    with pytest.raises(OpError):
        analyze.analyze(objs)


def test_the_triangle_pair_cap_truncates_with_exact_false() -> None:
    """Two dense, fully-overlapping cylinders, with *near* set large enough
    that the whole of each falls in one grid cell -- so the candidate count
    is close to the full product of their triangle counts, comfortably past
    :data:`analyze.MAX_TRIANGLE_PAIRS`."""
    a = _obj(bp.cylinder(radius=0.5, height=1.0, segments=400))
    b = _obj(bp.cylinder(radius=0.5, height=1.0, segments=400))

    result = analyze.analyze([a, b], near=1000.0)
    assert result.truncated is True
    assert len(result.pairs) == 1
    assert result.pairs[0].exact is False
