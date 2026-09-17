"""Facts :func:`~.analyze.analyze` reports, and the caps that keep it cheap.

Diagnose's own tests build a defect and check the finding; these build a
small scene and check a measurement. Nothing here asserts on a selection --
this module never makes one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from warlock.kernels.geom3d import math3d as m3
from warlock.kernels.mesh import analyze
from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import mesh as bm
from warlock.kernels.mesh import ops as clay_ops
from warlock.kernels.mesh import primitives as bp
from warlock.kernels.mesh.elements import OpError


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


def test_a_pair_past_the_triangle_pair_cap_never_reports_a_false_no_intersection() -> None:
    """The 2026-09-14 audit's clay-01: past MAX_TRIANGLE_PAIRS, analyze()
    used to hard-code intersects=False without ever running the SAT test,
    so two heavily-overlapping dense meshes read as "not touching" -- the
    same bits an honest "checked and clear" would produce. These two
    cylinders share a centre (one entirely inside the other, so they
    unambiguously intersect) and are dense enough that the grid-binned
    candidate count clears MAX_TRIANGLE_PAIRS at this test's `near`. Past
    the cap the SAT test itself is skipped, so the honest answer is
    "unknown" (``None``), never ``False``.
    """
    a = _obj(bp.cylinder(radius=0.5, height=1.0, segments=400))
    b = _obj(bp.cylinder(radius=0.5, height=1.0, segments=400))

    result = analyze.analyze([a, b], near=1000.0)
    assert result.truncated is True
    pair = result.pairs[0]
    assert pair.exact is False
    assert pair.intersects is None


def test_analyze_does_not_stall_on_one_large_flat_triangle() -> None:
    """The 2026-09-14 audit's clay-02: `_grid_candidates` registered a
    triangle into *every* grid cell its own AABB touched, in pure Python,
    with no ceiling -- an 80 m floor plate (two triangles) at the default
    0.05 m cell measured 2.35 s, and a 160 m one 10.6 s, with none of
    analyze()'s three other ceilings bounding it (14 triangles total is
    nowhere near MAX_ANALYZE_TRIANGLES). A single large quad -- exactly an
    authored "ground plane" -- paired against a small box must stay fast.
    """
    import time

    floor = _obj(bp.plane((160.0, 160.0)), name="floor")
    box = _obj(bp.box((0.5, 0.5, 0.5)), translation=(0.0, 0.25, 0.0), name="box")

    t0 = time.time()
    result = analyze.analyze([floor, box])
    dt = time.time() - t0

    # Comfortably above what the fix needs (milliseconds) and comfortably
    # below the 10.6 s the unfixed code measured on this exact shape.
    assert dt < 2.0
    assert len(result.pairs) == 1


def _grid_mesh(rows: int, cols: int, quad_size: float) -> bm.Mesh:
    """A flat floor tiled from `rows` x `cols` separate quad faces, each
    `quad_size` metres across -- an ordinary blockout floor built from many
    tiles, not one giant plate."""
    positions = [
        (c * quad_size, 0.0, r * quad_size)
        for r in range(rows + 1)
        for c in range(cols + 1)
    ]
    stride = cols + 1
    faces = []
    for r in range(rows):
        for c in range(cols):
            a = r * stride + c
            faces.append([a, a + 1, a + 1 + stride, a + stride])
    return bm.from_faces(positions, faces)


def test_analyze_does_not_stall_on_many_moderately_large_triangles() -> None:
    """The 2026-09-16 audit: `_MAX_CELLS_PER_TRIANGLE_AXIS` only bounds what
    *one* triangle can cost `_grid_candidates`, not the total across *many*
    triangles that are each individually under that cap. A floor tiled from
    20,000 separate 10 m quads -- each on its own saturating the per-triangle
    cap, the way an authored level's floor plates routinely do -- measured
    5.8 s in the unfixed pure-Python registration loops, at only 10% of
    MAX_ANALYZE_TRIANGLES, so none of analyze()'s other ceilings caught it
    either. A blockout floor built from many ordinary tiles, paired against a
    small box, must stay fast.
    """
    import time

    floor = _obj(_grid_mesh(rows=100, cols=100, quad_size=10.0), name="floor")
    box = _obj(bp.box((0.5, 0.5, 0.5)), translation=(5.0, 0.25, 5.0), name="box")

    t0 = time.time()
    result = analyze.analyze([floor, box])
    dt = time.time() - t0

    # Comfortably above what the fix needs and comfortably below the 5.8 s
    # the unfixed code measured on this exact shape.
    assert dt < 2.0
    assert len(result.pairs) == 1


def test_analyze_refuses_before_triangulating_past_max_analyze_triangles(monkeypatch) -> None:
    """The 2026-09-14 audit's clay-04: MAX_ANALYZE_TRIANGLES used to be
    checked only after _geometry() -- via cached_triangulation -- had
    already triangulated every object, so a refused call still paid the
    triangulation cost first. An n-cornered face's triangle count (n - 2) is
    knowable from mesh.loops and face_count alone, the same trick
    ops_boolean._refuse_complexity uses -- so cached_triangulation must never
    run for a call this refuses.
    """

    def _boom(mesh: object) -> None:
        raise AssertionError("cached_triangulation ran before the triangle-count refusal")

    monkeypatch.setattr(analyze, "cached_triangulation", _boom)

    # One n-gon face with enough corners that n - 2 alone clears the
    # ceiling -- no triangulation needed to know that, and this test never
    # lets one run.
    n = analyze.MAX_ANALYZE_TRIANGLES + 3
    positions = [(float(i), 0.0, 0.0) for i in range(n)]
    mesh = bm.from_faces(positions, [list(range(n))])
    obj = _obj(mesh)

    with pytest.raises(OpError):
        analyze.analyze([obj])
