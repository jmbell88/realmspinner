"""The deformation numbers, on shapes whose right answer is known by hand.

No Blender: ``deform_metrics`` takes arrays, so a cylinder twisted by linear
blending (which is exactly what a bad twist weighting does to a mesh) stands in
for the rigged mesh the worker would hand it.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.rig import deform_metrics as dm


def _cylinder(rings: int = 21, around: int = 16, *, capped: bool = True):
    """A unit-radius cylinder of height 2 about +Y, ``rings`` rings of ``around``."""
    verts = []
    for r in range(rings):
        y = -1.0 + 2.0 * r / (rings - 1)
        for a in range(around):
            t = 2.0 * np.pi * a / around
            verts.append((np.cos(t), y, np.sin(t)))
    tris = []
    for r in range(rings - 1):
        for a in range(around):
            b = (a + 1) % around
            i, j = r * around + a, r * around + b
            k, m = (r + 1) * around + a, (r + 1) * around + b
            tris += [(i, j, m), (i, m, k)]
    if capped:
        for end, flip in ((0, True), (rings - 1, False)):
            verts.append((0.0, -1.0 if end == 0 else 1.0, 0.0))
            centre = len(verts) - 1
            for a in range(around):
                b = (a + 1) % around
                p, q = end * around + a, end * around + b
                tris.append((centre, q, p) if flip else (centre, p, q))
    return np.array(verts), np.array(tris)


def _joint(name="elbow", y=0.0, posed_axis=(0.0, 1.0, 0.0)):
    return {
        "name": name,
        "rest_head": (0.0, y, 0.0),
        "rest_axis": (0.0, 1.0, 0.0),
        "posed_head": (0.0, y, 0.0),
        "posed_axis": posed_axis,
    }


def _twist_by_blending(rest: np.ndarray, degrees: float) -> np.ndarray:
    """Linear-blend two opposed rotations along the height: the candy wrapper."""
    posed = rest.copy()
    t = (rest[:, 1] + 1.0) / 2.0
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    twisted = np.stack(
        [c * rest[:, 0] + s * rest[:, 2], rest[:, 1], -s * rest[:, 0] + c * rest[:, 2]], 1
    )
    posed[:] = (1.0 - t)[:, None] * rest + t[:, None] * twisted
    return posed


def test_the_rest_pose_scores_as_nothing_happened():
    rest, tris = _cylinder()
    out = dm.score_pose(
        rest, rest, tris, np.zeros(len(rest), int), [_joint()], dm.is_closed(rest, tris), ["spine"]
    )
    assert out["volume_ratio"] == pytest.approx(1.0)
    assert out["stretch_p99"] == pytest.approx(1.0)
    assert out["stretched_fraction"] == 0.0 and out["collapsed_fraction"] == 0.0
    assert out["joints"]["elbow"]["radial_ratio"] == pytest.approx(1.0)


def test_twisted_cylinder_reports_radial_collapse_at_the_joint():
    """A half turn blended linearly wrings the middle of the tube to a point:
    the cross-section at the joint has all but vanished, while a joint at the
    unrotated end is untouched."""
    rest, tris = _cylinder()
    posed = _twist_by_blending(rest, 180.0)
    out = dm.score_pose(
        rest, posed, tris, np.zeros(len(rest), int),
        [_joint("waist", 0.0), _joint("ankle", -0.95)], True, ["spine"],
    )
    assert out["joints"]["waist"]["radial_ratio"] < 0.2
    assert out["joints"]["ankle"]["radial_ratio"] > 0.8
    assert out["min_radial_ratio"] == out["joints"]["waist"]["radial_ratio"]
    assert out["volume_ratio"] < 0.6, "the wrung-out tube encloses far less than it did"


def test_an_open_mesh_reports_no_volume_at_all():
    rest, tris = _cylinder(capped=False)
    assert dm.is_closed(rest, tris) is False
    out = dm.score_pose(rest, rest * 1.5, tris, np.zeros(len(rest), int), [], False)
    assert out["volume_ratio"] is None
    assert out["stretch_p99"] == pytest.approx(1.5)


def test_closure_welds_vertices_a_uv_seam_split_in_two():
    """Textured meshes duplicate vertices along seams; judged by index they are
    all 'open', and no textured rig would ever get a volume."""
    rest, tris = _cylinder(capped=False)
    cap_rest, cap_tris = _cylinder(capped=True)
    assert dm.is_closed(cap_rest, cap_tris) is True
    # Duplicate every vertex of the second half of the tris onto its own index.
    extra = cap_rest[cap_tris[len(cap_tris) // 2:].reshape(-1)]
    split_rest = np.concatenate([cap_rest, extra])
    split_tris = cap_tris.copy()
    split_tris[len(cap_tris) // 2:] = (
        len(cap_rest) + np.arange(extra.shape[0]).reshape(-1, 3)
    )
    assert dm.is_closed(split_rest, split_tris) is True


def test_one_dragged_vertex_shows_in_the_maximum_and_the_fraction_not_the_p99():
    rest, tris = _cylinder()
    posed = rest.copy()
    posed[5] += np.array([0.0, 0.0, 30.0])
    out = dm.score_pose(rest, posed, tris, np.arange(len(rest)) % 2, [], True, ["a", "b"])
    assert out["stretch_max"] > 5.0
    assert 0.0 < out["stretched_fraction"] < 0.05
    assert out["stretch_p99"] == pytest.approx(1.0, abs=0.01), "a single vertex is under 1%"
    assert out["worst_stretch_bone"] == "b"


def test_a_dragged_region_shows_in_the_p99():
    rest, tris = _cylinder()
    posed = rest.copy()
    posed[:40] += np.array([0.0, 0.0, 30.0])
    out = dm.score_pose(rest, posed, tris, np.zeros(len(rest), int), [], True)
    assert out["stretch_p99"] > 5.0


def test_a_flattened_limb_counts_the_collapsed_edges():
    """Squashing the height collapses the edges that ran along it (a third of
    this mesh's) and leaves the rings' own edges alone."""
    rest, tris = _cylinder()
    posed = rest.copy()
    posed[:, 1] *= 0.05
    out = dm.score_pose(rest, posed, tris, np.zeros(len(rest), int), [], True)
    assert 0.25 < out["collapsed_fraction"] < 0.5
    assert out["stretched_fraction"] == 0.0


def test_a_joint_the_mesh_does_not_cover_reports_none_not_a_number():
    rest, tris = _cylinder()
    out = dm.score_pose(rest, rest, tris, np.zeros(len(rest), int), [_joint("far", 50.0)], True)
    assert out["joints"]["far"]["radial_ratio"] is None
    assert out["min_radial_ratio"] is None


def test_mismatched_or_non_finite_input_is_refused_not_scored():
    rest, tris = _cylinder()
    with pytest.raises(ValueError):
        dm.score_pose(rest, rest[:-1], tris, np.zeros(len(rest), int), [], True)
    bad = rest.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        dm.score_pose(rest, bad, tris, np.zeros(len(rest), int), [], True)
