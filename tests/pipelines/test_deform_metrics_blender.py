"""The deformation numbers through real Blender: a rigged tube, rest and twisted.

``kernels.rig.deform_metrics`` is tested on arrays in ``tests/kernels/rig``; this
is the half only Blender can supply -- the depsgraph-evaluated vertices, the
dominant-bone lookup and the joint frames -- end to end through ``op_rig`` and
``op_sheet``.
"""

from __future__ import annotations

import pytest

from realmspinner.kernels.rig import blender_spec, poses, store
from realmspinner.pipelines import blender_run

pytestmark = pytest.mark.timeout(600)


@pytest.fixture(scope="module")
def rigged_tube(tmp_path_factory, gl):
    # ``gl`` is a skip, not a dependency: the sheet op renders through Blender's
    # GPU context, and on a runner with none (Windows CI) the worker dies with an
    # access violation (exit 3221225477) rather than raising. Where moderngl
    # cannot get a context neither can Blender, so these skip there as ``gl`` does.
    pytest.importorskip("bpy")
    import trimesh

    job = tmp_path_factory.mktemp("serpent-qa")
    tube = trimesh.creation.cylinder(radius=0.05, height=1.0, sections=16)
    # Uniform, not ``subdivide_to_size``: that splits faces one at a time and
    # leaves T-junctions, and a cracked tube is open.
    for _ in range(4):
        tube = tube.subdivide()
    tube.export(job / "model.glb")
    blender_run.run_worker(blender_spec.rig_spec(job, "serpent"), timeout=600)
    store.finalize_rig(job)
    return job


def _scores(job, pose_bones: dict[str, dict], tmp_path):
    cells = [
        {"index": i, "yaw": 0.0, "pose": name, "frame": 0, "bones": bones, "pose_space": "delta"}
        for i, (name, bones) in enumerate(pose_bones.items())
    ]
    spec = blender_spec.sheet_spec(
        job / "rig.glb", tmp_path, cells, frame_size=48, elevation=0.3,
        lighting="flat", deform_metrics=True,
    )
    return blender_run.run_worker(spec, timeout=600)


def test_a_rigged_tube_at_rest_scores_as_nothing_happened_and_a_twist_scores_below_it(
    rigged_tube, tmp_path
):
    twist = next(p for p in poses.deform_battery("serpent") if p["name"] == "roll twist")
    result = _scores(rigged_tube, {"rest": {}, "twist": twist["bones"]}, tmp_path)
    rest, twisted = result["deform_metrics"]["rest"], result["deform_metrics"]["twist"]
    assert rest is not None and twisted is not None
    assert rest["stretch_p99"] == pytest.approx(1.0, abs=1e-3)
    assert rest["min_radial_ratio"] == pytest.approx(1.0, abs=1e-2)
    assert rest["joints"], "the serpent's joints were measured"
    # A capped tube is closed once its seams are welded; the number must exist.
    assert rest["volume_ratio"] == pytest.approx(1.0, abs=1e-3)
    assert twisted["min_radial_ratio"] < rest["min_radial_ratio"]
    assert twisted["volume_ratio"] < 1.0


def test_a_sheet_that_does_not_ask_returns_no_metrics(rigged_tube, tmp_path):
    cells = [{"index": 0, "yaw": 0.0, "pose": "rest", "frame": 0, "bones": {}}]
    spec = blender_spec.sheet_spec(
        rigged_tube / "rig.glb", tmp_path, cells, frame_size=48, elevation=0.3, lighting="flat"
    )
    assert "deform_metrics" not in blender_run.run_worker(spec, timeout=600)
