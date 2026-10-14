"""The 2026-10-14 audit, finding create-panes-05: baking a posed GLB runs
Blender, and a rigged asset can exist on a host with none (Clay writes
``rig.glb`` itself). The door says so, in the app's one "needs Blender"
sentence, instead of the generic "could not bake this pose"."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from realmspinner import doctor
from realmspinner.kernels.rig import store
from realmspinner.pipelines import blender_run
from realmspinner.service import Invalid
from realmspinner.service import jobs as svc_jobs
from realmspinner.service import rig as svc_rig

BONES = ["hips", "spine", "head"]
IDENTITY = [0.0, 0.0, 0.0, 1.0]


def _rigged_job_with_a_pose(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a knight")["id"]
    job_dir = svc.config.data_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    (job_dir / "rig.json").write_text(
        json.dumps({"version": 1, "bones": [{"name": n} for n in BONES]})
    )
    svc.store.set_status(job_id, "done")
    record = svc_rig.save_pose(
        svc, job_id, {"name": "idle", "bones": {b: IDENTITY for b in BONES}}
    )
    return job_id, record["id"]


def _no_blender(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "blender_check",
        lambda **_kw: doctor.Check("Blender (rigging)", False, "bpy missing", fatal=False),
    )

    def never(spec, **kwargs):  # pragma: no cover - the failure being asserted
        raise AssertionError("Blender must not be spawned on a host without it")

    monkeypatch.setattr(blender_run, "run_worker", never)


def test_posed_model_refuses_with_the_blender_sentence_when_blender_is_not_installed(
    svc, monkeypatch
):
    job_id, pose_id = _rigged_job_with_a_pose(svc)
    _no_blender(monkeypatch)
    with pytest.raises(Invalid, match="needs Blender"):
        svc_rig.posed_model(svc, job_id, pose_id)
    assert not store.pose_glb_path(svc.job_dir(job_id), pose_id).exists()


def test_a_cached_posed_glb_is_still_served_without_blender(svc, monkeypatch):
    job_id, pose_id = _rigged_job_with_a_pose(svc)
    cached = Path(store.pose_glb_path(svc.job_dir(job_id), pose_id))
    cached.write_bytes(b"posed-glb")
    _no_blender(monkeypatch)
    assert svc_rig.posed_model(svc, job_id, pose_id).read_bytes() == b"posed-glb"
