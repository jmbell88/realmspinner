"""The 2026-10-14 audit, finding create-panes-03 (door half): a re-texture runs in Blender, so
``retexture_job`` refuses before the row is written when Blender is not installed, in
``remesh_job``'s own sentence shape. Without it the row was queued, took a GPU lease and ended
in an error row in history."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from realmspinner import doctor
from realmspinner.service import Invalid
from realmspinner.service import jobs as svc_jobs


def _finished_mesh(svc) -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    svc.store.set_status(job_id, "done")
    return job_id


def test_a_retexture_without_blender_is_refused_before_a_row_is_written(svc, monkeypatch):
    job_id = _finished_mesh(svc)
    monkeypatch.setattr(
        doctor, "blender_check", lambda *a, **k: SimpleNamespace(ok=False, detail="no bpy")
    )
    rows_before = len(svc.store.list(limit=1000))

    with pytest.raises(Invalid) as caught:
        svc_jobs.retexture_job(svc, job_id, "rusted iron")

    assert "Blender" in caught.value.message
    assert "re-texture" in caught.value.message
    assert caught.value.field == "prompt"
    assert len(svc.store.list(limit=1000)) == rows_before


def test_a_retexture_with_blender_still_queues(svc, monkeypatch):
    job_id = _finished_mesh(svc)
    monkeypatch.setattr(
        doctor, "blender_check", lambda *a, **k: SimpleNamespace(ok=True, detail="")
    )
    new_id = svc_jobs.retexture_job(svc, job_id, "rusted iron")["id"]
    assert svc.store.get(new_id)["kind"] == "retexture"
