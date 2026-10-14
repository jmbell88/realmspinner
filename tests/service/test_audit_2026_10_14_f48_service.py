"""The 2026-10-14 audit's finding pipelines-27 (the optimize door). The test's
name is the claim.
"""

from __future__ import annotations

import pytest

from realmspinner.service import Failed
from realmspinner.service import jobs as svc_jobs


def _mesh_job(svc) -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "source.glb").write_bytes(b"source-bytes")
    (job_dir / "model.glb").write_bytes(b"model-v1")
    svc.store.set_status(job_id, "done")
    return job_id


def test_optimize_job_backs_out_its_staged_version_when_the_output_cannot_be_measured(
    svc, monkeypatch
):
    """The 2026-10-14 audit, finding pipelines-mesh-03: ``optimize.run`` lets a
    trimesh failure escape as a bare ``ValueError``, and the door caught only
    ``OptimizeError``, so the version it had staged before the run stayed on
    disk with no index entry and the UI got a raw ValueError instead of Failed."""
    from realmspinner.pipelines import modelhistory, optimize

    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)

    def unreadable(source, out, **_kwargs):
        raise ValueError("this is not a mesh")

    monkeypatch.setattr(optimize, "run", unreadable)

    with pytest.raises(Failed):
        svc_jobs.optimize_job(svc, job_id, profile="raw")

    versions = job_dir / modelhistory.VERSIONS_DIR
    assert not versions.exists() or list(versions.iterdir()) == [], (
        "the version staged before the run must be backed out"
    )
    assert not (svc.store.get(job_id)["params"].get("model_history") or [])
    assert (job_dir / "model.glb").read_bytes() == b"model-v1"


def test_optimize_job_backs_out_its_staged_version_on_any_other_exception_too(svc, monkeypatch):
    from realmspinner.pipelines import modelhistory, optimize

    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)

    def boom(source, out, **_kwargs):
        raise MemoryError("out of memory")

    monkeypatch.setattr(optimize, "run", boom)
    with pytest.raises(MemoryError):
        svc_jobs.optimize_job(svc, job_id, profile="raw")
    versions = job_dir / modelhistory.VERSIONS_DIR
    assert not versions.exists() or list(versions.iterdir()) == []
