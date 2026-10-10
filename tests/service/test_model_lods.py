"""``model_lods.zip`` as a served artifact: derived on request, dropped by every
rework that makes it describe a mesh that is gone."""

from __future__ import annotations

import pytest

from realmspinner.pipelines import optimize, remesh, retexture
from realmspinner.service import NotReady
from realmspinner.service import derive as svc_derive
from realmspinner.service import files as svc_files
from realmspinner.service import jobs as svc_jobs
from realmspinner.studio import artifacts


def _job(svc) -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "source.glb").write_bytes(b"source")
    (job_dir / "model.glb").write_bytes(b"model")
    svc.store.set_status(job_id, "done")
    return job_id


def _fake_chain(monkeypatch, calls: list):
    def fake(model, out, *, exe, **_kwargs):
        calls.append((model, exe))
        out.write_bytes(b"PK-lods")
        return {}

    monkeypatch.setattr(optimize, "lod_chain", fake)


def test_the_lod_chain_is_one_more_derived_export_wherever_the_set_is_spelled():
    assert "model_lods.zip" in svc_files.DERIVED
    assert svc_files.MEDIA["model_lods.zip"] == "application/zip"
    assert ("model_lods.zip", "LOD chain (zip)") in artifacts.ARTIFACTS
    assert set(remesh.GEOMETRY_DERIVED) == set(svc_files.DERIVED)
    # It carries the skin (the textures are inside every level), so it goes with
    # the other surface exports when a re-texture lands.
    assert "model_lods.zip" in retexture.SURFACE_DERIVED


def test_the_lod_chain_is_derived_on_first_request_and_then_served_from_disk(svc, monkeypatch):
    calls: list = []
    _fake_chain(monkeypatch, calls)
    svc.config.gltfpack_exe = svc.job_dir("x").parent / "gltfpack.exe"
    svc.config.gltfpack_exe.write_bytes(b"exe")
    job_id = _job(svc)

    first = svc_derive.get_file(svc, job_id, "model_lods.zip")
    second = svc_derive.get_file(svc, job_id, "model_lods.zip")

    assert first == second == svc.job_dir(job_id) / "model_lods.zip"
    assert first.read_bytes() == b"PK-lods"
    assert len(calls) == 1, "existence is the freshness test; the second request is a read"
    assert calls[0][0] == svc.job_dir(job_id) / "model.glb"
    assert not [p for p in svc.job_dir(job_id).iterdir() if p.name.startswith(".")]


def test_without_gltfpack_the_chain_is_not_ready_and_says_why(svc):
    job_id = _job(svc)
    with pytest.raises(NotReady, match="gltfpack"):
        svc_derive.get_file(svc, job_id, "model_lods.zip")
    assert not (svc.job_dir(job_id) / "model_lods.zip").exists()


def test_a_mesh_too_small_to_have_a_chain_is_not_ready_with_the_reason(svc, monkeypatch):
    def refuse(*_a, **_k):
        raise ValueError("the mesh did not get smaller than LOD1 (40 triangles)")

    monkeypatch.setattr(optimize, "lod_chain", refuse)
    svc.config.gltfpack_exe = svc.job_dir("x").parent / "gltfpack.exe"
    svc.config.gltfpack_exe.write_bytes(b"exe")
    job_id = _job(svc)
    with pytest.raises(NotReady, match="no LOD chain.*did not get smaller"):
        svc_derive.get_file(svc, job_id, "model_lods.zip")
    assert not (svc.job_dir(job_id) / "model_lods.zip").exists()


def test_model_lods_zip_is_dropped_by_a_retarget_remesh_and_retexture(svc, monkeypatch):
    """A retarget (``optimize_job``) and a restore drop all of ``DERIVED``; a
    queued remesh deletes ``GEOMETRY_DERIVED``; a re-texture the surface set --
    each by name, so each list has to hold the new one."""
    from realmspinner.service import _jobs_rework

    job_id = _job(svc)
    job_dir = svc.job_dir(job_id)

    def fake_optimize(source, out, **_kwargs):
        out.write_bytes(b"model-optimized")
        return {"requested": 100, "achieved": 100, "source_triangles": 200, "bytes": 15}

    monkeypatch.setattr(optimize, "run", fake_optimize)
    svc.config.gltfpack_exe = job_dir.parent / "gltfpack.exe"
    svc.config.gltfpack_exe.write_bytes(b"exe")

    (job_dir / "model_lods.zip").write_bytes(b"old")
    svc_jobs.optimize_job(svc, job_id, profile="draft")
    assert not (job_dir / "model_lods.zip").exists(), "a retarget left a stale LOD chain"

    droppers = {"remesh": remesh.GEOMETRY_DERIVED, "retexture": retexture.SURFACE_DERIVED}
    for rework, names in droppers.items():
        assert "model_lods.zip" in names, f"a {rework} would serve a chain of the old mesh"
    assert _jobs_rework.stale_surface_artifacts(job_dir) == []
    (job_dir / "model_lods.zip").write_bytes(b"old")
    assert "model_lods.zip" in _jobs_rework.stale_surface_artifacts(job_dir)


def test_the_inspector_greys_the_lod_button_with_a_reason_when_gltfpack_is_missing():
    from types import SimpleNamespace

    from realmspinner.studio.panes import inspector

    ctx = SimpleNamespace(svc=SimpleNamespace(config=None), rigging_available=True)
    assert inspector._why_blocked(ctx, {"status": "done"}, "model_lods.zip", False, True) == (
        "needs gltfpack"
    )
