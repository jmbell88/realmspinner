"""Every mesh a rework replaced, kept until the cap, restorable through
``revert_model``.

Before 2026-09-22 three writers -- ``optimize_job``, ``_q_mesh._remesh``
and ``_q_sprite._retexture`` -- each simply overwrote ``model.glb``, so a
remesh-then-retexture lost the remesh and a retarget after either lost both.
``pipelines.modelhistory`` is what makes that recoverable; this file is its
regression suite.

``_publish_model_version`` is driven directly against a minimal stand-in for
``Worker`` rather than a real queue -- it is a synchronous method on
``_q_mesh.MeshPostOps``, mixed into ``Worker``, and all it needs from one is
``.store``, ``.config`` and ``.artifact_lock``; ``svc`` already has the first
two and ``svc.convert_lock`` *is* the lock the worker's own
``artifact_lock`` is injected as (``studio.runtime`` wires
``worker.artifact_lock = svc.convert_lock``).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import uuid
from pathlib import Path

import pytest

from realmspinner.service import Conflict, Failed
from realmspinner.service import jobs as svc_jobs
from realmspinner.service.validation import DERIVED_PARAMS


class _FakeWorker:
    """Just enough of ``Worker`` for ``_publish_model_version`` to run."""

    def __init__(self, svc) -> None:
        self.store = svc.store
        self.config = svc.config
        self.artifact_lock = svc.convert_lock


def _mesh_job(svc, **params) -> str:
    """A finished job with a reconstruction and a mesh already on disk."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel", **params)["id"]
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "source.glb").write_bytes(b"source-bytes")
    (job_dir / "model.glb").write_bytes(b"model-v1")
    svc.store.set_status(job_id, "done")
    return job_id


def _finished_job(svc, **params) -> str:
    """A finished job with every ``DERIVED_PARAMS`` key already set -- the
    same fixture ``tests/service/test_service.py::_finished_job`` uses, to
    prove a reroll strips ``model_history`` the same way it strips every
    other derived record.
    """
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel", **params)["id"]
    svc.store.merge_params(job_id, {k: "stale" for k in DERIVED_PARAMS})
    svc.store.set_status(job_id, "done")
    return job_id


def _fake_optimize(monkeypatch) -> None:
    from realmspinner.pipelines import optimize

    def fake_run(source, out, **_kwargs):
        out.write_bytes(b"model-optimized")
        return {"ok": True}

    monkeypatch.setattr(optimize, "run", fake_run)


def _publish(fake, job_id, job_dir, content: bytes, *, kind: str, geometry: bool, detail: str):
    from realmspinner import _q_mesh

    temp = job_dir / f".{kind}.tmp.glb"
    temp.write_bytes(content)
    _q_mesh.MeshPostOps._publish_model_version(
        fake, job_id, temp, kind=kind, geometry=geometry, detail=detail
    )


# --- keeping a version -------------------------------------------------------


def test_a_retexture_after_a_remesh_keeps_the_remeshed_mesh(svc):
    """The defect this whole feature exists to close: a re-texture used to
    overwrite whatever a prior remesh had baked, with no way back."""
    from realmspinner.pipelines import modelhistory

    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    fake = _FakeWorker(svc)

    _publish(
        fake, job_id, job_dir, b"model-remeshed",
        kind="remesh", geometry=True, detail="8,000 quads",
    )
    _publish(
        fake, job_id, job_dir, b"model-retextured",
        kind="retexture", geometry=False, detail="6 views",
    )

    assert (job_dir / "model.glb").read_bytes() == b"model-retextured"
    entries = svc.store.get(job_id)["params"]["model_history"]
    assert [e["kind"] for e in entries] == ["remesh", "retexture"]
    # Version 1 is the pre-remesh mesh; version 2 is the remeshed mesh the
    # re-texture would otherwise have destroyed with nothing to recover it.
    assert modelhistory.version_path(job_dir, 1).read_bytes() == b"model-v1"
    assert modelhistory.version_path(job_dir, 2).read_bytes() == b"model-remeshed"


def test_retarget_keeps_the_mesh_it_replaced(svc, monkeypatch):
    from realmspinner.pipelines import modelhistory

    _fake_optimize(monkeypatch)
    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    original = (job_dir / "model.glb").read_bytes()

    svc_jobs.optimize_job(svc, job_id, profile="raw")

    entries = svc.store.get(job_id)["params"].get("model_history") or []
    assert len(entries) == 1, "the retarget kept the mesh it replaced"
    assert entries[0]["kind"] == "optimize"
    assert entries[0]["geometry"] is True
    assert modelhistory.version_path(job_dir, entries[0]["n"]).read_bytes() == original
    assert (job_dir / "model.glb").read_bytes() == b"model-optimized"


def test_a_failed_retarget_leaves_no_history_entry(svc, monkeypatch):
    """``optimize.run`` only creates ``dest`` on success, so model.glb is
    untouched on this path -- the version ``optimize_job`` pushed before
    calling it would describe a replacement that never happened."""
    from realmspinner.pipelines import modelhistory, optimize

    def boom(source, out, **_kwargs):
        raise optimize.OptimizeError("gltfpack exploded")

    monkeypatch.setattr(optimize, "run", boom)
    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)

    with pytest.raises(Failed):
        svc_jobs.optimize_job(svc, job_id, profile="raw")

    assert not svc.store.get(job_id)["params"].get("model_history")
    assert not modelhistory.version_path(job_dir, 1).exists()
    versions_dir = job_dir / modelhistory.VERSIONS_DIR
    assert not versions_dir.exists() or not any(versions_dir.iterdir())


def test_a_retarget_tolerates_a_hand_edited_model_history(svc, monkeypatch):
    """``_finished_job`` (this file and ``test_service.py`` both) seeds every
    ``DERIVED_PARAMS`` key, ``model_history`` included, with the marker
    string ``"stale"`` -- not a list of entries. ``modelhistory.entries_of``
    is what stops that from blowing up ``keep()``'s own ``entries[-1]``
    indexing the way a bare ``.get(...) or []`` did."""
    _fake_optimize(monkeypatch)
    job_id = _mesh_job(svc)
    svc.store.merge_params(job_id, {"model_history": "stale"})

    svc_jobs.optimize_job(svc, job_id, profile="raw")

    entries = svc.store.get(job_id)["params"]["model_history"]
    assert len(entries) == 1
    assert entries[0]["n"] == 1


def test_a_retarget_forgets_the_retexture_report(svc, monkeypatch):
    """A retarget already dropped a stale ``remesh`` report; ``retexture`` had
    the same defect until this row joined the drop list (2026-09-22)."""
    _fake_optimize(monkeypatch)
    job_id = _mesh_job(svc)
    svc.store.merge_params(job_id, {"retexture": {"coverage": 0.9, "views": 10}})

    svc_jobs.optimize_job(svc, job_id, profile="raw")

    assert "retexture" not in svc.store.get(job_id)["params"]


# --- restoring a version -----------------------------------------------------


def test_revert_restores_the_bytes_and_the_params_that_described_them(svc):
    """``profile`` in particular: it is in ``vectors.VECTOR_PARAMS``, so a
    restore that left the wrong tier on the row would key a findings-corpus
    verdict to a triangle budget no longer on disk."""
    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    fake = _FakeWorker(svc)
    svc.store.merge_params(job_id, {"profile": "standard", "optimize": {"ok": True}})

    _publish(fake, job_id, job_dir, b"model-remeshed", kind="remesh", geometry=True, detail="x")
    n = svc.store.get(job_id)["params"]["model_history"][0]["n"]

    result = svc_jobs.revert_model(svc, job_id, version=n)

    assert result["ok"] is True
    assert (job_dir / "model.glb").read_bytes() == b"model-v1"
    params = svc.store.get(job_id)["params"]
    assert params["profile"] == "standard"
    assert params["optimize"] == {"ok": True}


def test_revert_across_geometry_drops_every_derived_export_and_reports_the_rig(svc):
    from realmspinner.service import files as svc_files

    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    fake = _FakeWorker(svc)
    _publish(fake, job_id, job_dir, b"model-remeshed", kind="remesh", geometry=True, detail="x")
    n = svc.store.get(job_id)["params"]["model_history"][0]["n"]

    derived_name = next(iter(svc_files.DERIVED))
    (job_dir / derived_name).write_bytes(b"stale")
    (job_dir / "rig.glb").write_bytes(b"rig")
    (job_dir / "rig.json").write_bytes(b"{}")

    result = svc_jobs.revert_model(svc, job_id, version=n)

    assert not (job_dir / derived_name).exists()
    assert result["surface"] is False
    assert result["stale"] == ["rig.glb", "rig.json"]


def test_revert_across_only_a_retexture_drops_only_surface_exports(svc):
    from realmspinner.pipelines import retexture
    from realmspinner.service import files as svc_files

    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    fake = _FakeWorker(svc)
    _publish(
        fake, job_id, job_dir, b"model-retextured",
        kind="retexture", geometry=False, detail="x",
    )
    n = svc.store.get(job_id)["params"]["model_history"][0]["n"]

    surface_name = next(iter(retexture.SURFACE_DERIVED))
    geometry_only = next(
        name for name in svc_files.DERIVED if name not in retexture.SURFACE_DERIVED
    )
    (job_dir / surface_name).write_bytes(b"stale")
    (job_dir / geometry_only).write_bytes(b"kept")
    (job_dir / "rig.glb").write_bytes(b"rig")

    result = svc_jobs.revert_model(svc, job_id, version=n)

    assert not (job_dir / surface_name).exists()
    assert (job_dir / geometry_only).exists()
    assert (job_dir / "rig.glb").exists()
    assert result["surface"] is True
    assert result["stale"] == []


def test_revert_writes_model_glb_through_a_staging_name(svc, monkeypatch):
    """``derive._staged`` is a stage-then-rename, never an in-place write --
    ``model.glb`` is served on mere existence once a job is done."""
    from realmspinner.service import derive as svc_derive

    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    fake = _FakeWorker(svc)
    _publish(fake, job_id, job_dir, b"model-remeshed", kind="remesh", geometry=True, detail="x")
    n = svc.store.get(job_id)["params"]["model_history"][0]["n"]

    # A list, not a dict overwritten in place: ``modelhistory.keep`` (also
    # called by this revert, to snapshot the mesh it is about to replace) does
    # its own staged ``os.replace`` calls through the same module-global
    # ``os``, so the interesting one has to be found rather than assumed last.
    calls: list[tuple[str, str]] = []
    original_replace = svc_derive.os.replace

    def spy(src, dst):
        calls.append((Path(src).name, Path(dst).name))
        return original_replace(src, dst)

    monkeypatch.setattr(svc_derive.os, "replace", spy)

    svc_jobs.revert_model(svc, job_id, version=n)

    onto_model_glb = [c for c in calls if c[1] == "model.glb"]
    assert len(onto_model_glb) == 1, calls
    src_name, _ = onto_model_glb[0]
    assert src_name != "model.glb" and src_name.startswith("."), "not a staged write"
    assert (job_dir / "model.glb").read_bytes() == b"model-v1"


def test_revert_refuses_a_running_job_and_one_with_dependents(svc):
    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    fake = _FakeWorker(svc)
    _publish(fake, job_id, job_dir, b"model-remeshed", kind="remesh", geometry=True, detail="x")
    n = svc.store.get(job_id)["params"]["model_history"][0]["n"]

    svc.store.set_status(job_id, "running")
    with pytest.raises(Conflict):
        svc_jobs.revert_model(svc, job_id, version=n)
    svc.store.set_status(job_id, "done")

    dep_id = svc.store.create("retexture", "x", {"source_job": job_id}, uuid.uuid4().hex[:12])
    svc.store.set_status(dep_id, "queued")
    with pytest.raises(Conflict):
        svc_jobs.revert_model(svc, job_id, version=n)


def test_revert_is_itself_kept_so_it_can_be_undone(svc):
    from realmspinner.pipelines import modelhistory

    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    fake = _FakeWorker(svc)
    _publish(fake, job_id, job_dir, b"model-remeshed", kind="remesh", geometry=True, detail="x")
    n = svc.store.get(job_id)["params"]["model_history"][0]["n"]
    replaced = (job_dir / "model.glb").read_bytes()

    svc_jobs.revert_model(svc, job_id, version=n)

    entries = svc.store.get(job_id)["params"]["model_history"]
    assert len(entries) == 2
    assert entries[-1]["kind"] == "revert"
    assert modelhistory.version_path(job_dir, entries[-1]["n"]).read_bytes() == replaced


# --- the cap, deletion and serving --------------------------------------------


def test_history_is_capped_and_evicts_the_oldest_file(svc):
    from realmspinner.pipelines import modelhistory

    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    fake = _FakeWorker(svc)

    for i in range(modelhistory.MAX_MODEL_VERSIONS + 1):
        _publish(
            fake, job_id, job_dir, f"model-{i}".encode(),
            kind="remesh", geometry=True, detail=f"step {i}",
        )

    entries = svc.store.get(job_id)["params"]["model_history"]
    assert len(entries) == modelhistory.MAX_MODEL_VERSIONS
    assert entries[0]["n"] == 2, "the oldest entry (n=1) should have been evicted"
    assert not modelhistory.version_path(job_dir, 1).exists()
    assert not modelhistory.meta_path(job_dir, 1).exists()


def test_deleting_a_job_deletes_its_versions(svc):
    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    fake = _FakeWorker(svc)
    _publish(fake, job_id, job_dir, b"model-remeshed", kind="remesh", geometry=True, detail="x")

    assert (job_dir / "versions").exists()

    from realmspinner.service import _jobs_lifecycle

    _jobs_lifecycle.delete_job(svc, job_id)

    assert not job_dir.exists()


def test_model_versions_are_never_served(svc):
    from realmspinner.service import NotFound
    from realmspinner.service import derive as svc_derive

    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    fake = _FakeWorker(svc)
    _publish(fake, job_id, job_dir, b"model-remeshed", kind="remesh", geometry=True, detail="x")

    with pytest.raises(NotFound):
        svc_derive.get_file(svc, job_id, "versions/1.model.glb")


def test_a_rerun_does_not_inherit_model_history(svc):
    job_id = _finished_job(svc)
    new_id = svc_jobs.rerun_job(svc, job_id)["id"]
    assert "model_history" not in svc.store.get(new_id)["params"]


# --- the lock ------------------------------------------------------------


def test_every_model_glb_writer_takes_the_model_lock():
    """Every writer of ``model.glb`` takes ``MODEL_LOCK``, either directly or
    by delegating to ``_publish_model_version``, which does. An AST walk
    rather than a substring test, since what matters is that the call is
    really there, not that its name merely appears somewhere in the body.
    """
    from realmspinner import _q_mesh, _q_sprite
    from realmspinner.service import _jobs_rework

    def _tree(fn):
        return ast.parse(textwrap.dedent(inspect.getsource(fn)))

    def calls_lock(fn) -> bool:
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("artifact_lock", "convert_lock")
            for node in ast.walk(_tree(fn))
        )

    def calls_publish(fn) -> bool:
        return any(
            isinstance(node, ast.Attribute) and node.attr == "_publish_model_version"
            for node in ast.walk(_tree(fn))
        )

    assert calls_lock(_q_mesh.MeshPostOps._publish_model_version), (
        "_publish_model_version no longer takes an artifact/convert lock"
    )
    assert calls_publish(_q_mesh.MeshPostOps._remesh), (
        "_remesh no longer publishes through _publish_model_version"
    )
    assert calls_publish(_q_sprite.SpriteOps._retexture), (
        "_retexture no longer publishes through _publish_model_version"
    )
    assert calls_lock(_jobs_rework.optimize_job), "optimize_job no longer takes a lock"
    assert calls_lock(_jobs_rework.revert_model), "revert_model no longer takes a lock"
