"""Regression tests for the 2026-09-23 audit's history-service findings
(service-01, service-03, muse-01).

Before this fix, ``pipelines.modelhistory.keep`` staged a new version *and*
evicted the oldest one past the cap in a single call, before the caller's own
write had even been attempted. Splitting it into ``stage`` (push only) and
``commit`` (evict only, called after success) means a caller with a failure
path -- ``optimize_job``, ``_q_mesh.MeshPostOps._publish_model_version`` --
can undo exactly its own staged push on failure via ``discard_last``, with no
eviction ever having happened to undo.

Fixture shapes follow ``tests/service/test_model_history.py`` (``_mesh_job``,
``_FakeWorker``, ``_publish``) so this file reads as that one's companion
rather than reinventing the same setup.
"""

from __future__ import annotations

import pytest

from realmspinner.service import Failed
from realmspinner.service import jobs as svc_jobs


class _FakeWorker:
    """Just enough of ``Worker`` for ``_publish_model_version`` to run."""

    def __init__(self, svc) -> None:
        self.store = svc.store
        self.config = svc.config
        self.artifact_lock = svc.convert_lock


def _mesh_job(svc, **params) -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel", **params)["id"]
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "source.glb").write_bytes(b"source-bytes")
    (job_dir / "model.glb").write_bytes(b"model-v1")
    svc.store.set_status(job_id, "done")
    return job_id


def _publish(fake, job_id, job_dir, content: bytes, *, kind: str, geometry: bool, detail: str):
    from realmspinner import _q_mesh

    temp = job_dir / f".{kind}.tmp.glb"
    temp.write_bytes(content)
    _q_mesh.MeshPostOps._publish_model_version(
        fake, job_id, temp, kind=kind, geometry=geometry, detail=detail
    )


def _fill_history_to_cap(svc, job_id, job_dir, fake) -> None:
    """Push ``MAX_MODEL_VERSIONS`` successful versions via the real publish
    path, so the history is exactly at the cap with all files on disk."""
    from realmspinner.pipelines import modelhistory

    for i in range(modelhistory.MAX_MODEL_VERSIONS):
        _publish(
            fake, job_id, job_dir, f"model-v{i + 2}".encode(),
            kind="optimize", geometry=True, detail=f"pass {i}",
        )


def test_optimize_job_failure_does_not_evict_the_oldest_kept_version_when_the_history_is_at_cap(
    svc, monkeypatch
):
    """2026-09-23 audit, finding service-01: with ``model_history`` already
    at the cap, a failed retarget used to evict the oldest kept version
    (inside ``keep()``, before ``optimize.run`` was even tried) and never
    recover it on failure -- ``discard_last`` only undid the push, not the
    eviction ``keep()`` performed to make room for it.
    """
    from realmspinner.pipelines import modelhistory, optimize

    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    fake = _FakeWorker(svc)

    _fill_history_to_cap(svc, job_id, job_dir, fake)
    entries_before = svc.store.get(job_id)["params"]["model_history"]
    assert len(entries_before) == modelhistory.MAX_MODEL_VERSIONS
    oldest_n = entries_before[0]["n"]
    assert modelhistory.version_path(job_dir, oldest_n).exists()

    def boom(source, out, **_kwargs):
        raise optimize.OptimizeError("gltfpack exploded")

    monkeypatch.setattr(optimize, "run", boom)

    with pytest.raises(Failed):
        svc_jobs.optimize_job(svc, job_id, profile="raw")

    entries_after = svc.store.get(job_id)["params"]["model_history"]
    assert [e["n"] for e in entries_after] == [e["n"] for e in entries_before], (
        "a failed retarget must not change the kept history at all"
    )
    assert modelhistory.version_path(job_dir, oldest_n).exists(), (
        "the oldest kept version must survive a failed retarget even at the cap"
    )


def test_publish_model_version_discards_the_staged_history_entry_when_the_replace_fails(
    svc, monkeypatch
):
    """2026-09-23 audit, finding service-03: ``keep()`` staged a version (and
    could evict one at the cap) before ``os.replace``; a failed replace left
    the staged pair orphaned and the eviction already applied, while
    ``merge_params`` -- never reached -- left the stored index still naming
    the version eviction had just deleted from disk.
    """
    from realmspinner import _q_mesh
    from realmspinner.pipelines import modelhistory

    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    fake = _FakeWorker(svc)

    entries_before = svc.store.get(job_id)["params"].get("model_history") or []

    def boom_replace(*_args, **_kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(_q_mesh.os, "replace", boom_replace)

    temp = job_dir / ".optimize.tmp.glb"
    temp.write_bytes(b"model-v2")
    with pytest.raises(OSError):
        _q_mesh.MeshPostOps._publish_model_version(
            fake, job_id, temp, kind="optimize", geometry=True, detail="x"
        )

    entries_after = svc.store.get(job_id)["params"].get("model_history") or []
    assert entries_after == entries_before, (
        "a failed replace must not add an entry to the stored index"
    )
    # And nothing the failed push staged should be left behind, orphaned.
    versions_dir = job_dir / modelhistory.VERSIONS_DIR
    assert not versions_dir.exists() or not any(versions_dir.iterdir())
    # model.glb itself is untouched -- the original mesh is still what's served.
    assert (job_dir / "model.glb").read_bytes() == b"model-v1"


def test_separate_conflict_message_does_not_call_a_music_take_a_mesh(svc, monkeypatch):
    """2026-09-23 audit, finding muse-01: ``_require_no_dependents``'s
    conflict message hardcoded "mesh", so refusing a second ``separate``
    (stems) job on a music take that already has one in flight told the user
    a job had been "started from this mesh" -- there is no mesh involved.
    """
    from realmspinner.service import Conflict, _jobs_lifecycle

    # ``create_job`` only accepts kind "text"/"image" (mesh jobs); a music
    # take is created directly on the store the way ``create_music_job``
    # itself does, which this test doesn't need the whole pipeline for.
    job_id = svc.store.create("music", "a jig", {}, stage="music", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "track.wav").write_bytes(b"wav-bytes")

    monkeypatch.setattr(
        _jobs_lifecycle, "dependent_jobs", lambda *_a, **_k: ["fake-blocking-id"]
    )

    with pytest.raises(Conflict) as excinfo:
        svc_jobs.separate_job(svc, job_id)

    assert "mesh" not in str(excinfo.value), str(excinfo.value)
    assert "music take" in str(excinfo.value)
