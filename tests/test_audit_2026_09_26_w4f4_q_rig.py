"""poser-rig-07 (2026-09-26 audit): ``finalize_rig``'s delete-then-rename
window raced a pose bake landing under the old rig.

``store.finalize_rig`` deletes every stale ``poses/*.glb`` up front and then
spends up to 5s retrying the rig.glb/rig.json rename against a lock a Windows
antivirus or indexer can hold that long. ``service.rig.posed_model`` bakes a
pose from whatever rig.glb is on disk *right now*, guarded by nothing but
that one pose id's own ``pose:<id>`` lock. A bake landing in that window reads
the *old* rig and writes straight back to ``poses/<id>.glb`` -- the very file
finalize just cleared -- so it is never invalidated again and is served under
the new rig's authority forever.

``_q_rig.RigOps._finalize_rig_locked`` is driven directly against a minimal
stand-in for ``Worker``, the same shape ``tests/service/test_model_history.py``
uses for ``_publish_model_version``: all it needs is ``.artifact_lock``, and
``svc.convert_lock`` *is* the lock ``studio.runtime`` injects it as.
"""

from __future__ import annotations

import json
import threading

from realmspinner import _q_rig
from realmspinner.kernels.rig import store


class _FakeWorker:
    """Just enough of ``Worker`` for ``_finalize_rig_locked`` to run."""

    def __init__(self, svc) -> None:
        self.artifact_lock = svc.convert_lock


POSE_ID = "0123456789ab"


def _pose_record(job_dir, pose_id: str) -> None:
    poses = job_dir / "poses"
    poses.mkdir(parents=True, exist_ok=True)
    (poses / f"{pose_id}.json").write_text(
        json.dumps({"id": pose_id, "bones": {}}), encoding="utf-8"
    )
    (poses / f"{pose_id}.glb").write_bytes(b"stale-bake")


def _rig_pending_rename(job_dir) -> None:
    (job_dir / store.RIG_GLB_TMP).write_bytes(b"new-rig")
    (job_dir / store.RIG_JSON_TMP).write_text("{}", encoding="utf-8")
    (job_dir / "rig.glb").write_bytes(b"old-rig")
    (job_dir / "rig.json").write_text("{}", encoding="utf-8")


def test_finalize_rig_locked_holds_the_pose_lock_across_the_delete_then_rename_window(
    svc, tmp_path, monkeypatch
):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _pose_record(job_dir, POSE_ID)
    _rig_pending_rename(job_dir)

    entered = threading.Event()
    release = threading.Event()
    real_finalize = store.finalize_rig

    def _slow_finalize(directory):
        entered.set()
        release.wait(timeout=5)
        real_finalize(directory)

    monkeypatch.setattr(_q_rig.store, "finalize_rig", _slow_finalize)

    worker = _FakeWorker(svc)
    job_id = "abcdef012345"
    thread = threading.Thread(
        target=_q_rig.RigOps._finalize_rig_locked, args=(worker, job_id, job_dir)
    )
    thread.start()
    try:
        assert entered.wait(timeout=2), "finalize_rig_locked never started"

        # While finalize is still inside its (simulated) slow rename window,
        # the exact lock ``posed_model`` would take for this pose must
        # already be held -- proving a concurrent ``posed_model`` call would
        # block rather than bake a pose under the stale rig.
        lock = svc.convert_lock(job_id, f"pose:{POSE_ID}")
        got = lock.acquire(timeout=0.2)
        try:
            assert not got, (
                "a pose bake could still start mid-finalize -- the exact "
                "race poser-rig-07 named"
            )
        finally:
            if got:
                lock.release()
    finally:
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()

    # And once finalize has actually finished, the lock is free again -- this
    # is a lock around the window, not a deadlock.
    lock = svc.convert_lock(job_id, f"pose:{POSE_ID}")
    assert lock.acquire(timeout=1), "the pose lock was never released"
    lock.release()

    # finalize_rig's own contract still holds: the rig published, the stale
    # pose bake is gone.
    assert (job_dir / "rig.glb").read_bytes() == b"new-rig"
    assert not (job_dir / "poses" / f"{POSE_ID}.glb").exists()


def test_finalize_rig_locked_ignores_a_pose_that_has_no_id_in_its_record(svc, tmp_path):
    """A pose record with no usable id must not crash the lock collection --
    ``store.list_poses`` already tolerates a corrupt record; this must too."""
    job_dir = tmp_path / "job2"
    job_dir.mkdir()
    _rig_pending_rename(job_dir)
    poses = job_dir / "poses"
    poses.mkdir()
    (poses / "not-an-id!!.json").write_text("not json at all", encoding="utf-8")

    worker = _FakeWorker(svc)
    _q_rig.RigOps._finalize_rig_locked(worker, "abcdef012345", job_dir)
    assert (job_dir / "rig.glb").read_bytes() == b"new-rig"
