"""Regression tests for the 2026-09-23 (second run) audit's service-layer
findings service-01 (revert_model half), service-02 and service-03.

``tests/service/test_audit_2026_09_23_history.py`` already covers the first
run's service-01/service-03 fixes for ``optimize_job`` and
``_publish_model_version``. This file covers what that pass left half done:
``revert_model`` still called the combined ``modelhistory.keep()`` directly,
so a restore's own staged write (copying a kept version's bytes back onto
``model.glb``) could still evict the oldest kept version before that write
had even been attempted. Fixture shapes follow that file's ``_mesh_job``/
``_FakeWorker``/``_publish`` so this reads as its companion.
"""

from __future__ import annotations

from pathlib import Path

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
    path, so the history is exactly at the cap with all files on disk, and
    ``model.glb`` ends on the last of them -- leaving version 1 (the very
    first snapshot, taken of the original ``model-v1``) as the oldest kept
    entry, the one a version-cap eviction would take first.
    """
    from realmspinner.pipelines import modelhistory

    for i in range(modelhistory.MAX_MODEL_VERSIONS):
        _publish(
            fake, job_id, job_dir, f"model-v{i + 2}".encode(),
            kind="optimize", geometry=True, detail=f"pass {i}",
        )


def test_a_failed_revert_does_not_evict_the_version_it_did_not_restore(svc, monkeypatch):
    """2026-09-23 (second run) audit, finding service-01: ``revert_model``
    called the combined ``modelhistory.keep()``, which evicted the oldest
    kept version unconditionally -- *before* the restore's own staged write
    (copying the chosen version's bytes onto ``model.glb``) had even been
    attempted. A write that then failed (disk full, a permissions error)
    left the restore refused but the evicted version already gone, and
    ``model_history`` was never merged on that path either, so the row went
    on listing the now-gone entry and a retry was refused with "version N is
    no longer available" -- for a version the failure never touched.
    """
    from realmspinner.pipelines import modelhistory

    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    fake = _FakeWorker(svc)

    _fill_history_to_cap(svc, job_id, job_dir, fake)
    entries_before = svc.store.get(job_id)["params"]["model_history"]
    assert len(entries_before) == modelhistory.MAX_MODEL_VERSIONS
    oldest_n = entries_before[0]["n"]
    restore_n = entries_before[-1]["n"]
    assert modelhistory.version_path(job_dir, oldest_n).exists()
    assert modelhistory.version_path(job_dir, restore_n).exists()

    import shutil

    real_copyfile = shutil.copyfile

    def boom_copyfile(src, *args, **kwargs):
        # Only the restore's own write fails -- copying *from* a kept
        # version's file back onto ``model.glb``. ``modelhistory.stage``'s
        # own snapshot copy (*from* ``model.glb`` into ``versions/``, taken
        # just before that write) must still succeed, the same way a real
        # disk-full failure would not discriminate by caller but this test
        # only wants to prove what happens when the *specific* write revert_model
        # depends on fails.
        if Path(src).parent.name == modelhistory.VERSIONS_DIR:
            raise OSError("simulated disk-full failure")
        return real_copyfile(src, *args, **kwargs)

    # ``revert_model`` imports ``shutil`` locally (inside the function body,
    # not at module scope), so the module-level attribute is what every call
    # resolves against -- patch the real module rather than
    # ``_jobs_rework.shutil``, which does not exist until the function runs.
    monkeypatch.setattr(shutil, "copyfile", boom_copyfile)

    with pytest.raises(Failed):
        svc_jobs.revert_model(svc, job_id, version=restore_n)

    entries_after = svc.store.get(job_id)["params"]["model_history"]
    assert [e["n"] for e in entries_after] == [e["n"] for e in entries_before], (
        "a failed restore must not change the kept history at all"
    )
    assert modelhistory.version_path(job_dir, oldest_n).exists(), (
        "the oldest kept version must survive a failed restore even at the cap"
    )

    # And the retry the audit's "no longer available" complaint is about:
    # the version this failure never touched must still be restorable.
    monkeypatch.undo()
    result = svc_jobs.revert_model(svc, job_id, version=restore_n)
    assert result["ok"] is True


def test_a_finished_lora_train_job_is_not_reported_as_missing_a_mesh(svc):
    """2026-09-23 (second run) audit, finding service-02: ``lora_train`` rows
    are minted with no stage of their own (``service/loras.py``), so they
    keep ``JobStore.create``'s default stage, "model" -- and
    ``service.library.verify()`` maps that stage to ``model.glb``, a file no
    LoRA run ever writes. Every finished LoRA training job was therefore
    reported as missing its mesh, forever.
    """
    from realmspinner.pipelines import lora_train
    from realmspinner.service import library

    # No ``stage`` kwarg -- matches ``service.loras.train_lora``'s own
    # ``svc.store.create("lora_train", text, params, new_id)`` exactly, which
    # names no stage and so gets ``JobStore.create``'s default, "model".
    job_id = svc.store.create("lora_train", "a knight", {}, status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / lora_train.WEIGHTS_NAME).write_bytes(b"lora-bytes")

    report = library.verify(svc)
    ids = [p["id"] for p in report["missing_artifacts"]]
    assert job_id not in ids, (
        f"a finished lora_train job with its weights on disk must not be "
        f"reported as missing an artifact: {report['missing_artifacts']}"
    )


def test_a_legacy_lora_train_row_minted_with_stage_model_is_still_verified_correctly(svc):
    """Companion to the above: a LoRA row minted before this fix (stage
    "model", ``JobStore.create``'s default, because ``loras.py`` never named
    one of its own) must not start failing verification once ``verify()``
    keys primarily on ``kind`` rather than ``stage``.
    """
    from realmspinner.pipelines import lora_train
    from realmspinner.service import library

    job_id = svc.store.create(
        "lora_train", "a knight", {}, stage="model", status="done"
    )
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / lora_train.WEIGHTS_NAME).write_bytes(b"lora-bytes")

    report = library.verify(svc)
    ids = [p["id"] for p in report["missing_artifacts"]]
    assert job_id not in ids, (
        f"a legacy lora_train row (stage='model') must also not be reported "
        f"as missing model.glb: {report['missing_artifacts']}"
    )
