"""Regression tests for the 2026-10-03 audit's High service findings (service-01..07)."""

from __future__ import annotations

import pytest


def test_to_png_applies_the_exif_orientation_before_dropping_it():
    """The 2026-10-03 audit (service-02): a portrait phone photo (EXIF
    orientation 6) was re-encoded sideways, the tag dropped."""
    import io

    from PIL import Image

    from realmspinner.service import files

    src = Image.new("RGB", (40, 20), (255, 0, 0))  # stored landscape
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW to display: shown 20 wide, 40 tall
    buf = io.BytesIO()
    src.save(buf, "JPEG", exif=exif)

    out = Image.open(io.BytesIO(files.to_png(buf.getvalue())))
    assert out.size == (20, 40)


def _text_job_with_files(svc, files, **params):
    from realmspinner.service import jobs as svc_jobs

    src = svc_jobs.create_job(svc, kind="text", prompt="a barrel", seed=42)["id"]
    svc.store.merge_params(src, params)
    job_dir = svc.job_dir(src)
    job_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        (job_dir / name).write_bytes(b"png-" + name.encode())
    return src


def test_reroll_carries_the_native_reference_files_it_names(svc):
    """service-04: the key was copied, the files were not."""
    from realmspinner.service import jobs as svc_jobs

    names = ["native_reference_0.png", "native_reference_1.png"]
    src = _text_job_with_files(svc, names, native_reference_files=names)
    new_id = svc_jobs.rerun_job(svc, src, mode="reroll")["id"]
    new_dir = svc.job_dir(new_id)
    for name in names:
        assert (new_dir / name).read_bytes() == b"png-" + name.encode()


def test_reroll_never_copies_a_native_reference_name_that_is_a_path(svc, tmp_path):
    from realmspinner.service import jobs as svc_jobs

    (tmp_path / "secret.png").write_bytes(b"secret")
    src = _text_job_with_files(
        svc, [], native_reference_files=["../../secret.png", "ref.png"]
    )
    new_id = svc_jobs.rerun_job(svc, src, mode="reroll")["id"]
    assert not svc.job_dir(new_id).exists() or not any(svc.job_dir(new_id).glob("*secret*"))


def test_reroll_of_a_masked_job_keeps_its_mask(svc):
    """service-05: a masked img2img was rerolled as a whole-picture img2img."""
    from realmspinner.service import jobs as svc_jobs

    src = _text_job_with_files(svc, ["ref.png", "mask.png"], init_image=True)
    new_id = svc_jobs.rerun_job(svc, src, mode="reroll")["id"]
    assert (svc.job_dir(new_id) / "mask.png").read_bytes() == b"png-mask.png"
    assert (svc.job_dir(new_id) / "ref.png").exists()


def _sheet_and_animated_job(svc):
    import json

    from test_character_exports import (
        _build_sheet,
        _fake_glb,
        _two_movement_layout,
    )

    from realmspinner import clips
    from realmspinner.service import derive as svc_derive

    job_id, sheet_id, _colors = _build_sheet(
        svc, _two_movement_layout(), with_troupe_block=True, name="Knight"
    )
    job_dir = svc.job_dir(job_id)
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    (job_dir / "rig.json").write_text(json.dumps({"template": "humanoid"}), "utf-8")
    glb = _fake_glb(
        ["idle", "walk"],
        loops=["idle", "walk"],
        digest=clips.library_digest("humanoid"),
        rig_digest=svc_derive._rig_digest(job_dir),
    )
    (job_dir / "animated.glb").write_bytes(glb)
    return job_id, sheet_id


def test_a_frames_export_and_a_godot_export_of_one_character_do_not_replace_each_other(
    svc, tmp_path
):
    from realmspinner.service import characters as svc_characters

    job_id, sheet_id = _sheet_and_animated_job(svc)
    out = tmp_path / "out"
    frames = svc_characters.export_frames(svc, job_id, sheet_id, dest_dir=out)
    godot = svc_characters.export_godot(svc, job_id, dest_dir=out)
    assert frames != godot
    assert frames.is_dir() and (frames / "manifest.json").exists()
    assert godot.is_dir() and list(godot.glob("*.tscn"))
    # and a re-export replaces its own folder only when told to
    assert (
        svc_characters.export_frames(svc, job_id, sheet_id, dest_dir=out, overwrite=True)
        == frames
    )


def test_an_export_never_replaces_a_folder_it_did_not_write(svc, tmp_path):
    from realmspinner.service import characters as svc_characters
    from realmspinner.service.errors import Conflict

    job_id, sheet_id = _sheet_and_animated_job(svc)
    out = tmp_path / "out"
    mine = out / "Knight-godot"
    mine.mkdir(parents=True)
    (mine / "hand_attached.gd").write_text("extends Node")
    # By default the folder is simply left alone and the export takes a new name.
    fresh = svc_characters.export_godot(svc, job_id, dest_dir=out)
    assert fresh != mine and (mine / "hand_attached.gd").exists()
    # Even an explicit overwrite refuses a folder this app did not write.
    with pytest.raises(Conflict):
        svc_characters.export_godot(svc, job_id, dest_dir=out, overwrite=True)
    assert (mine / "hand_attached.gd").exists()


def test_a_cancelled_model_job_cannot_be_retargeted_back_into_existence(svc, monkeypatch):
    """service-03: a cancelled finishing job keeps source.glb + model.glb, and
    every rework door accepted the row and published over it."""
    from realmspinner.service import jobs as svc_jobs
    from realmspinner.service.errors import Conflict

    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "source.glb").write_bytes(b"source")
    (job_dir / "model.glb").write_bytes(b"model")
    svc.store.set_status(job_id, "cancelled")
    with pytest.raises(Conflict):
        svc_jobs.optimize_job(svc, job_id, profile="raw")
    with pytest.raises(Conflict):
        svc_jobs.remesh_job(svc, job_id)
    with pytest.raises(Conflict):
        svc_jobs.retexture_job(svc, job_id, prompt="mossy")
    assert (job_dir / "model.glb").read_bytes() == b"model"


def test_a_cancel_after_the_stage_committed_still_records_done_and_keeps_the_served_artifact(
    svc,
):
    """service-06: cancel_job flipped a running row to cancelled even after the
    worker's stage had committed its cancel token, so finish(done) returned False
    and _discard_artifacts deleted the served sheet."""
    from realmspinner.queue import Worker, _Cancel
    from realmspinner.service import jobs as svc_jobs

    job_id = svc.store.create("charsheet", None, {}, stage="sheet")
    svc.store.claim(job_id)

    class _Committed:
        """The real request_cancel, over a worker whose running stage committed."""

        def __init__(self) -> None:
            self.worker = object.__new__(Worker)
            self.worker.current_job_id = job_id
            self.worker._cancel = _Cancel(job_id)
            self.worker._cancel.commit()
            self.worker.progress = type("P", (), {"snapshot": lambda s: None})()
            self.worker._blender = None

    stub = _Committed()
    svc.worker = stub.worker
    svc.loop = None
    try:
        result = svc_jobs.cancel_job(svc, job_id)
    finally:
        svc.worker = None
    assert result["ok"] is True
    assert svc.store.get(job_id)["status"] == "running"
    # ...so the worker's own terminal write lands.
    assert svc.store.finish(job_id, "done") is True
    assert svc.store.get(job_id)["status"] == "done"
    assert stub.worker._cancel.event.is_set()


async def test_a_store_whose_writes_fail_stops_the_worker_and_says_so(tmp_path):
    """service-07: the failure counter was reset as soon as next_queued (a read)
    succeeded, so a claim that kept raising never reached LOOP_FAILURE_LIMIT."""
    import asyncio

    from realmspinner import queue as queue_mod
    from realmspinner.config import Config
    from realmspinner.db import JobStore
    from realmspinner.queue import Worker

    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )
    worker = Worker(config, JobStore(config.db_path))
    worker.store.create("image", None, {"seed": 1})

    def full(job_id):
        raise RuntimeError("database or disk is full")

    worker.store.claim = full
    old = queue_mod.POLL_INTERVAL
    queue_mod.POLL_INTERVAL = 0.001
    try:
        worker.start()
        for _ in range(500):
            if worker.fatal is not None:
                break
            await asyncio.sleep(0.01)
    finally:
        queue_mod.POLL_INTERVAL = old
    assert worker.fatal is not None and "full" in str(worker.fatal)
    await worker.shutdown()
