"""Regression tests for the 2026-10-03 audit's Medium service findings (service-08..15).

Each test's name is the claim, and each one fails against the code it was
written for.
"""

from __future__ import annotations

import contextlib
import json
import os
import struct
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from realmspinner.service.errors import Invalid, NotReady, ServiceError

# --- service-08: a 2D export derived while the reference is hand-edited ------------


def _draw(job_dir: Path, box) -> None:
    im = Image.new("RGB", (128, 128), (200, 200, 200))
    ImageDraw.Draw(im).rectangle(box, fill=(40, 40, 40))
    im.save(job_dir / "input.png")


def test_an_export_derived_while_the_reference_is_edited_is_not_served_as_fresh(
    svc, monkeypatch
):
    """service-08: the matte takes seconds; a hand edit landing inside it left an
    export whose mtime was newer than the *edited* input.png but whose pixels were
    the pre-edit ones, so ``fresh_2d`` served it until the next edit."""
    from realmspinner.pipelines import matting
    from realmspinner.service import derive as svc_derive
    from realmspinner.service import files as svc_files

    job_id = svc.store.create("text", "a barrel", {"seed": 1}, stage="reference", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    _draw(job_dir, (32, 24, 96, 104))
    source = job_dir / "input.png"
    old = source.stat().st_mtime_ns - 20_000_000_000
    os.utime(source, ns=(old, old))

    real_mask = matting.mask

    def mask_while_the_user_saves_an_edit(image, config):
        out = real_mask(image, config)
        # The edit lands after the pixels were read and before the export is
        # published: a real write, newer than the stamp the derive took.
        _draw(job_dir, (8, 8, 40, 40))
        edited = old + 5_000_000_000
        os.utime(source, ns=(edited, edited))
        return out

    monkeypatch.setattr(matting, "mask", mask_while_the_user_saves_an_edit)
    with contextlib.suppress(NotReady):
        svc_derive.get_file(svc, job_id, "icon.png")
    icon = job_dir / "icon.png"
    assert not (icon.exists() and svc_files.fresh_2d(job_dir, "icon.png")), (
        "an icon cut from the pre-edit pixels is being served as fresh"
    )


# --- service-09: a pose baked across a re-rig -------------------------------------

_BONES = ["hips", "spine", "head"]
_IDENTITY = [0.0, 0.0, 0.0, 1.0]


def _rigged_job(svc) -> str:
    from realmspinner.service import jobs as svc_jobs

    job_id = svc_jobs.create_job(svc, kind="text", prompt="a knight")["id"]
    job_dir = svc.config.data_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    (job_dir / "rig.json").write_text(
        json.dumps({"version": 1, "bones": [{"name": n} for n in _BONES]})
    )
    svc.store.set_status(job_id, "done")
    return job_id


def test_a_pose_baked_across_a_rerig_is_not_published(svc, monkeypatch):
    """service-09: the bake imported the old rig.glb, finalize_rig deleted the
    stale pose GLBs, and the bake then published onto ``poses/<id>.glb`` -- a pose
    from the previous skeleton, served for good because existence is its whole
    freshness test."""
    from realmspinner.kernels.rig import store
    from realmspinner.pipelines import blender_run
    from realmspinner.service import rig as svc_rig

    job_id = _rigged_job(svc)
    job_dir = svc.config.data_dir / job_id
    record = svc_rig.save_pose(
        svc, job_id, {"name": "idle", "bones": {b: _IDENTITY for b in _BONES}}
    )

    def run_worker(spec, **kwargs):
        Path(spec["out_glb"]).write_bytes(b"posed-from-the-old-skeleton")
        # The re-rig's finalize lands while Blender is still running.
        (job_dir / "rig.json").write_text(
            json.dumps({"version": 1, "bones": [{"name": n} for n in (*_BONES, "tail")]})
        )

    monkeypatch.setattr(blender_run, "run_worker", run_worker)
    with pytest.raises(ServiceError):
        svc_rig.posed_model(svc, job_id, record["id"])
    assert not store.pose_glb_path(job_dir, record["id"]).exists()
    pose_dir = store.pose_glb_path(job_dir, record["id"]).parent
    assert [p.name for p in pose_dir.glob(".*.glb")] == []


def test_finalize_rig_leaves_an_in_flight_pose_staging_file_alone(tmp_path):
    """service-09's second half: ``glob("*.glb")`` matched the dot-prefixed staging
    file a bake was still writing, so the bake's own rename failed with only
    "could not bake this pose"."""
    from realmspinner.kernels.rig import store

    job_dir = tmp_path / "job"
    poses = job_dir / store.POSE_DIR_NAME
    poses.mkdir(parents=True)
    (poses / "0123456789ab.glb").write_bytes(b"stale bake")
    staging = poses / ".0123456789ab.tmp.glb"
    staging.write_bytes(b"half written")
    (job_dir / store.RIG_GLB_TMP).write_bytes(b"new rig")
    (job_dir / store.RIG_JSON_TMP).write_text(json.dumps({"version": 1, "bones": []}))

    store.finalize_rig(job_dir)

    assert not (poses / "0123456789ab.glb").exists()
    assert staging.exists()


# --- service-10: importing a style LoRA --------------------------------------------


def _archive(*, header=None, data_len=4, truncate=0) -> bytes:
    """A minimal safetensors file: u64 header length, JSON header, tensor bytes."""
    if header is None:
        header = {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, data_len]}}
    blob = json.dumps(header).encode("utf-8")
    body = struct.pack("<Q", len(blob)) + blob + b"\x00" * data_len
    return body[: len(body) - truncate] if truncate else body


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\x00" * 64,
        b"not a safetensors archive at all, just text",
        _archive(truncate=2),
        struct.pack("<Q", 1 << 40) + b"{}",
    ],
    ids=["empty", "zeros", "text", "truncated tensor data", "header longer than file"],
)
def test_import_lora_refuses_a_file_that_is_not_a_safetensors_archive(svc, tmp_path, content):
    from realmspinner import models
    from realmspinner.service import loras as svc_loras

    before = dict(models.STYLE_LORAS)
    path = tmp_path / "style.safetensors"
    path.write_bytes(content)
    try:
        with pytest.raises(Invalid) as info:
            svc_loras.import_lora(svc, path, label="Cosmos")
        assert info.value.field == "source"
        assert svc_loras.imported(svc) == []
    finally:
        models.STYLE_LORAS.clear()
        models.STYLE_LORAS.update(before)


def test_import_lora_still_accepts_a_whole_safetensors_archive(svc, tmp_path):
    from realmspinner import models
    from realmspinner.service import loras as svc_loras

    before = dict(models.STYLE_LORAS)
    path = tmp_path / "style.safetensors"
    path.write_bytes(_archive())
    try:
        out = svc_loras.import_lora(svc, path, label="Cosmos")
        assert out["key"] in models.STYLE_LORAS
    finally:
        models.STYLE_LORAS.clear()
        models.STYLE_LORAS.update(before)


# --- service-11: environment values the per-job doors range-check ------------------


@pytest.mark.parametrize(
    "name,raw,field,default",
    [
        ("REALMSPINNER_MESH_PROFILE", "standrad", "mesh_profile", "standard"),
        ("REALMSPINNER_MESH_PROFILE", "custom", "mesh_profile", "standard"),
        ("REALMSPINNER_LOWPOLY_TRIANGLES", "-5", "lowpoly_triangles", 5000),
        ("REALMSPINNER_LOWPOLY_TRIANGLES", "999999999", "lowpoly_triangles", 5000),
        ("REALMSPINNER_TRELLIS_BAND", "0", "trellis_band", None),
        ("REALMSPINNER_TRELLIS_TEX_RES", "7", "trellis_tex_res", 512),
        ("REALMSPINNER_TRELLIS_MAX_TOKENS", "-1", "trellis_max_tokens", None),
        ("REALMSPINNER_TRELLIS_DECIM", "-3", "trellis_decim", None),
        ("REALMSPINNER_TRELLIS_ATLAS", "10", "trellis_atlas", None),
        ("REALMSPINNER_TRELLIS_GSS", "0", "trellis_gss", None),
        ("REALMSPINNER_TRELLIS_GSH", "-2", "trellis_gsh", None),
    ],
)
def test_a_malformed_mesh_profile_or_lowpoly_budget_in_the_environment_is_recorded_and_defaulted(
    monkeypatch, name, raw, field, default
):
    """service-11: these reached ``Config`` unchecked, so one typo refused every
    default-profile submit with an error addressed to a control the user never
    touched, while ``INVALID_ENV`` (and so Doctor) said all was well."""
    import realmspinner.config as config_mod

    monkeypatch.setattr(config_mod, "INVALID_ENV", [])
    monkeypatch.setenv(name, raw)
    cfg = config_mod.Config()
    got = getattr(cfg, field)
    if field == "trellis_band":
        default = config_mod.DEFAULT_TRELLIS_BAND
    assert got == default
    assert any(entry[0] == name for entry in config_mod.INVALID_ENV)


def test_the_environment_bounds_agree_with_the_per_job_doors(monkeypatch):
    """The config layer cannot import ``service.validation``, so it carries its own
    copy of each bound; this is what keeps the two from drifting."""
    import realmspinner.config as config_mod
    from realmspinner.pipelines import optimize, remesh
    from realmspinner.service import validation

    monkeypatch.setattr(config_mod, "INVALID_ENV", [])
    for name, value in (
        ("REALMSPINNER_TRELLIS_BAND", validation.MIN_TRELLIS_BAND),
        ("REALMSPINNER_TRELLIS_BAND", validation.MAX_TRELLIS_BAND),
        ("REALMSPINNER_TRELLIS_TEX_RES", validation.MIN_TRELLIS_TEX_RES),
        ("REALMSPINNER_TRELLIS_TEX_RES", validation.MAX_TRELLIS_TEX_RES),
        ("REALMSPINNER_TRELLIS_MAX_TOKENS", 1),
        ("REALMSPINNER_TRELLIS_MAX_TOKENS", validation.MAX_TRELLIS_MAX_TOKENS),
        ("REALMSPINNER_TRELLIS_DECIM", 0),
        ("REALMSPINNER_TRELLIS_DECIM", validation.MAX_TRELLIS_DECIM),
        ("REALMSPINNER_TRELLIS_ATLAS", validation.MIN_TRELLIS_ATLAS),
        ("REALMSPINNER_TRELLIS_ATLAS", validation.MAX_TRELLIS_ATLAS),
        ("REALMSPINNER_LOWPOLY_TRIANGLES", 0),
        ("REALMSPINNER_LOWPOLY_TRIANGLES", remesh.TRIANGLES_MIN),
        ("REALMSPINNER_LOWPOLY_TRIANGLES", remesh.TRIANGLES_MAX),
    ):
        monkeypatch.setenv(name, str(value))
        config_mod.Config()
        assert config_mod.INVALID_ENV == [], (name, value)
    for profile in optimize.PROFILES:
        monkeypatch.setenv("REALMSPINNER_MESH_PROFILE", profile)
        assert config_mod.Config().mesh_profile == profile
    assert config_mod.INVALID_ENV == []


# --- service-12: a mesh verdict against a job that never made a mesh ---------------


@pytest.mark.parametrize("stage", ["reference", "tile"])
def test_a_mesh_verdict_refuses_a_job_that_never_made_a_mesh(svc, stage):
    """service-12: only the inspector's own ``job.stage == "model"`` check kept a
    +5 off a finished picture, and the vector snapshot outlives the job."""
    from realmspinner.service import verdicts as svc_verdicts

    job_id = svc.store.create("text", "a chest", {}, stage=stage, status="done")
    with pytest.raises(Invalid):
        svc_verdicts.record_verdict(svc, job_id, grade=5)
    assert svc.store.latest_verdicts() == []


# --- service-13: cancel loses the race to the worker's claim -----------------------


class _RecordingWorker:
    def __init__(self) -> None:
        self.current_job_id = None
        self.cancelled: list[str] = []

    async def request_cancel(self, job_id: str) -> bool:
        self.cancelled.append(job_id)
        return True


def test_cancel_signals_the_worker_when_the_job_is_claimed_between_read_and_write(svc):
    """service-13: the status was read as ``queued``, the worker's claim landed,
    and ``store.cancel`` then wrote ``cancelled`` over a running job without ever
    sending ``request_cancel`` -- minutes of GPU the user could not stop."""
    from realmspinner.service import jobs as svc_jobs

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    worker = _RecordingWorker()
    svc.worker = worker
    svc.loop = None
    real_cancel = svc.store.cancel

    def cancel_after_the_claim(jid):
        assert svc.store.claim(jid)  # the worker wins the race for the row
        worker.current_job_id = jid
        return real_cancel(jid)

    svc.store.cancel = cancel_after_the_claim
    try:
        svc_jobs.cancel_job(svc, job_id)
    finally:
        svc.worker = None
    assert svc.store.get(job_id)["status"] == "cancelled"
    assert worker.cancelled == [job_id]


def test_pressing_cancel_again_signals_a_worker_still_running_the_cancelled_row(svc):
    """service-13's second half: the idempotent branch answered ``ok`` without
    signalling, so a second press could never stop the job the first one missed."""
    from realmspinner.service import jobs as svc_jobs

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.claim(job_id)
    svc.store.cancel(job_id)  # the row already reads cancelled...
    worker = _RecordingWorker()
    worker.current_job_id = job_id  # ...but the worker is still on it
    svc.worker = worker
    svc.loop = None
    try:
        assert svc_jobs.cancel_job(svc, job_id) == {"ok": True}
    finally:
        svc.worker = None
    assert worker.cancelled == [job_id]


# --- service-14: promoting a reference that was never measured ---------------------


def _blank_reference(svc) -> str:
    """A finished reference drawn as an ordinary 2D image: no ``reference_report``,
    and nothing in the frame to reconstruct."""
    job_id = svc.store.create("text", "a field", {"seed": 1}, stage="reference", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (230, 230, 230)).save(job_dir / "input.png")
    assert "reference_report" not in svc.store.get(job_id)["params"]
    return job_id


def test_force_promotes_an_unmeasured_reference_through_the_workers_composition_gate(svc):
    """service-14: the cutout modal measured the reference on demand and offered
    Build anyway, but the promote door only read a *stored* report, so it granted
    no override and the worker's ``reference.prepare`` refused the queued job."""
    from realmspinner.pipelines import reference
    from realmspinner.service import jobs as svc_jobs

    job_id = _blank_reference(svc)
    with pytest.raises(Invalid):
        svc_jobs.promote_to_model(svc, job_id)

    child = svc_jobs.promote_to_model(svc, job_id, force=True)["id"]
    params = svc.store.get(child)["params"]
    assert reference.OVERRIDE_KEY in params
    assert "empty" in params[reference.OVERRIDE_KEY]["codes"]


# --- service-15: crediting a resident trellis --------------------------------------


def _worker(tmp_path, **overrides):
    from realmspinner.config import Config
    from realmspinner.db import JobStore
    from realmspinner.queue import Worker

    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        **overrides,
    )
    return Worker(config, JobStore(config.db_path))


@pytest.mark.parametrize(
    "stage,resolution",
    [("model", 1024), ("reference", 1024), ("model", 512)],
    ids=["text model 1024", "text reference stage", "text model 512"],
)
def test_check_resources_refuses_a_smaller_job_beside_a_trellis_primed_at_a_larger_resolution(
    tmp_path, fake_pipelines, monkeypatch, stage, resolution
):
    """service-15: the credit used the RESIDENT resolution (24 GiB after one 1536
    job) while the estimate charged this job's own price, so a smaller job was
    admitted on 8-10 GiB of headroom that was never free."""
    import realmspinner.queue as queue_mod
    from realmspinner import vram

    monkeypatch.setattr(queue_mod, "commit_fraction", lambda: None)
    monkeypatch.setattr(vram, "device_memory", lambda: vram.DeviceMemory(32.0, 2.0))
    worker = _worker(tmp_path)
    try:
        worker.trellis.running = True
        worker._trellis_resolution = 1536
        job = {"kind": "text", "stage": stage, "params": {"resolution": resolution}}
        with pytest.raises(RuntimeError, match="GiB of VRAM"):
            worker._check_resources(job)
    finally:
        worker.store.close()


def test_check_resources_still_credits_the_resident_trellis_when_a_handoff_will_stop_it(
    tmp_path, fake_pipelines, monkeypatch
):
    """The other direction: under exclusive the handoff stops the server before
    anything loads, so what it frees is the resident figure, whatever this job's
    own price is."""
    import realmspinner.queue as queue_mod
    from realmspinner import vram

    monkeypatch.setattr(queue_mod, "commit_fraction", lambda: None)
    monkeypatch.setattr(vram, "device_memory", lambda: vram.DeviceMemory(32.0, 2.0))
    worker = _worker(tmp_path, vram_exclusive=True)
    try:
        worker.trellis.running = True
        worker._trellis_resolution = 1536
        job = {"kind": "text", "stage": "reference", "params": {"resolution": 1024}}
        worker._check_resources(job)  # need 7 GiB; 2 + 24 freed by the stop
    finally:
        worker.store.close()


def test_the_trellis_charge_is_what_the_coexist_estimate_adds_for_trellis():
    """``vram.trellis_charge`` is derived beside ``estimate_parts`` and must not
    drift from it: coexist minus the non-trellis part is the charge."""
    from realmspinner import vram

    for kind in ("retexture", "pixel_sheet", "sprite_synthesis", "tile_sheet", "music"):
        params = {"base_model": "sdxl_cfg"}
        coexist = vram.estimate(kind, "model", params, exclusive=False)
        alone = vram.estimate(kind, "model", params, exclusive=True)
        assert coexist - alone == pytest.approx(vram.trellis_charge(kind, "model", params))
    for resolution in (512, 1024, 1536):
        params = {"resolution": resolution}
        sdxl = vram.estimate("text", "reference", params, exclusive=True)
        total = vram.estimate("text", "model", params, exclusive=False)
        assert total - sdxl == pytest.approx(vram.trellis_charge("text", "model", params))
        assert vram.trellis_charge("text", "reference", params) == vram.TRELLIS_GIB
    assert vram.trellis_charge("rig", "model", {}) == 0.0
