"""Installer first-run setup is a one-shot view over startup facts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock import doctor, fetch, models, vram
from warlock.service import downloads
from warlock.studio.panes import app_settings, first_run, model_gate
from warlock.studio.state import AppState


def _ctx(svc, *, checks=(), rows=(), plan=None, rigging=False, pack_rows=()):
    svc.vram_plan = plan
    return SimpleNamespace(
        svc=svc,
        runtime=SimpleNamespace(
            checks=list(checks),
            device_memory=vram.DeviceMemory(24.0, 20.0, "Test GPU"),
        ),
        state=AppState(),
        model_rows=list(rows),
        model_picks=set(),
        pack_rows=list(pack_rows),
        rigging_available=rigging,
        first_run=True,
        first_run_info={},
        gpu_name="",
        toast=lambda *_args: None,
    )


#: A pack row shaped like ``service.packs.rows``' own, standing in for one
#: read off a real ``ctx.pack_rows`` snapshot: a fresh install with the
#: Image generation pack (``packs.PACKS`` -- the one whose ``modes`` names
#: ``"create"``) not yet installed. The figure is illustrative, not measured
#: against the live wheel set -- ``tests/test_packs.py`` owns that number.
_TEXT2IMAGE_PACK_ROW = {
    "key": "text2image",
    "label": "Image generation",
    "modes": ["create"],
    "present": False,
    "download_gib": 3.3,
}


def test_a_marker_hides_the_overlay_on_the_next_start(svc, monkeypatch):
    ctx = _ctx(svc)
    assert first_run.pending(svc.config)
    monkeypatch.setattr(first_run.imgui, "close_current_popup", lambda: None)
    assert first_run.dismiss(ctx)
    assert first_run.marker_path(svc.config).is_file()
    assert not first_run.pending(svc.config)
    assert ctx.first_run is False


def test_the_snapshot_uses_startup_hardware_and_a_deduped_download_plan(
    svc, monkeypatch
):
    resolved = vram.plan(exclusive=False, total_gib=24.0)
    checks = (
        doctor.Check("CUDA", True, "available", fatal=False),
        doctor.Check("VRAM budget", True, resolved.reason, fatal=False),
    )
    rows = downloads.rows(svc)
    monkeypatch.setattr(fetch, "disk_refusal", lambda _jobs: None)
    ctx = _ctx(svc, checks=checks, rows=rows, plan=resolved, rigging=True)
    info = first_run.snapshot(ctx)
    expected = fetch.total_gib(
        fetch.plan(svc.config, [fetch.find(key) for key in first_run.GENERATION_ROWS])
    )
    assert info["gpu_name"] == "Test GPU"
    assert info["vram_total_gib"] == pytest.approx(24.0)
    assert info["three_d"]["ready"] is True
    assert info["images"]["ready"] is True
    assert info["rigging"]["ready"] is True
    assert info["total_gib"] == pytest.approx(expected)
    assert info["total_gib"] == pytest.approx(
        models.ENGINE_MODELS["trellis_gguf"].fetch[0].size_gib
        + models.ENGINE_MODELS["trellis_runtime"].fetch[0].size_gib
        + models.BASE_MODELS[models.DEFAULT_BASE_MODEL].fetch[0].size_gib
    )


def test_image_and_reconstruction_verdicts_have_separate_requirements(svc):
    plan = vram.plan(exclusive=False)
    ctx = _ctx(svc, checks=(), rows=downloads.rows(svc), plan=plan)
    info = first_run.snapshot(ctx)
    assert info["three_d"]["ready"] is False
    assert info["images"]["ready"] is True


def test_download_handoff_unions_picks_and_opens_settings_models(svc, monkeypatch):
    ctx = _ctx(svc)
    ctx.model_picks.add("lora:pixelxl")
    monkeypatch.setattr(first_run, "dismiss", lambda _ctx: True)
    first_run.download_models(ctx)
    assert ctx.model_picks == {"lora:pixelxl", *first_run.GENERATION_ROWS}
    assert ctx.state.mode == "settings"
    assert ctx.state.preview[app_settings.CATEGORY_SLOT] == "models"


# --- the pack half of "needed to generate" (2026-09-10) ---------------------
#
# Before this the panel knew only about weights: ``GENERATION_ROWS`` named two
# ``fetch`` rows and ``snapshot``/``draw`` had nothing to say about the 3.3 GB
# of Python those weights need in order to do anything. A user could tick
# every row here, download ~23 GB, and still have a greyed Create -- the F4
# defect, unfixed on this one surface.


def test_the_panel_now_counts_the_reconstruction_engines_own_binaries():
    """``engine:trellis_runtime`` stopped shipping in the installer on
    2026-09-10 and became a download the same way the GGUF weights already
    were -- so the panel's own list of what generation needs must grow with
    it, or the figure it shows undercounts what "Download models" actually
    fetches."""
    assert "engine:trellis_runtime" in first_run.GENERATION_ROWS


def test_snapshot_names_a_missing_pack_the_weights_alone_do_not_cover(svc, monkeypatch):
    """The other half of F4: a pack is not a ``fetch`` row, so it cannot live
    in ``GENERATION_ROWS`` -- it has to come from ``ctx.pack_rows`` through
    ``model_gate.missing_packs``, the same reader the rail and Settings use."""
    monkeypatch.setattr(fetch, "disk_refusal", lambda _jobs: None)
    ctx = _ctx(svc, pack_rows=[_TEXT2IMAGE_PACK_ROW])
    info = first_run.snapshot(ctx)
    assert info["packs"] == [
        {"key": "text2image", "label": "Image generation", "download_gib": 3.3}
    ]


def test_snapshot_says_nothing_about_a_pack_that_is_already_installed(svc, monkeypatch):
    monkeypatch.setattr(fetch, "disk_refusal", lambda _jobs: None)
    present = dict(_TEXT2IMAGE_PACK_ROW, present=True)
    ctx = _ctx(svc, pack_rows=[present])
    info = first_run.snapshot(ctx)
    assert info["packs"] == []


def test_a_missing_pack_routes_the_primary_button_to_packs_not_models(svc, monkeypatch):
    """F4's ordering, found again on this panel: a pack is the code and the
    weights are what the code reads, so weights installed without their pack
    buy the user nothing, and the panel must send them to the smaller,
    first-needed download rather than the ~23 GB of weights."""
    monkeypatch.setattr(fetch, "disk_refusal", lambda _jobs: None)
    ctx = _ctx(svc, pack_rows=[_TEXT2IMAGE_PACK_ROW])
    monkeypatch.setattr(first_run, "dismiss", lambda _ctx: True)
    packs = tuple(model_gate.missing_packs(ctx, "create"))
    first_run.install_packs(ctx, packs)
    assert ctx.state.mode == "settings"
    assert ctx.state.preview[app_settings.CATEGORY_SLOT] == "packs"
