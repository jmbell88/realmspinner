"""Installer first-run setup is a one-shot view over startup facts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from realmspinner import config as config_module
from realmspinner import doctor, fetch, models, vram
from realmspinner.service import downloads
from realmspinner.studio.modes.settings.ui.panes import app_settings
from realmspinner.studio.panes import first_run, model_gate
from realmspinner.studio.state import AppState


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
        + models.BASE_MODELS[config_module.DEFAULT_BASE_MODEL].fetch[0].size_gib
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


# --- the primary button's label and action (2026-09-26 audit) ---------------


def test_a_pc_with_nothing_missing_offers_continue_not_a_zero_byte_download():
    """The 2026-09-26 audit, finding shell-review-settings-07: every row
    already present and no pack missing used to fall through to "Download
    models (~0 GB)" -- a button that promised a download and would have
    started one for nothing."""
    info = {
        "packs": [],
        "rows": [
            {"label": "TRELLIS.2 GGUF weights", "present": True},
            {"label": "SDXL 1.0", "present": True},
        ],
        "download_gib": 0.0,
        "total_gib": 23.1,
    }
    assert first_run._primary_action(info) == ("Continue", "continue")


def test_a_pc_with_a_missing_row_still_offers_the_real_download():
    """The fix must not swallow the ordinary case: a row still missing keeps
    the download offer, sized off ``download_gib``."""
    info = {
        "packs": [],
        "rows": [
            {"label": "TRELLIS.2 GGUF weights", "present": False},
            {"label": "SDXL 1.0", "present": True},
        ],
        "download_gib": 12.3,
        "total_gib": 23.1,
    }
    label, action = first_run._primary_action(info)
    assert action == "models"
    assert label == "Download models (~12 GB)"


def test_a_missing_pack_still_wins_over_a_fully_present_row_set():
    """Packs before weights (F4's ordering) still applies even when every
    weight row happens to already be present."""
    info = {
        "packs": [{"key": "text2image", "label": "Image generation", "download_gib": 3.3}],
        "rows": [{"label": "SDXL 1.0", "present": True}],
        "download_gib": 0.0,
        "total_gib": 23.1,
    }
    label, action = first_run._primary_action(info)
    assert action == "packs"
    assert "Image generation" in label


def test_continue_dismisses_the_panel_rather_than_starting_a_download(svc, monkeypatch):
    """The button press itself: with nothing missing, pressing it must close
    the panel (``dismiss``), not reach for ``download_models``."""
    ctx = _ctx(svc)
    ctx.first_run_info = {
        "packs": [],
        "rows": [{"label": "SDXL 1.0", "present": True}],
        "download_gib": 0.0,
        "total_gib": 0.0,
    }
    calls: list[str] = []
    monkeypatch.setattr(first_run, "dismiss", lambda _ctx: calls.append("dismiss") or True)
    monkeypatch.setattr(
        first_run, "download_models", lambda _ctx: calls.append("download_models")
    )
    label, action = first_run._primary_action(ctx.first_run_info)
    assert label == "Continue"
    # The dispatch ``draw`` performs on a press, exercised directly since
    # ``draw`` itself needs an imgui frame.
    if action == "packs":
        first_run.install_packs(ctx, tuple(ctx.first_run_info["packs"]))
    elif action == "models":
        first_run.download_models(ctx)
    else:
        first_run.dismiss(ctx)
    assert calls == ["dismiss"]
