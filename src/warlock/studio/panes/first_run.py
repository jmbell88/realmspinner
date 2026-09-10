"""Installer first-run hardware and model setup.

The overlay is deliberately a renderer over a startup snapshot.  Hardware,
disk and model presence were already measured while Runtime and Ctx were being
built; a modal drawn sixty times a second must not repeat any of those probes.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from ... import fetch, models, vram
from .. import controls, theme, tokens, widgets
from ..tokens import sp
from . import model_gate

MARKER = "first-run.json"
POPUP = "Welcome to Warlock"
#: What *generation* needs -- not what the app needs.
#:
#: Renamed from ``REQUIRED_ROWS`` on 2026-09-04. Nothing here is required to
#: run Warlock: the eight non-generative workspaces (Inker, Clay, Plotter,
#: Packwright, Sirens and the rest) open and work with none of it on disk. The
#: old name was read as "the app requires these", which is what put a red
#: banner and a blocking-looking panel in front of a healthy first launch.
#:
#: ``engine:trellis_runtime`` joined 2026-09-10, beside ``modes.NEEDS_ROWS``'s
#: own addition: it is a *weights* row (a ``fetch`` entry with a size), so it
#: belongs on this list. The *pack* half of the same defect -- the 3.3 GB of
#: Python this panel says nothing about -- is not a row at all, which is why
#: :func:`snapshot` reads ``model_gate.missing_packs`` separately rather than
#: adding a fourth thing here that ``fetch.find`` cannot resolve.
GENERATION_ROWS: tuple[str, ...] = (
    "engine:trellis_gguf", "engine:trellis_runtime", "base:sdxl_cfg",
)


def marker_path(config: Any) -> Path:
    return Path(config.home) / MARKER


def pending(config: Any) -> bool:
    """Sampled once by App.setup_context; never used as a per-frame stat."""
    return not marker_path(config).is_file()


def _check(checks: Any, name: str) -> Any:
    return next((row for row in checks or () if row.name == name), None)


def _settled(check: Any) -> bool:
    """``ok``, and not the provisional "still checking" answer."""
    return bool(check and check.ok and "still checking" not in str(check.detail))


def snapshot(ctx: Any) -> dict[str, Any]:
    """Everything the overlay draws, computed once at startup."""
    checks = getattr(ctx.runtime, "checks", None) or ()
    cuda = _check(checks, "CUDA")
    budget = _check(checks, "VRAM budget")
    plan = getattr(ctx.svc, "vram_plan", None)
    default = models.BASE_MODELS[models.DEFAULT_BASE_MODEL]
    image_fit = vram.fits(plan, default) if plan is not None else vram.FIT_OK
    # A row that is still checking (``doctor._cuda_check`` before torch has
    # imported) reports ``ok=True`` so the banner stays quiet; here it is
    # *not* ready, or the panel says "No CUDA GPU detected" and "Ready" on one
    # screen. The snapshot is retaken when the health poll lands.
    three_d_ready = bool(_settled(cuda) and _settled(budget))

    entries = [fetch.find(key) for key in GENERATION_ROWS]
    chosen = [entry for entry in entries if entry is not None]
    jobs = fetch.plan(ctx.svc.config, chosen)
    missing_jobs = fetch.plan(
        ctx.svc.config,
        [entry for entry in chosen if not entry.is_present(ctx.svc.config)],
    )
    by_key = {
        str(row.get("row_key")): row
        for row in (getattr(ctx, "model_rows", None) or ())
    }
    rows = [
        {
            "row_key": entry.row_key,
            "label": entry.label,
            "present": bool((by_key.get(entry.row_key) or {}).get("present")),
        }
        for entry in chosen
    ]
    device = getattr(ctx.runtime, "device_memory", None)
    total = getattr(plan, "total_gib", None)
    free = getattr(device, "free_gib", None)
    gpu_name = str(getattr(device, "name", "") or getattr(ctx, "gpu_name", "") or "")
    # The pack half of "needed to generate" (F4's defect, found again here
    # 2026-09-10): ``rows`` above is weights, all of them ``fetch`` entries
    # with a size on disk, and until now that was the whole panel. A pack is
    # code rather than weights -- read from ``ctx.pack_rows`` through
    # ``model_gate``'s own reader rather than the disk, for its reason: this
    # runs on the frame thread. Keyed on "create" because every row in
    # ``GENERATION_ROWS`` is what Create needs; Muse's own pack has no row
    # here to sit beside.
    packs = model_gate.missing_packs(ctx, "create")
    return {
        "gpu_name": gpu_name or ("CUDA GPU" if cuda and cuda.ok else "No CUDA GPU detected"),
        "vram_total_gib": total,
        "vram_free_gib": free,
        "three_d": {
            "ready": three_d_ready,
            "detail": (
                "Ready"
                if three_d_ready
                else f"Requires an NVIDIA GPU with about {vram.TRELLIS_GIB:.0f} GiB "
                "free; there is no CPU fallback."
            ),
        },
        "images": {
            # Image generation has its own CPU/offload paths; unlike TRELLIS,
            # its first-run verdict is the default checkpoint's fit result.
            "ready": image_fit != vram.FIT_NO,
            "detail": {
                vram.FIT_OK: "Ready",
                vram.FIT_TIGHT: "Ready (the image model and reconstruction run separately)",
                vram.FIT_NO: f"{default.label} does not fit this GPU's VRAM budget.",
            }[image_fit],
        },
        "rigging": {
            "ready": bool(getattr(ctx, "rigging_available", False)),
            "detail": (
                "Ready"
                if getattr(ctx, "rigging_available", False)
                else "Needs Blender (bpy)."
            ),
        },
        "rows": rows,
        "packs": [
            {
                "key": row.get("key"),
                "label": row.get("label"),
                "download_gib": row.get("download_gib"),
            }
            for row in packs
        ],
        "total_gib": fetch.total_gib(jobs),
        "download_gib": fetch.total_gib(missing_jobs),
        "disk_refusal": fetch.disk_refusal(missing_jobs),
    }


def dismiss(ctx: Any) -> bool:
    """Write the durable marker, then close. A failed write leaves the offer."""
    path = marker_path(ctx.svc.config)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp.write_text(
            json.dumps({"version": 1, "dismissed_at": time.time()}, indent=2),
            encoding="utf-8",
        )
        os.replace(temp, path)
    except OSError as exc:
        with suppress(OSError):
            temp.unlink(missing_ok=True)
        ctx.toast(f"Could not save first-run setup: {exc}", "error")
        return False
    ctx.first_run = False
    imgui.close_current_popup()
    return True


def download_models(ctx: Any) -> None:
    if dismiss(ctx):
        model_gate.request_install(ctx, GENERATION_ROWS)


def install_packs(ctx: Any, packs: tuple[dict[str, Any], ...]) -> None:
    """Packs first (F4's ordering, found again here): a pack is the code and
    the weights are what the code reads, so offering ~23 GB of weights before
    the 3.3 GB pack that makes them do anything is the expensive way to find
    that out. Reuses ``model_gate.request_pack`` -- this is the same door the
    rail and the palette already send a gated mode through, not a third one.
    """
    if dismiss(ctx):
        model_gate.request_pack(ctx, tuple(str(row["key"]) for row in packs))


def take_the_tour(ctx: Any) -> None:
    """Close the setup question, then offer the app itself.

    A separate exit rather than a checkbox on the other two, because it answers
    a different question: those two are about *this machine*, and this is about
    the reader. Someone who has just pressed Download has twenty-three
    gigabytes to wait through and nothing to do, which is the best moment the
    app will ever have to explain itself.
    """
    if dismiss(ctx):
        from . import tour as tour_pane

        tour_pane.start(ctx, "first-hour")


def is_open(ctx: Any) -> bool:
    return bool(getattr(ctx, "first_run", False))


def _verdict(label: str, result: dict[str, Any]) -> None:
    colour = theme.OK if result.get("ready") else theme.WARN
    widgets.text_colored(colour, f"{label}: {result.get('detail')}")


def draw(ctx: Any) -> None:
    if not is_open(ctx):
        return
    if not imgui.is_popup_open(POPUP):
        imgui.open_popup(POPUP)
    centre = imgui.get_main_viewport().get_center()
    imgui.set_next_window_pos(centre, imgui.Cond_.always.value, (0.5, 0.5))
    imgui.set_next_window_size((sp(620), 0), imgui.Cond_.always.value)
    opened, _ = imgui.begin_popup_modal(
        POPUP,
        None,
        imgui.WindowFlags_.always_auto_resize.value | imgui.WindowFlags_.no_move.value,
    )
    if not opened:
        return
    widgets.window_shadow("overlay")
    info = getattr(ctx, "first_run_info", None) or {}
    imgui.text_wrapped("Set up this PC")
    widgets.muted_wrapped(
        "Warlock runs locally. Nothing here is needed to start work: drawing, "
        "modelling, tile maps, atlases and the tracker all work right now, "
        "with nothing downloaded. Generating references, meshes and music "
        "needs the pieces below, and you can fetch them whenever you like -- "
        "here, or later from Settings -> Packs or Models."
    )

    imgui.dummy((0, sp(tokens.SP_2)))
    widgets.field_label("Hardware")
    gpu = info.get("gpu_name") or "GPU not identified"
    total = info.get("vram_total_gib")
    free = info.get("vram_free_gib")
    memory = "VRAM not measured"
    if total is not None:
        memory = f"{float(total):.1f} GiB VRAM"
        if free is not None:
            memory += f", {float(free):.1f} GiB free now"
    imgui.text_wrapped(f"{gpu} — {memory}")
    _verdict("3D reconstruction", info.get("three_d") or {})
    _verdict("Image generation", info.get("images") or {})
    _verdict("Rigging", info.get("rigging") or {})

    imgui.dummy((0, sp(tokens.SP_2)))
    # "Required downloads", and each row "required", until 2026-09-04. Nothing
    # here is required to run the app -- only to generate -- and the word was
    # the panel's main reason for reading as a toll rather than an offer.
    widgets.field_label("Needed to generate")
    packs = tuple(info.get("packs") or ())
    for row in packs:
        gib = row.get("download_gib")
        suffix = f" (~{float(gib):.1f} GB)" if gib else ""
        imgui.text_wrapped(f"{row.get('label')} pack — not installed{suffix}")
    for row in info.get("rows") or ():
        suffix = "installed" if row.get("present") else "not downloaded"
        imgui.text_wrapped(f"{row.get('label')} — {suffix}")
    refusal = info.get("disk_refusal")
    if refusal:
        widgets.text_colored(theme.ERR, str(refusal))

    download_gib = float(info.get("download_gib", info.get("total_gib")) or 0.0)
    # Packs before weights, in the button as well as in the list above: see
    # ``install_packs``. The figure is the packs' own -- the smaller number,
    # deliberately, because it is also the download that has to happen first.
    if packs:
        pack_gib = sum(float(row.get("download_gib") or 0.0) for row in packs)
        names = ", ".join(str(row.get("label") or row.get("key")) for row in packs)
        noun = "pack" if len(packs) == 1 else "packs"
        label = f"Install the {names} {noun} (~{pack_gib:.1f} GB)"
        if widgets.primary_button(label, (-1, sp(36))):
            install_packs(ctx, packs)
    else:
        label = f"Download models (~{download_gib:.0f} GB)"
        if widgets.primary_button(label, (-1, sp(36))):
            download_models(ctx)
    # Not a deferral of something owed: it dismisses the panel for good and
    # leaves a fully usable app. Home keeps a quiet row offering the same
    # download, so declining here loses nothing.
    if controls.button("Not now", (-1, 0), role=controls.ButtonRole.SECONDARY):
        dismiss(ctx)
    if controls.button("Show me around first", (-1, 0), role=controls.ButtonRole.GHOST):
        take_the_tour(ctx)
    imgui.end_popup()
