"""The 2026-10-03 audit's Low findings owned by fixer ``service2``.

service-27: ``_pixel_sheet``'s per-band decode/crop/remask/paste ran on
``realmspinner-loop``. service-29: a failed seam measurement was recorded as a
perfect one. service-32: a locked/readonly store was offered the "damaged
file" reset. service-33: ``_lora_train`` was missing from ``PUBLISHERS``.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time

import pytest
from PIL import Image

from realmspinner.config import Config
from realmspinner.db import JobStore, StoreUnreadable
from realmspinner.kernels.rig import store as rig_store
from realmspinner.queue import Worker


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    yield w
    store.close()


async def _run(worker, job_id):
    worker.start()
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if worker.store.get(job_id)["status"] in ("done", "error", "cancelled"):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("job did not finish before timeout")
    finally:
        await worker.shutdown()
    return worker.store.get(job_id)


# --- service-27 --------------------------------------------------------------


def _rendered_sheet(worker, source, *, frame_size=128, columns=8, rows=1):
    source_dir = worker.config.job_dir(source)
    sheet_id = rig_store.new_id()
    png = rig_store.sheet_png_path(source_dir, sheet_id)
    png.parent.mkdir(parents=True, exist_ok=True)
    atlas = Image.new("RGBA", (frame_size * columns, frame_size * rows), (0, 0, 0, 0))
    for row in range(rows):
        for column in range(columns):
            block = Image.new("RGBA", (frame_size // 2,) * 2, (200, 60, 60, 255))
            atlas.paste(
                block,
                (column * frame_size + frame_size // 4, row * frame_size + frame_size // 4),
            )
    atlas.save(png)
    meta = {
        "version": 1, "id": sheet_id, "name": "turnaround", "source_job": source,
        "created": 1.0, "image": png.name, "frame_size": frame_size,
        "columns": columns, "rows": rows,
        "width": frame_size * columns, "height": frame_size * rows,
        "elevation": 30.0, "lighting": "flat",
        "yaws": [i * 360.0 / columns for i in range(columns)],
        "poses": [{"id": None, "name": "rest"}],
        "cells": [],
    }
    rig_store.sheet_path(source_dir, sheet_id).write_text(json.dumps(meta), encoding="utf-8")
    return sheet_id


@pytest.mark.asyncio
async def test_pixel_sheet_band_tail_runs_off_the_loop_thread(worker, monkeypatch):
    """The band tail (decode the generation, crop it back, remask against the
    atlas, paste) ran inline on ``realmspinner-loop`` -- the thread every job's
    progress and cancel is served from -- once per band, up to eight times.
    Spies on the work itself, not on ``to_thread`` (which the unfixed code
    already uses for the band's *other* steps)."""
    import PIL.Image as pil_image

    from realmspinner.pipelines import pixelsheet

    loop_thread = threading.get_ident()
    seen: dict[str, set[int]] = {"open": set(), "crop_back": set(), "remask": set()}

    real_open = pil_image.open

    def spying_open(*args, **kwargs):
        seen["open"].add(threading.get_ident())
        return real_open(*args, **kwargs)

    real_crop = pixelsheet.crop_back
    real_remask = pixelsheet.remask

    def spying_crop(*args, **kwargs):
        seen["crop_back"].add(threading.get_ident())
        return real_crop(*args, **kwargs)

    def spying_remask(*args, **kwargs):
        seen["remask"].add(threading.get_ident())
        return real_remask(*args, **kwargs)

    monkeypatch.setattr(pil_image, "open", spying_open)
    monkeypatch.setattr(pixelsheet, "crop_back", spying_crop)
    monkeypatch.setattr(pixelsheet, "remask", spying_remask)

    source = worker.store.create("text", "a knight", {"seed": 1})
    job_dir = worker.config.job_dir(source)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    worker.store.set_status(source, "done")
    sheet_id = _rendered_sheet(worker, source)
    job_id = worker.store.create(
        "pixel_sheet", "a knight",
        {"source_job": source, "sheet_id": sheet_id, "logical_size": 32,
         "colors": 8, "seed": 3},
    )

    row = await _run(worker, job_id)

    assert row["status"] == "done", row.get("error")
    for name, threads in seen.items():
        assert threads, f"{name} was never observed"
        assert loop_thread not in threads, f"{name} ran on the loop thread"


# --- service-29 --------------------------------------------------------------


def _materials_job(worker, *, tile_w=32, prompts=("moss", "gravel"), seed=100) -> str:
    from realmspinner.pipelines import tileatlas

    seeds = tileatlas.material_seeds(seed, len(prompts))
    geom = tileatlas.material_geometry(tile_w, "top_down", len(prompts))
    params = {
        "seed": seed,
        "base_model": "sdxl_cfg",
        "colors": 64,
        "negative_prompt": "",
        "sheet": {
            "version": 3, "mode": "materials",
            "tile_w": geom.tile_w, "tile_h": geom.tile_h, "projection": "top_down",
            "columns": geom.columns, "rows": geom.rows, "layout": "grid",
            "materials": [
                {"index": i, "prompt": p, "variant": 1, "seed": s}
                for i, (p, s) in enumerate(zip(prompts, seeds, strict=True))
            ],
            "terrains": [], "mask": None, "boundary": "", "variants": 1,
            "style_lock": False,
        },
    }
    return worker.store.create("tile_sheet", "a dungeon", params, stage="tilesheet")


@pytest.mark.asyncio
async def test_a_failed_seam_measurement_is_not_recorded_as_a_perfect_one(
    worker, monkeypatch
):
    """``seam.report`` raising used to append ``{}``, which the recipe read back
    as ``worst: 0.0, seamless: False`` -- and ``seam_worst`` as 0.0, the best
    possible score, for a set whose only measurements never ran."""
    from realmspinner.pipelines import seam

    calls = {"n": 0}
    real_report = seam.report

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic seam failure")
        return real_report(path)

    monkeypatch.setattr(seam, "report", flaky)

    row = await _run(worker, _materials_job(worker))

    assert row["status"] == "done", row.get("error")
    report = row["params"]["sheet_report"]
    failed, measured = report["seams"]
    assert failed["worst"] is None and failed["seamless"] is None
    assert measured["worst"] is not None
    # The set's number is the measured material's, not the failed one's zero.
    assert report["seam_worst"] == measured["worst"]


@pytest.mark.asyncio
async def test_a_set_with_no_measured_seam_has_no_seam_worst(worker, monkeypatch):
    from realmspinner.pipelines import seam

    def broken(path):
        raise RuntimeError("synthetic seam failure")

    monkeypatch.setattr(seam, "report", broken)

    row = await _run(worker, _materials_job(worker))

    assert row["status"] == "done", row.get("error")
    report = row["params"]["sheet_report"]
    assert report["seam_worst"] is None
    assert all(e["worst"] is None and e["seamless"] is None for e in report["seams"])


# --- service-32 --------------------------------------------------------------


def test_a_locked_or_readonly_store_is_not_offered_a_reset(tmp_path, monkeypatch):
    """``OperationalError`` ("database is locked", "disk is full", "readonly
    database") is a ``DatabaseError``, so a healthy-but-busy library was wrapped
    in ``StoreUnreadable`` and offered the rename-the-index reset. Only the
    malformed / not-a-database classes are that; a locked file surfaces as
    itself."""
    path = tmp_path / "jobs.sqlite"
    JobStore(path).close()

    # A real lock: another connection holds the database exclusively.
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("PRAGMA locking_mode=EXCLUSIVE")
    holder.execute("BEGIN EXCLUSIVE")
    real_connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **kw: real_connect(*a, **{**kw, "timeout": 0.05})
    )
    try:
        with pytest.raises(sqlite3.OperationalError) as caught:
            JobStore(path)
    finally:
        holder.close()
    assert not isinstance(caught.value, StoreUnreadable)
    assert "locked" in str(caught.value)


def test_a_malformed_store_is_still_offered_a_reset(tmp_path):
    """The other half: narrowing must not stop the genuine corruption case."""
    path = tmp_path / "jobs.sqlite"
    path.write_bytes(b"SQLite format 3\x00" + b"\xff" * 4096)
    with pytest.raises(StoreUnreadable):
        JobStore(path)


# --- service-33 --------------------------------------------------------------


def test_every_served_publish_commits_the_cancel_token_for_lora_train():
    """``_lora_train`` registers an adapter onto the shared model store through
    ``generation.import_lora`` and commits the cancel token, but was in no row
    of ``PUBLISHERS``, so deleting its commit failed nothing."""
    from tests.test_job_durability import PUBLISHERS

    assert ("realmspinner._q_lora", "_lora_train", "import_lora") in PUBLISHERS
