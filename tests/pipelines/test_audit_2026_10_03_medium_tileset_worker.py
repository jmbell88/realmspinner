"""plotter-23 (2026-10-03 audit): the worker's sidecar claimed a lock that never ran."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from realmspinner.config import Config
from realmspinner.db import JobStore
from realmspinner.pipelines import tileatlas
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


def _job(worker, prompts):
    seeds = tileatlas.material_seeds(100, len(prompts))
    geom = tileatlas.material_geometry(32, "top_down", len(prompts))
    params = {
        "seed": 100,
        "base_model": "sdxl_cfg",
        "colors": 64,
        "negative_prompt": "",
        "sheet": {
            "version": 3, "mode": "materials", "tile_w": geom.tile_w,
            "tile_h": geom.tile_h, "projection": "top_down",
            "columns": geom.columns, "rows": geom.rows, "layout": "grid",
            "materials": [
                {"index": i, "prompt": p, "variant": 1, "seed": s}
                for i, (p, s) in enumerate(zip(prompts, seeds, strict=True))
            ],
            "variants": 1, "style_lock": True,
        },
    }
    return worker.store.create("tile_sheet", "a damp dungeon", params, stage="tilesheet")


async def _run(worker, job_id):
    worker.start()
    try:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if worker.store.get(job_id)["status"] in ("done", "error", "cancelled"):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("job did not finish before timeout")
    finally:
        await worker.shutdown()
    return worker.store.get(job_id)


@pytest.mark.asyncio
async def test_a_one_material_sheet_does_not_record_a_style_lock_that_never_ran(worker):
    job_id = _job(worker, ("moss",))
    row = await _run(worker, job_id)
    assert row["status"] == "done", row["error"]

    recipe = json.loads((worker.config.job_dir(job_id) / "sheet.json").read_text())["recipe"]
    assert recipe["style_lock"] is False
    assert "ip_adapter" not in recipe


@pytest.mark.asyncio
async def test_a_two_material_sheet_still_records_the_lock_it_ran(worker):
    job_id = _job(worker, ("moss", "gravel"))
    row = await _run(worker, job_id)
    assert row["status"] == "done", row["error"]

    recipe = json.loads((worker.config.job_dir(job_id) / "sheet.json").read_text())["recipe"]
    assert recipe["style_lock"] is True
    assert recipe["ip_adapter"] == "plus"
