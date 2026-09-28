"""Three gaps in the tileset worker (``_q_tileset.py``), the 2026-09-26 audit.

**plotter-tiles-03 / service-kinds-07.** A terrain row's ``mask`` record and
its terrain-row count against the geometry were validated only inside
``tileatlas.atlas_sidecar``, at the very end -- after both materials had
already been generated (and, for a row-count mismatch, after
``tilemask.blob_atlas`` had run and ``input.png`` had already been
published). Moved up front, before the card is spent, the same rule the
palette check right above it already keeps.

**service-kinds-08.** ``compose_prompt`` (fallible) ran between
``_acquire_t2i`` and the ``try`` that releases the pipe, so a raise there
leaked it.

**service-kinds-10.** The per-material PNG decode after the release ran
straight on ``realmspinner-loop`` instead of behind ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from realmspinner import guidance
from realmspinner.config import Config
from realmspinner.db import JobStore
from realmspinner.pipelines import tileatlas, tilemask
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


def _terrain_job(worker, *, tile_w=32, seed=200, mask=None, terrains=None, **extra) -> str:
    seeds = tileatlas.material_seeds(seed, 2)
    geom = tileatlas.terrain_geometry(tile_w, "top_down")
    params = {
        "seed": seed,
        "base_model": "sdxl_cfg",
        "colors": 64,
        "negative_prompt": "",
        "sheet": {
            "version": 3,
            "mode": "terrain",
            "tile_w": geom.tile_w,
            "tile_h": geom.tile_h,
            "projection": "top_down",
            "columns": geom.columns,
            "rows": geom.rows,
            "layout": "blob47",
            "materials": [
                {"index": 0, "prompt": "grass", "variant": 1, "seed": seeds[0]},
                {"index": 1, "prompt": "still water", "variant": 1, "seed": seeds[1]},
            ],
            "terrains": (
                terrains
                if terrains is not None
                else [{"name": "grass", "fill": [40, 120, 40, 255], "outline": [20, 60, 20, 255]}]
            ),
            "mask": mask
            if mask is not None
            else {
                "version": tilemask.MASK_VERSION,
                "seed": 5,
                "inset": None,
                "amplitude": None,
                "feather": None,
            },
            "variants": 1,
            "style_lock": False,
        },
    }
    params.update(extra)
    return worker.store.create("tile_sheet", "a shoreline", params, stage="tilesheet")


def _materials_job(worker, *, tile_w=32, prompts=("moss", "gravel"), seed=100) -> str:
    seeds = tileatlas.material_seeds(seed, len(prompts))
    geom = tileatlas.material_geometry(tile_w, "top_down", len(prompts))
    params = {
        "seed": seed,
        "base_model": "sdxl_cfg",
        "colors": 64,
        "negative_prompt": "",
        "sheet": {
            "version": 3,
            "mode": "materials",
            "tile_w": geom.tile_w,
            "tile_h": geom.tile_h,
            "projection": "top_down",
            "columns": geom.columns,
            "rows": geom.rows,
            "layout": "grid",
            "materials": [
                {"index": i, "prompt": p, "variant": 1, "seed": s}
                for i, (p, s) in enumerate(zip(prompts, seeds, strict=True))
            ],
            "terrains": [],
            "mask": None,
            "boundary": "",
            "variants": 1,
            "style_lock": False,
        },
    }
    return worker.store.create("tile_sheet", "a dungeon", params, stage="tilesheet")


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


# --- plotter-tiles-03 / service-kinds-07 ------------------------------------


@pytest.mark.asyncio
async def test_a_terrain_row_count_mismatch_costs_no_generation(worker):
    """The row check used to live only in ``tileatlas.atlas_sidecar``, at the
    very end -- after both materials were drawn, the mask atlas composited
    and ``input.png`` published. A block declaring the wrong number of
    terrain rows must now be refused before either generation runs."""
    row = await _run(worker, _terrain_job(worker, terrains=[]))

    assert row["status"] == "error"
    assert "terrain" in (row["error"] or "")
    assert worker._text2image is None or not worker._text2image.prompts
    job_dir = worker.config.job_dir(row["id"])
    assert not (job_dir / "input.png").exists()


@pytest.mark.asyncio
async def test_a_missing_mask_record_costs_no_generation(worker):
    row = await _run(worker, _terrain_job(worker, mask={}))

    assert row["status"] == "error"
    assert "mask" in (row["error"] or "")
    assert worker._text2image is None or not worker._text2image.prompts


# --- service-kinds-08 --------------------------------------------------------


@pytest.mark.asyncio
async def test_a_compose_prompt_failure_still_releases_the_resident_pipe(
    worker, monkeypatch
):
    """``compose_prompt`` used to run between ``_acquire_t2i`` and the
    ``try`` that frees the pipe; a raise there leaked it, leaving the next
    job's admission wrongly crediting or debiting a resident checkpoint that
    was never released."""
    calls = {"n": 0}
    real_compose = guidance.compose_prompt

    def flaky_compose(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic compose_prompt failure")
        return real_compose(*args, **kwargs)

    import realmspinner._q_tileset as q_tileset

    monkeypatch.setattr(q_tileset.guidance, "compose_prompt", flaky_compose)

    released = {"n": 0}
    real_release = Worker._release_t2i

    async def spying_release(self, *a, **kw):
        released["n"] += 1
        return await real_release(self, *a, **kw)

    monkeypatch.setattr(Worker, "_release_t2i", spying_release)

    row = await _run(worker, _materials_job(worker))

    assert row["status"] == "error"
    assert "synthetic compose_prompt failure" in (row["error"] or "")
    assert released["n"] == 1, (
        "a raise between acquire and try must still release the resident pipe"
    )


# --- service-kinds-10 --------------------------------------------------------


@pytest.mark.asyncio
async def test_the_per_material_decode_runs_off_the_loop_thread(worker, monkeypatch):
    """The post-release per-material ``Image.open`` used to decode straight
    on ``realmspinner-loop``. Wrapping it in ``asyncio.to_thread`` is the
    fix; this spies on ``PIL.Image.open`` itself (not on ``to_thread``, which
    the unfixed code never calls for this decode at all -- a spy on
    ``to_thread`` alone would pass against the bug it is supposed to catch)
    and asserts the decode never runs on the loop thread."""
    import threading

    import PIL.Image as real_pil_image

    decode_threads: set[int] = set()
    real_open = real_pil_image.open

    def spying_open(*args, **kwargs):
        decode_threads.add(threading.get_ident())
        return real_open(*args, **kwargs)

    monkeypatch.setattr(real_pil_image, "open", spying_open)

    main_thread = threading.get_ident()
    row = await _run(worker, _materials_job(worker))

    assert row["status"] == "done", row.get("error")
    assert decode_threads, "no decode was observed at all"
    assert main_thread not in decode_threads, (
        "the material decode ran on the loop thread instead of a worker thread"
    )
