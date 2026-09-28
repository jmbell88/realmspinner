"""service-kinds-07 (2026-09-26 audit, folded into plotter-tiles-03): a named
palette used to be re-resolved only inside ``_charsheet``'s ``_quantise()``
closure, which runs *after* ``_render_charsheet``'s Blender subprocess -- the
expensive part -- has already finished. A palette deleted after the job was
queued failed it only once the render was paid for.

Self-contained, in ``test_charsheet_cancel.py``'s own words: a local
``worker`` fixture and its own render fake, rather than importing another
fix's private helpers.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from realmspinner.config import Config
from realmspinner.db import JobStore
from realmspinner.kernels.rig import store as rig_store
from realmspinner.pipelines import blender_run
from realmspinner.queue import Worker

pytestmark = pytest.mark.asyncio


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


async def _wait_until(predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


def _fake_render(monkeypatch):
    from PIL import Image

    calls: list[dict] = []

    def fake(spec, **kwargs):
        calls.append({"spec": spec, **kwargs})
        frames_dir = Path(spec["frames_dir"])
        for cell in spec["cells"]:
            frame = Image.new("RGBA", (spec["frame_size"],) * 2, (0, 0, 0, 0))
            frame.paste((10, 20, 30, 255), (2, 2, spec["frame_size"] - 2, spec["frame_size"] - 2))
            frame.save(frames_dir / f"{cell['index']:04d}.png")
        return {"ok": True, "pivot": [0.5, 0.9], "framing": {"extent": 1.0, "margin": 1.12}}

    monkeypatch.setattr(blender_run, "run_worker", fake)
    return calls


def _rigged_source(worker):
    source = worker.store.create("image", "a ranger", {}, stage="model")
    source_dir = worker.config.job_dir(source)
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "model.glb").write_bytes(b"fake-glb")
    (source_dir / "rig.glb").write_bytes(b"fake-rig")
    (source_dir / "rig.json").write_text(json.dumps({"template": "humanoid"}), "utf-8")
    worker.store.set_status(source, "done")
    return source, source_dir


_TINY_LAYOUT = {
    "version": 2,
    "movements": [{"key": "idle", "frames": 3, "directions": 1}],
}


def _queue_charsheet(worker, source, **extra):
    params = {
        "source_job": source,
        "sheet_id": rig_store.new_id(),
        "logical_size": 16,
        "colors": 8,
        "layout": _TINY_LAYOUT,
    }
    params.update(extra)
    return worker.store.create("charsheet", "a ranger", params)


async def test_a_palette_that_is_not_installed_costs_no_blender_render(worker, monkeypatch):
    calls = _fake_render(monkeypatch)
    source, _source_dir = _rigged_source(worker)
    job_id = _queue_charsheet(worker, source, palette="does-not-exist")

    worker.start()
    try:
        await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
    finally:
        await worker.shutdown()

    row = worker.store.get(job_id)
    assert row["status"] == "error"
    assert "installed" in (row["error"] or ""), row.get("error")
    assert calls == [], (
        "a since-deleted palette must be refused before the Blender render, "
        f"but {len(calls)} render(s) ran first"
    )


async def test_an_hd_request_never_checks_the_palette_at_all(worker, monkeypatch):
    """``pixel_art=False`` never reads ``colors``/``palette`` -- the palette
    check is gated the same way ``_quantise`` itself already gates them, so an
    HD request naming no real palette at all is unaffected."""
    calls = _fake_render(monkeypatch)
    source, _source_dir = _rigged_source(worker)
    job_id = _queue_charsheet(worker, source, pixel_art=False)

    worker.start()
    try:
        await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
    finally:
        await worker.shutdown()

    row = worker.store.get(job_id)
    assert row["status"] == "done", row.get("error")
    assert calls, "the HD render should still have run"
