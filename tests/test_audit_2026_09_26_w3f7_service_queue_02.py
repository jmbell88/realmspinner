"""service-queue-02 (2026-09-26 audit): a separate job never stopped trellis.

``vram.estimate_job_parts`` prices a ``separate`` job under ``vram_exclusive``
with no trellis term at all -- the same assumption ``_acquire_t2i`` and
``_acquire_music`` make for their own kinds, and *keep*, by actually stopping
trellis before they run. ``_q_music.py``'s ``_separate`` made the same
assumption but never paid for it: with a resident trellis left running,
``queue._check_resources`` still credited its footprint back into headroom as
memory the device did not actually have free, on top of a ``need`` already
quoted as if trellis were gone -- wrong in the same direction twice over.

Proven here by dispatching an actual ``separate`` job under
``vram_exclusive`` and watching whether it stops a resident trellis before it
finishes: if it does, the credit ``_check_resources`` gave at admission time
was honoured by the time the job's own work ran, exactly as it already is for
every other kind that reaches that credit line.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from realmspinner.pipelines import blender_run
from realmspinner.queue import Worker
from realmspinner.service import _jobs_rework as rework

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _admitted(monkeypatch):
    monkeypatch.setattr(rework, "check_weights", lambda svc, kind, params: None)
    monkeypatch.setattr(rework, "check_vram", lambda svc, kind, stage, params: None)


def _take(svc, status: str = "done") -> str:
    job_id = svc.store.create("music", "dark ambient", {}, stage="music")
    svc.config.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    (svc.config.job_dir(job_id) / "track.wav").write_bytes(b"x")
    svc.store.set_status(job_id, status)
    return job_id


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


async def test_a_separate_job_stops_a_warm_trellis_under_vram_exclusive(svc, monkeypatch):
    svc.config.vram_exclusive = True

    def fake_run_worker(spec, *, on_progress=None, on_start=None, timeout=0.0, **kwargs):
        Path(spec["out_dir"]).mkdir(parents=True, exist_ok=True)
        return {"ok": True, "files": [], "rate": 44100}

    monkeypatch.setattr(blender_run, "run_worker", fake_run_worker)

    worker = Worker(svc.config, svc.store)
    stop_calls: list[int] = []
    monkeypatch.setattr(worker.trellis, "stop", lambda: stop_calls.append(1))

    take = _take(svc)
    split_id = rework.separate_job(svc, take)["id"]

    worker.start()
    await _wait_until(lambda: worker.store.get(split_id)["status"] == "done")
    # Checked before shutdown(), which also stops trellis on its own -- the
    # claim is that *this job* stopped it, not that the worker eventually did.
    assert stop_calls, (
        "a separate job under vram_exclusive must stop a resident trellis "
        "itself, matching the assumption vram.estimate_job_parts already "
        "prices it under -- otherwise queue._check_resources's headroom "
        "credit for a warm trellis is never actually earned"
    )
    await worker.shutdown()
