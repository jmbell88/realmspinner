"""The worker records ``params["mesh_score"]`` for a finished mesh, and only that.

The conftest fake server returns a unit cube; a reference that is a square on a plain
background is its true silhouette, so the score is high, and a reference that is a thin
bar is not, so it is low. Nothing here starts a real engine.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from PIL import Image, ImageDraw

from realmspinner.config import Config
from realmspinner.db import JobStore
from realmspinner.queue import Worker


async def _wait_until(predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        gltfpack_exe=tmp_path / "no-gltfpack.exe",
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    yield w
    store.close()


def _reference(path, box):
    """A light-gray picture with one dark rectangle: the subject, as the matte sees it."""
    image = Image.new("RGB", (256, 256), (200, 200, 200))
    ImageDraw.Draw(image).rectangle(box, fill=(40, 40, 40))
    image.save(path)


async def _run(worker, box, **params):
    job_id = worker.store.create("image", None, {"seed": 1, "resolution": 512, **params})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    _reference(job_dir / "input.png", box)
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()
    return worker.store.get(job_id)


async def test_a_finished_mesh_records_its_score_against_its_own_reference(worker):
    row = await _run(worker, (64, 64, 192, 192))  # a square: the cube's own silhouette
    score = row["params"]["mesh_score"]
    assert score["version"] == 1
    assert score["iou"] is not None and score["iou"] > 0.8
    assert score["best_view"] is not None
    assert score["demoted_by"] == []


async def test_a_mesh_that_does_not_match_its_reference_scores_lower(worker, tmp_path):
    matching = (await _run(worker, (64, 64, 192, 192)))["params"]["mesh_score"]["iou"]

    other_config = Config(
        data_dir=tmp_path / "assets2",
        db_path=tmp_path / "assets2" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        gltfpack_exe=tmp_path / "no-gltfpack.exe",
    )
    store = JobStore(other_config.db_path)
    other = Worker(other_config, store)
    try:
        bar = (await _run(other, (16, 120, 240, 136)))["params"]["mesh_score"]["iou"]
    finally:
        store.close()
    assert bar < matching - 0.3


async def test_a_job_with_no_reference_picture_is_unscored_not_failed(worker):
    job_id = worker.store.create("image", None, {"seed": 1, "resolution": 512})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"not a picture")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()
    assert "mesh_score" not in worker.store.get(job_id)["params"]


async def test_a_missed_budget_demotes_in_the_stored_score(worker, monkeypatch):
    """``params["budget_miss"]`` (the shape gate's record) becomes a hard demotion."""
    from pathlib import Path

    import realmspinner.pipelines.blender_run as blender_run_mod
    from realmspinner import tiercheck
    from realmspinner.pipelines import postprocess, remesh

    def fake_blender(spec, **kw):
        # A readable mesh (the audit and the score both load it); only the reported
        # triangle count is the scripted lie.
        Path(spec["out_glb"]).write_bytes(Path(spec["source_glb"]).read_bytes())
        return {"ok": True, "method": "decimate", "faces": 13821, "texture_size": 1024}

    monkeypatch.setattr(blender_run_mod, "run_worker", fake_blender)
    monkeypatch.setattr(postprocess, "normalize_glb", lambda *a, **k: {"scale": 1.0})
    monkeypatch.setattr(tiercheck, "compare", lambda b, a: tiercheck.Verdict(True, (), ()))
    monkeypatch.setattr(remesh, "compare_geometry", lambda *a, **k: {"measured": False})
    row = await _run(
        worker, (64, 64, 192, 192), mesh_finishing="repair", lowpoly_triangles=5000,
        profile="raw",
    )
    assert row["params"]["budget_miss"]["achieved"] == 13821
    assert row["params"]["mesh_score"]["demoted_by"] == ["budget_miss"]
    assert row["params"]["mesh_score"]["budget_ratio"] == pytest.approx(2.764, abs=0.01)
