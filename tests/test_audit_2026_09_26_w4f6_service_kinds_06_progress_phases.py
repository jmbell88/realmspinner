"""service-kinds-06 (2026-09-26 audit): ``_retexture``/``_remesh``/
``_lora_train`` passed whole-bar fractions as ``inner`` to
``ProgressBus.update``, which treats ``inner`` as *phase-relative* (mapped
through ``lo + span * inner`` for that phase's own ``(lo, hi)`` in
``progress.PHASES_*``). A leftover pre-split absolute value like ``inner=0.95``
against a narrow tail phase such as ``PHASES_REMESH``'s ``"publish": (0.95,
1.00)`` got remapped a second time (``0.95 + 0.05 * 0.95`` = 99.75%), so the
bar jumped to ~100% before the phase's own work had even started.

``_q_lora.py``'s ``_lora_train`` and ``_q_mesh.py``'s ``_remesh`` each have a
cheap end-to-end harness (``tests/test_loras.py``, ``tests/test_remesh.py``),
so those two are proven by running the real job and inspecting the recorded
``progress.update`` call. ``_q_sprite.py``'s ``_retexture`` does not --
``tests/test_job_durability.py``'s own comment says why ("it wants a resident
SDXL pipe, ten Blender renders and a texture bake") -- so it is proven the
same way that file already proves things about ``_retexture``: by reading
its source.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from pathlib import Path

import pytest
from PIL import Image

from realmspinner.config import Config
from realmspinner.db import JobStore
from realmspinner.pipelines import blender_run, lora_train
from realmspinner.queue import Worker


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


def _spy_progress(worker):
    calls = []
    real_update = worker.progress.update

    def spying(job_id, **kwargs):
        calls.append(dict(kwargs))
        return real_update(job_id, **kwargs)

    worker.progress.update = spying
    return calls


# --- _q_lora.py: the "publish" phase ----------------------------------------


@pytest.fixture
def lora_worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models",
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    yield w
    store.close()


async def test_the_lora_train_publish_phase_is_reported_phase_relative(
    lora_worker, monkeypatch
):
    calls = _spy_progress(lora_worker)
    monkeypatch.setattr(lora_worker.trellis, "stop", lambda: None)

    def fake(spec, *, on_progress=None, on_start=None, timeout=0.0, **kw):
        out = Path(spec["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / lora_train.WEIGHTS_NAME).write_bytes(b"\x00" * 32)
        return {"ok": True, "steps": 10, "images": 3, "rank": 16, "loss": 0.05,
                "weights": str(out / lora_train.WEIGHTS_NAME)}

    monkeypatch.setattr(blender_run, "run_worker", fake)
    job_id = lora_worker.store.create(
        "lora_train", "Cosmos",
        {"base_model": "sdxl_cfg", "label": "Cosmos", "trigger": "cosmos style", "steps": 10},
    )
    train_dir = lora_worker.config.job_dir(job_id) / "train"
    train_dir.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (8, 8)).save(train_dir / f"{i:03d}.png")

    lora_worker.start()
    await _wait_until(lambda: lora_worker.store.get(job_id)["status"] in ("done", "error"))
    await lora_worker.shutdown()
    assert lora_worker.store.get(job_id)["status"] == "done"

    publish_calls = [c for c in calls if c.get("phase") == "publish"]
    assert publish_calls, "no publish-phase progress update was recorded"
    # Phase-relative: 0.0 at the phase's own floor, easing to 1.0 -- not the
    # pre-split whole-bar 0.95 that ``progress.update`` would remap a second
    # time against ``PHASES_LORA_TRAIN``'s ``"publish": (0.95, 1.00)``.
    assert publish_calls[0]["inner"] == 0.0
    assert publish_calls[0]["inner_next"] == 1.0


# --- _q_mesh.py: the "publish" phase -----------------------------------------


@pytest.fixture
def mesh_worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models",
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    yield w
    store.close()


def _mesh_job(worker: Worker) -> str:
    job_id = worker.store.create("text", "a crate", {"seed": 1, "size_m": 1.0})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"old-model")
    (job_dir / "source.glb").write_bytes(b"reconstruction")
    worker.store.set_status(job_id, "done")
    return job_id


async def test_the_remesh_publish_phase_is_reported_phase_relative(
    mesh_worker, monkeypatch
):
    from realmspinner.pipelines import postprocess

    monkeypatch.setattr(postprocess, "normalize_glb", lambda *a, **k: {"scale": 1.0})
    calls = _spy_progress(mesh_worker)

    def fake(spec, *, on_progress=None, on_start=None, timeout=0.0):
        if on_progress is not None:
            on_progress(0.5, "Remeshing")
        Path(spec["out_glb"]).write_bytes(b"new-model")
        return {"ok": True, "method": "quadriflow", "faces": 8000, "faces_before": 300000,
                "quads": 0.97, "texture_size": spec["texture_size"], "metallic": 0.0}

    monkeypatch.setattr(blender_run, "run_worker", fake)
    source = _mesh_job(mesh_worker)
    job_id = mesh_worker.store.create(
        "remesh", "a crate", {"source_job": source, "target_faces": 8000, "texture_size": 512}
    )
    mesh_worker.start()
    await _wait_until(lambda: mesh_worker.store.get(job_id)["status"] in ("done", "error"))
    await mesh_worker.shutdown()
    row = mesh_worker.store.get(job_id)
    assert row["status"] == "done", row.get("error")

    publish_calls = [c for c in calls if c.get("phase") == "publish"]
    assert publish_calls, "no publish-phase progress update was recorded"
    assert publish_calls[0]["inner"] == 0.0
    assert publish_calls[0]["inner_next"] == 1.0


# --- _q_sprite.py's _retexture: no cheap end-to-end harness ------------------


def test_retexture_progress_calls_are_phase_relative_not_whole_bar_literals():
    """Reads the source, the way ``tests/test_job_durability.py`` already
    proves things about ``_retexture`` for the same "no cheap harness"
    reason. Before the fix, the "views", "restyle", "project" and "assemble"
    ``self.progress.update`` calls carried whole-bar-looking literals
    (``inner=f * 0.2``, ``inner=0.2 + 0.55 * index / len(views)``,
    ``inner=0.75``, ``inner=0.95``) left over from before
    ``progress.PHASES_RETEXTURE`` split those stages into their own narrow
    ``(lo, hi)`` slices. Phase-relative calls stay in ``[0, 1]`` by
    construction and never repeat those literals against those phases."""
    import realmspinner._q_sprite as q_sprite

    source = inspect.getsource(q_sprite.SpriteOps._retexture)

    # The exact leftover whole-bar literals the 2026-09-26 audit found,
    # each paired with the phase it was wrongly computed against.
    assert "inner=f * 0.2" not in source
    assert "0.2 + 0.55 * index" not in source
    assert 'inner=0.75,' not in source
    assert 'label="Baking projections", inner=0.75' not in source
    assert 'label="Combining projections", inner=0.95' not in source
    assert "0.75 + f * 0.2" not in source

    # And the fix is actually there: the views/project on_progress lambdas
    # and the assemble marker now hand ``progress.update`` values relative
    # to their own phase.
    assert 'phase="views"' in source and "inner=f," in source
    assert 'phase="assemble"' in source and 'inner=0.0,' in source
