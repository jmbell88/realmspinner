"""Regression tests for two 2026-09-26 audit findings whose fix sites are the
worker's job-kind plumbing at the top of ``src/realmspinner``:

* **poser-jobs-01** -- ``_q_jobs.py``'s ``_maybe_queue_charsheet`` used to
  write ``colors: None``/``outline: None`` onto an HD (``pixel_art: False``)
  follow-up row and never carried ``pixel_art`` at all, so
  ``_q_troupe``'s read of it defaulted back to the ordinary pixel-art path
  and crashed on ``int(None)`` after the whole render.
* **service-kinds-01** -- ``_q_troupe.py``'s subset re-render pinned a
  re-rendered run's palette to the base sheet's exact colours only when
  ``_palette_entries`` returned ``None``, but an unnamed palette makes it
  return ``()`` instead, so the pin never ran and a re-rendered run could
  come back a different shade from the sheet beside it.

Named per the fixer brief's convention
(``tests/<mirrored dir>/test_audit_2026_09_26_w2f7_<mirrored dir>.py``); both
fixes sit at the root of ``src/realmspinner``, so this file does too.
"""

from __future__ import annotations

import json

import pytest

from realmspinner.config import Config
from realmspinner.db import JobStore
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


async def _wait_until(predicate, timeout: float = 60.0) -> None:
    import asyncio

    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


def _model_job(worker, **params):
    """A finished model-stage job with a rig and an HD Troupe follow-up
    block, the shape ``service.troupe._check_options`` mints for
    ``pixel_art: False`` -- ``colors``/``outline``/``palette``/``dither`` are
    all *absent*, never ``None``."""
    base = {
        "seed": 5,
        "rig": True,
        "rig_template": "humanoid",
        "troupe": {"logical_size": 32, "pixel_art": False},
    }
    base.update(params)
    job_id = worker.store.create("image", "a ranger", base, stage="model")
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    return job_id


async def test_an_hd_reference_chain_charsheet_row_carries_pixel_art_false_and_no_none_pixel_options(  # noqa: E501
    worker,
):
    """poser-jobs-01: an HD reference-chain follow-up must carry
    ``pixel_art: False`` forward and must never write ``colors``/``outline``
    as ``None`` -- a present ``None`` defeats ``_q_troupe``'s own
    ``params.get("colors", 64)`` default the same way it defeats
    ``elevation``'s.
    """
    job_id = _model_job(worker)
    job = worker.store.get(job_id)
    await worker._maybe_queue_charsheet(job)

    sheet = next(j for j in worker.store.list() if j["kind"] == "charsheet")
    params = sheet["params"]

    assert params["pixel_art"] is False
    assert "colors" not in params or params["colors"] is not None
    assert "outline" not in params or params["outline"] is not None


async def test_an_ordinary_reference_chain_charsheet_row_still_carries_its_colours(
    worker,
):
    """The fix must not turn an ordinary (pixel-art) follow-up's ``colors``/
    ``outline`` into absent keys -- only an HD block's ``None`` must stop
    being written."""
    job_id = _model_job(
        worker,
        troupe={"logical_size": 32, "colors": 16, "outline": "outer"},
    )
    job = worker.store.get(job_id)
    await worker._maybe_queue_charsheet(job)

    sheet = next(j for j in worker.store.list() if j["kind"] == "charsheet")
    params = sheet["params"]

    assert params["pixel_art"] is True
    assert params["colors"] == 16
    assert params["outline"] == "outer"


# -- service-kinds-01 ---------------------------------------------------------


_LAYOUT = {
    "version": 2,
    "movements": [
        {"key": "idle", "frames": 3, "directions": 1},
        {"key": "attack", "frames": 2, "directions": 1},
    ],
}


def _fake_render(monkeypatch):
    """A Blender fake that paints each cell a distinct flat colour (varying
    by cell index), so that median-cutting a two-cell subset alone lands on
    different quantised RGB triples than median-cutting the same two cells
    as part of the whole five-cell atlas would -- the "same shirt, two
    shades" failure this fix exists to prevent. A single flat colour shared
    by every cell would median-cut losslessly either way and prove nothing.
    """
    from pathlib import Path

    from PIL import Image

    from realmspinner.pipelines import blender_run

    def fake(spec, **kwargs):
        frames_dir = Path(spec["frames_dir"])
        for cell in spec["cells"]:
            idx = cell["index"]
            fill = (30 + idx * 40, 60 + idx * 20, 200 - idx * 15, 255)
            frame = Image.new("RGBA", (spec["frame_size"],) * 2, (0, 0, 0, 0))
            inset = spec["frame_size"] // 4
            frame.paste(
                fill,
                (inset, inset, spec["frame_size"] - inset, spec["frame_size"] - inset),
            )
            frame.save(frames_dir / f"{idx:04d}.png")
        return {
            "ok": True,
            "pivot": [0.5, 0.9],
            "framing": {"extent": 2.24, "margin": spec.get("margin") or 1.12},
        }

    monkeypatch.setattr(blender_run, "run_worker", fake)


async def test_a_subset_rerender_with_no_named_palette_is_quantised_to_the_base_sheets_exact_colours(  # noqa: E501
    worker, monkeypatch
):
    """service-kinds-01: with no palette named, a subset re-render must be
    pinned to the base sheet's own exact colour set (``_atlas_entries``) --
    the "same shirt, two shades" bug meant this pin never fired because
    ``_palette_entries("")`` returns ``()``, not ``None``.

    One ``worker.start()``/``shutdown()`` cycle covers both jobs, the same
    shape ``tests/service/test_troupe_chain.py``'s own subset-rerender tests
    use -- a ``Worker`` is not meant to be restarted after ``shutdown()``.
    """
    import realmspinner._q_troupe as q_troupe_mod

    calls: list[object] = []
    real_atlas_entries = q_troupe_mod._atlas_entries

    def spy(png, colors):
        calls.append(png)
        return real_atlas_entries(png, colors)

    monkeypatch.setattr(q_troupe_mod, "_atlas_entries", spy)
    _fake_render(monkeypatch)

    source = worker.store.create("image", "a ranger", {}, stage="model")
    source_dir = worker.config.job_dir(source)
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "model.glb").write_bytes(b"fake-glb")
    (source_dir / "rig.glb").write_bytes(b"fake-rig")
    (source_dir / "rig.json").write_text(json.dumps({"template": "humanoid"}), "utf-8")
    worker.store.set_status(source, "done")

    def _queue(**extra):
        return worker.store.create(
            "charsheet",
            "a ranger",
            {
                "source_job": source,
                "sheet_id": rig_store.new_id(),
                "logical_size": 16,
                "colors": 8,
                "layout": _LAYOUT,
                **extra,
            },
        )

    first_id = _queue()
    worker.start()
    try:
        await _wait_until(
            lambda: worker.store.get(first_id)["status"] in ("done", "error")
        )
        first = worker.store.get(first_id)
        assert first["error"] is None
        base_sheet = first["params"]["sheet_id"]

        calls.clear()
        rerun_id = _queue(
            subset=[{"animation": "attack", "direction": "front"}],
            base_sheet=base_sheet,
        )
        await _wait_until(
            lambda: worker.store.get(rerun_id)["status"] in ("done", "error")
        )
        rerun = worker.store.get(rerun_id)
        assert rerun["error"] is None
    finally:
        await worker.shutdown()

    # The whole claim: the pin ran (``_atlas_entries`` was called against the
    # base sheet's own PNG) rather than silently skipping to an independent
    # median cut of the two-cell subset.
    assert calls, "the base sheet's exact colours were never read back"
    assert calls[0] == rig_store.sheet_png_path(source_dir, base_sheet)
