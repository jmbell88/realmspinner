"""The 2026-10-10 audit's Create exits / quality / matte / sweep findings.

One file, six claims (create-14, -15, -16, -17, -29, -30). Each test's name is
its claim, and each fails against the code the audit read.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.studio import asset_exits, quality

SRC = Path(__file__).resolve().parents[1] / "src" / "realmspinner"


# --- create-14 / create-15: asset_exits -------------------------------------


class _Cache:
    def get(self, job_id: Any) -> Any:
        return None


class _Ctx:
    def __init__(self, svc: Any) -> None:
        self.svc = svc
        self.cache = _Cache()
        self.state = SimpleNamespace(mode="library", create=SimpleNamespace(stage="reference"))
        self.state.selected = None

    def job(self) -> Any:
        return None


def _row(svc, kind: str, *, stage: str, status: str, files: list[str], params=None) -> dict:
    job_id = svc.store.create(kind, "x", params or {}, stage=stage, status=status)
    job = svc.store.get(job_id)
    job["files"] = files
    return job


def test_a_lora_training_row_offers_no_mesh_exits(svc):
    """A ``lora_train`` row is minted with no stage, takes the column default
    ``model`` and carries no ``source_job``, so ``_is_mesh`` read it as a mesh
    and offered Clay, Mason and two Poser doors, each dimmed with "This mesh has
    no model yet." -- a reason that is false of a LoRA."""
    for status in ("done", "running", "error"):
        lora = _row(
            svc,
            "lora_train",
            stage="model",
            status=status,
            files=["lora.safetensors"],
            params={"base_model": "sdxl", "label": "x", "trigger": "t"},
        )
        exits = asset_exits.exits_for(_Ctx(svc), lora)
        mesh_modes = {e.mode for e in exits} & {"clay", "mason", "poser"}
        assert mesh_modes == set(), (status, [(e.mode, e.reason) for e in exits])
        assert not any("no model yet" in e.reason for e in exits)


@pytest.mark.parametrize("stage", ["tilesheet", "tile", "reference"])
@pytest.mark.parametrize("status", ["queued", "running", "error", "cancelled"])
def test_an_errored_tile_sheet_that_already_has_input_png_dims_the_plotter_and_packwright_doors(
    svc, stage, status
):
    """``_plotter_add``/``_packwright_add``'s ready branch tested the stage and
    ``input.png`` but never the row's status, so an unfinished tile row that had
    already published its picture was offered a live "Add to Plotter as a
    tileset" while Inker, on the same row, was dimmed by status."""
    job = _row(svc, "tile_sheet", stage=stage, status=status, files=["input.png"])
    by_mode = {e.mode: e for e in asset_exits.exits_for(_Ctx(svc), job)}
    assert by_mode["plotter"].reason, "an unfinished tile row got a live Plotter door"
    assert by_mode["packwright"].reason, "an unfinished tile row got a live Packwright door"


def test_a_finished_tile_sheet_with_input_png_keeps_live_plotter_and_packwright_doors(svc):
    job = _row(svc, "tile_sheet", stage="tilesheet", status="done", files=["input.png"])
    by_mode = {e.mode: e for e in asset_exits.exits_for(_Ctx(svc), job)}
    assert by_mode["plotter"].reason == ""
    assert by_mode["packwright"].reason == ""


# --- create-16: remesh_line --------------------------------------------------


def test_remesh_line_does_not_claim_the_lowest_when_the_restore_failed():
    """When the restore of the better attempt failed (``_restore_pair`` raised
    OSError) the mesh that ships is the last attempt, so "kept the lowest" was
    false. The attempt entry that shipped is marked, and the line names it."""
    attempts = [
        {"seed": 1, "worst": 0.05},
        {"seed": 2, "worst": 0.30, "kept": True},
    ]
    text = " ".join(quality.remesh_line(attempts))
    assert "kept the lowest" not in text
    assert "attempt 2" in text and "30.0%" in text
    # The lowest reading is still reported, as the one that could not be put back.
    assert "5.0%" in text and "restore" in text


def test_remesh_line_names_the_attempt_that_shipped_when_it_is_the_lowest():
    attempts = [
        {"seed": 1, "worst": 0.30},
        {"seed": 2, "worst": 0.05, "kept": True},
    ]
    text = " ".join(quality.remesh_line(attempts))
    assert "kept the lowest (5.0%)" in text


def test_remesh_line_without_a_kept_mark_keeps_the_old_wording_for_old_rows():
    text = " ".join(quality.remesh_line([{"worst": 0.123}, {"worst": 0.045}]))
    assert "kept the lowest (4.5%)" in text


def _retry_worker(tmp_path, **config_kwargs):
    from realmspinner.config import Config
    from realmspinner.db import JobStore
    from realmspinner.queue import Worker

    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        **config_kwargs,
    )
    assert str(tmp_path) in str(config.data_dir)
    store = JobStore(config.db_path)
    return Worker(config, store), store


async def _wait_until(predicate, timeout: float = 10.0) -> None:
    import asyncio

    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        await asyncio.sleep(0.02)


async def test_a_failed_restore_marks_the_last_attempt_as_the_one_that_shipped(
    tmp_path, fake_pipelines, monkeypatch
):
    """The restore of the better first attempt fails; ``params["mesh_attempts"]``
    must say the *last* attempt shipped (it is the mesh on disk), not leave the
    reader to assume the lowest one did."""
    import realmspinner.meshaudit as meshaudit_mod
    import realmspinner.queue as queue_mod

    readings = [0.2, 0.9]  # attempt 1 is better; attempt 2 ships when restore fails
    seen = {"n": 0}

    def fake(path, views, resolution):
        value = readings[min(seen["n"], len(readings) - 1)]
        seen["n"] += 1
        return {"worst": value, "mean": value / 2, "faces": 1000, "resolution": resolution,
                "views": []}

    monkeypatch.setattr(meshaudit_mod, "hole_fraction", fake)

    real_link = queue_mod._stage_link

    def link(source, dest):
        # Staging links *into* best.glb / best.source.glb; the restore links
        # *out of* them onto the served names. Only the restore fails.
        if str(source).endswith(("best.glb", "best.source.glb")):
            raise OSError("restore refused")
        return real_link(source, dest)

    monkeypatch.setattr(queue_mod, "_stage_link", link)

    w, store = _retry_worker(tmp_path, mesh_retries=1, mesh_hole_max=0.1)
    job_id = store.create("image", None, {"seed": 3, "resolution": 512, "mesh_seed": 3})
    job_dir = w.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    params = store.get(job_id)["params"]
    attempts = params["mesh_attempts"]
    assert [a["worst"] for a in attempts] == [0.2, 0.9]
    assert [bool(a.get("kept")) for a in attempts] == [False, True]
    assert params["mesh_seed"] == attempts[-1]["seed"]
    assert "kept the lowest" not in " ".join(quality.remesh_line(attempts))
    store.close()


async def test_a_good_restore_marks_the_better_attempt_as_the_one_that_shipped(
    tmp_path, fake_pipelines, monkeypatch
):
    import realmspinner.meshaudit as meshaudit_mod

    readings = [0.2, 0.9]
    seen = {"n": 0}

    def fake(path, views, resolution):
        value = readings[min(seen["n"], len(readings) - 1)]
        seen["n"] += 1
        return {"worst": value, "mean": value / 2, "faces": 1000, "resolution": resolution,
                "views": []}

    monkeypatch.setattr(meshaudit_mod, "hole_fraction", fake)
    w, store = _retry_worker(tmp_path, mesh_retries=1, mesh_hole_max=0.1)
    job_id = store.create("image", None, {"seed": 3, "resolution": 512, "mesh_seed": 3})
    job_dir = w.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    attempts = store.get(job_id)["params"]["mesh_attempts"]
    assert [bool(a.get("kept")) for a in attempts] == [True, False]
    assert "kept the lowest (20.0%)" in " ".join(quality.remesh_line(attempts))
    store.close()


# --- create-17: body plans counted from the registry -------------------------


def test_the_character_type_hint_and_no_species_refusal_count_body_plans_from_the_registry(
    monkeypatch,
):
    """Both sentences counted species from the registry and hard-coded "four
    body plans"; a fifth archetype would have shipped beside a sentence
    promising four."""
    from realmspinner.characters import family
    from realmspinner.characters import resolve as resolve_mod
    from realmspinner.studio.modes.create.engine import character as engine
    from realmspinner.studio.modes.create.ui import brief

    real = family.archetypes()
    fake = dict(real)
    fake["sixth"] = next(iter(real.values()))
    fake["seventh"] = next(iter(real.values()))
    assert len(fake) == len(real) + 2
    monkeypatch.setattr(family, "archetypes", lambda: dict(fake))

    hint = brief._character_hint()
    assert "six body plans" in hint and "four body plans" not in hint

    opts = {"families": [{"key": "human"}], "archetypes": [{"key": k} for k in fake]}
    sentence = engine._no_species(resolve_mod.Resolution(), opts)
    assert "six body plans" in sentence and "four body plans" not in sentence


def test_the_shipped_registry_still_reads_four_body_plans():
    from realmspinner.characters import resolve as resolve_mod
    from realmspinner.studio.modes.create.engine import character as engine
    from realmspinner.studio.modes.create.ui import brief

    assert "across four body plans" in brief._TYPE_HINTS["character"]
    opts = {"families": [{"key": "human"}], "archetypes": [{}, {}, {}, {}]}
    assert "across four body plans" in engine._no_species(resolve_mod.Resolution(), opts)


# --- create-29: matte preview stale completions ------------------------------


class _MatteCtx:
    def __init__(self, svc: Any) -> None:
        self.state = SimpleNamespace(matte=None)
        self.svc = svc
        self.submitted: list[tuple[str, dict]] = []

    def submit(self, key, fn, *args, **kwargs):
        self.submitted.append((key, kwargs))
        return True


def _png_job(svc) -> str:
    import io

    import numpy as np
    from PIL import Image

    from realmspinner.service import jobs as svc_jobs

    arr = np.full((64, 64, 3), 230, dtype=np.uint8)
    arr[16:48, 16:48] = (20, 30, 40)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, "PNG")
    return svc_jobs.import_reference(svc, buf.getvalue(), name="subject")["id"]


def test_a_failure_for_an_older_stamp_does_not_latch_the_newer_one(svc):
    """``on_task_failed`` latched ``failed_stamp = state.stamp`` -- the stamp
    ``pump`` had already advanced to after a rewrite of ``input.png`` -- so the
    failure of the task that read the *old* bytes suppressed the ask for the
    *new* ones. The failure is latched against the stamp its task was tagged
    with at submit."""
    from realmspinner.studio import matte_preview

    job_id = _png_job(svc)
    src = svc.job_dir(job_id) / "input.png"
    ctx = _MatteCtx(svc)
    matte_preview.open_for(ctx, job_id, {})
    state = matte_preview.pump(ctx)
    old_stamp = state.stamp
    submitted_tag = ctx.submitted[-1][1].get("tag")

    # input.png is rewritten (Fix matte round trip) and the frame notices.
    ns = old_stamp + 5_000_000_000
    os.utime(src, ns=(ns, ns))
    matte_preview.pump(ctx)
    assert state.stamp != old_stamp

    # Now the first task's failure lands, carrying what it was submitted with.
    matte_preview.on_task_failed(
        ctx, SimpleNamespace(key=matte_preview.key(job_id), tag=submitted_tag)
    )
    n = len(ctx.submitted)
    matte_preview.pump(ctx)
    assert len(ctx.submitted) == n + 1, "the new bytes were never asked about"


def test_a_failure_for_the_current_stamp_still_latches(svc):
    from realmspinner.studio import matte_preview

    job_id = _png_job(svc)
    ctx = _MatteCtx(svc)
    matte_preview.open_for(ctx, job_id, {})
    state = matte_preview.pump(ctx)
    tag = ctx.submitted[-1][1].get("tag")
    matte_preview.on_task_failed(ctx, SimpleNamespace(key=matte_preview.key(job_id), tag=tag))
    n = len(ctx.submitted)
    matte_preview.pump(ctx)
    assert len(ctx.submitted) == n
    assert state._tried_and_failed


def test_a_preview_that_remember_refused_as_racily_clean_is_not_adopted(svc):
    """``on_task_done`` adopted a preview ``svc_matte.remember`` had refused,
    which is the very case the racily-clean rule exists for: the cutout on
    screen could describe pixels a later write had already replaced, and the
    stamp would match forever."""
    from realmspinner.service import matte as svc_matte
    from realmspinner.studio import matte_preview

    job_id = _png_job(svc)
    ctx = _MatteCtx(svc)
    matte_preview.open_for(ctx, job_id, {})
    state = matte_preview.pump(ctx)
    fresh = svc_matte.replace_stamp(svc_matte.preview(svc, job_id), time.time_ns())
    matte_preview.on_task_done(
        ctx, SimpleNamespace(key=matte_preview.key(job_id), result=fresh)
    )
    assert job_id not in state.cache
    assert state.preview is None


# --- create-30: sweep comment -------------------------------------------------


def test_sweep_comments_quote_hole_worst_figures_with_their_corpus_date():
    source = (SRC / "sweep.py").read_text(encoding="utf-8")
    quoted = [line for line in source.splitlines() if "AUC" in line or "0.115" in line]
    assert quoted, "the comment this test polices has gone; update the test with it"
    text = "\n".join(source.splitlines())
    # A figure without its corpus means nothing: both corpora are named, dated.
    assert "2026-08-07" in text and "2026-08-09" in text
    assert "0.756" in text
    # The superseded figure is never presented as present state.
    assert "84-verdict" not in text
    # No plan-item id from src/ (quality.py shared the same citation).
    for name in ("sweep.py", "studio/quality.py"):
        body = (SRC / name).read_text(encoding="utf-8")
        assert not re.search(r"\(P\d{2,3}\)", body), name
