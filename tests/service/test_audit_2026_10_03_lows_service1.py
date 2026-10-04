"""Regression tests for the 2026-10-03 audit's Low service findings (service-17..36).

Each test's name is the claim, and each one fails against the code it was
written for -- except where the finding is an evidence gap (service-36, the
sweep-site table gate), whose new test *is* the fix.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from realmspinner import models
from realmspinner.service.errors import Conflict, Failed, Invalid, NotFound

SRC = Path(__file__).resolve().parents[2] / "src" / "realmspinner"

# --- service-17: remove_lora with a locked adapter file -----------------------------

_ARCHIVE = (
    len(b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}').to_bytes(8, "little")
    + b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    + b"\x00" * 4
)


@pytest.fixture
def clean_registry():
    before = dict(models.STYLE_LORAS)
    yield
    models.STYLE_LORAS.clear()
    models.STYLE_LORAS.update(before)


def _imported(svc, tmp_path):
    from realmspinner.service import loras as svc_loras

    adapter = tmp_path / "mystyle.safetensors"
    adapter.write_bytes(_ARCHIVE)
    return svc_loras.import_lora(svc, adapter, label="Cosmos")


def test_remove_lora_keeps_registry_and_manifest_in_step_when_the_file_is_locked(
    svc, tmp_path, monkeypatch, clean_registry
):
    """service-17: a sharing violation on the unlink raised a raw PermissionError
    after the manifest had been rewritten, so the picker kept a style whose
    manifest row was gone."""
    from realmspinner.service import loras as svc_loras

    out = _imported(svc, tmp_path)
    key, filename = out["key"], out["filename"]
    real_unlink = Path.unlink

    def locked(self, *args, **kwargs):
        if self.name == filename:
            raise PermissionError(32, "The process cannot access the file")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)
    with pytest.raises(Failed):
        svc_loras.remove_lora(svc, key)
    monkeypatch.undo()
    assert key in models.STYLE_LORAS
    assert [m["key"] for m in svc_loras.imported(svc)] == [key], (
        "the manifest row was removed although the file could not be"
    )
    assert (Path(svc.config.t2i_model_root) / "loras" / filename).exists()


def test_remove_lora_refuses_while_a_job_names_the_style(svc, tmp_path, clean_registry):
    """service-17: a queued job on the removed style failed later, at load."""
    from realmspinner.service import loras as svc_loras

    out = _imported(svc, tmp_path)
    key = out["key"]
    svc.store.create("text", "a barrel", {"seed": 1, "style_lora": key})
    with pytest.raises(Conflict) as caught:
        svc_loras.remove_lora(svc, key)
    assert caught.value.field == "key"
    assert key in models.STYLE_LORAS
    assert [m["key"] for m in svc_loras.imported(svc)] == [key]


# --- service-18: a truncated PNG is not a hand edit ---------------------------------


def test_save_edited_image_refuses_a_truncated_png(svc):
    """service-18: only the header was read, so a truncated PNG was published
    onto the served ``input.png`` and the call answered ``{"ok": True}``."""
    import io
    import os

    from realmspinner.service import files as svc_files

    job_id = svc.store.create("text", "a barrel", {"seed": 1}, stage="reference", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    original = Image.new("RGB", (32, 32), (200, 200, 200))
    original.save(job_dir / "input.png")
    before = (job_dir / "input.png").read_bytes()

    noisy = Image.frombytes("RGB", (64, 64), os.urandom(64 * 64 * 3))
    buffer = io.BytesIO()
    noisy.save(buffer, format="PNG")
    whole = buffer.getvalue()
    truncated = whole[: int(len(whole) * 0.7)]

    with pytest.raises(Invalid) as caught:
        svc_files.save_edited_image(svc, job_id, truncated)
    assert caught.value.field == "image"
    assert (job_dir / "input.png").read_bytes() == before, "the truncated bytes were published"
    # And the whole one still goes through: the refusal is the decode, not the door.
    assert svc_files.save_edited_image(svc, job_id, whole) == {"ok": True}


# --- service-19: the sweep refusals name a control ----------------------------------


def _sweep_plan(**changes):
    from realmspinner.service.sweeps import SweepPlan

    base = {"label": "t", "prompt": "a barrel"}
    base.update(changes)
    return SweepPlan(**base)


def test_a_sweep_that_plans_too_many_units_names_the_axes_field(svc):
    from realmspinner.service import sweeps as svc_sweeps

    plan = _sweep_plan(seeds=tuple(range(svc_sweeps.MAX_UNITS + 1)))
    with pytest.raises(Invalid) as caught:
        svc_sweeps.validate_sweep(svc, plan)
    assert "limit" in str(caught.value)
    assert caught.value.field == "axes"


def test_a_sweep_that_plans_no_units_names_the_axes_field(svc):
    from realmspinner.service import sweeps as svc_sweeps

    with pytest.raises(Invalid) as caught:
        svc_sweeps._validate(svc, _sweep_plan(), [])
    assert caught.value.field == "axes"


@pytest.mark.parametrize("seed", ["abc", 1.5, None])
def test_a_sweep_with_a_malformed_seed_is_refused_by_name_not_truncated_or_raised(svc, seed):
    """service-19: ``int(unit.seed)`` raised a raw ValueError for "abc" and
    silently ran 1.5 as seed 1."""
    from realmspinner.service import sweeps as svc_sweeps

    with pytest.raises(Invalid) as caught:
        svc_sweeps.validate_sweep(svc, _sweep_plan(seeds=(seed,)))
    assert caught.value.field == "seeds"


def test_a_sweep_with_no_prompt_text_is_refused_by_name_not_raised(svc):
    from realmspinner.service import sweeps as svc_sweeps

    with pytest.raises(Invalid) as caught:
        svc_sweeps.validate_sweep(svc, _sweep_plan(prompt=None))
    assert caught.value.field == "prompt"


# --- service-20: a blank REALMSPINNER_T2I_DIR is unset ------------------------------


def test_a_whitespace_t2i_dir_is_unset_not_the_checkout(monkeypatch):
    from realmspinner import config as config_module

    monkeypatch.setenv("REALMSPINNER_T2I_DIR", "   ")
    assert config_module.Config().t2i_turbo_dir is None
    monkeypatch.setenv("REALMSPINNER_T2I_DIR", "")
    assert config_module.Config().t2i_turbo_dir is None


def test_migrate_reads_a_whitespace_root_variable_as_unset_like_config(tmp_path, monkeypatch):
    """service-20: ``migrate._pending`` tested the raw variable, so a blank
    ``REALMSPINNER_DATA_DIR`` made it skip a root Config had already fallen back
    to home for."""
    from realmspinner import config as config_module
    from realmspinner import migrate

    checkout = tmp_path / "checkout"
    (checkout / "assets").mkdir(parents=True)
    (checkout / "assets" / "jobs.sqlite").write_bytes(b"x")
    monkeypatch.setattr(migrate, "PROJECT_ROOT", checkout)
    monkeypatch.setenv("REALMSPINNER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("REALMSPINNER_DATA_DIR", "   ")
    for name in ("REALMSPINNER_BENCH_DIR", "REALMSPINNER_PALETTE_DIR", "REALMSPINNER_T2I_ROOT"):
        monkeypatch.delenv(name, raising=False)
    pending = migrate._pending(config_module.Config())
    assert [legacy for legacy, _dest in pending] == [checkout / "assets"]


# --- service-21: an id with a trailing newline is not an id -------------------------


def test_an_id_with_a_trailing_newline_is_refused():
    from realmspinner.kernels.rig import store
    from realmspinner.service.validation import JOB_ID_RE, check_job_id

    assert check_job_id("0123456789ab") is None
    with pytest.raises(NotFound):
        check_job_id("0123456789ab\n")
    assert store.is_valid_id("0123456789ab")
    assert not store.is_valid_id("0123456789ab\n")
    assert JOB_ID_RE.match("0123456789ab\n") is None


# --- service-22: zero is a value, refused by name -----------------------------------


def _mesh_job(svc) -> str:
    from realmspinner.service import jobs as svc_jobs

    job_id = svc_jobs.create_job(svc, kind="text", prompt="a knight")["id"]
    svc.store.set_status(job_id, "done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"glb")
    return job_id


def test_create_sheet_refuses_a_zero_frame_size_rather_than_defaulting_it(svc):
    from realmspinner.service import sheets as svc_sheets

    job_id = _mesh_job(svc)
    with pytest.raises(Invalid) as caught:
        svc_sheets.create_sheet(svc, job_id, frame_size=0)
    assert caught.value.field == "frame_size"


def test_create_sheet_refuses_zero_directions_rather_than_defaulting_them(svc):
    from realmspinner.service import sheets as svc_sheets

    job_id = _mesh_job(svc)
    with pytest.raises(Invalid) as caught:
        svc_sheets.create_sheet(svc, job_id, yaws=0)
    assert caught.value.field == "yaws"


def test_create_pixel_sheet_refuses_a_non_numeric_strength_by_name(svc):
    from realmspinner.kernels.rig import store
    from realmspinner.service import sheets as svc_sheets

    job_id = _mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    sheet_id = store.new_id()
    png = store.sheet_png_path(job_dir, sheet_id)
    png.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (1024, 128), (0, 0, 0, 0)).save(png)
    meta = {
        "version": 1, "id": sheet_id, "name": "turnaround", "source_job": job_id,
        "created": 1.0, "image": png.name, "frame_size": 128, "columns": 8, "rows": 1,
        "width": 1024, "height": 128, "yaws": [i * 45.0 for i in range(8)],
        "poses": [{"id": None, "name": "rest"}], "cells": [],
    }
    store.sheet_path(job_dir, sheet_id).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(Invalid) as caught:
        svc_sheets.create_pixel_sheet(svc, job_id, sheet_id, strength="strong")
    assert caught.value.field == "strength"


# --- service-23: palettes refuse before reading -------------------------------------


def test_palettes_load_refuses_an_oversized_file_before_reading_it(svc, tmp_path, monkeypatch):
    from realmspinner.service import palettes

    directory = tmp_path / "palettes"
    directory.mkdir(exist_ok=True)
    svc.config.palette_dir = directory
    big = directory / "huge.hex"
    with big.open("wb") as fh:
        fh.truncate(palettes.MAX_PALETTE_BYTES + 1)
    read: list[str] = []
    real_read_text = Path.read_text

    def spy(self, *args, **kwargs):
        read.append(self.name)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    with pytest.raises(Invalid) as caught:
        palettes.load(svc.config, "huge")
    assert caught.value.field == "palette"
    assert "huge.hex" not in read, "the whole file was read before the size was checked"


# --- service-24: the entry points the text names exist ------------------------------


def test_every_cli_verb_a_source_comment_names_exists():
    """service-24: two comments justified a design with ``realmspinner library
    backup`` and ``realmspinner library verify``, neither of which ``cli.py`` has."""
    cli_text = (SRC / "cli.py").read_text(encoding="utf-8")
    choices = re.search(r"choices=\[([^\]]*)\]", cli_text)
    assert choices is not None
    verbs = set(re.findall(r'"([a-z]+)"', choices.group(1)))
    assert {"doctor", "sweep", "mcp"} <= verbs
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for verb in re.findall(r"`realmspinner ([a-z][a-z-]*)", line):
                if verb not in verbs:
                    offenders.append(f"{path.relative_to(SRC)}:{number}: realmspinner {verb}")
    assert not offenders, offenders


def test_the_sweep_cancel_reason_does_not_send_an_installed_build_to_a_repo_path(svc):
    """service-24: the reason (stored in ``error``, shown in the library) said only
    "re-queue with scripts/sweep_refill.py", which an installed build does not carry."""
    from realmspinner.service import sweeps as svc_sweeps

    # A style LoRA in the base, or ``normalize`` drops the weight and the two units
    # are the same job, which admission refuses.
    plan = _sweep_plan(
        seeds=(1,),
        base={"style_lora": "render3d"},
        axes=(svc_sweeps.Axis("lora_weight", (0.6,)),),
    )
    result = svc_sweeps.create_sweep(svc, plan)
    units = {u["sweep_unit"]: u for u in svc.store.sweep_jobs(result["id"])}
    svc.store.claim(units["baseline s1"]["id"])
    svc.store.finish(units["baseline s1"]["id"], "error", "boom")
    svc_sweeps.on_job_failed(svc, svc.store.get(units["baseline s1"]["id"]))
    reason = svc.store.get(units["lora_weight=0.6 s1"]["id"])["error"]
    assert "new sweep" in reason, reason
    assert "source checkout" in reason, reason


# --- service-25: the scoped bucket's documented field list --------------------------


def test_the_scoped_bucket_field_list_in_the_docstring_matches_the_code():
    from realmspinner.service import findings

    bucket = findings._marginals(
        [{"vector": {"lora_weight": 0.6}, "verdict": "accept", "grade": 2}],
        [{"vector": {"lora_weight": 0.6}, "metrics": {"hole": 0.1}}],
        full=False,
    )["lora_weight"]["0.6"]
    doc = findings._marginals.__doc__ or ""
    keys = [k for k in bucket if k != "metrics"]
    assert set(keys) == {"n", "accepts", "accept_rate", "wilson_low", "graded_n", "mean_grade"}
    for key in keys:
        assert f"``{key}``" in doc, f"{key} is kept per scoped bucket but not documented"
    assert "metrics" in bucket
    assert "``metrics``" in doc
    assert "only the four" not in doc


# --- service-26: one pack operation at a time ---------------------------------------


class _FakePackService:
    def __init__(self, home: Path) -> None:
        self.config = type("C", (), {"home": home})()


def test_a_second_pack_install_is_refused_while_one_runs(tmp_path, monkeypatch):
    from realmspinner import packs
    from realmspinner.service import packs as svc_packs

    manifest = tmp_path / packs.MANIFEST_NAME
    manifest.write_text(
        json.dumps(
            {
                "version": packs.MANIFEST_VERSION,
                "wheels": [
                    {
                        "filename": "bpy-5.2.0-cp313-cp313-win_amd64.whl",
                        "url": "https://example.invalid/bpy.whl",
                        "size_bytes": 10,
                        "sha256": "a" * 64,
                        "installed_bytes": 0,
                        "packs": ["rig"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(svc_packs, "manifest_path", lambda: manifest)
    monkeypatch.setattr(svc_packs, "installed_versions", dict)
    started, gate = tmp_path / "started", tmp_path / "gate"
    body = f"""
    import json, os, sys, time
    spec = json.loads(sys.stdin.read())
    open({str(started)!r}, "w").write("1")
    while not os.path.exists({str(gate)!r}):
        time.sleep(0.05)
    done = {{"ok": True, "collected": [], "installed": []}}
    open(spec["result_path"], "w").write(json.dumps(done))
    """
    monkeypatch.setattr(
        svc_packs, "worker_argv", lambda: [sys.executable, "-c", textwrap.dedent(body)]
    )
    svc = _FakePackService(tmp_path / "home")
    errors: dict[str, BaseException | None] = {"first": None, "second": None}

    def run(name: str, fn) -> None:
        try:
            fn(svc, ["rig"])
        except BaseException as exc:  # noqa: BLE001 - the test reads it back
            errors[name] = exc

    first = threading.Thread(target=run, args=("first", svc_packs.install))
    first.start()
    deadline = time.monotonic() + 20
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert started.exists(), "the first install never reached its child"
    second = threading.Thread(target=run, args=("second", svc_packs.repair))
    second.start()
    second.join(timeout=3)
    still_running = second.is_alive()
    gate.write_text("go")  # always release the child, whatever the second call did
    first.join(timeout=20)
    second.join(timeout=20)
    assert not still_running, "a second pack operation ran alongside the first"
    assert isinstance(errors["second"], Conflict), errors["second"]
    assert errors["first"] is None, errors["first"]


# --- service-28: the reroll loop spends one window across its draws -----------------


@pytest.mark.asyncio
async def test_the_reference_reroll_loop_spends_one_sampling_window_across_its_draws(
    tmp_path, fake_pipelines, monkeypatch
):
    """service-28: every redraw fed its own ``(i, n)`` to ``_t2i_step``, so the
    creep floor held the bar at the top of the window for draws two and three."""
    import asyncio

    import realmspinner.pipelines.reference as reference_mod
    from realmspinner.config import Config
    from realmspinner.db import JobStore
    from realmspinner.queue import Worker

    monkeypatch.setattr(
        reference_mod,
        "measure_file",
        lambda path: reference_mod.Report(ok=False, reasons=("the subject runs off the frame",)),
    )
    steps: list[tuple[int, int]] = []
    real = Worker._step_progress

    def record(self, job_id, phase, label, step, total):
        if phase == "t2i_sample":
            steps.append((step, total))
        return real(self, job_id, phase, label, step, total)

    monkeypatch.setattr(Worker, "_step_progress", record)
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        reference_retries=2,
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    job_id = store.create("text", "a barrel", {"seed": 5}, stage="reference")
    w.start()
    deadline = time.monotonic() + 20
    while store.get(job_id)["status"] != "done" and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    await w.shutdown()
    store.close()

    assert len(w._text2image.seeds) == 3, "the fixture was meant to draw three times"
    per_draw = w._text2image.steps
    assert steps == [(i + 1, 3 * per_draw) for i in range(3 * per_draw)], steps


# --- service-30: the facade names every sibling it re-exports -----------------------


def test_jobs_facade_docstring_names_every_sibling_it_reexports():
    from realmspinner.service import jobs

    source = (SRC / "service" / "jobs.py").read_text(encoding="utf-8")
    siblings = sorted(set(re.findall(r"^from \.(_jobs_\w+) import", source, re.MULTILINE)))
    doc = jobs.__doc__ or ""
    for name in siblings:
        assert f"``{name}``" in doc, f"{name} is re-exported but not named in the docstring"
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    assert f"over {words[len(siblings)]} siblings" in doc


# --- service-31: a request names a bounded number of references ---------------------


def test_a_request_with_too_many_references_is_refused_at_the_door():
    from realmspinner import generation

    request = generation.GenerationRequest(
        generation_type="3d_model",
        prompt="a knight",
        references=tuple(f"ref{i}.png" for i in range(generation.MAX_INPUT_REFERENCES + 1)),
    )
    issues = generation.validate_request(request)
    assert any(issue.field == "references" for issue in issues)
    ok = generation.GenerationRequest(
        generation_type="3d_model",
        prompt="a knight",
        references=tuple(f"ref{i}.png" for i in range(generation.MAX_INPUT_REFERENCES)),
    )
    assert not any(issue.field == "references" for issue in generation.validate_request(ok))


# --- service-36: every dispatched kind is in the three silent-failure tables --------

#: Kinds whose ``estimate_parts`` is deliberately zero: Blender or numpy, out of
#: process or CPU-side, costing the VRAM budget nothing (vram.py's own comment).
_DELIBERATE_ZERO_VRAM = {"rig", "sheet", "charsheet", "remesh"}

#: Kinds ``_discard_artifacts`` serves with its generic (mesh-filename) arm.
_GENERIC_DISCARD = {"text", "image"}


def _kinds_named_in(function: ast.AST) -> set[str]:
    """Every string a ``job["kind"]`` is compared with, ``==`` or ``in (...)``."""
    found: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        is_kind = (
            isinstance(left, ast.Subscript)
            and isinstance(left.slice, ast.Constant)
            and left.slice.value == "kind"
        )
        if not is_kind:
            continue
        for comparator in node.comparators:
            if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                found.add(comparator.value)
            elif isinstance(comparator, (ast.Tuple, ast.List)):
                found.update(
                    e.value
                    for e in comparator.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                )
    return found


def _function(path: Path, name: str) -> ast.AST:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def _site_gaps(
    kinds: set[str], discard_kinds: set[str], phases: set[str], zero: set[str], estimate
) -> list[str]:
    gaps: list[str] = []
    for kind in sorted(kinds):
        if kind not in phases:
            gaps.append(f"{kind}: no progress._PHASES_BY_KIND row")
        priced = estimate(kind)
        if kind in zero:
            if priced != 0.0:
                gaps.append(f"{kind}: listed as a deliberate zero but vram prices it {priced}")
        elif priced <= 0.0:
            gaps.append(f"{kind}: vram.estimate_parts prices it at zero (admission is a no-op)")
        if kind not in discard_kinds and kind not in _GENERIC_DISCARD:
            gaps.append(f"{kind}: no _discard_artifacts arm (cancel falls to the mesh arm)")
    return gaps


def _estimate(kind: str) -> float:
    from realmspinner import vram

    return vram.estimate_parts(kind, "model", {}, exclusive=False)[0]


def test_every_dispatched_job_kind_is_in_vram_discard_and_progress_tables():
    from realmspinner import progress

    dispatched = _kinds_named_in(_function(SRC / "_q_generate.py", "_generate"))
    # ``text`` and ``image`` are the fall-through the chain above them leaves.
    dispatched |= {"text", "image"}
    discard = _kinds_named_in(_function(SRC / "_q_jobs.py", "_discard_artifacts"))
    assert {"rig", "sheet", "charsheet", "pixel_sheet", "tile_sheet", "music"} <= dispatched
    gaps = _site_gaps(
        dispatched, discard, set(progress._PHASES_BY_KIND), _DELIBERATE_ZERO_VRAM, _estimate
    )
    assert not gaps, gaps
    # Bidirectional: a kind a table keeps that nothing dispatches is a stale row.
    assert set(progress._PHASES_BY_KIND) <= dispatched
    assert discard <= dispatched


def test_the_site_table_gate_catches_a_kind_missing_from_each_table():
    """The gate above passes today, so prove it is not vacuous: a made-up kind
    that is dispatched but registered nowhere fails all three tables at once."""
    gaps = _site_gaps({"hologram"}, set(), set(), set(), _estimate)
    assert len(gaps) == 3, gaps

