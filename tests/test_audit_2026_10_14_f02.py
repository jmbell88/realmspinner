"""The 2026-10-14 audit, fixer F02: the remesh bake size, the loop-roll temp, the
sheet-timeout wording, the tile jobs' ``ref.png`` probes, the changelog's
comments, the installed app's import failure, the sweep's numeric flags, the
environment readout and its validators, and the backup destination.

One file at the top of ``tests/``: ``tests/`` carries no ``__init__.py``, so a
second file sharing this basename in another directory would collide.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from realmspinner import changelog, cli, sweep
from realmspinner import config as config_mod
from realmspinner.config import Config, effective
from realmspinner.db import JobStore
from realmspinner.kernels.rig import blender_spec
from realmspinner.pipelines import blender_run, remesh, retexture, tileatlas, tilesheet
from realmspinner.queue import Worker
from realmspinner.service import jobs as svc_jobs
from realmspinner.service import library, tilesheets
from realmspinner.service.errors import Invalid

# --- helpers -------------------------------------------------------------------


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    cfg = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models",
    )
    store = JobStore(cfg.db_path)
    w = Worker(cfg, store)
    yield w
    store.close()


async def _wait_for_end(worker: Worker, job_id: str, timeout: float = 30.0) -> dict:
    worker.start()
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if worker.store.get(job_id)["status"] in ("done", "error", "cancelled"):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("job did not finish before timeout")
    finally:
        await worker.shutdown()
    return worker.store.get(job_id)


@pytest.fixture
def clean_invalid_env(monkeypatch):
    """``INVALID_ENV`` is process-global; every test that reads it starts empty."""
    monkeypatch.setattr(config_mod, "INVALID_ENV", [])
    return config_mod.INVALID_ENV


# --- pipelines-12: a remesh left on "match the mesh" is bounded -----------------


def test_the_remesh_and_retexture_ceilings_are_the_same_table():
    """The clamp lives in ``retexture`` and the door that refuses an explicit
    size lives in ``remesh``; they have to agree or "match the mesh" would
    bake at a size the remesh door itself would refuse."""
    assert tuple(remesh.TEXTURE_SIZES) == tuple(retexture.TEXTURE_SIZES)


async def test_a_remesh_left_on_match_the_mesh_bakes_no_larger_than_the_largest_texture_size(
    worker, monkeypatch
):
    """The 2026-10-14 audit, finding pipelines-mesh-01: with ``texture_size``
    unset the queued remesh baked at the raw albedo size, so a 4096 atlas sent
    Blender a five-map float bake at 4096 squared with nothing refusing it."""
    from realmspinner.pipelines import postprocess

    monkeypatch.setattr(retexture, "atlas_size", lambda path: 4096)
    monkeypatch.setattr(postprocess, "normalize_glb", lambda *a, **k: {"scale": 1.0})
    seen: list[dict] = []

    def fake(spec, *, on_progress=None, on_start=None, timeout=0.0):
        seen.append(spec)
        Path(spec["out_glb"]).write_bytes(b"new-model")
        return {"ok": True, "method": "quadriflow", "faces": 8000, "faces_before": 300000,
                "quads": 0.97, "texture_size": spec["texture_size"], "metallic": 0.0}

    monkeypatch.setattr(blender_run, "run_worker", fake)
    source = worker.store.create("text", "a crate", {"seed": 1, "size_m": 1.0})
    job_dir = worker.config.job_dir(source)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"old-model")
    (job_dir / "source.glb").write_bytes(b"reconstruction")
    worker.store.set_status(source, "done")
    job_id = worker.store.create(
        "remesh", "a crate", {"source_job": source, "target_triangles": 5000}
    )
    row = await _wait_for_end(worker, job_id)
    assert row["status"] == "done", row.get("error")
    assert seen[0]["texture_size"] == max(remesh.TEXTURE_SIZES)


async def test_the_lowpoly_repair_path_left_on_match_the_mesh_bakes_no_larger_than_the_largest_size(
    worker, monkeypatch, tmp_path
):
    """The same finding at the in-job ``_lowpoly`` site."""
    from realmspinner import tiercheck

    monkeypatch.setattr(retexture, "atlas_size", lambda path: 8192)
    monkeypatch.setattr(tiercheck, "survey", lambda path: {})
    seen: dict = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here; only the requested size is the subject")

    monkeypatch.setattr(blender_spec, "remesh_spec", spy)
    job_id = worker.store.create("text", "a crate", {"seed": 1})
    glb = tmp_path / "model.glb"
    glb.write_bytes(b"glb")
    worker.config.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    await worker._lowpoly(job_id, glb, {"lowpoly_triangles": 5000})
    assert seen["texture_size"] == max(remesh.TEXTURE_SIZES)


# --- service-kinds-07: a failed loop roll leaves no staging wav ----------------


def _wav_bytes() -> bytes:
    import io
    import wave

    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(bytes(range(256)) * 16)
    return out.getvalue()


@pytest.mark.parametrize("how", ["replace", "write"])
def test_a_failed_loop_roll_leaves_no_staging_wav_behind(tmp_path, monkeypatch, how):
    """The 2026-10-14 audit, finding service-kinds-07: the roll stages through
    ``.track.wav.tmp`` and the music discard arm never lists it, so a write or
    replace that raised (disk full, an antivirus lock) left tens of MB in the
    job directory until the row was deleted."""
    from realmspinner import _q_music

    track = tmp_path / "track.wav"
    track.write_bytes(_wav_bytes())
    before = track.read_bytes()
    real_write = Path.write_bytes

    def boom_replace(self, target):
        raise OSError("the replace failed")

    def half_write(self, data):
        real_write(self, data[: len(data) // 2])
        raise OSError("disk full")

    if how == "replace":
        monkeypatch.setattr(Path, "replace", boom_replace)
    else:
        monkeypatch.setattr(Path, "write_bytes", half_write)
    with pytest.raises(OSError):
        _q_music._stage_rolled_wav(track, 0.01)
    monkeypatch.undo()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["track.wav"]
    assert track.read_bytes() == before


# --- poser-jobs-06: the sheet timeout names what it bounds ---------------------


def _comment_block_above(source: str, needle: str) -> str:
    lines = source.splitlines()
    index = next(i for i, line in enumerate(lines) if needle in line)
    block = []
    for line in reversed(lines[:index]):
        if not line.strip().startswith("#"):
            break
        block.append(line)
    return "\n".join(reversed(block))


def test_the_sheet_timeout_documentation_names_the_runs_it_bounds():
    """The 2026-10-14 audit, finding poser-jobs-06: ``sheet_timeout`` was
    described as one sprite-sheet render ("8 yaws times however many poses"),
    but the sprite-sheet pipeline is gone: it bounds the two Blender runs of a
    re-texture and the rig's deformation-QA sheet. And the re-texture comment
    called its view render "six frames" while ``retexture.VIEWS`` is ten."""
    from realmspinner import _q_sprite

    comment = _comment_block_above(inspect.getsource(config_mod), "sheet_timeout: float")
    assert "yaws" not in comment
    assert "re-texture" in comment and "deformation" in comment

    sprite = inspect.getsource(_q_sprite)
    assert "A render of six frames" not in sprite
    assert len(retexture.VIEWS) == 10


# --- plotter-tiles-06: the tileset docstring cites by name ---------------------


def test_tileset_stage_docstring_cites_text2image_by_name_not_line():
    """The 2026-10-14 audit, finding plotter-tiles-06: ``text2image.py:99``,
    ``:975-988`` and ``:955`` had drifted onto unrelated code."""
    from realmspinner import _q_tileset
    from realmspinner.pipelines import text2image

    doc = _q_tileset.__doc__ or ""
    assert not re.search(r"text2image\.py:\d+", doc)
    assert "circular_padding" in doc
    assert callable(text2image.circular_padding)


# --- plotter-tiles-09: a directory named ref.png is not a reference ------------


def _tile_params(mode_block: dict) -> dict:
    return {
        "seed": 100,
        "base_model": "sdxl_cfg",
        "colors": 64,
        "negative_prompt": "",
        "sheet": mode_block,
    }


async def test_a_directory_named_ref_png_is_not_a_tile_set_reference(worker):
    seeds = tileatlas.material_seeds(100, 2)
    geom = tileatlas.material_geometry(32, "top_down", 2)
    block = {
        "version": 3, "mode": "materials", "tile_w": geom.tile_w, "tile_h": geom.tile_h,
        "projection": "top_down", "columns": geom.columns, "rows": geom.rows,
        "layout": "grid",
        "materials": [
            {"index": i, "prompt": p, "variant": 1, "seed": s}
            for i, (p, s) in enumerate(zip(("moss", "gravel"), seeds, strict=True))
        ],
        "variants": 1, "style_lock": False,
    }
    job_id = worker.store.create(
        "tile_sheet", "a dungeon", _tile_params(block), stage="tilesheet"
    )
    job_dir = worker.config.job_dir(job_id)
    (job_dir / "ref.png").mkdir(parents=True)
    row = await _wait_for_end(worker, job_id)
    assert row["status"] == "done", row.get("error")
    assert worker._text2image.conditionings == [None, None]


async def test_a_directory_named_ref_png_is_not_a_tile_sheet_reference(worker):
    geom = tilesheet.geometry(16, "top_down")
    block = {
        "version": 2, "tile_w": geom.tile_w, "tile_h": geom.tile_h,
        "projection": geom.view, "columns": geom.columns, "rows": geom.rows,
    }
    params = _tile_params(block)
    params.update(control="canny", colors=8)
    job_id = worker.store.create("tile_sheet", "a mine", params, stage="tilesheet")
    (worker.config.job_dir(job_id) / "ref.png").mkdir(parents=True)
    row = await _wait_for_end(worker, job_id)
    assert row["status"] == "done", row.get("error")
    assert worker._text2image.conditionings
    assert all(c.ip_image is None for c in worker._text2image.conditionings)


def test_a_directory_named_ref_png_does_not_make_a_rerun_demand_the_ip_adapter(svc):
    """The rerun door's ``with_reference`` read the same ``exists()``: a
    directory named ``ref.png`` made a plain reroll refuse for an adapter the
    worker would never load."""
    from realmspinner import models

    made = tilesheets.create_tile_sheet(
        svc,
        prompt="a temperate ruin",
        tile_size=32,
        view="top_down",
        mode=tilesheets.MODE_MATERIALS,
        prompt_items=("mossy cobblestone", "cracked dry mud"),
    )
    svc.store.set_status(made["id"], "done")
    (svc.job_dir(made["id"]) / "ref.png").mkdir(parents=True, exist_ok=True)
    adapter = models.IP_ADAPTERS["plus"]
    (
        svc.config.t2i_model_root / adapter.dir_name / adapter.subfolder / adapter.weight_name
    ).unlink()
    new = svc_jobs.rerun_job(svc, made["id"], mode="reroll")
    assert not (svc.job_dir(new["id"]) / "ref.png").exists()


# --- docs-A-B-15: the changelog's comments describe the real input -------------


def test_changelog_parse_hands_emphasis_regex_single_line_text(monkeypatch):
    """``flush`` joins a wrapped bullet's continuation lines with a space, so
    the emphasis regex never sees a newline -- whatever the file's wrapping."""
    seen: list[str] = []

    class Spy:
        def __init__(self, inner):
            self._inner = inner

        def sub(self, repl, text):
            seen.append(text)
            return self._inner.sub(repl, text)

    monkeypatch.setattr(changelog, "_EMPHASIS", Spy(changelog._EMPHASIS))
    releases = changelog.parse(
        "## 1.0.0 - 2026-01-01\n- **A bold lead that wraps\n  onto a second line.** The rest.\n"
    )
    assert releases[0].bullets == ("A bold lead that wraps onto a second line. The rest.",)
    assert seen and all("\n" not in text for text in seen)


def test_the_changelog_module_does_not_claim_a_wrap_the_file_does_not_have():
    doc = changelog.__doc__ or ""
    assert "hard-wrapped at 80 columns like everything else" not in doc
    # The ``re.S`` rationale is stated as defensive: ``flush`` joins the lines.
    comment = _comment_block_above(inspect.getsource(changelog), "    re.S,")
    assert "defensive" in comment
    assert "``.`` stopped\n    # at the newline, the opening" not in comment


# --- shell-boot-03: a failed window import is shown, not swallowed -------------


def _break_the_window_import(monkeypatch):
    from realmspinner import instance

    monkeypatch.setattr(sys, "argv", ["realmspinner"])
    # ``None`` in ``sys.modules`` makes ``from .studio.main import run`` raise
    # ``ImportError`` exactly as a missing DLL under it would.
    monkeypatch.setitem(sys.modules, "realmspinner.studio.main", None)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(instance, "alert", lambda title, message: alerts.append((title, message)))
    return alerts


def test_a_failed_studio_import_raises_a_native_alert_not_just_a_systemexit_message(monkeypatch):
    """The 2026-10-14 audit, finding shell-boot-03: the installed app runs under
    ``pythonw`` with stderr on the null device, so the ``SystemExit`` text went
    nowhere and double-clicking the icon did nothing at all."""
    alerts = _break_the_window_import(monkeypatch)
    monkeypatch.setattr(cli, "_installed_build", lambda: True)
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert len(alerts) == 1
    title, message = alerts[0]
    assert "Realmspinner" in title
    # The installed build has no ``uv``: the remedy is the installer.
    assert "uv sync" not in message and "uv sync" not in str(raised.value)
    assert "installer" in message.lower()


def test_a_source_checkout_keeps_the_uv_remedy_and_does_not_pop_a_box_over_a_console(
    monkeypatch,
):
    alerts = _break_the_window_import(monkeypatch)
    monkeypatch.setattr(cli, "_installed_build", lambda: False)
    monkeypatch.setattr(cli, "_launched_without_console", lambda: False)
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert alerts == []
    assert "uv sync --extra studio" in str(raised.value)


# --- create-workspace-05: the sweep's numeric flags ----------------------------


def test_sweep_refuses_a_resolution_the_service_door_refuses():
    """The 2026-10-14 audit, finding create-workspace-05: ``--resolution`` was
    any int although the door allows three values, and ``--audit-resolution``
    was any int, so a bad one failed minutes into a GPU run (0 measured every
    band as worst 0.0 and recommended nothing)."""
    from realmspinner.service.validation import ALLOWED_RESOLUTIONS

    for good in ALLOWED_RESOLUTIONS:
        assert sweep.parse_resolution(good) == good
    for bad in (0, -1, 700, 4096):
        with pytest.raises(ValueError, match="resolution"):
            sweep.parse_resolution(bad)
    assert sweep.parse_audit_resolution(1024) == 1024
    for bad in (0, -5, 10**7):
        with pytest.raises(ValueError, match="audit"):
            sweep.parse_audit_resolution(bad)


@pytest.mark.parametrize(
    ("resolution", "audit_resolution", "flag"),
    [(700, 1024, "--resolution"), (1024, 0, "--audit-resolution")],
)
def test_the_sweep_command_refuses_a_bad_numeric_flag_before_any_gpu_work(
    monkeypatch, resolution, audit_resolution, flag
):
    async def must_not_run(*args, **kwargs):
        raise AssertionError("the sweep started")

    monkeypatch.setattr(sweep, "sweep", must_not_run)
    args = SimpleNamespace(
        image=Path(__file__), bands="auto,4", seed=1, resolution=resolution,
        audit_resolution=audit_resolution, matte=None, out=Path("unused"),
    )
    with pytest.raises(SystemExit, match=re.escape(flag)):
        cli._run_sweep(args)


# --- service-gates-09: the readout does not call a rejected value applied ------


def test_effective_does_not_report_a_rejected_environment_value_as_applied(
    monkeypatch, clean_invalid_env
):
    """The 2026-10-14 audit, finding service-gates-09: ``from_env`` was
    ``env in os.environ``, so a variable the parser rejected (or a blank one)
    was drawn as set by the environment, in the accent colour, with the default
    it fell back to -- answering "did my setting take" with yes."""
    monkeypatch.setenv("REALMSPINNER_TRELLIS_PORT", "eighty")
    monkeypatch.setenv("REALMSPINNER_TRELLIS_IDLE", "-5")
    monkeypatch.setenv("REALMSPINNER_FAMILIAR_IDLE", "   ")
    monkeypatch.setenv("REALMSPINNER_RIG_TIMEOUT", "99")
    rows = {s.env: s for s in effective(Config())}
    for name in (
        "REALMSPINNER_TRELLIS_PORT", "REALMSPINNER_TRELLIS_IDLE", "REALMSPINNER_FAMILIAR_IDLE"
    ):
        assert rows[name].from_env is False, name
    assert rows["REALMSPINNER_TRELLIS_PORT"].rejected
    assert rows["REALMSPINNER_TRELLIS_IDLE"].rejected
    # A good one is still reported as applied, with nothing rejected.
    assert rows["REALMSPINNER_RIG_TIMEOUT"].from_env is True
    assert rows["REALMSPINNER_RIG_TIMEOUT"].rejected == ""
    assert rows["REALMSPINNER_RIG_TIMEOUT"].value == "99.0"


# --- service-gates-10: every numeric and registry variable is validated --------


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("REALMSPINNER_REFERENCE_RETRIES", "-1"),
        ("REALMSPINNER_REFERENCE_RETRIES", "100000"),
        ("REALMSPINNER_MESH_RETRIES", "-3"),
        ("REALMSPINNER_MESH_RETRIES", "100000"),
        ("REALMSPINNER_TRELLIS_PORT", "0"),
        ("REALMSPINNER_TRELLIS_PORT", "70000"),
        ("REALMSPINNER_FAMILIAR_PORT", "-4"),
        ("REALMSPINNER_FAMILIAR_EMBED_PORT", "99999"),
        ("REALMSPINNER_RIG_TEMPLATE", "humanoyd"),
        ("REALMSPINNER_T2I_MODEL", "sdxl_cfgg"),
    ],
)
def test_a_bad_rig_template_or_t2i_model_or_port_environment_value_is_recorded_as_invalid(
    monkeypatch, clean_invalid_env, name, raw
):
    """The 2026-10-14 audit, finding service-gates-10: these skipped the checks
    their neighbours have, so a typo'd ``REALMSPINNER_RIG_TEMPLATE`` made every
    rig request refuse "That skeleton is not available" at a control the user
    never touched."""
    monkeypatch.setenv(name, raw)
    cfg = Config()
    assert name in {entry[0] for entry in clean_invalid_env}
    monkeypatch.delenv(name)
    clean = Config()
    for field_name, env in config_mod.SETTINGS:
        if env == name:
            assert getattr(cfg, field_name) == getattr(clean, field_name)


def test_good_values_for_the_validated_fields_are_accepted(monkeypatch, clean_invalid_env):
    monkeypatch.setenv("REALMSPINNER_TRELLIS_PORT", "18000")
    monkeypatch.setenv("REALMSPINNER_MESH_RETRIES", "2")
    monkeypatch.setenv("REALMSPINNER_RIG_TEMPLATE", "Quadruped")
    monkeypatch.setenv("REALMSPINNER_T2I_MODEL", "flux_klein")
    cfg = Config()
    assert clean_invalid_env == []
    assert cfg.trellis_port == 18000 and cfg.mesh_retries == 2
    assert cfg.rig_template == "quadruped" and cfg.t2i_model == "flux_klein"


def test_the_environment_copies_of_the_registries_match_the_registries():
    """``config`` may not import ``models`` or the rig templates (layering), so
    it restates their keys; this is the pin that fails when one drifts."""
    from realmspinner import models
    from realmspinner.kernels.rig import templates

    assert set(config_mod._ENV_BASE_MODELS) == set(models.BASE_MODELS)
    assert set(config_mod._ENV_RIG_TEMPLATES) == set(templates.templates())


# --- service-queue-05: backing up into the live data directory -----------------


def test_backup_into_the_live_data_directory_is_refused_with_a_message(svc):
    """The 2026-10-14 audit, finding service-queue-05: the data directory is
    the most natural wrong folder, and the replace over the open store raised a
    raw ``PermissionError [WinError 5]`` with a long temp path."""
    live_dir = Path(svc.config.db_path).parent
    with pytest.raises(Invalid) as raised:
        library.backup(svc, live_dir)
    assert raised.value.field == "dest"
    assert "live" in str(raised.value).lower()
    assert Path(svc.config.db_path).is_file()


def test_the_store_itself_refuses_to_back_up_over_its_own_file(svc):
    with pytest.raises(ValueError, match="itself"):
        svc.store.backup_to(Path(svc.config.db_path))
