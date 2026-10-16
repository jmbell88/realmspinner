"""Pixal3D: an experimental second reconstruction model on the same engine.

Opt-in per job (``recon_model``), never the default, its own five weight files in the
same model directory, resolution 1024 or 1536 only (Pixal3D publishes no 512 texture
model). Nothing here starts a real engine: the door is exercised with a patched
``engine_caps`` and the worker with the conftest fake server.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from realmspinner import doctor, fetch, models, provenance, vectors, vram
from realmspinner.config import Config
from realmspinner.db import JobStore
from realmspinner.pipelines import trellis
from realmspinner.queue import Worker
from realmspinner.service import jobs as svc_jobs
from realmspinner.service.errors import Invalid
from realmspinner.studio.modes.create.engine import mesh as create_mesh
from realmspinner.studio.modes.create.ui.panes import settings_3d
from realmspinner.studio.state import DEFAULT_FORM_3D


def _engine(monkeypatch, *, pixal3d: bool = True) -> None:
    monkeypatch.setattr(
        trellis, "engine_caps",
        lambda exe, **kw: {**trellis.NO_CAPS, "pixal3d": pixal3d, "faces": True},
    )


def _install_weights(config) -> None:
    config.trellis_models_dir.mkdir(parents=True, exist_ok=True)
    for name in models.PIXAL3D_GGUF_FILES:
        (config.trellis_models_dir / name).write_bytes(b"gguf")


def _create(svc, **kwargs):
    out = svc_jobs.create_job(svc, kind="text", prompt="a chest", output="model", **kwargs)
    return svc.store.get(out["id"])["params"]


# --- the registry row ------------------------------------------------------------


def test_the_weights_row_is_the_five_pixal3d_files_in_the_trellis_directory(tmp_path):
    spec = models.ENGINE_MODELS["pixal3d_gguf"]
    assert set(spec.probe) == set(models.PIXAL3D_GGUF_FILES)
    assert len(models.PIXAL3D_GGUF_FILES) == 5
    assert all(name.startswith("pixal3d_") for name in spec.probe)
    assert spec.runtime is False
    # Lands beside the TRELLIS.2 weights: one model directory serves both families.
    config = Config(trellis_models_dir=tmp_path / "models")
    assert fetch.engine_dir(config, spec) == fetch.engine_dir(
        config, models.ENGINE_MODELS["trellis_gguf"]
    )


def test_the_weights_are_pinned_to_a_revision_and_selected_by_prefix():
    (item,) = models.ENGINE_MODELS["pixal3d_gguf"].fetch
    assert item.repo_id == "vegax87/Pixal3D"
    assert len(item.revision) == 40
    assert item.allow_patterns == ("pixal3d_*.gguf",)


# --- the door ----------------------------------------------------------------------


def test_pixal3d_at_512_is_refused_with_field_resolution(svc, monkeypatch):
    _engine(monkeypatch)
    _install_weights(svc.config)
    with pytest.raises(Invalid) as info:
        _create(svc, recon_model="pixal3d", resolution=512)
    assert info.value.field == "resolution"


def test_pixal3d_without_weights_is_refused_at_the_door(svc, monkeypatch):
    _engine(monkeypatch)
    with pytest.raises(Invalid) as info:
        _create(svc, recon_model="pixal3d", resolution=1024)
    assert info.value.field == "recon_model"
    assert "not downloaded" in str(info.value)
    # Carried to the Settings row, so the Download button can be offered.
    assert info.value.rows == ("engine:pixal3d_gguf",)


def test_pixal3d_on_an_engine_that_cannot_run_it_is_refused_not_silently_trellis(
    svc, monkeypatch
):
    """An exe older than v0.8.0 ignores the ``model`` field and would make a TRELLIS.2 mesh
    under Pixal3D's name -- into the findings corpus as evidence about Pixal3D."""
    _engine(monkeypatch, pixal3d=False)
    _install_weights(svc.config)
    with pytest.raises(Invalid) as info:
        _create(svc, recon_model="pixal3d", resolution=1024)
    assert info.value.field == "recon_model"
    assert "v0.8.0" in str(info.value)


def test_an_unknown_model_and_an_absurd_fov_are_refused_by_name(svc):
    with pytest.raises(Invalid) as info:
        _create(svc, recon_model="hunyuan")
    assert info.value.field == "recon_model"
    with pytest.raises(Invalid) as info:
        _create(svc, recon_model="pixal3d", recon_fov=400)
    assert info.value.field == "recon_fov"
    with pytest.raises(Invalid) as info:
        _create(svc, recon_model="pixal3d", recon_fov="wide")
    assert info.value.field == "recon_fov"


def test_a_pixal3d_job_records_its_model_and_its_camera(svc, monkeypatch):
    _engine(monkeypatch)
    _install_weights(svc.config)
    params = _create(svc, recon_model="pixal3d", resolution=1024)
    assert params["recon_model"] == "pixal3d"
    assert params["recon_fov"] == pytest.approx(49.13)
    assert _create(svc, recon_model="pixal3d", recon_fov=38.0)["recon_fov"] == 38.0


def test_a_trellis_job_records_neither_key(svc):
    """TRELLIS.2 is the default and leaves no trace, so every row from before Pixal3D and
    every row that never asked reads identically (and its vector is unchanged)."""
    for kwargs in ({}, {"recon_model": "trellis"}, {"recon_model": ""}):
        params = _create(svc, **kwargs)
        assert "recon_model" not in params and "recon_fov" not in params


def test_recon_model_is_a_vector_param():
    assert "recon_model" in vectors.VECTOR_PARAMS
    assert "recon_fov" in vectors.VECTOR_PARAMS
    # ...and a row with neither (every TRELLIS.2 row) contributes nothing to its vector.
    base = {"platform": "3d", "resolution": 1024}
    assert vectors.config_vector(base) == vectors.config_vector({**base, "recon_model": None})


def test_the_recipe_names_the_model_and_the_camera(monkeypatch):
    monkeypatch.setattr(trellis, "engine_caps", lambda exe, **kw: dict(trellis.NO_CAPS))
    plain = provenance.trellis_recipe(Config(), {}, mesh_seed=1)
    pixal = provenance.trellis_recipe(
        Config(), {"recon_model": "pixal3d", "recon_fov": 40.0}, mesh_seed=1
    )
    assert plain["recon_model"] == "trellis" and plain["recon_fov"] is None
    assert pixal["recon_model"] == "pixal3d" and pixal["recon_fov"] == 40.0


# --- VRAM ---------------------------------------------------------------------------


def test_pixal3d_is_charged_more_than_trellis_until_it_is_measured():
    assert pytest.approx(vram.TRELLIS_GIB * 1.25) == vram.PIXAL3D_GIB
    plain = vram._trellis_cost({"resolution": 1024})
    pixal = vram._trellis_cost({"resolution": 1024, "recon_model": "pixal3d"})
    assert plain == vram.TRELLIS_GIB and pixal == vram.PIXAL3D_GIB
    # Resolution scales it the same way for both.
    assert vram._trellis_cost({"resolution": 1536, "recon_model": "pixal3d"}) == pytest.approx(
        vram.PIXAL3D_GIB * vram.TRELLIS_RES_MULT[1536]
    )


# --- the worker ---------------------------------------------------------------------


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


async def _run(worker, **params):
    job_id = worker.store.create("image", None, {"seed": 1, "resolution": 1024, **params})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    worker.start()
    deadline = time.monotonic() + 20
    while worker.store.get(job_id)["status"] not in ("done", "error"):
        if time.monotonic() > deadline:
            pytest.fail("job did not settle")
        await asyncio.sleep(0.01)
    await worker.shutdown()
    return worker.store.get(job_id)


async def test_a_pixal3d_job_sends_its_model_and_camera_to_the_engine(worker, monkeypatch):
    _engine(monkeypatch)
    row = await _run(worker, recon_model="pixal3d", recon_fov=44.0)
    assert row["status"] == "done"
    call = worker.trellis.generate_calls[0]
    assert call["model"] == "pixal3d" and call["fov"] == 44.0


async def test_a_trellis_job_sends_neither_field(worker, monkeypatch):
    _engine(monkeypatch)
    row = await _run(worker)
    assert row["status"] == "done"
    call = worker.trellis.generate_calls[0]
    assert call["model"] is None and call["fov"] is None


async def test_a_pixal3d_job_on_an_engine_that_cannot_run_it_fails_instead_of_running_trellis(
    worker, monkeypatch
):
    _engine(monkeypatch, pixal3d=False)
    row = await _run(worker, recon_model="pixal3d")
    assert row["status"] == "error"
    assert "Pixal3D" in row["error"]
    assert worker.trellis.generate_calls == [], "the engine must never be asked"


# --- Create ---------------------------------------------------------------------------


def test_an_untouched_form_chooses_trellis_and_sends_no_model_field():
    assert DEFAULT_FORM_3D["recon_model"] == ""
    assert "recon_model" not in create_mesh.engine_kwargs(DEFAULT_FORM_3D)
    assert "recon_fov" not in create_mesh.engine_kwargs(DEFAULT_FORM_3D)


def test_a_pixal3d_form_sends_the_model_and_its_camera():
    form = {**DEFAULT_FORM_3D, "recon_model": "pixal3d", "recon_fov": 38.0}
    kwargs = create_mesh.engine_kwargs(form)
    assert kwargs["recon_model"] == "pixal3d" and kwargs["recon_fov"] == 38.0


def test_the_camera_alone_never_chooses_pixal3d():
    """A stray ``recon_fov`` with TRELLIS.2 selected is a number nobody reads."""
    kwargs = create_mesh.engine_kwargs({**DEFAULT_FORM_3D, "recon_fov": 20.0})
    assert "recon_model" not in kwargs and "recon_fov" not in kwargs


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (None, True),  # no snapshot yet: never lock an installed host
        ([], True),
        ([{"row_key": "engine:pixal3d_gguf", "present": True}], True),
        ([{"row_key": "engine:pixal3d_gguf", "present": False}], False),
        ([{"row_key": "engine:trellis_gguf", "present": False}], True),
    ],
)
def test_the_combo_is_greyed_only_when_the_pixal3d_row_is_known_absent(rows, expected):
    assert settings_3d._pixal3d_present(SimpleNamespace(model_rows=rows)) is expected


# --- doctor ---------------------------------------------------------------------------


def _doctor_config(tmp_path) -> Config:
    (tmp_path / "assets").mkdir(parents=True, exist_ok=True)
    return Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_models_dir=tmp_path / "models",
    )


def test_doctor_reports_absent_pixal3d_weights_as_optional_and_never_fatal(tmp_path):
    row = doctor._pixal3d_check(_doctor_config(tmp_path))
    assert row.ok is False and row.fatal is False and row.pending_install is True
    assert "optional" in row.detail and "TRELLIS.2 is unaffected" in row.detail


def test_doctor_is_green_when_the_five_files_are_there(tmp_path):
    config = _doctor_config(tmp_path)
    _install_weights(config)
    assert doctor._pixal3d_check(config).ok is True


# --- two weight rows, one directory ------------------------------------------------------


def _models_config(tmp_path, monkeypatch) -> Config:
    # Cleared: the override is the *user's* directory and removal refuses it by design;
    # these tests are about the directory Realmspinner fetched into.
    monkeypatch.delenv("REALMSPINNER_TRELLIS_MODELS", raising=False)
    config = Config(trellis_models_dir=tmp_path / "models" / "trellis2-gguf")
    config.trellis_models_dir.mkdir(parents=True)
    return config


def _row(key: str) -> fetch.Entry:
    entry = fetch.find(key)
    assert entry is not None
    return entry


def _write(config, spec_key: str) -> None:
    for name in models.ENGINE_MODELS[spec_key].probe:
        (config.trellis_models_dir / name).write_bytes(b"w")


def test_removing_pixal3d_deletes_only_its_own_files(tmp_path, monkeypatch):
    config = _models_config(tmp_path, monkeypatch)
    _write(config, "trellis_gguf")
    _write(config, "pixal3d_gguf")
    removal = fetch.removal_plan(config, [_row("engine:pixal3d_gguf")])
    assert removal.blocked == ()
    assert set(removal.paths) == {
        config.trellis_models_dir / name for name in models.PIXAL3D_GGUF_FILES
    }
    assert config.trellis_models_dir not in removal.paths


def test_the_trellis_weights_cannot_be_removed_out_from_under_installed_pixal3d(
    tmp_path, monkeypatch
):
    """Pixal3D shares the decoders, DINOv3 and BiRefNet with TRELLIS.2: taking the
    directory would leave the second model installed and broken."""
    config = _models_config(tmp_path, monkeypatch)
    _write(config, "trellis_gguf")
    _write(config, "pixal3d_gguf")
    assert fetch.removal_plan(config, [_row("engine:trellis_gguf")]).paths == ()


def test_the_trellis_weights_alone_still_take_the_whole_directory(tmp_path, monkeypatch):
    """Pixal3D not downloaded: its files are an absent claim, and an absent file is a
    download not made yet, which deleting the directory cannot hurt."""
    config = _models_config(tmp_path, monkeypatch)
    _write(config, "trellis_gguf")
    assert fetch.removal_plan(config, [_row("engine:trellis_gguf")]).paths == (
        config.trellis_models_dir,
    )


def test_choosing_both_rows_removes_the_directory_once(tmp_path, monkeypatch):
    config = _models_config(tmp_path, monkeypatch)
    _write(config, "trellis_gguf")
    _write(config, "pixal3d_gguf")
    removal = fetch.removal_plan(
        config, [_row("engine:trellis_gguf"), _row("engine:pixal3d_gguf")]
    )
    # The chosen files inside the chosen directory are not listed a second time.
    assert removal.paths == (config.trellis_models_dir,)


def test_both_rows_are_separate_downloads_into_one_directory(tmp_path):
    config = Config(trellis_models_dir=tmp_path / "gguf")
    jobs = fetch.plan(config, [_row("engine:trellis_gguf"), _row("engine:pixal3d_gguf")])
    assert len(jobs) == 2
    assert {job.dest for job in jobs} == {config.trellis_models_dir}
    assert {job.repo_id for job in jobs} == {"ilintar/trellis2-gguf", "vegax87/Pixal3D"}
