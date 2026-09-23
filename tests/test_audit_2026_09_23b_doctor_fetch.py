"""The 2026-09-23 (second run) audit, findings pipelines-02 and pipelines-03.

pipelines-02: :func:`realmspinner.fetch.suspect_files`'s zero-byte scan only
ever matched weight-file suffixes (``.safetensors``, ``.gguf``, ``.bin``,
``.pt``), so a zero-byte ``model_index.json``/``config.json`` -- the marker
``present()``/``base_model_state`` check with a bare ``is_file()``, true at
any size -- passed both ``present()`` and this scan while the checkpoint
behind it could not load.

pipelines-03: doctor's ``_exe_check`` downgraded ``trellis-server.exe`` on its
own zero-byte size alone, never on the eight CUDA/ggml DLLs beside it that
the same registry row (``models.ENGINE_MODELS["trellis_runtime"]``) probes --
so a zero-byte sibling DLL left doctor reporting OK while Settings -> Models,
reading the identical probe list through ``fetch.suspect_files``, reported
the row broken.
"""

from __future__ import annotations

import socket

from realmspinner import fetch, models
from realmspinner.config import Config
from realmspinner.doctor import run_checks

# --- pipelines-02: fetch.suspect_files must catch a truncated marker file ----


def _config(tmp_path):
    cfg = Config()
    cfg.t2i_model_root = tmp_path / "models"
    return cfg


def test_base_model_row_treats_a_zero_byte_model_index_json_as_missing_not_healthy(tmp_path):
    cfg = _config(tmp_path)
    spec = models.BASE_MODELS["sdxl_cfg"]
    base_dir = fetch.base_model_dir(cfg, spec)
    unet = base_dir / "unet"
    unet.mkdir(parents=True)
    (unet / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
    # A killed download's last write can land on the small marker file just
    # as readily as on the multi-GB weight -- and does not have to be the
    # weight, which is why the scan below must not assume it always is.
    (base_dir / "model_index.json").write_bytes(b"")

    # present()/base_model_state check the marker with a bare is_file(),
    # true at any size, so a truncated marker still reads as installed --
    # that is exactly the gap suspect_files exists to close.
    ok, _missing = fetch.base_model_state(cfg, spec)
    assert ok is True

    bad = fetch.suspect_files(cfg, "base", spec)
    assert str(base_dir / "model_index.json") in bad


def test_suspect_files_still_catches_a_zero_byte_weight_beside_a_healthy_marker(tmp_path):
    # The pre-existing weight-file coverage must survive the marker addition.
    cfg = _config(tmp_path)
    spec = models.BASE_MODELS["sdxl_cfg"]
    base_dir = fetch.base_model_dir(cfg, spec)
    unet = base_dir / "unet"
    unet.mkdir(parents=True)
    (unet / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"")
    (base_dir / "model_index.json").write_text("{}", encoding="utf-8")

    bad = fetch.suspect_files(cfg, "base", spec)
    assert str(unet / "diffusion_pytorch_model.fp16.safetensors") in bad
    assert str(base_dir / "model_index.json") not in bad


def test_suspect_files_leaves_music_and_separation_kinds_unaffected(tmp_path):
    # Music/separation have no top-level config.json/model_index.json at all
    # (present()'s own comment: "ACE-Step has no top-level config.json") --
    # the new marker candidate must simply never match, not invent a false
    # positive for a kind that was never meant to carry one.
    cfg = _config(tmp_path)
    spec = models.SEPARATION_MODELS[next(iter(models.SEPARATION_MODELS))]
    dest = cfg.t2i_model_root / spec.dir_name
    dest.mkdir(parents=True)
    for name in spec.probe:
        (dest / name).write_bytes(b"x")
    assert fetch.suspect_files(cfg, "separation", spec) == []


# --- pipelines-03: doctor's exe row must fold in the sibling DLL probes -----


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _doctor_config(tmp_path, **overrides) -> Config:
    (tmp_path / "assets").mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        trellis_port=_free_port(),
    )
    kwargs.update(overrides)
    return Config(**kwargs)


def test_exe_check_reports_broken_when_a_sibling_cuda_dll_is_zero_bytes(tmp_path):
    runtime_dir = tmp_path / "engine" / "trellis"
    runtime_dir.mkdir(parents=True)
    exe = runtime_dir / "trellis-server.exe"
    exe.write_bytes(b"a real binary, not empty")
    # One of TRELLIS_RUNTIME_FILES -- the exe itself is intact, but the
    # library it needs to actually run is a zero-byte leftover of a killed
    # unpack.
    (runtime_dir / "cublas64_13.dll").write_bytes(b"")

    checks = {
        c.name: c
        for c in run_checks(
            _doctor_config(tmp_path, trellis_server_exe=None, trellis_runtime_dir=runtime_dir)
        )
    }
    row = checks["trellis-server.exe"]
    assert row.ok is False
    assert row.pending_install is True
    assert "cublas64_13.dll" in row.detail
    assert "0 bytes" in row.detail


def test_exe_check_still_passes_when_every_probed_file_is_healthy(tmp_path):
    runtime_dir = tmp_path / "engine" / "trellis"
    runtime_dir.mkdir(parents=True)
    for name in models.ENGINE_MODELS["trellis_runtime"].probe:
        (runtime_dir / name).write_bytes(b"a real file")

    checks = {
        c.name: c
        for c in run_checks(
            _doctor_config(tmp_path, trellis_server_exe=None, trellis_runtime_dir=runtime_dir)
        )
    }
    row = checks["trellis-server.exe"]
    assert row.ok is True
