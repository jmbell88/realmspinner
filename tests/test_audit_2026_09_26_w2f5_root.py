"""Regression tests for the 2026-09-26 audit's root-level findings closed in
pass w2f5: pipelines-children-02 (``winjob.run``), pipelines-install-01
(forced health recheck), pipelines-install-02 (``_exe_check``),
pipelines-install-04 (``suspect_files``), pipelines-install-06
(``_registry_row``) and pipelines-mesh-01 (``meshaudit._project``).

One file because every fix here lives in a root-level module
(``winjob.py``, ``doctor.py``, ``fetch.py``, ``meshaudit.py``) with no
package of its own under ``tests/``, matching the shape of the existing
``tests/test_audit_2026_09_23b_doctor_fetch.py``.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from realmspinner import doctor, fetch, meshaudit, winjob
from realmspinner import models as model_registry
from realmspinner.config import Config
from realmspinner.doctor import run_checks


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _config(tmp_path, **overrides) -> Config:
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


# --- pipelines-children-02: winjob.run's input/check kwargs ------------------


def test_winjob_run_accepts_input_and_check_like_subprocess_run():
    """``winjob.run`` used to forward ``input``/``check`` straight into
    ``Popen(argv, **kwargs)``, which has neither parameter -- every Flourish
    text-model prompt (``input=...`` on stdin) and every Reveal in Explorer
    call (``check=False``) raised ``TypeError`` before the child was even
    assigned into the kill-on-close job."""
    proc = winjob.run(
        [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read().upper())"],
        input="hello",
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout == "HELLO"


def test_winjob_run_check_true_raises_like_subprocess_run_on_a_nonzero_exit():
    with pytest.raises(subprocess.CalledProcessError):
        winjob.run([sys.executable, "-c", "import sys; sys.exit(3)"], check=True)


# --- pipelines-install-01: forced health recheck reprobes the cached rows ----


def test_forced_health_recheck_reprobes_bpy_after_a_pack_install(monkeypatch):
    """Installing the Rigging pack from Settings -> Packs is supposed to turn
    Poser back on without a restart (manual 42/INSTALL). Before this fix,
    ``doctor.blender_check``'s module-level cache held the *first* answer for
    the life of the process regardless of ``force`` -- ``static_checks``
    rebuilt its list on a forced recheck, but this one row inside it kept
    answering from the stale global."""
    monkeypatch.setattr(doctor, "_blender", None)
    calls: list[None] = []

    def fake_probe():
        calls.append(None)
        # First probe: bpy missing. Second: the pack "installed" between calls.
        ok = len(calls) > 1
        return doctor.Check("Blender (rigging)", ok, "stub", fatal=False)

    monkeypatch.setattr(doctor, "_probe_blender", fake_probe)

    first = doctor.blender_check(force=True)
    assert first.ok is False
    second = doctor.blender_check(force=True)
    assert second.ok is True, "force=True must drop the cached answer, not just re-read it"


# --- pipelines-install-02: _exe_check wants every runtime file, not just the exe --


def test_exe_check_fails_when_a_sibling_runtime_dll_is_missing(tmp_path):
    """``_exe_check`` used to report OK from ``path.is_file()`` alone, so a
    runtime with the exe intact but a *missing* (not zero-byte) sibling CUDA
    DLL passed here while the identical probe list in Settings -> Models
    reported it broken -- and trellis-server failed to start with no row
    that had said why. ``fetch.present`` is the nine-file check both should
    share."""
    spec = model_registry.ENGINE_MODELS["trellis_runtime"]
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    for name in spec.probe:
        if name == "ggml.dll":
            continue  # the one sibling file this install never got
        (runtime_dir / name).write_bytes(b"binary")

    checks = {
        c.name: c
        for c in run_checks(
            _config(tmp_path, trellis_server_exe=None, trellis_runtime_dir=runtime_dir)
        )
    }
    row = checks["trellis-server.exe"]
    assert row.ok is False
    assert row.pending_install is True
    assert "ggml.dll" in row.detail


# --- pipelines-install-04: suspect_files must also weigh a base's own LoRA --


def test_a_zero_byte_base_lora_makes_the_base_model_suspect(tmp_path):
    """``suspect_files``'s ``base`` branch scanned the checkpoint directory
    for empty weight files but never looked at ``loras/<base_lora>`` --
    exactly the file ``base_model_state`` refuses without -- so a killed
    download that left a zero-byte ``Hyper-SDXL-4steps-lora.safetensors``
    kept every doctor row green."""
    spec = model_registry.BASE_MODELS["sdxl"]
    assert spec.base_lora, "fixture assumption: sdxl carries a step-distillation LoRA"

    config = _config(tmp_path)
    base_dir = fetch.base_model_dir(config, spec)
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "model_index.json").write_text("{}")
    lora_dir = config.t2i_model_root / "loras"
    lora_dir.mkdir(parents=True, exist_ok=True)
    zero_byte_lora = lora_dir / spec.base_lora
    zero_byte_lora.write_bytes(b"")

    bad = fetch.suspect_files(config, "base", spec)
    assert str(zero_byte_lora) in bad


# --- pipelines-install-06: a load failure is not "not installed yet" --------


def test_a_present_matting_model_that_fails_to_load_is_not_a_setup_row(tmp_path, monkeypatch):
    """``_registry_row`` used to set ``pending_install=not ok`` unconditionally,
    so a matting/pose model whose weights are fully present but fails to
    *load* was labelled SETUP -- identically to one never downloaded -- and
    dropped out of every failure counter ``dev/INVARIANTS.md`` keys on that
    distinction."""
    config = _config(tmp_path)
    spec = model_registry.MATTING_MODELS[model_registry.DEFAULT_MATTING]
    base = config.t2i_model_root / spec.dir_name
    base.mkdir(parents=True, exist_ok=True)
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(doctor, "_probes", {})
    monkeypatch.setattr(doctor, "_run_load_probe", lambda which, path: (False, "could not load"))
    monkeypatch.setattr(doctor.matting, "last_error", lambda: None)
    monkeypatch.setattr(doctor, "_missing_modules", lambda names: [])

    checks = {c.name: c for c in doctor._matting_checks(config, probe_slow=True)}
    row = checks[fetch.check_name("matting", spec.label)]
    assert row.ok is False
    assert row.pending_install is False, "present-but-broken must not read as a setup step"


# --- pipelines-mesh-01: a non-finite vertex must refuse, not corrupt indices -


def test_hole_fraction_refuses_a_mesh_with_a_non_finite_vertex_before_rasterising():
    """``meshaudit._project`` never checked that the vertices it projects are
    finite. A NaN turned ``sx.min()``/``sy.max()`` into NaN, the ``span <= 0``
    guard is False for NaN (every comparison with NaN is), and the
    NaN-poisoned scale that followed produced pixel coordinates the native
    rasteriser indexed with directly -- an access violation in the app
    process. The numpy fallback instead silently reports the mesh solid.
    Refusing inside ``_project`` protects both fills at once, before either
    ever sees the triangle."""
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [float("nan"), 0.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]])

    assert meshaudit._project(positions, faces, (0.0, 0.0, 1.0), 64) is None

    covered = meshaudit._coverage(positions, faces, (0.0, 0.0, 1.0), 64)
    assert covered.shape == (64, 64)
    assert not covered.any()
