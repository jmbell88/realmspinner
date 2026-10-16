"""The engine tells us what it can do; we do not infer it from a version string.

``trellis.engine_caps`` reads the exe's own ``--help``. v0.8.1 (the last upstream
release) lists ``--model`` and pixal3d but not ``--faces``; a build from a later
upstream commit (``scripts/build-trellis.ps1``) lists ``--faces``. The fixture is
the real v0.8.1 help text, captured 2026-10-15 from the installed engine -- on
*stderr*, which is why the probe reads both streams.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from realmspinner import doctor, models, provenance, winjob
from realmspinner.config import Config
from realmspinner.pipelines import trellis

_FIXTURES = Path(__file__).parent.parent / "fixtures"
HELP_V081 = (_FIXTURES / "trellis_help_v081.txt").read_text(encoding="utf-8")
#: The help of the engine ``scripts/build-trellis.ps1`` built from upstream commit 610c90e9
#: (2026-10-15): the first one that lists ``--faces``. Captured from the real binary.
HELP_WITH_FACES = (_FIXTURES / "trellis_help_610c90e9.txt").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _fresh_cache():
    trellis._caps_cache.clear()
    yield
    trellis._caps_cache.clear()


def _exe(tmp_path: Path, body: bytes = b"exe") -> Path:
    exe = tmp_path / "engine" / "trellis-server.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(body)
    return exe


def _answer(monkeypatch, *, stdout: str = "", stderr: str = "") -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout, stderr)

    monkeypatch.setattr(winjob, "run", fake_run)
    return calls


def test_the_two_real_help_texts_differ_exactly_where_the_probe_looks():
    """Not a synthetic line: the shipped fixtures are two real binaries' help."""
    assert "--faces" not in HELP_V081 and "--faces" in HELP_WITH_FACES
    assert "--model" in HELP_V081 and "pixal3d" in HELP_V081


def test_engine_caps_reads_faces_from_help_text(tmp_path, monkeypatch):
    exe = _exe(tmp_path)
    _answer(monkeypatch, stderr=HELP_WITH_FACES)
    assert trellis.engine_caps(exe) == {"faces": True, "post_replay": False, "pixal3d": True}


def test_engine_caps_reports_v081_as_without_faces(tmp_path, monkeypatch):
    exe = _exe(tmp_path)
    _answer(monkeypatch, stderr=HELP_V081)
    caps = trellis.engine_caps(exe)
    assert caps["faces"] is False
    assert caps["pixal3d"] is True


def test_engine_caps_reads_either_stream(tmp_path, monkeypatch):
    exe = _exe(tmp_path)
    _answer(monkeypatch, stdout=HELP_WITH_FACES, stderr="")
    assert trellis.engine_caps(exe)["faces"] is True


def test_engine_caps_sees_post_replay_beside_the_exe(tmp_path, monkeypatch):
    exe = _exe(tmp_path)
    _answer(monkeypatch, stderr=HELP_V081)
    assert trellis.engine_caps(exe)["post_replay"] is False
    (exe.parent / "post-replay.exe").write_bytes(b"x")
    # Not part of the cached help-derived answer: a file next to the exe can appear later.
    assert trellis.engine_caps(exe)["post_replay"] is True


def test_engine_caps_probes_once_per_file_and_again_when_it_changes(tmp_path, monkeypatch):
    exe = _exe(tmp_path)
    calls = _answer(monkeypatch, stderr=HELP_V081)
    trellis.engine_caps(exe)
    trellis.engine_caps(exe)
    assert len(calls) == 1
    # An engine replaced in place (new size) is a different binary.
    exe.write_bytes(b"a larger replacement exe")
    trellis.engine_caps(exe)
    assert len(calls) == 2


def test_engine_caps_does_not_cache_a_failed_probe(tmp_path, monkeypatch):
    exe = _exe(tmp_path)

    def hangs(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(winjob, "run", hangs)
    assert trellis.engine_caps(exe) == trellis.NO_CAPS
    # The engine is repaired; the next ask probes again instead of remembering the failure.
    _answer(monkeypatch, stderr=HELP_WITH_FACES)
    assert trellis.engine_caps(exe)["faces"] is True


def test_engine_caps_reads_nothing_from_output_that_is_not_a_usage_text(tmp_path, monkeypatch):
    exe = _exe(tmp_path)
    # A loader error that happens to mention the flag: not the engine's usage.
    _answer(monkeypatch, stderr="The code execution cannot proceed: --faces.dll was not found")
    assert trellis.engine_caps(exe) == trellis.NO_CAPS


def test_engine_caps_for_a_missing_exe_is_no_caps(tmp_path):
    assert trellis.engine_caps(tmp_path / "nope.exe") == trellis.NO_CAPS


@pytest.mark.skipif(sys.platform != "win32", reason="the probe is a Windows child")
def test_engine_caps_really_spawns_the_exe_with_help(tmp_path):
    exe = tmp_path / "fake-server.cmd"
    exe.write_text(
        "@echo off\r\necho usage: fake [--port P] --faces N 1>&2\r\n", encoding="ascii"
    )
    assert trellis.engine_caps(exe)["faces"] is True


# --- doctor -------------------------------------------------------------------


def _doctor_config(tmp_path: Path, **overrides) -> Config:
    (tmp_path / "assets").mkdir(parents=True, exist_ok=True)
    runtime = tmp_path / "engine" / "trellis"
    runtime.mkdir(parents=True, exist_ok=True)
    for name in models.ENGINE_MODELS["trellis_runtime"].probe:
        (runtime / name).write_bytes(b"a real file")
    kwargs = dict(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=None,
        trellis_runtime_dir=runtime,
        trellis_models_dir=tmp_path / "models",
    )
    kwargs.update(overrides)
    return Config(**kwargs)


def _row(config: Config):
    # The one row, not ``run_checks``: that runs every other probe too, and this is
    # about what this row says, not about the music-deps subprocess beside it.
    return doctor._faces_check(config)


def test_the_face_row_is_part_of_the_static_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "_faces_check", lambda config: "sentinel")
    for name in dir(doctor):
        # Every other static check answered cheaply: only membership is under test.
        if name.endswith("_check") and name not in {"_faces_check"} and callable(
            getattr(doctor, name)
        ):
            monkeypatch.setattr(doctor, name, lambda *a, **k: "other")
    monkeypatch.setattr(doctor, "_familiar_checks", lambda config: [])
    monkeypatch.setattr(doctor, "music_deps_check", lambda **k: "other")
    assert "sentinel" in doctor.static_checks(_doctor_config(tmp_path))


def test_doctor_warns_when_engine_lacks_faces(tmp_path, monkeypatch):
    monkeypatch.setattr(trellis, "engine_caps", lambda exe: dict(trellis.NO_CAPS))
    row = _row(_doctor_config(tmp_path))
    assert row.ok is False
    assert row.fatal is False
    assert "Settings -> Models" in row.detail
    assert "--faces" in row.detail


def test_doctor_names_the_override_when_the_override_lacks_faces(tmp_path, monkeypatch):
    monkeypatch.setattr(trellis, "engine_caps", lambda exe: dict(trellis.NO_CAPS))
    stale = tmp_path / "kept-v081" / "trellis-server.exe"
    stale.parent.mkdir()
    stale.write_bytes(b"x")
    row = _row(_doctor_config(tmp_path, trellis_server_exe=stale))
    assert row.ok is False and row.fatal is False
    assert "REALMSPINNER_TRELLIS_EXE" in row.detail
    assert str(stale) in row.detail


def test_doctor_is_green_when_the_engine_lists_faces(tmp_path, monkeypatch):
    monkeypatch.setattr(
        trellis, "engine_caps", lambda exe: {**trellis.NO_CAPS, "faces": True}
    )
    row = _row(_doctor_config(tmp_path))
    assert row.ok is True


def test_doctor_does_not_double_count_a_missing_engine(tmp_path):
    config = _doctor_config(tmp_path, trellis_runtime_dir=tmp_path / "no-engine-here")
    row = _row(config)
    assert row.ok is False
    assert row.pending_install is True
    assert row.fatal is False


# --- the pin and the recipe ----------------------------------------------------


def test_runtime_pin_digests_name_every_probe_file():
    """Every file the row probes is a file whose bytes are pinned, and the reverse.

    A probe file with no digest is a file ``provenance.native_engine_identity``
    never checks (a swapped DLL still reads ``version_verified``); a digest with
    no probe file is a pin nothing downloads.
    """
    digests = {name for name, _ in models.TRELLIS_RUNTIME_DIGESTS}
    assert digests == set(models.TRELLIS_RUNTIME_FILES)
    assert set(models.ENGINE_MODELS["trellis_runtime"].probe) == digests


def test_trellis_recipe_records_what_the_engine_could_do(tmp_path, monkeypatch):
    caps = {"faces": True, "post_replay": True, "pixal3d": True}
    monkeypatch.setattr(trellis, "engine_caps", lambda exe: caps)
    recipe = provenance.trellis_recipe(Config(), {}, mesh_seed=1)
    assert recipe["engine_caps"] == caps
