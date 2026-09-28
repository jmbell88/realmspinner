"""the 2026-09-26 audit, finding muse-engine-02.

``fileio.read_track`` resamples a legacy-rate take (anything below
``sirens_audio.RATE``) through ``scipy.signal.resample_poly`` so the mixer
always sees one fixed rate. Both export functions used to write straight from
that same resampled ``player.pcm``/``player.rate`` pair, so an exported loop
or track for a legacy-rate take was scipy's resampled arithmetic rather than
the take's own samples -- the one exported file in this app whose bytes
depended on a dependency that can change under a ``uv sync``, breaking the
byte-identity rule ``engine/__init__.py``'s scipy ban exists to hold.

These fail against the unfixed code: the export lands at ``player.rate``
(the mixer's fixed rate), not the take's own native rate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import soundfile as sf

from realmspinner.studio.modes.muse import fileio as muse_io
from realmspinner.studio.modes.muse import state as muse_state
from realmspinner.studio.modes.muse.engine import waveform
from realmspinner.studio.modes.sirens import audio as sirens_audio


class _Config:
    def __init__(self, root) -> None:
        self.root = root

    def job_dir(self, job_id: str):
        return self.root / job_id


class _Svc:
    def __init__(self, root) -> None:
        self.config = _Config(root)


class FakeCtx:
    """The same shape as ``test_muse_mode.py``'s own fixture -- kept local so
    this file has no import-order dependency on another test module."""

    def __init__(self, root) -> None:
        self.svc = _Svc(root)
        self.toasts: list[tuple[str, str]] = []

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info", **_extra: Any) -> None:
        self.toasts.append((message, kind))


def _write_legacy_take(job_dir, native_rate: int, seconds: float = 10.0) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * native_rate)
    rng = np.random.default_rng(0)
    data = rng.uniform(-0.5, 0.5, size=(frames, 2)).astype(np.float32)
    sf.write(str(job_dir / "track.wav"), data, native_rate, subtype="PCM_16")


def _load_player(job_id: str, job_dir) -> Any:
    """Exactly what ``mode.on_task_done`` builds, without importing ``mode``
    (which pulls in imgui-adjacent machinery this test does not need)."""
    decoded = muse_io.read_track(job_dir / "track.wav")
    return muse_state.Player(
        job=job_id,
        pcm=decoded["pcm"],
        rate=int(decoded["rate"]),
        env=decoded["env"],
        duration=float(decoded["duration"]),
    )


@pytest.fixture
def legacy_player(tmp_path):
    native_rate = 22050
    job_id = "legacy-take"
    job_dir = tmp_path / job_id
    _write_legacy_take(job_dir, native_rate)
    one = _load_player(job_id, job_dir)
    assert one.rate != native_rate, "the fixture must actually need a resample"
    return tmp_path, one, native_rate


def test_export_with_points_writes_the_takes_own_rate_not_the_resampled_one(
    legacy_player, monkeypatch
):
    tmp_path, one, native_rate = legacy_player
    one.loop_start, one.loop_end = 1.0, 9.0
    one.xfade_ms = 0.0

    out = tmp_path / "exported_track.wav"
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: out)

    ctx = FakeCtx(tmp_path)
    muse_io.export_with_points(ctx, one)

    assert out.exists()
    written, written_rate = sf.read(str(out), dtype="int16", always_2d=True)
    assert written_rate == native_rate, (
        "export must write at the take's own rate, not the mixer's resampled "
        f"one ({one.rate})"
    )


def test_export_loop_writes_the_takes_own_rate_not_the_resampled_one(
    legacy_player, monkeypatch
):
    tmp_path, one, native_rate = legacy_player
    one.loop_start, one.loop_end = 1.0, 9.0
    one.xfade_ms = 20.0

    out = tmp_path / "exported_loop.wav"
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: out)

    ctx = FakeCtx(tmp_path)
    muse_io.export_loop(ctx, one)

    assert out.exists()
    written, written_rate = sf.read(str(out), dtype="int16", always_2d=True)
    assert written_rate == native_rate, (
        "the crossfaded loop must be exported at the take's own rate, not "
        f"the mixer's resampled one ({one.rate})"
    )


def test_export_with_points_still_uses_the_players_rate_with_no_file_on_disk(
    tmp_path, monkeypatch
):
    """The fallback in ``_export_source``: a test double (or any caller with
    no real ``track.wav`` under that job) must not turn into a crash -- the
    probe fails, and export falls back to the buffer already in memory,
    exactly the pre-fix behaviour for the ordinary, non-legacy-rate case."""
    rate = sirens_audio.RATE
    pcm = np.zeros((int(10.0 * rate), 2), dtype=np.int16)
    one = muse_state.Player(
        job="no-file-on-disk", pcm=pcm, rate=rate, env=waveform.peaks(pcm), duration=10.0
    )
    one.loop_start, one.loop_end = 2.0, 8.0
    one.xfade_ms = 0.0

    out = tmp_path / "track.wav"
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: out)

    ctx = FakeCtx(tmp_path)
    muse_io.export_with_points(ctx, one)

    assert out.exists()
    assert not ctx.toasts
