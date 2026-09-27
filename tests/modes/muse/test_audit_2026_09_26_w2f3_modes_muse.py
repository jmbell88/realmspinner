"""Regression tests for the 2026-09-26 audit's Muse findings (pass w2f3).

Four independent claims, kept in one file because none of them is large
enough to earn its own and all of them live under ``modes/muse``:

* ``waveform.peaks`` scaling a stereo integer take into unit range
  (muse-engine-01).
* ``on_task_done`` starting a take switch's playback at the carried
  position, not sample 0 (muse-mode-01).
* the derive popup clearing a stale refusal on open and staying open until
  its own submit has actually come back (muse-mode-02).
* ``_refine``/``crossfade``/``loop_cache_key`` not producing a wrong or
  out-of-bounds answer from a degenerate input (muse-engine-03, muse-engine-04).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from realmspinner.studio.modes.muse import fileio as muse_io
from realmspinner.studio.modes.muse import mode as muse_mode
from realmspinner.studio.modes.muse.engine import loops, waveform

from .test_muse_mode import FakeCtx

# --- muse-engine-01 -----------------------------------------------------------


def test_peaks_scales_a_stereo_int16_take_into_unit_range():
    """muse-engine-01 (2026-09-26 audit, re-run: stereo 16384.0 vs mono
    0.50001526).

    ``peaks`` downmixed a stereo take *before* checking whether ``pcm`` was
    still an integer dtype -- ``ndarray.mean`` upcasts an int16 array to
    float64, so by the time the check ran a stereo take's dtype was never
    still ``np.integer``, and the whole scale-by-the-dtype's-peak step was
    skipped outright. Its envelope came back in raw sample units, not
    ``[-1, 1]``, and drew as a solid block rather than a waveform.

    Fails against the unfixed code: ``stereo[1, 0]`` comes back ``16384.0``
    instead of ``~0.50001526``.
    """
    mono = waveform.peaks(np.full(100, 16384, dtype="<i2"), columns=1)
    stereo = waveform.peaks(np.full((100, 2), 16384, dtype="<i2"), columns=1)
    assert mono[1, 0] == pytest.approx(16384 / 32767, abs=1e-4)
    assert stereo[1, 0] == pytest.approx(mono[1, 0], abs=1e-4)
    assert stereo[1, 0] != pytest.approx(16384.0)


# --- muse-mode-01 ---------------------------------------------------------


def test_switching_takes_starts_the_new_take_at_the_carried_position(monkeypatch, tmp_path):
    """muse-mode-01 (2026-09-26 audit).

    ``on_task_done`` carried the old ``play_offset`` onto the new ``Player``
    (W4) but handed ``sirens_audio.play`` the *whole* buffer regardless, so
    the channel always started the new take at sample 0 while the player's
    own state -- and the transport's readout -- said otherwise for the rest
    of the audition.

    Fails against the unfixed code: switching from a position 7s into a
    100 Hz, 1000-sample take to a fresh load plays the new buffer's full
    1000 samples, not the 300 left after sample 700.
    """
    from realmspinner.studio.modes.sirens import audio as sirens_audio

    played: dict[str, Any] = {}

    def _play(pcm, rate, *, tag="", loops=0):
        played["length"] = len(pcm)
        return True

    monkeypatch.setattr(sirens_audio, "play", _play)
    ctx = FakeCtx(tmp_path)
    state = muse_mode.ensure(ctx)

    def load(job: str, duration: float = 10.0) -> None:
        state.audition_job = job
        done = type(
            "_Done",
            (),
            {
                "key": f"{muse_mode.LOAD_PREFIX}{job}",
                "result": {
                    "pcm": np.zeros((1000, 2), dtype=np.int16),
                    "rate": 100,
                    "duration": duration,
                },
            },
        )()
        muse_mode.on_task_done(ctx, done)

    load("a")
    assert played["length"] == 1000, "a fresh load has nothing to carry, so it plays from 0"
    state.player.play_offset = 7.0

    load("b")
    assert played["length"] == 300, "the new take must start at the carried sample 700, not 0"


# --- muse-mode-02 -----------------------------------------------------------


def test_open_derive_clears_a_refusal_recorded_for_the_previous_derive(tmp_path):
    """muse-mode-02 (2026-09-26 audit).

    Field errors used to be cleared right before ``derive`` submitted, not
    when the popup opened -- so a refusal from a previous take's derive,
    whose popup had already closed on ``ctx.submit``'s mere *acceptance*,
    survived in ``ctx.state.field_errors`` until it rang a control on the
    *next* take's popup instead.

    Fails against the unfixed code: opening a fresh popup for take "b" still
    shows take "a"'s stale ring.
    """
    ctx = FakeCtx(tmp_path)
    ctx.state.note_field_error("repaint_end", "that window is not inside the take")
    muse_mode.open_derive(ctx, "b", "retake")
    assert ctx.state.field_errors == {}


def test_derive_stays_open_until_the_submit_actually_answers(monkeypatch, tmp_path):
    """muse-mode-02 (2026-09-26 audit).

    ``derive`` used to clear ``state.derive_job`` -- what the popup draws on
    -- the instant ``ctx.submit`` merely *accepted* the request, before
    ``derive_music_job`` (a door, on the task thread) had actually answered.
    A refusal landing a frame later then had no popup left to ring.

    Fails against the unfixed code: with the submit still busy,
    ``state.derive_job`` is already ``""`` right after ``derive`` returns.
    """
    from realmspinner.service import jobs as svc_jobs

    monkeypatch.setattr(svc_jobs, "derive_music_job", lambda svc, job_id, **kw: {"ids": ["x"]})
    ctx = FakeCtx(tmp_path)
    muse_mode.open_derive(ctx, "a", "retake")
    # The real ``TaskRunner`` already shows the key busy the instant
    # ``submit`` accepts it (``_pending`` is written inside the same call,
    # before any task has had a chance to finish) -- simulated here since
    # ``FakeCtx.submit`` runs ``run()`` synchronously.
    ctx.busy_keys.add("submit")

    assert muse_mode.derive(ctx) is True
    assert (
        muse_mode.ensure(ctx).derive_job == "a"
    ), "the popup must stay open until the submit has actually come back"

    ctx.busy_keys.discard("submit")
    assert muse_mode.derive_settled(ctx) is True
    assert muse_mode.ensure(ctx).derive_job == ""


# --- muse-engine-03 -----------------------------------------------------------


def test_refine_leaves_the_end_alone_when_the_target_window_is_silent():
    """muse-engine-03 (2026-09-26 audit).

    A silent ``target`` window is all zeros, so ``np.correlate`` against it
    is flat at every offset and ``np.argmax`` of a flat array returns its
    *first* index -- not "no offset preferred". That snapped ``end`` to
    roughly ``half`` the refine window before where it started, on every
    silent region this stage ever saw.

    Fails against the unfixed code: ``new_end`` comes back ``1900``, 100
    samples (``half`` at this rate/window) before the ``end`` the door asked
    to refine.
    """
    rate = 2000
    mono = np.zeros(4000, dtype=np.float32)
    start, end = 1000, 2000

    new_start, new_end = loops._refine(mono, start, end, rate)

    assert new_end == end
    assert new_start == start


# --- muse-engine-04 -----------------------------------------------------------


def test_crossfade_clamps_an_end_past_the_takes_length_instead_of_raising():
    """muse-engine-04 (2026-09-26 audit).

    Not reachable through today's UI -- every live caller already comes
    through ``loop_cache_key`` -- but ``crossfade`` trusted ``start``/``end``
    outright: an ``end`` past ``len(pcm)`` raised ``IndexError`` at
    ``data[end - 1]``.

    Fails against the unfixed code with that ``IndexError``.
    """
    pcm = np.zeros(100, dtype=np.int16)
    body = loops.crossfade(pcm, 10, 500, 4)
    assert body.shape[0] == 90


def test_crossfade_clamps_a_negative_start_instead_of_wrapping():
    """muse-engine-04 (2026-09-26 audit), the other half.

    A negative ``start`` fell through to ``data[start:end]``, which is valid
    Python slicing syntax -- but it silently wraps onto samples from the
    *tail* of the take instead of refusing what is obviously not a region of
    it, returning a body of the wrong length with no error at all.

    Fails against the unfixed code: with ``start=-10`` on a 100-sample take,
    the body wraps in the last 10 samples and comes back 110 samples long
    instead of the clamped 100.
    """
    pcm = np.arange(100, dtype=np.int16)
    body = loops.crossfade(pcm, -10, 100, 4)
    assert body.shape[0] == 100


def test_loop_cache_key_clamps_a_region_past_the_current_takes_length():
    """muse-engine-04 (2026-09-26 audit).

    Not reachable through today's UI -- a region remembered against one take
    (``loop_memory``, keyed by job id) would need to outlive a shorter file
    later written under the same id (a rerun) -- but ``loop_cache_key``
    computed ``end`` straight from ``loop_end * rate`` with nothing checked
    against ``len(player.pcm)``, and the returned key reached ``crossfade``
    unclamped.

    Fails against the unfixed code: ``key[1]`` comes back ``2000``, past the
    1000-sample take it was asked about.
    """
    player = SimpleNamespace(
        loop_start=0.0,
        loop_end=20.0,
        rate=100,
        xfade_ms=0.0,
        pcm=np.zeros(1000, dtype=np.int16),
    )
    key = muse_io.loop_cache_key(player)
    assert key is not None
    assert key[1] <= 1000
