"""The player's verbs: the playhead, the region, seeking and the exports.

``test_muse_mode``'s context, because these are the same controller -- but its
own file because every test here needs a *decoded take* on the state, which the
brief's tests have no use for.

Nothing here touches a sound card. ``sirens_audio`` is stubbed to a recorder, so
what is asserted is what the mode asks the device for -- which is the whole
contract between the two: the mixer owns one channel and Muse owns the offset
that makes its clock absolute.
"""

from __future__ import annotations

import io
import wave
from typing import Any

import numpy as np
import pytest

from realmspinner.studio.modes.muse import fileio as muse_io
from realmspinner.studio.modes.muse import mode as muse_mode
from realmspinner.studio.modes.muse import state as muse_state

from .test_muse_mode import FakeCtx

RATE = 44100


class _Device:
    """``sirens_audio``, as a recorder. One channel, exactly as the real one."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.tag_value = ""
        self.busy = False
        self.pos = 0.0
        self.level = 1.0
        # muse-01 (2026-09-11 audit): a device-less machine's whole failure
        # mode is ``play`` refusing -- this flag is what lets a test put the
        # recorder in that state without a real mixer.
        self.refuses = False

    RATE = RATE

    def play(self, pcm, rate=RATE, *, tag="", loops=0) -> bool:
        if self.refuses:
            return False
        self.calls.append(
            {"frames": len(pcm), "rate": rate, "tag": tag, "loops": loops, "pcm": np.array(pcm)}
        )
        self.tag_value = tag
        self.busy = True
        return True

    def stop(self) -> None:
        self.busy = False
        self.tag_value = ""

    def playing(self) -> bool:
        return self.busy

    def tag(self) -> str:
        return self.tag_value if self.busy else ""

    def position(self) -> float:
        return self.pos

    def volume(self) -> float:
        return self.level

    def set_volume(self, value) -> None:
        self.level = value

    def unavailable_reason(self) -> str:
        return "no device" if self.refuses else ""

    def available(self) -> bool:
        return not self.refuses


@pytest.fixture
def device(monkeypatch):
    one = _Device()
    monkeypatch.setattr(muse_mode, "sirens_audio", one)
    return one


@pytest.fixture
def ctx(tmp_path):
    one = FakeCtx(tmp_path)
    one.cache = type("_Cache", (), {"jobs": [{"id": "a"}]})()
    return one


def _loaded(ctx, seconds: float = 10.0, job: str = "a"):
    """Put a decoded take on the state, the way ``on_task_done`` would."""
    from realmspinner.studio.modes.muse.engine import waveform

    pcm = np.zeros((int(seconds * RATE), 2), dtype=np.int16)
    state = muse_mode.ensure(ctx)
    state.player = muse_state.Player(
        job=job, pcm=pcm, rate=RATE, env=waveform.peaks(pcm), duration=seconds
    )
    return state.player


def _settle(ctx, one) -> None:
    """Land the loop-cache blend, the way a marker grip's release or the
    crossfade slider's release does in the real UI -- both call
    ``precompute_loop`` right after ``set_region``/setting ``xfade_ms``, well
    before a user's next Play. Every test below that sets a region and then
    plays or seeks calls this in between, for the reason ``_play_from`` no
    longer blends inline on a stale cache (muse-02, the 2026-09-18 audit):
    calling ``set_region`` alone, as these tests used to and then playing
    straight through it, is not a sequence the real controls produce, and a
    stale-cache Play now defers rather than sounding anything immediately.
    """
    muse_mode.precompute_loop(ctx)
    if not ctx.submitted or not ctx.submitted[-1].startswith(muse_io.CACHE_PREFIX):
        return  # nothing to settle: no region, or the cache was already current
    done = type("_Done", (), {"key": ctx.submitted[-1], "result": ctx.result})()
    muse_mode.on_task_done(ctx, done)


# --- the playhead ------------------------------------------------------------


def test_there_is_no_player_before_the_first_audition():
    """``player`` never builds one -- ``active``'s rule, on the other field."""
    from pathlib import Path

    assert muse_mode.player(FakeCtx(Path("."))) is None


def test_the_position_is_the_offset_plus_the_mixers_clock(ctx, device):
    """The two halves of one number.

    ``sirens_audio.position`` answers where the playhead is in the *buffer*, and
    seeking is slice-and-replay -- so the offset the slice began at is what
    makes it a position in the take. The mixer deliberately does not track it:
    it does not own the caller's buffer.
    """
    one = _loaded(ctx)
    one.play_offset = 4.0
    device.busy, device.tag_value, device.pos = True, "a", 1.5
    assert muse_mode.position(ctx) == pytest.approx(5.5)


def test_a_stopped_player_reads_at_its_offset_rather_than_at_zero(ctx, device):
    one = _loaded(ctx)
    one.play_offset = 4.0
    assert muse_mode.position(ctx) == pytest.approx(4.0)


def test_the_position_never_runs_past_the_end(ctx, device):
    one = _loaded(ctx, seconds=10.0)
    one.play_offset = 9.0
    device.busy, device.tag_value, device.pos = True, "a", 5.0
    assert muse_mode.position(ctx) == pytest.approx(10.0)


# --- seeking -----------------------------------------------------------------


def test_seeking_while_silent_moves_the_playhead_without_making_a_noise(ctx, device):
    """A seek is not a play. Clicking the waveform to look at something must
    not start the mode making a sound."""
    _loaded(ctx)
    muse_mode.seek(ctx, 3.0)
    assert muse_mode.position(ctx) == pytest.approx(3.0)
    assert device.calls == []


def test_seeking_while_sounding_replays_the_remainder(ctx, device):
    """Slice-and-replay: there is no device-side seek in this mixer, so the
    buffer handed to it *is* the remainder."""
    one = _loaded(ctx, seconds=10.0)
    device.busy, device.tag_value = True, "a"
    muse_mode.seek(ctx, 4.0)
    assert device.calls[-1]["frames"] == pytest.approx(6 * RATE, rel=0.01)
    assert one.play_offset == pytest.approx(4.0)


def test_a_seek_is_clamped_to_the_take(ctx, device):
    _loaded(ctx, seconds=10.0)
    muse_mode.seek(ctx, -5.0)
    assert muse_mode.position(ctx) == 0.0
    muse_mode.seek(ctx, 999.0)
    assert muse_mode.position(ctx) == pytest.approx(10.0)


def test_playing_a_region_repeats_it_and_stops_at_its_end(ctx, device):
    """A seam is judged by hearing it come round again, so the region audition
    repeats -- and ``sirens_audio.position`` already wraps modulo the buffer
    when loops is non-zero, so the playhead falls out with no arithmetic."""
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 6.0)
    _settle(ctx, one)
    muse_mode.play_region(ctx)
    call = device.calls[-1]
    assert call["loops"] == -1
    assert call["frames"] == pytest.approx(4 * RATE, rel=0.01)


def test_playing_outside_a_region_does_not_repeat(ctx, device):
    _loaded(ctx)
    device.busy, device.tag_value = True, "a"
    muse_mode.seek(ctx, 1.0)
    assert device.calls[-1]["loops"] == 0


def test_seeking_inside_the_region_loops_the_whole_region_not_just_the_tail(ctx, device):
    """M10 repro: seeking to 5s inside a 2-8s region used to slice
    ``pcm[start:loop_end]`` and loop *that* -- a buffer of only three seconds,
    not the marked six-second region. Fails against the unfixed code, whose
    buffer length there is ``loop_end - seek`` rather than
    ``loop_end - loop_start``.
    """
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 8.0)
    _settle(ctx, one)
    muse_mode.play_region(ctx)  # start the loop, as "Play the loop" does
    muse_mode.seek(ctx, 5.0)
    call = device.calls[-1]
    assert call["loops"] == -1
    assert call["frames"] == pytest.approx(6 * RATE, rel=0.01)


def test_seeking_before_the_region_does_not_force_a_loop(ctx, device):
    """A region existing at all used to loop *any* seek, even one that lands
    outside it -- ``repeat`` depended only on whether a region was set, never
    on whether the seek landed inside it. Fails against the unfixed code,
    which reports ``loops == -1`` here.
    """
    _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 8.0)
    device.busy, device.tag_value = True, "a"
    muse_mode.seek(ctx, 0.5)
    assert device.calls[-1]["loops"] == 0


def test_seeking_after_the_region_does_not_force_a_loop(ctx, device):
    """The mirror of the above, and a worse instance of the same bug: seeking
    past the region's end used to loop the unbounded remainder to the end of
    the take, forever. Fails against the unfixed code (``loops == -1`` and a
    one-second buffer standing in for the whole remainder).
    """
    _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 8.0)
    device.busy, device.tag_value = True, "a"
    muse_mode.seek(ctx, 9.0)
    assert device.calls[-1]["loops"] == 0
    assert device.calls[-1]["frames"] == pytest.approx(1 * RATE, rel=0.01)


def test_shrinking_the_region_while_sounding_is_picked_up_on_the_next_seek(ctx, device):
    """Changing the markers mid-loop must be reflected the next time the take
    is actually replayed. Fails against the unfixed code, whose loop buffer is
    always "seek point to loop end" (1s here) rather than the full, narrowed
    region (2s).
    """
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 8.0)
    _settle(ctx, one)
    muse_mode.play_region(ctx)
    muse_mode.set_region(ctx, 3.0, 5.0)  # narrowed while sounding
    _settle(ctx, one)
    device.busy, device.tag_value = True, "a"
    muse_mode.seek(ctx, 4.0)
    assert device.calls[-1]["frames"] == pytest.approx(2 * RATE, rel=0.01)


def test_the_position_wraps_within_the_region_after_a_seek_inside_it(ctx, device):
    """Old code's ``play_offset`` was the seek point and its buffer began
    there too, so adding the mixer's raw clock straight to it was correct by
    accident, for a buffer that was the wrong length. With the loop rotated to
    start at the seek point, the wrap has to be computed against the
    *region's* length and re-based at ``loop_start`` -- this pins that
    arithmetic. Fails against the unfixed code, which returns 8.5 here
    (``play_offset`` 5.0 plus the raw clock 3.5, with no wrap at all).
    """
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 8.0)
    _settle(ctx, one)
    device.busy, device.tag_value = True, "a"
    muse_mode.seek(ctx, 5.0)
    device.pos = 3.5
    assert muse_mode.position(ctx) == pytest.approx(2.5)


# --- resuming a loaded take (M07) --------------------------------------------


def test_stop_captures_the_playhead_rather_than_losing_it(ctx, device):
    """M07: the playhead the mixer was actually at must survive Stop. Fails
    against the unfixed code, which never reads ``position(ctx)`` before
    stopping and leaves ``play_offset`` wherever the last ``_play_from`` call
    had started -- 3.0 here, not the 4.0 the take had actually reached.
    """
    one = _loaded(ctx, seconds=10.0)
    one.play_offset = 3.0
    device.busy, device.tag_value, device.pos = True, "a", 1.0
    muse_mode.stop(ctx)
    assert one.play_offset == pytest.approx(4.0)


def test_stop_then_play_resumes_the_same_take_without_losing_its_state(
    ctx, device, monkeypatch
):
    """M07: pressing Play on an already-loaded take must not rebuild the
    ``Player``. The old code always resubmitted the decode -- so a 2-8s
    region, a 200ms crossfade and a mid-take offset all reverted to nothing on
    the very next Play, since ``on_task_done`` builds a brand-new
    ``MusePlayer`` for every successful load. Fails against the unfixed code,
    which submits a fresh decode (``ctx.submitted`` is non-empty) for a take
    already sitting in memory with nothing to re-read.
    """
    from .test_muse_mode import _finished

    _finished(ctx, "a")
    monkeypatch.setattr(muse_mode, "_read_track", lambda path: {"pcm": [], "rate": RATE})
    one = _loaded(ctx, seconds=10.0, job="a")
    muse_mode.set_region(ctx, 2.0, 8.0)
    one.xfade_ms = 200.0
    _settle(ctx, one)
    ctx.submitted.clear()  # the settle's own precompute must not count below
    device.busy, device.tag_value, device.pos = True, "a", 1.0
    one.play_offset = 3.0
    muse_mode.stop(ctx)
    assert one.play_offset == pytest.approx(4.0)

    muse_mode.play(ctx, "a")
    assert ctx.submitted == [], "an already-decoded take must be resumed, not re-read"
    resumed = muse_mode.player(ctx)
    assert resumed is one, "the same Player, not a fresh one built by on_task_done"
    assert (resumed.loop_start, resumed.loop_end) == (2.0, 8.0)
    assert resumed.xfade_ms == pytest.approx(200.0)
    call = device.calls[-1]
    assert call["loops"] == -1  # 4.0 is inside the (2, 8) region
    assert call["frames"] == pytest.approx(6 * RATE, rel=0.01)


def test_pressing_play_with_no_audio_device_tells_the_user_rather_than_doing_nothing(
    ctx, device
):
    """muse-01 (2026-09-11 audit): ``_play_from``'s two ``sirens_audio.play``
    calls had no ``else`` branch at all, unlike the first-decode landing path
    in ``on_task_done`` (see ``test_a_device_that_refuses_leaves_nothing_
    claiming_to_play`` in ``test_muse_mode.py``). Resuming an already-decoded
    take -- the ordinary Stop-then-Play case -- is exactly the path that
    never goes through ``on_task_done`` at all, so on a device-less machine it
    silently did nothing. Fails against the unfixed code, whose ``ctx.toasts``
    stays empty after both presses below.
    """
    one = _loaded(ctx, seconds=10.0, job="a")
    device.refuses = True

    # The plain, out-of-region path: Stop, then Play again on a take already
    # sitting in memory (M07's resume branch), never re-reading the file.
    muse_mode.play(ctx, "a")
    assert ctx.toasts, "a refused resume must say so rather than doing nothing"
    assert ctx.toasts[-1] == ("no device", "warn")
    assert one.play_offset == pytest.approx(0.0), "a refused play must not move the playhead"

    # The looping, in-region path: "Play the loop". Settled first (a device
    # refusal is a mixer-side question, not a cache one, so this must not be
    # mistaken for muse-02's deferred-play toast instead).
    ctx.toasts.clear()
    muse_mode.set_region(ctx, 2.0, 6.0)
    _settle(ctx, one)
    muse_mode.play_region(ctx)
    assert ctx.toasts, "a refused loop play must say so rather than doing nothing"
    assert ctx.toasts[-1] == ("no device", "warn")


# --- one buffer for the audition and the export (M09) ------------------------


def test_playing_the_loop_hands_the_mixer_the_export_buffer_not_a_raw_slice(
    ctx, device, monkeypatch, tmp_path
):
    """M09: before this, ``play_region``/seeking inside the region handed the
    mixer a raw, uncrossfaded slice of ``pcm`` while ``export_loop``
    crossfaded independently on write -- so the crossfade slider could be
    dragged to any value with no audible difference through the advertised
    audition. Fails against the unfixed code, whose buffer here is the plain
    slice and does not match ``muse.loops.crossfade``'s output at all once the
    fade is non-zero.

    ``precompute_loop`` is called and its result routed through
    ``on_task_done`` before ``play_region`` here (muse-02, the 2026-09-18
    audit): a region set and a crossfade changed by hand, exactly as this test
    does, leave the cache stale, and a stale cache now makes ``play_region``
    defer rather than block the frame thread to blend it -- see
    ``test_play_pressed_immediately_after_a_region_change_does_not_block_the_
    frame_thread`` and ``test_a_deferred_play_starts_once_the_precompute_lands``
    below for that half. Landing the cache first here is what the real
    controls usually do too: the marker grip and the crossfade slider both
    call ``precompute_loop`` on release, well before a user's next Play.
    """
    from realmspinner.studio.modes.muse.engine import loops as loops_mod

    one = _loaded(ctx, seconds=10.0)
    rng = np.random.default_rng(0)
    one.pcm = (rng.standard_normal((10 * RATE, 2)) * 5000).astype(np.int16)
    muse_mode.set_region(ctx, 2.0, 8.0)
    one.xfade_ms = 200.0
    rate = one.rate
    expected = loops_mod.crossfade(
        one.pcm, int(2.0 * rate), int(8.0 * rate), int(200.0 * rate / 1000.0)
    )

    _settle(ctx, one)

    muse_mode.play_region(ctx)  # phase 0: starts at the region's own start
    assert np.array_equal(device.calls[-1]["pcm"], expected)

    out = tmp_path / "loop.wav"
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: out)
    muse_io.export_loop(ctx, one)
    with wave.open(str(out)) as handle:
        raw = handle.readframes(handle.getnframes())
    exported = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
    assert np.array_equal(exported, expected), "the export must write the same buffer"


def test_play_pressed_immediately_after_a_region_change_does_not_block_the_frame_thread(
    ctx, device, monkeypatch
):
    """muse-02 (2026-09-18 audit).

    ``_play_from`` used to call ``muse_io.loop_body`` unconditionally on a
    marked region -- and on a cache miss (exactly the state right after a
    region or crossfade changes, before ``precompute_loop``'s task lands)
    that function blends the whole region on whatever thread called it. Called
    from ``_play_from``, that thread is the frame thread: about 100 ms on a
    240 s take, the very stall muse-03 (2026-09-05 audit) fixed for every
    *other* caller of the blend, just left open for the one caller that runs
    on a Play press right after a marker moves.

    Asserted two ways: ``muse_io.loop_body`` -- the blocking blend -- must
    never be called while the cache is stale, and the mixer must not have
    been handed anything yet. Fails against the unfixed code, whose
    ``loop_body`` patch below is invoked and raises.
    """

    def _blend_must_not_run_here(*_a, **_kw):
        raise AssertionError("loop_body must not blend on the frame thread")

    _loaded(ctx, seconds=10.0)
    monkeypatch.setattr(muse_io, "loop_body", _blend_must_not_run_here)
    muse_mode.set_region(ctx, 2.0, 8.0)  # cache is stale: no precompute ran

    muse_mode.play_region(ctx)

    assert device.calls == [], "must not have started sounding an unblended region"
    assert any(
        key.startswith(muse_io.CACHE_PREFIX) for key in ctx.submitted
    ), "must kick off the precompute rather than leaving the stall permanent"
    assert ctx.toasts, "a deferred loop play must say so rather than doing nothing"


def test_a_deferred_play_starts_once_the_precompute_lands(ctx, device):
    """muse-02 (2026-09-18 audit), the second half of the same finding: a
    first pass refused outright ("try again in a moment"), which made a user
    press Play twice for something they asked for once. Deferred instead --
    the take actually starts sounding once ``on_task_done`` installs the
    cache this Play was waiting for, with no second press. Fails against a
    refuse-only fix, whose ``device.calls`` stays empty after the landing
    below.
    """
    one = _loaded(ctx, seconds=10.0)
    ctx.state.mode = "muse"  # on_task_done only starts sounding while Muse is up
    muse_mode.set_region(ctx, 2.0, 8.0)  # cache is stale: no precompute ran

    muse_mode.play_region(ctx)
    assert device.calls == [], "must not sound anything before the blend lands"
    assert one.pending_play is not None

    done = type("_Done", (), {"key": ctx.submitted[-1], "result": ctx.result})()
    muse_mode.on_task_done(ctx, done)

    assert device.calls, "the deferred play must start once the cache lands"
    call = device.calls[-1]
    assert call["loops"] == -1
    assert call["frames"] == pytest.approx(6 * RATE, rel=0.01)  # the (2, 8) region
    assert one.pending_play is None, "a landed request must not linger for a second play"


def test_a_region_change_before_the_precompute_lands_drops_the_pending_play(ctx, device):
    """muse-02 (2026-09-18 audit): the region the user was waiting to hear is
    not the region that has landed by the time it does. Dropped silently --
    no toast, no sound -- rather than starting a loop for a region the
    marker has since moved off of.
    """
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 8.0)
    muse_mode.play_region(ctx)  # pending_play waits on the (2, 8) key
    stale_done = type("_Done", (), {"key": ctx.submitted[-1], "result": ctx.result})()

    muse_mode.set_region(ctx, 3.0, 5.0)  # the marker moves before the blend lands
    ctx.toasts.clear()

    muse_mode.on_task_done(ctx, stale_done)

    assert device.calls == [], "a landing for an abandoned region must not start anything"
    assert one.pending_play is None, "the stale request must not linger for a later landing"
    assert ctx.toasts == [], "dropped silently -- the region already moved on with no help needed"


def test_switching_away_from_muse_before_the_precompute_lands_drops_the_pending_play(
    ctx, device
):
    """muse-02 (2026-09-18 audit): a Play request made in Muse must not start
    a take sounding out from under a user who has since moved to another
    mode -- ``on_task_done`` runs on every frame's tasks, whichever mode is
    drawn.
    """
    one = _loaded(ctx, seconds=10.0)
    ctx.state.mode = "muse"
    muse_mode.set_region(ctx, 2.0, 8.0)
    muse_mode.play_region(ctx)
    done = type("_Done", (), {"key": ctx.submitted[-1], "result": ctx.result})()

    ctx.state.mode = "library"  # the user switched away before it landed

    muse_mode.on_task_done(ctx, done)

    assert device.calls == [], "must not start sounding while another mode is up"
    assert one.pending_play is None, "the request must not linger for a later landing"


def test_stop_withdraws_a_deferred_play(ctx, device):
    """muse-02 (2026-09-18 audit): Stop is the user taking a Play request
    back. A press that never got as far as sounding anything must not start
    once the blend it was waiting on lands anyway.
    """
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 8.0)
    muse_mode.play_region(ctx)
    done = type("_Done", (), {"key": ctx.submitted[-1], "result": ctx.result})()

    muse_mode.stop(ctx)
    assert one.pending_play is None

    muse_mode.on_task_done(ctx, done)
    assert device.calls == [], "a withdrawn request must not start sounding on landing"


# --- the region --------------------------------------------------------------


def test_a_reversed_region_is_ordered_rather_than_refused(ctx, device):
    """One place clamps and orders, because four surfaces set these markers --
    the finder, the two grips and the two keys."""
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 8.0, 2.0)
    assert (one.loop_start, one.loop_end) == (2.0, 8.0)


def test_a_region_is_clamped_to_the_take(ctx, device):
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, -4.0, 99.0)
    assert (one.loop_start, one.loop_end) == (0.0, 10.0)


def test_clearing_takes_both_markers(ctx, device):
    one = _loaded(ctx)
    muse_mode.set_region(ctx, 1.0, 2.0)
    muse_mode.set_region(ctx, None, None)
    assert one.loop_start is None and one.loop_end is None


def test_choosing_a_candidate_adopts_it_as_the_region(ctx, device):
    one = _loaded(ctx, seconds=10.0)
    from realmspinner.studio.modes.muse.engine.loops import Candidate

    one.candidates = [Candidate(RATE, RATE * 5, 0.1), Candidate(0, RATE * 3, 0.2)]
    muse_mode.choose_candidate(ctx, 1)
    assert (one.loop_start, one.loop_end) == (0.0, 3.0)


def test_choosing_a_candidate_that_is_not_there_does_nothing(ctx, device):
    one = _loaded(ctx)
    muse_mode.choose_candidate(ctx, 7)
    assert one.loop_start is None


# --- the finder --------------------------------------------------------------


def test_the_finder_runs_on_a_task_and_its_answer_lands_in_on_task_done(ctx, device):
    one = _loaded(ctx, seconds=3.0)
    muse_mode.find_loops(ctx)
    assert ctx.submitted[-1] == f"{muse_io.FIND_PREFIX}a"
    assert one.finding is True

    from realmspinner.studio.modes.muse.engine.loops import Candidate

    done = type("_Done", (), {"key": f"{muse_io.FIND_PREFIX}a", "result": [
        Candidate(0, RATE * 2, 0.1)
    ]})()
    muse_mode.on_task_done(ctx, done)
    assert one.finding is False
    # The best one is adopted immediately: the finder's output is a ranking,
    # and a second press to hear the answer it already has is a step with no
    # decision in it.
    assert (one.loop_start, one.loop_end) == (0.0, 2.0)


def test_a_stale_find_loops_result_does_not_override_a_region_set_after_the_search_was_abandoned(
    ctx, device
):
    """muse-03 (2026-09-11 audit): the FIND_PREFIX branch matched on the
    player's job id alone, with no ``one.finding`` guard -- unlike the
    LOAD_PREFIX branch a few lines below it, which checks ``audition_job``
    before adopting. A search abandoned by switching to another take and back
    rebuilds a *fresh* ``Player`` (``on_task_done``'s LOAD_PREFIX branch
    always does), whose ``finding`` defaults back to ``False`` -- but it
    carries the same job id the abandoned search's key names, so the search's
    answer still matched and silently overwrote a region the user had since
    restored by hand. Fails against the unfixed code, whose region here ends
    up ``(0.0, 2.0)`` -- the finder's stale candidate -- rather than the
    hand-set ``(5.0, 9.0)``.
    """
    one = _loaded(ctx, seconds=10.0, job="a")
    # The search was requested and then abandoned -- switching away and back
    # built a fresh Player for the same job id, which is why ``finding`` is
    # false even though the outstanding task's key still names this job.
    one.finding = False
    muse_mode.set_region(ctx, 5.0, 9.0)  # restored/hand-set after the search

    from realmspinner.studio.modes.muse.engine.loops import Candidate

    done = type("_Done", (), {
        "key": f"{muse_io.FIND_PREFIX}a", "result": [Candidate(0, RATE * 2, 0.1)],
    })()
    muse_mode.on_task_done(ctx, done)

    assert (one.loop_start, one.loop_end) == (5.0, 9.0), (
        "a search nobody is waiting on any more must not override a region "
        "set since it was abandoned"
    )


def test_an_answer_for_a_different_take_is_ignored(ctx, device):
    """The player holds one take; a result that arrives after the user moved on
    describes samples that are no longer in memory."""
    one = _loaded(ctx, job="a")
    one.finding = True
    done = type("_Done", (), {"key": f"{muse_io.FIND_PREFIX}b", "result": []})()
    muse_mode.on_task_done(ctx, done)
    assert one.finding is True


def test_no_candidates_says_so_rather_than_leaving_a_spinner(ctx, device):
    one = _loaded(ctx)
    one.finding = True
    done = type("_Done", (), {"key": f"{muse_io.FIND_PREFIX}a", "result": []})()
    muse_mode.on_task_done(ctx, done)
    assert one.finding is False
    assert any("No loop points" in message for message, _ in ctx.toasts)


# --- lifecycle ---------------------------------------------------------------


def test_a_decoded_take_becomes_the_player(ctx, device):
    from realmspinner.studio.modes.muse.engine import waveform

    muse_mode.ensure(ctx).audition_job = "a"
    pcm = np.zeros((RATE, 2), dtype=np.int16)
    done = type("_Done", (), {
        "key": f"{muse_mode.LOAD_PREFIX}a",
        "result": {
            "pcm": pcm, "rate": RATE, "env": waveform.peaks(pcm), "duration": 1.0
        },
    })()
    muse_mode.on_task_done(ctx, done)
    one = muse_mode.player(ctx)
    assert one is not None and one.job == "a"
    assert one.duration == pytest.approx(1.0)
    assert device.calls[-1]["tag"] == "a"


def test_the_player_is_dropped_when_its_take_leaves_the_library(ctx, device):
    """~42 MB for four minutes. One take at a time, and this is the half that
    lets go of it."""
    _loaded(ctx, job="gone")
    muse_mode.sync(ctx)
    assert muse_mode.player(ctx) is None


def test_a_player_survives_a_frame_where_the_cache_is_empty(ctx, device):
    """An empty cache is "not loaded yet", not "every take was deleted" -- and
    dropping the buffer on a refresh frame would stop playback mid-audition."""
    _loaded(ctx, job="a")
    ctx.cache.jobs = []
    muse_mode.sync(ctx)
    assert muse_mode.player(ctx) is not None


class _CountingJobs(list):
    """A jobs list that counts how many rows a scan actually visits.

    Timing a per-frame cost is meaningless under contention (CLAUDE.md's own
    rule for the ``perf`` lane); counting work is what stays true regardless
    of the machine this test runs on.
    """

    def __init__(self, jobs) -> None:
        super().__init__(jobs)
        self.visited = 0

    def __iter__(self):
        for item in list.__iter__(self):
            self.visited += 1
            yield item


def test_sync_does_not_allocate_a_set_of_every_job_every_frame(ctx, device):
    """muse-06 (2026-09-11 audit): ``sync`` built a ``set`` of every job id in
    ``ctx.cache.jobs`` on every frame just to test whether one id -- the
    loaded player's -- was a member, so the cost scaled with the whole
    library's size and was paid 60 times a second for as long as the tray was
    drawn. The loaded job is first in the list here, so a short-circuiting
    membership scan visits exactly one row; a ``set`` comprehension has no
    short circuit and must still visit all four to build it. Fails against
    the unfixed code, which reports ``visited == 4``.
    """
    one = _loaded(ctx, job="target")
    jobs = _CountingJobs([{"id": "target"}, {"id": "b"}, {"id": "c"}, {"id": "d"}])
    ctx.cache.jobs = jobs
    muse_mode.sync(ctx)
    assert muse_mode.player(ctx) is one, "the loaded take must not have been dropped"
    assert jobs.visited == 1, (
        "a membership check for the first row in the list must stop there, "
        "not build a set of every row in the library"
    )


# --- export ------------------------------------------------------------------


def _read(data: bytes) -> tuple[int, int, int]:
    with wave.open(io.BytesIO(data)) as handle:
        return handle.getframerate(), handle.getnchannels(), handle.getnframes()


def test_the_loop_export_is_the_crossfaded_body(ctx, device, monkeypatch, tmp_path):
    one = _loaded(ctx, seconds=10.0)
    one.pcm = (np.ones((10 * RATE, 2)) * 1000).astype(np.int16)
    muse_mode.set_region(ctx, 2.0, 6.0)
    out = tmp_path / "loop.wav"
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: out)
    muse_io.export_loop(ctx, one)
    rate, channels, frames = _read(out.read_bytes())
    assert (rate, channels) == (RATE, 2)
    assert frames == pytest.approx(4 * RATE, rel=0.01)


def test_the_loop_export_carries_its_points_over_the_whole_file(ctx, device, monkeypatch, tmp_path):
    """The whole file *is* the loop, which is what tells an engine to repeat it
    seamlessly rather than to find the seam itself."""
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 6.0)
    out = tmp_path / "loop.wav"
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: out)
    muse_io.export_loop(ctx, one)
    assert b"smpl" in out.read_bytes()


def test_loop_points_are_refused_over_a_crossfade(ctx, device, monkeypatch, tmp_path):
    """The exclusivity, held at the door as well as at the button.

    The samples at a crossfaded seam do not exist in the take, so an ``smpl``
    chunk pointing into the untouched file is a loop that clicks wearing a label
    saying it does not.
    """
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 6.0)
    one.xfade_ms = 40.0
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: tmp_path / "x.wav")
    muse_io.export_with_points(ctx, one)
    assert not (tmp_path / "x.wav").exists()
    assert any("loop points" in message for message, _ in ctx.toasts)


def test_loop_points_over_a_plain_seam_write_the_whole_take(ctx, device, monkeypatch, tmp_path):
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 6.0)
    one.xfade_ms = 0.0
    out = tmp_path / "track.wav"
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: out)
    muse_io.export_with_points(ctx, one)
    _, _, frames = _read(out.read_bytes())
    assert frames == 10 * RATE
    assert b"smpl" in out.read_bytes()


def test_an_export_with_no_region_says_so_rather_than_writing_nothing(ctx, device):
    one = _loaded(ctx)
    muse_io.export_loop(ctx, one)
    assert any("loop region" in message for message, _ in ctx.toasts)
    assert ctx.submitted == []


def test_a_cancelled_picker_writes_no_file(ctx, device, monkeypatch, tmp_path):
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 6.0)
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: None)
    muse_io.export_loop(ctx, one)
    assert list(tmp_path.glob("*.wav")) == []


# --- the grips (muse-06) ------------------------------------------------------


def test_grip_at_picks_the_nearer_marker_when_both_are_in_reach():
    """muse-06: ``_grip_at`` used to return the *first* grip within reach in a
    fixed ``("start", "end")`` order, so a click plainly closer to the end
    grip was reported as a grab on start whenever the region was narrow enough
    -- on screen -- to put both grips within reach at once. That is exactly
    the region a user is trying to tighten: one narrower than two grips wide.

    10 px/second here, a 0.6 s region (6 px, under the 5 px-either-side reach
    that puts both grips in range at once), clicked one pixel from the end
    marker and five from the start. Fails against the unfixed code, which
    returns "start" because it is checked first and is still, barely, within
    reach.
    """
    from realmspinner.studio.modes.muse.ui.panes import player as muse_player

    one = muse_state.Player(job="a", duration=100.0, loop_start=10.0, loop_end=10.6)
    width = 1000.0  # 10 design pixels per second, at the tokens.SCALE default
    assert muse_player._grip_at(one, 105.0, width) == "end"


def test_grip_at_breaks_an_exact_tie_toward_start():
    """The mirror of the case above: equidistant from both grips, the pick
    must still be deterministic rather than whichever the loop visits last."""
    from realmspinner.studio.modes.muse.ui.panes import player as muse_player

    one = muse_state.Player(job="a", duration=100.0, loop_start=10.0, loop_end=10.6)
    width = 1000.0
    assert muse_player._grip_at(one, 103.0, width) == "start"


# --- the disabled reasons (muse-07) -------------------------------------------


def test_muse_disabled_control_reasons_are_pure_and_testable():
    """muse-07: every disabled-control sentence in the player strip and the
    brief used to be a string literal chosen inline in the ternary passed to
    the draw call, so a change to ``has_region`` or ``plain`` that made a
    reason stale or wrong would be caught by nothing -- the 2026-09-02
    review's T4 defect, in the one mode whose reasons had not been extracted
    the way ``plotter_menu._layer_reason``, ``inker_mode._no_document_reason``
    and ``overlay.cancel_reason`` were. These are asserted with no imgui frame
    at all, which is the point.
    """
    from realmspinner.studio.modes.muse.ui import brief as muse_brief
    from realmspinner.studio.modes.muse.ui.panes import player as muse_player

    assert muse_player._no_region_reason(True) == ""
    assert muse_player._no_region_reason(False) == "no loop region yet"

    assert muse_player._export_points_reason(True) == ""
    blocked = muse_player._export_points_reason(False)
    assert "loop points" in blocked and blocked != ""

    assert muse_brief._generate_reason(False) == ""
    missing = muse_brief._generate_reason(True)
    assert "not downloaded" in missing


def test_the_trays_disabled_reasons_are_pure_and_testable():
    """The 2026-09-08 audit, finding muse-04. The tray (``muse_results.py``)
    had four ``reason='' if ready else 'this take has not finished yet'``
    inline literals -- the same pattern **muse-07** (2026-09-05 audit) pulled
    out of the brief and the player strip in
    ``test_muse_disabled_control_reasons_are_pure_and_testable`` above, just
    above this test. This is that guard's missing third surface: a future
    edit to the tray's sentence, or a bug that greys a card button for the
    wrong reason, has nothing else in the suite to catch it.
    """
    from realmspinner.studio.modes.muse.ui.panes import results as muse_results

    assert muse_results._ready_reason(True) == ""
    not_ready = muse_results._ready_reason(False)
    assert "not finished yet" in not_ready and not_ready != ""


def test_the_stems_already_split_reason_is_a_pure_testable_function():
    """muse-05 (2026-09-11 audit): the Stems button's "already split" sentence
    was still chosen inline in ``_actions``' ternary -- ``"this take has
    already been split" if stems else _ready_reason(ready)`` -- the one
    literal **muse-04**/**muse-07** missed when they pulled every other
    disabled-control sentence in this mode into a pure function. Fails
    against the unfixed code, which has no ``_stems_reason`` at all.
    """
    from realmspinner.studio.modes.muse.ui.panes import results as muse_results

    assert muse_results._stems_reason(True, False) == ""
    not_ready = muse_results._stems_reason(False, False)
    assert "not finished yet" in not_ready and not_ready != ""
    split = muse_results._stems_reason(True, True)
    assert "already been split" in split and split != ""
    # Already-split wins even over "not ready" -- a status this file's own
    # ``has_stems``/``ready`` combination cannot actually produce (a queued
    # take has no stems), but the reason function should still be able to
    # say which fact it is reporting rather than picking one arbitrarily.
    assert muse_results._stems_reason(False, True) == "this take has already been split"

    # **Revert-sensitive.** The four calls above only exercise
    # ``_stems_reason``'s return value, and a re-inlined ternary at the
    # ``_actions`` call site -- keeping ``_stems_reason`` defined but unused,
    # which is muse-05's exact regression -- would still pass every one of
    # them, since nothing above asks *where* the sentence is written. Parsed
    # with ``ast`` and scoped to each function's own body so a coincidental
    # second appearance of the phrase elsewhere in the module (a docstring, a
    # comment) cannot pass this by accident: the sentence must live inside
    # ``_stems_reason`` and nowhere inside ``_actions``.
    import ast
    import inspect
    from pathlib import Path

    source = Path(inspect.getfile(muse_results)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in ("_stems_reason", "_actions")
    }
    assert set(functions) == {"_stems_reason", "_actions"}, "both functions must exist"

    def _mentions(node: ast.AST) -> int:
        return sum(
            1
            for sub in ast.walk(node)
            if isinstance(sub, ast.Constant)
            and isinstance(sub.value, str)
            and "already been split" in sub.value
        )

    assert _mentions(functions["_stems_reason"]) == 1, (
        "the sentence must be written exactly once, inside _stems_reason"
    )
    assert _mentions(functions["_actions"]) == 0, (
        "_actions must ask _stems_reason for the sentence, not spell it again "
        "inline at the call site"
    )


def test_the_extend_menu_item_reason_is_a_pure_testable_function():
    """The 2026-09-18 audit, finding muse-01, the menu half.

    ``derive_music_job`` refuses on the *combined* total,
    ``parent_duration + extend_left + extend_right`` against the sampler's
    frame ceiling -- so a take already at or past that ceiling has zero
    seconds of room for any nonzero extension at all. ``_extend_reason``
    is the pure function ``_derive_menu`` asks before greying "Extend" on
    the "Make more" menu, in the same shape this file's other
    ``*_reason`` tests already hold every other disabled control to.
    """
    from realmspinner.studio.modes.muse.ui.panes import results as muse_results

    assert muse_results._extend_reason(0.0) == ""
    assert muse_results._extend_reason(60.0) == ""
    at_ceiling = muse_results._extend_reason(muse_results._extend_ceiling())
    assert at_ceiling != "" and "extend ceiling" in at_ceiling
    past_ceiling = muse_results._extend_reason(600.0)
    assert past_ceiling != "" and "extend ceiling" in past_ceiling


# --- the untouched marker's anchor (muse-02) ----------------------------------


def test_dragging_a_grip_past_the_other_marker_keeps_the_untouched_one_fixed(
    ctx, monkeypatch
):
    """muse-02 (2026-09-07 audit): ``_input`` used to read the *other* marker's
    bound live off ``one`` every frame of a drag, rather than once at grab
    time. That is fine until the drag crosses it: ``muse_mode.set_region``
    sorts the pair, so the untouched marker's value moves onto the field the
    dragged one used to occupy -- and the next frame's "live" read of that
    field is actually last frame's *dragged* position, not the marker the user
    never touched. Each frame's sort fed back in as the next frame's anchor,
    so a drag that crossed the other marker and came back left it drifted
    rather than restored.

    Grabbing the start grip at 8.0s (region 8-12s), dragging out to 16.0s --
    past the end marker at 12.0s, forcing ``set_region`` to sort the pair --
    and back to the original grab point at 8.0s should leave the region
    exactly where it started, (8.0, 12.0): the end marker was never touched.
    Fails against the unfixed code, which reports (8.0, 16.0) instead --
    the crossing frame's dragged value, 16.0, in place of the true anchor.
    """
    from imgui_bundle import imgui

    from realmspinner.studio.modes.muse.engine import waveform
    from realmspinner.studio.modes.muse.ui.panes import player as muse_player

    one = _loaded(ctx, seconds=20.0)
    muse_mode.set_region(ctx, 8.0, 12.0)
    width = 1000.0  # 50 px/s at a 20s take -- well past GRIP_W's 5 px reach.
    origin = type("Origin", (), {"x": 0.0, "y": 0.0})()

    previous = imgui.get_current_context()
    gl_ctx = imgui.create_context()
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)

        def _at(seconds: float, *, active: bool, activated: bool, deactivated: bool):
            monkeypatch.setattr(imgui, "is_item_active", lambda: active)
            monkeypatch.setattr(imgui, "is_item_activated", lambda: activated)
            monkeypatch.setattr(imgui, "is_item_deactivated", lambda: deactivated)
            io.mouse_pos = (waveform.at(seconds, one.duration, width), 0.0)

        # Frame 1: press down exactly on the start grip. Grabs "start"; the
        # region is unchanged because the pointer has not moved yet.
        _at(8.0, active=True, activated=True, deactivated=False)
        muse_player._input(ctx, one, origin, width)
        assert (one.loop_start, one.loop_end) == (8.0, 12.0)

        # Frame 2: drag past the end marker. This is the crossing frame: the
        # pair gets sorted and the untouched marker's value (12.0) moves onto
        # ``loop_start``, the field this drag keeps writing to.
        _at(16.0, active=True, activated=False, deactivated=False)
        muse_player._input(ctx, one, origin, width)
        assert (one.loop_start, one.loop_end) == (12.0, 16.0)

        # Frame 3: release back at the original grab point. A drag that ends
        # where it began should leave the region exactly as it found it.
        _at(8.0, active=False, activated=False, deactivated=True)
        muse_player._input(ctx, one, origin, width)
        assert (one.loop_start, one.loop_end) == (8.0, 12.0)
    finally:
        imgui.destroy_context(gl_ctx)
        if previous is not None:
            imgui.set_current_context(previous)


def test_the_drawn_playhead_does_not_move_until_a_seek_drag_releases(ctx, monkeypatch):
    """The 2026-09-08 audit, finding muse-05. ``_input``'s docstring used to
    claim the drawn playhead is "drawn from the pending value" during a plain
    click-drag-to-seek and "the sound catches up when they let go" -- implying
    live visual feedback during the drag. For a plain seek (no grip grabbed),
    nothing in ``_input`` runs while ``imgui.is_item_active()`` is true:
    ``muse_mode.seek`` -- the only thing that moves ``play_offset``, which
    ``_playhead`` reads through ``muse_mode.position`` -- only fires from the
    ``is_item_deactivated()`` branch. So the position seen by a caller reading
    it mid-drag must equal the pre-drag value, and only becomes the drag's
    target once the gesture releases.
    """
    from imgui_bundle import imgui

    from realmspinner.studio.modes.muse.ui.panes import player as muse_player

    one = _loaded(ctx, seconds=20.0)
    width = 1000.0
    origin = type("Origin", (), {"x": 0.0, "y": 0.0})()
    before = one.play_offset

    previous = imgui.get_current_context()
    gl_ctx = imgui.create_context()
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)

        def _at(seconds: float, *, active: bool, activated: bool, deactivated: bool):
            monkeypatch.setattr(imgui, "is_item_active", lambda: active)
            monkeypatch.setattr(imgui, "is_item_activated", lambda: activated)
            monkeypatch.setattr(imgui, "is_item_deactivated", lambda: deactivated)
            io.mouse_pos = (seconds / one.duration * width, 0.0)

        # Press, away from either grip (there is no region at all): grabs "".
        _at(5.0, active=True, activated=True, deactivated=False)
        muse_player._input(ctx, one, origin, width)
        assert one.play_offset == before

        # Mid-drag, still held. The docstring's old claim was that the drawn
        # position tracks the pointer here -- it must not.
        _at(10.0, active=True, activated=False, deactivated=False)
        muse_player._input(ctx, one, origin, width)
        assert one.play_offset == before

        # Release: only now does the position catch up to where the drag let go.
        _at(10.0, active=False, activated=False, deactivated=True)
        muse_player._input(ctx, one, origin, width)
        assert one.play_offset == pytest.approx(10.0)
    finally:
        imgui.destroy_context(gl_ctx)
        if previous is not None:
            imgui.set_current_context(previous)


# --- the player's six keys (muse-05, the 2026-09-18 audit) -------------------
#
# ``handle_key``'s own docstring names six bindings for a loaded take --
# Left/Right (and Shift for the ten-times step), Home, ``[``/``]`` and ``L`` --
# and none of them had a test anywhere in this file before this section: an
# evidence gap, not a bug found, so nothing here is expected to fail against
# the unfixed code. What follows is the coverage the finding asked for.


def test_right_arrow_nudges_the_playhead_forward(ctx, device):
    import pygame

    from .test_muse_mode import _key

    one = _loaded(ctx, seconds=10.0)
    assert muse_mode.handle_key(ctx, _key(pygame.K_RIGHT)) is True
    assert one.play_offset == pytest.approx(muse_mode.NUDGE_SECONDS)


def test_left_arrow_nudges_the_playhead_backward_and_clamps_at_zero(ctx, device):
    import pygame

    from .test_muse_mode import _key

    one = _loaded(ctx, seconds=10.0)
    one.play_offset = 0.5
    assert muse_mode.handle_key(ctx, _key(pygame.K_LEFT)) is True
    assert one.play_offset == pytest.approx(0.0)


def test_shift_makes_the_nudge_ten_times_as_far(ctx, device):
    import pygame

    from .test_muse_mode import _key

    one = _loaded(ctx, seconds=30.0)
    assert muse_mode.handle_key(ctx, _key(pygame.K_RIGHT, pygame.KMOD_SHIFT)) is True
    assert one.play_offset == pytest.approx(
        muse_mode.NUDGE_SECONDS * muse_mode.NUDGE_MULTIPLIER
    )


def test_home_returns_the_playhead_to_the_start(ctx, device):
    import pygame

    from .test_muse_mode import _key

    one = _loaded(ctx, seconds=10.0)
    one.play_offset = 5.0
    assert muse_mode.handle_key(ctx, _key(pygame.K_HOME)) is True
    assert one.play_offset == pytest.approx(0.0)


def test_left_bracket_sets_the_region_start_at_the_playhead_and_precomputes(ctx, device):
    import pygame

    from .test_muse_mode import _key

    one = _loaded(ctx, seconds=10.0)
    one.play_offset = 3.0
    assert muse_mode.handle_key(ctx, _key(pygame.K_LEFTBRACKET)) is True
    # No end set yet, so ``[`` is ordered against the take's own duration --
    # the same default ``set_region`` gives every other caller with no
    # existing end.
    assert (one.loop_start, one.loop_end) == (3.0, 10.0)
    assert any(key.startswith(muse_io.CACHE_PREFIX) for key in ctx.submitted), (
        "settling a marker at the keyboard must precompute, exactly as a grip's "
        "release does"
    )


def test_right_bracket_sets_the_region_end_at_the_playhead_and_precomputes(ctx, device):
    import pygame

    from .test_muse_mode import _key

    one = _loaded(ctx, seconds=10.0)
    one.play_offset = 7.0
    assert muse_mode.handle_key(ctx, _key(pygame.K_RIGHTBRACKET)) is True
    assert (one.loop_start, one.loop_end) == (0.0, 7.0)
    assert any(key.startswith(muse_io.CACHE_PREFIX) for key in ctx.submitted)


def test_l_runs_the_finder(ctx, device):
    import pygame

    from .test_muse_mode import _key

    one = _loaded(ctx, seconds=3.0)
    assert muse_mode.handle_key(ctx, _key(pygame.K_l)) is True
    assert ctx.submitted[-1] == f"{muse_io.FIND_PREFIX}a"
    assert one.finding is True


def test_the_players_six_keys_are_a_no_op_with_no_player(ctx):
    """``handle_key``'s own floor: about a decoded take, and before the first
    audition there is none."""
    import pygame

    from .test_muse_mode import _key

    for key in (
        pygame.K_LEFT,
        pygame.K_RIGHT,
        pygame.K_HOME,
        pygame.K_LEFTBRACKET,
        pygame.K_RIGHTBRACKET,
        pygame.K_l,
    ):
        assert muse_mode.handle_key(ctx, _key(key)) is False


# --- the file round trip (the 2026-09-11 audit, finding muse-04) -------------
#
# ``tests/modes/muse/test_muse_io.py`` does not exist; this file already imports
# ``muse_io`` directly at module scope and is the one other test file that
# does, so the round-trip test lives here rather than starting a new module.


def test_read_track_round_trips_a_full_scale_int16_sample_exactly(tmp_path):
    """``read_track`` decodes with ``soundfile``, which normalises 16-bit PCM
    by dividing by 32768 -- the *negative* peak, the "conventional reading"
    ``sirens/wavout.py``'s own ``read_wav`` was hand-written to avoid, by its
    own comment there, for exactly this reason. ``read_track`` then
    re-quantised that float by multiplying by 32767 to match ``to_int16``'s
    encode constant -- reproducing, in this soundfile-based reader, precisely
    the mismatch ``wavout.py`` already worked around. A full-scale sample
    written by ``muse_io._wav`` came back one LSB quiet on the next
    ``read_track`` load, contradicting ``_wav``'s own docstring: "the pair is
    exact, so a take exported unchanged comes back sample for sample."

    ``-32768`` is left out of "full-scale" here on purpose: it is
    ``to_int16``'s own documented exception (the one value it cannot
    represent, by its own docstring), not this reader's mismatch, so testing
    it would conflate two separate things.

    Reproduced against the unfixed code: 32767 comes back as 32766.
    """
    pcm = np.array([[32767, -32767], [-1000, 1000], [0, 0]], dtype=np.int16)
    path = tmp_path / "take.wav"
    path.write_bytes(muse_io._wav(pcm, RATE))

    loaded = muse_io.read_track(path)

    assert loaded["rate"] == RATE
    np.testing.assert_array_equal(loaded["pcm"], pcm)
