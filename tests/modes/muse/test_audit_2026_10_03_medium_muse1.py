"""The 2026-10-03 audit, Medium batch muse-1 (muse-04 .. muse-12)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from realmspinner.studio.modes.muse import fileio as muse_io
from realmspinner.studio.modes.muse import mode as muse_mode
from realmspinner.studio.modes.muse.engine import loops

from .test_muse_mode import FakeCtx
from .test_muse_player import _loaded
from .test_muse_player import device as _device

RATE = 44100

# The recorder fixture, re-registered under its own name here.
device = _device


def _int_tone(seconds: float, hz: float = 440.0, amp: float = 12000.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE), dtype=np.float64) / RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16)


@pytest.fixture
def ctx(tmp_path):
    one = FakeCtx(tmp_path)
    one.cache = SimpleNamespace(jobs=[{"id": "a"}, {"id": "b"}], total=2)
    return one


def _space():
    import pygame

    return SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_SPACE, mod=0)


# --- muse-04 ----------------------------------------------------------------


def test_a_stereo_int16_take_is_analysed_at_the_same_scale_as_its_mono_int16_twin():
    """``data.mean(axis=1)`` on int16 returns float64, so the integer scaling
    that followed was skipped for every real (stereo int16) take."""
    rng = np.random.default_rng(4)
    mono = (rng.normal(0.0, 6000.0, RATE * 3)).astype(np.int16)
    stereo = np.stack([mono, mono], axis=1)

    scaled = loops._mono(stereo)
    assert float(np.abs(scaled).max()) <= 1.0
    np.testing.assert_allclose(scaled, loops._mono(mono), atol=1e-6)

    stereo_candidates = loops.find(stereo, RATE)
    mono_candidates = loops.find(mono, RATE)
    assert [(c.start, c.end) for c in stereo_candidates] == [
        (c.start, c.end) for c in mono_candidates
    ]


# --- muse-05 ----------------------------------------------------------------


def _worst_step(body: np.ndarray) -> float:
    repeated = np.concatenate([body, body]).astype(np.float64)
    return float(np.abs(np.diff(repeated, axis=0)).max())


def test_a_fade_clamped_to_a_few_samples_of_room_does_not_create_a_step_bigger_than_the_wrap_it_removed():  # noqa: E501
    """A region starting 3 samples into the take can only fade 3 samples of
    lead-out. That ramp used to be costed by the wrap it lands on alone, so it
    moved the click from the wrap into the body's last samples (7503 against
    the music's own 752). A fade that is applied must leave no step bigger
    than the music's own; declining is the other acceptable answer."""
    tone = _int_tone(1.0)
    start = 3
    # End on a crest, the case the audit measured.
    end = int(np.argmax(tone[20000:20200])) + 20000
    plain = tone[start:end]
    own = float(np.abs(np.diff(plain.astype(np.float64))).max())

    body = loops.crossfade(tone[:end], start, end, 2048)
    if not np.array_equal(body, plain):
        assert _worst_step(body) <= own * 2.0


# --- muse-06 ----------------------------------------------------------------


def _smpl_loop(data: bytes) -> tuple[int, int]:
    body = data.index(b"smpl") + 8
    return (
        int.from_bytes(data[body + 44 : body + 48], "little"),
        int.from_bytes(data[body + 48 : body + 52], "little"),
    )


def test_export_with_points_never_writes_loop_points_past_the_end_of_the_take(
    tmp_path, monkeypatch
):
    from realmspinner.studio.modes.muse import state as muse_state
    from realmspinner.studio.modes.muse.engine import waveform

    pcm = np.zeros((1000, 2), dtype=np.int16)
    one = muse_state.Player(
        job="a", pcm=pcm, rate=RATE, env=waveform.peaks(pcm), duration=1000 / RATE
    )
    one.loop_start, one.loop_end, one.xfade_ms = 0.5, 1.0, 0.0
    out = tmp_path / "track.wav"
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: out)

    muse_io.export_with_points(FakeCtx(tmp_path), one)

    if out.exists():
        start, end = _smpl_loop(out.read_bytes())
        assert end < 1000, f"loop {start}..{end} runs past the 1000 frames of data"


def test_a_remembered_region_is_clamped_to_the_take_it_is_restored_onto(ctx, device):
    state = muse_mode.ensure(ctx)
    state.loop_memory["a"] = (0.5, 99.0, 0.0)
    state.audition_job = "a"
    done = SimpleNamespace(
        key=f"{muse_mode.LOAD_PREFIX}a",
        result={"pcm": np.zeros((441, 2), dtype=np.int16), "rate": RATE, "duration": 0.01},
    )
    muse_mode.on_task_done(ctx, done)
    one = state.player
    # 0.5 s is past the 0.01 s take, so nothing of the region is left.
    assert one.loop_start is None and one.loop_end is None


# --- muse-07 ----------------------------------------------------------------


def test_the_first_down_from_no_selection_lands_on_the_newest_take():
    ctx = FakeCtx(Path("."))
    muse_mode.select(ctx, [{"id": "a"}, {"id": "b"}, {"id": "c"}], 1)
    assert muse_mode.ensure(ctx).selected_job == "a"


def test_the_first_up_from_no_selection_wraps_to_the_oldest_take():
    ctx = FakeCtx(Path("."))
    muse_mode.select(ctx, [{"id": "a"}, {"id": "b"}, {"id": "c"}], -1)
    assert muse_mode.ensure(ctx).selected_job == "c"


def test_clicking_a_take_card_selects_it_and_the_selected_card_is_drawn_marked(
    tmp_path, monkeypatch
):
    from imgui_bundle import imgui

    from realmspinner.studio import theme, widgets
    from realmspinner.studio.modes.muse.ui.panes import results as muse_results
    from realmspinner.studio.modes.sirens import audio as sirens_audio
    from realmspinner.studio.state import AppState

    from .test_muse_panes_smoke import _take

    monkeypatch.setattr(sirens_audio, "available", lambda: False)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)
    monkeypatch.setattr(sirens_audio, "tag", lambda: "")
    rings: list[Any] = []
    monkeypatch.setattr(widgets, "ring", lambda *a, **k: rings.append(a))

    ctx = FakeCtx(tmp_path)
    ctx.state = AppState()
    ctx.cache = SimpleNamespace(jobs=[_take("a")], total=1)

    previous = imgui.get_current_context()
    imgui_ctx = imgui.create_context()
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size = (1600, 950)
        io.delta_time = 1 / 60
        io.fonts.add_font_default()
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
        theme.apply(imgui)
        origin: dict[str, Any] = {}

        def frame() -> None:
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((900, 700))
            imgui.begin("smoke")
            origin["at"] = imgui.get_cursor_screen_pos()
            muse_results.draw(ctx)
            imgui.end()
            imgui.end_frame()
            imgui.render()

        frame()
        frame()
        assert muse_mode.ensure(ctx).selected_job == ""
        assert rings == []

        # The empty top-right corner of the card, clear of every button.
        x = origin["at"].x + muse_results.sp(muse_results.CARD_W) - 12
        y = origin["at"].y + 12
        io.add_mouse_pos_event(x, y)
        frame()
        io.add_mouse_button_event(0, True)
        frame()
        io.add_mouse_button_event(0, False)
        frame()
        frame()

        assert muse_mode.ensure(ctx).selected_job == "a"
        assert rings, "the selected card must be drawn marked"
    finally:
        imgui.destroy_context(imgui_ctx)
        if previous is not None:
            imgui.set_current_context(previous)


# --- muse-08 ----------------------------------------------------------------


def test_space_stops_the_take_a_card_started_even_with_nothing_selected(ctx, device):
    ctx.state.mode = "muse"
    state = muse_mode.ensure(ctx)
    one = _loaded(ctx, job="a")
    state.selected_job = ""  # as if the card's own Play never selected it
    muse_mode._play_from(ctx, one, 0.0)  # what the card's Play ends in
    state.playing_job = "a"
    assert device.playing()

    assert muse_mode.handle_key(ctx, _space()) is True

    assert not device.playing(), "Space must stop the sounding take"


def test_space_stops_the_sounding_take_rather_than_starting_the_selected_one(ctx, device):
    state = muse_mode.ensure(ctx)
    one = _loaded(ctx, job="a")
    muse_mode._play_from(ctx, one, 0.0)
    state.playing_job = "a"
    state.selected_job = "b"
    calls_before = len(device.calls)

    muse_mode.handle_key(ctx, _space())

    assert not device.playing()
    assert len(device.calls) == calls_before, "no second take may start"
    assert not ctx.submitted, "the selected take must not have been loaded"


def test_pressing_play_on_a_take_selects_it(ctx, device):
    state = muse_mode.ensure(ctx)
    muse_mode.play(ctx, "b")
    assert state.selected_job == "b"


# --- muse-09 ----------------------------------------------------------------


def test_switching_takes_mid_listen_carries_the_live_playhead_not_the_slice_base(
    ctx, device
):
    state = muse_mode.ensure(ctx)
    first = _loaded(ctx, seconds=60.0, job="a")
    muse_mode._play_from(ctx, first, 0.0)  # sliced from 0 s
    device.pos = 20.0  # ... and 20 s into the buffer by now
    state.audition_job = "b"
    done = SimpleNamespace(
        key=f"{muse_mode.LOAD_PREFIX}b",
        result={
            "pcm": np.zeros((60 * RATE, 2), dtype=np.int16),
            "rate": RATE,
            "duration": 60.0,
        },
    )

    muse_mode.on_task_done(ctx, done)

    assert state.player.play_offset == pytest.approx(20.0)
    assert device.calls[-1]["frames"] == pytest.approx(40 * RATE, rel=0.001)


# --- muse-10 ----------------------------------------------------------------


def test_a_play_deferred_behind_an_in_flight_stale_precompute_still_sounds_once_the_current_region_is_cached(  # noqa: E501
    ctx, device
):
    ctx.state.mode = "muse"
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 8.0)
    muse_mode.precompute_loop(ctx)
    stale = SimpleNamespace(key=ctx.submitted[-1], result=ctx.result)  # (2, 8), "in flight"

    # The marker moves, and the Play that follows defers on the new region; its
    # own precompute is refused because the old task still holds the key.
    muse_mode.set_region(ctx, 3.0, 5.0)
    ctx.accept = False
    muse_mode.play_region(ctx)
    assert one.pending_play is not None
    ctx.accept = True
    ctx.submitted.clear()

    muse_mode.on_task_done(ctx, stale)  # the old task lands with the old key

    assert ctx.submitted, "the current region must be recomputed, not abandoned"
    assert one.pending_play is not None, "the live Play must keep waiting for it"
    muse_mode.on_task_done(ctx, SimpleNamespace(key=ctx.submitted[-1], result=ctx.result))
    assert device.calls, "the Play must sound once the current region is cached"
    assert device.calls[-1]["frames"] == pytest.approx(2 * RATE, rel=0.01)  # (3, 5)


# --- muse-11 ----------------------------------------------------------------


def test_a_cancelled_re_split_leaves_the_previous_stems_in_place(svc, monkeypatch):
    from realmspinner.service import _jobs_rework as rework
    from realmspinner.service.errors import ServiceError

    monkeypatch.setattr(rework, "check_weights", lambda svc, kind, params: None)
    monkeypatch.setattr(rework, "check_vram", lambda svc, kind, stage, params: None)
    take = svc.store.create("music", "dark ambient", {}, stage="music")
    job_dir = svc.config.job_dir(take)
    (job_dir / "stems").mkdir(parents=True, exist_ok=True)
    (job_dir / "track.wav").write_bytes(b"x")
    (job_dir / "stems" / "stems.json").write_text("{}")
    (job_dir / "stems" / "drums.wav").write_bytes(b"stem")
    svc.store.set_status(take, "done")

    with pytest.raises(ServiceError):
        rework.separate_job(svc, take)

    # Refused at the door, so no row exists whose cancel could unlink the set.
    assert [j for j in svc.store.list() if j["kind"] == "separate"] == []
    assert (job_dir / "stems" / "stems.json").exists()
    assert (job_dir / "stems" / "drums.wav").read_bytes() == b"stem"


# --- muse-12 ----------------------------------------------------------------


def test_sync_keeps_the_player_when_its_take_has_only_scrolled_out_of_the_jobs_window(
    ctx, device
):
    _loaded(ctx, job="target")
    # 500 jobs exist; the window holds the newest two, and the take is older.
    ctx.cache = SimpleNamespace(jobs=[{"id": "x"}, {"id": "y"}], total=500)
    muse_mode.sync(ctx)
    assert muse_mode.player(ctx) is not None


def test_sync_still_drops_the_player_when_the_window_is_the_whole_library(ctx, device):
    _loaded(ctx, job="target")
    ctx.cache = SimpleNamespace(jobs=[{"id": "x"}, {"id": "y"}], total=2)
    muse_mode.sync(ctx)
    assert muse_mode.player(ctx) is None


def test_the_tray_knows_when_it_is_only_looking_at_a_window_of_the_library(tmp_path):
    from realmspinner.studio.modes.muse.ui.panes import results as muse_results

    ctx = FakeCtx(tmp_path)
    ctx.cache = SimpleNamespace(jobs=[{"id": "x", "kind": "image"}], total=500)
    assert muse_results._window_is_partial(ctx) is True
    ctx.cache = SimpleNamespace(jobs=[{"id": "x", "kind": "image"}], total=1)
    assert muse_results._window_is_partial(ctx) is False


# --- the extra: a snapshot reader never moves the shared uid counter ---------


def test_composing_from_sirens_reads_its_snapshot_without_moving_the_uid_counter(
    ctx, monkeypatch
):
    from realmspinner.service import jobs as svc_jobs
    from realmspinner.studio.modes.sirens.engine import document as D
    from realmspinner.studio.modes.sirens.engine import rsng, synth

    doc = D.new_song()
    assert doc.order, "the song needs something in its order list"
    monkeypatch.setattr(svc_jobs, "create_music_job", lambda svc, **kw: {"ids": ["x"]})
    monkeypatch.setattr(
        synth, "render_marked", lambda d: (np.zeros((100, 2), dtype=np.float32), None, [])
    )
    monkeypatch.setattr(muse_mode, "set_mode", lambda state, mode: None)
    muse_mode.ensure(ctx).form["prompt"] = "slow strings"
    data = rsng.rsng_bytes(doc)
    monkeypatch.setattr(rsng, "rsng_bytes", lambda d: data)
    monkeypatch.setattr(D, "_next_uid", 0)

    assert muse_mode.compose_from_sirens(ctx, SimpleNamespace(doc=doc)) is True

    assert D._next_uid == 0, "a throwaway snapshot must not reserve uids"
