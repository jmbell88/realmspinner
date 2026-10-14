"""The 2026-10-14 audit, Muse's controller and bar: findings muse-mode-04, -06,
-07, -09, -10, -11, -12, -13, -15, -18 and muse-engine-01.

No device and no imgui frame: ``_Device`` is the one-channel recorder the player
tests use, and the bar's key reads are driven through a stub ``imgui`` the way
``test_audit_2026_10_03_lows_muse1`` drives muse-23.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.studio.modes.muse import mode as muse_mode
from realmspinner.studio.modes.muse.ui import brief as muse_brief

from .test_muse_mode import FakeCtx
from .test_muse_player import _Device, _loaded, _settle

SRC = Path(__file__).resolve().parents[3] / "src" / "realmspinner" / "studio" / "modes"


# --- muse-mode-04: the region changes under a sounding loop -------------------


def _looping_take(tmp_path, monkeypatch):
    """A 10 s take looping 2-8 s, started from 5 s. -> (ctx, one, device)."""
    ctx = FakeCtx(tmp_path)
    ctx.cache = SimpleNamespace(jobs=[{"id": "a"}])
    device = _Device()
    monkeypatch.setattr(muse_mode, "sirens_audio", device)
    one = _loaded(ctx, seconds=10.0)
    one.xfade_ms = 0.0
    muse_mode.set_region(ctx, 2.0, 8.0)
    _settle(ctx, one)
    muse_mode.seek(ctx, 5.0)
    device.busy, device.tag_value = True, "a"
    muse_mode._play_from(ctx, one, 5.0)
    assert device.calls[-1]["loops"] == -1
    # Ten seconds of looping: 2 + (3 + 10) % 6 = 3.0 in the take, and the mixer's
    # own clock reads 10 % 6 inside the rotated buffer.
    device.pos = 10.0 % 6.0
    return ctx, one, device


def test_clearing_the_region_during_a_loop_keeps_the_playhead_where_it_was(
    tmp_path, monkeypatch
):
    ctx, one, device = _looping_take(tmp_path, monkeypatch)
    assert muse_mode.position(ctx) == pytest.approx(3.0, abs=1e-3)

    # The Clear button: the rotated buffer on the channel is untouched, so
    # what is audible is still the 2-8 s loop and the playhead must say so.
    muse_mode.set_region(ctx, None, None)
    assert muse_mode.position(ctx) == pytest.approx(3.0, abs=1e-3)

    muse_mode.stop(ctx)
    assert one.play_offset == pytest.approx(3.0, abs=1e-3)


def test_dragging_a_marker_during_a_loop_does_not_move_the_playhead(tmp_path, monkeypatch):
    ctx, _one, _device = _looping_take(tmp_path, monkeypatch)
    muse_mode.set_region(ctx, 1.0, 9.0)
    assert muse_mode.position(ctx) == pytest.approx(3.0, abs=1e-3)


# --- muse-mode-11: Generate snapshots the brief at the press ----------------------


def test_generate_submits_the_brief_as_it_stood_at_the_press(tmp_path, monkeypatch):
    from realmspinner.service import jobs as svc_jobs

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(
        svc_jobs, "create_music_job", lambda svc, **kw: seen.append(kw) or {"ids": ["x"]}
    )

    class _Deferred(FakeCtx):
        pending: Any = None

        def submit(self, key, run, *args, tag=None):
            self.pending = lambda: run(*args)
            return True

    ctx = _Deferred(tmp_path)
    ctx.model_rows = []
    form = muse_mode.ensure(ctx).form
    form["prompt"] = "lo-fi piano"
    assert muse_mode.generate(ctx) is True
    form["prompt"] = "death metal"  # typed between the press and the task starting
    ctx.pending()
    assert seen[0]["prompt"] == "lo-fi piano"


# --- muse-mode-12: Open in Sirens when the import is refused -----------------------


def test_open_in_sirens_says_so_when_the_import_is_refused(tmp_path, monkeypatch):
    from realmspinner.studio.modes.sirens import mode as sirens

    ctx = FakeCtx(tmp_path, accept=False)  # a decode is already in flight
    take = tmp_path / "abc123"
    take.mkdir()
    (take / "track.wav").write_bytes(b"x")
    monkeypatch.setattr(muse_mode, "track_path", lambda c, job: take / "track.wav")
    monkeypatch.setattr(sirens, "active", lambda c: SimpleNamespace(uid="u", busy=False))

    assert muse_mode.open_in_sirens(ctx, "abc123") is False
    assert ctx.toasts, "a refused import must not be a dead click"


def test_a_song_made_only_for_a_refused_import_is_closed_again(tmp_path, monkeypatch):
    from realmspinner.studio.modes.sirens import fileio as sirens_io
    from realmspinner.studio.modes.sirens import mode as sirens

    ctx = FakeCtx(tmp_path)
    take = tmp_path / "abc123"
    take.mkdir()
    (take / "track.wav").write_bytes(b"x")
    monkeypatch.setattr(muse_mode, "track_path", lambda c, job: take / "track.wav")
    made, closed = [], []
    monkeypatch.setattr(sirens, "active", lambda c: None)
    monkeypatch.setattr(
        sirens, "new_document", lambda c: made.append(1) or SimpleNamespace(uid="fresh")
    )
    monkeypatch.setattr(sirens, "close_tab", lambda c, uid: closed.append(uid))
    monkeypatch.setattr(sirens_io, "import_sample", lambda *a, **k: False)
    monkeypatch.setattr(muse_mode.sirens_io, "import_sample", sirens_io.import_sample)

    assert muse_mode.open_in_sirens(ctx, "abc123") is False
    assert made == [1] and closed == ["fresh"]


# --- muse-mode-13: a queued split is acknowledged ---------------------------------


def test_a_queued_split_is_acknowledged_and_refreshes_the_job_cache(tmp_path):
    ctx = FakeCtx(tmp_path)
    invalidated: list[int] = []
    ctx.cache = SimpleNamespace(invalidate=lambda: invalidated.append(1), jobs=[])
    done = SimpleNamespace(key="muse-separate:abc123", result={"id": "stems-job"})

    muse_mode.on_task_done(ctx, done)

    assert invalidated == [1]
    assert ctx.toasts and "queued" in ctx.toasts[-1][0].lower()


# --- muse-mode-18: keyboard selection scrolls the card into view -------------------


def test_keyboard_selection_scrolls_the_card_into_view(tmp_path):
    ctx = FakeCtx(tmp_path)
    jobs = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    state = muse_mode.ensure(ctx)
    assert state.scroll_to_selected is False

    muse_mode.select(ctx, jobs, 1)

    assert state.selected_job == "a"
    assert state.scroll_to_selected is True


# --- muse-mode-06/07/09/10: the bar's keys ------------------------------------------


class _Key:
    left_arrow = "left"
    right_arrow = "right"
    enter = "enter"
    keypad_enter = "kp_enter"


def _stub_imgui(monkeypatch, *, pressed: set[str], typing: bool) -> None:
    class _Io:
        want_text_input = typing

    class _Fake:
        Key = _Key

        def __getattr__(self, _name):
            return lambda *a, **k: None

        def is_key_pressed(self, key):
            return key in pressed

        def get_io(self):
            return _Io()

    @contextmanager
    def focused_item(*_a, **_k):
        yield True

    monkeypatch.setattr(muse_brief, "imgui", _Fake())
    monkeypatch.setattr(muse_brief.focus, "item", focused_item)
    monkeypatch.setattr(muse_brief.widgets, "field_label", lambda *a, **k: None)
    monkeypatch.setattr(muse_brief, "sp", lambda value: value)
    monkeypatch.setattr(
        muse_brief.controls, "segmented_choice", lambda _id, _opts, current, **k: (False, current)
    )
    monkeypatch.setattr(muse_brief, "_ring", lambda *a, **k: None)


def test_arrow_keys_in_a_text_field_do_not_change_the_takes_count(tmp_path, monkeypatch):
    _stub_imgui(monkeypatch, pressed={"right"}, typing=True)  # the caret is in Lyrics
    ctx = FakeCtx(tmp_path)
    ctx.state.clear_field_error = lambda *_a: None
    form = {"count": 1}

    muse_brief._count(ctx, form)

    assert form["count"] == 1


def test_enter_in_a_text_field_does_not_press_generate_when_the_ring_is_on_generate(
    tmp_path, monkeypatch
):
    from realmspinner.studio.panes import model_gate

    pressed: list[int] = []
    monkeypatch.setattr(muse_brief.muse_mode, "generate", lambda ctx: pressed.append(1))
    monkeypatch.setattr(model_gate, "missing", lambda *a, **k: [])
    monkeypatch.setattr(muse_brief.widgets, "primary_button", lambda *a, **k: False)
    monkeypatch.setattr(muse_brief.anchors, "mark", lambda *a, **k: None)
    ctx = FakeCtx(tmp_path)

    _stub_imgui(monkeypatch, pressed={"enter"}, typing=True)
    muse_brief._generate(ctx, enabled=True)
    assert pressed == [], "an Enter typed into Lyrics is a newline, not a press"

    # And the keyboard press the ring exists for still works with no caret.
    _stub_imgui(monkeypatch, pressed={"enter"}, typing=False)
    muse_brief._generate(ctx, enabled=True)
    assert pressed == [1]


def test_left_right_on_the_length_ring_steps_the_pills_in_a_reserved_nav_mode(tmp_path):
    import pygame

    ctx = FakeCtx(tmp_path)
    ctx.state.focus_key = {muse_brief.FOCUS_PANE: "duration"}
    ctx.state.focus_order = {muse_brief.FOCUS_PANE: ["tags", "duration", "count", "generate"]}
    form = muse_mode.ensure(ctx).form
    keys = [key for key, _ in muse_brief._DURATION_OPTIONS]
    start = muse_brief._current_duration_key(muse_mode.ensure(ctx), form)

    event = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0)
    assert muse_mode.handle_key(ctx, event) is True

    now = muse_brief._current_duration_key(muse_mode.ensure(ctx), form)
    assert now == keys[(keys.index(start) + 1) % len(keys)]


def test_left_right_on_the_takes_ring_steps_the_count_and_elsewhere_still_nudges(tmp_path):
    import pygame

    ctx = FakeCtx(tmp_path)
    ctx.state.focus_key = {muse_brief.FOCUS_PANE: "count"}
    ctx.state.focus_order = {muse_brief.FOCUS_PANE: ["tags", "duration", "count", "generate"]}
    form = muse_mode.ensure(ctx).form
    form["count"] = 1
    event = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0)
    assert muse_mode.handle_key(ctx, event) is True
    assert form["count"] == 2

    # Ring elsewhere: the arrows are the playhead's, as they always were.
    ctx.state.focus_key = {muse_brief.FOCUS_PANE: "generate"}
    one = _loaded(ctx, seconds=10.0)
    assert muse_mode.handle_key(ctx, event) is True
    assert one.play_offset == pytest.approx(1.0)
    assert form["count"] == 2


def test_generate_reason_names_an_in_flight_submit():
    assert muse_brief._generate_reason(False, True)
    assert "submitting" in muse_brief._generate_reason(False, True).lower()
    assert muse_brief._generate_reason(False, False) == ""
    # The missing model is the more lasting reason and keeps the sentence.
    assert "music model" in muse_brief._generate_reason(True, True)


# --- muse-engine-01 / muse-mode-15: docstrings that name removed things -----------

_OWNED = (
    "muse/state.py",
    "muse/fileio.py",
    "muse/engine/__init__.py",
    "muse/mode.py",
)
_REMOVED = (
    "Poser",
    "poser_mode",
    "Troupe",
    "muse_io",
    "muse_state",
    "sirens_state",
    "studio/muse/",
    "studio/muse_io",
    "studio/sirens/",
    "studio/inker/",
    "studio/clay/",
)


def test_muse_docstrings_name_no_removed_mode():
    offences: list[str] = []
    for rel in _OWNED:
        text = (SRC / rel).read_text(encoding="utf-8")
        for word in _REMOVED:
            # ``muse_io``, ``muse_state`` and ``sirens_state`` are mode.py's own
            # import aliases; only the other three owned files name them as
            # modules that no longer exist.
            if rel == "muse/mode.py" and word in ("muse_io", "muse_state", "sirens_state"):
                continue
            if word in text:
                offences.append(f"{rel}: {word}")
    test_text = (Path(__file__).parent / "test_muse_imports.py").read_text(encoding="utf-8")
    offences += [f"test_muse_imports.py: {w}" for w in _REMOVED if w in test_text]
    assert offences == []
    # ``muse_mode`` and ``muse_io`` are not modules any longer either.
    for rel in ("muse/state.py", "muse/fileio.py", "muse/engine/__init__.py"):
        assert "muse_mode" not in (SRC / rel).read_text(encoding="utf-8"), rel
