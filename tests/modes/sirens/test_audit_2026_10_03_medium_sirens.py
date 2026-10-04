"""The 2026-10-03 audit's Medium findings on Sirens playback and files (sirens-14, -15, -16)."""

from __future__ import annotations

from pathlib import Path

from realmspinner.studio.modes.sirens import fileio as sirens_io
from realmspinner.studio.modes.sirens import mode as sirens_mode

from .test_sirens_export import _song
from .test_sirens_mode import FakeCtx, _sounding, _tab, _two_pattern_song, _wav


def test_following_the_playhead_never_puts_the_caret_on_a_deleted_pattern(monkeypatch):
    ctx = FakeCtx()
    tab, first, second = _two_pattern_song(ctx)
    state = sirens_mode.ensure(ctx)
    state.follow = True
    state.pattern = first
    state.row = 0
    at = next(mark for mark in tab.marks if mark[2] == second and mark[3] == 2)
    _sounding(monkeypatch, tab, (at[0] + 1) / 44100.0)
    # The render the mixer is playing outlives the edit: delete the pattern it
    # is sounding while the song plays.
    assert tab.doc.remove_pattern(second)

    sirens_mode.follow_playhead(ctx)

    assert tab.doc.pattern(state.pattern) is not None
    assert state.pattern == first


def test_two_channels_whose_names_differ_only_by_case_are_two_files(tmp_path):
    ctx = FakeCtx()
    tab = _song(ctx)
    doc = tab.doc
    doc.update_channel(doc.channels[0].uid, name="Lead")
    doc.update_channel(doc.channels[1].uid, name="lead")

    stems = sirens_io.channel_stems(doc)
    assert len({stem.casefold() for stem in stems}) == len(stems)

    # ``Path`` equality is case-insensitive on Windows, which is where the map
    # collapsed: the planned count is the written count.
    plan = sirens_io.export_plan(doc, Path(tmp_path))
    assert len(plan) == 1 + len(doc.channels) + len(doc.oneshots)


def test_a_sample_dropped_on_a_saving_tab_is_refused_out_loud(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True

    started = sirens_io.import_sample(ctx, tab, _wav(tmp_path / "shk.wav"), switch=True)

    assert started is False
    assert not ctx.submitted
    assert ctx.toasts == [(sirens_io.SAMPLE_WHILE_SAVING, "error")]


def test_open_in_sirens_reports_false_when_the_tab_is_saving(tmp_path, monkeypatch):
    from realmspinner.studio.modes.muse import mode as muse_mode

    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    take = _wav(tmp_path / "take.wav")
    monkeypatch.setattr(muse_mode, "track_path", lambda _ctx, _job: take)

    assert muse_mode.open_in_sirens(ctx, "job1") is False
    assert not ctx.submitted
