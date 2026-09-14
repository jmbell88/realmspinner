"""Section 8 of the 2026-09-02 review, closed 2026-09-04.

The last section of that document, set aside on 2026-09-04 and built the same
day. What it named was one engine defect (row-scoped effects that every tracker
makes persistent), two invisible states an envelope marker could be dragged
into, a manifest that could name one sample twice and quietly keep the second,
and a list of verbs a FamiTracker user reaches for and did not find: note
preview, keyboard instrument selection, Home/End, Insert and shift-rows,
interpolate, and the channels past the right-hand edge of the pane.

**The split is what these assertions are for.** ``sirens_mode.py`` became
``sirens_edit``/``sirens_play``/``sirens_keys`` (T7's mechanism, its ``_MOVED``
table) *after* everything below was green, so the moves are pure code motion
over behaviour these tests pin -- which is the order T7 itself gives and the
reason it was done last. Several tests below deliberately reach through
``sirens_mode.<name>`` rather than through the file a function now lives in:
that address is the compatibility surface, and a test that named the new module
would stop noticing if the door back closed.

The six Sirens panes still cannot be driven headlessly -- this suite has no
imgui harness -- so the pattern is ``test_findings_blind_spots``': a draw
function stays covered by the smoke pass and its **decisions** are covered here,
as pure functions beside it (``sirens_patterns.first_channel``,
``sirens/envelope.marker_bounds``).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_sirens_mode import FakeCtx, _Event, _tab

from warlock.studio import sirens_mode
from warlock.studio.panes import sirens_envelopes, sirens_patterns
from warlock.studio.sirens import document as D
from warlock.studio.sirens import envelope, notes, synth, wsng
from warlock.studio.sirens import instruments as inst

# --- the engine: an effect runs until it is cancelled -------------------------


def _put(doc: Any, row: int, **values: Any) -> None:
    cells = doc.patterns[0].cells
    for key, value in values.items():
        cells[row, 0, getattr(D, key.upper())] = value


def _song(rows: int = 8) -> Any:
    """One note on one instrument that is still sounding several rows later.

    The volume envelope is the point: the default instrument is a six-step
    pluck that finishes inside one row, so a persistence test written against
    it would compare two silences and pass whatever the engine did.
    """
    doc = D.new_song()
    doc.patterns[0].cells = D.empty_cells(rows, doc.patterns[0].channels)
    instrument = doc.add_instrument()
    doc.update_instrument(
        instrument.uid, volume=inst.Sequence(values=(15,), loop=0)
    )
    _put(doc, 0, note=60, instrument=instrument.uid)
    return doc


@pytest.mark.parametrize(
    ("effect", "param", "cancel"),
    [
        (synth.FX_SLIDE_UP, 0x40, 0x00),
        (synth.FX_SLIDE_DOWN, 0x40, 0x00),
        (synth.FX_ARPEGGIO, 0x47, 0x00),
        # ``4x0`` is the cancel, so the *speed* nibble is what stays. 4 rather
        # than 8: a half-cycle per tick is sampled at the zero crossings and
        # sounds like no vibrato at all, which would make this test pass
        # against an engine that had dropped the effect entirely.
        (synth.FX_VIBRATO, 0x4F, 0x40),
        # One step per tick, not fifteen: a fade that reaches silence inside
        # its own row cannot show whether it kept going after it.
        (synth.FX_VOLUME_SLIDE, 0x01, 0x00),
    ],
)
def test_a_voice_effect_keeps_running_after_the_row_that_set_it(effect, param, cancel):
    """The defect, stated once per effect: these were reset at the top of every
    row, so ``103`` was a slide that lasted a sixteenth note and the user who
    typed it had nothing on screen saying why."""
    persisting = _song()
    _put(persisting, 0, effect=effect, param=param)

    stopped = _song()
    _put(stopped, 0, effect=effect, param=param)
    _put(stopped, 1, effect=effect, param=cancel)

    a, _loop = synth.render(persisting)
    b, _loop = synth.render(stopped)
    assert a.shape == b.shape
    assert not np.allclose(a, b), "the effect stopped at the end of its own row"


@pytest.mark.parametrize(
    ("effect", "param"),
    [
        (synth.FX_SLIDE_UP, 0x40),
        (synth.FX_ARPEGGIO, 0x47),
        (synth.FX_VIBRATO, 0x4F),
        (synth.FX_VOLUME_SLIDE, 0x01),
    ],
)
def test_a_zero_parameter_is_what_stops_it(effect, param):
    """The other half: cancelled on row 1, the rest of the song has to be the
    audio of a song that never had the effect at all."""
    cancelled = _song()
    _put(cancelled, 0, effect=effect, param=param)
    _put(cancelled, 1, effect=effect, param=0x00)

    plain = _song()
    _put(plain, 0, effect=effect, param=param)
    _put(plain, 1, effect=effect, param=0x00)
    # Same document; what is asserted is that the *tail* of the two renders
    # agrees with a render that stops, rather than with one that keeps going.
    running = _song()
    _put(running, 0, effect=effect, param=param)

    stopped, _l = synth.render(cancelled)
    kept, _l = synth.render(running)
    assert stopped.shape == kept.shape
    tail = stopped.shape[0] // 2
    assert not np.allclose(stopped[tail:], kept[tail:])
    assert np.allclose(stopped, synth.render(plain)[0])


def test_the_player_effects_are_events_and_do_not_persist():
    """``Bxx``/``Cxx``/``Dxx`` happen on their row and nothing carries over --
    a halt that persisted would stop the song on every subsequent row too,
    which is the same thing, and a *jump* that persisted would never end."""
    doc = _song(4)
    _put(doc, 1, effect=synth.FX_HALT, param=0)
    out, _loop = synth.render(doc)
    assert out.shape[0] > 0


def test_every_effect_says_in_its_own_name_whether_it_persists():
    """The manual's rule and the tooltip's have one source. Each of the six
    voice effects names how it is turned off; a seventh added to the engine
    without that sentence fails here."""
    for effect in (
        synth.FX_ARPEGGIO,
        synth.FX_SLIDE_UP,
        synth.FX_SLIDE_DOWN,
        synth.FX_PORTAMENTO,
        synth.FX_VIBRATO,
        synth.FX_VOLUME_SLIDE,
    ):
        _letter, description = synth.EFFECT_NAMES[effect]
        assert "until" in description.lower(), description


# --- the engine: one note, previewed ------------------------------------------


def test_a_previewed_note_is_the_voice_the_channel_is():
    """The preview goes through the ordinary renderer on a scratch document, so
    a noise channel previews as noise. Two kinds, two different buffers."""
    doc = _song()
    uid = doc.instruments[0].uid
    pulse = synth.render_note(doc, uid, 60, kind="pulse")
    noise = synth.render_note(doc, uid, 60, kind="noise")
    assert pulse.size and noise.size
    assert not np.allclose(pulse[: noise.shape[0]], noise[: pulse.shape[0]])


def test_a_preview_of_a_note_with_no_instrument_is_silence_rather_than_a_crash():
    doc = _song()
    assert synth.render_note(doc, 0x7E, 60).size == 0
    assert synth.render_note(doc, doc.instruments[0].uid, notes.NOTE_OFF).size == 0


# --- the engine: a manifest that names one sample twice ------------------------


def test_two_manifest_entries_for_one_sample_key_are_refused_by_name():
    """Collapsed silently before: the second entry won, and whichever
    instrument named that key played the wrong sound with nothing saying so."""
    import json
    import zipfile

    doc = D.new_song()
    doc.set_sample("hit", np.zeros(64, dtype=np.float32))
    raw = wsng.wsng_bytes(doc)

    out = bytearray()
    import io

    source = zipfile.ZipFile(io.BytesIO(raw))
    manifest = json.loads(source.read(wsng.MANIFEST))
    entry = dict(manifest["samples"][0])
    manifest["samples"] = [entry, dict(entry)]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for member in source.namelist():
            if member == wsng.MANIFEST:
                zf.writestr(member, json.dumps(manifest))
            else:
                zf.writestr(member, source.read(member))
    out = buffer.getvalue()

    with pytest.raises(ValueError, match="twice"):
        wsng.read_wsng(out)


# --- the envelope editor's two invisible states -------------------------------


def test_a_release_marker_cannot_be_dragged_onto_step_zero():
    """``release == 0`` makes the whole sequence tail material and a held note
    silent -- a state the engine tolerates from a file and one no drag should
    be able to produce."""
    sequence = inst.Sequence(values=(15, 12, 8, 4), release=2)
    assert envelope.moved(sequence, "release", 0).release == 1
    assert envelope.moved(sequence, "release", -5).release == 1


def test_a_loop_dragged_past_the_release_stops_at_the_last_step_before_it():
    """It vanished from the graph and stayed in the document: the editor drew
    the loop only inside the held half, and the engine never reaches a loop
    point in the tail."""
    sequence = inst.Sequence(values=(15, 12, 8, 4), loop=0, release=2)
    after = envelope.moved(sequence, "loop", 3)
    assert after.loop == 1
    assert 0 <= after.loop < after.release


def test_a_marker_with_no_room_is_left_where_it_was():
    """A one-step sequence has no held half to split off, so there is nowhere
    legal for a release to land -- and nowhere for a loop to clear it."""
    one = inst.Sequence(values=(15,))
    assert envelope.moved(one, "release", 0) == one
    assert envelope.toggled(one, "release") == one
    assert envelope.moved(inst.Sequence(values=(15, 12), release=1), "loop", 5).loop == 0


def test_shortening_a_sequence_cannot_create_either_invisible_state():
    short = envelope.resized(
        inst.Sequence(values=tuple(range(20)), loop=15, release=18), 1
    )
    assert short.values == (0,)
    assert (short.loop, short.release) == (0, -1)


def test_the_pane_still_answers_at_every_moved_name():
    """The pure half moved under ``studio/sirens/``; the pane is where every
    caller and every existing test names it."""
    for name in (
        "span", "columns", "painted", "moved", "toggled", "grabbed",
        "step_at", "value_at", "marker_bounds", "resized", "MIN_STEPS",
    ):
        assert getattr(sirens_envelopes, name) is getattr(envelope, name)


def test_the_envelope_arithmetic_is_reachable_with_no_imgui_frame():
    """Which is the whole reason it moved. ``tests/sirens/test_sirens_imports``
    is what pins the package's outward set; this asserts the consequence."""
    import sys

    assert "warlock.studio.sirens.envelope" in sys.modules
    assert envelope.marker_bounds(inst.Sequence(values=(1, 2, 3)), "release") == (1, 2)


# --- rows: insert, delete, interpolate ----------------------------------------


def _grid(ctx: FakeCtx, tab: Any) -> Any:
    return tab.doc.pattern(sirens_mode.ensure(ctx).pattern).cells


def test_insert_opens_a_row_and_keeps_the_pattern_the_length_it_was():
    ctx = FakeCtx()
    tab = _tab(ctx)
    rows = tab.doc.patterns[0].rows
    tab.doc.set_cell(tab.doc.patterns[0].uid, 0, 0, D.NOTE, 60)
    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.NOTE)
    assert sirens_mode.shift_rows(ctx, 1)
    cells = _grid(ctx, tab)
    assert cells[0, 0, D.NOTE] == notes.EMPTY
    assert cells[1, 0, D.NOTE] == 60
    assert tab.doc.patterns[0].rows == rows


def test_deleting_a_row_pulls_the_rest_up():
    ctx = FakeCtx()
    tab = _tab(ctx)
    uid = tab.doc.patterns[0].uid
    tab.doc.set_cell(uid, 1, 0, D.NOTE, 60)
    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.NOTE)
    assert sirens_mode.shift_rows(ctx, -1)
    assert _grid(ctx, tab)[0, 0, D.NOTE] == 60


def test_a_row_shift_is_one_undo_step_and_arms_the_renderer():
    ctx = FakeCtx()
    tab = _tab(ctx)
    head = tab.doc.history.head
    tab.render_dirty = False
    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.NOTE)
    tab.doc.set_cell(tab.doc.patterns[0].uid, 0, 0, D.NOTE, 60)
    head = tab.doc.history.head
    sirens_mode.shift_rows(ctx, 1)
    assert tab.doc.history.head == head + 1
    assert tab.render_dirty


def test_interpolate_fills_the_rows_between_a_blocks_ends():
    ctx = FakeCtx()
    tab = _tab(ctx)
    uid = tab.doc.patterns[0].uid
    tab.doc.set_cell(uid, 0, 0, D.VOLUME, 0)
    tab.doc.set_cell(uid, 4, 0, D.VOLUME, 12)
    state = sirens_mode.ensure(ctx)
    state.anchor = (0, 0)
    sirens_mode.set_caret(ctx, row=4, channel=0)
    state.anchor = (0, 0)
    assert sirens_mode.interpolate_selection(ctx)
    column = _grid(ctx, tab)[0:5, 0, D.VOLUME]
    assert list(column) == [0, 3, 6, 9, 12]


def test_interpolate_leaves_a_column_whose_ends_do_not_both_answer():
    """A ramp needs two endpoints. Inventing one from an empty cell is how a
    fade turns into notes nobody typed."""
    doc = D.new_song()
    uid = doc.patterns[0].uid
    doc.set_cell(uid, 0, 0, D.NOTE, 60)
    doc.interpolate(uid, 0, 0, 5, 1)
    assert list(doc.patterns[0].cells[1:5, 0, D.NOTE]) == [notes.EMPTY] * 4


def test_interpolate_never_ramps_the_instrument_column():
    """Ids are a set with no order: a line from 01 to 07 names five slots
    nobody chose."""
    assert D.INSTRUMENT not in D.SongDoc.RAMP_COLUMNS


def test_interpolate_refuses_out_loud_with_nothing_to_ramp_between():
    ctx = FakeCtx()
    _tab(ctx)
    assert not sirens_mode.interpolate_selection(ctx)
    assert ctx.toasts and "three rows" in ctx.toasts[-1][0]


# --- the keyboard -------------------------------------------------------------


def _press(ctx: FakeCtx, name: str, mod: int = 0) -> bool:
    import pygame

    return sirens_mode.handle_key(ctx, _Event(getattr(pygame, name), mod))


def test_home_and_end_reach_the_ends_of_the_pattern():
    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.set_caret(ctx, row=5, channel=0, column=D.NOTE)
    assert _press(ctx, "K_HOME")
    assert sirens_mode.ensure(ctx).row == 0
    assert _press(ctx, "K_END")
    assert sirens_mode.ensure(ctx).row == tab.doc.patterns[0].rows - 1


def test_shift_end_selects_to_the_end_rather_than_dropping_the_block():
    ctx = FakeCtx()
    tab = _tab(ctx)
    import pygame

    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.NOTE)
    assert _press(ctx, "K_END", pygame.KMOD_SHIFT)
    row, chan, rows, chans = sirens_mode.ensure(ctx).selection()
    assert (row, chan, rows, chans) == (0, 0, tab.doc.patterns[0].rows, 1)


def test_insert_and_shift_delete_are_bound():
    ctx = FakeCtx()
    tab = _tab(ctx)
    import pygame

    uid = tab.doc.patterns[0].uid
    tab.doc.set_cell(uid, 0, 0, D.NOTE, 60)
    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.NOTE)
    assert _press(ctx, "K_INSERT")
    assert _grid(ctx, tab)[1, 0, D.NOTE] == 60
    assert _press(ctx, "K_DELETE", pygame.KMOD_SHIFT)
    assert _grid(ctx, tab)[0, 0, D.NOTE] == 60


def test_plain_delete_still_blanks_the_column_it_is_on():
    ctx = FakeCtx()
    tab = _tab(ctx)
    uid = tab.doc.patterns[0].uid
    tab.doc.set_cell(uid, 0, 0, D.NOTE, 60)
    tab.doc.set_cell(uid, 0, 0, D.INSTRUMENT, 3)
    sirens_mode.ensure(ctx).step = 0
    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.INSTRUMENT)
    assert _press(ctx, "K_DELETE")
    cells = _grid(ctx, tab)
    assert cells[0, 0, D.INSTRUMENT] == notes.EMPTY
    assert cells[0, 0, D.NOTE] == 60


def test_ctrl_up_and_down_step_the_stamped_instrument():
    ctx = FakeCtx()
    tab = _tab(ctx)
    import pygame

    uids = [one.uid for one in tab.doc.instruments]
    assert len(uids) > 1, "a new song has an instrument list to step through"
    state = sirens_mode.ensure(ctx)
    state.instrument = uids[0]
    assert _press(ctx, "K_DOWN", pygame.KMOD_CTRL)
    assert state.instrument == uids[1]
    assert _press(ctx, "K_UP", pygame.KMOD_CTRL)
    assert state.instrument == uids[0]
    # Clamped, not wrapped: a step past the end landing on the other end is a
    # stamp nobody meant.
    _press(ctx, "K_UP", pygame.KMOD_CTRL)
    assert state.instrument == uids[0]
    state.instrument = uids[-1]
    _press(ctx, "K_DOWN", pygame.KMOD_CTRL)
    assert state.instrument == uids[-1]


def test_ctrl_g_interpolates():
    ctx = FakeCtx()
    tab = _tab(ctx)
    import pygame

    uid = tab.doc.patterns[0].uid
    tab.doc.set_cell(uid, 0, 0, D.VOLUME, 0)
    tab.doc.set_cell(uid, 2, 0, D.VOLUME, 8)
    state = sirens_mode.ensure(ctx)
    sirens_mode.set_caret(ctx, row=2, channel=0)
    state.anchor = (0, 0)
    assert _press(ctx, "K_g", pygame.KMOD_CTRL)
    assert _grid(ctx, tab)[1, 0, D.VOLUME] == 4


def test_a_busy_tab_refuses_the_interpolate_chord():
    ctx = FakeCtx()
    tab = _tab(ctx)
    import pygame

    tab.saving = True
    assert "g" in sirens_mode._MUTATING_CTRL
    assert _press(ctx, "K_g", pygame.KMOD_CTRL)
    assert tab.doc.history.head == 0


# --- the note preview ---------------------------------------------------------


def test_typing_a_note_asks_for_a_preview(monkeypatch):
    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "available", lambda: True)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.instrument = tab.doc.add_instrument().uid
    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.NOTE)
    assert _press(ctx, "K_z")
    assert any(key.startswith(sirens_mode.PREVIEW_PREFIX) for key in ctx.submitted)


def test_the_song_wins_over_a_preview(monkeypatch):
    """One reserved mixer channel, so a preview would cut whatever is on it --
    and typing into bar 3 while bar 1 plays is what follow mode is for."""
    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "available", lambda: True)
    monkeypatch.setattr(sirens_audio, "playing", lambda: True)
    monkeypatch.setattr(sirens_audio, "tag", lambda: "song")
    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.ensure(ctx).instrument = tab.doc.add_instrument().uid
    assert not sirens_mode.preview_note(ctx, 60)
    assert not ctx.submitted


def test_a_preview_switched_off_costs_nothing(monkeypatch):
    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "available", lambda: True)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.instrument = tab.doc.add_instrument().uid
    state.preview = False
    assert not sirens_mode.preview_note(ctx, 60)
    assert not ctx.submitted


def test_a_preview_never_lands_on_the_songs_buffer():
    """``AUDITION_PREFIX``'s rule, third instance: its own key and its own arm,
    so a note typed during a re-render neither is refused nor replaces the
    song with a single note until the next edit."""
    assert sirens_mode.PREVIEW_PREFIX not in (
        "sirens-render:",
        sirens_mode.AUDITION_PREFIX,
        sirens_mode.PATTERN_PREFIX,
    )


# --- the channels past the pane's right-hand edge ------------------------------


def test_the_channel_window_follows_the_caret_in_both_directions():
    """They were drawn nowhere and typed into all the same."""
    assert sirens_patterns.first_channel(caret=7, count=8, fits=3, scroll=0) == 5
    assert sirens_patterns.first_channel(caret=0, count=8, fits=3, scroll=5) == 0


def test_the_channel_window_stays_put_while_the_caret_is_inside_it():
    """Recomputing "centre on the caret" every frame would slide the whole grid
    sideways on every Right."""
    assert sirens_patterns.first_channel(caret=3, count=8, fits=3, scroll=2) == 2


def test_the_channel_window_never_runs_past_the_last_channel():
    assert sirens_patterns.first_channel(caret=0, count=3, fits=5, scroll=9) == 0
    assert sirens_patterns.first_channel(caret=7, count=8, fits=8, scroll=4) == 0


# --- the channel properties the model could always do -------------------------


def test_renaming_repanning_and_re_kinding_a_channel_all_reach_the_document():
    ctx = FakeCtx()
    tab = _tab(ctx)
    uid = tab.doc.channels[3].uid
    assert sirens_mode.update_channel(ctx, uid, name="Snare")
    assert sirens_mode.update_channel(ctx, uid, kind="triangle")
    assert sirens_mode.update_channel(ctx, uid, pan=-0.5)
    channel = tab.doc.channel(uid)
    assert (channel.name, channel.kind, channel.pan) == ("Snare", "triangle", -0.5)
    assert tab.render_dirty


def test_a_refused_channel_change_is_a_toast_rather_than_a_traceback():
    ctx = FakeCtx()
    tab = _tab(ctx)
    assert not sirens_mode.update_channel(ctx, tab.doc.channels[0].uid, kind="banjo")
    assert ctx.toasts and ctx.toasts[-1][1] == "error"


def test_the_notes_written_on_a_channel_survive_a_change_of_voice():
    """The voice is how they sound, not what they are."""
    doc = D.new_song()
    uid = doc.patterns[0].uid
    doc.set_cell(uid, 0, 3, D.NOTE, 60)
    doc.update_channel(doc.channels[3].uid, kind="triangle")
    assert doc.patterns[0].cells[0, 3, D.NOTE] == 60


# --- the split ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "clamp_caret", "move_caret", "jump_row", "set_caret", "write_cell",
        "write_note", "write_hex", "write_effect", "clear_cell", "transpose",
        "shift_rows", "interpolate_selection", "copy_selection", "paste",
        "set_sequence", "adopt_sample", "undo", "redo",
        "request_render", "pump", "play", "play_pattern", "play_from_caret",
        "toggle_play", "playhead_row", "follow_playhead", "audition",
        "preview_note", "handle_key", "release_all", "PIANO_KEYS",
        "AUDITION_PREFIX", "ENVELOPE_FIELDS",
    ],
)
def test_every_moved_name_still_answers_at_its_old_address(name):
    """The door back. ``sirens_mode.<name>`` is what the panes, the app's key
    routing and most of this mode's tests say."""
    assert getattr(sirens_mode, name) is not None


def test_the_moved_table_names_each_thing_once_and_no_ghosts():
    """A name in the table that its module does not define is a door onto
    nothing, and one that is *also* still defined here is two places to keep
    in step."""
    from importlib import import_module

    for name, module in sirens_mode._MOVED.items():
        assert name not in vars(sirens_mode), f"{name} is in both places"
        assert hasattr(import_module(f"warlock.studio.{module}"), name)


def test_dir_still_finds_the_moved_names():
    assert "handle_key" in dir(sirens_mode)


# --- the rest of section 8's "Tests" paragraph --------------------------------


def test_the_playhead_reads_a_multi_entry_order_rather_than_one_long_pattern():
    """The estimate this replaced was wrong the moment a song had two patterns.
    Asserted here against the *order*: the second entry's rows have to come
    back as rows of the pattern that entry names, not as rows past the end of
    the first."""
    doc = D.new_song()
    first = doc.patterns[0]
    second = doc.add_pattern(rows=4)
    doc.set_order([first.uid, second.uid])
    _samples, _loop, marks = synth.render_marked(doc)
    entries = {index for _offset, index, _uid, _row in marks}
    assert entries == {0, 1}
    for _offset, index, uid, row in marks:
        pattern = first if index == 0 else second
        assert uid == pattern.uid
        assert 0 <= row < pattern.rows


def test_a_paste_wider_than_the_remaining_channels_is_clipped_not_refused():
    """``set_cells``' rule, reached through the clipboard: what fits goes in."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    pattern = tab.doc.patterns[0]
    tab.doc.set_cell(pattern.uid, 0, 0, D.NOTE, 60)
    tab.doc.set_cell(pattern.uid, 0, 1, D.NOTE, 62)
    sirens_mode.set_caret(ctx, row=0, channel=0)
    state.anchor = (0, 0)
    sirens_mode.set_caret(ctx, row=0, channel=1)
    state.anchor = (0, 0)
    assert sirens_mode.copy_selection(ctx)
    state.anchor = None
    sirens_mode.set_caret(ctx, row=2, channel=pattern.channels - 1)
    assert sirens_mode.paste(ctx)
    cells = tab.doc.pattern(pattern.uid).cells
    assert cells[2, pattern.channels - 1, D.NOTE] == 60


def test_a_click_takes_the_column_it_landed_on_including_the_gap_after_it():
    """"Click on ``Fxx``, type, get a note" was the defect. The gap after a
    column belongs to the value on its left, the way a tracker's does."""
    widths = [30.0, 20.0, 20.0, 10.0, 20.0]
    assert sirens_patterns.column_at(0.0, widths, 6.0) == 0
    assert sirens_patterns.column_at(33.0, widths, 6.0) == 0
    assert sirens_patterns.column_at(95.0, widths, 6.0) == 3
    assert sirens_patterns.column_at(9999.0, widths, 6.0) == 4


def test_caret_span_rings_a_nibble_only_on_two_digit_columns():
    """the 2026-09-07 audit, finding sirens-07: ``_caret_span`` -- which picks
    the characters the caret rings for a two-digit column -- had no direct
    test, unlike its pulled-out neighbours ``first_channel``/``column_at``.
    An evidence gap rather than a reproduced defect, so this is coverage
    only: at the time of writing the function already does the right thing.
    """
    # A single-keystroke column (NOTE) always ranges the whole cell, whatever
    # ``digit`` says -- there is no sub-position to point a nibble caret at.
    assert sirens_patterns._caret_span(D.NOTE, 0, "C-4") == (0, 3)
    assert sirens_patterns._caret_span(D.NOTE, 1, "C-4") == (0, 3)

    # A two-digit column (INSTRUMENT, PARAM) rings one nibble while a digit is
    # mid-entry ...
    assert sirens_patterns._caret_span(D.INSTRUMENT, 0, "3F") == (0, 1)
    assert sirens_patterns._caret_span(D.INSTRUMENT, 1, "3F") == (1, 1)
    assert sirens_patterns._caret_span(D.PARAM, 0, "0A") == (0, 1)
    assert sirens_patterns._caret_span(D.PARAM, 1, "0A") == (1, 1)

    # ... and falls back to the whole cell once ``digit`` is not a position
    # inside it -- no half-typed entry in flight to narrow the caret to.
    assert sirens_patterns._caret_span(D.INSTRUMENT, -1, "3F") == (0, 2)
    assert sirens_patterns._caret_span(D.INSTRUMENT, 2, "3F") == (0, 2)

    # A one-digit column (VOLUME) and a zero-digit one (EFFECT) are both
    # ``COLUMN_DIGITS <= 1``, so neither ever narrows to a nibble.
    assert sirens_patterns._caret_span(D.VOLUME, 0, "F") == (0, 1)
    assert sirens_patterns._caret_span(D.EFFECT, 0, "A") == (0, 1)


def test_the_loop_point_follows_an_entry_that_moves_under_it():
    """``sirens_orders.moved_loop``, the order list's own pure half."""
    from warlock.studio.panes import sirens_orders

    assert sirens_orders.moved_loop(2, 2, 0) == 0
    assert sirens_orders.moved_loop(0, 2, 0) == 1


def test_reused_patterns_are_counted_in_the_order_list():
    """W1.11: an order row whose pattern appears more than once used to draw
    identically to one that appears once -- nothing on screen said the chorus
    at 00 and 03 was the same pattern rather than two coincidentally similar
    ones. ``reuse_counts`` is the row's own arithmetic, pulled out pure the
    way ``pattern_room`` and ``moved_loop`` beside it are."""
    from warlock.studio.panes import sirens_orders

    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    verse = doc.patterns[0].uid
    chorus = doc.add_pattern().uid
    assert doc.set_order([verse, chorus, verse, chorus, chorus])

    counts = sirens_orders.reuse_counts(list(doc.order))
    assert counts[verse] == 2
    assert counts[chorus] == 3

    # A pattern named once is not "reused" and the row draws no marker for it.
    solo = doc.add_pattern().uid
    assert doc.set_order([verse, chorus, verse, chorus, chorus, solo])
    counts = sirens_orders.reuse_counts(list(doc.order))
    assert counts[solo] == 1


def test_an_effect_can_be_picked_by_name():
    """W1.11: the fx-cell right-click popup's own verb. ``choose_effect`` is
    what a click on one of ``synth.EFFECT_NAMES``' rows does -- point the
    caret at the cell's effect column and let ``write_effect`` (the single
    authority over which ids the engine has a handler for) write it -- pulled
    out pure so this does not need a mouse to drive it, the
    ``column_at``/``first_channel`` idiom this file already uses for the grid.
    """
    from warlock.studio.panes import sirens_patterns

    ctx = FakeCtx()
    tab = _tab(ctx)
    assert sirens_patterns.choose_effect(ctx, row=2, channel=0, effect=synth.FX_TEMPO)
    cells = tab.doc.pattern(sirens_mode.ensure(ctx).pattern).cells
    assert cells[2, 0, D.EFFECT] == synth.FX_TEMPO
    # And the caret followed the pick to the cell the popup was opened on --
    # ``write_cell`` then steps the row the same way any other finished entry
    # does, which is why only the channel and column are pinned here.
    state = sirens_mode.ensure(ctx)
    assert (state.channel, state.column) == (0, D.EFFECT)

    # An id the engine has no handler for is refused, the same way an unknown
    # letter typed by hand is.
    unknown = max(synth.EFFECT_NAMES) + 1
    assert not sirens_patterns.choose_effect(ctx, row=2, channel=0, effect=unknown)
    assert cells[2, 0, D.EFFECT] == synth.FX_TEMPO


# --- Order and Instruments: the ceilings, and the reasons a row is dead -------


def test_add_pattern_button_refuses_gracefully_at_max_patterns():
    """the 2026-09-07 audit, finding sirens-03: "Add a pattern" and
    "Duplicate" called the document with no cap check and no try/except, so
    filling a song to its own documented ceiling raised a bare ``ValueError``
    out of ``draw()`` -- and ``guard.py`` replaces the whole pane after three
    presses. Reproduced against the unfixed code: ``sirens_orders`` has no
    ``pattern_room`` at all, so this fails with an ``AttributeError`` rather
    than the greyed button the fix provides."""
    from warlock.studio.panes import sirens_orders

    doc = D.new_song()
    while len(doc.patterns) < D.MAX_PATTERNS:
        doc.add_pattern()
    assert len(doc.patterns) == D.MAX_PATTERNS

    addable, reason = sirens_orders.pattern_room(doc, editable=True)
    assert not addable
    assert str(D.MAX_PATTERNS) in reason

    # Duplicate shares the same ceiling check -- one function, two call sites.
    assert sirens_orders.pattern_room(doc, editable=True) == (False, reason)

    # A song under the ceiling is not refused, and a busy one names the save
    # rather than the ceiling.
    under = D.new_song()
    assert sirens_orders.pattern_room(under, editable=True) == (True, "")
    assert sirens_orders.pattern_room(under, editable=False) == (
        False,
        sirens_orders._BUSY_WHY,
    )


def test_add_instrument_button_refuses_gracefully_at_max_instruments():
    """the 2026-09-07 audit, finding sirens-03: "Add" (instruments) called
    ``add_instrument`` with no cap check and no try/except, the same defect as
    the pattern buttons. Reproduced against the unfixed code: ``instrument_room``
    does not exist there."""
    from warlock.studio.panes import sirens_instruments

    doc = D.new_song()
    while len(doc.instruments) < D.MAX_INSTRUMENTS:
        doc.add_instrument()
    assert len(doc.instruments) == D.MAX_INSTRUMENTS

    addable, reason = sirens_instruments.instrument_room(doc, editable=True)
    assert not addable
    assert str(D.MAX_INSTRUMENTS) in reason

    under = D.new_song()
    assert sirens_instruments.instrument_room(under, editable=True) == (True, "")
    assert sirens_instruments.instrument_room(under, editable=False) == (
        False,
        sirens_instruments._BUSY_WHY,
    )


def test_delete_reasons_name_the_state_that_is_actually_true():
    """the 2026-09-07 audit, finding sirens-04: the disabled-reason ternary in
    ``sirens_effects.py`` and ``sirens_instruments.py`` tested ``editable``
    inverted, so a busy song showed "nothing selected" and an idle one with
    nothing selected showed the busy sentence -- each state naming the
    other's. Reproduced against the unfixed code: the busy case answered "No
    sound effect is selected." instead of the busy sentence."""
    from warlock.studio.panes import sirens_effects, sirens_instruments

    assert sirens_effects.delete_reason(True, False) == "No sound effect is selected."
    assert sirens_effects.delete_reason(False, False) == sirens_effects._BUSY_WHY
    assert sirens_effects.delete_reason(False, True) == sirens_effects._BUSY_WHY
    assert sirens_effects.delete_reason(True, True) == ""

    assert sirens_instruments.delete_reason(True, False) == "No instrument is selected."
    assert sirens_instruments.delete_reason(False, False) == sirens_instruments._BUSY_WHY
    assert sirens_instruments.delete_reason(False, True) == sirens_instruments._BUSY_WHY
    assert sirens_instruments.delete_reason(True, True) == ""


def test_follow_playhead_updates_order_index_so_a_reused_patterns_highlight_survives_the_next_entry(
    monkeypatch,
):
    """the 2026-09-08 audit, finding sirens-02: ``follow_playhead`` moved
    ``state.pattern``/``state.row`` onto the sounding row every frame but
    never touched ``state.order_index``, even though ``playhead_row`` (added
    to fix "a chorus at entries 00 and 03", S3) refuses to answer once
    ``state.order_index`` disagrees with the sounding order entry. A pattern
    reused at two order entries is the worst case: its pattern/row answer is
    identical at both entries, so the caret was already "on" the sounding
    row from an earlier click and the early-return above fired without ever
    correcting the stale order index -- the highlight then stayed dark for
    the rest of the session. Reproduced against the unfixed code: the caret
    starts on the pattern/row the mark also names, with only the order index
    stale (as a click on the order list, then Play, leaves it), and
    ``follow_playhead`` reports no movement and leaves ``playhead_row`` mute.
    """
    from warlock.studio import sirens_audio
    from warlock.studio.sirens_state import Sounding

    ctx = FakeCtx()
    tab = _tab(ctx)
    pattern_uid = tab.doc.patterns[0].uid
    sirens_mode.set_caret(ctx, pattern=pattern_uid, row=0, order_index=0)
    state = sirens_mode.ensure(ctx)
    assert state.follow, "follow mode is on by default -- this is the ordinary case"

    # The song is sounding order entry 1, which happens to reuse the caret's
    # own pattern at row 0 -- so pattern and row already match state, and only
    # the order index (still 0, from the earlier click) is wrong.
    tab.sounding = Sounding(marks=((0, 1, pattern_uid, 0),), anchor=0)
    monkeypatch.setattr(sirens_audio, "tag", lambda: tab.uid)
    monkeypatch.setattr(sirens_audio, "position", lambda: 0.0)

    assert sirens_mode.follow_playhead(ctx)
    assert state.order_index == 1
    assert sirens_mode.playhead_row(ctx) == 0


def test_the_sample_delete_reason_is_never_the_empty_string_while_busy():
    """the 2026-09-07 audit, finding sirens-05: the sample Delete button's
    reason fell through to "" while the song was saving -- the exact failure
    ``disabled_button``'s docstring exists to prevent, at the moment it most
    needs explaining. Reproduced against the unfixed code: the busy case
    answers "" rather than a sentence."""
    from warlock.studio.panes import sirens_instruments

    assert sirens_instruments.sample_delete_reason(False, False) == (
        sirens_instruments._BUSY_WHY
    )
    assert sirens_instruments.sample_delete_reason(False, True) == (
        sirens_instruments._BUSY_WHY
    )
    assert (
        sirens_instruments.sample_delete_reason(True, False)
        == "This instrument has no sample."
    )
    assert sirens_instruments.sample_delete_reason(True, True) == ""


def test_column_chars_length_agrees_with_document_columns(monkeypatch):
    """the 2026-09-08 audit, finding sirens-04: ``COLUMN_CHARS`` was defined
    and documented as "asserted by a test" and as the mechanism that "widens
    the group" when a sixth grid column lands, but nothing in the module or
    the suite ever read it -- so a column added to ``document.COLUMNS``
    without a matching entry here would draw a grid with that column silently
    missing, the exact hazard the comment claimed was already guarded
    against. Reproduced against the unfixed code by mismatching the two and
    reloading the module: with no assertion tying them together, the reload
    used to succeed silently instead of raising.
    """
    import importlib

    from warlock.studio.sirens import document as D

    original = D.COLUMNS
    monkeypatch.setattr(D, "COLUMNS", original + 1)
    try:
        with pytest.raises(AssertionError):
            importlib.reload(sirens_patterns)
    finally:
        monkeypatch.setattr(D, "COLUMNS", original)
        importlib.reload(sirens_patterns)


class _FakeOrderDoc:
    """A stand-in for ``SongDoc`` carrying only what ``add_to_order_reason``
    reads. The function is pure over ``doc.patterns``' truthiness alone, so a
    real document (and the pattern list it insists on keeping non-empty) is
    more setup than the claim needs."""

    def __init__(self, patterns: list[Any]) -> None:
        self.patterns = patterns


def test_add_to_order_reason_names_the_state_that_is_actually_true():
    """the 2026-09-08 audit, finding sirens-05: "Add to the order"'s disabled
    reason was an inline three-way ternary (effect selected, then no
    patterns, then busy), unlike every sibling disabled-reason in this file
    and its neighbours -- each pulled out pure and unit tested after the
    2026-09-07 audit found this same shape of bug in them. Reproduced against
    the unfixed code: ``sirens_orders`` has no ``add_to_order_reason`` at all,
    so this fails with an ``AttributeError``.
    """
    from warlock.studio.panes import sirens_orders

    with_pattern = _FakeOrderDoc([object()])
    without_pattern = _FakeOrderDoc([])

    assert sirens_orders.add_to_order_reason("Coin", with_pattern, True) == (
        "The grid is editing the sound effect Coin, and an effect's "
        "pattern is not part of the song. Pick a song pattern first."
    )
    # The effect reason wins even over "no patterns" and even while editable:
    # picking a song pattern first is the fix either way.
    assert sirens_orders.add_to_order_reason("Coin", without_pattern, True) == (
        "The grid is editing the sound effect Coin, and an effect's "
        "pattern is not part of the song. Pick a song pattern first."
    )
    assert sirens_orders.add_to_order_reason("", without_pattern, True) == (
        "There is no pattern to add yet."
    )
    assert sirens_orders.add_to_order_reason("", with_pattern, False) == (
        sirens_orders._BUSY_WHY
    )
    assert sirens_orders.add_to_order_reason("", with_pattern, True) == ""


def test_bridge_export_and_compose_reasons_are_pulled_out_and_tested():
    """the 2026-09-11 audit, finding sirens-04: ``sirens_bridge.py``'s "Export
    audio..." and "Compose in Muse..." buttons picked their ``reason=`` with
    an inline ternary (``"busy" if ready/doc.order else "nothing to
    export/compose"``) instead of a pulled-out, unit-tested ``*_reason``
    function -- the exact shape that produced sirens-03/04/05 (2026-09-07) and
    again ``sirens_orders.add_to_order_reason`` (2026-09-08), whose own
    docstring names an inline ternary with untested priority among competing
    disabled causes as precisely what produced those. Reproduced against the
    unfixed code: ``sirens_bridge`` has no ``export_reason``/``compose_reason``
    at all, so this fails with an ``AttributeError``.
    """
    from warlock.studio.panes import sirens_bridge

    assert sirens_bridge.export_reason(ready=True, busy=False) == ""
    assert sirens_bridge.export_reason(ready=True, busy=True) == (
        "This song is being written; the button comes back when it lands."
    )
    assert sirens_bridge.export_reason(ready=False, busy=False) == (
        "There is nothing in the order list to export yet."
    )
    # Both are true at once (nothing to export *and* busy): "nothing to
    # export" wins, because it is still true once the busy stretch ends and a
    # transient "busy" reason would mislead about what to do next.
    assert sirens_bridge.export_reason(ready=False, busy=True) == (
        "There is nothing in the order list to export yet."
    )

    assert sirens_bridge.compose_reason(has_order=True, busy=False) == ""
    assert sirens_bridge.compose_reason(has_order=True, busy=True) == (
        "This song is being written; the button comes back when it lands."
    )
    assert sirens_bridge.compose_reason(has_order=False, busy=False) == (
        "There is nothing in the order list to compose from."
    )
    assert sirens_bridge.compose_reason(has_order=False, busy=True) == (
        "There is nothing in the order list to compose from."
    )


# --- the 2026-09-13 audit -------------------------------------------------


def test_preview_note_does_not_reencode_the_song_while_a_preview_is_already_rendering(
    monkeypatch,
):
    """Finding sirens-01. ``audition``, ``preview_note`` and ``play_pattern``
    used to build ``wsng.wsng_bytes(tab.doc)`` -- a whole-document DEFLATE and
    sample encode, on the frame thread -- before asking ``ctx.busy(key)``,
    although ``submit`` refuses a key already in flight regardless and
    ``request_render`` already asks first for exactly this reason. Reproduced
    against the unfixed code by making ``wsng_bytes`` raise: with the busy
    check first, a preview asked for while one is already rendering never
    reaches it."""
    from warlock.studio import sirens_audio
    from warlock.studio.sirens import wsng

    monkeypatch.setattr(sirens_audio, "available", lambda: True)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)

    def _boom(_doc):
        raise AssertionError("re-encoded the song although the key was busy")

    monkeypatch.setattr(wsng, "wsng_bytes", _boom)

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.instrument = tab.doc.add_instrument().uid
    ctx.busy_keys.add(f"{sirens_mode.PREVIEW_PREFIX}{tab.uid}")
    assert not sirens_mode.preview_note(ctx, 60)


def test_audition_does_not_reencode_the_song_while_already_rendering(monkeypatch):
    """The same finding, sirens-01, for ``audition``."""
    from warlock.studio import sirens_audio
    from warlock.studio.sirens import wsng

    monkeypatch.setattr(sirens_audio, "available", lambda: True)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)

    def _boom(_doc):
        raise AssertionError("re-encoded the song although the key was busy")

    monkeypatch.setattr(wsng, "wsng_bytes", _boom)

    ctx = FakeCtx()
    tab = _tab(ctx)
    effect = tab.doc.add_oneshot("coin")
    ctx.busy_keys.add(f"{sirens_mode.AUDITION_PREFIX}{tab.uid}")
    assert not sirens_mode.audition(ctx, tab, effect.uid)


def test_play_pattern_does_not_reencode_the_song_while_already_rendering(monkeypatch):
    """The same finding, sirens-01, for ``play_pattern``."""
    from warlock.studio import sirens_audio
    from warlock.studio.sirens import wsng

    monkeypatch.setattr(sirens_audio, "available", lambda: True)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)

    def _boom(_doc):
        raise AssertionError("re-encoded the song although the key was busy")

    monkeypatch.setattr(wsng, "wsng_bytes", _boom)

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.pattern = tab.doc.patterns[0].uid
    ctx.busy_keys.add(f"{sirens_mode.PATTERN_PREFIX}{tab.uid}")
    assert not sirens_mode.play_pattern(ctx, tab)


@pytest.mark.parametrize("module", ["sirens_play", "sirens_keys", "sirens_edit"])
def test_every_name_defined_in_the_split_modules_is_in_the_moved_table(module):
    """Finding sirens-02. ``_MOVED`` promises every name the split-out modules
    define stays reachable as ``sirens_mode.<name>``, but the ghost test
    (``test_the_moved_table_names_each_thing_once_and_no_ghosts``) only checks
    that direction: a table entry resolves. It never checked the other way, so
    ``sirens_play._caret_offset`` and ``sirens_keys._piano_elsewhere`` could be
    -- and were -- missing from the table with nothing failing. Reproduced
    against the unfixed code: both names are module-level in their files and
    absent from ``sirens_mode._MOVED``.
    """
    import warlock.studio as studio_pkg

    path = Path(studio_pkg.__file__).parent / f"{module}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defined: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
    missing = {name for name in defined if sirens_mode._MOVED.get(name) != module}
    assert not missing, f"{module} defines {missing} but _MOVED does not point there"


def test_audition_reason_names_the_state_that_is_actually_true():
    """Finding sirens-05: the Audition button's disabled reason was an inline
    ternary, the shape behind findings sirens-03/04/05 of the 2026-09-07
    audit. Reproduced against the unfixed code: ``sirens_effects`` has no
    ``audition_reason`` at all, so this fails with an ``AttributeError``.
    """
    from warlock.studio import sirens_audio
    from warlock.studio.panes import sirens_effects

    assert sirens_effects.audition_reason(False) == sirens_effects._BUSY_WHY
    assert sirens_effects.audition_reason(True) == sirens_audio.unavailable_reason()


# --- the 2026-09-14 audit -------------------------------------------------


def test_the_playhead_goes_dark_during_the_release_tail_after_the_last_row():
    """Finding sirens-01. ``Sounding.mark_at``'s own docstring promises
    ``None`` "before the first row and after the last", but the bisect that
    answers it had no upper bound at all -- every offset past the final mark
    still landed on index ``len(marks) - 1``, so the highlight on the last
    row stayed lit through the whole release/decay tail once the song
    stopped advancing rows, exactly the state the docstring says never
    happens. Reproduced against the unfixed code (see the 2026-09-14 audit's
    probe): a query far past the last mark's offset still answered with that
    row rather than ``None``.
    """
    from warlock.studio.sirens_state import Sounding

    marks = ((0, 0, 100, 0), (1000, 0, 100, 1), (2000, 0, 100, 2))
    sounding = Sounding(marks=marks, anchor=0, wrap=None, generation=1)
    rate = 100

    # Inside the last row's own duration (the interval the row before it
    # took): still lit.
    assert sounding.mark_at(20.0, rate) == (0, 100, 2)
    # Deep in the release tail, well past that duration: dark.
    assert sounding.mark_at(100.0, rate) is None

    # A looping (wrapped) buffer has no tail to go dark through -- it rolls
    # straight back into row 0 -- so the new bound must not apply to it.
    looped = Sounding(marks=marks, anchor=0, wrap=3000, generation=1)
    assert looped.mark_at(100.0, rate) is not None


def test_add_to_order_reason_names_busy_even_when_the_caret_is_on_an_effect():
    """Finding sirens-04. ``add_to_order_reason`` checked the effect-column
    case before ``editable``, so a song that was busy saving while the caret
    happened to sit on an effect cell reported "pick a song pattern first"
    -- a fix that does nothing, since the button stays disabled either way
    until the save lands. Reproduced against the unfixed code: this same
    call answered the effect sentence instead of the busy one.
    """
    from warlock.studio.panes import sirens_orders

    with_pattern = _FakeOrderDoc([object()])
    assert sirens_orders.add_to_order_reason("Coin", with_pattern, False) == (
        sirens_orders._BUSY_WHY
    )
    # Idle again: the effect reason returns, as the pre-existing test above
    # already pins.
    assert sirens_orders.add_to_order_reason("Coin", with_pattern, True) != (
        sirens_orders._BUSY_WHY
    )
