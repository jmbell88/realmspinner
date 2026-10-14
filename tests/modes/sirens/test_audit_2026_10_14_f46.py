"""The 2026-10-14 audit, Sirens' keyboard, playback, envelope, order and grid panes (fixer F46).

One file for the slice, the way ``test_audit_2026_10_14_f45`` does the engine's.
Nothing here opens a sound device or an imgui frame: every claim is about a pure
function or a call through ``FakeCtx``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from realmspinner.studio.modes.sirens import mode as sirens_mode
from realmspinner.studio.modes.sirens.engine import document as D
from realmspinner.studio.modes.sirens.engine import envelope
from realmspinner.studio.modes.sirens.engine import instruments as inst

from .test_sirens_mode import FakeCtx, _Event, _tab

REPO = Path(__file__).resolve().parents[3]


def _press(ctx: FakeCtx, key_name: str, mod: int = 0) -> bool:
    import pygame

    return sirens_mode.handle_key(ctx, _Event(getattr(pygame, key_name), mod))


# --- sirens-26: the numeric keypad ---------------------------------------------


def test_keypad_digits_type_hex_into_the_numeric_columns():
    """``pygame.key.name(K_KP1)`` is ``"[1]"``, so the keypad's digits matched
    nothing: the instrument, volume and parameter columns and the octave keys
    were unreachable from it."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    cells = tab.doc.pattern(sirens_mode.ensure(ctx).pattern).cells

    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.INSTRUMENT)
    assert _press(ctx, "K_KP0")
    assert _press(ctx, "K_KP5")
    assert int(cells[0, 0, D.INSTRUMENT]) == 0x05

    sirens_mode.set_caret(ctx, row=1, channel=0, column=D.VOLUME)
    assert _press(ctx, "K_KP7")
    assert int(cells[1, 0, D.VOLUME]) == 0x07

    sirens_mode.set_caret(ctx, row=2, channel=0, column=D.PARAM)
    assert _press(ctx, "K_KP1")
    assert _press(ctx, "K_KP2")
    assert int(cells[2, 0, D.PARAM]) == 0x12


def test_keypad_plus_and_minus_move_the_octave():
    ctx = FakeCtx()
    _tab(ctx)
    state = sirens_mode.ensure(ctx)
    start = state.octave
    assert _press(ctx, "K_KP_PLUS")
    assert state.octave == start + 1
    assert _press(ctx, "K_KP_MINUS")
    assert _press(ctx, "K_KP_MINUS")
    assert state.octave == start - 1


# --- sirens-02: the caret may not follow the playhead off the end --------------


def _follow_setup(monkeypatch: Any) -> tuple[FakeCtx, Any, Any]:
    from realmspinner.studio.modes.sirens import audio as sirens_audio
    from realmspinner.studio.modes.sirens.state import Sounding

    ctx = FakeCtx()
    tab = _tab(ctx)
    uid = tab.doc.patterns[0].uid
    sirens_mode.set_caret(ctx, pattern=uid, row=0, channel=0, column=D.INSTRUMENT)
    state = sirens_mode.ensure(ctx)
    # The render the mixer is playing was made when the pattern had 64 rows; the
    # row map still names row 40 after the Rows slider has cut it to 16.
    tab.doc.resize_pattern(uid, 16)
    tab.sounding = Sounding(marks=((0, 0, uid, 40),), anchor=0)
    monkeypatch.setattr(sirens_audio, "tag", lambda: tab.uid)
    monkeypatch.setattr(sirens_audio, "position", lambda: 0.0)
    return ctx, tab, state


def test_follow_playhead_never_lands_the_caret_past_the_end_of_a_shortened_pattern(
    monkeypatch,
):
    ctx, tab, state = _follow_setup(monkeypatch)
    assert state.follow

    assert sirens_mode.follow_playhead(ctx) is False
    assert state.row < tab.doc.patterns[0].rows
    # The keystroke that used to raise IndexError.
    assert _press(ctx, "K_5") in (True, False)


def test_write_hex_refuses_a_caret_row_past_the_end_instead_of_raising():
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.INSTRUMENT)
    state.row = tab.doc.patterns[0].rows + 24
    assert sirens_mode.write_hex(ctx, 5) is False


# --- sirens-07: a typed note does not serialise the song -----------------------


def test_a_note_preview_does_not_serialise_the_samples_on_the_frame_thread(monkeypatch):
    """``preview_note`` took ``rsng.rsng_bytes(tab.doc)`` per typed note: a
    DEFLATE of every pattern and an encode of every sample, on the frame
    thread, to hand one instrument table and a tempo to ``render_note``."""
    from realmspinner.studio.modes.sirens import audio as sirens_audio
    from realmspinner.studio.modes.sirens.engine import rsng

    monkeypatch.setattr(sirens_audio, "available", lambda: True)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)

    def _boom(_doc: Any) -> bytes:
        raise AssertionError("a note preview serialised the whole document")

    monkeypatch.setattr(rsng, "rsng_bytes", _boom)

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.preview = True
    state.instrument = tab.doc.add_instrument().uid

    assert sirens_mode.preview_note(ctx, 60)
    assert ctx.result["pcm"].size > 0


def test_the_journal_encode_defers_the_expensive_half_to_the_task(monkeypatch):
    """Journal autosave did the same whole-document encode on the frame thread;
    ``Provider.encode`` may return a callable that the write task runs."""
    from realmspinner.studio.modes.sirens.engine import rsng

    ctx = FakeCtx()
    tab = _tab(ctx)
    real = rsng.rsng_bytes
    calls: list[int] = []

    def _spy(doc: Any) -> bytes:
        calls.append(1)
        return real(doc)

    monkeypatch.setattr(rsng, "rsng_bytes", _spy)
    deferred = sirens_mode.JOURNAL.encode(tab)
    assert callable(deferred)
    assert not calls
    # An edit landing after the snapshot is not in the bytes the task writes.
    tab.doc.set_cell(tab.doc.patterns[0].uid, 0, 0, D.NOTE, 48)
    data = deferred()
    assert calls
    assert int(rsng.read_rsng(data).patterns[0].cells[0, 0, D.NOTE]) != 48


# --- sirens-16: the envelope marker buttons say why they are dead --------------


def test_envelope_marker_buttons_name_the_busy_song_as_their_reason():
    from realmspinner.studio.modes.sirens.ui.panes import envelopes

    assert envelopes.marker_reason(editable=False, has_values=True) == envelopes._BUSY_WHY
    assert envelopes.marker_reason(editable=False, has_values=False) == envelopes._BUSY_WHY
    assert envelopes.marker_reason(editable=True, has_values=False)
    assert envelopes.marker_reason(editable=True, has_values=True) == ""


# --- sirens-19: painting past the end of a long sequence -----------------------


def test_the_envelope_graph_offers_a_column_past_the_end_of_any_sequence_it_says_painting_can_lengthen():  # noqa: E501
    for length in (0, 5, 15, 16, 20, 100, inst.MAX_SEQUENCE_LEN - 1):
        sequence = inst.Sequence(values=tuple([3] * length))
        count = envelope.paint_columns(sequence)
        assert count > length, length
        # The last column the pointer can reach is one the sequence does not
        # hold yet, so dragging onto it lengthens the sequence.
        last = envelope.step_at(9999.0, 1.0, count)
        grown = envelope.painted(sequence, last, 7)
        assert len(grown.values) > length or length == 0 and len(grown.values) >= 1


def test_a_sequence_at_the_ceiling_has_no_column_past_it():
    sequence = inst.Sequence(values=tuple([1] * inst.MAX_SEQUENCE_LEN))
    assert envelope.paint_columns(sequence) == inst.MAX_SEQUENCE_LEN


# --- sirens-06: the Rows slider does not eat the notes it passes over ----------


def test_dragging_the_rows_slider_down_and_back_up_keeps_the_notes_below_the_lowest_value():
    from realmspinner.studio.modes.sirens.ui.panes import orders

    doc = D.new_song()
    uid = doc.patterns[0].uid
    doc.resize_pattern(uid, 64)
    doc.set_cell(uid, 40, 0, D.NOTE, 48)

    held = None
    for rows in (60, 30, 8, 30, 64):
        changed, held = orders.resized_in_drag(doc, uid, rows, held)
        assert changed
    assert doc.pattern(uid).rows == 64
    assert int(doc.pattern(uid).cells[40, 0, D.NOTE]) == 48

    # A second gesture starts from what the first left, not from its snapshot:
    # the notes the first one cut are gone, and stay gone.
    changed, held = orders.resized_in_drag(doc, uid, 8, None)
    assert changed
    changed, held = orders.resized_in_drag(doc, uid, 64, None)
    assert int(doc.pattern(uid).cells[40, 0, D.NOTE]) == int(D.empty_cells(1, 1)[0, 0, D.NOTE])


# --- sirens-17: default pattern names ------------------------------------------


def test_unnamed_patterns_are_numbered_without_counting_sound_effect_patterns():
    from realmspinner.studio.modes.sirens.ui.panes import orders

    doc = D.new_song()
    first = doc.patterns[0]
    second = doc.add_pattern()
    doc.add_oneshot("coin")
    third = doc.add_pattern()
    assert [orders._name_of(doc, one.uid) for one in (first, second, third)] == [
        "Pattern 1",
        "Pattern 2",
        "Pattern 3",
    ]


# --- sirens-18: moving a looped entry past an identical neighbour --------------


def test_moving_a_looped_entry_past_an_identical_neighbour_moves_the_loop_point():
    doc = D.new_song()
    a = doc.patterns[0].uid
    b = doc.add_pattern().uid
    assert doc.set_order([a, a, b], loop_order=1)
    assert doc.loop_order == 1

    from realmspinner.studio.modes.sirens.ui.panes import orders

    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.doc = doc
    orders._reorder(ctx, tab, 1, 0)
    assert doc.order == [a, a, b]
    assert doc.loop_order == orders.moved_loop(1, 1, 0) == 0


def test_set_order_with_the_same_order_and_the_same_loop_is_still_a_no_op():
    doc = D.new_song()
    a = doc.patterns[0].uid
    assert doc.set_order([a, a], loop_order=1)
    assert doc.set_order([a, a], loop_order=1) is False
    assert doc.set_order([a, a]) is False


# --- sirens-20: the slack right of the last drawn channel ----------------------


def test_a_click_in_the_slack_right_of_the_last_drawn_channel_does_not_move_the_caret():
    from realmspinner.studio.modes.sirens.ui.panes import patterns

    # Eight channels, three drawn from channel 2: 2, 3, 4. Gutter 40, 100 px each.
    drawn = dict(gutter=40.0, chan_w=100.0, left=2, fits=3, channels=8)
    assert patterns.channel_at(45.0, **drawn) == 2
    assert patterns.channel_at(339.0, **drawn) == 4
    # 340 and beyond is the slack; the arithmetic alone names channel 5.
    assert patterns.channel_at(345.0, **drawn) is None
    assert patterns.channel_at(10.0, **drawn) is None
    # Fewer channels than fit: the cut is the pattern's own edge.
    assert patterns.channel_at(250.0, gutter=40.0, chan_w=100.0, left=0, fits=5, channels=2) is None


# --- sirens-22: the manual says what the right-click chooser is ----------------


def test_the_manual_mentions_the_right_click_effect_chooser():
    text = (REPO / "docs" / "manual" / "31-sirens.md").read_text(encoding="utf-8")
    effects = text.split("## Effects", 1)[1].split("| Letter", 1)[0].lower()
    assert "right-click" in effects
    assert "×N" in text
