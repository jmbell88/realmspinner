"""Regression tests for the 2026-09-26 audit's Sirens findings, wave 3 fixer 1.

Owns ``src/realmspinner/studio/modes/sirens/`` (id prefixes ``sirens-engine-``,
``sirens-panes-`` and ``sirens-playback-``). One file per this pass's
convention (``test_audit_2026_09_23_sirens.py``), one section per finding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from realmspinner.studio.modes.sirens import mode as sirens_mode
from realmspinner.studio.modes.sirens.engine import document as D
from realmspinner.studio.modes.sirens.engine import instruments as inst
from realmspinner.studio.modes.sirens.engine import rsng, synth

from .test_sirens_mode import FakeCtx, _Done, _tab

# --- sirens-panes-01: the channel popup was bound to the solo button -----------


def test_right_clicking_a_channel_name_opens_its_property_popup():
    """``_channel_popup``'s ``begin_popup_context_item`` binds to *the last
    item drawn*, whatever ``tag`` string it is given -- imgui docs this as
    "for id-based interaction (the last item)". The call used to sit after the
    20 px solo button, so right-clicking the name/mute button (all but the
    last 26 px of the column) tested the solo button's hover state, which is
    never true there, and the popup never opened. This does not simulate a
    real click (a real one crashed the headless imgui backend used by this
    suite's ``renderer_has_textures`` declaration two frames in, before the
    fix was even in the loop) -- it instead captures which item's rect was
    "last" at the moment ``begin_popup_context_item`` runs, which is exactly
    the thing imgui's own click detection reads. Failed against the unfixed
    code: the captured rect equalled the solo button's, not the mute button's.
    """
    from imgui_bundle import imgui

    from realmspinner.studio import theme
    from realmspinner.studio.modes.sirens.ui.panes import patterns as sirens_patterns

    previous = imgui.get_current_context()
    gui_ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)

    try:
        ctx = FakeCtx()
        tab = _tab(ctx)
        state = sirens_mode.ensure(ctx)
        pattern = state.active.doc.patterns[0]
        channel_uid = tab.doc.channels[0].uid

        button_rects: dict[str, tuple[Any, Any]] = {}
        popup_rects: dict[str, tuple[Any, Any]] = {}

        import realmspinner.studio.widgets as widgets_mod

        orig_button = widgets_mod.disabled_button

        def spy_button(label: str, *args: Any, **kwargs: Any) -> Any:
            result = orig_button(label, *args, **kwargs)
            button_rects[label] = (
                tuple(imgui.get_item_rect_min()),
                tuple(imgui.get_item_rect_max()),
            )
            return result

        widgets_mod.disabled_button = spy_button

        orig_popup = imgui.begin_popup_context_item

        def spy_popup(tag: str, flags: int) -> bool:
            popup_rects[tag] = (
                tuple(imgui.get_item_rect_min()),
                tuple(imgui.get_item_rect_max()),
            )
            return orig_popup(tag, flags)

        imgui.begin_popup_context_item = spy_popup

        try:
            imgui.new_frame()
            imgui.set_next_window_size((760.0, 900.0))
            imgui.begin("smoke")
            try:
                sirens_patterns._headers(ctx, state, tab, pattern, 0, 5)
            finally:
                imgui.end()
                imgui.end_frame()
                imgui.render()
        finally:
            widgets_mod.disabled_button = orig_button
            imgui.begin_popup_context_item = orig_popup

        mute_key = next(k for k in button_rects if f"sirens-mute-{channel_uid}" in k)
        solo_key = next(k for k in button_rects if f"sirens-solo-{channel_uid}" in k)
        popup_rect = popup_rects[f"sirens-chan-menu-{channel_uid}"]
        assert popup_rect == button_rects[mute_key], (
            "the channel popup is bound to "
            + ("the solo button" if popup_rect == button_rects[solo_key] else "neither button")
            + ", not the name/mute button"
        )
    finally:
        imgui.destroy_context(gui_ctx)
        if previous is not None:
            imgui.set_current_context(previous)


# --- sirens-panes-02: a sound effect's own pattern was in the Patterns list ----


def test_deleting_a_sound_effects_pattern_is_refused_or_takes_the_effect_with_it():
    """``add_oneshot`` mints its effect's pattern into ``doc.patterns`` (the
    same list ``orders._patterns`` draws), and that pane's Delete button
    counted only ``doc.order`` for "is this used" -- never ``doc.oneshots`` --
    so a fresh effect's pattern showed up in the Patterns list with a Delete
    button that took it with no confirmation at all, leaving the effect naming
    a pattern that no longer existed. Fixed by excluding a sound effect's own
    pattern from the pane's list (``sirens_orders.visible_patterns``), the same
    place ``pattern_room`` lives, so it can be asserted with no imgui frame.
    """
    from realmspinner.studio.modes.sirens.ui.panes import orders as sirens_orders

    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    song_pattern = doc.patterns[0].uid
    effect = doc.add_oneshot("boom", rows=4)

    visible = sirens_orders.visible_patterns(doc)
    visible_uids = {pattern.uid for pattern in visible}

    assert effect.pattern not in visible_uids, (
        "a sound effect's private pattern is still offered in the song's own"
        " Patterns list, where Delete takes it with no confirmation"
    )
    assert song_pattern in visible_uids
    # The document itself is untouched -- only the pane's list is filtered.
    assert doc.pattern(effect.pattern) is not None


# --- sirens-engine-01: an unbounded arpeggio/pitch value overflowed frequency --


def test_a_huge_arpeggio_or_pitch_sequence_value_still_renders():
    """``_NOTE_CLAMP`` (the 2026-09-23 audit's own fix, sirens-01) bounds only
    ``voice.note`` inside ``_advance``; the value actually handed to
    ``notes.frequency`` in ``_sound`` is ``cents(note + arp, bend)``, and
    neither ``arp`` (an instrument's own arpeggio sequence) nor ``bend``
    (``voice.pitch_bend``, which a pitch sequence adds to every tick with
    nothing capping it) passed through that clamp. A sequence value in the
    billions -- reachable from a hand-typed envelope or a round-tripped
    ``.rsng`` -- pushed the *combined* pitch past what ``math.pow`` can raise
    2 to, aborting render, playback and export for the rest of that song's
    life.

    Fails against the unfixed code (confirmed against ``git show HEAD``'s
    ``_sound``, run from the scratchpad rather than checked out): ``OverflowError:
    math range error`` out of ``notes.frequency``.
    """
    instrument = inst.Instrument(
        uid=0,
        name="Huge",
        kind="pulse",
        volume=inst.Sequence(values=(15,)),
        duty=inst.Sequence(values=(2,)),
        arpeggio=inst.Sequence(values=(2_000_000_000,)),
    )
    doc = D.SongDoc(
        channels=[D.Channel(uid=D.new_uid(), name="Pulse 1", kind="pulse")],
        instruments=[instrument],
        patterns=[D.Pattern(uid=D.new_uid(), cells=D.empty_cells(4, 1))],
        tempo=150,
        speed=6,
    )
    out = synth.render_note(doc, 0, 48, kind="pulse", rows=8)
    assert out.size > 0
    assert np.isfinite(out).all(), "a huge arpeggio value left non-finite samples in the buffer"


def test_render_pattern_survives_an_extreme_pitch_sequence_too():
    """The other contributor the clamp above must also cover: a pitch
    sequence (``voice.pitch_bend``) rather than an arpeggio one, held across
    several rows so it is the *accumulated* value, not a single lookup, that
    reaches ``notes.frequency``.
    """
    instrument = inst.Instrument(
        uid=0,
        name="Huge pitch",
        kind="pulse",
        volume=inst.Sequence(values=(15,), loop=0),
        duty=inst.Sequence(values=(2,)),
        pitch=inst.Sequence(values=(500_000_000,), loop=0),
    )
    cells = D.empty_cells(8, 1)
    cells[0, 0, D.NOTE] = 48
    cells[0, 0, D.INSTRUMENT] = 0
    pattern = D.Pattern(uid=D.new_uid(), cells=cells)
    doc = D.SongDoc(
        channels=[D.Channel(uid=D.new_uid(), name="Pulse 1", kind="pulse")],
        instruments=[instrument],
        patterns=[pattern],
        order=[pattern.uid],
        tempo=150,
        speed=6,
    )
    out = synth.render_pattern(doc, pattern.uid)
    assert out.size > 0
    assert np.isfinite(out).all()


def test_read_rsng_sequence_refuses_an_infinite_value_by_name():
    """The other half of the finding: ``1e999`` parses as ``float('inf')`` in
    JSON, and ``int(inf)`` raises ``OverflowError`` -- a type this function's
    ``except (TypeError, ValueError)`` did not list, so a malformed sequence
    value crashed the whole load instead of refusing with ``rsng._MALFORMED``
    like every other bad field in this manifest.
    """
    with pytest.raises(ValueError, match=rsng._MALFORMED):
        rsng._sequence_from({"values": [float("inf")]})
    with pytest.raises(ValueError, match=rsng._MALFORMED):
        rsng._sequence_from({"values": [float("nan")]})


# --- sirens-engine-02: undoing an order shrink lost the loop point -------------


def _song() -> D.SongDoc:
    return D.new_song()


def test_undoing_an_order_shrink_restores_the_loop_point():
    """``_apply_order`` resets ``doc.loop_order`` to -1 whenever the new order
    is too short to hold it -- arithmetic left over from the change, per its
    own comment, not a choice the user made. But neither ``OrderEdit`` nor
    ``PatternRemoveEdit`` carried the *old* loop point, so undoing back to the
    longer order restored the order list and left the loop point at -1 instead
    of putting it back where the user set it.
    """
    doc = _song()
    first = doc.patterns[0].uid
    second = doc.add_pattern().uid
    third = doc.add_pattern().uid
    doc.set_order([first, second, third])
    assert doc.set_song(loop_order=2)
    assert doc.loop_order == 2

    # Shrinking the order past the loop point resets it -- exercised via
    # ``set_order`` directly first.
    doc.set_order([first, second])
    assert doc.loop_order == -1
    doc.undo()
    assert doc.order == [first, second, third]
    assert doc.loop_order == 2, "OrderEdit.undo lost the loop point on the way back"

    doc.redo()
    assert doc.loop_order == -1


def test_undoing_a_pattern_removal_that_shrank_the_order_restores_the_loop_point():
    """The twin path: ``remove_pattern`` takes its order entries with it in the
    same step (``PatternRemoveEdit``), and that step can shrink the order past
    the loop point exactly the way a plain ``set_order`` can.
    """
    doc = _song()
    first = doc.patterns[0].uid
    second = doc.add_pattern().uid
    third = doc.add_pattern().uid
    doc.set_order([first, second, third])
    assert doc.set_song(loop_order=2)

    doc.remove_pattern(third)
    assert doc.order == [first, second]
    assert doc.loop_order == -1

    doc.undo()
    assert doc.order == [first, second, third]
    assert doc.pattern(third) is not None
    assert doc.loop_order == 2, "PatternRemoveEdit.undo lost the loop point on the way back"


# --- sirens-playback-01: a stale render resurrected an emptied song ------------


def test_a_render_landing_after_the_order_list_was_emptied_does_not_leave_a_playable_buffer():
    """``request_render``'s empty-order guard returns before its own in-flight
    check, so it never cancels a render already running: it only sets
    ``pcm=None``/``render_dirty=False`` on the assumption there is nothing
    left to play. ``adopt_render`` (``state.py``) then unconditionally
    overwrites ``tab.pcm`` when that stale render's result lands in
    ``on_task_done``, and deliberately never touches ``render_dirty`` (its own
    docstring explains why for the ordinary case) -- so the tab is left
    playing the removed song's audio with nothing left to notice and ask for
    a fresh, empty one.
    """
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.doc.set_order([])
    # What ``request_render``'s own early return leaves behind.
    tab.pcm, tab.loop, tab.render_dirty, tab.render_error = None, None, False, ""
    tab.rendering = True

    stale_pcm = np.zeros((10, 2), dtype=np.float32)
    sirens_mode.on_task_done(
        ctx, _Done(f"sirens-render:{tab.uid}", {"pcm": stale_pcm, "loop": None, "marks": ()})
    )

    assert tab.pcm is None, "a stale render resurrected playback of an emptied song"
    assert not tab.rendering
    assert not tab.render_dirty


# --- sirens-playback-02: a new tab never stopped the previous one's audio ------


def test_adding_a_tab_while_a_song_sounds_stops_the_device(monkeypatch):
    """S6 (2026-09-05)'s fix lived in an ``activate`` override, and that is
    only one of the two doors ``DocTabs`` has into "the active tab changed":
    ``add`` -- what a new song or an opened one goes through -- sets
    ``active_uid`` and calls ``_switched`` directly, never ``activate``. A
    song sounding under tab A when tab B is created (File > New while music is
    playing) kept sounding, with no Stop button able to reach it since the
    transport belongs to B now.
    """
    from realmspinner.studio.modes.sirens import audio as sirens_audio

    stopped: list[bool] = []
    monkeypatch.setattr(sirens_audio, "stop", lambda: stopped.append(True))

    ctx = FakeCtx()
    first = _tab(ctx)
    first.sounding = object()
    stopped.clear()  # only care about what happens from here on

    second = _tab(ctx)  # goes through DocTabs.add, not activate

    assert stopped, "creating a new tab did not stop the device"
    assert first.sounding is None
    assert second.uid != first.uid


# --- sirens-playback-03: the playhead moved the caret without dropping the anchor


def test_following_the_playhead_into_another_pattern_drops_the_selection(monkeypatch):
    """``move_caret``, ``set_caret`` and ``set_row`` (``edit.py``) all drop a
    block selection's anchor when the caret moves without being asked to
    extend it; ``follow_playhead`` moves the caret too (``state.pattern``,
    ``state.row``, ``state.order_index``) but never touched ``state.anchor``.
    A selection anchored before playback started then spans whatever the
    caret has since followed the playhead through -- across a pattern
    boundary, in this case -- and Ctrl+X/G/Delete act on that stale rectangle
    as if it had been chosen on purpose.
    """
    from .test_sirens_mode import _sounding, _two_pattern_song

    ctx = FakeCtx()
    tab, first, second = _two_pattern_song(ctx)
    state = sirens_mode.ensure(ctx)
    state.follow = True
    state.pattern = first
    state.row = 0
    state.channel = 0
    # A selection made before playback moved anywhere.
    state.anchor = (0, 0)

    at = next(mark for mark in tab.marks if mark[2] == second and mark[3] == 2)
    _sounding(monkeypatch, tab, (at[0] + 1) / 44100.0)

    assert sirens_mode.follow_playhead(ctx) is True
    assert (state.pattern, state.row) == (second, 2)
    assert state.anchor is None, (
        "the playhead moved the caret into another pattern but left a stale"
        " selection anchor behind"
    )


# --- sirens-engine-04: more rsng coercions raised a raw exception, not _MALFORMED


def test_a_null_channel_pan_is_reported_as_malformed_not_a_typeerror():
    """Every sibling numeric field in this manifest is read through
    ``rsng._int``, which turns a bad value into ``_MALFORMED`` -- ``pan`` was
    the one field read with a bare ``float()`` instead, so ``{"pan": null}``
    raised ``TypeError: float() argument must be a string or a real number,
    not 'NoneType'`` straight out of ``_channels_from``.
    """
    from .test_rsng import _repack, _song

    doc = _song()

    def edit(manifest: Any) -> None:
        manifest["channels"][0]["pan"] = None

    raw = _repack(doc, edit)
    with pytest.raises(ValueError, match="malformed"):
        rsng.read_rsng(raw)


def test_an_infinite_manifest_field_is_reported_as_malformed_not_an_overflowerror():
    """``_int``'s own guard: a JSON ``1e999`` (here written directly as
    ``float("inf")``, which round-trips through ``json.dumps``/``loads`` the
    same way) parses to an infinite float, and ``int(inf)`` is an
    ``OverflowError`` -- a type ``_int``'s ``except`` clause did not list.
    """
    from .test_rsng import _repack, _song

    doc = _song()

    def edit(manifest: Any) -> None:
        manifest["tempo"] = float("inf")

    raw = _repack(doc, edit)
    with pytest.raises(ValueError, match="malformed"):
        rsng.read_rsng(raw)


def test_an_infinite_order_entry_is_reported_as_malformed_not_an_overflowerror():
    """The order list's own coercion is a bare ``int()`` with nothing this
    file's other readers already have -- neither ``_int`` nor a local
    ``try``/``except`` of its own.
    """
    from .test_rsng import _repack, _song

    doc = _song()

    def edit(manifest: Any) -> None:
        manifest["order"] = [float("inf")]

    raw = _repack(doc, edit)
    with pytest.raises(ValueError, match="malformed"):
        rsng.read_rsng(raw)


# --- sirens-engine-05: a stale comment claimed a refusal nothing enforces -----


def test_a_pulse_instrument_plays_on_the_noise_channel_rather_than_being_refused():
    """Pins the behaviour ``instruments.KINDS``' own comment used to
    contradict: ``document.Channel``'s docstring already settled (the
    2026-09-02 review, theme T5) that the *channel* decides the waveform and
    an instrument written for another kind still plays on it -- nothing here
    is a refusal. This does not fail against the pre-fix tree (the comment was
    never enforced, so the behaviour never changed); it documents the same
    fact the comment now states correctly, so a future patch that actually
    added the refusal the old comment described would be caught here.
    """
    doc = D.SongDoc(
        channels=[D.Channel(uid=D.new_uid(), name="Noise", kind="noise")],
        instruments=[
            inst.Instrument(
                uid=0, name="Lead", kind="pulse",
                volume=inst.Sequence(values=(15,)),
                duty=inst.Sequence(values=(2,)),
            )
        ],
        patterns=[D.Pattern(uid=D.new_uid(), cells=D.empty_cells(4, 1))],
        tempo=150, speed=6,
    )
    out = synth.render_note(doc, 0, 48, kind="noise", rows=4)
    assert out.size > 0


# --- sirens-engine-06: set_cell/set_cells int16 overflow; update_instrument ---


def test_a_cell_value_past_int16_is_clipped_not_an_overflowerror():
    """``set_cells``'s own docstring promises a clip for a block that runs off
    the pattern's *edge*; a value past what an ``int16`` cell can hold got no
    such promise -- ``np.ascontiguousarray(values, dtype=np.int16)`` raised a
    bare ``OverflowError`` straight out of the document, past every
    ``ValueError`` refusal this module otherwise gives.
    """
    doc = D.new_song()
    uid = doc.patterns[0].uid
    assert doc.set_cell(uid, 0, 0, D.NOTE, 999_999)
    assert doc.pattern(uid).cells[0, 0, D.NOTE] == 32767
    assert doc.set_cell(uid, 1, 0, D.NOTE, -999_999)
    assert doc.pattern(uid).cells[1, 0, D.NOTE] == -32768


def test_a_pasted_block_past_int16_is_clipped_not_an_overflowerror():
    """The block path (a paste), not just the single-cell one."""
    doc = D.new_song()
    doc.add_channel()
    uid = doc.patterns[0].uid
    block = np.array([[[500_000], [-500_000]]])  # (1 row, 2 channels, 1 column)
    assert doc.set_cells(uid, 0, 0, D.NOTE, block)
    assert doc.pattern(uid).cells[0, 0, D.NOTE] == 32767
    assert doc.pattern(uid).cells[0, 1, D.NOTE] == -32768


def test_update_instrument_with_an_unknown_field_is_refused_not_a_typeerror():
    """``set_song`` already refuses an unknown field by name
    (``a song has no ...``); ``update_instrument`` went straight into
    ``dataclasses.replace``, which raises ``TypeError: __init__() got an
    unexpected keyword argument 'pitch_bend'`` instead.
    """
    doc = D.new_song()
    uid = doc.instruments[0].uid
    with pytest.raises(ValueError, match="instrument has no"):
        doc.update_instrument(uid, pitch_bend=5)


# --- sirens-playback-04: a failed render unlocked a save running alongside ----


def test_a_failed_render_does_not_unlock_a_save_running_alongside():
    """``on_task_failed``'s own comment already names the hazard for its three
    siblings (a sample, a pattern audition, a note preview): none of them lock
    the tab, so falling through to the unconditional ``tab.saving = False`` /
    ``tab.doc.busy = False`` would unlock a *save* genuinely running
    alongside. A render does not lock the tab either (only ``docmodes.
    start_save`` sets ``saving``/``doc.busy``), but render was missing from
    that carve-out and fell through to the same unlock anyway.
    """
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.rendering = True
    # A save genuinely in flight at the same time.
    tab.saving = True
    tab.doc.busy = True

    sirens_mode.on_task_failed(
        ctx, _Done(f"sirens-render:{tab.uid}", message="that song is too long")
    )

    assert not tab.rendering
    assert tab.render_error == "that song is too long"
    assert tab.saving, "a failed render unlocked a save that was still running"
    assert tab.doc.busy, "a failed render cleared doc.busy out from under a running save"


# --- sirens-playback-05: a forced .rsng suffix overwrote an unasked file ------


class _FakePfdResult:
    def __init__(self, answer: Any) -> None:
        self._answer = answer

    def result(self) -> Any:
        return self._answer


def test_save_as_asks_before_a_forced_rsng_suffix_replaces_a_different_file(monkeypatch, tmp_path):
    """The OS dialog's own overwrite prompt only ever saw whatever name the
    user actually typed or picked (``song.txt``, say); ``save_as`` then
    silently rewrote that to ``song.rsng`` and wrote there instead -- a
    completely different, and here pre-existing, file -- with nothing asking
    about *that* name.
    """
    from imgui_bundle import portable_file_dialogs as pfd

    from realmspinner.studio import dialogs

    ctx = FakeCtx()
    tab = _tab(ctx)
    existing = tmp_path / "song.rsng"
    existing.write_bytes(b"old bytes")
    picked = tmp_path / "song.txt"

    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: picked)
    asked: list[tuple[Any, ...]] = []

    def fake_message(title: str, text: str, choice: Any) -> Any:
        asked.append((title, text, choice))
        return _FakePfdResult(pfd.button.no)

    monkeypatch.setattr(pfd, "message", fake_message)

    sirens_mode.save_as(ctx, tab)

    assert asked, "no confirmation was asked before replacing a different existing file"
    assert ctx.result is None, "declining the replacement must read as a cancel"
    assert existing.read_bytes() == b"old bytes", "the existing song was overwritten anyway"


def test_save_as_replaces_the_file_once_confirmed(monkeypatch, tmp_path):
    """The other half: saying yes actually writes, at the ``.rsng`` path."""
    from imgui_bundle import portable_file_dialogs as pfd

    from realmspinner.studio import dialogs

    ctx = FakeCtx()
    tab = _tab(ctx)
    existing = tmp_path / "song.rsng"
    existing.write_bytes(b"old bytes")
    picked = tmp_path / "song.txt"

    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: picked)
    monkeypatch.setattr(pfd, "message", lambda *a, **k: _FakePfdResult(pfd.button.yes))

    sirens_mode.save_as(ctx, tab)

    assert ctx.result is not None
    assert Path(ctx.result["path"]) == existing
    assert existing.read_bytes() != b"old bytes"


def test_save_as_does_not_ask_again_when_the_suffix_was_already_right(monkeypatch, tmp_path):
    """No behaviour change for the ordinary case: the OS dialog's own prompt
    already covered a name that needed no correction.
    """
    from imgui_bundle import portable_file_dialogs as pfd

    from realmspinner.studio import dialogs

    ctx = FakeCtx()
    tab = _tab(ctx)
    picked = tmp_path / "song.rsng"

    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: picked)
    asked: list[bool] = []

    def fake_message(*_a: Any, **_k: Any) -> Any:
        asked.append(True)
        return _FakePfdResult(pfd.button.yes)

    monkeypatch.setattr(pfd, "message", fake_message)

    sirens_mode.save_as(ctx, tab)

    assert not asked
    assert ctx.result is not None
    assert Path(ctx.result["path"]) == picked
