"""The 2026-10-03 audit's Medium findings on the Sirens engine and panes (sirens-03 .. -12).

One test per claim, named for it. The pane tests press real controls inside a
real (GL-less) imgui frame and read the control census, the way
``tests/modes/inker/test_timeline_cel_opacity_input.py`` does, so a rect is read
and never computed.
"""

from __future__ import annotations

import sys
import threading
import time
import warnings
import zipfile

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.studio import probe
from realmspinner.studio.modes.sirens import mode as sirens_mode
from realmspinner.studio.modes.sirens.engine import document as D
from realmspinner.studio.modes.sirens.engine import envelope, rsng, synth
from realmspinner.studio.modes.sirens.engine import instruments as inst
from realmspinner.studio.modes.sirens.ui.panes import envelopes as env_pane
from realmspinner.studio.modes.sirens.ui.panes import orders as orders_pane
from realmspinner.studio.modes.sirens.ui.panes import patterns as patterns_pane

from .test_rsng import _repack, _song
from .test_sirens_mode import FakeCtx, _tab

# --- sirens-03: a noise voice past the int64 phase range ------------------------


def _held_noise_slide_doc() -> tuple[D.SongDoc, D.Pattern]:
    channels = [D.Channel(uid=D.new_uid(), name="Noise", kind="noise")]
    instrument = inst.Instrument(
        uid=0, name="Hiss", kind="noise",
        volume=inst.Sequence(values=(15,), loop=0),
        duty=inst.Sequence(values=(0,)),
    )
    cells = D.empty_cells(80, 1)  # 80 rows is 8 s at tempo 150
    cells[0, 0, D.NOTE] = 48
    cells[0, 0, D.INSTRUMENT] = 0
    cells[0, 0, D.EFFECT] = synth.FX_SLIDE_UP
    cells[0, 0, D.PARAM] = 0xFF
    pattern = D.Pattern(uid=D.new_uid(), name="p", cells=cells)
    doc = D.SongDoc(
        channels=channels, instruments=[instrument], patterns=[pattern],
        order=[pattern.uid], speed=D.MAX_SPEED, tempo=150,
    )
    return doc, pattern


def test_a_noise_slide_held_past_the_int64_phase_range_still_renders_hiss():
    doc, pattern = _held_noise_slide_doc()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = synth.render_pattern(doc, pattern.uid)
    tail = out[-synth.SAMPLE_RATE:, 0]
    assert tail.size == synth.SAMPLE_RATE
    # Hiss, not a constant plateau: the unfixed cast left one value for the
    # rest of the note.
    assert len(np.unique(tail)) > 100, "the noise collapsed to a constant"
    assert not [w for w in caught if "invalid value encountered in cast" in str(w.message)]


# --- sirens-04: a release below the loop ----------------------------------------


def test_a_release_cannot_be_dragged_or_toggled_to_or_below_the_loop_point():
    seq = inst.Sequence(values=tuple(range(8)), loop=4, release=6)
    dragged = envelope.moved(seq, "release", 2)
    assert dragged.release > dragged.loop

    looped = inst.Sequence(values=tuple(range(8)), loop=5)
    landed = envelope.toggled(looped, "release")
    assert landed.release < 0 or landed.release > landed.loop

    # No room above a loop on the last step: refused rather than put where the
    # loop would stop being drawn.
    last = inst.Sequence(values=tuple(range(8)), loop=7)
    assert envelope.toggled(last, "release") == last


# --- sirens-05: the uid counter -------------------------------------------------


def test_read_rsng_on_a_worker_thread_never_re_issues_a_uid_to_the_frame_thread(monkeypatch):
    # The counter is a process global and this test spins it into the millions:
    # restored afterwards, so no later test sees it move.
    monkeypatch.setattr(D, "_next_uid", D._next_uid)
    doc = _song()
    raw = _repack(doc, lambda manifest: None)
    # Hand the reader uids far above the counter so every open reserves hard.
    old_switch = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    stop = threading.Event()
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            while not stop.is_set():
                rsng.read_rsng(raw)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    thread = threading.Thread(target=worker)
    backwards = 0
    last = 0
    try:
        thread.start()
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            for _ in range(500):
                uid = D.new_uid()
                # Every uid the frame thread is handed must be larger than the
                # one before: a counter that moves backwards re-issues one.
                if uid <= last:
                    backwards += 1
                last = uid
    finally:
        stop.set()
        thread.join()
        sys.setswitchinterval(old_switch)
    assert not errors, errors
    assert backwards == 0, f"the frame thread was handed {backwards} uids twice"


# --- sirens-06: the decoded total of a document's samples -----------------------


def test_read_rsng_refuses_a_song_whose_samples_decode_past_a_document_ceiling(monkeypatch):
    doc = D.new_song()
    for key in ("a", "b", "c"):
        doc.set_sample(key, np.linspace(-1.0, 1.0, 2000, dtype=np.float32))
    raw = rsng.rsng_bytes(doc)
    monkeypatch.setattr(rsng, "MAX_DOC_SAMPLE_FRAMES", 3000, raising=False)
    with pytest.raises(ValueError, match="samples"):
        rsng.read_rsng(raw)


# --- sirens-07: a long sample is not recompressed on every snapshot -------------


def test_rsng_bytes_of_an_unchanged_long_sample_does_not_recompress_it(monkeypatch):
    doc = D.new_song()
    rng = np.random.default_rng(7)
    doc.set_sample("take", rng.uniform(-1.0, 1.0, 2_000_000).astype(np.float32))
    first = rsng.rsng_bytes(doc)  # warms the encoded-WAV cache

    seen = {"bytes": 0}
    real = zipfile._get_compressor

    class Counting:
        def __init__(self, inner):
            self._inner = inner

        def compress(self, data):
            seen["bytes"] += len(data)
            return self._inner.compress(data)

        def flush(self):
            return self._inner.flush()

    def counting(compress_type, compresslevel=None):
        inner = real(compress_type, compresslevel)
        return None if inner is None else Counting(inner)

    monkeypatch.setattr(zipfile, "_get_compressor", counting)
    second = rsng.rsng_bytes(doc)
    assert second == first
    assert seen["bytes"] < 1_000_000, f"{seen['bytes']} bytes were deflated again"
    # And what it wrote still opens.
    assert rsng.read_rsng(second).samples["take"].size == 2_000_000


# --- sirens-08: a sequence value past float range -------------------------------


def test_a_sequence_value_past_float_range_is_refused_at_read_rather_than_aborting_the_render():
    def huge(manifest):
        manifest["instruments"][0]["arpeggio"] = {"values": [10**400]}

    raw = _repack(_song(), huge)
    with pytest.raises(ValueError):
        rsng.read_rsng(raw)
    with pytest.raises(ValueError):
        inst.Sequence(values=(10**400,))


# --- the pane tests -------------------------------------------------------------


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _frame(imgui, build, *, pos=(400.0, 300.0), down=False):
    io = imgui.get_io()
    io.add_mouse_pos_event(pos[0], pos[1])
    io.add_mouse_button_event(0, down)
    probe.begin_frame()
    imgui.new_frame()
    imgui.set_next_window_size((900.0, 700.0))
    imgui.set_next_window_pos((0.0, 0.0))
    imgui.begin("##host")
    build()
    imgui.end()
    imgui.end_frame()
    return list(probe.FRAME_CONTROLS)


# --- sirens-09: the channel headers sit over their columns ----------------------


def test_channel_headers_sit_over_the_grid_columns_they_name(ui):
    from realmspinner.studio.tokens import sp

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    pattern = tab.doc.patterns[0]
    origin = {}

    def build():
        origin["x"] = ui.get_cursor_screen_pos().x
        patterns_pane._headers(ctx, state, tab, pattern, 0, 4)

    _frame(ui, build)
    controls = _frame(ui, build)
    mutes = {c.label: c for c in controls if "sirens-mute-" in c.label}
    assert len(mutes) == 4, [c.label for c in controls]
    xs = [c.hit[0] for c in sorted(mutes.values(), key=lambda c: c.hit[0])]
    column0 = origin["x"] + sp(patterns_pane.GUTTER_W)
    for index, x in enumerate(xs):
        expected = column0 + index * sp(patterns_pane.CHANNEL_W)
        assert abs(x - expected) < 1.0, (index, x, expected)


# --- sirens-10: the last song pattern cannot be deleted -------------------------


def test_pattern_delete_is_disabled_for_the_last_song_pattern_when_a_sound_effect_exists(ui):
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.doc.add_oneshot("jump")
    assert len(tab.doc.patterns) == 2
    state = sirens_mode.ensure(ctx)
    state.pattern = tab.doc.patterns[0].uid

    def build():
        orders_pane._patterns(ctx, state, tab, True)

    _frame(ui, build)
    controls = _frame(ui, build)
    drops = [c for c in controls if "sirens-pattern-drop-" in c.label]
    assert drops, [c.label for c in controls]
    assert not drops[0].enabled


# --- sirens-11: the envelope Steps field drags ----------------------------------


def test_dragging_the_envelope_steps_field_changes_the_sequence_length(ui):
    ctx = FakeCtx()
    tab = _tab(ctx)
    uid = tab.doc.instruments[0].uid
    tab.doc.update_instrument(uid, volume=inst.Sequence(values=(15,) * 8, loop=0))
    state = sirens_mode.ensure(ctx)

    def build():
        instrument = next(one for one in tab.doc.instruments if one.uid == uid)
        env_pane._header(ctx, state, tab, instrument, "volume", "Volume", instrument.volume)

    _frame(ui, build)
    controls = _frame(ui, build)
    field = next(c for c in controls if c.label == "##sirens-env-steps-volume")
    x, y, w, h = field.hit
    start = (x + 6.0, y + h * 0.5)
    _frame(ui, build, pos=start, down=True)
    for step in range(1, 21):
        _frame(ui, build, pos=(start[0] + step, start[1]), down=True)
    _frame(ui, build, pos=(start[0] + 20, start[1]), down=False)
    _frame(ui, build, pos=(start[0] + 20, start[1]), down=False)

    after = next(one for one in tab.doc.instruments if one.uid == uid).volume
    assert len(after.values) != 8, "the drag changed nothing"


# --- sirens-12: moving the looped order entry is one undo step ------------------


def test_moving_the_looped_order_entry_is_one_undo_step():
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    first = doc.patterns[0].uid
    second = doc.add_pattern().uid
    third = doc.add_pattern().uid
    doc.set_order([first, second, third], loop_order=1)
    before = (list(doc.order), doc.loop_order)

    orders_pane._reorder(ctx, tab, 1, 2)
    assert doc.order == [first, third, second]
    assert doc.loop_order == 2

    doc.undo()
    assert (list(doc.order), doc.loop_order) == before
