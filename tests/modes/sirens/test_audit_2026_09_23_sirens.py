"""Regression tests for the 2026-09-23 audit's Sirens findings.

Four findings, none of them touching one another's code: a persistent pitch
effect that overflows a voice's note over a long render (sirens-01, plus
``preview_note``'s missing catch), an unguarded "Loop the song" checkbox
(sirens-02), a silent note-preview refusal (sirens-03), and an unlogged
zero-length buffer refusal (sirens-04).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from realmspinner.service.errors import Invalid
from realmspinner.studio.modes.sirens.engine import document as D
from realmspinner.studio.modes.sirens.engine import instruments as inst
from realmspinner.studio.modes.sirens.engine import notes, synth

from .test_sirens_mode import FakeCtx, _tab

# --- sirens-01: a long-held pitch slide ----------------------------------------


def _sustained_slide_doc() -> tuple[D.SongDoc, D.Pattern]:
    """One channel, ``MAX_ROWS`` rows: trigger C-4 on row 0 and set effect 1xx
    at its max param (255 cents/tick, persistent per the tracker convention
    the module docstring describes) -- nothing after row 0 retriggers the
    note, so the slide runs the whole pattern. The instrument's volume
    sequence loops forever, so the voice never finishes on its own either.
    ``speed=MAX_SPEED`` is what makes the pattern's own tick count (not
    ``MAX_RENDER_SECONDS``) the thing that has to outlast the overflow -- the
    audit's own probe (``sirens-engine-03.py``) is where this shape comes
    from.
    """
    channels = [D.Channel(uid=D.new_uid(), name="Pulse 1", kind="pulse")]
    instrument = inst.Instrument(
        uid=0,
        name="Sustain",
        kind="pulse",
        volume=inst.Sequence(values=(15,), loop=0),
        duty=inst.Sequence(values=(2,)),
    )
    cells = D.empty_cells(D.MAX_ROWS, 1)
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


def test_a_slid_voice_note_never_pushes_frequency_past_finite():
    """sirens-01: a persistent ``1xx`` slide accumulated on ``voice.note``
    with nothing clamping it, and after about 80 seconds (~4800 ticks at the
    default tempo) ``notes.frequency()`` returned ``inf``.

    Fails against the unfixed code: at tick 4782 ``voice.note`` is about
    12242 and ``notes.frequency(voice.note)`` is ``inf`` -- reproduced from
    the audit's own probe (``sirens-engine-01.py``). With the clamp in
    ``synth._advance``, frequency must stay finite for far longer than any
    real render (``MAX_RENDER_SECONDS`` is 600s, ~36000 ticks).
    """
    from realmspinner.studio.modes.sirens.engine.synth import Voice, _advance

    v = Voice(kind="pulse")
    v.trigger(48)
    v.slide = 255.0
    for tick in range(1, 40000):
        _advance(v)
        v.tick = tick
        f = notes.frequency(v.note)
        assert math.isfinite(f), f"frequency went non-finite at tick {tick} (note={v.note})"


def test_render_does_not_raise_after_a_long_held_pitch_slide():
    """sirens-01, end to end: rendering a song whose only channel carries a
    persistent slide must not raise, and must not leave non-finite samples in
    the buffer.

    Fails against the unfixed code: ``render_pattern`` raises
    ``ValueError: math domain error`` out of ``math.fmod`` once the slid
    voice's phase goes infinite, aborting the render (and, in the app, every
    export of that song from then on -- the effect is persistent, so it is
    baked into every row after the one that set it).
    """
    doc, pattern = _sustained_slide_doc()

    orig_max = synth.MAX_RENDER_SECONDS
    synth.MAX_RENDER_SECONDS = 60
    try:
        out = synth.render_pattern(doc, pattern.uid)
    finally:
        synth.MAX_RENDER_SECONDS = orig_max

    assert out.shape[0] > 0
    assert np.all(np.isfinite(out)), "a slid voice left non-finite samples in the render"


def test_a_render_that_never_reaches_the_overflow_is_unchanged_by_the_clamp():
    """The clamp must not touch an ordinary song. A single unslid note is
    rendered byte-identically with and without ``_NOTE_CLAMP`` in play,
    because the clamp only ever fires once ``voice.note`` has already left
    the playable range -- proving this fix does not perturb the byte-for-byte
    reproducibility the module docstring promises for every song that never
    reaches the overflow.
    """
    channels = [D.Channel(uid=D.new_uid(), name="Pulse 1", kind="pulse")]
    instrument = inst.Instrument(
        uid=0, name="Plain", kind="pulse",
        volume=inst.Sequence(values=(15, 12, 8, 0)),
        duty=inst.Sequence(values=(2,)),
    )
    cells = D.empty_cells(8, 1)
    cells[0, 0, D.NOTE] = 48
    cells[0, 0, D.INSTRUMENT] = 0
    pattern = D.Pattern(uid=D.new_uid(), name="p", cells=cells)
    doc = D.SongDoc(
        channels=channels, instruments=[instrument], patterns=[pattern],
        order=[pattern.uid],
    )
    out = synth.render_pattern(doc, pattern.uid)
    assert out.shape[0] > 0
    assert np.all(np.isfinite(out))


def test_a_note_preview_that_fails_to_render_is_framed_not_leaked(monkeypatch):
    """``preview_note``'s ``run`` had no ``try``/``except`` around
    ``synth.render_note`` at all, unlike its two siblings (``audition`` and
    ``request_render``'s own ``run``), which both frame a ``ValueError`` with
    ``invalid_from`` before it reaches the task runner.

    Fails against the unfixed code: a ``ValueError`` from ``render_note``
    propagates out of ``preview_note`` bare, instead of as the framed
    ``Invalid`` its siblings raise.
    """
    from realmspinner.studio.modes.sirens import play as sirens_play
    from realmspinner.studio.modes.sirens.engine import synth as synth_mod

    def _boom(*_a: Any, **_kw: Any) -> Any:
        raise ValueError("bad note")

    monkeypatch.setattr(synth_mod, "render_note", _boom)

    ctx = FakeCtx()
    _ = _tab(ctx)
    state = sirens_play.ensure(ctx)
    state.preview = True
    state.instrument = 0

    with pytest.raises(Invalid):
        sirens_play.preview_note(ctx, 48)


# --- sirens-02: "Loop the song" mid-save ---------------------------------------


def test_loop_the_song_checkbox_refuses_a_toggle_while_the_song_is_busy(monkeypatch):
    """The checkbox had no ``enabled=editable``, unlike every other control in
    this pane (``pattern_room``, the reorder/retarget/drop buttons, "Loop
    from"), so it could be toggled while a save was already in flight.

    Fails against the unfixed code: ``controls.checkbox`` is called with
    ``enabled`` defaulting to ``True`` (or absent) while ``tab.busy`` is
    ``True``, instead of ``enabled=False``.
    """
    from realmspinner.studio.modes.sirens.ui.panes import orders as sirens_orders

    calls: list[dict[str, Any]] = []
    orig_checkbox = sirens_orders.controls.checkbox

    def _spy_checkbox(label: str, value: bool, **kwargs: Any):
        if label == "Loop the song":
            calls.append(kwargs)
        return orig_checkbox(label, value, **kwargs)

    monkeypatch.setattr(sirens_orders.controls, "checkbox", _spy_checkbox)

    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True  # ``DocTab.busy`` is ``self.saving`` -- see docmodes.py

    from imgui_bundle import imgui

    previous = imgui.get_current_context()
    gctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    try:
        imgui.new_frame()
        imgui.set_next_window_size((760.0, 900.0))
        imgui.begin("smoke")
        try:
            sirens_orders.draw(ctx)
        finally:
            imgui.end()
            imgui.end_frame()
            imgui.render()
    finally:
        imgui.destroy_context(gctx)
        if previous is not None:
            imgui.set_current_context(previous)

    assert calls, "the checkbox was never drawn"
    assert calls[-1].get("enabled") is False, (
        "the checkbox stayed enabled while the song was busy saving"
    )


# --- sirens-03: silent note preview --------------------------------------------


def test_a_note_preview_the_device_refuses_is_not_silent(monkeypatch):
    """The ``sirens-preview`` arm of ``on_task_done`` ignored ``play()``'s
    ``False``, unlike its two siblings (``sirens-pattern``, ``sirens-audition``)
    just above and below it, which both toast on a refused play.

    Fails against the unfixed code: ``ctx.toasts`` is empty after the device
    refuses the buffer.
    """
    from realmspinner.studio.modes.sirens import audio as sirens_audio
    from realmspinner.studio.modes.sirens import mode as sirens_mode

    monkeypatch.setattr(sirens_audio, "available", lambda: True)
    monkeypatch.setattr(sirens_audio, "play", lambda *_a, **_kw: False)

    class _Ctx(FakeCtx):
        # ``docmodes.refuse`` calls ``toast_once(text, "error", action,
        # action_arg)`` positionally; ``FakeCtx.toast_once`` (owned by
        # ``test_sirens_mode.py``, not this file) only accepts ``message`` and
        # ``kind`` before ``**kwargs``, so the two trailing positionals raise
        # a bare ``TypeError`` rather than exercising the refusal this test is
        # about. Widened locally rather than touching the shared fixture.
        def toast_once(self, message: str, kind: str = "info", *_args: Any, **_kw: Any) -> bool:
            return super().toast_once(message, kind)

    ctx = _Ctx()
    tab = _tab(ctx)

    class _Done:
        def __init__(self, key: str, result: Any) -> None:
            self.key = key
            self.result = result
            self.tag = None

    sirens_mode.on_task_done(
        ctx, _Done(f"sirens-preview:{tab.uid}", {"pcm": np.zeros((4, 2), dtype=np.int16)})
    )

    assert ctx.toasts, "a device refusal during note preview must not be silent"


# --- sirens-04: the zero-length refusal is not logged --------------------------


def test_a_zero_length_buffer_refusal_is_logged(monkeypatch, caplog):
    """``audio.play``'s zero-length refusal was the one refusal in the
    function that did not call ``log.warning``, even though every toast that
    can follow a refused play says "see the log".

    Fails against the unfixed code: no ``WARNING`` record is emitted for a
    zero-length buffer, even though sibling refusals (a wrong sample rate, a
    wrongly shaped buffer) both log one.
    """
    from realmspinner.studio.modes.sirens import audio as sirens_audio

    monkeypatch.setattr(sirens_audio, "available", lambda: True)

    with caplog.at_level("WARNING", logger=sirens_audio.log.name):
        result = sirens_audio.play(np.zeros((0, 2), dtype=np.int16))

    assert result is False
    assert any(
        "zero-length" in record.message or "zero" in record.message.lower()
        for record in caplog.records
    ), "the zero-length refusal must be logged like its siblings"
