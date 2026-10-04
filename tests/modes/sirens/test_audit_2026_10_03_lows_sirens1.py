"""The 2026-10-03 audit's Low findings on Sirens (sirens-17 .. sirens-30).

One test per claim, named for it. The pane tests draw real panes inside a real
(GL-less) imgui frame and read the control census, so a rect or a reason is
read and never computed.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.studio import probe
from realmspinner.studio.modes.sirens import audio as sirens_audio
from realmspinner.studio.modes.sirens import fileio as sirens_io
from realmspinner.studio.modes.sirens import mode as sirens_mode
from realmspinner.studio.modes.sirens.engine import document as D
from realmspinner.studio.modes.sirens.engine import notes, synth
from realmspinner.studio.modes.sirens.ui.panes import bridge as bridge_pane
from realmspinner.studio.modes.sirens.ui.panes import effects as effects_pane
from realmspinner.studio.modes.sirens.ui.panes import envelopes as envelopes_pane
from realmspinner.studio.modes.sirens.ui.panes import instruments as instruments_pane
from realmspinner.studio.modes.sirens.ui.panes import orders as orders_pane
from realmspinner.studio.modes.sirens.ui.panes import patterns as patterns_pane
from realmspinner.studio.modes.sirens.ui.panes import transport as transport_pane

from .test_sirens_export import _effect, _export, _song, _written
from .test_sirens_mode import FakeCtx, _render, _tab, _two_pattern_song
from .test_sirens_panes_smoke import _loaded

REPO = Path(__file__).resolve().parents[3]
ENGINE = REPO / "src" / "realmspinner" / "studio" / "modes" / "sirens" / "engine"


@pytest.fixture(autouse=True)
def _no_device(monkeypatch):
    monkeypatch.setattr(sirens_audio, "available", lambda: False)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)


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
    imgui.set_next_window_size((900.0, 1400.0))
    imgui.set_next_window_pos((0.0, 0.0))
    imgui.begin("##host")
    build()
    imgui.end()
    imgui.end_frame()
    return list(probe.FRAME_CONTROLS)


def _audible(monkeypatch):
    played: list[tuple[object, dict]] = []
    monkeypatch.setattr(sirens_audio, "available", lambda: True)
    monkeypatch.setattr(
        sirens_audio, "play", lambda pcm, **kw: played.append((pcm, kw)) or True
    )
    return played


# --- sirens-17: the caret after a pattern delete --------------------------------


def _song_effect_song(ctx):
    """Patterns [A, fx, B]: a sound effect's private pattern sits between two
    song patterns."""
    tab = _tab(ctx)
    doc = tab.doc
    first = doc.patterns[0].uid
    effect = doc.add_oneshot("coin")
    second = doc.add_pattern().uid
    assert [one.uid for one in doc.patterns] == [first, effect.pattern, second]
    return tab, first, effect, second


def test_deleting_a_song_pattern_does_not_move_the_caret_into_a_sound_effect():
    ctx = FakeCtx()
    tab, first, effect, second = _song_effect_song(ctx)
    sirens_mode.set_caret(ctx, pattern=first)

    sirens_mode.confirm_remove_pattern(ctx, tab, first, 0)

    assert tab.doc.pattern(first) is None
    state = sirens_mode.ensure(ctx)
    assert state.pattern != effect.pattern, "the grid silently became the effect editor"
    assert state.pattern == second


def test_clamp_caret_falls_back_to_a_song_pattern_not_a_sound_effect():
    ctx = FakeCtx()
    tab, first, effect, second = _song_effect_song(ctx)
    tab.doc.remove_pattern(first)
    state = sirens_mode.ensure(ctx)
    state.pattern = first  # stale: the pattern is gone

    sirens_mode.clamp_caret(ctx)

    assert state.pattern == second


# --- sirens-18: an effects-only export -------------------------------------------


def test_an_effects_only_export_writes_no_empty_song_or_stems(tmp_path):
    ctx = FakeCtx()
    tab = _song(ctx)
    tab.doc.set_order([])
    _effect(tab, "coin")
    out = tmp_path / "audio"
    result = _export(ctx, tab, out)
    assert _written(out) == {"sfx/coin.wav"}, "header-only WAVs a build script would glob"
    assert result["files"] == 1
    assert sirens_io.SONG_NAME == "song.wav"


# --- sirens-19: mutator refusals --------------------------------------------------


def test_update_channel_and_update_oneshot_refuse_unknown_fields_with_a_value_error():
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    channel = doc.channels[0]
    one = doc.add_oneshot("coin")
    other = doc.add_pattern().uid
    instrument = doc.instruments[0]

    with pytest.raises(ValueError):
        doc.update_channel(channel.uid, bogus=1)
    # ``uid=`` cannot get through at all: it is the mutator's own positional
    # parameter, so Python refuses the duplicate before the body runs.
    with pytest.raises(TypeError):
        doc.update_channel(channel.uid, uid=doc.channels[1].uid)
    with pytest.raises(ValueError):
        doc.update_oneshot(one.uid, bogus=1)
    with pytest.raises(TypeError):
        doc.update_oneshot(one.uid, uid=one.uid + 1000)
    with pytest.raises(ValueError):
        doc.update_oneshot(one.uid, pattern=987654)
    with pytest.raises(TypeError):
        doc.update_instrument(instrument.uid, uid=instrument.uid + 1)

    # Nothing moved, and the legitimate edits still work.
    assert doc.channel(channel.uid) == channel
    assert doc.oneshot(one.uid) == one
    assert doc.update_oneshot(one.uid, pattern=other)
    assert doc.update_channel(channel.uid, name="Lead")


# --- sirens-20: a sample voice at another rate ------------------------------------


def _crossings_per_second(pcm: np.ndarray, rate: int) -> float:
    mono = pcm.sum(axis=1)[: rate // 10]
    signs = np.signbit(mono[np.abs(mono) > 1e-4])
    return float(np.count_nonzero(signs[1:] != signs[:-1])) / 2.0 / (rate // 10 / rate)


def test_a_sample_voice_keeps_its_pitch_when_rendered_at_another_rate():
    heard = {}
    for rate in (44100, 22050):
        doc = D.new_song()
        doc.set_song(tempo=150, speed=6)
        doc.resize_pattern(doc.patterns[0].uid, 8)
        # A 441 Hz sine at the source rate: 100 samples a period.
        tone = 0.5 * np.sin(2 * np.pi * np.arange(40_000) / 100.0)
        doc.set_sample("sine", tone.astype(np.float32))
        sampler = next(one for one in doc.instruments if one.kind == "sample")
        doc.update_instrument(sampler.uid, sample="sine")
        index = next(i for i, one in enumerate(doc.channels) if one.kind == "sample")
        pattern = doc.patterns[0].uid
        doc.set_cell(pattern, 0, index, D.NOTE, notes.SAMPLE_BASE_NOTE)
        doc.set_cell(pattern, 0, index, D.INSTRUMENT, sampler.uid)
        pcm, _loop = synth.render(doc, rate=rate)
        heard[rate] = _crossings_per_second(pcm, rate)
    assert heard[44100] == pytest.approx(441.0, rel=0.1)
    assert heard[22050] == pytest.approx(heard[44100], rel=0.1), heard


# --- sirens-21: the engine package docstring --------------------------------------


def test_the_engine_package_docstring_lists_exactly_its_own_modules():
    import realmspinner.studio.modes.sirens.engine as engine

    listed = set(re.findall(r"^``(\w+)``", engine.__doc__ or "", flags=re.MULTILINE))
    own = {path.stem for path in ENGINE.glob("*.py") if path.stem != "__init__"}
    assert listed == own, (sorted(listed - own), sorted(own - listed))
    assert "kernels/audio/wavout" in (engine.__doc__ or "")


# --- sirens-22: Loop playback toggled while the song sounds -----------------------


def test_toggling_loop_playback_while_playing_keeps_the_position(ui, monkeypatch):
    played = _audible(monkeypatch)
    ctx = FakeCtx()
    tab, _first, _second = _two_pattern_song(ctx)
    state = sirens_mode.ensure(ctx)
    assert sirens_mode.play(ctx, tab) is True
    assert len(played) == 1
    # One second in: the playhead is not at the top.
    monkeypatch.setattr(sirens_audio, "tag", lambda: tab.uid)
    monkeypatch.setattr(sirens_audio, "position", lambda: 1.0)
    offset = int(1.0 * sirens_audio.RATE)

    def build():
        transport_pane.draw(ctx)

    controls = _frame(ui, build)
    box = next(c for c in controls if c.label.startswith("Loop playback"))
    _frame(ui, build, pos=box.centre, down=True)
    _frame(ui, build, pos=box.centre, down=False)

    assert state.loop_playback is True
    assert len(played) == 2, "the toggle is applied to what is sounding"
    pcm, kwargs = played[-1]
    assert kwargs["loops"] == -1
    assert tab.sounding.anchor == offset, "the song jumped back to the top"
    assert len(pcm) == len(tab.pcm)
    assert np.array_equal(pcm[0], tab.pcm[offset])


# --- sirens-23: greyed fields name why --------------------------------------------

_FIELDS = {"slider_int", "slider_float", "drag_int", "drag_float", "input_text", "combo"}


def test_every_sirens_field_greyed_while_busy_carries_a_reason(ui, monkeypatch):
    ctx = FakeCtx()
    tab = _loaded(ctx)
    tab.saving = True  # the song is being written
    tab.doc.set_song(loop_order=0)
    state = sirens_mode.ensure(ctx)
    sample = next(one for one in tab.doc.instruments if one.kind == "sample")
    # The channel menu's body, forced open (a right-click on a header opens it).
    monkeypatch.setattr(ui, "begin_popup_context_item", lambda *_a, **_k: True)
    monkeypatch.setattr(ui, "end_popup", lambda: None)

    seen: list = []
    for instrument in (tab.doc.instruments[0].uid, sample.uid):
        state.instrument = instrument
        state.pattern = tab.doc.patterns[0].uid

        def build():
            for pane in (
                transport_pane, orders_pane, instruments_pane, envelopes_pane,
                effects_pane, patterns_pane,
            ):
                pane.draw(ctx)

        _frame(ui, build)
        seen.extend(probe.FRAME_CONTROLS)
    fields = [c for c in seen if c.kind in _FIELDS]
    assert fields, "no field was drawn"
    mute = [c for c in fields if not c.enabled and not c.reason]
    assert not mute, [(c.label, c.kind) for c in mute]


# --- sirens-24: a cell naming an instrument the song does not hold ---------------


def test_the_grid_marks_a_cell_naming_an_instrument_the_song_does_not_hold():
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    pattern = doc.patterns[0]
    doc.set_cell(pattern.uid, 0, 0, D.NOTE, 48)
    doc.set_cell(pattern.uid, 0, 0, D.INSTRUMENT, doc.instruments[0].uid)
    doc.set_cell(pattern.uid, 1, 0, D.NOTE, 50)
    doc.set_cell(pattern.uid, 1, 0, D.INSTRUMENT, 0x7E)
    known = {one.uid for one in doc.instruments}
    assert 0x7E not in known

    mark = patterns_pane.cell_names_missing_instrument
    assert mark(known, pattern.cells, 1, 0) is True
    assert mark(known, pattern.cells, 0, 0) is False
    assert mark(known, pattern.cells, 2, 0) is False, "an empty cell names nothing"


# --- sirens-25 / sirens-29: the manual ---------------------------------------------


def _chapter(name: str) -> str:
    return (REPO / "docs" / "manual" / name).read_text(encoding="utf-8")


def test_manual_34_names_every_control_the_song_file_panel_draws():
    text = _chapter("34-sirens.md")
    flat = " ".join(text.split())
    assert "Compose in Muse" in flat
    assert "Closeness" in flat
    assert "35-muse.md" in text
    assert "is every pattern the document holds" not in flat
    # And the pane really draws both, so the chapter is being held to the code.
    source = (Path(bridge_pane.__file__)).read_text(encoding="utf-8")
    assert "Compose in Muse" in source and "_CLOSENESS" in source


def _sirens_section(text: str) -> str:
    head = text.index("\n## Sirens\n")
    rest = text[head + 1:]
    nxt = rest.find("\n## ", 5)
    return rest if nxt < 0 else rest[:nxt]


def test_chapter_38_lists_the_sirens_panic_key():
    from realmspinner.studio.shortcuts import shortcut_sections

    assert "Shift+Esc" in _sirens_section(_chapter("38-shortcuts.md"))
    sheet = dict(shortcut_sections())["Sirens"]
    keys = {key for key, _what in sheet}
    assert "Shift+Esc" in keys
    assert "Ctrl+G" in keys
    assert "Ctrl+Up / Ctrl+Down" in keys


# --- sirens-26: play-from-caret on an effect's pattern -----------------------------


def test_play_from_caret_on_an_effect_pattern_does_not_advise_adding_it_to_the_order(
    monkeypatch,
):
    played = _audible(monkeypatch)
    ctx = FakeCtx()
    tab = _tab(ctx)
    effect = tab.doc.add_oneshot("coin")
    _render(ctx, tab)
    sirens_mode.set_caret(ctx, pattern=effect.pattern, row=0)

    assert sirens_mode.play_from_caret(ctx, tab) is False
    assert played == []
    assert ctx.toasts, "the refusal said nothing"
    message = ctx.toasts[-1][0]
    assert "order list" not in message, "advice the order pane then refuses"
    assert "coin" in message and "sound effect" in message


# --- sirens-27: an effect edit does not re-render the song --------------------------


def test_editing_an_effects_pattern_does_not_rearm_the_song_render():
    ctx = FakeCtx()
    tab = _tab(ctx)
    effect = tab.doc.add_oneshot("coin")
    tab.render_dirty = False
    sirens_mode.set_caret(ctx, pattern=effect.pattern, row=0, channel=0, column=D.NOTE)

    assert sirens_mode.write_note(ctx, 0) is True
    assert tab.doc.pattern(effect.pattern).cells[0, 0, D.NOTE] != notes.EMPTY
    assert not tab.render_dirty, "the song's buffer cannot have changed"

    # And a song pattern, which the render does read, still re-arms it.
    sirens_mode.set_caret(ctx, pattern=tab.doc.order[0], row=0, channel=0, column=D.NOTE)
    assert sirens_mode.write_note(ctx, 0) is True
    assert tab.render_dirty


# --- sirens-28: the render_generation comment ---------------------------------------


def test_render_generation_comment_does_not_claim_playback_keys_on_it():
    source = (
        REPO / "src" / "realmspinner" / "studio" / "modes" / "sirens" / "state.py"
    ).read_text(encoding="utf-8")
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("# Bumped once"))
    block = []
    for line in lines[start:]:
        if not line.strip().startswith("#"):
            break
        block.append(line)
    comment = " ".join(block)
    assert "Playback keys on it" not in comment
    assert "Sounding" in comment


# --- sirens-30: the two popup bodies a right-click opens ----------------------------


def test_the_channel_menu_and_effect_chooser_bodies_draw(ui, monkeypatch):
    ctx = FakeCtx()
    tab = _loaded(ctx)
    channel = tab.doc.channels[0]
    monkeypatch.setattr(ui, "begin_popup_context_item", lambda *_a, **_k: True)
    monkeypatch.setattr(ui, "begin_popup", lambda *_a, **_k: True)
    monkeypatch.setattr(ui, "end_popup", lambda: None)

    def build():
        patterns_pane._channel_popup(ctx, tab, channel, 0)
        patterns_pane._effect_popup(ctx, "sirens-fx-popup", 0, 0)

    controls = _frame(ui, build)
    labels = [c.label for c in controls]
    assert any("sirens-chan-menu" in label and "kind" in label for label in labels), labels
    assert any(label.startswith("Pan##sirens-chan-menu") for label in labels), labels
    assert sum("sirens-fx-choice-" in label for label in labels) == len(synth.EFFECT_NAMES)
