"""Regression tests for the 2026-09-15 audit's sirens fixer batch.

sirens-06 is a comment/docstring fix with no observable behaviour, so it has
no test here (its own finding says so). The rest close one finding each:
sirens-01 (a dangling instrument reference), sirens-02 and sirens-05 (Play's
refusal order and its one shared function), sirens-03 (the WAV cache's byte
budget), and sirens-04 (the Name fields disabling while the song saves).
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest
from test_sirens_mode import FakeCtx, _audible, _tab

from warlock.studio.modes.sirens import play as sirens_play
from warlock.studio.modes.sirens.engine import document as D
from warlock.studio.modes.sirens.engine import synth, wsng
from warlock.studio.modes.sirens.ui.panes import effects as sirens_effects
from warlock.studio.modes.sirens.ui.panes import instruments as sirens_instruments

# --- sirens-01 ------------------------------------------------------------


def _song(rows: int = 16) -> D.SongDoc:
    doc = D.new_song()
    doc.resize_pattern(doc.patterns[0].uid, rows)
    return doc


def _put(doc: D.SongDoc, row: int, channel: int = 0, **columns: int) -> None:
    lookup = {
        "note": D.NOTE,
        "instrument": D.INSTRUMENT,
        "volume": D.VOLUME,
        "effect": D.EFFECT,
        "param": D.PARAM,
    }
    for name, value in columns.items():
        doc.set_cell(doc.patterns[0].uid, row, channel, lookup[name], value)


def _rms(pcm: np.ndarray) -> float:
    return float(np.sqrt((pcm.astype(np.float64) ** 2).mean())) if pcm.size else 0.0


def test_a_dangling_instrument_reference_is_silent_even_after_a_different_instrument_played_earlier_on_the_same_voice():  # noqa: E501
    """``_apply_row`` used to leave ``voice.instrument`` untouched when the
    cell's uid did not resolve, so a note whose instrument column named a
    deleted instrument kept sounding on whichever instrument the voice played
    last -- instead of the silence ``remove_instrument``'s docstring promises
    ("a pattern cell holding a uid nothing answers to plays silently"). The
    2026-09-15 audit, finding sirens-01.
    """
    doc = _song(rows=16)
    kind = doc.channels[0].kind
    first = next(one for one in doc.instruments if one.kind == kind)
    second = doc.add_instrument(kind=kind)
    _put(doc, 0, note=48, instrument=first.uid)
    _put(doc, 8, note=48, instrument=second.uid)
    doc.remove_instrument(second.uid)

    pcm, _loop = synth.render(doc)
    half = pcm.shape[0] // 2

    assert _rms(pcm[:half]) > 0.01
    assert _rms(pcm[half:]) == pytest.approx(0.0, abs=1e-6)


# --- sirens-02 / sirens-05 --------------------------------------------------


def test_play_after_a_failed_render_reports_the_render_error_not_an_empty_order_list(
    monkeypatch,
):
    """``play`` and ``_playable`` checked ``tab.pcm is None`` before
    ``tab.render_error``, and a failed render leaves ``pcm`` at ``None`` too --
    so a song that had an order list and had tried and failed to render was
    told there was nothing in its order list, while the transport pane right
    beside it read the real error. The 2026-09-15 audit, finding sirens-02.
    """
    played = _audible(monkeypatch)
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.render_dirty = False
    tab.pcm = None
    tab.render_error = "That song did not render: a made-up render failure."

    assert sirens_play.play(ctx, tab) is False
    assert played == []
    assert ctx.toasts[-1] == (tab.render_error, "error")


def test_play_and_play_from_caret_share_one_refusal_function():
    """``_playable``'s docstring already claimed to be the refusals ``play``
    and ``play_from_caret`` share; ``play`` restated the same three checks
    inline instead of calling it, so the two could disagree (as sirens-02
    shows they did). The 2026-09-15 audit, finding sirens-05.
    """
    source = inspect.getsource(sirens_play.play)
    assert "_playable(ctx, tab)" in source
    assert "sirens_audio.available()" not in source
    assert "render_dirty" not in source


# --- sirens-03 --------------------------------------------------------------


def test_wav_cache_evicts_on_total_bytes_not_just_entry_count(monkeypatch):
    """``_WAV_CACHE`` only evicted once the entry count reached 128, and one
    sample's ``pcm`` plus its encoded WAV can run to ~173 MB -- so the worst
    case was ~22 GB resident at once. Priced by bytes instead, the way
    ``undo.py`` prices an array, a handful of large entries evict long before
    the count cap ever would. The 2026-09-15 audit, finding sirens-03.
    """
    monkeypatch.setattr(wsng, "_WAV_CACHE", {})
    monkeypatch.setattr(wsng, "_WAV_CACHE_MAX", 1000)
    monkeypatch.setattr(wsng, "_WAV_CACHE_BUDGET", 10_000)

    arrays = [np.zeros(2000, dtype=np.int16) for _ in range(5)]
    for arr in arrays:
        wsng._wav_of(arr)

    total = sum(wsng._wav_cache_cost(*entry) for entry in wsng._WAV_CACHE.values())
    assert total <= wsng._WAV_CACHE_BUDGET
    assert len(wsng._WAV_CACHE) < len(arrays)


# --- sirens-04 --------------------------------------------------------------


@pytest.mark.parametrize(
    "pane,field_id",
    [
        (sirens_effects, "##sirens-fx-name"),
        (sirens_instruments, "##sirens-inst-name"),
    ],
    ids=["effects", "instruments"],
)
def test_the_effect_and_instrument_name_fields_disable_while_the_song_is_saving(
    pane, field_id
):
    """Both Name fields used ``widgets.input_text``, which has no ``enabled``,
    so they stayed live and kept pushing undo steps while ``tab.busy`` -- a
    rename typed mid-save landed on a document a save was still reading. The
    2026-09-15 audit, finding sirens-04: both now go through
    ``controls.input_text(enabled=editable, ...)``.
    """
    source = inspect.getsource(pane)
    match = re.search(
        r'controls\.input_text\(\s*"' + re.escape(field_id) + r'".*?\)',
        source,
        re.DOTALL,
    )
    assert match is not None, f"{field_id} is not drawn by controls.input_text"
    assert "enabled=editable" in match.group(0)
