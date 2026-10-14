"""The 2026-10-14 audit, Sirens engine and file door (fixer F45).

One file for the whole slice: the findings are all in the song document, its
``.rsng`` container, the renderer and the save door, and a test that reads
better beside its neighbours than in a file of its own.
"""

from __future__ import annotations

import io
import json
import struct
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

from realmspinner.core.safeio import zipguard
from realmspinner.kernels.audio import wavout
from realmspinner.studio import dialogs
from realmspinner.studio.modes.sirens import edit as sirens_edit
from realmspinner.studio.modes.sirens import mode as sirens_mode
from realmspinner.studio.modes.sirens.engine import document as D
from realmspinner.studio.modes.sirens.engine import instruments as inst
from realmspinner.studio.modes.sirens.engine import rsng, synth

from .test_rsng import _repack_all, _song
from .test_sirens_mode import FakeCtx, _tab

# --- sirens-25: a no-op undo must not re-arm the renderer ---------------------


def test_a_noop_undo_does_not_rearm_the_renderer():
    """Ctrl+Z on an untouched song set ``render_dirty`` unconditionally, so Play
    refused "Still rendering your latest edits" until a render nobody needed
    landed. Undo, redo and the history popover all armed it with the stack
    empty (or the step not moving)."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.render_dirty = False

    sirens_edit.undo(ctx, tab)
    assert not tab.render_dirty

    sirens_edit.redo(ctx, tab)
    assert not tab.render_dirty

    assert sirens_edit.step_history(ctx, tab, 0) is False
    assert not tab.render_dirty


def test_a_moving_undo_and_redo_still_rearm_the_renderer():
    ctx = FakeCtx()
    tab = _tab(ctx)
    uid = tab.doc.patterns[0].uid
    tab.doc.set_cell(uid, 0, 0, D.NOTE, 48)

    tab.render_dirty = False
    sirens_edit.undo(ctx, tab)
    assert tab.render_dirty

    tab.render_dirty = False
    sirens_edit.redo(ctx, tab)
    assert tab.render_dirty

    tab.render_dirty = False
    assert sirens_edit.step_history(ctx, tab, 0) is True
    assert tab.render_dirty


# --- sirens-04: the writer may not make what the reader refuses ---------------


def test_set_sample_refuses_what_read_rsng_would_refuse_to_open(monkeypatch):
    """``set_sample`` bounded the sample *count* and nothing bounded their total
    frames, so five Muse-length takes (each legal alone) made a song that saved
    and journalled and then could not be rendered, played, exported or
    reopened. The same sentence the reader uses, at the door."""
    monkeypatch.setattr(rsng, "MAX_DOC_SAMPLE_FRAMES", 3000, raising=False)
    doc = D.new_song()
    doc.set_sample("a", np.linspace(-1.0, 1.0, 2000, dtype=np.float32))
    with pytest.raises(ValueError, match="samples decode to more than"):
        doc.set_sample("b", np.linspace(-1.0, 1.0, 2000, dtype=np.float32))
    assert "b" not in doc.samples

    # What is left is a song the reader opens.
    assert rsng.read_rsng(rsng.rsng_bytes(doc)).samples.keys() == {"a"}


def test_set_sample_counts_a_replaced_sample_once_not_twice(monkeypatch):
    monkeypatch.setattr(rsng, "MAX_DOC_SAMPLE_FRAMES", 3000, raising=False)
    doc = D.new_song()
    doc.set_sample("a", np.linspace(-1.0, 1.0, 2000, dtype=np.float32))
    # 2800 in place of the 2000, not on top of it.
    assert doc.set_sample("a", np.linspace(-1.0, 1.0, 2800, dtype=np.float32))
    # Removing is always allowed, and so is a sample that fits what is left.
    assert doc.set_sample("a", None)
    assert doc.set_sample("b", np.linspace(-1.0, 1.0, 3000, dtype=np.float32))


# --- sirens-12: a sound effect's pattern ---------------------------------------


def test_remove_pattern_refuses_a_pattern_a_sound_effect_names():
    """``remove_pattern`` left a one-shot pointing at a pattern the song no
    longer held: the export raised "that pattern is not in this song" and
    ``read_rsng`` dropped the effect on the next open without a word."""
    doc = D.new_song()
    effect = doc.add_oneshot("jump")
    before = list(doc.patterns)
    with pytest.raises(ValueError, match="sound effect"):
        doc.remove_pattern(effect.pattern)
    assert doc.patterns == before
    assert synth.render_oneshot(doc, effect.uid).shape[1] == 2  # still renders


def test_update_oneshot_comment_describes_what_read_rsng_does():
    """The docstring said a file naming a missing pattern is one ``read_rsng``
    refuses; ``read_rsng`` drops the effect. Prose, so pinned by its words."""
    source = D.SongDoc.update_oneshot.__code__.co_filename
    text = Path(source).read_text(encoding="utf-8")
    assert "is one ``read_rsng`` refuses" not in text


# --- sirens-10: volume past four bits -----------------------------------------


def _held_note(volume: tuple[int, ...]) -> np.ndarray:
    doc = D.new_song()
    uid = doc.patterns[0].uid
    instrument = doc.instruments[0].uid
    doc.set_cell(uid, 0, 0, D.NOTE, 48)
    doc.set_cell(uid, 0, 0, D.INSTRUMENT, instrument)
    doc.update_instrument(instrument, volume=inst.Sequence(values=volume, loop=0))
    return synth.render(doc)[0]


def test_a_volume_step_past_fifteen_renders_no_louder_than_fifteen():
    """A volume value of 30 played at twice the level and 1000 hard-clipped,
    from a hand-edited or damaged file the envelope editor (fixed 0..15 span)
    cannot draw. ``MAX_VOLUME`` is documented as four bits."""
    full = _held_note((15,))
    assert np.abs(full).max() > 0.0
    assert np.array_equal(_held_note((30,)), full)
    assert np.array_equal(_held_note((1000,)), full)


def test_a_negative_volume_step_is_silent():
    assert not np.abs(_held_note((-5,))).any()


# --- sirens-01: a removed instrument ------------------------------------------


def test_a_song_with_a_removed_instrument_renders_the_same_after_a_save_and_open():
    """``read_rsng`` blanked every cell naming an instrument the file does not
    hold, but the renderer plays an unknown id as a *clear* while a blank cell
    keeps the voice's previous instrument -- so the snapshot every play, typed
    note, export and save is built from sounded the removed instrument's notes
    with a different instrument. ``remove_instrument`` promises silence."""
    doc = D.new_song()
    uid = doc.patterns[0].uid
    first = doc.instruments[0].uid
    second = doc.add_instrument("pulse", name="Gone").uid
    doc.set_cell(uid, 0, 0, D.NOTE, 48)
    doc.set_cell(uid, 0, 0, D.INSTRUMENT, first)
    doc.set_cell(uid, 8, 0, D.NOTE, 55)
    doc.set_cell(uid, 8, 0, D.INSTRUMENT, second)
    doc.remove_instrument(second)

    live, _ = synth.render(doc)
    snapshot, _ = synth.render(rsng.read_rsng(rsng.rsng_bytes(doc), reserve=False))
    assert np.array_equal(live, snapshot)

    # And the reference itself survives the round trip rather than being
    # stripped by a save.
    back = rsng.read_rsng(rsng.rsng_bytes(doc), reserve=False)
    assert back.patterns[0].cells[8, 0, D.INSTRUMENT] == second


# --- sirens-03: a member is bounded by what it can be --------------------------


def _with_member(doc: D.SongDoc, name: str, payload: bytes) -> bytes:
    raw = rsng.rsng_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        members = {one: zf.read(one) for one in zf.namelist()}
    members[name] = payload
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for one, data in members.items():
            zf.writestr(one, data)
    return out.getvalue()


def test_read_rsng_refuses_a_pattern_member_larger_than_any_pattern_can_be():
    """A 150 KiB ``.rsng`` whose pattern member was 150 MiB of padding opened
    (the bound was the generic 1 GiB archive ceiling, read whole before the
    member was known to be a pattern). No legal pattern is past
    ``MAX_ROWS * MAX_CHANNELS * COLUMNS * 2`` bytes plus a header."""
    doc = _song()
    with zipfile.ZipFile(io.BytesIO(rsng.rsng_bytes(doc))) as zf:
        good = zf.read("patterns/0.npy")
    ceiling = D.MAX_ROWS * D.MAX_CHANNELS * D.COLUMNS * 2 + 256
    raw = _with_member(doc, "patterns/0.npy", good + b"\x00" * (ceiling + 4096))
    with pytest.raises(ValueError, match="patterns/0.npy"):
        rsng.read_rsng(raw)


def test_read_rsng_refuses_a_manifest_past_the_manifest_ceiling(monkeypatch):
    monkeypatch.setattr(rsng, "MAX_MANIFEST_BYTES", 2_000, raising=False)
    doc = _song()
    with zipfile.ZipFile(io.BytesIO(rsng.rsng_bytes(doc))) as zf:
        manifest = zf.read(rsng.MANIFEST)
    raw = _with_member(doc, rsng.MANIFEST, manifest + b" " * 4_000)
    with pytest.raises(ValueError, match=rsng.MANIFEST):
        rsng.read_rsng(raw)


def test_read_rsng_refuses_a_sample_member_larger_than_any_sample_can_be(monkeypatch):
    monkeypatch.setattr(rsng, "MAX_SAMPLE_MEMBER_BYTES", 1_000, raising=False)
    with pytest.raises(ValueError, match="samples/0.wav"):
        rsng.read_rsng(rsng.rsng_bytes(_song()))  # the 600-frame kick is 1.2 KB


def test_read_wav_bounds_the_decoded_size_not_only_the_frame_count(monkeypatch):
    """A WAV of 8-bit 255-channel audio decodes to ~9x its own bytes, and the
    only door counted frames."""
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(255)
        handle.setsampwidth(1)
        handle.setframerate(44100)
        handle.writeframes(b"\x80" * 255 * 40)
    monkeypatch.setattr(wavout, "MAX_DECODED_VALUES", 1_000, raising=False)
    with pytest.raises(ValueError, match="channel"):
        wavout.read_wav(out.getvalue(), 44100)


# --- sirens-05: the wav cache and two long samples ----------------------------


def test_two_ten_minute_samples_do_not_defeat_the_wav_cache(monkeypatch):
    """The cache was priced by ``pcm.nbytes`` plus the WAV, so one ten-minute
    sample cost ~160 MiB and two overran the 256 MiB budget: each insert evicted
    the entry the next ``rsng_bytes`` needed, and a song with two long takes
    re-encoded both on every edit. The pcm is the document's own and is held
    there already; the cache is priced by what it adds, the WAV. Scaled down
    here (the arithmetic is the same at any size) and pinned at full size below.
    """
    monkeypatch.setattr(rsng, "_WAV_CACHE", {})
    first = np.zeros(10_000, dtype=np.float32)
    second = np.zeros(10_000, dtype=np.float32)
    wav = len(wavout.wav_bytes(first, synth.SAMPLE_RATE))
    monkeypatch.setattr(rsng, "_WAV_CACHE_BUDGET", 2 * wav + wav // 2)

    rsng._wav_of(first)
    rsng._wav_of(second)
    assert id(first) in rsng._WAV_CACHE and id(second) in rsng._WAV_CACHE


def test_the_wav_cache_budget_holds_a_documents_worth_of_long_samples():
    """The real figures: every sample a document may hold, encoded."""
    encoded = 2 * rsng.MAX_DOC_SAMPLE_FRAMES + 64 * 44
    assert encoded <= rsng._WAV_CACHE_BUDGET


# --- sirens-09: refusals are ValueError ---------------------------------------


def _patched_zip(raw: bytes, *, flags: int | None = None, method: int | None = None) -> bytes:
    """Every local and central header of *raw* with a flag word or method set."""
    data = bytearray(raw)
    for sig, flag_at, method_at in ((b"PK\x03\x04", 6, 8), (b"PK\x01\x02", 8, 10)):
        at = data.find(sig)
        while at != -1:
            if flags is not None:
                data[at + flag_at : at + flag_at + 2] = struct.pack("<H", flags)
            if method is not None:
                data[at + method_at : at + method_at + 2] = struct.pack("<H", method)
            at = data.find(sig, at + 4)
    return bytes(data)


def test_read_rsng_refuses_a_deeply_nested_manifest_by_name():
    """A manifest nested ~100k deep raised ``RecursionError`` out of
    ``json.loads``, past the ``ValueError`` every caller frames."""
    doc = _song()
    nested = b"[" * 100_000 + b"]" * 100_000
    with pytest.raises(ValueError, match="not a Realmspinner song"):
        rsng.read_rsng(_with_member(doc, rsng.MANIFEST, nested))


def test_read_rsng_refuses_an_encrypted_member_by_name():
    raw = _patched_zip(rsng.rsng_bytes(_song()), flags=1)
    with pytest.raises(ValueError, match="not a Realmspinner song"):
        rsng.read_rsng(raw)


def test_read_rsng_refuses_an_unknown_compression_method_by_name():
    raw = _patched_zip(rsng.rsng_bytes(_song()), method=99)
    with pytest.raises(ValueError, match="not a Realmspinner song"):
        rsng.read_rsng(raw)


def test_read_rsng_refuses_an_encrypted_pattern_member_by_name():
    """Not only the manifest: the pattern and sample reads carry the same
    translation."""
    doc = _song()
    raw = rsng.rsng_bytes(doc)
    # Mark only the pattern's headers: find the member by its name.
    data = bytearray(raw)
    for sig, flag_at, len_at, name_at in (
        (b"PK\x03\x04", 6, 26, 30),
        (b"PK\x01\x02", 8, 28, 46),
    ):
        at = data.find(sig)
        while at != -1:
            length = struct.unpack("<H", data[at + len_at : at + len_at + 2])[0]
            if bytes(data[at + name_at : at + name_at + length]).startswith(b"patterns/"):
                data[at + flag_at : at + flag_at + 2] = struct.pack("<H", 1)
            at = data.find(sig, at + 4)
    with pytest.raises(ValueError, match="not a Realmspinner song"):
        rsng.read_rsng(bytes(data))


# --- sirens-11: renumbering and the empty cell --------------------------------


def test_renumbering_never_assigns_an_instrument_to_an_empty_cell():
    """An instrument stored as uid -1 put ``{-1: 0}`` in the renumbering map,
    applied to the instrument plane without excluding EMPTY (-1): every empty
    cell of every pattern became instrument 0."""
    doc = _song()
    raw = rsng.rsng_bytes(doc)
    zf = zipguard.BoundedZip(io.BytesIO(raw))
    with zf:
        manifest = json.loads(zf.read(rsng.MANIFEST))
        patterns = rsng._patterns_from(zf, manifest, len(doc.channels), {-1: 0, 5: 1})
    plane = patterns[0].cells[:, :, D.INSTRUMENT]
    original = doc.patterns[0].cells[:, :, D.INSTRUMENT]
    assert (original == -1).any()
    assert (plane[original == -1] == -1).all()


def test_read_rsng_refuses_a_negative_instrument_uid_by_name():
    doc = _song()

    def edit(manifest, arrays):
        manifest["instruments"][0]["uid"] = -1

    with pytest.raises(ValueError, match="instrument"):
        rsng.read_rsng(_repack_all(doc, edit))


# --- sirens-13: a halt before the loop point ----------------------------------


def test_a_halt_before_the_loop_point_keeps_the_release_tail():
    """``tail_left`` was zeroed whenever ``loop_order`` named an order entry, but
    a ``Cxx`` that ended the song first meant no loop point ever existed: the
    render ended with no tail and no ``smpl`` chunk, cut off mid-note."""
    doc = D.new_song()
    uid = doc.patterns[0].uid
    doc.set_order([uid, uid])
    doc.set_cell(uid, 0, 0, D.NOTE, 48)
    doc.set_cell(uid, 0, 0, D.INSTRUMENT, doc.instruments[0].uid)
    doc.set_cell(uid, 2, 0, D.EFFECT, synth.FX_HALT)
    doc.set_cell(uid, 2, 0, D.PARAM, 0)

    doc.set_song(loop_order=-1)
    plain, plain_loop = synth.render(doc)
    doc.set_song(loop_order=1)
    halted, halted_loop = synth.render(doc)

    assert plain_loop is None and halted_loop is None
    assert halted.shape == plain.shape
    assert np.array_equal(halted, plain)


# --- sirens-08 / sirens-23: Save As -------------------------------------------


def test_save_as_refuses_a_path_another_open_song_already_holds(monkeypatch, tmp_path):
    """Two tabs bound to one ``.rsng`` overwrite each other's saves; Clay's
    Save As refuses it (clay-49) and ``open_path`` focuses rather than forks."""
    from realmspinner.service.errors import Conflict

    ctx = FakeCtx()
    other = _tab(ctx)
    taken = tmp_path / "taken.rsng"
    other.path = taken
    mine = _tab(ctx)
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: taken)

    with pytest.raises(Conflict) as raised:
        sirens_mode.save_as(ctx, mine)
    assert "taken.rsng" in str(raised.value)
    assert not taken.exists()


def test_save_as_appends_rsng_rather_than_replacing_a_dotted_tail(monkeypatch, tmp_path):
    """``with_suffix`` replaced everything after the last dot: "Song 1.5" was
    written as "Song 1.rsng", a name the picker never offered."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    picked = tmp_path / "Song 1.5"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: picked)

    sirens_mode.save_as(ctx, tab)

    assert ctx.result is not None
    assert (tmp_path / "Song 1.5.rsng").exists()
    assert not (tmp_path / "Song 1.rsng").exists()


def test_save_as_keeps_a_name_that_already_ends_in_rsng(monkeypatch, tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx)
    picked = tmp_path / "Song.RSNG"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: picked)

    sirens_mode.save_as(ctx, tab)

    assert ctx.result is not None
    assert ctx.result["path"].lower().endswith("song.rsng")
    assert not (tmp_path / "Song.RSNG.rsng").exists()
