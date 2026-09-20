"""The format a take is written in, and the reader that has to open it.

Muse wrote 48 kHz IEEE-float WAVs for its whole life, and *nothing caught it*:
``tests/modes/muse/test_muse_mode.py`` stubs ``import_sample`` out, so the bridge to Sirens
-- the headline pairing of the two audio modes -- was asserted against a mock
while the real one raised ``unknown format: 3`` on every take. This file is what
stops that coming back, and it is deliberately three cheap tests rather than one
expensive one: none of them needs weights, a card, or the ``music`` extra for
the scan.

The vendored change they pin is ``REALMSPINNER 5/6``; see
``pipelines/acestep/ATTRIBUTION.md`` for the argument, including why 16-bit
rather than a float branch in the tracker's reader.
"""

from __future__ import annotations

import io
import re
import wave
from pathlib import Path

import numpy as np
import pytest

from realmspinner.kernels.audio import wavout

_ACESTEP = Path(__file__).resolve().parents[2] / "src" / "realmspinner" / "pipelines" / "acestep"
_PIPELINE = _ACESTEP / "pipeline_ace_step.py"
_ATTRIBUTION = _ACESTEP / "ATTRIBUTION.md"


def test_the_two_call_sites_carry_the_format_kwargs():
    """A source scan, the ``winjob.assign`` precedent.

    The defect is two absent keyword arguments, so the regression is two absent
    keyword arguments -- and a re-vendoring drops them silently, because it is a
    file copy over the top rather than a merge. Scanning the source is the only
    check that survives someone re-pinning the model without reading
    ATTRIBUTION.md's "Updating" section.
    """
    source = _PIPELINE.read_text(encoding="utf-8")

    save = re.search(r"torchaudio\.save\((.*?)\n        \)", source, re.S)
    assert save is not None, "torchaudio.save call not found -- was the vendor copy reshaped?"
    assert 'encoding="PCM_S"' in save.group(1)
    assert "bits_per_sample=16" in save.group(1)

    call = re.search(r"self\.latents2audio\((.*?)\n        \)", source, re.S)
    assert call is not None, "the latents2audio call site was not found"
    assert "sample_rate=44100" in call.group(1)


def test_the_marker_count_matches_the_attribution_document():
    """``REALMSPINNER n/N`` and the document's numbered list must agree.

    They did not: the document said "Four ... `REALMSPINNER n/4`" while the source
    comments said ``n/3``, and the "Updating" section said "the three
    modifications". Drift in a file whose entire job is telling a future
    re-vendorer what to re-apply is the drift that costs a feature, so it is
    pinned rather than proofread.
    """
    sources = [_PIPELINE.read_text(encoding="utf-8")]
    sources.append((_ACESTEP / "__init__.py").read_text(encoding="utf-8"))
    markers = set()
    total = set()
    for text in sources:
        for n, of in re.findall(r"REALMSPINNER (\d+)/(\d+):", text):
            markers.add(int(n))
            total.add(int(of))

    assert len(total) == 1, f"the markers disagree on how many there are: {sorted(total)}"
    count = total.pop()
    assert markers == set(range(1, count + 1)), (
        f"markers {sorted(markers)} do not number 1..{count} -- one was added or"
        " removed without renumbering the rest"
    )

    doc = _ATTRIBUTION.read_text(encoding="utf-8")
    assert f"`REALMSPINNER n/{count}:`" in doc
    entries = re.findall(r"^(\d+)\. \*\*", doc, re.M)
    assert [int(e) for e in entries] == list(range(1, count + 1)), (
        f"ATTRIBUTION.md lists {entries} modifications but the source carries {count}"
    )


_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every non-vendored file the 2026-09-13 audit found citing an ACE-Step
#: ``REALMSPINNER n/N`` marker in prose. Not a repo-wide sweep: the marker string
#: is reused, with its own independent numbering, by other vendored packages
#: (BiRefNet's ``ATTRIBUTION.md``, ``pipelines/_workerio.py``'s own note) that
#: this finding never touched and this module does not own, so a blind
#: ``rglob`` over ``src/``/``tests/`` false-positives on those instead of
#: catching drift here. This list is exactly muse-04's file set (its own
#: files plus the two the orchestrator also saw), kept beside the finding
#: rather than discovered fresh each run.
_MUSE_PROSE_FILES = (
    "src/realmspinner/_q_music.py",
    "src/realmspinner/pipelines/audioout.py",
    "src/realmspinner/pipelines/separation_worker.py",
    "src/realmspinner/pipelines/_workerio.py",
    "src/realmspinner/studio/modes/muse/fileio.py",
    "src/realmspinner/studio/modes/muse/mode.py",
    "tests/pipelines/test_music_format.py",
    "tests/modes/muse/test_muse_bridge.py",
    "tests/test_q_music_tasks.py",
    "tests/test_music_gpu.py",
)


def test_realmspinner_marker_prose_outside_acestep_matches_attribution_count():
    """The 2026-09-13 audit, finding muse-04.

    ``test_the_marker_count_matches_the_attribution_document`` above only
    scans the vendored files themselves -- ``pipeline_ace_step.py`` and
    ``__init__.py`` -- so when ``ATTRIBUTION.md`` moved from five
    modifications to six, ``_MUSE_PROSE_FILES`` kept naming the fifth and
    fourth markers by their old five-of-five denominator in prose and
    nothing caught it. This pins every
    ``REALMSPINNER n/N`` mention in those files to the document's real
    modification count, so the next renumbering fails here instead of
    shipping stale prose again.
    """
    doc = _ATTRIBUTION.read_text(encoding="utf-8")
    entries = re.findall(r"^(\d+)\. \*\*", doc, re.M)
    count = len(entries)
    assert count > 0, "ATTRIBUTION.md's numbered modification list is empty"

    stale: list[str] = []
    for rel in _MUSE_PROSE_FILES:
        path = _REPO_ROOT / rel
        text = path.read_text(encoding="utf-8")
        for n, of in re.findall(r"REALMSPINNER (\d+)/(\d+)", text):
            if int(of) != count:
                stale.append(f"{rel}: REALMSPINNER {n}/{of}")

    assert not stale, (
        "prose references a stale modification count -- ATTRIBUTION.md now "
        f"has {count}:\n" + "\n".join(stale)
    )


def _saved_bytes(tmp_path: Path) -> bytes:
    """Write 0.1 s of tone through the vendored writer. -> the file's bytes.

    ``save_wav_file`` reads no model state -- it is a path calculation and a
    ``torchaudio.save`` -- so it is called unbound on ``None`` rather than
    through a pipeline nobody wants to construct for a format assertion.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    module = pytest.importorskip("realmspinner.pipelines.acestep.pipeline_ace_step")

    t = torch.linspace(0.0, 0.1, 4410)
    wave_ = torch.stack([torch.sin(t * 440.0), torch.sin(t * 660.0)]) * 0.5
    out = tmp_path / "track.wav"
    module.ACEStepPipeline.save_wav_file(
        None, wave_, 0, save_path=str(out), sample_rate=44100
    )
    return out.read_bytes()


def test_a_written_take_is_44100_hz_16_bit_stereo(tmp_path):
    with wave.open(io.BytesIO(_saved_bytes(tmp_path))) as handle:
        assert handle.getframerate() == 44100
        assert handle.getsampwidth() == 2
        assert handle.getnchannels() == 2


# -- pipelines/audioout: the re-encodings a finished take can become --------
#
# A plain ``wave`` write rather than ``_saved_bytes``'s vendored pipeline: the
# module docstring's whole point is that these three are cheap, and pulling in
# torch/torchaudio for a format-conversion test would undo that for no reason
# -- ``pipelines/audioout`` never imports either.


def _silent_wav(tmp_path, seconds=0.05, rate=44100) -> Path:
    import wave

    path = tmp_path / "track.wav"
    frames = np.zeros((int(seconds * rate), 2), dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())
    return path


def test_track_wav_is_a_byte_identical_copy_not_a_re_encode(tmp_path, monkeypatch):
    # Decoding 16-bit PCM to float32 and writing it back as 16-bit PCM is a
    # second quantisation for no change a listener would ever hear -- see
    # audioout.convert's docstring.
    #
    # **A byte comparison alone does not pin this**, which was measured rather
    # than assumed: a 16-bit round trip through float32 is lossless, so the
    # bytes match whether the WAV was copied or re-encoded, and this test
    # passed unchanged against a deliberately re-encoding version. Both halves
    # are asserted -- the bytes, because that is the property the user has, and
    # that soundfile is never reached, because that is the one the name claims.
    import soundfile as sf

    from realmspinner.pipelines import audioout

    source = _silent_wav(tmp_path)
    out = tmp_path / "out.wav"

    def refuse(*a, **k):  # pragma: no cover - the point is that it never runs
        raise AssertionError("track.wav went through soundfile instead of a copy")

    monkeypatch.setattr(sf, "write", refuse)
    audioout.convert(source, out, "track.wav")
    assert out.read_bytes() == source.read_bytes()


def test_track_aiff_decodes_back_to_the_same_samples(tmp_path):
    sf = pytest.importorskip("soundfile")
    from realmspinner.pipelines import audioout

    source = _silent_wav(tmp_path)
    out = tmp_path / "track.aiff"
    audioout.convert(source, out, "track.aiff")

    wav_data, wav_rate = sf.read(str(source), dtype="int16", always_2d=True)
    aiff_data, aiff_rate = sf.read(str(out), dtype="int16", always_2d=True)
    assert aiff_rate == wav_rate
    assert (aiff_data == wav_data).all()


def test_convert_refuses_a_name_outside_its_format_table(tmp_path):
    from realmspinner.pipelines import audioout

    source = _silent_wav(tmp_path)
    with pytest.raises(ValueError):
        audioout.convert(source, tmp_path / "out.opus", "track.opus")


def test_the_tracker_can_read_a_written_take(tmp_path):
    """The assertion "Open in Sirens" needs and nothing else made.

    ``wavout.read_wav`` is the reader behind ``sirens_io.import_sample``, which
    is what the button calls. Feeding it the writer's own bytes is the whole
    bridge, minus the file dialog.
    """
    mono = wavout.read_wav(_saved_bytes(tmp_path), 44100)
    assert mono.dtype == np.float32
    # The render rate asked for is the take's own, so no resampling happens and
    # the frame count survives -- which is the second half of the fix: at the
    # old 48 kHz this would have been a resample even had the width been right.
    assert len(mono) == 4410
    assert np.abs(mono).max() <= 1.0
