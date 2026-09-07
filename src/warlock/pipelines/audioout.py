"""Re-encoding a finished take into the formats an engine actually imports.

**No new dependency, which is the opposite of the obvious assumption.** The
``soundfile`` already in the core dependencies is libsndfile 1.2.2, which
encodes WAV, FLAC, AIFF, MP3 (LAME) and OGG (Vorbis) on its own. No ffmpeg, no
``lameenc``, no second binary to ship or to sign. It is worth saying plainly,
because every plan for this reaches for a converter first.

**Deliberately not ``pipeline_ace_step.save_wav_file``**, which dispatches on a
``format`` string and would look like the reuse. It selects ``backend="sox"``
for ogg, and ``torchaudio.list_audio_backends()`` here returns ``['soundfile']``
alone -- sox is not built on Windows. So the worker keeps ``format="wav"`` and
every other format is a *derived artifact* of ``track.wav``, which is also the
shape ``files.MEDIA`` requires: that allowlist is literal filenames, so a
per-job format would be a per-job artifact name, which is exactly what an
allowlist exists to prevent.

**No staleness rule, and that is stated rather than omitted.** ``input.png`` has
three writers, which is ``files.fresh_2d``'s whole reason for existing;
``track.wav`` has one and nothing rewrites it after the run. So existence is the
freshness test here, as it is for the mesh exports.

No torch: this module is imported in the app process, on a path that has no
reason to pay for it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

#: Which libsndfile format and subtype each re-encoded name is written as.
#:
#: The subtypes are chosen, not defaulted. FLAC and AIFF at ``PCM_16`` match
#: what ``WARLOCK 5/5`` writes, so either is *lossless with respect to the
#: file it came from* rather than lossless with respect to a re-quantisation.
#: MP3 and Vorbis take libsndfile's own VBR default, and there is deliberately
#: no bitrate knob: one would cost a Config field, a SETTINGS row and the
#: bidirectional test that pairs them, and no measurement says the default is
#: insufficient for what these are for.
#:
#: ``track.wav`` is deliberately absent from this table -- see :func:`convert`,
#: which handles it as a plain file copy rather than a libsndfile round trip.
#: Opus is absent too, and that is not an oversight: libsndfile refuses it
#: outright at a take's 44.1 kHz (it accepts only 8/12/16/24/48 kHz for Opus),
#: so offering it here would need a resample smuggled into what is supposed to
#: be a format change and nothing else -- out of scope until something asks for
#: 48 kHz takes.
FORMATS: dict[str, tuple[str, str]] = {
    "track.flac": ("FLAC", "PCM_16"),
    "track.mp3": ("MP3", "MPEG_LAYER_III"),
    "track.ogg": ("OGG", "VORBIS"),
    "track.aiff": ("AIFF", "PCM_16"),
}


def convert(source: Path, out: Path, name: str) -> None:
    """Re-encode ``source`` into ``name``'s format, writing to ``out``. Blocking.

    ``name`` is passed rather than read off ``out``, and that is not redundancy:
    every derivation here is staged through ``.{name}.tmp``, so the path being
    written to is called ``.track.flac.tmp`` and has no format in its suffix at
    all. Dispatching on the artifact name keeps the choice on the allowlisted
    string rather than on a filename this function was handed.

    Read and written at the file's own rate and channel count: this is a format
    change and nothing else, so resampling or downmixing here would be a second,
    undeclared transformation riding along inside an export.

    ``track.wav`` -- the one name here that is not in :data:`FORMATS` -- is a
    plain :func:`shutil.copyfile` rather than a decode-and-re-encode through
    libsndfile. Decoding 16-bit PCM to float32 and writing it back as 16-bit
    PCM is not lossless *in practice*: it is a second quantisation of a signal
    already at its final bit depth, for no change in the bytes a listener
    would ever get. It is slower for the same reason -- two format
    conversions where a copy does the one job asked for. ``track.wav`` earns a
    place in :data:`~warlock.service.files.DERIVED_AUDIO` anyway (rather than
    being served only through its own, separate ``ready`` branch) so the
    Library's Convert door and the Downloads grid can treat it exactly like
    every other row in the format list, with one lock and one code path,
    instead of a WAV-shaped special case in each of them.
    """
    if name == "track.wav":
        shutil.copyfile(source, out)
        return
    import soundfile as sf

    if name not in FORMATS:
        raise ValueError(f"{name} is not a format this build writes")
    fmt, subtype = FORMATS[name]
    data, rate = sf.read(str(source), dtype="float32", always_2d=True)
    sf.write(str(out), data, int(rate), format=fmt, subtype=subtype)


__all__ = ["FORMATS", "convert"]
