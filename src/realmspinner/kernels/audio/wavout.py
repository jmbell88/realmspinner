"""16-bit PCM WAV, written by hand so it can carry loop points.

**Why not ``wave`` from the standard library.** It writes ``fmt `` and ``data``
and nothing else, and the one thing a game soundtrack needs a WAV to carry is
where it loops. That lives in the ``smpl`` chunk -- the chunk Unity, Godot,
FMOD, Wwise and every sampler read -- and a format that cannot express the loop
pushes the answer into a sidecar JSON the engine does not know to look for. So
the RIFF is assembled here, in about forty lines, and ``wave`` is used for
*reading* only, where its chunk skipping is exactly what is wanted.

**The bytes are a pure function of the samples.** No timestamps, no writer
string, no padding that depends on anything but the length -- so two exports of
an unchanged document are byte-identical, which is the rule ``.rpack``,
``.rmap`` and ``.rblk`` already follow and is what makes a content hash mean
something.
"""

from __future__ import annotations

import io
import struct
import wave

import numpy as np

#: Loop type 0 in the ``smpl`` chunk: forward. The other two (alternating and
#: backward) are sampler behaviours no game engine implements, and writing one
#: would produce a file that loops differently depending on who reads it.
LOOP_FORWARD = 0

#: The ceiling on a sample file this build will decode: the longest thing this
#: build will treat as a sample. It is sized by what Muse can hand it --
#: ``service._jobs_music.MAX_DURATION`` is ten minutes, and the Open-in-Sirens
#: bridge lands a take through this same decoder, so a ceiling below that made
#: the bridge work at Muse's default length and nowhere else. The two
#: constants move together.
#:
#: The cost is real and worth stating, not a footnote: ten minutes of mono
#: ``float32`` at the 44.1 kHz render rate is about 106 MB in the document, and
#: a ``SampleEdit`` copies both ends, so one such import charges roughly 212 MB
#: against ``undo.UNDO_BYTES`` (192 MiB soft, ``UNDO_HARD_BYTES`` 768 MiB
#: hard). That is now *over* the soft budget and under the hard ceiling, so it
#: is kept as the one step the stack guarantees while evicting everything else
#: -- ``undo.py`` states that contract as "one step, or under the ceiling,
#: never both."
MAX_SAMPLE_FRAMES = 48_000 * 600


class SampleTooLong(ValueError):
    """A sample's frame count is past the limit the caller allowed.

    A ``ValueError`` so every caller that already frames one still does; its own
    type so a reader that handed :func:`read_wav` a *smaller* limit than
    :data:`MAX_SAMPLE_FRAMES` (``rsng``'s running document total) can tell this
    refusal from a malformed file and name the right ceiling.
    """


def to_int16(samples: np.ndarray) -> np.ndarray:
    """Float samples in ``[-1, 1]`` as ``int16``.

    Clipped before scaling, and scaled by 32767 rather than 32768: the latter
    turns a full-scale ``-1.0`` into ``-32768`` and a full-scale ``+1.0`` into a
    value that does not exist, so the positive peak wraps. Asymmetry in the
    integer range is not a reason to produce an asymmetric waveform.
    """
    return np.round(np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0) * 32767.0).astype(
        np.int16
    )


def _chunk(tag: bytes, payload: bytes) -> bytes:
    """One RIFF chunk, padded to an even length as the format requires."""
    pad = b"\x00" if len(payload) % 2 else b""
    return tag + struct.pack("<I", len(payload)) + payload + pad


def _smpl(rate: int, loop: tuple[int, int]) -> bytes:
    """The loop chunk. One forward loop, which is what a game track has.

    ``end`` is the last sample *inside* the loop, not one past it -- the RIFF
    spec is inclusive here and an exclusive value produces a loop one sample
    long at the seam, which is audible as a click on every repeat.
    """
    start, end = int(loop[0]), max(int(loop[0]), int(loop[1]) - 1)
    header = struct.pack(
        "<9I",
        0,  # manufacturer: none
        0,  # product: none
        int(round(1_000_000_000 / rate)),  # sample period, nanoseconds
        60,  # MIDI unity note: middle C, the conventional filler
        0,  # pitch fraction
        0,  # SMPTE format
        0,  # SMPTE offset
        1,  # one loop
        0,  # no sampler-specific data
    )
    body = struct.pack("<6I", 0, LOOP_FORWARD, start, end, 0, 0)
    return header + body


def wav_bytes(
    samples: np.ndarray, rate: int, *, loop: tuple[int, int] | None = None
) -> bytes:
    """Stereo or mono float samples as a complete WAV file.

    ``samples`` is ``(n,)`` or ``(n, channels)``.
    """
    data = np.asarray(samples)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError("samples are (n,) or (n, channels)")
    frames, channels = data.shape
    # int16 is already the file's own format: passed through untouched rather
    # than divided and re-quantised, which cannot hold ``-32768`` (muse-14,
    # 2026-10-03 audit: ``to_int16`` clips to [-1, 1] and scales by 32767, so a
    # full-scale negative sample was written one LSB short).
    pcm = (data if data.dtype == np.int16 else to_int16(data)).astype("<i2").tobytes()
    fmt = struct.pack(
        "<HHIIHH",
        1,  # PCM
        channels,
        int(rate),
        int(rate) * channels * 2,  # byte rate
        channels * 2,  # block align
        16,  # bits per sample
    )
    body = _chunk(b"fmt ", fmt)
    if loop is not None and frames:
        body += _chunk(b"smpl", _smpl(int(rate), loop))
    body += _chunk(b"data", pcm)
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body


def write(path, samples: np.ndarray, rate: int, *, loop: tuple[int, int] | None = None) -> None:
    """Straight to a file. **Not staged** -- the caller stages it.

    Deliberately not doing the temp-and-replace here: the mode above writes
    every artefact through ``core/safeio/atomic.py``, and a second staging rule
    inside the encoder would be a second answer to where a partial file can
    appear.
    """
    with open(path, "wb") as handle:
        handle.write(wav_bytes(samples, rate, loop=loop))


def read_wav(data: bytes, rate: int, *, max_frames: int = MAX_SAMPLE_FRAMES) -> np.ndarray:
    """A WAV file's bytes as mono ``float32`` at ``rate``. For sample import.

    Three conversions, each of which the caller would otherwise have to know
    about: **mono**, because a chip voice has one waveform and a stereo sample
    played through one is two samples fighting; **float32 in [-1, 1]**, because
    that is what :func:`~.voices.sampled` interpolates; and **resampled to the
    render rate**, because :mod:`.synth` advances a sample's phase in output
    samples and a 22 kHz source would otherwise play an octave out.

    Linear resampling, matching :func:`~.voices.sampled`'s own interpolation --
    a sample is a drum hit or a bass note, and the decimation filter over the
    mix sits downstream of both.

    ``max_frames`` is the most this one decode may produce, at most
    :data:`MAX_SAMPLE_FRAMES`; a caller summing several samples against a
    document-wide ceiling passes what is left of it (the 2026-10-03 audit,
    finding sirens-06). Past it the refusal is a :class:`SampleTooLong`, raised
    before the array that would pass it is allocated.
    """
    max_frames = min(int(max_frames), MAX_SAMPLE_FRAMES)
    try:
        with wave.open(io.BytesIO(data)) as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            source_rate = handle.getframerate()
            frames = handle.getnframes()
            if frames > max_frames:
                raise SampleTooLong(
                    f"this sample is {frames} frames, past the"
                    f" {max_frames} this build will load"
                )
            # The 2026-09-07 audit found sirens-01: a WAV declaring a 0 Hz rate
            # sailed through here and hit the resample branch's division by
            # ``source_rate`` below, raising an uncaught ``ZeroDivisionError``
            # instead of the ``ValueError`` both call sites (sample import,
            # the Open-in-Sirens bridge) catch and reframe -- so an otherwise
            # intact song failed to open with no field to point at.
            if source_rate <= 0:
                raise ValueError(f"this WAV declares a {source_rate} Hz rate")
            raw = handle.readframes(frames)
            # The 2026-09-26 audit, finding sirens-engine-03: a data chunk
            # shorter than the header's own frame count is not an error to
            # ``wave`` -- ``Chunk.read`` returns whatever bytes are actually
            # there rather than raising -- so a cut file sailed past this
            # point with a ``raw`` too short for ``frames``, then hit
            # ``reshape``/``frombuffer`` below with numpy's own message
            # ("cannot reshape array of size ...") instead of one that says
            # what was actually wrong with the file. Checked here, once,
            # before any array is built.
            expected = frames * channels * width
            if len(raw) != expected:
                raise ValueError(
                    f"this WAV's data chunk holds {len(raw)} bytes, short of"
                    f" the {expected} its header declares for {frames} frames"
                )
    except (wave.Error, EOFError, struct.error) as exc:
        # The 2026-09-26 audit, finding sirens-engine-03: ``wave`` raises a
        # bare ``EOFError`` (from ``_read_fmt_chunk``, e.g. a ``fmt `` chunk
        # truncated before its 14 required bytes) or a bare ``struct.error``
        # for a header cut in other ways -- neither is a ``wave.Error`` and
        # neither is a ``ValueError``, so both used to sail past this
        # function's only ``except`` and past the ``except ValueError`` every
        # caller (sample import, song open, the Open-in-Sirens bridge) wraps
        # it in, reaching the user as "Something went wrong; see the log for
        # details" with no field pointing at the file that failed.
        detail = str(exc)
        suffix = f": {detail}" if detail else ""
        raise ValueError(f"this is not a WAV file this build reads{suffix}") from exc

    # **Scaled by the positive peak, not by the negative one.** The conventional
    # reading divides a 16-bit sample by 32768, which makes this the inverse of
    # nothing: :func:`to_int16` multiplies by 32767, so a sample written and
    # read back would come out a hair quiet -- and a document saved, opened and
    # saved again would produce different bytes each time, which is exactly the
    # byte-identity ``.rsng`` depends on. Dividing by the same number the writer
    # multiplies by makes the pair exact. The one value it cannot represent is
    # a full-scale negative, which lands at -1.00003 and is clipped.
    if width == 1:
        # 8-bit WAV is unsigned, alone among the widths. Every other one is
        # two's complement, which is why this case exists at all.
        mono = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 127.0
    elif width == 2:
        mono = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    elif width == 4:
        mono = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483647.0
    elif width == 3:
        # 24-bit has no numpy dtype. Sign-extended through the top byte of a
        # 32-bit view rather than by hand: the bytes are already little-endian,
        # and the arithmetic shift back down carries the sign.
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        wide = np.zeros((packed.shape[0], 4), dtype=np.uint8)
        wide[:, 1:] = packed
        mono = (wide.view("<i4").ravel() >> 8).astype(np.float32) / 8388607.0
    else:
        raise ValueError(f"a {width * 8}-bit WAV is not one this build reads")
    np.clip(mono, -1.0, 1.0, out=mono)

    if channels > 1:
        mono = mono.reshape(-1, channels).mean(axis=1)
    if source_rate != rate and mono.size:
        count = max(1, int(round(mono.size * rate / source_rate)))
        # The 2026-09-08 audit found sirens-01: the only size door above
        # checks the *header's* declared frame count against
        # ``MAX_SAMPLE_FRAMES``, but a low source rate scales the decoded
        # duration up by ``rate / source_rate`` -- a WAV that declares a low
        # enough rate passed both that door and the byte-size door in
        # ``sirens_io._sample_ceiling`` (sized on the same frame count) while
        # projecting to an array many times the ~10-minute budget, allocated
        # here by ``np.interp`` before anything else could refuse it.
        if count > max_frames:
            raise SampleTooLong(
                f"this sample would decode to {count} frames at {rate} Hz,"
                f" past the {max_frames} this build will load"
            )
        mono = np.interp(
            np.arange(count, dtype=np.float64) * (source_rate / rate),
            np.arange(mono.size, dtype=np.float64),
            mono.astype(np.float64),
        ).astype(np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32)
