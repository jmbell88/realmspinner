"""The loop finder, on material whose right answer is known.

A heuristic over real music cannot be asserted -- there is no ground truth for
"where does this piece repeat best". So every test here builds a signal whose
loop point is a *fact*: a tone that repeats exactly, a piece with a seam
deliberately placed, silence. What is being pinned is that the machinery finds
what is genuinely there and does not invent what is not.

The three score weights are unmeasured and say so in the source; nothing here
asserts a ranking that depends on their exact values, which is what keeps this
file from freezing a figure the module admits it guessed.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.muse import loops

RATE = 44100


def _tone(seconds: float, hz: float = 220.0, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _phrase(seconds: float, hz: float, rate: int = RATE) -> np.ndarray:
    """A tone with an onset, so the lead-in term has something to match on."""
    body = _tone(seconds, hz, rate)
    attack = np.minimum(np.arange(body.size, dtype=np.float32) / (rate * 0.02), 1.0)
    return (body * attack).astype(np.float32)


# --- features ----------------------------------------------------------------


def test_every_frames_feature_is_unit_length():
    """Per-frame normalisation is what makes the score about content.

    Level is not discarded -- it comes back as its own term, where it can be
    weighted on purpose rather than buried inside the spectral distance.
    """
    bands, levels = loops.features(_tone(2.0, rate=loops.ANALYSIS_RATE))
    assert bands.shape[1] == loops.BANDS
    assert np.allclose(np.linalg.norm(bands, axis=1), 1.0, atol=1e-4)
    assert levels.shape[0] == bands.shape[0]


def test_two_identical_passages_have_the_same_feature():
    one = _tone(1.0, 440.0, loops.ANALYSIS_RATE)
    bands, _ = loops.features(np.concatenate([one, one]))
    half = bands.shape[0] // 2
    # A frame in the first copy and the matching frame in the second.
    assert bands[half // 2] @ bands[half + half // 2] == pytest.approx(1.0, abs=1e-3)


def test_features_bands_span_40hz_to_nyquist_not_dc_to_second_highest_edge():
    """The 2026-09-08 audit, finding muse-01. An off-by-one in how ``_bands``'s
    boundary array was fed to ``np.add.reduceat`` made kept band 0 the
    DC/sub-bass segment ``[0, ~40 Hz)`` and silently dropped the true top
    band, ``[~4.5 kHz, Nyquist)`` -- the "sizzle" the module's own docstring
    says the analysis exists to judge seams on.

    Fails against the unfixed code: a tone inside the true top band splits
    across kept bands 21-22 (band 23 all but empty) instead of dominating
    band 23, and a tone inside the true first log-spaced band (~38-48 Hz)
    peaks in kept band 1, not band 0.
    """
    rate = loops.ANALYSIS_RATE
    bin_hz = rate / loops.FFT

    sizzle = _tone(2.0, 5000.0, rate)
    bands, _ = loops.features(sizzle)
    frame = bands[bands.shape[0] // 2]
    assert int(np.argmax(frame)) == loops.BANDS - 1
    assert frame[loops.BANDS - 1] > 0.9

    # Bin-aligned to FFT/ANALYSIS_RATE so Hanning leakage stays concentrated
    # on this bin: bin 8 is ~43 Hz, squarely inside the true first log-spaced
    # band (edges[0]..edges[1] of the 25-point geomspace, ~38-48 Hz).
    low = _tone(2.0, 8 * bin_hz, rate)
    bands, _ = loops.features(low)
    frame = bands[bands.shape[0] // 2]
    assert int(np.argmax(frame)) == 0


def test_a_loud_passage_and_a_quiet_one_of_the_same_material_match_on_content():
    """And differ on level, which is the split the two return values are for.

    The match is 0.99-something rather than exactly 1: ``log1p`` is not linear,
    so scaling a frame by 20 dB does not scale its band vector, it *compresses*
    it slightly -- normalising afterwards removes almost all of the level and
    not quite all. That residue is deliberate and small, and it errs the useful
    way: two passages that differ only in level still read as the same content,
    while the level term below carries the difference at full weight.
    """
    quiet = _tone(1.0, 440.0, loops.ANALYSIS_RATE)
    bands, levels = loops.features(np.concatenate([quiet, quiet * 0.1]))
    half = bands.shape[0] // 2
    assert bands[half // 2] @ bands[half + half // 2] > 0.99
    assert levels[half // 2] - levels[half + half // 2] == pytest.approx(20.0, abs=1.0)


# --- _refine ------------------------------------------------------------------


def test_refine_does_not_misalign_the_phase_when_the_target_window_clips_at_the_buffer_start(
    monkeypatch,
):
    """The 2026-09-16 audit, finding muse-engine-01.

    ``_refine`` builds ``target = mono[max(start - half, 0) : start + half]``
    and then assumes sample ``start`` always sits at offset ``half`` inside
    it -- true only when ``start >= half``. When ``start`` is inside the
    buffer's first ``half`` samples, the window clips at 0 and ``start``
    actually sits at offset ``start``, so the old ``end = lo + argmax(...) +
    half`` is off by ``half - start``.

    Isolated from the zero-crossing snap (monkeypatched to identity here, as
    the audit's own probe did) so only the correlation math is under test. A
    copy of ``target`` is planted at a known position ``k`` so the *true*
    match is known exactly, rather than inferred from a period.
    """
    monkeypatch.setattr(loops, "_snap", lambda mono, at, rate: at)

    rate = 2000
    half = int(loops.REFINE_MS * rate / 1000.0) // 2
    assert half == 100  # keeps the arithmetic below legible

    rng = np.random.RandomState(0)
    mono = (rng.standard_normal(1000) * 0.1).astype(np.float32)

    start = 20  # < half: the clipped case
    lo_target = max(start - half, 0)
    assert lo_target == 0, "start must be inside the window's clipped left edge"
    origin = start - lo_target  # true offset of ``start`` inside ``target``
    target = mono[lo_target : start + half]

    # Plant an exact copy of ``target`` at k, so mono[k + origin] is the one
    # sample that truly corresponds to mono[start].
    k = 500
    mono[k : k + target.size] = target
    true_end = k + origin

    _, end = loops._refine(mono, start, true_end, rate)
    assert end == true_end


# --- find --------------------------------------------------------------------


def test_a_take_shorter_than_a_second_has_no_loop_points():
    assert loops.find(_tone(0.5), RATE) == []


def test_silence_produces_no_crash_and_at_most_a_ranking():
    """Silence is the degenerate case: every moment matches every other, so any
    answer is as good as any other. What must not happen is an exception."""
    out = loops.find(np.zeros(RATE * 3, dtype=np.float32), RATE)
    assert all(c.end > c.start for c in out)


def test_every_candidate_is_ordered_and_inside_the_take():
    pcm = np.concatenate([_phrase(2.0, 220.0), _phrase(2.0, 330.0), _phrase(2.0, 220.0)])
    out = loops.find(pcm, RATE)
    assert out, "a three-phrase piece should offer at least one loop"
    for candidate in out:
        assert 0 <= candidate.start < candidate.end <= pcm.size
        assert candidate.frames == candidate.end - candidate.start


def test_it_returns_alternatives_rather_than_one_answer():
    """A single answer with no alternatives would claim a confidence this
    method does not have -- see the module docstring."""
    pcm = np.concatenate([_phrase(1.5, hz) for hz in (220, 330, 220, 440, 220)])
    out = loops.find(pcm, RATE)
    assert 1 < len(out) <= loops.TOP_N


def test_candidates_are_distinct_rather_than_five_readings_of_one_basin():
    """'Same basin' is ``_coarse``'s own dedup rule: both endpoints agree to
    within a lead-in. Two candidates may legitimately share a start (or an
    end) with the other endpoint far apart -- that is a different loop, not
    the same one twice -- so the check is on the pair, not on raw distinctness
    of starts alone.

    The 2026-09-08 audit, finding muse-01, corrected ``features()``'s band
    boundaries; the fix legitimately changes which candidates ``find()``
    returns for this material, including which ones share a start, so the
    raw-distinctness form of this assertion (checked before the fix) no
    longer holds for this input even though no basin is duplicated.
    """
    pcm = np.concatenate([_phrase(1.5, hz) for hz in (220, 330, 220, 440, 220)])
    out = loops.find(pcm, RATE)
    reach = loops.CONTEXT_FRAMES * loops.HOP * (RATE / loops.ANALYSIS_RATE)
    for i, a in enumerate(out):
        for b in out[i + 1 :]:
            assert abs(a.start - b.start) > reach or abs(a.end - b.end) > reach
    starts = [c.start for c in out]
    assert len(out) == 1 or max(starts) - min(starts) > reach


def test_a_loop_covers_a_real_fraction_of_the_take():
    """A two-second window of a four-minute piece is a sample, not a loop.

    The penalty is soft, so this asserts the effect rather than a hard floor:
    the best-ranked candidate is not a sliver.
    """
    pcm = np.concatenate([_phrase(2.0, hz) for hz in (220, 330, 440, 220)])
    out = loops.find(pcm, RATE)
    assert out
    assert out[0].frames > pcm.size * 0.2


# --- crossfade ---------------------------------------------------------------
#
# M08. The seam a repeated loop plays across is the body's *last* sample
# meeting its *first*, and that is the only number these tests measure. Two
# earlier versions each blended material in without measuring the wrap: the
# second ended the body on ``head[fade - 1]``, so playback wrapped to
# ``head[0]`` and jumped backwards by a whole fade every repeat.
#
# Every wrap test below uses *non-constant* material, deliberately. The old
# tests used a constant plateau, where ``head[fade - 1] == head[0]`` makes that
# defect structurally invisible -- which is why it shipped.


def _wrap(body: np.ndarray) -> tuple[float, float]:
    """``(seam step, largest interior step)`` of ``body`` played on repeat.

    The claim every test here asserts is the comparison of the two: a loop
    whose wrap is no bigger a step than the music's own biggest step has no
    seam to hear, whatever the absolute numbers are.
    """
    repeated = np.concatenate([body, body])
    n = body.shape[0]
    steps = np.abs(np.diff(repeated.astype(np.float64), axis=0))
    steps = steps.reshape(steps.shape[0], -1).sum(axis=1)
    seam = float(np.abs(repeated[n].astype(np.float64) - repeated[n - 1].astype(np.float64)).sum())
    return seam, float(steps[: n - 1].max())


def _int_tone(seconds: float, hz: float = 440.0, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    return (20000.0 * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def test_the_crossfade_is_equal_power_rather_than_linear():
    """The blend is exactly ``lead * cos + own * sin`` -- weights whose
    *squares* sum to 1, which is what keeps two decorrelated signals from
    dipping ~3 dB at mid-fade the way a linear (weights summing to 1) blend
    would. Asserted as the exact per-sample formula rather than a statistical
    approximation over random material, which is fragile by construction: any
    single sample's instantaneous value depends on both signals' signs, not
    only on their power.
    """
    pcm = np.arange(5000, dtype=np.float32)
    fade = 400
    # A ramp's cheapest join by far is to continue past ``end`` into the head,
    # so this is the post-end fade: the body starts on ``pcm[4000]``.
    body = loops.crossfade(pcm, 0, 4000, fade)
    angle = np.linspace(0.0, np.pi / 2.0, fade, dtype=np.float32)
    expected_head = pcm[4000 : 4000 + fade] * np.cos(angle) + pcm[:fade] * np.sin(angle)
    np.testing.assert_allclose(body[:fade], expected_head, rtol=1e-4)


def test_the_body_is_the_requested_span():
    pcm = _tone(3.0)
    assert loops.crossfade(pcm, 1000, 5000, 100).shape[0] == 4000


def test_the_repeat_seam_is_no_worse_than_the_musics_own_biggest_step():
    """M08, the headline claim, on the audit's own measurement: a 440 Hz tone,
    region ``[1000, 60000)``, 2048-sample fade. Against the unfixed code the
    seam steps 14502 with an interior maximum of 1320 -- eleven times the
    largest step the music itself contains, and worse than the 11695 that
    applying no fade at all would have left.
    """
    body = loops.crossfade(_int_tone(2.0), 1000, 60000, 2048)
    seam, interior = _wrap(body)
    assert seam <= interior


def test_a_region_starting_at_the_top_of_the_take_still_gets_a_seamless_wrap():
    """M08 with no material *before* ``start``: the join has to be found on the
    other side, by beginning the body on ``data[end]``. Fails against the
    unfixed code, whose seam steps 9238 against an interior maximum of 1401.
    """
    body = loops.crossfade(_int_tone(2.0), 0, 60000, 2048)
    seam, interior = _wrap(body)
    assert seam <= interior


def test_a_stereo_takes_wrap_is_seamless_in_both_channels_at_once():
    """M08 across channels. The two channels share one seam, so the join is
    costed on both together rather than chosen for one and inflicted on the
    other. Fails against the unfixed code, whose summed seam steps 42945
    against an interior maximum of 2579.
    """
    tone = _int_tone(2.0)
    pcm = np.stack([tone, np.roll(tone, 7)], axis=1)
    body = loops.crossfade(pcm, 1000, 60000, 2048)
    assert body.shape == (59000, 2)
    seam, interior = _wrap(body)
    assert seam <= interior


def test_a_body_shorter_than_twice_the_fade_still_wraps_cleanly():
    """Duration policy under M08: the fade is capped at half the body, so the
    faded region can never cover the body twice, and the join is still chosen
    by measurement. Fails against the unfixed code, whose seam steps 22404 on
    this fifty-sample body against an interior maximum of 1253.
    """
    body = loops.crossfade(_int_tone(2.0), 1000, 1050, 1000)
    assert body.shape[0] == 50
    seam, interior = _wrap(body)
    assert seam <= interior


def test_an_already_seamless_loop_is_left_exactly_alone():
    """M08's own repro, and the reason declining is a real branch. A constant
    plateau's untreated wrap is already a zero-sample step: no join can improve
    on that, so no fade is applied and the body is the plain slice. Fails
    against the unfixed code, which blends anyway and leaves a 157-unit step
    wobbling through the *interior* of a signal that was flat.
    """
    pcm = np.concatenate(
        [np.full(500, -20000, dtype=np.int16), np.full(2000, 20000, dtype=np.int16)]
    )
    body = loops.crossfade(pcm, 500, 2500, 200)
    assert np.array_equal(body, pcm[500:2500])
    assert _wrap(body) == (0.0, 0.0)


def test_a_one_sample_available_room_still_wraps_cleanly_or_declines_rather_than_silently_doing_nothing():  # noqa: E501
    """The 2026-09-13 audit, finding muse-02.

    ``np.linspace(0, pi/2, 1)`` is a single angle of 0, whose sin/cos are 0
    and 1 -- so a "tail" fade with only one sample of room blends in 100%
    the original sample and 0% the lead-out, a no-op. The cost model still
    believed that side would land the wrap on the join it costed, so it
    picked the cheapest-*looking* side (the one-sample tail) over a genuinely
    usable twenty-sample head candidate that would have worked.

    Fails against the unfixed code: it picks the one-sample tail (cost
    0.001) over the real head candidate (cost 0.1, twenty samples of room),
    applies no actual blend, and the wrap seam (5.0) lands far above the
    take's own biggest interior step (~0.26) -- the click the fade exists to
    remove is still there. Fixed, the one-sample candidate is excluded and
    the real head fade is used instead, landing the seam under the interior
    step.
    """
    n = 120
    pcm = np.zeros(n, dtype=np.float32)
    pcm[0] = 0.001  # lead-out: a hair from pcm[1], so "tail" costs ~0
    pcm[80:100] = np.linspace(0.0, 5.0, 20, dtype=np.float32)  # body's own ramp
    pcm[100:120] = 5.1  # lead-in: close to pcm[99], so "head" costs ~0.1
    start, end = 1, 100

    body = loops.crossfade(pcm, start, end, 2048)
    seam, interior = _wrap(body)
    assert seam <= interior


def test_a_region_with_no_material_on_either_side_declines_rather_than_inventing_one():
    """Both fades need material from outside the region; with none available
    there is nothing to measure a better join against, so the body is returned
    untouched rather than blended into itself. Fails against the unfixed code,
    which blends the head into the tail and returns a body that is not the
    slice.
    """
    pcm = np.concatenate([np.full(25, 1000, dtype=np.int16), np.full(25, -1000, dtype=np.int16)])
    body = loops.crossfade(pcm, 0, 50, 1000)
    assert np.array_equal(body, pcm)


def test_a_rotated_body_is_still_seamless_so_seeking_inside_the_region_keeps_the_loop():
    """M10 rotates the loop body with ``np.roll`` so playback can start at the
    seek point, which turns every interior join into a potential seam. A body
    that is wrap-continuous by construction survives that; the unfixed code's
    does not -- rotating it merely moves its 14502-step defect into the
    interior, where the seek lands on it.
    """
    body = loops.crossfade(_int_tone(2.0), 1000, 60000, 2048)
    seam, interior = _wrap(np.roll(body, -5000, axis=0))
    assert seam <= interior


def test_no_fade_is_the_plain_slice():
    pcm = _tone(1.0)
    assert np.array_equal(loops.crossfade(pcm, 100, 900, 0), pcm[100:900])


def test_an_integer_take_comes_back_in_its_own_dtype_and_in_range():
    pcm = np.concatenate([np.full(1000, 30000, dtype="<i2"), _int_tone(0.1)])
    body = loops.crossfade(pcm, 100, 1000, 200)
    assert body.dtype == np.dtype("<i2")
    assert body.max() <= np.iinfo("<i2").max
    assert body.min() >= np.iinfo("<i2").min
