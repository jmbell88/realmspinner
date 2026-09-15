"""``_q_music``'s two pure helpers: what the sampler is told, and the roll.

Both are module level and pure precisely so they can be asserted here -- with
no client, no card and no queue -- which is the reason ``_music_needs_handoff``
gives for being a function rather than an inlined expression.
"""

from __future__ import annotations

import io
import threading
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from warlock import _q_music as q
from warlock import models, packs


def _dir() -> Path:
    return Path("C:/jobs/abc")


class _FakeWorker:
    """Just enough of ``Worker`` for ``MusicOps._get_music_client``.

    Not a client, a card or a queue either -- the point of muse-03's fix is
    that the pack is probed before any of those exist.
    """

    _music_client = None


# --- _get_music_client ---------------------------------------------------


async def test_missing_music_extra_reports_pack_guidance_not_a_child_traceback(
    monkeypatch,
):
    """The 2026-09-13 audit, finding muse-03.

    ``music_client.py`` imports no torch, so the old ``except ImportError``
    around its import could never fire on a host missing the ``music`` extra
    -- the import only fails *inside the spawned child*, which surfaced as a
    raw ``ChildFailed("the music worker exited during startup")`` instead of
    the pack guidance. Failing this test against the unfixed code means the
    old code raised no ``RuntimeError`` at all here (nothing probed the pack
    before constructing the client), because ``MusicClient`` never even
    imports torch to fail on.
    """
    monkeypatch.setattr(packs, "installed", lambda pack: False)
    spec = models.MUSIC_MODELS[models.DEFAULT_MUSIC_MODEL]

    with pytest.raises(RuntimeError) as excinfo:
        await q.MusicOps._get_music_client(_FakeWorker(), spec)

    message = str(excinfo.value)
    assert "Music generation pack" in message
    assert "Settings -> Packs" in message
    assert "ChildFailed" not in message
    assert "exited during startup" not in message


# --- _task_kwargs ------------------------------------------------------------


def test_a_row_with_no_task_sends_nothing_at_all():
    """The guarantee, not a convenience.

    Every music row minted before tasks existed must take a byte-identical path
    through ``client.generate``, and an empty dict is what makes that true by
    construction rather than by inspection of the call site.
    """
    assert q._task_kwargs({"duration": 60.0}, _dir()) == {}
    assert q._task_kwargs({"task": ""}, _dir()) == {}


def test_a_stored_row_naming_an_unknown_task_is_refused():
    """``_music``'s rule for an unknown model, on the other string it dispatches
    on: silently downgrading to text2music would record a recipe never run."""
    with pytest.raises(RuntimeError, match="unknown music task"):
        q._task_kwargs({"task": "remix"}, _dir())


def test_audio2audio_never_sends_a_src_audio_path():
    """``__call__`` asserts that path implies repaint/edit/extend.

    Sending both would trip an assertion *inside the child*, with the weights
    already resident -- which is the failure the door and this table exist to
    move forward. The reference travels as ``ref_audio_input`` instead.
    """
    out = q._task_kwargs(
        {"task": "audio2audio", "ref_audio_strength": 0.3}, _dir()
    )
    assert out["task"] == "text2music"
    assert out["audio2audio_enable"] is True
    assert out["ref_audio_input"].endswith("source.wav")
    assert out["ref_audio_strength"] == pytest.approx(0.3)
    assert "src_audio_path" not in out


def test_a_retake_sends_no_source_because_it_re_runs_from_the_seed():
    """Amended for the 2026-09-07 audit, finding muse-01: the old assertion
    (``out == {"task": "retake", "retake_variance": 0.2}``) pinned the very
    bug the finding reports -- no ``retake_seed`` in the input meant nothing to
    forward, which this case still covers, but the identical dict shape also
    passed when a stored ``retake_seed`` was silently dropped. This case is
    now only about the no-seed row (one minted before ``retake_seed`` existed,
    or a task_kwargs call with none supplied); the seed being forwarded is
    ``test_a_retake_forwards_its_stored_retake_seed_to_the_sampler`` below.
    """
    out = q._task_kwargs({"task": "retake", "retake_variance": 0.2}, _dir())
    assert out == {"task": "retake", "retake_variance": 0.2}
    assert "retake_seeds" not in out


def test_a_retake_forwards_its_stored_retake_seed_to_the_sampler():
    """the 2026-09-07 audit, finding muse-01: ``derive_music_job`` draws a
    fresh ``retake_seed`` for every retake row (the seed-walk INVARIANTS.md's
    Muse paragraph describes), but the retake branch here sent only
    ``retake_variance`` -- so at the default variance roughly 70% of the
    blended noise was unseeded and no retake could ever be reproduced."""
    out = q._task_kwargs(
        {"task": "retake", "retake_variance": 0.2, "retake_seed": 777}, _dir()
    )
    assert out == {
        "task": "retake",
        "retake_variance": 0.2,
        "retake_seeds": [777],
    }


def test_an_extend_is_encoded_as_a_negative_repaint_window():
    """Upstream's spelling: the head pad runs from -left to 0 and the tail from
    the parent's duration to duration+right."""
    out = q._task_kwargs(
        {
            "task": "extend",
            "extend_left": 5.0,
            "extend_right": 10.0,
            "parent_duration": 60.0,
        },
        _dir(),
    )
    assert out["task"] == "extend"
    assert out["repaint_start"] == pytest.approx(-5.0)
    assert out["repaint_end"] == pytest.approx(70.0)
    assert out["src_audio_path"].endswith("source.wav")


def test_a_loop_is_sent_as_a_repaint_under_muses_own_name():
    out = q._task_kwargs(
        {"task": "loop", "repaint_start": 26.0, "repaint_end": 34.0}, _dir()
    )
    assert out["task"] == "repaint"
    assert out["repaint_start"] == pytest.approx(26.0)
    assert out["repaint_end"] == pytest.approx(34.0)


def test_an_edit_swaps_only_the_target_conditioning():
    out = q._task_kwargs(
        {"task": "edit", "edit_prompt": "bright strings", "edit_lyrics": "la"},
        _dir(),
    )
    assert out["edit_target_prompt"] == "bright strings"
    assert out["edit_target_lyrics"] == "la"
    assert out["src_audio_path"].endswith("source.wav")


# --- _roll_wav ---------------------------------------------------------------


def _wav(frames: np.ndarray, rate: int = 44100) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(frames.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.astype("<i2").tobytes())
    return out.getvalue()


def _frames(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as handle:
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").reshape(-1, channels)


def test_a_roll_and_its_inverse_return_the_identical_bytes():
    """Lossless is the whole argument for the rolled repaint.

    If the roll cost anything, a loop job would degrade the take it was trying
    to join up -- and the joint the model wrote would sit inside a piece of
    music that was no longer the one the user chose.
    """
    frames = np.arange(44100 * 2, dtype="<i2").reshape(-1, 2)
    original = _wav(frames)
    rolled = q._roll_wav(original, 0.25)
    assert rolled != original
    assert q._roll_wav(rolled, -0.25) == original


def test_the_roll_moves_the_joint_to_the_middle():
    rate = 1000
    frames = np.arange(2000, dtype="<i2").reshape(-1, 2)  # 1000 frames, 1 s
    rolled = _frames(q._roll_wav(_wav(frames, rate), 0.5))
    # What was frame 500 is now frame 0, so what was the head/tail joint --
    # the wrap from frame 999 to frame 0 -- now sits at frame 500.
    assert rolled[0].tolist() == frames[500].tolist()
    assert rolled[500].tolist() == frames[0].tolist()


def test_a_file_this_build_did_not_write_is_refused_rather_than_mangled():
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)  # 8-bit, which WARLOCK 5/6 never writes
        handle.setframerate(44100)
        handle.writeframes(b"\x00" * 100)
    with pytest.raises(RuntimeError, match="16-bit PCM"):
        q._roll_wav(out.getvalue(), 0.5)


# --- _stage_rolled_wav ---------------------------------------------------------


def test_a_loop_takes_roll_back_is_staged_not_written_in_place(tmp_path, monkeypatch):
    """The 2026-09-05 audit (muse-04): a loop's roll-back wrote to
    ``track.wav`` -- the job's served artifact name -- by reading it and
    writing the rolled bytes straight back in place. A process killed between
    that read and that write left a truncated or corrupt ``track.wav`` on
    disk, exactly the failure every other writer onto a served name in this
    module (``_write_stems_sidecar``, ``separation_worker``'s per-stem
    ``.tmp``) stages against.

    The fault is injected at the boundary between the read and the on-disk
    swap -- ``Path.replace`` raises -- so the assertion distinguishes staged
    from in-place: an in-place write has already clobbered ``track.wav``
    before ``replace`` is ever called, so ``track.wav`` would come back
    rolled (silently corrupt-on-crash); a staged write leaves it byte-for-byte
    the original, because the rolled bytes only ever touched the temp
    sibling.
    """
    frames = np.arange(44100 * 2, dtype="<i2").reshape(-1, 2)
    original = _wav(frames)
    output = tmp_path / "track.wav"
    output.write_bytes(original)

    def _boom(self, target):
        raise OSError("simulated kill between read and replace")

    monkeypatch.setattr(Path, "replace", _boom)

    with pytest.raises(OSError, match="simulated kill"):
        q._stage_rolled_wav(output, 0.25)

    assert output.read_bytes() == original


# --- _wav_duration_seconds ----------------------------------------------------


def test_wav_duration_seconds_reads_the_headers_real_length_not_a_guess(tmp_path):
    """The 2026-09-14 audit, finding muse-03's pure half.

    ``getnframes()``/``getframerate()`` off the header, not the caller's own
    idea of how long the file should be -- the ``fallback`` argument exists
    only for a file this build cannot read at all.
    """
    rate = 1000
    frames = np.arange(2500 * 2, dtype="<i2").reshape(-1, 2)  # 2500 frames, 2.5 s
    path = tmp_path / "track.wav"
    path.write_bytes(_wav(frames, rate))

    assert q._wav_duration_seconds(path, fallback=999.0) == pytest.approx(2.5)


def test_wav_duration_seconds_falls_back_for_a_file_it_cannot_read(tmp_path):
    path = tmp_path / "track.wav"
    path.write_bytes(b"not a wav")
    assert q._wav_duration_seconds(path, fallback=42.0) == 42.0


# --- _music (the full stage) --------------------------------------------------


class _FakeCancel:
    """Just enough of the queue's cancel token for ``MusicOps._music``.

    ``event`` used to be a bare ``object()`` -- fine as long as nothing but
    ``commit()`` ever touched this fake, which stopped being true the moment
    ``_music`` started checking ``self._cancel.event.is_set()`` after
    ``client.generate`` returns (the 2026-09-15 audit, finding muse-01).
    ``queue._Cancel.event`` is always a real ``threading.Event`` in
    production; a bare ``object()`` was this fake behind the real contract,
    not a shape production has to tolerate, so the fix is here rather than a
    ``getattr``/``hasattr`` guard in ``_music`` for an event that can never
    actually be missing ``.is_set()``. Unset by construction, matching every
    test in this class that never asks for a cancel.
    """

    def __init__(self) -> None:
        self.event = threading.Event()
        self.committed = False

    def commit(self) -> None:
        self.committed = True


class _FakeConfig:
    def __init__(self, root: Path) -> None:
        self.root = root

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id


class _FakeStore:
    def __init__(self) -> None:
        self.saved: dict[str, Any] = {}

    def set_params(self, job_id: str, params: dict[str, Any]) -> None:
        self.saved[job_id] = dict(params)


class _FakeClient:
    """Stands in for ``MusicClient``: ``generate`` writes a WAV of a length
    the caller chooses -- an audio2audio job's own reference-sized render,
    for this test -- regardless of what ``audio_duration`` asked for."""

    def __init__(self, rendered_seconds: float, rate: int = 44100) -> None:
        self.rendered_seconds = rendered_seconds
        self.rate = rate
        self.last_recipe = None

    def generate(self, prompt, output, *, audio_duration, **_kw):
        n = int(round(self.rendered_seconds * self.rate))
        frames = np.zeros((max(n, 1), 2), dtype="<i2")
        output.write_bytes(_wav(frames, self.rate))


async def test_audio2audio_actual_duration_matches_the_rendered_file_not_the_request(
    tmp_path,
):
    """muse-03 (2026-09-14 audit).

    ``pipeline_ace_step.py`` sizes an audio2audio render's ``frame_length``
    off the *reference*'s own latents (``ref_latents.shape[-1]``, lines
    936-941), not off the requested ``duration`` -- so a take's file can run
    for minutes longer or shorter than the request. This job asks for 30 s
    and the fake sampler (standing in for that reference-sized render) writes
    a 96 s file; the take card's ``actual_duration`` has to read the file, not
    echo the request back.

    Fails against the unfixed code, which sets
    ``params["actual_duration"] = float(params.get("duration", 60.0))`` --
    30.0, the request -- with nothing that ever opens ``track.wav``.
    """
    rendered_seconds = 96.0
    client = _FakeClient(rendered_seconds)

    # ``worker`` is a bare namespace, not a ``MusicOps``/``Worker`` instance,
    # so these are plain attributes ``_music`` reaches through ``self.`` --
    # not class methods -- which is why they take no ``self`` of their own.
    async def _acquire_music(spec):
        return client, False

    async def _release_music(client, spec, *, handoff):
        return None

    worker = SimpleNamespace(
        config=_FakeConfig(tmp_path),
        store=_FakeStore(),
        _cancel=_FakeCancel(),
        _music_state=lambda job_id, s: None,
        _music_step=lambda job_id, i, n: None,
        _music_client=None,
        _acquire_music=_acquire_music,
        _release_music=_release_music,
    )
    job_id = "abc123"
    params = {
        "task": "audio2audio",
        "duration": 30.0,
        "reference_wav": b"",
        "ref_audio_strength": 0.5,
    }
    job = {"id": job_id, "prompt": "x", "params": params}

    await q.MusicOps._music(worker, job)

    saved = worker.store.saved[job_id]
    assert saved["actual_duration"] == pytest.approx(rendered_seconds)
    assert saved["actual_duration"] != pytest.approx(30.0)
