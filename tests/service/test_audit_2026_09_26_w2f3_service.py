"""Regression tests for the 2026-09-26 audit's service-layer Muse findings
(pass w2f3): the edit door's source/target conditioning (muse-jobs-01), the
files cache noticing a stems separation land (service-assets-02), and the
reference-bytes ceiling being per copy rather than per request (muse-jobs-02).
"""

from __future__ import annotations

import os
import time
import wave
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from realmspinner import _q_music as q
from realmspinner.kernels.audio import wavout
from realmspinner.service import _jobs_music as door
from realmspinner.service.errors import Invalid

# --- muse-jobs-01 --------------------------------------------------------------


class _FakeCancel:
    """Just enough of the queue's cancel token for ``MusicOps._music``."""

    def __init__(self) -> None:
        import threading

        self.event = threading.Event()
        self.committed = False

    def commit(self) -> None:
        self.committed = True


class _FakeConfig:
    def __init__(self, root) -> None:
        self.root = root

    def job_dir(self, job_id: str):
        return self.root / job_id


class _FakeStore:
    def __init__(self) -> None:
        self.saved: dict[str, Any] = {}

    def set_params(self, job_id: str, params: dict[str, Any]) -> None:
        self.saved[job_id] = dict(params)


class _FakeClient:
    """Stands in for ``MusicClient``: records what ``generate`` was called
    with and writes a short, valid WAV so ``_music``'s own post-processing
    (reading ``track.wav``'s header back for ``actual_duration``) succeeds."""

    last_recipe = None

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt, output, *, audio_duration, **kw):
        self.calls.append({"prompt": prompt, **kw})
        frames = np.zeros((100, 2), dtype="<i2")
        with wave.open(str(output), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(100)
            handle.writeframes(frames.tobytes())


async def test_an_edit_sends_the_parents_prompt_as_source_and_the_new_words_as_target(
    tmp_path,
):
    """muse-jobs-01 (2026-09-26 audit).

    An edit's row ``prompt`` is deliberately the *new* words --
    ``derive_music_job`` puts them there so the library shows a derived row's
    own brief -- but ``_music`` used to send that same column to the sampler
    as the *source* conditioning too, identical to ``edit_target_prompt``
    (also the new words), so a prompt-only edit had nothing to edit away
    from.

    Fails against the unfixed code: the positional ``prompt``
    ``client.generate`` receives is "bright strings, major" (the new words),
    not "dark ambient, dungeon" (the parent's) -- so ``calls[0]["prompt"]``
    equals ``calls[0]["edit_target_prompt"]``.
    """
    client = _FakeClient()

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
        "task": "edit",
        "duration": 30.0,
        "edit_prompt": "bright strings, major",
        "edit_source_prompt": "dark ambient, dungeon",
        "edit_lyrics": "",
        "edit_n_min": 0.0,
        "edit_n_max": 1.0,
    }
    job = {"id": job_id, "prompt": "bright strings, major", "params": params}

    await q.MusicOps._music(worker, job)

    assert client.calls[0]["prompt"] == "dark ambient, dungeon"
    assert client.calls[0]["edit_target_prompt"] == "bright strings, major"
    assert client.calls[0]["prompt"] != client.calls[0]["edit_target_prompt"]


# --- service-assets-02 ----------------------------------------------------


def test_attach_files_cache_notices_stems_json_landing_in_the_stems_subdirectory(svc):
    """service-assets-02 (2026-09-26 audit, re-run: cached call still
    ``['track.wav']``).

    Stems land inside a ``stems/`` subdirectory that a separation job's own
    ``mkdir`` creates once -- the one moment that touches *job_dir*'s own
    mtime. Every stem WAV, and the ``stems.json`` gate that lands last, is
    written *inside* ``stems/`` instead, which never moves job_dir's mtime
    again. A listing cached between that mkdir and ``stems.json`` landing
    matched the stamp forever after, no matter what landed inside ``stems/``.

    Fails against the unfixed code: the second call still returns
    ``['track.wav']`` even though ``stems.json`` is now on disk.
    """
    from realmspinner.service import files as svc_files

    job_id = "mus1c0000001"
    job = {"id": job_id, "kind": "music", "stage": "music", "status": "done"}
    job_dir = svc.job_dir(job_id)
    stems_dir = job_dir / svc_files.STEMS_DIR
    stems_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "track.wav").write_bytes(b"RIFF....WAVEfmt ")

    # Both directories settled well outside the race window, so the first
    # listing is trusted to cache -- ``test_studio_frame.py``'s own pattern
    # for making the timing a decision rather than a coin toss.
    now = time.time_ns()
    settled = now - 10 * svc_files.MTIME_RACE_NS
    os.utime(job_dir, ns=(settled, settled))
    os.utime(stems_dir, ns=(settled, settled))

    cache: dict = {}
    svc_files.attach_files(job, job_dir, cache=cache)
    assert job["files"] == ["track.wav"]
    assert job_id in cache, "both mtimes are safely in the past -- this must be cached"

    # The separation job's own writes: four stems, then the completion gate,
    # all inside ``stems/`` -- job_dir itself is never touched.
    for name in ("drums", "bass", "other", "vocals"):
        (stems_dir / f"{name}.wav").write_bytes(b"RIFF")
    (stems_dir / "stems.json").write_bytes(b"{}")

    svc_files.attach_files(job, job_dir, cache=cache)
    assert any(name.endswith("drums.wav") for name in job["files"]), job["files"]


# --- muse-jobs-02 --------------------------------------------------------


@pytest.fixture(autouse=True)
def _admitted(monkeypatch):
    monkeypatch.setattr(door, "check_weights", lambda svc, kind, params: None)
    monkeypatch.setattr(door, "check_vram", lambda svc, kind, stage, params: None)


def _reference_wav(seconds: float, rate: int) -> bytes:
    n = int(seconds * rate)
    silence = np.zeros((n, 2), dtype=np.float32)
    return wavout.wav_bytes(silence, rate)


def test_create_music_job_refuses_a_reference_whose_total_across_takes_is_too_large(
    svc, monkeypatch
):
    """muse-jobs-02 (2026-09-26 audit).

    ``MAX_REFERENCE_BYTES`` bounded one *copy* of ``reference_wav``, but the
    door then writes that same buffer once per row -- up to ``MAX_COUNT``,
    four -- with nothing that ever added the copies up: a reference just
    under the ceiling and ``count=4`` wrote up to four times the ceiling to
    disk with the door having only ever checked one copy of it.

    Fails against the unfixed code: a reference at 80% of a (lowered, for
    this test) ceiling, asked for at ``count=4``, is accepted -- 320% of the
    ceiling written to disk -- instead of refused.
    """
    monkeypatch.setattr(door, "MAX_REFERENCE_BYTES", 500_000)
    reference = _reference_wav(10.0, rate=4000)
    assert len(reference) < 500_000, "a single copy must still be under the (lowered) ceiling"

    with pytest.raises(Invalid) as caught:
        door.create_music_job(
            svc,
            prompt="dark ambient, dungeon",
            duration=30.0,
            count=4,
            reference_wav=reference,
        )
    assert caught.value.field == "count"


def test_a_reference_within_the_total_ceiling_is_still_accepted(svc, monkeypatch):
    """The other half of muse-jobs-02: the new check must not refuse a
    request the old per-copy one would have accepted and that still fits
    within the ceiling once multiplied out."""
    monkeypatch.setattr(door, "MAX_REFERENCE_BYTES", 500_000)
    reference = _reference_wav(10.0, rate=4000)

    out = door.create_music_job(
        svc,
        prompt="dark ambient, dungeon",
        duration=30.0,
        count=1,
        reference_wav=reference,
    )
    assert len(out["ids"]) == 1
