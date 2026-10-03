"""Audit 2026-10-03 muse-01..03 regressions."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from realmspinner.service import _jobs_music as door
from realmspinner.service import jobs

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _admitted(monkeypatch):
    monkeypatch.setattr(door, "check_weights", lambda svc, kind, params: None)
    monkeypatch.setattr(door, "check_vram", lambda svc, kind, stage, params: None)
    from realmspinner.service import _jobs_resubmit as resub

    monkeypatch.setattr(resub, "check_weights", lambda svc, kind, params: None)
    monkeypatch.setattr(resub, "check_vram", lambda svc, kind, stage, params: None)


def _wav(seconds: float, rate: int = 44100) -> bytes:
    frames = np.zeros((int(seconds * rate), 2), dtype="<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())
    return out.getvalue()


def _finished(svc, **kw):
    made = door.create_music_job(svc, prompt="dark ambient, dungeon", duration=60.0, **kw)
    jid = made["id"]
    (svc.config.job_dir(jid) / "track.wav").write_bytes(_wav(60.0))
    svc.store.set_status(jid, "done")
    return jid


def test_a_derivation_of_an_edited_take_sends_the_edited_lyrics_not_the_originals(svc):
    parent = _finished(svc, lyrics="old words")
    edited = door.derive_music_job(svc, parent, task="edit", edit_lyrics="new words")["id"]
    (svc.config.job_dir(edited) / "track.wav").write_bytes(_wav(60.0))
    svc.store.set_status(edited, "done")
    assert svc.store.get(edited)["params"]["lyrics"] == "new words"
    child = door.derive_music_job(svc, edited, task="retake")["id"]
    assert svc.store.get(child)["params"]["lyrics"] == "new words"
    # The worker's source conditioning for the edit itself stays the old words.
    assert svc.store.get(edited)["params"]["edit_source_lyrics"] == "old words"
    # An edit back to the old words is a real change now.
    back = door.derive_music_job(svc, edited, task="edit", edit_lyrics="old words")
    assert back["id"]


def test_a_reroll_of_a_derived_take_keeps_the_parents_seed_and_walks_the_retake_seed(svc):
    parent = _finished(svc)
    was = svc.store.get(parent)["params"]["seed"]
    child = door.derive_music_job(svc, parent, task="retake")["id"]
    (svc.config.job_dir(child) / "track.wav").write_bytes(_wav(60.0))
    svc.store.set_status(child, "done")
    before = svc.store.get(child)["params"]["retake_seed"]
    out = jobs.rerun_job(svc, child, mode="reroll")
    params = svc.store.get(out["id"])["params"]
    assert params["seed"] == was
    assert params["retake_seed"] != before


def test_the_soundtrack_tutorial_does_not_promise_a_warm_pipeline_between_takes():
    text = (ROOT / "docs/manual/16-generating-a-soundtrack.md").read_text(encoding="utf-8")
    assert "reuses the loaded pipeline" not in text
