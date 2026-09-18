"""Regression tests for the 2026-09-18 audit's Muse findings, ``_q_music.py``
side. Sibling of ``test_audit_2026_09_15_muse.py`` (same shape, same reason
for a dated file: a fix batch that closes several unrelated findings in one
mode, each worth its own section rather than a shared, unlabelled one).
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from warlock import _q_music as q

# --- muse-03: cancel during a loop take's roll-back --------------------------


class _FakeCancel:
    """Just enough of ``queue._Cancel`` for ``MusicOps._music``."""

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


def _wav(seconds: float, rate: int = 44100) -> bytes:
    import io
    import wave

    frames = np.zeros((int(seconds * rate), 2), dtype="<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())
    return out.getvalue()


class _OrdinaryLoopClient:
    """A loop take that finishes cleanly -- the cancel this finding is about
    lands *after* ``generate`` returns, during the roll-back, not during
    sampling. See ``_q_music.py``'s own muse-01 (2026-09-15 audit) comment for
    the sibling window this one is not: that one is between the last sampling
    step and ``generate`` returning; this one is between ``generate``
    returning and the commit, and specifically inside the roll.
    """

    def __init__(self) -> None:
        self.last_recipe = {"model": "ace_step_v1"}

    def generate(self, prompt, output, *, on_step=None, **_kw):
        if on_step is not None:
            on_step(1, 1)
        output.write_bytes(_wav(4.0))
        return output


async def test_a_cancel_during_the_loop_roll_is_not_committed(tmp_path, monkeypatch):
    """muse-03 (2026-09-18 audit).

    ``_music``'s loop branch runs ``_stage_rolled_wav`` -- an awaited thread
    hop -- *after* the one cancel check the 2026-09-15 audit's muse-01 added,
    and then falls through to ``self._cancel.commit()`` with no check in
    between. A Cancel pressed while the roll is running lands in exactly that
    gap and used to be published anyway.

    The roll is stood in for by a stub that flips the cancel event partway
    through -- reproducing "the roll is still in flight when Cancel is
    pressed" without depending on the real WAV-rewriting arithmetic in
    ``_stage_rolled_wav``, which is not this finding's claim.

    Fails against the unfixed code: ``worker._cancel.committed`` comes back
    ``True`` and ``worker.store.saved`` holds a ``done``-shaped params write,
    because the old ``_music`` called ``self._cancel.commit()`` right after
    the roll with no check of the event in between.
    """
    from warlock.pipelines.music_client import MusicCancelled

    cancel = _FakeCancel()
    client = _OrdinaryLoopClient()
    roll_calls: list[tuple[Any, float]] = []

    def _fake_roll(output, seconds):
        # The roll is genuinely mid-flight work in the real code (a thread
        # hop via ``asyncio.to_thread``) -- setting the event here is what a
        # Cancel pressed during that window actually looks like from
        # ``_music``'s point of view: the roll completes, but the event is
        # now set by the time control returns to the coroutine.
        roll_calls.append((output, seconds))
        cancel.event.set()

    monkeypatch.setattr(q, "_stage_rolled_wav", _fake_roll)

    async def _acquire_music(spec):
        return client, False

    async def _release_music(client, spec, *, handoff):
        return None

    worker = SimpleNamespace(
        config=_FakeConfig(tmp_path),
        store=_FakeStore(),
        _cancel=cancel,
        _music_state=lambda job_id, s: None,
        _music_step=lambda job_id, i, n: None,
        _music_client=None,
        _acquire_music=_acquire_music,
        _release_music=_release_music,
    )
    job_id = "loopcancel1"
    job = {
        "id": job_id,
        "prompt": "dark ambient, dungeon",
        "params": {
            "task": "loop",
            "duration": 30.0,
            "roll": 15.0,
            "repaint_start": 0.0,
            "repaint_end": 8.0,
        },
    }

    with pytest.raises(MusicCancelled):
        await q.MusicOps._music(worker, job)

    # The roll actually ran -- proving this test exercises the gap *after*
    # it, not the earlier, already-fixed window before it.
    assert roll_calls, "the loop roll must have run for this to be the right window"

    # The token is left uncommitted -- the dispatch loop's own ``finally``
    # (``queue.py``) is what turns that into a cancelled row and a call to
    # ``_discard_artifacts``; this stage's only job is to not commit past it.
    assert cancel.committed is False
    assert job_id not in worker.store.saved


async def test_a_loop_take_with_no_late_cancel_still_commits_normally(tmp_path, monkeypatch):
    """The fix's other half: a loop that runs to completion with no Cancel
    anywhere near the end must still publish -- the new check must not grey
    the ordinary path."""
    cancel = _FakeCancel()
    client = _OrdinaryLoopClient()

    def _fake_roll(output, seconds):
        pass  # no cancel -- the ordinary case

    monkeypatch.setattr(q, "_stage_rolled_wav", _fake_roll)

    async def _acquire_music(spec):
        return client, False

    async def _release_music(client, spec, *, handoff):
        return None

    worker = SimpleNamespace(
        config=_FakeConfig(tmp_path),
        store=_FakeStore(),
        _cancel=cancel,
        _music_state=lambda job_id, s: None,
        _music_step=lambda job_id, i, n: None,
        _music_client=None,
        _acquire_music=_acquire_music,
        _release_music=_release_music,
    )
    job_id = "loopok1"
    job = {
        "id": job_id,
        "prompt": "dark ambient, dungeon",
        "params": {
            "task": "loop",
            "duration": 30.0,
            "roll": 15.0,
            "repaint_start": 0.0,
            "repaint_end": 8.0,
        },
    }

    await q.MusicOps._music(worker, job)

    assert cancel.committed is True
    assert job_id in worker.store.saved
