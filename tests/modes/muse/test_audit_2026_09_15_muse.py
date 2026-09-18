"""Regression tests for the 2026-09-15 audit's Muse findings.

Five records, five sections below: service-02 (a derive with "How many" > 1
draws one take repeated rather than several distinct ones), muse-01 (a Cancel
that lands during the vocoder decode still publishes the take), muse-02
(``_music``/``_separate`` were never rows in ``PUBLISHERS``), muse-03 (a
sub-sample loop region opens the export picker and then writes nothing), and
muse-04/muse-05 (a hand-listed stem set, and two functions with no coverage at
all).
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from warlock import _q_jobs, models
from warlock import _q_music as q
from warlock.service import _jobs_music as door
from warlock.service.errors import Invalid
from warlock.studio.modes.muse import fileio as muse_io
from warlock.studio.modes.muse import mode as muse_mode
from warlock.studio.modes.muse import state as muse_state

from .test_muse_mode import FakeCtx

# --- service-02: a derive with "How many" > 1 ------------------------------


def test_a_repaint_derivation_with_count_greater_than_one_draws_distinct_takes(
    tmp_path,
):
    """``_task_kwargs`` used to stop at ``src_audio_path`` for repaint/loop/
    extend/edit and never forward the stored ``retake_seed`` -- only
    ``retake`` did. Every one of those four tasks draws its own variation
    from the sampler's ``retake_random_generators``
    (``pipeline_ace_step.__call__``'s ``add_retake_noise`` for the first two,
    ``flowedit_diffusion_process``'s ``random_generators=
    retake_random_generators`` for edit -- see that call's "more diversity"
    comment), and ``manual_seeds`` -- the only thing that *was* seeded -- is
    the take's own draw, deliberately inherited unchanged by
    ``derive_music_job``. With no ``retake_seeds`` on the wire, two rows of
    the same repaint asked for with ``count=4`` were four requests for the
    sampler's own fresh, unrecorded draw: indistinguishable from four reruns
    of the same row, and never reproducible.

    Fails against the unfixed code with a ``KeyError: 'retake_seeds'`` on the
    first assertion below: the pre-fix repaint/loop branch returns
    ``{"task": "repaint", "src_audio_path": ..., "repaint_start": ...,
    "repaint_end": ...}`` and nothing else.
    """
    job_dir = tmp_path
    (job_dir / "source.wav").touch()

    row_a = q._task_kwargs(
        {
            "task": "repaint",
            "repaint_start": 10.0,
            "repaint_end": 20.0,
            "retake_seed": 111,
        },
        job_dir,
    )
    row_b = q._task_kwargs(
        {
            "task": "repaint",
            "repaint_start": 10.0,
            "repaint_end": 20.0,
            "retake_seed": 222,
        },
        job_dir,
    )
    assert row_a["retake_seeds"] == [111]
    assert row_b["retake_seeds"] == [222]
    assert row_a["retake_seeds"] != row_b["retake_seeds"]


@pytest.mark.parametrize("task", ["extend", "loop", "edit"])
def test_every_task_that_shares_the_retake_noise_path_also_forwards_its_seed(
    tmp_path, task
):
    """The same gap, the other three tasks it applied to. Not ``retake``
    (already forwarded before this audit) and not ``audio2audio`` (refused a
    count above 1 below, precisely because it has no such path)."""
    base = {
        "task": task,
        "retake_seed": 42,
        "extend_left": 5.0,
        "extend_right": 10.0,
        "parent_duration": 60.0,
        "repaint_start": 26.0,
        "repaint_end": 34.0,
        "edit_prompt": "brighter",
        "edit_lyrics": "",
    }
    out = q._task_kwargs(base, tmp_path)
    assert out["retake_seeds"] == [42]


@pytest.fixture(autouse=True)
def _admitted(monkeypatch):
    monkeypatch.setattr(door, "check_weights", lambda svc, kind, params: None)
    monkeypatch.setattr(door, "check_vram", lambda svc, kind, stage, params: None)


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


@pytest.fixture
def parent(svc):
    made = door.create_music_job(svc, prompt="dark ambient, dungeon", duration=60.0)
    job_id = made["id"]
    (svc.config.job_dir(job_id) / "track.wav").write_bytes(_wav(60.0))
    svc.store.set_status(job_id, "done")
    return job_id


def test_an_audio2audio_derive_with_count_above_one_is_refused_at_the_door(
    svc, parent
):
    """audio2audio's upstream ``task`` is reassigned to ``"audio2audio"``
    *inside* ``__call__`` itself (``if audio2audio_enable and ref_audio_input
    is not None``), which ``add_retake_noise = task in ("retake", "repaint",
    "extend")`` never matches -- so, unlike every sibling task, it never
    touches ``retake_random_generators`` at all. With ``seed`` inherited
    unchanged, nothing distinguishes one row of a count > 1 audio2audio
    derive from another: it queues N generations of one request and returns
    one take N times. Refused by name rather than silently degraded.

    Fails against the unfixed code, which has no ``count`` check in the
    ``else:  # audio2audio`` branch at all and queues four identical rows.
    """
    with pytest.raises(Invalid) as caught:
        door.derive_music_job(svc, parent, task="audio2audio", count=4)
    assert caught.value.field == "count"

    # One is still fine -- the refusal is about the multiplied request, not
    # the task itself.
    out = door.derive_music_job(svc, parent, task="audio2audio", count=1)
    assert len(out["ids"]) == 1


def test_the_derive_popup_does_not_offer_how_many_for_audio2audio():
    """``muse_mode.SINGLE_TAKE_TASKS`` is what ``muse_results`` reads to skip
    the "How many" slider -- read here directly rather than through a full
    imgui frame, which ``test_muse_panes_smoke.py`` (not owned by this fix)
    already drives for every task's popup and would be the wrong place to
    duplicate a second, disagreeing assertion about the same widget.
    """
    assert "audio2audio" in muse_mode.SINGLE_TAKE_TASKS
    # Every task the popup can actually reach is in ``DERIVE_CONTROLS``; the
    # single-take set must be a subset of it, or the door would be refusing a
    # task the popup can never open.
    assert set(muse_mode.DERIVE_CONTROLS) >= muse_mode.SINGLE_TAKE_TASKS
    # And the only one *today*: every other task varies with count (proved by
    # the two tests above), so widening this set to a task that does vary
    # would be exactly the wrong fix in the other direction.
    assert frozenset({"audio2audio"}) == muse_mode.SINGLE_TAKE_TASKS

    # And the popup's own draw function actually consults it -- the constant
    # existing unused would pass every assertion above.
    import inspect

    from warlock.studio.modes.muse.ui.panes import results as muse_results

    assert "SINGLE_TAKE_TASKS" in inspect.getsource(muse_results.derive_popup)


# --- muse-01: Cancel near the end of a take ---------------------------------


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


class _CancelsAfterTheLastStepClient:
    """Stands in for ``MusicClient``: writes a real ``track.wav`` -- the
    vocoder decode and ``save_wav_file`` this finding is about are exactly
    the work between the last ``on_step`` call and ``generate`` returning --
    then sets the cancel event the caller handed it, simulating a Cancel
    that lands in that unchecked window.
    """

    def __init__(self, cancel_event: threading.Event, total_steps: int = 3) -> None:
        self.cancel_event = cancel_event
        self.total_steps = total_steps
        self.last_recipe = {"model": "ace_step_v1"}

    def generate(self, prompt, output, *, on_step=None, **_kw):
        for i in range(1, self.total_steps + 1):
            if on_step is not None:
                on_step(i, self.total_steps)
        # The window itself: nothing between the last step and here checks
        # the event in either the vendored sampler or ``music_worker.
        # op_generate``, so a Cancel pressed right now used to reach
        # ``_music``'s ``self._cancel.commit()`` unconditionally.
        self.cancel_event.set()
        output.write_bytes(_wav(1.0))
        return output


async def test_cancel_set_immediately_after_the_last_sampling_step_still_raises_before_the_wav_is_written(  # noqa: E501
    tmp_path,
):
    """muse-01 (2026-09-15 audit).

    Fails against the unfixed code: ``worker._cancel.committed`` comes back
    ``True`` and ``worker.store.saved`` holds a ``done``-shaped params write,
    because the old ``_music`` called ``self._cancel.commit()`` right after
    ``client.generate`` returned with no check in between -- publishing a
    take the Cancel had already asked to stop.
    """
    from warlock.pipelines.music_client import MusicCancelled

    cancel = _FakeCancel()
    client = _CancelsAfterTheLastStepClient(cancel.event)

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
    job_id = "cancel123"
    job = {
        "id": job_id,
        "prompt": "dark ambient, dungeon",
        "params": {"task": "retake", "duration": 30.0, "retake_variance": 0.5},
    }

    with pytest.raises(MusicCancelled):
        await q.MusicOps._music(worker, job)

    # The token is left uncommitted -- the dispatch loop's own ``finally``
    # (``queue.py``) is what turns that into a cancelled row and a call to
    # ``_discard_artifacts``; this stage's only job is to not commit past it.
    assert cancel.committed is False
    assert job_id not in worker.store.saved


def test_discard_artifacts_removes_the_track_a_late_cancel_can_now_leave_behind(
    tmp_path,
):
    """The other half of muse-01. ``_discard_artifacts``'s ``"music"`` branch
    used to delete nothing, on the claim that a row could never reach it with
    a track on disk -- true only as long as ``_music`` committed unconditionally.
    Now that a late Cancel can leave a fully rendered ``track.wav`` sitting
    uncommitted, the branch has to remove it or a cancelled take leaks a full
    render every time this window is hit.

    Fails against the unfixed code: the file still exists after the call.
    """
    config = _FakeConfig(tmp_path)
    job_dir = config.job_dir("cancel123")
    job_dir.mkdir(parents=True)
    track = job_dir / "track.wav"
    track.write_bytes(_wav(1.0))
    (job_dir / "source.wav").write_bytes(b"not-touched")

    worker = SimpleNamespace(config=config)
    job = {"id": "cancel123", "kind": "music", "params": {}}

    _q_jobs.JobOps._discard_artifacts(worker, job)

    assert not track.exists()
    # The input the door wrote before the row existed -- untouched, same as
    # every other kind's rule for its own inputs.
    assert (job_dir / "source.wav").exists()


# --- muse-02: PUBLISHERS ------------------------------------------------------


def test_music_and_separate_are_rows_in_the_publishers_cancel_commit_scan():
    """``_music`` and ``_separate`` publish onto a served name and commit the
    cancel token correctly -- but neither was ever a row in ``PUBLISHERS``,
    so ``test_every_served_publish_commits_the_cancel_token`` never looked at
    either. Read the live list rather than re-implementing the scan: the
    claim here is only that the two rows exist, not that the scan itself
    (owned by ``test_job_durability.py``, and exercised there) is correct.

    Fails against the unfixed list, which has nine rows and none of them
    name ``warlock._q_music``.
    """
    from test_job_durability import PUBLISHERS

    named = {(module, func) for module, func, _ in PUBLISHERS}
    assert ("warlock._q_music", "_music") in named
    assert ("warlock._q_music", "_separate") in named


# --- muse-03: export a sub-sample loop ---------------------------------------


def _player(seconds: float = 10.0, rate: int = 44100) -> Any:
    from warlock.studio.modes.muse.engine import waveform

    pcm = np.zeros((int(seconds * rate), 2), dtype=np.int16)
    return muse_state.Player(
        job="a", pcm=pcm, rate=rate, env=waveform.peaks(pcm), duration=seconds
    )


def test_export_loop_toasts_instead_of_silently_writing_nothing_for_a_sub_sample_region(
    tmp_path, monkeypatch
):
    """muse-03 (2026-09-15 audit).

    A region that passes ``_has_region`` in seconds (``loop_end >
    loop_start``) can still quantise to the same sample index at the take's
    rate, which is exactly what dragging both grips to within a float
    rounding error of each other produces. ``loop_cache_key`` correctly
    refuses that (``end <= start`` in samples), but ``export_loop`` used to
    let the picker open, call ``make()``, get ``None`` back, and return --
    indistinguishable from the user cancelling the save dialog.

    Fails against the unfixed code: ``dialogs.save_file`` is called (the
    picker "opens") and ``ctx.toasts`` is empty afterwards.
    """
    one = _player()
    # Same float rounding error a drag can produce: under one sample apart at
    # 44100 Hz, so ``_has_region`` (seconds) passes and ``loop_cache_key``
    # (samples) does not.
    one.loop_start = 2.0
    one.loop_end = 2.0 + (1.0 / 44100.0) / 2.0

    picker_calls: list[Any] = []
    monkeypatch.setattr(
        muse_io.dialogs,
        "save_file",
        lambda *a, **k: picker_calls.append(1) or (tmp_path / "loop.wav"),
    )

    ctx = FakeCtx(tmp_path)
    muse_io.export_loop(ctx, one)

    assert not picker_calls, "the picker must not open for a region with no usable body"
    assert ctx.toasts, "a region that cannot be exported must say so"
    assert not (tmp_path / "loop.wav").exists()


def test_export_loop_still_writes_an_ordinary_region(tmp_path, monkeypatch):
    """The fix's other half: a real region must not be caught by the new
    guard. Same shape as ``test_muse_player.py``'s M09 export test, kept
    local and minimal rather than importing that file's fixtures."""
    one = _player()
    one.loop_start, one.loop_end = 2.0, 8.0
    out = tmp_path / "loop.wav"
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: out)

    ctx = FakeCtx(tmp_path)
    muse_io.export_loop(ctx, one)

    assert out.exists()
    assert not ctx.toasts


# --- muse-04: cancelled stem split cleanup -----------------------------------


def test_discard_artifacts_stem_list_matches_the_separation_models_sources(
    tmp_path, monkeypatch
):
    """``_discard_artifacts``'s ``"separate"`` branch used to hand-list the
    four stem names instead of reading ``SEPARATION_MODELS[...].sources`` --
    the same tuple that names these files on disk in the first place
    (``SeparationModel``'s own docstring). Right only by coincidence today,
    because the registry happens to have one entry carrying the default
    tuple; proved here against a *second*, differently-ordered model rather
    than against the registry's real (matching) one, so the test does not
    pass for the same reason the bug shipped.

    Fails against the unfixed code: it deletes ``drums.wav``/``bass.wav``/
    ``other.wav``/``vocals.wav`` regardless of what ``fake.sources`` says, so
    ``stem_one.wav``/``stem_two.wav`` below survive the call.
    """
    fake = models.SeparationModel(
        key="_audit_fake_separator",
        label="fake",
        dir_name="fake",
        sources=("stem_one", "stem_two"),
    )
    monkeypatch.setitem(models.SEPARATION_MODELS, fake.key, fake)

    source_id = "abc123abc123"
    config = _FakeConfig(tmp_path)
    source_dir = config.job_dir(source_id)
    stems = source_dir / "stems"
    stems.mkdir(parents=True)
    (stems / "stems.json").write_text("{}")
    (stems / "stem_one.wav").write_bytes(b"x")
    (stems / "stem_two.wav").write_bytes(b"x")

    worker = SimpleNamespace(config=config)
    job = {
        "id": "def456def456",
        "kind": "separate",
        "params": {"source_job": source_id, "separation_model": fake.key},
    }

    _q_jobs.JobOps._discard_artifacts(worker, job)

    assert not (stems / "stem_one.wav").exists()
    assert not (stems / "stem_two.wav").exists()
    assert not (stems / "stems.json").exists()


# --- muse-05: Muse Stems -----------------------------------------------------


def test_has_stems_reads_the_stems_prefix_from_the_cached_files_list():
    """No test anywhere called ``has_stems`` before this audit."""
    assert muse_mode.has_stems(None, {"files": ["track.wav", "stems/drums.wav"]})
    assert not muse_mode.has_stems(None, {"files": ["track.wav"]})
    assert not muse_mode.has_stems(None, {})
    assert not muse_mode.has_stems(None, {"files": []})


def test_separate_reaches_separate_job_and_refuses_a_second_press_while_busy(
    tmp_path, monkeypatch
):
    """The other half of muse-05: ``separate`` itself, never exercised."""
    from warlock.service import jobs as svc_jobs

    seen: dict[str, Any] = {}

    def _fake_separate(svc, job_id):
        seen["svc"], seen["job_id"] = svc, job_id
        return {"id": "stems-job"}

    monkeypatch.setattr(svc_jobs, "separate_job", _fake_separate)

    ctx = FakeCtx(tmp_path)
    assert muse_mode.separate(ctx, "take123") is True
    assert seen == {"svc": ctx.svc, "job_id": "take123"}
    assert ctx.result == {"id": "stems-job"}

    busy_ctx = FakeCtx(tmp_path, accept=False)
    assert muse_mode.separate(busy_ctx, "take123") is False
    assert busy_ctx.toasts and busy_ctx.toasts[-1][0] == "Already splitting that take."
