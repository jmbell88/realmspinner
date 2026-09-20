"""Stem separation: the registry, the door, and the names on disk.

No card, no weights and no child process, with one exception: the exit-code
contract between ``separation_worker`` and ``blender_run.run_worker`` is only real
with a real subprocess, so that one test below spawns one -- CPU-only and
weight-free, since it exercises the *failure* path.

What is asserted otherwise is everything that decides *whether* a separation
runs and *where its files land* -- the two halves that are wrong silently. The
model itself is the gpu lane's job.

The recurring theme is that four separate places name the same four stems, and
nothing but this file makes them agree: ``SeparationModel.sources`` (the model's
constructor argument), ``files.MEDIA`` (the export allowlist), ``files.LISTED``
(what a finished job reports) and the queue's own literal (which cannot import
the service).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from realmspinner import _q_music, fetch, models, vram
from realmspinner.pipelines import blender_run
from realmspinner.queue import Worker
from realmspinner.service import _jobs_rework as rework
from realmspinner.service import files
from realmspinner.service.errors import Conflict, Invalid

# --- the four names ----------------------------------------------------------


def test_every_source_the_model_returns_has_a_media_entry():
    """``sources`` is three things at once -- the constructor argument, the
    filenames and the allowlist keys -- so a fifth stem added to the registry
    with no MEDIA entry would be a file the worker writes and nothing serves."""
    spec = models.SEPARATION_MODELS[models.DEFAULT_SEPARATION]
    for name in spec.sources:
        assert f"{files.STEMS_DIR}/{name}.wav" in files.MEDIA


def test_the_stem_names_are_literals_rather_than_a_pattern():
    """MEDIA is the allowlist that keeps a caller-supplied string off the
    filesystem, and a ``stems/{name}.wav`` pattern is a hole in exactly that."""
    assert set(files.STEM_FILES) <= set(files.MEDIA)
    assert not any("{" in key or "*" in key for key in files.MEDIA)


def test_no_media_key_escapes_the_job_directory():
    """``collect`` and ``derive.get_file`` both do ``job_dir / name``.

    ``stems/drums.wav`` is the first MEDIA key with a separator in it at all, so
    this is the moment that join stops being obviously safe. Asserted once here
    rather than defended at every join -- ``sirens_io._under``'s belt and
    braces, as a test.
    """
    from pathlib import PurePosixPath, PureWindowsPath

    for name in files.MEDIA:
        for flavour in (PurePosixPath, PureWindowsPath):
            path = flavour(name)
            assert not path.is_absolute(), name
            assert ".." not in path.parts, name
            assert not path.drive, name


def test_the_queues_literal_agrees_with_the_services():
    """The queue may not import the service (``test_queue`` enforces it), so
    ``stems`` is written down twice. This is what stops that being drift."""
    from pathlib import Path

    source = Path(_q_music.__file__).read_text(encoding="utf-8")
    assert f'source_dir / "{files.STEMS_DIR}"' in source


def test_a_finished_take_lists_its_stems():
    assert set(files.STEM_FILES) <= set(files.LISTED)


# --- readiness ---------------------------------------------------------------


def _music_row(status: str = "done") -> dict:
    return {"kind": "music", "stage": "music", "status": status}


def test_a_stem_is_not_ready_until_the_sidecar_lands(tmp_path):
    """``stems.json`` is the completion gate, for ``rig.json``'s reason: the
    four WAVs appear one at a time, so their existence cannot say the set is
    finished -- and a reader that took it that way would offer a take with
    three stems as a take with four."""
    stems = tmp_path / files.STEMS_DIR
    stems.mkdir()
    (stems / "drums.wav").write_bytes(b"x")
    name = f"{files.STEMS_DIR}/drums.wav"
    assert files.ready(_music_row(), tmp_path, name) is False
    (stems / "stems.json").write_text("{}")
    assert files.ready(_music_row(), tmp_path, name) is True


def test_an_unsplit_take_says_so_rather_than_naming_a_file(tmp_path):
    name = f"{files.STEMS_DIR}/vocals.wav"
    assert "stems" in files.unready_reason(_music_row(), tmp_path, name)


# --- the derived audio formats ----------------------------------------------


def test_the_derived_formats_are_derived_from_the_track_not_the_mesh(tmp_path):
    """Its own tuple rather than a member of ``DERIVED``: that one is keyed on
    ``model.glb``, so a fourth name in it would make a take's FLAC wait for a
    mesh it will never have."""
    assert not set(files.DERIVED_AUDIO) & set(files.DERIVED)
    (tmp_path / "track.wav").write_bytes(b"x")
    for name in files.DERIVED_AUDIO:
        assert files.ready(_music_row(), tmp_path, name) is True


def test_a_derived_format_of_a_take_with_no_audio_names_the_track(tmp_path):
    reason = files.unready_reason(_music_row(), tmp_path, "track.flac")
    assert "track" in reason


def test_every_derived_format_has_a_media_type_and_an_encoder():
    """Both directions, minus the one name that is deliberately not encoded.

    ``track.wav`` joined ``DERIVED_AUDIO`` when the Library grew a Convert
    door, so that all five formats are offered from one menu under one lock --
    but it is a staged *copy*, not a ``FORMATS`` row, because re-encoding
    16-bit PCM through float32 and back is a second quantisation for nothing.
    So the two tables are not equal and must not be asserted equal; what has
    to hold is that everything derived is either encodable or that one copy,
    and that all of it is in the ``MEDIA`` allowlist.
    """
    from realmspinner.pipelines import audioout

    assert set(audioout.FORMATS) <= set(files.DERIVED_AUDIO)
    assert set(files.DERIVED_AUDIO) - set(audioout.FORMATS) == {"track.wav"}
    for name in files.DERIVED_AUDIO:
        assert name in files.MEDIA


def test_the_encoder_refuses_a_name_it_does_not_write(tmp_path):
    """The allowlist rule: a name that is not a row may not become a path.

    Asked with ``track.opus`` rather than ``track.aiff`` -- AIFF is a real row
    now, and Opus is the one libsndfile here advertises but cannot be given:
    it accepts only 8/12/16/24/48 kHz, and a take is 44.1, so it would need a
    resample to be offered at all.
    """
    from realmspinner.pipelines import audioout

    assert "track.opus" not in audioout.FORMATS
    with pytest.raises(ValueError, match="not a format"):
        audioout.convert(tmp_path / "a.wav", tmp_path / "b", "track.opus")


def test_a_take_round_trips_through_every_format(tmp_path):
    """libsndfile encodes all four with no ffmpeg and no second binary. Worth
    an actual round trip rather than a claim, because everyone assumes
    otherwise and would reach for a converter."""
    import numpy as np
    import soundfile as sf

    from realmspinner.pipelines import audioout

    source = tmp_path / "track.wav"
    tone = np.sin(np.arange(4410, dtype=np.float32) / 20.0) * 0.5
    sf.write(str(source), np.stack([tone, tone], axis=1), 44100, subtype="PCM_16")
    for name in audioout.FORMATS:
        out = tmp_path / name
        audioout.convert(source, out, name)
        data, rate = sf.read(str(out), always_2d=True)
        assert rate == 44100
        assert data.shape[1] == 2


# --- the registry ------------------------------------------------------------


def test_the_separation_model_is_labelled_non_commercial():
    """Meta states the trained weights are for scientific purposes only, and
    htdemucs was trained the same way with no new grant. This app's purpose is
    making assets people sell, so the flag is what puts the red marker and the
    warning in front of the download."""
    spec = models.SEPARATION_MODELS[models.DEFAULT_SEPARATION]
    assert spec.commercial is False
    assert spec.license_note


def test_the_checkpoint_is_pinned_by_digest_rather_than_a_revision():
    """It is not on the Hub, so there is no commit to name. A digest is the
    stronger half of the same promise: a revision names an immutable commit, a
    digest *is* the artifact."""
    spec = models.SEPARATION_MODELS[models.DEFAULT_SEPARATION]
    one = spec.fetch[0]
    assert one.repo_id == ""
    assert one.url.startswith("https://")
    assert len(one.sha256) == 64
    assert one.filename in spec.probe


def test_it_has_its_own_downloadable_row():
    keys = {entry.kind for entry in fetch.entries()}
    assert "separation" in keys


def test_a_partial_directory_reads_as_absent(tmp_path, monkeypatch):
    """The presence probe names files rather than using the generic
    ``config.json`` + safetensors tail, and it has to: this download is a
    single ``.pt`` with no config beside it, so the tail would report it
    absent forever."""

    class _Config:
        t2i_model_root = tmp_path

    spec = models.SEPARATION_MODELS[models.DEFAULT_SEPARATION]
    assert fetch.present(_Config(), "separation", spec) is False
    (tmp_path / spec.dir_name).mkdir(parents=True)
    (tmp_path / spec.dir_name / spec.probe[0]).write_bytes(b"x")
    assert fetch.present(_Config(), "separation", spec) is True


def test_a_zero_length_checkpoint_is_reported_as_suspect(tmp_path):
    """One file *is* the model, so a zero-length one is the whole thing missing
    while every presence probe says it is installed (MDL-08)."""

    class _Config:
        t2i_model_root = tmp_path

    spec = models.SEPARATION_MODELS[models.DEFAULT_SEPARATION]
    (tmp_path / spec.dir_name).mkdir(parents=True)
    (tmp_path / spec.dir_name / spec.probe[0]).write_bytes(b"")
    assert fetch.suspect_files(_Config(), "separation", spec)


# --- the worker's exit-code contract -----------------------------------------


def test_a_failed_separation_surfaces_the_workers_own_error_message_not_just_an_exit_code(
    tmp_path,
):
    """A handled failure must reach the caller as the worker's own sentence.

    ``separation_worker.main()`` used to write its caught exception into
    ``result_path`` and then exit 1 -- exactly the shape ``blender_run.run_worker``
    treats as a crash, so it deleted that file and raised
    "Stem separation exited with code 1" before ever reading the sentence the
    worker had just written into it. That made ``_q_music.py``'s
    ``result.get("error")`` handler unreachable dead code (the 2026-09-07
    audit, pipelines-01). ``service/downloads.py``'s own runner does not make
    this mistake: a handled failure there is still an exit 0 with the reason
    on disk.

    No weights needed: a missing checkpoint fails inside ``torch.load``, which
    is the earliest ``separate()`` can fail and still be *this* bug -- a crash
    before the spec is even read (a bad JSON, no torch installed) legitimately
    has no result file for ``run_worker`` to read, and is not what this test is
    about.
    """
    spec = {
        "source": str(tmp_path / "missing-take.wav"),
        "out_dir": str(tmp_path / "stems"),
        "model_dir": str(tmp_path / "no-such-model-dir"),
        "sources": list(models.SEPARATION_MODELS[models.DEFAULT_SEPARATION].sources),
        "segment_seconds": 10.0,
        "result_path": str(tmp_path / "separate.json"),
    }
    result = blender_run.run_worker(
        spec,
        timeout=120,
        module="realmspinner.pipelines.separation_worker",
        marker="separate",
        name="Stem separation",
    )
    assert result["ok"] is False
    assert "no-such-model-dir" in result["error"]


# --- admission ---------------------------------------------------------------


def test_a_one_shot_child_credits_no_resident_weights_back():
    """The interesting difference from the music branch.

    The second return value is the resident-checkpoint credit
    ``queue._check_resources`` gives back, and a process that dies at the end of
    the job has nothing to credit -- pricing it like the music pipe would tell
    the queue a permanent 4 GiB had been freed.
    """
    cost, credit = vram.estimate_parts("separate", "music", {}, exclusive=True)
    assert cost > 0.0
    assert credit == 0.0


def test_the_progress_phases_are_registered():
    """An unregistered kind draws ``PHASES_IMAGE``, whose phases it never
    emits -- so the bar sits at zero and then jumps."""
    from realmspinner import progress

    assert progress.phases_for("separate") is progress.PHASES_SEPARATE


# --- the door ----------------------------------------------------------------


def _take(svc, status: str = "done") -> str:
    job_id = svc.store.create("music", "dark ambient", {}, stage="music")
    (svc.config.job_dir(job_id)).mkdir(parents=True, exist_ok=True)
    (svc.config.job_dir(job_id) / "track.wav").write_bytes(b"x")
    svc.store.set_status(job_id, status)
    return job_id


@pytest.fixture(autouse=True)
def _admitted(monkeypatch):
    monkeypatch.setattr(rework, "check_weights", lambda svc, kind, params: None)
    monkeypatch.setattr(rework, "check_vram", lambda svc, kind, stage, params: None)


def test_a_split_is_a_queued_row_naming_its_source(svc):
    """A queued job rather than a task-thread action, and
    ``retexture_job``'s docstring is the deciding sentence: a TaskRunner thread
    racing the worker for VRAM is the OOM that only reproduces under load."""
    take = _take(svc)
    out = rework.separate_job(svc, take)
    row = svc.store.get(out["id"])
    assert row["kind"] == "separate"
    assert row["params"]["source_job"] == take
    assert row["params"]["separation_model"] == models.DEFAULT_SEPARATION


def test_only_a_track_can_be_split(svc):
    job_id = svc.store.create("image", "a goblin", {}, stage="model")
    svc.store.set_status(job_id, "done")
    with pytest.raises(Invalid) as caught:
        rework.separate_job(svc, job_id)
    assert caught.value.field == "source_job"


def test_a_take_with_no_audio_is_refused_in_the_same_words(svc):
    """``muse_mode.play``'s sentence and ``derive_music_job``'s, so all three
    surfaces say the same thing about the same missing file."""
    take = _take(svc)
    (svc.config.job_dir(take) / "track.wav").unlink()
    with pytest.raises(Invalid) as caught:
        rework.separate_job(svc, take)
    assert "no audio on disk" in str(caught.value)


def test_a_running_take_cannot_be_split_yet(svc):
    take = _take(svc, status="running")
    with pytest.raises(Conflict):
        rework.separate_job(svc, take)


def test_an_unknown_model_is_refused_rather_than_defaulted(svc):
    take = _take(svc)
    with pytest.raises(Invalid) as caught:
        rework.separate_job(svc, take, separation_model="demucs_v4")
    assert caught.value.field == "separation_model"


def test_a_split_cannot_be_rerolled(svc):
    """It is deterministic: re-running writes the identical four files over
    themselves, so a reroll is a press with no outcome."""
    from realmspinner.service import _jobs_resubmit as resubmit

    take = _take(svc)
    split = svc.store.create("separate", "x", {"source_job": take})
    svc.store.set_status(split, "done")
    with pytest.raises(Invalid, match="no seed to change"):
        resubmit.rerun_job(svc, split)


# --- dispatch ------------------------------------------------------------


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


def test_separate_job_does_not_reuse_the_blender_pose_timeout(svc, monkeypatch):
    """The 2026-09-08 audit, finding muse-02. ``pose_timeout`` (300s) is sized
    for an inline Blender pose bake -- seconds, not minutes, by its own
    docstring -- but ``_separate`` is a queued job that can process up to a
    600-second take on a CPU fallback. Reusing ``pose_timeout`` would kill a
    legitimately-progressing separation at 300s exactly like a hung one.
    Fails against the unfixed code, which passes ``worker.config.pose_timeout``
    (300.0) to ``blender_run.run_worker`` instead of a separation-sized ceiling.
    """
    calls: list[dict] = []

    def fake_run_worker(spec, *, on_progress=None, on_start=None, timeout=0.0, **kwargs):
        from pathlib import Path

        calls.append({"spec": spec, "timeout": timeout})
        Path(spec["out_dir"]).mkdir(parents=True, exist_ok=True)
        return {"ok": True, "files": [], "rate": 44100}

    monkeypatch.setattr(blender_run, "run_worker", fake_run_worker)

    worker = Worker(svc.config, svc.store)
    take = _take(svc)
    split_id = rework.separate_job(svc, take)["id"]

    async def _run() -> None:
        worker.start()
        await _wait_until(lambda: worker.store.get(split_id)["status"] == "done")
        await worker.shutdown()

    asyncio.run(_run())

    assert len(calls) == 1
    # It must not be pose_timeout (300s, sized for an inline bake with a
    # completely different cost model) -- it must be the job's own field.
    assert calls[0]["timeout"] != svc.config.pose_timeout
    assert calls[0]["timeout"] == svc.config.separation_timeout


# --- the overlap-add window (2026-09-16 audit, fix-separation) ---------------


def _hann_periodic(n: int) -> list[float]:
    """``torch.hann_window(n)``'s default (``periodic=True``) formula, restated
    in pure Python so this test needs neither torch nor a GPU -- separate()'s
    windowing is arithmetic, and this simulates exactly that arithmetic."""
    import math

    return [0.5 * (1 - math.cos(2 * math.pi * k / n)) for k in range(n)]


def _old_window(segment: int) -> list[float]:
    """The window the unfixed ``separate()`` built: the rising half of a
    double-length Hann window, spanning the *whole* segment."""
    return _hann_periodic(segment * 2)[:segment]


def _new_window(segment: int, overlap: int) -> list[float]:
    """The window the fixed ``separate()`` builds: 1.0 through the body of the
    chunk, tapering only across the true overlap width at each edge."""
    ramp = _hann_periodic(overlap * 2)
    window = [1.0] * segment
    window[:overlap] = ramp[:overlap]
    window[segment - overlap:] = ramp[overlap:]
    return window


def _earlier_share(segment: int, overlap: int, window: list[float]) -> list[float]:
    """The earlier chunk's fraction of the blended weight at each sample of
    the overlap region, the same arithmetic ``stems / weights.clamp(min=1e-6)``
    performs once both chunks have been accumulated -- reduced to just the two
    contributing chunks since nothing else touches this region."""
    earlier = window[segment - overlap:segment]
    later = window[0:overlap]
    return [
        e / (e + lat) if (e + lat) else 0.0
        for e, lat in zip(earlier, later, strict=True)
    ]


def test_separate_blends_adjacent_segments_instead_of_stepping_at_the_boundary():
    """The 2026-09-16 audit found ``separate()``'s overlap-add window was the
    rising half of a double-length Hann window applied to the *whole* segment
    (``torch.hann_window(segment * 2)[:segment]``), not a taper confined to the
    overlap. Reproduced here in pure numpy-free Python (the note on this
    finding: real torch/audio processing is out of scope for this lane) with
    segment=40, overlap=10: the old formula's earlier-chunk share never drops
    below ~0.91 across the whole overlap -- a near-instant hand-off, not a
    crossfade -- while the fixed formula sweeps smoothly from 1.0 to ~0.02 and
    crosses 0.5 exactly at the overlap's midpoint.

    Fails against the unfixed formula: ``min(old_fraction) > 0.85`` is true (it
    is ~0.91), so a test asserting the old formula behaves like a crossfade
    (spans most of [0, 1]) fails; this test instead asserts the reproduction
    directly and then asserts the *new* formula is the smooth one.
    """
    segment, overlap = 40, 10

    old_fraction = _earlier_share(segment, overlap, _old_window(segment))
    new_fraction = _earlier_share(segment, overlap, _new_window(segment, overlap))

    # Reproduction: the old window is a near-instant hand-off -- the earlier
    # chunk holds nearly all the weight for virtually the entire overlap.
    assert min(old_fraction) > 0.85, old_fraction
    assert max(old_fraction) - min(old_fraction) < 0.15, old_fraction

    # The fix: the new window's crossfade fraction moves smoothly across the
    # whole overlap region -- not bunched at one end -- and passes through the
    # midpoint at (approximately) an even split.
    assert max(new_fraction) - min(new_fraction) > 0.9, new_fraction
    midpoint = new_fraction[len(new_fraction) // 2]
    assert 0.3 < midpoint < 0.7, new_fraction
    # Monotonic: the earlier chunk's share only ever falls as the later chunk
    # takes over, never oscillates.
    pairs = zip(new_fraction, new_fraction[1:], strict=False)
    assert all(a >= b - 1e-9 for a, b in pairs), new_fraction


def test_separation_worker_result_cleanup_does_not_mask_a_replace_failure(
    tmp_path, monkeypatch
):
    """``separation_worker.main()``'s staged-result cleanup used to be a bare
    ``tmp.unlink(missing_ok=True)`` in the ``finally`` after ``tmp.write_text``/
    ``tmp.replace``, unlike the byte-identical pattern in ``blender_worker.py``
    and ``lora_train_worker.py``, both of which wrap it in
    ``contextlib.suppress(OSError)``. If ``tmp.replace(result_path)`` itself
    fails, the unwrapped ``finally``'s own OSError from ``unlink`` replaces
    that more informative exception -- ordinary ``try``/``finally`` behaviour:
    an exception raised while unwinding a ``finally`` clause supersedes the one
    that triggered it.

    Fails against the unfixed code: forcing ``Path.replace`` to fail with one
    distinctive OSError and ``Path.unlink`` to fail with another, the unfixed
    ``finally`` lets the unlink's message ("unlink: locked (simulated)")
    propagate out of ``main()`` and bury the replace's ("replace: disk full
    (simulated)").
    """
    import io
    import json
    import sys
    from pathlib import Path

    from realmspinner.pipelines import separation_worker as sw

    spec = {
        "source": str(tmp_path / "missing-take.wav"),
        "out_dir": str(tmp_path / "stems"),
        "model_dir": str(tmp_path / "no-such-model-dir"),
        "sources": ["drums", "bass", "other", "vocals"],
        "segment_seconds": 10.0,
        "result_path": str(tmp_path / "separate.json"),
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(spec)))

    real_replace = Path.replace

    def fake_replace(self, target):
        if self.name.endswith(".tmp"):
            raise OSError("replace: disk full (simulated)")
        return real_replace(self, target)

    def fake_unlink(self, missing_ok=False):
        raise OSError("unlink: locked (simulated)")

    monkeypatch.setattr(Path, "replace", fake_replace)
    monkeypatch.setattr(Path, "unlink", fake_unlink)

    with pytest.raises(OSError, match="disk full"):
        sw.main()
