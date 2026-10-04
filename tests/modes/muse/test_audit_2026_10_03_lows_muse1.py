"""The 2026-10-03 audit, Low batch muse1 (muse-13 .. muse-24).

One file for the whole batch even though the findings sit in different trees
(the studio mode, the separation child, the queue, the service): the fixer rule
is one new test file per agent, and every one of them is a Muse take's problem.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.studio.modes.muse import fileio as muse_io
from realmspinner.studio.modes.muse import mode as muse_mode

from .test_muse_mode import FakeCtx
from .test_muse_player import _loaded
from .test_muse_player import device as _device

RATE = 44100

# The recorder fixture, re-registered under its own name here.
device = _device


@pytest.fixture
def ctx(tmp_path):
    one = FakeCtx(tmp_path)
    one.cache = SimpleNamespace(jobs=[{"id": "a"}], total=1)
    return one


def _data_chunk(wav: bytes) -> bytes:
    at = wav.index(b"data")
    size = int.from_bytes(wav[at + 4 : at + 8], "little")
    return wav[at + 8 : at + 8 + size]


def _smpl_loop(data: bytes) -> tuple[int, int]:
    body = data.index(b"smpl") + 8
    return (
        int.from_bytes(data[body + 44 : body + 48], "little"),
        int.from_bytes(data[body + 48 : body + 52], "little"),
    )


# --- muse-13 ------------------------------------------------------------------


def _truncating_positions(count: int) -> list[int]:
    """Sample offsets that ``int((n / RATE) * RATE)`` hands back one short."""
    found = [n for n in range(44100, 44100 + 4000) if int((n / RATE) * RATE) != n]
    assert len(found) >= count, "the probe found no positions that truncate"
    return found[:count]


def test_a_candidate_chosen_from_the_finder_comes_back_as_the_same_sample_offsets(ctx):
    """``choose_candidate`` hands ``samples / rate`` seconds to ``set_region`` and
    ``loop_cache_key`` turned them back with ``int(seconds * rate)``, which
    drops one sample on about 7% of positions (44128 came back as 44127)."""
    one = _loaded(ctx, seconds=3.0)
    starts = _truncating_positions(5)
    for start in starts:
        end = start + 20000
        one.candidates = [SimpleNamespace(start=start, end=end)]
        muse_mode.choose_candidate(ctx, 0)
        key = muse_io.loop_cache_key(one)
        assert key is not None
        assert key[0] == start and key[1] == end, (start, end, key)


def test_export_with_points_writes_the_same_sample_offsets_the_finder_chose(
    ctx, monkeypatch, tmp_path
):
    one = _loaded(ctx, seconds=3.0)
    start = _truncating_positions(1)[0]
    end = start + 20000
    one.candidates = [SimpleNamespace(start=start, end=end)]
    muse_mode.choose_candidate(ctx, 0)
    one.xfade_ms = 0.0
    out = tmp_path / "pts.wav"
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: out)

    muse_io.export_with_points(ctx, one)

    loop_start, loop_end = _smpl_loop(out.read_bytes())
    # ``smpl`` stores the last sample *inside* the loop, so ``end - 1``.
    assert (loop_start, loop_end) == (start, end - 1)


# --- muse-14 ------------------------------------------------------------------


def test_an_exported_unchanged_take_matches_sample_for_sample_including_the_negative_peak():
    pcm = np.array([[0, 0], [-32768, 32767], [-32767, 1], [-1, -32768]], dtype=np.int16)

    wav = muse_io._wav(pcm, RATE)

    out = np.frombuffer(_data_chunk(wav), dtype="<i2").reshape(-1, 2)
    np.testing.assert_array_equal(out, pcm)


# --- muse-15 ------------------------------------------------------------------


def test_the_first_samples_of_a_stem_are_not_attenuated_by_the_overlap_window(
    tmp_path, monkeypatch
):
    """The first chunk's Hann window starts at exactly 0 and ``weights.clamp(
    min=1e-6)`` caps the divisor, so the head of every stem was silenced and
    the four stems did not sum back to the take there."""
    sf = pytest.importorskip("soundfile")
    torch = pytest.importorskip("torch")
    torchaudio_models = pytest.importorskip("torchaudio.models")

    from realmspinner.pipelines import separation_worker as sw

    rate = 8000
    seconds = 3.0
    total = int(rate * seconds)
    t = np.arange(total) / rate
    # A cosine over whole periods: zero mean (the worker adds the mean back to
    # every stem, so an offset would be counted four times by this identity
    # model, which says nothing about the head) and non-zero at sample 0.
    audio = (0.5 * np.cos(2 * np.pi * 220 * t)).astype(np.float32)
    stereo = np.stack([audio, audio], axis=1)
    source = tmp_path / "track.wav"
    sf.write(str(source), stereo, rate, subtype="FLOAT", format="WAV")

    class _Identity:
        def __init__(self, sources):
            self.n = len(sources)

        def load_state_dict(self, _state):
            return None

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, chunk):
            # (1, 2, T) -> (1, sources, 2, T), each stem a quarter of the mix.
            return chunk[:, None].repeat(1, self.n, 1, 1) / self.n

    monkeypatch.setattr(
        torchaudio_models, "hdemucs_high", lambda sources: _Identity(sources)
    )
    monkeypatch.setattr(torch, "load", lambda *a, **k: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sw, "_emit", lambda *a, **k: None)

    out_dir = tmp_path / "stems"
    sources = ["drums", "bass", "other", "vocals"]
    result = sw.separate(
        {
            "source": str(source),
            "out_dir": str(out_dir),
            "model_dir": str(tmp_path),
            "sources": sources,
            "segment_seconds": 1.0,
            "result_path": str(tmp_path / "r.json"),
        }
    )
    assert result["ok"]

    total_sum = sum(
        sf.read(str(out_dir / f"{name}.wav"), dtype="float32", always_2d=True)[0]
        for name in sources
    )
    # 16-bit quantisation of four stems is the only allowed difference.
    head = slice(0, 200)
    np.testing.assert_allclose(total_sum[head, 0], audio[head], atol=4e-4)
    tail = slice(total - 200, total)
    np.testing.assert_allclose(total_sum[tail, 0], audio[tail], atol=4e-4)


# --- muse-16 ------------------------------------------------------------------


def test_a_refused_play_with_a_working_device_does_not_claim_there_is_no_audio_device(
    ctx, device, monkeypatch
):
    """``unavailable_reason()`` returns its "No audio device" sentence
    unconditionally, so the ``or "could not play that take"`` branch was dead and
    an empty take was reported as missing hardware."""
    from realmspinner.studio.modes.sirens import audio as real_audio

    device.refuses = True  # play() returns False ...
    device.available = lambda: True  # ... but the device itself is fine
    device.unavailable_reason = real_audio.unavailable_reason  # the real, unconditional one

    one = _loaded(ctx, seconds=1.0)
    muse_mode._play_from(ctx, one, 0.0)

    assert ctx.toasts, "a refused play must say something"
    for message, _kind in ctx.toasts:
        assert "No audio device" not in message


def test_a_refused_play_without_a_device_still_says_there_is_no_audio_device(ctx, device):
    from realmspinner.studio.modes.sirens import audio as real_audio

    device.refuses = True  # available() is False in the recorder
    device.unavailable_reason = real_audio.unavailable_reason
    one = _loaded(ctx, seconds=1.0)
    muse_mode._play_from(ctx, one, 0.0)

    assert any("No audio device" in message for message, _ in ctx.toasts)


# --- muse-17 ------------------------------------------------------------------


def test_a_cancelled_split_removes_its_staging_files(tmp_path):
    from realmspinner import _q_jobs

    class _Config:
        def job_dir(self, job_id):
            return tmp_path / job_id

    source_id = "abc123abc123"
    stems = tmp_path / source_id / "stems"
    stems.mkdir(parents=True)
    names = ("drums", "bass", "other", "vocals")
    staged = [stems / f".{name}.wav.tmp" for name in names] + [stems / ".stems.json.tmp"]
    for path in staged:
        path.write_bytes(b"x" * 8)

    job = {"id": "def456def456", "kind": "separate", "params": {"source_job": source_id}}
    _q_jobs.JobOps._discard_artifacts(SimpleNamespace(config=_Config()), job)

    assert [p.name for p in staged if p.exists()] == []


# --- muse-18 ------------------------------------------------------------------


def test_the_stem_resplit_refusal_names_an_action_a_door_provides():
    """The refusal told the user to "delete these stems", but no door deletes a
    take's ``stems/`` on its own (deleting the ``separate`` row removes only that
    row's own directory)."""
    from realmspinner.service import _jobs_resubmit as resubmit
    from realmspinner.service.errors import Invalid

    svc = SimpleNamespace(
        require_job=lambda job_id: {
            "id": job_id,
            "kind": "separate",
            "status": "done",
            "params": {},
        }
    )
    with pytest.raises(Invalid) as caught:
        resubmit.rerun_job(svc, "abc123abc123")

    message = str(caught.value)
    assert "no seed to change" in message  # the phrase tests/test_separation.py pins
    assert "delete" not in message.lower()


# --- muse-19 ------------------------------------------------------------------

_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}


def test_music_worker_docstring_names_the_same_modification_count_as_the_attribution():
    from realmspinner.pipelines import music_worker

    root = Path(music_worker.__file__).parent
    attribution = (root / "acestep" / "ATTRIBUTION.md").read_text(encoding="utf-8")
    section = attribution.split("## The modifications", 1)[1].split("\n## ", 1)[0]
    declared = len(re.findall(r"^\d+\. ", section, flags=re.MULTILINE))
    assert declared >= 1

    said = re.search(r"the (\w+) modifications", music_worker.__doc__ or "")
    assert said is not None, "the docstring should name how many modifications there are"
    assert _WORDS[said.group(1)] == declared


def test_the_music_stage_docstring_does_not_deny_its_follow_up():
    from realmspinner import _q_music

    doc = _q_music.__doc__ or ""
    assert "no follow-up" not in doc
    assert "separate" in doc


# --- muse-20 ------------------------------------------------------------------


def test_separate_checks_host_commit_headroom_against_the_models_declared_peak(
    tmp_path, monkeypatch
):
    from realmspinner import _q_music, models
    from realmspinner import queue as queue_mod
    from realmspinner.pipelines import blender_run

    asked: list[float] = []

    async def fake_settled(when, remedy, need_gib):
        asked.append(need_gib)

    class _Boom(Exception):
        pass

    def fake_run_worker(*_a, **_k):
        raise _Boom

    monkeypatch.setattr(queue_mod, "_require_commit_headroom_settled", fake_settled)
    monkeypatch.setattr(blender_run, "run_worker", fake_run_worker)

    class _Config:
        vram_exclusive = False
        t2i_model_root = tmp_path / "models"
        separation_timeout = 10.0

        def job_dir(self, job_id):
            return tmp_path / job_id

    worker = SimpleNamespace(
        config=_Config(),
        trellis=SimpleNamespace(stop=lambda: None),
        progress=SimpleNamespace(update=lambda *a, **k: None),
        _note_blender=lambda *a, **k: None,
        _cancel=None,
    )
    job = {
        "id": "def456def456",
        "kind": "separate",
        "params": {"source_job": "abc123abc123"},
    }

    with pytest.raises(_Boom):
        asyncio.run(_q_music.MusicOps._separate(worker, job))

    spec = models.SEPARATION_MODELS[models.DEFAULT_SEPARATION]
    assert asked == [queue_mod._host_peak_gib(spec)]


# --- muse-21 ------------------------------------------------------------------


def _fake_pipe(wrote: list[Path]):
    def pipe(*, save_path, **_kwargs):
        track = Path(save_path)
        track.write_bytes(b"RIFF")
        stray = track.with_name(track.stem + "_input_params.json")
        stray.write_text("{}", encoding="utf-8")
        wrote.append(stray)

    return pipe


def test_a_finished_take_leaves_no_input_params_json(tmp_path):
    from realmspinner.pipelines import music_worker

    server = music_worker._Server("ace_step_v1", str(tmp_path))
    wrote: list[Path] = []
    server._pipe = _fake_pipe(wrote)
    out = tmp_path / "job" / "track.wav"

    reply = server.op_generate({"output": str(out), "prompt": "x"}, lambda _m: None)

    assert reply["kind"] == "done"
    assert out.exists()
    assert wrote and not wrote[0].exists()


def test_a_cancelled_take_leaves_no_input_params_json(tmp_path):
    from realmspinner import _q_jobs

    class _Config:
        def job_dir(self, job_id):
            return tmp_path / job_id

    job_dir = tmp_path / "def456def456"
    job_dir.mkdir()
    (job_dir / "track.wav").write_bytes(b"x")
    (job_dir / "track_input_params.json").write_text("{}", encoding="utf-8")

    job = {"id": "def456def456", "kind": "music", "params": {}}
    _q_jobs.JobOps._discard_artifacts(SimpleNamespace(config=_Config()), job)

    assert not (job_dir / "track_input_params.json").exists()


# --- muse-22 ------------------------------------------------------------------


def test_a_failed_or_cancelled_take_does_not_say_it_has_not_finished_yet():
    from realmspinner.studio.modes.muse.ui.panes import results

    for status, word in (("error", "failed"), ("cancelled", "cancelled")):
        reason = results._ready_reason(False, status)
        assert reason and "not finished yet" not in reason, (status, reason)
        assert word in reason
        assert word in results._stems_reason(False, False, status)
    for status in ("queued", "running", ""):
        assert "not finished yet" in results._ready_reason(False, status)
    assert results._ready_reason(True, "done") == ""


# --- muse-23 ------------------------------------------------------------------


def test_arrow_keys_in_the_custom_seconds_field_do_not_cycle_the_duration_pills(
    ctx, monkeypatch
):
    from contextlib import contextmanager

    from realmspinner.studio.modes.muse.ui import brief as muse_brief

    class _Key:
        left_arrow = "left"
        right_arrow = "right"

    class _Io:
        want_text_input = True  # the Custom seconds field has the caret

    class _FakeImgui:
        Key = _Key

        def __getattr__(self, _name):
            return lambda *a, **k: None

        def is_key_pressed(self, key):
            return key == _Key.right_arrow

        def get_io(self):
            return _Io()

    @contextmanager
    def focused_item(*_a, **_k):
        yield True

    monkeypatch.setattr(muse_brief, "imgui", _FakeImgui())
    monkeypatch.setattr(muse_brief.focus, "item", focused_item)
    monkeypatch.setattr(muse_brief.widgets, "field_label", lambda *a, **k: None)
    monkeypatch.setattr(muse_brief, "sp", lambda value: value)
    monkeypatch.setattr(
        muse_brief.controls, "segmented_choice", lambda _id, _opts, current, **k: (False, current)
    )
    monkeypatch.setattr(muse_brief, "_ring", lambda *a, **k: None)
    monkeypatch.setattr(muse_brief, "_duration_custom", lambda *a, **k: None)
    ctx.state.clear_field_error = lambda *_a: None

    state = muse_mode.ensure(ctx)
    state.duration_custom = True
    form = {"duration": 143.0}

    muse_brief._duration(ctx, form)

    assert form["duration"] == 143.0
    assert state.duration_custom is True


# --- muse-24 ------------------------------------------------------------------


def test_muse_mode_docstring_names_the_real_sirens_engine_path():
    doc = muse_mode.__doc__ or ""
    assert "``studio/sirens/``" not in doc
    assert "studio/modes/sirens/engine/" in doc
    repo = Path(__file__).resolve().parents[3]
    assert (repo / "src" / "realmspinner" / "studio" / "modes" / "sirens" / "engine").is_dir()

