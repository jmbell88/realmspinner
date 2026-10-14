"""The 2026-10-14 audit, fixer F51: the service-layer and music findings it closed.

One file for the batch, each test named for the claim it pins. The music door
tests copy the admission patches ``test_jobs_music_derive.py`` uses, so what is
under test is the door's own rule and not the machine's weights or card.
"""

from __future__ import annotations

import io
import re
import threading
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import realmspinner
from realmspinner import _q_music, vram
from realmspinner.service import Conflict, Invalid
from realmspinner.service import _jobs_music as music_door
from realmspinner.service import _jobs_rework as rework
from realmspinner.service import characters as svc_characters
from realmspinner.service import jobs as svc_jobs


@pytest.fixture
def admitted(monkeypatch):
    monkeypatch.setattr(music_door, "check_weights", lambda svc, kind, params: None)
    monkeypatch.setattr(music_door, "check_vram", lambda svc, kind, stage, params: None)
    monkeypatch.setattr(rework, "check_weights", lambda svc, kind, params: None)
    monkeypatch.setattr(rework, "check_vram", lambda svc, kind, stage, params: None)


def _wav(seconds: float, rate: int = 44100) -> bytes:
    frames = np.zeros((int(seconds * rate), 2), dtype="<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())
    return out.getvalue()


def _take(svc, status: str = "done", *, lyrics: str = "", with_audio: bool = True) -> str:
    """A 30 s take on disk in the given status. -> its job id."""
    job_id = svc.store.create(
        "music", "dark ambient", {"duration": 30.0, "lyrics": lyrics}, stage="music"
    )
    svc.config.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    if with_audio:
        (svc.config.job_dir(job_id) / "track.wav").write_bytes(_wav(30.0))
    svc.store.set_status(job_id, status)
    return job_id


# --- service-26: the busy refusal names what is running ----------------------


def test_deleting_a_mesh_with_a_running_remesh_names_the_remesh(svc):
    source = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(source, "done")
    svc.store.create(
        "remesh", "", {"source_job": source}, stage="model", status="running"
    )

    with pytest.raises(Conflict) as caught:
        svc_jobs.delete_job(svc, source)

    message = str(caught.value)
    assert "remesh" in message, message
    assert "rig" not in message and "sprite sheet" not in message, message
    assert svc.store.get(source) is not None


def test_the_busy_refusal_names_every_kind_that_is_running(svc):
    source = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(source, "done")
    svc.store.create("rig", "", {"source_job": source}, stage="model", status="running")
    svc.store.create(
        "retexture", "", {"source_job": source}, stage="model", status="queued"
    )

    with pytest.raises(Conflict) as caught:
        svc_jobs.delete_job(svc, source)

    message = str(caught.value)
    assert "rig" in message and "re-texture" in message, message


# --- muse-11: a failed take is not told it has no image ----------------------


@pytest.mark.parametrize("status", ["error", "cancelled"])
def test_deriving_from_a_failed_take_does_not_say_image(svc, admitted, status):
    take = _take(svc, status)

    with pytest.raises(Invalid) as caught:
        music_door.derive_music_job(svc, take, task="retake")

    message = str(caught.value)
    assert "image" not in message, message
    assert "audio" in message, message
    # "yet" promises a retry that a failed take will never allow.
    assert "yet" not in message, message
    assert caught.value.field == "parent_id"


def test_deriving_from_a_take_still_in_the_queue_keeps_its_yet(svc, admitted):
    take = _take(svc, "queued")

    with pytest.raises(Invalid) as caught:
        music_door.derive_music_job(svc, take, task="retake")

    assert "yet" in str(caught.value)


def test_the_image_sentences_still_say_image_for_a_reference():
    from realmspinner.service.validation import not_done_message

    assert not_done_message("That reference", "error") == (
        "That reference failed, so it has no image."
    )
    assert not_done_message("That reference", "paused") == "That reference is paused."


# --- muse-14: blank lyrics are an instrumental -------------------------------


@pytest.mark.parametrize("blank", ["   ", "\n", " \n\t\n "])
def test_whitespace_only_lyrics_are_stored_as_an_instrumental(svc, admitted, blank):
    made = music_door.create_music_job(svc, prompt="dark ambient, dungeon", lyrics=blank)

    assert svc.store.get(made["id"])["params"]["lyrics"] == ""


def test_real_lyrics_keep_their_whitespace(svc, admitted):
    lyrics = "[verse]\nla la la\n"
    made = music_door.create_music_job(svc, prompt="folk", lyrics=lyrics)

    assert svc.store.get(made["id"])["params"]["lyrics"] == lyrics


def test_whitespace_only_edit_lyrics_are_stored_as_an_instrumental(svc, admitted):
    take = _take(svc, lyrics="la la la")

    made = music_door.derive_music_job(svc, take, task="edit", edit_lyrics=" \n ")

    params = svc.store.get(made["id"])["params"]
    assert params["edit_lyrics"] == ""
    assert params["lyrics"] == ""


# --- muse-10: an errored take cannot be split --------------------------------


@pytest.mark.parametrize("status", ["error", "cancelled"])
def test_an_errored_take_cannot_be_split_into_stems(svc, admitted, status):
    take = _take(svc, status)  # a partial track.wav sits beside the failed row

    with pytest.raises(Invalid) as caught:
        rework.separate_job(svc, take)

    assert caught.value.field == "source_job"
    assert "audio" in str(caught.value)
    assert not [j for j in svc.store.list(100) if j["kind"] == "separate"]


# --- service-24: a reroll of a Blender kind is refused without Blender -------


def _no_blender(monkeypatch):
    monkeypatch.setattr(
        "realmspinner.doctor.blender_check",
        lambda *a, **k: SimpleNamespace(ok=False, detail="no bpy"),
    )


@pytest.mark.parametrize(
    ("kind", "extra"),
    [("rig", {}), ("remesh", {"remesh_profile": "medium"}), ("retexture", {})],
)
def test_rerun_job_refuses_a_blender_reroll_when_blender_is_absent(
    svc, monkeypatch, kind, extra
):
    _no_blender(monkeypatch)
    mesh = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(mesh, "done")
    row = svc.store.create(
        kind, "x", {"source_job": mesh, **extra}, stage="model", status="done"
    )
    before = len(svc.store.list(1000))

    with pytest.raises(Invalid) as caught:
        svc_jobs.rerun_job(svc, row)

    assert "Blender" in str(caught.value)
    assert len(svc.store.list(1000)) == before, "a refused reroll minted a row"


def test_rerun_job_refuses_a_rig_reroll_when_blender_is_absent(svc, monkeypatch):
    _no_blender(monkeypatch)
    mesh = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(mesh, "done")
    row = svc.store.create("rig", "x", {"source_job": mesh}, stage="model", status="done")

    with pytest.raises(Invalid, match="Rigging needs Blender"):
        svc_jobs.rerun_job(svc, row)


def test_a_text_reroll_is_not_refused_for_want_of_blender(svc, monkeypatch):
    """The refusal is for the kinds whose worker runs Blender, not for every row."""
    source = svc_jobs.create_job(svc, kind="text", prompt="a goblin")["id"]
    svc.store.set_status(source, "done")
    _no_blender(monkeypatch)

    assert svc_jobs.rerun_job(svc, source)["id"]


# --- poser-04: the character dry run prices what is built --------------------


def test_a_character_plan_estimate_equals_creates_and_is_not_refused_by_a_sheet_cell_limit(
    svc,
):
    result = svc_characters.recipe_from_prompt(svc, "a wolf")
    assert result["estimate_minutes"] == pytest.approx(svc_characters.estimate_minutes(0))
    assert result["estimate_minutes"] == pytest.approx(svc_characters.RIG_MINUTES)

    _a_plan_the_retired_sheet_could_not_hold(svc)


def test_a_plan_the_retired_sheet_could_not_hold_is_not_refused(svc):
    _a_plan_the_retired_sheet_could_not_hold(svc)


def _a_plan_the_retired_sheet_could_not_hold(svc) -> None:
    """Sixteen directions at 256 px with every humanoid clip: a sheet of that
    would be over both the cell and the atlas limit, and no sheet is built."""
    from realmspinner.clips import clip_timing

    every_clip = {name: None for name in clip_timing("humanoid")}
    big = svc_characters.recipe_from_prompt(
        svc,
        "a knight",
        overrides={"logical_size": 256, "directions": 16, "animations": every_clip},
    )
    assert big["estimate_minutes"] == pytest.approx(svc_characters.RIG_MINUTES)
    assert big["recipe"]["directions"] == 16
    assert big["recipe"]["logical_size"] == 256


def test_recipe_from_prompt_prices_a_character_without_sheet_cells_and_never_refuses_on_a_cell_limit(  # noqa: E501
    svc,
):
    result = svc_characters.recipe_from_prompt(svc, "a fire ogre")

    # The cleanup round dropped the ``cells`` key altogether once the Familiar plan summary
    # stopped reading it.
    assert "cells" not in result
    assert result["estimate_minutes"] == pytest.approx(svc_characters.estimate_minutes(0))


def test_a_dry_run_refuses_only_what_create_character_refuses(svc):
    """The dry run still refuses a recipe the build would refuse."""
    with pytest.raises(Invalid) as caught:
        svc_characters.recipe_from_prompt(
            svc, "a knight", overrides={"animations": {"moonwalk": None}}
        )
    assert caught.value.field == "animations"


# --- poser-13: the options carry what a form reads ---------------------------


def test_the_character_options_hold_exactly_the_keys_a_form_reads(svc):
    options = svc_characters.character_options(svc)

    # ``default_recipe`` went too, once the last test that asserted it did (the
    # 2026-10-14 audit, cleanup round).
    assert set(options) == {"archetypes", "families", "channels", "themes"}
    for row in options["archetypes"]:
        assert set(row) == {"key", "label", "template"}


# --- docs-79 / poser-19: prose that cites what the tree lacks ----------------


def test_service_docstrings_cite_no_function_the_tree_lacks():
    """``poselib.<name>`` and ``poser_mode.<name>`` in a comment or docstring must
    name something that exists. ``poser_mode`` is gone as a module, and
    ``poselib`` lost ``validate_record`` and ``next_copy_name`` with Poser."""
    from realmspinner import poselib

    root = Path(realmspinner.__file__).parent
    files = sorted((root / "service").glob("*.py")) + [
        root / "clips.py",
        root / "clipmaps.py",
    ]
    missing = []
    for path in files:
        text = path.read_text("utf-8")
        for match in re.finditer(r"\bposer_mode\.\s*\w+", text):
            missing.append(f"{path.name}: {match.group(0)!r}")
        for match in re.finditer(r"\bposelib\.(\w+)", text):
            name = match.group(1)
            if name != "py" and not hasattr(poselib, name):
                missing.append(f"{path.name}: poselib.{name}")
    assert not missing, missing


def test_the_clip_door_no_longer_calls_itself_poser_clip_editors_door():
    root = Path(realmspinner.__file__).parent
    source = (root / "service" / "clips.py").read_text("utf-8")
    assert "the door Poser's clip editor calls" not in source
    assert "a character sheet needs a clip called" not in source
    assert "cannot fill a character sheet" not in source


# --- muse-13 / muse-15: the music queue and its three task tables ------------

#: ``pipeline_ace_step.__call__``'s own arithmetic for a task with no source
#: audio: ``frame_length = int(duration * 44100 / 512 / 8)``.
_SAMPLES_PER_LATENT_FRAME = 512 * 8
_LONGEST_TAKE_FRAMES = int(music_door.MAX_DURATION * 44100 / _SAMPLES_PER_LATENT_FRAME)


def _sampler_frames(duration: float) -> int:
    return int(duration * 44100 / 512 / 8)


def _retake_params(frames: int) -> dict[str, Any]:
    # What ``derive_music_job`` records: the parent's duration read back off
    # its file as ``samples / rate``.
    return {"task": "retake", "duration": frames * _SAMPLES_PER_LATENT_FRAME / 44100}


def test_a_retake_has_the_same_frame_length_as_its_parent_for_every_whole_frame_count():
    # The premise: sent as recorded, the sampler's int() drops a frame for some
    # lengths -- 11.1 s is one -- or this test would prove nothing.
    short = [
        f
        for f in range(1, _LONGEST_TAKE_FRAMES + 1)
        if _sampler_frames(_retake_params(f)["duration"]) != f
    ]
    assert short, "the recorded duration no longer drops a latent frame"

    wrong = [
        f
        for f in range(1, _LONGEST_TAKE_FRAMES + 1)
        if _sampler_frames(_q_music._sampler_duration(_retake_params(f))) != f
    ]
    assert not wrong, wrong[:10]


def test_only_a_retake_is_padded():
    for task in ("extend", "repaint", "edit", "loop", "audio2audio", ""):
        params = {"task": task, "duration": 30.0}
        assert _q_music._sampler_duration(params) == 30.0, task
    assert _q_music._sampler_duration({}) == 60.0


def _wav_frames(frames_n: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(100)
        handle.writeframes(np.zeros((frames_n, 2), dtype="<i2").tobytes())
    return out.getvalue()


class _Cancel:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.committed = False

    def commit(self) -> None:
        self.committed = True


class _Config:
    def __init__(self, root: Path) -> None:
        self.root = root

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id


class _Store:
    def __init__(self) -> None:
        self.saved: dict[str, Any] = {}

    def set_params(self, job_id: str, params: dict[str, Any]) -> None:
        self.saved[job_id] = dict(params)


class _Client:
    last_recipe = None

    def __init__(self) -> None:
        self.asked: list[float] = []

    def generate(self, prompt, output, *, audio_duration, **_kw):
        self.asked.append(audio_duration)
        output.write_bytes(_wav_frames(100))


async def test_a_retake_asks_the_sampler_for_its_parents_frame_count(tmp_path):
    client = _Client()

    async def _acquire_music(spec):
        return client, False

    async def _release_music(client, spec, *, handoff):
        return None

    worker = SimpleNamespace(
        config=_Config(tmp_path),
        store=_Store(),
        _cancel=_Cancel(),
        _music_state=lambda job_id, s: None,
        _music_step=lambda job_id, i, n: None,
        _music_client=None,
        _acquire_music=_acquire_music,
        _release_music=_release_music,
    )
    # 120 frames is 11.1 s, a length whose recorded duration floors to 119.
    for frames in (120, 184, 646):
        job = {"id": f"take{frames}", "prompt": "x", "params": _retake_params(frames)}
        await _q_music.MusicOps._music(worker, job)
        assert _sampler_frames(client.asked[-1]) == frames, frames


# --- muse-jobs-07: three hand-written task tables, one sweep ------------------


def test_every_muse_task_is_known_to_the_door_the_worker_and_the_pricer():
    # A task in TASKS and not in _UPSTREAM_TASK clears the door and then raises
    # "unknown music task" in the worker after it has queued; one missing from
    # TASKS_WITH_SOURCE_AUDIO is under-priced -- admitted, then out of memory at
    # load. The three tables are written by hand in three files.
    assert set(music_door.TASKS) == set(_q_music._UPSTREAM_TASK)
    assert {t for t in music_door.TASKS if t != "retake"} == vram.TASKS_WITH_SOURCE_AUDIO
