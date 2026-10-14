"""The 2026-10-14 audit, fixer F52: the service gates, assets and Familiar host.

Each test names the claim it pins and fails against the code the audit read.
The prose findings (stale comments and docstrings) live in
``tests/test_audit_2026_10_14_f52_root.py``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image, ImageOps

from realmspinner import fetch
from realmspinner.kernels.rig import cliplib
from realmspinner.service import clips as svc_clips
from realmspinner.service import downloads as svc_downloads
from realmspinner.service import familiar as svc_familiar
from realmspinner.service import generation_log
from realmspinner.service import judge as svc_judge
from realmspinner.service import loras as svc_loras
from realmspinner.service import verdicts as svc_verdicts
from realmspinner.service.core import RealmspinnerService
from realmspinner.service.errors import Failed
from realmspinner.service.familiar import FamiliarRefusal

TEMPLATE = "humanoid"


@pytest.fixture(autouse=True)
def _fresh_clip_state():
    cliplib.set_user_clip_dir(None)
    cliplib.invalidate_clips()
    yield
    cliplib.set_user_clip_dir(None)
    cliplib.invalidate_clips()


# --- poser-poses-02 -----------------------------------------------------------


def test_an_orphaned_user_clip_library_does_not_override_the_shipped_clips_while_nothing_can_revert_it(  # noqa: E501
    svc,
):
    """Every service start pointed ``cliplib`` at ``data_dir/poser/clips``, so a
    clip edit saved under the old Poser beat the shipped library in every
    animated bake and nothing in the app could show or revert it."""
    view = svc_clips.library(svc, TEMPLATE)
    payload = {"space": view["space"], "poses": view["poses"], "clips": view["clips"]}
    current = payload["clips"][0].get("easing") or "linear"
    payload["clips"][0]["easing"] = next(e for e in sorted(cliplib.EASINGS) if e != current)
    svc_clips.save(svc, TEMPLATE, payload)
    edited = cliplib.clip_library(TEMPLATE)["clips"][0]["easing"]
    assert edited != current

    # A later process: the file is on disk, nothing has asked a clip door yet.
    cliplib.set_user_clip_dir(None)
    cliplib.invalidate_clips()
    RealmspinnerService(svc.config, svc.store)

    assert cliplib.user_clip_dir() is None
    assert cliplib.clip_library(TEMPLATE)["clips"][0]["easing"] == current
    # The clip doors still arm it themselves, so the editor and the importer
    # keep working on the user's copy.
    assert svc_clips.library(svc, TEMPLATE)["edited"] is True
    assert cliplib.clip_library(TEMPLATE)["clips"][0]["easing"] == edited


# --- service-gates-02 ---------------------------------------------------------


def _staging(tmp_path: Path, name: str, files: dict[str, bytes]) -> Path:
    staging = tmp_path / name
    staging.mkdir()
    for file, data in files.items():
        (staging / file).write_bytes(data)
    return staging


def test_a_second_download_into_a_shared_directory_keeps_the_first_downloads_manifest_record(
    tmp_path,
):
    """``repos[job.repo_id] = record`` rewrote the whole entry, so the mmproj row
    (same repo and directory as the Gemma weights) replaced the weights' file
    list and digests, and ``verify_manifest`` stopped covering the 7 GB file."""
    dest = tmp_path / "familiar"
    dest.mkdir()
    first = fetch.Job(repo_id="owner/gemma", dest=dest, filenames=("big.gguf",))
    second = fetch.Job(repo_id="owner/gemma", dest=dest, filenames=("mmproj.gguf",))

    one = _staging(tmp_path, "s1", {"big.gguf": b"weights"})
    svc_downloads._stage_manifest_for(first, one)
    # Published the way the move does, so the next job reads it.
    (dest / "big.gguf").write_bytes(b"weights")
    (dest / fetch.MANIFEST_NAME).write_bytes((one / fetch.MANIFEST_NAME).read_bytes())

    two = _staging(tmp_path, "s2", {"mmproj.gguf": b"projector"})
    svc_downloads._stage_manifest_for(second, two)

    repos = json.loads((two / fetch.MANIFEST_NAME).read_text())["repos"]
    record = repos["owner/gemma"]
    assert set(record["digests"]) == {"big.gguf", "mmproj.gguf"}
    assert set(record["files"]) == {"big.gguf", "mmproj.gguf"}


def test_a_manifest_does_not_carry_forward_a_file_that_is_gone(tmp_path):
    dest = tmp_path / "familiar"
    dest.mkdir()
    first = fetch.Job(repo_id="owner/gemma", dest=dest)
    second = fetch.Job(repo_id="owner/gemma", dest=dest)
    one = _staging(tmp_path, "s1", {"gone.gguf": b"x"})
    svc_downloads._stage_manifest_for(first, one)
    (dest / fetch.MANIFEST_NAME).write_bytes((one / fetch.MANIFEST_NAME).read_bytes())
    two = _staging(tmp_path, "s2", {"new.gguf": b"y"})
    svc_downloads._stage_manifest_for(second, two)
    record = json.loads((two / fetch.MANIFEST_NAME).read_text())["repos"]["owner/gemma"]
    assert set(record["digests"]) == {"new.gguf"}


# --- service-gates-06 ---------------------------------------------------------


def test_disk_usage_includes_the_familiar_runtime_and_models_directories(svc, tmp_path):
    """``disk_usage`` walked three roots and never Familiar's runtime directory,
    nor its models directory once ``REALMSPINNER_FAMILIAR_MODELS`` moved it out
    from under the model root."""
    svc.config.t2i_model_root.mkdir(parents=True, exist_ok=True)
    svc.config.familiar_runtime_dir = tmp_path / "elsewhere" / "llama"
    svc.config.familiar_models_dir = tmp_path / "elsewhere" / "familiar-models"
    svc.config.familiar_runtime_dir.mkdir(parents=True)
    svc.config.familiar_models_dir.mkdir(parents=True)
    before = svc_downloads.disk_usage(svc)

    (svc.config.familiar_runtime_dir / "llama-server.exe").write_bytes(b"x" * 1000)
    (svc.config.familiar_models_dir / "gemma.gguf").write_bytes(b"y" * 500)

    found = svc_downloads.disk_usage(svc)
    assert found["files"] - before["files"] == 2
    assert found["bytes"] - before["bytes"] == 1500


# --- service-gates-14 ---------------------------------------------------------


def test_a_failed_delete_after_staging_is_reclaimed_by_the_next_sweep(svc, monkeypatch):
    """A delete that failed after the rename-aside left a hidden multi-GB
    ``.trash-*`` tree that only the next *uninstall* would ever clear."""
    root = Path(svc.config.t2i_model_root)
    victim = root / "some-model"
    victim.mkdir(parents=True)
    (victim / "weights.bin").write_bytes(b"w" * 64)

    def locked(path, trash):
        raise Failed("another process holds these files")

    monkeypatch.setattr(svc_downloads, "_delete_staged", locked)
    removal = SimpleNamespace(paths=(victim,), freed_gib=0.0)
    with pytest.raises(Failed):
        svc_downloads._uninstall(svc, removal, None)
    leftovers = [p for p in root.iterdir() if p.name.startswith(svc_downloads.TRASH_PREFIX)]
    assert leftovers, "the failed delete should have left its trash tree behind"

    monkeypatch.undo()
    svc_downloads.sweep_staging(svc)

    assert [p for p in root.iterdir() if p.name.startswith(svc_downloads.TRASH_PREFIX)] == []


# --- familiar-host-07 ---------------------------------------------------------


class _Svc:
    def __init__(self, familiar: object | None = None) -> None:
        self.worker = SimpleNamespace(familiar=familiar if familiar is not None else object())

    def call_on_loop(self, factory, timeout: float = 30.0):
        return asyncio.run(factory())


def _has_image(messages) -> bool:
    last = messages[-1]["content"]
    return isinstance(last, list) and any(part.get("type") == "image_url" for part in last)


def _record_chat(monkeypatch, route: str, door_reply: str, seen: list):
    async def fake_chat(
        server, messages, *, slot, sampling, skill=None, expected_card_sha=None,
        response_format=None, transport=None,
    ):
        if skill == "router":
            return json.dumps({"skill": route})
        seen.append((response_format is not None, _has_image(messages)))
        return door_reply

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)


def test_a_routed_door_that_falls_back_to_chat_still_carries_the_attached_image(monkeypatch):
    """ask() forwarded the picture only to the last fallback and Clay Build: a
    navigate/create/character route whose door reply was unusable answered with
    a model that never saw the image, and said nothing."""
    from realmspinner.familiar import doors

    destinations = (doors.Destination(key="go:clay", label="Go to Clay"),)
    seen: list = []
    _record_chat(monkeypatch, "navigate", '{"target": "none"}', seen)
    svc_familiar.ask(
        _Svc(), "what is this?", mode="home", history=(), destinations=destinations,
        image=b"png",
    )
    # (door request, chat fallback): only the fallback is the user's own turn
    # and it must carry the picture.
    assert seen[-1] == (False, True)

    seen.clear()
    _record_chat(monkeypatch, "create", '{"asset_type": "nope", "prompt": ""}', seen)
    svc_familiar.ask(
        _Svc(), "what is this?", mode="home", history=(),
        asset_types=(("image", "Image"),), image=b"png",
    )
    assert seen[-1] == (False, True)

    seen.clear()
    options = {
        "families": [{"key": "humanoid", "label": "Humanoid", "themes": []}],
        "movements": ["idle"],
        "directions": [1],
        "size_range": (8, 256),
    }
    _record_chat(monkeypatch, "character", '{"family": "nope"}', seen)
    svc_familiar.ask(
        _Svc(), "what is this?", mode="home", history=(), character_options=options,
        image=b"png",
    )
    assert seen[-1] == (False, True)


# --- familiar-host-11 ---------------------------------------------------------


def test_a_request_killed_by_the_gpu_lease_is_refused_with_the_lease_reason(monkeypatch):
    """A chat in flight when a queued GPU job took the lease died with a
    transport error that read as "did not answer in time", and a retry then hit
    the lease refusal. While the server is leased, a failure is the lease."""

    async def fake_chat(*args, **kwargs):
        raise httpx.ReadError("connection reset")

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    leased = _Svc(SimpleNamespace(leased=True))
    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.chat_reply(leased, "hello")
    assert excinfo.value.reason == "lease"
    assert "GPU job holds the card" in excinfo.value.message

    # And not leased: still the timeout sentence it always was.
    free = _Svc(SimpleNamespace(leased=False))
    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.chat_reply(free, "hello")
    assert excinfo.value.reason == "unhealthy"


# --- create-workspace-01 ------------------------------------------------------


def test_a_graded_rig_row_that_still_exists_is_not_listed_as_deleted(svc):
    """``history_jobs`` selected only text/image rows, so a live rig row that
    carried a grade was listed with ``exists=False`` and Open disabled."""
    job_id = svc.store.create("rig", "a knight", {}, stage="model", status="done")
    # The store's writer, not ``record_verdict``: that door refuses a mesh grade for a rig
    # row now (the 2026-10-14 audit, finding service-gates-03), but a row graded before the
    # refusal existed is still in the log and still has to be listed as existing.
    svc.store.add_verdict(
        job_id,
        source="human",
        verdict=svc_verdicts.verdict_for_grade(4),
        reasons=[],
        vector={},
        stage="model",
        grade=4,
    )

    rows = {row["job_id"]: row for row in generation_log.history(svc)}

    assert rows[job_id]["exists"] is True
    assert rows[job_id]["prompt"] == "a knight"

    svc.store.delete(job_id)
    rows = {row["job_id"]: row for row in generation_log.history(svc)}
    assert rows[job_id]["exists"] is False


# --- service-gates-11 ---------------------------------------------------------


def test_judge_status_counts_only_labels_whose_image_still_exists(svc):
    """``status`` counted every human label, so after a prune the panel showed
    the threshold met while ``train`` fitted nothing."""

    def label(verdict: str, *, image: bool) -> str:
        job_id = svc.store.create("image", "a rogue", {}, stage="model", status="done")
        job_dir = svc.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "reference.png").write_bytes(b"png")
        svc_verdicts.record_verdict(svc, job_id, verdict=verdict, stage="blank")
        if not image:
            (job_dir / "reference.png").unlink()
        return job_id

    label("accept", image=True)
    label("accept", image=False)
    label("reject", image=False)

    found = svc_judge.status(svc, "blank")

    assert found["positives"] == 1
    assert found["negatives"] == 0
    assert found["labels"] == 1


# --- service-assets-03 --------------------------------------------------------


def test_the_training_door_keeps_a_photos_exif_orientation(svc, tmp_path, monkeypatch):
    """``train_lora`` re-encoded each image to PNG without applying the EXIF
    orientation, which dropped the tag and made the worker's own transpose a
    no-op: every portrait phone photo trained sideways."""
    monkeypatch.setattr(svc_loras, "check_base_model_weights", lambda *a, **k: None)
    folder = tmp_path / "photos"
    folder.mkdir()
    paths = []
    for index in range(4):
        path = folder / f"{index}.jpg"
        photo = Image.new("RGB", (20, 40), (200, 10, 10))
        exif = photo.getexif()
        exif[0x0112] = 6  # stored rotated: upright it is 40 wide and 20 tall
        photo.save(path, exif=exif)
        paths.append(path)
    with Image.open(paths[0]) as raw:
        upright = ImageOps.exif_transpose(raw).size
    assert upright == (40, 20)

    out = svc_loras.train_lora(svc, paths, label="Cosmos", trigger="cosmos style", steps=200)

    copied = sorted((svc.job_dir(out["id"]) / "train").glob("*.png"))
    assert len(copied) == 4
    with Image.open(copied[0]) as trained:
        assert trained.size == upright
