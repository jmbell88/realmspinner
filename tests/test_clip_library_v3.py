"""Clip library schema v3: per-clip ``duration_ms``, ``provisional`` and
``source``, the raised pose-count ceiling, and the direction-name guard.

Before this, per-frame timing lived only in ``charsheet.ANIMATIONS`` (then
``pipelines.charsheet.ANIMATIONS``), keyed by the five shipped animation
names -- so a clip library could never carry a clip with any other name and
still be timed. v3 moves timing into the library itself (``duration_ms`` on
every clip) so any clip name can exist, and a v2 file (everything shipped
before this parser existed) is migrated at read time from
:data:`cliplib.LEGACY_CLIP_DURATION_MS`, which is derived from
``charsheet.ANIMATIONS``. It was a pinned hand copy while ``charsheet`` lived
in ``pipelines/`` out of the rig kernel's reach; the restructure moved it to
``kernels/`` and the copy became an import, so the two tests that pinned the
copies to their originals went with them.

Design decision D2: ``closed`` stays the one fact about looping -- no ``loop``
field is added alongside it.
"""

from __future__ import annotations

import json

import pytest

from warlock import poselib
from warlock.kernels.rig import cliplib
from warlock.service import Invalid
from warlock.service import clips as svc_clips

TEMPLATE = "humanoid"


@pytest.fixture(autouse=True)
def _fresh_clip_cache():
    """Same isolation ``tests/service/test_clip_editing.py`` uses: the library caches
    are module globals filled once, and a test that edits one must not leak
    into the next."""
    cliplib.invalidate_clips()
    yield
    cliplib.invalidate_clips()


def _raw(clips: list[dict], *, version: int | None = 2, poses: list[dict] | None = None) -> dict:
    document: dict = {
        "poses": poses if poses is not None else [{"name": "rest", "bones": {}}],
        "clips": clips,
    }
    if version is not None:
        document["version"] = version
    return document


def _shipped_payload(svc) -> dict:
    view = svc_clips.library(svc, TEMPLATE)
    return {"space": view["space"], "poses": view["poses"], "clips": view["clips"]}


# --- migration ---------------------------------------------------------------


def test_a_version_2_library_reads_with_the_legacy_frame_times():
    clips = [
        {"name": name, "keys": ["rest"], "segments": [1]}
        for name in ("idle", "walk", "run", "attack", "jump", "some-other-clip")
    ]
    # Both spellings of "this is a v2 file": an explicit version and none at
    # all -- every file shipped before this parser existed says nothing.
    for raw in (_raw(clips, version=2), _raw(clips, version=None)):
        parsed = cliplib.parse_clip_library(raw)
        by_name = {c["name"]: c["duration_ms"] for c in parsed["clips"]}
        assert by_name["idle"] == 150
        assert by_name["walk"] == 100
        assert by_name["run"] == 60
        assert by_name["attack"] == 80
        assert by_name["jump"] == 100
        # A name the legacy table does not carry defaults to 100, same as
        # ``clips.animation_tracks``'s own fallback for an untabled clip.
        assert by_name["some-other-clip"] == 100


def test_a_version_3_clip_without_a_frame_time_is_refused():
    raw = _raw([{"name": "walk", "keys": ["rest"], "segments": [1]}], version=3)
    with pytest.raises(ValueError, match="duration_ms"):
        cliplib.parse_clip_library(raw)


@pytest.mark.parametrize("bad", [85, 0, 1010])
def test_a_frame_time_off_the_animation_timebase_is_refused(bad):
    raw = _raw(
        [{"name": "walk", "keys": ["rest"], "segments": [1], "duration_ms": bad}], version=3
    )
    with pytest.raises(ValueError):
        cliplib.parse_clip_library(raw)


# --- the timebase ----------------------------------------------------------


def test_the_clip_duration_step_divides_the_animation_timebase():
    from warlock import clips

    timebase_ms = 1000 / clips.ANIMATION_FPS
    assert timebase_ms == int(timebase_ms), "the timebase itself must be a whole ms count"
    assert int(timebase_ms) % cliplib.CLIP_DURATION_STEP_MS == 0


# --- the save door -------------------------------------------------------------


def test_saving_a_version_2_user_library_writes_version_3_with_the_times_it_was_read_with(
    svc, monkeypatch, tmp_path
):
    """Built by hand rather than read off the shipped files: those ship as v3
    now (the provisional clips added on top of the legacy five), so this test
    supplies its own v2 fixture -- the legacy five plus one name the legacy
    table does not carry, to exercise the migration's 100 ms fallback too."""
    two_pose_clip = {"keys": ["rest", "rest"], "segments": [1, 1], "closed": True}
    raw_v2 = {
        "version": 2,
        "space": "node",
        "poses": [{"name": "rest", "bones": {"root": [0.0, 0.0, 0.0, 1.0]}}],
        "clips": [
            {"name": "idle", **two_pose_clip},
            {"name": "walk", **two_pose_clip},
            {"name": "run", **two_pose_clip},
            {"name": "attack", **two_pose_clip},
            {"name": "jump", **two_pose_clip},
            {"name": "some-other-clip", **two_pose_clip},
        ],
    }
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir()
    (clip_dir / f"{TEMPLATE}.json").write_text(json.dumps(raw_v2), encoding="utf-8")
    monkeypatch.setattr(cliplib, "CLIP_DIR", clip_dir)
    cliplib.invalidate_clips()

    payload = _shipped_payload(svc)  # reads the v2 fixture above, migrated
    svc_clips.save(svc, TEMPLATE, payload)

    raw = json.loads(poselib.clip_path(svc.config, TEMPLATE).read_text(encoding="utf-8"))
    assert raw["version"] == 3
    for clip in raw["clips"]:
        assert clip["duration_ms"] == cliplib.LEGACY_CLIP_DURATION_MS.get(clip["name"], 100)


def test_a_clip_named_after_a_direction_is_refused(svc):
    # The parser: a v3 raw file that happens to carry a direction-suffixed
    # name. v3 only -- see test_a_version_2_library_with_a_turn_left_clip_
    # still_loads for the v2 half of this rule.
    raw = _raw(
        [{"name": "fall_back", "keys": ["rest"], "segments": [1], "duration_ms": 100}],
        version=3,
    )
    with pytest.raises(ValueError, match="fall_back"):
        cliplib.parse_clip_library(raw)

    # The save door: refused before it ever reaches the parser, with a field
    # the UI can point a control at.
    payload = _shipped_payload(svc)
    payload["clips"][0]["name"] = "fall_back"
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert caught.value.field


def test_a_version_2_library_with_a_turn_left_clip_still_loads():
    """Defect, fixed 2026-09-13: the direction-suffix guard used to apply to
    every version, so an existing v2 user library naming a clip like
    ``turn_left`` -- legal before the guard existed -- was refused *whole* on
    read, logged and silently skipped by the loader. The guard is a v3-only
    rule (the save door always writes v3, so such a name still cannot be
    *saved* -- see test_saving_a_turn_left_clip_asks_for_a_rename); a v2 file
    must keep reading exactly as it always did."""
    raw = _raw([{"name": "turn_left", "keys": ["rest"], "segments": [1]}], version=2)
    parsed = cliplib.parse_clip_library(raw)
    assert [c["name"] for c in parsed["clips"]] == ["turn_left"]

    # And the same is true with no "version" key at all -- every file shipped
    # before this parser existed.
    raw_unversioned = _raw(
        [{"name": "strafe_right", "keys": ["rest"], "segments": [1]}], version=None
    )
    parsed = cliplib.parse_clip_library(raw_unversioned)
    assert [c["name"] for c in parsed["clips"]] == ["strafe_right"]


def test_saving_a_turn_left_clip_asks_for_a_rename(svc):
    """The save door always writes v3, so it keeps refusing a direction-
    suffixed name -- but the refusal has to say what to do about it, not just
    restate the collision."""
    payload = _shipped_payload(svc)
    payload["clips"][0]["name"] = "turn_left"
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert caught.value.field == "clips"
    assert "rename" in caught.value.message.lower()


def test_provisional_and_source_survive_a_save_round_trip(svc):
    payload = _shipped_payload(svc)
    payload["clips"][0]["provisional"] = True
    payload["clips"][0]["source"] = {"file": "import.png", "map": "walk", "imported": 12345}
    saved = svc_clips.save(svc, TEMPLATE, payload)

    saved_clip = next(c for c in saved["clips"] if c["name"] == payload["clips"][0]["name"])
    assert saved_clip["provisional"] is True
    assert saved_clip["source"] == {"file": "import.png", "map": "walk", "imported": 12345}

    # And the file on disk, read back through the renderer's own parser.
    raw = json.loads(poselib.clip_path(svc.config, TEMPLATE).read_text(encoding="utf-8"))
    parsed = cliplib.parse_clip_library(raw)
    parsed_clip = next(c for c in parsed["clips"] if c["name"] == payload["clips"][0]["name"])
    assert parsed_clip["provisional"] is True
    assert parsed_clip["source"] == {"file": "import.png", "map": "walk", "imported": 12345}


# --- the raised pose cap -------------------------------------------------------


def test_a_library_may_hold_more_than_256_key_poses():
    poses = [{"name": f"p{i}", "bones": {}} for i in range(300)]
    raw = _raw(
        [{"name": "walk", "keys": ["p0"], "segments": [1]}],
        version=2,
        poses=poses,
    )
    parsed = cliplib.parse_clip_library(raw)
    assert len(parsed["poses"]) == 300


# --- the save/_commit_locked refactor ------------------------------------------


def test_save_and_commit_locked_write_the_same_file(svc):
    payload = _shipped_payload(svc)
    payload["clips"][0]["easing"] = "ease_out"

    document = svc_clips._check_shape({**payload, "template": TEMPLATE})
    with svc_clips._lock(svc):
        svc_clips._commit_locked(svc, TEMPLATE, document)
    direct_bytes = poselib.clip_path(svc.config, TEMPLATE).read_bytes()

    svc_clips.revert(svc, TEMPLATE)
    svc_clips.save(svc, TEMPLATE, payload)
    saved_bytes = poselib.clip_path(svc.config, TEMPLATE).read_bytes()

    assert direct_bytes == saved_bytes


# --- every shipped library still carries the legacy five's timing -------------


@pytest.mark.parametrize("key", ("humanoid", "quadruped", "bird", "blob"))
def test_every_shipped_library_still_parses_with_its_legacy_timing(key):
    """The shipped libraries are v3 now and carry provisional clips beyond the
    original five (attack_02, cast, fall, hit, death) -- timed on their own
    terms now that the vocabulary is open, not held to a table that was only
    ever about the five. This only checks that the five keep the times
    ``LEGACY_CLIP_DURATION_MS`` says they always had."""
    raw = json.loads((cliplib.CLIP_DIR / f"{key}.json").read_text(encoding="utf-8"))
    parsed = cliplib.parse_clip_library(raw)
    assert parsed["clips"], f"{key} shipped no clips to check"
    by_name = {c["name"]: c["duration_ms"] for c in parsed["clips"]}
    for name, expected in cliplib.LEGACY_CLIP_DURATION_MS.items():
        assert name in by_name, f"{key} no longer ships a {name!r} clip"
        assert by_name[name] == expected
