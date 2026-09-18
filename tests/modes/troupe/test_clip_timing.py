"""``clips.py``'s timing comes from the clip library now, not
``charsheet.ANIMATIONS`` -- its one home, since the open clip vocabulary
(``dev/measurements/2026-09-12-troupe-open-clip-vocabulary.md``) means a
library can carry a clip ``ANIMATIONS`` never heard of."""

from __future__ import annotations

import json

import pytest

from warlock import clips
from warlock.kernels import charsheet as cs
from warlock.kernels.rig import cliplib

TEMPLATES = ("humanoid", "quadruped", "bird", "blob")


@pytest.fixture(autouse=True)
def _fresh_clip_cache():
    """Isolation ``tests/test_clip_library_v3.py`` already needs: the library
    caches are module globals filled once, and a test that edits one must not
    leak into the next."""
    cliplib.invalidate_clips()
    cliplib.set_user_clip_dir(None)
    yield
    cliplib.set_user_clip_dir(None)
    cliplib.invalidate_clips()


def _v3_copy_of_shipped(template_key: str, edits: dict[str, int]) -> dict:
    """The shipped library for *template_key*, migrated to v3 with every
    clip's legacy duration (``edits`` overrides some of them)."""
    raw = json.loads((cliplib.CLIP_DIR / f"{template_key}.json").read_text(encoding="utf-8"))
    raw["version"] = 3
    for clip in raw["clips"]:
        name = clip["name"]
        clip["duration_ms"] = edits.get(name, cliplib.LEGACY_CLIP_DURATION_MS.get(name, 100))
    return raw


def test_animation_tracks_take_their_timing_from_the_clip_library(tmp_path):
    """Edit a user library's ``walk`` to 120ms and the baked track follows --
    the timing's one home is the library, not ``charsheet.ANIMATIONS``."""
    raw = _v3_copy_of_shipped("humanoid", {"walk": 120})
    (tmp_path / "humanoid.json").write_text(json.dumps(raw), encoding="utf-8")
    cliplib.set_user_clip_dir(tmp_path)

    tracks = clips.animation_tracks("humanoid")
    walk = next(t for t in tracks if t["name"] == "walk")
    assert walk["step"] == pytest.approx(clips.ANIMATION_FPS * 120 / 1000.0)
    assert walk["step"] == pytest.approx(12.0)
    assert walk["loop"] is True

    # Every other clip kept its legacy timing -- only walk moved.
    idle = next(t for t in tracks if t["name"] == "idle")
    assert idle["step"] == pytest.approx(clips.ANIMATION_FPS * 150 / 1000.0)


def test_the_timebase_divides_every_frame_time_in_every_library():
    for template_key in TEMPLATES:
        library = cliplib.clip_library(template_key)
        for clip in library["clips"]:
            step = clips.ANIMATION_FPS * clip["duration_ms"] / 1000.0
            assert step == int(step), (template_key, clip["name"], clip["duration_ms"])


def test_the_library_digest_changes_when_a_clip_does(tmp_path):
    before = clips.library_digest("humanoid")

    raw = _v3_copy_of_shipped("humanoid", {"walk": 120})
    (tmp_path / "humanoid.json").write_text(json.dumps(raw), encoding="utf-8")
    cliplib.set_user_clip_dir(tmp_path)

    after = clips.library_digest("humanoid")
    assert after != before

    # And a library that says nothing new digests the same as the shipped one.
    cliplib.set_user_clip_dir(None)
    cliplib.invalidate_clips()
    assert clips.library_digest("humanoid") == before


def test_clip_timing_reproduces_the_legacy_frame_table():
    timing = clips.clip_timing("humanoid")
    for name, frames, loop, duration_ms in cs.ANIMATIONS:
        assert timing[name] == cs.ClipTiming(frames=frames, loop=loop, duration_ms=duration_ms)
