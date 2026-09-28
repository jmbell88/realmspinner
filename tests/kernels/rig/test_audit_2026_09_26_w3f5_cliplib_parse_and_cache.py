"""Regression tests for the 2026-09-26 audit, findings poser-rig-05,
poser-rig-06 and poser-rig-09 (an evidence gap: no source change, just the
missing sync test).
"""

from __future__ import annotations

import pytest

from realmspinner.kernels.rig import cliplib
from realmspinner.service import clips as svc_clips


@pytest.fixture(autouse=True)
def _fresh_clip_cache():
    """The library caches are module globals filled once -- a test that
    edits or invalidates one must not leak into the next, the isolation
    ``tests/service/test_clip_editing.py`` already uses."""
    cliplib.invalidate_clips()
    yield
    cliplib.invalidate_clips()


def _raw(clips: list[dict], *, poses: list[dict] | None = None) -> dict:
    return {
        "version": 3,
        "poses": poses if poses is not None else [{"name": "rest", "bones": {}}],
        "clips": clips,
    }


# --- poser-rig-05: "closed" truthiness and an unchecked "easing" ------------


def test_parse_clip_library_refuses_a_string_typed_closed_instead_of_reading_it_as_true():
    """``bool("false")`` is ``True`` in Python -- any non-empty string is
    truthy -- so a hand-edited ``"closed": "false"`` used to close a clip its
    author meant to leave open, with no error anywhere in the chain."""
    raw = _raw(
        [
            {
                "name": "walk",
                "keys": ["rest"],
                "segments": [1],
                "closed": "false",
                "duration_ms": 100,
            }
        ]
    )
    with pytest.raises(ValueError, match="closed"):
        cliplib.parse_clip_library(raw)


def test_parse_clip_library_accepts_a_real_bool_for_closed():
    raw = _raw(
        [
            {
                "name": "walk",
                "keys": ["rest"],
                "segments": [1],
                "closed": False,
                "duration_ms": 100,
            }
        ]
    )
    parsed = cliplib.parse_clip_library(raw)
    assert parsed["clips"][0]["closed"] is False


def test_parse_clip_library_refuses_an_unknown_easing():
    """``service.clips._check_shape`` (the write door) already refuses this
    by name; the read door -- the one a hand-edited or externally produced
    file goes through -- accepted anything and only ``sheet.interpolate_clip``
    caught it, and only for whichever movement a layout happened to name."""
    raw = _raw(
        [
            {
                "name": "walk",
                "keys": ["rest"],
                "segments": [1],
                "easing": "bouncy",
                "duration_ms": 100,
            }
        ]
    )
    with pytest.raises(ValueError, match="easing"):
        cliplib.parse_clip_library(raw)


# --- poser-rig-06: a concurrent invalidate_clips() mid-read -----------------


def test_clip_library_survives_an_invalidate_landing_before_the_user_directory_is_read(
    monkeypatch,
):
    """``clip_library`` used to keep reading the module globals ``_clips``/
    ``_user_clips`` directly after their ``is None`` checks; a concurrent
    ``invalidate_clips()`` (Poser's save door, on another thread) landing in
    the gap -- simulated here deterministically via a hook on
    ``user_clip_dir`` rather than a real race -- reset them back to ``None``,
    and the final ``.get`` then raised ``AttributeError`` instead of serving
    the library this call had already committed to loading."""
    real_user_clip_dir = cliplib.user_clip_dir

    def hook():
        result = real_user_clip_dir()
        cliplib.invalidate_clips()
        return result

    monkeypatch.setattr(cliplib, "user_clip_dir", hook)
    library = cliplib.clip_library("humanoid")
    assert isinstance(library, dict)
    assert "clips" in library


# --- poser-rig-09: the read-door ceilings are restated by hand "in sync" ---


def test_cliplib_read_door_ceilings_stay_in_sync_with_the_write_door():
    """dev/INVARIANTS.md and this module's own comments say these four
    constants mirror ``service.clips``'s write-door ceilings by hand; nothing
    pinned that claim, so the 2026-09-11/2026-09-23 fixes that raised or
    added one pair could silently drift from the other with no test to
    notice (the 2026-09-26 audit, finding poser-rig-09)."""
    assert cliplib.MAX_CLIP_LIBRARY_POSES == svc_clips.MAX_LIBRARY_KEYS
    assert cliplib.MAX_CLIP_KEYS == svc_clips.MAX_KEYS
    assert cliplib.MIN_CLIP_SEGMENT == svc_clips.MIN_SEGMENT
    assert cliplib.MAX_CLIP_SEGMENT == svc_clips.MAX_SEGMENT
