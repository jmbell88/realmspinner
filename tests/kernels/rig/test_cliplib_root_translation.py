"""``parse_clip_library`` and the service door agree on a falsy ``root_translation``.

The 2026-10-03 audit's poser-49 made ``service.clips`` refuse a present-but-falsy
non-list (``0``, ``""``, ``{}``) rather than skip it as if absent. The read door
kept ``if pose.get("root_translation"):`` -- so the same field in a hand-edited
file was dropped silently, root motion and all, while the editor refused it on
save (the 2026-10-10 audit's F18-7). Absent, null and an empty list stay "no
translation" at both.
"""

from __future__ import annotations

import pytest

from realmspinner.kernels.rig import cliplib


def _library(translation):
    pose = {"name": "a", "bones": {}}
    if translation is not ...:
        pose["root_translation"] = translation
    return {
        "version": 3,
        "poses": [pose, {"name": "b", "bones": {}}],
        "clips": [
            {"name": "c", "keys": ["a", "b"], "segments": [4], "closed": False, "duration_ms": 100}
        ],
    }


@pytest.mark.parametrize("bad", [0, "", {}, "0,0,0", 5, False])
def test_a_present_falsy_non_list_root_translation_is_refused_not_dropped(bad):
    with pytest.raises(ValueError, match="root_translation"):
        cliplib.parse_clip_library(_library(bad))


@pytest.mark.parametrize("quiet", [..., None, [], ()])
def test_an_absent_null_or_empty_root_translation_is_still_no_translation(quiet):
    out = cliplib.parse_clip_library(_library(quiet))
    assert "root_translation" not in out["poses"]["a"]


def test_a_real_root_translation_is_still_kept():
    out = cliplib.parse_clip_library(_library([0.0, 0.1, 0.0]))
    assert out["poses"]["a"]["root_translation"] == [0.0, 0.1, 0.0]
