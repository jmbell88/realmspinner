"""Regression tests for two 2026-09-26 audit findings in ``pipelines/``:

* **inker-sheets-03** -- ``pixel.map_palette`` had no ceiling of its own on
  how many entries it would search: a palette at the *reader*'s ceiling
  (``MAX_PALETTE_ROWS`` = 65,536, a legal read) turned the dither branch's
  pairwise gap matrix and the no-native-kernel fallback's per-chunk search
  into a many-gigabyte allocation with no named refusal anywhere on the way.
* **poser-rig-01** -- ``blender_worker.op_animate`` never called
  ``_reset_pose`` between frames, so a bone a frame's pose omits kept
  whatever the previous frame (or the previous clip's last frame) left on
  it, instead of returning to rest the way the sheet render loop
  (``_measure``, also in this file) always does.

Named per the fixer brief's convention; the ``op_animate`` bake itself needs
real Blender and cannot run here (this repo's default lane has no ``bpy``),
so that half is proven the way ``tests/test_sheet.py`` already proves
adjacent ``op_sheet`` behaviour without one: a fake ``bpy``/armature plus
``monkeypatch`` spies on ``_reset_pose``/``_apply_pose`` that record the call
order the fixed loop must produce.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from realmspinner.pipelines import blender_worker, pixel

# -- inker-sheets-03 -----------------------------------------------------------


def test_map_palette_refuses_a_palette_too_large_to_search_by_name() -> None:
    """A palette past ``MAX_SEARCHABLE_PALETTE`` must be refused by name,
    before any O(entries) or O(entries^2) array is built -- not turn into an
    unbounded allocation. The image is tiny on purpose: if the refusal did
    not fire before the allocation, this test would hang or OOM rather than
    fail cleanly, which is itself part of the proof.
    """
    image = Image.new("RGBA", (4, 4), (10, 20, 30, 255))
    too_many = tuple((i % 256, (i * 7) % 256, (i * 13) % 256) for i in range(4097))
    assert len(too_many) > pixel.MAX_SEARCHABLE_PALETTE

    with pytest.raises(ValueError, match="4097"):
        pixel.map_palette(image, too_many)


def test_map_palette_still_accepts_an_ordinary_palette() -> None:
    """The new ceiling must not have tightened the door for any real
    palette -- Aseprite's own indexed mode tops out at 256 entries, and this
    build's designed-palette ladder never exceeds 64."""
    image = Image.new("RGBA", (4, 4), (200, 40, 40, 255))
    palette = tuple((i, 0, 255 - i) for i in range(0, 256, 4))  # 64 entries
    out = pixel.map_palette(image, palette)
    assert out.size == image.size


# -- poser-rig-01 ---------------------------------------------------------------


class _FakePoseBone:
    def __init__(self, name: str) -> None:
        self.name = name
        self.rotation_mode: str | None = None
        self.rotation_quaternion: tuple[float, float, float, float] | None = None
        self.location: tuple[float, float, float] | None = None
        self.scale: tuple[float, float, float] | None = None
        self.keyframes: list[tuple[str, float]] = []

    def keyframe_insert(self, data_path: str, frame: float) -> None:
        self.keyframes.append((data_path, frame))


class _BoneList(list):
    """``arm_obj.pose.bones``: iterated directly by both ``_reset_pose`` and
    ``op_animate``'s own closing keyframe loop -- a plain list already
    satisfies that, unlike a dict (which would iterate its keys)."""


class _FakeArmObj:
    def __init__(self, names: list[str]) -> None:
        self.pose = SimpleNamespace(bones=_BoneList(_FakePoseBone(n) for n in names))
        self.animation_data: SimpleNamespace | None = None

    def animation_data_create(self) -> None:
        self.animation_data = SimpleNamespace(action=None)


class _FakeAction:
    def __init__(self, name: str) -> None:
        self.name = name
        self.use_fake_user = False


class _FakeBpy:
    def __init__(self) -> None:
        self.data = SimpleNamespace(actions=SimpleNamespace(new=_FakeAction))
        self.context = SimpleNamespace(scene=SimpleNamespace(render=SimpleNamespace(fps=30)))


def _animate_spec(tmp_path: Path, tracks: list[dict[str, Any]]) -> dict[str, Any]:
    rig_glb = tmp_path / "rig.glb"
    rig_glb.write_bytes(b"not-really-a-glb")
    return {
        "rig_glb": str(rig_glb),
        "out_glb": str(tmp_path / "animated.glb"),
        "fps": 30,
        "clips": tracks,
    }


def test_animate_resets_omitted_bones_to_rest_between_frames_and_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """poser-rig-01: every frame ``op_animate`` bakes must be reset to rest
    before that frame's own pose is applied -- otherwise a bone one frame
    omits silently keeps whatever an earlier frame (or an earlier clip's
    last frame) left rotated, and ``animated.glb`` disagrees with the sheet
    the same frames were rendered for.
    """
    arm_obj = _FakeArmObj(["hips", "spine"])
    calls: list[tuple[str, tuple[str, ...] | None]] = []

    def fake_reset_pose(obj) -> None:
        assert obj is arm_obj
        calls.append(("reset", None))

    def fake_apply_pose(obj, bones, space="node"):
        assert obj is arm_obj
        calls.append(("apply", tuple(sorted(bones))))
        return len(bones), []

    monkeypatch.setattr(blender_worker, "_reset_scene", lambda _bpy: None)
    monkeypatch.setattr(blender_worker, "_import_rig", lambda _bpy, _path: arm_obj)
    monkeypatch.setattr(blender_worker, "_export", lambda _bpy, _path, **kw: None)
    monkeypatch.setattr(blender_worker, "_reset_pose", fake_reset_pose)
    monkeypatch.setattr(blender_worker, "_apply_pose", fake_apply_pose)

    tracks = [
        {
            "name": "walk",
            "space": "delta",
            "step": 1.0,
            "frames": [
                {"bones": {"hips": [0, 0, 0, 1]}},
                {"bones": {"spine": [0, 0, 0, 1]}},  # omits "hips"
            ],
        },
        {
            "name": "idle",
            "space": "delta",
            "step": 1.0,
            "frames": [
                {"bones": {}},  # omits both -- must still land at rest
            ],
        },
    ]
    result = blender_worker.op_animate(_FakeBpy(), _animate_spec(tmp_path, tracks))
    assert result["ok"] is True

    # Exactly one reset immediately before each of the three frames' apply,
    # in order -- the "no _reset_pose in the frame loop" gap, closed.
    assert calls == [
        ("reset", None),
        ("apply", ("hips",)),
        ("reset", None),
        ("apply", ("spine",)),
        ("reset", None),
        ("apply", ()),
    ]
