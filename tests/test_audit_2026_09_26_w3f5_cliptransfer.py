"""Regression tests for the 2026-09-26 audit, findings poser-poses-02 and
poser-poses-03.

Fixtures follow ``tests/test_cliptransfer.py``'s own baseline (a synthetic,
Blender-free sample whose rest geometry copies the shipped ``humanoid``
template, so a mapped bone's basis round-trips essentially unchanged) --
duplicated in miniature here rather than imported, since ``tests/`` carries
no ``__init__.py`` and cross-test-module imports are not this suite's
convention.
"""

from __future__ import annotations

from realmspinner import cliptransfer
from realmspinner.kernels.rig import templates

TEMPLATE = templates.get_template("humanoid")
TARGET_BONES = {b["name"]: b for b in TEMPLATE.bones}

MIXAMO_SINGLE = {
    "hips": "Hips",
    "spine": "Spine",
    "neck": "Neck",
    "head": "Head",
    "shoulder.L": "LeftShoulder",
    "upper_arm.L": "LeftArm",
    "forearm.L": "LeftForeArm",
    "hand.L": "LeftHand",
    "shoulder.R": "RightShoulder",
    "upper_arm.R": "RightArm",
    "forearm.R": "RightForeArm",
    "hand.R": "RightHand",
    "thigh.L": "LeftUpLeg",
    "shin.L": "LeftLeg",
    "foot.L": "LeftFoot",
    "thigh.R": "RightUpLeg",
    "shin.R": "RightLeg",
    "foot.R": "RightFoot",
}
CHEST_FIRST, CHEST_LAST = "Spine1", "Spine2"
_IDENTITY = [0.0, 0.0, 0.0, 1.0]


def _target_sample():
    bones = {}
    for name, b in TARGET_BONES.items():
        bones[name] = {
            "parent": b["parent"],
            "rest_rotation": list(_IDENTITY),
            "head": list(b["head"]),
            "tail": list(b["tail"]),
        }
    return {"template": "humanoid", "bones": bones}


def _baseline_source_bones():
    bones = {}
    for target, src_name in MIXAMO_SINGLE.items():
        t = TARGET_BONES[target]
        bones[src_name] = {
            "rest_rotation": list(_IDENTITY), "head": list(t["head"]), "tail": list(t["tail"])
        }
    chest = TARGET_BONES["chest"]
    mid = [(chest["head"][i] + chest["tail"][i]) / 2.0 for i in range(3)]
    bones[CHEST_FIRST] = {
        "rest_rotation": list(_IDENTITY), "head": list(chest["head"]), "tail": mid
    }
    bones[CHEST_LAST] = {
        "rest_rotation": list(_IDENTITY), "head": mid, "tail": list(chest["tail"])
    }
    return bones


def _rest_frame_bones(source_bones):
    out = {}
    for src_name in list(MIXAMO_SINGLE.values()) + [CHEST_LAST]:
        b = source_bones[src_name]
        out[src_name] = {"rotation": list(b["rest_rotation"]), "head": list(b["head"])}
    return out


def _make_sample(source_bones, actions):
    return {
        "source_bones": source_bones,
        "all_bone_names": list(source_bones),
        "actions": actions,
        "target": _target_sample(),
    }


def _one_frame_action(name, frame_bones, **kw):
    kw.setdefault("fps", 30.0)
    kw.setdefault("frame_start", 0)
    kw.setdefault("frame_end", 0)
    return {"name": name, "frames": [{"frame": 0, "bones": frame_bones}], **kw}


def test_two_actions_that_slug_to_one_name_are_imported_under_distinct_names():
    """Mixamo's own "Run!"/"Run?" both slug to "run" through
    ``_slug_from_action_name`` -- before the 2026-09-26 fix, ``transfer``
    handed both actions ``clip_name=None`` unchanged, so both clips in its
    result carried the identical name "run". ``import_into_library``'s own
    collision check then either raised a misleading "already exists" for a
    name this file never repeated, or, with ``replace=True``, dropped the
    first clip when the second overwrote it in the same merge loop."""
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    actions = [
        _one_frame_action("Run!", rest),
        _one_frame_action("Run?", rest),
    ]
    sample = _make_sample(source_bones, actions)
    results = cliptransfer.transfer(
        sample, template="humanoid", frames=2, loop="off", root_motion="none"
    )
    names = [r["clip"]["name"] for r in results]
    assert names[0] == "run"
    assert len(set(names)) == len(names), f"expected distinct names, got {names}"


def test_transfer_with_frames_1_and_loop_on_yields_an_importable_two_key_clip():
    """``frames=1`` on an *open* clip is refused by name, with the message
    telling the caller to ask for ``loop="on"`` instead for "a single held
    pose" -- but before the 2026-09-26 fix, a genuinely one-sampled-frame
    closed clip reduced to exactly one key (there is only one frame to
    reduce), which ``service.clips``' own ``MIN_KEYS=2`` door then refused,
    with no way to satisfy it: the message offered a path the function's own
    output could not walk."""
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    action = _one_frame_action("Idle", rest)
    sample = _make_sample(source_bones, [action])
    [result] = cliptransfer.transfer(
        sample, template="humanoid", frames=1, loop="on", root_motion="none"
    )
    assert result["clip"]["closed"] is True
    # service.clips.MIN_KEYS is 2; a one-key clip is refused at the save
    # door with no way for this caller to reach 2 through any parameter.
    assert len(result["clip"]["keys"]) >= 2
    assert len(result["clip"]["segments"]) == len(result["clip"]["keys"])
    assert all(n >= 1 for n in result["clip"]["segments"])
