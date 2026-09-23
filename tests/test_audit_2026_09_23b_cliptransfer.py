"""Regression tests for the 2026-09-23 (second run) audit's cliptransfer
findings (poser-01, poser-02).

A trimmed copy of ``tests/test_cliptransfer.py``'s own fixture-building
helpers -- that module is not owned by this fix and is not imported from,
per the fix brief -- built the same way: Mixamo-named source bones whose
rest pose copies the shipped ``humanoid`` template's own rest positions, so
facing alignment and every rest alignment come out to identity by
construction and an injected per-frame delta round-trips essentially
unchanged. See that module's docstring for the full derivation.
"""

from __future__ import annotations

import math

from realmspinner import cliptransfer
from realmspinner.kernels.rig import templates
from realmspinner.service import clips as service_clips

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


def _axis_angle(axis, degrees):
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / n, ay / n, az / n
    half = math.radians(degrees) / 2.0
    s = math.sin(half)
    return [ax * s, ay * s, az * s, math.cos(half)]


def _entry(rotation, head):
    return {"rotation": rotation, "head": head}


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


# --- poser-01 ----------------------------------------------------------------


def test_import_clip_of_a_low_motion_closed_loop_keeps_at_least_two_keys():
    """A closed loop whose motion never exceeds KEY_TOLERANCE_DEG (1.5deg) used
    to reduce to a single key: ``analyse`` previewed it as an ordinary closed
    clip, and only ``service.clips``' own MIN_KEYS floor refused it later, on
    save, naming no cause. See cliptransfer.py's ``_transfer_action``, Step 9.
    """
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    frames = []
    # A tiny sway (0 -> 1 -> 0 degrees) that matches first and last frame
    # (closing the loop under loop="auto") and never itself exceeds
    # KEY_TOLERANCE_DEG, so RDP has nothing to add past frame 0.
    for i, deg in enumerate((0.0, 1.0, 0.0)):
        arm = _entry(_axis_angle((0, 0, 1), deg), source_bones["LeftArm"]["head"])
        bones = {**rest, "LeftArm": arm}
        frames.append({"frame": i, "bones": bones})
    action = {"name": "Idle", "fps": 30.0, "frame_start": 0, "frame_end": 2, "frames": frames}
    sample = _make_sample(source_bones, [action])
    [result] = cliptransfer.transfer(
        sample, template="humanoid", frames=6, root_motion="none"
    )
    assert result["clip"]["closed"] is True
    keys = result["clip"]["keys"]
    # Same floor ``service.clips.save`` -> ``_check_shape`` refuses under:
    # the bug this test guards against is a clip host code accepts on
    # preview and then refuses to save.
    assert len(keys) >= service_clips.MIN_KEYS
    assert len(result["clip"]["segments"]) == len(keys)


# --- poser-02 ----------------------------------------------------------------


def test_transfer_refuses_frames_one_for_an_open_clip_instead_of_dropping_all_motion():
    """``frames=1`` on an open (non-looping) clip used to be accepted and
    silently sample only phase 0, dropping every later frame's motion --
    the panel never passes ``frames`` itself, so this was only reachable
    through the API, which refused it downstream ("needs at least 2 keys")
    without ever naming ``frames`` as the cause.
    """
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    moved = {
        **rest,
        "LeftArm": _entry(_axis_angle((0, 1, 0), 40.0), source_bones["LeftArm"]["head"]),
    }
    action = {
        "name": "Swing",
        "fps": 30.0,
        "frame_start": 0,
        "frame_end": 1,
        "frames": [{"frame": 0, "bones": rest}, {"frame": 1, "bones": moved}],
    }
    sample = _make_sample(source_bones, [action])
    try:
        cliptransfer.transfer(
            sample, template="humanoid", frames=1, loop="off", root_motion="none"
        )
    except cliptransfer.ClipTransferError as exc:
        assert exc.field == "frames"
    else:
        raise AssertionError("frames=1 on an open clip was accepted")
