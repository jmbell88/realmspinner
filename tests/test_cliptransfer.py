"""Regression tests for "Import clip"'s pure host math (``cliptransfer.py``).

Every fixture here is a **synthetic** sample built to satisfy the input
contract ``cliptransfer.transfer`` documents, not a real Blender export --
this module claims (like ``kernels.rig``, ``poselib`` and ``clipmaps`` before
it) to be decidable with no Blender at all, so this suite is what stands on
that claim.

The baseline fixture (``_baseline_source_bones`` with no overrides) copies
the shipped ``humanoid`` template's own rest positions onto Mixamo-named
source bones, and gives every target bone an identity ``rest_rotation`` --
which makes the facing alignment (``C``) and every rest alignment (``G``)
come out to the identity rotation by construction (worked out in comments
below), so an injected per-frame delta round-trips through the pipeline
essentially unchanged. Individual tests perturb one piece of that geometry
at a time to exercise ``C`` or ``G`` on their own.
"""

from __future__ import annotations

import math
import re

import pytest

from warlock import clipmaps, cliptransfer
from warlock.kernels.rig import cliplib, templates
from warlock.pipelines import sheet

TEMPLATE = templates.get_template("humanoid")
TARGET_BONES = {b["name"]: b for b in TEMPLATE.bones}

#: The shipped ``mixamo.json`` clip map's own table, restated here (not
#: imported -- there is nothing to import it from) so the fixtures below
#: build a source skeleton it actually resolves.
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


def _quat_angle_deg(q):
    """The rotation's own angle magnitude, computed independently of
    ``cliptransfer._angle_deg`` (same formula, but a rotation's angle off
    identity is just ``2*acos(|w|)`` and is worth stating plainly here rather
    than reusing the module under test for its own assertions)."""
    w = max(-1.0, min(1.0, abs(q[3])))
    return math.degrees(2.0 * math.acos(w))


def _target_sample():
    """The ``target`` half of a sample: every humanoid bone, identity rest
    rotations, real shipped head/tail positions."""
    bones = {}
    for name, b in TARGET_BONES.items():
        bones[name] = {
            "parent": b["parent"],
            "rest_rotation": list(_IDENTITY),
            "head": list(b["head"]),
            "tail": list(b["tail"]),
        }
    return {"template": "humanoid", "bones": bones}


def _baseline_source_bones(overrides=None):
    """Mixamo-named source bones whose rest pose copies the target's own --
    C and every G come out to identity for a skeleton built this way (see
    the module docstring). ``overrides`` maps a *source* bone name to its
    own ``(head, tail)`` pair, for tests that need one bone's rest direction
    to differ from the target's."""
    overrides = overrides or {}
    bones = {}
    for target, src_name in MIXAMO_SINGLE.items():
        t = TARGET_BONES[target]
        head, tail = list(t["head"]), list(t["tail"])
        if src_name in overrides:
            head, tail = overrides[src_name]
        bones[src_name] = {"rest_rotation": list(_IDENTITY), "head": head, "tail": tail}
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
    """Every mapped bone (chain-last only) held at its own rest -- the
    "nothing moves" frame every test starts from."""
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


def _entry(rotation, head):
    return {"rotation": rotation, "head": head}


def _one_frame_action(name, frame_bones, **kw):
    kw.setdefault("fps", 30.0)
    kw.setdefault("frame_start", 0)
    kw.setdefault("frame_end", 0)
    return {"name": name, "frames": [{"frame": 0, "bones": frame_bones}], **kw}


# --- the round trip ----------------------------------------------------


def test_a_source_whose_rest_equals_the_target_rest_round_trips_its_bases_exactly():
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    delta = _axis_angle((0, 1, 0), 40.0)
    # The whole arm swings as one rigid unit (elbow and wrist unposed): every
    # rest-relative offset in this fixture is identity (see the module
    # docstring), so a *world* rotation that cascades unchanged down the
    # chain is exactly what an unposed elbow/wrist looks like in Blender's
    # own world-space sample -- setting only ``LeftArm`` and leaving
    # ``LeftForeArm``/``LeftHand`` at their *own* rest would instead claim
    # the elbow counter-rotates to stay fixed in world space, which is not
    # what "only the upper arm moved" means.
    moved = {
        **rest,
        "LeftArm": _entry(delta, source_bones["LeftArm"]["head"]),
        "LeftForeArm": _entry(delta, source_bones["LeftForeArm"]["head"]),
        "LeftHand": _entry(delta, source_bones["LeftHand"]["head"]),
    }
    action = {
        "name": "Test",
        "fps": 30.0,
        "frame_start": 0,
        "frame_end": 1,
        "frames": [{"frame": 0, "bones": rest}, {"frame": 1, "bones": moved}],
    }
    sample = _make_sample(source_bones, [action])
    [result] = cliptransfer.transfer(
        sample, template="humanoid", frames=2, loop="off", root_motion="none"
    )
    keys = result["clip"]["keys"]
    assert len(keys) == 2
    first = result["poses"][keys[0]]["bones"]
    last = result["poses"][keys[1]]["bones"]
    assert _quat_angle_deg(first.get("upper_arm.L", _IDENTITY)) == pytest.approx(0.0, abs=0.5)
    assert _quat_angle_deg(last["upper_arm.L"]) == pytest.approx(40.0, abs=0.5)
    # Nothing else moved -- every other mapped bone is static and omitted.
    assert set(last) == {"upper_arm.L"}


# --- rest-direction alignment (G) ---------------------------------------


def test_a_t_pose_rest_frame_puts_the_upper_arm_at_the_source_direction_not_the_a_pose():
    head = list(TARGET_BONES["upper_arm.L"]["head"])
    tail = [head[0] + 0.20, head[1], head[2]]  # purely horizontal, a T-pose
    source_bones = _baseline_source_bones(overrides={"LeftArm": (head, tail)})
    rest = _rest_frame_bones(source_bones)
    action = _one_frame_action("Tpose", rest)
    sample = _make_sample(source_bones, [action])
    [result] = cliptransfer.transfer(
        sample, template="humanoid", frames=2, loop="off", root_motion="none"
    )
    pose = result["poses"][result["clip"]["keys"][0]]

    def world(name):
        b = pose["bones"].get(name, _IDENTITY)
        parent = TARGET_BONES[name]["parent"]
        if parent is None:
            return tuple(b)
        return cliptransfer._mul(world(parent), tuple(b))

    w = world("upper_arm.L")
    arm = TARGET_BONES["upper_arm.L"]
    d_t = cliptransfer._sub3(arm["tail"], arm["head"])
    direction = cliptransfer._rotate_vector(w, d_t)
    # The template's own A-pose direction dips well below horizontal
    # (dz/len ~ -0.78); the T-pose source direction should not.
    assert abs(direction[2]) < 0.02


def test_spine1_and_spine2_collapse_into_chest_as_their_composed_rotation():
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    delta = _axis_angle((1, 0, 0), 15.0)
    moved = dict(rest)
    moved[CHEST_LAST] = _entry(delta, source_bones[CHEST_LAST]["head"])
    # Spine1 (the chain's first bone) is read only for its *rest* direction,
    # never per frame -- a wildly different frame value for it must have no
    # effect on chest's orientation.
    bogus = _axis_angle((0, 0, 1), 111.0)
    moved[CHEST_FIRST] = _entry(bogus, source_bones[CHEST_FIRST]["head"])
    action = _one_frame_action("Chest", moved)
    sample = _make_sample(source_bones, [action])
    [result] = cliptransfer.transfer(
        sample, template="humanoid", frames=2, loop="off", root_motion="none"
    )
    pose = result["poses"][result["clip"]["keys"][0]]
    assert _quat_angle_deg(pose["bones"]["chest"]) == pytest.approx(15.0, abs=0.5)


# --- facing alignment (C) ------------------------------------------------


def test_a_source_facing_plus_y_is_turned_to_face_minus_y():
    source_bones = _baseline_source_bones()
    # Rotate every rest position 180 degrees about Z: a source whose own
    # forward, left and up were the baseline's (already matching the
    # template's) now faces +Y instead of -Y.
    for b in source_bones.values():
        b["head"] = [-b["head"][0], -b["head"][1], b["head"][2]]
        b["tail"] = [-b["tail"][0], -b["tail"][1], b["tail"][2]]
    resolved = clipmaps.match(list(source_bones), template="humanoid").resolved
    c = cliptransfer._facing_alignment(source_bones, resolved)
    forward = cliptransfer._rotate_vector(c, (0.0, 1.0, 0.0))
    assert forward == pytest.approx((0.0, -1.0, 0.0), abs=1e-6)
    left = cliptransfer._rotate_vector(c, (-1.0, 0.0, 0.0))
    assert left == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)


def test_a_source_facing_minus_y_with_left_at_plus_x_gives_c_identity():
    """The fixed point named in the algorithm spec: a source already using
    the template's own convention gets the identity facing correction."""
    source_bones = _baseline_source_bones()
    resolved = clipmaps.match(list(source_bones), template="humanoid").resolved
    c = cliptransfer._facing_alignment(source_bones, resolved)
    assert c == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-9)


def test_a_mirrored_source_does_not_swap_left_and_right():
    source_bones = _baseline_source_bones()
    # A mirrored FBX export (a -1 X-scale some exporters apply): every rest
    # position reflected in X, consistently across the *whole* skeleton, while
    # every bone still names the same anatomical side. Consistency matters --
    # mirroring only a few landmark bones would leave the rig internally
    # contradictory (facing one way per the hips, built another way per the
    # arms), which is not what a real mirrored export looks like and would
    # make ``G`` (the per-bone rest-direction correction) carry the
    # contradiction instead of ``C`` alone. Mirrored consistently, ``G`` comes
    # out identity for the untouched arm chain and only ``C`` changes -- and a
    # pure conjugation ``C . delta . C^-1`` never changes a rotation's own
    # angle, whatever axis ``C`` turns out to be.
    for b in source_bones.values():
        b["head"] = [-b["head"][0], b["head"][1], b["head"][2]]
        b["tail"] = [-b["tail"][0], b["tail"][1], b["tail"][2]]
    rest = _rest_frame_bones(source_bones)
    delta = _axis_angle((0, 1, 0), 25.0)
    moved = {**rest, "LeftArm": {"rotation": delta, "head": source_bones["LeftArm"]["head"]}}
    action = _one_frame_action("Raise", moved)
    sample = _make_sample(source_bones, [action])
    [result] = cliptransfer.transfer(
        sample, template="humanoid", frames=2, loop="off", root_motion="none"
    )
    pose = result["poses"][result["clip"]["keys"][0]]
    assert "upper_arm.L" in pose["bones"]
    assert _quat_angle_deg(pose["bones"]["upper_arm.L"]) == pytest.approx(25.0, abs=1.0)
    assert "upper_arm.R" not in pose["bones"]


# --- root motion ---------------------------------------------------------


def test_in_place_removes_travel_but_keeps_the_vertical_bob():
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    hips_rest = source_bones["Hips"]["head"]
    path = [
        [hips_rest[0] + 0.00, hips_rest[1], hips_rest[2] + 0.00],
        [hips_rest[0] + 0.25, hips_rest[1], hips_rest[2] + 0.07],
        [hips_rest[0] + 0.50, hips_rest[1], hips_rest[2] + 0.00],
        [hips_rest[0] + 0.75, hips_rest[1], hips_rest[2] + 0.07],
        [hips_rest[0] + 1.00, hips_rest[1], hips_rest[2] + 0.00],
    ]
    frames = []
    for i, head in enumerate(path):
        bones = {**rest, "Hips": {"rotation": list(_IDENTITY), "head": head}}
        frames.append({"frame": i, "bones": bones})
    action = {"name": "Walk", "fps": 30.0, "frame_start": 0, "frame_end": 4, "frames": frames}
    sample = _make_sample(source_bones, [action])
    [result] = cliptransfer.transfer(
        sample, template="humanoid", frames=5, loop="off", root_motion="in_place"
    )
    keys = result["clip"]["keys"]
    poses = result["poses"]
    first_root = poses[keys[0]]["root_translation"]
    last_root = poses[keys[-1]]["root_translation"]
    assert first_root[0] == pytest.approx(0.0, abs=1e-6)
    assert last_root[0] == pytest.approx(0.0, abs=1e-6)
    # The bob survives somewhere in the clip -- height is in character
    # heights and the baseline skeleton is exactly one unit tall (see the
    # module docstring's ``_root_height`` derivation), so 0.07 units is
    # 0.07 character heights, unchanged by in-place XY correction.
    peak_z = max(p["root_translation"][2] for p in poses.values())
    assert peak_z == pytest.approx(0.07, abs=0.01)


def test_keep_root_motion_past_the_limit_is_refused_naming_root_motion():
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    hips_rest = source_bones["Hips"]["head"]
    far_head = [hips_rest[0] + 3.0, hips_rest[1], hips_rest[2]]
    far = {**rest, "Hips": _entry(list(_IDENTITY), far_head)}
    action = {
        "name": "Sprint",
        "fps": 30.0,
        "frame_start": 0,
        "frame_end": 1,
        "frames": [{"frame": 0, "bones": rest}, {"frame": 1, "bones": far}],
    }
    sample = _make_sample(source_bones, [action])
    with pytest.raises(cliptransfer.ClipTransferError) as excinfo:
        cliptransfer.transfer(
            sample, template="humanoid", frames=2, loop="off", root_motion="keep"
        )
    assert excinfo.value.field == "root_motion"
    assert "3.000" in str(excinfo.value)  # names the peak, not just the field


# --- looping ---------------------------------------------------------------


def test_a_matching_first_and_last_frame_closes_the_clip_without_a_duplicate_seam():
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    frames = []
    for i, deg in enumerate((0.0, 20.0, 0.0)):
        arm = _entry(_axis_angle((0, 0, 1), deg), source_bones["LeftArm"]["head"])
        bones = {**rest, "LeftArm": arm}
        frames.append({"frame": i, "bones": bones})
    action = {"name": "Cycle", "fps": 30.0, "frame_start": 0, "frame_end": 2, "frames": frames}
    sample = _make_sample(source_bones, [action])
    [result] = cliptransfer.transfer(sample, template="humanoid", frames=6, root_motion="none")
    assert result["report"]["loop"]["closed"] is True
    assert result["report"]["loop"]["residual_deg"] == pytest.approx(0.0, abs=0.5)
    assert result["clip"]["closed"] is True


def test_loop_on_distributes_the_residual():
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    frames = []
    for i, deg in enumerate((0.0, 60.0, 30.0)):
        arm = _entry(_axis_angle((0, 0, 1), deg), source_bones["LeftArm"]["head"])
        bones = {**rest, "LeftArm": arm}
        frames.append({"frame": i, "bones": bones})
    action = {"name": "Spin", "fps": 30.0, "frame_start": 0, "frame_end": 2, "frames": frames}
    sample = _make_sample(source_bones, [action])
    [result] = cliptransfer.transfer(
        sample, template="humanoid", frames=3, loop="on", root_motion="none"
    )
    assert result["clip"]["closed"] is True
    keys = result["clip"]["keys"]
    poses = result["poses"]
    # Frame 0 is untouched (weight 0); the residual (-30, taking the
    # raw-authored last frame's 30 degrees onto frame 0's 0) is distributed
    # by weight i/n over n=3 raw frames -- 1/3 at the middle frame, 2/3 at
    # the last -- so the last frame lands one step short of frame 0's value
    # rather than exactly on it (see test_loop_on_leaves_no_held_frame_at_
    # the_seam for why landing exactly on it is the bug this weighting
    # fixes). Same-axis rotations compose exactly additively: middle =
    # 60 - 30*(1/3) = 50, last = 30 - 30*(2/3) = 10.
    angle0 = _quat_angle_deg(poses[keys[0]]["bones"].get("upper_arm.L", _IDENTITY))
    angle_mid = _quat_angle_deg(poses[keys[1]]["bones"]["upper_arm.L"])
    angle_last = _quat_angle_deg(poses[keys[-1]]["bones"]["upper_arm.L"])
    assert angle0 == pytest.approx(0.0, abs=0.5)
    assert angle_mid == pytest.approx(50.0, abs=1.0)
    assert angle_last == pytest.approx(10.0, abs=1.0)


def test_loop_on_leaves_no_held_frame_at_the_seam():
    """``loop="on"``'s residual correction used to weight the last frame by
    ``i/(n-1)``, landing it exactly on frame 0's own value -- so once the
    clip is treated as closed (wrapping last back to first), that whole wrap
    segment covers zero motion: a 1/n-of-the-cycle hitch every time the loop
    seams. Weighting by ``i/n`` instead leaves the last frame one step short
    of frame 0, so the wrap segment carries the final step like every other
    one does.
    """
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    # Front-loaded, non-linear motion (0 -> 50 -> 55 -> 58 -> 60) rather than
    # an even ramp: an even ramp's raw values are exactly collinear with the
    # correction line either weighting produces, which would cancel every
    # frame to zero motion under the old weighting too and mask the bug
    # behind an unrelated "clip has no motion" failure instead of the wrong
    # seam angle this test means to catch. Only the first (0) and last (60)
    # values decide the residual and, with it, the corrected last frame.
    n = 5
    raw_deg = (0.0, 50.0, 55.0, 58.0, 60.0)
    frames = []
    for i, deg in enumerate(raw_deg):
        arm = _entry(_axis_angle((0, 0, 1), deg), source_bones["LeftArm"]["head"])
        bones = {**rest, "LeftArm": arm}
        frames.append({"frame": i, "bones": bones})
    action = {"name": "Ramp", "fps": 30.0, "frame_start": 0, "frame_end": n - 1, "frames": frames}
    sample = _make_sample(source_bones, [action])
    [result] = cliptransfer.transfer(
        sample, template="humanoid", frames=n, loop="on", root_motion="none"
    )
    clip = result["clip"]
    assert clip["closed"] is True
    key_poses = [result["poses"][k] for k in clip["keys"]]
    expanded = sheet.resample_clip(
        key_poses, clip["segments"], n, closed=True, easing=clip["easing"], space="delta"
    )
    first_deg = _quat_angle_deg(expanded[0]["bones"].get("upper_arm.L", _IDENTITY))
    last_deg = _quat_angle_deg(expanded[-1]["bones"].get("upper_arm.L", _IDENTITY))
    assert first_deg == pytest.approx(0.0, abs=0.5)
    # The held-frame bug lands this at ~0 (the same value as frame 0, a
    # duplicate seam) because the old weighting applies the *full* residual
    # (raw_deg[-1] - raw_deg[0] = 60) at the last frame. The fix applies only
    # (n-1)/n of it, leaving raw_deg[-1] * (1 - (n-1)/n) = 60 * 1/5 = 12
    # degrees standing -- one step short of frame 0, not zero. Reexpansion
    # through key reduction approximates this like any other frame (see
    # test_reduced_keys_reexpand_through_interpolate_clip_within_tolerance),
    # hence the same ``KEY_TOLERANCE_DEG``-sized slack.
    assert last_deg == pytest.approx(raw_deg[-1] / n, abs=cliptransfer.KEY_TOLERANCE_DEG + 0.5)
    assert last_deg > raw_deg[-1] * 0.1


# --- resampling and key reduction -----------------------------------------


def test_reduced_keys_reexpand_through_interpolate_clip_within_tolerance():
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    # A dense, exactly piecewise-linear ramp (0 -> 60 -> 0 in steps of 10):
    # every interior sample lies exactly on the straight line between its
    # two neighbours, in the same-axis-rotation case where slerp *is* exact
    # linear angle interpolation -- so RDP has zero error to spend and keeps
    # only the two ends and the peak, a clean, deterministic reduction.
    bump = tuple(range(0, 70, 10)) + tuple(range(50, -10, -10))
    frames = []
    for i, deg in enumerate(bump):
        arm = _entry(_axis_angle((0, 0, 1), deg), source_bones["LeftArm"]["head"])
        bones = {**rest, "LeftArm": arm}
        frames.append({"frame": i, "bones": bones})
    action = {
        "name": "Bump",
        "fps": 30.0,
        "frame_start": 0,
        "frame_end": len(bump) - 1,
        "frames": frames,
    }
    sample = _make_sample(source_bones, [action])
    [result] = cliptransfer.transfer(
        sample, template="humanoid", frames=len(bump), loop="off", root_motion="none"
    )
    clip = result["clip"]
    # Fewer keys than raw frames -- the whole point of key reduction.
    assert len(clip["keys"]) < len(bump)

    key_poses = [result["poses"][k] for k in clip["keys"]]
    expanded = sheet.resample_clip(
        key_poses,
        clip["segments"],
        len(bump),
        closed=clip["closed"],
        easing=clip["easing"],
        space="delta",
    )
    assert len(expanded) == len(bump)
    for frame, wanted_deg in zip(expanded, bump, strict=True):
        got = frame["bones"].get("upper_arm.L", _IDENTITY)
        got_deg = _quat_angle_deg(got)
        assert got_deg == pytest.approx(wanted_deg, abs=cliptransfer.KEY_TOLERANCE_DEG + 0.5)


# --- the wider contract ----------------------------------------------------


def test_the_output_parses_through_the_renderers_own_clip_library_parser():
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    arm = _entry(_axis_angle((0, 1, 0), 20.0), source_bones["LeftArm"]["head"])
    moved = {**rest, "LeftArm": arm}
    action = {
        "name": "Wave",
        "fps": 30.0,
        "frame_start": 0,
        "frame_end": 1,
        "frames": [{"frame": 0, "bones": rest}, {"frame": 1, "bones": moved}],
    }
    sample = _make_sample(source_bones, [action])
    [result] = cliptransfer.transfer(
        sample, template="humanoid", frames=4, loop="off", root_motion="none"
    )
    poses = [{"name": name, **pose} for name, pose in result["poses"].items()]
    library = {
        "version": 3,
        "template": "humanoid",
        "space": "delta",
        "poses": poses,
        "clips": [result["clip"]],
    }
    parsed = cliplib.parse_clip_library(library)
    assert parsed["clips"][0]["name"] == result["clip"]["name"]
    assert set(parsed["poses"]) == {p["name"] for p in poses}


def test_a_frame_count_above_the_cap_is_refused_on_frames():
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    action = _one_frame_action("Rest", rest)
    sample = _make_sample(source_bones, [action])
    with pytest.raises(cliptransfer.ClipTransferError) as excinfo:
        cliptransfer.transfer(sample, template="humanoid", frames=cliptransfer.MAX_CLIP_FRAMES + 1)
    assert excinfo.value.field == "frames"


def test_an_action_name_becomes_a_legal_clip_name():
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    action = _one_frame_action("Armature|mixamo.com|Layer0", rest)
    sample = _make_sample(source_bones, [action])
    [result] = cliptransfer.transfer(sample, template="humanoid", frames=2, root_motion="none")
    name = result["clip"]["name"]
    assert re.fullmatch(r"[a-z0-9_]+", name)
    cliplib.reject_direction_named_clip(name)  # must not raise

    action2 = _one_frame_action("Walk_back", rest)
    sample2 = _make_sample(source_bones, [action2])
    with pytest.raises(cliptransfer.ClipTransferError) as excinfo:
        cliptransfer.transfer(sample2, template="humanoid", frames=2, root_motion="none")
    assert excinfo.value.field == "name"


def test_an_imported_clip_keeps_its_source_length():
    """``duration_ms`` is the time PER RENDERED FRAME (``clips.animation_
    tracks``'s ``step = ANIMATION_FPS * duration_ms / 1000``, one hop per
    authored frame; ``charsheet``'s per-cell durations are the same idea).
    A closed (looping) clip of ``N`` resampled frames makes ``N`` hops in one
    full cycle (the last hop is the wrap back to frame 0), so the default
    ``duration_ms`` must satisfy ``N * duration_ms ~= source_seconds * 1000``
    -- not ``(N / fps) * 1000``, which is roughly the *whole source clip's*
    length and used to make an imported clip play N times too slowly.
    """
    step = cliplib.CLIP_DURATION_STEP_MS
    for duration_s, n_frames in ((1.0, 10), (2.5, 25)):
        source_bones = _baseline_source_bones()
        rest = _rest_frame_bones(source_bones)
        frame_end = round(duration_s * 30)
        action = {
            "name": "Walk",
            "fps": 30.0,
            "frame_start": 0,
            "frame_end": frame_end,
            "frames": [
                {"frame": 0, "bones": rest},
                {"frame": frame_end, "bones": rest},
            ],
        }
        sample = _make_sample(source_bones, [action])
        [result] = cliptransfer.transfer(
            sample, template="humanoid", frames=n_frames, loop="on", root_motion="none"
        )
        clip = result["clip"]
        assert clip["closed"] is True
        total_ms = n_frames * clip["duration_ms"]
        assert total_ms == pytest.approx(duration_s * 1000.0, abs=step)


def test_the_duration_is_a_legal_clip_duration():
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    frames = [{"frame": i, "bones": rest} for i in range(48)]
    action = {"name": "Long", "fps": 24.0, "frame_start": 0, "frame_end": 47, "frames": frames}
    sample = _make_sample(source_bones, [action])
    [result] = cliptransfer.transfer(sample, template="humanoid", root_motion="none")
    cliplib.validate_clip_duration_ms(result["clip"]["duration_ms"], result["clip"]["name"])

    [explicit] = cliptransfer.transfer(
        sample, template="humanoid", root_motion="none", duration_ms=250
    )
    assert explicit["clip"]["duration_ms"] == 250


# --- clip-map matching, surfaced in the report ------------------------


def test_cliptransfer_report_names_a_duplicate_normalized_source_bone():
    """The 2026-09-16 audit: the 2026-09-15 fix (finding poser-04) taught
    ``clipmaps.MatchResult`` to record a normalized name more than one raw
    source bone collapsed onto, so a colliding duplicate no longer vanished
    from ``clipmaps.match``'s own result -- but nothing between there and
    ``cliptransfer.transfer``'s returned ``report`` ever read the field, so
    "Import clip"'s report still carried no trace of it one call frame
    further downstream. Reproduces the same duplicate
    ``test_clip_map_match_does_not_silently_drop_a_duplicate_normalized_source_bone_name``
    (``tests/test_audit_2026_09_15_poser.py``) does directly against
    ``clipmaps.match``, but through the whole ``transfer`` call."""
    source_bones = _baseline_source_bones()
    # A second raw spelling of "Hips" that the shipped mixamo clip map's
    # strip pattern normalizes onto the same name as "Hips" itself -- exactly
    # the hand-renamed-duplicate (or merged second Mixamo export) shape
    # poser-04 was written against.
    source_bones["mixamorig:Hips"] = dict(source_bones["Hips"])
    rest = _rest_frame_bones(source_bones)
    action = _one_frame_action("Idle", rest)
    sample = _make_sample(source_bones, [action])

    [result] = cliptransfer.transfer(
        sample, template="humanoid", frames=2, loop="off", root_motion="none"
    )

    clip_map = clipmaps.load_clip_maps()["mixamo"]
    normalized = clipmaps.normalise("Hips", clip_map)
    report = result["report"]
    assert "duplicate_source_names" in report
    assert report["duplicate_source_names"][normalized] == ["Hips", "mixamorig:Hips"]


# --- the restated constants, pinned to their sources ------------------


def test_max_clip_frames_matches_sheets_own_ceiling():
    assert cliptransfer.MAX_CLIP_FRAMES == sheet.MAX_CLIP_FRAMES


def test_max_root_translation_matches_poselibs_own_bound():
    from warlock import poselib

    assert cliptransfer.MAX_ROOT_TRANSLATION == poselib.MAX_ROOT_TRANSLATION


def test_the_restated_slerp_agrees_with_sheets_own():
    a = [0.0, 0.0, 0.0, 1.0]
    b = _axis_angle((0, 0, 1), 90.0)
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert cliptransfer._slerp(a, b, t) == pytest.approx(sheet.slerp(a, b, t), abs=1e-9)
