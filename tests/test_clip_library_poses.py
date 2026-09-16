"""Data regressions for the shipped clip libraries themselves.

``rigging._load_clip_library``/``parse_clip_library`` validate *shape*
(quaternions are finite unit-ish 4-vectors, bone names are legal) but have no
opinion on whether a pose's numbers describe an anatomically sane bend --
that is a modelling convention this test pins by data, not something the
parser can check.

The convention, read off the poses and confirmed with forward kinematics on
the humanoid template (``-Y`` is forward): a leg bone's pose quaternion is a
pure rotation about X, ``[x, 0, 0, w]``. A **negative** thigh X swings the
thigh forward (hip flexion) and a positive one swings it back; a **positive**
shin X folds the heel back toward the buttock (knee flexion). So a crouch is a
negative thigh with a positive shin -- opposite signs -- and a leg planted
behind the body at a walk's contact carries the larger thigh X of the pair.
"""

from __future__ import annotations

import math

import pytest

from warlock import rigging


def _angle(quat: list[float]) -> float:
    """A pure-X pose quaternion's rotation, in degrees."""
    return math.degrees(2.0 * math.asin(max(-1.0, min(1.0, quat[0]))))


def test_jump_crouch_and_land_flex_the_hip_forward_and_the_knee_back_with_the_foot_flat():
    """TODO.md's F7, finished. The clip shipped "jump crouch" as thigh
    ``+0.4384``, shin ``-0.6428``, foot ``+0.2079`` (and "jump land" as
    ``+0.3746``/``-0.5299``/``+0.1736``) -- every leg bone's sign inverted, so
    the thighs swept back and the knees bent the wrong way. The 2026-09-14
    audit flipped the shin alone, under a "thigh and shin share a sign" rule
    that is true of a walk's swing leg and false of any crouch; a render of
    Quaternius's Superhero Male the same day showed the result floating
    face-down with its ankles at hip height (ankle z 0.42 against a hip at
    0.52). With thigh and foot flipped too, forward kinematics puts the
    ankle back under the hips at rest height (z 0.064 against 0.060) and the
    toe on the ground."""
    poses = rigging.clip_library("humanoid")["poses"]
    for pose_name in ("jump crouch", "jump land"):
        bones = poses[pose_name]["bones"]
        for side in ("L", "R"):
            thigh = _angle(bones[f"thigh.{side}"])
            shin = _angle(bones[f"shin.{side}"])
            foot = _angle(bones[f"foot.{side}"])
            assert thigh < 0, f"{pose_name!r} swings thigh.{side} back ({thigh:+.0f} deg)"
            assert shin > 0, f"{pose_name!r} bends knee.{side} backward ({shin:+.0f} deg)"
            assert abs(thigh + shin + foot) <= 15.0, (
                f"{pose_name!r} leaves foot.{side} {thigh + shin + foot:+.0f} deg off flat"
            )


@pytest.mark.parametrize("library", ["humanoid", "bird"])
def test_no_authored_knee_bends_backward_past_fifteen_degrees(library):
    """TODO.md's F10. The forward-kinematics pass that settled F7 and F8 also
    measured the signed knee bend of every leg in the humanoid library:
    ``jump rise`` (L -40, R -14), ``jump apex`` (L -56, R -34), ``jump fall``
    (L -22, R -44), ``fall a``/``fall b`` (-90 on the lifted leg) and the three
    deaths (-70/-100/-110) all bent the knee backward -- a positive shin X
    should fold the heel toward the buttock, and these went negative instead.
    A straight planted leg reads a few degrees either side of zero (a walk's
    swing leg, an attack's stance) and that is fine; -15 is the line between
    "nearly straight" and "reverse-jointed".

    Only "humanoid" and "bird" are parametrized here: quadruped's
    ``rear_lower`` is a hock, not a knee, and bends the opposite way of every
    other joint by design (its rest pose is already bent), so it would fail a
    rule that does not apply to it; blob has no legs at all."""
    library_data = rigging.clip_library(library)
    poses = library_data["poses"]
    checked = 0
    for pose in poses.values():
        bones = pose["bones"]
        for side in ("L", "R"):
            shin_key = f"shin.{side}"
            if shin_key not in bones:
                continue
            shin = _angle(bones[shin_key])
            assert shin >= -15.0, (
                f"{library} {pose['name']!r} bends shin.{side} backward ({shin:+.0f} deg)"
            )
            checked += 1
    assert checked > 0


def _rest_offset(bone: dict) -> tuple[float, float]:
    """A bone's rest-frame (head -> tail) offset, Y/Z only -- the plane a
    pure-X pose quaternion rotates within (see this module's docstring)."""
    head, tail = bone["head"], bone["tail"]
    return tail[1] - head[1], tail[2] - head[2]


def _rotate_yz(y: float, z: float, degrees: float) -> tuple[float, float]:
    """Standard rotation about +X by ``degrees``, acting on a (Y, Z) offset."""
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    return y * c - z * s, y * s + z * c


def _leg_ground_heights(
    template_bones: dict[str, dict], pose_bones: dict, side: str, root_dz: float
) -> tuple[float, float]:
    """Planar FK for one leg: hips -> thigh -> shin -> foot, world Z only.

    Every rotation involved is pure-X (see this module's docstring), so world
    orientation is just the running sum of each ancestor's own angle -- no
    matrix needed. ``hips`` is folded in only when it too is pure-X (a walk's
    turn keys it about Z instead, which this ignores, per this test's brief).
    Verified before trusting it against two known points: the rest pose's toe
    lands at z=0, and the already-correct "jump crouch" pins an ankle z of
    ~0.064 (docstring above, F7).
    """
    thigh_b = template_bones[f"thigh.{side}"]
    shin_b = template_bones[f"shin.{side}"]
    foot_b = template_bones[f"foot.{side}"]

    hips_q = pose_bones.get("hips", [0.0, 0.0, 0.0, 1.0])
    hips_angle = _angle(hips_q) if hips_q[1] == 0.0 and hips_q[2] == 0.0 else 0.0

    cum_thigh = hips_angle + _angle(pose_bones.get(f"thigh.{side}", [0.0, 0.0, 0.0, 1.0]))
    cum_shin = cum_thigh + _angle(pose_bones.get(f"shin.{side}", [0.0, 0.0, 0.0, 1.0]))
    cum_foot = cum_shin + _angle(pose_bones.get(f"foot.{side}", [0.0, 0.0, 0.0, 1.0]))

    y, z = 0.0, thigh_b["head"][2] + root_dz

    dy, dz = _rest_offset(thigh_b)
    ry, rz = _rotate_yz(dy, dz, cum_thigh)
    y, z = y + ry, z + rz

    dy, dz = _rest_offset(shin_b)
    ry, rz = _rotate_yz(dy, dz, cum_shin)
    y, z = y + ry, z + rz
    ankle_z = z

    dy, dz = _rest_offset(foot_b)
    ry, rz = _rotate_yz(dy, dz, cum_foot)
    y, z = y + ry, z + rz
    toe_z = z

    return ankle_z, toe_z


def test_no_humanoid_pose_puts_an_ankle_or_toe_below_the_ground():
    """TODO.md's F10 and F11. The forward-kinematics pass found the three
    death poses' ankles below the ground plane (z=0) on top of their
    backward knees, and "death stagger" -- already knee-forward -- left its
    toe at z~-0.026 because nothing rotated the foot to follow the folded
    leg back up (F10). The same pass, widened to the whole library, later
    found six more: "run contact A", "run contact B", "attack strike",
    "attack follow", "attack_02 strike" and "attack_02 follow" each pushed
    the planted leg's toe 0.014-0.016 of character height below z=0 because
    the foot bone was left flat (or absent) under a thigh/shin bend that
    tips the toe down (F11). This now checks every pose in the humanoid
    library, not a "jump"/"fall"/"death" subset -- a planted foot has to
    clear the ground everywhere, including mid-stride and mid-swing, and
    F8's contact/passing rule (checked separately, below) is about which
    leg is behind, not about ground clearance.

    Sanity-checked before trusting it: the rest pose (no keys at all) puts
    the toe exactly at z=0, and "jump crouch" -- already fixed under F7 --
    lands its ankle at z~0.064, matching this module's docstring.
    """
    template = rigging.get_template("humanoid")
    template_bones = {b["name"]: b for b in template.bones}
    library_data = rigging.clip_library("humanoid")
    poses = library_data["poses"]

    # Sanity check: rest pose, no keys, toe on the ground.
    rest_ankle_z, rest_toe_z = _leg_ground_heights(template_bones, {}, "L", 0.0)
    assert abs(rest_toe_z) < 1e-6, f"rest toe should be at z=0, got {rest_toe_z}"

    checked = 0
    for pose in poses.values():
        bones = pose["bones"]
        root_dz = pose.get("root_translation", [0.0, 0.0, 0.0])[2]
        for side in ("L", "R"):
            if f"thigh.{side}" not in bones and f"shin.{side}" not in bones:
                continue
            ankle_z, toe_z = _leg_ground_heights(template_bones, bones, side, root_dz)
            name = pose["name"]
            assert ankle_z >= -0.01, f"{name!r} ankle.{side} is below ground (z={ankle_z:+.4f})"
            assert toe_z >= -0.01, f"{name!r} toe.{side} is below ground (z={toe_z:+.4f})"
            checked += 1
    assert checked > 0


# --- the deformation battery (templates/deform_qa/humanoid.json) -----------
#
# A separate file and a separate loader (rigging.deform_battery, not
# clip_library), but the same sign convention -- and it shipped with the same
# F7 mistake this module's first test names: every leg sign inverted.


def test_deform_battery_squat_flexes_the_hip_forward_and_the_knee_back():
    """The 2026-09-16 human QA pass on a real ``measured`` rig (Quaternius's
    Superhero Male): the shipped squat was thigh ``+0.5``, shin ``-0.7071``,
    foot ``+0.2588`` -- every leg sign inverted, exactly the F7 mistake above,
    just in the battery instead of the clip library -- and it rendered the
    legs folded up behind the head instead of a crouch."""
    poses = {p["name"]: p for p in rigging.deform_battery("humanoid")}
    bones = poses["squat"]["bones"]
    for side in ("L", "R"):
        thigh = _angle(bones[f"thigh.{side}"])
        shin = _angle(bones[f"shin.{side}"])
        assert thigh < 0, f"squat swings thigh.{side} back ({thigh:+.0f} deg)"
        assert shin > 0, f"squat bends knee.{side} backward ({shin:+.0f} deg)"


def test_no_deform_battery_pose_bends_a_knee_backward_past_fifteen_degrees():
    """The battery's own version of
    ``test_no_authored_knee_bends_backward_past_fifteen_degrees`` above:
    "elbow and knee 90" shipped shin ``-0.7071``, the knee bending backward
    the same wrong way as the squat's."""
    poses = rigging.deform_battery("humanoid")
    checked = 0
    for pose in poses:
        bones = pose["bones"]
        for side in ("L", "R"):
            shin_key = f"shin.{side}"
            if shin_key not in bones:
                continue
            shin = _angle(bones[shin_key])
            assert shin >= -15.0, (
                f"{pose['name']!r} bends shin.{side} backward ({shin:+.0f} deg)"
            )
            checked += 1
    assert checked > 0


def test_deform_qa_humanoid_comment_cites_a_docstring_that_actually_states_the_arm_convention():
    """The 2026-09-16 audit: this file's own "comment" field stated its sign
    convention as "negative thigh/upper_arm X = forward/up flexion" and
    attributed the whole parenthetical to this module's docstring above --
    but that docstring defines the convention only for thigh/shin/spine and
    never mentions upper_arm at all, so a future author adding an arm pose
    to the battery who followed this file's own stated convention literally
    would author the opposite sign from what templates/clips/humanoid.json's
    -58 degree rest correction and this file's own "arms overhead" pose
    actually use -- the F7 sign-inversion mistake, reproduced by the file's
    own documentation rather than its data this time."""
    import json

    raw = json.loads((rigging.BATTERY_DIR / "humanoid.json").read_text(encoding="utf-8"))
    comment = raw["comment"]

    cite = "tests/test_clip_library_poses.py's docstring"
    assert cite in comment, "the comment should still cite this module's leg convention"

    this_docstring = __doc__ or ""
    assert "upper_arm" not in this_docstring, (
        "this module's docstring now documents upper_arm -- update this test, "
        "or the comment it pins, to match"
    )

    # The sentence that cites this module's docstring must not claim
    # upper_arm is part of what it states.
    cited_sentence = comment.split(cite, 1)[1].split(".", 1)[0]
    assert "upper_arm" not in cited_sentence, (
        "the deform_qa comment attributes an upper_arm sign convention to "
        "tests/test_clip_library_poses.py's docstring, but that docstring "
        "never states one"
    )

    # The comment must still document the arm convention itself, separately
    # -- and it must match what templates/clips/humanoid.json's rest
    # correction and this file's own "arms overhead" pose actually do:
    # negative swings the arm down, positive swings it up.
    assert "upper_arm" in comment
    arms_overhead_x = next(p for p in raw["poses"] if p["name"] == "arms overhead")[
        "bones"
    ]["upper_arm.L"][0]
    assert arms_overhead_x > 0, "arms overhead should be a positive upper_arm X"


@pytest.mark.parametrize(
    ("library", "clip_name"),
    [("humanoid", "walk"), ("humanoid", "run"), ("bird", "walk"), ("bird", "run")],
)
def test_the_leg_behind_at_a_contact_is_the_leg_the_next_passing_pose_lifts(library, clip_name):
    """TODO.md's F8. The humanoid walk played its stride backward: "walk
    contact A" plants the left leg behind, but "walk passing A" -- the key
    after it -- lifts the *right* leg and plants the left under the hips, so
    forward kinematics has the planted foot sliding 0.2 of a body height
    forward and the swinging foot travelling back. Every pose was a sane
    pose; the two passing poses were simply each other's. "run" had the
    same swap. The bird's walk and run already obey the rule and are here
    as the control, so the check is not a statement about one skeleton."""
    library_data = rigging.clip_library(library)
    clip = next(each for each in library_data["clips"] if each["name"] == clip_name)
    poses = library_data["poses"]
    keys = clip["keys"]
    checked = 0
    for index, key in enumerate(keys):
        if "contact" not in key:
            continue
        following = keys[(index + 1) % len(keys)]
        assert "passing" in following, f"{clip_name}: {key!r} is not followed by a passing pose"
        contact, passing = poses[key]["bones"], poses[following]["bones"]
        behind = max("LR", key=lambda side: contact[f"thigh.{side}"][0])
        lifted = max("LR", key=lambda side: passing[f"shin.{side}"][0])
        assert lifted == behind, (
            f"{library} {clip_name}: {key!r} plants {behind} behind, but {following!r} "
            f"lifts {lifted} -- the stride plays backward"
        )
        checked += 1
    assert checked == 2
