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
