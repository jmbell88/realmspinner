"""Data regressions for the shipped clip libraries themselves.

``rigging._load_clip_library``/``parse_clip_library`` validate *shape*
(quaternions are finite unit-ish 4-vectors, bone names are legal) but have no
opinion on whether a pose's numbers describe an anatomically sane bend --
that is a modelling convention this test pins by data, not something the
parser can check.
"""

from __future__ import annotations

from warlock import rigging


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def test_jump_crouch_and_land_bend_the_knee_forward_not_backward():
    """The 2026-09-14 audit, finding docs-05 (TODO.md's F7): every other
    flexed pose in this library rotates a leg's thigh and shin in the *same*
    sign about X when the knee bends -- e.g. walk's "passing A", `thigh.R
    +0.0698` with `shin.R +0.2924`, a normal swing-through. "jump crouch"
    (`thigh +0.4384`) shipped with `shin -0.6428` and "jump land"
    (`thigh +0.3746`) shipped with `shin -0.5299` -- opposite sign to their
    own thigh while every other bent-knee pose agrees, which bends the knee
    backward. Confirmed visually on a rendered character sheet (TODO.md F7);
    fixed here to the exact replacement values F7 names (`+0.6428`/
    `+0.5299`), matching this library's own sign convention.

    No Blender render is available in this test environment to re-confirm
    the visual fix -- this pins the data only, as F7 asks when no render is
    available.
    """
    poses = rigging.clip_library("humanoid")["poses"]
    for pose_name in ("jump crouch", "jump land"):
        bones = poses[pose_name]["bones"]
        for side in ("L", "R"):
            thigh_x = bones[f"thigh.{side}"][0]
            shin_x = bones[f"shin.{side}"][0]
            assert _sign(thigh_x) == _sign(shin_x), (
                f'{pose_name!r} bends thigh.{side} ({thigh_x}) and shin.{side} '
                f"({shin_x}) in opposite directions -- a backward knee"
            )
