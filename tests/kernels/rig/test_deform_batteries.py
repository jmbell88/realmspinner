"""Every skeleton the app ships either has a deformation battery or is named
as having none, and every battery is well-formed.

The QA sheet is human-reviewed, so these are the checks that cannot be left to
the eye: a bone the skeleton does not have is silently skipped by the worker
(``_apply_pose``), and a left/right row that is not an exact mirror reads as
plausible in a still.
"""

from __future__ import annotations

import pytest

from realmspinner.kernels.rig import poses, templates

# Skeletons with no battery, and why. A battery is authored per skeleton ("a
# squat means nothing to a fish"), so adding a template without one must be a
# decision someone wrote down here, not an omission.
EXEMPT = {
    "insect": "six jointed legs and mandibles: no row of the other families reads on it yet",
    "fish": "a spine and fins: nothing to bend that a sprite sheet would show",
    "blob": "a core and lobes with no bone chain; the clip library alone exercises it",
    "blank": "one root bone; there is nothing to deform",
}
FAMILIES = sorted(set(templates.templates()) - set(EXEMPT))


def test_every_rig_template_has_a_battery_or_is_named_exempt():
    keys = set(templates.templates())
    assert set(EXEMPT) <= keys, "an exempt name that is not a template is stale"
    missing = [k for k in sorted(keys - set(EXEMPT)) if not poses.deform_battery(k)]
    assert not missing, f"templates with no battery and no exemption: {missing}"


def test_an_exempt_template_really_has_no_battery():
    assert [k for k in EXEMPT if poses.deform_battery(k)] == []


@pytest.mark.parametrize("key", FAMILIES)
def test_every_battery_bone_exists_in_its_family_skeleton(key):
    names = {b["name"] for b in templates.get_template(key).bones}
    battery = poses.deform_battery(key)
    assert battery, key
    for pose in battery:
        assert pose["bones"], f"{key}/{pose['name']} poses nothing"
        assert pose["space"] == "delta", f"{key}/{pose['name']} is not authored in delta space"
        for bone, quat in pose["bones"].items():
            assert bone in names, f"{key}/{pose['name']} poses {bone}, which {key} lacks"
            assert len(quat) == 4
            assert sum(v * v for v in quat) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("key", FAMILIES)
def test_symmetric_battery_rows_are_exact_mirrors(key):
    """A pair that appears in a row appears on both sides, and the right is
    ``poses.mirror_quaternion`` of the left, to the digit the file carries."""
    for pose in poses.deform_battery(key):
        bones = pose["bones"]
        for left, right in templates.get_template(key).mirror_pairs:
            if left not in bones and right not in bones:
                continue
            assert left in bones and right in bones, f"{key}/{pose['name']}: {left}/{right}"
            assert bones[right] == pytest.approx(poses.mirror_quaternion(bones[left]), abs=1e-6)


@pytest.mark.parametrize("key", FAMILIES)
def test_battery_pose_names_and_ids_are_unique(key):
    battery = poses.deform_battery(key)
    assert len({p["name"] for p in battery}) == len(battery)
    assert len({p["id"] for p in battery}) == len(battery), (
        "a sheet row is keyed by pose id; duplicates would repeat the first pose"
    )


def test_the_biped_tail_battery_is_the_humanoid_one_plus_the_tail():
    """The two skeletons share every humanoid bone's head and tail, so the
    humanoid rows are reused verbatim -- if the skeletons ever diverge this
    stops being true and the rows need re-deriving, which is what this says."""
    human = {p["name"]: p["bones"] for p in poses.deform_battery("humanoid")}
    biped = {p["name"]: p["bones"] for p in poses.deform_battery("biped_tail")}
    for name, bones in human.items():
        assert biped[name] == bones
    assert {"tail whip", "tail lift"} <= set(biped)
    h = {b["name"]: b for b in templates.get_template("humanoid").bones}
    for b in templates.get_template("biped_tail").bones:
        if b["name"] in h:
            assert (b["head"], b["tail"]) == (h[b["name"]]["head"], h[b["name"]]["tail"])
