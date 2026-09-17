"""Bone-name mapping tables for "Import clip": the pure parser and matcher.

The Blender sampling op and the rotation math are later steps and are not
tested here -- see ``clipmaps.py``'s module docstring for the split. This
covers the shipped tables, ``parse_clip_map``'s refusals, and ``match``'s
prefix-stripping and chain-completeness rules.
"""

from __future__ import annotations

import json

import pytest

from warlock import clipmaps
from warlock.kernels.rig import templates

HUMANOID_BONES = {b["name"] for b in templates.get_template("humanoid").bones}


def _raw(key: str) -> dict:
    return json.loads((clipmaps.CLIP_MAP_DIR / f"{key}.json").read_text(encoding="utf-8"))


MIXAMO_BONES = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot",
    "RightUpLeg", "RightLeg", "RightFoot",
]

RIGIFY_BONES = [
    "DEF-spine", "DEF-spine.001", "DEF-spine.002", "DEF-spine.003",
    "DEF-spine.004", "DEF-spine.005", "DEF-spine.006",
    "DEF-shoulder.L", "DEF-upper_arm.L", "DEF-upper_arm.L.001",
    "DEF-forearm.L", "DEF-forearm.L.001", "DEF-hand.L",
    "DEF-shoulder.R", "DEF-upper_arm.R", "DEF-upper_arm.R.001",
    "DEF-forearm.R", "DEF-forearm.R.001", "DEF-hand.R",
    "DEF-thigh.L", "DEF-thigh.L.001", "DEF-shin.L", "DEF-shin.L.001", "DEF-foot.L",
    "DEF-thigh.R", "DEF-thigh.R.001", "DEF-shin.R", "DEF-shin.R.001", "DEF-foot.R",
]


def _mixamo_skeleton(prefix: str = "mixamorig:") -> list[str]:
    return [f"{prefix}{n}" for n in MIXAMO_BONES]


def _rigify_skeleton() -> list[str]:
    return list(RIGIFY_BONES)


# --- shipped tables ----------------------------------------------------------


def test_every_shipped_clip_map_parses_and_names_only_template_bones():
    maps = clipmaps.load_clip_maps()
    assert set(maps) == {"mixamo", "rigify"}
    for clip_map in maps.values():
        assert set(clip_map.bones) <= HUMANOID_BONES


def test_every_shipped_map_covers_every_humanoid_bone():
    for clip_map in clipmaps.load_clip_maps().values():
        assert set(clip_map.bones) == HUMANOID_BONES


# --- parse_clip_map refusals --------------------------------------------------


def test_a_map_claiming_one_source_bone_twice_is_refused():
    raw = _raw("mixamo")
    raw["bones"]["spine"] = raw["bones"]["hips"]  # both now claim "Hips"
    with pytest.raises(clipmaps.ClipMapError, match="claimed"):
        clipmaps.parse_clip_map(raw, source="mixamo")


def test_a_map_naming_a_bone_its_template_lacks_is_refused():
    raw = _raw("mixamo")
    raw["bones"]["tail"] = ["Tail"]  # humanoid has no "tail" bone
    with pytest.raises(clipmaps.ClipMapError, match="tail"):
        clipmaps.parse_clip_map(raw, source="mixamo")


def test_a_required_bone_outside_the_map_is_refused():
    raw = _raw("mixamo")
    raw["required"] = [*raw["required"], "tail"]  # "tail" is not a bones key
    with pytest.raises(clipmaps.ClipMapError, match="required"):
        clipmaps.parse_clip_map(raw, source="mixamo")


# --- normalise / prefix stripping --------------------------------------------


@pytest.mark.parametrize(
    "hips_name",
    ["mixamorig:Hips", "mixamorig1:Hips", "mixamorig_Hips", "Hips"],
)
def test_mixamo_prefix_variants_all_match(hips_name):
    clip_map = clipmaps.load_clip_maps()["mixamo"]
    assert clipmaps.normalise(hips_name, clip_map) == "Hips"


# --- match() -------------------------------------------------------------


def test_unmapped_source_bones_are_reported_not_fatal():
    extras = ["mixamorig:LeftHandThumb1", "mixamorig:HeadTop_End", "mixamorig:LeftToeBase"]
    result = clipmaps.match(_mixamo_skeleton() + extras, template="humanoid")
    assert result.clip_map.key == "mixamo"
    assert not result.left_at_rest
    assert set(result.ignored) == set(extras)


def test_a_skeleton_missing_a_required_bone_is_refused_naming_the_bones():
    skeleton = [
        n for n in _mixamo_skeleton() if n not in ("mixamorig:Hips", "mixamorig:LeftUpLeg")
    ]
    with pytest.raises(clipmaps.ClipMapError) as exc:
        clipmaps.match(skeleton, template="humanoid")
    assert "Hips" in str(exc.value)
    assert "LeftUpLeg" in str(exc.value)


def test_rigify_deform_names_match_the_rigify_map_and_not_mixamo():
    result = clipmaps.match(_rigify_skeleton(), template="humanoid")
    assert result.clip_map.key == "rigify"
    assert not result.left_at_rest
    assert result.resolved["hips"] == ("DEF-spine",)
    assert result.resolved["upper_arm.L"] == ("DEF-upper_arm.L", "DEF-upper_arm.L.001")


def test_a_partial_chain_leaves_its_target_bone_at_rest():
    skeleton = [n for n in _rigify_skeleton() if n != "DEF-upper_arm.L.001"]
    result = clipmaps.match(skeleton, template="humanoid")
    assert result.clip_map.key == "rigify"
    assert "upper_arm.L" in result.left_at_rest
    assert "upper_arm.L" not in result.resolved
    # DEF-upper_arm.L is still claimed by upper_arm.L's chain -- the chain just
    # did not complete -- so it is not reported as an orphaned bone.
    assert "DEF-upper_arm.L" not in result.ignored


def test_no_candidate_targeting_an_unknown_template_is_refused():
    with pytest.raises(clipmaps.ClipMapError, match="quadruped"):
        clipmaps.match(["Hips"], template="quadruped")


# --- load_clip_maps isolation --------------------------------------------


def test_a_bad_shipped_map_costs_that_entry_not_the_loader(tmp_path, monkeypatch):
    (tmp_path / "mixamo.json").write_text(
        json.dumps(_raw("mixamo")), encoding="utf-8"
    )
    (tmp_path / "broken.json").write_text('{"key": "broken"}', encoding="utf-8")
    monkeypatch.setattr(clipmaps, "CLIP_MAP_DIR", tmp_path)
    monkeypatch.setattr(clipmaps, "_clip_maps", None)
    assert set(clipmaps.load_clip_maps()) == {"mixamo"}
