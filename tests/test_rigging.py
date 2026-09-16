"""Everything about rigging that is decidable without Blender.

The bpy-dependent half lives behind importorskip at the bottom, matching how
tests/test_offline.py guards the heavy paths.
"""

from __future__ import annotations

import itertools
import json
import math
import subprocess
import sys
import time

import pytest

from warlock import rigging

# --- template registry ------------------------------------------------------

EXPECTED_TEMPLATES = {
    "humanoid",
    "quadruped",
    "bird",
    "fish",
    "insect",
    "serpent",
    "biped_tail",
    "blob",
}


def test_both_templates_load():
    keys = set(rigging.templates())
    assert {"humanoid", "quadruped"} <= keys


def test_every_shipped_template_parses():
    assert set(rigging.templates()) == EXPECTED_TEMPLATES


@pytest.mark.parametrize("key", sorted(EXPECTED_TEMPLATES))
def test_template_is_well_formed(key):
    """Parsing already rejects unknown parents and bad roots; this pins the
    things _parse_template does not check and that a hand-authored file gets
    wrong: bones inside the unit box, and mirror pairs naming real bones."""
    template = rigging.get_template(key)
    names = {b["name"] for b in template.bones}
    for bone in template.bones:
        for end in ("head", "tail"):
            x, y, z = bone[end]
            assert -0.5 <= x <= 0.5, f"{key}/{bone['name']} {end} x out of box"
            assert -0.5 <= y <= 0.5, f"{key}/{bone['name']} {end} y out of box"
            assert 0.0 <= z <= 1.0, f"{key}/{bone['name']} {end} z out of box"
    for a, b in template.mirror_pairs:
        assert a in names and b in names, f"{key} mirrors a bone it does not have"


@pytest.mark.parametrize("key", sorted(EXPECTED_TEMPLATES))
def test_fitting_produces_no_zero_length_bones(key):
    fitted = rigging.fit_template(rigging.get_template(key), [-1, -1, 0], [1, 1, 2])
    for bone in fitted:
        assert rigging._distance(bone["head"], bone["tail"]) > 0


@pytest.mark.parametrize("key", sorted(EXPECTED_TEMPLATES))
def test_template_hierarchy_is_well_formed(key):
    t = rigging.get_template(key)
    names = {b["name"] for b in t.bones}
    assert len(names) == len(t.bones)
    assert t.root in names
    for bone in t.bones:
        assert bone["parent"] is None or bone["parent"] in names
    # Exactly one root: a second parentless bone would export as a detached
    # skeleton the pose editor has no way to present.
    assert sum(1 for b in t.bones if b["parent"] is None) == 1
    for a, b in t.mirror_pairs:
        assert a in names and b in names


@pytest.mark.parametrize("key", sorted(EXPECTED_TEMPLATES))
def test_mirror_pairs_are_actually_mirrored(key):
    """A .L/.R pair must be reflections in X, or the future mirror-pose button
    silently produces a lopsided pose."""
    t = rigging.get_template(key)
    by_name = {b["name"]: b for b in t.bones}
    for left, right in t.mirror_pairs:
        for end in ("head", "tail"):
            lx, ly, lz = by_name[left][end]
            rx, ry, rz = by_name[right][end]
            assert lx == pytest.approx(-rx)
            assert ly == pytest.approx(ry)
            assert lz == pytest.approx(rz)


def test_get_template_rejects_unknown():
    with pytest.raises(ValueError, match="unknown skeleton template"):
        rigging.get_template("dragon")


def test_catalog_shape():
    entries = rigging.catalog()
    assert all(set(e) == {"key", "label"} for e in entries)


def test_malformed_template_is_skipped_not_fatal(tmp_path, monkeypatch):
    (tmp_path / "good.json").write_text(
        json.dumps(
            {
                "key": "good",
                "label": "Good",
                "root": "a",
                "bones": [{"name": "a", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]}],
            }
        )
    )
    (tmp_path / "broken.json").write_text('{"key": "broken"}')
    monkeypatch.setattr(rigging, "TEMPLATE_DIR", tmp_path)
    monkeypatch.setattr(rigging, "_templates", None)
    assert set(rigging.templates()) == {"good"}


def test_a_template_whose_key_mismatches_its_filename_is_skipped(tmp_path, monkeypatch):
    """The key comes out of the JSON body and is interpolated into the preview
    cache's filenames -- and ``poselib.template_digest`` reads
    ``TEMPLATE_DIR/<key>.json`` back, so a key naming a different file would
    hash the wrong bytes or none at all. Enforced at load, log-and-skip like
    every other malformed template."""
    (tmp_path / "renamed.json").write_text(
        json.dumps(
            {
                "key": "original",
                "label": "Original",
                "root": "a",
                "bones": [{"name": "a", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]}],
            }
        )
    )
    monkeypatch.setattr(rigging, "TEMPLATE_DIR", tmp_path)
    monkeypatch.setattr(rigging, "_templates", None)
    assert rigging.templates() == {}


def test_a_path_unsafe_template_key_is_rejected():
    with pytest.raises(ValueError, match="not a safe path component"):
        rigging._parse_template(
            {
                "key": "../evil",
                "label": "X",
                "root": "a",
                "bones": [{"name": "a", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]}],
            }
        )


def test_a_parented_root_is_rejected():
    """``blender_worker._apply_root_translation`` inverts only the root's own
    rest frame, which is sound exactly while no parent's pose sits above it --
    so the registry refuses the template rather than the worker guessing."""
    with pytest.raises(ValueError, match="must be parentless"):
        rigging._parse_template(
            {
                "key": "x",
                "label": "X",
                "root": "b",
                "bones": [
                    {"name": "a", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]},
                    {"name": "b", "parent": "a", "head": [0, 0, 1], "tail": [0, 0, 2]},
                ],
            }
        )


def test_every_shipped_template_root_is_parentless():
    for key, template in rigging.templates().items():
        parent = next(b["parent"] for b in template.bones if b["name"] == template.root)
        assert parent is None, f"{key}: root {template.root!r} is parented"


def test_a_template_with_a_disconnected_parent_cycle_is_rejected():
    """The 2026-09-14 audit, finding poser-02: every existing check here
    resolves each bone's own parent and confirms one root, but none of them
    walk the chain from a bone to prove it actually reaches that root -- a
    cycle among two bones neither of which is the root (``a``/``b`` parent
    each other) satisfies every prior check. ``_build_armature`` parents
    bones in one pass with no cycle check of its own, so this must be caught
    here, at load, the same guarantee ``check_skeleton_structure`` and
    ``validate_skeleton`` already give a caller-supplied skeleton."""
    with pytest.raises(ValueError, match="cycle"):
        rigging._parse_template(
            {
                "key": "x",
                "label": "X",
                "root": "root",
                "bones": [
                    {
                        "name": "root",
                        "parent": None,
                        "head": [0, 0, 0],
                        "tail": [0, 0, 1],
                    },
                    {"name": "a", "parent": "b", "head": [0, 0, 0], "tail": [0, 0, 1]},
                    {"name": "b", "parent": "a", "head": [0, 0, 0], "tail": [0, 0, 1]},
                ],
            }
        )


def test_template_with_unknown_parent_is_rejected():
    with pytest.raises(ValueError, match="unknown parent"):
        rigging._parse_template(
            {
                "key": "x",
                "label": "X",
                "root": "a",
                "bones": [{"name": "a", "parent": "ghost", "head": [0, 0, 0], "tail": [0, 0, 1]}],
            }
        )


# --- fitting ----------------------------------------------------------------


def test_fit_maps_normalized_coords_onto_the_bbox():
    t = rigging.get_template("humanoid")
    lo, hi = [-1.0, -0.5, 0.0], [1.0, 0.5, 4.0]
    fitted = rigging.fit_template(t, lo, hi)
    by_name = {b["name"]: b for b in fitted}
    # head's tail is z=1.0 normalized -> the top of the bbox; hips sit on the
    # bbox centre in x and y.
    assert by_name["head"]["tail"][2] == pytest.approx(4.0)
    assert by_name["hips"]["head"][0] == pytest.approx(0.0)
    assert by_name["hips"]["head"][2] == pytest.approx(0.53 * 4.0)
    # x spans -0.5..0.5 of a 2.0-wide box, so 0.07 normalized is 0.14 world.
    assert by_name["thigh.L"]["head"][0] == pytest.approx(0.14)


def test_fit_respects_an_off_origin_bbox():
    t = rigging.get_template("humanoid")
    fitted = rigging.fit_template(t, [10.0, 20.0, 30.0], [11.0, 21.0, 31.0])
    for bone in fitted:
        for point in (bone["head"], bone["tail"]):
            assert 9.9 <= point[0] <= 11.1
            assert 19.9 <= point[1] <= 21.1
            assert 29.9 <= point[2] <= 31.1


@pytest.mark.parametrize("key", ["humanoid", "quadruped"])
def test_fit_never_produces_a_zero_length_bone(key):
    """Blender deletes zero-length bones on leaving edit mode, taking their
    children with them -- so a flat bbox must not be able to produce one."""
    t = rigging.get_template(key)
    for lo, hi in [
        ([-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]),
        ([-1.0, 0.0, 0.0], [1.0, 0.0, 2.0]),  # zero depth
        ([0.0, 0.0, 0.0], [0.0, 0.0, 2.0]),  # a line
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),  # a point
    ]:
        for bone in rigging.fit_template(t, lo, hi):
            assert rigging._distance(bone["head"], bone["tail"]) > 0.0


def test_fit_preserves_parent_names():
    t = rigging.get_template("quadruped")
    fitted = rigging.fit_template(t, [-1, -2, 0], [1, 2, 1])
    assert [b["parent"] for b in fitted] == [b["parent"] for b in t.bones]


# --- corrected joints ---------------------------------------------------------


def test_validate_joints_accepts_a_full_corrected_skeleton():
    template = rigging.get_template("humanoid")
    fitted = rigging.fit_template(template, [-1, -1, 0], [1, 1, 2])
    payload = {
        "bones": [
            {"name": b["name"], "head": b["head"], "tail": b["tail"]} for b in fitted
        ]
    }
    out = rigging.validate_joints(payload, template)
    assert [b["name"] for b in out] == [b["name"] for b in template.bones]
    assert out[0]["parent"] == template.bones[0]["parent"]


def test_validate_joints_rejects_a_missing_bone():
    template = rigging.get_template("humanoid")
    with pytest.raises(ValueError):
        rigging.validate_joints(
            {"bones": [{"name": "hips", "head": [0, 0, 0], "tail": [0, 0, 1]}]}, template
        )


def test_validate_joints_rejects_an_unknown_bone():
    template = rigging.get_template("humanoid")
    fitted = rigging.fit_template(template, [-1, -1, 0], [1, 1, 2])
    payload = {
        "bones": [
            {"name": b["name"], "head": b["head"], "tail": b["tail"]} for b in fitted
        ]
        + [{"name": "wing.L", "head": [0, 0, 0], "tail": [0, 0, 1]}]
    }
    with pytest.raises(ValueError, match="unknown bone"):
        rigging.validate_joints(payload, template)


def test_validate_joints_rejects_a_zero_length_bone():
    template = rigging.get_template("humanoid")
    payload = {
        "bones": [
            {"name": b["name"], "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 0.0]}
            for b in template.bones
        ]
    }
    with pytest.raises(ValueError):
        rigging.validate_joints(payload, template)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_validate_joints_rejects_non_finite_coordinates(bad):
    """NaN makes the zero-length comparison False, so it used to reach rig.json
    and only fail inside Blender."""
    template = rigging.get_template("humanoid")
    fitted = rigging.fit_template(template, [-1, -1, 0], [1, 1, 2])
    bones = [{"name": b["name"], "head": b["head"], "tail": b["tail"]} for b in fitted]
    bones[0] = dict(bones[0], head=[bad, 0.0, 0.0])
    with pytest.raises(ValueError, match="not numeric"):
        rigging.validate_joints({"bones": bones}, template)


def test_rig_spec_carries_corrected_bones_when_given_them(tmp_path):
    template = rigging.get_template("humanoid")
    fitted = rigging.fit_template(template, [-1, -1, 0], [1, 1, 2])
    assert "bones" not in rigging.rig_spec(tmp_path, "humanoid")
    assert rigging.rig_spec(tmp_path, "humanoid", fitted)["bones"] == fitted


# --- image-informed landmarks ------------------------------------------------
#
# The host reads the reference image, converts what it finds into the
# template's own normalized space, and hands the result over as
# ``template_bones``. The worker then fits *that* template exactly as it fits
# the shipped one, so the bbox scaling stays in one place and host and worker
# still cannot disagree about where a joint goes.


def _landmarks() -> list[dict]:
    """A normalized bone list shaped like a template's, with one joint moved
    somewhere the shipped template would never put it."""
    template = rigging.get_template("humanoid")
    moved = []
    for bone in template.bones:
        bone = {k: list(v) if isinstance(v, list) else v for k, v in bone.items()}
        if bone["name"] == "forearm.L":
            bone["head"] = [0.40, 0.0, 0.20]
        moved.append(bone)
    return moved


def test_rig_spec_leaves_out_the_landmark_fields_when_there_are_none(tmp_path):
    spec = rigging.rig_spec(tmp_path, "humanoid")
    assert "template_bones" not in spec
    assert "fit" not in spec


def test_rig_spec_carries_landmarks_and_how_they_were_found(tmp_path):
    bones = _landmarks()
    fit = {"method": "pose2d", "model": "vitpose", "confidence": 0.81}
    spec = rigging.rig_spec(tmp_path, "humanoid", template_bones=bones, fit=fit)
    assert spec["template_bones"] == bones
    assert spec["fit"] == fit


def test_the_worker_fits_the_landmark_template_not_the_shipped_one(tmp_path):
    """The whole seam in one assertion: with landmarks in the spec, the joints
    the armature is built from are the ones measured off the reference image,
    scaled onto the mesh bbox by the same fit_template every rig uses."""
    from warlock.pipelines import blender_worker

    spec = rigging.rig_spec(
        tmp_path, "humanoid", template_bones=_landmarks(), fit={"method": "pose2d"}
    )
    bones, fit = blender_worker._rig_bones(spec, [-1.0, -1.0, 0.0], [1.0, 1.0, 2.0])
    informed = {b["name"]: b for b in bones}["forearm.L"]["head"]
    shipped = {b["name"]: b for b in blender_worker._rig_bones(
        rigging.rig_spec(tmp_path, "humanoid"), [-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]
    )[0]}["forearm.L"]["head"]
    assert informed != shipped
    # 0.40 of a 2-unit-wide bbox centred on 0, and 0.20 of a 2-unit-tall one
    # off a floor at 0 -- fit_template's arithmetic, unchanged.
    assert informed == pytest.approx([0.80, 0.0, 0.40])
    assert fit == {"method": "pose2d"}


def test_a_rig_with_no_landmarks_records_the_bbox_fit(tmp_path):
    from warlock.pipelines import blender_worker

    _, fit = blender_worker._rig_bones(
        rigging.rig_spec(tmp_path, "humanoid"), [-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]
    )
    assert fit == {"method": "bbox"}


def test_joints_the_user_moved_still_beat_the_landmarks(tmp_path):
    """Adjust-joints is the user overruling the fit, whichever fit it was."""
    from warlock.pipelines import blender_worker

    template = rigging.get_template("humanoid")
    corrected = rigging.fit_template(template, [-5.0, -5.0, 0.0], [5.0, 5.0, 10.0])
    spec = rigging.rig_spec(
        tmp_path, "humanoid", bones=corrected, template_bones=_landmarks()
    )
    bones, fit = blender_worker._rig_bones(spec, [-1.0, -1.0, 0.0], [1.0, 1.0, 2.0])
    assert bones == corrected
    assert fit == {"method": "manual"}


def test_landmarks_that_are_not_this_templates_bones_are_ignored(tmp_path):
    """The worker reads its spec off a pipe. A bone list that does not name
    this template's bones would build an armature whose parents do not resolve,
    so it falls back to the fit that is always available and says so."""
    from warlock.pipelines import blender_worker

    spec = rigging.rig_spec(
        tmp_path, "humanoid", template_bones=[{"name": "tentacle", "parent": None,
                                               "head": [0, 0, 0], "tail": [0, 0, 1]}]
    )
    bones, fit = blender_worker._rig_bones(spec, [-1.0, -1.0, 0.0], [1.0, 1.0, 2.0])
    assert [b["name"] for b in bones] == [b["name"] for b in rigging.get_template("humanoid").bones]
    assert fit == {"method": "bbox"}


# --- mirroring ---------------------------------------------------------------


def test_mirror_quaternion_reflects_across_the_yz_plane():
    # A rotation about Z becomes the opposite rotation about Z under an X mirror.
    half = math.radians(30) / 2
    q = [0.0, 0.0, math.sin(half), math.cos(half)]
    assert rigging.mirror_quaternion(q) == pytest.approx(
        [0.0, 0.0, -math.sin(half), math.cos(half)]
    )


def test_mirroring_twice_is_the_identity():
    q = [0.1830, 0.2588, 0.3536, 0.8810]
    twice = rigging.mirror_quaternion(rigging.mirror_quaternion(q))
    assert twice == pytest.approx(q)


def test_a_rotation_about_x_survives_mirroring():
    """A limb swinging forward/back mirrors to the same swing, not its opposite."""
    half = math.radians(40) / 2
    q = [math.sin(half), 0.0, 0.0, math.cos(half)]
    assert rigging.mirror_quaternion(q) == pytest.approx(q)


def test_mirror_pose_fills_in_the_other_side_and_leaves_centre_bones_alone():
    pairs = [["upper_arm.L", "upper_arm.R"]]
    posed = {"upper_arm.L": [0.0, 0.5, 0.0, 0.8660], "spine": [0.1, 0.0, 0.0, 0.9950]}
    out = rigging.mirror_pose(posed, pairs)
    assert out["upper_arm.R"] == pytest.approx([0.0, -0.5, 0.0, 0.8660])
    assert out["spine"] == pytest.approx([0.1, 0.0, 0.0, 0.9950])


# --- shipped pose libraries -------------------------------------------------


def test_preset_poses_validate_against_their_template():
    for key in rigging.templates():
        bone_names = [b["name"] for b in rigging.get_template(key).bones]
        for preset in rigging.preset_poses(key):
            # The same validation a browser-saved pose goes through: unknown
            # bones and non-unit quaternions are rejected identically, so a
            # shipped preset can never be a thing the API would refuse.
            rigging.validate_pose(preset, bone_names)


def test_a_template_with_no_preset_file_returns_an_empty_list():
    assert rigging.preset_poses("serpent") == []


def test_unknown_template_presets_raise():
    with pytest.raises(ValueError):
        rigging.preset_poses("not-a-template")


def test_the_deformation_battery_declares_delta_space():
    """The 2026-09-16 QA sheet finding: templates/deform_qa/humanoid.json had
    no top-level ``space``, so ``_load_pose_library`` dropped it silently
    (unlike ``parse_clip_library``, which has always read the same field off
    a clip library) and every deform-QA cell rendered in the pose editor's
    ``node`` frame regardless. Broken specifically on a *measured* rig (a
    real render of Quaternius's Superhero Male showed the squat folding the
    legs up behind the head and "arms overhead" pointing the arms straight
    forward) -- see blender_worker.POSE_SPACES for why a node-local value
    bakes in the rest orientation of the skeleton it was authored against."""
    poses = rigging.deform_battery("humanoid")
    assert poses, "the humanoid battery should not be empty"
    for pose in poses:
        assert pose.get("space") == "delta", f"{pose['name']!r} has no delta space"


# --- clip libraries -----------------------------------------------------


def test_parse_clip_library_rejects_duplicate_pose_names():
    """The same rule ``_parse_template`` states for bone names: a library
    internally inconsistent by name is a malformed file, not two poses
    silently collapsing into one (the last write into ``poses[name]`` winning
    over the first, with nothing said)."""
    raw = {
        "poses": [{"name": "rest", "bones": {}}, {"name": "rest", "bones": {}}],
        "clips": [],
    }
    with pytest.raises(ValueError, match="duplicate pose names"):
        rigging.parse_clip_library(raw)


def test_parse_clip_library_rejects_duplicate_clip_names():
    raw = {
        "poses": [{"name": "a", "bones": {}}, {"name": "b", "bones": {}}],
        "clips": [
            {"name": "walk", "keys": ["a", "b"], "segments": [1]},
            {"name": "walk", "keys": ["b", "a"], "segments": [1]},
        ],
    }
    with pytest.raises(ValueError, match="duplicate clip names"):
        rigging.parse_clip_library(raw)


# --- the read-door ceilings the 2026-09-11 audit added (poser-03/poser-04) --
#
# _load_templates, _load_pose_library and _load_clip_library each did a bare
# json.loads(path.read_text(...)) with no byte ceiling at all -- unlike
# read_record, which stats the file and refuses anything over MAX_RECORD_BYTES
# *before* reading it. parse_clip_library separately had no per-item count
# ceiling, unlike service.clips._check_shape's write-side caps. Both are
# "a malformed/oversized file costs you that entry, not the app": every
# loader here already wraps its body in try/except Exception and logs, so a
# refusal from the new guards is swallowed exactly like a bad body always was.


def test_an_oversized_template_is_skipped_not_fully_read(tmp_path, monkeypatch):
    """A file that *would* parse cleanly -- a real, well-formed template --
    costs a log line instead of a read once it is over the ceiling, the same
    way an oversized pose.json already does. A malformed file would be
    skipped either way and prove nothing about the size guard specifically."""
    monkeypatch.setattr(rigging, "TEMPLATE_DIR", tmp_path)
    monkeypatch.setattr(rigging, "MAX_TEMPLATE_BYTES", 200)
    (tmp_path / "giant.json").write_text(
        json.dumps(
            {
                "key": "giant",
                "label": "Giant",
                "root": "root",
                "bones": [
                    {"name": "root", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 0.5]}
                ],
                # Ignored by _parse_template, which reads only key/label/root/
                # bones/mirror_pairs -- exactly the kind of harmless bulk a
                # hand-edited or synced file could carry by accident.
                "padding": "x" * 1000,
            }
        ),
        encoding="utf-8",
    )
    # _load_templates, not the cached templates(): the module-level cache
    # would otherwise still hold whatever earlier tests already loaded.
    assert rigging._load_templates() == {}


def test_an_oversized_pose_library_file_is_skipped_not_fully_read(tmp_path, monkeypatch):
    monkeypatch.setattr(rigging, "MAX_TEMPLATE_BYTES", 32)
    (tmp_path / "humanoid.json").write_text(
        json.dumps({"poses": [{"name": "x" * 200, "bones": {}}]}), encoding="utf-8"
    )
    assert rigging._load_pose_library(tmp_path) == {}


def test_an_oversized_clip_library_file_is_skipped_not_fully_read(tmp_path, monkeypatch):
    """poser-03: same gap, the one directory of the three that is genuinely
    user-editable (rigging.user_clip_dir's own docstring)."""
    monkeypatch.setattr(rigging, "MAX_CLIP_LIBRARY_BYTES", 32)
    (tmp_path / "humanoid.json").write_text(
        json.dumps(
            {
                "poses": [{"name": "rest", "bones": {}}],
                "clips": [{"name": "idle", "keys": ["rest"], "segments": [1]}],
            }
        ),
        encoding="utf-8",
    )
    assert rigging._load_clip_library(tmp_path) == {}


def test_an_oversized_user_clip_library_is_refused_before_being_read(tmp_path):
    """poser-04: parse_clip_library had no per-item count ceiling at all, so a
    library with more poses than service.clips._check_shape's own write-side
    MAX_LIBRARY_KEYS cap would allow parsed in full instead of being refused
    the way the write door already refuses it. The cap itself rose from 256
    to 1024 on 2026-09-12 (schema v3, design decision D2) so that clips
    imported from Mixamo/Rigify have room to sit beside the shipped ones
    without silently landing on a ceiling sized for the original five --
    the ``"at most 1024 key poses"`` text below moves if that cap ever does."""
    raw = {
        "poses": [
            {"name": f"p{i}", "bones": {}} for i in range(rigging.MAX_CLIP_LIBRARY_POSES + 1)
        ],
        "clips": [{"name": "idle", "keys": ["p0"], "segments": [1]}],
    }
    with pytest.raises(ValueError, match="at most 1024 key poses"):
        rigging.parse_clip_library(raw)

    # And the read door built on it: a library this shape sitting on disk --
    # exactly where service.clips.save would have written a user's edited
    # library, under rigging.user_clip_dir() -- costs itself, not the app.
    (tmp_path / "humanoid.json").write_text(json.dumps(raw), encoding="utf-8")
    assert rigging._load_clip_library(tmp_path) == {}


def test_a_clips_key_list_over_the_cap_is_refused():
    """The other half of poser-04's cap: service.clips._check_shape's
    MAX_KEYS, restated here so the two doors agree on a single clip's key
    count too, not just the library's pose count."""
    raw = {
        "poses": [{"name": f"p{i}", "bones": {}} for i in range(2)],
        "clips": [
            {
                "name": "long",
                "keys": ["p0"] * (rigging.MAX_CLIP_KEYS + 1),
                "segments": [1] * rigging.MAX_CLIP_KEYS,
            }
        ],
    }
    with pytest.raises(ValueError, match="at most 64 keys"):
        rigging.parse_clip_library(raw)


def test_shipped_clips_survive_a_cache_invalidation_mid_read(monkeypatch):
    """``shipped_clip_library`` used to check ``_clips is None`` and then
    read the module global a *second* time (``library = _clips.get(...)``)
    to actually fetch the library. Between those two reads is exactly the
    window Poser's save door's ``invalidate_clips()`` can land in, on another
    thread, once the cache is already warm -- and when it does, the second
    read finds ``None`` and raises ``AttributeError: 'NoneType' object has
    no attribute 'get'`` instead of simply serving the library this call had
    already found present.

    A real second thread would only land in that window *sometimes*, so this
    forces it deterministically with ``sys.settrace``: fire
    ``invalidate_clips()`` from a line-event callback right before the
    function's own read of the cache executes, the same interruption point a
    genuinely concurrent ``invalidate_clips()`` call would have to hit.
    """
    import inspect
    import sys as sys_mod

    # Warm the cache first -- the crash only reaches the vulnerable line when
    # the None-check is skipped because the library is already loaded.
    monkeypatch.setattr(rigging, "_clips", None)
    rigging.shipped_clip_library("humanoid")
    assert rigging._clips is not None

    source_lines, start_line = inspect.getsourcelines(rigging.shipped_clip_library)
    target_line = next(
        start_line + offset
        for offset, line in enumerate(source_lines)
        if "library = " in line and ".get(template_key)" in line
    )

    fired = False

    def tracer(frame, event, arg):
        nonlocal fired
        if (
            not fired
            and event == "line"
            and frame.f_code is rigging.shipped_clip_library.__code__
            and frame.f_lineno == target_line
        ):
            fired = True
            rigging.invalidate_clips()
        return tracer

    sys_mod.settrace(tracer)
    try:
        result = rigging.shipped_clip_library("humanoid")
    finally:
        sys_mod.settrace(None)

    assert fired, "the interruption point was never reached; the test no longer matches the source"
    assert result["clips"]


# --- the authored clip libraries --------------------------------------------
#
# A template can only produce a character sheet if its skeleton has clips --
# ``service.troupe.create_charsheet`` refuses on exactly that, and for a long
# time ``humanoid`` was the only template that had any. These are the families
# that can be sent to Troupe, so the list is a claim and not a fixture: adding
# a library is a deliberate act and so is losing one.

AUTHORED_CLIP_LIBRARIES = ("humanoid", "quadruped", "bird", "blob")

#: The frame counts a library has to survive, not just the one it was authored
#: at. ``resample_clip`` re-times an authored clip onto whatever the layout
#: asks for, including a single frame (which is the first key) and counts that
#: cannot give every authored segment a frame of its own.
FRAME_COUNTS = (1, 2, 4, 6, 8, 12)


def test_the_authored_clip_libraries_are_exactly_the_four_families():
    with_clips = {k for k in rigging.templates() if rigging.clip_library(k)["clips"]}
    assert with_clips == set(AUTHORED_CLIP_LIBRARIES)


@pytest.mark.parametrize("key", AUTHORED_CLIP_LIBRARIES)
@pytest.mark.parametrize("frames", FRAME_COUNTS)
def test_every_template_with_a_clip_library_expands_to_the_frame_table(key, frames):
    """The whole point of authoring a library: the frame table asks, and the
    library fills it exactly. A clip that expanded to one frame too few would
    reach the renderer as ``check_frame_counts``' ValueError an hour and 256
    EEVEE frames later, so it is asked here, at every count Troupe offers.

    Every clip the library defines, open-vocabulary and all -- not only the
    legacy five -- and one direction each: the claim here is about the
    per-clip frame table, and a library with ten clips at ``directions=8``
    would trip ``MAX_CELLS`` for a reason this test is not about."""
    from warlock import clips

    library = rigging.clip_library(key)
    names = [c["name"] for c in library["clips"]]
    layout = {
        "movements": [{"name": name, "frames": frames, "directions": 1} for name in names]
    }
    records = clips.expand_clips(key, layout)
    assert set(records) == set(names)
    for name, rows in records.items():
        assert len(rows) == frames, f"{key}/{name} at {frames} frames"
        assert all(r["bones"] for r in rows), f"{key}/{name} expanded to empty poses"


@pytest.mark.parametrize("key", AUTHORED_CLIP_LIBRARIES)
def test_every_authored_library_fills_troupes_default_frame_table(key):
    """The layout a user gets without touching anything asks for all five
    animations, so a library carrying only three refuses the default sheet."""
    from warlock import clips
    from warlock.pipelines import charsheet

    records = clips.expand_clips(key)
    assert set(records) == {name for name, *_rest in charsheet.ANIMATIONS}


@pytest.mark.parametrize("key", AUTHORED_CLIP_LIBRARIES)
def test_every_authored_clip_library_is_delta_space(key):
    """``space`` is per file and it is the whole meaning of every quaternion in
    it: in ``delta`` a value is a rotation from the bone's own rest, in ``node``
    it is absolute against the parent joint. The two are not interchangeable --
    the same numbers read the other way contort the skeleton silently -- and a
    library that said nothing at all would be read as ``node``, which is what
    ``parse_clip_library`` defaults to for the files written before the field."""
    library = rigging.clip_library(key)
    assert library["space"] == "delta"
    # Per file, so every clip in it agrees, and the records the renderer sees
    # carry it too -- ``_blend`` treats a missing bone differently in each
    # space, so a record that lost the field would blend a folded wing open.
    assert {c["space"] for c in library["clips"]} == {"delta"}


def test_a_quadruped_walk_moves_the_diagonal_pairs_out_of_phase():
    """The gait claim, and the one a plausible-looking file gets wrong.

    A four-beat lateral-sequence walk puts the footfalls in the order LH, LF,
    RH, RF -- the four legs a quarter cycle apart, the two legs of each girdle
    in strict antiphase. A walk whose legs swing together is a rocking horse,
    and it is invisible in a single frame: it only shows in the expansion, so
    that is what this reads rather than the authored file.

    Every quadruped bone lies in a sagittal plane (head and tail share an x),
    so Blender's roll-free bone frame puts the bone's local X on world X and
    the delta quaternion's x component is ``sin(swing / 2)`` -- negative
    forward. That is why the swing can be read straight off the record.
    """
    from warlock import clips

    records = clips.expand_clips(
        "quadruped", {"movements": [{"name": "walk", "frames": 8}]}
    )["walk"]
    legs = ("rear_upper.L", "front_upper.L", "rear_upper.R", "front_upper.R")
    swing = {bone: [r["bones"][bone][0] for r in records] for bone in legs}

    # Furthest forward -- the moment that foot strikes -- one per quarter.
    reach = {bone: values.index(min(values)) for bone, values in swing.items()}
    assert reach == {
        "rear_upper.L": 0,
        "front_upper.L": 2,
        "rear_upper.R": 4,
        "front_upper.R": 6,
    }
    # No two legs move together, which is the failure the ordering exists to
    # avoid: four legs sharing one curve is one leg drawn four times.
    for a, b in itertools.combinations(legs, 2):
        assert swing[a] != swing[b], f"{a} and {b} swing in phase"
    # And each girdle's pair is exactly half a cycle apart, not merely unequal.
    for left, right in (("front_upper.L", "front_upper.R"), ("rear_upper.L", "rear_upper.R")):
        shifted = swing[left][4:] + swing[left][:4]
        assert swing[right] == pytest.approx(shifted), f"{left}/{right} are not antiphase"


# The forward kinematics the assertion below stands on. ``bpy`` is installed on
# no machine the default lane runs on, so Blender's two composition rules are
# reproduced here rather than imported:
#
#   pose(b) = pose(parent) @ rest(parent)^-1 @ rest(b) @ basis(b)
#
# with ``rest(parent)^-1 @ rest(b) @ basis(b)`` being exactly
# ``rigging.node_from_delta`` applied to the parent-relative rest -- the one
# definition both ends of the pose feature already share -- and the bone's own
# rest frame coming from ``vec_roll_to_mat3`` with roll 0, which is what
# ``_build_armature`` leaves every bone at. Which way a bone's local X points
# is what every delta quaternion in a clip library *means*, and Blender's
# special case for a bone pointing along -Y flips it: the quadruped's ``spine``
# points exactly -Y, so an approximation here would not be an approximation.


def _bone_frame(direction):
    """Blender's roll-free bone matrix as a quaternion. Local Y is the bone."""
    x, y, z = direction
    theta = 1.0 + y
    alt = x * x + z * z
    if theta > 6.1e-3 or alt > 2.5e-4**2:
        if theta <= 6.1e-3:
            theta = 0.5 * alt + alt * alt * 0.125
        m00, m22, m02 = 1.0 - x * x / theta, 1.0 - z * z / theta, -x * z / theta
    else:
        # A bone pointing straight along -Y: local X is *minus* world X.
        m00, m22, m02 = -1.0, 1.0, 0.0
    return _quat_from_matrix([[m00, x, m02], [-x, y, -z], [m02, z, m22]])


def _quat_from_matrix(m):
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [(m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s]
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        q = [0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s]
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        q = [(m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s]
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        q = [(m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s]
    norm = math.sqrt(sum(v * v for v in q))
    return [v / norm for v in q]


def _rotate(q, v):
    x, y, z, w = q
    return [
        (1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y - z * w) * v[1] + 2 * (x * z + y * w) * v[2],
        2 * (x * y + z * w) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z - x * w) * v[2],
        2 * (x * z - y * w) * v[0] + 2 * (y * z + x * w) * v[1] + (1 - 2 * (x * x + y * y)) * v[2],
    ]


def _compose(a, b):
    """Two rotations, one after the other. ``mathutils``' ``@``, by hand."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _joint_points(key, bones=None, root_translation=None):
    """Every joint of ``key``'s unit-box fit, posed by one frame's deltas.

    The unit box is the frame the library was authored in, and the one
    ``fit_template`` maps onto a real mesh, so a height of 1.0 makes the
    root offset (which is in character-height units) directly comparable.
    """
    fitted = rigging.fit_template(rigging.get_template(key), [-0.5, -0.5, 0.0], [0.5, 0.5, 1.0])
    by_name = {b["name"]: b for b in fitted}
    rest, posed = {}, {}
    points = []
    for bone in fitted:
        head, tail = bone["head"], bone["tail"]
        length = rigging._distance(head, tail)
        direction = [(t - h) / length for h, t in zip(head, tail, strict=True)]
        rest[bone["name"]] = _bone_frame(direction)
        delta = (bones or {}).get(bone["name"]) or [0.0, 0.0, 0.0, 1.0]
        parent = bone["parent"]
        if parent is None:
            rotation = rigging.node_from_delta(rest[bone["name"]], delta)
            origin = list(head)
        else:
            inverse = [-v for v in rest[parent][:3]] + [rest[parent][3]]
            relative = _compose(inverse, rest[bone["name"]])
            parent_rotation, parent_origin = posed[parent]
            rotation = _compose(parent_rotation, rigging.node_from_delta(relative, delta))
            offset = [h - p for h, p in zip(head, by_name[parent]["head"], strict=True)]
            moved = _rotate(parent_rotation, _rotate(inverse, offset))
            origin = [p + m for p, m in zip(parent_origin, moved, strict=True)]
        posed[bone["name"]] = (rotation, origin)
        lift = float((root_translation or [0.0, 0.0, 0.0])[2])
        end = [o + d for o, d in zip(origin, _rotate(rotation, [0.0, length, 0.0]), strict=True)]
        points.append([origin[0], origin[1], origin[2] + lift])
        points.append([end[0], end[1], end[2] + lift])
    return points


def test_the_forward_kinematics_this_file_uses_leaves_a_rest_pose_where_it_found_it():
    """The FK above is only worth as much as its rest case: an empty pose has
    to reproduce ``fit_template``'s own joints, or every claim built on it is
    measuring the arithmetic rather than the clip."""
    for key in AUTHORED_CLIP_LIBRARIES:
        fitted = rigging.fit_template(
            rigging.get_template(key), [-0.5, -0.5, 0.0], [0.5, 0.5, 1.0]
        )
        expected = [p for bone in fitted for p in (bone["head"], bone["tail"])]
        for got, want in zip(_joint_points(key), expected, strict=True):
            assert got == pytest.approx(want, abs=1e-6), key


@pytest.mark.parametrize("key", AUTHORED_CLIP_LIBRARIES)
def test_every_one_shot_attack_leaves_the_rest_bounding_box(key):
    """An attack that fits inside the rest silhouette is not an attack.

    Rearing, spreading wings, swinging overhead: the apex is supposed to leave
    the box the character stands in, which is precisely why the sheet renderer
    frames a row from the *union* of its cells rather than from the rest pose.
    Asserted on the expansion through ``fit_template`` and
    ``rigging.node_from_delta``, with no bpy, so it stays in the default lane.
    """
    from warlock import clips

    rest_crown = max(p[2] for p in _joint_points(key))
    records = clips.expand_clips(key, {"movements": [{"name": "attack", "frames": 6}]})["attack"]
    peak = max(
        p[2]
        for record in records
        for p in _joint_points(key, record["bones"], record.get("root_translation"))
    )
    # A fifth of a hair over would be an accident; 0.02 of the character's own
    # height is the smallest rise that reads as one at 64 pixels.
    assert peak > rest_crown + 0.02, (
        f"{key}: attack peaks at {peak:.3f}, rest crown {rest_crown:.3f}"
    )


# --- pose payloads ----------------------------------------------------------


def test_validate_pose_accepts_a_unit_quaternion():
    pose = rigging.validate_pose({"name": "idle", "bones": {"hips": [0, 0, 0, 1]}})
    assert pose == {"name": "idle", "bones": {"hips": [0.0, 0.0, 0.0, 1.0]}}


def test_validate_pose_renormalizes_drifted_quaternions():
    """Accumulated gizmo drags leave the unit sphere by float noise; refusing
    the save over that would be indefensible."""
    pose = rigging.validate_pose({"name": "wave", "bones": {"hips": [0, 0, 0, 2.0]}})
    assert pose["bones"]["hips"] == pytest.approx([0.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"name": "", "bones": {"hips": [0, 0, 0, 1]}}, "requires a name"),
        ({"name": "x" * 65, "bones": {"hips": [0, 0, 0, 1]}}, "at most"),
        ({"name": "a", "bones": {}}, "non-empty"),
        ({"name": "a"}, "non-empty"),
        ({"name": "a", "bones": {"hips": [0, 0, 1]}}, "4-element"),
        ({"name": "a", "bones": {"hips": ["x", 0, 0, 1]}}, "not numeric"),
        ({"name": "a", "bones": {"hips": [0, 0, 0, 0]}}, "degenerate"),
        # NaN passes the norm check (abs(nan - 1.0) > eps is False) and used to
        # be stored verbatim; inf normalizes to nan.
        ({"name": "a", "bones": {"hips": [float("nan"), 0, 0, 1]}}, "not numeric"),
        ({"name": "a", "bones": {"hips": [float("inf"), 0, 0, 1]}}, "not numeric"),
        ({"name": "a", "bones": {"hips": [0, 0, 0, float("-inf")]}}, "not numeric"),
    ],
)
def test_validate_pose_rejects_bad_payloads(payload, match):
    with pytest.raises(ValueError, match=match):
        rigging.validate_pose(payload)


def test_validate_pose_rejects_unknown_bones_when_told_the_skeleton():
    with pytest.raises(ValueError, match="unknown bone"):
        rigging.validate_pose({"name": "a", "bones": {"tentacle": [0, 0, 0, 1]}}, ["hips"])


# --- ids and paths ----------------------------------------------------------


def test_new_id_is_accepted_by_its_own_validator():
    assert rigging.is_valid_id(rigging.new_id())


@pytest.mark.parametrize(
    "bad", ["..", "../../etc", "ABCDEF012345", "0123456789ab/x", "", "0123456789abc"]
)
def test_traversal_and_malformed_ids_are_rejected(bad):
    assert not rigging.is_valid_id(bad)


# --- the worker boundary ----------------------------------------------------


def test_rig_spec_validates_the_template_before_spawning_anything(tmp_path):
    with pytest.raises(ValueError):
        rigging.rig_spec(tmp_path, "dragon")


def test_rig_spec_paths_all_land_in_the_job_dir(tmp_path):
    spec = rigging.rig_spec(tmp_path, "humanoid")
    assert spec["op"] == "rig"
    for key in ("source_glb", "out_glb", "out_json", "result_path"):
        assert tmp_path in type(tmp_path)(spec[key]).parents


def test_read_rig_returns_none_without_a_rig(tmp_path):
    assert rigging.read_rig(tmp_path) is None
    assert rigging.rig_bone_names(tmp_path) is None


def test_read_rig_survives_corrupt_json(tmp_path):
    (tmp_path / "rig.json").write_text("{not json")
    assert rigging.read_rig(tmp_path) is None


def test_read_rig_round_trips(tmp_path):
    (tmp_path / "rig.json").write_text(json.dumps({"bones": [{"name": "hips"}]}))
    assert rigging.rig_bone_names(tmp_path) == ["hips"]


# --- why the weights are what they are --------------------------------------
#
# Envelope weights are a degraded outcome, not a second success, and until this
# the only trace of *why* the bone-heat solve did not take was a print into a
# subprocess's stdout. Every test below is host-side: ``_skin`` takes bpy as an
# argument and ``_rig_meta`` is pure, which is exactly what makes the decision
# reachable without Blender installed.


class _FakeOps:
    def __init__(self, fail: str | None, weights: bool) -> None:
        self._fail = fail
        self._weights = weights
        self.binds: list[str] = []

    # bpy.ops.object.*
    @property
    def object(self):  # noqa: D401 - mimics bpy's namespace
        return self

    def select_all(self, action: str) -> None:
        pass

    def parent_set(self, type: str) -> None:  # noqa: A002 - bpy's own keyword
        self.binds.append(type)
        if type == "ARMATURE_AUTO" and self._fail is not None:
            raise RuntimeError(self._fail)


class _FakeBpy:
    def __init__(self, fail: str | None = None, weights: bool = True) -> None:
        self.ops = _FakeOps(fail, weights)

        class _View:
            objects = type("O", (), {"active": None})()

        self.context = type("C", (), {"view_layer": _View()})()


class _FakeMesh:
    def __init__(self, weights: bool) -> None:
        self._weights = weights
        self.modifiers: list[object] = []
        self.parent = None
        self.vertex_groups = _FakeGroups(weights)

    def select_set(self, value: bool) -> None:
        pass


class _FakeGroups(list):
    def __init__(self, weights: bool) -> None:
        super().__init__([type("G", (), {"index": 0})()] if weights else [])

    def clear(self) -> None:
        del self[:]


class _FakeArm:
    def select_set(self, value: bool) -> None:
        pass


def _skin_with(fail: str | None, weights: bool):
    from warlock.pipelines import blender_worker

    bpy = _FakeBpy(fail, weights)
    mesh = _FakeMesh(weights)
    if weights:
        weighted = type("VG", (), {"group": 0, "weight": 1.0})()
        mesh.data = type("D", (), {"vertices": [type("V", (), {"groups": [weighted]})()]})()
    else:
        mesh.data = type("D", (), {"vertices": []})()
    return blender_worker._skin(bpy, mesh, _FakeArm()), bpy.ops.binds


def test_a_heat_solve_that_works_reports_no_reason():
    (weighting, reason), binds = _skin_with(None, True)
    assert weighting == "automatic"
    assert reason is None
    assert binds == ["ARMATURE_AUTO"]


def test_a_raising_heat_solve_carries_its_message_back_as_the_reason():
    (weighting, reason), binds = _skin_with("non-manifold input", False)
    assert weighting == "envelope"
    assert "non-manifold input" in reason
    assert binds == ["ARMATURE_AUTO", "ARMATURE_ENVELOPE"]


def test_a_silent_heat_solve_failure_is_a_reason_too():
    """The 'FINISHED' that leaves every vertex group empty is the case that
    reached the user as nothing at all, so it has to name itself."""
    (weighting, reason), _ = _skin_with(None, False)
    assert weighting == "envelope"
    assert "no vertex weights" in reason


def test_rig_meta_round_trips_the_weighting_reason_through_rig_json(tmp_path):
    """The whole point of the field: what the solve did has to survive the
    subprocess boundary as a file, not as a print nobody reads.

    Host-side on purpose -- ``_rig_meta`` is pure, so this pins the contract
    on a machine with no bpy, which is every machine the app ships on.
    """
    from warlock.pipelines import blender_worker

    template = rigging.get_template("humanoid")
    meta = blender_worker._rig_meta(
        template,
        bones=[{"name": "hips", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]}],
        lo=[0.0, 0.0, 0.0],
        hi=[1.0, 1.0, 1.0],
        weighting="envelope",
        weighting_reason="bone-heat weighting failed: produced no vertex weights",
        adjusted=False,
        fit={"method": "bbox"},
    )
    (tmp_path / "rig.json").write_text(json.dumps(meta), encoding="utf-8")

    rig = rigging.read_rig(tmp_path)
    assert rig["weighting"] == "envelope"
    assert rig["weighting_reason"] == "bone-heat weighting failed: produced no vertex weights"
    # Additive: no version bump travels with it, because every reader is
    # .get-based and a file written before the field stays readable.
    assert rig["version"] == 1


def test_a_successful_rig_records_no_reason(tmp_path):
    from warlock.pipelines import blender_worker

    meta = blender_worker._rig_meta(
        rigging.get_template("humanoid"),
        bones=[],
        lo=[0.0, 0.0, 0.0],
        hi=[1.0, 1.0, 1.0],
        weighting="automatic",
        weighting_reason=None,
        adjusted=False,
        fit={"method": "bbox"},
    )
    assert meta["weighting_reason"] is None


def test_the_weighting_reason_is_derived_and_cannot_be_inherited_by_a_reroll():
    from warlock.service.validation import DERIVED_PARAMS

    assert "weighting" in DERIVED_PARAMS
    assert "weighting_reason" in DERIVED_PARAMS


def test_the_inspector_calls_envelope_a_degraded_outcome():
    """A pure function for the reason ``seam_verdict`` is one: the wording is
    the feature, and it has to be assertable without a GL context."""
    from warlock.studio.panes import inspector

    assert inspector.weighting_verdict({}) is None
    assert inspector.weighting_verdict({"weighting": "automatic"})[1] == "weighting: automatic"
    colour, text = inspector.weighting_verdict({"weighting": "envelope"})
    assert text == "weighting: envelope - needs review"
    # Everything drawn goes through imgui's default Basic-Latin+Latin-1 atlas,
    # so an em dash would render as the missing-glyph box.
    text.encode("latin-1")


def test_rigging_stays_importable_with_no_bpy_anywhere():
    """The host half may never import bpy -- it is process-global, not thread
    safe, and takes the interpreter down rather than raising on the kind of
    geometry trellis produces. A stub that raises on import proves it."""
    import pathlib
    import re

    source = pathlib.Path(rigging.__file__).read_text(encoding="utf-8")
    # Statements only -- the module's own prose argues about bpy at length.
    assert not re.search(r"^\s*(import bpy|from bpy)", source, re.MULTILINE)
    # And transitively, which the scan cannot see. In a subprocess rather than
    # by reloading in-process: reloading rigging mints new function objects and
    # breaks the identity tests/test_viewer_pose.py asserts about
    # ``mirror_quaternion``. ``sys.modules['bpy'] = None`` makes any attempt to
    # import it raise, so a hidden import fails loudly instead of succeeding on
    # a machine that happens to have Blender.
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['bpy'] = None; import warlock.rigging",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_run_worker_failure_is_a_blender_error_not_a_traceback(tmp_path):
    """Whether bpy is installed or not, rigging a mesh that isn't there must
    surface as one typed error carrying the worker's output."""
    spec = rigging.rig_spec(tmp_path, "humanoid")
    with pytest.raises(rigging.BlenderError):
        rigging.run_worker(spec, timeout=300)


def test_run_worker_kills_a_worker_that_hangs_without_closing_stdout(tmp_path, monkeypatch):
    """The timeout has to cover the *run*, not just the wait after EOF.

    A bpy process wedged in a weight solve or an EEVEE render writes nothing
    more and never closes stdout, so reading it inline would block forever and
    take the single serial job queue down with it -- no rig, pose or sheet job
    would ever start again.
    """
    real_popen = subprocess.Popen

    def fake_popen(_cmd, **kw):
        # Holds stdout open and never exits, exactly like a wedged worker.
        return real_popen([sys.executable, "-c", "import time; time.sleep(120)"], **kw)

    monkeypatch.setattr(rigging.subprocess, "Popen", fake_popen)
    started: list[subprocess.Popen] = []
    began = time.monotonic()
    with pytest.raises(rigging.BlenderError, match="timed out"):
        rigging.run_worker(
            {"op": "rig", "result_path": str(tmp_path / "r.json")},
            timeout=1.0,
            on_start=started.append,
        )
    assert time.monotonic() - began < 30, "the deadline did not fire"
    assert started and started[0].poll() is not None, "the hung process was left running"


def test_run_worker_never_waits_unbounded_after_a_kill(tmp_path, monkeypatch):
    import io

    waits: list[float | None] = []

    class StuckProcess:
        args = ["blender-worker"]
        pid = 12345

        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("")

        def wait(self, timeout=None):
            waits.append(timeout)
            raise subprocess.TimeoutExpired(self.args, timeout)

        def poll(self):
            return None

        def kill(self):
            pass

    monkeypatch.setattr(rigging.subprocess, "Popen", lambda *_a, **_kw: StuckProcess())
    monkeypatch.setattr(rigging.winjob, "assign", lambda _pid: None)
    monkeypatch.setattr(rigging.winjob, "track", lambda _pid, _label: None)
    monkeypatch.setattr(rigging.winjob, "untrack", lambda _pid: None)

    with pytest.raises(rigging.BlenderError, match="timed out"):
        rigging.run_worker(
            {"op": "rig", "result_path": str(tmp_path / "r.json")}, timeout=0.01
        )

    assert waits
    assert all(timeout is not None for timeout in waits)


def test_run_worker_honours_a_result_already_queued_when_the_deadline_elapses(
    tmp_path, monkeypatch
):
    """A child that finished legitimately must not be treated as timed out
    just because its EOF sentinel and the deadline land in the same tick.

    ``threading.Thread`` is faked to run the reader synchronously, so the
    EOF sentinel is already sitting in ``lines`` before the wait loop's
    first ``remaining <= 0`` check ever runs -- exactly the race described
    at rigging.py's run_worker. Before the fix this raised
    ``TimeoutExpired`` and deleted the result the worker had already
    written; now the drain-before-raise must let it through.
    """
    import io

    result_path = tmp_path / "r.json"

    class DoneProcess:
        args = ["blender-worker"]
        pid = 54321

        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("")  # closed immediately, like a finished worker

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

        def kill(self):
            pass

    class SyncThread:
        """Runs its target inline on start() instead of on a real thread, so
        the reader has fully drained stdout (and queued the None sentinel)
        before run_worker's wait loop takes its first look at the clock."""

        def __init__(self, target=None, args=(), kwargs=None, **_kw):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            self._target(*self._args, **self._kwargs)

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(rigging.subprocess, "Popen", lambda *_a, **_kw: DoneProcess())
    monkeypatch.setattr(rigging, "threading", type("T", (), {"Thread": SyncThread}))
    monkeypatch.setattr(rigging.winjob, "assign", lambda _pid: None)
    monkeypatch.setattr(rigging.winjob, "track", lambda _pid, _label: None)
    monkeypatch.setattr(rigging.winjob, "untrack", lambda _pid: None)

    def on_start(_proc):
        # Stands in for the worker having already written its result --
        # run_worker unlinks result_path before spawning, so this is the
        # earliest point in the call a fake worker can supply one.
        result_path.write_text(json.dumps({"ok": True}))

    # A deadline already in the past: the very first "remaining <= 0" check
    # must fire, with the sentinel already queued by the synchronous reader.
    payload = rigging.run_worker(
        {"op": "rig", "result_path": str(result_path)},
        timeout=-1.0,
        on_start=on_start,
    )
    assert payload == {"ok": True}


def test_run_worker_rejects_an_unknown_op(tmp_path):
    with pytest.raises(rigging.BlenderError, match="code 2"):
        rigging.run_worker(
            {"op": "sculpt", "result_path": str(tmp_path / "r.json")}, timeout=60
        )


# --- import-then-strip, without a live Blender -------------------------------
#
# The 2026-09-07 audit (poser-02) found _import_glb feeding _world_bounds
# directly in op_remesh and _retexture_frame, skipping _strip_incoming_rig --
# the same double Y-up -> Z-up bug that function exists to fix for op_rig.
# Reproduced against real Blender as a supplied rigged mesh measuring
# (0.505, 0.896, 1.458) against a true (1.138, 0.312, 1.507). These tests pin
# the *order* of the two calls without needing mathutils or a live scene: a
# fake matrix_world stands in for the still-parented mesh's, and a spy on
# _world_bounds records whether the mesh had already been unparented by the
# time it was measured.


class _StopHere(Exception):
    """Raised by a spy once it has recorded what the test needs -- the rest of
    the op (real quadriflow/bake/export calls) is not this test's business."""


def _fake_bpy_with_incoming_armature(scene_objects):
    import types

    return types.SimpleNamespace(
        context=types.SimpleNamespace(scene=types.SimpleNamespace(objects=scene_objects)),
        data=types.SimpleNamespace(
            objects=types.SimpleNamespace(
                remove=lambda obj, do_unlink=True: scene_objects.remove(obj)
            )
        ),
        ops=types.SimpleNamespace(
            wm=types.SimpleNamespace(read_factory_settings=lambda use_empty: None)
        ),
    )


def test_op_remesh_measures_a_supplied_rigged_meshs_bounds_correctly(monkeypatch, tmp_path):
    import types

    from warlock.pipelines import blender_worker

    armature = types.SimpleNamespace(type="ARMATURE", data=types.SimpleNamespace(bones=[1, 2, 3]))
    mesh = types.SimpleNamespace(
        type="MESH",
        parent=armature,  # a supplied humanoid usually arrives skinned and parented
        modifiers=[],
        vertex_groups=[],
        data=types.SimpleNamespace(polygons=[]),
    )
    bpy = _fake_bpy_with_incoming_armature([mesh, armature])

    monkeypatch.setattr(blender_worker, "_import_glb", lambda _bpy, _path: mesh)

    seen_parent = []

    def spy_world_bounds(measured_mesh):
        seen_parent.append(measured_mesh.parent)
        raise _StopHere()

    monkeypatch.setattr(blender_worker, "_world_bounds", spy_world_bounds)

    source = tmp_path / "model.glb"
    source.write_bytes(b"fake-glb")
    spec = {
        "source_glb": str(source),
        "out_glb": str(tmp_path / "out.glb"),
        "target_faces": 100,
        "texture_size": 64,
    }

    with pytest.raises(_StopHere):
        blender_worker.op_remesh(bpy, spec)

    assert seen_parent == [None], (
        "op_remesh measured a mesh that was still parented to its incoming "
        "armature -- the same double-rotation bug _strip_incoming_rig fixes "
        "for op_rig"
    )


def test_retexture_frame_measures_a_supplied_rigged_meshs_bounds_correctly(monkeypatch, tmp_path):
    """The other caller poser-02 named: _retexture_frame, shared by op_views
    and op_project."""
    import types

    from warlock.pipelines import blender_worker

    armature = types.SimpleNamespace(type="ARMATURE", data=types.SimpleNamespace(bones=[1]))
    mesh = types.SimpleNamespace(type="MESH", parent=armature, modifiers=[], vertex_groups=[])
    bpy = _fake_bpy_with_incoming_armature([mesh, armature])

    monkeypatch.setattr(blender_worker, "_import_glb", lambda _bpy, _path: mesh)

    seen_parent = []

    def spy_world_bounds(measured_mesh):
        seen_parent.append(measured_mesh.parent)
        raise _StopHere()

    monkeypatch.setattr(blender_worker, "_world_bounds", spy_world_bounds)
    monkeypatch.setattr(blender_worker, "_setup_render", lambda _bpy, _size: None)

    source = tmp_path / "model.glb"
    source.write_bytes(b"fake-glb")

    with pytest.raises(_StopHere):
        blender_worker._retexture_frame(bpy, source, 64)

    assert seen_parent == [None], (
        "_retexture_frame measured a mesh that was still parented to its "
        "incoming armature -- the same double-rotation bug _strip_incoming_rig "
        "fixes for op_rig"
    )


def test_op_rig_validate_joints_failure_falls_back_to_the_bbox_fit(monkeypatch, tmp_path):
    """poser-04: validate_joints raises ValueError too (a jointfit measurement
    that does not match a non-humanoid template's bone set), and it used to
    sit in the try's ``else`` rather than the try itself -- so a clean
    measurement that failed validation crashed the worker instead of falling
    back, against the comment above it: 'Costs the measurement, never the
    rig'."""
    import types

    from warlock.pipelines import blender_worker, jointfit

    class _IdentityMatrix:
        def __matmul__(self, other):
            return other

    mesh = types.SimpleNamespace(
        type="MESH",
        parent=None,
        modifiers=[],
        vertex_groups=[],
        matrix_world=_IdentityMatrix(),
        data=types.SimpleNamespace(vertices=[types.SimpleNamespace(co=(0.0, 0.0, 0.0))]),
    )
    bpy = _fake_bpy_with_incoming_armature([mesh])

    monkeypatch.setattr(blender_worker, "_import_glb", lambda _bpy, _path: mesh)
    monkeypatch.setattr(blender_worker, "_world_bounds", lambda _mesh: ([0.0] * 3, [1.0] * 3))
    monkeypatch.setattr(jointfit, "payload", lambda _points: {"bones": []})

    def rejecting_validate(_measured, _template):
        raise ValueError("joints payload is missing bone(s)")

    monkeypatch.setattr(rigging, "validate_joints", rejecting_validate)

    seen_spec = {}

    def spy_rig_bones(spec, _lo, _hi):
        seen_spec["spec"] = spec
        raise _StopHere()

    monkeypatch.setattr(blender_worker, "_rig_bones", spy_rig_bones)

    source = tmp_path / "model.glb"
    source.write_bytes(b"fake-glb")
    spec = {
        "template": "humanoid",
        "source_glb": str(source),
        "out_glb": str(tmp_path / "rig.glb"),
        "out_json": str(tmp_path / "rig.json"),
        "joints": "measured",
    }

    with pytest.raises(_StopHere):
        blender_worker.op_rig(bpy, spec)

    assert "bones" not in seen_spec["spec"], (
        "a measurement validate_joints rejected must never reach _rig_bones -- "
        "the bbox fit is still a rig"
    )


# --- with Blender actually installed ----------------------------------------


@pytest.mark.gpu
def test_end_to_end_rig_of_a_generated_cube(tmp_path):
    pytest.importorskip("bpy")
    import bpy  # noqa: F401  -- only to confirm the same interpreter has it

    from warlock.pipelines import blender_worker

    # A cube is manifold, so this exercises the heat-weighting path rather
    # than the envelope fallback.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2)
    source = tmp_path / "model.glb"
    bpy.ops.export_scene.gltf(filepath=str(source), export_format="GLB")

    spec = rigging.rig_spec(tmp_path, "humanoid")
    result = blender_worker.op_rig(bpy, spec)
    # The worker writes temp names; publishing them is the queue's job.
    rigging.finalize_rig(tmp_path)
    assert result["ok"] is True
    # ``automatic-welded`` joined this list when weld-before-heat landed, and a
    # cube reaches it: the glTF export splits every vertex by normal, so even
    # this mesh arrives with 24 vertices and welds back to 8 before the solve.
    assert result["weighting"] in ("automatic", "automatic-welded", "envelope")
    assert (tmp_path / "rig.glb").exists()
    rig = rigging.read_rig(tmp_path)
    assert rig["template"] == "humanoid"
    assert len(rig["bones"]) == result["bones"]
    # Which fit produced those joints, recorded beside them: a rig fitted to
    # landmarks and one scaled onto the bbox are not the same artifact, and
    # nothing else in the file distinguishes them.
    assert rig["fit"] == {"method": "bbox"}
    assert rig["adjusted"] is False


@pytest.mark.gpu
def test_a_landmark_informed_rig_builds_the_armature_from_the_landmarks(tmp_path):
    """The seam, all the way through a real Blender.

    ``_rig_bones`` is unit-tested above, but what it returns then has to
    survive ``_build_armature`` and the export, and rig.json has to say which
    fit produced it -- that file is the only record, and the pose editor and
    the adjust-joints pass both start from it.
    """
    pytest.importorskip("bpy")
    import bpy

    from warlock.pipelines import blender_worker

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2)
    bpy.ops.export_scene.gltf(filepath=str(tmp_path / "model.glb"), export_format="GLB")

    landmarks = _landmarks()  # forearm.L's head moved to [0.40, 0.0, 0.20]
    spec = rigging.rig_spec(
        tmp_path,
        "humanoid",
        template_bones=landmarks,
        fit={"method": "pose2d", "model": "vitpose", "confidence": 0.77},
    )
    blender_worker.op_rig(bpy, spec)
    rigging.finalize_rig(tmp_path)

    rig = rigging.read_rig(tmp_path)
    assert rig["fit"]["method"] == "pose2d"
    assert rig["fit"]["confidence"] == 0.77
    assert rig["adjusted"] is False, "a measured fit is not a correction the user made"
    # A default cube spans -1..1 on every axis, so fit_template's arithmetic
    # here is x*2 and z*2 off a floor at -1 -- the landmark, scaled, and not
    # the shipped template's own [0.22, 0.0, 0.66].
    head = {b["name"]: b for b in rig["bones"]}["forearm.L"]["head"]
    assert head == pytest.approx([0.80, 0.0, -0.60], abs=1e-4)


# --- the pose library's engine half ------------------------------------------
#
# pose_spec's root kwargs, save_pose's extra merge, root_offset_world and
# armature_spec are what service.poses stands on; all four are host-side pure.


def test_pose_spec_without_root_kwargs_is_byte_identical_to_the_old_shape(tmp_path):
    """Backward compatibility is structural: a spec with no root offset must be
    exactly the dict every pose bake has always sent, key for key."""
    pose_id = "0123456789ab"
    bones = {"hips": [0.0, 0.0, 0.0, 1.0]}
    assert rigging.pose_spec(tmp_path, pose_id, bones) == {
        "op": "pose",
        "rig_glb": str(tmp_path / "rig.glb"),
        "out_glb": str(rigging.pose_glb_path(tmp_path, pose_id)),
        "result_path": str(rigging.pose_dir(tmp_path) / f".{pose_id}.result.json"),
        "bones": bones,
    }


def test_pose_spec_adds_no_keys_for_a_zero_offset(tmp_path):
    spec = rigging.pose_spec(
        tmp_path, "0123456789ab", {}, root_bone="hips", root_offset=[0.0, 0.0, 0.0]
    )
    assert "root_bone" not in spec
    assert "root_offset" not in spec


def test_pose_spec_carries_a_nonzero_root_offset(tmp_path):
    spec = rigging.pose_spec(
        tmp_path, "0123456789ab", {}, root_bone="hips", root_offset=[0.1, 0, -0.2]
    )
    assert spec["root_bone"] == "hips"
    assert spec["root_offset"] == [0.1, 0.0, -0.2]


def test_save_pose_merges_extra_into_the_record(tmp_path):
    pose = rigging.validate_pose({"name": "snap", "bones": {"hips": [0, 0, 0, 1]}})
    record = rigging.save_pose(
        tmp_path,
        pose,
        extra={"root_translation": [0.1, 0.0, 0.0], "source_pose": {"id": "abc"}},
    )
    back = rigging.read_pose(tmp_path, record["id"])
    assert back["root_translation"] == [0.1, 0.0, 0.0]
    assert back["source_pose"] == {"id": "abc"}
    assert back["name"] == "snap"


def test_save_pose_extra_may_not_override_what_the_record_owns(tmp_path):
    pose = rigging.validate_pose({"name": "snap", "bones": {"hips": [0, 0, 0, 1]}})
    for key in ("id", "name", "bones", "created"):
        with pytest.raises(ValueError, match="may not override"):
            rigging.save_pose(tmp_path, pose, extra={key: "x"})


# --- pose storage against a hand-edited job directory ------------------------
#
# The read door, not the write door. Everything below arrives as a file some
# other program wrote: nothing here may cost the caller more than the one
# record it is about, because the pane that lists them has no other recourse.


def _pose_file(job_dir, pose_id: str):
    directory = job_dir / rigging.POSE_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{pose_id}.json"


def test_a_pose_that_is_not_utf8_costs_itself(tmp_path):
    pose_id = rigging.new_id()
    _pose_file(tmp_path, pose_id).write_bytes(b"\xff\xfe not text at all")
    assert rigging.read_pose(tmp_path, pose_id) is None
    assert rigging.list_poses(tmp_path) == []


def test_a_pose_that_is_a_json_array_costs_itself(tmp_path):
    """json.loads succeeds; every caller then does record["bones"] on a list."""
    pose_id = rigging.new_id()
    _pose_file(tmp_path, pose_id).write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert rigging.read_pose(tmp_path, pose_id) is None


def test_an_oversized_pose_is_refused_without_being_parsed(tmp_path, monkeypatch):
    monkeypatch.setattr(rigging, "MAX_RECORD_BYTES", 32)
    pose_id = rigging.new_id()
    _pose_file(tmp_path, pose_id).write_text(
        json.dumps({"id": pose_id, "name": "x" * 200, "bones": {}}), encoding="utf-8"
    )
    assert rigging.read_pose(tmp_path, pose_id) is None


# The same three guards on the three readers that never had them. ``read_sheet``,
# ``read_sheet_pixel`` and ``read_sprite_draft`` each caught only
# ``json.JSONDecodeError`` and checked neither the size nor the shape --
# ``UnicodeDecodeError`` is a ``ValueError`` and is *not* a ``JSONDecodeError``,
# so a sidecar holding bytes that are not UTF-8 raised straight out of the
# reader, through ``list_sheets``, and into the pane that called it.


def _reader_cases(job_dir):
    sheet_id, draft_id = rigging.new_id(), rigging.new_id()
    return (
        (
            rigging.sheet_path(job_dir, sheet_id),
            lambda: rigging.read_sheet(job_dir, sheet_id),
        ),
        (
            rigging.sheet_pixel_path(job_dir, sheet_id),
            lambda: rigging.read_sheet_pixel(job_dir, sheet_id),
        ),
        (
            rigging.sprite_draft_path(job_dir, draft_id),
            lambda: rigging.read_sprite_draft(job_dir, draft_id),
        ),
    )


def test_a_sidecar_that_is_not_utf8_costs_itself(tmp_path):
    for path, read in _reader_cases(tmp_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([0xFF, 0xFE]) + b" not text at all")
        assert read() is None, path.name


def test_a_sidecar_that_is_a_json_array_costs_itself(tmp_path):
    for path, read in _reader_cases(tmp_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        assert read() is None, path.name


def test_an_oversized_sidecar_is_refused_without_being_parsed(tmp_path, monkeypatch):
    monkeypatch.setattr(rigging, "MAX_RECORD_BYTES", 32)
    for path, read in _reader_cases(tmp_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": "x" * 200}), encoding="utf-8")
        assert read() is None, path.name


def test_a_pose_whose_id_disagrees_with_its_filename_lists_under_the_stem(tmp_path):
    """The stem is the address every caller uses, so the stem wins: the pose
    stays readable and deletable instead of naming a file that isn't there."""
    pose_id = rigging.new_id()
    _pose_file(tmp_path, pose_id).write_text(
        json.dumps({"id": "somethingelse", "name": "drift", "bones": {}, "created": 1.0}),
        encoding="utf-8",
    )
    [listed] = rigging.list_poses(tmp_path)
    assert listed["id"] == pose_id
    assert rigging.delete_pose(tmp_path, listed["id"]) is True


def test_a_string_created_stamp_does_not_break_the_whole_list(tmp_path):
    for pose_id, created in ((rigging.new_id(), 2.0), (rigging.new_id(), "yesterday")):
        _pose_file(tmp_path, pose_id).write_text(
            json.dumps({"id": pose_id, "name": "p", "bones": {}, "created": created}),
            encoding="utf-8",
        )
    assert len(rigging.list_poses(tmp_path)) == 2


def test_a_rig_json_that_is_not_an_object_costs_itself(tmp_path):
    (tmp_path / "rig.json").write_text(json.dumps(["hips"]), encoding="utf-8")
    assert rigging.read_rig(tmp_path) is None
    assert rigging.rig_bone_names(tmp_path) is None


def test_an_oversized_rig_json_is_refused_without_being_parsed(tmp_path, monkeypatch):
    monkeypatch.setattr(rigging, "MAX_RECORD_BYTES", 32)
    (tmp_path / "rig.json").write_text(
        json.dumps({"bones": [{"name": "hips"}] * 20}), encoding="utf-8"
    )
    assert rigging.read_rig(tmp_path) is None


def test_list_sheets_skips_a_sidecar_whose_png_name_is_a_directory(tmp_path):
    """troupe-06, the 2026-09-11 audit: list_sheets guarded a sheet's PNG with
    ``.exists()``, which is also True for a directory -- unlike the identical
    presence check everywhere else in this area (sheet.pack,
    pixelize.reduce_frames, troupe_mode.scores/atlas_texture), fixed to
    ``.is_file()`` by the 2026-09-07/2026-09-08 audits for the same reason."""
    sheet_id = rigging.new_id()
    rigging.sheet_dir(tmp_path).mkdir(parents=True, exist_ok=True)
    rigging.sheet_path(tmp_path, sheet_id).write_text(
        json.dumps({"id": sheet_id, "created": 1.0}), encoding="utf-8"
    )
    # A directory where the completed sheet's PNG belongs -- exists() is True
    # for this, is_file() is not.
    rigging.sheet_png_path(tmp_path, sheet_id).mkdir()
    assert rigging.list_sheets(tmp_path) == []


def test_root_offset_world_scales_by_the_rig_height():
    bounds = {"min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 2.0]}
    assert rigging.root_offset_world([0.1, 0.0, -0.25], bounds) == pytest.approx(
        [0.2, 0.0, -0.5]
    )


def test_root_offset_world_degenerate_height_falls_back_to_the_largest_extent():
    bounds = {"min": [-3.0, -1.0, 1.0], "max": [3.0, 1.0, 1.0]}  # flat in z
    assert rigging.root_offset_world([0.5, 0.0, 0.0], bounds) == pytest.approx([3.0, 0.0, 0.0])


def test_root_offset_world_point_box_degrades_to_as_authored():
    bounds = {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    assert rigging.root_offset_world([0.5, -0.5, 0.25], bounds) == [0.5, -0.5, 0.25]


def test_root_offset_world_zero_stays_zero():
    bounds = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 5.0]}
    assert rigging.root_offset_world([0.0, 0.0, 0.0], bounds) == [0.0, 0.0, 0.0]


def test_armature_spec_validates_the_template_before_spawning_anything(tmp_path):
    with pytest.raises(ValueError, match="unknown skeleton template"):
        rigging.armature_spec("dragon", tmp_path / "a.glb", tmp_path)


def test_armature_spec_shape(tmp_path):
    spec = rigging.armature_spec("humanoid", tmp_path / ".preview.tmp.glb", tmp_path)
    assert spec == {
        "op": "armature",
        "template": "humanoid",
        "out_glb": str(tmp_path / ".preview.tmp.glb"),
        "result_path": str(tmp_path / ".humanoid.armature_result.json"),
    }


def test_armature_specs_for_different_templates_use_different_result_files(tmp_path):
    """The preview lock is per template, so two templates may build at once --
    and ``run_worker`` unlinks and then watches the result path, so a shared
    name would let each build eat the other's answer."""
    a = rigging.armature_spec("humanoid", tmp_path / "a.glb", tmp_path)
    b = rigging.armature_spec("fish", tmp_path / "b.glb", tmp_path)
    assert a["result_path"] != b["result_path"]


def test_interpolate_carries_endpoint_root_offsets_through_the_clip():
    """This was a refusal by name -- the TMX rule, applied to a feature the
    lerp did not model. It models it now: a root offset is what a walk's
    vertical bob and a jump's rise are made of, and every consumer downstream
    (``root_offset_world``, ``_sheet_root_offsets``, ``op_sheet``'s per-cell
    ``root_offset``) was already keyed per frame and ready for it."""
    from warlock.pipelines import sheet as sheetlib

    plain = {"id": "a" * 12, "name": "A", "bones": {"hips": [0, 0, 0, 1]}}
    offset = {
        "id": "b" * 12,
        "name": "Leap",
        "bones": {"hips": [0, 0, 0, 1]},
        "root_translation": [0.0, 0.0, 0.3],
    }
    out = sheetlib.interpolate(plain, offset, 4)
    assert [round(r["root_translation"][2], 4) for r in out] == [0.0, 0.075, 0.15, 0.225]
    assert [round(r["root_translation"][2], 4) for r in sheetlib.interpolate(offset, plain, 4)] == [
        0.3, 0.225, 0.15, 0.075
    ]
    # A zero offset is still no offset: every record written before the field,
    # and every pose that never touched the root, produces the byte-identical
    # records it always did, with no ``root_translation`` key at all.
    zeroed = dict(plain, root_translation=[0.0, 0.0, 0.0])
    records = sheetlib.interpolate(zeroed, plain, 4)
    assert len(records) == 4
    assert all("root_translation" not in r for r in records)


def _glb_node_rotations(path) -> dict[str, list[float]]:
    """Every named node's local rotation, read straight out of the GLB.

    Deliberately not via bpy: the point of the test below is that the *file*
    the browser downloads carries the rotations the browser asked for, so
    reading it back through Blender would beg the question.
    """
    import struct

    data = path.read_bytes()
    offset = 12
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        if kind == 0x4E4F534A:   # 'JSON'
            doc = json.loads(data[offset + 8 : offset + 8 + length])
            return {
                n["name"]: n.get("rotation", [0.0, 0.0, 0.0, 1.0])
                for n in doc["nodes"]
                if "name" in n
            }
        offset += 8 + length + (-length % 4)
    raise AssertionError("GLB has no JSON chunk")


@pytest.mark.gpu
def test_a_posed_glb_carries_back_exactly_the_rotations_it_was_given(tmp_path):
    """The contract the whole pose feature rests on.

    The browser only ever sees glTF node-local rotations, and Blender only ever
    wants pose-bone bases. blender_worker._rest_local_rotation claims the two
    are one composition apart with no per-bone axis correction; if that were
    wrong, poses would come back subtly rotated and nothing else would notice.
    """
    pytest.importorskip("bpy")
    import bpy

    from warlock.pipelines import blender_worker

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    bpy.context.object.scale = (0.3, 0.2, 1.0)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(tmp_path / "model.glb"), export_format="GLB")
    blender_worker.op_rig(bpy, rigging.rig_spec(tmp_path, "humanoid"))
    rigging.finalize_rig(tmp_path)

    # A parent and its child, so bone-chain accumulation is exercised too.
    want = {
        "upper_arm.L": [0.3826834, 0.0, 0.0, 0.9238795],   # 45 deg about local X
        "forearm.L": [0.0, 0.0, 0.2588190, 0.9659258],     # 30 deg about local Z
    }
    out = tmp_path / "posed.glb"
    result = blender_worker.op_pose(
        bpy, {"rig_glb": str(tmp_path / "rig.glb"), "out_glb": str(out), "bones": want}
    )
    assert result == {"ok": True, "bones": 2, "unknown": []}

    got = _glb_node_rotations(out)
    for name, target in want.items():
        # q and -q are the same rotation, and the exporter is free to pick either.
        deltas = [abs(a - b) for a, b in zip(got[name], target, strict=True)]
        sums = [abs(a + b) for a, b in zip(got[name], target, strict=True)]
        assert max(deltas) < 1e-4 or max(sums) < 1e-4, f"{name}: {got[name]} != {target}"


@pytest.mark.gpu
def test_posing_a_bone_the_rig_lacks_is_reported_not_fatal(tmp_path):
    """A pose saved against one skeleton should still mostly apply after a
    re-rig, so an unknown name is data, not an error."""
    pytest.importorskip("bpy")
    import bpy

    from warlock.pipelines import blender_worker

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(tmp_path / "model.glb"), export_format="GLB")
    blender_worker.op_rig(bpy, rigging.rig_spec(tmp_path, "humanoid"))
    rigging.finalize_rig(tmp_path)

    result = blender_worker.op_pose(
        bpy,
        {
            "rig_glb": str(tmp_path / "rig.glb"),
            "out_glb": str(tmp_path / "posed.glb"),
            "bones": {"hips": [0.0, 0.0, 0.0, 1.0], "tail_01": [0.0, 0.0, 0.0, 1.0]},
        },
    )
    assert result["bones"] == 1
    assert result["unknown"] == ["tail_01"]
    assert (tmp_path / "posed.glb").exists()


def test_a_glb_round_trips_through_the_worker_into_a_real_fbx(tmp_path):
    """op_fbx actually produces a file Unity/Unreal would accept.

    The spec-shape test above only pins the wire format; this runs the real
    Blender subprocess, because the failure mode worth catching is an export
    operator that rejects one of its keyword arguments -- which no amount of
    dict-shape assertion would find.
    """
    pytest.importorskip("bpy")
    trimesh = pytest.importorskip("trimesh")

    source = tmp_path / "model.glb"
    trimesh.Scene(trimesh.creation.box(extents=(1.0, 2.0, 1.0))).export(source)
    out = tmp_path / "model.fbx"

    rigging.run_worker(rigging.fbx_spec(source, out, tmp_path), timeout=300)

    assert out.exists()
    # "Kaydara FBX Binary" is the magic every FBX reader looks for.
    assert out.read_bytes()[:18] == b"Kaydara FBX Binary"


def _glb_doc(path) -> dict:
    """The GLB's whole JSON chunk, read directly for _glb_node_rotations'
    reason: reading it back through Blender would beg the question."""
    import struct

    data = path.read_bytes()
    offset = 12
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        if kind == 0x4E4F534A:  # 'JSON'
            return json.loads(data[offset + 8 : offset + 8 + length])
        offset += 8 + length + (-length % 4)
    raise AssertionError("GLB has no JSON chunk")


def _glb_node_translations(path) -> dict[str, list[float]]:
    return {
        n["name"]: n.get("translation", [0.0, 0.0, 0.0])
        for n in _glb_doc(path)["nodes"]
        if "name" in n
    }


def _world_positions(doc) -> dict[str, list[float]]:
    """Every named node's world position, composing T*R*S down the tree."""
    import numpy as np

    nodes = doc["nodes"]

    def local(n):
        m = np.eye(4)
        r = n.get("rotation")
        if r is not None:
            x, y, z, w = (float(v) for v in r)
            m[:3, :3] = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
        s = n.get("scale")
        if s is not None:
            m[:3, :3] = m[:3, :3] @ np.diag([float(v) for v in s])
        t = n.get("translation")
        if t is not None:
            m[:3, 3] = [float(v) for v in t]
        return m

    out: dict[str, list[float]] = {}

    def walk(i, parent):
        m = parent @ local(nodes[i])
        name = nodes[i].get("name")
        if name:
            out[name] = [float(v) for v in m[:3, 3]]
        for c in nodes[i].get("children", []):
            walk(c, m)

    scene = doc["scenes"][doc.get("scene", 0)]
    for root in scene.get("nodes", []):
        walk(root, np.eye(4))
    return out


@pytest.mark.gpu
def test_op_armature_exports_a_meshless_skeleton(tmp_path):
    """The empirical gate the whole Poser preview stands on: Blender's glTF
    exporter emits an armature with no mesh at all as one node per bone, named
    exactly, correctly parented, with no skin -- so the preview the editor
    rotates is built by the same code path (and therefore the same bone
    frames) as every real rig."""
    pytest.importorskip("bpy")
    import bpy

    from warlock import poselib
    from warlock.pipelines import blender_worker

    out = tmp_path / ".preview.tmp.glb"
    result = blender_worker.op_armature(
        bpy, rigging.armature_spec("humanoid", out, tmp_path)
    )
    assert result["ok"] is True
    assert out.exists()

    doc = _glb_doc(out)
    # Measured, not assumed: Blender 5.2 exports a meshless armature with a
    # joints-only ``skins`` palette but no mesh and nothing skinned -- so the
    # assertion that matters is that nothing draws, not that the word "skins"
    # is absent from the file.
    assert "meshes" not in doc
    assert not any("mesh" in n or "skin" in n for n in doc["nodes"])

    template = rigging.get_template("humanoid")
    bone_names = {b["name"] for b in template.bones}
    named = [n["name"] for n in doc["nodes"] if n.get("name") in bone_names]
    assert sorted(named) == sorted(bone_names), "one node per template bone, named exactly"

    parent_of: dict[int, int] = {}
    for i, node in enumerate(doc["nodes"]):
        for child in node.get("children", []):
            parent_of[child] = i
    index_by_name = {n.get("name"): i for i, n in enumerate(doc["nodes"])}
    for bone in template.bones:
        parent_index = parent_of.get(index_by_name[bone["name"]])
        parent_name = (
            doc["nodes"][parent_index].get("name") if parent_index is not None else None
        )
        if bone["parent"] is not None:
            assert parent_name == bone["parent"], f"{bone['name']} parented wrong"
        else:
            assert parent_name not in bone_names, "the root bone hangs off the armature node"

    # Head positions land on the unit-box fit -- the canonical armature is
    # exactly one character-height tall, which is what makes a stored
    # root_translation mean character-height units literally. Blender world
    # (x, y, z) reads back as glTF (x, z, -y).
    world = _world_positions(doc)
    for bone in rigging.fit_template(template, poselib.UNIT_LO, poselib.UNIT_HI):
        bx, by, bz = bone["head"]
        assert world[bone["name"]] == pytest.approx([bx, bz, -by], abs=1e-4), bone["name"]


@pytest.mark.gpu
def test_op_pose_bakes_the_root_offset_and_only_when_asked(tmp_path):
    """A library pose's root translation reaches the baked GLB as exactly
    ``rest + d`` on the root node, and a spec without the key bakes the same
    root the rig was exported with."""
    pytest.importorskip("bpy")
    import bpy

    from warlock.pipelines import blender_worker

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(tmp_path / "model.glb"), export_format="GLB")
    blender_worker.op_rig(bpy, rigging.rig_spec(tmp_path, "humanoid"))
    rigging.finalize_rig(tmp_path)

    rest_id, moved_id, ignored_id = "aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc"
    spec = rigging.pose_spec(tmp_path, rest_id, {})
    assert "root_offset" not in spec
    blender_worker.op_pose(bpy, spec)

    offset = [0.15, -0.1, 0.2]
    blender_worker.op_pose(
        bpy,
        rigging.pose_spec(tmp_path, moved_id, {}, root_bone="hips", root_offset=offset),
    )

    rest = _glb_node_translations(rigging.pose_glb_path(tmp_path, rest_id))["hips"]
    moved = _glb_node_translations(rigging.pose_glb_path(tmp_path, moved_id))["hips"]
    # The exporter emits the root joint's frame in glTF axes, so the world
    # displacement _apply_root_translation guarantees reads back as
    # (dx, dz, -dy) -- measured, and exactly the m3.blender_delta_to_gltf
    # mapping the viewer already uses.
    delta = [m - r for m, r in zip(moved, rest, strict=True)]
    assert delta == pytest.approx([offset[0], offset[2], -offset[1]], abs=1e-4)

    # A root the rig lacks is reported, never fatal -- the _apply_pose rule.
    blender_worker.op_pose(
        bpy,
        rigging.pose_spec(
            tmp_path, ignored_id, {}, root_bone="tail_99", root_offset=[0.5, 0.0, 0.0]
        ),
    )
    ignored = _glb_node_translations(rigging.pose_glb_path(tmp_path, ignored_id))["hips"]
    assert ignored == pytest.approx(rest, abs=1e-6)


@pytest.mark.gpu
def test_op_rig_builds_the_armature_from_supplied_joints_not_the_fit(tmp_path):
    """The adjust pass is only worth anything if the worker actually uses the
    joints it is handed. Checked against rig.json, which records what was
    built, and against a fit that is deliberately nothing like the override."""
    pytest.importorskip("bpy")
    import bpy

    from warlock.pipelines import blender_worker

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(tmp_path / "model.glb"), export_format="GLB")

    template = rigging.get_template("humanoid")
    # A skeleton fitted to a box ten times the sphere: every joint lands
    # somewhere the sphere's own fit never would.
    override = rigging.fit_template(template, [-10, -10, 0], [10, 10, 20])
    blender_worker.op_rig(bpy, rigging.rig_spec(tmp_path, "humanoid", override))
    rigging.finalize_rig(tmp_path)

    rig = json.loads((tmp_path / "rig.json").read_text(encoding="utf-8"))
    assert rig["adjusted"] is True
    built = {b["name"]: b for b in rig["bones"]}
    for bone in override:
        assert built[bone["name"]]["head"] == pytest.approx(bone["head"])
        assert built[bone["name"]]["tail"] == pytest.approx(bone["tail"])


def test_a_garbage_result_from_an_exit_zero_worker_is_a_blender_error(tmp_path, monkeypatch):
    """The one way run_worker could still raise a raw JSON decoder error at the
    caller: the worker exits 0 and the file it left is not JSON. Typed like
    every other way the worker can disappoint, and the file does not survive."""
    real_popen = subprocess.Popen
    result = tmp_path / "r.json"

    def fake_popen(_cmd, **kw):
        result.write_text("half a resu", encoding="utf-8")
        return real_popen([sys.executable, "-c", "pass"], **kw)

    monkeypatch.setattr(rigging.subprocess, "Popen", fake_popen)
    with pytest.raises(rigging.BlenderError, match="unreadable result"):
        rigging.run_worker({"op": "rig", "result_path": str(result)}, timeout=30)
    assert not result.exists()


# --- the skeleton editor (P1: rigging.py, pure) -------------------------------
#
# Poser's skeleton editor: move/add/remove pivots, split a bone, graft a limb
# preset. Everything here is pure -- list[dict] in, a new list[dict] out -- so
# an editor's undo stack is just the sequence of dicts each edit returned.


def test_max_skeleton_bones_matches_the_viewer_shader_uniform():
    """The viewer's skinning shader has a fixed-size joint uniform array
    (programs.MAX_JOINTS); a skin with more joints draws at rest with no error
    on screen (viewer.scene's warning). The two constants must never drift
    apart independently, or the skeleton editor could build a rig the viewer
    silently fails to animate."""
    from warlock.studio.viewer.programs import MAX_JOINTS

    assert rigging.MAX_SKELETON_BONES == MAX_JOINTS


def _humanoid_rig_bones():
    template = rigging.get_template("humanoid")
    return rigging.fit_template(template, [-1, -1, 0], [1, 1, 2])


def _humanoid_bounds():
    return {"min": [-1, -1, 0], "max": [1, 1, 2]}


class TestValidateSkeleton:
    def test_the_unedited_fit_reports_as_template(self):
        base = rigging.get_template("humanoid")
        result = rigging.validate_skeleton(
            {
                "bones": _humanoid_rig_bones(),
                "mirror_pairs": [list(p) for p in base.mirror_pairs],
            },
            base=base,
            bounds=_humanoid_bounds(),
        )
        assert result["skeleton"] == "template"
        assert result["root"] == "hips"
        assert {tuple(p) for p in result["mirror_pairs"]} == set(base.mirror_pairs)

    def test_a_renamed_bone_is_custom(self):
        base = rigging.get_template("humanoid")
        bones = _humanoid_rig_bones()
        bones[0] = dict(bones[0], name="pelvis")
        for b in bones[1:]:
            if b["parent"] == "hips":
                b["parent"] = "pelvis"
        result = rigging.validate_skeleton({"bones": bones}, base=base, bounds=_humanoid_bounds())
        assert result["skeleton"] == "custom"

    def test_an_extra_bone_is_custom(self):
        base = rigging.get_template("humanoid")
        bones = rigging.add_bone(
            _humanoid_rig_bones(), "hips", "tail_01", [0, -0.1, 0.5], [0, -0.3, 0.5]
        )
        result = rigging.validate_skeleton({"bones": bones}, base=base, bounds=_humanoid_bounds())
        assert result["skeleton"] == "custom"
        assert any(b["name"] == "tail_01" for b in result["bones"])

    def test_rejects_an_empty_skeleton(self):
        base = rigging.get_template("humanoid")
        with pytest.raises(rigging.RigError, match="non-empty"):
            rigging.validate_skeleton({"bones": []}, base=base, bounds=_humanoid_bounds())

    def test_rejects_too_many_bones(self):
        base = rigging.get_template("humanoid")
        bones = [
            {
                "name": f"b{i}",
                "parent": None if i == 0 else f"b{i - 1}",
                "head": [0, 0, float(i)],
                "tail": [0, 0, float(i) + 1],
            }
            for i in range(rigging.MAX_SKELETON_BONES + 1)
        ]
        with pytest.raises(rigging.RigError, match="at most") as exc:
            rigging.validate_skeleton(
                {"bones": bones}, base=base, bounds={"min": [0, 0, 0], "max": [0, 0, 100]}
            )
        assert exc.value.field == "bones"

    def test_rejects_a_bad_name(self):
        base = rigging.get_template("humanoid")
        bones = _humanoid_rig_bones()
        bones[0] = dict(bones[0], name="has a space")
        with pytest.raises(rigging.RigError, match="not usable"):
            rigging.validate_skeleton({"bones": bones}, base=base, bounds=_humanoid_bounds())

    def test_rejects_a_name_over_63_bytes(self):
        base = rigging.get_template("humanoid")
        bones = _humanoid_rig_bones()
        bones[0] = dict(bones[0], name="x" * 64)
        with pytest.raises(rigging.RigError, match="not usable"):
            rigging.validate_skeleton({"bones": bones}, base=base, bounds=_humanoid_bounds())

    def test_rejects_a_duplicate_name(self):
        base = rigging.get_template("humanoid")
        bones = _humanoid_rig_bones()
        bones[1] = dict(bones[1], name=bones[0]["name"])
        with pytest.raises(rigging.RigError, match="duplicate"):
            rigging.validate_skeleton({"bones": bones}, base=base, bounds=_humanoid_bounds())

    def test_rejects_an_unknown_parent(self):
        base = rigging.get_template("humanoid")
        bones = _humanoid_rig_bones()
        bones[1] = dict(bones[1], parent="nonexistent")
        with pytest.raises(rigging.RigError, match="unknown parent"):
            rigging.validate_skeleton({"bones": bones}, base=base, bounds=_humanoid_bounds())

    def test_rejects_two_roots(self):
        base = rigging.get_template("humanoid")
        bones = _humanoid_rig_bones()
        bones[1] = dict(bones[1], parent=None)
        with pytest.raises(rigging.RigError, match="exactly one root") as exc:
            rigging.validate_skeleton({"bones": bones}, base=base, bounds=_humanoid_bounds())
        assert exc.value.field == "root"

    def test_rejects_a_cycle(self):
        """A single root satisfies the root check; the cycle is a *second*,
        disconnected pair of bones pointing at each other."""
        base = rigging.get_template("humanoid")
        bones = [
            {"name": "root", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]},
            {"name": "a", "parent": "b", "head": [0, 0, 1], "tail": [0, 0, 2]},
            {"name": "b", "parent": "a", "head": [0, 0, 2], "tail": [0, 0, 3]},
        ]
        with pytest.raises(rigging.RigError, match="cycle"):
            rigging.validate_skeleton(
                {"bones": bones}, base=base, bounds={"min": [0, 0, 0], "max": [0, 0, 3]}
            )

    def test_rejects_a_zero_length_bone(self):
        base = rigging.get_template("humanoid")
        bones = _humanoid_rig_bones()
        bones[0] = dict(bones[0], tail=list(bones[0]["head"]))
        with pytest.raises(rigging.RigError, match="zero-length"):
            rigging.validate_skeleton({"bones": bones}, base=base, bounds=_humanoid_bounds())

    def test_rejects_a_joint_far_outside_the_mesh(self):
        base = rigging.get_template("humanoid")
        bones = _humanoid_rig_bones()
        bones[0] = dict(bones[0], head=[1000.0, 1000.0, 1000.0])
        with pytest.raises(rigging.RigError, match="far outside"):
            rigging.validate_skeleton({"bones": bones}, base=base, bounds=_humanoid_bounds())

    def test_accepts_a_limb_reaching_past_the_bbox_diagonal(self):
        """A deliberately extended limb (a tail, a reaching wing) is expected
        to stick out past the mesh's own bounds -- the box is widened by its
        own diagonal before anything is checked against it."""
        base = rigging.get_template("humanoid")
        bones = _humanoid_rig_bones()
        bones = rigging.add_bone(bones, "hips", "tail_01", [0, -0.1, 0.5], [0, -1.5, 0.5])
        result = rigging.validate_skeleton({"bones": bones}, base=base, bounds=_humanoid_bounds())
        assert any(b["name"] == "tail_01" for b in result["bones"])

    def test_mirror_pairs_must_name_real_distinct_bones(self):
        base = rigging.get_template("humanoid")
        bones = _humanoid_rig_bones()
        with pytest.raises(rigging.RigError, match="does not have"):
            rigging.validate_skeleton(
                {"bones": bones, "mirror_pairs": [["hips", "nonexistent"]]},
                base=base,
                bounds=_humanoid_bounds(),
            )
        with pytest.raises(rigging.RigError, match="itself"):
            rigging.validate_skeleton(
                {"bones": bones, "mirror_pairs": [["hips", "hips"]]},
                base=base,
                bounds=_humanoid_bounds(),
            )

    def test_a_bone_cannot_be_in_two_mirror_pairs(self):
        base = rigging.get_template("humanoid")
        bones = _humanoid_rig_bones()
        with pytest.raises(rigging.RigError, match="more than one"):
            rigging.validate_skeleton(
                {
                    "bones": bones,
                    "mirror_pairs": [
                        ["upper_arm.L", "upper_arm.R"],
                        ["upper_arm.L", "hand.L"],
                    ],
                },
                base=base,
                bounds=_humanoid_bounds(),
            )


# --- pure bone-list edits -----------------------------------------------------


def test_add_bone_appends_and_never_mutates_the_input():
    bones = _humanoid_rig_bones()
    before = [dict(b) for b in bones]
    out = rigging.add_bone(bones, "hips", "tail_01", [0, 0, 0], [0, -1, 0])
    assert bones == before, "add_bone must not mutate its input"
    assert len(out) == len(bones) + 1
    assert out[-1] == {
        "name": "tail_01", "parent": "hips", "head": [0.0, 0.0, 0.0], "tail": [0.0, -1.0, 0.0],
    }


def test_add_bone_rejects_a_duplicate_name():
    bones = _humanoid_rig_bones()
    with pytest.raises(rigging.RigError, match="duplicate"):
        rigging.add_bone(bones, "hips", bones[0]["name"], [0, 0, 0], [0, 0, 1])


def test_add_bone_rejects_an_unknown_parent():
    bones = _humanoid_rig_bones()
    with pytest.raises(rigging.RigError, match="unknown parent"):
        rigging.add_bone(bones, "nonexistent", "new", [0, 0, 0], [0, 0, 1])


def test_split_bone_inserts_a_child_at_the_midpoint():
    bones = [{"name": "a", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 2]}]
    out = rigging.split_bone(bones, "a", "a2")
    by_name = {b["name"]: b for b in out}
    assert by_name["a"]["tail"] == pytest.approx([0, 0, 1])
    assert by_name["a2"]["parent"] == "a"
    assert by_name["a2"]["head"] == pytest.approx([0, 0, 1])
    assert by_name["a2"]["tail"] == pytest.approx([0, 0, 2])


def test_split_bone_moves_children_onto_the_new_half():
    bones = [
        {"name": "a", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 2]},
        {"name": "child", "parent": "a", "head": [0, 0, 2], "tail": [0, 0, 3]},
    ]
    out = rigging.split_bone(bones, "a", "a2")
    by_name = {b["name"]: b for b in out}
    assert by_name["child"]["parent"] == "a2"


def test_remove_pivot_reparents_children_and_keeps_their_heads():
    bones = [
        {"name": "a", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]},
        {"name": "b", "parent": "a", "head": [0, 0, 1], "tail": [0, 0, 2]},
        {"name": "c", "parent": "b", "head": [0, 0, 2], "tail": [0, 0, 3]},
    ]
    out = rigging.remove_pivot(bones, "b")
    names = {b["name"] for b in out}
    assert names == {"a", "c"}
    by_name = {b["name"]: b for b in out}
    assert by_name["c"]["parent"] == "a"
    assert by_name["c"]["head"] == [0, 0, 2]


def test_remove_pivot_promotes_the_sole_child_when_removing_the_root():
    bones = [
        {"name": "root", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]},
        {"name": "child", "parent": "root", "head": [0, 0, 1], "tail": [0, 0, 2]},
    ]
    out = rigging.remove_pivot(bones, "root")
    by_name = {b["name"]: b for b in out}
    assert by_name["child"]["parent"] is None


def test_remove_pivot_refuses_the_root_with_more_than_one_child():
    bones = [
        {"name": "root", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]},
        {"name": "a", "parent": "root", "head": [0, 0, 1], "tail": [0, 0, 2]},
        {"name": "b", "parent": "root", "head": [0, 0, 1], "tail": [1, 0, 1]},
    ]
    with pytest.raises(rigging.RigError, match="exactly one child") as exc:
        rigging.remove_pivot(bones, "root")
    assert exc.value.field == "root"


def test_remove_subtree_takes_every_descendant():
    bones = [
        {"name": "a", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]},
        {"name": "b", "parent": "a", "head": [0, 0, 1], "tail": [0, 0, 2]},
        {"name": "c", "parent": "b", "head": [0, 0, 2], "tail": [0, 0, 3]},
        {"name": "d", "parent": "a", "head": [0, 0, 1], "tail": [1, 0, 1]},
    ]
    out = rigging.remove_subtree(bones, "b")
    assert {b["name"] for b in out} == {"a", "d"}


def test_remove_subtree_refuses_the_root():
    bones = [{"name": "a", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]}]
    with pytest.raises(rigging.RigError, match="root"):
        rigging.remove_subtree(bones, "a")


def test_rename_bone_updates_children_and_pairs():
    bones = [
        {"name": "arm.L", "parent": None, "head": [0, 0, 0], "tail": [1, 0, 0]},
        {"name": "hand.L", "parent": "arm.L", "head": [1, 0, 0], "tail": [1.3, 0, 0]},
    ]
    pairs = [("arm.L", "arm.R")]
    out_bones, out_pairs = rigging.rename_bone(bones, pairs, "arm.L", "upper_arm.L")
    by_name = {b["name"]: b for b in out_bones}
    assert "upper_arm.L" in by_name
    assert by_name["hand.L"]["parent"] == "upper_arm.L"
    assert out_pairs == [("upper_arm.L", "arm.R")]


def test_rename_bone_rejects_a_collision():
    bones = [
        {"name": "a", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]},
        {"name": "b", "parent": "a", "head": [0, 0, 1], "tail": [0, 0, 2]},
    ]
    with pytest.raises(rigging.RigError, match="duplicate"):
        rigging.rename_bone(bones, [], "a", "b")


def test_prune_pairs_drops_pairs_naming_a_gone_bone():
    bones = [{"name": "a", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]}]
    pairs = [("a", "b"), ("c", "d")]
    assert rigging.prune_pairs(bones, pairs) == []


def test_mirror_partner_name():
    assert rigging.mirror_partner_name("hand.L") == "hand.R"
    assert rigging.mirror_partner_name("hand.R") == "hand.L"
    assert rigging.mirror_partner_name("spine") is None


def test_unique_name_disambiguates():
    bones = [{"name": "tail_01", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]}]
    assert rigging.unique_name(bones, "tail_01") == "tail_01.2"
    assert rigging.unique_name(bones, "wing") == "wing"


# --- limb presets --------------------------------------------------------------


EXPECTED_LIMB_PRESETS = {"arm", "leg", "tail", "wing", "antenna"}


def test_every_shipped_limb_preset_loads():
    assert set(rigging.limb_presets()) == EXPECTED_LIMB_PRESETS


def test_no_limb_preset_key_collides_with_a_template_key():
    assert not (set(rigging.limb_presets()) & set(rigging.templates()))


def test_a_limb_preset_with_a_disconnected_parent_cycle_is_skipped_not_fatal(tmp_path, monkeypatch):
    """The 2026-09-14 audit, finding poser-02: ``_parse_limb_preset`` checked
    that every bone's parent resolves and that there is exactly one root, but
    never that walking those links from every bone actually reaches that
    root -- so a schema-valid preset naming a cycle disconnected from the
    root (``a``/``b`` each other's parent, neither reachable from ``root``)
    was accepted at load and would only misbehave later, wherever a consumer
    walked the chain. Malformed presets are load-and-skip, like every other
    template file (``_load_templates``'s stated policy, restated for limb
    presets by ``_load_limb_presets``'s own comment) -- so this is skipped,
    not a crash, at the loader.
    """
    (tmp_path / "good.json").write_text(
        json.dumps(
            {
                "key": "good",
                "label": "Good",
                "bones": [{"name": "root", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]}],
            }
        )
    )
    (tmp_path / "cyclic.json").write_text(
        json.dumps(
            {
                "key": "cyclic",
                "label": "Cyclic",
                "bones": [
                    {"name": "root", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]},
                    {"name": "a", "parent": "b", "head": [0, 0, 0], "tail": [0, 0, 1]},
                    {"name": "b", "parent": "a", "head": [0, 0, 0], "tail": [0, 0, 1]},
                ],
            }
        )
    )
    monkeypatch.setattr(rigging, "LIMB_DIR", tmp_path)
    assert set(rigging._load_limb_presets()) == {"good"}


def test_a_limb_preset_with_out_of_order_bones_is_rejected():
    """``attach_limb``'s ``_attach_one`` walks ``preset_data["bones"]`` in
    file order, building ``name_map`` as it goes and looking a bone's parent
    up in it (``name_map[b["parent"]]``) -- a schema-valid preset that lists
    a child before its parent throws an uncaught ``KeyError`` there, on the
    frame thread, instead of the ``RigError`` Poser's ``_skeleton_call``
    catches. Refused at parse time instead, the same log-and-skip malformed
    presets already get.
    """
    with pytest.raises(ValueError, match="listed before its parent"):
        rigging._parse_limb_preset(
            {
                "key": "x",
                "label": "X",
                "bones": [
                    {"name": "root", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]},
                    {"name": "child", "parent": "mid", "head": [0, 0, 0], "tail": [0, 0, 1]},
                    {"name": "mid", "parent": "root", "head": [0, 0, 0], "tail": [0, 0, 1]},
                ],
            }
        )


def test_limb_dir_is_not_swept_up_by_the_template_loader():
    """``_load_templates`` globs ``TEMPLATE_DIR / "*.json"`` non-recursively;
    ``templates/limbs/`` is a subdirectory precisely so a preset never shows up
    in the skeleton template catalogue."""
    assert "arm" not in rigging.templates()
    assert "leg" not in rigging.templates()


@pytest.mark.parametrize("preset", sorted(EXPECTED_LIMB_PRESETS))
def test_every_preset_attaches_to_the_humanoid_chest_as_a_valid_skeleton(preset):
    base = rigging.get_template("humanoid")
    bones = _humanoid_rig_bones()
    is_centre = preset == "tail"
    out_bones, out_pairs = rigging.attach_limb(
        bones, [], preset, "chest", None if is_centre else "L", mirror=not is_centre
    )
    result = rigging.validate_skeleton(
        {"bones": out_bones, "mirror_pairs": out_pairs}, base=base, bounds=_humanoid_bounds()
    )
    assert result["skeleton"] == "custom"
    assert len(result["bones"]) > len(bones)


def test_attach_limb_mirror_names_both_sides_and_pairs_them():
    bones = _humanoid_rig_bones()
    out_bones, out_pairs = rigging.attach_limb(bones, [], "wing", "chest", "L", mirror=True)
    names = {b["name"] for b in out_bones}
    assert "wing_base.L" in names
    assert "wing_base.R" in names
    assert ("wing_base.L", "wing_base.R") in out_pairs


def test_attach_limb_refuses_mirroring_a_centre_limb():
    bones = _humanoid_rig_bones()
    with pytest.raises(rigging.RigError, match="no side to mirror"):
        rigging.attach_limb(bones, [], "tail", "hips", None, mirror=True)


def test_attach_limb_rejects_an_unknown_parent():
    bones = _humanoid_rig_bones()
    with pytest.raises(rigging.RigError, match="unknown parent"):
        rigging.attach_limb(bones, [], "arm", "nonexistent", "L", mirror=False)


def test_attach_limb_rejects_an_unknown_preset():
    bones = _humanoid_rig_bones()
    with pytest.raises(ValueError, match="unknown limb preset"):
        rigging.attach_limb(bones, [], "tentacle", "chest", "L", mirror=False)


# --- clip coverage -------------------------------------------------------------


def test_clip_coverage_is_empty_for_an_unedited_template_rig():
    bones = _humanoid_rig_bones()
    rig = {"bones": bones}
    library = rigging.clip_library("humanoid")
    if library["poses"]:
        assert rigging.clip_coverage(rig, "humanoid") == []


def test_clip_coverage_names_bones_a_custom_rig_dropped():
    library = rigging.clip_library("humanoid")
    if not library["poses"]:
        pytest.skip("humanoid ships no clip library to test coverage against")
    animated_bone = next(iter(next(iter(library["poses"].values()))["bones"]))
    bones = rigging.remove_pivot(_humanoid_rig_bones(), animated_bone)
    rig = {"bones": bones}
    assert animated_bone in rigging.clip_coverage(rig, "humanoid")


# --- rig_spec: skeleton/root/mirror_pairs (P2) --------------------------------


def test_rig_spec_leaves_out_skeleton_fields_by_default(tmp_path):
    """Byte-identical to before the skeleton editor existed: an ordinary rig
    or joint-move spec must not gain any of the three new keys."""
    spec = rigging.rig_spec(tmp_path, "humanoid")
    assert "skeleton" not in spec
    assert "root" not in spec
    assert "mirror_pairs" not in spec


def test_rig_spec_carries_a_custom_skeletons_structure(tmp_path):
    spec = rigging.rig_spec(
        tmp_path, "humanoid", skeleton="custom", root="hips", mirror_pairs=[("a.L", "a.R")]
    )
    assert spec["skeleton"] == "custom"
    assert spec["root"] == "hips"
    assert spec["mirror_pairs"] == [["a.L", "a.R"]]


# --- the worker's own re-check of a custom skeleton (P2) ----------------------


def test_check_skeleton_structure_returns_the_root():
    bones = [
        {"name": "a", "parent": None},
        {"name": "b", "parent": "a"},
    ]
    assert rigging.check_skeleton_structure(bones) == "a"


def test_check_skeleton_structure_rejects_a_bad_parent():
    with pytest.raises(rigging.RigError, match="unknown parent"):
        rigging.check_skeleton_structure([{"name": "a", "parent": "nonexistent"}])


def test_a_custom_skeleton_with_a_broken_structure_falls_back_to_the_bbox_fit(capsys):
    """The spec crosses a pipe as plain JSON; a custom skeleton that failed to
    round-trip must cost the informed placement, never the rig -- the same
    rule ``template_bones`` already follows for a mismatched landmark set."""
    from warlock.pipelines import blender_worker

    template = rigging.get_template("humanoid")
    broken = [{"name": "a", "parent": "nonexistent", "head": [0, 0, 0], "tail": [0, 0, 1]}]
    spec = {"template": "humanoid", "bones": broken, "skeleton": "custom"}
    bones, fit = blender_worker._rig_bones(spec, [-1.0, -1.0, 0.0], [1.0, 1.0, 2.0])
    assert bones is not broken
    assert [b["name"] for b in bones] == [b["name"] for b in template.bones]
    assert fit["method"] == "bbox"
    assert "unusable" in capsys.readouterr().out


def test_a_valid_custom_skeleton_is_used_unchanged():
    from warlock.pipelines import blender_worker

    custom = rigging.add_bone(
        _humanoid_rig_bones(), "hips", "tail_01", [0, -0.1, 0.5], [0, -0.3, 0.5]
    )
    spec = {"template": "humanoid", "bones": custom, "skeleton": "custom"}
    bones, fit = blender_worker._rig_bones(spec, [-1.0, -1.0, 0.0], [1.0, 1.0, 2.0])
    assert bones is custom
    assert fit == {"method": "manual"}


def test_rig_meta_carries_a_custom_skeletons_root_and_pairs():
    from warlock.pipelines import blender_worker

    template = rigging.get_template("humanoid")
    meta = blender_worker._rig_meta(
        template,
        bones=[{"name": "pelvis", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]}],
        lo=[0.0, 0.0, 0.0],
        hi=[1.0, 1.0, 1.0],
        weighting="automatic",
        weighting_reason=None,
        adjusted=True,
        fit={"method": "manual"},
        root="pelvis",
        mirror_pairs=[("a.L", "a.R")],
        skeleton="custom",
    )
    assert meta["root"] == "pelvis"
    assert meta["mirror_pairs"] == [["a.L", "a.R"]]
    assert meta["skeleton"] == "custom"


def test_rig_meta_defaults_to_the_template_shape():
    """Every rig before the skeleton editor existed, and every ordinary joint
    move today, must write exactly what it always wrote."""
    from warlock.pipelines import blender_worker

    template = rigging.get_template("humanoid")
    meta = blender_worker._rig_meta(
        template,
        bones=[],
        lo=[0.0, 0.0, 0.0],
        hi=[1.0, 1.0, 1.0],
        weighting="automatic",
        weighting_reason=None,
        adjusted=False,
        fit={"method": "bbox"},
    )
    assert meta["root"] == template.root
    assert meta["mirror_pairs"] == [list(p) for p in template.mirror_pairs]
    assert meta["skeleton"] == "template"


def test_op_rig_records_a_custom_skeleton_end_to_end(tmp_path):
    """The seam whole: a valid custom skeleton reaches rig.json's root,
    mirror_pairs and skeleton fields untouched by the template."""
    pytest.importorskip("bpy")
    import bpy

    from warlock.pipelines import blender_worker

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(tmp_path / "model.glb"), export_format="GLB")

    custom = rigging.add_bone(
        _humanoid_rig_bones(), "hips", "tail_01", [0, -0.1, 0.5], [0, -0.3, 0.5]
    )
    spec = rigging.rig_spec(
        tmp_path, "humanoid", custom, skeleton="custom", root="hips",
        mirror_pairs=[("upper_arm.L", "upper_arm.R")],
    )
    blender_worker.op_rig(bpy, spec)
    rigging.finalize_rig(tmp_path)

    rig = json.loads((tmp_path / "rig.json").read_text(encoding="utf-8"))
    assert rig["skeleton"] == "custom"
    assert rig["root"] == "hips"
    assert rig["mirror_pairs"] == [["upper_arm.L", "upper_arm.R"]]
    assert any(b["name"] == "tail_01" for b in rig["bones"])


# --- stale bakes are invalidated by a re-rig (P2.4) ---------------------------


def test_finalize_rig_deletes_a_stale_pose_bake(tmp_path):
    """A pose baked under the previous skeleton depicts joints this rig no
    longer has; existence is ``posed_model``'s whole freshness test, so a
    stale bake left behind would be served, silently wrong, forever."""
    job_dir = tmp_path
    poses_dir = job_dir / rigging.POSE_DIR_NAME
    poses_dir.mkdir()
    (poses_dir / "abc123456789.glb").write_bytes(b"stale bake")
    (poses_dir / "abc123456789.json").write_text('{"name": "a pose"}', encoding="utf-8")
    (job_dir / rigging.RIG_GLB_TMP).write_bytes(b"new-rig")
    (job_dir / rigging.RIG_JSON_TMP).write_text("{}", encoding="utf-8")

    rigging.finalize_rig(job_dir)

    assert not (poses_dir / "abc123456789.glb").exists()
    # The pose *record* survives -- only its cached bake is invalidated.
    assert (poses_dir / "abc123456789.json").exists()


def test_finalize_rig_deletes_a_stale_animated_glb(tmp_path):
    job_dir = tmp_path
    (job_dir / "animated.glb").write_bytes(b"stale clip bake")
    (job_dir / rigging.RIG_GLB_TMP).write_bytes(b"new-rig")
    (job_dir / rigging.RIG_JSON_TMP).write_text("{}", encoding="utf-8")

    rigging.finalize_rig(job_dir)

    assert not (job_dir / "animated.glb").exists()


def test_finalize_rig_is_fine_with_no_poses_dir_at_all(tmp_path):
    job_dir = tmp_path
    (job_dir / rigging.RIG_GLB_TMP).write_bytes(b"new-rig")
    (job_dir / rigging.RIG_JSON_TMP).write_text("{}", encoding="utf-8")
    rigging.finalize_rig(job_dir)  # must not raise
