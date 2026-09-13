"""Poser's skeleton editor (P5): a DRAFT bone list edited in place, over a mesh
that stays at rest -- because a bone this session adds has no glTF node, so
the loaded model cannot be reposed to show it the way joints mode can.

Every mutator here is a thin wrapper over a pure function already pinned in
``tests/test_rigging.py``; what is pinned here is the editor's own contract:
one undo step per operation, a refusal that changes nothing, the mirror, and
the Blender <-> glTF handle mapping a draft bone (no node to read a world
matrix from) needs a fixed anchor for.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock import rigging
from warlock.studio.viewer import bonelines
from warlock.studio.viewer import math3d as m3
from warlock.studio.viewer.gltf import Model, Node
from warlock.studio.viewer.pose import PoseEditor


def _model() -> Model:
    """A meshless armature, like the one the Poser preview loads: joints in a
    straight line so each one's world position is easy to reason about."""
    nodes = [
        Node(name="hip", translation=m3.vec3(0.0, 0.0, 0.0)),
        Node(name="upper_arm.L", translation=m3.vec3(0.0, 1.0, 0.0)),
    ]
    nodes[0].children = [1]
    return Model(nodes, roots=[0], meshes=[], skins=[])


def _editor() -> PoseEditor:
    editor = PoseEditor()
    editor.bind(_model(), ["hip", "upper_arm.L"])
    return editor


def _rig() -> dict:
    return {
        "bones": [
            {"name": "hip", "parent": None, "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0]},
            {
                "name": "upper_arm.L",
                "parent": "hip",
                "head": [0.0, 0.0, 1.0],
                "tail": [0.5, 0.0, 1.0],
            },
            {
                "name": "upper_arm.R",
                "parent": "hip",
                "head": [0.0, 0.0, 1.0],
                "tail": [-0.5, 0.0, 1.0],
            },
        ],
        "root": "hip",
        "mirror_pairs": [["upper_arm.L", "upper_arm.R"]],
    }


@pytest.fixture
def skel():
    editor = _editor()
    editor.enter_skeleton_mode(_rig())
    return editor


# --- entering / exiting ------------------------------------------------------


def test_enter_skeleton_mode_requires_a_bound_model():
    editor = PoseEditor()
    editor.enter_skeleton_mode(_rig())
    assert editor.mode == "pose"
    assert editor.draft == []


def test_enter_skeleton_mode_seeds_a_deep_copy_of_the_rig(skel):
    assert skel.mode == "skeleton"
    assert [b["name"] for b in skel.draft] == ["hip", "upper_arm.L", "upper_arm.R"]
    assert skel.draft_pairs == [["upper_arm.L", "upper_arm.R"]]
    assert skel.draft_root == "hip"
    assert skel.draft_dirty is False
    assert skel.mirror_edit is False
    assert skel.selected is None
    # A deep copy: mutating the draft must never reach back into the caller's
    # rig dict.
    rig = _rig()
    skel.draft[0]["head"][0] = 99.0
    assert rig["bones"][0]["head"][0] == 0.0


def test_enter_skeleton_mode_resets_the_pose_like_joints_mode_does():
    editor = _editor()
    editor.apply({"hip": [0, 0, 0.3, 0.954]})
    editor.enter_skeleton_mode(_rig())
    assert editor.pose()["hip"] == pytest.approx([0, 0, 0, 1])
    assert editor.has_unsaved_edits() is False


def test_exit_skeleton_mode_discards_the_draft(skel):
    skel.skel_add_child("upper_arm.L")
    skel.exit_skeleton_mode()
    assert skel.mode == "pose"
    assert skel.draft == []
    assert skel.draft_root is None
    assert skel.has_unsaved_edits() is False


# --- handle mapping (Blender draft -> glTF world) ----------------------------


def test_handles_map_the_roots_own_head_to_its_known_gltf_position(skel):
    # ``hip`` sits at glTF (0, 0, 0), matching its Blender head of (0, 0, 0):
    # the anchor is itself, so the mapping is the identity here.
    assert skel.handles["hip"] == pytest.approx([0.0, 0.0, 0.0])


def test_handles_convert_a_non_root_head_through_the_anchor(skel):
    # anchor_blender=(0,0,0), anchor_gltf=(0,0,0) (hip's own position), so the
    # mapping is exactly blender_delta_to_gltf: [x, y, z] -> [x, z, -y].
    assert skel.handles["upper_arm.L"] == pytest.approx(
        m3.blender_delta_to_gltf(np.array([0.0, 0.0, 1.0]))
    )


def test_a_leaf_bone_gets_its_own_tail_handle_but_a_parent_does_not(skel):
    assert "upper_arm.L@tail" in skel.handles
    assert "upper_arm.R@tail" in skel.handles
    # hip is not a leaf -- its own tail has no handle of its own, since it is
    # drawn as its first child's head instead (draft_segments' rule).
    assert "hip@tail" not in skel.handles


def test_selected_bone_strips_the_tail_suffix(skel):
    skel.selected = "upper_arm.L@tail"
    assert skel.selected_bone() == "upper_arm.L"
    skel.selected = "hip"
    assert skel.selected_bone() == "hip"
    skel.selected = None
    assert skel.selected_bone() is None


# --- moving a handle ----------------------------------------------------------


def test_moving_a_head_handle_writes_the_draft_in_blender_space(skel):
    world = m3.blender_delta_to_gltf(np.array([0.0, 0.0, 2.0]))
    skel.move_handle("upper_arm.L", world)
    by_name = {b["name"]: b for b in skel.draft}
    assert by_name["upper_arm.L"]["head"] == pytest.approx([0.0, 0.0, 2.0])
    assert skel.draft_dirty is True
    assert skel.selected == "upper_arm.L"


def test_moving_a_first_childs_head_also_moves_the_parents_tail(skel):
    """The same continuity rule ``corrected_bones`` applies in joints mode,
    restated as a write: a structural edit has no separate 'corrected' view."""
    world = m3.blender_delta_to_gltf(np.array([0.0, 0.0, 3.0]))
    skel.move_handle("upper_arm.L", world)
    by_name = {b["name"]: b for b in skel.draft}
    assert by_name["hip"]["tail"] == pytest.approx([0.0, 0.0, 3.0])
    # The second child is unaffected -- only the *first* child drives it.
    assert by_name["upper_arm.R"]["head"] == pytest.approx([0.0, 0.0, 1.0])


def test_moving_a_leaf_tail_handle_only_touches_that_bones_tail(skel):
    world = m3.blender_delta_to_gltf(np.array([1.0, 0.0, 1.0]))
    skel.move_handle("upper_arm.L@tail", world)
    by_name = {b["name"]: b for b in skel.draft}
    assert by_name["upper_arm.L"]["tail"] == pytest.approx([1.0, 0.0, 1.0])
    assert by_name["upper_arm.L"]["head"] == pytest.approx([0.0, 0.0, 1.0])


def test_mirror_edit_moves_the_partner_with_x_negated(skel):
    skel.mirror_edit = True
    world = m3.blender_delta_to_gltf(np.array([1.0, -1.0, 1.0]))
    skel.move_handle("upper_arm.L@tail", world)
    by_name = {b["name"]: b for b in skel.draft}
    assert by_name["upper_arm.L"]["tail"] == pytest.approx([1.0, -1.0, 1.0])
    assert by_name["upper_arm.R"]["tail"] == pytest.approx([-1.0, -1.0, 1.0])


def test_without_mirror_edit_the_partner_is_untouched(skel):
    skel.mirror_edit = False
    world = m3.blender_delta_to_gltf(np.array([1.0, -1.0, 1.0]))
    skel.move_handle("upper_arm.L@tail", world)
    by_name = {b["name"]: b for b in skel.draft}
    assert by_name["upper_arm.R"]["tail"] == pytest.approx([-0.5, 0.0, 1.0])


def test_moving_an_unknown_handle_is_a_no_op(skel):
    before = [dict(b) for b in skel.draft]
    skel.move_handle("no_such_bone", np.array([1.0, 2.0, 3.0]))
    assert skel.draft == before
    assert skel.draft_dirty is False


# --- draft_segments -----------------------------------------------------------


def test_draft_segments_connect_parent_to_child(skel):
    pairs = bonelines.draft_segments(skel.draft)
    assert ("hip", "upper_arm.L") in pairs
    assert ("hip", "upper_arm.R") in pairs


def test_draft_segments_add_a_tail_segment_for_every_leaf(skel):
    pairs = bonelines.draft_segments(skel.draft)
    assert ("upper_arm.L", "upper_arm.L@tail") in pairs
    assert ("upper_arm.R", "upper_arm.R@tail") in pairs
    # hip is not a leaf: no synthetic tail segment for it.
    assert ("hip", "hip@tail") not in pairs


def test_draft_segments_positions_resolve_through_the_same_handles(skel):
    """Every name ``draft_segments`` mentions must be a key ``handles`` has --
    that is what lets ``BoneLines.draws`` (built for name -> position lookups)
    draw a draft with no change of its own."""
    pairs = bonelines.draft_segments(skel.draft)
    names = {n for pair in pairs for n in pair}
    assert names <= set(skel.handles)


# --- undo/redo, one step per operation ----------------------------------------


def test_skel_add_child_pushes_one_step_and_undoes(skel):
    before = len(skel.history)
    name = skel.skel_add_child("upper_arm.L")
    assert len(skel.history) == before + 1
    assert name in {b["name"] for b in skel.draft}
    assert skel.draft_dirty is True
    assert skel.selected == name
    skel.undo()
    assert name not in {b["name"] for b in skel.draft}
    assert skel.draft_dirty is False
    skel.redo()
    assert name in {b["name"] for b in skel.draft}


def test_skel_add_child_continues_the_parents_own_direction():
    editor = _editor()
    rig = {
        "bones": [
            {"name": "hip", "parent": None, "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 2.0]},
        ],
        "root": "hip",
        "mirror_pairs": [],
    }
    editor.enter_skeleton_mode(rig)
    name = editor.skel_add_child("hip")
    by_name = {b["name"]: b for b in editor.draft}
    assert by_name[name]["head"] == pytest.approx([0.0, 0.0, 2.0])
    assert by_name[name]["tail"] == pytest.approx([0.0, 0.0, 3.0])


def test_skel_add_child_refuses_an_unknown_parent_without_pushing(skel):
    before = len(skel.history)
    draft_before = [dict(b) for b in skel.draft]
    with pytest.raises(rigging.RigError) as exc:
        skel.skel_add_child("no_such_bone")
    assert exc.value.field == "parent"
    assert len(skel.history) == before
    assert skel.draft == draft_before
    assert skel.draft_dirty is False


def test_skel_add_child_refuses_past_the_bone_cap():
    editor = _editor()
    bones = [{"name": "root", "parent": None, "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0]}]
    for i in range(1, rigging.MAX_SKELETON_BONES):
        bones.append(
            {
                "name": f"bone{i}",
                "parent": bones[-1]["name"],
                "head": [0.0, 0.0, float(i)],
                "tail": [0.0, 0.0, float(i + 1)],
            }
        )
    assert len(bones) == rigging.MAX_SKELETON_BONES
    editor.enter_skeleton_mode({"bones": bones, "root": "root", "mirror_pairs": []})
    before = len(editor.history)
    with pytest.raises(rigging.RigError) as exc:
        editor.skel_add_child("root")
    assert exc.value.field == "bones"
    assert len(editor.history) == before
    assert len(editor.draft) == rigging.MAX_SKELETON_BONES


def test_skel_split_pushes_one_step_and_undoes(skel):
    before = len(skel.history)
    new_name = skel.skel_split("upper_arm.L")
    assert len(skel.history) == before + 1
    by_name = {b["name"]: b for b in skel.draft}
    assert by_name[new_name]["parent"] == "upper_arm.L"
    skel.undo()
    assert new_name not in {b["name"] for b in skel.draft}
    skel.redo()
    assert new_name in {b["name"] for b in skel.draft}


def test_skel_split_refuses_an_unknown_bone_without_pushing(skel):
    before = len(skel.history)
    with pytest.raises(rigging.RigError):
        skel.skel_split("no_such_bone")
    assert len(skel.history) == before


def test_skel_remove_pivot_reparents_children_and_selects_the_parent(skel):
    editor = _editor()
    rig = {
        "bones": [
            {"name": "hip", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]},
            {"name": "mid", "parent": "hip", "head": [0, 0, 1], "tail": [0, 0, 2]},
            {"name": "tip", "parent": "mid", "head": [0, 0, 2], "tail": [0, 0, 3]},
        ],
        "root": "hip",
        "mirror_pairs": [],
    }
    editor.enter_skeleton_mode(rig)
    before = len(editor.history)
    editor.skel_remove_pivot("mid")
    assert len(editor.history) == before + 1
    by_name = {b["name"]: b for b in editor.draft}
    assert "mid" not in by_name
    assert by_name["tip"]["parent"] == "hip"
    assert editor.selected == "hip"
    assert editor.draft_dirty is True
    editor.undo()
    by_name = {b["name"]: b for b in editor.draft}
    assert by_name["tip"]["parent"] == "mid"
    editor.redo()
    by_name = {b["name"]: b for b in editor.draft}
    assert by_name["tip"]["parent"] == "hip"


def test_skel_remove_pivot_refuses_an_unknown_bone_without_pushing(skel):
    before = len(skel.history)
    with pytest.raises(rigging.RigError):
        skel.skel_remove_pivot("no_such_bone")
    assert len(skel.history) == before


def test_subtree_size_counts_the_bone_and_its_descendants():
    editor = _editor()
    rig = {
        "bones": [
            {"name": "hip", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]},
            {"name": "mid", "parent": "hip", "head": [0, 0, 1], "tail": [0, 0, 2]},
            {"name": "tip", "parent": "mid", "head": [0, 0, 2], "tail": [0, 0, 3]},
        ],
        "root": "hip",
        "mirror_pairs": [],
    }
    editor.enter_skeleton_mode(rig)
    assert editor.subtree_size("mid") == 2
    assert editor.subtree_size("hip") == 3


def test_skel_remove_subtree_removes_the_whole_branch_and_selects_the_parent(skel):
    before = len(skel.history)
    count = skel.skel_remove_subtree("upper_arm.L")
    assert count == 1
    assert len(skel.history) == before + 1
    names = {b["name"] for b in skel.draft}
    assert "upper_arm.L" not in names
    assert skel.selected == "hip"
    # The mirror pair naming it is pruned along with it.
    assert skel.draft_pairs == []
    skel.undo()
    names = {b["name"] for b in skel.draft}
    assert "upper_arm.L" in names
    assert skel.draft_pairs == [["upper_arm.L", "upper_arm.R"]]


def test_skel_remove_subtree_refuses_the_root_without_pushing(skel):
    before = len(skel.history)
    with pytest.raises(rigging.RigError) as exc:
        skel.skel_remove_subtree("hip")
    assert exc.value.field == "root"
    assert len(skel.history) == before


def test_skel_rename_pushes_one_step_and_undoes(skel):
    before = len(skel.history)
    skel.selected = "upper_arm.L"
    skel.skel_rename("upper_arm.L", "shoulder.L")
    assert len(skel.history) == before + 1
    names = {b["name"] for b in skel.draft}
    assert "shoulder.L" in names and "upper_arm.L" not in names
    assert skel.draft_pairs == [["shoulder.L", "upper_arm.R"]]
    assert skel.selected == "shoulder.L"
    skel.undo()
    names = {b["name"] for b in skel.draft}
    assert "upper_arm.L" in names and "shoulder.L" not in names
    assert skel.selected == "upper_arm.L"


def test_skel_rename_refuses_a_bad_name_with_field_name_and_pushes_nothing(skel):
    before = len(skel.history)
    with pytest.raises(rigging.RigError) as exc:
        skel.skel_rename("upper_arm.L", "not a legal name!")
    assert exc.value.field == "name"
    assert len(skel.history) == before
    assert "upper_arm.L" in {b["name"] for b in skel.draft}


def test_skel_rename_refuses_a_duplicate_with_field_name(skel):
    before = len(skel.history)
    with pytest.raises(rigging.RigError) as exc:
        skel.skel_rename("upper_arm.L", "upper_arm.R")
    assert exc.value.field == "name"
    assert len(skel.history) == before


def test_skel_rename_updates_the_root_name_when_the_root_is_renamed(skel):
    skel.skel_rename("hip", "pelvis")
    assert skel.draft_root == "pelvis"


def test_skel_attach_limb_pushes_one_step_and_undoes(skel):
    before = len(skel.history)
    names = skel.skel_attach_limb("antenna", "upper_arm.L", "L", False)
    assert len(skel.history) == before + 1
    assert names
    assert set(names) <= {b["name"] for b in skel.draft}
    assert skel.selected == names[0]
    skel.undo()
    assert not (set(names) & {b["name"] for b in skel.draft})
    skel.redo()
    assert set(names) <= {b["name"] for b in skel.draft}


def test_skel_attach_limb_mirror_adds_a_pair_for_each_grafted_bone(skel):
    names = skel.skel_attach_limb("antenna", "hip", "L", True)
    assert len(skel.draft_pairs) == 3  # the original pair plus two new ones
    assert len(names) == 4  # both the L- and R-side grafted bones


def test_skel_attach_limb_refuses_past_the_bone_cap():
    editor = _editor()
    bones = [{"name": "root", "parent": None, "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0]}]
    for i in range(1, rigging.MAX_SKELETON_BONES - 1):
        bones.append(
            {
                "name": f"bone{i}",
                "parent": bones[-1]["name"],
                "head": [0.0, 0.0, float(i)],
                "tail": [0.0, 0.0, float(i + 1)],
            }
        )
    editor.enter_skeleton_mode({"bones": bones, "root": "root", "mirror_pairs": []})
    before_len = len(editor.draft)
    before_history = len(editor.history)
    # "antenna" adds 2 bones, which would cross MAX_SKELETON_BONES by one.
    with pytest.raises(rigging.RigError) as exc:
        editor.skel_attach_limb("antenna", "root", "L", False)
    assert exc.value.field == "bones"
    assert len(editor.history) == before_history
    assert len(editor.draft) == before_len


# --- has_unsaved_edits ---------------------------------------------------------


def test_has_unsaved_edits_reflects_draft_dirty(skel):
    assert skel.has_unsaved_edits() is False
    skel.skel_add_child("upper_arm.L")
    assert skel.has_unsaved_edits() is True
    skel.undo()
    assert skel.has_unsaved_edits() is False


# --- skeleton_payload -----------------------------------------------------------


def test_skeleton_payload_matches_service_rig_edit_skeletons_shape(skel):
    skel.skel_add_child("upper_arm.L")
    payload = skel.skeleton_payload()
    assert set(payload) == {"bones", "root", "mirror_pairs"}
    assert payload["root"] == "hip"
    assert payload["bones"] == skel.draft
    assert payload["bones"] is not skel.draft  # a copy, not a live alias
