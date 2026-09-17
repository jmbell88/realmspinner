"""``mason/document.py``: ``MasonDoc`` -- the tree's mutators, the sculpt
session, the prefab table, and the state a save or an undo panel reads.

Every test name is a claim, written to fail against the version of
``document.py`` it would have caught: an index-addressed edit, a
``move_node`` that clamps a cycle instead of refusing it, a sculpt drag that
pushes forty steps, a ``TerrainEdit`` billed for a whole height field.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from warlock.core.undo import Edit
from warlock.kernels.geom3d.gltf import Material
from warlock.studio.mason import document as doc
from warlock.studio.mason import edits as ed
from warlock.studio.mason import nodes as nd
from warlock.studio.mason import refs
from warlock.studio.mason import terrain as tr


def _terrain(side: int = 4, size: float = 8.0) -> tr.Terrain:
    heights = np.zeros((side + 1, side + 1), dtype="f4")
    return tr.Terrain(heights=heights, size_x=size, size_z=size, material=Material())


# --- uid addressing ------------------------------------------------------


def test_undo_addresses_a_node_by_uid_and_survives_a_reorder_of_the_tree():
    """This must fail against an index-addressed ``TransformEdit``: the
    reorder below happens *after* the transform is recorded and is not itself
    on the undo stack (it stands in for whatever else in the app moved the
    list -- an unrelated compound folded elsewhere), so an implementation
    that captured "index 0" at push time would restore the wrong node.
    """
    d = doc.MasonDoc()
    a = nd.GroupNode(uid=nd.new_uid(), name="A")
    b = nd.GroupNode(uid=nd.new_uid(), name="B")
    d.add_node(a)
    d.add_node(b)
    d.set_transform(a.uid, translation=[5.0, 0.0, 0.0])
    d.roots[0], d.roots[1] = d.roots[1], d.roots[0]  # reorder, off the undo stack
    d.undo()
    assert a.translation.tolist() == [0.0, 0.0, 0.0]
    assert b.translation.tolist() == [0.0, 0.0, 0.0]


# --- move_node -------------------------------------------------------------


def test_move_node_refuses_moving_a_node_into_itself():
    d = doc.MasonDoc()
    a = nd.GroupNode(uid=nd.new_uid())
    d.add_node(a)
    with pytest.raises(ValueError, match="itself"):
        d.move_node(a.uid, 0, parent_uid=a.uid)


def test_move_node_refuses_making_a_node_a_child_of_its_own_descendant():
    d = doc.MasonDoc()
    parent = nd.GroupNode(uid=nd.new_uid())
    child = nd.GroupNode(uid=nd.new_uid())
    parent.children.append(child)
    d.add_node(parent)
    with pytest.raises(ValueError, match="descendant"):
        d.move_node(parent.uid, 0, parent_uid=child.uid)


def test_move_node_reparent_and_reorder_round_trips_through_one_undo():
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid())
    leaf = nd.GroupNode(uid=nd.new_uid())
    other = nd.GroupNode(uid=nd.new_uid())
    d.add_node(group)
    d.add_node(leaf)
    d.add_node(other)
    before_len = len(d.history)
    d.move_node(leaf.uid, 0, parent_uid=group.uid)
    assert len(d.history) == before_len + 1  # one step, not two
    assert d.parent_uid_of(leaf.uid) == group.uid
    assert d.index_of(leaf.uid) == 0
    d.undo()
    assert d.parent_uid_of(leaf.uid) is None
    assert [n.uid for n in d.roots] == [group.uid, leaf.uid, other.uid]


# --- add / remove / move round trips ---------------------------------------


def test_add_then_undo_leaves_the_tree_exactly_as_it_was():
    d = doc.MasonDoc()
    node = nd.GroupNode(uid=nd.new_uid())
    d.add_node(node)
    d.undo()
    assert d.roots == []
    assert d.node(node.uid) is None


def test_remove_then_undo_restores_a_subtree_whole_with_its_children():
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), name="group")
    child_a = nd.GroupNode(uid=nd.new_uid(), name="child-a")
    child_b = nd.GroupNode(uid=nd.new_uid(), name="child-b")
    group.children.extend([child_a, child_b])
    d.add_node(group)

    d.remove_node(group.uid)
    assert d.roots == []

    d.undo()
    assert [n.uid for n in d.roots] == [group.uid]
    restored = d.node(group.uid)
    assert restored is not None
    assert [c.uid for c in restored.children] == [child_a.uid, child_b.uid]
    assert restored.children[0] is child_a  # the very object, not a rebuild


def test_removing_a_node_removes_it_from_selection_and_does_not_restore_it_on_undo():
    """The decided rule: selection is not undoable (the module docstring's
    own argument), so a detach always drops the uid from ``selection`` and a
    re-attach -- whichever direction put it back -- never re-selects it."""
    d = doc.MasonDoc()
    node = nd.GroupNode(uid=nd.new_uid())
    d.add_node(node)
    d.select([node.uid])
    assert node.uid in d.selection

    d.remove_node(node.uid)
    assert node.uid not in d.selection

    d.undo()
    assert d.node(node.uid) is not None
    assert node.uid not in d.selection  # not restored -- selection stays empty


# --- dirty -------------------------------------------------------------------


def test_dirty_is_a_comparison_not_a_flag_edit_save_edit_undo_is_not_dirty():
    d = doc.MasonDoc()
    node = nd.GroupNode(uid=nd.new_uid())
    d.add_node(node)  # edit
    d.mark_saved()  # save
    assert not d.dirty
    d.set_transform(node.uid, translation=[1.0, 0.0, 0.0])  # edit
    assert d.dirty
    d.undo()  # undo back to the saved head
    assert not d.dirty


# --- add_nodes ---------------------------------------------------------------


def test_add_nodes_of_six_is_one_undo_step():
    d = doc.MasonDoc()
    six = [nd.GroupNode(uid=nd.new_uid()) for _ in range(6)]
    d.add_nodes(six)
    assert len(d.history) == 1
    assert len(d.roots) == 6
    d.undo()
    assert d.roots == []


def test_add_nodes_of_none_pushes_nothing():
    d = doc.MasonDoc()
    assert d.add_nodes([]) == []
    assert len(d.history) == 0


def test_add_nodes_refuses_before_building_past_max_placed_rather_than_after(monkeypatch):
    """The 2026-09-14 audit's mason-01: an array op with a count someone typed
    an extra zero into used to build every copy and attach it before anything
    checked ``scene.MAX_PLACED`` -- the ceiling only ever fired downstream, in
    ``scene.resolve()``, by which point the document already had the extra
    nodes attached and every future resolve()/walk() refused for good. This
    must fail against a version of ``add_nodes`` with no such check: nothing
    would raise, and both the roots list and the history would grow.
    """
    from warlock.studio.mason import scene as sc

    monkeypatch.setattr(sc, "MAX_PLACED", 5)
    d = doc.MasonDoc()
    d.add_nodes([nd.GroupNode(uid=nd.new_uid()) for _ in range(3)])
    with pytest.raises(ValueError, match="MAX_PLACED"):
        d.add_nodes([nd.GroupNode(uid=nd.new_uid()) for _ in range(3)])
    # Refused before building: nothing from the refused call was attached,
    # and it pushed no undo step either.
    assert len(d.roots) == 3
    assert len(d.history) == 1


def test_add_nodes_counts_each_added_nodes_whole_subtree_not_just_the_top_level_count(
    monkeypatch,
):
    """The 2026-09-16 audit's mason-engine-01: ``add_nodes`` counted
    ``len(added)`` -- the number of top-level nodes handed in -- rather than
    each one's whole subtree, so a single "array" of a GroupNode with
    children (exactly what ``_spawn_array`` hands it, via
    ``copy_subtree()``) silently attached far more nodes than the ceiling
    check saw. This must fail against a version of ``add_nodes`` that counts
    only the top-level list: one added node, each carrying three children,
    is 4 nodes against a ceiling of 3, but ``len(added) == 1`` would pass it
    straight through.
    """
    from warlock.studio.mason import scene as sc

    monkeypatch.setattr(sc, "MAX_PLACED", 3)
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid())
    group.children.append(nd.GroupNode(uid=nd.new_uid()))
    group.children.append(nd.GroupNode(uid=nd.new_uid()))
    group.children.append(nd.GroupNode(uid=nd.new_uid()))
    # group + 3 children == 4 nodes, past a ceiling of 3, from one add_nodes
    # call carrying a single top-level node.
    with pytest.raises(ValueError, match="MAX_PLACED"):
        d.add_nodes([group])
    assert d.roots == []
    assert len(d.history) == 0


# --- set_transform / set_props -----------------------------------------------


def test_set_transform_with_was_records_the_values_the_caller_started_with():
    """A gizmo writes the live value before it ever calls this, so reading
    "before" off the node itself would compare a value against itself."""
    d = doc.MasonDoc()
    node = nd.GroupNode(uid=nd.new_uid())
    d.add_node(node)
    was = node.trs()  # the true original, captured before the live write below
    node.translation = np.array([9.0, 9.0, 9.0])  # the gizmo's own live write
    pushed = d.set_transform(node.uid, translation=[9.0, 9.0, 9.0], was=was)
    assert pushed
    d.undo()
    assert node.translation.tolist() == [0.0, 0.0, 0.0]


def test_set_props_that_changes_nothing_pushes_no_step():
    d = doc.MasonDoc()
    node = nd.GroupNode(uid=nd.new_uid(), name="same")
    d.add_node(node)
    before_len = len(d.history)
    pushed = d.set_props(node.uid, name="same")
    assert not pushed
    assert len(d.history) == before_len


# --- sculpting ---------------------------------------------------------------


def test_a_forty_call_sculpt_drag_is_one_undo_step_restoring_the_exact_original():
    d = doc.MasonDoc()
    d.set_terrain(_terrain(side=8))  # its own TerrainSwapEdit step, not part of this claim
    original = d.terrain.heights.copy()
    before_len = len(d.history)
    d.begin_sculpt()
    for i in range(40):
        rect = (0, 0, 2, 2)
        sub = np.full((2, 2), float(i), dtype="f4")
        d.sculpt(rect, sub)
    assert d.end_sculpt()
    assert len(d.history) == before_len + 1
    d.undo()
    np.testing.assert_array_equal(d.terrain.heights, original)


def test_end_sculpt_twice_pushes_only_one_step():
    d = doc.MasonDoc()
    d.set_terrain(_terrain())
    before_len = len(d.history)
    d.begin_sculpt()
    d.sculpt((0, 0, 2, 2), np.ones((2, 2), dtype="f4"))
    assert d.end_sculpt()
    assert not d.end_sculpt()  # nothing open a second time
    assert len(d.history) == before_len + 1


def test_end_sculpt_with_no_session_open_is_false_and_harmless():
    d = doc.MasonDoc()
    d.set_terrain(_terrain())
    before_len = len(d.history)
    assert not d.end_sculpt()
    assert len(d.history) == before_len


def test_undo_mid_sculpt_commits_the_open_session_first():
    d = doc.MasonDoc()
    d.set_terrain(_terrain(side=8))
    original = d.terrain.heights.copy()
    d.begin_sculpt()
    d.sculpt((0, 0, 2, 2), np.ones((2, 2), dtype="f4"))
    assert d.sculpting
    d.undo()  # must commit the session, not step over it with nothing to undo
    assert not d.sculpting
    np.testing.assert_array_equal(d.terrain.heights, original)


def test_terrain_edit_cost_is_kilobytes_not_megabytes_for_a_dab_on_a_256_side_terrain():
    d = doc.MasonDoc()
    d.set_terrain(_terrain(side=256))
    d.begin_sculpt()
    cx, cz, radius = 128.0, 128.0, 4.0
    rect = (
        max(0, int(cx - radius)),
        max(0, int(cz - radius)),
        min(257, int(cx + radius) + 1),
        min(257, int(cz + radius) + 1),
    )
    x0, y0, x1, y1 = rect
    patch = np.ones((y1 - y0, x1 - x0), dtype="f4")
    d.sculpt(rect, patch)
    d.end_sculpt()
    step = d.history.top
    assert isinstance(step, ed.TerrainEdit)
    # A whole 257x257 f4 height field is ~264 KB; a radius-4 dab's own rect
    # must be nowhere near that.
    assert step.cost < 8192


# --- installing / removing the terrain (TerrainSwapEdit) ---------------------


def test_installing_a_terrain_is_one_undoable_step():
    d = doc.MasonDoc()
    made = _terrain(side=4)
    before_len = len(d.history)
    d.set_terrain(made)
    assert len(d.history) == before_len + 1
    assert d.terrain is made
    d.undo()
    assert d.terrain is None
    d.redo()
    assert d.terrain is made


def test_removing_the_terrain_can_be_undone_and_the_heights_come_back_exactly():
    """This must fail against a ``set_terrain`` that just assigns: with
    nothing on the stack to restore it, clearing the terrain after a real
    sculpt would lose that work outright the moment anyone pressed Ctrl+Z --
    checked by hand against the pre-fix implementation, which raised
    ``ValueError`` here instead of restoring anything."""
    d = doc.MasonDoc()
    d.set_terrain(_terrain(side=4))
    d.begin_sculpt()
    d.sculpt((0, 0, 2, 2), np.array([[1.0, 2.0], [3.0, 4.0]], dtype="f4"))
    d.end_sculpt()
    sculpted = d.terrain.heights.copy()

    d.set_terrain(None)
    assert d.terrain is None

    d.undo()
    assert d.terrain is not None
    np.testing.assert_array_equal(d.terrain.heights, sculpted)


def test_setting_the_terrain_to_what_it_already_is_pushes_nothing():
    d = doc.MasonDoc()
    made = _terrain(side=4)
    d.set_terrain(made)
    before_len = len(d.history)
    d.set_terrain(made)  # the very same instance, not merely an equal-looking one
    assert len(d.history) == before_len


def test_a_terrain_swap_costs_the_height_fields_it_holds():
    made = _terrain(side=4)
    installed = ed.TerrainSwapEdit(None, made)
    removed = ed.TerrainSwapEdit(made, None)
    both_none = ed.TerrainSwapEdit(None, None)
    assert installed.cost == made.heights.nbytes
    assert removed.cost == made.heights.nbytes
    assert both_none.cost == 0


# --- prefabs -----------------------------------------------------------------


def test_unpack_instance_gives_a_fresh_uid_keeps_the_instance_name_and_transform():
    d = doc.MasonDoc()
    template = nd.GroupNode(uid=nd.new_uid(), name="template")
    template.children.append(nd.GroupNode(uid=nd.new_uid(), name="template-child"))
    d.define_prefab("thing", template)

    instance = nd.PrefabNode(
        uid=nd.new_uid(),
        name="my instance",
        template="thing",
        translation=np.array([1.0, 2.0, 3.0]),
    )
    d.add_node(instance)
    parent_history_len = len(d.history)

    unpacked = d.unpack_instance(instance.uid)

    assert len(d.history) == parent_history_len + 1  # one compound step
    assert unpacked.uid != instance.uid
    assert unpacked.name == "my instance"
    assert unpacked.translation.tolist() == [1.0, 2.0, 3.0]
    assert d.node(instance.uid) is None
    assert d.node(unpacked.uid) is unpacked
    assert len(unpacked.children) == 1
    assert unpacked.children[0].uid != template.children[0].uid

    d.undo()  # reverses the whole unpack in one press
    assert d.node(instance.uid) is instance
    assert d.node(unpacked.uid) is None
    assert len(d.history) == parent_history_len  # back to just the add_node step


def test_editing_the_template_after_unpacking_does_not_reach_the_unpacked_copy():
    d = doc.MasonDoc()
    template = nd.GroupNode(uid=nd.new_uid(), name="template")
    d.define_prefab("thing", template)
    instance = nd.PrefabNode(uid=nd.new_uid(), template="thing")
    d.add_node(instance)
    unpacked = d.unpack_instance(instance.uid)

    # Redefine the prefab -- as if a later authoring edit changed the template.
    changed_template = nd.GroupNode(uid=nd.new_uid(), name="changed")
    d.define_prefab("thing", changed_template)

    assert unpacked.name == ""  # the unpacked copy is untouched by the redefinition
    assert d.prefabs["thing"].name == "changed"


def test_define_prefab_refuses_a_self_recursive_template():
    d = doc.MasonDoc()
    template = nd.GroupNode(uid=nd.new_uid())
    template.children.append(nd.PrefabNode(uid=nd.new_uid(), template="loop"))
    with pytest.raises(ValueError, match="itself"):
        d.define_prefab("loop", template)


def test_define_prefab_refuses_a_template_that_refers_to_itself_through_another():
    d = doc.MasonDoc()
    inner = nd.GroupNode(uid=nd.new_uid())
    inner.children.append(nd.PrefabNode(uid=nd.new_uid(), template="outer"))
    d.define_prefab("inner", inner)

    outer = nd.GroupNode(uid=nd.new_uid())
    outer.children.append(nd.PrefabNode(uid=nd.new_uid(), template="inner"))
    with pytest.raises(ValueError, match="itself"):
        d.define_prefab("outer", outer)


def test_remove_prefab_is_one_step_and_round_trips():
    d = doc.MasonDoc()
    template = nd.GroupNode(uid=nd.new_uid())
    d.define_prefab("thing", template)
    assert d.remove_prefab("thing")
    assert "thing" not in d.prefabs
    d.undo()
    assert "thing" in d.prefabs


# --- missing refs --------------------------------------------------------------


def test_missing_refs_is_empty_when_nothing_is_missing():
    d = doc.MasonDoc()
    node = nd.MeshNode(uid=nd.new_uid(), ref=refs.primitive_ref("box", {}))
    d.add_node(node)
    assert d.missing_refs() == []


def test_missing_refs_lists_nodes_whose_ref_key_is_marked_missing_in_walk_order():
    d = doc.MasonDoc()
    ref_a = refs.LibraryRef(job_id="job-a")
    ref_b = refs.LibraryRef(job_id="job-b")
    a = nd.MeshNode(uid=nd.new_uid(), name="a", ref=ref_a)
    b = nd.MeshNode(uid=nd.new_uid(), name="b", ref=ref_b)
    group = nd.GroupNode(uid=nd.new_uid())
    group.children.append(a)
    d.add_node(group)
    d.add_node(b)
    d.missing = {refs.ref_key(ref_a), refs.ref_key(ref_b)}

    found = d.missing_refs()
    assert [node.name for node, _ref in found] == ["a", "b"]


# --- the sweep: every edit type's undo/redo round trips -----------------------


def _scenario_node_add() -> None:
    d = doc.MasonDoc()
    node = nd.GroupNode(uid=nd.new_uid())
    d.add_node(node)
    after = [n.uid for n in d.roots]
    d.undo()
    assert d.roots == []
    d.redo()
    assert [n.uid for n in d.roots] == after


def _scenario_node_remove() -> None:
    d = doc.MasonDoc()
    node = nd.GroupNode(uid=nd.new_uid())
    d.add_node(node)
    d.remove_node(node.uid)
    assert d.roots == []
    d.undo()
    assert [n.uid for n in d.roots] == [node.uid]
    d.redo()
    assert d.roots == []


def _scenario_node_move() -> None:
    d = doc.MasonDoc()
    a = nd.GroupNode(uid=nd.new_uid())
    b = nd.GroupNode(uid=nd.new_uid())
    d.add_node(a)
    d.add_node(b)
    d.move_node(b.uid, 0)
    after = [n.uid for n in d.roots]
    d.undo()
    assert [n.uid for n in d.roots] == [a.uid, b.uid]
    d.redo()
    assert [n.uid for n in d.roots] == after


def _scenario_transform() -> None:
    d = doc.MasonDoc()
    node = nd.GroupNode(uid=nd.new_uid())
    d.add_node(node)
    d.set_transform(node.uid, translation=[1.0, 2.0, 3.0])
    after = node.translation.tolist()
    d.undo()
    assert node.translation.tolist() == [0.0, 0.0, 0.0]
    d.redo()
    assert node.translation.tolist() == after


def _scenario_props() -> None:
    d = doc.MasonDoc()
    node = nd.GroupNode(uid=nd.new_uid(), name="orig")
    d.add_node(node)
    d.set_props(node.uid, name="changed")
    d.undo()
    assert node.name == "orig"
    d.redo()
    assert node.name == "changed"


def _scenario_ref() -> None:
    d = doc.MasonDoc()
    node = nd.MeshNode(uid=nd.new_uid())
    d.add_node(node)
    ref = refs.primitive_ref("box", {"size": (1.0, 1.0, 1.0)})
    d.set_ref(node.uid, ref)
    d.undo()
    assert node.ref is None
    d.redo()
    assert node.ref == ref


def _scenario_terrain() -> None:
    d = doc.MasonDoc()
    d.set_terrain(_terrain(side=4))
    original = d.terrain.heights.copy()
    d.begin_sculpt()
    d.sculpt((1, 1, 3, 3), np.ones((2, 2), dtype="f4"))
    d.end_sculpt()
    after = d.terrain.heights.copy()
    d.undo()
    np.testing.assert_array_equal(d.terrain.heights, original)
    d.redo()
    np.testing.assert_array_equal(d.terrain.heights, after)


def _scenario_terrain_config() -> None:
    d = doc.MasonDoc()
    d.set_terrain(_terrain())
    d.set_terrain_config(size_x=16.0)
    d.undo()
    assert d.terrain.size_x == 8.0
    d.redo()
    assert d.terrain.size_x == 16.0


def _scenario_terrain_swap() -> None:
    d = doc.MasonDoc()
    made = _terrain(side=4)
    d.set_terrain(made)
    d.undo()
    assert d.terrain is None
    d.redo()
    assert d.terrain is made


def _scenario_prefab() -> None:
    d = doc.MasonDoc()
    template = nd.GroupNode(uid=nd.new_uid(), name="tmpl")
    d.define_prefab("thing", template)
    d.undo()
    assert "thing" not in d.prefabs
    d.redo()
    assert "thing" in d.prefabs


#: One scenario per :class:`~.edits.Edit` subclass, keyed by class name so the
#: *next* one shows up in the parametrize below (derived from the module) and
#: fails loudly -- a ``KeyError`` naming the type -- rather than the sweep
#: silently covering the rest forever. ``TerrainSwapEdit`` is here because it
#: enrolled itself exactly this way: adding it to ``edits.py`` alone was
#: enough for :func:`_edit_type_names` to notice it before this dict did.
_SCENARIOS = {
    "NodeAddEdit": _scenario_node_add,
    "NodeRemoveEdit": _scenario_node_remove,
    "NodeMoveEdit": _scenario_node_move,
    "TransformEdit": _scenario_transform,
    "NodePropsEdit": _scenario_props,
    "RefEdit": _scenario_ref,
    "TerrainEdit": _scenario_terrain,
    "TerrainSwapEdit": _scenario_terrain_swap,
    "TerrainConfigEdit": _scenario_terrain_config,
    "PrefabEdit": _scenario_prefab,
}


def _edit_type_names() -> list[str]:
    return sorted(
        name
        for name, obj in vars(ed).items()
        if inspect.isclass(obj) and issubclass(obj, Edit) and obj is not Edit
    )


@pytest.mark.parametrize("name", _edit_type_names())
def test_undo_then_redo_returns_the_document_to_the_same_state(name: str) -> None:
    if name not in _SCENARIOS:
        raise KeyError(
            f"{name} has no undo/redo round-trip scenario in _SCENARIOS -- "
            "add one rather than letting the sweep quietly skip it"
        )
    _SCENARIOS[name]()


def test_the_sweep_covers_all_ten_edit_types():
    """A derived list that ever drifted from what ``_SCENARIOS`` covers would
    mean either an edit type this test file has not enrolled, or one this
    sweep has lost."""
    names = _edit_type_names()
    assert len(names) == 10
    assert set(names) == set(_SCENARIOS)
