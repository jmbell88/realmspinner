"""Tranche 3's OPS registry rows: parenting/groups, separate, set origin, lock.

The document doors (``ClayDoc.group``/``set_parent``/``remove_object``/
``set_origin``/``separate``) and the pure ``kernels.mesh.separate`` splitters
are tested on their own terms elsewhere (``tests/modes/clay/test_document.py``,
``tests/modes/clay/test_separate.py``); what belongs here is the registry
wiring -- that the context menu, the tools pane and the keyboard all reach the
new rows through one list, the way the rest of ``clay_ops`` already is, and
that the handful of call sites this tranche touches (Bake, Merge/Union/
Difference/Intersection, Mirror Copy, Place Between, Align/Distribute/Drop to
Ground) now measure a *parented* object correctly rather than reading its
local TRS as if it were the world.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import elements as el
from warlock.kernels.mesh import mesh as bm
from warlock.kernels.mesh import modifiers as mod
from warlock.kernels.mesh import ops as clay_ops_geom
from warlock.kernels.mesh import primitives as bp
from warlock.studio.modes.clay import ops as clay_ops


class _Toasts:
    def __init__(self) -> None:
        self.errors: list[str] = []


class _Ctx:
    def __init__(self) -> None:
        self.toasts = _Toasts()

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.toasts.errors.append(message)


def _doc() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box"))
    return doc, obj.uid


def _offset_box(size: tuple[float, float, float], offset: tuple[float, float, float]) -> bm.Mesh:
    """A box mesh whose own vertices sit away from its local origin -- the
    same helper ``test_clay_ops.py`` uses to prove an op reads the actual
    mesh rather than a pivot that happens to already sit at the answer."""
    mesh = bp.box(size=size)
    return replace(mesh, positions=mesh.positions + np.array(offset, dtype="f4"))


def _two_loose_boxes(offset: tuple[float, float, float] = (5.0, 0.0, 0.0)) -> bm.Mesh:
    """Two boxes concatenated with no shared vertex index -- two loose parts
    in one mesh. ``tests/modes/clay/test_separate.py``'s own helper."""
    a = bp.box()
    b = bp.box()
    b_positions = np.asarray(b.positions, dtype="f4") + np.asarray(offset, dtype="f4")
    merged = bm.Mesh(
        positions=np.concatenate([a.positions, b_positions]),
        loops=np.concatenate([a.loops, b.loops + len(a.positions)]),
        starts=np.concatenate([a.starts, a.starts[-1] + b.starts[1:]]),
        material=np.concatenate([a.material, b.material]),
        smooth=np.concatenate([a.smooth, b.smooth]),
    )
    bm.validate(merged)
    return merged


def _two_toned_box() -> bm.Mesh:
    box = bp.box()
    material = np.zeros(bm.face_count(box), dtype="i4")
    material[: bm.face_count(box) // 2] = 1
    out = bm.Mesh(
        positions=box.positions,
        loops=box.loops,
        starts=box.starts,
        material=material,
        smooth=box.smooth,
    )
    bm.validate(out)
    return out


NEW_OBJECT_ROWS = (
    "group",
    "ungroup",
    "parent-to-last",
    "clear-parent",
    "separate-loose",
    "separate-material",
    "origin-to-bounds",
    "origin-to-base",
    "origin-to-world",
    "lock",
    "unlock",
)


# --- registry wiring: menu membership and greyed reasons ---------------------


def test_every_new_object_row_is_in_the_object_menu_and_greyed_on_an_empty_document() -> None:
    doc = bd.ClayDoc()
    object_menu = {op.name for op in clay_ops.menu("object")}
    for name in NEW_OBJECT_ROWS:
        assert name in object_menu, f"{name}: missing from the object menu"
        op = clay_ops.get(name)
        assert not op.enabled(doc), f"{name}: expected refused on an empty document"
        assert clay_ops.reason_for(op, doc), f"{name}: refused with no reason"


def test_separate_selection_is_a_face_mode_row_reachable_by_p() -> None:
    face_menu = {op.name for op in clay_ops.menu("face")}
    assert "separate-selection" in face_menu
    assert "separate-selection" not in {op.name for op in clay_ops.menu("object")}
    op = clay_ops.by_key("face", "P")
    assert op is not None and op.name == "separate-selection"


def test_origin_to_selection_is_an_element_mode_row_in_all_three_modes() -> None:
    for mode in ("vertex", "edge", "face"):
        assert "origin-to-selection" in {op.name for op in clay_ops.menu(mode)}
    assert "origin-to-selection" not in {op.name for op in clay_ops.menu("object")}

    doc, uid = _doc()
    op = clay_ops.get("origin-to-selection")
    doc.set_element_mode("face")
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc)
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    assert op.enabled(doc)


def test_group_and_parent_to_last_need_two_selected() -> None:
    for name in ("group", "parent-to-last"):
        doc, uid = _doc()
        op = clay_ops.get(name)
        assert not op.enabled(doc)
        assert "0 selected" in clay_ops.reason_for(op, doc)
        doc.select([uid])
        assert not op.enabled(doc)
        assert "1 selected" in clay_ops.reason_for(op, doc)


def test_ungroup_needs_a_group_selected_not_merely_an_object() -> None:
    doc, uid = _doc()
    op = clay_ops.get("ungroup")
    doc.select([uid])
    assert not op.enabled(doc), "an ordinary box is not a group"
    assert clay_ops.reason_for(op, doc)


# --- group / ungroup ----------------------------------------------------------


def test_group_makes_one_empty_with_the_selection_as_children_and_nothing_moves() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box(), translation=[2.0, 0.0, 0.0])
    )
    b = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box(), translation=[-2.0, 0.0, 0.0])
    )
    a_world_before = np.array(doc.world_matrix(a.uid), copy=True)
    b_world_before = np.array(doc.world_matrix(b.uid), copy=True)
    doc.select([a.uid, b.uid])
    depth = len(doc.history)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("group")) is True

    assert len(doc.history) == depth + 1, "group is one undo step"
    empties = [o for o in doc.objects if o.uid not in (a.uid, b.uid)]
    assert len(empties) == 1
    empty = empties[0]
    assert len(empty.mesh.positions) == 0, "a group is mesh-less"
    assert set(doc.children_of(empty.uid)) == {a.uid, b.uid}
    assert doc.by_uid(a.uid).parent == empty.uid
    assert doc.by_uid(b.uid).parent == empty.uid
    assert np.allclose(doc.world_matrix(a.uid), a_world_before), "A did not move"
    assert np.allclose(doc.world_matrix(b.uid), b_world_before), "B did not move"
    assert doc.selection == {empty.uid}, "the new group is what stays selected"


def test_ungroup_keeps_grandchildren_in_place() -> None:
    """Ungrouping the outer group re-parents its *direct* child (Mid) onto
    the group's own parent -- the document root here -- but Mid's own child
    (Leaf, a grandchild of the group) is not itself touched by that step at
    all, and must read the identical world placement afterwards."""
    doc = bd.ClayDoc()
    mid = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Mid", mesh=bp.box(), translation=[3.0, 0.0, 0.0])
    )
    leaf = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Leaf", mesh=bp.box(), translation=[1.0, 0.0, 0.0])
    )
    doc.set_parent(leaf.uid, mid.uid, keep_world=False)
    # ``ClayDoc.group`` directly -- one object is enough for the *door*, and
    # this test is about Ungroup, not about Group Selected's own 2+ gate
    # (covered by ``test_group_and_parent_to_last_need_two_selected``).
    doc.group([mid.uid])
    empty = next(o for o in doc.objects if o.uid not in (mid.uid, leaf.uid))
    assert doc.by_uid(mid.uid).parent == empty.uid
    assert doc.by_uid(leaf.uid).parent == mid.uid, "Leaf is still Mid's own child"

    mid_world_before = np.array(doc.world_matrix(mid.uid), copy=True)
    leaf_world_before = np.array(doc.world_matrix(leaf.uid), copy=True)
    doc.select([empty.uid])
    depth = len(doc.history)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("ungroup")) is True

    assert len(doc.history) == depth + 1, "ungroup is one undo step"
    assert empty.uid not in [o.uid for o in doc.objects]
    assert doc.by_uid(mid.uid).parent is None
    assert doc.by_uid(leaf.uid).parent == mid.uid, "the grandchild's own parent is untouched"
    assert np.allclose(doc.world_matrix(mid.uid), mid_world_before)
    assert np.allclose(doc.world_matrix(leaf.uid), leaf_world_before)


def test_ungroup_leaves_a_non_group_object_in_the_selection_alone() -> None:
    """A mixed selection -- one real group, one ordinary box -- ungroups
    only the group; the box is not what "Ungroup" means and is left as is,
    the same "whichever of the selection this applies to" shape ``_shade``'s
    own face-mode branch already uses."""
    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    doc.group([a.uid])  # the door directly -- see the identical note above
    empty = next(o for o in doc.objects if o.uid != a.uid)
    plain = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Plain", mesh=bp.box()))

    doc.select([empty.uid, plain.uid])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("ungroup")) is True

    assert empty.uid not in [o.uid for o in doc.objects]
    assert plain.uid in [o.uid for o in doc.objects], "the ordinary box was left alone"


# --- parent-to-last / clear-parent --------------------------------------------


def test_parent_to_last_parents_the_rest_onto_the_topmost_selected_object() -> None:
    doc = bd.ClayDoc()
    top = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Top", mesh=bp.box()))
    mid = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Mid", mesh=bp.box(), translation=[1.0, 0.0, 0.0])
    )
    bottom = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Bottom", mesh=bp.box(), translation=[2.0, 0.0, 0.0])
    )
    mid_world_before = np.array(doc.world_matrix(mid.uid), copy=True)
    bottom_world_before = np.array(doc.world_matrix(bottom.uid), copy=True)
    # Selection order deliberately not document order -- the op must read
    # ``doc.objects``, not whichever order the set iterates in.
    doc.select([bottom.uid, mid.uid, top.uid])
    depth = len(doc.history)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("parent-to-last")) is True

    assert len(doc.history) == depth + 1
    assert doc.by_uid(top.uid).parent is None
    assert doc.by_uid(mid.uid).parent == top.uid
    assert doc.by_uid(bottom.uid).parent == top.uid
    assert np.allclose(doc.world_matrix(mid.uid), mid_world_before), "kept world placement"
    assert np.allclose(doc.world_matrix(bottom.uid), bottom_world_before)


def test_parent_to_last_refuses_a_cycle_with_a_toast() -> None:
    doc = bd.ClayDoc()
    # Child is added first, so it is topmost in document order and becomes
    # the target ``parent-to-last`` would parent everything else onto --
    # including its own actual parent, Grandparent, which is exactly the
    # cycle ``set_parent`` exists to refuse.
    child = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Child", mesh=bp.box()))
    grandparent = doc.add_object(bd.Obj(uid=bd.new_uid(), name="GP", mesh=bp.box()))
    doc.set_parent(child.uid, grandparent.uid, keep_world=False)
    doc.select([child.uid, grandparent.uid])
    depth = len(doc.history)
    ctx = _Ctx()

    clay_ops.run(ctx, doc, clay_ops.get("parent-to-last"))

    assert ctx.toasts.errors, "the document's own cycle refusal reached the toast"
    assert "descendant" in ctx.toasts.errors[0] or "itself" in ctx.toasts.errors[0]
    assert doc.by_uid(child.uid).parent == grandparent.uid, "unchanged"
    assert doc.by_uid(grandparent.uid).parent is None, "unchanged"
    assert len(doc.history) == depth, "the refused reparent pushed nothing"


def test_clear_parent_makes_selected_objects_roots_keeping_world() -> None:
    doc = bd.ClayDoc()
    parent = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="P", mesh=bp.box(), translation=[5.0, 0.0, 0.0])
    )
    child = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="C", mesh=bp.box(), translation=[1.0, 0.0, 0.0])
    )
    doc.set_parent(child.uid, parent.uid, keep_world=False)
    child_world_before = np.array(doc.world_matrix(child.uid), copy=True)
    doc.select([child.uid])
    depth = len(doc.history)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("clear-parent")) is True

    assert len(doc.history) == depth + 1
    assert doc.by_uid(child.uid).parent is None
    assert np.allclose(doc.world_matrix(child.uid), child_world_before)


# --- separate: loose parts, material, selection -------------------------------


def test_separate_loose_makes_one_object_per_part_as_one_step_and_copies_the_stack() -> None:
    doc = bd.ClayDoc()
    stack = (mod.make("weld", id=1),)
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Both",
            mesh=_two_loose_boxes(),
            translation=[1.0, 0.0, 0.0],
            modifiers=stack,
        )
    )
    doc.select([obj.uid])
    depth = len(doc.history)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("separate-loose")) is True

    assert len(doc.history) == depth + 1, "one undo step"
    assert obj.uid not in [o.uid for o in doc.objects]
    assert len(doc.objects) == 2
    for piece in doc.objects:
        assert piece.modifiers == stack
        assert np.allclose(piece.translation, [1.0, 0.0, 0.0])
        assert bm.face_count(piece.mesh) == 6


def test_separate_material_makes_one_object_per_material_slot() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="TwoTone", mesh=_two_toned_box()))
    doc.select([obj.uid])
    depth = len(doc.history)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("separate-material")) is True

    assert len(doc.history) == depth + 1
    assert len(doc.objects) == 2
    materials = sorted(int(o.mesh.material[0]) for o in doc.objects)
    assert materials == [0, 1]


def test_separate_material_refuses_a_single_material_object_as_a_toast() -> None:
    doc, uid = _doc()
    doc.select([uid])
    ctx = _Ctx()
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("separate-material")) is False
    assert ctx.toasts.errors
    assert len(doc.objects) == 1
    assert len(doc.history) == depth


def test_separate_selection_splits_the_picked_faces_out_as_one_step() -> None:
    doc, uid = _doc()
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=[0, 1]))
    total_faces = bm.face_count(doc.by_uid(uid).mesh)
    depth = len(doc.history)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("separate-selection")) is True

    assert len(doc.history) == depth + 1
    assert len(doc.objects) == 2
    faces = sorted(bm.face_count(o.mesh) for o in doc.objects)
    assert faces == [2, total_faces - 2]


# --- set origin ----------------------------------------------------------------


def test_origin_to_bounds_moves_the_pivot_without_moving_geometry_or_children() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="A",
            mesh=_offset_box((1.0, 1.0, 1.0), (2.0, 0.0, 0.0)),
            translation=[3.0, 0.0, 0.0],
        )
    )
    child = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Child", mesh=bp.box(), translation=[0.5, 0.0, 0.0])
    )
    doc.set_parent(child.uid, obj.uid, keep_world=False)
    box_before = clay_ops_geom.world_box(doc.by_uid(obj.uid), world=doc.world_matrix(obj.uid))
    child_world_before = np.array(doc.world_matrix(child.uid), copy=True)
    old_translation = np.array(obj.translation, copy=True)
    doc.select([obj.uid])
    depth = len(doc.history)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("origin-to-bounds")) is True

    assert len(doc.history) == depth + 1
    assert not np.allclose(doc.by_uid(obj.uid).translation, old_translation), "the pivot moved"
    box_after = clay_ops_geom.world_box(doc.by_uid(obj.uid), world=doc.world_matrix(obj.uid))
    assert np.allclose(box_before[0], box_after[0]) and np.allclose(box_before[1], box_after[1])
    assert np.allclose(doc.world_matrix(child.uid), child_world_before), "the child stayed put"


def test_origin_to_base_puts_the_pivot_at_the_box_bottom_centre() -> None:
    doc, uid = _doc()
    doc.select([uid])
    lo, hi = clay_ops_geom.world_box(doc.by_uid(uid))

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("origin-to-base")) is True

    expected = [(lo[0] + hi[0]) / 2.0, lo[1], (lo[2] + hi[2]) / 2.0]
    assert np.allclose(doc.by_uid(uid).translation, expected)
    lo_after, hi_after = clay_ops_geom.world_box(doc.by_uid(uid))
    assert np.allclose(lo, lo_after) and np.allclose(hi, hi_after), "geometry did not move"


def test_origin_to_world_moves_the_pivot_to_the_world_origin() -> None:
    doc, uid = _doc()
    doc.set_transform(uid, translation=[3.0, 4.0, 5.0])
    doc.select([uid])

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("origin-to-world")) is True

    assert np.allclose(doc.by_uid(uid).translation, [0.0, 0.0, 0.0])


def test_origin_to_selection_uses_the_element_selections_centroid() -> None:
    doc, uid = _doc()
    doc.set_element_mode("face")
    mesh_before = doc.by_uid(uid).mesh
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    face0_verts = mesh_before.loops[mesh_before.starts[0] : mesh_before.starts[1]]
    expected = np.asarray(mesh_before.positions, dtype="f8")[face0_verts].mean(axis=0)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("origin-to-selection")) is True

    assert np.allclose(doc.by_uid(uid).translation, expected)


# --- lock / unlock ---------------------------------------------------------------


def test_lock_unlock_round_trip() -> None:
    doc, uid = _doc()
    doc.select([uid])

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("lock")) is True
    assert doc.by_uid(uid).locked is True

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("unlock")) is True
    assert doc.by_uid(uid).locked is False


def test_a_locked_object_refuses_a_geometry_op_with_the_documents_sentence() -> None:
    doc, uid = _doc()
    doc.select([uid])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("lock")) is True
    assert doc.by_uid(uid).locked is True

    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("extrude")) is False
    assert ctx.toasts.errors == [f"{doc.by_uid(uid).name!r} is locked."]
    assert bm.face_count(doc.by_uid(uid).mesh) == bm.face_count(bp.box()), "nothing was extruded"


# --- world-space fixes: merge, align, drop to ground on a parented object -----


def test_merging_a_parented_child_uses_its_world_placement() -> None:
    """``_join`` used to hand ``ops.join`` each selected object's own local
    TRS, which is only ever that object's *world* placement for a root. A
    child's local TRS is relative to its parent, so a merge that reads it
    unconverted places the child's geometry at its local numbers instead of
    where it actually sits on screen. This fails against the unfixed code
    (``ops.join`` called with no ``world=``): the child then merges at local
    (0, 0, 0), landing on top of the target rather than ~10m away.
    """
    doc = bd.ClayDoc()
    target = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Target", mesh=bp.box()))
    parent = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Parent", mesh=bp.box(), translation=[10.0, 0.0, 0.0])
    )
    child = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Child", mesh=bp.box(), translation=[0.0, 0.0, 0.0])
    )
    doc.set_parent(child.uid, parent.uid, keep_world=False)
    doc.select([target.uid, child.uid])

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("join"), weld=0.0) is True

    merged = doc.by_uid(target.uid).mesh
    _lo, hi = bm.bounds(merged)
    assert float(hi[0]) > 9.0, (
        "the child's geometry landed at its true world position (~x=10), not its "
        "local one (~x=0) -- fails against the unfixed code"
    )


def test_drop_to_ground_rests_a_parented_objects_world_box_on_y_zero() -> None:
    """``_world_boxes``/``_apply_deltas`` used to read and write a parented
    object's *local* translation as if it were the world one. This fails
    against the unfixed code: the delta would be computed from a box read at
    the child's local (0, 5, 0) origin instead of its true world position
    five metres above the parent, and applying it to the local field alone
    leaves the object floating instead of resting on the ground.
    """
    doc = bd.ClayDoc()
    parent = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Parent", mesh=bp.box(), translation=[0.0, 5.0, 0.0])
    )
    child = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Child", mesh=bp.box(), translation=[0.0, 0.0, 0.0])
    )
    doc.set_parent(child.uid, parent.uid, keep_world=False)
    doc.select([child.uid])

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("drop-to-ground")) is True

    lo, _hi = clay_ops_geom.world_box(doc.by_uid(child.uid), world=doc.world_matrix(child.uid))
    assert float(lo[1]) == pytest.approx(0.0, abs=1e-6), (
        "the child's *world* box bottom should rest on y=0 -- fails against the "
        "unfixed code, which drops it by its local box instead and leaves it "
        "floating at its parent's height"
    )


def test_align_centres_a_parented_and_a_root_object_on_the_same_world_axis() -> None:
    """``_world_boxes`` again: aligning a root object (world Y centre 0) and
    an object parented five metres above the origin must bring their *world*
    Y centres together. Against the unfixed code the parented object's box
    is read at its local translation (0, 0, 0) -- already believed to match
    the root -- so no delta is applied to it at all, and the two centres
    stay five metres apart.
    """
    doc = bd.ClayDoc()
    root = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Root", mesh=bp.box()))
    parent = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Parent", mesh=bp.box(), translation=[0.0, 20.0, 0.0])
    )
    child = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Child", mesh=bp.box(), translation=[0.0, 0.0, 0.0])
    )
    doc.set_parent(child.uid, parent.uid, keep_world=False)
    doc.select([root.uid, child.uid])

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("align"), axis=1, mode=1) is True

    centres = []
    for uid in (root.uid, child.uid):
        lo, hi = clay_ops_geom.world_box(doc.by_uid(uid), world=doc.world_matrix(uid))
        centres.append(float((lo[1] + hi[1]) / 2.0))
    assert centres[0] == pytest.approx(centres[1]), (
        "both world Y centres should now agree -- fails against the unfixed "
        "code, which leaves the parented object's centre 20m away"
    )
