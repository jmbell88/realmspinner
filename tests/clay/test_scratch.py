"""Familiar's preview mechanics: clone, diff and transplant.

See ``studio/clay/scratch.py``'s own module docstring for the design this
pins -- a clone that shares meshes and materials by identity but copies
everything an op mutates in place, a diff by identity for meshes/materials
and by value for transforms, and a transplant that moves the scratch's own
objects and edits onto the real document through its ordinary doors rather
than replaying any tool call.
"""

from __future__ import annotations

from warlock.studio.clay import document as bd
from warlock.studio.clay import primitives as bp
from warlock.studio.clay import scratch as clay_scratch


def _obj(name: str = "obj", mesh=None, **kwargs) -> bd.Obj:
    return bd.Obj(uid=bd.new_uid(), name=name, mesh=mesh or bp.box(), **kwargs)


def _doc(count: int = 2) -> bd.ClayDoc:
    doc = bd.ClayDoc()
    for i in range(count):
        doc.add_object(_obj(f"obj{i}", params={"size": [1.0, 1.0, 1.0]}))
    return doc


# --- clone -------------------------------------------------------------------


def test_a_clone_shares_meshes_but_not_params_dicts():
    doc = _doc(2)
    scratch = clay_scratch.clone(doc)

    assert [o.uid for o in scratch.objects] == [o.uid for o in doc.objects]
    for base_obj, scratch_obj in zip(doc.objects, scratch.objects, strict=True):
        assert scratch_obj.mesh is base_obj.mesh
        assert scratch_obj.params == base_obj.params
        assert scratch_obj.params is not base_obj.params
        assert scratch_obj.translation is not base_obj.translation

    # Editing the clone's params (as a real op handler does: read, mutate,
    # write back) must never reach the base document's own dict.
    scratch.objects[0].params["size"] = [9.0, 9.0, 9.0]
    assert doc.objects[0].params["size"] == [1.0, 1.0, 1.0]

    # The palette is a fresh list holding the same material objects.
    assert scratch.materials is not doc.materials
    assert scratch.materials == doc.materials
    for a, b in zip(doc.materials, scratch.materials, strict=True):
        assert a is b

    # A fresh, empty history -- nothing in the clone has "already happened".
    assert len(scratch.history) == 0


def test_clone_copies_selection_and_element_state():
    doc = _doc(2)
    uid = doc.objects[0].uid
    doc.select([uid])
    doc.set_element_mode("vertex")

    scratch = clay_scratch.clone(doc)
    assert scratch.selection == doc.selection
    assert scratch.selection is not doc.selection
    assert scratch.element_mode == "vertex"


def test_a_new_object_minted_during_a_scratch_run_never_collides_with_the_base():
    doc = _doc(3)
    scratch = clay_scratch.clone(doc)
    base_uids = {o.uid for o in doc.objects}

    new_obj = _obj("fresh")
    scratch.add_objects([new_obj])

    # ``new_uid`` is a process-global counter (document.py's own comment: "per
    # process, not per document"), so nothing here has to reserve anything --
    # the new uid simply cannot already be one of the base's.
    assert new_obj.uid not in base_uids


# --- diff ----------------------------------------------------------------


def test_diff_reports_added_and_removed_uids():
    doc = _doc(2)
    scratch = clay_scratch.clone(doc)
    removed_uid = scratch.objects[0].uid
    scratch.remove_object(removed_uid)
    added = scratch.add_objects([_obj("new")])[0]

    result = clay_scratch.diff(doc, scratch)
    assert result.removed == {removed_uid}
    assert result.added == {added.uid}
    assert not result.empty


def test_diff_compares_meshes_by_identity_and_transforms_by_value():
    doc = _doc(1)
    uid = doc.objects[0].uid
    scratch = clay_scratch.clone(doc)

    # Writing back the same numbers is not a change.
    obj = scratch.by_uid(uid)
    scratch.set_transform(uid, translation=obj.translation, rotation=obj.rotation,
                           scale=obj.scale)
    no_op_diff = clay_scratch.diff(doc, scratch)
    assert uid not in no_op_diff.transform_changed
    assert no_op_diff.empty

    scratch.set_transform(uid, translation=[5.0, 0.0, 0.0])
    scratch.set_mesh(uid, bp.box(size=(2.0, 2.0, 2.0)))
    moved_diff = clay_scratch.diff(doc, scratch)
    assert uid in moved_diff.transform_changed
    assert uid in moved_diff.mesh_changed


def test_diff_captures_the_base_snapshot_at_preview_time():
    doc = _doc(1)
    uid = doc.objects[0].uid
    doc.select([uid])
    doc.set_element_mode("object")
    scratch = clay_scratch.clone(doc)

    result = clay_scratch.diff(doc, scratch)
    assert result.base_head == doc.history.head
    assert result.base_selection == {uid}
    assert result.base_element_mode == "object"


# --- transplant ------------------------------------------------------------


def test_transplant_applies_exactly_what_the_diff_reports_as_one_step():
    doc = _doc(2)
    before_head = doc.history.head
    original_uids = {o.uid for o in doc.objects}
    scratch = clay_scratch.clone(doc)
    kept_uid = scratch.objects[0].uid
    removed_uid = scratch.objects[1].uid
    scratch.remove_object(removed_uid)
    scratch.set_transform(kept_uid, translation=[3.0, 0.0, 0.0])
    added = scratch.add_objects([_obj("brand_new")])[0]

    result = clay_scratch.diff(doc, scratch)
    changed = clay_scratch.transplant(doc, scratch, result)

    assert changed
    assert doc.history.head != before_head
    live_uids = {o.uid for o in doc.objects}
    assert removed_uid not in live_uids
    assert added.uid in live_uids
    assert (doc.by_uid(kept_uid).translation == [3.0, 0.0, 0.0]).all()

    # Undo restores the whole preview in one press.
    assert doc.undo()
    assert doc.history.head == before_head
    assert {o.uid for o in doc.objects} == original_uids


def test_transplant_added_objects_keep_the_scratch_uids():
    doc = _doc(1)
    scratch = clay_scratch.clone(doc)
    added = scratch.add_objects([_obj("agent_made")])[0]
    result = clay_scratch.diff(doc, scratch)
    clay_scratch.transplant(doc, scratch, result)
    assert doc.by_uid(added.uid) is not None
    assert doc.by_uid(added.uid).name == "agent_made"


def test_transplant_is_a_no_op_on_an_empty_diff():
    doc = _doc(1)
    before_head = doc.history.head
    scratch = clay_scratch.clone(doc)
    result = clay_scratch.diff(doc, scratch)
    assert result.empty
    assert clay_scratch.transplant(doc, scratch, result) is False
    assert doc.history.head == before_head


def test_transplant_with_material_removal_keeps_face_indices_right():
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("multi"))
    extra = doc.add_material()  # index 1, unused -- the one the scratch drops
    third_material = doc.materials[doc.add_material()]  # index 2

    # Paint half the faces with the third slot -- what the removal's own
    # renumbering has to get right when the *middle* slot goes.
    faces = obj.mesh.material.copy()
    third_index = next(i for i, m in enumerate(doc.materials) if m is third_material)
    faces[: len(faces) // 2] = third_index
    from warlock.studio.clay import mesh as bm

    repainted = bm.Mesh(
        positions=obj.mesh.positions, loops=obj.mesh.loops, starts=obj.mesh.starts,
        smooth=obj.mesh.smooth, material=faces,
    )
    doc.set_mesh(obj.uid, repainted, keep_generator=True)

    scratch = clay_scratch.clone(doc)
    assert scratch.remove_material(extra)  # drops the unused middle slot

    result = clay_scratch.diff(doc, scratch)
    assert result.materials_changed
    assert clay_scratch.transplant(doc, scratch, result)

    # One slot gone, and the faces that pointed at "third" now point at
    # whatever index it landed on -- but at the *material object itself*,
    # not a number that happened to match before the shift.
    assert len(doc.materials) == 2
    assert any(m is third_material for m in doc.materials)
    new_third_index = next(i for i, m in enumerate(doc.materials) if m is third_material)
    live_mesh_material = doc.by_uid(obj.uid).mesh.material
    painted = live_mesh_material[: len(live_mesh_material) // 2]
    assert (painted == new_third_index).all()
