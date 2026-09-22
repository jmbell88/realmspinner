"""Familiar's preview mechanics: clone, diff and transplant.

See ``studio/clay/scratch.py``'s own module docstring for the design this
pins -- a clone that shares meshes and materials by identity but copies
everything an op mutates in place, a diff by identity for meshes/materials
and by value for transforms, and a transplant that moves the scratch's own
objects and edits onto the real document through its ordinary doors rather
than replaying any tool call.
"""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import scratch as clay_scratch


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


def test_scratch_clone_carries_every_obj_field():
    """A self-adjusting gate against the exact defect ``clone``'s own
    docstring now names: it silently dropped ``parent``/``locked``/``tags``
    (tranche 3 fields) until this 2026-09-19 field sweep caught it -- a
    scratch preview of anything reading a parented object's world matrix saw
    every object as a root, and a locked object's own doors happily let a
    preview edit it, with the mismatch only surfacing as an unexplained
    transplant refusal three calls later. ``modifiers`` had already been
    caught and fixed the same way once before (see ``clone``'s own comment).

    Rather than re-deriving "did clone forget a field" from ``clone``'s own
    source, this builds one :class:`~realmspinner.kernels.mesh.document.Obj` with
    a value that differs from every field's own dataclass default, clones
    it, and checks every value survived. The value table's keys are checked
    against ``dataclasses.fields(Obj)`` first: adding a field to ``Obj``
    without adding it here fails this test immediately (a clear "update me"
    signal) rather than leaving a silent gap the rest of the test cannot see.
    """
    import dataclasses

    from realmspinner.kernels.mesh import modifiers as mod

    non_default: dict[str, object] = {
        "name": "distinctive",
        "translation": np.array([1.0, 2.0, 3.0]),
        "rotation": np.array([0.0, 0.0, 0.70710678, 0.70710678]),
        "scale": np.array([2.0, 2.0, 2.0]),
        "generator": "box",
        "params": {"size": [2.0, 2.0, 2.0]},
        "visible": False,
        "material": 3,
        "modifiers": (mod.make("mirror", {}, id=1),),
        # A fictitious uid -- nothing here exercises hierarchy semantics
        # (``world_matrix``, ``ancestors``), only whether the plain value
        # ``clone`` is handed comes back unchanged, so there is no need for
        # a second real object to parent onto.
        "parent": 999,
        "locked": True,
        "tags": ("hero", "prop"),
        "seams": ((0, 1), (2, 3)),
        "role": "collider",
        "collider_kind": "box",
    }
    obj_fields = {f.name for f in dataclasses.fields(bd.Obj)} - {"uid", "mesh"}
    assert non_default.keys() == obj_fields, (
        "Obj gained or lost a field without this test's own value table being "
        "updated to match -- see this test's own docstring"
    )

    doc = bd.ClayDoc()
    source = doc.add_object(bd.Obj(uid=bd.new_uid(), mesh=bp.box(), **non_default))
    scratch = clay_scratch.clone(doc)
    cloned = scratch.by_uid(source.uid)

    assert cloned.mesh is source.mesh  # shared, per clone()'s own contract
    for field_name, value in non_default.items():
        cloned_value = getattr(cloned, field_name)
        if isinstance(value, np.ndarray):
            assert np.array_equal(cloned_value, value), field_name
        else:
            assert cloned_value == value, field_name


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


def test_transplant_surfaces_a_material_the_real_document_refused_to_drop():
    """``remove_material``'s ``material_users`` also counts faces on objects
    the *undo stack* still holds (for redo) -- and the scratch clone's own
    stack starts empty (``clone``'s own comment), so a removal that succeeded
    on the scratch can still be refused when replayed against the real
    document, whose undo stack holds a deleted object that named the slot.
    Before this fix the refusal was silently swallowed: the palette just
    didn't shrink, with no signal on the result."""
    from realmspinner.kernels.mesh import mesh as bm

    doc = bd.ClayDoc()  # materials: [default] at index 0
    extra = doc.add_material()  # index 1 -- the slot this test drops

    box = bp.box()
    painted = bm.Mesh(
        positions=box.positions, loops=box.loops, starts=box.starts,
        smooth=box.smooth, material=np.full_like(box.material, extra),
    )
    doomed = doc.add_object(bd.Obj(uid=bd.new_uid(), name="doomed", mesh=painted))
    assert doc.remove_object(doomed.uid)  # gone from .objects, held by the undo stack

    # Cloned *after* the delete -- the scratch's live objects never used slot
    # 1 at all, and its own undo stack starts empty, so its removal succeeds
    # cleanly while the real document's undo-held object still blocks it.
    scratch = clay_scratch.clone(doc)
    assert scratch.remove_material(extra)

    result = clay_scratch.diff(doc, scratch)
    assert result.materials_changed

    changed = clay_scratch.transplant(doc, scratch, result)

    assert changed
    assert changed.kept_materials == [extra]
    assert len(doc.materials) == 2  # the palette did not shrink


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
    from realmspinner.kernels.mesh import mesh as bm

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


def test_transplant_does_not_change_a_locked_objects_modifiers_or_seams():
    """The 2026-09-19 audit, finding clay-19: ``transplant`` applied a scratch
    run's ``modifiers``/``seams`` edit through ``set_props``, which is
    deliberately not a locking door (a rename or an unlock must still work on
    a locked object -- see its own docstring), instead of ``set_modifiers``/
    ``set_seams``, which both call ``_refuse_if_locked``. An agent preview
    built while an object was unlocked, then applied after the user locked
    that object, walked straight past the lock and overwrote its modifier
    stack and marked seams anyway.

    The object is locked on the *real* document only after the preview is
    built, the same "state can move between preview and apply" path
    ``test_transplant_surfaces_a_material_the_real_document_refused_to_drop``
    already exercises for materials -- a scratch run has no way to know the
    base will be locked later, so the refusal has to happen at transplant
    time, and it must not abort the rest of the transplant.
    """
    from realmspinner.kernels.mesh import modifiers as mod

    doc = _doc(1)
    uid = doc.objects[0].uid
    scratch = clay_scratch.clone(doc)

    scratch.set_modifiers(uid, (mod.make("mirror", {}, id=1),))
    scratch.set_seams(uid, [(0, 1)])
    scratch.set_props(uid, name="renamed")  # a plain prop should still land

    result = clay_scratch.diff(doc, scratch)
    assert uid in result.props_changed
    assert {"modifiers", "seams", "name"} <= result.props_changed[uid]

    doc.set_props(uid, locked=True)

    changed = clay_scratch.transplant(doc, scratch, result)

    assert changed
    live = doc.by_uid(uid)
    assert live.modifiers == (), "a locked object's modifier stack must not change"
    assert live.seams == (), "a locked object's seams must not change"
    assert live.name == "renamed", "a non-locking prop still transplants"


def test_transplant_does_not_abort_the_rest_of_the_apply_when_a_locked_objects_mesh_or_transform_changed():  # noqa: E501
    """The 2026-09-20 audit's clay-13: ``transplant`` tolerated a locked
    object only for the ``modifiers``/``seams`` door (clay-19 above) --
    ``set_mesh`` and ``set_transform`` were called unguarded, so a mesh or
    transform change on an object locked between preview and apply raised
    ``OpError`` uncaught and aborted the *whole* transplant, against this
    module's own docstring ("the rest of the transplant still lands").
    ``preview.apply``'s own head check catches a real user's lock first, so
    this is only reachable by calling ``transplant`` directly -- the same way
    ``test_transplant_does_not_change_a_locked_objects_modifiers_or_seams``
    reaches its own otherwise-unreachable path.
    """
    doc = _doc(2)
    locked_uid = doc.objects[0].uid
    other_uid = doc.objects[1].uid
    original_mesh = doc.by_uid(locked_uid).mesh
    scratch = clay_scratch.clone(doc)

    scratch.set_mesh(locked_uid, bp.box(size=[2.0, 2.0, 2.0]), keep_generator=True)
    scratch.set_transform(locked_uid, translation=[3.0, 0.0, 0.0])
    scratch.set_transform(other_uid, translation=[0.0, 5.0, 0.0])

    result = clay_scratch.diff(doc, scratch)
    assert locked_uid in result.mesh_changed
    assert locked_uid in result.transform_changed
    assert other_uid in result.transform_changed

    doc.set_props(locked_uid, locked=True)

    changed = clay_scratch.transplant(doc, scratch, result)

    assert changed, "the transform change on the unlocked object must still push a step"
    assert doc.by_uid(locked_uid).mesh is original_mesh, (
        "a locked object's mesh must not change"
    )
    assert (doc.by_uid(locked_uid).translation == [0.0, 0.0, 0.0]).all(), (
        "a locked object's transform must not change"
    )
    assert (doc.by_uid(other_uid).translation == [0.0, 5.0, 0.0]).all(), (
        "a locked object aborting its own mesh/transform must not stop the rest of the transplant"
    )


def test_transplant_does_not_abort_the_rest_of_the_apply_when_a_removed_objects_lock_changed_before_apply():  # noqa: E501
    """The 2026-09-22 audit's clay-17: the ``removed`` loop is the first
    thing ``transplant`` does, and unlike the mesh/transform/props loops
    below it (the 2026-09-19 audit's clay-19 and the 2026-09-20 audit's
    clay-13), it called ``doc.remove_object`` with no
    ``contextlib.suppress(el.OpError)`` around it. Running first, an
    uncaught refusal here aborted the *whole* Apply before any of the
    changes queued behind it -- including an unrelated object's own
    transform change -- ever landed, which is exactly the "state can move
    between preview and apply" gap clay-19/clay-13 closed for their own
    doors.
    """
    doc = _doc(2)
    removed_uid = doc.objects[0].uid
    other_uid = doc.objects[1].uid
    scratch = clay_scratch.clone(doc)

    scratch.remove_object(removed_uid)
    scratch.set_transform(other_uid, translation=[3.0, 0.0, 0.0])

    result = clay_scratch.diff(doc, scratch)
    assert removed_uid in result.removed

    # Locked on the real document only after the preview was built -- the
    # scratch run had no way to know.
    doc.set_props(removed_uid, locked=True)

    changed = clay_scratch.transplant(doc, scratch, result)

    assert changed, "the other object's transform change must still push a step"
    live_uids = {o.uid for o in doc.objects}
    assert removed_uid in live_uids, "a locked object must not be removed"
    assert (doc.by_uid(other_uid).translation == [3.0, 0.0, 0.0]).all(), (
        "a locked removal aborting its own delete must not stop the rest of the transplant"
    )
