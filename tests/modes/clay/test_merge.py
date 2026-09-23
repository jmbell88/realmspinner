"""``kernels.mesh.merge``: folding a second document into the one on screen.

Pure kernel tests -- no ctx, no task thread, no imgui. The controller half
(polling a generation job, landing it in the right tab) is
``tests/modes/clay/test_clay_generate.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import merge
from realmspinner.kernels.mesh import ops as mesh_ops
from realmspinner.kernels.mesh import primitives as bp

ZERO = np.zeros(3, dtype="f8")


def _doc_with_box(name: str = "Box") -> bd.ClayDoc:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box()))
    return doc


def _incoming_two_boxes() -> bd.ClayDoc:
    """A document with two *root* objects -- what a multi-primitive assembly
    (a figure preset, several objects in one ``model.glb``) looks like."""
    incoming = bd.ClayDoc()
    incoming.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    incoming.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    return incoming


def test_merge_into_is_one_undo_step_and_one_redo_step():
    doc = _doc_with_box()
    before_count = len(doc.objects)
    before_head = doc.history.head

    incoming = _incoming_two_boxes()
    added = merge.merge_into(doc, incoming, offset=ZERO)

    assert len(added) == 2
    # Two boxes plus the group empty they were folded under.
    assert len(doc.objects) == before_count + 3
    assert doc.history.head != before_head

    assert doc.undo() is True
    assert len(doc.objects) == before_count
    assert doc.history.head == before_head

    assert doc.redo() is True
    assert len(doc.objects) == before_count + 3


def test_merge_into_offsets_face_materials_past_the_existing_palette():
    doc = _doc_with_box()
    assert len(doc.materials) == 1

    custom = gltf.Material(name="Custom")
    incoming = bd.ClayDoc(materials=[bd.default_material("M0"), custom])
    box_mesh = bp.box()
    faces = len(box_mesh.starts) - 1
    painted = dataclasses.replace(box_mesh, material=np.ones(faces, dtype="i4"))
    incoming.add_object(bd.Obj(uid=bd.new_uid(), name="Painted", mesh=painted, material=1))

    added = merge.merge_into(doc, incoming, offset=ZERO)

    # Existing slot 0 is untouched; the two incoming slots land at 1 and 2.
    assert len(doc.materials) == 3
    assert doc.materials[0] is not custom
    assert doc.materials[2] is custom
    merged = added[0]
    assert merged.material == 2
    assert set(doc.by_uid(merged.uid).mesh.material.tolist()) == {2}


def test_merge_into_keeps_the_incoming_texture_through_an_rblk_round_trip():
    from realmspinner.kernels.mesh import serialize

    doc = _doc_with_box()
    pixels = bytes(2 * 2 * 4)
    textured = gltf.Material(name="Tex", base_color=(2, 2, pixels))
    incoming = bd.ClayDoc(materials=[textured])
    incoming.add_object(bd.Obj(uid=bd.new_uid(), name="Textured", mesh=bp.box()))

    merge.merge_into(doc, incoming, offset=ZERO)

    reloaded = serialize.read_rblk(serialize.rblk_bytes(doc))
    landed = reloaded.materials[-1]
    assert landed.base_color is not None
    assert landed.base_color[2] == pixels


def test_merge_into_renames_objects_that_clash_with_the_document():
    doc = _doc_with_box("Box")
    incoming = bd.ClayDoc()
    incoming.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))

    added = merge.merge_into(doc, incoming, offset=ZERO)

    assert added[0].name != "Box"
    assert added[0].name.startswith("Box.")
    # And the original is left exactly as it was.
    assert doc.objects[0].name == "Box"


def test_a_multi_primitive_import_arrives_as_one_group():
    doc = bd.ClayDoc()
    incoming = _incoming_two_boxes()

    added = merge.merge_into(doc, incoming, offset=ZERO, group_name="Generated")

    assert len(added) == 2
    parents = {doc.by_uid(obj.uid).parent for obj in added}
    assert len(parents) == 1
    (parent_uid,) = parents
    assert parent_uid is not None
    group = doc.by_uid(parent_uid)
    assert group.name == "Generated"
    assert len(group.mesh.positions) == 0  # an empty, not a third mesh


def test_a_single_root_import_is_not_wrapped_in_a_group():
    doc = bd.ClayDoc()
    incoming = bd.ClayDoc()
    incoming.add_object(bd.Obj(uid=bd.new_uid(), name="Solo", mesh=bp.box()))

    added = merge.merge_into(doc, incoming, offset=ZERO)

    assert len(added) == 1
    assert doc.by_uid(added[0].uid).parent is None
    assert len(doc.objects) == 1


def test_placement_puts_the_import_beside_the_selection_on_the_ground():
    doc = bd.ClayDoc()
    selected = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Table", mesh=bp.box(), translation=(0.0, 0.5, 0.0))
    )
    incoming = bd.ClayDoc()
    incoming.add_object(bd.Obj(uid=bd.new_uid(), name="Lamp", mesh=bp.box((0.2, 0.2, 0.2))))

    offset = merge.placement_offset(doc, incoming, [selected.uid])

    sel_lo, sel_hi = mesh_ops.world_box(selected)
    in_lo, in_hi = mesh_ops.world_box(incoming.objects[0])
    new_lo, new_hi = in_lo + offset, in_hi + offset

    assert new_lo[0] >= sel_hi[0]  # beside, not overlapping
    assert new_lo[1] == pytest.approx(0.0)  # standing on the ground
    sel_z = (sel_lo[2] + sel_hi[2]) * 0.5
    new_z = (new_lo[2] + new_hi[2]) * 0.5
    assert new_z == pytest.approx(sel_z)


def test_placement_with_nothing_selected_is_the_origin_grounded():
    doc = bd.ClayDoc()
    incoming = bd.ClayDoc()
    incoming.add_object(
        bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), translation=(3.0, 5.0, -2.0))
    )

    offset = merge.placement_offset(doc, incoming, [])

    in_lo, in_hi = mesh_ops.world_box(incoming.objects[0])
    new_lo, new_hi = in_lo + offset, in_hi + offset

    assert (new_lo[0] + new_hi[0]) * 0.5 == pytest.approx(0.0)
    assert (new_lo[2] + new_hi[2]) * 0.5 == pytest.approx(0.0)
    assert new_lo[1] == pytest.approx(0.0)


def test_placement_with_an_empty_document_is_the_zero_vector():
    """Nothing to place -- ``merge_into`` still runs with no translation."""
    doc = bd.ClayDoc()
    incoming = bd.ClayDoc()  # no objects at all

    offset = merge.placement_offset(doc, incoming, [])

    assert np.array_equal(offset, ZERO)
