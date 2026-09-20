"""The 2026-09-19 audit, finding clay-03.

Pressing Delete in vertex or edge mode was a **completely silent no-op**
whenever the selection did not happen to cover every corner of at least one
whole face: ``selection.delete_selected`` converted the selection to faces,
found the result empty, and ``continue``d past the object with nothing
recorded -- so ``ops_topo.delete_faces``'s own refusal ("Select at least one
face to delete.") never had a chance to fire, and the toast at
``modes/clay/ops.py``'s ``_delete`` never appeared. This is the ordinary case
(two of a box face's four vertices, say), not an edge case, and it broke
``elements.py``'s documented contract that every refusal is toasted.
"""

from __future__ import annotations

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import selection


def _obj(name: str) -> bd.Obj:
    return bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box())


def test_delete_on_a_partial_vertex_selection_refuses_with_a_message_not_a_silent_no_op() -> (
    None
):
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_element_mode("vertex")
    # Face 0's corners are verts 0-3 (box()'s own loops); two of the four is
    # a selection that converts to zero faces, not one.
    doc.set_element_sel(a.uid, el.ElementSel(verts=[0, 1]))
    before_faces = len(a.mesh.starts) - 1
    depth = len(doc.history)

    refusals = selection.delete_selected(doc)

    assert refusals == ["Select at least one face to delete."]
    assert len(a.mesh.starts) - 1 == before_faces, "nothing should have been removed"
    assert len(doc.history) == depth, "a refusal records no undo step"


def test_a_multi_object_delete_refuses_only_the_object_that_did_not_cover_a_face() -> (
    None
):
    """One object with a whole face selected and one with a partial vertex
    selection: the object that converts cleanly still gets deleted, and the
    other is refused rather than either aborting the whole press or being
    silently skipped -- the same "a refusal on one object does not abandon
    the others" contract ``delete_selected``'s own docstring already
    documents for a raised ``OpError``.
    """
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    doc.set_element_mode("vertex")
    doc.set_element_sel(a.uid, el.ElementSel(verts=[0, 1, 2, 3]))  # all of face 0
    doc.set_element_sel(b.uid, el.ElementSel(verts=[0, 1]))  # half of face 0
    a_faces_before = len(a.mesh.starts) - 1
    b_faces_before = len(b.mesh.starts) - 1

    refusals = selection.delete_selected(doc)

    assert refusals == ["Select at least one face to delete."]
    assert len(a.mesh.starts) - 1 == a_faces_before - 1, "A's whole-face selection was deleted"
    assert len(b.mesh.starts) - 1 == b_faces_before, "B's partial selection was refused"
