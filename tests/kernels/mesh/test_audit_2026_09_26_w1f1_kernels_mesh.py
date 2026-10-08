"""Regressions for the 2026-09-26 audit's kernels/mesh findings (fixer w1f1).

Four findings still live, each closed in the module its own record names
(the other three -- a boolean modifier's cached error, ``analyze``'s coplanar
SAT and ``uvunwrap``'s seam-aware closedness -- went with the modifier stack,
the game check and the LSCM unwrap when Clay was cut to a picoCAD-level
modeller):

clay-mesh-core-01: ``selection.delete_selected``'s object-mode branch popped
straight out of ``doc.objects`` instead of going through ``remove_object``,
so deleting a parent left each child's ``parent`` naming a uid the document
no longer carries.

clay-document-01: ``column`` at radius 0 (with no base/capital) puts both
profile ends at radius 0, which ``_revolve`` reads as poles -- a shape
``column``'s own uv builder does not expect -- and the mismatch fails
``validate``.

clay-io-01: GLB import's ``scale``/``up`` remap rescaled only each object's
mesh, never its translation/rotation, so a placed child detached from the
shape it used to sit on.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import glbimport, meshimport, selection
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser
from realmspinner.kernels.mesh.mesh import validate


def _obj(name: str, mesh: object = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box() if mesh is None else mesh, **kwargs)


# --- clay-mesh-core-01 -------------------------------------------------------


def test_deleting_a_parent_object_reparents_its_children_and_the_document_still_round_trips() -> (
    None
):
    doc = bd.ClayDoc()
    parent = doc.add_object(_obj("Parent"))
    child = doc.add_object(_obj("Child", translation=(1.0, 0.0, 0.0)))
    doc.set_parent(child.uid, parent.uid, keep_world=True)
    assert child.parent == parent.uid
    doc.select([parent.uid])

    refusals = selection.delete_selected(doc)

    assert refusals == []
    assert parent.uid not in {o.uid for o in doc.objects}
    # The 2026-09-26 audit, finding clay-mesh-core-01: the unfixed code left
    # ``child.parent`` naming ``parent.uid``, a uid the document no longer
    # carries at all.
    assert child.parent is None, "the child must be re-parented, not left dangling"
    assert parent.uid not in doc._mesh_stamps

    # A dangling parent reference is exactly what serialize.read_rblk refuses
    # to reload -- round-tripping the document proves the fix rather than
    # just the in-memory state.
    reloaded = ser.read_rblk(ser.rblk_bytes(doc))
    reloaded_child = next(o for o in reloaded.objects if o.name == "Child")
    assert reloaded_child.parent is None


def test_deleting_two_selected_objects_in_object_mode_is_still_one_undo_step() -> None:
    """The fold this fix now goes through (``mark``/``collapse_since`` around
    per-uid ``remove_object`` calls) must not regress the one-undo-step
    promise ``delete_selected``'s own docstring makes for a multi-object
    Delete (the 2026-09-06 audit's clay-01)."""
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    doc.select([a.uid, b.uid])
    depth = len(doc.history)

    refusals = selection.delete_selected(doc)

    assert refusals == []
    assert len(doc.objects) == 0
    assert len(doc.history) == depth + 1
    assert doc.undo() is True
    assert {o.uid for o in doc.objects} == {a.uid, b.uid}


# --- clay-document-01 --------------------------------------------------------


def test_a_column_at_zero_radius_builds_a_mesh_that_validates_and_round_trips_through_rblk() -> (
    None
):
    mesh = bp.column(radius=0.0)

    validate(mesh)  # must not raise -- reproduced: uv sized for 288 corners, 352 rows

    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Column", mesh=mesh))
    reloaded = ser.read_rblk(ser.rblk_bytes(doc))  # must not raise
    assert len(reloaded.objects) == 1


# --- clay-io-01 --------------------------------------------------------------


def test_glb_import_scale_moves_object_translations_along_with_their_meshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``import_file`` reads a GLB through ``glb_to_claydoc_report`` (the
    # 2026-10-07 audit's clay-07), which is what is stubbed here.
    def _fake_glb_to_claydoc(data: bytes, name: str) -> tuple[bd.ClayDoc, int]:
        del data, name
        doc = bd.ClayDoc()
        doc.add_object(_obj("Root", translation=(10.0, 0.0, 0.0)))
        return doc, 0

    monkeypatch.setattr(glbimport, "glb_to_claydoc_report", _fake_glb_to_claydoc)

    doc = meshimport.import_file(b"", ".glb", scale=0.01)

    obj = doc.objects[0]
    # The 2026-09-26 audit, finding clay-io-01: the unfixed code left
    # ``obj.translation`` at ``(10.0, 0.0, 0.0)`` -- only the mesh itself
    # shrank, so the object jumped away from where its now-tiny mesh sits.
    assert np.allclose(obj.translation, (0.1, 0.0, 0.0)), obj.translation
    assert np.allclose(np.abs(obj.mesh.positions).max(), 0.005)
