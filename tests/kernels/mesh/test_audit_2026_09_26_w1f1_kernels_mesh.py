"""Regressions for the 2026-09-26 audit's kernels/mesh findings (fixer w1f1).

Seven findings, each closed in the module its own record names:

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

clay-mesh-core-03: ``select.grow`` computed its two adjacency passes off the
same mutating array, so a vertex adjacent to one of this call's own
additions grew a second ring.

clay-mesh-model-01: a boolean modifier's cached "target no longer exists"
error recorded no dependency at all, so undoing the delete that caused it
never invalidated the cache.

clay-mesh-model-02: the 11-axis SAT in ``analyze._tri_tri_intersect`` has no
axis that lies inside a shared plane, so two disjoint coplanar triangles
read as intersecting.

clay-mesh-uv-01: ``uvunwrap._island_is_closed`` decided closedness from raw
edge-uses-per-island counts alone, ignoring the seam set, so a sphere cut
along one meridian (structurally still "every edge used twice", the seam
being pure authoring metadata) read as an uncut closed surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import glbimport, meshimport, select, selection
from realmspinner.kernels.mesh import modifiers as mod
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser
from realmspinner.kernels.mesh import uvunwrap as lscm
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
    assert parent.uid not in doc._evaluated

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
    def _fake_glb_to_claydoc(data: bytes, name: str) -> bd.ClayDoc:
        del data, name
        doc = bd.ClayDoc()
        doc.add_object(_obj("Root", translation=(10.0, 0.0, 0.0)))
        return doc

    monkeypatch.setattr(glbimport, "glb_to_claydoc", _fake_glb_to_claydoc)

    doc = meshimport.import_file(b"", ".glb", scale=0.01)

    obj = doc.objects[0]
    # The 2026-09-26 audit, finding clay-io-01: the unfixed code left
    # ``obj.translation`` at ``(10.0, 0.0, 0.0)`` -- only the mesh itself
    # shrank, so the object jumped away from where its now-tiny mesh sits.
    assert np.allclose(obj.translation, (0.1, 0.0, 0.0)), obj.translation
    assert np.allclose(np.abs(obj.mesh.positions).max(), 0.005)


# --- clay-mesh-core-03 -------------------------------------------------------


def _grid_mesh(n: int = 5) -> bd.Obj:
    """An *n* x *n* grid of quads in the XZ plane, one vertex per grid point,
    row-major -- enough rings around an interior vertex to tell "one ring"
    from "two" apart."""
    xs, zs = np.meshgrid(np.arange(n, dtype="f8"), np.arange(n, dtype="f8"))
    positions = np.stack([xs.ravel(), np.zeros(n * n), zs.ravel()], axis=1)
    faces = []
    for row in range(n - 1):
        for col in range(n - 1):
            a = row * n + col
            b = a + 1
            c = a + n + 1
            d = a + n
            faces.append([a, b, c, d])
    from realmspinner.kernels.mesh.mesh import from_faces

    return from_faces(positions, faces)


def test_grow_from_an_interior_vertex_takes_exactly_its_edge_neighbours() -> None:
    mesh = _grid_mesh(5)
    # Vertex 12 is the exact centre of the 5x5 grid (row 2, col 2): its four
    # edge neighbours are 7, 11, 13, 17.
    grown = select.grow(mesh, np.array([12], dtype="i4"))

    # The 2026-09-26 audit, finding clay-mesh-core-03: the unfixed code
    # returned [7, 8, 11, 12, 13, 16, 17] -- two rings, because the second
    # adjacency pass read the first pass's own in-place writes.
    assert sorted(grown.tolist()) == [7, 11, 12, 13, 17]


# --- clay-mesh-model-01 -------------------------------------------------------


def test_undoing_a_target_delete_revives_the_boolean_modifier_result() -> None:
    pytest.importorskip("manifold3d")
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    # Disjoint (no overlap at all): a union just concatenates both meshes, so
    # the "good" result has exactly 16 verts against A's own 8 -- the same
    # 16 -> 8 shape the audit's own reproduction names.
    b = doc.add_object(_obj("B", translation=(3.0, 0.0, 0.0)))
    doc.set_modifiers(a.uid, (mod.make("boolean", {"target": b.uid, "operation": "union"}, id=1),))

    good = doc.evaluation(a.uid)
    assert good.errors == (), good.errors
    good_verts = len(good.mesh.positions)
    assert good_verts == 16, "sanity: a union of two disjoint boxes must keep both"

    doc.remove_object(b.uid)
    errored = doc.evaluation(a.uid)
    assert errored.errors, "the target is gone -- this must record an error"
    assert len(errored.mesh.positions) == len(a.mesh.positions), (
        "with the target missing, the modifier falls back to the base mesh"
    )

    assert doc.undo() is True  # brings B back, same uid
    # The 2026-09-26 audit, finding clay-mesh-model-01: the unfixed code
    # recorded no dependency at all for a missing target, so this second
    # evaluation kept serving the cached error (8 verts) instead of noticing
    # B had come back and recomputing (16 verts, reproduced).
    revived = doc.evaluation(a.uid)
    assert revived.errors == (), revived.errors
    assert len(revived.mesh.positions) == good_verts


# --- clay-mesh-model-02 -------------------------------------------------------


def test_two_boxes_with_a_gap_and_coplanar_faces_do_not_intersect() -> None:
    from realmspinner.kernels.mesh import analyze

    a = _obj("A")
    # A's +X face sits at world x = 0.5 (a unit box centred on the origin).
    # B's -X face is coplanar with it (also at world x = 0.5, up to the gap)
    # but offset 0.04 m further out, so the two boxes never actually touch --
    # axis-aligned level-kit pieces set near, not against, each other.
    b = _obj("B", translation=(1.04, 0.0, 0.0))

    result = analyze.analyze([a, b])
    assert len(result.pairs) == 1
    pair = result.pairs[0]

    # The 2026-09-26 audit, finding clay-mesh-model-02: the unfixed 11-axis
    # SAT found no separating axis for this coplanar-disjoint pair and
    # reported ``intersects=True, distance=0.0`` (reproduced).
    assert pair.intersects is False, pair
    assert pair.distance is not None and pair.distance > 0.0, pair


# --- clay-mesh-uv-01 ----------------------------------------------------------


def test_unwrap_lscm_accepts_a_closed_sphere_cut_along_a_meridian_seam() -> None:
    n, m = 16, 8
    sphere = bp.uv_sphere(0.5, n, m)
    top, bottom = 0, 1 + (m - 1) * n

    def row(j: int) -> int:
        return 1 + (j - 1) * n

    seam_pairs = [(top, row(1))]
    seam_pairs.extend((row(j), row(j + 1)) for j in range(1, m - 1))
    seam_pairs.append((row(m - 1), bottom))
    seams = np.array(seam_pairs, dtype="i4")

    # The 2026-09-26 audit, finding clay-mesh-uv-01: the unfixed
    # ``_island_is_closed`` ignored the seam set entirely and refused this
    # with "A closed surface cannot be flattened with no seam", even though
    # one is plainly marked.
    result = lscm.unwrap_lscm(sphere, seams)

    assert result.uv is not None
    assert result.uv.shape == (len(result.loops), 2)
    assert np.isfinite(result.uv).all()
