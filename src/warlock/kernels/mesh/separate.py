"""Splitting one mesh into several -- by loose parts, by material, or by a
selection -- pure functions over a :class:`~.mesh.Mesh`, with no document in
the loop.

Each function here answers "how would this mesh split", never "make it so":
:meth:`~.document.ClayDoc.separate` is the door that turns a ``list[Mesh]``
into new objects, as one undo step, with the source's transform, parent and
modifier stack carried onto every piece. That split mirrors every other
kernel/document boundary in this package (:mod:`.ops_boolean` computes a
mesh, ``ClayDoc.join_objects`` adopts it) -- the geometry and the bookkeeping
are two different kinds of correctness, and mixing them would make each
harder to test on its own.

**Every piece is a standalone mesh, positions and all -- not a view sharing
the source's full vertex array.** :func:`~.document._submesh` (the per-material
split :func:`~.document.to_primitives` uses) deliberately keeps the *whole*
array, because that split is a cheap, short-lived render slice of the same
object; a separated piece becomes its own :class:`~.document.Obj` and lives in
the document from here on, so an uncompacted array would carry the rest of the
original mesh's positions around for the life of the document for nothing.
:func:`_piece` compacts: only the vertices a piece's own faces still use, with
``loops`` remapped to match.

**A face's material and shading survive a split**, and so does every corner's
UV -- a piece is exactly the sub-mesh the source's own per-face/per-corner
arrays already describe for the faces it keeps, the same "restricted, not
recomputed" answer :func:`~.document._submesh` gives for the same fields.

**Refuses -- :class:`~.elements.OpError`, and computes nothing past the
check -- a split that would produce a single piece.** "Separate" whose result
is the mesh it started with is not a separation; the message names why
(everything is one connected piece / one material / the whole selection or
none of it), which is more useful than a silent single-element list.
"""

from __future__ import annotations

import numpy as np

from . import elements as el
from . import mesh as bm
from .adjacency import adjacency as mesh_adjacency

__all__ = ["by_loose_parts", "by_material", "by_selection"]


def _piece(mesh: bm.Mesh, faces: np.ndarray) -> bm.Mesh:
    """*mesh* restricted to *faces*, with unused vertices dropped and
    ``loops`` remapped to the compacted vertex array. See the module
    docstring for why this compacts rather than keeping the full array the
    way :func:`~.document._submesh`'s per-material render slice does.
    """
    counts = np.diff(mesh.starts).astype("i8")[faces]
    starts = np.concatenate([[0], np.cumsum(counts)]).astype("i4")
    total = int(starts[-1]) if len(starts) else 0
    if total:
        face_of_corner = np.repeat(np.arange(len(faces), dtype="i8"), counts)
        within = np.arange(total, dtype="i8") - starts[:-1].astype("i8")[face_of_corner]
        corners = mesh.starts[:-1].astype("i8")[faces][face_of_corner] + within
    else:
        corners = np.zeros(0, dtype="i8")
    old_vertex = mesh.loops[corners].astype("i8")
    kept_vertices, remap = np.unique(old_vertex, return_inverse=True)
    return bm.Mesh(
        positions=mesh.positions[kept_vertices],
        loops=remap.astype("i4"),
        starts=starts,
        material=mesh.material[faces],
        smooth=mesh.smooth[faces],
        uv=None if mesh.uv is None else mesh.uv[corners],
    )


def by_loose_parts(mesh: bm.Mesh) -> list[bm.Mesh]:
    """One piece per connected group of faces (shared *vertex*, not merely
    shared *position* -- two shells that touch without sharing an index, the
    way :func:`~.ops.join` at ``eps=0`` deliberately leaves two meshes it
    concatenated, stay separate pieces).

    Refuses (:class:`~.elements.OpError`) a mesh that is already one piece:
    zero or one face trivially is, and any larger mesh whose faces are all
    reachable from one another through shared vertices is too.
    """
    n = len(mesh.positions)
    edges = mesh_adjacency(mesh).edge_verts
    if len(edges):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        row = np.concatenate([edges[:, 0], edges[:, 1]]).astype("i8")
        col = np.concatenate([edges[:, 1], edges[:, 0]]).astype("i8")
        graph = csr_matrix((np.ones(len(row), dtype="i1"), (row, col)), shape=(n, n))
        _n_comp, labels = connected_components(graph, directed=False)
    else:
        labels = np.arange(n, dtype="i8")

    n_faces = bm.face_count(mesh)
    # Every corner of one face already shares a vertex-graph component with
    # every other corner of that face (the face's own edges are in the
    # graph), so the first corner's label speaks for the whole face.
    face_labels = (
        np.zeros(0, dtype="i8") if n_faces == 0 else labels[mesh.loops[mesh.starts[:-1]]]
    )

    distinct = np.unique(face_labels)
    if len(distinct) <= 1:
        raise el.OpError("Nothing to separate: this object is one connected piece.")
    return [_piece(mesh, np.flatnonzero(face_labels == comp)) for comp in distinct]


def by_material(mesh: bm.Mesh) -> list[bm.Mesh]:
    """One piece per distinct material slot the mesh's faces use.

    Refuses (:class:`~.elements.OpError`) a mesh whose faces all name one
    slot: there is nothing to separate by.
    """
    if bm.face_count(mesh) == 0:
        raise el.OpError("Nothing to separate: this object has no faces.")
    materials = np.unique(mesh.material)
    if len(materials) <= 1:
        raise el.OpError("Nothing to separate: every face uses the same material.")
    return [_piece(mesh, np.flatnonzero(mesh.material == m)) for m in materials]


def by_selection(mesh: bm.Mesh, sel: el.ElementSel) -> list[bm.Mesh]:
    """Two pieces: *sel*'s faces (converted up from vertex/edge, Wings3D's
    "a face is selected when all its corners are" rule -- :func:`~.elements.
    convert`), and everything else.

    Refuses (:class:`~.elements.OpError`) a selection touching none of the
    mesh's faces, or all of them: either leaves a single piece.
    """
    n_faces = bm.face_count(mesh)
    face_sel = el.convert(mesh, sel, "face")
    faces = np.asarray(face_sel.faces, dtype="i8")
    if len(faces) == 0 or len(faces) >= n_faces:
        raise el.OpError("Nothing to separate: select some faces, but not all of them.")
    rest = np.setdiff1d(np.arange(n_faces, dtype="i8"), faces, assume_unique=True)
    return [_piece(mesh, faces), _piece(mesh, rest)]
