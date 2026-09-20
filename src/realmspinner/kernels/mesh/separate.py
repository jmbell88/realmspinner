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

#: The largest number of pieces one `by_loose_parts`/`by_material` call will
#: build. The 2026-09-20 audit's clay-04 found neither had any ceiling, and
#: that the closing comprehension re-scanned the *whole* face array once per
#: distinct group (`flatnonzero(face_labels == comp)` inside a Python loop
#: over every group) -- reproduced at 0.94s/16k pieces, 6.8s/64k, 26.0s/128k,
#: superlinear because that alone was `pieces * n_faces`. A second copy of the
#: same shape of bug was hiding one call deeper: `_piece` itself recomputed
#: `np.diff(mesh.starts)` -- the *whole* mesh's per-face corner counts -- from
#: scratch on every call, rather than once per split. `_grouped_pieces` below
#: fixes both: the grouping is one `argsort` and split, the same "compact
#: each label to a run of a sorted key, then `np.split`" trick
#: `ops_dissolve._group_by_label` already uses for exactly this shape of
#: problem, and `counts_all` is computed once and threaded through every
#: `_piece` call rather than per piece.
#:
#: A ceiling is still worth keeping past that fix, because `_piece` mints a
#: brand-new `Mesh` per group -- its own `np.unique` remap, its own six-field
#: dataclass, its own arrays -- and that per-piece Python/numpy overhead,
#: measured on this machine on the same many-disjoint-quads shape the audit
#: reproduced with (each piece a single quad, so this is the per-piece floor
#: rather than a per-face cost), only grows *mildly* superlinear rather than
#: with mesh size -- allocator/GC pressure from the sheer object count, not
#: an algorithmic term either fix above left behind:
#:
#: | pieces  | by_loose_parts() | us/piece |
#: |--------:|------------------:|---------:|
#: |   4,000 |             98 ms |    24.6 |
#: |   8,000 |            216 ms |    27.0 |
#: |  16,000 |            477 ms |    29.8 |
#: |  24,000 |            786 ms |    32.7 |
#: |  32,000 |          1,118 ms |    35.0 |
#:
#: Set well under the ~30,000-piece point where that curve crosses a second,
#: the same "well under a second" bar every sibling ceiling in this package
#: uses.
MAX_SEPARATE_PIECES = 20_000


def _refuse_piece_count(n_pieces: int) -> None:
    """Refuse before :func:`_grouped_pieces` builds a `Mesh` per piece --
    see :data:`MAX_SEPARATE_PIECES` for the measurements this ceiling is set
    under.
    """
    if n_pieces > MAX_SEPARATE_PIECES:
        raise el.OpError(
            f"Separating this mesh would build {n_pieces:,} pieces, past the "
            f"{MAX_SEPARATE_PIECES:,} Separate works with before it would "
            "stall the frame it runs on. Separate a mesh with fewer pieces."
        )


def _grouped_pieces(mesh: bm.Mesh, labels: np.ndarray) -> list[bm.Mesh]:
    """One :func:`_piece` per distinct value of *labels* (one entry per face),
    groups found with one ``argsort`` and split rather than a ``flatnonzero``
    rescan of the whole array per group -- see :data:`MAX_SEPARATE_PIECES` for
    the incident this replaced. Groups come back in ascending label order,
    each group's own faces ascending -- identical to what
    ``[flatnonzero(labels == v) for v in np.unique(labels)]`` produced, since
    ``argsort`` with ``kind="stable"`` preserves a tied group's original
    (ascending) order exactly as ``flatnonzero`` did.
    """
    order = np.argsort(labels, kind="stable")
    splits = np.flatnonzero(np.diff(labels[order])) + 1
    # ``counts_all`` is the whole mesh's per-face corner count, computed once
    # here rather than inside ``_piece`` -- see that function's own comment
    # for why a per-call ``np.diff(mesh.starts)`` was the other half of
    # clay-04's superlinear cost, on top of the grouping this function fixes.
    counts_all = np.diff(mesh.starts).astype("i8")
    return [
        _piece(mesh, group.astype("i8"), counts_all) for group in np.split(order, splits)
    ]


def _piece(mesh: bm.Mesh, faces: np.ndarray, counts_all: np.ndarray | None = None) -> bm.Mesh:
    """*mesh* restricted to *faces*, with unused vertices dropped and
    ``loops`` remapped to the compacted vertex array. See the module
    docstring for why this compacts rather than keeping the full array the
    way :func:`~.document._submesh`'s per-material render slice does.

    *counts_all* is the whole mesh's per-face corner count
    (``np.diff(mesh.starts)``), computed once by a caller that builds several
    pieces from the same mesh and passed through -- the 2026-09-20 audit's
    clay-04 found this function computing it fresh, from scratch, on *every*
    call, which made a piece-per-connected-component split cost
    ``pieces * n_faces`` even after :func:`_grouped_pieces` stopped
    rescanning the face array per group to find that piece's faces in the
    first place. ``None`` (the single-piece callers, :func:`by_selection`'s
    two-piece split among them) computes it locally, unchanged from before.
    """
    if counts_all is None:
        counts_all = np.diff(mesh.starts).astype("i8")
    counts = counts_all[faces]
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
    _refuse_piece_count(len(distinct))
    return _grouped_pieces(mesh, face_labels)


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
    _refuse_piece_count(len(materials))
    return _grouped_pieces(mesh, mesh.material)


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
    counts_all = np.diff(mesh.starts).astype("i8")
    return [_piece(mesh, faces, counts_all), _piece(mesh, rest, counts_all)]
