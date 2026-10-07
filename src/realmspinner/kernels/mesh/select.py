"""Selection verbs: linked, grow, and the by-property queries.

Pure, and over the adjacency. Nothing here touches a document, a selection
object or a view: each takes a mesh and a set of indices and returns a set of
indices, which is what lets "does linked stop at a shell boundary" be a plain
assertion.

Two conventions travel with the package. **An edge is a vertex pair**, not an
id: ids renumber globally whenever the topology changes and pairs do not, which
is why ``ElementSel`` stores pairs and why everything here returns them.
And **a refusal is empty, never an exception**: these are reached by a keystroke
during a selection, and a key that raises is a key that takes the window down.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from . import elements as el
from . import mesh as bm
from . import topo
from .adjacency import adjacency
from .mesh import Mesh

# --- everything connected -----------------------------------------------------


def linked(mesh: Mesh, verts: np.ndarray) -> np.ndarray:
    """Every vertex reachable from ``verts`` along edges. -> vertex indices.

    The verb that makes two objects welded into one mesh separable again: L
    over one of them takes the whole shell.

    ``scipy.sparse.csgraph.connected_components`` over the edge graph, not
    label propagation. Propagation's pass count is a shell's *length* along
    the mesh, not the square root of its size, so four disconnected quad
    strips two faces wide totalling 200k vertices took 11.1 s -- 11 seconds on
    the frame thread for one L key (dev/measurements/
    2026-09-13-native-batch-10-candidates.md, §1). ``connected_components``
    answers the same question -- which vertices share a component with the
    seeds -- in one pass over the whole graph regardless of its shape, and
    ``adjacency`` is already cached per mesh so the graph itself costs
    nothing extra to build here.
    """
    seeds = np.unique(np.asarray(verts, dtype="i8").reshape(-1))
    count = len(mesh.positions)
    if not len(seeds) or count == 0:
        return np.zeros(0, dtype="i4")
    seeds = seeds[(seeds >= 0) & (seeds < count)]
    if not len(seeds):
        return np.zeros(0, dtype="i4")
    a = adjacency(mesh)
    if a.n_edges == 0:
        return np.unique(seeds).astype("i4")
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = a.edge_verts.astype("i8")
    graph = coo_matrix(
        (np.ones(len(edges), dtype="i1"), (edges[:, 0], edges[:, 1])),
        shape=(count, count),
    )
    labels = connected_components(graph, directed=False)[1]
    inside = np.isin(labels, np.unique(labels[seeds]))
    return np.flatnonzero(inside).astype("i4")


def grow(mesh: Mesh, verts: np.ndarray) -> np.ndarray:
    """One ring outward. -> vertex indices."""
    seeds = np.unique(np.asarray(verts, dtype="i8").reshape(-1))
    count = len(mesh.positions)
    if not len(seeds) or count == 0:
        return np.zeros(0, dtype="i4")
    a = adjacency(mesh)
    inside = np.zeros(count, dtype=bool)
    inside[seeds[(seeds >= 0) & (seeds < count)]] = True
    if a.n_edges:
        lo = a.edge_verts[:, 0].astype("i8")
        hi = a.edge_verts[:, 1].astype("i8")
        # Both passes read a snapshot taken before either writes, so a vertex
        # can only join ``inside`` for being adjacent to a vertex that was
        # *already* selected, never to one this same call just added (the
        # 2026-09-26 audit's clay-mesh-core-03: one ring came back as two).
        base = inside.copy()
        inside[lo[base[hi]]] = True
        inside[hi[base[lo]]] = True
    return np.flatnonzero(inside).astype("i4")


def by_material(mesh: Mesh, slot: int) -> np.ndarray:
    """Every face using palette slot ``slot``. -> face indices.

    The one selection verb that is about the *document* rather than the
    topology, and it earns its place for the reason a material slot exists at
    all: "show me everything painted with this" is how a slot gets reassigned,
    and there was no way to ask.
    """
    material = np.asarray(mesh.material, dtype="i8")
    return np.flatnonzero(material == int(slot)).astype("i4")


# --- currency conversion --------------------------------------------------
#
# Moved down from ``studio/modes/clay/ops.py`` (the 2026-09-10 groundwork pass): both
# functions were already pure ``(mesh, sel, mode)`` -> currency conversions
# with no opinion about a document, a toast or a key binding, which is
# everything that belongs in this module and nothing that belonged one level
# up. ``_sel_from_verts`` in particular reached into ``elements.
# _face_corner_mask`` -- a private of *this* package's sibling -- from the ops
# layer, which is the reach that said "this function is sitting one level too
# high" as plainly as the lazy ``from .clay import elements as el`` /
# ``from .clay.adjacency import adjacency`` imports it needed to get there did.
# ``clay_ops``'s five ``_verb_*`` wrappers now import these from here instead.


def verts_of(mesh: Mesh, sel: el.ElementSel, mode: str) -> np.ndarray:
    """Whatever is selected, as a set of vertices.

    The common currency: linking from a face selection and from a vertex one
    are the same walk over the same graph, and converting once here is what
    keeps the rest of this module free of a mode argument.
    """
    if mode == "vertex":
        return np.asarray(sel.verts, dtype="i8")
    if mode == "edge":
        return np.unique(np.asarray(sel.edges, dtype="i8").reshape(-1))
    faces = np.asarray(sel.faces, dtype="i8")
    if not len(faces):
        return np.zeros(0, dtype="i8")
    starts = np.asarray(mesh.starts, dtype="i8")
    faces = faces[(faces >= 0) & (faces < len(starts) - 1)]
    if not len(faces):
        return np.zeros(0, dtype="i8")
    # The 2026-09-26 audit (clay-mesh-core-05): this list comprehension
    # gathered corners one face at a time (0.41s at 490,000 faces); the same
    # vectorised CSR gather `elements.affected_verts` now uses replaces it.
    loops = np.asarray(mesh.loops, dtype="i8")
    return np.unique(loops[topo.corner_spans(starts, faces)])


def sel_from_verts(mesh: Mesh, verts: np.ndarray, mode: str) -> el.ElementSel:
    """A vertex set back into the mode's own currency.

    An edge or a face is included when **every** one of its vertices is, which
    is the only definition that makes a vertex set and the selection it came
    from round-trip on the inside of a selection: "partly selected" is not a
    state an element selection can be in.
    """
    verts = np.unique(np.asarray(verts, dtype="i8"))
    if mode == "vertex":
        return el.ElementSel(verts=verts)
    inside = np.zeros(len(mesh.positions), dtype=bool)
    inside[verts[(verts >= 0) & (verts < len(inside))]] = True
    if mode == "face":
        # ``elements._face_corner_mask``, which is this question vectorised and
        # is the same definition ``convert`` uses to go *up* a level. This had
        # a Python loop over every face of the mesh -- so a 200k-face
        # import walked all of them per press, for an answer numpy already had
        # -- and a second spelling of "a face is selected only when all of its
        # corners are".
        mask = el._face_corner_mask(mesh, verts)
        return el.ElementSel(faces=np.flatnonzero(mask).astype("i4"))
    a = adjacency(mesh)
    if a.n_edges == 0:
        return el.ElementSel()
    both = inside[a.edge_verts[:, 0]] & inside[a.edge_verts[:, 1]]
    return el.ElementSel(edges=a.edge_verts[both])


# --- selecting by a property of the geometry, not by a walk over it ----------
#
# ``linked`` and ``grow`` above walk the adjacency graph outward from a seed. The
# two below ask a different kind of question -- "which faces face roughly this
# way" and "which faces sit inside this box" -- that is answered from a face's
# own geometry rather than from its neighbours, but it is the same vocabulary
# of pure ``mesh -> indices`` functions and the same "a refusal is empty, never
# an exception" convention.


def faces_by_normal(
    mesh: Mesh, direction: Sequence[float], max_angle: float = 45.0
) -> np.ndarray:
    """Faces whose normal points within *max_angle* degrees of *direction*.
    -> face indices.

    Newell normals (:func:`~.mesh.face_normals`), normalised with the same
    ``np.divide(..., where=lengths > 1e-12)`` guard :func:`~.shading.
    auto_smooth` uses on the identical array -- a Newell normal's magnitude is
    twice the face's area, so an unnormalised dot product is not a cosine and
    comparing it against one would silently call every face's angle wrong.

    **A degenerate face -- zero area, or a normal that collapsed to nothing --
    matches nothing, whatever *max_angle* is asked for**, rather than matching
    every direction. That is the trap: dividing a zero vector by a guarded-to-
    zero length leaves it ``[0, 0, 0]``, whose dot product with *any* unit
    direction is exactly ``0`` -- which is ``>= cos(max_angle)`` the moment
    *max_angle* reaches 90 degrees, so a wide-angle query would otherwise pick
    up every sliver face in the mesh as "facing every direction at once". The
    validity mask below is checked independently of the angle comparison for
    exactly that reason.

    *direction* is normalised here too, and a *direction* with no length (the
    caller's own degenerate input) selects nothing rather than raising.
    """
    n_faces = bm.face_count(mesh)
    if n_faces == 0:
        return np.zeros(0, dtype="i4")
    normals = np.asarray(bm.face_normals(mesh), dtype="f8")
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1e-12
    unit = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)
    d = np.asarray(direction, dtype="f8")
    d_length = float(np.linalg.norm(d))
    if d_length <= 1e-12:
        return np.zeros(0, dtype="i4")
    cos = unit @ (d / d_length)
    limit = float(np.cos(np.radians(max(0.0, min(180.0, float(max_angle))))))
    return np.flatnonzero(valid & (cos >= limit)).astype("i4")


def faces_in_bounds(
    mesh: Mesh,
    lo: Sequence[float],
    hi: Sequence[float],
    *,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Faces every one of whose corners lies within ``[lo, hi]``. -> face
    indices.

    The same conjunction :func:`sel_from_verts` and ``elements.
    _face_corner_mask`` use to go *up* a level from a vertex set to a face
    one, and the only rule consistent with this package's "partly selected is
    not a state" claim -- a face with one corner in the box and the rest of it
    hanging out of the far side is not "in bounds".

    *positions* defaults to ``mesh.positions`` (local space) and exists so a
    caller can pass **world**-space positions instead -- see
    :func:`~.ops.world_positions`, which is the only correct way to get them:
    a box cannot be transformed into local space when the object is rotated,
    because the local axis-aligned box of a rotated *world* box is not itself
    a box, so it is the *positions* that have to move to answer this question
    in another space, never the bounds.
    """
    n_faces = bm.face_count(mesh)
    if n_faces == 0:
        return np.zeros(0, dtype="i4")
    pts = np.asarray(mesh.positions if positions is None else positions, dtype="f8")
    lo_arr = np.asarray(lo, dtype="f8")
    hi_arr = np.asarray(hi, dtype="f8")
    inside = np.all((pts >= lo_arr) & (pts <= hi_arr), axis=1)
    starts = np.asarray(mesh.starts, dtype="i8")
    mask = np.minimum.reduceat(inside[mesh.loops].astype("i1"), starts[:-1]) > 0
    return np.flatnonzero(mask).astype("i4")


# --- QUERIES: selection answered from parameters ------------------------------
#
# ``OPS`` (``studio/modes/clay/ops.py``) is invocable verbs and the agent's
# derived tool list is built from it, never hand-listed. A query is a different
# shape from a verb: it does not *change* the selection in place, it *answers*
# one from parameters an agent read out of a scene report ("slot 3", "faces
# facing up"), which ``OPS``'s ``run(ctx, doc, **params)`` shape has no natural
# place for. It lives here, next to the verbs it is built from, and the agent
# surface derives its tool list from :data:`QUERIES`. ``all``, ``none``,
# ``invert`` and ``linked`` are not here: they already exist as ``select-*``
# rows in ``OPS`` and a second door for a verb that has one is exactly the
# drift "one list, not three" exists to prevent.
#
# ``args`` names the vocabulary the JSON-schema layer offers for each query --
# not necessarily ``run``'s own keyword names. ``bounds``'s ``space`` is the
# clearest case: whether the box an agent supplies is in world or local
# coordinates is a decision the *agent surface* makes, converting through
# :func:`~.ops.world_positions` before ever calling in here, because this
# module has no opinion about anything but the mesh's own local space.


@dataclass(frozen=True)
class Query:
    """One way to answer "what should be selected", from a seed or parameters.

    ``modes`` is which element mode(s) the answer can come back in -- ``face``
    for ``material``, both ``face`` and ``vertex`` for ``bounds``, since a box
    is a sensible answer in either currency and :func:`_q_bounds` returns both
    at once rather than forcing a caller to pick.
    """

    name: str
    modes: tuple[str, ...]
    args: tuple[str, ...]
    run: Callable[..., el.ElementSel]
    hint: str


def _q_material(mesh: Mesh, slot: int) -> el.ElementSel:
    return el.ElementSel(faces=by_material(mesh, slot))


def _q_normal(mesh: Mesh, direction: Sequence[float], max_angle: float = 45.0) -> el.ElementSel:
    return el.ElementSel(faces=faces_by_normal(mesh, direction, max_angle))


def _q_bounds(mesh: Mesh, lo: Sequence[float], hi: Sequence[float]) -> el.ElementSel:
    """Both currencies at once: every vertex in the box, and every face all of
    whose corners are (:func:`faces_in_bounds`) -- the two things "select
    what's in this box" can honestly mean, computed from the one scan."""
    pts = np.asarray(mesh.positions, dtype="f8")
    lo_arr = np.asarray(lo, dtype="f8")
    hi_arr = np.asarray(hi, dtype="f8")
    if len(pts) == 0:
        return el.ElementSel()
    inside = np.all((pts >= lo_arr) & (pts <= hi_arr), axis=1)
    return el.ElementSel(
        verts=np.flatnonzero(inside).astype("i4"),
        faces=faces_in_bounds(mesh, lo_arr, hi_arr),
    )


QUERIES: dict[str, Query] = {
    "material": Query(
        name="material",
        modes=("face",),
        args=("slot",),
        run=_q_material,
        hint="Every face painted with the given palette slot.",
    ),
    "normal": Query(
        name="normal",
        modes=("face",),
        args=("direction", "max_angle"),
        run=_q_normal,
        hint="Every face whose normal points within max_angle degrees of "
        "direction -- 'the upward-facing faces', read out of a scene report.",
    ),
    "bounds": Query(
        name="bounds",
        modes=("face", "vertex"),
        args=("min", "max", "space"),
        run=_q_bounds,
        hint="Every vertex, and every face wholly, inside an axis-aligned box.",
    ),
}
