"""Selection verbs: loops, rings, linked, grow, shrink, boundary, mirror.

Every one of these is a thing a modeller does dozens of times an hour and none
of them existed. Clay could select an element and add another with Shift, and
that was the whole vocabulary -- so selecting the ring of edges round a cylinder
meant clicking each of them, and selecting one of two objects welded into one
mesh was not possible at all.

**Pure, and over the adjacency.** Nothing here touches a document, a selection
object or a view: each takes a mesh and a set of indices and returns a set of
indices, which is what lets "does a loop stop at a pole" be a plain assertion.
``ops_bevel`` already walks a quad strip for its own purposes and this shares
that walk rather than writing a second one -- an edge ring and a bevel's strip
are the same traversal, and two of them would disagree the first time either
was fixed.

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
from .adjacency import Adjacency, adjacency
from .mesh import Mesh


def _arity(mesh: Mesh) -> np.ndarray:
    """How many corners each face has."""
    starts = np.asarray(mesh.starts, dtype="i8")
    return (starts[1:] - starts[:-1]).astype("i8")


def _pairs(edge_ids: np.ndarray, a: Adjacency) -> np.ndarray:
    """Edge ids back to the vertex pairs a selection stores."""
    ids = np.asarray(edge_ids, dtype="i8").reshape(-1)
    if not len(ids):
        return np.zeros((0, 2), dtype="i4")
    ids = np.unique(ids[(ids >= 0) & (ids < a.n_edges)])
    return a.edge_verts[ids].astype("i4")


# --- loops and rings ----------------------------------------------------------


def quad_strip(mesh: Mesh, a: Adjacency, corner: int) -> tuple[list[int], list[int]]:
    """The strip of quads reached from ``corner``. -> ``(faces, edge ids)``.

    The traversal an edge ring and a face loop are both made of, and the one
    ``ops_bevel`` already does for its own cuts: step across a quad to the
    opposite edge, through its twin, and on until a triangle, a boundary or the
    starting face stops it.

    Written here rather than imported from ``ops_bevel`` because that one also
    records the endpoint each cut is measured from -- a bevel needs to know
    which way along an edge it is going and a selection does not -- and a
    function that returns what half its callers throw away is a function two
    callers are reading differently.
    """
    arity = _arity(mesh)
    faces: list[int] = []
    edges: list[int] = []
    seen: set[int] = set()
    at = int(corner)
    while at >= 0:
        face = int(a.corner_face[at])
        if face in seen or arity[face] != 4:
            break
        seen.add(face)
        faces.append(face)
        edges.append(int(a.corner_edge[at]))
        exit_corner = int(a.next_corner[a.next_corner[at]])
        edges.append(int(a.corner_edge[exit_corner]))
        at = int(a.twin[exit_corner])
    return faces, edges


def _corners_of_edge(a: Adjacency, edge: int) -> list[int]:
    """Every corner that leaves along ``edge``. One or two on a sane mesh."""
    return [int(c) for c in np.flatnonzero(a.corner_edge == int(edge))]


def edge_ring(mesh: Mesh, edge: tuple[int, int]) -> np.ndarray:
    """Every edge parallel to ``edge`` across the quad strip. -> vertex pairs.

    The ring, not the loop: a cylinder's ring is the band of edges running
    *round* it, each one the far side of a quad from the last. Both directions
    from the seed, so a seed in the middle of a strip reaches both ends.
    """
    a = adjacency(mesh)
    ids = a.edge_ids(np.asarray([edge], dtype="i4"))
    if not len(ids) or ids[0] < 0:
        return np.zeros((0, 2), dtype="i4")
    found: set[int] = {int(ids[0])}
    for corner in _corners_of_edge(a, int(ids[0])):
        _faces, walked = quad_strip(mesh, a, corner)
        found.update(walked)
    return _pairs(np.fromiter(found, dtype="i8", count=len(found)), a)


def edge_loop(mesh: Mesh, edge: tuple[int, int]) -> np.ndarray:
    """Every edge continuing ``edge`` end to end. -> vertex pairs.

    The loop, not the ring: it runs *along* the seed rather than across it, and
    it is what Alt+click gives in every modelling package.

    The rule at each vertex is the one that makes a loop stop where a modeller
    expects it to. Continue through a vertex of **exactly four edges** to the
    edge opposite the one arrived on; stop at anything else. A pole -- the tip
    of a cone, the centre of a fan -- has some other number, and a loop that
    ran through one would wander off round the mesh.
    """
    a = adjacency(mesh)
    ids = a.edge_ids(np.asarray([edge], dtype="i4"))
    if not len(ids) or ids[0] < 0:
        return np.zeros((0, 2), dtype="i4")
    seed = int(ids[0])
    found: set[int] = {seed}
    for end in (0, 1):
        current = seed
        vertex = int(a.edge_verts[seed][end])
        while True:
            nxt = _opposite_edge(a, vertex, current)
            if nxt is None or nxt in found:
                break
            found.add(nxt)
            pair = a.edge_verts[nxt]
            vertex = int(pair[1]) if int(pair[0]) == vertex else int(pair[0])
            current = nxt
    return _pairs(np.fromiter(found, dtype="i8", count=len(found)), a)


def _opposite_edge(a: Adjacency, vertex: int, edge: int) -> int | None:
    """The edge across ``vertex`` from ``edge``, or None at anything but a
    four-edge vertex. See :func:`edge_loop` for why four."""
    corners = a.vertex_corners(int(vertex))
    around: set[int] = set()
    for corner in corners:
        around.add(int(a.corner_edge[corner]))
        around.add(int(a.corner_edge[a.prev_corner[corner]]))
    if len(around) != 4 or int(edge) not in around:
        return None
    # The two edges of the face the seed is in are its neighbours; the fourth
    # is the one opposite. Two of the four share a face with the seed at this
    # vertex, and the remaining one is the answer.
    neighbours: set[int] = set()
    for corner in corners:
        pair = {int(a.corner_edge[corner]), int(a.corner_edge[a.prev_corner[corner]])}
        if int(edge) in pair:
            neighbours |= pair
    rest = around - neighbours
    return int(next(iter(rest))) if len(rest) == 1 else None


def face_loop(mesh: Mesh, face: int) -> np.ndarray:
    """The strip of faces running through ``face``. -> face indices.

    Both directions, which for a quad means both of its two strips: a face sits
    on two loops at right angles and picking one arbitrarily would make the
    verb's result depend on corner order rather than on anything the user can
    see. Both is the honest answer and is what Blender's face loop gives from a
    face rather than from an edge.
    """
    a = adjacency(mesh)
    face = int(face)
    if not (0 <= face < len(mesh.starts) - 1) or _arity(mesh)[face] != 4:
        return np.zeros(0, dtype="i4")
    found: set[int] = {face}
    start = int(mesh.starts[face])
    for offset in (0, 1):
        for corner in (start + offset, int(a.next_corner[a.next_corner[start + offset]])):
            faces, _edges = quad_strip(mesh, a, corner)
            found.update(faces)
    return np.array(sorted(found), dtype="i4")


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


# --- more, less, and the edge of it -------------------------------------------


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
        inside[lo[inside[hi]]] = True
        inside[hi[inside[lo]]] = True
    return np.flatnonzero(inside).astype("i4")


def shrink(mesh: Mesh, verts: np.ndarray) -> np.ndarray:
    """One ring inward: every selected vertex whose neighbours are all selected.

    The exact inverse of :func:`grow` only on an infinite lattice, and this is
    the definition that matters rather than the symmetry: shrinking peels the
    *boundary* off a selection, which is what a user reaching for it wants --
    "the middle of what I have", not "whatever grow would undo".
    """
    seeds = np.unique(np.asarray(verts, dtype="i8").reshape(-1))
    count = len(mesh.positions)
    if not len(seeds) or count == 0:
        return np.zeros(0, dtype="i4")
    a = adjacency(mesh)
    inside = np.zeros(count, dtype=bool)
    inside[seeds[(seeds >= 0) & (seeds < count)]] = True
    if a.n_edges == 0:
        return np.flatnonzero(inside).astype("i4")
    lo = a.edge_verts[:, 0].astype("i8")
    hi = a.edge_verts[:, 1].astype("i8")
    # A vertex is peeled when it has a neighbour outside the selection **or**
    # it sits on the mesh's own border.
    #
    # The second half is not a refinement, it is the case the verb is mostly
    # used in: Select Less over a fully selected grid has no unselected
    # neighbour anywhere, so without it the answer is "everything" and the key
    # appears to do nothing. A closed solid has no border and so is unchanged,
    # which is also right -- there is nothing to peel off a cube.
    edge_of = np.zeros(count, dtype=bool)
    edge_of[lo[~inside[hi]]] = True
    edge_of[hi[~inside[lo]]] = True
    border = a.edge_uses == 1
    if bool(border.any()):
        edge_of[lo[border]] = True
        edge_of[hi[border]] = True
    return np.flatnonzero(inside & ~edge_of).astype("i4")


def boundary(mesh: Mesh) -> np.ndarray:
    """Every edge with exactly one face on it. -> vertex pairs.

    The mesh's open border, which is what a hole is and what a Fill Hole is
    about to act on -- so selecting it is how you look at what you are about to
    close.
    """
    a = adjacency(mesh)
    if a.n_edges == 0:
        return np.zeros((0, 2), dtype="i4")
    return _pairs(np.flatnonzero(a.edge_uses == 1), a)


def by_material(mesh: Mesh, slot: int) -> np.ndarray:
    """Every face using palette slot ``slot``. -> face indices.

    The one selection verb that is about the *document* rather than the
    topology, and it earns its place for the reason a material slot exists at
    all: "show me everything painted with this" is how a slot gets reassigned,
    and there was no way to ask.
    """
    material = np.asarray(mesh.material, dtype="i8")
    return np.flatnonzero(material == int(slot)).astype("i4")


# --- symmetry -----------------------------------------------------------------


def mirror_pairs(mesh: Mesh, axis: int = 0, eps: float = 1e-4) -> dict[int, int]:
    """``{vertex: its mirror}`` across the plane ``axis == 0``.

    What X-mirror editing needs and what it can only be as good as: a mesh that
    is not actually symmetric has no pairs to find, and this reports the ones it
    can rather than pretending. A vertex *on* the plane maps to itself, which is
    the case that has to be handled rather than excluded -- those are the ones a
    mirrored drag must slide along the plane instead of moving off it.

    ``eps`` is a distance in the mesh's own units. Bucketed on the rounded
    coordinate rather than compared pairwise, because pairwise is O(V^2) and a
    50k-vertex import would take minutes.
    """
    positions = np.asarray(mesh.positions, dtype="f8")
    if not len(positions):
        return {}
    axis = int(axis)
    quantum = max(float(eps), 1e-9)
    mirrored = positions.copy()
    mirrored[:, axis] *= -1.0
    keys = np.round(positions / quantum).astype("i8")
    wanted = np.round(mirrored / quantum).astype("i8")
    lookup: dict[tuple[int, int, int], int] = {}
    for index, row in enumerate(keys):
        lookup.setdefault((int(row[0]), int(row[1]), int(row[2])), index)
    out: dict[int, int] = {}
    for index, row in enumerate(wanted):
        twin = lookup.get((int(row[0]), int(row[1]), int(row[2])))
        if twin is not None:
            out[index] = int(twin)
    return out


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

    The common currency: growing a face selection and growing a vertex one are
    the same walk over the same graph, and converting once here is what keeps
    the rest of this module free of a mode argument.
    """
    if mode == "vertex":
        return np.asarray(sel.verts, dtype="i8")
    if mode == "edge":
        return np.unique(np.asarray(sel.edges, dtype="i8").reshape(-1))
    faces = np.asarray(sel.faces, dtype="i8")
    if not len(faces):
        return np.zeros(0, dtype="i8")
    starts = np.asarray(mesh.starts, dtype="i8")
    loops = np.asarray(mesh.loops, dtype="i8")
    out = [loops[starts[f] : starts[f + 1]] for f in faces if 0 <= f < len(starts) - 1]
    return np.unique(np.concatenate(out)) if out else np.zeros(0, dtype="i8")


def sel_from_verts(mesh: Mesh, verts: np.ndarray, mode: str) -> el.ElementSel:
    """A vertex set back into the mode's own currency.

    An edge or a face is included when **every** one of its vertices is, which
    is the only definition that makes grow and shrink inverses of each other on
    the inside of a selection: "partly selected" is not a state an element
    selection can be in.
    """
    verts = np.unique(np.asarray(verts, dtype="i8"))
    if mode == "vertex":
        return el.ElementSel(verts=verts)
    inside = np.zeros(len(mesh.positions), dtype=bool)
    inside[verts[(verts >= 0) & (verts < len(inside))]] = True
    if mode == "face":
        # ``elements._face_corner_mask``, which is this question vectorised and
        # is the same definition ``convert`` uses to go *up* a level. This had
        # a Python loop over every face of the mesh -- so Select More on a
        # 200k-face sculpt walked all of them per press, for an answer numpy
        # already had -- and, worse, a second spelling of "a face is selected
        # only when all of its corners are", which is the rule those two verbs
        # rest on being inverses of each other.
        mask = el._face_corner_mask(mesh, verts)
        return el.ElementSel(faces=np.flatnonzero(mask).astype("i4"))
    a = adjacency(mesh)
    if a.n_edges == 0:
        return el.ElementSel()
    both = inside[a.edge_verts[:, 0]] & inside[a.edge_verts[:, 1]]
    return el.ElementSel(edges=a.edge_verts[both])


# --- selecting by a property of the geometry, not by a walk over it ----------
#
# ``edge_loop``, ``edge_ring``, ``face_loop``, ``linked``, ``grow``, ``shrink``
# and ``boundary`` above all walk the adjacency graph outward from a seed. The
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


# --- similar: a property of a seed set, matched across the whole mesh --------
#
# Tranche 5's "select similar" verbs. Every other verb in this module answers
# from a *walk* (a loop, a ring, everything linked) or from an *absolute*
# question (which slot, which direction, which box); these answer "what else
# is like *this*", where "this" is a caller-supplied seed set -- in practice
# the object's own current selection, handed in by whichever caller wires the
# query up, exactly as ``by_material``'s ``slot`` or ``faces_by_normal``'s
# ``direction`` are handed in rather than read off a document this module has
# no notion of (see the module docstring: "nothing here touches a document, a
# selection object or a view"). A seed set of several elements compares
# against the *mean* of their own property, a single deterministic number
# rather than a per-seed nearest-match search -- simpler, and it is what makes
# two calls with the same seed set produce the same answer regardless of which
# member happens to be "active".


def similar_area(mesh: Mesh, faces: Sequence[int], tolerance: float = 0.1) -> np.ndarray:
    """Faces whose area is within ``tolerance`` of the seed faces' own mean
    area, as a fraction of that mean. -> face indices, the seed's own faces
    included (they trivially match themselves).
    """
    n_faces = bm.face_count(mesh)
    seed = np.unique(np.asarray(faces, dtype="i8").reshape(-1))
    seed = seed[(seed >= 0) & (seed < n_faces)]
    if n_faces == 0 or len(seed) == 0:
        return np.zeros(0, dtype="i4")
    areas = 0.5 * np.linalg.norm(bm.face_normals(mesh), axis=1)
    ref = float(areas[seed].mean())
    if ref <= 1e-12:
        return np.flatnonzero(areas <= 1e-12).astype("i4")
    within = np.abs(areas - ref) / ref <= float(tolerance)
    return np.flatnonzero(within).astype("i4")


def similar_normal(mesh: Mesh, faces: Sequence[int], tolerance: float = 5.0) -> np.ndarray:
    """Faces whose normal points within ``tolerance`` degrees of the seed
    faces' own mean unit normal. -> face indices.

    Built on :func:`faces_by_normal`, so a seed whose own normals cancel to
    (near) zero -- two seed faces pointing opposite ways -- matches nothing,
    the same degenerate-direction behaviour that function's own docstring
    states rather than hides.
    """
    n_faces = bm.face_count(mesh)
    seed = np.unique(np.asarray(faces, dtype="i8").reshape(-1))
    seed = seed[(seed >= 0) & (seed < n_faces)]
    if n_faces == 0 or len(seed) == 0:
        return np.zeros(0, dtype="i4")
    normals = np.asarray(bm.face_normals(mesh), dtype="f8")
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    unit = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)
    mean = unit[seed].mean(axis=0)
    return faces_by_normal(mesh, mean, max_angle=tolerance)


def similar_material(mesh: Mesh, faces: Sequence[int]) -> np.ndarray:
    """Every face sharing a material slot with any of the seed faces. -> face
    indices. No tolerance: a slot is a discrete index, and "close to slot 3"
    has no meaning ``by_material`` does not already give it directly.
    """
    n_faces = bm.face_count(mesh)
    seed = np.unique(np.asarray(faces, dtype="i8").reshape(-1))
    seed = seed[(seed >= 0) & (seed < n_faces)]
    if n_faces == 0 or len(seed) == 0:
        return np.zeros(0, dtype="i4")
    slots = np.unique(np.asarray(mesh.material, dtype="i8")[seed])
    return np.flatnonzero(np.isin(mesh.material, slots)).astype("i4")


def similar_sides(mesh: Mesh, faces: Sequence[int], tolerance: int = 0) -> np.ndarray:
    """Faces whose corner count is within ``tolerance`` of *any* arity
    present in the seed faces. -> face indices.

    Bucketed on the seed's own *set* of arities rather than their mean --
    side count is discrete, and a seed selection spanning a triangle and a
    quad has two legitimate reference counts, not one fractional one.
    """
    n_faces = bm.face_count(mesh)
    seed = np.unique(np.asarray(faces, dtype="i8").reshape(-1))
    seed = seed[(seed >= 0) & (seed < n_faces)]
    if n_faces == 0 or len(seed) == 0:
        return np.zeros(0, dtype="i4")
    counts = _arity(mesh)
    ref_counts = np.unique(counts[seed])
    tol = abs(int(tolerance))
    diffs = np.abs(counts[:, None] - ref_counts[None, :])
    within = (diffs <= tol).any(axis=1)
    return np.flatnonzero(within).astype("i4")


def similar_length(mesh: Mesh, edges: np.ndarray, tolerance: float = 0.1) -> np.ndarray:
    """Every mesh edge whose length is within ``tolerance`` of the seed
    edges' own mean length, as a fraction of that mean. -> vertex pairs.
    """
    a = adjacency(mesh)
    raw = np.asarray(edges, dtype="i4")
    seed_pairs = raw.reshape(-1, 2) if raw.size else np.zeros((0, 2), dtype="i4")
    ids = a.edge_ids(seed_pairs) if len(seed_pairs) else np.zeros(0, dtype="i4")
    ids = ids[ids >= 0]
    if a.n_edges == 0 or len(ids) == 0:
        return np.zeros((0, 2), dtype="i4")
    ends0 = mesh.positions[a.edge_verts[:, 0]].astype("f8")
    ends1 = mesh.positions[a.edge_verts[:, 1]].astype("f8")
    lengths = np.linalg.norm(ends0 - ends1, axis=1)
    ref = float(lengths[ids].mean())
    if ref <= 1e-12:
        return _pairs(np.flatnonzero(lengths <= 1e-12), a)
    within = np.abs(lengths - ref) / ref <= float(tolerance)
    return _pairs(np.flatnonzero(within), a)


def similar_valence(mesh: Mesh, verts: Sequence[int], tolerance: int = 0) -> np.ndarray:
    """Every vertex whose incident-edge count is within ``tolerance`` of the
    seed vertices' own set of valences. -> vertex indices.
    """
    n_verts = len(mesh.positions)
    seed = np.unique(np.asarray(verts, dtype="i8").reshape(-1))
    seed = seed[(seed >= 0) & (seed < n_verts)]
    if n_verts == 0 or len(seed) == 0:
        return np.zeros(0, dtype="i4")
    a = adjacency(mesh)
    valence = (
        np.bincount(a.edge_verts.reshape(-1), minlength=n_verts)
        if a.n_edges
        else np.zeros(n_verts, dtype="i8")
    )
    ref_valences = np.unique(valence[seed])
    tol = abs(int(tolerance))
    diffs = np.abs(valence[:, None].astype("i8") - ref_valences[None, :].astype("i8"))
    within = (diffs <= tol).any(axis=1)
    return np.flatnonzero(within).astype("i4")


# --- QUERIES: the fourth derived registry -------------------------------------
#
# ``OPS`` (``studio/modes/clay/ops.py``) is invocable verbs and the agent's derived tool
# list is built from it, never hand-listed -- an invariant, test-gated in both
# directions. A query is a different shape from a verb: it does not *change*
# the selection in place, it *answers* one from a seed or a set of parameters
# an agent read out of a scene report ("the edge at these two vertices", "slot
# 3", "faces facing up"), which is a request ``OPS``'s ``run(ctx, doc,
# **params)`` shape -- built to mutate ``doc.element_sel`` for the object
# already on screen -- has no natural place for. Writing that vocabulary out by
# hand in the agent surface would be a *fifth* place to remember it exists (the
# menu, the tools pane, the key handler and ``OPS`` already being four ways
# ``studio/modes/clay/ops.py`` used to answer "is X available", before this registry
# collapsed them to one) -- so it lives here, next to the verbs it is built
# from, and the agent surface derives its tool list from :data:`QUERIES` the
# same way it already derives one from ``primitives.GENERATORS``,
# ``presets.ASSEMBLIES`` and ``clay_ops.OPS``.
#
# **Deliberately six entries, not thirteen.** ``all``, ``none``, ``invert``,
# ``linked``, ``more``, ``less`` and ``boundary`` are not here, because they
# already exist as ``select-*`` rows in ``clay_ops.OPS`` and are already in the
# agent's derived ``clay_op`` enum -- dead only because no element mode can be
# set from an agent yet, which is a wiring gap the agent surface closes later,
# not a reason to open a second door for a verb that already has one. A
# ``QUERIES`` entry named ``linked`` would be exactly the drift this codebase's
# "one list, not three" rule (see ``studio/modes/clay/ops.py``'s own module docstring)
# exists to prevent, one file over.
#
# Every entry here is seeded or parametric -- it takes something an agent
# supplies (an edge, a face, a material slot, a direction, a box) and answers
# a *fresh* selection from it, which is the shape ``all``/``none``/``invert``
# and the rest do not have: they act on whatever is *already* selected, which
# only makes sense once a selection exists to act on.
#
# ``args`` names the vocabulary the eventual JSON-schema layer offers for each
# query -- not necessarily ``run``'s own keyword names. ``bounds``'s ``space``
# is the clearest case: whether the box an agent supplies is in world or local
# coordinates is a decision the *agent surface* makes, converting through
# :func:`~.ops.world_positions` before ever calling in here, because this
# module has no opinion about anything but the mesh's own local space (see
# :func:`faces_in_bounds`). Keeping that mapping out of this file is what
# keeps this module -- and the rest of this pure, headless package -- free of
# JSON-schema knowledge.


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


def _q_loop(mesh: Mesh, edge: Sequence[int]) -> el.ElementSel:
    return el.ElementSel(edges=edge_loop(mesh, edge))


def _q_ring(mesh: Mesh, edge: Sequence[int]) -> el.ElementSel:
    return el.ElementSel(edges=edge_ring(mesh, edge))


def _q_face_loop(mesh: Mesh, face: int) -> el.ElementSel:
    return el.ElementSel(faces=face_loop(mesh, face))


def _q_material(mesh: Mesh, slot: int) -> el.ElementSel:
    return el.ElementSel(faces=by_material(mesh, slot))


def _q_normal(mesh: Mesh, direction: Sequence[float], max_angle: float = 45.0) -> el.ElementSel:
    return el.ElementSel(faces=faces_by_normal(mesh, direction, max_angle))


def _q_similar_area(mesh: Mesh, faces: Sequence[int], tolerance: float = 0.1) -> el.ElementSel:
    return el.ElementSel(faces=similar_area(mesh, faces, tolerance))


def _q_similar_normal(mesh: Mesh, faces: Sequence[int], tolerance: float = 5.0) -> el.ElementSel:
    return el.ElementSel(faces=similar_normal(mesh, faces, tolerance))


def _q_similar_material(mesh: Mesh, faces: Sequence[int]) -> el.ElementSel:
    return el.ElementSel(faces=similar_material(mesh, faces))


def _q_similar_sides(mesh: Mesh, faces: Sequence[int], tolerance: int = 0) -> el.ElementSel:
    return el.ElementSel(faces=similar_sides(mesh, faces, tolerance))


def _q_similar_length(mesh: Mesh, edges: Sequence[int], tolerance: float = 0.1) -> el.ElementSel:
    return el.ElementSel(edges=similar_length(mesh, np.asarray(edges), tolerance))


def _q_similar_valence(mesh: Mesh, verts: Sequence[int], tolerance: int = 0) -> el.ElementSel:
    return el.ElementSel(verts=similar_valence(mesh, verts, tolerance))


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
    "loop": Query(
        name="loop",
        modes=("edge",),
        args=("edge",),
        run=_q_loop,
        hint="Every edge continuing the seed end to end -- an Alt+click loop, "
        "found from two vertex indices instead of a mouse position.",
    ),
    "ring": Query(
        name="ring",
        modes=("edge",),
        args=("edge",),
        run=_q_ring,
        hint="Every edge parallel to the seed across the quad strip -- the "
        "band running *round* a cylinder rather than along it.",
    ),
    "face_loop": Query(
        name="face_loop",
        modes=("face",),
        args=("face",),
        run=_q_face_loop,
        hint="The strip of faces running through the seed face, both directions.",
    ),
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
    # Tranche 5's "select similar" family -- see the "similar" section above
    # this dict's own module for why each takes a seed *set* (in practice the
    # object's current selection) rather than one index the way loop/ring/
    # face_loop/material do.
    "similar_area": Query(
        name="similar_area",
        modes=("face",),
        args=("faces", "tolerance"),
        run=_q_similar_area,
        hint="Faces whose area is within tolerance of the seed faces' own "
        "mean area, as a fraction of it.",
    ),
    "similar_normal": Query(
        name="similar_normal",
        modes=("face",),
        args=("faces", "tolerance"),
        run=_q_similar_normal,
        hint="Faces whose normal points within tolerance degrees of the "
        "seed faces' own mean normal.",
    ),
    "similar_material": Query(
        name="similar_material",
        modes=("face",),
        args=("faces",),
        run=_q_similar_material,
        hint="Every face sharing a material slot with any seed face.",
    ),
    "similar_sides": Query(
        name="similar_sides",
        modes=("face",),
        args=("faces", "tolerance"),
        run=_q_similar_sides,
        hint="Faces whose corner count is within tolerance of any arity "
        "present in the seed faces.",
    ),
    "similar_length": Query(
        name="similar_length",
        modes=("edge",),
        args=("edges", "tolerance"),
        run=_q_similar_length,
        hint="Edges whose length is within tolerance of the seed edges' own "
        "mean length, as a fraction of it.",
    ),
    "similar_valence": Query(
        name="similar_valence",
        modes=("vertex",),
        args=("verts", "tolerance"),
        run=_q_similar_valence,
        hint="Vertices whose incident-edge count is within tolerance of any "
        "valence present in the seed vertices.",
    ),
}
