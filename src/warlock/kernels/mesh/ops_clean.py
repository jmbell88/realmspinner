"""Repair verbs for an imported or hand-authored mesh: measure, then fix.

:mod:`.adjacency`'s ``check_manifold`` and :mod:`.diagnose` already answer "what
is wrong" for a panel to draw; this module answers "how wrong, as one number
per defect" (:func:`survey`, returning a :class:`Survey`) and "make it not
wrong" (six individual ops plus :func:`clean`, which runs a fixed sequence of
them). Both halves exist because they serve two different callers: a
properties-panel badge or an agent's `clay_diagnose` wants counts without
touching the mesh, and a **Clean** button or a `clay_op` call wants the mesh
actually fixed -- and the two must agree on what they are counting, which is
why every op's own defect count is computed the same way :func:`survey`
computes it rather than being re-derived independently.

**Every op here is `Mesh -> Mesh`, and every op is a no-op when there is
nothing to fix** -- returning the exact same object, not an equal one. The
viewport's GPU cache and the undo mechanism both key on `id(mesh)` (see
`mesh.py`'s own docstring), so a "clean" that rebuilt an already-clean mesh
byte-for-byte would still evict every cached buffer and push an undo step for
an edit that changed nothing. Every internal helper below therefore returns
`(mesh_or_same_object, count_fixed)`, and the public op is a thin wrapper that
drops the count; :func:`clean` keeps the count to build its
:class:`CleanReport` without measuring anything twice.

**What each op reuses rather than reimplements.** :func:`merge_by_distance`
calls :func:`~.ops_topo.weld` for the actual merge -- it does not walk vertices
itself -- but a *count* of how many vertices would merge is wanted whether or
not the merge is worth paying for (`survey`'s `coincident_vertices`, and
`merge_by_distance`'s own identity check), and `weld` has no way to answer that
without doing the merge. Rather than a second clustering implementation, both
call `ops_topo._clusters` directly -- the module-private helper `weld` itself
is built from -- and say so here once: it is not `merge_by_distance` or
`survey` under a different name, it is the one clustering `weld` already does,
asked for a count as well as a mesh. :func:`fill_all_holes` calls
:func:`~.ops_topo.fill_hole` with every boundary edge in the mesh at once
rather than looping per ring, because `fill_hole` already resolves the seed
edges it is given into whichever distinct boundary rings they touch
(`adjacency.boundary_ring_from`) and caps each one -- one call therefore fills
every hole in the mesh, refusing (as `fill_hole` already does) on a pinched
vertex, a self-crossing ring, or a ring past
`ops_dissolve.MAX_DISSOLVED_RING`. :func:`recalc_outside` calls
:func:`~.ops_topo.flip_normals` to actually reverse the faces it decides need
reversing, rather than permuting corners itself a second time.

**Winding consistency is a BFS over manifold edges, and it is the one
Python-level loop in this module.** A face's winding only means anything
relative to its neighbours: two faces sharing an edge are "consistent" when
they traverse it in opposite directions (`adjacency.twin`) and "inconsistent"
when they traverse it the same way (`adjacency.flipped_pairs`). Propagating
that relation with a breadth-first walk from one face per connected shell
assigns every face a boolean relative to its shell's own arbitrary starting
point; the *minority* boolean in each shell is what disagrees with the
majority, which is exactly :class:`Survey`'s `flipped_faces` and what
`recalc_outside` corrects first. A closed shell is then oriented outward by
the sign of its own volume (the divergence theorem over each face's own fan
triangulation, which needs no earclip-grade concavity handling because a fan
sum from a point on the polygon's own boundary is the correct signed
contribution for *any* simple polygon, convex or not -- the same identity the
shoelace formula rests on); an open shell has no volume to sign, so its global
orientation is instead the majority of its own face normals pointing away from
its centroid, the closest an open sheet has to "outward". The walk is over
faces of one shell with a queue, never a face pair matched against every other
face, so it is linear in the mesh rather than quadratic in it.

**MAX_CLEAN_CORNERS**, below, is the ceiling every Clay walking op carries
(`ops_dissolve.MAX_DISSOLVED_RING`, `ops_topo.MAX_BRIDGED_RING`,
`ops_topo.MAX_COLLAPSED_PAIRS`, `ops_bevel.MAX_BEVELED_CORNERS`...), chosen the
same way theirs were: measured from what the operation actually grows, with a
margin under the point a single press stops being well under a second.
`clean()`'s own BFS is O(corners), same order as the rest of the pipeline
(`weld`, `compact_vertices`, `fill_hole`'s own ceiling), so the ceiling here is
sized off `clean()`'s own wall-clock, not off the BFS in isolation. Measured on
this machine, `uv_sphere` at increasing resolution (as `primitives.uv_sphere`
actually builds it -- every primitive it ships carries a `uv` array, so this
is already the UV-bearing case), default params (`fill_holes=False`,
`recalc=True`, so the BFS and the divergence-theorem volume pass both run):

| corners | faces   | clean() |
|--------:|--------:|--------:|
|  32,512 |   8,192 |   35 ms |
|  73,344 |  18,432 |   80 ms |
| 130,560 |  32,768 |  155 ms |
| 204,160 |  51,200 |  245 ms |
| 319,200 |  80,000 |  378 ms |

...and the same shapes with `uv` stripped to `None` measure within noise of
these (35 / 76 / 154 / 245 / 378 ms) -- unlike `ops_bevel.MAX_BEVELED_CORNERS`
(dev/INVARIANTS.md: "A Clay ceiling measured on a mesh without UVs is halved
for a mesh that carries them"), this module never does a per-corner float
computation whose cost UVs would double; every step here that touches `uv` at
all (`merge_by_distance`'s `weld`, the two `remove_*` compactions,
`recalc_outside`'s `flip_normals`) only carries it along an existing
`np.concatenate`/`np.insert`/fancy-index, a cost dwarfed by the adjacency
build, the SciPy `cKDTree`/`connected_components` calls and the BFS that
dominate every one of these runs regardless of `uv`. **A ceiling is a
measurement of one shape of input, and this one measured no UV multiplier
worth giving its own factor** -- one constant serves both cases here, rather
than a halving copied from a different op's different cost shape.

A single closed sphere understates the worst case, though: `MAX_CLEAN_CORNERS`
exists for `recalc_outside`'s outer `for seed in range(n_faces)` restarting
its BFS once per shell, and an imported mesh with many small separate shells
(a decal, a kitbash, a scene flattened to one mesh) pays that restart far more
often than one sphere does. Measured on many small boxes concatenated into one
mesh (`n_faces = 300 * n`, one closed shell each), the shape this ceiling
actually has to answer for:

| boxes  | corners   | faces   | clean() |
|-------:|----------:|--------:|--------:|
|  2,000 |    48,000 |  12,000 |   56 ms |
|  8,000 |   192,000 |  48,000 |  247 ms |
| 20,000 |   480,000 | 120,000 |  636 ms |
| 40,000 |   960,000 | 240,000 | 1306 ms |

...again within noise between UV-bearing and UV-less. Growth is linear
throughout; 480,000 corners (20,000 boxes) is the last point still comfortably
under a second, and 960,000 crosses it. `MAX_CLEAN_CORNERS` is set well below
that crossing -- the same margin `MAX_BRIDGED_RING` keeps under its own
measured stall point -- rather than at it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from . import ops_topo, topo
from .adjacency import adjacency, boundary_loops
from .elements import ElementSel, OpError
from .mesh import Mesh, face_count, face_normals

__all__ = [
    "CleanReport",
    "FaceDefectMasks",
    "Survey",
    "clean",
    "face_defect_masks",
    "fill_all_holes",
    "merge_by_distance",
    "recalc_outside",
    "remove_degenerate",
    "remove_duplicate_faces",
    "remove_loose_vertices",
    "survey",
]

#: Faces below this area (positions cast to f8, the same precision
#: `face_normals` already works in) count as degenerate regardless of how many
#: distinct vertices they have -- three distinct but collinear points make a
#: triangle with zero area, which "distinct vertices >= 3" alone would miss.
#: f4 positions round-trip to f8 with error far below this, so a face this
#: small in a metre-scale document is a construction defect, not a rounding
#: artefact of the storage format.
AREA_EPS = 1e-10

#: See the module docstring's measurement tables -- one constant for both a
#: UV-less and a UV-bearing mesh, since neither table found a UV multiplier
#: here worth a separate factor.
MAX_CLEAN_CORNERS = 300_000


def _refuse_if_too_large(mesh: Mesh) -> None:
    if len(mesh.loops) > MAX_CLEAN_CORNERS:
        raise OpError(
            f"This mesh has {len(mesh.loops):,} corners, past the "
            f"{MAX_CLEAN_CORNERS:,} Clean can process without stalling. Clean "
            "a smaller selection, or split the mesh first."
        )


# --- shared measurement: faces --------------------------------------------


def _face_of_corner(mesh: Mesh) -> np.ndarray:
    counts = np.diff(mesh.starts.astype("i8"))
    return np.repeat(np.arange(len(counts), dtype="i8"), counts)


def _distinct_counts(mesh: Mesh) -> np.ndarray:
    """Distinct vertex count per face, vectorised via a per-face lexsort.

    The same "sort corners within their own face" trick
    `adjacency.check_manifold` uses for `repeated_corner_faces`, generalised
    from "does any vertex repeat" to "how many distinct vertices are there",
    which is what a face with a repeat *twice* over (a five-corner face
    reusing one vertex three times) needs to be told apart from one reusing
    it only once.
    """
    n_faces = face_count(mesh)
    if n_faces == 0:
        return np.zeros(0, dtype="i8")
    loops = mesh.loops.astype("i8")
    face_of = _face_of_corner(mesh)
    order = np.lexsort((loops, face_of))
    sorted_loops = loops[order]
    sorted_face = face_of[order]
    is_new = np.ones(len(sorted_loops), dtype=bool)
    if len(sorted_loops) > 1:
        same_face = sorted_face[1:] == sorted_face[:-1]
        is_new[1:] = ~(same_face & (sorted_loops[1:] == sorted_loops[:-1]))
    return np.bincount(sorted_face, weights=is_new.astype("i8"), minlength=n_faces).astype("i8")


def _face_areas(mesh: Mesh) -> np.ndarray:
    """Each face's own area -- half the magnitude of its Newell normal."""
    if face_count(mesh) == 0:
        return np.zeros(0, dtype="f8")
    return np.linalg.norm(face_normals(mesh), axis=1) / 2.0


def _degenerate_mask(mesh: Mesh) -> np.ndarray:
    """A face with fewer than three distinct vertices, or effectively no area."""
    if face_count(mesh) == 0:
        return np.zeros(0, dtype=bool)
    return (_distinct_counts(mesh) < 3) | (_face_areas(mesh) < AREA_EPS)


def _duplicate_mask(mesh: Mesh) -> np.ndarray:
    """True for the *later* face of each vertex-set repeat, arity by arity.

    Same row-grouping `adjacency.check_manifold`'s own `duplicate_faces` uses
    (sort each face's corners, then `np.unique` the fixed-width rows within one
    arity), but that report names every member of a duplicate group; the
    contract here is narrower -- "a face whose vertex set equals an *earlier*
    face's", so only the later member(s) of each group count, matching the
    face a `remove_duplicate_faces` pass would actually delete.
    """
    n_faces = face_count(mesh)
    out = np.zeros(n_faces, dtype=bool)
    if n_faces == 0:
        return out
    loops = mesh.loops.astype("i8")
    starts = mesh.starts.astype("i8")
    counts = np.diff(starts)
    face_of = np.repeat(np.arange(n_faces, dtype="i8"), counts)
    order = np.lexsort((loops, face_of))
    sorted_loops = loops[order]
    offs = np.concatenate([[0], np.cumsum(counts)])[:-1]
    for arity in np.unique(counts).tolist():
        arity = int(arity)
        which = np.flatnonzero(counts == arity)
        if arity == 0 or len(which) < 2:
            continue
        rows = sorted_loops[offs[which][:, None] + np.arange(arity, dtype="i8")[None, :]]
        _, inv, _cnt = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1)
        # ``which`` is ascending by face index already, so a stable sort by
        # group id keeps each group's members in original face order -- the
        # first is the earliest face, every later one in the sort is a
        # duplicate of it.
        group_order = np.argsort(inv, kind="stable")
        sorted_inv = inv[group_order]
        is_first = np.ones(len(group_order), dtype=bool)
        is_first[1:] = sorted_inv[1:] != sorted_inv[:-1]
        out[which[group_order[~is_first]]] = True
    return out


def _take_kept(mesh: Mesh, keep: np.ndarray) -> Mesh:
    """*mesh* with only the faces *keep* marks, vertices compacted after.

    Mirrors `ops_topo.delete_faces`'s own empty-mesh special case: dropping
    every face is legal and leaves an empty mesh of the right shape (a `uv`
    array present but zero-length, never `None` on a mesh that had one),
    rather than routing through `topo.take_faces` with an empty face list,
    which would leave every vertex behind uncompacted.
    """
    if keep.all():
        return mesh
    if not keep.any():
        return topo.rebuild(
            np.zeros((0, 3), dtype="f4"),
            np.zeros(0, dtype="i4"),
            np.zeros(1, dtype="i4"),
            np.zeros(0, dtype="i4"),
            np.zeros(0, dtype=bool),
            uv=None if mesh.uv is None else np.zeros((0, 2), dtype="f4"),
        )
    kept, _old_to_new = topo.compact_vertices(topo.take_faces(mesh, np.flatnonzero(keep)))
    return kept


# --- shared measurement: winding consistency --------------------------------


def _shell_and_flip(mesh: Mesh, a) -> tuple[np.ndarray, np.ndarray, int]:
    """``(shell, flip, n_shells)`` -- see the module docstring's BFS paragraph.

    ``shell[f]`` is which connected component of "faces sharing a 2-use edge"
    face *f* belongs to; ``flip[f]`` is whether *f*'s own winding, as stored,
    disagrees with the arbitrary reference its shell's BFS happened to start
    from. Neither number means anything about *correctness* on its own --
    :func:`_minority_mask` is what turns "disagrees with the seed" into
    "disagrees with the majority", which is the only reading that does not
    depend on which face the walk started from.
    """
    n_faces = face_count(mesh)
    shell = np.full(n_faces, -1, dtype="i8")
    flip = np.zeros(n_faces, dtype=bool)
    if n_faces == 0:
        return shell, flip, 0

    twin = a.twin.astype("i8")
    has_twin = twin >= 0
    tc1 = np.flatnonzero(has_twin)
    tc2 = twin[tc1]
    tfa = a.corner_face[tc1].astype("i8")
    tfb = a.corner_face[tc2].astype("i8")

    fp = a.flipped_pairs
    if len(fp):
        ffa = a.corner_face[fp[:, 0]].astype("i8")
        ffb = a.corner_face[fp[:, 1]].astype("i8")
    else:
        ffa = np.zeros(0, dtype="i8")
        ffb = np.zeros(0, dtype="i8")

    # Both directions, so a CSR built from the sorted "from" face lets the walk
    # step either way across an edge.
    all_a = np.concatenate([tfa, ffa, tfb, ffb])
    all_b = np.concatenate([tfb, ffb, tfa, ffa])
    all_same = np.concatenate(
        [
            np.ones(len(tfa), dtype=bool),
            np.zeros(len(ffa), dtype=bool),
            np.ones(len(tfb), dtype=bool),
            np.zeros(len(ffb), dtype=bool),
        ]
    )

    order = np.argsort(all_a, kind="stable")
    nbr_face = all_b[order]
    nbr_same = all_same[order]
    csr_starts = np.searchsorted(all_a[order], np.arange(n_faces + 1))

    n_shells = 0
    for seed in range(n_faces):
        if shell[seed] != -1:
            continue
        shell[seed] = n_shells
        queue: deque[int] = deque([seed])
        while queue:
            f = queue.popleft()
            for k in range(csr_starts[f], csr_starts[f + 1]):
                g = int(nbr_face[k])
                if shell[g] != -1:
                    continue
                shell[g] = n_shells
                flip[g] = flip[f] if nbr_same[k] else not flip[f]
                queue.append(g)
        n_shells += 1
    return shell, flip, n_shells


def _minority_mask(shell: np.ndarray, flip: np.ndarray, n_shells: int) -> np.ndarray:
    """Per face, whether it is on the *minority* side of its own shell.

    Ties (an even split) are broken toward flagging the ``True`` side, so a
    50/50 shell still gets one deterministic answer rather than one that
    depends on BFS visitation order -- the same reasoning `weld`'s cluster
    representative (the lowest member, not an arbitrary one) is picked under.
    """
    if n_shells == 0:
        return np.zeros(0, dtype=bool)
    count_true = np.bincount(shell, weights=flip.astype("i8"), minlength=n_shells)
    count_total = np.bincount(shell, minlength=n_shells)
    count_false = count_total - count_true
    minority_is_true = count_true <= count_false
    return np.where(minority_is_true[shell], flip, ~flip)


def _shell_closed(mesh: Mesh, a, shell: np.ndarray, n_shells: int) -> np.ndarray:
    """Per shell: true iff every edge any of its faces touches has exactly two
    uses -- no boundary, no non-manifold edge, anywhere in the shell."""
    if n_shells == 0:
        return np.zeros(0, dtype=bool)
    bad_corner = a.edge_uses[a.corner_edge] != 2
    bad_face = np.zeros(face_count(mesh), dtype=bool)
    np.logical_or.at(bad_face, a.corner_face, bad_corner)
    has_bad = np.zeros(n_shells, dtype=bool)
    np.logical_or.at(has_bad, shell, bad_face)
    return ~has_bad


def _face_centroids(mesh: Mesh) -> np.ndarray:
    starts = mesh.starts.astype("i8")
    counts = np.diff(starts)
    pos = mesh.positions.astype("f8")
    sums = np.add.reduceat(pos[mesh.loops], starts[:-1], axis=0)
    return sums / counts[:, None]


def _face_fan_volume(mesh: Mesh) -> np.ndarray:
    """Six-times-signed-volume contribution per face, fan-triangulated from
    each face's own first corner.

    ``v0 . (v_i x v_{i+1})`` summed over the fan is the divergence theorem's
    flux term for that face alone, and it is exactly the fan-from-a-boundary-
    point identity the shoelace formula rests on: it is the correct signed
    contribution for *any* simple polygon, convex or not, with no earclip-grade
    concavity handling needed the way `mesh.triangulate` needs it for
    rendering. A closed, consistently wound mesh's true volume is this array's
    sum divided by six; this function stops one step short of that division
    because every caller only cares about the *sign* of a per-shell sum.
    """
    n_faces = face_count(mesh)
    if n_faces == 0:
        return np.zeros(0, dtype="f8")
    starts = mesh.starts.astype("i8")
    counts = np.diff(starts)
    n_tri = np.maximum(counts - 2, 0)
    total_tri = int(n_tri.sum())
    if total_tri == 0:
        return np.zeros(n_faces, dtype="f8")
    tri_face = np.repeat(np.arange(n_faces, dtype="i8"), n_tri)
    tri_offsets = np.concatenate([[0], np.cumsum(n_tri)])[:-1]
    local = np.arange(total_tri, dtype="i8") - np.repeat(tri_offsets, n_tri)
    c0 = starts[tri_face]
    ci = c0 + 1 + local
    cip1 = ci + 1
    pos = mesh.positions.astype("f8")
    v0 = pos[mesh.loops[c0]]
    vi = pos[mesh.loops[ci]]
    vip1 = pos[mesh.loops[cip1]]
    contrib = np.einsum("ij,ij->i", v0, np.cross(vi, vip1))
    return np.bincount(tri_face, weights=contrib, minlength=n_faces)


def _shell_volume(
    face_contrib: np.ndarray, shell: np.ndarray, to_flip: np.ndarray, n_shells: int
) -> np.ndarray:
    """Per-shell signed volume, with each minority face's contribution negated
    -- i.e. as if its winding were already corrected."""
    if n_shells == 0:
        return np.zeros(0, dtype="f8")
    signed = np.where(to_flip, -face_contrib, face_contrib)
    return np.bincount(shell, weights=signed, minlength=n_shells)


# --- Survey ------------------------------------------------------------------


@dataclass(frozen=True)
class Survey:
    """One count per defect :mod:`.ops_clean` can fix. Pure -- measures, does
    not touch the mesh. See each op's own docstring for exactly what it counts
    and why; this dataclass exists so a caller can ask "how much is wrong" at
    the cost of one `check_manifold`-sized pass, the same "measure once, act or
    don't" split `diagnose.findings` already follows.
    """

    degenerate_faces: int
    duplicate_faces: int
    loose_vertices: int
    coincident_vertices: int
    flipped_faces: int
    inside_out_shells: int
    open_edges: int


def _coincident_count(mesh: Mesh, distance: float) -> int:
    """How many vertices a :func:`merge_by_distance` at *distance* would
    remove -- `ops_topo._clusters`'s own vertex count minus its cluster count,
    the same clustering `weld` merges by, asked here for a count rather than a
    mesh. See the module docstring for why this reaches the private helper
    directly instead of calling `weld` (which always rebuilds a mesh, even
    when nothing would merge) just to read a length off the result."""
    n_verts = len(mesh.positions)
    if n_verts < 2:
        return 0
    labels = ops_topo._clusters(mesh.positions.astype("f8"), float(distance))
    n_clusters = int(labels.max()) + 1 if len(labels) else 0
    return n_verts - n_clusters


class FaceDefectMasks(NamedTuple):
    """Per-face boolean masks for the four face-level defects, one adjacency
    build and one BFS shared between all four -- what :func:`survey` counts
    and what :mod:`.diagnose` selects, computed once rather than per finding.

    ``degenerate`` and ``duplicate`` need no shell information at all;
    ``flipped`` and ``inside_out`` are both read off the same
    :func:`_shell_and_flip` BFS, ``inside_out`` marking every face of a shell
    :func:`survey`'s own ``inside_out_shells`` counted, not only the faces that
    individually disagree with their neighbours -- a shell that is uniformly
    wound but net inside-out has no minority face at all, and every one of its
    faces is still part of the defect.
    """

    degenerate: np.ndarray
    duplicate: np.ndarray
    flipped: np.ndarray
    inside_out: np.ndarray


def _compute_face_defects(mesh: Mesh) -> tuple[FaceDefectMasks, np.ndarray]:
    """``(masks, bad_shell)`` -- the shared computation behind both
    :func:`face_defect_masks` and :func:`survey`'s ``inside_out_shells`` count,
    so one `survey()` call does the BFS and the volume pass exactly once.
    ``bad_shell`` is per *shell*, not per face -- :func:`survey` sums it
    directly for a shell count, `face_defect_masks` broadcasts it onto faces.
    """
    n_faces = face_count(mesh)
    degenerate = _degenerate_mask(mesh)
    duplicate = _duplicate_mask(mesh)
    if n_faces == 0:
        empty = np.zeros(0, dtype=bool)
        return FaceDefectMasks(degenerate, duplicate, empty, empty), np.zeros(0, dtype=bool)

    a = adjacency(mesh)
    shell, flip, n_shells = _shell_and_flip(mesh, a)
    flipped = _minority_mask(shell, flip, n_shells)
    closed = _shell_closed(mesh, a, shell, n_shells)
    contrib = _face_fan_volume(mesh)
    volume = _shell_volume(contrib, shell, flipped, n_shells)
    bad_shell = closed & (volume < 0.0)
    inside_out = bad_shell[shell] if n_shells else np.zeros(n_faces, dtype=bool)
    return FaceDefectMasks(degenerate, duplicate, flipped, inside_out), bad_shell


def face_defect_masks(mesh: Mesh) -> FaceDefectMasks:
    """The four per-face defect masks :func:`survey` and :mod:`.diagnose` both
    read, computed once. See :class:`FaceDefectMasks`."""
    return _compute_face_defects(mesh)[0]


def survey(mesh: Mesh, *, distance: float = 1e-5) -> Survey:
    """Measure every defect :mod:`.ops_clean` can fix, without changing *mesh*.

    O(corners), the same order `adjacency.check_manifold` already costs (one
    BFS over the shell graph on top of it) -- safe to call from a button, not
    from a per-frame draw. See :class:`Survey`'s own field-by-field docstrings
    on each `remove_*`/`merge_by_distance`/`recalc_outside` op below for what
    each count actually measures.
    """
    a = adjacency(mesh)
    masks, bad_shell = _compute_face_defects(mesh)

    used = np.zeros(len(mesh.positions), dtype=bool)
    if len(mesh.loops):
        used[mesh.loops] = True
    loose = int((~used).sum())

    coincident = _coincident_count(mesh, distance)
    open_edges = int((a.edge_uses == 1).sum())
    # A shell counts once toward inside_out_shells however many faces it has --
    # `masks.inside_out` is the per-face broadcast of this same `bad_shell`.
    inside_out_shells = int(bad_shell.sum())

    return Survey(
        degenerate_faces=int(masks.degenerate.sum()),
        duplicate_faces=int(masks.duplicate.sum()),
        loose_vertices=loose,
        coincident_vertices=coincident,
        flipped_faces=int(masks.flipped.sum()),
        inside_out_shells=inside_out_shells,
        open_edges=open_edges,
    )


# --- individual ops ----------------------------------------------------------


def _remove_degenerate(mesh: Mesh) -> tuple[Mesh, int]:
    mask = _degenerate_mask(mesh)
    n = int(mask.sum())
    if n == 0:
        return mesh, 0
    return _take_kept(mesh, ~mask), n


def remove_degenerate(mesh: Mesh) -> Mesh:
    """Drop every face with fewer than three distinct vertices, or (near) zero
    area, then compact the vertices that leaves unused.

    UV **dropped** for the removed faces, **preserved** for every surviving
    one -- the same rule `ops_topo.delete_faces` states for the same reason: a
    corner that survives keeps the uv it had.

    Identity when nothing is degenerate: :func:`Survey.degenerate_faces` and
    this op's own count are the same mask, so a caller can check the survey
    first and skip the call, or call this unconditionally and get the same
    mesh object back either way.
    """
    return _remove_degenerate(mesh)[0]


def _remove_duplicate_faces(mesh: Mesh) -> tuple[Mesh, int]:
    mask = _duplicate_mask(mesh)
    n = int(mask.sum())
    if n == 0:
        return mesh, 0
    return _take_kept(mesh, ~mask), n


def remove_duplicate_faces(mesh: Mesh) -> Mesh:
    """Drop the later face of every pair (or run) of faces over the same
    vertex set, then compact.

    UV **dropped** for the removed faces, **preserved** for the rest -- same
    rule as :func:`remove_degenerate`.
    """
    return _remove_duplicate_faces(mesh)[0]


def _remove_loose_vertices(mesh: Mesh) -> tuple[Mesh, int]:
    out, old_to_new = topo.compact_vertices(mesh)
    removed = int((old_to_new < 0).sum())
    return out, removed


def remove_loose_vertices(mesh: Mesh) -> Mesh:
    """Drop every vertex no face references.

    This *is* `topo.compact_vertices` -- faces are untouched, so there is
    nothing this op does that the shared compaction helper does not already
    do, including its own identity check when every vertex is used.
    """
    return _remove_loose_vertices(mesh)[0]


def _merge_by_distance(mesh: Mesh, distance: float) -> tuple[Mesh, int]:
    n_verts = len(mesh.positions)
    if n_verts < 2:
        return mesh, 0
    removed = _coincident_count(mesh, distance)
    if removed == 0:
        return mesh, 0
    out, _sel = ops_topo.weld(
        mesh, ElementSel(verts=np.arange(n_verts, dtype="i4")), eps=float(distance)
    )
    return out, removed


def merge_by_distance(mesh: Mesh, distance: float) -> Mesh:
    """Merge every pair of vertices within *distance* into one, at their
    cluster's centroid.

    All of the real work is `ops_topo.weld`'s -- this only decides, from the
    same clustering weld itself does (`ops_topo._clusters`, reached directly
    rather than through `weld` -- see the module docstring), whether calling
    it would change anything, so a mesh with nothing to merge comes back as
    the identical object rather than paying for a rebuild that welds nothing.

    UV **preserved**: `weld` -> `merge_vertices` keeps each surviving corner's
    own uv, which is `merge_vertices`'s own documented policy.
    """
    return _merge_by_distance(mesh, distance)[0]


def _fill_all_holes(mesh: Mesh) -> tuple[Mesh, int]:
    a = adjacency(mesh)
    boundary = a.edge_verts[a.edge_uses == 1]
    if len(boundary) == 0:
        return mesh, 0
    rings, _pinched = boundary_loops(mesh)
    out, _sel = ops_topo.fill_hole(mesh, ElementSel(edges=boundary))
    # A ring `boundary_loops`' own walk could not close is still real holes to
    # report -- the same "holes or 1" fallback `diagnose.rows_for` uses for the
    # identical reason.
    return out, len(rings) or 1


def fill_all_holes(mesh: Mesh) -> Mesh:
    """Cap every boundary loop in the mesh with one n-gon each.

    One `ops_topo.fill_hole` call, seeded with every boundary edge in the mesh
    at once: `fill_hole` already resolves its seed edges into whichever
    distinct rings they touch (`adjacency.boundary_ring_from`) and caps each
    one, so there is no reason to call it once per ring. Refuses exactly as
    `fill_hole` does -- a pinched vertex, a self-crossing ring, a ring past
    `ops_dissolve.MAX_DISSOLVED_RING` -- and refuses for the *whole* call
    rather than filling the holes it could and skipping the one it could not,
    because a partial `clean()` step that silently dropped a hole would be a
    worse surprise than a named refusal is.

    UV: **copied from an adjacent boundary corner** at each new rim vertex,
    the same documented placeholder `fill_hole` uses.
    """
    return _fill_all_holes(mesh)[0]


def _recalc(mesh: Mesh) -> tuple[Mesh, int]:
    n_faces = face_count(mesh)
    if n_faces == 0:
        return mesh, 0
    a = adjacency(mesh)
    shell, flip, n_shells = _shell_and_flip(mesh, a)
    to_flip = _minority_mask(shell, flip, n_shells)

    closed = _shell_closed(mesh, a, shell, n_shells)
    contrib = _face_fan_volume(mesh)
    volume = _shell_volume(contrib, shell, to_flip, n_shells)
    shell_global_flip = closed & (volume < 0.0)

    # Open shells: after the same minority correction, orient the whole shell
    # by which way most of its own faces already point relative to its own
    # centroid -- there is no volume sign for a sheet with a boundary, so this
    # is the closest analogue to "outward" it has.
    if n_shells:
        raw = face_normals(mesh)
        logical = np.where(to_flip[:, None], -raw, raw)
        centroids = _face_centroids(mesh)
        sums = np.zeros((n_shells, 3))
        np.add.at(sums, shell, centroids)
        face_totals = np.bincount(shell, minlength=n_shells)
        shell_centroid = sums / np.maximum(face_totals, 1)[:, None]
        direction = np.einsum("ij,ij->i", logical, centroids - shell_centroid[shell])
        inward = np.zeros(n_shells)
        outward = np.zeros(n_shells)
        np.add.at(inward, shell, (direction < 0.0).astype("f8"))
        np.add.at(outward, shell, (direction > 0.0).astype("f8"))
        shell_global_flip = shell_global_flip | (~closed & (inward > outward))

    final_flip = to_flip ^ shell_global_flip[shell]
    n_flipped = int(final_flip.sum())
    if n_flipped == 0:
        return mesh, 0
    faces = np.flatnonzero(final_flip).astype("i4")
    out, _sel = ops_topo.flip_normals(mesh, ElementSel(faces=faces))
    return out, n_flipped


def recalc_outside(mesh: Mesh) -> Mesh:
    """Make every shell's winding consistent, then orient each shell outward.

    Two passes folded into the one set of face flips they add up to: first,
    every shell's faces are made to agree with their own majority (the minority
    side is what :class:`Survey.flipped_faces` counts); second, each shell as a
    whole is oriented -- a closed shell by the sign of its own volume, an open
    one by which way most of its faces already point relative to its own
    centroid. See the module docstring for why a fan-triangulated divergence
    theorem needs no concavity handling to get the sign right.

    UV **preserved**: every flip here is `ops_topo.flip_normals`, whose own
    documented policy is that a flipped face's corners keep their uvs, in the
    reversed order the flip puts the corners in.

    Identity when every shell is already consistent and outward: the flip mask
    this computes is empty, and `flip_normals` is never called.
    """
    return _recalc(mesh)[0]


# --- clean ---------------------------------------------------------------


@dataclass(frozen=True)
class CleanReport:
    """What one :func:`clean` call actually changed, one count per step it ran.

    ``degenerate_removed`` is the *sum* of the two degenerate passes clean()
    runs (see its own docstring for why there are two) -- a caller wanting the
    two separately can call :func:`remove_degenerate` around
    :func:`merge_by_distance` directly and read each op's own count instead of
    going through `clean`.
    """

    degenerate_removed: int
    merged_vertices: int
    duplicate_removed: int
    loose_removed: int
    holes_filled: int
    faces_flipped: int


def clean(
    mesh: Mesh, *, distance: float = 1e-5, fill_holes: bool = False, recalc: bool = True
) -> tuple[Mesh, CleanReport]:
    """Run every repair in the one order that is actually correct, and report
    what each step did.

    **Degenerate, then merge, then degenerate again.** Merging two vertices
    that a face used twice over (or that collapse a triangle's three corners to
    two positions) can *create* a degenerate face that did not exist before the
    merge -- so the first pass clears out what was already degenerate (which
    would otherwise pollute the coincident-vertex count merge is about to act
    on), and the second catches whatever the merge itself produced. Duplicates
    and loose vertices are then swept after the mesh has stopped changing shape
    from beneath them, and the two optional, more expensive passes -- filling
    holes and re-orienting shells -- run last, on the smallest, already-cleaned
    mesh clean() is going to hand back.

    Refuses past :data:`MAX_CLEAN_CORNERS` **before any step runs**, on the
    *input* mesh's own corner count -- not
    partway through, when a caller has already paid for the cheaper steps and
    only `recalc_outside`'s BFS was going to be the expensive one. See the
    module docstring for the measurements the ceiling is read off.
    """
    _refuse_if_too_large(mesh)

    mesh, degenerate_1 = _remove_degenerate(mesh)
    mesh, merged = _merge_by_distance(mesh, distance)
    mesh, degenerate_2 = _remove_degenerate(mesh)
    mesh, duplicate = _remove_duplicate_faces(mesh)
    mesh, loose = _remove_loose_vertices(mesh)

    holes = 0
    if fill_holes:
        mesh, holes = _fill_all_holes(mesh)

    flipped = 0
    if recalc:
        mesh, flipped = _recalc(mesh)

    return mesh, CleanReport(
        degenerate_removed=degenerate_1 + degenerate_2,
        merged_vertices=merged,
        duplicate_removed=duplicate,
        loose_removed=loose,
        holes_filled=holes,
        faces_flipped=flipped,
    )
