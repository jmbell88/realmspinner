"""Winding repair and vertex merging: recalculate normals, merge by distance.

**Every op here is `Mesh -> Mesh`, and every op is a no-op when there is
nothing to fix** -- returning the exact same object, not an equal one. The
viewport's GPU cache and the undo mechanism both key on `id(mesh)` (see
`mesh.py`'s own docstring), so a repair that rebuilt an already-clean mesh
byte-for-byte would still evict every cached buffer and push an undo step for
an edit that changed nothing. Every internal helper below therefore returns
`(mesh_or_same_object, count_fixed)`, and the public op is a thin wrapper that
drops the count.

**What each op reuses rather than reimplements.** :func:`merge_by_distance`
calls :func:`~.ops_topo.weld` for the actual merge -- it does not walk vertices
itself -- but a *count* of how many vertices would merge is wanted to decide
whether the merge is worth paying for, and `weld` has no way to answer that
without doing the merge. Rather than a second clustering implementation it
calls `ops_topo._clusters` directly -- the module-private helper `weld` itself
is built from. :func:`recalc_outside` calls :func:`~.ops_topo.flip_normals` to
actually reverse the faces it decides need reversing, rather than permuting
corners itself a second time.

**Winding consistency is a BFS over manifold edges, and it is the one
Python-level loop in this module.** A face's winding only means anything
relative to its neighbours: two faces sharing an edge are "consistent" when
they traverse it in opposite directions (`adjacency.twin`) and "inconsistent"
when they traverse it the same way (`adjacency.flipped_pairs`). Propagating
that relation with a breadth-first walk from one face per connected shell
assigns every face a boolean relative to its shell's own arbitrary starting
point; the *minority* boolean in each shell is what disagrees with the
majority, which is what `recalc_outside` corrects first. A closed shell is then
oriented outward by the sign of its own volume (the divergence theorem over
each face's own fan triangulation, which needs no earclip-grade concavity
handling because a fan sum from a point on the polygon's own boundary is the
correct signed contribution for *any* simple polygon, convex or not -- the same
identity the shoelace formula rests on); an open shell has no volume to sign,
so its global orientation is instead the majority of its own face normals
pointing away from its centroid, the closest an open sheet has to "outward".
The walk is over faces of one shell with a queue, never a face pair matched
against every other face, so it is linear in the mesh rather than quadratic.

**MAX_CLEAN_CORNERS**, below, is the ceiling every Clay walking op carries,
chosen the same way theirs were: measured from what the operation actually
grows, with a margin under the point a single press stops being well under a
second. `MAX_CLEAN_CORNERS` exists for `recalc_outside`'s outer
`for seed in range(n_faces)` restarting its BFS once per shell, and an imported
mesh with many small separate shells (a decal, a kitbash, a scene flattened to
one mesh) pays that restart far more often than one sphere does. Measured on
many small boxes concatenated into one mesh (`n_faces = 300 * n`, one closed
shell each), with the UV-bearing and UV-less cases within noise of each other
(so one constant serves both):

| boxes  | corners   | faces   | recalc  |
|-------:|----------:|--------:|--------:|
|  2,000 |    48,000 |  12,000 |   56 ms |
|  8,000 |   192,000 |  48,000 |  247 ms |
| 20,000 |   480,000 | 120,000 |  636 ms |
| 40,000 |   960,000 | 240,000 | 1306 ms |

Growth is linear throughout; 480,000 corners (20,000 boxes) is the last point
still comfortably under a second, and 960,000 crosses it. `MAX_CLEAN_CORNERS` is
set well below that crossing rather than at it.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import ops_topo
from .adjacency import adjacency
from .elements import ElementSel, OpError
from .mesh import Mesh, face_count, face_normals

__all__ = [
    "merge_by_distance",
    "recalc_outside",
]

#: See the module docstring's measurement table -- one constant for both a
#: UV-less and a UV-bearing mesh.
MAX_CLEAN_CORNERS = 300_000


def _refuse_if_too_large(mesh: Mesh) -> None:
    if len(mesh.loops) > MAX_CLEAN_CORNERS:
        raise OpError(
            f"This mesh has {len(mesh.loops):,} corners, past the "
            f"{MAX_CLEAN_CORNERS:,} Recalculate Normals can process without "
            "stalling. Split the mesh first."
        )


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
    every shell's faces are made to agree with their own majority; second,
    each shell as a whole is oriented -- a closed shell by the sign of its own
    volume, an open one by which way most of its faces already point relative
    to its own centroid. See the module docstring for why a fan-triangulated
    divergence theorem needs no concavity handling to get the sign right.

    UV **preserved**: every flip here is `ops_topo.flip_normals`, whose own
    documented policy is that a flipped face's corners keep their uvs, in the
    reversed order the flip puts the corners in.

    Identity when every shell is already consistent and outward: the flip mask
    this computes is empty, and `flip_normals` is never called.

    Refuses past :data:`MAX_CLEAN_CORNERS`: the BFS/adjacency/volume pass
    is unbounded as the mesh grows (the 2026-09-19 audit's clay-14: a
    523,264-corner mesh ran to completion in 429 ms with no warning).
    """
    _refuse_if_too_large(mesh)
    return _recalc(mesh)[0]
