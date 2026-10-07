"""Distances, angles, areas and volumes -- pure numbers, world matrices
passed in rather than composed here.

The element HUD and the agent's ``clay_measure`` tool both read this module
rather than each carrying its own arithmetic, the same "one function, several
callers" shape: a distance the HUD and an agent disagree about is worse than
neither having one.

**No document, no object -- every function takes exactly the numbers it
needs.** :func:`distance` and :func:`angle` take bare points (already in
world space, however a caller got them there); :func:`face_area` and
:func:`volume` take a :class:`~.mesh.Mesh` plus an
optional *world* matrix, composed with the mesh's own local positions the
same way every world-space function in :mod:`.ops` does (tranche 3: scene
structure) -- ``world=None`` measures the mesh in its own local space, which
is what a caller with no document in hand, or an unparented object, wants.

**Area and volume are computed over the mesh's own triangulation**
(:func:`~.mesh.triangulate`), the one every render already uses, so a
measurement never disagrees with what is on screen. :func:`volume` is the
divergence-theorem volume over *every* triangle regardless of whether the
mesh is actually closed -- a meaningful number for an open mesh is not
this module's problem to solve; a caller that cares checks
``adjacency.check_manifold`` first (:func:`volume_if_closed` does).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import mesh as bm
from .adjacency import check_manifold

__all__ = [
    "angle",
    "distance",
    "face_area",
    "is_closed",
    "volume",
    "volume_if_closed",
]


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    """The straight-line distance between two points, whatever space they
    are already in (world, if that is what a caller measured)."""
    return float(np.linalg.norm(np.asarray(b, dtype="f8") - np.asarray(a, dtype="f8")))


def angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """The angle at *b*, between the rays *b -> a* and *b -> c*, in degrees.

    ``0.0`` for a degenerate ray (*a* or *c* coincident with *b*): there is no
    angle to report, and a caller asking about three points that collapsed to
    two gets a number rather than a division by zero.
    """
    b_arr = np.asarray(b, dtype="f8")
    v1 = np.asarray(a, dtype="f8") - b_arr
    v2 = np.asarray(c, dtype="f8") - b_arr
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 0.0
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def _world_positions(mesh: bm.Mesh, world: np.ndarray | None) -> np.ndarray:
    positions = np.asarray(mesh.positions, dtype="f8")
    if world is None or len(positions) == 0:
        return positions
    homogeneous = np.hstack([positions, np.ones((len(positions), 1))])
    return (np.asarray(world, dtype="f8") @ homogeneous.T).T[:, :3]


def face_area(
    mesh: bm.Mesh, faces: Sequence[int] | np.ndarray, world: np.ndarray | None = None
) -> float:
    """The summed area of *faces* (a face-mode selection's worth), in world
    space when *world* is given."""
    face_idx = np.asarray(list(faces), dtype="i8") if not isinstance(faces, np.ndarray) else (
        faces.astype("i8")
    )
    if len(face_idx) == 0:
        return 0.0
    tri_corners, tri_face = bm.triangulate(mesh)
    if len(tri_corners) == 0:
        return 0.0
    mask = np.isin(tri_face, face_idx)
    if not mask.any():
        return 0.0
    positions = _world_positions(mesh, world)
    tri_pts = positions[tri_corners[mask]]
    e1 = tri_pts[:, 1] - tri_pts[:, 0]
    e2 = tri_pts[:, 2] - tri_pts[:, 0]
    return float(0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum())


def volume(mesh: bm.Mesh, world: np.ndarray | None = None) -> float:
    """The divergence-theorem volume enclosed by every triangle of *mesh*, in
    world space when *world* is given. See the module docstring for why this
    does not itself check that the mesh is closed."""
    tri_corners, _tri_face = bm.triangulate(mesh)
    if len(tri_corners) == 0:
        return 0.0
    positions = _world_positions(mesh, world)
    tri_pts = positions[tri_corners]
    v0, v1, v2 = tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2]
    signed = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()) / 6.0
    return abs(signed)


def is_closed(mesh: bm.Mesh) -> bool:
    """No hole and no non-manifold edge -- the reading ``clay_add_mesh``'s own
    ``closed`` uses, so a volume gate here agrees with it. An empty or
    single-face-less mesh is not closed."""
    if len(mesh.starts) <= 1:
        return False
    report = check_manifold(mesh)
    return len(report.boundary_edges) == 0 and len(report.nonmanifold_edges) == 0


def volume_if_closed(mesh: bm.Mesh, world: np.ndarray | None = None) -> float | None:
    """:func:`volume`, or ``None`` when *mesh* is not a closed manifold.

    The 2026-10-03 audit's clay-25: ``clay_measure kind=volume`` called
    :func:`volume` directly and answered ``0.6667`` for a five-faced open box
    while ``clay_add_mesh`` said not closed for the same object -- the divergence
    sum over an open surface depends on where the object sits, so the number
    is meaningless and an agent used it as a real volume. :func:`volume`
    itself is unchanged (the element HUD reads it and a caller with no
    document may want the raw sum); a caller that answers a person or an
    agent with the number goes through this.
    """
    if not is_closed(mesh):
        return None
    # The 2026-10-07 audit's clay-14: ``is_closed`` counts only boundary and
    # non-manifold edges, so a cube with one face wound the wrong way is
    # "closed" and the divergence sum answered 0.667 for a unit cube -- a
    # confident wrong number, the failure this gate exists to prevent. A
    # flipped edge (two faces traversing it the same way) means the shell has
    # no consistent outside, so the signed sum is not a volume either.
    if len(check_manifold(mesh).flipped_edges):
        return None
    return volume(mesh, world)
