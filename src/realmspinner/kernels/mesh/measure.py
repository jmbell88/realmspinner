"""Distances, angles, areas and volumes -- pure numbers, world matrices
passed in rather than composed here.

The element HUD and the agent's ``clay_measure`` tool both read this module
rather than each carrying its own arithmetic, the same "one function, several
callers" shape :mod:`.readiness` states for :func:`~.readiness.validate`: a
distance the HUD and an agent disagree about is worse than neither having one.

**No document, no object -- every function takes exactly the numbers it
needs.** :func:`distance` and :func:`angle` take bare points (already in
world space, however a caller got them there); :func:`face_area`,
:func:`volume` and :func:`edge_length` take a :class:`~.mesh.Mesh` plus an
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
``adjacency.check_manifold`` first, the same way :mod:`.readiness` does
before it ever calls this.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import mesh as bm

__all__ = ["angle", "distance", "edge_length", "face_area", "volume"]


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


def edge_length(
    mesh: bm.Mesh, edges: Sequence[Sequence[int]] | np.ndarray, world: np.ndarray | None = None
) -> float:
    """The summed length of *edges* (vertex-index pairs, an edge-mode
    selection's own shape), in world space when *world* is given."""
    rows = (
        np.asarray(edges, dtype="i8").reshape(-1, 2) if len(edges) else np.zeros((0, 2), dtype="i8")
    )
    if len(rows) == 0:
        return 0.0
    positions = _world_positions(mesh, world)
    a = positions[rows[:, 0]]
    b = positions[rows[:, 1]]
    return float(np.linalg.norm(b - a, axis=1).sum())
