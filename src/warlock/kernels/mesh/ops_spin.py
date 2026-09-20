"""Spin and screw: lathe a profile of edges around an axis into new topology.

A different family from :mod:`.ops_model`'s edit-in-place ops. Both of these
mint every face and every ring past the first from scratch -- closer in
spirit to a primitive generator revolving :func:`~.primitives.lathe`'s own
``profile`` than to an op that rewrites faces that already exist -- which is
why UV here is **generated** rather than one of the ``preserved`` /
``interpolated`` / ``inherited`` three every edit op in this package states:
there is no existing face to draw a uv from, only a position along the
profile and a position along the spin.

Neither op touches an existing face. Both concatenate new rings and new bands
onto the mesh they were given -- the original faces are copied through
verbatim -- so there is nothing for either to refuse over the mesh's own
existing geometry, only over the profile selection and the step count.
"""

from __future__ import annotations

import math

import numpy as np

from . import topo
from .adjacency import adjacency
from .elements import ElementSel, OpError
from .mesh import Mesh, face_count

__all__ = ["screw", "spin"]

#: The largest ``bands * profile-edges`` grid of quads either op will build.
#: Both walk that grid in a plain Python loop -- one iteration per quad, to
#: gather four vertex ids and, when the mesh has uv, four generated
#: coordinates -- so the cost is exactly that product. New code with nothing
#: measured yet, so the bound mirrors ``primitives.MAX_DIVISIONS``'s own
#: order of magnitude rather than a rate taken from a real run.
MAX_SPIN_QUADS = 65_536


def _profile_order(edges: np.ndarray) -> tuple[list[int], bool]:
    """``(vertices in walk order, is_closed)`` for one chain or loop of
    undirected edges.

    Every vertex touched must have degree at most 2 within the selection --
    a fork has no single walk order -- and the selection must be one
    connected run: refused by name otherwise, the same "a walk has no
    defined next step" refusal :mod:`.ops_dissolve` and
    :func:`~.ops_topo.bridge_edges` already give a forked or
    multi-component selection. Degree at most 2 also means the candidate list
    at each step of the walk never actually holds more than one vertex once
    the one just arrived from is excluded; ``min`` is kept as the same
    "deterministic over a case that cannot occur" defence
    :func:`~.ops_topo._entry_corners` gives its own tie-break, not a
    real ambiguity this function resolves.
    """
    adj: dict[int, list[int]] = {}
    for u, v in np.asarray(edges, dtype="i8").tolist():
        adj.setdefault(int(u), []).append(int(v))
        adj.setdefault(int(v), []).append(int(u))
    for v, nbrs in adj.items():
        if len(nbrs) > 2:
            raise OpError(
                f"The profile forks at vertex {v}; select a single open "
                "chain or closed loop of edges."
            )
    ends = sorted(v for v, nbrs in adj.items() if len(nbrs) == 1)
    closed = not ends
    start = ends[0] if ends else min(adj)
    order = [start]
    prev: int | None = None
    cur = start
    while True:
        candidates = [n for n in adj[cur] if n != prev]
        if not candidates:
            break
        nxt = min(candidates)
        if nxt == start and len(order) > 1:
            break
        order.append(nxt)
        prev, cur = cur, nxt
    if len(order) != len(adj):
        raise OpError(
            "The selected edges do not form a single connected chain or loop."
        )
    return order, closed


def _rotate(points: np.ndarray, axis: int, center: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate ``(N, 3)`` points about the line through ``center`` parallel to
    the given axis (0 = X, 1 = Y, 2 = Z).

    A positive angle turns the lower-numbered of the two remaining axes
    toward the higher-numbered one -- X toward Y about Z, Y toward Z about X,
    Z toward X about Y -- an arbitrary but stated convention, exactly the
    kind :func:`~.ops_topo.bridge_edges`'s own rotation search names its tie-
    break. Plain 2-D rotation rather than Rodrigues' formula: the axis is
    always exactly X, Y or Z here, never an arbitrary direction, so the two
    components that move are known up front and there is nothing a general
    axis-angle formula would buy that a rotation matrix in the other two
    columns does not already give for free.
    """
    other = [a for a in range(3) if a != axis]
    u_axis, v_axis = other
    out = points.copy()
    u = points[:, u_axis] - center[u_axis]
    v = points[:, v_axis] - center[v_axis]
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    out[:, u_axis] = center[u_axis] + u * c - v * s
    out[:, v_axis] = center[v_axis] + u * s + v * c
    return out


def _validate_profile(mesh: Mesh, sel: ElementSel) -> list[int]:
    if len(sel.edges) == 0:
        raise OpError("Select the edges of a profile to spin.")
    a = adjacency(mesh)
    ids = a.edge_ids(sel.edges)
    if (ids < 0).any():
        raise OpError("That edge is not part of this mesh.")
    order, closed = _profile_order(sel.edges)
    if len(order) < 2:
        raise OpError("Select at least one edge to spin.")
    return order, closed


def _profile_pairs(order: list[int], closed: bool) -> list[tuple[int, int]]:
    pairs = [(i, i + 1) for i in range(len(order) - 1)]
    if closed:
        pairs.append((len(order) - 1, 0))
    return pairs


def _new_quad_starts(starts: np.ndarray, n_new: int) -> np.ndarray:
    """``starts`` grown by ``n_new`` freshly appended quads (4 corners each)."""
    grown = int(starts[-1]) + 4 * np.arange(1, n_new + 1, dtype="i8")
    return np.concatenate([starts.astype("i8"), grown])


def _quad_uv_array(quad_uv: list[tuple[float, float]]) -> np.ndarray:
    return np.array(quad_uv, dtype="f4").reshape(-1, 2)


def _refuse_spin_size(n_bands: int, n_pairs: int, what: str) -> None:
    total = n_bands * n_pairs
    if total > MAX_SPIN_QUADS:
        raise OpError(
            f"{what} this profile would build {total:,} quads, past the "
            f"{MAX_SPIN_QUADS:,} it works with before stalling the frame it "
            "runs on. Spin fewer steps, or select a shorter profile."
        )


def spin(
    mesh: Mesh, sel: ElementSel, *, axis: int, angle: float, steps: int, center
) -> tuple[Mesh, ElementSel]:
    """Lathe the selected edges (a profile) around ``axis`` through
    ``center``, ``steps`` times across ``angle`` degrees total.

    **A whole turn closes the ring by reusing the first ring's own vertex
    indices for the last band**, rather than minting a duplicate coincident
    ring and welding it away: ``abs(angle) % 360 == 0`` (within float noise)
    makes the final band connect back to ring zero directly, so the seam is
    closed by construction and there is nothing left to weld. Any other
    angle leaves ``steps + 1`` distinct rings and an open seam at both ends,
    exactly what the profile's own two ends deserve when the spin does not
    come all the way around.

    A **closed** profile (a loop rather than a chain) spins into a tube-of-
    rings the same way either way -- its own wrap-around edge is just one
    more entry in the band grid -- so lathing a closed silhouette all the way
    around builds a shape closed in both directions, a legitimate if unusual
    request nothing here refuses.

    UV is **generated**: ``u`` runs along the profile by station fraction,
    ``v`` by spin-step fraction, supplied only when the source mesh already
    carries uv elsewhere (a mesh with none stays with none, the same default
    every primitive generator in this package ships).

    Selection out: the newly built faces.
    """
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise OpError("Axis must be 0 (X), 1 (Y) or 2 (Z).")
    steps = int(steps)
    if steps < 1:
        raise OpError("Spin needs at least one step.")
    angle_f = float(angle)
    if angle_f == 0.0:
        raise OpError("The spin angle must not be zero.")
    order, closed_profile = _validate_profile(mesh, sel)

    remainder = abs(angle_f) % 360.0
    full_turn = (remainder < 1e-6 or remainder > 360.0 - 1e-6) and abs(angle_f) > 1e-9
    n_copies = steps if full_turn else steps + 1
    n_bands = n_copies if full_turn else n_copies - 1
    pairs = _profile_pairs(order, closed_profile)
    _refuse_spin_size(n_bands, len(pairs), "Spinning")

    center_v = np.asarray(center, dtype="f8").reshape(3)
    profile_pos = mesh.positions[order].astype("f8")
    step_angle = math.radians(angle_f / steps)

    n_verts = len(mesh.positions)
    ring_index: list[np.ndarray] = [np.asarray(order, dtype="i8")]
    new_rows: list[np.ndarray] = []
    for k in range(1, n_copies):
        new_rows.append(_rotate(profile_pos, axis, center_v, step_angle * k))
        ring_index.append(n_verts + (k - 1) * len(order) + np.arange(len(order), dtype="i8"))
    new_positions = np.concatenate(new_rows) if new_rows else np.zeros((0, 3))

    quads: list[int] = []
    quad_uv: list[tuple[float, float]] = []
    p_span = max(len(order) - 1, 1)
    for k in range(n_bands):
        k2 = (k + 1) % n_copies
        v0 = k / n_bands
        v1 = k2 / n_bands if k2 != 0 else 1.0
        for i, j in pairs:
            va0, vb0 = int(ring_index[k][i]), int(ring_index[k][j])
            va1, vb1 = int(ring_index[k2][i]), int(ring_index[k2][j])
            quads.extend([va0, vb0, vb1, va1])
            if mesh.uv is not None:
                u0, u1 = i / p_span, j / p_span
                quad_uv.extend([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])

    n_faces0 = face_count(mesh)
    n_new = len(quads) // 4
    positions_all = (
        np.concatenate([mesh.positions.astype("f8"), new_positions])
        if len(new_positions)
        else mesh.positions.astype("f8")
    )
    out = topo.rebuild(
        positions_all,
        np.concatenate([mesh.loops.astype("i8"), np.array(quads, dtype="i8")]),
        _new_quad_starts(mesh.starts, n_new),
        np.concatenate([mesh.material, np.zeros(n_new, dtype="i8")]),
        np.concatenate([mesh.smooth, np.zeros(n_new, dtype=bool)]),
        uv=None if mesh.uv is None else np.concatenate([mesh.uv, _quad_uv_array(quad_uv)]),
    )
    return out, ElementSel(faces=np.arange(n_faces0, n_faces0 + n_new))


def screw(
    mesh: Mesh, sel: ElementSel, *, axis: int, angle: float, steps: int, height: float, center
) -> tuple[Mesh, ElementSel]:
    """:func:`spin` plus a per-step offset along ``axis``, and **never**
    closes the seam.

    A screw is a helix: even a whole-turn ``angle`` leaves the last ring
    ``height / steps`` further along the axis than the first, so the two
    never coincide and there is nothing to weld -- ``steps + 1`` rings, every
    time, whatever ``angle`` is. That is the one way this differs from
    :func:`spin` in shape; everything else (the profile's own validation,
    the band grid, the generated uv) is identical.
    """
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise OpError("Axis must be 0 (X), 1 (Y) or 2 (Z).")
    steps = int(steps)
    if steps < 1:
        raise OpError("Screw needs at least one step.")
    angle_f = float(angle)
    if angle_f == 0.0 and float(height) == 0.0:
        raise OpError("The screw angle and height cannot both be zero.")
    order, closed_profile = _validate_profile(mesh, sel)

    n_copies = steps + 1
    pairs = _profile_pairs(order, closed_profile)
    _refuse_spin_size(n_copies - 1, len(pairs), "Screwing")

    center_v = np.asarray(center, dtype="f8").reshape(3)
    profile_pos = mesh.positions[order].astype("f8")
    step_angle = math.radians(angle_f / steps)
    step_lift = float(height) / steps

    n_verts = len(mesh.positions)
    ring_index: list[np.ndarray] = [np.asarray(order, dtype="i8")]
    new_rows: list[np.ndarray] = []
    for k in range(1, n_copies):
        ring = _rotate(profile_pos, axis, center_v, step_angle * k)
        ring[:, axis] += step_lift * k
        new_rows.append(ring)
        ring_index.append(n_verts + (k - 1) * len(order) + np.arange(len(order), dtype="i8"))
    new_positions = np.concatenate(new_rows) if new_rows else np.zeros((0, 3))

    quads: list[int] = []
    quad_uv: list[tuple[float, float]] = []
    p_span = max(len(order) - 1, 1)
    n_bands = n_copies - 1
    for k in range(n_bands):
        v0, v1 = k / n_bands, (k + 1) / n_bands
        for i, j in pairs:
            va0, vb0 = int(ring_index[k][i]), int(ring_index[k][j])
            va1, vb1 = int(ring_index[k + 1][i]), int(ring_index[k + 1][j])
            quads.extend([va0, vb0, vb1, va1])
            if mesh.uv is not None:
                u0, u1 = i / p_span, j / p_span
                quad_uv.extend([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])

    n_faces0 = face_count(mesh)
    n_new = len(quads) // 4
    positions_all = (
        np.concatenate([mesh.positions.astype("f8"), new_positions])
        if len(new_positions)
        else mesh.positions.astype("f8")
    )
    out = topo.rebuild(
        positions_all,
        np.concatenate([mesh.loops.astype("i8"), np.array(quads, dtype="i8")]),
        _new_quad_starts(mesh.starts, n_new),
        np.concatenate([mesh.material, np.zeros(n_new, dtype="i8")]),
        np.concatenate([mesh.smooth, np.zeros(n_new, dtype=bool)]),
        uv=None if mesh.uv is None else np.concatenate([mesh.uv, _quad_uv_array(quad_uv)]),
    )
    return out, ElementSel(faces=np.arange(n_faces0, n_faces0 + n_new))
