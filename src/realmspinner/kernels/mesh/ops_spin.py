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
#:
#: **This product alone does not bound the wall clock.** The 2026-09-19
#: audit's clay-21 found the comment above was never actually measured, and
#: is false: at the identical 65,536-quad product, a 1-edge-profile spun
#: 65,536 steps measured 576-658 ms (bands-only and identical-product series,
#: this fix's own scratch measurement, reproducing the audit's 689-692 ms),
#: while a 256-edge-profile spun 256 steps -- same product -- measured only
#: 48 ms. The outer per-band loop (slicing ``ring_index[k]``/``ring_index
#: [k2]``, the modulo, the fixed Python-level overhead of one more iteration)
#: costs roughly 10us *per band* almost independently of how many profile
#: edges that band carries, while the inner per-quad loop costs roughly
#: 0.7-2.5us *per quad* -- so a selection with many bands and few edges pays
#: for bands it is not amortising over, and :data:`MAX_SPIN_QUADS` alone
#: cannot see that: it only ever saw the two multiplied together. See
#: :data:`MAX_SPIN_BANDS`, which bounds the factor this ceiling cannot.
MAX_SPIN_QUADS = 65_536

#: The largest step/band count either op will walk, **regardless of profile
#: length** -- the other half of clay-21's fix, refusing on ``n_bands`` and
#: ``n_pairs`` (via :data:`MAX_SPIN_QUADS`) separately rather than only on
#: their product. Measured on this machine, a one-edge profile (``n_pairs``
#: pinned at 1, so this is the per-band cost in isolation) at increasing step
#: counts, uv-bearing (the slightly more expensive case):
#:
#: | bands  | quads  | spin() |
#: |-------:|-------:|-------:|
#: |    256 |    256 |  2.7 ms |
#: |  4,096 |  4,096 | 39.9 ms |
#: | 16,384 | 16,384 |156.1 ms |
#: | 32,768 | 32,768 |320.4 ms |
#: | 65,536 | 65,536 |657.8 ms |
#:
#: ...linear, about 10us/band -- so 65,536 bands alone, whatever the profile,
#: already measures within noise of a second. The ceiling sits at half that,
#: comfortably under the point (~100,000 bands) where "well under a second"
#: stops being true, the same margin ``ops_topo.MAX_BRIDGED_RING`` keeps
#: under its own measured stall point. The UI's own ``steps`` Param caps at
#: 256 either way, so this closes a latent hazard (an agent, or a future UI
#: control, asking for more steps) without touching anything reachable today.
MAX_SPIN_BANDS = 32_768


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
    """Refuse on ``n_bands`` and the ``n_bands * n_pairs`` quad count
    separately -- see :data:`MAX_SPIN_BANDS`'s own comment for why a single
    product ceiling cannot see the band-heavy case on its own.
    """
    if n_bands > MAX_SPIN_BANDS:
        raise OpError(
            f"{what} this profile would walk {n_bands:,} steps, past the "
            f"{MAX_SPIN_BANDS:,} it works with before stalling the frame it "
            "runs on, whatever the profile's own length. Spin fewer steps."
        )
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
