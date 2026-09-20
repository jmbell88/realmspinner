"""The geometry behind each modifier kind in :mod:`.modifiers`.

Split out of that module for the same reason ``ops_clean``/``ops_subdiv``/
``ops_bevel`` are split out of ``document.py``: this is pure mesh algebra --
``Mesh -> Mesh`` (or ``Mesh, params -> Mesh``) -- with no notion of an object,
a document, a stack or a cache, and :mod:`.modifiers` is the layer that knows
about those. Every function here trusts its ``params`` dict to already be
coerced and clamped, because :func:`.modifiers.make`/:func:`.modifiers.
with_params` are the *only* doors a param value comes through on its way into
a stack -- there is no second place range-checking could be skipped.

Every kind reuses an existing kernel rather than re-deriving one:
:func:`.ops_clean.merge_by_distance` welds, :func:`.ops_subdiv.catmull_clark`
smooths, :func:`.ops_bevel.bevel_edges` bevels, and mirror/array/radial-array
share one small concatenation helper (:func:`_concat`) that is
:func:`.ops.join`'s own concatenation with the cross-object frame transform
and the automatic weld both removed -- every copy here already lives in one
object's local space, and whether to weld at all is each kind's own
parameter, not a rule this module imposes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from ..geom3d import math3d as m3
from . import elements as el
from . import mesh as bm
from . import ops_bevel, ops_clean, ops_subdiv
from .adjacency import adjacency
from .earclip import corner_triangles
from .mesh import Mesh

__all__ = [
    "array_linear",
    "array_radial",
    "bevel_by_angle",
    "laplacian_smooth",
    "mirror",
    "solidify",
    "subdivide",
    "triangulate",
    "weld",
]


# --- shared growth guard ------------------------------------------------


def _triangle_count(mesh: Mesh) -> int:
    """The triangle count :func:`.mesh.triangulate` would produce.

    The same ``starts[i+1] - starts[i] - 2`` sum ``serialize.read_wblk`` counts
    a loaded document's meshes with, kept in step deliberately: a modifier
    stack is exactly the other way triangles can arrive past the ceiling that
    matters to the same downstream consumers (the viewport, the exporter).
    """
    counts = np.diff(mesh.starts).astype("i8") - 2
    return int(np.clip(counts, 0, None).sum())


def _refuse_growth(verb: str, predicted: int) -> None:
    """Refuse before or after the fact, from a triangle count either way.

    Called *before* an allocation that scales with a parameter Clay does not
    otherwise bound (array/radial-array's ``count``, up to 200), and again,
    centrally, by :mod:`.modifiers` after every kind's ``apply`` runs -- the
    backstop for the kinds that have no growth parameter of their own to
    pre-check (mirror, solidify) and a second net under the ones that do.
    """
    from .glbimport import MAX_TRIANGLES

    if predicted > MAX_TRIANGLES:
        raise el.OpError(
            f"{verb} would make {predicted:,} triangles, past the "
            f"{MAX_TRIANGLES:,} Clay works with."
        )


# --- concatenation --------------------------------------------------------


def _concat(meshes: Sequence[Mesh]) -> Mesh:
    """Several meshes sharing one local frame, as one CSR mesh.

    :func:`.ops.join`'s own concatenation, minus the cross-object frame
    transform (every copy here is already expressed in the object's own local
    space -- there is only one object) and minus the automatic weld (each
    kind's own ``weld``/``distance`` parameter decides that, or there is none
    to decide with at all, as for radial-array).
    """
    offsets = np.cumsum([0] + [len(m.positions) for m in meshes[:-1]])
    corner_offsets = np.cumsum([0] + [len(m.loops) for m in meshes[:-1]])
    keep_uv = any(m.uv is not None for m in meshes)
    return Mesh(
        positions=np.concatenate([m.positions for m in meshes]),
        loops=np.concatenate(
            [m.loops.astype("i8") + off for m, off in zip(meshes, offsets, strict=True)]
        ),
        starts=np.concatenate(
            [
                m.starts[:-1].astype("i8") + off
                for m, off in zip(meshes, corner_offsets, strict=True)
            ]
            + [[len(meshes[-1].loops) + corner_offsets[-1]]]
        ),
        material=np.concatenate([m.material for m in meshes]),
        smooth=np.concatenate([m.smooth for m in meshes]),
        uv=(
            np.concatenate(
                [
                    m.uv if m.uv is not None else np.zeros((len(m.loops), 2), dtype="f4")
                    for m in meshes
                ]
            )
            if keep_uv
            else None
        ),
    )


# --- mirror / array / radial-array ----------------------------------------


def mirror(mesh: Mesh, params: dict) -> Mesh:
    """The base plus its reflection across the local plane through the origin.

    :func:`.mesh.transformed` reverses the copy's loops for us (the module's
    own rule: any transform with a negative determinant is mirrored *and*
    rewound, never left inside-out), so the concatenation only has to weld the
    seam -- which is what ``weld`` names, defaulted on, unlike every other
    kind's weld-style parameter, because an unwelded mirror seam is the
    common defect and a seam this small is never intentional geometry.
    """
    axis = int(params.get("axis", 0))
    weld_distance = float(params.get("weld", 0.0001))
    matrix = np.eye(4)
    matrix[axis, axis] = -1.0
    reflected = bm.transformed(mesh, matrix)
    merged = _concat([mesh, reflected])
    if weld_distance > 0.0:
        merged = ops_clean.merge_by_distance(merged, weld_distance)
    return merged


def array_linear(mesh: Mesh, params: dict) -> Mesh:
    """``count`` copies, copy *k* translated by ``k * offset``, concatenated."""
    count = max(1, int(params.get("count", 3)))
    offset = np.array(
        [
            float(params.get("offset_x", 1.0)),
            float(params.get("offset_y", 0.0)),
            float(params.get("offset_z", 0.0)),
        ],
        dtype="f8",
    )
    weld_distance = float(params.get("weld", 0.0))
    _refuse_growth("Arraying this object", _triangle_count(mesh) * count)
    copies = [
        mesh if k == 0 else replace(mesh, positions=mesh.positions.astype("f8") + offset * k)
        for k in range(count)
    ]
    merged = _concat(copies)
    if weld_distance > 0.0:
        merged = ops_clean.merge_by_distance(merged, weld_distance)
    return merged


def _closes_a_ring(angle: float) -> bool:
    """:func:`~.studio.modes.clay.ops._closes_a_ring`'s own rule, duplicated.

    Kernels may not import ``studio`` -- the layering ``tests/test_layering.py``
    enforces runs the other way -- so the rule that decides whether a sweep
    divides by ``count`` or by ``count - 1`` is repeated here rather than
    shared. See that function's docstring for the reasoning; it is not
    restated because nothing about it changed crossing the boundary.
    """
    remainder = abs(float(angle)) % 360.0
    return remainder < 1e-6 or remainder > 360.0 - 1e-6


def array_radial(mesh: Mesh, params: dict) -> Mesh:
    """``count`` copies spun about the *local* origin, closed-ring aware.

    Same divisor rule as the object-level radial array op: a sweep that closes
    back on itself spaces every copy, the original included, evenly around the
    whole turn (divide by ``count``); an open arc reaches its far end exactly
    (divide by ``count - 1``). See :func:`_closes_a_ring`.
    """
    count = max(1, int(params.get("count", 6)))
    angle = float(params.get("angle", 360.0))
    axis = int(params.get("axis", 1))
    _refuse_growth("Arraying this object", _triangle_count(mesh) * count)
    if count <= 1:
        return mesh
    divisor = count if _closes_a_ring(angle) else count - 1
    axis_vec = np.zeros(3, dtype="f8")
    axis_vec[axis] = 1.0
    copies = []
    for k in range(count):
        degrees = k * angle / divisor if divisor else 0.0
        if degrees == 0.0:
            copies.append(mesh)
            continue
        quat = m3.quat_from_axis_angle(axis_vec, math.radians(degrees))
        matrix = m3.compose(m3.vec3(), quat, m3.vec3(1.0, 1.0, 1.0))
        copies.append(bm.transformed(mesh, matrix))
    return _concat(copies)


# --- solidify ---------------------------------------------------------------


def _vertex_normals(mesh: Mesh) -> np.ndarray:
    """Area-weighted per-vertex normals, ignoring the ``smooth`` flag.

    A shell offset wants one consistent outward direction per vertex, not the
    split-by-shading-group answer :func:`.mesh.render_arrays` computes for the
    GPU -- so this is its own small accumulation rather than a reuse of that
    one, built the same way (:func:`.mesh.accumulate` over each corner's own
    face normal) for the reason its own docstring gives: unbuffered ``np.
    add.at`` is an order of magnitude slower.
    """
    raw = bm.face_normals(mesh).astype("f8")
    counts = np.diff(mesh.starts).astype("i8")
    per_corner = np.repeat(raw, counts, axis=0)
    accumulated = bm.accumulate(mesh.loops, per_corner, len(mesh.positions))
    lengths = np.linalg.norm(accumulated, axis=1, keepdims=True)
    return np.divide(accumulated, lengths, out=np.zeros_like(accumulated), where=lengths > 1e-12)


def solidify(mesh: Mesh, params: dict) -> Mesh:
    """A shell: the surface offset along vertex normals, plus a reversed copy
    offset the other way, plus a rim quad along every open edge.

    ``offset`` places the *original* surface between the two new ones rather
    than moving it: ``-1`` keeps it as the outer wall and grows a new inner
    one, ``0`` centres a wall of ``thickness`` on it, ``1`` keeps it as the
    inner wall and grows a new outer one. Both new positions are therefore
    ``thickness * (offset +- 1) / 2`` -- the formula, not two branches, because
    every value in between is a real, continuous answer, not a choice among
    three cases.

    The rim quad per open edge is wound ``[outer_b, outer_a, inner_a,
    inner_b]`` for an original edge read ``a -> b``: checked numerically in
    the tests (a single quad solidifies into a closed box, every face normal
    pointing off the box) rather than argued here.
    """
    thickness = float(params.get("thickness", 0.05))
    offset = float(params.get("offset", -1.0))
    if bm.face_count(mesh) == 0:
        return mesh
    outer_amount = thickness * (offset + 1.0) / 2.0
    inner_amount = thickness * (offset - 1.0) / 2.0
    normals = _vertex_normals(mesh)
    positions = mesh.positions.astype("f8")
    outer = replace(mesh, positions=positions + normals * outer_amount)
    perm = bm.reversed_corner_perm(mesh.starts)
    inner = Mesh(
        positions=positions + normals * inner_amount,
        loops=mesh.loops[perm],
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
        uv=None if mesh.uv is None else mesh.uv[perm],
    )
    n = len(mesh.positions)
    shells = _concat([outer, inner])

    a = adjacency(mesh)
    boundary_corners = np.flatnonzero(a.edge_uses[a.corner_edge] == 1)
    if len(boundary_corners) == 0:
        return shells  # closed mesh: no rim, per the kind's own table entry

    rim_material = []
    rim_loops: list[int] = []
    for c in boundary_corners.tolist():
        va = int(mesh.loops[c])
        vb = int(mesh.loops[a.next_corner[c]])
        rim_loops.extend((vb, va, n + va, n + vb))
        rim_material.append(int(mesh.material[int(a.corner_face[c])]))
    n_rim = len(rim_material)
    rim_loops_arr = np.array(rim_loops, dtype="i8")
    rim_starts_local = np.arange(n_rim + 1, dtype="i8") * 4
    rim_material_arr = np.array(rim_material, dtype="i4")
    rim_smooth_arr = np.zeros(n_rim, dtype=bool)
    rim_uv = np.zeros((len(rim_loops_arr), 2), dtype="f4") if shells.uv is not None else None

    return Mesh(
        positions=shells.positions,
        loops=np.concatenate([shells.loops.astype("i8"), rim_loops_arr]),
        starts=np.concatenate(
            [shells.starts[:-1].astype("i8"), rim_starts_local + len(shells.loops)]
        ),
        material=np.concatenate([shells.material, rim_material_arr]),
        smooth=np.concatenate([shells.smooth, rim_smooth_arr]),
        uv=None if shells.uv is None else np.concatenate([shells.uv, rim_uv]),
    )


# --- bevel / subdivide / weld / triangulate / smooth ------------------------


def bevel_by_angle(mesh: Mesh, params: dict) -> Mesh:
    """Bevel every interior edge whose dihedral angle exceeds ``angle``.

    "Interior" excludes a boundary edge (the table's own rule -- there is no
    second face to measure an angle against) and, one step further, a
    non-manifold or a flipped-winding edge: both also leave
    :attr:`.adjacency.Adjacency.twin` at -1, and both have no *single* other
    face to compare against either (three-or-more meeting, or a pair that
    disagrees about which way the shared edge runs), so there is no dihedral
    angle to measure any more than there is on a boundary. Left in, either
    would reach :func:`.ops_bevel.bevel_edges` and be refused there for a
    reason that has nothing to do with what this parameter means.

    No sharp edge over the threshold is not an error -- an unbeveled mesh is
    a legitimate answer to "nothing here is sharp enough."
    """
    width = float(params.get("width", 0.02))
    threshold = float(params.get("angle", 30.0))
    if bm.face_count(mesh) == 0:
        return mesh
    a = adjacency(mesh)
    normals = bm.face_normals(mesh)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    unit = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)
    paired = np.flatnonzero(a.twin >= 0)
    if len(paired) == 0:
        return mesh
    left = a.corner_face[paired].astype("i8")
    right = a.corner_face[a.twin[paired]].astype("i8")
    dots = np.clip(np.einsum("ij,ij->i", unit[left], unit[right]), -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    sharp_corners = paired[angles > threshold]
    if len(sharp_corners) == 0:
        return mesh
    edge_ids = np.unique(a.corner_edge[sharp_corners])
    sel = el.ElementSel(edges=a.edge_verts[edge_ids])
    result, _sel = ops_bevel.bevel_edges(mesh, sel, width=width)
    return result


def subdivide(mesh: Mesh, params: dict) -> Mesh:
    levels = int(params.get("levels", 1))
    result, _sel = ops_subdiv.catmull_clark(mesh, el.empty(), levels=levels)
    return result


def weld(mesh: Mesh, params: dict) -> Mesh:
    distance = float(params.get("distance", 0.0001))
    if distance <= 0.0:
        return mesh
    return ops_clean.merge_by_distance(mesh, distance)


def triangulate(mesh: Mesh, params: dict) -> Mesh:
    """Every n-gon fanned or ear-clipped, whichever rendering itself uses.

    :func:`.mesh.triangulate` only ever hands back *vertex* indices, having
    already thrown its own corner indices away -- fine for a renderer that
    reads positions, wrong for this kind, which also has to carry each
    triangle's per-corner uv. So this calls :func:`.earclip.corner_triangles`
    directly, the one call :func:`.mesh.triangulate` itself wraps, and keeps
    the corner indices to gather ``uv`` with as well as ``loops``.
    """
    del params
    if bm.face_count(mesh) == 0:
        return mesh
    corners, tri_face = corner_triangles(
        mesh.positions, mesh.loops, mesh.starts, bm.face_normals(mesh)
    )
    if len(corners) == 0:
        return mesh
    tris = mesh.loops[corners]
    n_tris = len(tris)
    return Mesh(
        positions=mesh.positions,
        loops=tris.reshape(-1),
        starts=np.arange(n_tris + 1, dtype="i4") * 3,
        material=mesh.material[tri_face],
        smooth=mesh.smooth[tri_face],
        uv=None if mesh.uv is None else mesh.uv[corners].reshape(-1, 2),
    )


def laplacian_smooth(mesh: Mesh, params: dict) -> Mesh:
    """Vertex positions averaged toward their neighbours; topology untouched.

    A boundary vertex is held fixed -- an unconstrained Laplacian pulls an open
    border inward (the same crease ``catmull_clark`` names for exactly this
    reason), and this kind is not the smoothing subdivision, which has its own
    boundary rule and its own topology change.
    """
    factor = float(params.get("factor", 0.5))
    iterations = int(params.get("iterations", 1))
    if bm.face_count(mesh) == 0 or factor <= 0.0 or iterations <= 0:
        return mesh
    a = adjacency(mesh)
    if a.n_edges == 0:
        return mesh
    ev = a.edge_verts.astype("i8")
    u, v = ev[:, 0], ev[:, 1]
    n = len(mesh.positions)
    boundary_edge = a.edge_uses == 1
    is_boundary = np.zeros(n, dtype=bool)
    if boundary_edge.any():
        is_boundary[ev[boundary_edge].reshape(-1)] = True
    idx = np.concatenate([u, v]).astype("i4")
    degree = np.bincount(idx, minlength=n).astype("f8")
    movable = (~is_boundary) & (degree > 0.0)
    if not movable.any():
        return mesh
    positions = mesh.positions.astype("f8").copy()
    for _ in range(iterations):
        vals = np.concatenate([positions[v], positions[u]])
        sums = bm.accumulate(idx, vals, n)
        avg = np.divide(sums, degree[:, None], out=np.zeros_like(sums), where=degree[:, None] > 0.0)
        positions[movable] = positions[movable] * (1.0 - factor) + avg[movable] * factor
    return replace(mesh, positions=positions)
