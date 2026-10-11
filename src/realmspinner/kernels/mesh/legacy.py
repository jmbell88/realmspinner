"""Opening a ``.rblk`` from before format 4: bake the old stack, drop what Clay lost.

Format 3 documents carried things Clay no longer models -- a non-destructive
modifier stack on each object, collider objects, seams, tags and locks -- and a
file written by 0.0.51 or earlier must keep opening (INVARIANTS: a ``.rblk``
Clay wrote must reopen). This module is the **one place those survive**, and
only as a one-way migration: :func:`read_fields` lifts the dead fields off each
object entry as the reader walks them, and :func:`migrate` then turns the
document into the format-4 shape in a single pass and returns the sentences the
mode shows the user.

* **Modifier stacks are baked.** Each object's base mesh is run through its
  enabled modifiers, in order, exactly as the old evaluator did, and the result
  becomes ``Obj.mesh``. A boolean modifier's target is its *own evaluated* mesh,
  so the targets are resolved first (an explicit post-order, never Python
  recursion: a chain of 1,500 targets is inside ``glbimport.MAX_OBJECTS``). An
  object whose bake changed its mesh is frozen -- no generator, no params -- the
  claim "this is a box, size 1" stopped being true. A disabled modifier never
  contributed to what the user saw and is dropped unapplied. **A modifier that
  cannot run is dropped, not refused**: the old evaluator skipped it and carried
  on with the next modifier over whatever the previous one produced, and the bake
  does the same, naming it in the notice.
* **Collider objects are dropped**, children lifted onto the collider's own
  parent keeping their place in the world. Seams, tags and locks are discarded.
* **An object whose generator Clay's Add palette no longer offers**
  (:data:`~.primitives.CLAY_GENERATOR_NAMES`) is frozen the same way -- the mesh
  stays, the params go.

The kernels below (``mirror`` through ``laplacian_smooth``) are the old
``ops_modifiers`` module, relocated whole: they are the only thing that knows
how to reproduce an old document's shape, so they live here and nowhere else.
Nothing outside :mod:`.serialize` imports this module.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ...core.safeio import npyguard
from ..geom3d import math3d as m3
from . import elements as el
from . import mesh as bm
from . import ops_bevel, ops_boolean, ops_clean, ops_subdiv
from .adjacency import adjacency
from .document import ClayDoc, Obj
from .earclip import corner_triangles
from .mesh import Mesh
from .primitives import CLAY_GENERATOR_NAMES, F4_MAX, GENERATORS

__all__ = ["LegacyFields", "freeze_unknown_generators", "migrate", "read_fields"]



# --- the old modifier kernels, relocated whole ----------------------------


# --- shared growth guard ------------------------------------------------


def _triangle_count(mesh: Mesh) -> int:
    """The triangle count :func:`.mesh.triangulate` would produce.

    The same ``starts[i+1] - starts[i] - 2`` sum ``serialize.read_rblk`` counts
    a loaded document's meshes with, kept in step deliberately: a modifier
    stack is exactly the other way triangles can arrive past the ceiling that
    matters to the same downstream consumers (the viewport, the exporter).
    """
    counts = np.diff(mesh.starts).astype("i8") - 2
    return int(np.clip(counts, 0, None).sum())


def _refuse_growth(verb: str, predicted: int, limit: int | None = None) -> None:
    """Refuse before or after the fact, from a triangle count either way.

    Called *before* an allocation that scales with a parameter Clay does not
    otherwise bound (array/radial-array's ``count``, up to 200), and again,
    centrally, by :func:`_bake` after every kind's apply runs -- the backstop
    for the kinds that have no growth parameter of their own to pre-check
    (mirror, solidify) and a second net under the ones that do.

    *limit* is the room the *document* has left for this one mesh (see
    :func:`_bake`): the 2026-10-07 audit's clay-05 found the ceiling was
    charged per modifier only, so a file of a few small objects each baking to
    just under it opened as millions of triangles that Save then refused.
    ``None`` is the whole ceiling.
    """
    from .glbimport import MAX_TRIANGLES

    ceiling = MAX_TRIANGLES if limit is None else min(limit, MAX_TRIANGLES)
    if predicted > ceiling:
        if ceiling == MAX_TRIANGLES:
            raise el.OpError(
                f"{verb} would make {predicted:,} triangles, past the "
                f"{MAX_TRIANGLES:,} Clay works with."
            )
        raise el.OpError(
            f"{verb} would make {predicted:,} triangles, past the {max(ceiling, 0):,} "
            f"this model has room for ({MAX_TRIANGLES:,} is the most Clay works with "
            "in all)."
        )


def _largest_array_bytes(vertices: int, corners: int, has_uv: bool) -> int:
    """The byte size of the biggest array a mesh of this shape stores: positions
    are ``(V, 3)`` f4, a corner array is i4 (or ``(L, 2)`` f4 for the uv)."""
    return max(vertices * 12, corners * (8 if has_uv else 4))


def _refuse_vertex_growth(verb: str, mesh: Mesh, vertices: int, corners: int) -> None:
    """Refuse a result of *vertices* and *corners* that Clay cannot hold or save.

    The 2026-10-07 audit's clay-25: the guard above counts triangles only, and a
    triangle ceiling says nothing about the vertices a mesh carries without
    referencing them -- one triangle over 100,000 unreferenced vertices times an
    array count of 200 became 20,000,000 vertices with no refusal, and a smaller
    multiple could bake a positions array past ``npyguard.MAX_ARRAY_BYTES``,
    which the writer accepted and the reader then refused to reopen. Called
    *before* any copy is built (the array kinds) and again on every result
    (:func:`_bake`). A mesh that is already over a ceiling when it arrives is
    not charged for staying that size, only for growing.
    """
    from .glbimport import MAX_VERTICES

    had = len(mesh.positions)
    if vertices > MAX_VERTICES and vertices > had:
        raise el.OpError(
            f"{verb} would make {vertices:,} vertices, past the "
            f"{MAX_VERTICES:,} Clay works with."
        )
    has_uv = mesh.uv is not None
    ceiling = npyguard.MAX_ARRAY_BYTES
    if _largest_array_bytes(vertices, corners, has_uv) > ceiling and (
        _largest_array_bytes(vertices, corners, has_uv)
        > _largest_array_bytes(had, len(mesh.loops), has_uv)
    ):
        raise el.OpError(
            f"{verb} would make a mesh array past the {ceiling:,} bytes a saved "
            "document may hold."
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


def array_linear(mesh: Mesh, params: dict, limit: int | None = None) -> Mesh:
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
    _refuse_growth("Arraying this object", _triangle_count(mesh) * count, limit)
    _refuse_vertex_growth(
        "Arraying this object", mesh, len(mesh.positions) * count, len(mesh.loops) * count
    )
    copies = [
        mesh if k == 0 else replace(mesh, positions=mesh.positions.astype("f8") + offset * k)
        for k in range(count)
    ]
    merged = _concat(copies)
    if weld_distance > 0.0:
        merged = ops_clean.merge_by_distance(merged, weld_distance)
    return merged


def _closes_a_ring(angle: float) -> bool:
    """Whether a sweep of *angle* degrees closes into a ring, so the steps divide
    by ``count`` rather than ``count - 1``."""
    remainder = abs(float(angle)) % 360.0
    return remainder < 1e-6 or remainder > 360.0 - 1e-6


def array_radial(mesh: Mesh, params: dict, limit: int | None = None) -> Mesh:
    """``count`` copies spun about the *local* origin, closed-ring aware.

    Same divisor rule as the object-level radial array op: a sweep that closes
    back on itself spaces every copy, the original included, evenly around the
    whole turn (divide by ``count``); an open arc reaches its far end exactly
    (divide by ``count - 1``). See :func:`_closes_a_ring`.
    """
    count = max(1, int(params.get("count", 6)))
    angle = float(params.get("angle", 360.0))
    axis = int(params.get("axis", 1))
    _refuse_growth("Arraying this object", _triangle_count(mesh) * count, limit)
    _refuse_vertex_growth(
        "Arraying this object", mesh, len(mesh.positions) * count, len(mesh.loops) * count
    )
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

#: The largest number of boundary corners :func:`solidify` will build a rim
#: for. The 2026-09-20 audit's clay-18 found the rim-building Python loop
#: (one iteration per boundary corner, minting a wound quad and reading its
#: owning face's material) had no ceiling of its own -- the only check,
#: :func:`_refuse_growth`, runs *after* ``apply()`` has already built the
#: whole result, which is too late for a cost that lives in the loop itself
#: rather than in the triangle count it produces.
#:
#: Measured on this machine with a mesh of disjoint quads (every edge a
#: boundary edge, so ``boundary_corners`` is exactly ``4 * quad count`` and
#: isolated from any interior-mesh cost the way a solidified surface's own
#: rim rarely is):
#:
#: | boundary corners | solidify() |
#: |------------------:|-----------:|
#: |            200,000 |    233 ms |
#: |            300,000 |    357 ms |
#: |            400,000 |    475 ms |
#: |            500,000 |    584 ms |
#:
#: ...linear, about 1.2us/corner, matching the audit's own ~1.1us/corner
#: closely. Set at 400,000 -- comfortably under the ~850,000-corner point
#: where that rate would cross a second, the same "well under a second" bar
#: every sibling ceiling in this package uses.
MAX_SOLIDIFY_RIM_CORNERS = 400_000


def _refuse_solidify_rim(n_corners: int) -> None:
    """Refuse before the rim loop below runs -- see
    :data:`MAX_SOLIDIFY_RIM_CORNERS` for the measurements.
    """
    if n_corners > MAX_SOLIDIFY_RIM_CORNERS:
        raise el.OpError(
            f"Solidifying this mesh means building a rim of {n_corners:,} "
            f"boundary corners, past the {MAX_SOLIDIFY_RIM_CORNERS:,} "
            "solidify works with before it would stall the frame it runs "
            "on. Solidify a mesh with a smaller open boundary."
        )


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

    # Found and refused before building either shell -- not just before the
    # rim loop below -- so a mesh past the ceiling does not first pay for two
    # full copies of itself it will never get to use. See
    # MAX_SOLIDIFY_RIM_CORNERS for why _refuse_growth's own post-apply check
    # (below, in :func:`_bake`) is too late to protect this loop.
    a = adjacency(mesh)
    boundary_corners = np.flatnonzero(a.edge_uses[a.corner_edge] == 1)
    _refuse_solidify_rim(len(boundary_corners))

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


#: The dihedral angle below which an edge is flat to float32 precision
#: (:func:`bevel_by_angle`'s noise floor).
_FLAT_EDGE_NOISE_DEGREES = 0.1


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
    # A floor under the threshold: the dihedral comes from float32 positions
    # through arccos, which turns a one-ulp error in a dot near 1 into a few
    # hundredths of a degree, so at ``angle = 0`` (the parameter's own minimum)
    # every coplanar edge read as sharper than the threshold and was beveled
    # (the 2026-10-03 audit's clay-92: a flat grid tilted 0.3 rad went from 16
    # faces to 49). Below this an edge is flat, whatever was asked.
    sharp_corners = paired[angles > max(threshold, _FLAT_EDGE_NOISE_DEGREES)]
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


# --- the old vocabulary, read-only ---------------------------------------


@dataclass(frozen=True)
class _Param:
    """One parameter's shape. ``target=True`` marks a stored object uid."""

    name: str
    default: float
    low: float = 0.0
    high: float = 1e6
    integer: bool = False
    choices: tuple[str, ...] = ()
    target: bool = False


@dataclass(frozen=True)
class _Modifier:
    id: int
    kind: str
    params: tuple[tuple[str, float | int | bool], ...]
    enabled: bool = True

    def get(self, name: str, default: Any = None) -> Any:
        for key, value in self.params:
            if key == name:
                return value
        return default

    def as_dict(self) -> dict[str, float | int | bool]:
        return dict(self.params)


#: kind -> (label, parameters, ``(mesh, params) -> mesh``). The labels are what a
#: notice names; there is no menu order to keep now that nothing offers one.
_KINDS: dict[str, tuple[str, tuple[_Param, ...], Any]] = {
    "mirror": (
        "Mirror",
        (_Param("axis", 0.0, choices=("X", "Y", "Z")), _Param("weld", 0.0001, 0.0, 1.0)),
        mirror,
    ),
    "array": (
        "Array",
        (
            _Param("count", 3.0, 2.0, 200.0, integer=True),
            _Param("offset_x", 1.0, -1e4, 1e4),
            _Param("offset_y", 0.0, -1e4, 1e4),
            _Param("offset_z", 0.0, -1e4, 1e4),
            _Param("weld", 0.0, 0.0, 1.0),
        ),
        array_linear,
    ),
    "radial-array": (
        "Radial Array",
        (
            _Param("count", 6.0, 2.0, 200.0, integer=True),
            _Param("angle", 360.0, -3600.0, 3600.0),
            _Param("axis", 1.0, choices=("X", "Y", "Z")),
        ),
        array_radial,
    ),
    "solidify": (
        "Solidify",
        (_Param("thickness", 0.05, 0.0, 10.0), _Param("offset", -1.0, -1.0, 1.0)),
        solidify,
    ),
    "bevel": (
        "Bevel",
        (_Param("width", 0.02, 0.0, 10.0), _Param("angle", 30.0, 0.0, 180.0)),
        bevel_by_angle,
    ),
    "subdivide": ("Subdivide", (_Param("levels", 1.0, 1.0, 3.0, integer=True),), subdivide),
    "weld": ("Weld", (_Param("distance", 0.0001, 0.0, 1.0),), weld),
    "triangulate": ("Triangulate", (), triangulate),
    "smooth": (
        "Smooth",
        (_Param("factor", 0.5, 0.0, 1.0), _Param("iterations", 1.0, 1.0, 50.0, integer=True)),
        laplacian_smooth,
    ),
    "boolean": (
        "Boolean",
        (
            _Param("target", 0.0, 0.0, 1e9, target=True),
            _Param(
                "operation",
                float(ops_boolean.KINDS.index("difference")),
                choices=ops_boolean.KINDS,
            ),
        ),
        None,  # resolved by the bake: it needs the target's evaluated mesh
    ),
}


def _coerce(p: _Param, raw: Any) -> float | int | bool:
    """*raw* into the shape *p* declares: a choice index, or a clamped number."""
    if p.choices:
        if isinstance(raw, str):
            try:
                return p.choices.index(raw)
            except ValueError:
                raise ValueError(f"{p.name} must be one of {', '.join(p.choices)}") from None
        idx = int(raw)
        if not 0 <= idx < len(p.choices):
            raise ValueError(f"{p.name} index {idx} is out of range")
        return idx
    number = float(raw)
    if not math.isfinite(number):
        # The 2026-10-07 audit's clay-54: ``max(low, min(high, nan))`` is the
        # maximum, so a stored NaN loaded silently at the top of the range.
        raise ValueError(f"{p.name} is not a finite number")
    value = max(p.low, min(p.high, number))
    return int(round(value)) if p.integer else value


def _parse_modifier(item: Any) -> _Modifier:
    """One stored modifier, rebuilt the way the old ``make`` did: a value past a
    parameter's range loads at the clamp. Raises ``ValueError`` for anything this
    migration cannot run -- the caller drops it and says so."""
    if not isinstance(item, dict):
        raise ValueError("it is malformed")
    try:
        mid = int(item["id"])
        kind = str(item["kind"])
        params = item.get("params") or {}
        enabled = bool(item.get("enabled", True))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("it is malformed") from exc
    if not isinstance(params, dict):
        raise ValueError("its parameters are malformed")
    entry = _KINDS.get(kind)
    if entry is None:
        raise ValueError(f"{kind!r} is not a modifier this version knows")
    given = dict(params)
    declared = {p.name for p in entry[1]}
    unknown = sorted(set(given) - declared)
    if unknown:
        raise ValueError(f"it has no parameter {unknown[0]!r}")
    try:
        out = {p.name: _coerce(p, given.get(p.name, p.default)) for p in entry[1]}
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"its parameters are malformed ({exc})") from exc
    return _Modifier(id=mid, kind=kind, params=tuple(sorted(out.items())), enabled=enabled)


# --- what the reader lifts off each object entry --------------------------


@dataclass
class LegacyFields:
    """The format-3 fields of one object entry that format 4 no longer has."""

    modifiers: tuple[_Modifier, ...] = ()
    #: One sentence for every stored modifier (or modifier list) not rebuilt.
    unreadable: list[str] = field(default_factory=list)
    role: str = "mesh"
    has_seams: bool = False
    has_tags: bool = False
    locked: bool = False


def read_fields(entry: dict[str, Any]) -> LegacyFields:
    """Lift an object entry's dead fields. Never raises: a hand-edited field of
    the wrong shape is dropped, and named in the notice."""
    out = LegacyFields()
    raw = entry.get("modifiers")
    if isinstance(raw, list):
        seen: set[int] = set()
        kept: list[_Modifier] = []
        for item in raw:
            try:
                built = _parse_modifier(item)
                if built.id in seen:
                    raise ValueError(f"it shares id {built.id} with another")
            except ValueError as exc:
                label = item.get("kind", "a") if isinstance(item, dict) else "a"
                out.unreadable.append(f"the {label} modifier was dropped: {exc}")
                continue
            seen.add(built.id)
            kept.append(built)
        out.modifiers = tuple(kept)
    elif raw is not None:
        out.unreadable.append("its modifier list was malformed and was dropped")
    out.role = "collider" if entry.get("role") == "collider" else "mesh"
    out.has_seams = bool(entry.get("seams"))
    out.has_tags = bool(entry.get("tags"))
    out.locked = bool(entry.get("locked", False))
    return out


# --- the bake ----------------------------------------------------------------


#: The most work the bake spends on one file, in *smooth passes*: one pass is
#: the cost of one Laplacian-smooth iteration over one triangle, 0.15 us. The
#: 2026-10-07 audit's probe: a 50-iteration smooth over 90,000 quads (180,000
#: triangles, 9M passes) took 1.3 s, so 10,000,000 passes is on the order of a
#: second and a half -- the "well under a few seconds" bar the sibling ceilings
#: in this module use. A new refusal, not a constant any stored file is keyed on:
#: the v3 fixtures under ``tests/modes/clay/data`` still convert with no budget
#: notice.
#:
#: **The unit was calibrated on ``smooth`` alone, and every other kind was
#: charged as if it cost the same.** The 2026-10-09 second audit's clay-85: forty
#: tiny booleans against a 199k-triangle target took 21 s to open against the
#: claimed second and a half, and mirror, array, solidify and subdivide run 5 to
#: 300 times a smooth's cost per triangle. :data:`BAKE_WEIGHTS` charges each kind
#: its measured cost in smooth passes per triangle, so the one budget really is
#: about 1.5 s. Measured 2026-10-09 on this machine (the maintainer's, a warm
#: process, best of three) over UV spheres of 1.1k to 261k triangles, the worst
#: figure per kind in microseconds per input triangle, then divided by 0.15:
#:
#:   kind          worst us/triangle     charged, in passes per triangle
#:   smooth         0.19 (1 iteration)    1.3 per iteration
#:   radial-array   0.43 (count 6)        1 per copy
#:   bevel          0.34                  3
#:   solidify       0.57                  4
#:   weld           0.61                  5
#:   triangulate    0.81                  6
#:   subdivide      2.0 / 8.8 / 47        14 / 60 / 320 for levels 1 / 2 / 3
#:   mirror         3.1 (weld on)         20; 1 with no weld
#:   array          4.4 (weld on)         30 + count; count with no weld
#:   boolean        2.5 (owner + target)  18 per triangle of owner plus target
#:
#: (The charges sit at or just above the measurement everywhere: the small
#: meshes run several times dearer per triangle than the large ones, so the
#: table errs toward refusing a file early rather than holding the window open.)
MAX_BAKE_WORK = 10_000_000

#: Smooth passes charged per input triangle, by kind. ``smooth`` is per
#: iteration; ``subdivide`` is by level (below); ``array``, ``radial-array`` and
#: ``mirror`` have a cheaper weld-free row (see :func:`_bake_cost`).
BAKE_WEIGHTS: dict[str, float] = {
    "smooth": 1.3,
    "radial-array": 1.0,
    "bevel": 3.0,
    "solidify": 4.0,
    "weld": 5.0,
    "triangulate": 6.0,
    "mirror": 20.0,
    "array": 30.0,
    "boolean": 18.0,
}
#: ``subdivide`` costs more than linearly in its levels: each level quadruples
#: the faces the next one runs over.
SUBDIVIDE_WEIGHTS: dict[int, float] = {1: 14.0, 2: 60.0, 3: 320.0}

#: The least one modifier is charged, in smooth passes, whatever it runs over.
#: **The per-triangle table cannot see a fixed cost.** The 2026-10-11 audit's
#: clay-09: a modifier over a 12-triangle box costs 60 to 2,400 us no matter how
#: few triangles it has (numpy setup, adjacency, the loop's own growth checks), and
#: the table charged it 6 to 36 passes, about a microsecond, so a hand-made file of
#: 2,000 tiny weld, triangulate or smooth modifiers measured 172, 135 and 119 us
#: each while the budget admitted 166,666 to 641,025 of them -- 19 to 76 s to open
#: one file, on a task with no cancel. A floor, not a surcharge: a modifier costs
#: about the fixed figure until its per-triangle charge overtakes it, so a large
#: mesh is priced exactly as before. Measured 2026-10-11 on this machine (a warm
#: process, best of three) as one modifier through ``_bake`` over a 12-triangle box
#: and a 48-triangle sphere, the worst of the two, in microseconds, then divided
#: by 0.15 and rounded up (the 12 and 48 triangles it already pays for are inside
#: the figure, which is why this is a floor and not an addend):
#:
#:   kind          worst us per modifier    floor, in passes
#:   weld, smooth   60 / 135                1,000 (the default)
#:   triangulate    150                     1,000
#:   solidify       270                     2,000
#:   radial-array   460                     3,000
#:   subdivide      500                     3,500
#:   mirror, array  1,070                   7,500
#:   bevel          1,350                   9,500
#:   boolean        2,400                   16,000
#:
#: Each floor admits a stack of its kind in about a second and a half or less
#: (10,000,000 over 1,000 is 10,000 smooth modifiers at 135 us; over 16,000 it is
#: 625 booleans at 2.4 ms). Bevel's *per-triangle* figure is a separate question:
#: it was measured on spheres, which have no sharp edge to bevel, and a box costs
#: it about a hundred microseconds a triangle.
BAKE_FLOOR: dict[str, float] = {
    "solidify": 2_000.0,
    "radial-array": 3_000.0,
    "subdivide": 3_500.0,
    "mirror": 7_500.0,
    "array": 7_500.0,
    "bevel": 9_500.0,
    "boolean": 16_000.0,
}
BAKE_FLOOR_DEFAULT = 1_000.0


#: How a notice that already says what became of its modifiers ends; ``migrate``
#: leaves the "and was dropped" tail off such a sentence.
_DROPPED_WORD = "dropped"


def _bake_cost(mod: _Modifier, triangles: int, target_triangles: int = 0) -> float:
    """What running *mod* over *triangles* (plus a boolean's *target_triangles*)
    is charged against :data:`MAX_BAKE_WORK`, in smooth passes: its per-triangle
    cost, but never under its kind's :data:`BAKE_FLOOR`."""
    return max(
        _triangle_cost(mod, triangles, target_triangles),
        BAKE_FLOOR.get(mod.kind, BAKE_FLOOR_DEFAULT),
    )


def _triangle_cost(mod: _Modifier, triangles: int, target_triangles: int = 0) -> float:
    """:func:`_bake_cost` before the floor: what the modifier costs per triangle."""
    base = max(1, triangles)
    kind = mod.kind
    if kind == "smooth":
        return base * BAKE_WEIGHTS["smooth"] * int(mod.get("iterations", 1))
    if kind == "subdivide":
        levels = min(max(int(mod.get("levels", 1)), 1), max(SUBDIVIDE_WEIGHTS))
        return base * SUBDIVIDE_WEIGHTS[levels]
    if kind == "boolean":
        return (base + target_triangles) * BAKE_WEIGHTS["boolean"]
    if kind == "radial-array":
        return base * BAKE_WEIGHTS["radial-array"] * max(1, int(mod.get("count", 1)))
    welds = float(mod.get("weld", 0.0)) > 0.0
    if kind == "mirror":
        return base * (BAKE_WEIGHTS["mirror"] if welds else 1.0)
    if kind == "array":
        copies = max(1, int(mod.get("count", 1)))
        return base * (copies + (BAKE_WEIGHTS["array"] if welds else 0.0))
    return base * BAKE_WEIGHTS.get(kind, 1.0)


def _bake(
    doc: ClayDoc, stacks: dict[int, tuple[_Modifier, ...]]
) -> tuple[dict[int, Mesh], dict[int, list[str]]]:
    """Every object's base mesh run through its enabled modifiers.

    -> ``(meshes, errors)``: the result for each uid whose mesh changed, and the
    sentences for each modifier that refused. A refusing modifier is skipped and
    the next one runs over whatever the previous produced, the old evaluator's
    own rule. A boolean takes its target's *evaluated* mesh, so targets are
    visited first, in post-order with an explicit stack; a target that would
    close a cycle is refused on the modifier that closes it.

    **The whole document is bounded, not just each modifier.** The 2026-10-07
    audit's clay-05: ``read_rblk`` charges ``MAX_TRIANGLES`` against the *base*
    meshes before this runs, and the bake only ever bounded one modifier's
    result, so a 19.6 KB file of a few arrays opened as 15.5M triangles that
    Save, Save As and the journal all then refused. A running total of every
    object's current triangles rides the loop; a modifier whose result would
    push it past the ceiling is refused like any other that cannot run --
    dropped and named -- so the document this hands back is one the writer
    accepts. (Colliders are dropped afterwards and only ever lower the total.)
    """
    from .glbimport import MAX_TRIANGLES

    result: dict[int, Mesh] = {}
    errors: dict[int, list[str]] = {}
    cyclic: set[tuple[int, int]] = set()

    def targets_of(uid: int) -> list[tuple[int, int]]:
        found = []
        for mod in stacks.get(uid, ()):
            if mod.enabled and mod.kind == "boolean":
                target = int(mod.get("target", 0))
                if target:
                    found.append((mod.id, target))
        return found

    present = {o.uid for o in doc.objects}
    white, gray, black = 0, 1, 2
    color = dict.fromkeys(present, white)
    order: list[int] = []
    for start in [o.uid for o in doc.objects]:
        if color[start] != white:
            continue
        color[start] = gray
        frames = [(start, iter(targets_of(start)))]
        while frames:
            node, edges = frames[-1]
            descended = False
            for mid, target in edges:
                if target not in present:
                    continue
                if color[target] == gray:
                    cyclic.add((node, mid))
                elif color[target] == white:
                    color[target] = gray
                    frames.append((target, iter(targets_of(target))))
                    descended = True
                    break
            if not descended:
                color[node] = black
                order.append(node)
                frames.pop()

    counts = {o.uid: _triangle_count(o.mesh) for o in doc.objects}
    total = sum(counts.values())
    spent = 0
    over_first: int | None = None
    over_more = 0
    for uid in order:
        stack = stacks.get(uid, ())
        if not any(m.enabled for m in stack):
            continue
        obj = doc.by_uid(uid)
        mesh = obj.mesh
        for mod in stack:
            if not mod.enabled:
                continue
            label = _KINDS[mod.kind][0]
            room = MAX_TRIANGLES - (total - counts[uid])
            # The 2026-10-07 audit's clay-53: size is bounded, time was not --
            # a few KB of ``scene.json`` held fifty-iteration smooths in a
            # stack as long as the file allowed, each a full pass over a
            # 90,000-face mesh (1.3 s apiece, linear in stack length), and the
            # bake has no cancel. A running budget over the whole file, charged
            # before the modifier runs, in triangle passes.
            #
            # The 2026-10-09 audit's clay-20: a boolean works over its target
            # as well as its owner, and charging only the owner let six tiny
            # cutters against a 199k-triangle target run 13 s past the budget
            # this exists to hold. The target is charged at its evaluated size.
            # Its second run's clay-85: the unit was a smooth pass and every
            # kind was charged one per triangle, so :func:`_bake_cost` prices
            # each kind at what it measured.
            tid = int(mod.get("target", 0)) if mod.kind == "boolean" else 0
            cost = _bake_cost(
                mod, counts[uid], counts[tid] if tid in present and tid != uid else 0
            )
            if spent + cost > MAX_BAKE_WORK:
                # The 2026-10-11 audit's clay-09: every modifier past the budget
                # added its own copy of this sentence, so a stack of thousands
                # opened under a screenful of identical notices. The first is
                # named in full, on the object it first stopped; the rest are
                # counted, and one "and N more" sentence is added at the end.
                if over_first is None:
                    over_first = uid
                    errors.setdefault(uid, []).append(
                        f"the {label} modifier could not run (this file's modifiers are "
                        f"past the {MAX_BAKE_WORK:,} triangle passes Clay spends opening one)"
                    )
                else:
                    over_more += 1
                continue
            try:
                spent += cost
                if mod.kind == "boolean":
                    target = int(mod.get("target", 0))
                    if target == 0:
                        raise el.OpError("Choose a target object.")
                    if (uid, mod.id) in cyclic or target == uid:
                        raise el.OpError("Its boolean target would create a cycle.")
                    if target not in present:
                        raise el.OpError(f"Target object {target} no longer exists.")
                    target_obj = doc.by_uid(target)
                    target_mesh = result.get(target, target_obj.mesh)
                    operation = ops_boolean.KINDS[int(mod.get("operation", 1))]
                    grown = ops_boolean.boolean(
                        [replace(obj, mesh=mesh), replace(target_obj, mesh=target_mesh)],
                        operation,
                        world=[doc.world_matrix(uid), doc.world_matrix(target)],
                    )
                elif mod.kind in ("array", "radial-array"):
                    grown = _KINDS[mod.kind][2](mesh, mod.as_dict(), limit=room)
                else:
                    grown = _KINDS[mod.kind][2](mesh, mod.as_dict())
                grown_count = _triangle_count(grown)
                _refuse_growth("This modifier", grown_count, room)
                _refuse_vertex_growth(
                    "This modifier", mesh, len(grown.positions), len(grown.loops)
                )
            except el.OpError as error:
                errors.setdefault(uid, []).append(f"the {label} modifier could not run ({error})")
                continue
            mesh = grown
            total += grown_count - counts[uid]
            counts[uid] = grown_count
        if mesh is not obj.mesh:
            result[uid] = mesh
    if over_first is not None and over_more:
        many = over_more != 1
        errors[over_first].append(
            f"{over_more:,} more modifier{'s' if many else ''} in this file could not run "
            f"for the same reason, and {'were' if many else 'was'} {_DROPPED_WORD}"
        )
    return result, errors


# --- the migration --------------------------------------------------------


def _names(objs: Sequence[Obj], limit: int = 4) -> str:
    shown = ", ".join(f'"{o.name}"' for o in objs[:limit])
    return shown + (f" and {len(objs) - limit} more" if len(objs) > limit else "")


def _collider_targets(
    kept: set[int], stacks: dict[int, tuple[_Modifier, ...]], colliding: set[int]
) -> set[int]:
    """The colliders (with a stack of their own) that an enabled boolean of a kept
    object -- or of another such collider -- cuts with, so their stacks are
    baked as targets. Transitive, and cycle-safe: each uid joins once."""
    needed: set[int] = set()
    todo = list(kept)
    while todo:
        for mod in stacks.get(todo.pop(), ()):
            if not (mod.enabled and mod.kind == "boolean"):
                continue
            target = int(mod.get("target", 0))
            if target in colliding and target not in needed and stacks.get(target):
                needed.add(target)
                todo.append(target)
    return needed


def migrate(doc: ClayDoc, fields: dict[int, LegacyFields]) -> list[str]:
    """Rewrite *doc* -- a freshly read format-3 document -- into the format-4 shape.

    Mutates ``doc.objects`` in place (there is no history yet) and returns the
    notices, one sentence each. *fields* is keyed by object uid; an object the
    reader found nothing dead on may be absent.
    """
    notices: list[str] = []
    # The 2026-10-09 audit's clay-36: a collider is dropped below, so baking its
    # stack spent budget on a mesh nobody keeps and named it in the "applied to"
    # sentence of a model it is not part of.
    # The second audit's clay-97 settled what clay-36 left open: a collider's
    # stack is still baked when a kept object's boolean cuts with it, because the
    # old evaluator gave a boolean its target's *evaluated* mesh, collider or
    # not -- but only as that target. Nothing of it is applied, and no sentence
    # about it is written: the collider is removed below, and a notice that "the
    # Weld modifier was dropped" from a shape the file no longer has is noise.
    kept = {uid for uid, f in fields.items() if f.modifiers and f.role != "collider"}
    colliding = {uid for uid, f in fields.items() if f.role == "collider"}
    needed = _collider_targets(kept, {uid: f.modifiers for uid, f in fields.items()}, colliding)
    baked_uids = kept | needed
    stacks = {uid: f.modifiers for uid, f in fields.items() if uid in baked_uids}

    baked, errors = _bake(doc, stacks)
    for obj in doc.objects:
        if obj.uid in colliding:
            continue
        for sentence in fields.get(obj.uid, LegacyFields()).unreadable:
            notices.append(f'On "{obj.name}", {sentence}.')
        for sentence in errors.get(obj.uid, ()):
            # A sentence that already says what became of its modifiers (the "and N
            # more" one) is not given the tail the single-modifier sentences get.
            tail = "" if sentence.endswith(_DROPPED_WORD) else " and was dropped"
            notices.append(f'On "{obj.name}", {sentence}{tail}.')

    changed: list[Obj] = []
    for obj in doc.objects:
        mesh = baked.get(obj.uid)
        if mesh is not None and obj.uid not in colliding:
            obj.mesh = mesh
            obj.generator, obj.params = None, {}
            changed.append(obj)
    if changed:
        notices.insert(
            0,
            "This model used modifiers, which Clay no longer has. They were applied "
            f"to the mesh of {_names(changed)}.",
        )

    colliders = [o for o in doc.objects if fields.get(o.uid, LegacyFields()).role == "collider"]
    if colliders:
        _drop(doc, colliders)
        many = len(colliders) != 1
        notices.append(
            f"{len(colliders)} collision shape{'s' if many else ''} ({_names(colliders)}) "
            f"{'were' if many else 'was'} removed: Clay no longer models colliders."
        )

    lost = [
        o
        for o in doc.objects
        if (f := fields.get(o.uid)) is not None and (f.has_seams or f.has_tags or f.locked)
    ]
    if lost:
        notices.append(
            f"Seams, tags and locks are gone from Clay; they were cleared from {_names(lost)}."
        )

    frozen = freeze_unknown_generators(doc)
    if frozen is not None:
        notices.append(frozen)
    return notices


def _param_is_unusable(default: Any, value: Any) -> bool:
    """Whether a stored generator param cannot stand in for its default's type."""

    # The 2026-10-09 second audit's clay-94: ``math.isfinite`` raises
    # ``OverflowError`` on an integer past float range (a 400-digit ``segments``
    # escaped the reader as a crash, not a freeze), and a number that is finite
    # but outside the range the Properties panel draws it in loaded as a live
    # generator whose ``input_int`` raised every frame -- an int outside int32,
    # a float outside float32, which is also the range ``clamp_params`` refuses.
    def number(v: Any, low: float, high: float) -> bool:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False
        try:
            return math.isfinite(v) and low <= v <= high
        except OverflowError:
            return False

    int32 = (-(2.0**31), 2.0**31 - 1)
    float32 = (-F4_MAX, F4_MAX)
    if isinstance(default, bool):
        return not isinstance(value, bool)
    if isinstance(default, (tuple, list)):
        return (
            not isinstance(value, (tuple, list))
            or len(value) != len(default)
            or not all(number(v, *float32) for v in value)
        )
    return not number(value, *(int32 if isinstance(default, int) else float32))


def _malformed_params(obj: Obj) -> bool:
    """A Clay-offered generator whose stored params carry a value the Properties
    panel cannot draw: a string, null or bool where a number goes, or a NaN."""
    defaults = GENERATORS[obj.generator][0]
    return any(
        key in defaults and _param_is_unusable(defaults[key], value)
        for key, value in obj.params.items()
    )


def freeze_unknown_generators(doc: ClayDoc) -> str | None:
    """Freeze every object whose generator Clay's Add palette no longer offers --
    the mesh stays, the generator and its params go -- and return the notice
    naming them, or ``None`` when there was nothing to freeze.

    Public because it is a rule about the generator, not the file's age: the
    2026-10-07 audit's clay-43 found it ran only for format 3, so a version-4
    file naming ``lathe`` kept a live generator panel whose edits are refused.
    ``serialize.read_rblk`` calls it for every version.

    The 2026-10-09 audit's clay-19 widened it to a Clay-offered generator whose
    stored params are not numbers of the default's type: ``"segments": "x"``,
    null or NaN loaded cleanly and then raised in the Properties panel every
    frame, and this reader is also what crash recovery runs with no user to ask.
    Freezing keeps the mesh the file carries, which is the part that is real.
    """
    unknown = [
        o
        for o in doc.objects
        if o.generator is not None and o.generator not in CLAY_GENERATOR_NAMES
    ]
    broken = [
        o
        for o in doc.objects
        if o.generator in CLAY_GENERATOR_NAMES and _malformed_params(o)
    ]
    for obj in unknown + broken:
        obj.generator, obj.params = None, {}
    notices = []
    if unknown:
        notices.append(
            f"{_names(unknown)} used a shape Clay no longer builds; "
            "the mesh is kept as a plain mesh."
        )
    if broken:
        notices.append(
            f"{_names(broken)} had shape settings that are not numbers; "
            "the mesh is kept as a plain mesh."
        )
    return " ".join(notices) or None


def _drop(doc: ClayDoc, doomed: list[Obj]) -> None:
    """Remove *doomed* from a history-less document. A survivor under a removed
    object is lifted onto its nearest surviving ancestor, keeping its place in the
    world; one that cannot be lifted (a zero-scale ancestor) keeps its local
    transform, the best a degenerate document allows."""
    gone = {o.uid for o in doomed}
    orphans = {o.uid: doc.world_matrix(o.uid) for o in doc.objects if o.parent in gone}
    for obj in doc.objects:
        while obj.parent in gone:
            obj.parent = doc.by_uid(obj.parent).parent
    doc.objects = [o for o in doc.objects if o.uid not in gone]
    for uid, world in orphans.items():
        if uid in gone:
            continue
        child = doc.by_uid(uid)
        # ``_kept_place``, not ``_local_relative``: the 2026-10-07 audit's
        # clay-26 found a surviving ancestor of denormal scale (finite, so past
        # the singular test) inverted to ``inf`` here and wrote NaN onto the
        # child, a migrated document that could never be saved or journalled.
        with contextlib.suppress(el.OpError):
            child.translation, child.rotation, child.scale = doc._kept_place(
                child.name, world, child.parent
            )
