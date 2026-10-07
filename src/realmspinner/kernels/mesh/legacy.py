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

from ..geom3d import math3d as m3
from . import elements as el
from . import mesh as bm
from . import ops_bevel, ops_boolean, ops_clean, ops_subdiv
from .adjacency import adjacency
from .document import ClayDoc, Obj
from .earclip import corner_triangles
from .mesh import Mesh
from .primitives import CLAY_GENERATOR_NAMES

__all__ = ["LegacyFields", "migrate", "read_fields"]



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
    """Whether a sweep of *angle* degrees closes into a ring, so the steps divide
    by ``count`` rather than ``count - 1``."""
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
    # (below, via .modifiers) is too late to protect this loop.
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
    value = max(p.low, min(p.high, float(raw)))
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
    """
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
            try:
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
                else:
                    grown = _KINDS[mod.kind][2](mesh, mod.as_dict())
                _refuse_growth("This modifier", _triangle_count(grown))
            except el.OpError as error:
                errors.setdefault(uid, []).append(f"the {label} modifier could not run ({error})")
                continue
            mesh = grown
        if mesh is not obj.mesh:
            result[uid] = mesh
    return result, errors


# --- the migration --------------------------------------------------------


def _names(objs: Sequence[Obj], limit: int = 4) -> str:
    shown = ", ".join(f'"{o.name}"' for o in objs[:limit])
    return shown + (f" and {len(objs) - limit} more" if len(objs) > limit else "")


def migrate(doc: ClayDoc, fields: dict[int, LegacyFields]) -> list[str]:
    """Rewrite *doc* -- a freshly read format-3 document -- into the format-4 shape.

    Mutates ``doc.objects`` in place (there is no history yet) and returns the
    notices, one sentence each. *fields* is keyed by object uid; an object the
    reader found nothing dead on may be absent.
    """
    notices: list[str] = []
    stacks = {uid: f.modifiers for uid, f in fields.items() if f.modifiers}

    baked, errors = _bake(doc, stacks)
    for obj in doc.objects:
        for sentence in fields.get(obj.uid, LegacyFields()).unreadable:
            notices.append(f'On "{obj.name}", {sentence}.')
        for sentence in errors.get(obj.uid, ()):
            notices.append(f'On "{obj.name}", {sentence} and was dropped.')

    changed: list[Obj] = []
    for obj in doc.objects:
        mesh = baked.get(obj.uid)
        if mesh is not None:
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

    unknown = [
        o
        for o in doc.objects
        if o.generator is not None and o.generator not in CLAY_GENERATOR_NAMES
    ]
    for obj in unknown:
        obj.generator, obj.params = None, {}
    if unknown:
        notices.append(
            f"{_names(unknown)} used a shape Clay no longer builds; "
            "the mesh is kept as a plain mesh."
        )
    return notices


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
        with contextlib.suppress(el.OpError):
            child.translation, child.rotation, child.scale = doc._local_relative(
                world, child.parent
            )
