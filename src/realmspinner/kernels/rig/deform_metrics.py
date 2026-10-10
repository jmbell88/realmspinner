"""Numbers for the deformation review: what a bent mesh did to itself.

The deformation QA sheet is a picture a person looks at. This module is the
arithmetic that sits beside it, and it is **advisory in the same sense the sheet
is**: it measures, it gates nothing, and it carries no pass/fail colour until a
dated measurement under ``dev/measurements/`` has calibrated thresholds from
real rigs. The constants below are therefore *provisional* and are reported as
raw numbers, never compared against in the UI.

Pure numpy and no Blender: the worker (the only place a posed, evaluated mesh
exists) hands it arrays, so every claim here is testable on a synthetic cylinder.

Four things are measured, each the arithmetic of a failure a reviewer looks for:

* **Stretch** -- the ratio of each triangle edge's posed length to its rest
  length. A bad weight map drags a vertex away from the body and a few edges
  grow by an order of magnitude; the maximum, the 99th percentile and the share
  past :data:`STRETCH_LIMIT` find them where a mean would not.
* **Collapse** -- the share of edges shorter than :data:`COLLAPSE_LIMIT` of
  their rest length, the opposite failure (a limb folded into the torso).
* **Volume** -- posed over rest enclosed volume, for a closed mesh only. An open
  mesh has no enclosed volume, and reporting one anyway would be a number with
  nothing behind it.
* **Radial ratio at each joint** (the candy-wrapper test) -- the median distance
  of the vertices in a thin slab around a joint from the bone's axis, posed over
  rest. A twist that blends two opposed rotations pinches the cross-section
  towards the axis; the ratio goes to zero where the mesh is wrung out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# Provisional. Nothing in the stored corpus is keyed on these and no measurement
# backs them; they decide only which edges the two fractions below count, and the
# raw p99 beside them is what a reader should look at first.
STRETCH_LIMIT = 2.0
COLLAPSE_LIMIT = 0.25
# Half-width of the slab around a joint, as a fraction of the rest bounding-box
# diagonal. Wide enough to hold a ring of vertices on a coarse mesh, narrow
# enough not to reach the next joint on a hand.
SLAB_FRACTION = 0.03
MIN_SLAB_VERTICES = 8
# Welding tolerance for deciding whether a mesh is closed: UV seams split
# vertices that coincide in space, and judging closure by index would call every
# textured mesh open.
WELD_FRACTION = 1e-6


def edges_of(tris: np.ndarray) -> np.ndarray:
    """Unique undirected edges of a triangle index array, ``(E, 2)`` sorted."""
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    pairs = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    pairs.sort(axis=1)
    return np.unique(pairs, axis=0)


def is_closed(rest: np.ndarray, tris: np.ndarray) -> bool:
    """Whether every edge of the mesh is shared by exactly two triangles, once
    vertices that coincide in space are treated as one."""
    rest = np.asarray(rest, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if len(tris) == 0:
        return False
    span = float(np.linalg.norm(rest.max(axis=0) - rest.min(axis=0)))
    step = max(span * WELD_FRACTION, 1e-12)
    keys = np.round(rest / step).astype(np.int64)
    _unique, welded = np.unique(keys, axis=0, return_inverse=True)
    welded = np.asarray(welded).reshape(-1)
    wt = welded[tris]
    # A triangle collapsed by the weld has no area and no business in the count.
    wt = wt[(wt[:, 0] != wt[:, 1]) & (wt[:, 1] != wt[:, 2]) & (wt[:, 2] != wt[:, 0])]
    if len(wt) == 0:
        return False
    pairs = np.concatenate([wt[:, [0, 1]], wt[:, [1, 2]], wt[:, [2, 0]]])
    pairs.sort(axis=1)
    _edges, counts = np.unique(pairs, axis=0, return_counts=True)
    return bool((counts == 2).all())


def signed_volume(verts: np.ndarray, tris: np.ndarray) -> float:
    v = np.asarray(verts, dtype=np.float64)[np.asarray(tris, dtype=np.int64).reshape(-1, 3)]
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)


def _radial(points: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    rel = points - origin
    along = rel @ axis
    return np.linalg.norm(rel - np.outer(along, axis), axis=1)


def _unit(vec: Sequence[float]) -> np.ndarray | None:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    return None if not np.isfinite(norm) or norm < 1e-12 else arr / norm


def radial_ratio(
    rest: np.ndarray, posed: np.ndarray, joint: Mapping[str, Any], *, slab: float
) -> tuple[float | None, int]:
    """The candy-wrapper ratio for one joint, and how many vertices backed it.

    ``joint`` carries ``rest_head``/``rest_axis`` and ``posed_head``/``posed_axis``
    -- the bone's head and unit direction before and after the pose. ``None`` when
    the slab holds too few vertices to say anything (a joint on a bare stick, or
    one the mesh does not cover).
    """
    rest_axis, posed_axis = _unit(joint["rest_axis"]), _unit(joint["posed_axis"])
    if rest_axis is None or posed_axis is None:
        return None, 0
    rest_head = np.asarray(joint["rest_head"], dtype=np.float64)
    posed_head = np.asarray(joint["posed_head"], dtype=np.float64)
    along = (rest - rest_head) @ rest_axis
    chosen = np.abs(along) <= slab
    r_rest = _radial(rest[chosen], rest_head, rest_axis)
    keep = r_rest > 1e-9
    if int(keep.sum()) < MIN_SLAB_VERTICES:
        return None, int(keep.sum())
    r_posed = _radial(posed[chosen][keep], posed_head, posed_axis)
    base = float(np.median(r_rest[keep]))
    return float(np.median(r_posed) / base), int(keep.sum())


def score_pose(
    rest: np.ndarray,
    posed: np.ndarray,
    tris: np.ndarray,
    dominant_bone: np.ndarray,
    joints: Sequence[Mapping[str, Any]],
    closed: bool,
    bone_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Score one pose of a rig against its rest shape.

    ``rest`` and ``posed`` are ``(V, 3)`` world positions of the same vertices
    in the same order; ``tris`` is ``(T, 3)`` indices; ``dominant_bone`` is the
    index, per vertex, of the bone that weighs most on it (``-1`` for none) into
    ``bone_names``. ``joints`` is a sequence of mappings with ``name`` and the
    four vectors :func:`radial_ratio` reads. ``closed`` says whether the volume
    ratio means anything; it is the caller's to compute (see :func:`is_closed`)
    because it is a property of the rest mesh, not of the pose.
    """
    rest = np.asarray(rest, dtype=np.float64)
    posed = np.asarray(posed, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    dominant_bone = np.asarray(dominant_bone, dtype=np.int64).reshape(-1)
    if rest.shape != posed.shape or rest.ndim != 2 or rest.shape[1] != 3:
        raise ValueError("rest and posed must be matching (V, 3) arrays")
    if len(tris) == 0 or not np.isfinite(posed).all():
        raise ValueError("a mesh with no triangles or a non-finite pose cannot be scored")

    edges = edges_of(tris)
    rest_len = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    posed_len = np.linalg.norm(posed[edges[:, 0]] - posed[edges[:, 1]], axis=1)
    # Zero-length rest edges (a seam welded shut, a degenerate sliver) have no
    # ratio to speak of; counting them would put an infinity in the p99.
    usable = rest_len > 1e-9 * max(float(rest_len.max()), 1e-12)
    ratio = posed_len[usable] / rest_len[usable]

    worst: str | None = None
    if len(ratio) and bone_names:
        owner = dominant_bone[edges[usable, 0]]
        sums = {}
        for index in np.unique(owner):
            if 0 <= index < len(bone_names):
                sums[bone_names[int(index)]] = float(np.mean(ratio[owner == index]))
        if sums:
            worst = max(sums, key=sums.get)

    volume: float | None = None
    if closed:
        before = signed_volume(rest, tris)
        if abs(before) > 1e-12:
            volume = signed_volume(posed, tris) / before

    span = float(np.linalg.norm(rest.max(axis=0) - rest.min(axis=0)))
    slab = SLAB_FRACTION * span
    per_joint: dict[str, dict[str, Any]] = {}
    for joint in joints:
        value, samples = radial_ratio(rest, posed, joint, slab=slab)
        per_joint[str(joint["name"])] = {
            "radial_ratio": None if value is None else round(value, 4),
            "samples": samples,
        }
    measured = [v["radial_ratio"] for v in per_joint.values() if v["radial_ratio"] is not None]
    have = len(ratio) > 0
    return {
        "volume_ratio": None if volume is None else round(volume, 4),
        "stretch_p99": round(float(np.percentile(ratio, 99)), 4) if have else None,
        # The p99 cannot see a handful of dragged vertices on a dense mesh (a
        # vertex has six edges in several thousand); the maximum and the share
        # past the limit are what do.
        "stretch_max": round(float(ratio.max()), 4) if have else None,
        "stretched_fraction": round(float(np.mean(ratio > STRETCH_LIMIT)), 5) if have else None,
        "collapsed_fraction": round(float(np.mean(ratio < COLLAPSE_LIMIT)), 5) if have else None,
        "worst_stretch_bone": worst,
        "min_radial_ratio": min(measured) if measured else None,
        "joints": per_joint,
    }
