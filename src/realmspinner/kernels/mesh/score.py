"""Rank a generated mesh against the picture it was made from -- no model, no GPU.

Create can make several attempts at one reference (different mesh seeds). Which one
to keep used to be a pure eyeball job; this gives the picker an order. It is a
**sort key**, never a verdict: ``rank_key`` orders candidates and nothing here may be
read as "this mesh is good" (``dev/INVARIANTS.md``, the candidate-order entry).

What it measures is the one thing a reconstruction is supposed to preserve and that a
mask comparison can see: the silhouette of the subject the mesh was made from. The mesh
is rasterised from a sweep of views (the caller does that -- this module is numpy-only
and may not import ``meshaudit``), and the best view's IoU against the reference's
subject mask is the score. A *sweep*, because TRELLIS has no fixed camera-matched yaw:
``bench/views.py`` records the 2026-08-05 calibration, where the per-job best yaw
scattered by 330 degrees. Taking the best over the sweep is what that note says any
fidelity metric has to do.

Ordering is lexicographic, not a weighted blend -- a blend invents an exchange rate
between "has holes" and "matches the silhouette" that nothing here measured:

1. a **hard demotion** sorts a mesh after every undemoted one (it missed its triangle
   budget, or it is more see-through than ``mesh_hole_max`` allows, or it is above a
   component cap when one is set);
2. then **silhouette IoU**, higher first;
3. then fewer non-manifold and boundary edges.

The component cap is ``None`` by default and that is deliberate. The 2026-10-15 sweep
(``dev/measurements/2026-10-15-lowpoly-silhouette-gate.md``) found components scale with
triangle count on every finisher (~0.25-0.35 per triangle) and do not separate good
meshes from bad ones, so a hard cap would demote by size, not by quality.

Pure functions over arrays and dicts. The only imports are numpy and the standard library.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

#: Bumped when the meaning of a stored ``mesh_score`` changes, so a reader never
#: compares scores made by different rules.
SCORE_VERSION = 1

#: Side of the square both masks are normalised to before they are compared. A 64 grid is
#: coarse enough to ignore pixel-level matte noise and fine enough to see a missing leg.
MASK_SIZE = 64

#: The view sweep for a model with no fixed camera: 24 yaws x 3 elevations = 72 views.
YAW_STEPS = 24
ELEVATIONS_DEG = (0.0, 20.0, 40.0)


def view_directions(
    yaws: int = YAW_STEPS, elevations: Sequence[float] = ELEVATIONS_DEG
) -> list[tuple[float, float, float]]:
    """Camera directions (from the origin toward the camera, Y up) for the sweep.

    Yaw 0 is +Z, the front view ``meshaudit.DEFAULT_VIEWS`` starts from; elevations are
    above the horizon. Row-major by elevation then yaw, so a view's index names it.
    """
    out: list[tuple[float, float, float]] = []
    for elevation in elevations:
        lift = math.radians(elevation)
        for step in range(yaws):
            yaw = 2.0 * math.pi * step / yaws
            out.append(
                (
                    math.cos(lift) * math.sin(yaw),
                    math.sin(lift),
                    math.cos(lift) * math.cos(yaw),
                )
            )
    return out


def normalise(mask: Any, size: int = MASK_SIZE) -> np.ndarray | None:
    """``mask`` cropped to its own bounding box, padded square (never stretched -- stretching
    would let a tall sword score 1.0 against a wide shield) and resized to ``size``.

    ``None`` for an empty mask: an empty subject is "not measured", never an all-zero
    picture. Nearest-neighbour index sampling, the same choice ``bench.metrics`` makes.
    """
    array = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(array)
    if not len(xs):
        return None
    cropped = array[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    side = max(cropped.shape)
    square = np.zeros((side, side), dtype=bool)
    top = (side - cropped.shape[0]) // 2
    left = (side - cropped.shape[1]) // 2
    square[top : top + cropped.shape[0], left : left + cropped.shape[1]] = cropped
    index = np.minimum(((np.arange(size) + 0.5) * side / size).astype(np.int64), side - 1)
    return square[np.ix_(index, index)]


def silhouette_iou(a: Any, b: Any) -> float | None:
    """Intersection over union of two masks, each normalised first. ``None`` if either is empty."""
    na, nb = normalise(a), normalise(b)
    if na is None or nb is None:
        return None
    union = int(np.logical_or(na, nb).sum())
    if not union:
        return None
    return float(np.logical_and(na, nb).sum()) / float(union)


def best_view_iou(views: Sequence[Any], target: Any) -> tuple[float | None, int | None]:
    """The best IoU of any of ``views`` against ``target``, and which view it was.

    ``target`` is normalised once. ``(None, None)`` when nothing could be compared (the
    target is empty, or every view is).
    """
    goal = normalise(target)
    if goal is None:
        return None, None
    best: float | None = None
    best_index: int | None = None
    for index, view in enumerate(views):
        candidate = normalise(view)
        if candidate is None:
            continue
        union = int(np.logical_or(candidate, goal).sum())
        if not union:
            continue
        value = float(np.logical_and(candidate, goal).sum()) / float(union)
        if best is None or value > best:
            best, best_index = value, index
    return best, best_index


def build(
    *,
    iou: float | None,
    best_view: int | None,
    holes: float | None,
    hole_max: float | None,
    components: int | None,
    nonmanifold: int | None,
    boundary: int | None,
    budget_ratio: float | None,
    budget_miss: Any,
    components_max: int | None = None,
) -> dict[str, Any]:
    """The record ``params["mesh_score"]`` keeps. Every field is a fact; ``demoted_by`` names
    which hard rule, if any, sends the mesh to the back of the order."""
    demoted: list[str] = []
    if budget_miss:
        demoted.append("budget_miss")
    if holes is not None and hole_max is not None and holes > hole_max:
        demoted.append("holes")
    if components_max is not None and components is not None and components > components_max:
        demoted.append("components")
    return {
        "version": SCORE_VERSION,
        "iou": None if iou is None else round(float(iou), 4),
        "best_view": best_view,
        "holes": None if holes is None else round(float(holes), 4),
        "components": components,
        "nonmanifold": nonmanifold,
        "boundary": boundary,
        "budget_ratio": None if budget_ratio is None else round(float(budget_ratio), 3),
        "demoted_by": demoted,
    }


def rank_key(score: Any) -> tuple[int, int, float, int]:
    """A sort key for ``params["mesh_score"]``: smaller sorts first.

    ``(unscored, demoted, -iou, open edges)``. A mesh with no score (not finished, no
    reference to compare, the measurement broke) sorts after every scored one, and among
    scored ones an unmeasured IoU sorts after a measured one inside its demotion class.
    """
    if not isinstance(score, dict) or score.get("version") != SCORE_VERSION:
        return (1, 0, 0.0, 0)
    iou = score.get("iou")
    open_edges = int(score.get("nonmanifold") or 0) + int(score.get("boundary") or 0)
    return (
        0,
        1 if score.get("demoted_by") else 0,
        # An unmeasured IoU is worse than any measured one, and finite so the key stays sortable.
        1.0 if iou is None else -float(iou),
        open_edges,
    )
