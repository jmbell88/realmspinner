"""Bézier handles over a polyline, flattened back to a polyline.

``lathe``'s ``profile``, ``sweep``'s ``outline`` and ``tube``'s ``path`` are
polylines, and every consumer of them -- the clamps and the builders -- already
speaks polyline. A curve is therefore *stored* as the polyline plus per-anchor
handles and *flattened into that same polyline at build time*, so nothing
downstream learns that curves exist.

Clay no longer offers those three shapes, so it has no curve canvas, viewport
overlay or agent tool for these handles (the 2026-10-07 audit's clay-56: this
paragraph used to name them as consumers). The generators stay whole in
``primitives.GENERATORS`` because a Mason scene records placed objects by
generator name; a ``.rblk`` that names one is frozen to its mesh on open.

**A handle of zero length is a corner.** Handles are per-anchor ``(in, out)``
offsets from the anchor; a segment whose two relevant handles are both zero is
straight and contributes no interior points. A document that carries no handles
at all -- every document written before this module existed -- therefore
flattens to *exactly* the polyline it was, the same floats in the same order, and
the module's first test pins that identity.

The segment from anchor ``a`` to anchor ``b`` is the cubic
``a, a + out_a, b + in_b, b``. ``in`` is the handle on the *incoming* side, so
dragging it away from the anchor along the way the curve arrives is what a
modeller expects of either end.

**Adaptive, and capped.** Each curved segment is cut into ``n`` equal parameter
steps, ``n`` chosen from the cubic's own second difference so that the polyline
stays within ``tol`` of the curve (``err <= 6M / 8n^2`` for ``M`` the larger
second difference of the control points); a straight segment is ``n = 1``. The
whole result is held to ``cap`` stations -- ``primitives.MAX_PROFILE_STATIONS``,
512, the ceiling a profile, outline or path may already not exceed -- by scaling
every segment's interior count down together, deterministically, so a
pathological curve costs resolution rather than a refusal.

Pure: numpy only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "MAX_STATIONS",
    "MAX_HANDLE",
    "DEFAULT_TOL",
    "flatten",
    "has_curves",
    "normalise_handles",
]

#: The station ceiling a flattened curve is held to. Kept equal to
#: ``primitives.MAX_PROFILE_STATIONS`` (a test holds the two equal) rather than
#: imported, because this module sits below it.
MAX_STATIONS = 512

#: The default flatness, in the units of the points (metres for a tube's path, a
#: lathe's silhouette). A millimetre-scale error on a prop that is metres across
#: is invisible, and tight enough that a 128-segment lathe reads as round.
DEFAULT_TOL = 0.002

#: The largest magnitude a handle component is flattened with. The 2026-10-07
#: audit's clay-54: a finite but huge handle (``1e308``) is accepted by
#: :func:`normalise_handles`, but the second difference built from it overflows
#: to ``inf`` and ``math.ceil(inf)`` raised a bare ``OverflowError`` out of a
#: ``clamp_params`` call -- and two opposite huge handles made ``nan``, a
#: ``ValueError``. A million units is far past any anchor the primitives accept
#: (their clamps are metres wide), so no drawn handle is touched; the curve only
#: stops being able to ask for more resolution than ``cap`` already allows.
MAX_HANDLE = 1.0e6


def normalise_handles(handles: Any, count: int, dim: int) -> list[list[list[float]]]:
    """*handles* as ``count`` rows of ``[in, out]`` ``dim``-vectors, or ``[]``.

    ``[]`` for anything that does not fit -- the wrong length, a ragged row, a
    non-number, a non-finite value -- and for handles that are all zero, which are
    the same as none. "Mismatched lengths reset to nothing" is what lets a caller
    that changed the anchors without touching the handles stay valid: the curve
    falls back to its polyline rather than to a refusal.
    """
    try:
        rows = [
            [[float(v) for v in side] for side in (row[0], row[1])] for row in (handles or [])
        ]
    except (TypeError, ValueError, IndexError, OverflowError):
        return []
    if len(rows) != count or not rows:
        return []
    for row in rows:
        if any(len(side) != dim or not all(math.isfinite(v) for v in side) for side in row):
            return []
    if not any(any(side) for row in rows for side in row):
        return []
    return rows


def has_curves(handles: Any) -> bool:
    """Whether any handle is non-zero (``handles`` already normalised)."""
    return bool(handles) and any(any(side) for row in handles for side in row)


def _segment_count(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, tol: float
) -> int:
    if np.array_equal(p0, p1) and np.array_equal(p2, p3):
        return 1  # both handles zero: a straight segment, however it was written
    second = max(
        float(np.linalg.norm(p0 - 2.0 * p1 + p2)), float(np.linalg.norm(p1 - 2.0 * p2 + p3))
    )
    return max(1, math.ceil(math.sqrt(0.75 * second / max(tol, 1e-12))))


def flatten(
    points: Sequence[Sequence[float]],
    handles: Any,
    closed: bool = False,
    tol: float = DEFAULT_TOL,
    cap: int = MAX_STATIONS,
) -> list[list[float]]:
    """The polyline a curve describes: anchors, with the curve's own points between.

    ``handles`` is ``[]`` or one ``[in, out]`` row per anchor (see
    :func:`normalise_handles`). ``closed`` adds the segment from the last anchor
    back to the first, and does **not** repeat the first point at the end -- an
    outline is closed by its own wrap, as ``sweep`` expects it.
    """
    anchors = [[float(v) for v in p] for p in points]
    count = len(anchors)
    if count < 2:
        return anchors
    dim = len(anchors[0])
    rows = normalise_handles(handles, count, dim)
    if not rows:
        return anchors

    pts = np.asarray(anchors, dtype="f8")
    inn = np.asarray([row[0] for row in rows], dtype="f8")
    out = np.asarray([row[1] for row in rows], dtype="f8")
    inn = np.clip(inn, -MAX_HANDLE, MAX_HANDLE)
    out = np.clip(out, -MAX_HANDLE, MAX_HANDLE)
    segments = count if closed else count - 1
    controls = []
    for i in range(segments):
        j = (i + 1) % count
        p0, p3 = pts[i], pts[j]
        controls.append((p0, p0 + out[i], p3 + inn[j], p3))
    steps = [_segment_count(*c, tol) for c in controls]

    budget = cap - count
    interior = sum(n - 1 for n in steps)
    if interior > max(budget, 0):
        scale = max(budget, 0) / interior
        steps = [1 + int((n - 1) * scale) for n in steps]

    result: list[list[float]] = []
    for (p0, p1, p2, p3), n in zip(controls, steps, strict=True):
        result.append(p0.tolist())
        for k in range(1, n):
            t = k / n
            u = 1.0 - t
            point = u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t * p3
            result.append(point.tolist())
    if not closed:
        result.append(pts[-1].tolist())
    return result
