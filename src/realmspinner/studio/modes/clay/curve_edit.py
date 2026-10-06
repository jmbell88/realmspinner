"""The curve editor's model: what a drag, an insert and a delete *do* to a curve.

Pure and drawn-nothing, so every claim the editor makes about the data -- a
point inserted into a curve keeps its shape, deleting a point drops its handles
with it, a figure-eight is found -- is provable without a frame. The widget
(``ui/panes/curve_editor.py``) is the thin layer that turns mouse events into
these calls and sends the result through the same ``set_generator_params`` door
the generic parameter loop uses.

A :class:`Curve` is the polyline a generator stores (``profile``, ``outline`` or
``path``) plus its per-anchor ``[in, out]`` handles. **Handles are always full
here** -- one zero row per anchor when there are none -- so every operation
indexes them without a case; :meth:`Curve.stored_handles` collapses an
all-zero set back to ``[]``, which is what a document with no curves stores and
what makes an untouched curve a no-op write.

Every operation returns a **new** curve. The stored lists belong to the document
(and to undo), so nothing here mutates what it was handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ....kernels.mesh import curves

__all__ = [
    "PLANES",
    "Curve",
    "crossings",
    "delete_point",
    "insert_point",
    "min_points",
    "move_anchor",
    "nearest_on_curve",
    "pull_symmetric",
    "set_handle",
]

#: The three planes a 3D path is edited in: name -> the two coordinate indices
#: drawn as (horizontal, vertical). ``ZY`` puts Y up with Z across, the side view.
PLANES: dict[str, tuple[int, int]] = {"XY": (0, 1), "XZ": (0, 2), "ZY": (2, 1)}

#: The fewest anchors each kind may keep: a lathe needs two stations, a path two
#: points, a closed outline three corners to be a polygon at all.
_MIN_POINTS = {"profile": 2, "outline": 3, "path": 2}


def min_points(key: str) -> int:
    return _MIN_POINTS.get(key, 2)


@dataclass
class Curve:
    """Anchors, their handles, and whether the last joins back to the first."""

    points: list[list[float]]
    handles: list[list[list[float]]]
    closed: bool = False

    @classmethod
    def from_params(cls, points: Any, handles: Any, *, closed: bool = False) -> Curve:
        pts = [[float(v) for v in p] for p in points]
        dim = len(pts[0]) if pts else 2
        rows = curves.normalise_handles(handles, len(pts), dim)
        if not rows:
            rows = [[[0.0] * dim, [0.0] * dim] for _ in pts]
        return cls(pts, rows, closed)

    @property
    def dim(self) -> int:
        return len(self.points[0]) if self.points else 2

    def copy(self) -> Curve:
        return Curve(
            [list(p) for p in self.points],
            [[list(side) for side in row] for row in self.handles],
            self.closed,
        )

    def stored_handles(self) -> list:
        """The handles as stored: ``[]`` when every one is zero."""
        return [[list(side) for side in row] for row in self.handles] if (
            curves.has_curves(self.handles)
        ) else []

    def flat(self, tol: float = curves.DEFAULT_TOL) -> list[list[float]]:
        return curves.flatten(self.points, self.stored_handles(), closed=self.closed, tol=tol)


# --- edits ----------------------------------------------------------------------------


def move_anchor(curve: Curve, index: int, position: list[float], axes: tuple[int, ...]) -> Curve:
    """Anchor *index* to *position* along *axes* only (a path edited in a plane
    leaves its third coordinate where it was). Handles are offsets, so they ride along."""
    out = curve.copy()
    for slot, axis in enumerate(axes):
        out.points[index][axis] = float(position[slot])
    return out


def set_handle(
    curve: Curve,
    index: int,
    side: int,
    offset: list[float],
    axes: tuple[int, ...],
    *,
    mirror: bool = False,
) -> Curve:
    """Handle *side* (0 = in, 1 = out) of anchor *index* to *offset* along *axes*.

    ``mirror`` sets the opposite handle to the negation -- an aligned, equal-length
    pair, which is what a smooth point is.
    """
    out = curve.copy()
    for slot, axis in enumerate(axes):
        out.handles[index][side][axis] = float(offset[slot])
        if mirror:
            out.handles[index][1 - side][axis] = -float(offset[slot])
    return out


def pull_symmetric(curve: Curve, index: int, offset: list[float], axes: tuple[int, ...]) -> Curve:
    """Alt-drag: pull a smooth handle pair out of an anchor, ``out = offset``, ``in = -offset``."""
    return set_handle(curve, index, 1, offset, axes, mirror=True)


def _split(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float
) -> tuple[np.ndarray, ...]:
    q0 = p0 + t * (p1 - p0)
    q1 = p1 + t * (p2 - p1)
    q2 = p2 + t * (p3 - p2)
    r0 = q0 + t * (q1 - q0)
    r1 = q1 + t * (q2 - q1)
    s = r0 + t * (r1 - r0)
    return q0, q2, r0, r1, s


def insert_point(curve: Curve, segment: int, t: float) -> Curve:
    """Split *segment* (anchor ``segment`` to the next) at parameter *t*.

    By de Casteljau, so **the curve's shape does not change** -- the new anchor
    sits on it and the neighbours' handles shorten to match. A straight segment
    splits into two straight ones. For a closed curve the wrap segment (last to
    first) is ``segment == len - 1`` and the new anchor is appended.
    """
    n = len(curve.points)
    j = (segment + 1) % n
    pts = np.asarray(curve.points, dtype="f8")
    hs = np.asarray(curve.handles, dtype="f8")
    p0, p3 = pts[segment], pts[j]
    out = curve.copy()
    if not hs[segment][1].any() and not hs[j][0].any():
        # A straight segment splits into two straight ones, the new point at the
        # linear position. (The cubic with both handles zero is the same line but
        # ease-parametrised, and its de Casteljau handles would be collinear
        # non-zero ones -- a "curve" that is a line, which would then store
        # handles and flatten into points for no reason.)
        s = p0 + t * (p3 - p0)
        out.points.insert(segment + 1, s.tolist())
        out.handles.insert(segment + 1, [[0.0] * pts.shape[1], [0.0] * pts.shape[1]])
        return out
    p1, p2 = p0 + hs[segment][1], p3 + hs[j][0]
    q0, q2, r0, r1, s = _split(p0, p1, p2, p3, t)

    out.handles[segment][1] = (q0 - p0).tolist()
    out.handles[j][0] = (q2 - p3).tolist()
    at = segment + 1
    out.points.insert(at, s.tolist())
    out.handles.insert(at, [(r0 - s).tolist(), (r1 - s).tolist()])
    return out


def delete_point(curve: Curve, index: int) -> Curve:
    """Anchor *index* and its handle row, and nothing else."""
    out = curve.copy()
    del out.points[index]
    del out.handles[index]
    return out


# --- picking ----------------------------------------------------------------------------


def nearest_on_curve(
    curve: Curve, point: tuple[float, float], axes: tuple[int, int], samples: int = 32
) -> tuple[int, float, float] | None:
    """The nearest place on the curve to *point* (in the plane of *axes*).

    -> ``(segment, t, distance)`` or ``None`` for a curve with no segment. Sampled
    per segment and then refined once around the best sample, which is plenty for
    a click and keeps the answer in the *segment/t* terms ``insert_point`` wants.
    """
    n = len(curve.points)
    count = n if curve.closed else n - 1
    if count < 1:
        return None
    pts = np.asarray(curve.points, dtype="f8")
    hs = np.asarray(curve.handles, dtype="f8")
    target = np.asarray(point, dtype="f8")
    best: tuple[int, float, float] | None = None

    def at(seg: int, t: float) -> np.ndarray:
        j = (seg + 1) % n
        p0, p3 = pts[seg], pts[j]
        if not hs[seg][1].any() and not hs[j][0].any():
            return (p0 + t * (p3 - p0))[list(axes)]  # straight: linear, as insert_point splits it
        p1, p2 = p0 + hs[seg][1], p3 + hs[j][0]
        u = 1.0 - t
        full = u**3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t**3 * p3
        return full[list(axes)]

    for seg in range(count):
        ts = np.linspace(0.0, 1.0, samples + 1)
        dists = [float(np.linalg.norm(at(seg, float(t)) - target)) for t in ts]
        k = int(np.argmin(dists))
        lo, hi = ts[max(k - 1, 0)], ts[min(k + 1, samples)]
        for t in np.linspace(lo, hi, 9):
            d = float(np.linalg.norm(at(seg, float(t)) - target))
            if best is None or d < best[2]:
                best = (seg, float(t), d)
    return best


def crossings(poly: list[list[float]], closed: bool, axes: tuple[int, int] = (0, 1)) -> set[int]:
    """Indices of the flattened polyline's segments that cross a non-adjacent one.

    The manual's *uncaught figure-eight*: a self-crossing outline survives every
    clamp and builds a self-intersecting solid. The editor draws these segments in
    the warning colour. Vectorised (every pair at once) because a flattened
    outline can be hundreds of segments and this runs on a redraw.
    """
    n = len(poly)
    if n < 4:
        return set()
    pts = np.asarray(poly, dtype="f8")[:, list(axes)]
    a = pts
    b = np.roll(pts, -1, axis=0) if closed else np.vstack([pts[1:], pts[-1:]])
    count = n if closed else n - 1
    a, b = a[:count], b[:count]

    d = b - a  # (m, 2)
    # Segment i against segment j: solve a_i + s d_i = a_j + u d_j.
    cross = d[:, None, 0] * d[None, :, 1] - d[:, None, 1] * d[None, :, 0]
    diff = a[None, :, :] - a[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (diff[..., 0] * d[None, :, 1] - diff[..., 1] * d[None, :, 0]) / cross
        u = (diff[..., 0] * d[:, None, 1] - diff[..., 1] * d[:, None, 0]) / cross
    eps = 1e-9
    hit = (np.abs(cross) > 1e-12) & (s > eps) & (s < 1 - eps) & (u > eps) & (u < 1 - eps)
    index = np.arange(count)
    gap = np.abs(index[:, None] - index[None, :])
    if closed:
        gap = np.minimum(gap, count - gap)
    hit &= gap > 1  # neighbours share an endpoint; that is a corner, not a crossing
    return {int(i) for i in np.nonzero(hit.any(axis=1))[0]}


# --- the gesture state machine ---------------------------------------------------------
#
# The widget turns pygame/imgui mouse state into a :class:`Pointer` once a frame
# and calls :func:`step`; everything about *what a press, a drag and a release
# mean* is decided here, in model coordinates, so the headless tests drive the
# same code the pane does.


@dataclass
class CurveUi:
    """One object's curve editor, between frames. Display state, never document state."""

    selected: int = -1
    #: ``("anchor"|"pull"|"handle", index, side)`` while the mouse is held on something.
    drag: tuple[str, int, int] | None = None
    #: The curve being dragged, **unclamped**: the clamp may reorder, dedupe or
    #: re-centre what the document stores mid-drag, and drawing from the stored
    #: copy would slide the point out from under the cursor. Dropped at release,
    #: after which the stored (clamped) params are what is read.
    working: Curve | None = None
    #: The history mark the open gesture folds back to (``-1`` when none).
    mark: int = -1
    plane: str = "XY"
    centre: tuple[float, float] | None = None
    scale: float = 0.0
    cross_key: Any = None
    cross_value: frozenset[int] = frozenset()


@dataclass
class Pointer:
    """The mouse, in model coordinates, for one frame."""

    pos: tuple[float, float]
    #: Model units per pixel, so hit radii are pixels however far the view is zoomed.
    per_pixel: float = 1.0
    pressed: bool = False
    down: bool = False
    released: bool = False
    alt: bool = False
    shift: bool = False
    delete: bool = False
    right_pressed: bool = False


#: Pixels. An anchor is easier to hit than a handle dot, and a segment easiest of all.
ANCHOR_PX = 8.0
HANDLE_PX = 7.0
SEGMENT_PX = 7.0


def _anchor_hit(
    curve: Curve, axes: tuple[int, int], pos: tuple[float, float], radius: float
) -> int:
    best, best_d = -1, radius
    for i, point in enumerate(curve.points):
        d = float(np.hypot(point[axes[0]] - pos[0], point[axes[1]] - pos[1]))
        if d <= best_d:
            best, best_d = i, d
    return best


def _handle_hit(
    curve: Curve, axes: tuple[int, int], pos: tuple[float, float], radius: float
) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    best_d = radius
    for i, point in enumerate(curve.points):
        for side in (0, 1):
            offset = curve.handles[i][side]
            if not any(offset):
                continue  # a zero handle sits on its anchor; Alt-drag the anchor to pull it out
            d = float(
                np.hypot(point[axes[0]] + offset[axes[0]] - pos[0],
                         point[axes[1]] + offset[axes[1]] - pos[1])
            )
            if d <= best_d:
                best, best_d = (i, side), d
    return best


def step(
    ui: CurveUi, curve: Curve, key: str, pointer: Pointer, axes: tuple[int, int]
) -> tuple[Curve, list[str]]:
    """One frame of editing. -> ``(curve, events)``.

    ``events`` says what the caller owes: ``"begin"`` (open a history gesture),
    ``"change"`` (the returned curve differs -- apply it), ``"end"`` (fold the
    gesture). A drag is ``begin``, many ``change``, ``end``; an insert or a
    delete is all three at once and so is one undo step like any other edit.
    """
    events: list[str] = []
    lathe = key == "profile"

    def clamped(pos: tuple[float, float]) -> tuple[float, float]:
        # A lathe's radius is an extent; dragging past the axis would be stored as
        # its mirror image by the clamp and read as the point jumping across.
        return (max(pos[0], 0.0), pos[1]) if lathe else pos

    if ui.drag is not None:
        kind, index, side = ui.drag
        if index >= len(curve.points):
            ui.drag = None
            return curve, events
        if pointer.down:
            if kind == "anchor":
                new = move_anchor(curve, index, list(clamped(pointer.pos)), axes)
            else:
                anchor = curve.points[index]
                offset = [pointer.pos[0] - anchor[axes[0]], pointer.pos[1] - anchor[axes[1]]]
                if kind == "pull":
                    new = pull_symmetric(curve, index, offset, axes)
                else:
                    new = set_handle(curve, index, side, offset, axes, mirror=pointer.shift)
            if new.points != curve.points or new.handles != curve.handles:
                events.append("change")
                return new, events
            return curve, events
        ui.drag = None
        events.append("end")
        return curve, events

    radius = pointer.per_pixel
    if pointer.delete or pointer.right_pressed:
        index = (
            ui.selected
            if pointer.delete
            else _anchor_hit(curve, axes, pointer.pos, ANCHOR_PX * radius)
        )
        if 0 <= index < len(curve.points) and len(curve.points) > min_points(key):
            ui.selected = min(index, len(curve.points) - 2)
            events += ["begin", "change", "end"]
            return delete_point(curve, index), events
        return curve, events

    if not pointer.pressed:
        return curve, events

    handle = _handle_hit(curve, axes, pointer.pos, HANDLE_PX * radius)
    if handle is not None:
        ui.selected = handle[0]
        ui.drag = ("handle", handle[0], handle[1])
        events.append("begin")
        return curve, events
    index = _anchor_hit(curve, axes, pointer.pos, ANCHOR_PX * radius)
    if index >= 0:
        ui.selected = index
        ui.drag = ("pull" if pointer.alt else "anchor", index, 0)
        events.append("begin")
        return curve, events
    near = nearest_on_curve(curve, pointer.pos, axes)
    if near is not None and near[2] <= SEGMENT_PX * radius:
        segment, t, _ = near
        if len(curve.points) < _MAX_POINTS:
            new = insert_point(curve, segment, t)
            ui.selected = segment + 1
            ui.drag = ("anchor", segment + 1, 0)
            events += ["begin", "change"]
            return new, events
    ui.selected = -1
    return curve, events


#: The anchor ceiling the editor will insert up to (the primitives' own cap).
_MAX_POINTS = curves.MAX_STATIONS
