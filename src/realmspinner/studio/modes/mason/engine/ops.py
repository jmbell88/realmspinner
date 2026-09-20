"""Placement arithmetic: array, scatter, align, distribute, drop-to-ground, snap.

Pure functions over transforms, the way ``clay/ops.py`` is pure functions over
``Obj``: **none of them knows that a document, a history or a selection
exists.** Each one takes what it needs -- a base TRS, a count, a set of world
boxes -- and hands back new TRS values or new translations; the document layer
(``document.py``, not this module) is what turns a returned value into a
:class:`~.edits.TransformEdit`. That boundary is deliberate and it is the same
one ``clay/ops.py``'s own module docstring draws: an op here moves things
around, it never records that they moved.

**``align``/``distribute``/``drop_to_ground`` take world boxes, never a
``GeometrySource``.** Aligning or distributing a selection is a question about
world extremes -- the lowest edge, the gap between two neighbours' surfaces --
and answering it needs each item's world axis-aligned box, not just its
translation. This package cannot resolve one: turning a :class:`~.refs.Ref`
into triangles is exactly the one thing ``refs.py``'s module docstring refuses
to let any module here do. ``scene.world_bounds`` already knows how to ask a
host's ``GeometrySource`` for that box, so these three take the *answer* --
``{owner: (lo, hi)}`` -- as a plain argument instead of reaching for the
question's machinery themselves. That is also why each one hands back a
**translation delta** rather than an absolute new translation: a delta is
correct however the caller ultimately applies it -- add it straight to a
root-level node's translation, or carry it through a parent's inverse rotation
first for a nested one -- where an absolute world position would be flatly
wrong the instant a node has a parent transform, since translation lives in
the parent's frame, not world space. Keyed by *owner* rather than node uid for
the same reason every other selection-shaped answer in this package is: a hit
inside a prefab instance and a click in the outliner both address the
instance, not its internals, and ``owner`` is the uid the document actually
has a row for (see ``scene.py``'s "owner" section).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

from .....kernels.geom3d import math3d as m3
from .terrain import Terrain, height_at

#: Translation, rotation (XYZW) and scale -- the same three-array shape
#: ``nodes.Node.trs()`` hands back, so a caller can splat one of these
#: straight into ``MasonDoc.set_transform``.
TRS = tuple[np.ndarray, np.ndarray, np.ndarray]

#: ``{owner_uid: (lo, hi)}`` world axis-aligned boxes, exactly what
#: ``scene.world_bounds`` answers per owner. The type :func:`align`,
#: :func:`distribute` and :func:`drop_to_ground` all read.
Boxes = Mapping[int, tuple[np.ndarray, np.ndarray]]

AlignMode = Literal["min", "centre", "max"]

_AXES = (0, 1, 2)


def _check_axis(axis: int) -> None:
    if axis not in _AXES:
        raise ValueError(f"axis must be 0, 1 or 2, got {axis!r}")


def _own_trs(trs: TRS) -> TRS:
    """A defensive copy of a TRS triple, the same reason ``Node.__post_init__``
    copies its own three arrays: every function below hands back fresh values,
    never a view into the ``base_trs`` the caller is still holding."""
    t, r, s = trs
    return (
        np.array(t, dtype="f8", copy=True),
        np.array(r, dtype="f8", copy=True),
        np.array(s, dtype="f8", copy=True),
    )


# --- arrays ------------------------------------------------------------------


def array_linear(count: int, offset: Sequence[float], *, base_trs: TRS) -> list[TRS]:
    """``count`` copies stepped by ``offset``, the first exactly ``base_trs``.

    Rotation and scale are carried through unchanged on every copy -- a linear
    array is "more of the same thing in a row", not a rotation, and
    :func:`array_radial` is where facing direction ever moves.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count!r}")
    t0, r0, s0 = _own_trs(base_trs)
    step = np.asarray(offset, dtype="f8")
    out: list[TRS] = []
    for i in range(count):
        out.append((t0 + step * i, r0.copy(), s0.copy()))
    return out


def array_radial(
    count: int,
    *,
    centre: Sequence[float],
    axis: Sequence[float],
    degrees: float,
    base_trs: TRS,
    orient: bool = True,
) -> list[TRS]:
    """``count`` copies of ``base_trs`` carried around ``centre`` by ``degrees``
    total, split evenly.

    **Orients by default, and that is the whole decision.** Twelve copies
    around a circle that all keep the base's own rotation is a ring of
    identically-facing props -- a fence with every post staring the same way
    regardless of where it sits on the circle -- which is almost never the
    point of a radial array. So each copy's rotation is carried by the same
    spin that carries its translation (``clay.ops.rotated_about_origin``'s own
    move, generalized from the world origin to an arbitrary ``centre``/
    ``axis``): a base authored facing outward at its own angle keeps facing
    outward all the way around. ``orient=False`` is the escape hatch for the
    ring-of-identical-props case, when that *is* what is wanted.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count!r}")
    t0, r0, s0 = _own_trs(base_trs)
    centre_arr = np.asarray(centre, dtype="f8")
    axis_arr = np.asarray(axis, dtype="f8")
    step = degrees / count
    out: list[TRS] = []
    for i in range(count):
        spin = m3.quat_from_axis_angle(axis_arr, math.radians(step * i))
        translation = centre_arr + m3.quat_rotate(spin, t0 - centre_arr)
        rotation = m3.quat_mul(spin, r0) if orient else r0.copy()
        out.append((translation, rotation, s0.copy()))
    return out


def scatter(
    count: int,
    *,
    area: tuple[float, float, float, float],
    base_trs: TRS,
    seed: int,
    rotate: bool = True,
) -> list[TRS]:
    """``count`` copies at random positions in ``area`` (``x0, z0, x1, z1``),
    keeping ``base_trs``'s own height and scale.

    **Deterministic from ``seed``, via ``numpy.random.default_rng`` --  never
    the legacy global state.** ``terrain.noise`` already makes this argument
    for a sculpt brush and it applies unchanged to a scatter: two runs of the
    same document must place the same props in the same places, and a
    generator seeded fresh here cannot have been advanced by anything else the
    session did first, the way the global RNG could have been.

    ``rotate`` gives each copy an independent random yaw about world +Y
    (``base_trs``'s own rotation is discarded for a rotated copy, not composed
    with it, since a scatter's whole point is that the props no longer share
    one orientation); ``rotate=False`` keeps every copy facing exactly as
    ``base_trs`` did, for a prop whose silhouette does not read as randomly
    placed unless it stays upright and aligned -- a road sign, say.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count!r}")
    t0, r0, s0 = _own_trs(base_trs)
    x0, z0, x1, z1 = area
    rng = np.random.default_rng(seed)
    xs = rng.uniform(x0, x1, size=count)
    zs = rng.uniform(z0, z1, size=count)
    y = float(t0[1])
    out: list[TRS] = []
    for i in range(count):
        translation = np.array([xs[i], y, zs[i]], dtype="f8")
        if rotate:
            yaw = rng.uniform(0.0, 2.0 * math.pi)
            rotation = m3.quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), yaw)
        else:
            rotation = r0.copy()
        out.append((translation, rotation, s0.copy()))
    return out


# --- align / distribute / drop-to-ground -------------------------------------


def align(boxes: Boxes, axis: int, mode: AlignMode) -> dict[int, np.ndarray]:
    """A world-space translation delta per owner that lines up every box's
    edge (or centre) along ``axis``.

    **Works on box extremes, not on origins -- and that is the whole reason
    ``align(..., "centre")`` disagrees with aligning bare translations.** Two
    boxes of different sizes sharing a translation do not share a centre, and
    "align to centre" means the visible middle of the shape, not wherever its
    pivot happens to sit inside it.
    """
    _check_axis(axis)
    if not boxes:
        return {}
    los = {owner: np.asarray(lo, dtype="f8") for owner, (lo, _hi) in boxes.items()}
    his = {owner: np.asarray(hi, dtype="f8") for owner, (_lo, hi) in boxes.items()}
    if mode == "min":
        target = min(lo[axis] for lo in los.values())
        current = {owner: lo[axis] for owner, lo in los.items()}
    elif mode == "max":
        target = max(hi[axis] for hi in his.values())
        current = {owner: hi[axis] for owner, hi in his.items()}
    elif mode == "centre":
        lowest = min(lo[axis] for lo in los.values())
        highest = max(hi[axis] for hi in his.values())
        target = (lowest + highest) / 2.0
        current = {owner: (los[owner][axis] + his[owner][axis]) / 2.0 for owner in boxes}
    else:
        raise ValueError(f"mode must be 'min', 'centre' or 'max', got {mode!r}")
    out: dict[int, np.ndarray] = {}
    for owner in boxes:
        delta = np.zeros(3, dtype="f8")
        delta[axis] = target - current[owner]
        out[owner] = delta
    return out


def distribute(boxes: Boxes, axis: int) -> dict[int, np.ndarray]:
    """A world-space translation delta per owner that leaves equal gaps
    between neighbouring box *edges* along ``axis``, the two extreme items held
    fixed.

    **Fewer than three items is a no-op, not an error.** Two items have a
    single gap between them and nothing to distribute it against -- there is
    no "even" or "uneven" with one interval -- and a refusal mid-gesture (the
    user dragged a marquee over two props and clicked Distribute) has nothing
    useful to show. An empty dict is returned rather than raising, exactly the
    register ``terrain.py``'s brushes return ``None`` in for "nothing to do".
    """
    _check_axis(axis)
    if len(boxes) < 3:
        return {}
    items = sorted(
        boxes.items(), key=lambda kv: float(kv[1][0][axis] + kv[1][1][axis]) / 2.0
    )
    first_lo = float(items[0][1][0][axis])
    last_hi = float(items[-1][1][1][axis])
    widths = [float(hi[axis] - lo[axis]) for _owner, (lo, hi) in items]
    span = last_hi - first_lo
    gap = (span - sum(widths)) / (len(items) - 1)
    out: dict[int, np.ndarray] = {}
    cursor = first_lo
    for (owner, (lo, _hi)), width in zip(items, widths, strict=True):
        delta = np.zeros(3, dtype="f8")
        delta[axis] = cursor - float(lo[axis])
        out[owner] = delta
        cursor += width + gap
    return out


def drop_to_ground(
    boxes: Boxes, *, terrain: Terrain | None = None, ground: float = 0.0
) -> dict[int, np.ndarray]:
    """A world-space translation delta per owner that rests each box's
    *bottom* on the ground, sampled under the box's own footprint centre.

    **The box's bottom, never its origin.** An origin-dropped prop sinks into
    the ground (or floats above it) by however much of its mesh hangs below
    its own pivot -- a barrel authored with its pivot at the middle floats
    half its height in the air, one authored at the rim sinks half its height
    in. Reading the box's own ``lo[1]`` and moving *that* onto the ground is
    the entire reason this op earns its keep over a properties-panel Y field:
    it is correct regardless of where any given asset's pivot happens to sit.

    Samples ``terrain.height_at`` (local space) or the flat ``ground`` plane
    otherwise, at the box's own XZ centre -- this module has no node transform
    to carry a terrain's own placement through, so a document with a terrain
    that is not sitting at the world origin is the caller's problem to solve
    before calling this, the same way a caller already has to for ``ground``
    meaning anything but world Y zero.
    """
    out: dict[int, np.ndarray] = {}
    for owner, (lo, hi) in boxes.items():
        lo_arr = np.asarray(lo, dtype="f8")
        hi_arr = np.asarray(hi, dtype="f8")
        cx = float(lo_arr[0] + hi_arr[0]) / 2.0
        cz = float(lo_arr[2] + hi_arr[2]) / 2.0
        target = height_at(terrain, cx, cz) if terrain is not None else float(ground)
        delta = np.zeros(3, dtype="f8")
        delta[1] = target - float(lo_arr[1])
        out[owner] = delta
    return out


# --- snapping -----------------------------------------------------------------


def _snap_value(value: float, step: float) -> float:
    """``value`` to the nearest multiple of ``step``, half away from zero --
    ``clay.ops.snap_value``'s own tie-break, restated: breaking a tie by
    parity (Python's ``round``) makes some half-way points on the grid stick
    backwards and others forwards with nothing visible to say which, where
    away-from-zero keeps the grid symmetric about the origin.
    """
    step = abs(float(step))
    if step == 0.0:
        return float(value)
    return math.copysign(math.floor(abs(value) / step + 0.5), value) * step


def snap_translation(vec: Sequence[float], step: float) -> np.ndarray:
    """Each component of a position onto a grid of ``step`` metres.

    **``step == 0`` is the identity, not a division by zero.** It reads like a
    degenerate case to guard against and it is really the *off switch*:
    ``step`` comes straight from a settings field, and the user turns
    snapping off by clearing it -- a ``round(v / step) * step`` written
    without this check would divide by zero the instant they did.
    ``clay.ops.snap_translation`` makes the identical argument for the
    identical reason, one document over.
    """
    out = np.asarray(vec, dtype="f8")
    if step == 0.0:
        return out.copy()
    return np.array([_snap_value(float(v), step) for v in out], dtype="f8")


def snap_rotation(quat_xyzw: Sequence[float], degrees: float) -> np.ndarray:
    """The rotation's *angle* quantised to ``degrees``, its axis kept exactly.

    Snapping the angle rather than decomposing to Euler angles and snapping
    each is the whole decision, and ``clay.ops.snap_rotation`` states why:
    Euler components are not independent, so quantising each one moves the
    axis as well as the angle, and an object tilted off the cardinal axes
    would visibly swing the moment snapping is turned on even though nothing
    asked it to. **``degrees == 0`` is the same off switch** ``snap_translation``
    has, for the same reason -- unchanged input, not a division.
    """
    q = np.asarray(quat_xyzw, dtype="f8")
    if degrees == 0.0:
        return q.copy()
    axis_len = float(np.linalg.norm(q[:3]))
    if axis_len == 0.0:
        # The identity rotation has no axis to keep; any answer other than
        # the identity would invent one out of nothing.
        return m3.quat_identity()
    angle = 2.0 * math.atan2(axis_len, float(q[3]))
    snapped = math.radians(_snap_value(math.degrees(angle), degrees))
    return m3.quat_from_axis_angle(q[:3] / axis_len, snapped)
