"""Where a shape the user just asked for should land, and whether the camera sees it.

Pure arithmetic over axis-aligned boxes: no imgui, no document, no camera. The
human Add door (``ui/panes/tools.place_primitive``) feeds it the document's
world boxes and gets back a translation; the camera step feeds
:func:`in_frustum` the live view-projection. Kept out of ``add_primitive`` on
purpose -- the agent places through that door and the frozen Familiar card says
a box of height ``h`` stands at ``[0, h/2, 0]`` *after the agent's own
translation*, so the pure door stays origin-placed and only the click gets a
better spot.

**Why it exists.** A new shape used to land at the origin with the middle of
it on the ground plane -- half of it buried, on top of whatever was already
there, and nothing framed it -- so the first thing a new user saw of their own
work was a half-sunk box they could not see all of. Three rules, in order:

1. **Ground.** The lowest point sits on ``y = 0`` exactly (``-lo.y``, not a
   rounded number, so ``lo.y + t.y`` is ``0.0`` and a test can say ``==``).
2. **Beside, never inside.** The footprint searches outward from the origin --
   or from the selected object, when exactly one is selected -- in
   :data:`STEP` rings and takes the first spot that keeps :data:`GAP` clear of
   every box in plan view, snapped to :data:`SNAP` so a placed shape sits on
   the grid the Move tool snaps to.
3. **Rest on top** only when no spot within :data:`RINGS` rings is free (a
   slab 100 m across, say): the shape then stands on the highest box under its
   footprint rather than inside it.

Plan view only (X and Z) for rule 2, deliberately: a box floating above the
ground is still something a new shape should not land underneath.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: A world axis-aligned box, ``(lo, hi)``, each a length-3 array.
Box = tuple[np.ndarray, np.ndarray]

#: The clear space kept between a new shape's footprint and every other
#: footprint, in metres.
GAP = 0.15
#: The search step, in metres: half a metre is the size of the smallest thing a
#: modeller blocks out, so a ring is a visible move rather than a nudge.
STEP = 0.5
#: Where the final X and Z are snapped, in metres -- an eighth of a metre, the
#: step Clay's snap-to-grid offers, so a placed shape is already on the grid.
SNAP = 0.125
#: How many rings of :data:`STEP` the search covers (12 m each way).
RINGS = 24


def _offsets() -> np.ndarray:
    """Every ring offset, nearest first, ties broken counter-clockwise from +X.

    +X first is what makes the second shape land to the right of the first --
    the same side every time, so the layout a user builds is predictable
    rather than scattered. Computed once; the table is data.
    """
    axis = np.arange(-RINGS, RINGS + 1)
    ix, iz = np.meshgrid(axis, axis, indexing="ij")
    ix, iz = ix.ravel(), iz.ravel()
    angle = np.mod(np.arctan2(iz, ix), 2.0 * np.pi)
    # arctan2 of (0, 0) is 0 and a rounding-edge angle of 2*pi - eps would sort
    # last among its ring; fold anything within a hair of a full turn to 0.
    angle = np.where(angle > 2.0 * np.pi - 1e-9, 0.0, angle)
    order = np.lexsort((angle, ix * ix + iz * iz))
    return np.stack([ix[order], iz[order]], axis=1).astype("f8") * STEP


_OFFSETS = _offsets()


def snap(value: float, grid: float = SNAP) -> float:
    """*value* rounded to the nearest multiple of *grid*."""
    return float(np.round(value / grid) * grid)


def _footprints(boxes: Sequence[Box]) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` as ``(n, 2)`` arrays over X and Z, finite boxes only."""
    lo = np.empty((len(boxes), 2))
    hi = np.empty((len(boxes), 2))
    keep = 0
    for box_lo, box_hi in boxes:
        a = np.asarray(box_lo, dtype="f8")
        b = np.asarray(box_hi, dtype="f8")
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        lo[keep] = a[[0, 2]]
        hi[keep] = b[[0, 2]]
        keep += 1
    return lo[:keep], hi[:keep]


def support_height(lo: np.ndarray, hi: np.ndarray, others: Sequence[Box]) -> float:
    """The top of the highest box whose footprint overlaps ``lo``..``hi`` in plan
    view, or ``0.0`` (the ground) when none does.

    *lo* and *hi* are the footprint's X and Z (length-2). Touching is not
    overlapping: a shape that merely abuts another's side stands on the ground.
    """
    top = 0.0
    for box_lo, box_hi in others:
        a = np.asarray(box_lo, dtype="f8")
        b = np.asarray(box_hi, dtype="f8")
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        if lo[0] < b[0] and hi[0] > a[0] and lo[1] < b[2] and hi[1] > a[2]:
            top = max(top, float(b[1]))
    return top


def spawn_translation(
    box: Box,
    others: Sequence[Box],
    *,
    beside: Box | None = None,
) -> np.ndarray:
    """The translation that puts a new shape on the ground, clear of *others*.

    *box* is the new shape's own bounds at the identity transform (a new shape
    has none), *others* every box the document already holds that is in the way
    (visible ones -- the caller decides), and *beside* the one selected object's
    box when exactly one is selected: the search then starts at its centre
    rather than at the origin, so a shape added while another is selected lands
    next to it. -> a length-3 translation; Y is ``-lo.y`` exactly.
    """
    lo = np.asarray(box[0], dtype="f8")
    hi = np.asarray(box[1], dtype="f8")
    ground = -float(lo[1])
    own_lo, own_hi = lo[[0, 2]], hi[[0, 2]]

    anchor = np.zeros(2)
    if beside is not None:
        near_lo, near_hi = np.asarray(beside[0], dtype="f8"), np.asarray(beside[1], dtype="f8")
        if np.isfinite(near_lo).all() and np.isfinite(near_hi).all():
            # The new shape's *centre* on the selected shape's centre, so the
            # first free ring is the nearest clear side whatever either pivot is.
            anchor = ((near_lo + near_hi) * 0.5)[[0, 2]] - (own_lo + own_hi) * 0.5

    spots = np.round((anchor + _OFFSETS) / SNAP) * SNAP
    other_lo, other_hi = _footprints(others)
    if len(other_lo) == 0:
        return np.array([spots[0, 0], ground, spots[0, 1]])

    chunk = 64
    for start in range(0, len(spots), chunk):
        part = spots[start : start + chunk]
        new_lo = (own_lo + part)[:, None, :]
        new_hi = (own_hi + part)[:, None, :]
        clash = ((new_lo < other_hi + GAP) & (new_hi > other_lo - GAP)).all(axis=2).any(axis=1)
        free = np.flatnonzero(~clash)
        if len(free):
            x, z = part[free[0]]
            return np.array([x, ground, z])

    # Nothing free in reach: stand on whatever is under the first spot instead
    # of inside it.
    x, z = spots[0]
    rest = support_height(own_lo + spots[0], own_hi + spots[0], others)
    return np.array([x, ground + rest, z])


def in_frustum(
    lo: np.ndarray,
    hi: np.ndarray,
    view_proj: np.ndarray,
    eye: np.ndarray,
    *,
    margin: float = 0.0,
) -> bool:
    """Whether all of the box ``lo``..``hi`` is in front of the camera and inside
    the picture, *margin* (a fraction of the picture's size) in from every edge.

    Reuses :func:`~realmspinner.kernels.mesh.pick.project`, the projection the
    picker already trusts, rather than a second copy of the divide: it keeps a
    vertex behind the eye out of the picture instead of mirroring it through
    the origin, which is the failure a hand-rolled clip-space test would have.
    """
    from ....kernels.mesh import pick

    a = np.asarray(lo, dtype="f8")
    b = np.asarray(hi, dtype="f8")
    corners = np.array(
        [[x, y, z] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])]
    )
    screen = pick.project(corners, np.eye(4), view_proj, eye, 1, 1)
    if not screen.front.all():
        return False
    margin = max(0.0, min(float(margin), 0.49))
    return bool(((screen.xy >= margin) & (screen.xy <= 1.0 - margin)).all())
