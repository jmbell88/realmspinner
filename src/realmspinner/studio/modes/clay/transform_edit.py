"""The arithmetic behind the Properties panel's rotation and size fields.

Pure and drawn-nothing, so the two claims that need proving -- a typed angle is
not rewritten while it is being typed, and a size edit lands on the scale that
reproduces it -- are provable without a frame.

**Rotation.** The document stores a quaternion; the panel shows Euler degrees
in the one order the app has (XYZ, the MCP surface's). Decomposing a quaternion
is not stable from frame to frame -- 190 degrees comes back as -170 -- so the
panel remembers, per object, the Euler it last showed *and the quaternion that
Euler produced*. While the object's quaternion is still that one the remembered
angles are shown verbatim; the moment anything else moves it (a gizmo drag, an
undo, an agent) the cache misses and the angles are read afresh.

**Size.** ``width/height/depth`` is the mesh's *local* extent times
``|scale|`` -- the one number that maps back to a scale without ambiguity. (The
world box of a rotated object is a different number and no scale reproduces it,
which is why that stays a read-only line underneath.)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ....kernels.geom3d import math3d as m3
from ....kernels.mesh import mesh as bm

#: ``uid -> (quaternion the angles were made from, the angles)``.
EulerCache = dict[int, tuple[tuple[float, ...], tuple[float, float, float]]]

#: A mesh flatter than this on an axis has nothing to scale a size *from*.
FLAT_EXTENT = 1e-9
#: The smallest size an edit may ask for, in metres. Zero would collapse the
#: axis to a plane the field could never be edited back out of.
MIN_SIZE = 1e-6

_SAME_QUAT = 1e-9


def displayed_euler(
    cache: EulerCache, uid: int, quat: Sequence[float]
) -> tuple[float, float, float]:
    """The angles to show for *uid*'s rotation, stable while it has not moved."""
    q = tuple(float(v) for v in quat)
    hit = cache.get(uid)
    if hit is not None and np.allclose(hit[0], q, atol=_SAME_QUAT, rtol=0.0):
        return hit[1]
    euler = m3.euler_xyz_from_quat(q)
    cache[uid] = (q, euler)
    return euler


def remember_euler(
    cache: EulerCache, uid: int, quat: Sequence[float], euler: Sequence[float]
) -> None:
    """Pin what was just typed against the quaternion it produced."""
    x, y, z = (float(v) for v in euler)
    cache[uid] = (tuple(float(v) for v in quat), (x, y, z))


def local_extent(mesh: Any) -> np.ndarray:
    """The mesh's own box edge lengths, ``(3,)`` f8, before any transform."""
    lo, hi = bm.bounds(mesh)
    return hi - lo


def size_of(extent: Sequence[float], scale: Sequence[float]) -> np.ndarray:
    return np.asarray(extent, dtype="f8") * np.abs(np.asarray(scale, dtype="f8"))


def axis_refusal(extent: Sequence[float], axis: int) -> str | None:
    """Why *axis* cannot be edited, or ``None``. Names the axis the way the row does."""
    if float(extent[axis]) <= FLAT_EXTENT:
        name = ("width", "height", "depth")[axis]
        return f"The mesh has no {name} to scale -- it is flat on this axis."
    return None


def resized_scale(
    extent: Sequence[float],
    scale: Sequence[float],
    axis: int,
    new_size: float,
    *,
    lock_aspect: bool = False,
) -> np.ndarray | None:
    """The scale that gives *axis* the size *new_size*, or ``None`` for a no-op.

    ``None`` for a flat axis, a non-finite or non-positive size -- the caller
    leaves the document alone rather than inventing a value. The sign of the
    existing scale is kept (a negative scale is whatever the document already
    allowed; this edit must not be the thing that flips it). With
    ``lock_aspect`` the other two axes follow by the same ratio the edited one
    moved by, so the shape keeps its proportions.
    """
    if axis_refusal(extent, axis) is not None:
        return None
    new_size = float(new_size)
    if not math.isfinite(new_size) or new_size < MIN_SIZE:
        return None
    ext = np.asarray(extent, dtype="f8")
    out = np.array(scale, dtype="f8", copy=True)
    old = float(ext[axis] * abs(out[axis]))
    sign = -1.0 if out[axis] < 0 else 1.0
    out[axis] = sign * new_size / float(ext[axis])
    if lock_aspect and old > FLAT_EXTENT:
        ratio = new_size / old
        for other in range(3):
            if other != axis:
                out[other] *= ratio
    return out
