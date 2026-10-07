"""Axis constraints and typed values for a transform in progress.

A gizmo drag answers "how far" with the mouse, which is the right control for
placing something by eye and the wrong one for placing it *exactly*. Every
modelling package therefore lets the keyboard join a drag already under way:
``X`` locks it to an axis, digits type the number outright, and the pair
compose -- ``X`` then ``2`` is "two metres along X" with no dragging left in it
at all.

**It is a filter on the delta, not a second kind of drag.** Everything here is
``(quantity, lock) -> quantity``: the gizmo still produces a displacement, a
factor triple or a quaternion exactly as it did, and this narrows it on the way
past. That is what keeps the constraint out of the three gizmos, out of the
element-drag preview and out of the commit -- all of which go on knowing only
that they were handed a delta -- and it is what makes every rule below
assertable from three numbers rather than from a viewport.

**A typed value replaces the quantity; an axis lock only projects it.** The two
are deliberately different, and the difference is what a user means by each: a
lock is a statement about *direction* and leaves the mouse in charge of the
amount, while a number is the amount and leaves nothing. So a typed value with
no axis is still meaningful -- it sets the length of the displacement the mouse
chose the direction of, and the size of a uniform scale -- which is why the
lock is not a precondition for typing.

Pure: numpy plus ``kernels.geom3d.math3d`` for the one quaternion conversion, which is
the same reason every other module in this package reaches for it -- an XYZW
quaternion is built in exactly one place in this project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..geom3d import math3d as m3

__all__ = [
    "AXES",
    "DragInput",
    "constrain_rotation",
    "constrain_scale",
    "constrain_translation",
    "readout",
]

AXES: tuple[str, str, str] = ("x", "y", "z")

#: What a number may be built out of. A bare ``-`` or a trailing ``.`` parses to
#: nothing and is *kept*, because both are states a half-typed number passes
#: through and discarding them would make the minus sign unreachable.
_NUMERIC = set("0123456789.-")


def _index(axis: str) -> int | None:
    return AXES.index(axis) if axis in AXES else None


def _unit(axis: str) -> np.ndarray:
    out = np.zeros(3)
    out[AXES.index(axis)] = 1.0
    return out


@dataclass
class DragInput:
    """What the keyboard has said about the drag currently under way.

    Held on the view rather than in the app's Clay state, because it is about
    one drag: it is created empty at the press and dropped at the release, and a
    value that outlived a drag would silently constrain the next one.
    """

    axis: str = ""
    typed: str = ""
    #: The unit direction ``axis == "normal"`` locks a translation to. Set by the
    #: viewport when an Extrude begins its drag on faces (the average face
    #: normal, in world space); meaningless for any other axis value.
    normal: np.ndarray | None = field(default=None, repr=False, compare=False)
    #: Set once a number has been typed *and* is parseable, so a caller can tell
    #: "the user is typing" from "the user has typed a value", which read the
    #: same on the string alone at the moment it holds only ``-``.
    _cache: dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def active(self) -> bool:
        return bool(self.axis) or bool(self.typed)

    def value(self) -> float | None:
        """The typed number, or ``None`` while there is not one yet."""
        try:
            return float(self.typed)
        except ValueError:
            return None

    def key(self, name: str) -> bool:
        """Feed one key by pygame name. -> whether it was consumed.

        Pressing the locked axis again *clears* the lock rather than being
        ignored, which is the convention every package shares and the only one
        that leaves a mistyped ``X`` recoverable without abandoning the drag.
        """
        if name in AXES:
            self.axis = "" if self.axis == name else name
            return True
        if name == "backspace":
            self.typed = self.typed[:-1]
            return True
        if len(name) == 1 and name in _NUMERIC:
            self.typed += name
            return True
        return False


def constrain_translation(displacement: np.ndarray, drag: DragInput) -> np.ndarray:
    """A world-space displacement, narrowed by the axis lock and the typed value.

    With a number and no axis the *direction the mouse chose* is kept and only
    the length is replaced -- a zero-length displacement has no direction to
    keep, so it stays zero rather than inventing one.
    """
    out = np.asarray(displacement, dtype="f8").reshape(3).copy()
    value = drag.value()
    if drag.axis == "normal" and drag.normal is not None:
        # An extrude pulls along the face's own normal: the mouse sets how far,
        # the normal says which way, and a typed number is that distance.
        direction = np.asarray(drag.normal, dtype="f8").reshape(3)
        return direction * (value if value is not None else float(np.dot(out, direction)))
    index = _index(drag.axis)
    if value is not None:
        if index is not None:
            return _unit(drag.axis) * value
        length = float(np.linalg.norm(out))
        return out / length * value if length > 1e-12 else out
    if index is not None:
        kept = np.zeros(3)
        kept[index] = out[index]
        return kept
    return out


def constrain_scale(factors: np.ndarray, drag: DragInput) -> np.ndarray:
    """A per-axis scale triple, narrowed the same way.

    An unlocked axis goes back to **one**, not to whatever the gizmo said: the
    point of locking is that nothing else moves, and leaving the other two at
    their dragged values would be a lock that only renamed the readout.
    """
    out = np.asarray(factors, dtype="f8").reshape(3).copy()
    value = drag.value()
    index = _index(drag.axis)
    if value is not None:
        if index is not None:
            kept = np.ones(3)
            kept[index] = value
            return kept
        return np.full(3, value)
    if index is not None:
        kept = np.ones(3)
        kept[index] = out[index]
        return kept
    return out


def constrain_rotation(quat: np.ndarray, drag: DragInput) -> np.ndarray:
    """A rotation, narrowed to one axis and/or to a typed angle in **degrees**.

    Locked with no number, the drag's angle is *projected* onto the axis rather
    than kept whole -- the component of the sweep about that axis, which is what
    a user turning a free rotation into a constrained one is asking for and what
    keeps the result continuous as the lock goes on.

    With a number and no lock the drag's own axis is kept and only the angle is
    replaced, which mirrors the translation rule exactly. The identity rotation
    has no axis, so a number typed against one leaves it identity rather than
    inventing an axis to turn about.
    """
    q = np.asarray(quat, dtype="f8").reshape(4)
    value = drag.value()
    index = _index(drag.axis)
    if value is None and index is None:
        return q
    length = float(np.linalg.norm(q[:3]))
    angle = 2.0 * float(np.arctan2(length, float(q[3])))
    axis = q[:3] / length if length > 1e-12 else None

    if index is not None:
        wanted = _unit(drag.axis)
        if value is not None:
            return m3.quat_from_axis_angle(wanted, np.radians(value))
        if axis is None:
            return m3.quat_identity()
        return m3.quat_from_axis_angle(wanted, angle * float(axis @ wanted))
    if axis is None:
        return m3.quat_identity()
    return m3.quat_from_axis_angle(axis, np.radians(value))


# The unit each tool's readout is quoted in. Scale is a ratio and has none,
# which is stated here rather than by an empty branch in the formatter.
_UNITS = {"move": " m", "rotate": " deg", "scale": ""}


def readout(tool: str, quantity: np.ndarray, drag: DragInput) -> str:
    """The one line the HUD draws: what the drag currently amounts to.

    It shows the *result* rather than what was typed, so a locked axis with no
    number still reads as a number and a half-typed one reads as the value it
    will replace -- with the raw text appended while it is being entered, since
    a field you cannot see is not a field.
    """
    values = np.asarray(quantity, dtype="f8").reshape(-1)
    unit = _UNITS.get(tool, "")
    if tool == "rotate":
        body = f"{float(values[0]):.1f}{unit}"
    elif len(values) == 1:
        body = f"{float(values[0]):.3f}{unit}"
    else:
        body = "  ".join(f"{a.upper()} {float(v):.3f}" for a, v in zip(AXES, values, strict=True))
        body += unit
    parts = [f"[{drag.axis.upper()}]"] if drag.axis else []
    parts.append(body)
    if drag.typed:
        parts.append(f"= {drag.typed}")
    return "  ".join(parts)


def screen_angle(
    pivot: np.ndarray, normal: np.ndarray, was: np.ndarray, now: np.ndarray
) -> float:
    """The signed angle from ``was`` to ``now`` about ``normal``, in radians.

    Both points are projected into the plane through ``pivot`` first, so a ray
    hit that is a hair off the plane -- which every ray hit is, at float
    precision -- does not tilt the answer.

    Signed rather than absolute, and that is the whole of why it is a function:
    ``arccos`` of the dot product gives the angle and loses the direction, so a
    rotate drag turned the same way whichever way the mouse went round. The
    cross product against the axis is what carries the sign.

    Zero for a degenerate pair rather than a refusal: a pointer exactly on the
    pivot has no angle to report, and the caller is a live drag that must go on
    drawing.
    """
    pivot = np.asarray(pivot, dtype="f8")
    axis = np.asarray(normal, dtype="f8")
    first = np.asarray(was, dtype="f8") - pivot
    second = np.asarray(now, dtype="f8") - pivot
    first = first - axis * float(np.dot(first, axis))
    second = second - axis * float(np.dot(second, axis))
    len_first = float(np.linalg.norm(first))
    len_second = float(np.linalg.norm(second))
    if len_first < 1e-9 or len_second < 1e-9:
        return 0.0
    first = first / len_first
    second = second / len_second
    return float(
        math.atan2(float(np.dot(np.cross(first, second), axis)), float(np.dot(first, second)))
    )
