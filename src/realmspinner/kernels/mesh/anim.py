"""Keyframe animation for a Clay rig: the data, the caps and the sampling.

An animation is a set of **channels**, each one property of one bone over time: a bone's
``rotation`` (a quaternion, XYZW), ``translation`` or ``scale``. A channel's values are the
**absolute local transform** of the bone at that moment -- what a glTF node channel holds -- so
exporting one is a copy, importing one is a copy, and a bone with no channel for a property plays
its rest value (the bone object's own transform). There is one source of truth: the pose a
viewport shows at time ``t``, the pose ``Export Posed`` writes and the frame a glTF player
reaches are all :func:`pose` of the same animation at the same time.

Everything here is pure and immutable. An :class:`Animation` is a frozen value whose channels hold
read-only arrays, and a document edits animation by *replacing* it (``ClayDoc`` pushes a
``RigEdit`` holding the before and after), the way a ``Mesh`` is edited: undo is a snapshot swap,
and a reference taken by a save snapshot keeps meaning what it meant.

**The caps are the reader's and the writer's both** (``serialize`` holds a document to the same
numbers, and ``gltf`` the same for an imported file): :data:`MAX_ANIMATIONS`,
:data:`MAX_KEYS_PER_CHANNEL`, :data:`MAX_ANIMATION_SECONDS`.

**Interpolation is one of two** (:data:`INTERPOLATIONS`): ``linear`` (``slerp`` for rotation,
which takes the shorter way round) and ``step``. A glTF ``CUBICSPLINE`` channel is resampled to
linear on import by :func:`resample_cubic` and the importer says so; Clay has no tangent handles to
keep.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "FPS_CHOICES",
    "INTERPOLATIONS",
    "MAX_ANIMATIONS",
    "MAX_ANIMATION_SECONDS",
    "MAX_KEYS_PER_CHANNEL",
    "PROPS",
    "Animation",
    "Channel",
    "RigState",
    "pose",
    "resample_cubic",
    "sample",
    "wrap",
]

MAX_ANIMATIONS = 64
MAX_KEYS_PER_CHANNEL = 4096
MAX_ANIMATION_SECONDS = 600.0
FPS_CHOICES = (12, 24, 30, 60)
INTERPOLATIONS = ("linear", "step")

#: The three properties a bone animates, and how many numbers each holds.
PROPS: dict[str, int] = {"rotation": 4, "translation": 3, "scale": 3}

#: Two keys closer than this are one key: a time is a float typed or dragged, and ``0.1 + 0.2``
#: must not make a second key where the first already is.
TIME_EPSILON = 1e-6


@dataclass(frozen=True, eq=False)
class Channel:
    """One property of one bone: ``times`` (K,) ascending seconds and ``values`` (K, 3|4)."""

    times: Any
    values: Any

    def __post_init__(self) -> None:
        times = np.array(self.times, dtype="f8", copy=True).reshape(-1)
        values = np.array(self.values, dtype="f8", copy=True)
        if values.ndim != 2 or values.shape[0] != len(times) or values.shape[1] not in (3, 4):
            raise ValueError(
                f"a channel needs one row of 3 or 4 numbers per key, got {values.shape} "
                f"for {len(times)} key(s)"
            )
        if len(times) == 0:
            raise ValueError("a channel must have at least one key")
        if len(times) > MAX_KEYS_PER_CHANNEL:
            raise ValueError(
                f"a channel has {len(times):,} keys, past the {MAX_KEYS_PER_CHANNEL:,} Clay holds"
            )
        if not (np.isfinite(times).all() and np.isfinite(values).all()):
            raise ValueError("a channel has a time or a value that is not a finite number")
        if times[0] < 0.0 or times[-1] > MAX_ANIMATION_SECONDS:
            raise ValueError(f"a key lies outside 0..{MAX_ANIMATION_SECONDS:g} seconds")
        if len(times) > 1 and not (np.diff(times) > 0.0).all():
            raise ValueError("a channel's key times must strictly increase")
        if values.shape[1] == 4 and not (np.linalg.norm(values, axis=1) > 1e-9).all():
            raise ValueError("a rotation key is a zero quaternion")
        times.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    def __len__(self) -> int:
        return len(self.times)

    @property
    def end(self) -> float:
        return float(self.times[-1])


def _validate_prop(prop: str, channel: Channel) -> None:
    if prop not in PROPS:
        raise ValueError(f"{prop!r} is not an animatable property ({', '.join(PROPS)})")
    if channel.values.shape[1] != PROPS[prop]:
        raise ValueError(f"a {prop} channel holds {PROPS[prop]} numbers per key")


@dataclass(frozen=True, eq=False)
class Animation:
    """A named set of channels: ``channels[bone uid][property] -> Channel``.

    ``loop`` says whether time wraps at :attr:`duration` when played; ``interpolation`` is
    between keys. Frozen; ``with_key`` and friends return a new animation.
    """

    name: str
    loop: bool = True
    interpolation: str = "linear"
    channels: Mapping[int, Mapping[str, Channel]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {', '.join(INTERPOLATIONS)}")
        if not str(self.name).strip():
            raise ValueError("an animation needs a name")
        frozen: dict[int, dict[str, Channel]] = {}
        for uid, props in self.channels.items():
            kept: dict[str, Channel] = {}
            for prop, channel in props.items():
                _validate_prop(prop, channel)
                kept[prop] = channel
            if kept:
                frozen[int(uid)] = kept
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "loop", bool(self.loop))
        object.__setattr__(self, "channels", frozen)

    # -- questions ----------------------------------------------------------------------

    @property
    def duration(self) -> float:
        """The last key's time over every channel, or 0 for an animation with none."""
        ends = [c.end for props in self.channels.values() for c in props.values()]
        return max(ends) if ends else 0.0

    def key_times(self, uid: int | None = None) -> list[float]:
        """Every distinct key time, for one bone or the whole animation, ascending."""
        found: set[float] = set()
        for bone, props in self.channels.items():
            if uid is not None and bone != uid:
                continue
            for channel in props.values():
                found.update(round(float(t), 6) for t in channel.times)
        return sorted(found)

    def bones(self) -> list[int]:
        return sorted(self.channels)

    # -- editing: each returns a new animation ------------------------------------------------

    def with_key(self, uid: int, prop: str, time: float, value: Any) -> Animation:
        """A key set at *time* (replacing one within :data:`TIME_EPSILON`)."""
        if not np.isfinite(time) or not 0.0 <= time <= MAX_ANIMATION_SECONDS:
            raise ValueError(f"a key time must lie in 0..{MAX_ANIMATION_SECONDS:g} seconds")
        row = np.array(value, dtype="f8").reshape(-1)
        if prop == "rotation":
            norm = float(np.linalg.norm(row)) if len(row) == 4 else 0.0
            if norm <= 1e-9:
                raise ValueError("a rotation key needs a non-zero quaternion")
            row = row / norm
        old = self.channels.get(uid, {}).get(prop)
        if old is None:
            times, values = np.array([time]), row[None, :]
        else:
            near = np.flatnonzero(np.abs(old.times - time) <= TIME_EPSILON)
            times, values = old.times.copy(), old.values.copy()
            if len(near):
                values[near[0]] = row
                times[near[0]] = time
            else:
                at = int(np.searchsorted(times, time))
                times, values = np.insert(times, at, time), np.insert(values, at, row, axis=0)
        return self._replace_channel(uid, prop, Channel(times, values))

    def without_key(self, uid: int, prop: str, time: float) -> Animation:
        """The key at *time* removed; a channel left with none is dropped."""
        old = self.channels.get(uid, {}).get(prop)
        if old is None:
            return self
        keep = np.abs(old.times - time) > TIME_EPSILON
        if keep.all():
            return self
        if not keep.any():
            return self._replace_channel(uid, prop, None)
        return self._replace_channel(uid, prop, Channel(old.times[keep], old.values[keep]))

    def moved(self, uid: int, prop: str, time: float, to: float) -> Animation:
        """The key at *time* moved to *to*, taking the place of any key already there."""
        old = self.channels.get(uid, {}).get(prop)
        if old is None:
            return self
        match = np.flatnonzero(np.abs(old.times - time) <= TIME_EPSILON)
        if not len(match):
            return self
        value = old.values[match[0]]
        return self.without_key(uid, prop, time).with_key(uid, prop, to, value)

    def without_bones(self, uids: Any) -> Animation:
        drop = {int(u) for u in uids}
        if not drop & set(self.channels):
            return self
        return Animation(
            self.name,
            self.loop,
            self.interpolation,
            {u: p for u, p in self.channels.items() if u not in drop},
        )

    def renamed(self, name: str) -> Animation:
        return Animation(name, self.loop, self.interpolation, self.channels)

    def with_loop(self, loop: bool) -> Animation:
        return Animation(self.name, loop, self.interpolation, self.channels)

    def with_interpolation(self, interpolation: str) -> Animation:
        return Animation(self.name, self.loop, interpolation, self.channels)

    def _replace_channel(self, uid: int, prop: str, channel: Channel | None) -> Animation:
        channels = {u: dict(p) for u, p in self.channels.items()}
        props = channels.setdefault(uid, {})
        if channel is None:
            props.pop(prop, None)
        else:
            props[prop] = channel
        return Animation(self.name, self.loop, self.interpolation, channels)


@dataclass(frozen=True, eq=False)
class RigState:
    """The animations of a document and the frame rate they are shown at: everything about
    a rig's motion that an undo step restores.

    Frozen, and replaced wholesale by an edit (``RigEdit`` swaps one for another). Which
    animation is *selected*, and where the playhead is, are not here: like the object
    selection they are not edits, so a click on a different animation never makes a document
    ask to be saved.
    """

    animations: tuple[Animation, ...] = ()
    fps: int = 24

    def __post_init__(self) -> None:
        if self.fps not in FPS_CHOICES:
            raise ValueError(f"fps must be one of {', '.join(str(f) for f in FPS_CHOICES)}")
        animations = tuple(self.animations)
        if len(animations) > MAX_ANIMATIONS:
            raise ValueError(
                f"{len(animations)} animations, past the {MAX_ANIMATIONS} Clay holds"
            )
        names = [a.name for a in animations]
        if len(set(names)) != len(names):
            raise ValueError("two animations share a name")
        object.__setattr__(self, "animations", animations)

    def with_animations(self, animations: Any) -> RigState:
        return RigState(tuple(animations), self.fps)

    def with_fps(self, fps: int) -> RigState:
        return RigState(self.animations, int(fps))

    def __bool__(self) -> bool:
        return bool(self.animations) or self.fps != 24

    def nbytes(self) -> int:
        """The arrays this state holds, each distinct channel once -- what an undo step pins."""
        seen: set[int] = set()
        total = 0
        for animation in self.animations:
            for props in animation.channels.values():
                for channel in props.values():
                    if id(channel) not in seen:
                        seen.add(id(channel))
                        total += channel.times.nbytes + channel.values.nbytes
        return total


# -- sampling ----------------------------------------------------------------------------


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation along the shorter arc, normalised."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 0.9995:
        out = a + t * (b - a)
    else:
        theta = np.arccos(min(dot, 1.0))
        out = (np.sin((1.0 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)
    return out / np.linalg.norm(out)


def sample(channel: Channel, time: float, interpolation: str = "linear") -> np.ndarray:
    """The channel's value at *time*: held before the first key and after the last."""
    times, values = channel.times, channel.values
    rotation = values.shape[1] == 4
    if time <= times[0] or len(times) == 1:
        out = values[0]
    elif time >= times[-1]:
        out = values[-1]
    else:
        right = int(np.searchsorted(times, time, side="right"))
        left = right - 1
        if interpolation == "step":
            out = values[left]
        else:
            span = times[right] - times[left]
            t = (time - times[left]) / span if span > 0 else 0.0
            if rotation:
                return _slerp(values[left], values[right], float(t))
            out = values[left] + (values[right] - values[left]) * t
    out = np.array(out, dtype="f8", copy=True)
    if rotation:
        out /= np.linalg.norm(out)
    return out


def wrap(animation: Animation, time: float) -> float:
    """*time* folded into the animation: wrapped at its duration when it loops, clamped to
    ``0..duration`` when it does not."""
    duration = animation.duration
    if duration <= 0.0:
        return 0.0
    if animation.loop:
        return float(time % duration)
    return float(min(max(time, 0.0), duration))


def pose(animation: Animation | None, time: float) -> dict[int, dict[str, np.ndarray]]:
    """Every animated bone's local transform at *time*: ``{uid: {prop: value}}``.

    Only the properties that have a channel are present; the rest are the bone's rest value,
    which the caller reads off the bone itself. ``None`` (no animation selected) is the rest pose.
    """
    if animation is None:
        return {}
    at = wrap(animation, time)
    return {
        uid: {
            prop: sample(channel, at, animation.interpolation) for prop, channel in props.items()
        }
        for uid, props in animation.channels.items()
    }


def resample_cubic(
    times: np.ndarray, values: np.ndarray, *, fps: float = 30.0, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """A glTF ``CUBICSPLINE`` channel evaluated onto a regular grid, as linear keys.

    *values* is ``(3 * K, width)`` in the spec's order (in-tangent, value, out-tangent per key).
    The Hermite spline is sampled every ``1/fps`` seconds across its span, plus its last key
    exactly, and a rotation is normalised. Clay keeps no tangent handles; this is the honest
    flattening, and the importer says it happened.
    """
    times = np.asarray(times, dtype="f8")
    flat = np.asarray(values, dtype="f8").reshape(len(times), 3, width)
    value_k, in_t, out_t = flat[:, 1], flat[:, 0], flat[:, 2]
    if len(times) == 1:
        return times.copy(), value_k.copy()
    grid = np.unique(np.concatenate([np.arange(times[0], times[-1], 1.0 / fps), times]))
    out = np.empty((len(grid), width))
    for n, t in enumerate(grid):
        k = min(max(int(np.searchsorted(times, t, side="right")) - 1, 0), len(times) - 2)
        h = times[k + 1] - times[k]
        s = (t - times[k]) / h
        s2, s3 = s * s, s * s * s
        out[n] = (
            (2 * s3 - 3 * s2 + 1) * value_k[k]
            + (s3 - 2 * s2 + s) * h * out_t[k]
            + (-2 * s3 + 3 * s2) * value_k[k + 1]
            + (s3 - s2) * h * in_t[k + 1]
        )
    if width == 4:
        out /= np.linalg.norm(out, axis=1, keepdims=True)
    return grid, out
