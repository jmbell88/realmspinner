"""The timeline's arithmetic: what is on the strip, where the playhead goes, how playback advances.

The drawing is :mod:`~realmspinner.studio.modes.clay.ui.panes.timeline`; this is the half a test can
assert without an imgui frame, which is why the ruler's ticks, the frame snap, the rows and the
playback step are plain functions of a document. Nothing here writes a document except through
its own doors (``set_rig_time``, ``move_bone_keys``, ``delete_bone_keys``), and playback is
session state: it is never an undo step and never reaches a file.

**A row is a bone, not a property.** The strip shows one diamond per bone per time, however many of
translation, rotation and scale are keyed there; sliding or deleting it takes all three, which is
what the person pressing it means. A bone appears when it has any key *or* is selected, so a bone
about to be keyed has a row to key it on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ....kernels.mesh import anim

#: The shortest span the ruler shows, so an empty animation still has room to put a first key.
MIN_SPAN = 1.0

FPS_CHOICES = anim.FPS_CHOICES


@dataclass
class Sheet:
    """Where the dope sheet is on screen: the geometry every hit test and every diamond shares.

    ``x0..x1`` is the track area (time 0 to :attr:`visible` seconds), ``gutter`` the name column
    to its left, ``y0`` the ruler's top; the rows start one ruler below. ``scroll`` is the first
    bone row shown, clamped to what exists.
    """

    x0: float
    x1: float
    y0: float
    ruler_h: float
    row_h: float
    visible: float
    gutter: float
    scroll: int
    rows_fit: int
    rows: int

    def __post_init__(self) -> None:
        self.scroll = self.clamp_scroll(self.scroll)

    def clamp_scroll(self, scroll: int) -> int:
        return int(min(max(scroll, 0), max(self.rows - self.rows_fit, 0)))

    def to_x(self, time: float) -> float:
        return self.x0 + (time / self.visible) * (self.x1 - self.x0)

    def to_t(self, x: float) -> float:
        return (x - self.x0) / max(self.x1 - self.x0, 1e-9) * self.visible

    def row_top(self, slot: int) -> float:
        """The y of the *slot*-th row on screen (0 is the first one shown)."""
        return self.y0 + self.ruler_h + slot * self.row_h

    def row_at(self, y: float) -> int | None:
        """The bone row (an index into the rows, scroll included) under *y*, or ``None`` over
        the ruler or past the last row."""
        offset = y - (self.y0 + self.ruler_h)
        if offset < 0.0:
            return None
        slot = int(offset // self.row_h)
        row = self.scroll + slot
        return row if slot < self.rows_fit and row < self.rows else None


@dataclass(frozen=True)
class Row:
    """One bone on the strip: ``times`` are every time any of its properties is keyed."""

    uid: int
    name: str
    times: tuple[float, ...]
    selected: bool


def length(doc: Any) -> float:
    """The selected animation's last key, or 0 with none selected or no keys."""
    animation = doc.animation()
    return 0.0 if animation is None else float(animation.duration)


def span(doc: Any) -> float:
    """The seconds the ruler shows: the animation's length rounded up to a whole second, never
    less than :data:`MIN_SPAN`, and never short of the playhead."""
    return float(max(MIN_SPAN, math.ceil(max(length(doc), doc.rig_time) - 1e-9)))


def frame_of(doc: Any, time: float) -> int:
    """The frame *time* falls on at the document's frame rate (nearest)."""
    return int(round(float(time) * doc.rig_state.fps))


def time_of(doc: Any, frame: int) -> float:
    return max(0, int(frame)) / float(doc.rig_state.fps)


def snap(doc: Any, time: float) -> float:
    """*time* on the nearest frame, within the ruler. Frames are how keys are authored and
    shown; a key between two of them is only ever imported."""
    return min(max(time_of(doc, frame_of(doc, time)), 0.0), anim.MAX_ANIMATION_SECONDS)


def rows(doc: Any) -> list[Row]:
    """The bones the strip lists, in document order."""
    animation = doc.animation()
    out = []
    for obj in doc.objects:
        if not obj.bone:
            continue
        selected = obj.uid in doc.selection
        times = tuple(animation.key_times(obj.uid)) if animation is not None else ()
        if times or selected:
            out.append(Row(obj.uid, obj.name, times, selected))
    return out


def ruler(visible: float, width: float) -> list[tuple[float, str, bool]]:
    """``(time, label, major)`` ticks for *visible* seconds across *width* pixels.

    Majors fall on a "round" step that keeps them at least ~80 px apart (0.1, 0.25, 0.5, 1, 2, 5,
    10... s), minors are a fifth of it, and only majors carry a label.
    """
    if visible <= 0.0 or width <= 0.0:
        return []
    per_px = visible / width
    want = 80.0 * per_px
    step = 600.0
    for candidate in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0):
        if candidate >= want:
            step = candidate
            break
    minor = step / 5.0
    out: list[tuple[float, str, bool]] = []
    for i in range(int(visible / minor) + 1):
        time = round(i * minor, 6)
        major = i % 5 == 0
        out.append((time, f"{time:g}s" if major else "", major))
    return out


def nearest_key(
    rows_: list[Row], x: float, row: int, to_x: Any, radius: float = 7.0
) -> tuple[int, float] | None:
    """The ``(bone, time)`` of the diamond on *row* within *radius* pixels of *x*, nearest first.

    *to_x* maps a time to its pixel, so the same rule serves any zoom.
    """
    if not 0 <= row < len(rows_):
        return None
    best: tuple[float, float] | None = None
    for time in rows_[row].times:
        gap = abs(to_x(time) - x)
        if gap <= radius and (best is None or gap < best[0]):
            best = (gap, time)
    return None if best is None else (rows_[row].uid, best[1])


# -- the transport ---------------------------------------------------------------------------


def go_to(doc: Any, time: float) -> None:
    """Park the playhead, on a frame."""
    doc.set_rig_time(snap(doc, time))


def go_start(doc: Any) -> None:
    doc.playing = False
    go_to(doc, 0.0)


def go_end(doc: Any) -> None:
    doc.playing = False
    go_to(doc, length(doc))


def step(doc: Any, frames: int) -> None:
    """One frame (or *frames*) forward or back, stopping playback. Never before 0."""
    doc.playing = False
    doc.set_rig_time(time_of(doc, frame_of(doc, doc.rig_time) + int(frames)))


def toggle_play(doc: Any) -> bool:
    """Start or stop playback. -> whether it is playing now.

    Starting needs an animation with something to play; at its end a one-shot restarts from 0,
    which is what pressing Play on a finished clip means.
    """
    if doc.playing:
        doc.playing = False
        doc.pose_rev += 1
        return False
    end = length(doc)
    if end <= 0.0:
        return False
    animation = doc.animation()
    if animation is not None and not animation.loop and doc.rig_time >= end - 1e-9:
        doc.set_rig_time(0.0)
    doc.playing = True
    doc.pose_rev += 1
    return True


def tick(doc: Any, dt: float) -> bool:
    """Advance playback by *dt* seconds. -> whether the playhead moved.

    A looping animation wraps past its end; a one-shot stops on its last key. Stops by itself
    when the animation has gone (deleted, or its keys all removed) or the document left Pose mode.
    """
    if not doc.playing:
        return False
    end = length(doc)
    animation = doc.animation()
    if not doc.pose_mode or animation is None or end <= 0.0:
        doc.playing = False
        doc.pose_rev += 1
        return False
    target = doc.rig_time + max(float(dt), 0.0)
    if target >= end:
        if animation.loop:
            target = target % end
        else:
            target = end
            doc.playing = False
    doc.set_rig_time(target)
    return True
