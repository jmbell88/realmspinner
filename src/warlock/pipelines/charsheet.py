"""Troupe's sheet planning: clip x direction -> frame table -> ``sheet.Plan``.

Pure arithmetic. No torch, no Blender, no ``service``, no filesystem -- the
same bargain ``sheet.py`` makes, and for the same reason: the browser preview,
the Blender renderer and the sidecar must never disagree about what cell 137
depicts, and the way to guarantee that is for one testable function to decide.

**The frame table is held twice.** ``studio.troupe.spec`` holds it as the
studio's answer and this module holds it as the pipeline's, because a
``pipelines`` module runs inside worker and Blender processes where ``studio``
is not importable at all, and ``studio/troupe`` imports nothing outward. That is
the ``spritesynth`` / ``inker.animation`` ``DIRECTION_ORDER`` arrangement at its
second instance, and it takes the same safeguard:
``tests/troupe/test_troupe_geometry_agreement.py`` is the **sole owner** of the
agreement between the two copies. A change to one is a change to both plus that
test, or a Troupe sheet and the editor that opens it come to mean different
things by ``walk_left``.

Cell order is grouped by ``(animation, direction)`` and dense -- the argument is
written out in ``studio.troupe.spec``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import sheet

__all__ = [
    "ANIMATIONS",
    "CAMERA_PRESETS",
    "COLUMNS",
    "COMPASS_16",
    "DEFAULT_CAMERA_PRESET",
    "DIRECTION_PRESETS",
    "DIRECTIONS",
    "FPS_CHOICES",
    "LAYOUT_VERSION",
    "MAX_CELLS",
    "MAX_FRAME_SIZE",
    "MAX_FRAMES",
    "MIN_FRAME_SIZE",
    "MOVEMENT_MIN_FRAMES",
    "WARN_CELLS",
    "SIZES",
    "ClipTiming",
    "LayoutSpec",
    "MovementSpec",
    "TroupeCell",
    "animation_block",
    "check_frame_counts",
    "check_subset",
    "compass_name",
    "frame_table",
    "movement_min_frames",
    "plan",
    "point_in_cell",
    "resolve_layout",
    "spans",
    "subset_indices",
]

#: ``(name, frames, loop, duration_ms)``, as a literal table rather than a
#: loop, so the frame table is readable in one glance.
ANIMATIONS: tuple[tuple[str, int, bool, int], ...] = (
    ("idle", 4, True, 150),
    ("walk", 8, True, 100),
    ("run", 8, True, 60),
    ("attack", 6, False, 80),
    ("jump", 6, False, 100),
)

#: ``(name, yaw)``, degrees clockwise from the front view. The four the rest of
#: the repo already names keep the yaws they already have.
DIRECTIONS: tuple[tuple[str, float], ...] = (
    ("front", 0.0),
    ("front_left", 45.0),
    ("left", 90.0),
    ("back_left", 135.0),
    ("back", 180.0),
    ("back_right", 225.0),
    ("right", 270.0),
    ("front_right", 315.0),
)

COLUMNS = 8
SIZES = (16, 24, 32, 48, 64, 96, 128, 256)
RENDER_SIZE = 512

#: **Task G, 2026-09-12: the custom sprite size's actual range.** ``SIZES`` is
#: the ladder of presets a form offers and stays the ladder the tests pin
#: (``tests/troupe/test_troupe_geometry_agreement.py`` ties
#: ``studio.troupe.spec`` to it); these two are the wider question ``plan``
#: itself answers, matching ``service.troupe.TROUPE_CUSTOM_SIZE_RANGE`` -- see
#: that constant's comment for why 8 and 256 are the floor and ceiling.
MIN_FRAME_SIZE = 8
MAX_FRAME_SIZE = 256

#: ``(key, label, elevation)`` -- the camera angles a character sheet may be
#: framed from, as a literal table for the same reason ``ANIMATIONS`` is one.
#:
#: ``isometric`` is 30.0 because that is :data:`sheet.DEFAULT_ELEVATION`, the
#: elevation every sheet this program has ever rendered used; naming it
#: "Isometric" rather than "2:1 dimetric" is deliberate, because "isometric" is
#: what players and every engine's tileset documentation call that projection
#: and a technically-correct label nobody searches for is a label nobody finds.
#:
#: ``top_down`` stops at 60 rather than reaching a true 90 overhead: a humanoid
#: seen straight down is a pair of shoulders and a hat brim -- it reads as a
#: blob, and every "top-down" sprite anyone actually ships is tilted.
#:
#: **The table lives in ``pipelines`` rather than in ``service.troupe``**
#: because the worker frames from it too, and ``pipelines`` is the one layer
#: both the door and the worker can see -- ``service`` is importable in neither
#: the Blender process nor the worker.
CAMERA_PRESETS: tuple[tuple[str, str, float], ...] = (
    ("three_quarter_top_down", "3/4 top-down", 35.0),
    ("isometric", "Isometric", 30.0),
    ("side", "Side", 0.0),
    ("top_down", "Top-down", 60.0),
)

DEFAULT_CAMERA_PRESET = "three_quarter_top_down"
#: 3 adds an open clip vocabulary: a movement may carry its own ``loop`` and
#: ``duration_ms`` (or resolve them from a rig's clip library via
#: ``resolve_layout``'s ``timing`` argument) instead of being one of
#: :data:`ANIMATIONS`' five names. ``resolve_layout`` still reads a 2, and
#: ``LayoutSpec.as_dict`` still writes one whenever the result is expressible
#: in it -- see ``docs/measurements/2026-09-12-troupe-open-clip-vocabulary.md``.
LAYOUT_VERSION = 3
WARN_CELLS = 256
MAX_CELLS = 512
MAX_FRAMES = sheet.MAX_CLIP_FRAMES

#: A layout-wide frame rate (Settings, an export choice) rather than an
#: arbitrary number: every legal ``fps`` has to divide cleanly enough that
#: ``round(1000 / fps)`` names a duration a person would recognise, and the
#: ladder is Troupe's own -- restated nowhere else, unlike
#: ``CLIP_DURATION_STEP_MS``, because no other module needs it.
FPS_CHOICES: tuple[int, ...] = (6, 8, 10, 12, 15, 24, 30)

#: A v3 movement's own ``loop``/``duration_ms`` are held to the same bounds a
#: clip library's ``duration_ms`` is (``rigging.MIN_CLIP_DURATION_MS`` /
#: ``MAX_CLIP_DURATION_MS`` / ``CLIP_DURATION_STEP_MS``), restated rather than
#: imported for the same reason ``rigging.LEGACY_CLIP_DURATION_MS`` restates
#: ``ANIMATIONS``: the two modules do not import each other.
MIN_MOVEMENT_DURATION_MS = 10
MAX_MOVEMENT_DURATION_MS = 1000
MOVEMENT_DURATION_STEP_MS = 10

_DIRECTIONS_16: tuple[tuple[str, float], ...] = (
    ("front", 0.0),
    ("front_front_left", 22.5),
    ("front_left", 45.0),
    ("left_front_left", 67.5),
    ("left", 90.0),
    ("left_back_left", 112.5),
    ("back_left", 135.0),
    ("back_back_left", 157.5),
    ("back", 180.0),
    ("back_back_right", 202.5),
    ("back_right", 225.0),
    ("right_back_right", 247.5),
    ("right", 270.0),
    ("right_front_right", 292.5),
    ("front_right", 315.0),
    ("front_front_right", 337.5),
)

DIRECTION_PRESETS: dict[int, tuple[tuple[str, float], ...]] = {
    1: (_DIRECTIONS_16[0],),
    4: tuple(_DIRECTIONS_16[i] for i in (0, 4, 8, 12)),
    8: tuple(_DIRECTIONS_16[i] for i in range(0, 16, 2)),
    16: _DIRECTIONS_16,
}

#: The sixteen-point compass, ``bearing 0 = N`` and clockwise from there --
#: the ordinary reading of a compass rose, used only as the lookup table
#: :func:`compass_name` indexes into.
_COMPASS_POINTS: tuple[str, ...] = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)


def compass_name(yaw: float) -> str:
    """The compass point a frame-folder export names direction ``yaw`` with.

    See ``docs/measurements/2026-09-12-troupe-open-clip-vocabulary.md``
    ("Compass names follow the camera arithmetic, not the docstring"):
    **bearing = (180 + yaw) mod 360**, derived from
    :func:`blender_worker._view_forward` -- the camera sits at
    ``centre - forward * distance``, the templates face -Y, and turning the
    subject clockwise as the camera sees it is what turning ``yaw`` up does.
    ``_aim_camera``'s own docstring ("yaw increases clockwise seen from
    above") describes the camera's *position*, which orbits the other way;
    this function follows the measurement's derivation, not that sentence.

    Raises for a ``yaw`` that is not a multiple of 22.5 degrees -- the
    sixteen-point rose has no finer answer to give.
    """
    index = yaw / 22.5
    if abs(index - round(index)) > 1e-6:
        raise ValueError(f"yaw must be a multiple of 22.5 degrees, not {yaw}")
    bearing = (180.0 + yaw) % 360.0
    return _COMPASS_POINTS[round(bearing / 22.5) % 16]


#: Every Troupe facing key, named by the compass point a frame-folder export
#: uses for it. Built from :data:`_DIRECTIONS_16` rather than hand-written,
#: so the two cannot silently disagree about which key is which point.
COMPASS_16: dict[str, str] = {key: compass_name(yaw) for key, yaw in _DIRECTIONS_16}

_ANIMATION_BY_NAME = {a[0]: a for a in ANIMATIONS}
MOVEMENT_MIN_FRAMES = {name: 1 for name, *_rest in ANIMATIONS}


def movement_min_frames(name: str) -> int:
    """The fewest frames any movement may have -- always 1.

    A function rather than a dict lookup, now that a layout may name a
    movement :data:`MOVEMENT_MIN_FRAMES` (built from the closed
    :data:`ANIMATIONS` table) has never heard of. ``characters/recipe.py``
    and ``service/troupe.py`` both already call this function rather than
    indexing the dict directly -- the 2026-09-14 audit (troupe-04) found this
    docstring still claiming otherwise. ``MOVEMENT_MIN_FRAMES`` itself is
    kept only because it is part of this module's exported surface
    (``__all__``); nothing in ``src/`` indexes it directly any more.
    """
    return 1


@dataclass(frozen=True, slots=True)
class ClipTiming:
    """One clip's timing, as a rig's clip library states it.

    Passed to :func:`resolve_layout` as its ``timing`` argument -- the
    service door's own answer to "what does this rig call a walk", built by
    ``clips.clip_timing`` from ``rigging.clip_library``. Naming it lets a
    layout resolve any clip a rig's library defines, not just the five
    :data:`ANIMATIONS` names. See
    ``docs/measurements/2026-09-12-troupe-open-clip-vocabulary.md``.
    """

    frames: int
    loop: bool
    duration_ms: int


@dataclass(frozen=True, slots=True)
class MovementSpec:
    name: str
    frames: int
    loop: bool
    duration_ms: int
    directions: tuple[tuple[str, float], ...]


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _validate_movement_loop(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} loop must be true or false")
    return value


def _validate_movement_duration_ms(
    value: Any, name: str, *, expected: int | None = None
) -> int:
    """A movement's own ``duration_ms``, validated one of two ways.

    ``expected`` is *not* ``None`` exactly when the layout carries a global
    ``fps``: every movement's duration is then ``round(1000 / fps)`` by
    construction (``resolve_layout`` recomputes it right after this call), and
    that number is almost never a multiple of 10 -- 167, 125, 83, 67, 42, 33ms
    for 6/8/12/15/24/30 fps. The step-of-10 rule below is a rule for
    *authored* clip durations (a human typing a number into a form), and
    applying it to an fps-derived snapshot is what made
    ``resolve_layout(as_dict_output)`` refuse its own output for every fps but
    10 -- the worker, a subset re-render and ``export_frames`` all resolve a
    stored layout with no ``timing`` in hand, so this was a stored sheet that
    could never be rendered or exported again. So an fps-carrying snapshot is
    held to the weaker, honest claim instead: this movement's own number must
    equal what its fps says it should be, not some other rule about round numbers.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} duration_ms must be a whole number of milliseconds")
    if expected is not None:
        if value != expected:
            raise ValueError(
                f"{name} duration_ms must be {expected} at this layout's fps, "
                f"not {value}"
            )
        return value
    if not (MIN_MOVEMENT_DURATION_MS <= value <= MAX_MOVEMENT_DURATION_MS):
        raise ValueError(
            f"{name} duration_ms must be {MIN_MOVEMENT_DURATION_MS}-"
            f"{MAX_MOVEMENT_DURATION_MS}, not {value}"
        )
    if value % MOVEMENT_DURATION_STEP_MS != 0:
        raise ValueError(
            f"{name} duration_ms must be a multiple of {MOVEMENT_DURATION_STEP_MS}, "
            f"not {value}"
        )
    return value


def _reject_direction_named_movement(name: str) -> None:
    """Raise if *name* ends in ``_<direction>`` for one of Troupe's 16 facings.

    Restated from ``rigging.reject_direction_named_clip`` -- ``pipelines``
    and the poser modules do not import each other, the same reason
    :data:`_ANIMATION_BY_NAME`'s legacy table is restated in ``rigging``
    rather than shared. A movement name is exactly the same trap a clip name
    is: Inker's tag parser reads ``walk_front`` as clip ``walk`` facing
    ``front``, whether the name came from a clip library or a layout.
    """
    for key, _yaw in _DIRECTIONS_16:
        if name.endswith(f"_{key}"):
            raise ValueError(
                f'{name!r} ends in "_{key}"; Troupe reads a name like that as '
                f'movement {name[: -len(key) - 1]!r} facing {key!r}'
            )


def _movement_base_timing(
    name: str,
    raw: Mapping[str, Any],
    version: int,
    timing: Mapping[str, ClipTiming] | None,
    fps: int | None,
) -> tuple[int, bool, int]:
    """``(frames, loop, duration_ms)`` before any global ``fps`` override.

    The precedence
    ``docs/measurements/2026-09-12-troupe-open-clip-vocabulary.md`` lays out:
    a rig's own clip library (``timing``) first; then a v3 snapshot's own
    literal ``loop``/``duration_ms``; then the closed legacy table --
    **always** the table for a v2 payload, even though a v2 movement dict may
    carry ``loop``/``duration_ms`` fields too (every ``as_dict`` output does),
    because that is what keeps a v2 layout resolving byte-identically to
    before this vocabulary opened.

    ``fps`` is the layout's own global rate, when it carries one -- passed
    down so the middle branch can validate a v3 movement's own ``duration_ms``
    against *that* number (``round(1000 / fps)``) instead of the multiple-of-10
    authoring rule: an fps-derived snapshot's duration is round(1000 / fps),
    almost never a multiple of 10, and it is a fact about ``fps`` rather than
    about a human typing a number into a form. See
    :func:`_validate_movement_duration_ms`.
    """
    if timing is not None:
        clip = timing.get(name)
        if clip is None:
            raise ValueError(f"{name!r} is not a clip of this skeleton")
        return clip.frames, clip.loop, clip.duration_ms
    if version >= 3 and "loop" in raw and "duration_ms" in raw:
        loop = _validate_movement_loop(raw["loop"], name)
        expected = round(1000 / fps) if fps is not None else None
        duration_ms = _validate_movement_duration_ms(
            raw["duration_ms"], name, expected=expected
        )
        legacy = _ANIMATION_BY_NAME.get(name)
        default_frames = legacy[1] if legacy is not None else 1
        return default_frames, loop, duration_ms
    legacy = _ANIMATION_BY_NAME.get(name)
    if legacy is not None:
        _n, frames, loop, ms = legacy
        return frames, loop, ms
    raise ValueError(f"{name!r} is not a Troupe movement")


def _expresses_as_v2(movements: tuple[MovementSpec, ...], fps: int | None) -> bool:
    """Whether *movements* need nothing v3 offers.

    A global ``fps`` always forces v3. Otherwise: every movement must be a
    legacy name whose *frames-independent* timing (``loop``, ``duration_ms``)
    equals :data:`ANIMATIONS`' -- frames themselves are not part of the
    check, because v2 already let a movement pick its own frame count.
    """
    if fps is not None:
        return False
    for movement in movements:
        legacy = _ANIMATION_BY_NAME.get(movement.name)
        if legacy is None:
            return False
        _n, _f, loop, ms = legacy
        if movement.loop != loop or movement.duration_ms != ms:
            return False
    return True


@dataclass(frozen=True, slots=True)
class LayoutSpec:
    """A validated, immutable Troupe frame-table snapshot."""

    version: int
    columns: int
    movements: tuple[MovementSpec, ...]
    fps: int | None = None

    @property
    def cell_count(self) -> int:
        return sum(m.frames * len(m.directions) for m in self.movements)

    @property
    def directions(self) -> tuple[tuple[str, float], ...]:
        out: list[tuple[str, float]] = []
        seen: set[str] = set()
        for movement in self.movements:
            for direction in movement.directions:
                if direction[0] not in seen:
                    seen.add(direction[0])
                    out.append(direction)
        return tuple(out)

    def as_dict(self) -> dict[str, Any]:
        """The wire form. Writes ``"version": 2`` whenever the layout needs
        nothing v3 offers (see :func:`_expresses_as_v2`) so a build from
        before this vocabulary opened still opens every sheet that uses only
        the five legacy movements -- byte-identical to what this method
        produced before v3 existed. ``"fps"`` appears only when the layout
        carries one."""
        runs = []
        index = 0
        movements = []
        for movement in self.movements:
            directions = [
                {"key": name, "label": name.replace("_", " ").title(), "yaw": yaw}
                for name, yaw in movement.directions
            ]
            movements.append(
                {
                    "key": movement.name,
                    "label": movement.name.title(),
                    "frames": movement.frames,
                    "loop": movement.loop,
                    "duration_ms": movement.duration_ms,
                    "directions": directions,
                }
            )
            for direction, yaw in movement.directions:
                runs.append(
                    {
                        "movement": movement.name,
                        "direction": direction,
                        "yaw": yaw,
                        "start": index,
                        "end": index + movement.frames - 1,
                    }
                )
                index += movement.frames
        out: dict[str, Any] = {
            "version": 2 if _expresses_as_v2(self.movements, self.fps) else 3,
            "columns": self.columns,
            "movements": movements,
            "runs": runs,
            "cell_count": self.cell_count,
        }
        if self.fps is not None:
            out["fps"] = self.fps
        return out


def resolve_layout(
    payload: Mapping[str, Any] | None = None,
    *,
    timing: Mapping[str, ClipTiming] | None = None,
) -> LayoutSpec:
    """Validate a v2 or v3 request/snapshot; absence is the immutable legacy
    layout (``timing`` is ignored in that case -- there is no request to
    resolve it against).

    Each movement's timing follows one precedence, in order: *(a)* ``timing``
    given -- the name must be one of its keys, or the movement is refused by
    name; *(b)* a v3 movement carrying its own ``loop`` and ``duration_ms``;
    *(c)* a legacy :data:`ANIMATIONS` name -- always this path for a v2
    payload; *(d)* otherwise refused. A movement named after one of Troupe's
    sixteen facings (``walk_front``, and so on) is refused outright, the same
    trap ``rigging.reject_direction_named_clip`` guards a clip name against.

    A v3 payload may also carry a top-level ``fps`` (one of
    :data:`FPS_CHOICES`): every movement's ``duration_ms`` becomes
    ``round(1000 / fps)``, and a movement that omits ``frames`` gets one
    derived from its own base timing --
    ``clamp(round(base_frames * base_ms * fps / 1000), 1, MAX_FRAMES)`` --
    rather than the legacy/timing frame count verbatim, so it keeps its real
    length at the new rate. See
    ``docs/measurements/2026-09-12-troupe-open-clip-vocabulary.md``.
    """

    if not payload:
        movements = tuple(
            MovementSpec(name, frames, loop, ms, DIRECTIONS)
            for name, frames, loop, ms in ANIMATIONS
        )
        return LayoutSpec(LAYOUT_VERSION, COLUMNS, movements)
    raw_version = payload.get("version")
    version = LAYOUT_VERSION if raw_version is None else int(raw_version)
    if version not in (2, 3):
        raise ValueError(f"Troupe layout version {version} is not supported")
    fps: int | None = None
    if version >= 3:
        raw_fps = payload.get("fps")
        if raw_fps is not None:
            fps = int(raw_fps)
            if fps not in FPS_CHOICES:
                raise ValueError(f"fps must be one of {list(FPS_CHOICES)}")
    raw_movements = payload.get("movements")
    if not isinstance(raw_movements, Sequence) or isinstance(raw_movements, (str, bytes)):
        raise ValueError("a Troupe layout needs at least one movement")
    movements: list[MovementSpec] = []
    seen: set[str] = set()
    for raw in raw_movements:
        if not isinstance(raw, Mapping):
            raise ValueError("every Troupe movement must be an object")
        name = str(raw.get("key") or raw.get("name") or "").strip()
        _reject_direction_named_movement(name)
        if name in seen:
            raise ValueError(f"two Troupe movements are named {name!r}")
        seen.add(name)
        base_frames, loop, base_ms = _movement_base_timing(
            name, raw, version, timing, fps
        )
        raw_frames = raw.get("frames")
        if fps is not None:
            duration_ms = round(1000 / fps)
            if raw_frames is None:
                frames = _clamp(
                    round(base_frames * base_ms * fps / 1000.0), 1, MAX_FRAMES
                )
            else:
                frames = int(raw_frames)
        else:
            duration_ms = base_ms
            frames = base_frames if raw_frames is None else int(raw_frames)
        minimum = movement_min_frames(name)
        if not minimum <= frames <= sheet.MAX_CLIP_FRAMES:
            raise ValueError(
                f"{name} must have {minimum}-{sheet.MAX_CLIP_FRAMES} frames"
            )
        raw_directions = raw.get("directions", raw.get("direction_preset", 8))
        if isinstance(raw_directions, Sequence) and not isinstance(
            raw_directions, (str, bytes)
        ):
            if any(not isinstance(d, Mapping) for d in raw_directions):
                raise ValueError("every Troupe direction must be an object")
            # A v2 direction object with no ``yaw`` used to hit
            # ``float(None)`` and surface a raw TypeError instead of a named
            # refusal -- the 2026-09-13 audit, finding troupe-01. Wrapped the
            # same way the sibling branch below refuses an out-of-range
            # direction preset by name.
            try:
                directions = tuple(
                    (
                        str(d.get("key") or d.get("name") or ""),
                        float(d.get("yaw")),
                    )
                    for d in raw_directions
                )
            except (TypeError, ValueError):
                raise ValueError(
                    "every Troupe direction must carry a numeric yaw"
                ) from None
            if directions not in DIRECTION_PRESETS.values():
                raise ValueError("directions must use the 1, 4, 8, or 16 direction preset")
        else:
            try:
                directions = DIRECTION_PRESETS[int(raw_directions)]
            except (KeyError, TypeError, ValueError):
                raise ValueError("directions must be 1, 4, 8, or 16") from None
        movements.append(MovementSpec(name, frames, loop, duration_ms, directions))
    if not movements:
        raise ValueError("a Troupe layout needs at least one movement")
    raw_columns = payload.get("columns")
    columns = COLUMNS if raw_columns is None else int(raw_columns)
    if columns != COLUMNS:
        raise ValueError(f"Troupe sheets use exactly {COLUMNS} columns")
    result = LayoutSpec(LAYOUT_VERSION, columns, tuple(movements), fps)
    if result.cell_count > MAX_CELLS:
        raise ValueError(f"a Troupe sheet may contain at most {MAX_CELLS} cells")
    return result


@dataclass(frozen=True, slots=True)
class TroupeCell:
    index: int
    animation: str
    direction: str
    yaw: float
    frame: int


def frames_per_direction(layout: LayoutSpec | Mapping[str, Any] | None = None) -> int:
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    return sum(m.frames for m in resolved.movements)


def frame_table(
    layout: LayoutSpec | Mapping[str, Any] | None = None,
) -> tuple[TroupeCell, ...]:
    """Every cell of a character sheet, in pack-and-play order."""
    out: list[TroupeCell] = []
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    for movement in resolved.movements:
        for direction, yaw in movement.directions:
            for frame in range(movement.frames):
                out.append(
                    TroupeCell(
                        index=len(out),
                        animation=movement.name,
                        direction=direction,
                        yaw=yaw,
                        frame=frame,
                    )
                )
    return tuple(out)


def spans(
    layout: LayoutSpec | Mapping[str, Any] | None = None,
) -> tuple[tuple[str, str, int, int, bool], ...]:
    """``(animation, direction, start, end, loop)`` per contiguous run."""
    out = []
    index = 0
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    for movement in resolved.movements:
        for direction, _yaw in movement.directions:
            out.append(
                (
                    movement.name,
                    direction,
                    index,
                    index + movement.frames - 1,
                    movement.loop,
                )
            )
            index += movement.frames
    return tuple(out)


def check_subset(
    subset: Sequence[Mapping[str, Any]] | None,
    layout: LayoutSpec | Mapping[str, Any] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Normalise a re-render request to ``((animation, direction), ...)``.

    A subset names whole *runs* -- one animation seen from one direction --
    because a run is the unit a person judges and re-authors. Half a walk cycle
    re-rendered against the other half is not a thing anybody wants and would be
    a seam by construction.

    Refused by name rather than filtered, which is ``check_frame_counts``' rule
    one function up: a request naming an animation this sheet does not carry is
    a mistake about the sheet, and silently rendering the rest of it would leave
    the user looking for a change that never happened.

    **A subset naming every run is refused too**, and that one is not
    pedantry: it is a full render taking the slower path, with a copy step and
    a pinned palette it does not need. The ordinary door is right there.
    """
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    available = {(animation, direction) for animation, direction, *_ in spans(resolved)}
    if not subset:
        raise ValueError("name at least one animation and direction to re-render")
    wanted: list[tuple[str, str]] = []
    for entry in subset:
        animation = str((entry or {}).get("animation") or "")
        direction = str((entry or {}).get("direction") or "")
        if not animation or not direction:
            raise ValueError("each run needs an animation and a direction")
        if animation not in {name for name, _d in available}:
            raise ValueError(f"{animation} is not an animation on this sheet")
        if (animation, direction) not in available:
            raise ValueError(f"this sheet has no {animation} facing {direction}")
        if (animation, direction) in wanted:
            raise ValueError(f"{animation} facing {direction} is named twice")
        wanted.append((animation, direction))
    if len(wanted) == len(available):
        raise ValueError(
            "that is every run on the sheet -- build a new sheet instead of "
            "re-rendering all of it"
        )
    return tuple(wanted)


def subset_indices(
    subset: Sequence[Mapping[str, Any]] | None,
    layout: LayoutSpec | Mapping[str, Any] | None = None,
) -> tuple[int, ...]:
    """The cell indices a subset covers, ascending.

    Expanded from ``spans`` rather than from arithmetic of its own, so this
    answer and ``sheetscope.runs``' Inker-side answer come from the two copies
    ``tests/troupe/test_troupe_geometry_agreement.py`` already owns rather than
    from a third nothing owns.
    """
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    wanted = set(check_subset(subset, resolved))
    out: list[int] = []
    for animation, direction, start, end, _loop in spans(resolved):
        if (animation, direction) in wanted:
            out.extend(range(start, end + 1))
    return tuple(sorted(out))


def check_frame_counts(
    records: Mapping[str, Sequence[Any]],
    layout: LayoutSpec | Mapping[str, Any] | None = None,
) -> None:
    """Refuse a set of expanded clips that does not fill the frame table.

    By name and with both numbers, rather than by padding or truncating: a
    seven-frame walk laid into an eight-frame table renders one cell of some
    other animation, and the user would go looking at the rig.
    """
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    for movement in resolved.movements:
        got = len(records.get(movement.name) or ())
        if got != movement.frames:
            raise ValueError(
                f"the {movement.name} clip expands to {got} frames and the table wants "
                f"{movement.frames}"
            )
    extra = sorted(set(records) - {m.name for m in resolved.movements})
    if extra:
        raise ValueError(f"not Troupe animations: {extra}")


def plan(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    frame_size: int = 128,
    elevation: float = sheet.DEFAULT_ELEVATION,
    lighting: str = "flat",
    layout: LayoutSpec | Mapping[str, Any] | None = None,
) -> sheet.Plan:
    """The whole character sheet as one ``sheet.Plan``.

    ``records`` is ``animation name -> the expanded pose records`` that
    ``sheet.interpolate_clip`` produced for it; this module never expands a
    clip itself, because that needs the pose library and this file is meant to
    stay loadable anywhere.

    Built directly rather than through ``sheet.plan``: that one lays out poses
    down and yaws across, one row per pose, which for 256 cells would be a
    32-row-by-8-column grid whose rows are *poses* -- and Troupe's rows are a
    dense run of ``(animation, direction)`` groups instead. Both go through
    ``check_atlas_size``, which is the part that must not be approximated
    twice.
    """
    if lighting not in sheet.LIGHTING:
        raise ValueError(f"lighting must be one of {list(sheet.LIGHTING)}")
    if not -89.0 <= elevation <= 89.0:
        raise ValueError("elevation must be between -89 and 89 degrees")
    # Task G: any whole number in [MIN_FRAME_SIZE, MAX_FRAME_SIZE] is a size
    # this function lays out correctly -- SIZES/FRAME_SIZES are presets, not
    # the limit -- so a caller offering a custom size (service.troupe) is
    # answered rather than refused for being off two ladders it was never on.
    if (
        frame_size not in SIZES
        and frame_size not in sheet.FRAME_SIZES
        and not MIN_FRAME_SIZE <= frame_size <= MAX_FRAME_SIZE
    ):
        raise ValueError(
            f"frame_size must be between {MIN_FRAME_SIZE} and {MAX_FRAME_SIZE}, "
            f"or one of {list(SIZES)}"
        )
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    check_frame_counts(records, resolved)

    table = frame_table(resolved)
    rows = (len(table) + resolved.columns - 1) // resolved.columns
    sheet.check_atlas_size(resolved.columns * frame_size, rows * frame_size)

    cells: list[sheet.Cell] = []
    poses: list[dict[str, Any]] = []
    for cell in table:
        record = records[cell.animation][cell.frame]
        column, row = cell.index % resolved.columns, cell.index // resolved.columns
        cells.append(
            sheet.Cell(
                index=cell.index,
                row=row,
                column=column,
                x=column * frame_size,
                y=row * frame_size,
                pose=record.get("id"),
                pose_name=cell.animation,
                yaw=cell.yaw,
                frame=cell.frame,
            )
        )
        poses.append(dict(record))
    return sheet.Plan(
        frame_size=frame_size,
        columns=resolved.columns,
        rows=rows,
        # Every direction the sheet contains, in table order -- ``Plan.yaws``
        # is the sheet's set of view directions, and here it is not the same
        # thing as "one per column" the way it is for a pose-per-row sheet.
        yaws=tuple(y for _n, y in resolved.directions),
        elevation=float(elevation),
        lighting=lighting,
        poses=tuple(poses),
        cells=tuple(cells),
    )


def animation_block(
    layout: LayoutSpec | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The sidecar's ``animation`` block: durations and per-direction tags.

    Gap 4 of the plan, closed. The block has been defined in ``sheet.sidecar``
    since the format was written and populated only by the Inker exporter, so a
    *rendered* sheet reached an engine as frame indices with no fps and no loop
    tags -- and the fps was the one thing the renderer knew and the importer
    could not guess.

    ``repeat: 1`` on the one-shots is the same spelling Inker's exporter uses
    for a play-once tag, so the two writers produce one format rather than two
    dialects of it.

    Carries a top-level ``"fps"`` only when the layout has one -- a legacy
    layout, and any v3 layout that never named a global rate, states no
    fps here for the same reason ``LayoutSpec.as_dict`` states no ``"fps"``:
    absence is not the same claim as "60", and every reader of this block
    already treats a missing key as "no opinion" for ``repeat``.
    """
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    table = frame_table(resolved)
    durations = {m.name: m.duration_ms for m in resolved.movements}
    block: dict[str, Any] = {
        "frames": [
            {"cell_index": c.index, "duration_ms": durations[c.animation]}
            for c in table
        ],
        "tags": [
            {
                "name": f"{animation}_{direction}",
                "start": start,
                "end": end,
                "loop": loop,
                "direction": "forward",
                **({} if loop else {"repeat": 1}),
            }
            for animation, direction, start, end, loop in spans(resolved)
        ],
    }
    if resolved.fps is not None:
        block["fps"] = resolved.fps
    return block


def point_in_cell(
    point: tuple[float, float] | None, frame_size: int
) -> tuple[float, float] | None:
    """A point the worker projected at ``RENDER_SIZE``, in *cell* pixels.

    The sidecar documents its pivot as pixels within a cell, and every reader
    of it assumes exactly that -- ``sheet.sidecar`` defaults the field to
    ``(cell_w / 2, cell_h)`` for the same reason. ``blender_worker.op_sheet``
    projects the ground origin in the pixels it rendered at, which on every
    *other* sheet path is the cell size and on this one deliberately is not:
    Troupe renders at :data:`RENDER_SIZE` and packs at the logical size,
    because a 256-cell atlas at 512 would be refused at the 8192 ceiling.

    So the number needs converting, and the conversion had been missing: a
    32px cell recorded a pivot near ``(256, 470)``, sixteen times outside
    itself, and an engine placing sprites from the sidecar put the character's
    feet far below the sprite. Placing without drift is the one property the
    field exists for.

    **Generalised from ``pivot_in_cell`` on 2026-09-05**, when the worker began
    projecting *sockets* as well as the ground origin: they are the same render
    at the same size and want the same conversion, and a second copy of it
    would put a composited flame sixteen cells away from the hand rather than
    the feet.

    Here rather than inline in ``_q_troupe`` because this module is the
    filesystem-free half of the character sheet -- it decides what cell 137
    depicts and never reads a file -- which is what makes the arithmetic
    testable at all.
    """
    if point is None:
        return None
    scale = float(frame_size) / float(RENDER_SIZE)
    return (float(point[0]) * scale, float(point[1]) * scale)


def pivot_in_cell(
    pivot: tuple[float, float] | None, frame_size: int
) -> tuple[float, float] | None:
    """The sheet's ground origin, in cell pixels. :func:`point_in_cell` by its
    older name, kept because "pivot" is what the sidecar field is called."""
    return point_in_cell(pivot, frame_size)
