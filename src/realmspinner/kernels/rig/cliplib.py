"""The shipped/user clip *library* -- an ordered list of key poses that plays
back as one of Troupe's walk cycles -- loaded, validated and merged.

Split out of the former ``rigging.py`` (P4 of ``dev/RESTRUCTURE.md``).
Deliberately named ``cliplib``, not ``clips``: ``realmspinner/clips.py`` is a
different module already (the shipped library joined to Troupe's frame
table), and a later restructure wave folds it into this same package
(``dev/RESTRUCTURE.md``'s own table lists ``clips`` among ``rig/``'s
eventual members) -- so this file claims the name that will not collide with
it, rather than costing that wave a second rename to get out of this one's way.

The Troupe clip library is the same argument :mod:`.poses`' shipped pose
libraries make, one level higher: a *clip* is an ordered list of those poses
plus how many frames each step holds -- so the thing that is portable across
rigs is now a whole walk cycle rather than a single silhouette, and authoring
cost moves from 256 frames per character to ~22 keyframes once ever.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .. import charsheet
from ..sheet import POSE_SPACES
from .poses import validate_bones
from .templates import TEMPLATE_DIR, _read_json_capped, catalog, get_template

log = logging.getLogger(__name__)

CLIP_DIR = TEMPLATE_DIR / "clips"

# A shipped clip library is up to ~44 KB today, but unlike a template it is
# also something a user edits and re-saves through service.clips.save, whose
# own write-door caps (MAX_LIBRARY_KEYS=1024 poses, MAX_KEYS=64 keys/clip)
# allow a file substantially larger once every pose carries a full skeleton's
# worth of bones at JSON's verbosity. 4 MiB leaves real headroom above that
# legitimate maximum while still refusing anything that is not a hand-authored
# or program-written clip library.
#
# The 2026-09-18 audit, finding docs-06: this comment still cited
# MAX_LIBRARY_KEYS=256, the value the 2026-09-12 measurement raised to 1024
# (see the ``Raised from 256 to 1024`` comment on MAX_KEYS, below) -- updated
# to match, docstring only, no cap here actually moved.
MAX_CLIP_LIBRARY_BYTES = 4 << 20

_clips: dict[str, dict[str, Any]] | None = None
#: The same shape, read from ``user_clip_dir()``. A template present here
#: *replaces* the shipped entry rather than merging with it: a clip library
#: is internally consistent by construction (every clip names poses the same
#: file carries), and half of one file's poses under another file's clips is
#: exactly the hole ``parse_clip_library`` refuses within a single file.
_user_clips: dict[str, dict[str, Any]] | None = None

#: ``template key -> str(exc)`` for a user clip library file that exists but
#: failed to parse, filled by the same read that builds ``_user_clips``. A
#: parse failure there still costs that template only -- ``_load_clip_library``
#: logs and skips it exactly as it always has, and the merged, rendering-facing
#: ``clip_library()`` still falls back to the shipped library for it. But
#: without this, ``service.clips.library()`` had no way to tell "no edits"
#: from "edits this build can no longer read" and presented the fallback as
#: the user's own, ``edited: True`` -- so the next Poser Save would silently
#: overwrite the very file that failed to parse. :func:`user_clip_error` is
#: the honest answer that door needs; nothing else reads this dict.
_user_clip_errors: dict[str, str] | None = None


# Mirrors service.clips.MAX_LIBRARY_KEYS/MAX_KEYS -- this module may import no
# more of ``realmspinner`` than the rest of ``kernels/rig`` plus ``poselib`` (this
# file's own docstring, function-scoped below for the reason stated there);
# ``tests/modes/poser/test_poser_imports.py`` pins the package's own outward reach, so the
# write door's caps cannot be imported here and are restated as their own
# constants instead. The two must be kept in sync by hand: the 2026-09-11
# audit (poser-04) found this parser applied neither, so a hand-edited library
# under user_clip_dir() (writable by any program, exactly like a pose file)
# with no count ceiling at all parsed in full where service.clips.save would
# have refused it at the write door.
#
# Raised from 256 to 1024 alongside service.clips.MAX_LIBRARY_KEYS when the
# clip-library schema moved to v3 (design decision D2): a library that can
# hold any number of named clips, not just the shipped five, wants more room
# to author key poses in without a hand-authored or agent-generated set
# hitting a ceiling picked for the original five-animation table.
# MAX_CLIP_LIBRARY_BYTES was checked against this: the humanoid library
# averages ~1.5 KB/pose at 19 bones and the largest authored skeleton (bird)
# carries 20, so 1024 poses lands around 1.6 MB -- comfortably inside the 4 MiB
# ceiling, which is left alone.
MAX_CLIP_LIBRARY_POSES = 1024
MAX_CLIP_KEYS = 64

#: A step's frame count, mirrored from ``service.clips.MIN_SEGMENT``/
#: ``MAX_SEGMENT`` -- the write door's own numbers -- the same way
#: :data:`MAX_CLIP_KEYS` above already tracks ``service.clips.MAX_KEYS``.
#: The 2026-09-23 audit, finding poser-05: this read door checked neither a
#: clip's segment *count* against its keys nor each segment's range, so a
#: hand-edited or externally produced library with a ``segments`` list too
#: short, too long, zero-length or absurdly large parsed clean and reached
#: ``sheet.interpolate_clip`` only for whichever movement a layout happened
#: to name -- everything else sat parsed-but-wrong until something rendered
#: it.
MIN_CLIP_SEGMENT = 1
MAX_CLIP_SEGMENT = 64

#: The versions :func:`parse_clip_library` understands. 2 is the shipped shape
#: (no ``duration_ms``; timing lived in ``kernels.charsheet.ANIMATIONS``
#: instead) and 3 adds it per clip -- see :data:`LEGACY_CLIP_DURATION_MS``.
#: A file naming any other version is refused the same way a malformed one
#: always was: ``ValueError`` out of the parser, caught and logged by
#: ``_load_clip_library`` for the shipped/user read paths and turned into an
#: ``Invalid`` by ``service.clips.save`` for the write door.
CLIP_LIBRARY_VERSIONS = (2, 3)

#: Per-rendered-frame duration, in whole milliseconds. The floor is a guard on
#: a typo (a clip cannot render faster than one frame a tick), the ceiling on
#: a clip so slow it is really a stall, and the step keeps every value evenly
#: divisible into ``realmspinner.clips.ANIMATION_FPS``'s 10 ms scene tick -- this
#: module cannot import ``realmspinner.clips`` (see the module docstring above on
#: the name collision, and the import pin), so that relationship is only
#: asserted, not imported;
#: ``tests/test_clip_library_v3.py::test_the_clip_duration_step_divides_the_animation_timebase``
#: pins ``1000 / clips.ANIMATION_FPS`` (10) against this constant so the two
#: cannot drift apart silently.
#:
#: The numbers themselves are ``charsheet``'s movement bounds: a v3 movement's
#: ``duration_ms`` and a clip's are the same quantity, and until the
#: restructure put both modules in ``kernels/`` each restated the other's
#: under a layer ban that no longer exists.
MIN_CLIP_DURATION_MS = charsheet.MIN_MOVEMENT_DURATION_MS
MAX_CLIP_DURATION_MS = charsheet.MAX_MOVEMENT_DURATION_MS
CLIP_DURATION_STEP_MS = charsheet.MOVEMENT_DURATION_STEP_MS

#: A v2 clip carries no ``duration_ms`` of its own, so migrating one to v3
#: needs somewhere to read the time it was actually rendered at: the time
#: ``charsheet.ANIMATIONS`` rendered it at. Derived rather than restated. It
#: was a hand copy while ``charsheet`` lived in ``pipelines/`` (Layer 2, out
#: of this kernel's reach), pinned back to the original by a test; the
#: restructure moved ``charsheet`` beside this package, and a copy that can
#: be an import is a copy that can drift for nothing.
LEGACY_CLIP_DURATION_MS: dict[str, int] = {
    name: duration_ms for name, _frames, _loop, duration_ms in charsheet.ANIMATIONS
}

#: A clip named for one of Troupe's 16 facing directions -- or, worse, ending
#: in ``_<direction>`` -- collides with the tag Inker's importer already reads
#: off a filename: ``fall_back.png`` is read as clip ``fall`` facing ``back``.
#: Once any clip name can exist (not just the shipped five), authoring one
#: that *looks* like ``<clip>_<direction>`` is a trap the parser refuses
#: instead of leaving for Inker to silently mis-tag. The names are Troupe's
#: sixteen-direction preset, derived for the reason ``LEGACY_CLIP_DURATION_MS``
#: above is.
TROUPE_DIRECTION_KEYS: tuple[str, ...] = tuple(
    name for name, _yaw in charsheet.DIRECTION_PRESETS[16]
)

#: A clip's optional ``source`` is a small provenance note (e.g. ``{"file":
#: ..., "map": ..., "imported": ...}``) an importer or an agent can leave on a
#: clip it wrote, not free-form data -- so it is capped and type-checked like
#: every other field a clip carries, not merely round-tripped as opaque JSON.
MAX_CLIP_SOURCE_KEYS = 16
MAX_CLIP_SOURCE_BYTES = 2048


def validate_clip_duration_ms(value: Any, label: str) -> int:
    """One clip's ``duration_ms``, or raise. Shared by :func:`parse_clip_library`
    (the renderer's own check) and ``service.clips._check_shape`` (the write
    door's field-addressed one), so the two cannot quietly disagree about what
    a valid duration is -- the same argument ``validate_bones`` already makes
    for a pose's quaternions.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'clip {label!r} "duration_ms" must be a whole number of milliseconds')
    if not (MIN_CLIP_DURATION_MS <= value <= MAX_CLIP_DURATION_MS):
        raise ValueError(
            f'clip {label!r} "duration_ms" must be {MIN_CLIP_DURATION_MS}-'
            f"{MAX_CLIP_DURATION_MS}, not {value}"
        )
    if value % CLIP_DURATION_STEP_MS != 0:
        raise ValueError(
            f'clip {label!r} "duration_ms" must be a multiple of '
            f"{CLIP_DURATION_STEP_MS}, not {value}"
        )
    return value


def reject_direction_named_clip(name: str) -> None:
    """Raise if *name* ends in ``_<direction>`` for one of Troupe's 16 facings.

    See :data:`TROUPE_DIRECTION_KEYS`. Suffix only -- a clip named exactly
    ``back`` is not read as anything facing anywhere, because there is no clip
    name before it for the tag parser to split off.
    """
    for direction in TROUPE_DIRECTION_KEYS:
        if name.endswith(f"_{direction}"):
            raise ValueError(
                f'clip {name!r} ends in "_{direction}"; Troupe reads a name '
                f'like that as clip {name[: -len(direction) - 1]!r} facing {direction!r}'
            )


def validate_clip_source(value: Any, label: str) -> dict[str, Any]:
    """A clip's optional ``source`` object, or raise. See its cap comment above."""
    if not isinstance(value, dict):
        raise ValueError(f'clip {label!r} "source" must be an object')
    if len(value) > MAX_CLIP_SOURCE_KEYS:
        raise ValueError(
            f'clip {label!r} "source" holds at most {MAX_CLIP_SOURCE_KEYS} fields'
        )
    out: dict[str, Any] = {}
    for k, v in value.items():
        if not isinstance(k, str):
            raise ValueError(f'clip {label!r} "source" keys must be strings')
        if isinstance(v, bool) or not isinstance(v, (str, int, float)):
            raise ValueError(f'clip {label!r} "source.{k}" must be a string or a number')
        out[k] = v
    if len(json.dumps(out)) > MAX_CLIP_SOURCE_BYTES:
        raise ValueError(f'clip {label!r} "source" is too large')
    return out


def parse_clip_library(raw: dict[str, Any]) -> dict[str, Any]:
    """One clip library file's contents, validated. Raises on a bad one.

    Split out of :func:`_load_clip_library` so the shipped library and a
    user-authored one go through exactly one parser: the editor writes files
    this reads back, and a second, laxer parse on the authoring side is how an
    editor comes to save something the renderer cannot open.

    v3 added per-clip ``duration_ms``: a v2 file (or one naming no version at
    all -- every file shipped before this parser existed) is migrated at read
    time from :data:`LEGACY_CLIP_DURATION_MS`, defaulting to 100 ms for a name
    that table does not carry. A v3 file states its own timing and a v3 clip
    that omits ``duration_ms`` is refused outright: unlike v2, there is no
    table left to fall back to once any clip name can exist.

    **The direction-suffix guard (** :func:`reject_direction_named_clip` **)
    applies to v3 files only.** It exists because once any clip name can
    exist, one that looks like ``<clip>_<direction>`` is a trap for Inker's
    tag parser -- but a v2 file predates that guard by definition, and a
    2026-09-13 fix found this parser applying it to v2 reads too: an existing,
    previously-legal user library naming a clip like ``turn_left`` was refused
    *whole* on read, logged and silently skipped, with rendering falling back
    to the shipped clips and the editor showing them marked as the user's own
    edited copy. A v2 file must read exactly as it always did; the save door
    (``service.clips._check_shape``, which always writes v3) is the one place
    such a name is still refused, because every save is a v3 write.
    """
    raw_version = raw.get("version")
    version = 2 if raw_version is None else int(raw_version)
    if version not in CLIP_LIBRARY_VERSIONS:
        raise ValueError(f"unknown clip library version {version}")
    raw_poses = raw["poses"]
    if len(raw_poses) > MAX_CLIP_LIBRARY_POSES:
        raise ValueError(
            f"a clip library holds at most {MAX_CLIP_LIBRARY_POSES} key poses, "
            f"not {len(raw_poses)}"
        )
    poses = {}
    pose_names = [str(pose["name"]) for pose in raw["poses"]]
    if len(set(pose_names)) != len(pose_names):
        raise ValueError("duplicate pose names")
    for pose in raw["poses"]:
        # **Through ``validate_bones``, like every other door a pose comes in
        # by.** A clip library is a file -- shipped, or authored in the editor
        # and written back -- and its quaternions were taken verbatim: a
        # 3-element list raised out of the middle of ``sheet._blend`` with no
        # mention of the file or the bone, and a NaN was interpolated into a
        # clip and written to disk. The bone names are *not* checked against a
        # rig here: a library is authored against a template and applied to
        # whatever is loaded, and the worker already reports an unknown bone
        # without failing the bake.
        row = {
            "name": str(pose["name"]),
            "bones": validate_bones(pose["bones"]),
        }
        if pose.get("root_translation"):
            # Function-level: ``poselib`` imports this package at its own top,
            # so a top-level import here would be circular. Shared rather than
            # a bare ``float(v)`` -- the 2026-09-07 audit (poser-05) found this
            # door (and ``service.clips._check_shape``) round-tripped
            # ``[nan, 1e30, 0.0]`` with no finite or magnitude check, unlike
            # the identical field on a library pose.
            from ... import poselib

            row["root_translation"] = poselib.validate_root_translation(
                pose["root_translation"]
            )
        poses[row["name"]] = row
    # The frame every clip in this file is authored in. Per file and not per
    # clip: a library mixing the two would be one edit away from a clip that
    # silently means the other thing.
    space = str(raw.get("space") or "node")
    # The 2026-09-20 audit's poser-02: read with no membership test, unlike
    # ``sheet.py``'s own doors for the identical field. An unrecognised value
    # reached ``blender_worker._apply_pose``, which treats anything but the
    # exact string "delta" as "node" -- so a misspelled space applied every
    # clip in this file in the wrong rotation frame, silently. Raised here
    # like every other malformed-file case this parser already refuses on.
    if space not in POSE_SPACES:
        raise ValueError(f"space must be one of {list(POSE_SPACES)}, not {space!r}")
    clips = []
    clip_names = [str(clip["name"]) for clip in raw["clips"]]
    if len(set(clip_names)) != len(clip_names):
        raise ValueError("duplicate clip names")
    for clip in raw["clips"]:
        name = str(clip["name"])
        if version >= 3:
            # v2-only libraries predate this guard (see the docstring above):
            # a name like ``turn_left`` was legal before any file could name a
            # clip beyond the shipped five, and a v2 file must keep reading
            # exactly as it always did. Every *save* still refuses it, because
            # ``service.clips.save`` always writes v3.
            reject_direction_named_clip(name)
        keys = [str(k) for k in clip["keys"]]
        if len(keys) > MAX_CLIP_KEYS:
            raise ValueError(
                f"clip {clip['name']!r} holds at most {MAX_CLIP_KEYS} keys, not {len(keys)}"
            )
        missing = [k for k in keys if k not in poses]
        if missing:
            raise ValueError(f"clip {clip['name']!r} names {missing}")
        # The 2026-09-23 audit, finding poser-05: ``service.clips._check_shape``
        # (the write door) has always refused a segments list whose length
        # does not match its keys and a segment outside 1-64 frames; this
        # read door -- the one both the shipped files and a hand-edited or
        # externally produced one pass through -- checked neither, so a file
        # that never went through the editor's save could carry a mismatched
        # or out-of-range ``segments`` and parse clean. Gated on
        # ``len(keys) >= 2``, like the write door's own ``MIN_KEYS`` gate
        # (this parser does not repeat ``MIN_KEYS`` itself -- a sub-2-key
        # clip is a legal, if useless, filler shape several read-door tests
        # already rely on, and ``sheet.interpolate_clip`` refuses one by name
        # ("a clip needs at least two keyframes") the moment it is actually
        # expanded, so nothing reaches the renderer unchecked).
        closed = bool(clip.get("closed", False))
        raw_segments = clip["segments"]
        if len(keys) >= 2:
            wanted = len(keys) if closed else len(keys) - 1
            if len(raw_segments) != wanted:
                raise ValueError(
                    f'clip {name!r} is {"closed" if closed else "open"} with {len(keys)} keys, '
                    f"so it needs {wanted} segment lengths, not {len(raw_segments)}"
                )
        segments = [int(n) for n in raw_segments]
        for n in segments:
            if not MIN_CLIP_SEGMENT <= n <= MAX_CLIP_SEGMENT:
                raise ValueError(
                    f"clip {name!r} has a segment of {n} frames; each is "
                    f"{MIN_CLIP_SEGMENT}-{MAX_CLIP_SEGMENT}"
                )
        if version >= 3:
            if "duration_ms" not in clip:
                raise ValueError(f'clip {name!r} needs "duration_ms" in a version 3 library')
            duration_ms = clip["duration_ms"]
        else:
            duration_ms = LEGACY_CLIP_DURATION_MS.get(name, 100)
        duration_ms = validate_clip_duration_ms(duration_ms, name)
        record: dict[str, Any] = {
            "name": name,
            "keys": keys,
            "segments": segments,
            "closed": closed,
            "easing": str(clip.get("easing") or "linear"),
            "space": space,
            "duration_ms": duration_ms,
        }
        if "provisional" in clip:
            provisional = clip["provisional"]
            if not isinstance(provisional, bool):
                raise ValueError(f'clip {name!r} "provisional" must be true or false')
            record["provisional"] = provisional
        if clip.get("source") is not None:
            record["source"] = validate_clip_source(clip["source"], name)
        clips.append(record)
    return {"poses": poses, "clips": clips, "space": space}


def _load_clip_library(
    directory: Path, *, errors: dict[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    """``template key -> {"poses": {name: record}, "clips": [record]}``.

    A malformed library costs you that library and not the app, the rule
    ``templates._load_templates`` and ``poses._load_pose_library`` both
    follow -- and a clip naming a pose the file does not carry is exactly
    that: the file is internally inconsistent, and half-loading it would hand
    the renderer a clip with a hole in it.

    ``errors``, when given, is filled with ``path.stem -> str(exc)`` for every
    file this drops -- :func:`clip_library` passes its own ``_user_clip_errors``
    here so :func:`user_clip_error` can answer honestly without a second read
    of the same directory.
    """
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            found[path.stem] = parse_clip_library(
                _read_json_capped(path, MAX_CLIP_LIBRARY_BYTES)
            )
        except Exception as exc:
            log.exception("skipping unusable clip library %s", path)
            if errors is not None:
                errors[path.stem] = str(exc)
    return found


#: Where an *edited* clip library lives, or None when nothing has said.
#:
#: **Told, never discovered.** This module is one the host and the worker
#: share and it has deliberately never depended on the app's configuration --
#: ``tests/modes/poser/test_poser_imports.py`` pins ``kernels/rig``'s whole outward
#: ``realmspinner`` import set, and reaching for ``config`` here (even inside a
#: function body) would be a real architectural change dressed up as a
#: convenience. So the layer that *has* a config sets this:
#: ``service.core.RealmspinnerService`` does it on construction, which is the one
#: object every app process builds before anything asks for a clip.
#:
#: None means "the shipped library and nothing else", which is the correct
#: answer for a bare import, a test that never built a service, and the worker.
_user_clip_root: Path | None = None


def set_user_clip_dir(path: Any) -> None:
    """Point the loader at the user's editable clip libraries.

    Drops the caches, because the directory changing means every answer this
    module has already given about clips may now be the wrong one -- which is
    exactly what happens in a test suite that builds one service per test.
    """
    global _user_clip_root
    new = None if path is None else Path(path)
    if new != _user_clip_root:
        _user_clip_root = new
        invalidate_clips()


def user_clip_dir() -> Path | None:
    """Where an *edited* clip library lives, or None with no configured home.

    Beside the pose library under ``data_dir/poser/``, because it is the same
    kind of thing: authored content that belongs to the user rather than to the
    build. The shipped library under ``templates/clips/`` stays a factory
    default and is never written to -- which is what keeps an installed build
    (where the package tree may be read-only, and is replaced wholesale by the
    next installer) working the same way a checkout does.
    """
    return _user_clip_root


def invalidate_clips() -> None:
    """Drop the cached libraries so the next read sees a just-saved edit.

    The cache is a module global filled once, which is right for a library that
    ships with the build and wrong the moment one can be authored. Called by the
    save door rather than by a timestamp check: the writer knows.
    """
    global _clips, _user_clips, _user_clip_errors
    _clips = None
    _user_clips = None
    _user_clip_errors = None


def clip_library(template_key: str) -> dict[str, Any]:
    """The shipped clips for a template, or an empty library if it has none.

    Empty is a normal answer for the same reason ``deform_battery``'s is: a
    walk cycle authored for a humanoid means nothing to a fish, and a template
    without one should cost the character sheet, never the rig.
    """
    global _clips, _user_clips, _user_clip_errors
    get_template(template_key)
    if _clips is None:
        _clips = _load_clip_library(CLIP_DIR)
    if _user_clips is None:
        directory = user_clip_dir()
        _user_clip_errors = {}
        _user_clips = (
            _load_clip_library(directory, errors=_user_clip_errors)
            if directory is not None and directory.is_dir()
            else {}
        )
    # The user's file wins whole, never field by field -- see ``_user_clips``.
    # An unusable one has already been logged and dropped by the loader, so
    # this falls back to the shipped library rather than to nothing, which is
    # the same "costs you that library and not the app" rule one level up.
    # ``service.clips.library()`` is what tells the two cases apart for the
    # user, through :func:`user_clip_error` -- this function stays the
    # renderer's own "give me something to draw" door and keeps falling back.
    library = _user_clips.get(template_key) or _clips.get(template_key)
    return library or {"poses": {}, "clips": [], "space": "node"}


def user_clip_error(template_key: str) -> str | None:
    """The parse error for *template_key*'s user clip library file, or None.

    None both when the user has no file for this template and when the one
    they have parsed cleanly -- callers that need to tell those two apart
    already have ``poselib.clip_path(...).is_file()`` for the first half, the
    same test :func:`~realmspinner.service.clips.library`'s own ``edited`` flag
    uses. This is the answer for the other half: a file that exists but this
    build can no longer read, which used to make ``clip_library`` fall back to
    the shipped clips *silently* -- so the editor's ``edited`` flag lied,
    showing the shipped library marked as the user's own, and the next Poser
    Save would overwrite the very file that failed to parse.

    Populated by the same read :func:`clip_library` already does, so this
    never re-reads the directory and can never disagree with what that
    function actually served.
    """
    clip_library(template_key)  # ensures _user_clip_errors is populated
    return (_user_clip_errors or {}).get(template_key)


def shipped_clip_library(template_key: str) -> dict[str, Any]:
    """The template's *shipped* clips only -- never a user edit.

    :func:`clip_library` is the renderer's own door, and the user's file wins
    whole there by design (see ``_user_clips``' own comment): a hand-edited
    library is meant to change what a sheet renders the moment it is saved.
    An agent's movement vocabulary needs the opposite promise -- the catalogue
    it was told about at connect time must still be the catalogue a call
    against it means, for as long as the connection lasts -- so this reads
    straight out of the shipped ``_clips`` cache and never consults
    ``user_clip_dir()`` at all. Same empty-library fallback as
    :func:`clip_library`, and the same ``ValueError`` for an unknown template,
    both through :func:`get_template`.
    """
    global _clips
    get_template(template_key)
    # Bound to a local once and read only through it below: a concurrent
    # ``invalidate_clips()`` (Poser's save door, on another thread) between
    # the None check and the read used to be able to set the module global
    # back to None in the gap, and the second read raised
    # ``AttributeError: 'NoneType' object has no attribute 'get'`` instead of
    # simply serving the library this call had already committed to loading.
    clips = _clips
    if clips is None:
        clips = _load_clip_library(CLIP_DIR)
        _clips = clips
    library = clips.get(template_key)
    return library or {"poses": {}, "clips": [], "space": "node"}


def shipped_clip_templates() -> tuple[str, ...]:
    """Every template with at least one shipped clip, in ``templates.catalog``
    order."""
    return tuple(
        row["key"] for row in catalog() if shipped_clip_library(row["key"])["clips"]
    )


def shipped_clip_names(template_key: str) -> tuple[str, ...]:
    """A template's shipped clip names, library order."""
    return tuple(str(c["name"]) for c in shipped_clip_library(template_key)["clips"])


def clip_keys(template_key: str, clip_name: str) -> list[dict[str, Any]]:
    """One clip's key poses, in order. Raises KeyError for an unknown clip."""
    library = clip_library(template_key)
    for clip in library["clips"]:
        if clip["name"] == clip_name:
            return [dict(library["poses"][k]) for k in clip["keys"]]
    raise KeyError(f"{clip_name!r} is not a clip of the {template_key} template")
