"""The shipped clip library joined to Troupe's frame table.

Two functions -- one for the sheet's grid, one for an exported animation
track -- in a module of its own, for the reason ``vectors.py`` is a module
of its own: **the worker may not import ``service``**, and the door may not
reimplement the worker. Both need to turn "the humanoid template's clips" into
"the expanded pose records the resolved frame table wants", and if either one
owned it the other would have to grow a second copy that could disagree about
what a walk is. This module is that one expansion, importable by both.

It is not folded into ``kernels.charsheet`` because that module is
deliberately filesystem-free -- it decides what cell 137 depicts and never
reads a file to do it -- while this one reads the shipped/user clip library
off disk (``kernels.rig.store``) before it can expand anything. It is not
folded into ``kernels.rig`` either, even though both ``kernels.charsheet``
and ``kernels.sheet`` are kernels this module could now import directly
(the 2026-09-17 restructure moved both out of ``pipelines/``, closing what
used to be a real layering objection): ``kernels.rig`` is the rig's own
geometry and storage, one layer down from "what a walk looks like when
rendered", and moving this module's two functions into it would mean the
rig kernel importing the sheet/charsheet kernels for a job that is really
about Troupe's frame table, not the rig. Staying a separate module -- the
same shape ``vectors.py`` already has beside ``queue.py`` -- keeps that
layer boundary a fact about the directory, not a convention two callers
have to remember.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .kernels import charsheet, sheet
from .kernels.rig import blender_spec, cliplib, store, templates

log = logging.getLogger(__name__)


def expand_clips(
    template_key: str,
    layout: charsheet.LayoutSpec | Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """``animation name -> expanded pose records``, for every Troupe animation.

    A missing clip raises ``KeyError``, which is the honest answer: the library
    ships with the package, so a Troupe animation with no clip is a broken
    build and not a user mistake. A library that expands to the wrong number of
    frames raises ``ValueError`` out of ``check_frame_counts`` -- named here
    rather than left to the renderer, because a seven-frame walk laid into an
    eight-frame table renders one cell of some other animation and sends the
    user to look at the rig.

    ``layout`` is resolved with this rig's own :func:`clip_timing` as
    ``resolve_layout``'s ``timing`` -- the open-vocabulary door, precedence
    *(a)* -- so a movement naming any clip *template_key*'s library defines
    resolves, not only one of the five legacy names.
    """
    resolved = (
        layout
        if isinstance(layout, charsheet.LayoutSpec)
        else charsheet.resolve_layout(layout, timing=clip_timing(template_key))
    )
    library = cliplib.clip_library(template_key)
    by_name = {c["name"]: c for c in library["clips"]}
    records: dict[str, list[dict[str, Any]]] = {}
    for movement in resolved.movements:
        animation = movement.name
        clip = by_name.get(animation)
        if clip is None:
            raise KeyError(animation)
        keys = cliplib.clip_keys(template_key, animation)
        records[animation] = sheet.resample_clip(
            keys,
            clip["segments"],
            movement.frames,
            closed=clip["closed"],
            easing=clip["easing"],
            space=clip["space"],
            # The animation's name *is* the row identity here. Left to
            # ``_expand``'s default it was derived from the keys' ``id``
            # fields, which a clip library's key poses do not have -- so every
            # four-key clip shared one id and ``_charsheet``'s ``(id, frame)``
            # lookup let run overwrite walk. Names are unique by construction:
            # ``by_name`` above is keyed on them.
            clip_id=animation,
        )
    charsheet.check_frame_counts(records, resolved)
    return records


#: The timebase every baked animation track is written on, in frames per
#: second. 100 is not arbitrary: ``charsheet.ANIMATIONS`` states each movement's
#: frame duration in whole milliseconds (150, 100, 60, 80, 100), so a 10 ms
#: frame divides every one of them exactly -- each clip's keyframes land on
#: integer scene frames and no clip's tempo is rounded. glTF stores sample
#: times in seconds against the scene's own rate, which is why the base has to
#: be one number for the whole file rather than one per track.
ANIMATION_FPS = 100


def _timing_of(library: Mapping[str, Any]) -> dict[str, charsheet.ClipTiming]:
    """``clip_timing``'s and :func:`shipped_clip_timing`'s shared arithmetic.

    Frame count is not stored on the clip -- it is what the clip's own
    ``segments`` (plus one more sample for a one-shot, which lands on its last
    key rather than looping back to its first) expand to.
    """
    out: dict[str, charsheet.ClipTiming] = {}
    for clip in library["clips"]:
        closed = bool(clip["closed"])
        frames = sum(int(n) for n in clip["segments"]) + (0 if closed else 1)
        out[str(clip["name"])] = charsheet.ClipTiming(
            frames=frames, loop=closed, duration_ms=int(clip["duration_ms"])
        )
    return out


def clip_timing(template_key: str) -> dict[str, charsheet.ClipTiming]:
    """``clip name -> charsheet.ClipTiming``, from a rig's own clip library.

    ``resolve_layout``'s ``timing`` door: the service layer builds this once
    per rig and passes it down, so a layout can name any clip the library
    defines instead of one of :data:`charsheet.ANIMATIONS`' five. User-first,
    like :func:`~realmspinner.cliplib.clip_library` itself -- see
    :func:`shipped_clip_timing` for the agent-facing library that never moves
    under a hand edit.
    """
    return _timing_of(cliplib.clip_library(template_key))


def shipped_clip_timing(template_key: str) -> dict[str, charsheet.ClipTiming]:
    """:func:`clip_timing`'s shape, off the *shipped* library only.

    ``cliplib.shipped_clip_library``'s reason applied to timing: the agent
    catalogue's movement/frame-bound vocabulary must not move the moment a
    user edits their own copy of a template's clips.
    """
    return _timing_of(cliplib.shipped_clip_library(template_key))


def loop_names(template_key: str) -> tuple[str, ...]:
    """The names of a template's closed (looping) clips, library order."""
    library = cliplib.clip_library(template_key)
    return tuple(str(c["name"]) for c in library["clips"] if c["closed"])


def library_digest(template_key: str) -> str:
    """A hash of everything that decides how a template's clips play.

    Changes whenever a clip's keys, segments, easing, space, ``closed`` or
    ``duration_ms`` change -- or a pose's rotations do, or ``ANIMATION_FPS``
    itself does -- and not otherwise. Meant for a later stage to compare
    against a value it recorded on an ``animated.glb``'s sidecar at bake time,
    so a stale bake (the library edited since) is detectable without
    re-baking to find out.
    """
    library = cliplib.clip_library(template_key)
    canonical = json.dumps(
        {"library": library, "animation_fps": ANIMATION_FPS},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def animation_tracks(template_key: str) -> list[dict[str, Any]]:
    """Every authored clip of a template, resolved to frames. -> track list.

    The host half of the animated-GLB bake, and deliberately *not*
    :func:`expand_clips`: that one resamples each clip to the frame count the
    sheet's layout asks for, because a sprite sheet has a grid to fill. An
    exported animation has no grid, so this uses the clip's **own** segment
    lengths -- the animation carries the author's timing rather than Troupe's.

    Timing comes from the clip library's own ``closed``/``duration_ms``
    fields now -- their one home, since the vocabulary opened past
    ``charsheet.ANIMATIONS``' five names (see
    ``dev/measurements/2026-09-12-troupe-open-clip-vocabulary.md``). A
    second copy would be one edit from disagreeing about how fast a walk
    cycle is.

    Blender does no interpolation: it receives resolved frames, which is the
    same host/worker split ``fit_template`` establishes and what keeps the
    interpolation under test with no ``bpy``.
    """
    library = cliplib.clip_library(template_key)
    tracks: list[dict[str, Any]] = []
    for clip in library["clips"]:
        name = str(clip["name"])
        frames = sheet.interpolate_clip(
            cliplib.clip_keys(template_key, name),
            clip["segments"],
            closed=bool(clip["closed"]),
            easing=str(clip["easing"]),
            space=str(clip["space"]),
            # The clip's name is its identity here, ``expand_clips``' rule and
            # for its reason: a clip library's key poses carry no ``id``.
            clip_id=name,
        )
        loop = bool(clip["closed"])
        duration_ms = int(clip["duration_ms"])
        tracks.append(
            {
                "name": name,
                "loop": bool(loop),
                "space": str(clip["space"]),
                # Whole scene frames per animation frame. See ANIMATION_FPS.
                "step": ANIMATION_FPS * int(duration_ms) / 1000.0,
                # ``root_translation`` forwarded raw (character-height units,
                # ``sheet._record``'s own unit) when the frame carries one --
                # the 2026-09-08 audit (poser-01) found this dict built as
                # ``{"bones": record["bones"]}`` alone, so a clip's authored
                # vertical motion (the whole of a jump's crouch/launch/apex/
                # land arc) was computed by ``sheet.interpolate_clip`` and then
                # thrown away before it ever reached a spec. Scaled into a
                # world-space ``root_offset`` by :func:`animate_spec`, which is
                # the one caller with a job directory to read a rig's bounds
                # from -- this function stays host-pure and job-independent.
                "frames": [
                    {
                        "bones": record["bones"],
                        **(
                            {"root_translation": record["root_translation"]}
                            if record.get("root_translation")
                            else {}
                        ),
                    }
                    for record in frames
                ],
            }
        )
    return tracks


def _attach_root_offsets(tracks: list[dict[str, Any]], job_dir: Path) -> None:
    """Scale every frame's raw ``root_translation`` into a world-space
    ``root_offset``, in place -- ``op_sheet``'s per-cell ``root_offset``,
    mirrored for a track's per-frame one.

    ``clips.py`` may not import ``queue.py`` (``queue`` imports ``_q_troupe``,
    which imports ``clips`` -- a cycle), so this is this module's own copy of
    ``queue._sheet_root_offsets``'s arithmetic against
    ``blender_spec.root_offset_world``, and tolerant the same way
    ``service.rig._pose_bake_spec`` is: a rig.json this job directory does not
    have yet, or one built before bounds/root were recorded, costs every
    frame's offset rather than the bake. The 2026-09-08 audit (poser-01)'s own
    rule -- "costs the measurement, never the rig" -- applied to a root offset
    instead of a joint measurement.
    """
    carries_root = any(
        frame.get("root_translation") for track in tracks for frame in track["frames"]
    )
    if not carries_root:
        return
    rig_meta = store.read_rig(job_dir) or {}
    bounds, root_bone = rig_meta.get("bounds"), rig_meta.get("root")
    if not (isinstance(bounds, dict) and "min" in bounds and "max" in bounds and root_bone):
        log.warning("a clip carries a root offset but %s cannot scale it", job_dir / "rig.json")
        return
    for track in tracks:
        for frame in track["frames"]:
            root_translation = frame.pop("root_translation", None)
            if not root_translation:
                continue
            try:
                offset = blender_spec.root_offset_world(root_translation, bounds)
            except (TypeError, ValueError, IndexError):
                log.warning("a clip has a root offset rig.json cannot scale")
                continue
            frame["root_offset"] = offset
            frame["root_bone"] = str(root_bone)


def animate_spec(
    job_dir: Path, template_key: str, out_glb: Path, result_dir: Path
) -> dict[str, Any]:
    """The worker spec for baking every authored clip into one animated GLB.

    ``blender_spec.pose_spec``'s shape one step up: a pose is one set of bone
    rotations, and this is a named sequence of them per clip, resolved **here**
    rather than in Blender. :func:`animation_tracks` does the interpolation on the host, so
    the timing stays under test with no ``bpy`` and the worker only does the
    thing only Blender can do -- keying an armature and writing glTF animation
    samplers.

    Here rather than beside ``pose_spec`` in ``kernels.rig.blender_spec`` for
    that module's own pinned reason (``tests/test_poser_imports``): it may not
    import ``pipelines``, and resolving frames needs
    ``sheet.interpolate_clip``. Which is the argument this module was created
    on.

    Raises ``ValueError`` for a template with no clips, before a subprocess is
    spent: an animated GLB with no animations in it is a file that answers the
    question wrongly rather than not at all.
    """
    templates.get_template(template_key)  # fail here, not three seconds into a subprocess
    tracks = animation_tracks(template_key)
    if not tracks:
        raise ValueError(f"nothing is authored for the {template_key} rig")
    _attach_root_offsets(tracks, job_dir)
    return {
        "op": "animate",
        "rig_glb": str(job_dir / "rig.glb"),
        "out_glb": str(out_glb),
        # Named per job for ``armature_spec``'s reason: ``run_worker`` unlinks
        # and then watches this path, so two bakes sharing one name would let
        # each eat the other's answer.
        "result_path": str(result_dir / ".animate_result.json"),
        "fps": ANIMATION_FPS,
        "clips": tracks,
    }
