"""Import clip: posing a template rig from an externally authored animation.

**A door, not a job kind.** The heavy step is a Blender subprocess (``blender_run.
run_worker`` with ``op="clip_sample"``), but it samples a handful of actions --
a few seconds of CPU, no GPU, no VRAM admission to reason about -- and it
writes nothing durable of its own: the sampled transforms live only long
enough to be converted (``cliptransfer.transfer``, pure host math) and handed
back or merged into a clip library. There is no ``job_dir`` for this to write
into and nothing for a queue row, a progress bar or a cancel button to track.
``service.poses.template_preview`` is the same shape for the same reason --
Blender work with no artifact of its own -- and this module follows it: a
plain synchronous call a pane (or a script, or a future MCP tool) makes on the
``TaskRunner`` thread and waits out, not a ``_q_*`` module.

**Two doors, two audiences.** :func:`analyse` never touches the library --
it is what a preview panel calls to show the user what would be imported
before they commit to anything. :func:`import_into_library` calls
:func:`analyse` and then, only if every clip it proposes is acceptable, folds
the result into the template's clip library under the one lock
``service.clips`` already defines (:func:`~warlock.service.clips._lock`) --
using its private ``_check_shape``/``_commit_locked`` pair rather than
restating them, so an import and a hand-edited save are validated and
committed by the exact same code.
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import cliptransfer, doctor
from ..kernels.rig import blender_spec, cliplib, store
from ..pipelines import blender_run
from . import clips as _clips
from .core import WarlockService
from .errors import Conflict, Failed, Invalid, invalid_from

log = logging.getLogger(__name__)

#: What "Import clip" will read. Case-insensitive, checked on the suffix alone
#: -- the worker (not this door) is what actually opens the file and finds out
#: whether Blender's importers can make sense of it.
SOURCE_EXTENSIONS = (".fbx", ".glb", ".gltf")

#: A big Mixamo export with many actions baked in is a few tens of MB; 200 MiB
#: leaves headroom for that while still refusing, before a subprocess ever
#: starts, the much more likely mistake -- a multi-hundred-MB source .blend or
#: an unrelated video dragged onto the same control.
MAX_SOURCE_BYTES = 200 * 1024 * 1024

#: Where a sampled-and-converted preview's scratch result file goes. A sibling
#: of ``poser/previews/`` and ``poser/clips/`` (see ``poselib``'s path
#: functions) for the same reason: beside the job directories, never inside
#: one -- there is no job here to belong to.
_IMPORTS_SUBDIR = ("poser", "imports")


def _imports_dir(svc: WarlockService) -> Path:
    return Path(svc.config.data_dir).joinpath(*_IMPORTS_SUBDIR)


def _check_source(path: str | Path) -> Path:
    source = Path(path)
    if not source.is_file():
        raise Invalid(f'"{source.name or path}" does not exist', field="source")
    if source.suffix.lower() not in SOURCE_EXTENSIONS:
        raise Invalid(
            f'"{source.name}" is not an animation file this build can read; '
            f"import expects {', '.join(SOURCE_EXTENSIONS)}",
            field="source",
        )
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise Invalid(f'"{source.name}" could not be read: {exc}', field="source") from exc
    if size > MAX_SOURCE_BYTES:
        raise Invalid(
            f'"{source.name}" is over the {MAX_SOURCE_BYTES // (1024 * 1024)} MB import limit',
            field="source",
        )
    return source


def analyse(
    svc: WarlockService,
    template: str,
    path: str | Path,
    *,
    clip_name: str | None = None,
    frames: int | None = None,
    loop: str = "auto",
    root_motion: str = "in_place",
) -> dict[str, Any]:
    """Sample *path* and convert it onto *template*, writing nothing.

    Every action the source file carries comes back converted -- what
    :func:`~warlock.cliptransfer.transfer` returns, unchanged, one entry per
    action. The caller (a preview panel, :func:`import_into_library`) decides
    what to do with the result; this door's whole job is to be safe to call
    on every keystroke of "pick a file" with no side effect to undo.
    """
    key = _clips._template_or_invalid(template)
    source = _check_source(path)

    check = doctor.blender_check()
    if not check.ok:
        raise Failed(check.detail)

    directory = _imports_dir(svc)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / f".{store.new_id()}.clip_sample.json"
    try:
        spec = blender_spec.clip_sample_spec(source, key, result_path)
        try:
            payload = blender_run.run_worker(spec, timeout=svc.config.pose_timeout)
        except blender_run.BlenderError as exc:
            log.error("sampling %s for %s failed: %s", source, key, exc)
            raise Failed("That file could not be read by Blender") from exc
        if not payload.get("ok", False):
            raise Failed(str(payload.get("error") or "That file could not be read by Blender"))
        try:
            converted = cliptransfer.transfer(
                payload,
                template=key,
                clip_name=clip_name,
                frames=frames,
                loop=loop,
                root_motion=root_motion,
            )
        except cliptransfer.ClipTransferError as exc:
            raise invalid_from(exc, "That animation cannot be imported") from exc
    finally:
        # Belt and braces over ``run_worker``'s own cleanup: it unlinks the
        # result file itself on every path that reads it, but this is a
        # scratch file with no reader of its own once this call returns, and
        # a raise between writing the spec and calling the worker (or a
        # worker that dies before producing one) must not strand it, the same
        # rule ``service.poses.template_preview``'s ``tmp.unlink`` follows.
        with contextlib.suppress(OSError):
            result_path.unlink(missing_ok=True)

    return {"template": key, "clips": converted}


def _dedupe_pose_name(name: str, taken: set[str]) -> str:
    """The first ``"<name> N"`` (N >= 2) not already in *taken*.

    Exact-match, matching ``service.clips._check_shape``'s own pose-name
    check -- not ``poselib.next_copy_name``'s case-folded ``.NNN`` scheme,
    which is a different collision universe (library pose *records*, not the
    keys inside one clip library document).
    """
    if name not in taken:
        return name
    n = 2
    while f"{name} {n}" in taken:
        n += 1
    return f"{name} {n}"


def import_into_library(
    svc: WarlockService,
    template: str,
    path: str | Path,
    *,
    replace: bool = False,
    clip_name: str | None = None,
    frames: int | None = None,
    loop: str = "auto",
    root_motion: str = "in_place",
) -> dict[str, Any]:
    """Sample, convert and merge *path*'s actions into *template*'s library.

    The Blender sample and the pure conversion (:func:`analyse`) run *outside*
    the clip-store lock -- they take seconds, and nothing else needs excluding
    while they run. Only the read-merge-write is done under one hold of
    ``service.clips``'s own lock, through its own ``_check_shape`` and
    ``_commit_locked`` -- so an import is validated and committed by exactly
    the code path a hand-edited save is, and a save landing between the read
    and the write here is exactly what the lock rules out.

    **Name collisions.** A clip whose name is already in the library is a
    refusal (``Conflict``, field ``"name"``) unless *replace* is set, in which
    case the old clip entry is dropped and the new one takes its place -- its
    now-unreferenced key poses are simply left in the library rather than
    swept, the same "deleting is the whole implementation" bias
    ``service.clips.revert`` takes: chasing down whether some *other* clip
    still points at one of them is a second traversal this door does not need
    to be correct, and an orphaned pose costs nothing but a few bytes of JSON.
    A pose name that collides with one already in the library (imported or
    original) is never overwritten -- it is renamed with a numbered suffix
    (:func:`_dedupe_pose_name`) and every key in the imported clip that named
    it is rewritten to match.

    On any refusal -- a name clash, a shape the parser or the character-sheet
    frame check refuses -- nothing is written: the merge happens on local
    lists, and ``_commit_locked`` itself restores the previous bytes if the
    render check is what refuses.

    **Rotation space.** :func:`~warlock.cliptransfer.transfer` always emits
    delta-space bases (rest-relative deltas the target's own rig applies on
    top of its own rest pose) -- there is no code path in it that produces a
    node-space (whole world-space) quaternion. Folding that into a
    node-space library would silently reinterpret every pose it holds, the
    same hazard ``service.clips.save`` already refuses by name for a hand
    edit that tries to *change* a library's stored space; this door refuses
    it too, before the (Blender-backed) sample and conversion even run.
    """
    key = _clips._template_or_invalid(template)
    current_space = _clips.library(svc, key)["space"]
    if current_space != "delta":
        raise Invalid(
            f'"{key}" is a {current_space}-space clip library; '
            "imports need a delta-space clip library",
            field="template",
        )
    result = analyse(
        svc, key, path, clip_name=clip_name, frames=frames, loop=loop, root_motion=root_motion
    )
    key = result["template"]
    source_name = Path(path).name
    imported_on = datetime.now(UTC).date().isoformat()

    added: list[str] = []
    replaced: list[str] = []
    reports: list[dict[str, Any]] = []

    with _clips._lock(svc):
        current = _clips.library(svc, key)
        poses = list(current["poses"])
        clip_rows = list(current["clips"])
        pose_names = {str(p["name"]) for p in poses}
        clip_names = {str(c["name"]) for c in clip_rows}

        for entry in result["clips"]:
            clip = dict(entry["clip"])
            report = entry["report"]
            reports.append(report)

            rename: dict[str, str] = {}
            for pose_name, pose_body in entry["poses"].items():
                new_name = _dedupe_pose_name(pose_name, pose_names)
                rename[pose_name] = new_name
                pose_names.add(new_name)
                poses.append({"name": new_name, **pose_body})
            clip["keys"] = [rename[k] for k in clip["keys"]]
            clip["source"] = {"file": source_name, "map": report["map"], "imported": imported_on}

            name = str(clip["name"])
            if name in clip_names:
                if not replace:
                    raise Conflict(f'a clip named "{name}" already exists', field="name")
                clip_rows = [c for c in clip_rows if str(c["name"]) != name]
                replaced.append(name)
            else:
                added.append(name)
            clip_rows.append(clip)
            clip_names.add(name)

        payload = {"space": current["space"], "poses": poses, "clips": clip_rows}
        document = _clips._check_shape({**payload, "template": key})
        try:
            cliplib.parse_clip_library(document)
        except Exception as exc:
            raise invalid_from(exc, "That clip library cannot be saved") from exc

        # The 2026-09-14 audit, finding poser-01: this door merges into the
        # same document shape service.clips.save writes, through the same
        # _check_shape/_commit_locked pair, but save gained a check against
        # cliplib.MAX_CLIP_LIBRARY_BYTES on 2026-09-13 (finding poser-02) that
        # this door never did -- so a source file with enough bones (a dense
        # facial or cloth rig baked into the animation) could still commit a
        # library the read door (_load_clip_library) then refuses forever,
        # silently reverting the template to its shipped clips. Same check,
        # same constant, same place in the sequence save uses it: after the
        # renderer's own parser accepts the document, before a byte commits.
        size = len(json.dumps(document, indent=2).encode("utf-8"))
        if size > cliplib.MAX_CLIP_LIBRARY_BYTES:
            raise Conflict(
                f"this clip library is {size} bytes, over the "
                f"{cliplib.MAX_CLIP_LIBRARY_BYTES}-byte limit the reader enforces",
                field="poses",
            )
        _clips._commit_locked(svc, key, document)

    return {"template": key, "added": added, "replaced": replaced, "reports": reports}
