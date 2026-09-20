"""Rigs, joint adjustment and poses.

A pose is a bone -> local rotation map saved against a job's rig, stored as a
file in that job's directory. No job kind and no DB row: a pose is small,
instant to write, and belongs to the mesh the same way its rig does. Baking one
into a GLB is the only slow part, and that is derived on demand and cached,
exactly like the STL and OBJ exports.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import uuid
from pathlib import Path
from typing import Any

from .. import doctor, poselib
from ..kernels.rig import blender_spec, poses, skeleton, store, templates
from ..pipelines import blender_run
from .core import RealmspinnerService
from .errors import Conflict, Failed, Invalid, NotFound, invalid_from
from .validation import check_job_id, check_pose_id, valid_template

log = logging.getLogger(__name__)


def rig_templates(svc: RealmspinnerService) -> dict[str, Any]:
    """Which skeletons a rig request may ask for, plus whether rigging works at
    all. The UI hides the rig controls when available is false rather than
    offering a button that can only fail."""
    check = doctor.blender_check()
    return {
        "available": check.ok,
        "detail": check.detail,
        "default": svc.config.rig_template,
        "templates": templates.catalog(),
    }


def template_presets(key: str) -> dict[str, Any]:
    """The shipped pose library for a skeleton.

    Read-only and job-independent: applying one saves an ordinary pose through
    the normal save path, so a preset and a hand-made pose are the same thing
    by the time anything else sees them.
    """
    try:
        return {"poses": poses.preset_poses(key)}
    except ValueError as exc:
        raise invalid_from(
            exc, "That skeleton has no pose library", field="rig_template"
        ) from exc


def create_rig(
    svc: RealmspinnerService, job_id: str, *, template: str | None = None
) -> dict[str, Any]:
    """Queue a rig for a finished job's mesh.

    A queue job rather than an inline call: automatic weights on a 300k-face
    mesh is minutes of CPU, and going through the queue buys cancellation,
    progress and history for free -- and guarantees it never overlaps a trellis
    or SDXL run.
    """
    source = svc.require_job(job_id)
    if source["kind"] == "rig":
        raise Invalid("cannot rig a rig job; rig its source mesh")
    if source["status"] != "done" or not (svc.job_dir(job_id) / "model.glb").exists():
        raise Invalid("job has no finished mesh to rig")
    # The 2026-09-14 audit (service-03): this door minted a fresh rig job with
    # no check at all -- only troupe.send_to_troupe and the agent wrapper
    # called rig_in_flight, so Library's "Rig this mesh" and Poser's "Re-rig"
    # (different ctx.submit keys, rig:<id> vs poser-asset-rerig:<id>) could
    # both queue one for the same mesh. Both finalize_rig into the same
    # job_dir, and history showed two "done" rigs for one served result. Moved
    # into the door itself so every caller is covered, not just the two that
    # remembered to ask first -- send_to_troupe and the agent wrapper still
    # call it too, which is harmless double-checking.
    if rig_in_flight(svc, job_id) is not None:
        # send_to_troupe's own sentence, verbatim (troupe.py:981) -- one
        # wording for "there is already a rig job for this mesh" wherever it
        # is met, mirroring create_rig's own Blender refusal below. Unlike
        # that door's comment, this one carries ``job_id``: this is the
        # direct door (Library's "Rig this mesh", Poser's "Re-rig", and the
        # agent's own character_rig, which already checks first and so never
        # reaches this line) rather than the sheet reservation's own door,
        # which draws no job_id control to ring.
        raise Conflict("a rig for this mesh is already running", field="job_id")
    params = {"source_job": job_id, "template": valid_template(template, svc.config.rig_template)}
    # After every other refusal, and still before the row is written: this UI
    # hides the Rig button when ``rig_templates``' own probe says bpy is
    # absent, so the only paths that reach here on such a host are the MCP
    # agent surface and a stale frame -- exactly the reachable-by-an-agent case
    # worth refusing at the door rather than leaving to queue a job that dies
    # in ``pipelines/blender_worker.py`` with exit code 3. ``troupe.py``'s own
    # gate's sentence, verbatim, so the app has one wording for "this needs
    # Blender" wherever it is met.
    if not doctor.blender_check().ok:
        raise Invalid("Rigging needs Blender, which is not installed.")
    new_id = svc.store.create("rig", source["prompt"], params, uuid.uuid4().hex[:12])
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id, "template": params["template"]}


def rig_in_flight(svc: RealmspinnerService, job_id: str) -> str | None:
    """The id of a queued or running rig job for *job_id*'s mesh, or None.

    ``send_to_troupe``'s guard against minting a second rig -- and so a second
    ``troupe_sheet`` reservation -- for the same mesh while the first is still
    in flight (the 2026-09-13 audit's own finding: it did not check before
    this existed). ``store.active_jobs()`` is already oldest-first, so the
    first match is the request the user is already waiting on, never a later
    duplicate.
    """
    check_job_id(job_id)
    for row in svc.store.active_jobs():
        if row.get("kind") != "rig":
            continue
        params = row.get("params") or {}
        if str(params.get("source_job") or "") == job_id:
            return str(row["id"])
    return None


def adjust_joints(svc: RealmspinnerService, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Re-rig a mesh with joints the user moved.

    A queued rig job rather than an inline call, for the same reason the
    original rig is: skinning is minutes of CPU and must never overlap a
    trellis run.

    A rig whose ``skeleton`` is already ``"custom"`` (built by
    :func:`edit_skeleton`) is checked against *its own* structure, not the
    base template's -- ``skeleton.validate_joints`` only ever accepts a joint
    move that keeps the structure it is handed, so validating against the
    template here would refuse a plain joint move on a rig with, say, an
    extra tail bone the template never had. ``skeleton``/``root``/
    ``mirror_pairs`` are forwarded into the new job's params the same way, so
    a joint move never resets a custom rig back to template shape.
    """
    source = svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    rig = store.read_rig(job_dir)
    if rig is None or not (job_dir / "model.glb").exists():
        raise Invalid("job is not rigged")
    template = templates.get_template(str(rig.get("template") or svc.config.rig_template))
    is_custom = rig.get("skeleton") == "custom"
    try:
        if is_custom:
            store.validate_rig_bones(rig.get("bones", []))
            structure = rig["bones"]
        else:
            structure = template
        bones = skeleton.validate_joints(payload, structure)
    except (ValueError, KeyError, TypeError) as exc:
        raise invalid_from(exc, "Those joint positions cannot be used") from exc

    # The 2026-09-15 audit, finding poser-02: service-03 (the 2026-09-14
    # audit) moved this check into ``create_rig`` on the claim that every
    # caller was then covered, but this door and ``edit_skeleton``'s below
    # mint a rig job of their own and never called it -- so the Joints pane's
    # "Apply joint positions" (submit key ``joints:<id>``, unshared with
    # ``create_rig``'s callers) could still queue a second rig for a mesh
    # that already had one in flight.
    if rig_in_flight(svc, job_id) is not None:
        raise Conflict("a rig for this mesh is already running", field="job_id")
    # Same door as ``create_rig``'s, for the same reason: a re-rig queues a
    # fresh job that runs Blender exactly like the first one did.
    if not doctor.blender_check().ok:
        raise Invalid("Rigging needs Blender, which is not installed.")
    params = {
        "source_job": job_id,
        "template": template.key,
        "bones": bones,
        "adjusted": True,
    }
    if is_custom:
        params["skeleton"] = "custom"
        params["root"] = rig.get("root")
        params["mirror_pairs"] = rig.get("mirror_pairs")
    new_id = svc.store.create("rig", source["prompt"], params, uuid.uuid4().hex[:12])
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id}


def edit_skeleton(svc: RealmspinnerService, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Re-rig a mesh with a skeleton whose *shape* the user edited.

    ``adjust_joints``'s shape, for a caller that may have added or removed a
    pivot, split a bone, renamed one or grafted a limb preset on -- anything
    :func:`skeleton.validate_skeleton` accepts, not only a joint moved within
    the base template's own structure. The queue is serial, exactly like
    every other rig job: two edits queued against the same source job can
    never race each other's temp names (``store.RIG_GLB_TMP``/
    ``RIG_JSON_TMP``), because only one rig job ever runs at a time.

    ``rig.json`` keeps naming the *base* template this rig started from
    (``rig["template"]``) even once its shape has diverged from it --
    ``skeleton.clip_coverage`` and the pose library both need to know which
    template's poses/clips this rig might still play.
    """
    source = svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    rig = store.read_rig(job_dir)
    if rig is None or not (job_dir / "model.glb").exists():
        raise Invalid("job is not rigged")
    base = templates.get_template(str(rig.get("template") or svc.config.rig_template))
    bounds = rig.get("bounds")
    try:
        if not isinstance(bounds, dict) or "min" not in bounds or "max" not in bounds:
            raise store.RigError("rig.json has no usable bounds", field="bounds")
        result = skeleton.validate_skeleton(payload, base=base, bounds=bounds)
    except ValueError as exc:
        raise invalid_from(exc, "That skeleton cannot be used") from exc

    # The 2026-09-15 audit, finding poser-02: the same missing check as
    # ``adjust_joints``'s above, and for the same reason -- this door mints
    # its own rig job and was never on service-03's "every caller is
    # covered" list either.
    if rig_in_flight(svc, job_id) is not None:
        raise Conflict("a rig for this mesh is already running", field="job_id")
    # Same door as ``create_rig``'s and ``adjust_joints``'s, for the same
    # reason: this queues a fresh job that runs Blender exactly like they do.
    if not doctor.blender_check().ok:
        raise Invalid("Rigging needs Blender, which is not installed.")
    params = {
        "source_job": job_id,
        "template": base.key,
        "bones": result["bones"],
        "root": result["root"],
        "mirror_pairs": [list(p) for p in result["mirror_pairs"]],
        "skeleton": result["skeleton"],
        "adjusted": True,
    }
    new_id = svc.store.create("rig", source["prompt"], params, uuid.uuid4().hex[:12])
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id, "skeleton": result["skeleton"]}


def limb_presets() -> list[dict[str, Any]]:
    """The shipped limb presets the skeleton editor can graft onto a bone.

    Read-only and job-independent, like :func:`rig_templates`'s own
    ``templates`` list: ``{key, label, bone_count}`` is enough for a menu. The
    coordinate frame a preset is authored in, and what ``side``/``mirror``
    mean, are documented on :func:`skeleton.attach_limb`, which is what
    actually places one -- this is only the catalogue.
    """
    return [
        {"key": p["key"], "label": p["label"], "bone_count": len(p["bones"])}
        for p in templates.limb_presets().values()
    ]


def get_rig(svc: RealmspinnerService, job_id: str) -> dict[str, Any]:
    check_job_id(job_id)
    rig = store.read_rig(svc.job_dir(job_id))
    if rig is None:
        raise NotFound("job is not rigged")
    try:
        # The 2026-09-08 audit (poser-02): a rig.json that passes read_record's
        # file-level guards but carries a bone with no "name" used to reach a
        # caller as an uncaught KeyError instead of a field-addressed refusal.
        store.validate_rig_bones(rig.get("bones", []))
    except ValueError as exc:
        raise invalid_from(exc, "That rig cannot be read") from exc
    return rig


def _rig_bones(svc: RealmspinnerService, job_id: str) -> list[str]:
    try:
        bones = store.rig_bone_names(svc.job_dir(job_id))
    except ValueError as exc:
        # rig_bone_names now validates the bone list itself (poser-02); see
        # get_rig's own comment above for the incident.
        raise invalid_from(exc, "That rig cannot be read") from exc
    if bones is None:
        raise NotFound("job is not rigged")
    return bones


def list_poses(svc: RealmspinnerService, job_id: str) -> dict[str, Any]:
    check_job_id(job_id)
    bones = _rig_bones(svc, job_id)
    job_dir = svc.job_dir(job_id)
    # Named for what it is rather than for the return key it fills: the
    # ``kernels.rig.poses`` module is imported into this same file, and a
    # local variable named the same reads fine but shadows it for the rest
    # of this function -- ``poses.validate_bones(raw)`` below silently meant
    # ``list.validate_bones`` once this was called ``poses``, an
    # ``AttributeError`` only a real run surfaces.
    kept = []
    for record in store.list_poses(job_dir):
        # The same "costs you that pose, not the app" rule every sibling
        # loader in kernels.rig follows: a listing must survive one hand-edited
        # file, unlike posed_model, which addresses a single pose and can
        # refuse it outright (poser-01, the 2026-09-11 audit).
        raw = record.get("bones")
        if not isinstance(raw, dict) or not raw:
            log.warning(
                "pose %s/%s has no bones; omitting it from the list", job_id, record.get("id")
            )
            continue
        try:
            poses.validate_bones(raw)
        except ValueError:
            log.warning(
                "pose %s/%s has an unusable bones map; omitting it from the list",
                job_id,
                record.get("id"),
            )
            continue
        kept.append(record)
    return {"bones": bones, "poses": kept}


def save_pose(svc: RealmspinnerService, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Create a pose, or overwrite one by passing its id."""
    check_job_id(job_id)
    known = _rig_bones(svc, job_id)
    job_dir = svc.job_dir(job_id)
    try:
        pose = poses.validate_pose(payload, known)
    except ValueError as exc:
        raise invalid_from(exc, "That pose cannot be saved") from exc

    # A root offset, when the caller sends one -- poser_mode.save_pose_to_asset
    # now does, mirroring the shared library's own save (poselib.validate_record
    # via _payload/apply_pose). Absent from the payload stores nothing, so
    # every save that never touched the root (and every payload built before
    # this field existed) is unaffected; the same field on the shared library
    # goes through the identical validator, so a root offset means the same
    # thing -- character-height units, +/-2.0 -- wherever it is authored
    # (poser-02, the 2026-09-11 audit: this door used to drop the field on the
    # floor, silently losing a crouch or hop saved directly onto an asset).
    extra: dict[str, Any] = {}
    if payload.get("root_translation") is not None:
        try:
            extra["root_translation"] = poselib.validate_root_translation(
                payload["root_translation"]
            )
        except ValueError as exc:
            raise invalid_from(exc, "That pose cannot be saved") from exc

    pose_id = str(payload["id"]) if payload.get("id") else None
    if pose_id is not None:
        check_pose_id(pose_id)
        if not store.pose_path(job_dir, pose_id).exists():
            raise NotFound("no such pose")
        # Under the pose's bake lock: an in-flight bake of the old rotations
        # must finish (and be deleted here) before the new rotations land, or
        # the stale GLB gets cached under this id.
        with svc.convert_lock(job_id, f"pose:{pose_id}"):
            return store.save_pose(job_dir, pose, pose_id, extra=extra or None)
    # The cap is a check-then-write, so the count and the write that depends on
    # it happen under one hold -- exactly the rule the library's own cap in
    # service/poses.py states. Lock-free, two callers saving at once both read
    # MAX_POSES - 1 and both save, and the job ends up over its cap with no way
    # to notice. A job-wide key rather than a per-pose one, because what is
    # being guarded is the *set* of poses, not any one of them; it is a
    # different lock from the f"pose:{id}" bake locks and never nests with one.
    with svc.convert_lock(job_id, "poses"):
        if len(store.list_poses(job_dir)) >= store.MAX_POSES:
            raise Conflict(f"a job may hold at most {store.MAX_POSES} poses")
        return store.save_pose(job_dir, pose, pose_id, extra=extra or None)


def delete_pose(svc: RealmspinnerService, job_id: str, pose_id: str) -> dict[str, Any]:
    check_job_id(job_id)
    check_pose_id(pose_id)
    # Under the same lock the overwrite and the bake take: deleting while a
    # bake is in flight let the bake recreate <pose_id>.glb after its .json
    # was gone, leaving an orphan GLB nothing could ever reach or clean up.
    with svc.convert_lock(job_id, f"pose:{pose_id}"):
        try:
            deleted = store.delete_pose(svc.job_dir(job_id), pose_id)
        except OSError as exc:
            # The wrap lives here, not in rigging: the storage half may not
            # import service, and a file another program is holding open is a
            # real failure with a real remedy, not a generic error.
            log.error("deleting pose %s/%s failed: %s", job_id, pose_id, exc)
            raise Failed(
                "That pose could not be deleted; a file may be locked by another program."
            ) from exc
        if not deleted:
            raise NotFound("no such pose")
    return {"ok": True}


def _pose_or_not_found(job_dir: Path, pose_id: str) -> dict[str, Any]:
    """The semantic read door for a job's own saved pose.

    ``service.poses._record_or_not_found``'s shape, for a job-scoped pose
    instead of a library one: ``store.read_pose`` only gives ``read_record``'s
    three file-level guards (valid JSON, valid dict, under the byte ceiling),
    never the bones check a *library* pose gets through
    ``poselib.validate_record``. A hand-edited pose file missing "bones" used
    to reach ``_pose_bake_spec``'s ``pose["bones"]`` as a bare ``KeyError``, and
    one with a malformed quaternion was forwarded straight into the Blender
    worker spec with nothing to catch ``blender_worker.main()``'s unwrapped
    ``op(bpy, spec)`` (the 2026-09-11 audit, finding poser-01).

    Not checked against the rig's *current* bone names -- unlike ``save_pose``'s
    write-time check -- because a rig can be rebuilt with a different template
    after a pose was saved, and that is a stale pose, not a corrupt one; only
    the record's own shape (a non-empty bones map of valid quaternions) is
    re-verified here.
    """
    record = store.read_pose(job_dir, pose_id)
    if record is None:
        raise NotFound("no such pose")
    raw = record.get("bones")
    if not isinstance(raw, dict) or not raw:
        raise Invalid("pose has no bones", field="bones")
    try:
        bones = poses.validate_bones(raw)
    except ValueError as exc:
        raise invalid_from(exc, "That pose cannot be read", field="bones") from exc
    return dict(record, bones=bones)


def _pose_bake_spec(job_dir: Path, pose_id: str, pose: dict[str, Any]) -> dict[str, Any]:
    """The bake spec, with a snapshot's root translation scaled onto this rig.

    A pose applied from the global library can carry ``root_translation`` in
    character-height units; the rig's own height comes from rig.json's bounds
    and the offset lands on its root bone. Absent or zero yields today's spec
    exactly -- ``pose_spec`` adds no keys -- and a rig.json that cannot answer
    (pre-template, unreadable) costs the offset, never the bake.

    Every question asked of those two files is asked tolerantly, because both
    are read off disk: a string where a number belongs, a NaN, a ``bounds``
    that is a list rather than a box -- all fall through to the same
    offset-less spec the warn path below already documents.
    ``root_offset_world`` itself stays pure and raising; the tolerance belongs
    at the boundary, not in the arithmetic.
    """
    root = pose.get("root_translation")
    values: list[float] = []
    if isinstance(root, (list, tuple)) and len(root) == 3:
        try:
            values = [float(v) for v in root]
        except (TypeError, ValueError):
            values = []
        if not all(math.isfinite(v) for v in values):
            values = []
    if values and any(values):
        rig = store.read_rig(job_dir) or {}
        bounds, bone = rig.get("bounds"), rig.get("root")
        if isinstance(bounds, dict) and "min" in bounds and "max" in bounds and bone:
            try:
                return blender_spec.pose_spec(
                    job_dir,
                    pose_id,
                    pose["bones"],
                    root_bone=str(bone),
                    root_offset=blender_spec.root_offset_world(values, bounds),
                )
            except (TypeError, ValueError, IndexError):
                # IndexError as well as the plan's two: a two-element "min" is
                # exactly as plausible a hand edit as a string one, and
                # root_offset_world indexes [2] for the height.
                log.warning("pose %s has a root offset rig.json cannot scale", pose_id)
                return blender_spec.pose_spec(job_dir, pose_id, pose["bones"])
        log.warning("pose %s carries a root offset but rig.json cannot scale it", pose_id)
    return blender_spec.pose_spec(job_dir, pose_id, pose["bones"])


def posed_model(svc: RealmspinnerService, job_id: str, pose_id: str) -> Path:
    """The rig with one saved pose baked into it.

    Derived on first request and cached beside the pose, under the same
    per-artifact lock the STL/OBJ exports use -- posing runs Blender, and two
    callers asking at once should wait for one subprocess, not start two.
    """
    check_job_id(job_id)
    check_pose_id(pose_id)
    job_dir = svc.job_dir(job_id)
    path = store.pose_glb_path(job_dir, pose_id)
    # Existence is only checked under the lock, and the pose is *read* under it
    # too -- a delete landing between the read and the bake would otherwise
    # recreate the GLB with no .json beside it.
    with svc.convert_lock(job_id, f"pose:{pose_id}"):
        pose = _pose_or_not_found(job_dir, pose_id)
        if not path.exists():
            if not (job_dir / "rig.glb").exists():
                raise NotFound("job is not rigged")
            # Baked through a staging file and renamed, like every other
            # derivation in this codebase. Existence *is* this artifact's
            # freshness test, so a Blender that dies part way through -- a
            # pose_timeout, a kill-on-close at shutdown, a rig it cannot
            # weight -- would otherwise leave a truncated GLB that is served
            # under this pose id forever. The finally matters as much as the
            # replace: nothing ever looks at a dotfile in a pose directory
            # again, so a stranded one lives as long as the job.
            spec = _pose_bake_spec(job_dir, pose_id, pose)
            tmp = path.with_name(f".{pose_id}.tmp.glb")
            spec["out_glb"] = str(tmp)
            try:
                blender_run.run_worker(spec, timeout=svc.config.pose_timeout)
                os.replace(tmp, path)
            except blender_run.BlenderError as exc:
                log.error("posing %s/%s failed: %s", job_id, pose_id, exc)
                raise Failed("could not bake this pose") from exc
            except OSError as exc:
                # The rename after a Blender that succeeded: an antivirus
                # holding the staging file, a full disk. The pose is still what
                # was not baked, so it says the same thing.
                log.error("renaming the bake of %s/%s into place failed: %s", job_id, pose_id, exc)
                raise Failed("could not bake this pose") from exc
            finally:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
    return path
