"""Reworking a finished asset in place: retarget its mesh, restyle its skin.

Split out of ``service/jobs.py``, which had grown to 1,446 lines over five
unrelated subjects; ``jobs.py`` stays as the facade every caller still imports
and calls by attribute.

What separates these from a rerun is that **no new job row is minted for the
mesh** -- the asset keeps its identity and its history, and what changes is
one artifact of it. That is also what makes them the dangerous pair: they
write onto files that are being served, so both refuse a job that is queued or
running, both stage their writes, and both are followed by deleting the
derived exports that no longer describe the thing on disk. The two ``stale_*``
helpers are that list, stated rather than globbed.

``optimize_job`` runs the optimizer inline rather than queueing it, which is
why it shares ``resolve_profile`` with ``create_job`` instead of re-deriving
the budget: one implementation of the gltfpack gate, or a job finishes wearing
a tier that never ran.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from .. import models
from ..kernels.rig import store
from ._jobs_create import resolve_profile
from .core import RealmspinnerService
from .errors import Conflict, Failed, Invalid
from .validation import (
    ARTIFACT_HEALTH,
    check_job_id,
    check_prompt,
    check_seed,
    check_vram,
    check_weights,
    note_degraded,
    random_seed,
)

log = logging.getLogger(__name__)


def optimize_job(
    svc: RealmspinnerService,
    job_id: str,
    *,
    profile: str | None = None,
    custom_triangles: int | None = None,
) -> dict[str, Any]:
    """Rebuild model.glb from source.glb at a different triangle budget.

    Inline rather than on the queue, under the same per-artifact lock the
    STL/OBJ exports use: gltfpack is a two-second subprocess, and putting it
    behind the serial GPU queue would make it wait on a trellis run.

    A terminal status is required. The two writers are otherwise genuinely
    concurrent -- this rewrites model.glb while the worker's own
    _optimize/_apply_scale/_audit_mesh are still writing it and recording what
    they measured -- and no lock here can help, because the worker's half
    doesn't take one. Refusing beats racing.
    """
    import contextlib
    import time

    from ..pipelines import modelhistory, optimize, postprocess
    from . import files

    job = svc.require_job(job_id)
    if job["status"] in ("queued", "running"):
        raise Conflict(f"job is {job['status']}; re-optimize it once it finishes")
    _require_no_dependents(svc, job_id, "re-optimize")
    job_dir = svc.job_dir(job_id)
    source = job_dir / "source.glb"
    if not source.exists():
        raise Invalid("this job has no source reconstruction to re-optimize")
    # Through the one implementation, not a second copy of it: this used to
    # resolve the budget itself, which raised a fieldless Invalid the UI could
    # not point at anything and -- the half that mattered -- skipped the
    # gltfpack-presence refusal. With a broken REALMSPINNER_GLTFPACK a retarget then
    # shipped the raw copy while params["profile"] named a tier that never ran,
    # and `profile` is in VECTOR_PARAMS, so the corpus learned it.
    #
    # ``profile`` is passed through *as given*, including ``None`` -- a
    # retarget with nothing named -- rather than pre-filled with
    # ``svc.config.mesh_profile`` here the way this used to. Pre-filling would
    # turn "nothing named" into an *explicit* request the moment it reached
    # ``resolve_profile``, which on a machine with no gltfpack refuses instead
    # of downgrading (dev/measurements/2026-09-23-default-mesh-budget.md's
    # "defaulted tier downgrades, named tier refuses" split). A retarget
    # button with no tier picked must not start refusing every press the day
    # the default moved off ``raw`` -- so ``resolve_profile`` is left to do its
    # own defaulting, and the *resolved* name (not the argument) is what gets
    # recorded below.
    resolved: dict[str, Any] = {}
    budget = resolve_profile(svc, resolved, profile, custom_triangles)
    profile = resolved["profile"]

    # Read from the row rather than started empty: a step that failed on the
    # *original* run is still true of this mesh unless this run fixes it, and
    # the successful branch below is what clears it. Held out here because the
    # ``changes``/``drop`` pair that consumes it is out here.
    inherited = job["params"].get(ARTIFACT_HEALTH)
    # ``note_degraded``'s guard, for its reason: the value on a hand-edited row
    # (or one of the test fixtures that fills every ``DERIVED_PARAMS`` key with
    # a marker string) is not a dict, and starting fresh beats raising over it.
    health = {
        ARTIFACT_HEALTH: dict(inherited) if isinstance(inherited, dict) else {}
    }
    with svc.convert_lock(job_id, modelhistory.MODEL_LOCK):
        # Staged *before* the run, not after: the old model.glb is only still
        # on disk here, and stage() has to snapshot it while it is still the
        # file optimize.run is about to overwrite. A budget label rather than
        # a face count -- the count model.glb actually has needs a trimesh
        # load this door does not otherwise pay for.
        #
        # stage() only, not keep() -- 2026-09-23 audit, finding service-01:
        # keep() used to evict the oldest kept version unconditionally, before
        # optimize.run had even been tried, so a failed retarget at the
        # version cap permanently destroyed a recoverable mesh even though
        # nothing on disk had actually changed. Eviction now happens in
        # commit(), called only after optimize.run has actually succeeded;
        # discard_last() below undoes exactly this stage() call on failure,
        # with no eviction to undo because none has happened yet.
        entries = modelhistory.stage(
            job_dir,
            modelhistory.entries_of(job["params"]),
            job["params"],
            kind="optimize",
            geometry=True,
            detail=f"{budget:,} triangles" if budget else "raw reconstruction",
            now=time.time(),
        )
        try:
            result = optimize.run(
                source,
                job_dir / "model.glb",
                target_triangles=budget,
                exe=svc.config.gltfpack_exe,
            )
        except optimize.OptimizeError as exc:
            # model.glb is untouched on this path (optimize.run only creates
            # dest on success), so the version just staged describes a
            # replacement that never happened -- back it out rather than let
            # the history claim a retarget that failed.
            entries = modelhistory.discard_last(job_dir, entries)
            svc.store.merge_params(job_id, {"model_history": entries})
            raise Failed(str(exc)) from exc
        # optimize.run succeeded: model.glb now is the retargeted mesh, so the
        # version just staged really does describe what it replaced. Only now
        # is it safe to evict past the cap (2026-09-23 audit, finding
        # service-01 -- see the stage() call above).
        entries = modelhistory.commit(job_dir, entries)
        # The optimizer rewrote the node graph, so the grounding transform went
        # with it and has to be reapplied. Failure is logged and swallowed,
        # same rule as the queue path (_apply_scale): the new GLB is already on
        # disk and serving it unnormalized beats an error that leaves the
        # caches and params half-updated.
        transform = None
        try:
            transform = postprocess.normalize_glb(
                job_dir / "model.glb",
                float(job["params"]["size_m"]) if job["params"].get("size_m") else None,
            )
        except Exception as exc:
            log.exception("normalize failed after optimize for job %s", job_id)
            # Recorded, not only logged. This is the half that was missing, and
            # the half ``_q_mesh._apply_scale``, ``_q_mesh._audit_mesh`` and
            # ``_jobs_create.import_mesh`` all get right for the identical
            # swallow; ``validation`` states the reason in as many words --
            # without it "a user could export a visibly successful asset with
            # the wrong pivot or the wrong scale and have no way to find out".
            # The row stays ``done`` and serves ``model.glb`` either way; the
            # difference is whether anything says so.
            note_degraded(health, "normalize", str(exc))
        # Derived artifacts describe the old mesh; drop them once model.glb is
        # *finished*, and under each artifact's own lock so an in-flight
        # conversion of the old mesh can't rename a stale copy into place after
        # the unlink. Deleting before the normalize left a multi-second window
        # in which an STL or OBJ export -- which takes only its own artifact's
        # lock, never this one -- rebuilt itself from the ungrounded mesh and
        # cached that answer indefinitely.
        for name in files.DERIVED:
            with svc.convert_lock(job_id, name), contextlib.suppress(OSError):
                (job_dir / name).unlink()

    changes: dict[str, Any] = {
        "profile": profile,
        "optimize": result,
        "model_history": entries,
    }
    if custom_triangles is not None:
        changes["custom_triangles"] = custom_triangles
    # The old audit/report describe a mesh that no longer exists. "remesh" and
    # "retexture" are in the same boat: a retarget rebuilds model.glb from
    # source.glb, which overwrites whatever a prior remesh or re-texture
    # baked onto it. "remesh" was added by the 2026-09-18 audit (finding
    # service-01) -- until then this drop list left the old report in params,
    # so remesh_panel's "Last remesh: 8,000 faces" line (params["remesh"], set
    # by _q_mesh's worker path) went on describing quads that were no longer
    # on disk. "retexture" is the same defect, found while wiring
    # ``model_history`` (2026-09-22): the row went on claiming a skin
    # that a retarget had just discarded -- now recoverable under Earlier
    # meshes, but no longer true of the row's own top-level "retexture" key.
    drop = ["mesh_audit", "mesh_report", "remesh", "retexture"]
    if transform is None:
        drop += ["transform", "scale_factor"]
    else:
        changes["transform"] = transform
        changes["scale_factor"] = transform["scale"]
        # And the other direction, which was unhandled too: a stale
        # ``degraded["normalize"]`` inherited from the original run described a
        # mesh that no longer exists, so it left the asset flagged after the
        # step that fixed it.
        health[ARTIFACT_HEALTH].pop("normalize", None)
    if health[ARTIFACT_HEALTH]:
        changes[ARTIFACT_HEALTH] = health[ARTIFACT_HEALTH]
    else:
        drop.append(ARTIFACT_HEALTH)
    # Merged rather than written from the copy read at the top: params is one
    # JSON blob, and a full-blob write from a stale read silently discards
    # anything committed in between.
    svc.store.merge_params(job_id, changes, remove=tuple(drop))
    return {
        "ok": True,
        "optimize": result,
        "transform": transform,
        "stale": stale_rig_artifacts(job_dir),
    }


def revert_model(
    svc: RealmspinnerService,
    job_id: str,
    *,
    version: int,
) -> dict[str, Any]:
    """Put an earlier mesh -- one a retarget, a remesh or a re-texture
    replaced -- back as this job's ``model.glb``.

    Inline, like ``optimize_job``, and under the same lock: a restore writes
    onto the same served name every other rework here does, so it refuses
    exactly what they refuse (a job that is queued or running, a dependent
    job still writing into this one's directory) and for the same reason --
    the worker's own ``_optimize``/``_apply_scale``/``_publish_model_version``
    write ``model.glb`` without taking a lock of their own, so an unlocked
    restore racing one of them would be a second, unordered writer.

    It is a publish rather than a pure derivation -- the bytes come from
    ``versions/<n>.model.glb`` on disk, not from recomputing anything off
    ``source.glb`` -- so it goes through ``service.derive._staged`` by hand,
    the module's own canonical stage-then-rename, rather than through
    ``get_file``'s ``derived`` table of pure functions of ``model.glb``.

    Whether the restore drops the rig, its poses and its sheets, or only the
    surface exports, depends on what changed **between** the restored version
    and the mesh on disk now -- not on what kind of rework produced the
    version being restored. ``modelhistory.crossed_geometry`` walks the whole
    chain from that version forward for exactly that reason: a version from
    two reworks back can still be a pure surface restore if neither rework
    since touched geometry, and a version saved just before a re-texture can
    still cross geometry if a remesh landed after it.

    And the restore is itself kept, the same as every other publish here --
    so restoring one version is not a one-way trip either.
    """
    import contextlib
    import json
    import shutil
    import time

    from ..pipelines import modelhistory, retexture
    from . import files
    from .derive import _staged

    check_job_id(job_id)
    job = svc.require_job(job_id)
    if job["status"] in ("queued", "running"):
        raise Conflict(f"job is {job['status']}; restore an earlier mesh once it finishes")
    _require_no_dependents(svc, job_id, "restore an earlier mesh of")
    job_dir = svc.job_dir(job_id)

    with svc.convert_lock(job_id, modelhistory.MODEL_LOCK):
        # Re-read inside the lock, not the ``job`` fetched at the door: a
        # queued rework's own publish, or a concurrent restore, may have
        # pushed or evicted an entry in between.
        row = svc.require_job(job_id)
        params = row["params"]
        entries = modelhistory.entries_of(params)
        by_n = {e["n"]: e for e in entries}
        entry = by_n.get(version)
        version_glb = modelhistory.version_path(job_dir, version)
        if entry is None or not version_glb.exists():
            if entry is not None:
                # A dangling entry whose file is gone (an eviction that died
                # mid-delete, a hand-edited job directory) -- drop it rather
                # than fail forever on a version nothing can ever restore.
                entries = [e for e in entries if e["n"] != version]
                svc.store.merge_params(job_id, {"model_history": entries})
            raise Invalid(
                f"version {version} is no longer available", field="version"
            )
        geometry = modelhistory.crossed_geometry(entries, version)
        restored_params = json.loads(
            modelhistory.meta_path(job_dir, version).read_text("utf-8")
        )

        # The mesh being replaced is kept too, exactly like every other
        # rework's publish -- so a restore can itself be undone.
        #
        # stage() only, not keep() -- 2026-09-23 (second run) audit, finding
        # service-01: this used to call the combined keep(), which evicted the
        # oldest kept version unconditionally *before* the restore's own
        # staged write to model.glb had even been attempted. A write that then
        # failed (disk full, a permissions error) left the restore refused but
        # the evicted version already gone -- and model_history was never
        # merged on that path either, so the row went on listing the entry
        # that no longer existed, and the retry was refused with "version N is
        # no longer available" from the dangling-entry check above. Eviction
        # now happens in commit(), called only once the write below has
        # actually succeeded; discard_last() undoes exactly this stage() call
        # on failure, with no eviction to undo because none has happened yet
        # -- the same shape optimize_job already uses, for the same reason.
        entries = modelhistory.stage(
            job_dir,
            entries,
            params,
            kind="revert",
            geometry=geometry,
            detail=f"restored version {version}",
            now=time.time(),
        )

        def _write(tmp: Path) -> None:
            shutil.copyfile(version_glb, tmp)

        try:
            _staged(job_dir, "model.glb", _write)
        except OSError as exc:
            # model.glb is untouched on this path -- _staged writes through a
            # temp sibling and os.replace's it only once ``write`` returns --
            # so the version just staged describes a replacement that never
            # happened. Back it out, and merge the un-evicted entries back
            # onto the row so a retry never meets a "no longer available"
            # entry for a version this failure never touched.
            entries = modelhistory.discard_last(job_dir, entries)
            svc.store.merge_params(job_id, {"model_history": entries})
            raise Failed(f"could not restore version {version}: {exc}") from exc
        # The write succeeded: model.glb now is the restored mesh, so the
        # version just staged really does describe what it replaced. Only now
        # is it safe to evict past the cap.
        entries = modelhistory.commit(job_dir, entries)

        # Geometry crossed means every derived export describes a mesh that
        # no longer exists, the same rule optimize_job and remesh_job follow;
        # otherwise only the surface exports do, retexture_job's rule --
        # each under its own lock, so an in-flight conversion of the old mesh
        # cannot rename a stale copy into place after the unlink.
        drop_names = files.DERIVED if geometry else retexture.SURFACE_DERIVED
        for name in drop_names:
            with svc.convert_lock(job_id, name), contextlib.suppress(OSError):
                (job_dir / name).unlink()

        changes = dict(restored_params)
        changes["model_history"] = entries
        # Any MODEL_PARAMS key the restored version never had (a run before
        # the key existed, or one whose step failed) has to come off the row
        # too -- restoring is putting the mesh's whole description back, not
        # only adding what this version recorded.
        remove = tuple(k for k in modelhistory.MODEL_PARAMS if k not in restored_params)
        svc.store.merge_params(job_id, changes, remove=remove)

    return {
        "ok": True,
        "version": version,
        "stale": stale_rig_artifacts(job_dir) if geometry else [],
        "surface": not geometry,
    }


def retexture_job(
    svc: RealmspinnerService,
    job_id: str,
    prompt: str,
    *,
    strength: float | None = None,
    texture_size: int | None = None,
    seed: int | None = None,
    base_model: str | None = None,
    control: str | None = None,
    control_scale: float | None = None,
) -> dict[str, Any]:
    """Queue a new surface for a finished mesh, from a prompt.

    ``optimize_job``'s near-sibling in everything it *refuses* -- a terminal
    status is required, ``source.glb`` is never touched, and the exports that
    describe the old skin go -- and its opposite in where the work runs. A
    retarget is a two-second gltfpack subprocess and belongs inline; a
    re-texture is six SDXL passes around two Blender ops, so it takes the queue
    for the reason every other GPU path does: it needs the resident pipe, and a
    TaskRunner thread racing the worker for VRAM is the OOM that only
    reproduces under load. Every refusal is still *here* rather than in the
    worker, exactly as ``create_pixel_sheet`` states: a mesh with no atlas to
    replace should cost the request, not a place in the queue and a minute of
    GPU.

    The ``Conflict`` is the same one and for the same reason: the worker's own
    ``_optimize``/``_apply_scale`` write ``model.glb`` without taking a lock, so
    a re-texture queued against a running job would be a second writer with no
    ordering between them. Refusing beats racing.
    """
    from ..pipelines import retexture

    check_job_id(job_id)
    job = svc.require_job(job_id)
    if job["status"] in ("queued", "running"):
        raise Conflict(f"job is {job['status']}; re-texture it once it finishes")
    _require_no_dependents(svc, job_id, "re-texture")
    job_dir = svc.job_dir(job_id)
    if not (job_dir / "model.glb").exists():
        raise Invalid("this job has no mesh to re-texture")
    text = (prompt or "").strip()
    if not text:
        raise Invalid("describe the surface you want", field="prompt")
    check_prompt(text)

    value = models.RETEXTURE_DEFAULT_STRENGTH if strength is None else float(strength)
    # The re-texture's own bounds and default, not the sheets': with the depth
    # anchor the ceiling is the 2026-08-08 measurement's positive control and
    # the default is the 2026-08-15 ladder's pick (see models.py).
    if not models.RETEXTURE_STRENGTH_MIN <= value <= models.RETEXTURE_STRENGTH_MAX:
        raise Invalid(
            f"strength must be between {models.RETEXTURE_STRENGTH_MIN} "
            f"and {models.RETEXTURE_STRENGTH_MAX}",
            field="strength",
        )
    # Only the depth control, and only by name: its hint is rendered from the
    # mesh by the worker's own Blender pass. A canny here would be an edge map
    # of the very render being restyled -- a ControlNet's VRAM spent locking
    # the restyle to what it already looks like -- and anything unknown is
    # unknown. The scale is orphaned without its selection for
    # guidance.normalize's reason: it would read as a live setting on rerun.
    if control is not None and control != "depth":
        raise Invalid(
            'the only control a re-texture can render a hint for is "depth"',
            field="control",
        )
    if control_scale is not None and control is None:
        raise Invalid(
            "control_scale needs control to scale", field="control_scale"
        )
    scale = None
    if control is not None:
        spec = models.CONTROLNETS[control]
        scale = spec.default_scale if control_scale is None else float(control_scale)
        if not models.CONTROL_SCALE_MIN <= scale <= models.CONTROL_SCALE_MAX:
            raise Invalid(
                f"control_scale must be between {models.CONTROL_SCALE_MIN} "
                f"and {models.CONTROL_SCALE_MAX}",
                field="control_scale",
            )
    # None means "match the mesh's own atlas", resolved by the worker against
    # the file rather than here: it is a property of the GLB, and reading it at
    # the door would open a 26 MB file to answer a question the run is about to
    # ask anyway. It rides in params as an absent key, which is also what makes
    # a job row written before this existed still mean what it said.
    size = None if texture_size is None else int(texture_size)
    if size is not None and size not in retexture.TEXTURE_SIZES:
        raise Invalid(
            f"texture_size must be one of {list(retexture.TEXTURE_SIZES)}",
            field="texture_size",
        )
    if seed is not None:
        check_seed("seed", seed)
    base_key = str(base_model or svc.config.t2i_model)
    if base_key not in models.BASE_MODELS:
        raise Invalid(f"unknown base model {base_key!r}", field="base_model")
    _check_retexture_family(models.BASE_MODELS[base_key])

    params = {
        # Inputs, every one of them: what the re-texture recorded about the
        # atlas it produced goes on the *mesh's* row under "retexture", which is
        # in DERIVED_PARAMS. Nothing here is derived, so a rerun copies it
        # verbatim.
        "source_job": job_id,
        "strength": value,
        "seed": random_seed() if seed is None else int(seed),
        "base_model": base_key,
    }
    if size is not None:
        params["texture_size"] = size
    if control is not None:
        params["control"] = control
        params["control_scale"] = scale
    # At the door and before the row exists, as everywhere: the img2img passes
    # run through one resident pipe and that is a real budget question beside a
    # warm trellis -- and with "control" in params, check_vram prices the
    # ControlNet and check_weights refuses it undownloaded, both for free.
    check_weights(svc, "text", params)
    check_vram(svc, "retexture", "model", params)
    new_id = svc.store.create("retexture", text, params, uuid.uuid4().hex[:12])
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id, "stale": stale_surface_artifacts(job_dir)}


def remesh_job(
    svc: RealmspinnerService,
    job_id: str,
    *,
    profile: str | None = None,
    custom_triangles: int | None = None,
    texture_size: int | None = None,
    close_holes: bool = True,
) -> dict[str, Any]:
    """Queue a game-ready remesh of a finished mesh: a fresh surface at a
    triangle budget, a fresh unwrap, and the old surface baked onto the new.

    The third rework, and it sits between the other two. Like a retarget it
    changes geometry, so it invalidates *every* derived export and makes a rig
    describe a mesh that no longer exists -- ``stale_rig_artifacts`` is
    reported here for the same reason. Like a re-texture it is minutes of an
    out-of-process Blender run, so it takes the queue rather than the inline
    path: not for the resident pipe (it needs none) but because the serial
    worker is what keeps a multi-minute bake from overlapping a trellis run,
    exactly as ``_rig`` states.

    ``close_holes`` defaults to True, unlike every other bool-flag default in
    this door: measured 2026-09-23, without the voxel pre-pass the decimate
    fallback a trellis mesh always takes collapses the mesh rather than
    producing something usable -- so the panel and this door's own default now
    agree with what actually works, rather than each silently declining the
    pass a plain quadriflow run cannot survive on this input.

    ``source.glb`` is never touched: a remesh reads ``model.glb`` -- the mesh
    as the user sees it, current skin included -- and publishes over it. A
    later retarget rebuilds from the reconstruction and discards the remesh,
    which is the standing rule that ``model.glb`` is derived and
    ``source.glb`` is the authority.
    """
    from .. import doctor
    from ..pipelines import remesh

    check_job_id(job_id)
    job = svc.require_job(job_id)
    if job["status"] in ("queued", "running"):
        raise Conflict(f"job is {job['status']}; remesh it once it finishes")
    _require_no_dependents(svc, job_id, "remesh")
    job_dir = svc.job_dir(job_id)
    if not (job_dir / "model.glb").exists():
        raise Invalid("this job has no mesh to remesh")
    # At the door, where refusing is cheap: without bpy the worker would take a
    # queue slot to exit 3. The UI hides the panel on the same answer.
    check = doctor.blender_check()
    if not check.ok:
        raise Invalid(
            "a remesh runs in Blender, which is not installed "
            "(`uv sync --extra rig` on Python 3.13)",
            field="remesh_profile",
        )
    key = profile or remesh.DEFAULT_TRIANGLE_PROFILE
    try:
        triangles = remesh.resolve(key, custom_triangles)
    except ValueError as exc:
        raise Invalid(
            str(exc), field="custom_triangles" if key == "custom" else "remesh_profile"
        ) from exc
    size = None if texture_size is None else int(texture_size)
    if size is not None and size not in remesh.TEXTURE_SIZES:
        raise Invalid(
            f"texture_size must be one of {list(remesh.TEXTURE_SIZES)}",
            field="texture_size",
        )
    params: dict[str, Any] = {
        # ``remesh_profile`` rather than ``profile``: the latter is the gltfpack
        # tier and is in VECTOR_PARAMS, and a triangle budget wearing that key
        # would land in the findings corpus as a gltfpack tier.
        "source_job": job_id,
        "remesh_profile": key,
        "target_triangles": triangles,
        "close_holes": bool(close_holes),
    }
    if key == "custom":
        params["custom_triangles"] = triangles
    if size is not None:
        params["texture_size"] = size
    # Zero on this kind -- Blender is out of process -- but held for the
    # uniformity every queued kind has: a budget question is asked at the door.
    check_vram(svc, "remesh", "model", params)
    new_id = svc.store.create("remesh", job["prompt"], params, uuid.uuid4().hex[:12])
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id, "stale": stale_rig_artifacts(job_dir)}


def separate_job(
    svc: RealmspinnerService,
    job_id: str,
    *,
    separation_model: str | None = None,
) -> dict[str, Any]:
    """Queue a split of a finished take into its instrument stems.

    **A fourth sibling here rather than an action on a task thread**, and
    ``retexture_job``'s docstring is the deciding sentence: "a TaskRunner thread
    racing the worker for VRAM is the OOM that only reproduces under load."
    ``remesh_job`` makes the parallel point for a job that needs the queue but
    not the resident pipe, which is this one exactly -- the separation child is
    short-lived and holds nothing, but it wants a card the music pipe may still
    be giving back.

    That single decision inherits admission, cancellation, the progress phases,
    the per-card progress bar, ``_require_no_dependents`` and ``asset_open``
    routing for free.

    The stems land in the **source take's** directory, at
    ``stems/{name}.wav``, with ``stems.json`` written last as the completion
    gate -- ``rig.json``'s rule and ``sheet.json``'s, stated identically. So
    this is a follow-up in ``asset_open``'s sense: it writes into another job's
    directory and its own is never created.

    A note for the manual rather than for the code: **Sirens also exports into
    a folder called ``stems/``.** Same word, two unrelated places. They
    deliberately do not share a constant -- ``service/files.py`` must not reach
    into ``studio/`` -- so one sentence in the chapter is what prevents the
    confusion.
    """
    check_job_id(job_id)
    job = svc.require_job(job_id)
    if job.get("kind") != "music":
        raise Invalid("only a track can be split into stems", field="source_job")
    if job["status"] in ("queued", "running"):
        raise Conflict(f"job is {job['status']}; split it once it finishes")
    _require_no_dependents(svc, job_id, "separate", noun="music take")
    job_dir = svc.job_dir(job_id)
    if not (job_dir / "track.wav").exists():
        # ``muse_mode.play``'s sentence and ``derive_music_job``'s, so all
        # three surfaces say the same thing about the same missing file.
        raise Invalid("that take has no audio on disk", field="source_job")

    key = str(separation_model or models.DEFAULT_SEPARATION)
    if key not in models.SEPARATION_MODELS:
        raise Invalid(
            "that separation model is not one this build knows about",
            field="separation_model",
        )
    params: dict[str, Any] = {"source_job": job_id, "separation_model": key}
    # Both halves of admission, in ``create_job``'s order. ``check_weights``
    # refuses *this* job by name when the model is absent and never the music
    # job -- see ``SeparationModel``: a missing separation model costs a
    # feature, not a job.
    check_weights(svc, "separate", params)
    check_vram(svc, "separate", "music", params)
    new_id = svc.store.create("separate", job["prompt"], params, uuid.uuid4().hex[:12])
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id}


def _require_no_dependents(
    svc: RealmspinnerService, job_id: str, what: str, *, noun: str = "mesh"
) -> None:
    """Refuse while another job is still writing into this one's directory.

    Both doors here refuse on the *target row's* own status, which is the wrong
    question by itself: a re-texture, a rig or a sheet is a **separate row**
    whose artifacts land in the done job's directory, so the target reads `done`
    the whole time one is in flight -- which is exactly why it could be queued
    for it in the first place.

    The failure is silent and it inverts an explicit user choice. Queue a
    re-texture for done mesh J, then retarget J: the re-texture's ``os.replace``
    publishes a skin baked from the *pre-retarget* geometry over the retargeted
    mesh, so the triangle budget quietly reverts while ``params["profile"]``
    goes on claiming the tier ran. A rig in flight binds the old mesh and
    ``stale_rig_artifacts`` reports nothing, because nothing it looks at
    changed. Per-artifact locks do not cover this: the two writers are different
    jobs writing different names (CON-01).

    ``dependent_jobs`` already exists and ``_jobs_lifecycle``'s docstring
    already names it as the answer to this shape.

    ``noun`` names the thing being started from, not only meshes any more --
    ``separate_stems`` calls this on a music take, and the hardcoded "mesh" in
    the message used to call a finished track's take a mesh (2026-09-23 audit,
    finding muse-01).
    """
    from ._jobs_lifecycle import dependent_jobs

    blocking = dependent_jobs(svc, job_id)
    if blocking:
        raise Conflict(
            f"{len(blocking)} job(s) started from this {noun} are still queued or "
            f"running and will write into its directory; wait for them to finish "
            f"before you {what} it."
        )


def _check_retexture_family(base: models.BaseModel) -> None:
    """Refuse a non-SDXL checkpoint here, where refusing is still cheap.

    A re-texture is six conditioned img2img passes, and ``Text2Image._conditioned``
    refuses a non-SDXL family outright -- but it does so at *runtime*, which for
    this kind means after the job has queued, rendered all six Blender views,
    stopped trellis and loaded a ~16 GiB checkpoint into host commit. Minutes of
    the serial worker plus a trellis restart, to arrive at a refusal the
    registry could have given instantly (MDL-15).

    Reachable without anyone picking an exotic model, too: ``base_model`` is
    optional here and falls back to ``config.t2i_model``, so a host whose
    ``REALMSPINNER_T2I_MODEL`` names a klein entry qualified merely by leaving the
    field alone -- and ``rerun_job --reroll`` repeated the whole thing.

    Mirrors ``create_pixel_sheet``'s refusal in shape and in field, including
    naming the models that *would* work: a door that only says no is a door the
    user has to guess their way past.
    """
    if base.family == models.FAMILY_SDXL:
        return
    # Spelled out rather than borrowed from ``tile_bases()``, on that list's own
    # reasoning: a seamless tile and a conditioned re-texture are two different
    # questions whose answers happen to coincide today, and sharing the
    # derivation is how one of them silently acquires the other's rule.
    usable = sorted(k for k, m in models.BASE_MODELS.items() if m.family == models.FAMILY_SDXL)
    raise Invalid(
        f"base_model {base.key!r} is {base.family}; a re-texture conditions "
        f"every view on its render, which only SDXL-family checkpoints support. "
        f"Pick one of {usable}",
        field="base_model",
    )


def stale_surface_artifacts(job_dir: Path) -> list[str]:
    """What a re-texture will invalidate, so the panel can say so beforehand.

    Deliberately *not* ``stale_rig_artifacts``, and the difference is the whole
    point: a rig, its poses and its sheets reference geometry, and a re-texture
    changes no geometry -- so none of them appears here, and
    ``tests/test_retexture.py`` asserts that rather than leaving it to be
    "fixed" later. What does appear is the exports that carry the skin.

    Reported rather than merely deleted after the fact, the rule the retarget
    panel already follows: a warning before the button beats a list afterwards.
    """
    from ..pipelines import retexture

    return [n for n in retexture.SURFACE_DERIVED if (job_dir / n).exists()]


def stale_rig_artifacts(job_dir: Path) -> list[str]:
    """What still describes the old mesh after a retarget.

    Reported, not deleted: the rig and its poses are user work, and a warning
    the caller can act on beats silently destroying them over a triangle
    retarget.
    """
    # ``animated.glb`` is in the list because it is baked *from* rig.glb, so a
    # retarget makes it describe a skeleton skinned to a mesh that no longer
    # exists -- the one derived artifact whose source is the rig rather than
    # the mesh (``files.DERIVED_RIG``).
    out = [n for n in ("rig.glb", "rig.json", "animated.glb") if (job_dir / n).exists()]
    if out:
        out += sorted(
            f"{store.POSE_DIR_NAME}/{p.name}" for p in store.pose_dir(job_dir).glob("*.glb")
        )
        out += sorted(
            f"{store.SHEET_DIR_NAME}/{p.name}" for p in store.sheet_dir(job_dir).glob("*.png")
        )
    return out
