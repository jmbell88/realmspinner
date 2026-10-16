"""``Worker``'s mesh post-processing: retarget, ground, rank, measure.

Everything that happens to a mesh *after* the reconstruction is on disk, and
nothing that happens before it. Split out of ``queue.py`` for the reason every
mixin here is: the class had 35 methods and five unrelated subjects, and the
worker's loop core -- which is the part with the invariants -- was buried in
the middle of them.

Six of these share a rule the rest of the worker does not: **none of them may
fail the job**. The reconstruction is on disk and usable by the time any of
them runs, so a lost triangle budget costs file size, a lost remesh costs a
larger export, a lost transform costs a manual fixup, and a lost measurement
costs a badge -- while raising would cost the user the mesh. Every one of them
logs and returns instead.

No module-scope import of ``.queue`` (that would be circular) and no queue
module-level names are referenced, so there is no ``queue_mod`` indirection
here -- the logger is this module's own.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .queue import Worker

log = logging.getLogger(__name__)

# The params key recording a swallowed post-processing failure.
#
# Spelled here rather than imported: the queue layer may not import ``service``,
# which is the same reason ``VECTOR_PARAMS`` lives in ``vectors.py``. The
# authority is ``service.validation.ARTIFACT_HEALTH`` -- and its membership of
# ``DERIVED_PARAMS``, without which a reroll would wear a degradation belonging
# to the run it replaced. A test pins the two spellings together.
ARTIFACT_HEALTH = "degraded"


def _note_degraded(params: dict[str, Any], step: str, detail: str) -> None:
    """Record that a canonical step did not run, on the job's own row.

    ART-01: normalization and the mesh report are wrapped in catch-everything
    handlers, deliberately -- a mesh trimesh cannot parse must not fail a job
    whose ``source.glb`` is a good reconstruction. But the job then stayed
    ``done`` with the only evidence in a log line, so a user could export a
    visibly successful asset with the wrong pivot or scale and never find out.

    The non-fatal decision stands; the silence does not. Steps accumulate rather
    than overwrite, because a mesh can fail more than one of them and the second
    failure is not a correction of the first.
    """
    health = params.get(ARTIFACT_HEALTH)
    if not isinstance(health, dict):  # a hand-edited row; start fresh
        health = {}
    # **Rebound, never mutated in place**, which is the rule ``_optimize``,
    # ``_apply_scale`` and ``_audit_mesh`` already follow and the reason
    # ``_q_generate``'s remesh loop can snapshot the winning attempt with a
    # shallow ``dict(params)``. That loop's own comment names this exact
    # hazard -- "a callee that grew an in-place nested mutation would defeat
    # it" -- and ``setdefault`` plus ``health[step] =`` was that callee: with
    # ``REALMSPINNER_MESH_RETRIES`` set, attempt 2's failures wrote straight through
    # the shared dict into ``best["params"]["degraded"]``, so the job that
    # shipped attempt 1's mesh carried attempt 2's health record.
    params[ARTIFACT_HEALTH] = {**health, step: detail}


def _effective_triangle_budget(params: dict[str, Any]) -> int | None:
    """The triangle ceiling this job's own generate pipeline actually built to.

    The 2026-09-23 audit, finding pipelines-01: ``_audit_mesh`` called
    ``meshreport.build`` with no ``triangle_budget``, so a mesh retargeted or
    lowpoly-remeshed to a custom budget above the module's 150k default was
    recorded as over budget in the very report meant to describe it.
    ``params["lowpoly"]["requested"]`` and ``params["optimize"]["requested"]``
    are both already triangle counts (``_lowpoly`` and ``optimize.run`` record
    them that way), so no quad conversion is needed here the way ``_remesh``'s
    own call site needs one. Lowpoly wins when both ran: it is the last,
    stricter remesh the shipped mesh was actually built to.
    """
    for record in (params.get("lowpoly"), params.get("optimize")):
        if isinstance(record, dict):
            requested = record.get("requested")
            if isinstance(requested, int) and requested > 0:
                return requested
    return None


class _TierCheckRejected(RuntimeError):
    """The Game-ready remesh failed its check against the mesh it replaced."""

    def __init__(self, verdict: Any) -> None:
        self.verdict = verdict
        super().__init__(
            "the remesh failed its check against the optimized mesh: "
            + ", ".join(verdict.failures)
        )


class _NoUsableRemesh(RuntimeError):
    """Every finisher on the ladder failed or lost a texture, UV or material."""

    def __init__(self, attempts: list[dict[str, Any]]) -> None:
        detail = "; ".join(
            f"{a['rung']}: {a.get('error') or 'no mesh'}" for a in attempts
        ) or "no finisher could run"
        super().__init__(f"no finisher produced a usable mesh ({detail})")


class MeshPostOps:
    """Mesh post-processing, mixed into :class:`~.queue.Worker`."""

    def _publish_model_version(
        self: Worker, source_id: str, temp: Path, *, kind: str, geometry: bool, detail: str
    ) -> None:
        """Snapshot the mesh a rework is about to replace, then publish over it.

        The one place ``_remesh`` and ``_retexture`` rename their staging file
        onto another job's ``model.glb`` -- both used to call ``os.replace``
        directly, which is how a remesh-then-retexture lost the remesh and a
        retarget silently discarded either: the old bytes were simply gone the
        moment the new ones landed. ``pipelines.modelhistory.keep`` is what
        makes that recoverable, and this method is only the plumbing that gets
        it the lock and the fresh row it needs.

        Synchronous and meant to be called through ``asyncio.to_thread``, like
        every other blocking write in this file -- a job's directory can hold a
        multi-hundred-megabyte ``model.glb``, and ``shutil.copyfile`` of one is
        not a coroutine's business on the event loop thread.

        Under ``self.artifact_lock(source_id, modelhistory.MODEL_LOCK)`` --
        the same lock ``optimize_job`` takes as ``svc.convert_lock(job_id,
        "optimize")`` (``studio.runtime`` wires
        ``worker.artifact_lock = svc.convert_lock``, so the two spellings name
        one lock) -- and the row is re-read *inside* that lock rather than
        trusted from the caller's own copy: a retarget or another rework's
        publish can be racing this one for the same ``model_history`` list,
        and a stale read here would silently drop whichever entry landed
        first, the exact hazard ``db.merge_params`` exists to close for a
        params write in general.

        ``stage()``/``commit()``, not ``keep()`` -- 2026-09-23 audit, finding
        service-03: ``keep()`` stages a version (and may evict one at the cap)
        before the ``os.replace`` below; a failed replace used to leave the
        staged pair orphaned *and* the eviction already applied, while
        ``merge_params`` -- never reached -- left the stored index still
        naming the version eviction had just deleted from disk. Eviction now
        waits for ``commit()``, called only once ``os.replace`` has actually
        succeeded; a failed replace instead calls ``discard_last()``, which
        undoes only the ``stage()`` call, then re-raises so the caller learns
        its publish did not happen.
        """
        import time

        from .pipelines import modelhistory

        source_dir = self.config.job_dir(source_id)
        with self.artifact_lock(source_id, modelhistory.MODEL_LOCK):
            row = self.store.get(source_id) or {}
            params = row.get("params") or {}
            entries, staged = modelhistory.stage(
                source_dir,
                modelhistory.entries_of(params),
                params,
                kind=kind,
                geometry=geometry,
                detail=detail,
                now=time.time(),
            )
            try:
                os.replace(temp, source_dir / "model.glb")
            except OSError:
                entries = modelhistory.discard_last(source_dir, entries, staged)
                raise
            entries = modelhistory.commit(source_dir, entries)
            self.store.merge_params(source_id, {"model_history": entries})

    def _drop_surface_artifacts(self: Worker, source_dir: Path) -> None:
        """Delete the exports that carry the old skin, and only those.

        ``model.stl`` and ``collision.glb`` are geometry with no texture in
        them at all, and a re-texture changes no geometry -- deleting them would
        cost the user a re-export to produce a byte-identical file. The rig, its
        poses and its sheets are the same argument one level up and are the
        subject of a written assertion elsewhere: a rig references geometry, not
        pixels.

        Under each artifact's own lock where one is available. The worker holds
        no service, so ``artifact_lock`` is injected by ``studio.runtime`` and
        falls back to no lock at all -- which is the pre-existing behaviour of
        every other write the worker makes to a model.glb, and is why
        ``optimize_job`` refuses a job that is queued or running.
        """
        from .pipelines import retexture

        for name in retexture.SURFACE_DERIVED:
            with self.artifact_lock(source_dir.name, name), contextlib.suppress(OSError):
                (source_dir / name).unlink()

    async def _remesh(self: Worker, job: dict[str, Any]) -> None:
        """Remesh a finished mesh to a quad budget and rebake its surface.

        One Blender op (``op_remesh``) and a host-side publish. Like a rig and
        a re-texture the product lands in the *source* job's directory, over
        its ``model.glb``, by rename -- and like a re-texture the rename is the
        point of no return, so a cancel that lands after it commits rather
        than lying about the file on disk.

        What it invalidates is the union of the other two reworks': every
        derived export (geometry changed) *and* every surface export (the
        atlas is new). The rig is reported stale at the door, never deleted.
        """
        from . import tiercheck
        from .kernels.rig import blender_spec, store
        from .pipelines import blender_run, postprocess, remesh

        job_id = job["id"]
        params = job["params"]
        source_id = str(params.get("source_job") or "")
        if not store.is_valid_id(source_id):
            raise ValueError(f"source_job is not a job id: {source_id!r}")
        source_dir = self.config.job_dir(source_id)
        model_glb = source_dir / "model.glb"
        if not model_glb.exists():
            raise RuntimeError("source job has no mesh to remesh")
        source_row = await asyncio.to_thread(self.store.get, source_id)
        source_params = (source_row or {}).get("params") or {}

        # ``target_triangles`` is the triangle ladder's own key; ``target_faces``
        # is read only as a fallback for a row queued before it existed, and a
        # legacy row's number is already Blender's own quad count (what a
        # quadriflow ``target_faces`` means) -- halving it again would ask for
        # half the budget the user actually picked, so it is passed straight
        # through rather than run back through remesh.target_faces.
        requested_triangles = params.get("target_triangles")
        legacy_faces = params.get("target_faces")
        if requested_triangles is not None:
            triangles: int | None = int(requested_triangles)
            target = remesh.target_faces(triangles)
        elif legacy_faces is not None:
            triangles = None
            target = int(legacy_faces)
        else:
            triangles = remesh.TRIANGLE_PROFILES[remesh.DEFAULT_TRIANGLE_PROFILE]
            target = remesh.target_faces(triangles)
        asked = params.get("texture_size")
        if asked:
            texture_size = int(asked)
        else:
            from .pipelines import retexture

            # ``match_the_mesh_size``, not ``atlas_size``: the 2026-10-14 audit
            # (finding pipelines-mesh-01) found a remesh left on "match the mesh"
            # baking at the raw albedo size, so a 4096 atlas sent Blender a
            # five-map float bake at 4096 squared with nothing refusing it --
            # the 2026-10-03 pipelines-13 defect, fixed on the re-texture side
            # only. The remesh door refuses an explicit size outside
            # ``remesh.TEXTURE_SIZES``; this holds the unset one to the same
            # ceiling.
            texture_size = (
                await asyncio.to_thread(retexture.match_the_mesh_size, model_glb)
                or remesh.DEFAULT_TEXTURE_PX
            )
        temp = source_dir / store.REMESH_GLB_TMP
        job_dir = self.config.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        self.progress.update(
            job_id, phase="remesh", label="Starting Blender", inner=0.0,
            inner_next=0.05, nominal=60.0,
            detail=f"{triangles:,} triangles" if triangles is not None else f"{target:,} quads",
        )

        def on_progress(frac: float, label: str) -> None:
            self.progress.update(
                job_id, phase="remesh", label=label, inner=frac,
                inner_next=min(frac + 0.1, 1.0), nominal=120.0, detail="",
            )

        before = await asyncio.to_thread(tiercheck.survey, model_glb)
        try:
            result = await asyncio.to_thread(
                functools.partial(
                    blender_run.run_worker,
                    blender_spec.remesh_spec(
                        model_glb,
                        temp,
                        job_dir,
                        target_faces=target,
                        texture_size=texture_size,
                        close_holes=bool(params.get("close_holes")),
                        seed=remesh.QUADRIFLOW_SEED,
                    ),
                    on_progress=on_progress,
                    on_start=self._note_blender,
                    timeout=self.config.rig_timeout,
                )
            )
            if self._cancel is not None and self._cancel.event.is_set():
                return
            if not temp.exists():
                raise RuntimeError("Blender reported success but wrote no mesh")
            # The 2026-09-26 audit (service-kinds-06): same fix as
            # ``_q_lora.py``'s ``publish`` phase -- ``inner`` is
            # phase-relative to ``PHASES_REMESH``'s ``"publish": (0.95,
            # 1.00)``, and the whole-bar-looking 0.95 here was getting
            # remapped a second time (99.75%), jumping the bar to ~100%
            # before grounding/publishing had even started.
            self.progress.update(
                job_id, phase="publish", label="Grounding and publishing", inner=0.0,
                inner_next=1.0, nominal=5.0, detail="",
            )
            # The exporter wrote a fresh node graph, so the grounding transform
            # has to be reapplied -- ``optimize_job``'s reason. Swallowed the
            # same way: the mesh is good and serving it ungrounded beats losing
            # it, and the health record says which.
            transform = None
            health: dict[str, Any] = {}
            try:
                size_m = source_params.get("size_m")
                transform = await asyncio.to_thread(
                    postprocess.normalize_glb, temp, float(size_m) if size_m else None
                )
            except Exception as exc:
                log.exception("normalize failed after remesh for job %s", source_id)
                health["normalize"] = str(exc)
            after = await asyncio.to_thread(tiercheck.survey, temp)
            verdict = tiercheck.compare(before, after)
            # The mesh this replaces is kept under Earlier meshes
            # (2026-09-22) rather than simply overwritten -- a bare
            # ``os.replace`` here is what used to make a remesh-then-retexture
            # lose the remesh with no way back.
            await asyncio.to_thread(
                self._publish_model_version,
                source_id,
                temp,
                kind="remesh",
                geometry=True,
                detail=f"{result.get('faces', target):,} faces ({result.get('method', 'remesh')})",
            )
        finally:
            with contextlib.suppress(OSError):
                temp.unlink(missing_ok=True)
        if self._cancel is not None:
            self._cancel.commit()

        # Geometry *and* skin changed, so every export goes -- the tuple is
        # ``files.DERIVED`` restated where the queue may look (the worker may
        # not import ``service``; ``tests/test_remesh.py`` pins the pair), and
        # each goes under its own lock.
        for name in remesh.GEOMETRY_DERIVED:
            with self.artifact_lock(source_dir.name, name), contextlib.suppress(OSError):
                (source_dir / name).unlink()

        report = dict(result)
        report["target_faces"] = target
        if triangles is not None:
            report["target_triangles"] = triangles
        report["tiercheck"] = {
            "ok": verdict.ok,
            "failures": list(verdict.failures),
            "notes": list(verdict.notes),
        }
        params["remesh"] = report
        await asyncio.to_thread(self.store.set_params, job_id, params)
        changes: dict[str, Any] = {"remesh": report}
        drop = ["mesh_audit", "mesh_report", "optimize", "lowpoly"]
        if transform is not None:
            changes["transform"] = transform
            changes["scale_factor"] = transform["scale"]
        else:
            drop += ["transform", "scale_factor"]
        inherited = source_params.get(ARTIFACT_HEALTH)
        merged = dict(inherited) if isinstance(inherited, dict) else {}
        merged.pop("normalize", None)
        merged.update(health)
        if merged:
            changes[ARTIFACT_HEALTH] = merged
        else:
            drop.append(ARTIFACT_HEALTH)
        await asyncio.to_thread(
            self.store.merge_params, source_id, changes, remove=tuple(drop)
        )
        # The audit and the report describe the mesh that is now on disk.
        # ``triangles`` is this remesh's own requested budget -- ``target`` is
        # Blender's quad count (target_faces = triangles // 2), and the 2026-09-23
        # audit (finding pipelines-01) is what a report judged against the
        # quad figure, or against no figure at all, would get wrong: the
        # module default (150k) rather than the up-to-200k ceiling this mesh
        # was actually built to.
        await self._audit_published(
            source_id, model_glb, source_params.get("size_m"), triangle_budget=triangles
        )
        log.info(
            "remeshed job %s from %s: %s faces via %s, tiercheck %s",
            source_id, job_id, report.get("faces"), report.get("method"),
            "ok" if verdict.ok else "/".join(verdict.failures),
        )

    async def _audit_published(
        self: Worker,
        source_id: str,
        glb_path: Path,
        size_m: Any,
        *,
        triangle_budget: int | None = None,
    ) -> None:
        """Re-measure a mesh another job just published over, by merge.

        ``_audit_mesh`` writes the whole params blob of the job it is handed,
        which is right for a job's own row mid-run and wrong for a *source*
        row another writer may be touching -- so this is the same two
        measurements, merged in.

        ``triangle_budget``, when given, is the budget the publishing job
        actually built this mesh to -- see :func:`_remesh`'s call site and the
        2026-09-23 audit, finding pipelines-01: with nothing passed here this
        report fell back to ``meshreport.TRIANGLE_BUDGET`` (150k) even for a
        mesh remeshed to a larger, explicitly accepted budget.
        """
        try:
            from . import meshaudit, meshreport

            audit = await asyncio.to_thread(
                meshaudit.hole_fraction,
                glb_path,
                meshaudit.DEFAULT_VIEWS,
                meshaudit.REQUEST_PATH_RESOLUTION,
            )
            summary = {k: audit[k] for k in ("worst", "mean", "faces", "resolution")}
            report = await asyncio.to_thread(
                functools.partial(
                    meshreport.build,
                    glb_path,
                    target_size_m=size_m,
                    silhouette=summary,
                    triangle_budget=triangle_budget,
                )
            )
        except Exception:
            log.exception("audit after publish failed for job %s", source_id)
            return
        await asyncio.to_thread(
            self.store.merge_params,
            source_id,
            {"mesh_audit": summary, "mesh_report": report},
        )

    async def _optimize(
        self: Worker, job_id: str, source: Path, dest: Path, params: dict[str, Any]
    ) -> None:
        """Retarget the reconstruction to the job's triangle budget.

        Before the transform, not after: gltfpack rewrites the node graph, and
        running it over an already-grounded model would discard the transform
        node normalize_glb inserted. Optimizing first and transforming second is
        the only ordering where both survive.

        A failure here is not fatal. The reconstruction is on disk and usable;
        losing the budget costs the user file size, and failing the job would
        cost them the mesh.
        """
        if self._cancel is not None and self._cancel.event.is_set():
            return
        from .pipelines import optimize

        try:
            # A missing "profile" key is a *legacy* row, not an unset one:
            # since resolve_profile (service/_jobs_create.py) started
            # recording the resolved tier on every submit
            # (dev/measurements/2026-09-23-default-mesh-budget.md [lost; re-measure to change]),
            # every row this worker claims carries the key. A row with none predates that change and
            # ran "raw" -- the config default at the time -- so the fallback is the literal string,
            # not `self.config .mesh_profile`, which today reads "standard" and would silently
            # reinterpret an old row as having asked for a budget it never did.
            budget = optimize.resolve(
                str(params.get("profile") or "raw"),
                params.get("custom_triangles"),
            )
        except ValueError:
            log.warning("job %s has an unusable profile; shipping raw", job_id)
            budget = None
        self.progress.update(
            job_id, phase="optimize", label="Optimizing mesh", inner=0.0,
            inner_next=1.0, nominal=4.0, detail=f"{budget:,} tris" if budget else "raw",
        )
        try:
            result = await asyncio.to_thread(
                functools.partial(
                    optimize.run,
                    source,
                    dest,
                    target_triangles=budget,
                    exe=self.config.gltfpack_exe,
                )
            )
        except Exception as exc:
            log.exception("optimize failed for job %s; shipping the reconstruction", job_id)
            # Staged, not copyfile: on the /optimize route dest is a done
            # job's model.glb that a concurrent GET may be serving.
            await asyncio.to_thread(optimize.staged_copy, source, dest)
            _note_degraded(
                params,
                "optimize",
                f"the triangle budget was not applied ({exc}); this mesh is the "
                f"raw reconstruction",
            )
            await asyncio.to_thread(self.store.set_params, job_id, params)
            return
        params["optimize"] = result
        await asyncio.to_thread(self.store.set_params, job_id, params)

    async def _lowpoly(
        self: Worker, job_id: str, glb_path: Path, params: dict[str, Any]
    ) -> None:
        """Remesh the optimized mesh to a game-ready triangle budget, in place.

        Modelled on ``_remesh``'s Blender call, but it runs *inside* the model
        job rather than as a follow-up: a follow-up would race
        ``_maybe_queue_rig``, which would rig the 268k reconstruction and be
        stale the moment this remesh replaced it
        (dev/measurements/2026-09-23-default-mesh-budget.md [lost; re-measure to change]). Between
        ``_optimize`` and ``_apply_scale`` in ``_q_generate``'s loop, so it
        bakes from whatever the triangle budget already produced (or the raw
        reconstruction, when no budget was asked for) and the grounding
        transform that follows applies to the mesh actually shipped.

        Runs only when ``params["lowpoly_triangles"]`` names a budget --
        ``resolve_lowpoly`` (service/_jobs_create.py) is the only door that
        sets it, and a legacy row with no key at all runs none, exactly as a
        row with no ``profile`` key ran raw before ``resolve_profile`` existed.

        A failure here is not fatal, the rule every mesh post-processing step
        in this file follows: the optimized mesh is already on disk, and
        losing the remesh costs the user a larger file, not the asset.
        """
        if self._cancel is not None and self._cancel.event.is_set():
            return
        triangles = params.get("lowpoly_triangles")
        if not triangles:
            return
        from . import tiercheck
        from .pipelines import remesh

        triangles = int(triangles)
        # Missing on a stored row means the previous repair path.
        finishing = params.get("mesh_finishing", "repair")
        if finishing == "engine":
            if params.get("engine_faces") and await self._lowpoly_from_engine(
                job_id, glb_path, params, triangles
            ):
                return
            # The engine was never asked for the budget (an engine without
            # --faces, a pinned --decim) or it did not honour the field: the
            # reconstruction is the engine's full ~300k mesh, and Blender is
            # what finishes it. ``preserve_shape`` is the first Blender rung and
            # what "Finishing" meant before "engine" existed.
            finishing = "preserve_shape"
        target = remesh.target_faces(triangles)
        asked = params.get("texture_size")
        if asked:
            texture_size = int(asked)
        else:
            from .pipelines import retexture

            # ``match_the_mesh_size``, not ``atlas_size``: the 2026-10-14 audit
            # (finding pipelines-mesh-01) found a remesh left on "match the mesh"
            # baking at the raw albedo size, so a 4096 atlas sent Blender a
            # five-map float bake at 4096 squared with nothing refusing it --
            # the 2026-10-03 pipelines-13 defect, fixed on the re-texture side
            # only. The remesh door refuses an explicit size outside
            # ``remesh.TEXTURE_SIZES``; this holds the unset one to the same
            # ceiling.
            texture_size = (
                await asyncio.to_thread(retexture.match_the_mesh_size, glb_path)
                or remesh.DEFAULT_TEXTURE_PX
            )
        job_dir = self.config.job_dir(job_id)
        ladder = remesh.finishing_ladder(finishing)
        # Staged per rung, as dotfiles beside the served model.glb, not served names
        # of their own -- ``_publish_model_version``'s reason doesn't apply here
        # (this is the same, still-running job's own file, never a different job's
        # served artifact), but the extension still has to be .glb: Blender's exporter
        # and gltfpack both pick their writer from it.
        staged = {rung: job_dir / f".model.lowpoly.{rung}.glb" for rung in ladder}

        self.progress.update(
            job_id, phase="lowpoly", label="Starting Blender", inner=0.0,
            inner_next=0.05, nominal=60.0, detail=f"{triangles:,} triangles",
        )

        def on_progress(frac: float, label: str) -> None:
            self.progress.update(
                job_id, phase="lowpoly", label=label, inner=frac,
                inner_next=min(frac + 0.1, 1.0), nominal=120.0, detail="",
            )

        attempts: list[dict[str, Any]] = []
        try:
            before = await asyncio.to_thread(tiercheck.survey, glb_path)
            for index, rung in enumerate(ladder):
                if rung == "gltfpack" and not self.config.gltfpack_exe.is_file():
                    continue
                if index:
                    self.progress.update(
                        job_id, phase="lowpoly", label=f"Trying {rung.replace('_', ' ')}",
                        inner=0.0, inner_next=0.05, nominal=60.0,
                        detail=f"{triangles:,} triangles",
                    )
                attempt = await self._lowpoly_attempt(
                    rung, glb_path, staged[rung], job_dir,
                    requested=triangles, target=target, texture_size=texture_size,
                    before=before, on_progress=on_progress,
                )
                if self._cancel is not None and self._cancel.event.is_set():
                    params["lowpoly"] = {
                        "requested": triangles, "status": "cancelled", "source": "source.glb",
                        "finishing": rung,
                    }
                    await asyncio.to_thread(self.store.set_params, job_id, params)
                    return
                attempts.append(attempt)
                # Every step is gated: the first finisher to pass ends the ladder.
                if attempt["usable"] and attempt["miss"] <= 1.0:
                    break
            usable = [a for a in attempts if a["usable"]]
            if not usable:
                raise _NoUsableRemesh(attempts)
            # Nothing passed -> the least-missed one ships and says so; ``miss`` is
            # the largest excess over any limit (``remesh.gate``), so this prefers a
            # mesh that is a little over on one limit to one that is far over on another.
            winner = min(usable, key=lambda a: a["miss"])
            await asyncio.to_thread(os.replace, winner["path"], glb_path)
        except Exception as exc:
            log.exception(
                "lowpoly remesh failed for job %s; keeping the optimized mesh", job_id
            )
            _note_degraded(
                params,
                "lowpoly",
                f"the game-ready remesh did not run ({exc}); this mesh is the "
                f"optimized reconstruction, not a {triangles:,}-triangle remesh",
            )
            params["lowpoly"] = {
                "requested": triangles,
                "status": "cancelled" if self._cancel and self._cancel.event.is_set() else "failed",
                "error": str(exc),
                "finishing": finishing,
                "source": "source.glb",
            }
            # The first tier-check refusal among the attempts, in the shape the
            # record always carried: what the remesh dropped, by name.
            rejected = next(
                (a for a in attempts if (a.get("tiercheck") or {}).get("ok") is False), None
            )
            if rejected is not None:
                params["lowpoly"]["tiercheck"] = rejected["tiercheck"]
            await asyncio.to_thread(self.store.set_params, job_id, params)
            return
        finally:
            for path in staged.values():
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
        achieved = winner["achieved"]
        passed = winner["miss"] <= 1.0
        params["lowpoly"] = {
            "status": (
                "over_budget"
                if isinstance(achieved, int) and achieved > triangles * remesh.OVER_BUDGET_RATIO
                else "complete"
            ),
            "finishing": winner["rung"],
            "source": "source.glb",
            "requested": triangles,
            "achieved": achieved,
            "method": winner["method"],
            "triangles": achieved,
            "geometry": winner["geometry"],
            "texture_size": winner["texture_size"],
            "tiercheck": winner["tiercheck"],
            # What each finisher did, in the order tried. The winner is the last
            # one when it passed; when nothing did, it is the one that missed least.
            "attempts": [
                {k: a[k] for k in ("rung", "achieved", "miss", "reasons", "error") if k in a}
                for a in attempts
            ],
        }
        if finishing != params.get("mesh_finishing", "repair"):
            params["lowpoly"]["fell_back_from"] = params.get("mesh_finishing", "repair")
        if passed:
            params.pop("budget_miss", None)
        else:
            reason = "; ".join(winner["reasons"]) or "missed the Game-ready gate"
            params["budget_miss"] = {"requested": triangles, "achieved": achieved, "reason": reason}
            _note_degraded(
                params,
                "lowpoly",
                f"no finisher met the {triangles:,}-triangle budget cleanly ({reason}); "
                f"this is the closest of {len(attempts)} tried ({winner['rung']})",
            )
        await asyncio.to_thread(self.store.set_params, job_id, params)

    async def _lowpoly_attempt(
        self: Worker,
        rung: str,
        source: Path,
        out: Path,
        job_dir: Path,
        *,
        requested: int,
        target: int,
        texture_size: int,
        before: Any,
        on_progress: Any,
    ) -> dict[str, Any]:
        """Run one finisher and grade what it wrote against the Game-ready gate.

        -> a dict with ``usable`` (the file at ``path`` may ship: it exists and kept
        its UVs and materials -- ``tiercheck``), ``miss`` (``remesh.gate``: <= 1.0
        passes), ``reasons``, and the figures the record keeps. A finisher that
        raises is an attempt that is not usable, not an error: the next rung may
        still work, and only a ladder with nothing usable fails the step.
        """
        from . import tiercheck
        from .kernels.rig import blender_spec
        from .pipelines import blender_run, optimize, remesh

        outcome: dict[str, Any] = {
            "rung": rung, "path": out, "usable": False, "miss": float("inf"), "reasons": [],
            "achieved": None, "method": rung, "geometry": None, "tiercheck": None,
            "texture_size": None,
        }
        try:
            if rung == "gltfpack":
                result = await asyncio.to_thread(
                    functools.partial(
                        optimize.run, source, out,
                        target_triangles=requested, exe=self.config.gltfpack_exe,
                    )
                )
                # optimize.run checks its own output (tiercheck) before it writes.
                verdict = None
                if result.get("achieved") == result.get("source_triangles"):
                    # A copy: the mesh was already inside the budget (or the
                    # simplifier could not reduce it at all), so nothing was
                    # finished. Not a rung that "passed" -- a dead Blender and a
                    # small reconstruction must still read as the failure they are.
                    outcome["error"] = "gltfpack left the mesh unchanged"
                    return outcome
            else:
                result = await asyncio.to_thread(
                    functools.partial(
                        blender_run.run_worker,
                        blender_spec.remesh_spec(
                            source,
                            out,
                            job_dir,
                            target_faces=target,
                            texture_size=texture_size,
                            close_holes=rung == "repair",
                            preserve_shape=rung == "preserve_shape",
                            seed=remesh.QUADRIFLOW_SEED,
                        ),
                        on_progress=on_progress,
                        on_start=self._note_blender,
                        timeout=self.config.rig_timeout,
                    )
                )
                if self._cancel is not None and self._cancel.event.is_set():
                    return outcome
                if not out.exists():
                    raise RuntimeError("Blender reported success but wrote no mesh")
                after = await asyncio.to_thread(tiercheck.survey, out)
                verdict = tiercheck.compare(before, after)
                outcome["method"] = result.get("method")
                outcome["texture_size"] = result.get("texture_size", texture_size)
            geometry = await asyncio.to_thread(remesh.compare_geometry, source, out)
            achieved = result.get("triangles", result.get("faces", result.get("achieved")))
            if achieved is None and geometry.get("measured"):
                achieved = geometry.get("triangles")
            outcome.update(achieved=achieved, geometry=geometry)
            if verdict is not None:
                outcome["tiercheck"] = {
                    "ok": verdict.ok,
                    "failures": list(verdict.failures),
                    "notes": list(verdict.notes),
                }
                if not verdict.ok:
                    # Manual 23: a remesh that dropped UVs, a PBR map or a material
                    # assignment is checked "against the mesh it replaced, and a
                    # failure there keeps the optimized mesh" (2026-10-03 audit,
                    # pipelines-05) -- here, by not being a usable rung.
                    outcome["error"] = str(_TierCheckRejected(verdict))
                    return outcome
            else:
                outcome["tiercheck"] = result.get("tiercheck")
            miss, reasons = remesh.gate(requested, achieved, geometry)
            outcome.update(usable=True, miss=miss, reasons=reasons)
        except Exception as exc:
            log.exception("lowpoly finisher %s failed", rung)
            outcome["error"] = str(exc)
        return outcome

    async def _lowpoly_from_engine(
        self: Worker,
        job_id: str,
        glb_path: Path,
        params: dict[str, Any],
        requested: int,
    ) -> bool:
        """Verify a mesh the engine simplified to the budget itself.

        The engine ran QEM to ``requested`` *before* its own UV and PBR bake,
        so the texture was baked onto the mesh that ships and there is nothing
        left to remesh: this only counts what came back and records it.
        -> ``True`` when the mesh stands, ``False`` when it is more than
        ``remesh.BUDGET_SLACK`` times the request (an exe that ignored the
        field returns its built-in ~300k) and the caller should remesh in Blender.

        ``source_faces`` is recorded either way: it is what ``source.glb``
        really holds, and the retarget door refuses to go above it -- a mesh
        already at 5,000 cannot be re-derived at 20,000.

        No silhouette is measured. The gate compares a finished mesh with the
        higher-resolution one it was made from, and the engine path keeps no
        such mesh (the high-poly companion is a later slice), so ``geometry`` is
        recorded as unmeasured rather than as a pass.
        """
        from . import meshaudit
        from .pipelines import remesh

        try:
            _, faces = await asyncio.to_thread(meshaudit.load_mesh, glb_path)
        except Exception as exc:
            # An unreadable mesh fails the audit that follows with its own reason;
            # remeshing from a file Blender would not read either helps nothing.
            log.exception("could not count the engine's mesh for job %s", job_id)
            params["lowpoly"] = {
                "requested": requested, "status": "failed", "error": str(exc),
                "finishing": "engine", "source": "source.glb", "method": "engine",
            }
            await asyncio.to_thread(self.store.set_params, job_id, params)
            return True
        achieved = int(len(faces))
        params["source_faces"] = achieved
        if achieved > requested * remesh.BUDGET_SLACK:
            log.warning(
                "job %s: asked the engine for %d triangles and got %d; remeshing in Blender",
                job_id, requested, achieved,
            )
            await asyncio.to_thread(self.store.set_params, job_id, params)
            return False
        params["lowpoly"] = {
            "status": "complete",
            "finishing": "engine",
            "source": "source.glb",
            "requested": requested,
            "achieved": achieved,
            "method": "engine",
            "triangles": achieved,
            "geometry": {
                "measured": False,
                "reason": "the engine simplified before baking; no higher-resolution mesh was kept",
            },
            "texture_size": None,
        }
        await asyncio.to_thread(self.store.set_params, job_id, params)
        return True

    async def _apply_scale(
        self: Worker, job_id: str, glb_path: Path, params: dict[str, Any]
    ) -> None:
        """Resize the finished mesh, centre it in X/Z and sit it on the floor.

        The grounding half runs even when no ``size_m`` was requested -- a
        pivot at the reconstruction volume's centre is a manual fixup on every
        import regardless of whether the asset also needed rescaling.

        Runs after the GLB is on disk and holds no GPU memory, so it sits
        outside the VRAM handoff entirely. A cancel that landed during the
        trellis stage skips it -- the artifact is about to be deleted anyway.
        """
        if self._cancel is not None and self._cancel.event.is_set():
            return
        size_m = params.get("size_m")
        self.progress.update(
            job_id,
            phase="scale",
            label="Scaling and grounding",
            inner=0.0,
            inner_next=1.0,
            nominal=2.0,
            detail=f"{size_m} m" if size_m else "grounding",
        )
        # Imported here, not at module scope: postprocess pulls in trimesh,
        # which app startup should not pay for.
        from .pipelines import postprocess

        try:
            transform = await asyncio.to_thread(
                postprocess.normalize_glb, glb_path, float(size_m) if size_m else None
            )
        except Exception as exc:
            # Grounding runs on every job now, not just sized ones, so a mesh
            # trimesh cannot parse must not fail a job whose GLB is already on
            # disk -- the same rule _audit_mesh and the report follow. The
            # report's achieved_size_m is what tells the user the size did not
            # land, rather than a job that errored after producing a model.
            log.exception("normalize failed for job %s; leaving the mesh as-is", job_id)
            _note_degraded(
                params,
                "normalize",
                f"the mesh was not centred, grounded{' or resized' if size_m else ''} "
                f"({exc}); its pivot and scale are the engine's",
            )
            await asyncio.to_thread(self.store.set_params, job_id, params)
            return
        params["scale_factor"] = transform["scale"]
        params["transform"] = transform
        await asyncio.to_thread(self.store.set_params, job_id, params)

    def _rank_reference(self: Worker, image_path: Path, params: dict[str, Any]) -> dict[str, Any]:
        """Score a finished reference. Blocking -- called through to_thread.

        ``image_path`` is the same input.png the report was measured from and is
        passed in rather than re-derived, so the two halves of the score can
        never end up describing different files.

        The anchor half is opportunistic in three separate ways, and every one
        of them is a "leave the number out", never a failure: ranking can be
        switched off, ref.png -- the conditioning reference, whether the
        profile's style anchor or an image the user attached themselves -- is
        only there on a run that had one, and DINOv2 is an optional download.
        What is left is the composition score, which is free: the report was
        measured either way.

        The preference half (PickScore) is opportunistic the same way -- an
        optional download, gated on the same switch, every failure a missing
        number -- and is scored against the *composed* prompt, because "would
        a person pick this for what was asked" has to be asked about the text
        the sampler actually received, expansion and template included.
        """
        from .bench import metrics
        from .pipelines import rank

        report = params.get("reference_report")
        cosine = None
        anchor = image_path.parent / "ref.png"
        if self.config.rank_candidates and anchor.exists():
            try:
                if metrics.dino_available(self.config):
                    # CPU deliberately: this runs on the job queue beside a
                    # resident trellis and a resident SDXL pipe, and a metric
                    # must not take VRAM from the models making the asset.
                    cosine = metrics.reference_cosine(
                        anchor, image_path, self.config, device="cpu"
                    )
            except Exception:
                log.exception("anchor similarity failed; ranking on composition alone")
        preference = None
        if self.config.rank_candidates:
            try:
                if metrics.pickscore_available(self.config):
                    # CPU for the anchor's reason exactly.
                    preference = metrics.preference_score(
                        str(params.get("composed_prompt") or ""),
                        image_path,
                        self.config,
                        device="cpu",
                    )
            except Exception:
                log.exception("preference scoring failed; ranking without it")
        return rank.score(report, cosine, preference)

    async def _audit_mesh(
        self: Worker, job_id: str, glb_path: Path, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Measure how see-through the finished mesh is and record it.

        trellis-server's narrow-band remesh can emit a crust of disconnected
        plates that passes every integrity check while being visibly
        perforated, and the user currently finds that out in Blender. Measuring
        it here turns it into something they see on the job the moment it
        finishes. Runs after _apply_scale so it measures the mesh that will
        actually be downloaded; like scaling it is CPU-only and holds no GPU
        memory, so it sits outside the VRAM handoff.

        Returns the summary it stored, or None whenever it stored nothing --
        because it was cancelled, or because the measurement itself broke.
        The remesh loop in _generate reads that to decide whether the mesh is
        worth redoing, and None is deliberately not a bad verdict: no
        measurement means no retry, exactly as with the retry switched off.
        """
        if self._cancel is not None and self._cancel.event.is_set():
            return None
        self.progress.update(
            job_id,
            phase="audit",
            label="Checking mesh",
            inner=0.0,
            inner_next=1.0,
            nominal=6.0,
            detail="",
        )
        try:
            # Imported here for the same reason as postprocess above: numpy and
            # trimesh at module scope would land in app startup.
            from . import meshaudit

            report = await asyncio.to_thread(
                meshaudit.hole_fraction,
                glb_path,
                meshaudit.DEFAULT_VIEWS,
                meshaudit.REQUEST_PATH_RESOLUTION,
            )
        except Exception:
            # A diagnostic must never be able to fail a job whose mesh is
            # already on disk and fine. Log it and leave mesh_audit unset --
            # the UI renders no badge rather than a wrong one.
            log.exception("mesh audit failed for job %s", job_id)
            return None
        # Only the summary is stored: the per-view detail would ride along on
        # every row of the 100-job list for no one to read.
        params["mesh_audit"] = {
            "worst": report["worst"],
            "mean": report["mean"],
            "faces": report["faces"],
            "resolution": report["resolution"],
        }
        # The silhouette number stays exactly as it was -- it is the only thing
        # that catches trellis's disconnected-plate crust. The report adds what
        # the silhouette cannot see: topology, materials, budget, and whether
        # the thing will sit on an engine's floor.
        try:
            from . import meshreport

            params["mesh_report"] = await asyncio.to_thread(
                functools.partial(
                    meshreport.build,
                    glb_path,
                    target_size_m=params.get("size_m"),
                    silhouette=params["mesh_audit"],
                    triangle_budget=_effective_triangle_budget(params),
                )
            )
        except Exception as exc:
            log.exception("mesh report failed for job %s", job_id)
            # The measurements, not the mesh: the asset is fine and only the
            # description of it is missing. Recorded all the same, because an
            # inspector with no dimensions and no explanation reads as a bug in
            # the panel rather than as a step that did not run (ART-01).
            _note_degraded(
                params,
                "report",
                f"the mesh could not be measured ({exc}); size, triangle count "
                f"and watertightness are unknown for this asset",
            )
        await self._score_mesh(job_id, glb_path, params)
        log.info(
            "job %s mesh audit: worst %.3f, mean %.3f over %d faces",
            job_id, report["worst"], report["mean"], report["faces"],
        )
        await asyncio.to_thread(self.store.set_params, job_id, params)
        return params["mesh_audit"]

    async def _score_mesh(
        self: Worker, job_id: str, glb_path: Path, params: dict[str, Any]
    ) -> None:
        """Record ``params["mesh_score"]``: how well the mesh matches its own reference.

        A sort key for the candidate picker (``kernels.mesh.score``), never a verdict, and
        like every step in this file it can only ever record or stay silent: no reference
        image to compare against, or a measurement that broke, leaves the mesh unscored,
        and an unscored mesh sorts after every scored one. Derived (``DERIVED_PARAMS``),
        not a vector: it measures one run's artifact.
        """
        try:
            from PIL import Image

            from . import meshaudit
            from .kernels.mesh import score
            from .pipelines import reference

            job_dir = self.config.job_dir(job_id)
            # The approved matte when there is one, else the picture TRELLIS was given.
            source = next(
                (p for p in (job_dir / "cutout.png", job_dir / "reference.png") if p.is_file()),
                None,
            )
            if source is None:
                return

            def _measure() -> tuple[float | None, int | None]:
                with Image.open(source) as image:
                    target = reference.subject_mask(image)
                views = meshaudit.view_masks(glb_path, score.view_directions())
                return score.best_view_iou(views, target)

            iou, best_view = await asyncio.to_thread(_measure)
            def _record(key: str) -> dict[str, Any]:
                value = params.get(key)
                return value if isinstance(value, dict) else {}

            report, audit = _record("mesh_report"), _record("mesh_audit")
            lowpoly = _record("lowpoly")
            requested, achieved = lowpoly.get("requested"), lowpoly.get("achieved")
            ratio = (
                achieved / requested
                if isinstance(requested, int) and requested > 0 and isinstance(achieved, int)
                else None
            )
            params["mesh_score"] = score.build(
                iou=iou,
                best_view=best_view,
                holes=audit.get("worst"),
                hole_max=self.config.mesh_hole_max,
                components=report.get("welded_components", report.get("components")),
                nonmanifold=report.get("welded_nonmanifold_edges"),
                boundary=report.get("welded_boundary_edges"),
                budget_ratio=ratio,
                budget_miss=params.get("budget_miss"),
            )
        except Exception:
            log.exception("mesh score failed for job %s", job_id)
            params.pop("mesh_score", None)
