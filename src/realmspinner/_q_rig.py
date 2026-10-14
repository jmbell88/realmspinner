"""``Worker``'s Blender stages: rig and its deformation QA.

The jobs that run bpy out-of-process, plus the render-and-pack step the QA
sheet uses. Split out of ``queue.py`` for the reason every mixin here is:
the class had 35 methods and five unrelated subjects, and the loop core --
which is the part with the invariants -- was buried among them.

What these share, and why they are one module: none of them touches a resident
model or the VRAM handoff at all. They are CPU and an EEVEE render, and the
serial queue is the only thing keeping them from overlapping a trellis run.
They also all write into *another* job's directory -- a rig and its QA sheet
belong to the mesh they depict, not to the job that asked for them -- which is
what makes the staged-write and completion-marker rules in here load-bearing
rather than tidy.

``bpy`` never runs in this process: ``blender_run.run_worker`` spawns
``pipelines/blender_worker.py``, and ``_note_blender`` (which stays on
``Worker``) holds the live ``Popen`` so a cancel can kill it.

Queue.py module-level names are reached through ``queue_mod`` rather than
imported: a call-time lookup keeps ``monkeypatch.setattr(queue_mod, ...)``
landing, and a module-scope import of ``.queue`` here would be circular.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .kernels.rig import blender_spec, store
from .pipelines import blender_run, pose2d

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .queue import Worker

log = logging.getLogger(__name__)


class RigOps:
    """The Blender stages, mixed into :class:`~.queue.Worker`."""

    def _wants_landmarks(
        self: Worker, source_dir: Path, template: str, params: dict[str, Any]
    ) -> bool:
        """Whether reading joints off the reference image is worth attempting.

        Every one of these is a case where the answer could not be used, not a
        case where it might be poor -- how *good* a detection is is decided by
        pose2d's own sanity gates, which see the landmarks. Cheap checks first;
        ``available`` stats a directory and goes last because it is the only
        one that touches the disk outside the job's own.
        """
        return (
            template in pose2d.POSE_FIT_TEMPLATES
            and self.config.pose_fit
            # Adjust-joints. The user has already overruled one fit; running a
            # detector to produce a second one they would also overrule is pure
            # cost.
            and not params.get("bones")
            # An imported or hand-modelled mesh has no reference to read.
            and (source_dir / "input.png").exists()
            and pose2d.available(self.config)
        )

    async def _rig(self: Worker, job: dict[str, Any]) -> None:
        """Fit a skeleton to a finished job's mesh, out-of-process in Blender.

        A rig job is a job in its own right, but its artifacts land in the
        *source* job's directory alongside model.glb -- the rig belongs to the
        mesh, not to the request that produced it, and a UI that had to chase a
        second job id to find rig.glb would be the wrong shape. params carries
        source_job so the history list can attach it to the parent card.

        Nothing here touches the GPU or either resident model, so the VRAM
        handoff in _generate is not involved. The queue being serial is what
        keeps a multi-minute weighting solve from overlapping a trellis run.
        """
        job_id = job["id"]
        params = job["params"]
        source_id = str(params.get("source_job") or "")
        # Validated before it becomes a path.
        # This is the directory the Blender worker is pointed at and writes its
        # temps into, and job_dir("") is the assets root -- the whole reason
        # check_job_id and pose_path exist is to keep params off the filesystem
        # unchecked.
        if not store.is_valid_id(source_id):
            raise ValueError(f"source_job is not a job id: {source_id!r}")
        source_dir = self.config.job_dir(source_id)
        template = str(params.get("template") or self.config.rig_template)

        self.progress.update(
            job_id, phase="rig", label="Starting Blender", inner=0.0,
            inner_next=0.05, nominal=12.0, detail="",
        )

        def on_progress(frac: float, label: str) -> None:
            # The worker's fractions are already the whole bar (PHASES_RIG),
            # and it emits them from its own thread-free subprocess; this
            # runs on the to_thread worker, which ProgressBus locks for.
            self.progress.update(
                job_id, phase="rig", label=label, inner=frac,
                inner_next=min(frac + 0.1, 1.0), nominal=30.0, detail="",
            )

        # Blocking (a PIL decode, a silhouette measurement and a torch forward
        # on the CPU), so it goes through to_thread exactly as run_worker below
        # does. Skipped entirely -- not merely ignored -- when any gate is
        # closed, because the point of the gates is that the model is never
        # loaded on a job that could not use its answer.
        from . import queue as queue_mod

        landmarks, fit = None, None
        if self._wants_landmarks(source_dir, template, params):
            landmarks, fit = await asyncio.to_thread(
                queue_mod._landmark_bones, self.config, source_dir, template
            )

        spec = blender_spec.rig_spec(
            source_dir,
            template,
            params.get("bones"),
            template_bones=landmarks,
            fit=fit,
            # A custom skeleton (service.rig.rig_with_skeleton) carries its own
            # structure across the pipe -- see rig_spec's own docstring. Absent
            # for every other rig job (an ordinary rig, or adjust_joints's
            # template-shaped correction), which keeps their specs exactly
            # what they always were.
            skeleton=params.get("skeleton"),
            root=params.get("root"),
            mirror_pairs=params.get("mirror_pairs"),
        )
        try:
            result = await asyncio.to_thread(
                functools.partial(
                    blender_run.run_worker,
                    spec,
                    on_progress=on_progress,
                    on_start=self._note_blender,
                    timeout=self.config.rig_timeout,
                )
            )
            # A cancel that landed after the solve finished must not publish
            # the artifacts of a job about to be recorded as cancelled -- the
            # finally below throws the temps away instead.
            if self._cancel is not None and self._cancel.event.is_set():
                # Nothing was published, so nothing gets recorded either: the
                # weighting/bone_count of a discarded rig must not end up in
                # the params of a job recorded as cancelled.
                return
            await asyncio.to_thread(self._finalize_rig_locked, source_id, source_dir)
            # Published onto the served rig.glb/rig.json. From here a cancel
            # cannot take the artifact back -- ``_discard_artifacts`` removes
            # only the temps, because a cancelled *re*-rig must not destroy an
            # earlier successful one -- so the row must not say "cancelled"
            # either. It would leave a real rig under a row claiming it never
            # happened, and the follow-ups are gated on "done", so a
            # cancel during the QA tail below would silently stall whatever
            # chain waited on a perfectly good rig nobody pointed at.
            if self._cancel is not None:
                self._cancel.commit()
        finally:
            # No-op on success (finalize renamed them away); on failure or
            # cancel it removes the half-written temps and never touches the
            # served rig.glb/rig.json, which may belong to an earlier,
            # successful rig job.
            await asyncio.to_thread(store.discard_rig_temps, source_dir)
        # Recorded on the rig job so the history row can say "envelope
        # weights" without the UI having to fetch rig.json for every card.
        params["weighting"] = result.get("weighting")
        # Beside it rather than only in rig.json: "envelope" is a degraded
        # outcome and the question it raises is immediately "why", which the
        # inspector cannot answer without opening a file per card.
        params["weighting_reason"] = result.get("weighting_reason")
        # How many joints the fitted skeleton ended up with (an int -- the
        # worker returns len(bones)), for the same reason as the two above: it
        # is in rig.json, and a card that wants to say "17 bones" should not
        # have to open a file to do it. None only from a worker whose result
        # predates the key.
        params["bone_count"] = result.get("bones")
        await asyncio.to_thread(self.store.set_params, job_id, params)
        log.info(
            "rigged job %s from %s: %s weights, %s bones",
            job_id, source_id, result.get("weighting"), result.get("bones"),
        )

        # Log-and-swallow, the ``_audit_mesh`` rule: the rig is already
        # published, and a review sheet that could not be rendered must not
        # fail the job whose artifact is on disk. Recorded with merge_params
        # rather than a second set_params off the copy above, because the two
        # writes are a read-modify-write sequence the lock does not cover.
        try:
            qa = await self._deform_qa(job_id, source_id, source_dir, template)
        except Exception:
            log.exception("deformation QA sheet failed for job %s", job_id)
            qa = None
        if qa is not None:
            await asyncio.to_thread(self.store.merge_params, job_id, {"deform_qa": qa})

    def _finalize_rig_locked(self: Worker, source_id: str, source_dir: Path) -> None:
        """Publish the freshly baked rig.glb/rig.json, with every existing
        pose's bake frozen for the whole delete-then-rename window.

        The 2026-09-26 audit, finding poser-rig-07: ``store.finalize_rig``
        deletes every stale ``poses/*.glb`` up front and then spends up to 5s
        retrying the rig.glb/rig.json rename against a lock a Windows
        antivirus or indexer can hold that long (``store.py``'s own
        docstring). ``service.rig.posed_model`` bakes a pose from whatever
        rig.glb is on disk *right now*, guarded by nothing but that one pose
        id's own ``pose:<id>`` lock. A bake landing in that window reads the
        *old* rig -- finalize has not renamed the new one in yet -- and writes
        straight back to ``poses/<id>.glb``, the very file finalize just
        cleared, so it is never invalidated again and is served under the new
        rig's authority forever.

        Closed by taking every pose id's own lock -- the same
        ``self.artifact_lock(job_id, f"pose:{pose_id}")`` spelling
        ``posed_model`` uses as ``svc.convert_lock(job_id, f"pose:{pose_id}")``
        (``studio.runtime`` wires ``worker.artifact_lock = svc.convert_lock``,
        so the two spellings name one lock) -- across the whole delete-then-
        rename call, rather than touching ``finalize_rig``'s own
        delete-before-rename order, which its docstring says is deliberate: a
        crash there must cost a rebake under the *old* rig, never serve a
        stale one under the new rig.json's authority.
        """
        pose_ids = [str(p.get("id")) for p in store.list_poses(source_dir) if p.get("id")]
        with contextlib.ExitStack() as poses_locked:
            for pose_id in pose_ids:
                poses_locked.enter_context(self.artifact_lock(source_id, f"pose:{pose_id}"))
            store.finalize_rig(source_dir)

    async def _deform_qa(
        self: Worker, job_id: str, source_id: str, source_dir: Path, template: str
    ) -> dict[str, Any] | None:
        """Render the deformation battery for a freshly rigged mesh, or None.

        Human-reviewable and nothing more: this scores nothing and gates
        nothing. It exists because a weighting method that *reports* success
        says only that the solve produced numbers, and the way to see whether
        those numbers deform the mesh sensibly is to look at it bent.

        The poses are template data (``poses.deform_battery``) and the render
        is the ordinary sheet path -- ``sheetlib.plan``/``pack``/``sidecar``
        around ``op_sheet`` -- because a second renderer is a second set of
        camera conventions to keep in agreement with the first. The output is
        not a sheet in ``sheets/``: it belongs to the rig, and is deleted with
        it.
        """
        from . import queue as queue_mod
        from .kernels import sheet as sheetlib
        from .kernels.rig import deform_metrics
        from .kernels.rig import poses as rig_poses

        poses = rig_poses.deform_battery(template)
        rig_glb = source_dir / "rig.glb"
        if not poses or not self.config.deform_qa or not rig_glb.exists():
            return None
        # The rig is published and the row will say "done" either way; what a
        # cancel buys here is not paying for the QA battery. Checked at the top
        # and again after the render, which are the two places there is
        # anything left to skip -- every sibling stage checks the same way.
        if self._cancel is not None and self._cancel.event.is_set():
            return None

        layout = sheetlib.plan(
            poses,
            frame_size=queue_mod.DEFORM_QA_FRAME_SIZE,
            elevation=sheetlib.DEFAULT_ELEVATION,
            # Lit, where a sprite sheet defaults to flat: a collapsed elbow or
            # a candy-wrapper twist is a shading artefact, and unlit albedo is
            # exactly the render that hides it.
            lighting="lit",
            yaws=queue_mod.DEFORM_QA_YAWS,
        )
        bones = {(p["id"], 0): p["bones"] for p in poses}
        # Per pose: the battery is authored as deltas
        # from rest (see templates/deform_qa/humanoid.json's ``space``), and
        # dropping it here left every cell rendering in the pose editor's
        # ``node`` frame instead -- broken on a ``measured`` rig, the
        # 2026-09-16 QA sheet finding.
        spaces = {(p["id"], 0): p["space"] for p in poses if p.get("space")}
        cells = [
            {
                "index": c.index,
                "yaw": c.yaw,
                "pose": c.pose,
                "frame": c.frame,
                "bones": bones.get((c.pose, c.frame)) or {},
                **(
                    {"pose_space": space}
                    if (space := spaces.get((c.pose, c.frame)))
                    else {}
                ),
            }
            for c in layout.cells
        ]

        def on_progress(frac: float, label: str) -> None:
            self.progress.update(
                job_id, phase="rig", label=f"Deformation QA: {label}", inner=frac,
                inner_next=min(frac + 0.1, 1.0), nominal=20.0, detail="",
            )

        png = store.rig_qa_png_path(source_dir)
        # Both halves of the QA sheet are staged and renamed in. The sidecar is
        # the completion marker (files.ready keys on the .json, not the .png),
        # so on a *re-rig* the previous run's sidecar already says "ready" while
        # this run's atlas is being packed over the served PNG -- a reader in
        # that window gets a torn sheet that nothing marks as suspect. Renaming
        # both keeps the marker's promise true at every instant.
        png_tmp = png.with_name(f".{png.name}.tmp")
        qa_json = store.rig_qa_path(source_dir)
        json_tmp = qa_json.with_name(f".{qa_json.name}.tmp")
        try:
            result, _trims = await self._render_sheet_atlas(
                rig_glb,
                layout,
                cells,
                prefix="realmspinner-rigqa-",
                on_progress=on_progress,
                pack_target=png_tmp,
                deform_metrics=True,
            )
            # The second of the two checks the comment above already promised:
            # the 2026-09-12 audit (docs-10) found this docstring claimed a
            # cancel is caught "again after the render" when the body never
            # made a second check, so a cancel landing mid-render still
            # published the QA sheet and its sidecar. There is exactly as much
            # left to skip here as there was for the rig's own render a few
            # lines up in ``_rig``: png_tmp already holds the packed atlas, so
            # skipping the publish below leaves nothing but a temp for the
            # ``finally`` to discard.
            if self._cancel is not None and self._cancel.event.is_set():
                return None
            # Not the job's completion marker, so no commit here (classified
            # 2026-09-15): ``_rig`` commits right after ``finalize_rig``,
            # before it calls this, so a cancel landing now only skips the QA
            # sheet and cannot un-commit the row.
            # test_job_durability.py pins that ordering.
            await asyncio.to_thread(os.replace, png_tmp, png)
        finally:
            with contextlib.suppress(OSError):
                png_tmp.unlink(missing_ok=True)

        pivot = result.get("pivot") if isinstance(result, dict) else None
        meta = sheetlib.sidecar(
            layout,
            sheet_id="rig_qa",
            source_job=source_id,
            image=png.name,
            created=time.time(),
            name="deformation QA",
            pivot=(float(pivot[0]), float(pivot[1])) if pivot else None,
        )
        # The numbers beside the picture, keyed by the pose's name. Advisory and
        # uncalibrated (``kernels.rig.deform_metrics``), so they ride in the
        # sidecar as data and nothing reads a verdict out of them. A pose the
        # worker could not score is simply absent, and so is the whole key when
        # none could be -- the sheet is published either way.
        scored = result.get("deform_metrics") if isinstance(result, dict) else None
        names = {str(p["id"]): str(p["name"]) for p in poses}
        by_name = {
            names[pose_id]: scores
            for pose_id, scores in (scored or {}).items()
            if scores and pose_id in names
        }
        if by_name:
            meta["deform_metrics"] = {
                "poses": by_name,
                "limits": {
                    "stretch": deform_metrics.STRETCH_LIMIT,
                    "collapse": deform_metrics.COLLAPSE_LIMIT,
                },
            }
        # Written last, so it is the completion marker the file rules key on --
        # the same ordering rig.json has for rig.glb. The QA sheet's marker,
        # not the rig job's, which ``_rig`` has already committed.
        try:
            await asyncio.to_thread(
                json_tmp.write_text,
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )
            await asyncio.to_thread(os.replace, json_tmp, qa_json)
        finally:
            with contextlib.suppress(OSError):
                json_tmp.unlink(missing_ok=True)
        log.info(
            "rendered deformation QA for job %s: %d pose(s) x %d view(s)",
            source_id, layout.rows, layout.columns,
        )
        return {"poses": [str(p["name"]) for p in poses], "cells": len(cells)}

    async def _render_sheet_atlas(
        self: Worker,
        glb: Path,
        layout: Any,
        cells: list[dict[str, Any]],
        *,
        prefix: str,
        on_progress: Any,
        pack_target: Path,
        deform_metrics: bool = False,
    ) -> tuple[Any, Any]:
        """Render one cell per frame into a scratch directory and pack them.

        A temporary directory, ``blender_spec.sheet_spec``, one ``run_worker``
        under the sheet timeout, and ``sheetlib.pack`` over the frames it wrote.
        Blender only ever renders one square transparent frame per cell into a
        scratch directory that goes away either way; the grid and the packing
        are pure host-side code.

        ``pack_target`` is a staging name: the sidecar is the completion
        marker, so a re-run's pack would otherwise spend a minute writing over a
        PNG a *previous* run's sidecar still calls ready.
        """
        from .kernels import sheet as sheetlib

        with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
            frames_dir = Path(tmp)
            spec = blender_spec.sheet_spec(
                glb,
                frames_dir,
                cells,
                frame_size=layout.frame_size,
                elevation=layout.elevation,
                lighting=layout.lighting,
                deform_metrics=deform_metrics,
            )
            result = await asyncio.to_thread(
                functools.partial(
                    blender_run.run_worker,
                    spec,
                    on_progress=on_progress,
                    on_start=self._note_blender,
                    timeout=self.config.sheet_timeout,
                )
            )
            frames = {c.index: frames_dir / f"{c.index:04d}.png" for c in layout.cells}
            trims = await asyncio.to_thread(sheetlib.pack, layout, frames, pack_target)
        return result, trims
