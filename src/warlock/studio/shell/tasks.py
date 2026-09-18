"""The task pump: landing a finished ``ctx.tasks`` result on the document or
viewport it belongs to.

A **mixin on** :class:`~.app.App`, ``clay_viewport.ClayViewport``'s idiom
restated at shell scale -- ``self`` here is the App and every method's body is
unchanged from the line it stood on in ``studio/main.py`` before the P4
restructure split that module into ``shell/{app,frame,events,tasks,quit}.py``.

``_compare_key`` and ``_import_mesh_key`` travel here rather than to
``events.py``, where the split plan named them: neither is read by an input
router, only by :meth:`TasksMixin._on_task_done`'s own landing dispatch two
lines below each definition, so this is the file whose code the plan
disagreed with.

The library-card thumbnail capture, ``_capture_thumbnail_from``, is here
rather than on ``clay_viewport.ClayViewport`` despite drawing from Clay's own
viewport in the common case: ``_on_task_done`` calls it directly for Mason's
export too (``self._capture_thumbnail_from(done.result["job_id"],
self.mason_view)``), so the function is shared shell plumbing between two
mode mixins, not Clay's alone. ``_capture_clay_thumbnail``, the one-argument
convenience that always draws from ``self.clay_view``, stayed with Clay in
``studio/modes/clay/ui/viewport.py`` instead, and calls this one across the mixin boundary --
which works precisely because both are methods of the one assembled
:class:`~.app.App`.

The shell names this module reaches are imported *inside* the methods that use
them, ``studio/modes/clay/ui/viewport.py``'s own rule restated: ``main`` imports
:class:`~.app.App` (which assembles this mixin) to build the class, so a
module-scope import back would be a cycle -- and not only a stylistic one
here: ``main.py`` imports :mod:`shell.app` at its own module top so that
``main.App`` is a real, constructible attribute (``_run_locked`` builds one),
which means this module is on the import chain *before* ``main.py`` has
finished defining its own constants. A module-scope ``from ..main import
VIEWER_KEY`` failed with exactly that ``ImportError`` the first time this file
was written; every task-key constant below is read with a local import inside
the method that switches on it. ``app_ctx``/``jobs_cache`` are the one
exception, imported at module scope the way ``main.py`` always imported them:
they are plain state modules that do not import ``main`` back, so nothing
about them can cycle.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .. import app_ctx as app_ctx_mod
from .. import jobs_cache as jobs_cache_mod

log = logging.getLogger(__name__)


def _compare_key() -> str:
    """``library.COMPARE_KEY``, looked up lazily.

    A function rather than a module-level import: ``panes.library`` imports a
    great deal of the app and every other reference to it in this file is
    already deferred to its call site for that reason.
    """
    from ..panes import library

    return library.COMPARE_KEY


def _import_mesh_key() -> str:
    """``library.IMPORT_MESH_KEY``, looked up lazily -- :func:`_compare_key`'s
    reason, and the same shape so the two read as one convention."""
    from ..panes import library

    return library.IMPORT_MESH_KEY


class TasksMixin:
    """Landing a finished task, mixed into :class:`~.app.App`.

    The shell names it reaches are imported *inside* the methods that use
    them: ``main`` imports :class:`~.app.App` (which assembles this mixin) to
    build the class, so a module-scope import back would be a cycle. Same
    shape as ``clay_viewport.ClayViewport``.
    """

    def _collect_tasks(self) -> None:
        from ..main import REVIEW_MESH_KEY, VIEWER_KEY

        ctx = self.app_ctx
        for done in ctx.tasks.poll():
            if not done.ok:
                from ..app_ctx import UPDATE_CHECK_KEY

                if done.key == UPDATE_CHECK_KEY and done.tag == "auto":
                    # The one task in the app the user did not ask for that can
                    # fail for a reason that is none of their business. A
                    # background check on a flaky connection has nothing to
                    # report and nothing to retry, so it says nothing; the
                    # Settings button submits the same call with tag "manual"
                    # and its failure is toasted like every other refusal.
                    continue
                # The refusal's *address*, where it has one (UX.md Phase 3).
                # ``ServiceError.field`` has been carried since the class was
                # written and read by nothing, so a refusal about the seed and
                # one about the style LoRA arrived as the same red toast in the
                # corner. Recorded here rather than in each pane because this is
                # the one place every task failure passes through -- and the
                # toast still goes up either way: the ring says *which control*,
                # not *that something happened*, and a pane the user has since
                # navigated away from can draw no ring at all.
                named = getattr(done.error, "field", None)
                if isinstance(named, str):
                    # ``rows`` is the refusal's other half: which registry rows
                    # would fix it. Written since the class was, and until now
                    # read by nothing outside the tests -- so "you haven't got
                    # these weights" arrived as a sentence with no action.
                    rows = tuple(getattr(done.error, "rows", ()) or ())
                    gib = 0.0
                    if rows:
                        try:
                            from ...service import downloads as svc_downloads

                            gib = svc_downloads.needed_gib(ctx.svc, list(rows))
                        except Exception:
                            # Path arithmetic over ~17 registry entries, but a
                            # figure is a courtesy: an unknown row must not
                            # cost the user the button as well as the number.
                            log.exception("could not size a refusal's install")
                    # ``ServiceError.packs`` is ``rows``' sibling for "the code
                    # for this is not installed" (F4's job-door half). No gib
                    # to size: a pack has no ``downloads.needed_gib`` entry,
                    # only Settings -> Packs' own figure, which the pane reads
                    # off ``ctx.pack_rows`` when it draws the button.
                    packs = tuple(getattr(done.error, "packs", ()) or ())
                    ctx.state.note_field_error(named, done.message or "", rows, gib, packs)
                elif done.key == "submit":
                    # A refusal with no control to point at -- the VRAM door is
                    # the one of these, by its own recorded argument. It is
                    # kept so the plan block can say it, because a toast cannot
                    # hold ``vram.shortfall_message``'s list of remedies and
                    # the block was going on saying "Ready to generate."
                    ctx.state.create.submit_refusal = done.message or ""
                message = done.message or "That did not work."
                action = done.action
                if done.key.startswith("journal:"):
                    # The eighth prefix, and the one nobody had claimed. A
                    # journal write is the only task in the app the user did
                    # not start, and since the mark stopped advancing on
                    # *submit* (``journal.write``) its failure is also the only
                    # signal that the crash copy the app promised does not
                    # exist. "Something went wrong; see the log for details"
                    # names neither half of that, and it is the sentence that
                    # was being shown.
                    #
                    # No routing call beside it: the journal has no per-mode
                    # state to unlock, and the retry is already the debounce's
                    # -- ``_write_if_due`` comes back to this slot in
                    # JOURNAL_SECONDS whether or not anything is told here.
                    message = (
                        "Autosave could not write a recovery copy. Save your "
                        "work somewhere you choose."
                    )
                    action = "log"
                ctx.toast(message, "error", action)
                # A failed save must not leave the document locked: saving
                # disables every editing control, so without this one bad
                # write makes the tab read-only until it is closed. Each
                # editor claims its own key prefix.
                if done.key.startswith("inker-"):
                    from .. import inker_mode

                    inker_mode.on_task_failed(ctx, done)
                elif done.key.startswith("clay-"):
                    from ..modes.clay import mode as clay_mode

                    clay_mode.on_task_failed(ctx, done)
                elif done.key.startswith("plotter-"):
                    from .. import plotter_mode

                    plotter_mode.on_task_failed(ctx, done)
                elif done.key.startswith("packwright-"):
                    from .. import packwright_mode

                    # Same rule, plus one of its own: a failed *pack* has
                    # to clear ``packing`` and record why, or the items
                    # pane shows an empty list that reads as success.
                    packwright_mode.on_task_failed(ctx, done)
                elif done.key.startswith("sirens-"):
                    from .. import sirens_mode

                    # Same rule, plus one of its own: a failed *render* has to
                    # clear ``rendering`` and record why, or the transport
                    # shows a dead Play button with nothing beside it.
                    sirens_mode.on_task_failed(ctx, done)
                elif done.key.startswith("mason-"):
                    from .. import mason_mode

                    mason_mode.on_task_failed(ctx, done)
                elif done.key.startswith("muse-"):
                    from .. import muse_mode

                    # muse-03 (2026-09-07 audit): this chain had no branch for
                    # Muse at all, so a failed loop search left ``finding``
                    # set from ``find_loops`` -- the only place that turns it
                    # on -- and the strip spun forever, since only a
                    # *successful* ``on_task_done`` ever turned it back off.
                    muse_mode.on_task_failed(ctx, done)
                elif done.key.startswith("troupe-"):
                    from .. import troupe_mode

                    # Both of Troupe's tasks are *doors*, so a failure here is
                    # always a refusal with a sentence in it -- and one the
                    # user is owed, since neither door's button can know in
                    # advance which of its options the service will object to.
                    troupe_mode.on_task_failed(ctx, done)
                elif done.key.startswith(("download:", "remove:")):
                    # A failed fetch has to be *routed* somewhere, not merely
                    # toasted: the rows carry a presence flag, and a fetch that
                    # got partway before failing has changed what is on disk.
                    # Re-probing costs a few stats and is the only thing that
                    # stops the pane staying optimistic about a download that
                    # did not happen. (Only the rows, not doctor's whole
                    # suite -- nothing succeeded, so the health state has
                    # nothing new to say.)
                    #
                    # A failed *removal* is the same fact from the other side,
                    # and more sharply so: ``uninstall`` renames a directory
                    # out of the way before it deletes it, so a failure part
                    # way through has already made the model absent.
                    self._refresh_model_answers()
                elif done.key.startswith("pack:"):
                    # shell-07 (2026-09-11 audit): ``_resume_deferred_quit``
                    # used to be called only from ``_on_task_done``'s
                    # success-only "pack:" branch, and a failed task is routed
                    # away from there entirely -- so a quit ``_ask_quit``
                    # deferred for a pack's commit phase (disk full, a locked
                    # file, a network hiccup mid-write) was never resumed once
                    # that install then failed, even though the toast it
                    # showed promised "Quitting once it finishes."
                    self._resume_deferred_quit()
                elif done.key == REVIEW_MESH_KEY:
                    self._adopt_review_model(done)
                elif done.key == VIEWER_KEY:
                    # ``pending`` still names the file, so nothing would retry
                    # it; that is the intent (the parse would fail again), but
                    # the flag has to come down or the next *different* asset
                    # is refused as a duplicate of this one.
                    if self.viewer.pending == done.tag:
                        self.viewer.pending = None
                        self.viewer.clear()
                        self.viewer.path = done.tag
                elif done.key.startswith("review-"):
                    from .. import review_mode

                    # Same rule: ``scanning`` gates every button and key, so a
                    # failed scan that left it set would make the mode inert.
                    review_mode.on_task_failed(ctx, done)
                elif done.key.startswith("poser-"):
                    from .. import poser_mode

                    # Same rule again: ``loading``/``building`` gate the pane
                    # and the viewport's progress row.
                    poser_mode.on_task_failed(ctx, done)
                elif done.key.startswith("matte-"):
                    from .. import matte_preview

                    # The seventh, and it was missing. ``matte_preview.pump``
                    # re-submits whenever there is no cached cutout for the
                    # current stamp, and ``settings_3d.matte_modal`` runs it
                    # every frame -- so a failure that left no note behind was
                    # re-submitted, re-failed and re-toasted at the frame rate.
                    matte_preview.on_task_failed(ctx, done)
                continue
            try:
                self._on_task_done(done)
            except Exception:
                # A handler that raises here used to end the session: this is
                # the frame loop's own thread, and the outer ``try`` in
                # ``App.run`` treats any escaped exception as the app dying,
                # not one task's landing going wrong (finding #1). Logged and
                # toasted the way ``guard`` announces a tripped pane -- the
                # task itself already finished, so there is nothing left to
                # retry, only the fact that its landing broke to report.
                log.exception("landing task %r raised", done.key)
                self.app_ctx.toast(f"That did not finish landing: {done.key}.", "error", "log")

    def _on_task_done(self, done: Any) -> None:
        from ..main import (
            CHARACTER_PREVIEW_LOAD_KEY,
            REVIEW_MESH_KEY,
            SILENT_TASK_KEYS,
            VERIFY_KEY,
            VIEWER_KEY,
        )

        ctx = self.app_ctx
        key = done.key
        if key == "preview" and isinstance(done.result, dict):
            ctx.state.preview.update(done.result)
            return
        if key == "health":
            # The status surfaces read runtime.checks each frame; replacing the
            # list wholesale is atomic enough for all of them.
            if isinstance(done.result, list):
                self.runtime.checks = done.result
                # The rows that were "still checking" at startup have their
                # answer now: a newly fatal one (no CUDA) joins the banner, and
                # the first-run panel's verdicts are retaken from the same list.
                self._report_failed_checks()
                if getattr(ctx, "first_run", False):
                    from ..panes import first_run

                    ctx.first_run_info = first_run.snapshot(ctx)
                # The first poll is also what pays for the deferred bpy probe
                # (C30). If it says rigging works and the ctx does not yet,
                # re-ask for the templates -- the probe's answer is cached, so
                # the re-ask costs a directory read.
                blender_ok = any(c.name == "Blender (rigging)" and c.ok for c in done.result)
                if blender_ok and not ctx.rigging_available:
                    from ...service import rig as svc_rig

                    ctx.submit("rig-templates", svc_rig.rig_templates, self.svc)
            return
        if key == "model-storage":
            if isinstance(done.result, dict):
                ctx.model_storage = done.result
            return
        if key == "evidence-storage":
            if isinstance(done.result, dict):
                ctx.evidence_storage = done.result
            return
        if key == app_ctx_mod.UPDATE_CHECK_KEY:
            if isinstance(done.result, dict):
                # Onto the state rather than left in the task's progress: the
                # opt-in startup check lands while Settings is almost certainly
                # closed, and the pane has to be able to draw the answer
                # whenever it is next opened.
                ctx.state.update_check = done.result
                if done.tag == "auto" and done.result.get("available"):
                    ctx.toast(
                        f"Warlock {done.result['latest']} is available -- "
                        f"see Settings -> Updates."
                    )
            return
        if key == app_ctx_mod.UPDATE_DOWNLOAD_KEY:
            # Claimed rather than left to the unclaimed-key log, because the
            # pane below it draws from ``staged_installer`` -- a question about
            # the filesystem -- and the toast is the only thing that says the
            # long download is over to a user who has since navigated away.
            if isinstance(done.result, dict) and done.result.get("path"):
                ctx.toast(
                    "The update installer has been downloaded and verified. "
                    "Settings -> Updates has the button that runs it.",
                    "success",
                )
            return
        if key == "library-verify":
            self._report_library_check(done.result)
            return
        if key == "trash-size":
            # Adopts ``library.measure_trash``'s reading on the frame thread
            # it was answered for -- shell-05 in the 2026-09-13 audit found
            # the task itself writing this slot, the same T3 hazard
            # ``jobs_cache.adopt_storage`` exists to avoid. ``done.tag`` is
            # the trash's job-id tuple at submit time, so an answer for a
            # trash that has since moved on (a restore, another empty) is
            # dropped instead of shown against the wrong list.
            if isinstance(done.result, dict):
                from ..panes import library

                ctx.state.preview[library.TRASH_SIZE_SLOT] = (done.tag, done.result)
            return
        if key == "library-backup":
            out = done.result if isinstance(done.result, dict) else {}
            if out:
                from ..state import format_bytes

                ctx.toast(
                    f"Library index backed up to {out['dir']} "
                    f"({format_bytes(int(out['store_bytes']))}).",
                    "success",
                )
            return
        if key == "sweep-staging":
            # Silent when there was nothing to reclaim, which is every launch
            # that did not follow a cancelled fetch. Said out loud when there
            # was: disk quietly reappearing is the kind of thing a user should
            # be told about rather than discover in a folder listing.
            removed = done.result if isinstance(done.result, list) else []
            if removed:
                noun = "tree" if len(removed) == 1 else "trees"
                ctx.toast(f"Reclaimed {len(removed)} staging {noun} left by a cancelled download.")
            return
        if key == "rig-templates":
            templates = done.result if isinstance(done.result, dict) else {}
            ctx.rigging_available = bool(templates.get("available"))
            ctx.rig_templates = list(templates.get("templates") or [])
            ctx.rig_default = templates.get("default") or ctx.rig_default
            return
        # The side data the pose and sheet panels read. Keyed by job so a
        # result that arrives after the selection moved on can be dropped
        # rather than shown against the wrong asset.
        if key.startswith(("poses:", "sheets:", "presets:")):
            name, _, job_id = key.partition(":")
            if name != "presets" and job_id != ctx.state.selected:
                return
            if name == "poses" and isinstance(done.result, dict):
                ctx.state.preview["poses"] = done.result.get("poses") or []
                ctx.state.preview["bones"] = done.result.get("bones") or []
            elif name == "sheets" and isinstance(done.result, dict):
                ctx.state.preview["sheets"] = done.result.get("sheets") or []
            elif name == "presets" and isinstance(done.result, dict):
                ctx.state.preview["presets"] = done.result.get("poses") or []
            return
        if key.startswith("sprite:") and isinstance(done.result, dict):
            # Seeded from the create result so the panel can show *this* job's
            # bar. Keyed by the source reference, because the panel is drawn
            # against that row and not against the synthesis job.
            ctx.state.preview["sprite_active"] = dict(done.result)
            return
        if key.startswith("sprite-del:"):
            # The listing is stamped on the directory's mtime, so the delete
            # shows up on its own -- but the cached textures are keyed by draft
            # id and would otherwise outlive the files they decoded.
            ctx.cache.invalidate()
            return
        if key.startswith("pack:"):
            # What a finished install *means*, decided rather than left to the
            # user to discover: the same wholesale re-probe a finished download
            # gets (``force=True``, off the frame thread), because the child
            # wrote into the site-packages this process is running out of and
            # every model, rigging and mode answer in the ctx was derived from
            # a probe taken before it did.
            #
            # The re-probe is not always enough, and this says so rather than
            # leaving a mode grey after an install that reported success:
            # ``service.packs`` invalidates the import caches, so a pack whose
            # modules still do not resolve here is one that genuinely needs a
            # restart -- and that is a sentence, not a silence.
            from ...service import packs as svc_packs
            from ...service import system as svc_system

            ctx.submit(VERIFY_KEY, svc_system.current_checks, self.svc, force=True)
            ctx.tasks.set_progress(VERIFY_KEY, 0.0, "Verifying installation...")
            # Comma-joined for Restore packs (M02, ``app_settings._restore_packs``),
            # which installs several packs under one task key so the pane's
            # one-install-at-a-time rule still holds; a single key here is the
            # same list of one.
            keys = [name for name in key.partition(":")[2].split(",") if name]
            pack = ", ".join(keys)
            try:
                stubborn = svc_packs.unresolved(keys)
            except Exception:  # noqa: BLE001 -- a toast must not end the frame
                log.exception("could not re-probe the %r pack(s)", pack)
                stubborn = []
            if stubborn:
                ctx.toast(
                    f"The {pack} pack installed, but Warlock has to restart "
                    "before it can use it.",
                    "warn",
                )
            else:
                ctx.toast(f"The {pack} pack is installed.", "success")
            self._resume_deferred_quit()
            return
        if key.startswith(("download:", "remove:")):
            # Re-probe wholesale, exactly as the "health" task above replaces
            # runtime.checks: the fetch wrote files doctor has never looked at,
            # and every model answer in the ctx is derived from that list. A
            # removal is the same wholesale change with the sign flipped, so it
            # takes the same body rather than a second one that could drift.
            from ...service import system as svc_system

            # Off the frame thread. ``force=True`` re-runs *every* probe,
            # including the slow ones the startup path deliberately defers --
            # the torch import and the bpy subprocess, which is seconds of
            # frozen window on the frame that is supposed to say "Download
            # finished" (UX-09). The model answers below are refreshed when the
            # probe lands, so the pane catches up a moment later instead of the
            # whole app stopping for it.
            ctx.submit(VERIFY_KEY, svc_system.current_checks, self.svc, force=True)
            ctx.tasks.set_progress(VERIFY_KEY, 0.0, "Verifying installation...")
            # The untick happens when the probe lands (see ``VERIFY_KEY``
            # above), because the rows it reads are derived from the checks it
            # is still computing. Unticking against the *old* answers would
            # leave every row exactly as it was: the plan is deduped, so
            # fetching one SDXL 1.0 recipe satisfies the other three, and
            # leaving them ticked offers to download 7 GB that is already there.
            ctx.toast("Model removed." if key.startswith("remove:") else "Download finished.")
            return
        if key == "upload" and done.result is not None:
            from ..modes.create.ui.panes import settings_3d

            settings_3d.upload(ctx, Path(done.result))
            return
        if key == "ref-upload" and done.result is not None:
            # Only the path is kept here; the bytes are read in the submit
            # task, so picking a 20 MB image never touches the frame thread.
            ctx.state.form_2d["ref_path"] = str(done.result)
            return
        if key.startswith("journal:"):
            # The mark, on the frame thread. A copy that landed changes nothing
            # on screen, which is why this key was in ``SILENT_TASK_KEYS`` until
            # the review's T3 moved the three slot attributes off the task
            # thread -- the write still says nothing, but the slot has to be
            # told here, beside ``drop``, rather than from the pool.
            from .. import journal

            journal.on_task_done(ctx, done)
            return
        if key.startswith("familiar/"):
            from ..assistant import ui as familiar_ui

            familiar_ui.on_task_done(ctx, done)
            return
        if key.startswith("clay-"):
            from ..modes.clay import mode as clay_mode

            clay_mode.on_task_done(ctx, done)
            if isinstance(done.result, dict) and done.result.get("exported"):
                # The card appears in the library like any other asset, so it
                # needs the thumbnail every other asset gets -- and that is an
                # offscreen GL draw, which belongs on the frame thread rather
                # than in the task that minted the row.
                self._capture_clay_thumbnail(done.result["job_id"])
            return
        if key.startswith("mason-"):
            from .. import mason_mode

            # ``mason_mode.on_task_done`` gives ``mason_assets`` first
            # refusal itself: a ``mason-asset:`` key is a background parse,
            # never a document task, and that module claims the prefix.
            mason_mode.on_task_done(ctx, done)
            if isinstance(done.result, dict) and done.result.get("exported_asset"):
                # The card appears in the library like any other asset, so it
                # needs the thumbnail every other asset gets -- and that is an
                # offscreen GL draw, which belongs on the frame thread rather
                # than in the task that minted the row. From *Mason's* viewport,
                # not Clay's: the picture has to be of the scene that was
                # exported, and ``self.clay_view`` is either a different
                # document or (in a session that never opened Clay) None, which
                # would silently leave the card on its placeholder.
                self._capture_thumbnail_from(done.result["job_id"], self.mason_view)
            return
        if key.startswith("inker-"):
            from .. import inker_mode

            inker_mode.on_task_done(ctx, done)
            return
        if key.startswith("plotter-"):
            from .. import plotter_mode

            plotter_mode.on_task_done(ctx, done)
            if isinstance(done.result, dict) and done.result.get("exported_asset"):
                # The card appears in the library like any other asset, so
                # it needs the thumbnail every other asset gets -- and that
                # is an offscreen GL draw, which belongs on the frame thread
                # rather than in the task that minted the row.
                self._capture_clay_thumbnail(done.result["job_id"])
            return
        if key.startswith("packwright-"):
            from .. import packwright_mode

            packwright_mode.on_task_done(ctx, done)
            if isinstance(done.result, dict) and done.result.get("exported_asset"):
                self._capture_clay_thumbnail(done.result["job_id"])
            return
        if key.startswith("muse-"):
            from .. import muse_mode

            muse_mode.on_task_done(ctx, done)
            return
        if key.startswith("sirens-"):
            from .. import sirens_mode

            sirens_mode.on_task_done(ctx, done)
            return
        if key.startswith("troupe-"):
            from .. import troupe_mode

            troupe_mode.on_task_done(ctx, done)
            return
        if key.startswith("review-"):
            from .. import review_mode

            review_mode.on_task_done(ctx, done)
            return
        if key.startswith("poser-"):
            from .. import poser_mode

            poser_mode.on_task_done(ctx, done)
            return
        if key.startswith("matte-"):
            from .. import matte_preview

            matte_preview.on_task_done(ctx, done)
            return
        if key == "character-preview":
            self._dispatch_character_preview(done.result)
            return
        if key == CHARACTER_PREVIEW_LOAD_KEY:
            self._adopt_character_preview(done)
            return
        if key == "submit":
            # The press was taken, so whatever the last one was refused for is
            # no longer the state of things.
            ctx.state.create.submit_refusal = ""
            ctx.cache.invalidate()
            if isinstance(done.result, dict) and done.result.get("kind") == "character":
                self._landed_character(done.result)
                return
            # Say where in line it landed: five rapid submits used to produce
            # five identical "Queued." toasts and no sense of depth.
            # ``+ 1`` for the job this submit just created. ``invalidate()``
            # above only marks the cache dirty -- it does not reread -- so
            # ``cache.jobs`` here is still the page from *before* this submit,
            # and the count was short by exactly one every time: the first
            # submit of an idle queue counted 0 and said "Queued." while two
            # jobs were in line (UX-25).
            waiting = sum(1 for j in ctx.cache.jobs if j.get("status") == "queued") + 1
            ctx.toast("Queued." if waiting <= 1 else f"Queued - {waiting} jobs in line.")
            return
        if key.startswith(("save:", "bake:", "sheet-save:")) or key == "screenshot":
            # ``screenshot`` was outside this and outside every other branch,
            # so a viewport capture the user had just chosen a destination for
            # finished in silence -- the one shape a save must never have,
            # since the only other thing that looks like it is a save that did
            # not happen. ``None`` is a cancelled picker and says nothing,
            # which is what the other three already rely on.
            if done.result is not None:
                ctx.toast(f"Saved to {done.result}")
            return
        if key == "trellis-log":
            # The one diagnostic for "the 3D engine stopped unexpectedly". The
            # button submitted this and nothing ever stored the answer, so the
            # box under it stayed empty forever.
            if isinstance(done.result, dict):
                ctx.state.preview["trellis_log"] = done.result.get("text") or ""
            return
        if key.startswith("export-"):
            # A bulk export finishing with no visible outcome reads as a
            # failure; single-artifact saves have always toasted.
            if done.result is not None:
                ctx.toast(f"Exported to {done.result}")
            return
        if key == "home-unreviewed":
            # Home's status block. A count only, and the last one stands until
            # a newer one lands -- a failed read leaves the previous figure up
            # rather than blanking a row somebody is reading.
            if isinstance(done.result, int):
                ctx.state.home_unreviewed = done.result
            return
        if key == "jobs-list":
            # A2: the frame-thread half of the split ``request``/``read``
            # started -- the one place ``jobs``, ``by_id`` and
            # ``_last_status`` are ever assigned, and the one place a status
            # transition is ever announced, same as the old inline ``tick``.
            if ctx.cache.adopt(done.result, self._announce_job_transition):
                self._sync_viewer()
            return
        if key == jobs_cache_mod.SEARCH_KEY:
            # shell-01 (the 2026-09-08 audit): the frame-thread half of the
            # split ``request_widen``/``_search`` -- the library's search used
            # to run its store query inline, here, on the frame thread.
            ctx.cache.adopt_widen(done.result)
            return
        if key == "storage" or key.startswith("storage:"):
            # Both the full walk and the per-job incremental re-measure (C33)
            # land here, each as a *reading* -- the sizes, or the one directory
            # to fold in, or why neither. The amendment happens here rather
            # than in the task because the library's size sort reads those
            # sizes and the memo key reads their generation (T3).
            ctx.cache.adopt_storage(done.result)
            return
        if (
            key.startswith(
                ("delete:", "prune", "rename:", "name:", "tags:", "fav:", "restore:", "purge:")
            )
            or key == "empty-trash"
        ):
            ctx.cache.invalidate()
            # ``restore:`` and ``purge:`` were missing from both lists, and each
            # absence showed differently. Restore (the toast's Undo, and the
            # trash's own button) left the row looking untouched for up to the
            # 3 s cache backstop, so the action read as inert and users clicked
            # it twice. Purge and Empty trash are worse: they are the actions
            # that actually *free disk*, and nothing re-measured -- so the
            # "N jobs - X GB" footer kept the pre-delete figure for the rest of
            # the session, which is the one number the whole affordance exists
            # to move (UX-11).
            if key.startswith(("delete:", "prune", "purge:")) or key == "empty-trash":
                self._request_storage()
            return
        if key == _import_mesh_key():
            # A new finished row, so the list has to refetch exactly as it does
            # after a delete or a rename. ``None`` is the picker being
            # cancelled and is silent: a toast saying nothing happened, after
            # the user chose for nothing to happen, is noise.
            if done.result is None:
                return
            ctx.cache.invalidate()
            self._request_storage()
            # Selected, because an import is a thing the user just *made* and
            # the next click is always on it -- the same reasoning ``create``
            # follows when a job it queued lands.
            ctx.state.select(done.result["id"])
            ctx.toast("Mesh imported. It is an ordinary asset now.", "success")
            return
        if key.startswith("retarget:"):
            # model.glb was rewritten under the viewer, and the params it is
            # described by changed with it: drop the mesh verdicts on screen and
            # reload what is now on disk.
            ctx.cache.invalidate()
            self._reload_viewer()
            stale = (done.result or {}).get("stale") or []
            ctx.toast(
                f"Mesh rebuilt. {len(stale)} rig artifact(s) now describe the old mesh."
                if stale
                else "Mesh rebuilt."
            )
            return
        if key.startswith("sheet-del:"):
            # Not covered by the "sheet:" prefix below, and _sync_viewer's
            # early-return means nothing else refetches the list: a deleted
            # sheet stayed on screen with live-looking buttons.
            self._refresh_rig_side_data()
            return
        if key.startswith(("cancel:", "rerun:", "remesh:", "retry:", "rig:", "joints:", "sheet:")):
            ctx.cache.invalidate()
            if key.startswith("sheet:"):
                # A rendered sheet is side data, not a job-row change, so the
                # cache invalidation above does not bring it back.
                self._refresh_rig_side_data()
            if key.startswith(("rig:", "joints:")):
                # Poser's own "Rigged assets" picker is throttled like
                # ``troupe_mode.sendable_meshes`` -- up to CAST_REFRESH_LIVE
                # stale on its own -- but a rig landing while the mode is
                # already open should not need a restart to appear in it.
                from .. import poser_mode

                poser_mode.invalidate_riggable(ctx)
            return
        if key == VIEWER_KEY:
            self._adopt_model(done)
            return
        if key == REVIEW_MESH_KEY:
            self._adopt_review_model(done)
            return
        if key == _compare_key():
            self._adopt_compare(done)
            return
        if key == VERIFY_KEY:
            # The slow half of the post-download refresh, landing off the frame
            # thread (UX-09). Everything that reads the checks happens here, so
            # the pane updates in one step rather than half now and half later.
            if done.result is not None:
                self.runtime.checks = done.result
            self._refresh_model_answers()
            present = {row["row_key"] for row in ctx.model_rows if row.get("present")}
            ctx.model_picks -= present
            return
        if key.startswith("pose-library:"):
            # The global pose library rows the asset Pose panel offers, keyed
            # by job like poses:/sheets: so an answer that lands after the
            # selection moved on is dropped rather than shown against the
            # wrong asset.
            job_id = key.partition(":")[2]
            if job_id == ctx.state.selected and isinstance(done.result, dict):
                ctx.state.preview["library_poses"] = done.result.get("poses") or []
            return
        if key.startswith("pose-"):
            if key.startswith("pose-save:") and self.viewer.pose_mode:
                # Only now is the pose actually on disk. A failed save leaves
                # the flag set, so the guard still stops the user walking away
                # from work that was never written.
                self.viewer.editor.dirty = False
            self._refresh_rig_side_data()
            ctx.cache.invalidate()
            return
        if key.startswith(SILENT_TASK_KEYS):
            return
        # Nothing claimed it. This module's own rule -- "the app claims results
        # by prefix, and a key without one is a result delivered nowhere" --
        # was enforced by nothing whatever, so a result arriving under an
        # unclaimed key was indistinguishable from one that had been handled,
        # in the one place where the difference is invisible from outside:
        # the *success* path. The two ways to get here are a mode closing
        # while its own task was in flight, and a new key whose author forgot
        # this file exists.
        #
        # Once per key rather than once per arrival: several of these are
        # resubmitted by a pane that runs every frame, and a line a frame is a
        # log nobody can read. ``log.info`` and not a warning, because a
        # deliberately silent key that nobody added to ``SILENT_TASK_KEYS``
        # would otherwise cry wolf for the life of the session.
        if key not in self._unclaimed:
            self._unclaimed.add(key)
            log.info("a %r task finished with nowhere to deliver its result", key)

    def _announce_job_transition(self, job: Any, previous: str | None) -> None:
        """``JobsCache``'s ``on_transition`` callback -- shared between
        :meth:`_refresh` (which submits the read) and :meth:`_on_task_done`
        (which lands it), so a status change is announced identically
        whichever call happens to be the one that adopted it.
        """
        from .. import review_mode
        from ..jobs_cache import sweep_summary, transition_message

        ctx = self.app_ctx
        sweep_id = job.get("sweep_id")
        if sweep_id:
            # One toast per sweep, not one per unit (N109). A twenty-unit
            # sweep otherwise raises twenty notices, which is exactly the
            # burst the "+N more" line exists to count -- and the useful
            # message ("how did it go") is the one nothing was raising.
            summary = sweep_summary(ctx.cache.jobs, sweep_id)
            if summary is not None:
                ctx.toast(*summary, action="review", action_arg=sweep_id)
        else:
            message = transition_message(job, previous)
            if message is not None:
                # "Show" selects it (N108): a toast that names a job and
                # offers no way to it makes the user find it by hand,
                # which after an overnight batch is the whole problem.
                ctx.toast(*message, action="show", action_arg=job["id"])
        if job["status"] == "done":
            # Incremental (C33): only this job's directory changed, so only
            # it is re-walked; delete and prune still trigger the full one.
            self._request_storage(job["id"])
            # The worker has just appended an observation for this job
            # (queue._observe_finished, same condition), and it has no way
            # to ask for the recompute itself -- it runs on the asyncio
            # thread and knows nothing about tasks or panes. This is the
            # only place a finished generation is noticed, so it is where
            # the machine half of the findings corpus enters the file:
            # without it, evidence recorded on every run would reach
            # findings.json only when somebody next filed a verdict.
            if job.get("stage") == "model" and job.get("kind") in ("text", "image"):
                review_mode.refresh_findings(ctx)
                self._select_finished_mesh_if_waiting(job)
            # 2026-09-05 audit, finding create-02: a remesh or a re-texture
            # is a *queued* job, unlike a retarget's foreground task, so the
            # "remesh:"/"retexture:" keys the panels submit fire when the
            # panel enqueues the row, not when the worker finishes it -- by
            # the time this job reaches "done" nothing is waiting on that
            # key any more. This transition, noticed the same way a
            # finished generation is noticed above, is the only place left.
            self._reload_viewer_after_rework(job)

    def _report_library_check(self, report: Any) -> None:
        """One line on screen, the whole report in the log.

        A findings *list* wants a modal with columns and a way to act on each
        row, and none of the five findings has an action this app should take on
        the user's behalf -- deleting an orphan directory or a stale verdict is
        exactly the bulk gesture that caused the 2026-08-09 loss the check
        exists to surface. So the pane says how many and where to read them, and
        ``warlock library verify --json`` is the surface that hands the detail
        to something that can act.
        """
        ctx = self.app_ctx
        if not isinstance(report, dict):
            return
        checked = report.get("checked", 0)
        noun = "asset" if checked == 1 else "assets"
        if report.get("ok"):
            ctx.toast(f"Library intact - {checked} {noun} checked.", "success")
            return
        log.warning("library verify: %s", json.dumps(report, default=str))
        findings = report.get("findings", 0)
        thing = "finding" if findings == 1 else "findings"
        ctx.toast(
            f"{findings} {thing} across {checked} {noun}. The detail is in warlock.log.",
            "warn",
            action="log",
        )

    def _check_worker(self) -> None:
        """Say so, once, when the GPU worker dies.

        Two plain attribute reads, so this is frame-loop safe. It used to be
        reported only through ``/api/health``, which the browser build polled;
        with the HTTP layer gone a mid-session worker crash became invisible
        outside the log file, and every job queued afterwards simply sat there.
        """
        if self._fatal_reported:
            return
        ctx = self.app_ctx
        fatal = self.runtime.fatal
        if fatal is not None:
            self._fatal_reported = True
            ctx.state.note_error(f"The GPU worker stopped: {fatal}. Restart Warlock.")
            ctx.toast("The GPU worker stopped. Nothing new will run.", "error")
        elif not self.runtime.alive:
            self._fatal_reported = True
            ctx.state.note_error("The GPU worker is not running. Restart Warlock.")
            ctx.toast("The GPU worker is not running.", "error")

    def _request_update_check(self) -> None:
        """Ask whether there is a newer Warlock, if the user asked us to ask.

        Off the frame thread for ``_request_storage``'s reason: nothing on the
        first frame needs the answer, and it simply appears in Settings ->
        Updates (or as one toast) once the task lands. Silent when the setting
        is off, which is the default -- this is the only thing in the app that
        would ever reach the network without a click, so it does not.
        """
        if not self.app_ctx.settings.get("auto_check_updates", False):
            return
        from ...service import updates as svc_updates

        self.app_ctx.submit(
            app_ctx_mod.UPDATE_CHECK_KEY, svc_updates.check, self.svc, tag="auto"
        )

    def _request_storage(self, job_id: str | None = None) -> None:
        """Re-measure the data directory off the frame thread.

        A recursive stat walk of every job directory is not something to do
        between ``new_frame`` and ``render`` -- and the moment it was being
        asked for is the worst one: the frame that should be showing a job
        finishing. ``submit`` refuses a duplicate key, so a burst of jobs
        completing coalesces into one walk rather than queuing several.

        With ``job_id`` only that job's directory is re-measured and folded
        into the running totals (C33); a delete or prune, which can touch any
        number of directories, still asks for the full walk.
        """
        ctx = self.app_ctx
        if job_id is not None:
            ctx.submit(f"storage:{job_id}", ctx.cache.measure_one, job_id)
            return
        ctx.submit("storage", ctx.cache.measure)

    def _reload_viewer(self) -> None:
        """Re-read whatever the viewport is showing, in place.

        ``_sync_viewer`` short-circuits when the path it wants is the path
        already loaded, which is right for a selection change and wrong after a
        retarget: model.glb was rewritten under the same name, so the file to
        reload is the one it is convinced is current.

        ``pending`` is cleared for the same reason and one more: a parse of the
        *pre*-retarget bytes may be in flight, and adopting it afterwards would
        put the old mesh back. Clearing it makes that result unwanted.
        """
        self.viewer.path = None
        self.viewer.pending = None
        self._sync_viewer()

    def _reload_viewer_after_rework(self, job: Any) -> None:
        """Reload the viewport once a queued remesh or re-texture finishes.

        2026-09-05 audit, finding create-02. A retarget runs as a foreground
        task under its own ``"retarget:{job_id}"`` key and reaches
        ``_on_task_done`` -> ``_reload_viewer`` the moment the rebuild lands.
        A remesh or a re-texture is a *queued* job instead: the
        ``"remesh:"``/``"retexture:"`` task key (``texture_panel.py``'s
        ``_submit``) fires when the panel enqueues the row, not when the
        worker finishes rewriting ``model.glb`` minutes later, so that key
        was never going to reload anything. Both also publish over the
        *source* job's directory by rename rather than their own -- like a
        rig -- so the id to compare against the selection is
        ``params["source_job"]``, never ``job["id"]``.
        """
        if job.get("status") != "done" or job.get("kind") not in ("remesh", "retexture"):
            return
        source_job = str((job.get("params") or {}).get("source_job") or "")
        if source_job and source_job == self.app_ctx.state.selected:
            self._reload_viewer()

    def _select_finished_mesh_if_waiting(self, job: dict[str, Any]) -> None:
        """A finished ``Make 3D`` becomes the selection, for a user still
        watching it land.

        2026-09-07 Create review, item 5.8: the Character build is the only
        auto-advance Create has (:meth:`_landed_character`); every other
        outcome, this one included, lands as a fading toast with a "Show"
        action -- so a user who pressed Make 3D, waited on the Mesh stage,
        and got a mesh was still looking at the reference. Fixed narrowly:
        all three conditions have to hold, or a user who has navigated away,
        changed the selection, or is standing on some other stage entirely
        would have their selection stolen out from under them mid-click.

        ``create_stages.parent`` is the lineage walk this reuses rather than
        re-deriving "the reference this mesh was promoted from" a fourth
        time, and the move itself goes through ``create_stages.go`` -- the
        one stage switch -- never a bare ``state.selected =``.
        """
        from ..modes.create.ui import stages as create_stages

        ctx = self.app_ctx
        if not create_stages.at(ctx.state, "mesh"):
            return
        parent = create_stages.parent(ctx, job)
        if parent is None or str(parent["id"]) != str(ctx.state.selected):
            return
        create_stages.go(ctx, "mesh", select=str(job["id"]))

    def _landed_character(self, result: dict[str, Any]) -> None:
        """A finished character, straight to the stage that can draw it.

        **Not the Reference stage.** ``create_character`` mints the model row
        finished and it has no ``input.png`` at all -- there is no generator
        behind a character, no seed a reconstruction could re-roll and no
        picture to approve -- so landing on Reference would put the user in
        front of an empty canvas with a row selected. The Mesh stage is where a
        body is drawn, and the row wears the placeholder glyph every unrendered
        row wears until its thumbnail exists.

        ``create_stages.go`` rather than a bare ``select``: it is the one stage
        switch, and it is what moves the selection along with the stage.
        """
        from ..modes.create.ui import stages as create_stages
        from ..modes.create.ui.panes import settings_character

        ctx = self.app_ctx
        job_id = str(result.get("id") or "")
        if job_id:
            create_stages.go(ctx, "mesh", select=job_id)
        # Composed at *submit* time, on the frame thread, where the form and
        # the species registry are both to hand -- the result carries two ids
        # and a kind, which is all a door should have to know about a sentence.
        ctx.toast(
            str(ctx.state.preview.pop(settings_character.TOAST_SLOT, ""))
            or "Building the character."
        )

    def _dispatch_character_preview(self, path: Any) -> None:
        """Queue a just-built character preview's parse, off the frame thread.

        create-02 (2026-09-11 audit): this used to call ``viewer.load_model``
        directly here -- the blocking glTF parse plus a PNG texture decode per
        slot -- on the very frame the "character-preview" build task lands,
        which is exactly the T2-class stall the 2026-09-02 review already
        fixed for the reference-PNG and mesh-load paths (see
        ``_sync_viewer``/``_adopt_model``). The build itself
        (``svc_characters.preview_character``, under ``PREVIEW_KEY``) already
        runs off-thread; only the load of its result was skipping the split.
        """
        from ..main import CHARACTER_PREVIEW_LOAD_KEY

        ctx = self.app_ctx
        if path is None or self.viewer is None:
            return
        wanted = Path(path)
        # create-03: the selection this dispatch was made under, carried as
        # the task's tag and compared again at landing in
        # ``_adopt_character_preview`` -- the same way ``viewer.pending``
        # gates ``_adopt_model`` -- so a build that lands after the user has
        # selected a different asset, entered Poser, or left Create's
        # Reference stage cannot silently replace whatever the viewport shows
        # by then.
        token = str(getattr(ctx.state, "selected", "") or "")
        ctx.submit(CHARACTER_PREVIEW_LOAD_KEY, self.viewer.parse_model, wanted, tag=(wanted, token))

    def _adopt_character_preview(self, done: Any) -> None:
        """Take a parsed character preview as the viewer's current model.

        Frame thread only -- the GPU upload has to be, same reason as
        ``_adopt_model``. ``_sync_viewer`` decides what to show from the
        *selection* alone, so a preview adopted here is swapped back out on
        the next cache tick unless something says otherwise; that something
        is the pin set below, ``_clear_viewport``'s idiom in the other
        direction.
        """
        from ..main import CHARACTER_PIN
        from ..modes.create.ui import stages as create_stages

        ctx = self.app_ctx
        if self.viewer is None or not isinstance(done.tag, tuple) or len(done.tag) != 2:
            return
        wanted, token = done.tag
        # create-03: dropped, not adopted, once the token no longer matches --
        # see ``_dispatch_character_preview`` for what it names.
        if (
            not create_stages.at(ctx.state, "reference")
            or str(getattr(ctx.state, "selected", "") or "") != token
        ):
            return
        try:
            self.viewer.adopt_model(done.result, wanted)
        except Exception:
            log.exception("could not open the character preview %s", wanted)
            ctx.toast("Could not show that character preview.", "error")
            return
        ctx.state.preview[CHARACTER_PIN] = (str(wanted), token)

    def _clear_viewport(self) -> None:
        """Empty the canvas of whichever Create stage is on screen.

        The hard part is not the clearing but making it *stay* clear.
        ``_sync_viewer`` runs from four places and again within ~3s off the
        cache tick, and it decides what to show from the selection alone -- so a
        plain ``clear()`` is undone by the next tick, which is exactly the
        "reference wants the picture, mesh wants the mesh" rule doing its job.
        What stops it is the short-circuit ``viewer.path == wanted``: leave the
        path naming the thing that *would* be loaded and the sync agrees there
        is nothing to do. This is ``_review_load``'s idiom -- clear, then pin --
        applied at the other end of the same mechanism.

        So the two stages clear differently and neither is arbitrary.
        ``clear_reference`` releases just the texture (forgetting the backend's
        registration before the release, which is the whole reason to call it
        rather than release the texture here), while ``clear()`` empties the
        mesh side and nulls path *and* pending. Nulling ``pending`` is not
        incidental: a parse may be in flight, and ``_adopt_model`` would
        otherwise land it on the emptied viewport a moment later. The pin is
        then written back explicitly in both branches, rather than relying on
        one of the two clears happening to leave ``path`` alone.

        The pin releases on its own the moment the selection or the stage
        changes, because ``wanted`` changes with it. Reselecting the same job
        re-shows it, which is what the tooltip says.
        """
        from .. import modes
        from ..modes.create.ui import stages as create_stages

        ctx = self.app_ctx
        viewer = self.viewer
        if viewer is None or ctx.state.mode not in modes.VIEWPORT_MODES:
            return
        # A strip renders off the mesh that is about to go; finishing it would
        # spend frames on an image of something no longer on screen.
        viewer.cancel_sheet_strip()
        if ctx.state.comparing:
            ctx.state.comparing = None
            viewer.exit_compare()
        job = ctx.job()
        job_dir = None if job is None else ctx.job_dir(job["id"])
        files = [] if job is None else (job.get("files") or [])
        # ``wanted``, computed exactly as ``_sync_viewer`` computes it -- the
        # two must agree or the pin names something the sync does not want and
        # the canvas refills on the next tick.
        if create_stages.at(ctx.state, "reference"):
            viewer.clear_reference()
            name = "input.png"
        else:
            viewer.clear()
            name = "model.glb"
        if job_dir is not None and name in files:
            viewer.path = job_dir / name

    def _sync_viewer(self) -> None:
        """Show whatever the selection implies, when it changes.

        Driven off the cache rather than off the click so a job that finishes
        while it is selected starts showing its mesh without another click.
        """
        from .. import modes
        from ..main import CHARACTER_PIN, VIEWER_KEY
        from ..modes.create.ui import stages as create_stages

        ctx = self.app_ctx
        if ctx.state.mode not in modes.VIEWPORT_MODES:
            # Only two modes draw a viewport. Everywhere else there is
            # nothing to sync, and loading a mesh for the selection would
            # be work nothing shows.
            return
        job = ctx.job()
        if job is None:
            return
        if self.viewer.pose_mode:
            # The pose editor is showing rig.glb on purpose. Without this the
            # next cache tick decides the selection "should" be showing
            # model.glb and reloads it, which drops the editor -- half a second
            # after it was opened.
            return
        if ctx.state.preview.get(CHARACTER_PIN) == (
            str(self.viewer.path or ""),
            str(getattr(ctx.state, "selected", "") or ""),
        ):
            # A character preview is a build of the *form*, not of the
            # selection, so this sync has nothing to say about it. The pin
            # names the selection it was taken under, which is what releases it
            # the moment the user selects something else -- the tuple stops
            # matching and the ordinary rule takes over again.
            return
        job_dir = ctx.job_dir(job["id"])
        files = job.get("files") or []
        wanted = None
        # Reference wants the picture; Mesh, Rig and Pose all want the mesh --
        # the rig and its poses are fitted to *that* geometry, and a stage that
        # framed something else would be describing a different object. (The
        # pose editor swaps in ``rig.glb`` itself once it is entered, which is
        # why this returns early while ``viewer.pose_mode`` is on.)
        if create_stages.at(ctx.state, "reference"):
            if "input.png" in files:
                wanted = job_dir / "input.png"
        elif "model.glb" in files:
            wanted = job_dir / "model.glb"
        if wanted is None or self.viewer.path == wanted or self.viewer.pending == wanted:
            return
        # Both kinds are decoded off-thread and adopted when they land. This
        # runs on a *timer*, on the frame a job transitions to done -- which is
        # when the file is largest and coldest -- so doing the parse and the
        # texture decode here froze the frame that was meant to show the job
        # finishing. The GPU upload stays on the frame thread; see
        # ``_adopt_model``, which tells the two apart by the tag's suffix.
        parse = self.viewer.parse_reference if wanted.suffix == ".png" else self.viewer.parse_model
        self.viewer.pending = wanted
        if not ctx.submit(VIEWER_KEY, parse, wanted, tag=wanted):
            # Another load is already in flight. Its result is checked against
            # ``pending`` before it is adopted, so this one is simply retried
            # on the next tick rather than queued.
            self.viewer.pending = None

    def _adopt_compare(self, done: Any) -> None:
        """Take a parsed comparison mesh. Frame thread only, and contained.

        The mirror of ``_adopt_model`` and for the same three reasons: the GPU
        upload has to be on the frame thread, a result nobody is waiting for
        any more must be dropped rather than shown, and a failure must produce
        a toast rather than take the process out.

        That last one is what UX-04 is actually about. ``Viewer.compare`` used
        to parse *and* upload inline from the menu handler with no error
        boundary at all, so a corrupt GLB -- or an upload that failed on a
        driver hiccup -- propagated out of the frame loop and exited Studio. A
        comparison is a *look* at something; it must never be able to cost the
        session.
        """
        ctx = self.app_ctx
        wanted = done.tag
        if wanted is None or ctx.state.compare_pending != wanted:
            ctx.state.compare_pending = None
            return
        ctx.state.compare_pending = None
        try:
            self.viewer.adopt_compare(done.result)
        except Exception:
            log.exception("could not open %s for comparison", wanted)
            ctx.state.comparing = None
            ctx.toast("Could not open that asset to compare.", "error")

    def _adopt_model(self, done: Any) -> None:
        """Take a parsed GLB as the viewer's current model. Frame thread only.

        The upload is what has to be here -- ``GpuModel`` creates buffers and
        textures on the one GL context, and releasing the old one does too.

        Checked against ``pending`` first: the selection can move while a parse
        is in flight, and adopting a result nobody is waiting for any more
        would put the previous asset back on screen.
        """
        ctx = self.app_ctx
        wanted = done.tag
        if wanted is None or self.viewer.pending != wanted:
            self.viewer.pending = None
            return
        self.viewer.pending = None
        try:
            if wanted.suffix == ".png":
                self.viewer.clear()
                self.viewer.adopt_reference(done.result)
                self.viewer.path = wanted
                self._refresh_rig_side_data()
                return
            self.viewer.adopt_model(done.result, wanted)
        except Exception:
            log.exception("could not open %s", wanted)
            ctx.toast("Could not open that asset.", "error")
            return
        job = ctx.job()
        # The thumbnail is free here: the model is loaded and framed, and a
        # server-side render would need the serial GPU queue for something
        # purely cosmetic.
        if job is not None and "thumb.png" not in (job.get("files") or []):
            ctx.capture_thumbnail(job["id"])
        self._refresh_rig_side_data()

    def _refresh_rig_side_data(self) -> None:
        """Poses, sheets and the shipped preset library, off-thread.

        Cleared first: these belong to a job, and leaving the previous one's
        poses on screen while the new one's are read would offer a list that
        applies to nothing.
        """
        from ...service import poses as svc_poses
        from ...service import rig as svc_rig
        from ...service import sheets as svc_sheets

        ctx = self.app_ctx
        job = ctx.job()
        for key in (
            "poses",
            "sheets",
            "bones",
            "library_poses",
            # The sprite panel's three, for the same reason: a form
            # holds this attempt's seeds, a draft listing holds one
            # reference's drafts, and the running bar names one job.
            "sprite_active",
            "sprite_drafts",
            "sprite_form",
        ):
            ctx.state.preview.pop(key, None)
        if job is None:
            return
        job_id = job["id"]
        if "rig.glb" in (job.get("files") or []):
            ctx.submit(f"poses:{job_id}", svc_rig.list_poses, ctx.svc, job_id)
            # The global pose library, filtered to this rig's own skeleton --
            # what the Pose panel's "Library poses" section offers.
            ctx.submit(f"pose-library:{job_id}", svc_poses.library_for_job, ctx.svc, job_id)
        ctx.submit(f"sheets:{job_id}", svc_sheets.list_sheets, ctx.svc, job_id)

    def load_presets(self, template: str | None) -> None:
        """The shipped pose library for a skeleton.

        Read once per template rather than per job: it is a property of the
        rig, not of the mesh, and applying one saves an ordinary pose through
        the same path a hand-made one does.
        """
        from ...service import rig as svc_rig

        if not template:
            self.app_ctx.state.preview["presets"] = []
            return
        self.app_ctx.submit(f"presets:{template}", svc_rig.template_presets, template)

    def _capture_thumbnail_from(self, job_id: str, view: Any) -> None:
        """The library card's picture, from the viewport that is already drawn.

        On the frame thread because it reads a framebuffer, which is the same
        reason ``ctx.capture_thumbnail`` is -- and it is the one deliberate
        exception to "the frame loop never blocks", being a single offscreen
        read rather than work.

        Takes the viewport rather than reaching for ``self.clay_view``, because
        there are two of them now: Mason exports its own scene from
        ``self.mason_view``, and a version of this that knew only about Clay's
        would have photographed whatever Clay happened to be holding -- or
        nothing at all in a session that never opened it.
        """
        from ...service import files as svc_files

        ctx = self.app_ctx
        if view is None:
            return
        try:
            # The GL readback only; the PNG encode joins the save on the task
            # thread (D41), exactly as ctx.capture_thumbnail does.
            image = view.screenshot()
        except Exception:
            # A warning rather than an error (E48): the export itself succeeded
            # and the asset is in the library -- what failed is its picture. The
            # card falls back to the placeholder, which on its own reads as "the
            # build produced nothing".
            log.exception("could not capture a thumbnail for built asset %s", job_id)
            # Level ``warn``, not ``error``: the export succeeded and only the
            # picture did not, so a red card over a build that worked would
            # overclaim -- but something *did* fail and the log has the
            # traceback, which is exactly the middle level H68 added. (It sat
            # at ``info`` only while info and error were the whole vocabulary.)
            ctx.toast("The asset was built, but its thumbnail could not be made.", "warn", "log")
            return

        def run() -> Any:
            import io

            buf = io.BytesIO()
            image.convert("RGB").save(buf, "PNG")
            return svc_files.save_thumbnail(ctx.svc, job_id, buf.getvalue())

        ctx.submit(f"thumb:{job_id}", run)
