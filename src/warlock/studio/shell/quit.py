"""The quit chain, teardown, and the window caption's dirty marker.

A **mixin on** :class:`~.app.App`, ``clay_viewport.ClayViewport``'s idiom
restated at shell scale -- ``self`` here is the App and every method's body is
unchanged from the line it stood on in ``studio/main.py`` before the P4
restructure split that module into ``shell/{app,frame,events,tasks,quit}.py``.

``_sync_title`` and ``_on_pose_dirty`` sit here rather than in ``frame.py``:
both are about the one question a quit's guard chain also answers -- is
anything unsaved -- and the window caption they maintain is the ambient
version of the same fact ``_quit_summary`` states in full sentences when a
quit is actually asked for.

The shell names this module reaches are imported *inside* the methods that use
them, ``studio/modes/clay/ui/viewport.py``'s own rule restated: ``main`` imports
:class:`~.app.App` (which assembles this mixin) to build the class, so a
module-scope import back would be a cycle.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class QuitMixin:
    """The quit chain and teardown, mixed into :class:`~.app.App`.

    The shell names it reaches are imported *inside* the methods that use
    them: ``main`` imports :class:`~.app.App` (which assembles this mixin) to
    build the class, so a module-scope import back would be a cycle. Same
    shape as ``clay_viewport.ClayViewport``.
    """

    def _quit_summary(self) -> str:
        """What quitting would actually interrupt, in one sentence per thing.

        Empty when nothing is going on, which is the common case and the whole
        point: the confirm this feeds used to say "Anything still generating is
        cancelled" *unconditionally*, on an idle app with nothing unsaved --
        a warning about a thing that was not happening, which teaches people to
        click through warnings (UX-21).
        """
        from ..app_ctx import UPDATE_DOWNLOAD_KEY

        ctx = self.app_ctx
        lines: list[str] = []
        if self.runtime.current_job_id is not None or ctx.cache.active is not None:
            lines.append("A job is still generating and will be cancelled.")
        # Named, not counted: "3 tasks" is not something a user can weigh, and
        # a 16 GB download is a very different thing to interrupt than a
        # thumbnail. Downloads and exports are the two worth calling out.
        busy = set(ctx.tasks.busy_keys)
        if any(k.startswith("download:") for k in busy):
            lines.append("A model download is in progress and will be stopped.")
        # muse-05: this used to be ``k.startswith(("export", "save:", "bake:"))``,
        # which only ever matched the Library's bulk "export-folder"/"export-zip"
        # keys -- every per-mode export queues as "<mode>-export:<name>" (Muse,
        # Sirens, Clay, Inker, Packwright, Plotter all do this), and none of
        # those *start with* "export", so a quit mid-export got no warning at
        # all. Matching "-export:" anywhere in the key, not just as a prefix,
        # catches every mode that follows this naming convention -- present or
        # future -- without keeping a hand-maintained list of prefixes here.
        if any(k.startswith(("export", "save:", "bake:")) or "-export:" in k for k in busy):
            lines.append("An export is still being written.")
        # H02: named alongside downloads and exports rather than omitted. The
        # commit phase (pip writing into ``site-packages``) is not offered
        # here at all -- see ``_ask_quit``, which blocks the ask outright
        # rather than warning about a quit it would then let through.
        if any(k.startswith("pack:") for k in busy):
            lines.append("A dependency pack is downloading and will be stopped.")
        # shell-03 (2026-09-13 audit): "-export:" caught the per-mode
        # in-editor exports (muse-05 above) but not a *library* export --
        # Packwright, Mason and Plotter each queue theirs as
        # "<mode>-library:<name>", which matches neither "-export:" nor any
        # prefix above. Quitting mid-write left an asset whose sidecar never
        # landed, unopenable in its editor afterwards. Matched the same way
        # as "-export:": anywhere in the key, so a future mode that follows
        # the convention is covered without a hand-list here.
        if any("-library:" in k for k in busy):
            lines.append("An export to the library is still being written.")
        # shell-03 (2026-09-13 audit): an update-installer download has no
        # resume marker, so a quit mid-download loses the whole thing, same
        # as a model download above -- but it was never checked here because
        # it does not start with "download:".
        if UPDATE_DOWNLOAD_KEY in busy:
            lines.append("An update download is in progress and will be stopped.")
        # shell-14 (2026-09-11 audit): the same shape as the three lines
        # above, missing until now. ``review-launch`` fires twenty to forty
        # job creations at once and ``review-delete``/``cleanup``/``remove``
        # sweep them back out again -- a quit mid-either got none of the "X
        # will be stopped" context downloads, exports and packs already give.
        # ``review-scan``/``findings``/``scores`` are read-only and left out
        # on purpose: interrupting a scan or a re-score loses nothing on disk.
        from .. import review_mode

        if any(
            k
            in (
                review_mode.LAUNCH_KEY,
                review_mode.DELETE_KEY,
                review_mode.CLEANUP_KEY,
                review_mode.REMOVE_KEY,
                review_mode.LABELS_KEY,
                review_mode.TRAIN_KEY,
            )
            for k in busy
        ):
            lines.append("A review sweep is launching or being cleaned up and will be interrupted.")
        return "\n".join(lines)

    def _ask_quit(self) -> None:
        """Ask once, about what is actually true, then run the guards.

        The chain below still asks per unsaved document, because each of those
        is a genuine question with a genuine answer ("discard *this* one?").
        What is gone is the unconditional preamble in front of it: on an idle
        app with nothing dirty -- the common state -- quitting used to raise a
        warning about generating that was not happening, and confirming it
        could then raise up to six more (UX-21). Now the generic question is
        asked only when it has something to say.
        """
        from .. import dialogs

        # H02: the one thing a quit is not allowed to interrupt, checked
        # before anything else asks a question it cannot honour. Every normal
        # quit route reaches this method -- the window's X and Alt+F4 (through
        # the ``pygame.QUIT`` handlers), the command palette's Quit command,
        # and the splash screen's own guard all go through ``ctx.ask_quit`` or
        # this method directly -- so this is the one place the guard has to
        # live, not one summary line among several.
        #
        # A dialog offers a promise ("Quit" means quit); the commit phase is
        # the one moment this process cannot keep that promise, because
        # honouring it either kills pip mid-write into the runtime's own
        # ``site-packages`` (immediately) or leaves the app sitting on a
        # confirmed quit for however long pip takes (soon after) -- neither of
        # which is what clicking "Quit" said would happen. So the ask itself
        # is withheld, not answered, until the commit phase clears.
        if self.app_ctx.tasks.commit_busy("pack:"):
            self._quit_deferred = True
            self.app_ctx.toast(
                "A dependency pack is being installed and cannot be "
                "interrupted safely. Quitting once it finishes.",
                "warn",
            )
            return
        summary = self._quit_summary()
        if not summary:
            self._request_quit()
            return
        self.app_ctx.confirms.ask(
            dialogs.Confirm(
                title="Quit Warlock Studio?",
                message=summary,
                confirm_label="Quit",
                cancel_label="Stay",
                on_confirm=self._request_quit,
            )
        )

    def _resume_deferred_quit(self) -> None:
        """Finish a quit that ``_ask_quit`` withheld for a pack's commit phase.

        Called after every ``pack:`` task completes, which is the only source
        of a state change this could be waiting on. Re-asks rather than
        quitting outright: a second pack could have started installing, or
        another reason to ask (an unsaved document) could exist by now, and
        ``_ask_quit`` already knows how to weigh both.
        """
        if not getattr(self, "_quit_deferred", False):
            return
        self._quit_deferred = False
        self._ask_quit()

    def _request_quit(self) -> None:
        """One chain, in order: painted pixels, then built geometry, then a pose.

        A list walked by index rather than three lambdas nested by hand (I78).
        ``ConfirmQueue`` is a real queue now, so the old reason for the nesting
        -- three questions at once would have dropped two -- is gone; the chain
        stays because it is the *semantics*, not the workaround. Asking all
        three side by side and quitting once all three said yes would mean
        clicking "Keep editing" on the first still left two more questions to
        dismiss, after the user has already said they are not quitting.
        """
        from .. import (
            inker_mode,
            mason_mode,
            packwright_mode,
            plotter_mode,
            poser_mode,
            sirens_mode,
        )
        from ..modes.clay import mode as clay_mode
        from ..panes import pose_panel

        ctx = self.app_ctx
        # The two pose guards are mutually exclusive by construction: the
        # inspector's asks about the shared viewer's editor, the Poser's about
        # its own instance, so no press ever answers one question twice.
        guards = (
            inker_mode.guard,
            clay_mode.guard,
            mason_mode.guard,
            plotter_mode.guard,
            packwright_mode.guard,
            sirens_mode.guard,
            pose_panel.guard,
            poser_mode.guard,
        )

        def step(index: int) -> None:
            if index == len(guards):
                self._quit()
                return
            guards[index](ctx, "quit", lambda: step(index + 1))

        step(0)

    def _quit(self) -> None:
        self._running = False

    # -- teardown ----------------------------------------------------------

    def teardown(self) -> None:
        """Unwind everything, and let no step stop a later one.

        Each stage is independent, and the last of them -- runtime.shutdown --
        is the one that stops the worker loop and the trellis child. A GL
        release raising on a lost context used to skip it, which is a stranded
        trellis-server and a process that will not exit.
        """
        import pygame

        from ..main import _step

        if self.fps.frames:
            log.info("frame loop: %s", self.fps.summary())
        ctx = self.app_ctx
        if ctx is not None:
            # Each mode's persist is its own step, so one raising cannot cost
            # the others -- but the *write* is one flush at the end, because
            # Settings holds the whole document and flushing per step wrote the
            # same file five times on the way out.
            #
            # The list of modes to persist is asked of ``mode_manifest``
            # rather than hand-called one line per mode: that hand list is
            # the bug this phase exists to fix -- it named five of the six
            # modules that define ``persist`` and never grew a sixth line for
            # Sirens, whose own docstring says it is called after every open
            # and save. ``persisting_modes`` asks each module with
            # ``getattr`` instead of copying the answer down here, so a
            # seventh mode that grows a ``persist`` is covered the moment it
            # exists rather than the next time somebody remembers this list.
            from .. import mode_manifest

            _step("persist settings", lambda: self._persist(ctx))
            for entry in mode_manifest.persisting_modes():
                _step(
                    f"persist {entry.key}",
                    lambda entry=entry: mode_manifest.call_persist(ctx, entry),
                )
            _step("write settings", ctx.settings.flush)
            if ctx.textures is not None:
                _step("release textures", ctx.textures.release)
            from .. import troupe_mode
            from ..panes import sheet_panel

            _step("release sheet strip", lambda: sheet_panel.release_strip_texture(ctx))
            # Troupe's atlas, for the same reason and by the same rule: it is
            # registered with the imgui backend by ``widgets.texture_ref``, so
            # it must be forgotten before it is released.
            _step("release troupe atlas", lambda: troupe_mode.release_texture(ctx))
            from .. import inker_mode, packwright_mode, plotter_mode

            _step("release inker textures", lambda: inker_mode.release_all(ctx))
            _step("release plotter textures", lambda: plotter_mode.release_all(ctx))
            _step("release atlas textures", lambda: packwright_mode.release_all(ctx))
        if self.viewer is not None:
            _step("release viewer", self.viewer.release)
        # ``getattr``, not an attribute access: teardown runs after a *failed*
        # setup too, and Clay's viewport is one of the last things constructed
        # -- an AttributeError here would skip runtime.shutdown, which is the
        # step that stops the worker loop and the trellis child.
        clay_view = getattr(self, "clay_view", None)
        if clay_view is not None:
            _step("release clay view", clay_view.release)
            # The ctx mirror dies with the view: a released view left on it
            # would hand the call sites dead GL objects, where None is the
            # answer every one of them already refuses.
            if ctx is not None:
                ctx.clay_view = None
        mason_view = getattr(self, "mason_view", None)
        if mason_view is not None:
            _step("release mason view", mason_view.release)
            if ctx is not None:
                ctx.mason_view = None
        if ctx is not None:
            from .. import mason_mode

            _step("release mason", lambda: mason_mode.release_all(ctx))
        poser_viewer = getattr(self, "poser_viewer", None)
        if poser_viewer is not None:
            _step("release poser viewer", poser_viewer.release)
        # ``getattr``, for ``clay_view``'s own reason above: a setup that
        # failed before ``setup_context`` built this must not skip
        # ``runtime.shutdown`` over an ``AttributeError``. Stopped before its
        # view is released, not after: a call still in flight when the host
        # stops must not be racing ``agent_clay.release`` for the same GL
        # objects.
        agent_host = getattr(self, "agent_host", None)
        if agent_host is not None:
            _step("stop agent host", agent_host.stop)
        from ..modes.clay.agent import dispatch as agent_clay

        _step("release agent view", agent_clay.release)
        if self.imgui_renderer is not None:
            _step("shutdown imgui", self.imgui_renderer.shutdown)
        _step("pygame.quit", pygame.quit)
        _step("runtime shutdown", self.runtime.shutdown)
        # The line whose *absence* is evidence: a session that ends without it
        # died somewhere no `except` could see.
        log.info("teardown complete")

    def _persist(self, ctx: Any) -> None:
        """The app's own settings. The write itself is teardown's last step.

        No mode: the app opens on Home every launch, so storing the one it
        happened to quit in would have no reader -- and quitting from the
        Manual or Settings would store a mode nothing would want restored.
        """
        from ..settings import sanitise_form
        from ..state import filters_to_store

        ctx.settings.set("show_fps", ctx.state.show_fps)
        ctx.settings.set("show_resources", ctx.state.show_resources)
        ctx.settings.set("form_2d", sanitise_form(ctx.state.form_2d))
        ctx.settings.set("form_3d", sanitise_form(ctx.state.form_3d))
        ctx.settings.set("history", ctx.state.history)
        # Not ``vars``: the trash is a *view* rather than a filter, so quitting
        # from it must not reopen in it. See ``state.VOLATILE_FILTERS``.
        ctx.settings.set("filters", filters_to_store(ctx.state.filters))

    # -- the unsaved-work signals a quit's guards also answer ---------------

    def _on_pose_dirty(self, dirty: bool) -> None:
        """The one reader of ``Viewer.on_pose_dirty``.

        The viewer reports on every pose edit, every gizmo release and both
        ends of the editor's life; this mirrors it onto ``AppState`` and marks
        the window. Cheap by construction: the callback fires on a *change*
        rather than per frame, and the caption is only touched when the answer
        actually moves -- ``set_caption`` is an OS call and the pose editor's
        rotate gizmo would otherwise make one per mouse-motion event.

        A mirror, not the authority. ``pose_panel.guard`` -- which is what
        stands between unsaved rotations and losing them, on the Done button,
        on a mode switch and in the quit chain -- goes on asking the editor
        itself, so the worst a missed notification can do is leave a marker on
        a title bar. The indicator exists because that guard is the only sign
        the app gives, and it appears *after* the user has asked to leave: the
        banner saying so is inside the very pane you have to be looking at.
        """
        ctx = self.app_ctx
        if ctx is None or bool(dirty) == ctx.state.pose_dirty:
            return
        ctx.state.pose_dirty = bool(dirty)
        self._sync_title()

    def _sync_title(self) -> None:
        """The window caption, marked while something is unsaved.

        **Every kind of unsaved, not only a pose.** The mark used to be
        ``pose_dirty`` alone, so a dirty drawing, sculpt, map, atlas or song
        left the caption clean -- the one place in the app that answers "have I
        saved this" at a glance, saying no to five of the six things it could
        be about. Derived from the quit-guard predicate
        (``docmodes.any_unsaved``), so the caption and the question asked on
        the way out cannot disagree.

        Swallows its own failure: a caption is not worth taking a frame down
        for, and this is reachable from a viewer callback that knows nothing
        about whether a display still exists (teardown releases the viewer
        after pygame has quit).
        """
        from .. import docmodes
        from ..main import WINDOW_TITLE

        ctx = self.app_ctx
        state = ctx.state if ctx is not None else None
        marked = bool(
            state is not None and (state.pose_dirty or docmodes.any_unsaved(ctx))
        )
        # Called once a frame now that it tracks five more things; setting the
        # caption is a window-manager round trip, so only a *change* is sent.
        if marked == self._title_marked:
            return
        self._title_marked = marked
        try:
            import pygame

            pygame.display.set_caption(f"{WINDOW_TITLE} *" if marked else WINDOW_TITLE)
        except Exception:  # a lost display, a headless run
            log.debug("could not set the window caption", exc_info=True)
