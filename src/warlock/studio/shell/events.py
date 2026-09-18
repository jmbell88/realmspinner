"""Input: the pygame event pump, the global shortcut table, mode switching,
and a file dropped on the window.

A **mixin on** :class:`~.app.App`, ``clay_viewport.ClayViewport``'s idiom
restated at shell scale -- ``self`` here is the App and every method's body is
unchanged from the line it stood on in ``studio/main.py`` before the P4
restructure split that module into ``shell/{app,frame,events,tasks,quit}.py``.

Three module-level functions travel here alongside the routers even though
their only caller, ``setup_context``, is on :class:`~.app.App` in
``shell/app.py`` rather than on this mixin: ``initial_mode``, the answer to
"which mode a fresh window opens on", and ``_leave_sirens_if_needed``/
``_leave_mode_if_needed``, the callbacks a mode switch runs on the way out of
whatever it is leaving. All three are about a mode *changing*, which is this
module's subject even where its own routers are not the ones asking.

``_compare_key`` and ``_import_mesh_key`` are **not** here, despite the split
plan naming them for this module: neither is read by anything below, only by
``shell.tasks.TasksMixin._on_task_done``'s landing dispatch, so they moved
there instead -- see that module's own docstring for the same note.

The shell names this module reaches are imported *inside* the methods that use
them, ``clay_viewport.py``'s own rule restated: ``main`` imports
:class:`~.app.App` (which assembles this mixin) to build the class, so a
module-scope import back from here would be a cycle.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def initial_mode(settings: Any, available: Callable[[str], bool]) -> str:
    """Which mode a fresh window opens on.

    Home, unless Settings says to reopen the last workspace and that
    workspace's door is still open. **The default stays Home for a fresh
    install** -- an absent ``startup_mode`` reads the same as an explicit
    "home", so a settings file written before this existed, or a user who
    never opened the choice, gets exactly the launch they always had.

    A remembered mode that is gated (weights or a pack not installed) falls
    back to Home through the same refusal every other switch already goes
    through: ``available`` is asked the identical question
    ``state.set_mode``'s ``_MODE_AVAILABLE`` hook asks, so this cannot answer
    "yes" to a door that switch would then refuse.

    And a remembered mode that no longer *exists* falls back the same way,
    checked against ``modes.KEYS`` before ``available`` is ever asked --
    ``_escape_mode`` already makes this check for its own history and this was
    the one reader of a persisted mode name that did not (shell-04, the
    2026-09-08 audit). Left unchecked, a stale or hand-edited
    ``last_workspace`` naming a retired mode reached ``_build_ui``'s else
    branch and opened Create while ``state.mode`` held a value nothing else in
    the app -- the rail's selected-item highlight, the status bar, Esc's
    history -- recognises as a member of ``modes.KEYS``.
    """
    from .. import modes
    from ..main import LAST_WORKSPACE_SETTING, STARTUP_HOME, STARTUP_LAST, STARTUP_MODE_SETTING

    if str(settings.get(STARTUP_MODE_SETTING) or STARTUP_HOME) != STARTUP_LAST:
        return STARTUP_HOME
    remembered = str(settings.get(LAST_WORKSPACE_SETTING) or "")
    if not remembered or remembered not in modes.KEYS or not available(remembered):
        return STARTUP_HOME
    return remembered


def _leave_sirens_if_needed(ctx: Any, old: str) -> None:
    """``state.set_mode_leave``'s installed callback: silence Sirens on the
    way out (sirens-03, the 2026-09-11 audit).

    Leaving the mode mid-song used to leave it sounding with no visible
    transport and no way to stop it short of returning to Sirens --
    ``sirens_keys.release_all`` (the panic key's own verb, already routed
    through ``sirens_play.stop`` rather than ``sirens_audio.stop`` alone,
    sirens-02) was reachable from nowhere else. ``state.py`` cannot call
    ``sirens_play.stop`` itself -- it must not import an editor module -- so
    ``set_mode`` calls this hook instead, installed next to ``set_mode_gate``
    in :meth:`~.app.App.setup_context`.

    A module-level function rather than a closure folded into that
    installation line, so a test can call it directly against a fake ``ctx``
    with no App to boot -- the same reason :func:`initial_mode` above is one.
    """
    if old != "sirens":
        return
    from .. import sirens_play

    sirens_play.stop(ctx)


def _leave_mode_if_needed(ctx: Any, old: str) -> None:
    """``state.set_mode_leave`` accepts one callback, so this is the single
    dispatcher installed in :meth:`~.app.App.setup_context` -- it replaces the
    Sirens-only hook with one that also closes out Muse and Plotter.

    shell-04 in the 2026-09-13 audit: leaving Muse kept an auditioned take
    sounding on the shared mixer (nothing but Sirens' leave was wired up, so
    the same fix that made ``sirens_keys.release_all`` reachable never
    reached Muse), and leaving Plotter mid-drag left the stroke, object or
    tile-metadata edit session open -- its write already in the document with
    no undo step recorded for it, so Ctrl+Z after a mode switch undid the
    *previous* action instead. All three ``end_*`` methods are idempotent
    (see their own docstrings), so calling one that has nothing open is a
    no-op rather than a wrong pop.
    """
    from ..panes import inspector

    inspector.flush_unsent_on_mode_change(ctx)
    _leave_sirens_if_needed(ctx, old)
    if old == "muse":
        from .. import muse_mode

        muse_mode.stop(ctx)
    elif old == "plotter":
        from .. import plotter_state

        tab = plotter_state.active(ctx)
        if tab is not None:
            tab.doc.end_stroke()
            tab.doc.end_object_edit()
            tab.doc.end_tile_meta_edit()


class EventsMixin:
    """Input routing and mode switching, mixed into :class:`~.app.App`.

    The shell names it reaches are imported *inside* the methods that use
    them: ``main`` imports :class:`~.app.App` (which assembles this mixin) to
    build the class, so a module-scope import back would be a cycle. Same
    shape as ``clay_viewport.ClayViewport``.
    """

    def _events(self) -> None:
        import pygame
        from imgui_bundle import imgui

        from .. import imgui_backend
        from .frame import _takes_pointer

        ctx = self.app_ctx
        io = imgui.get_io()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Through the *preflight* (the UI redesign, wave 3). This went
                # straight to ``_request_quit``, which walks the per-document
                # guards and asks nothing about a run in flight -- survivable
                # only while the header's power icon existed to carry
                # ``_ask_quit``'s generic summary. The header is gone, so the
                # window's X is the only interactive way out and it has to be
                # the one that asks.
                self._ask_quit()
                continue
            if event.type == pygame.VIDEORESIZE:
                # The *clamped* size is persisted, not the requested one: the
                # window that comes back is the clamped one, so storing the
                # raw event meant next launch opened below the resize floor
                # with no event to correct it.
                sized = (max(event.w, self._min_size[0]), max(event.h, self._min_size[1]))
                pygame.display.set_mode(sized, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
                ctx.settings.set("window_size", list(sized))
                continue
            if event.type in (pygame.WINDOWDISPLAYCHANGED, pygame.WINDOWMOVED):
                # UX-22. Both, because neither is sufficient: SDL2 reports the
                # display change when a window is dragged to another monitor,
                # but a display whose *own* scale is changed in Windows'
                # settings raises no such event and the window simply starts
                # being drawn at the wrong size. WINDOWMOVED catches the first
                # case again and costs one Win32 call, which is cheaper than
                # being wrong until the next restart.
                self._resample_display_scale()
                continue
            if event.type == pygame.DROPFILE:
                self._on_drop(Path(event.file))
                continue
            imgui_backend.process_event(event)
            if event.type in (pygame.KEYDOWN, pygame.KEYUP):
                # A modal owns the keyboard while it is up (I77): Esc cancels
                # it and Enter confirms it, and letting the same press through
                # here would also leave the mode behind the dialog, or submit
                # the form the dialog is a question about. Releases still pass,
                # because Inker's space-to-pan is a hold and would otherwise
                # latch on whenever a dialog opened mid-drag.
                #
                # A focused text field takes the *plain* keys only, so letters
                # still reach it. Modifier chords and the F-keys pass through:
                # the manual and the settings pane both promise Ctrl+K works
                # everywhere, and it used to die the moment the 2D prompt box
                # had focus -- which is exactly where you are when you want it.
                if not (event.type == pygame.KEYDOWN and self._modal_open()) and (
                    not io.want_text_input or self._passes_text_field(event)
                ):
                    self._shortcut(event)
                continue
            # Clay owns its own centre pane, so its viewport takes the mouse
            # in that mode and the asset viewer never sees it -- the two would
            # otherwise both orbit on one drag.
            if ctx.state.mode == "clay":
                self._build_event(event)
                continue
            # Mason owns its own centre pane too, ``clay``'s reason above.
            if ctx.state.mode == "mason":
                self._mason_event(event)
                continue
            # Poser too, and for a stronger reason: it has its own Viewer
            # instance, so the shared-viewer path below must never see its
            # events or one drag would orbit both cameras.
            if ctx.state.mode == "poser":
                self._poser_event(event)
                continue
            # The viewer sees the mouse when it is over the viewport image, and
            # a drag already in progress keeps it wherever the cursor goes.
            if _takes_pointer(self.viewer, self._viewport_hovered):
                self.viewer.handle_event(event, hovered=self._viewport_hovered)

    def _build_event(self, event: Any) -> None:
        """Route the mouse to Clay's viewport, on the same hover rule.

        A drag already in progress ignores the hover, so crossing onto a panel
        mid-orbit does not drop it -- which is exactly what ``_grab`` is for in
        the asset viewer.
        """
        from .. import clay_mode
        from .frame import _takes_pointer

        tab = clay_mode.active(self.app_ctx)
        if tab is None or self.clay_view is None:
            return
        # Every panel refuses edits while a save is in flight; a gizmo or
        # element drag pushes history steps too, so the viewport must as well.
        # Only new presses are refused: a drag already in progress keeps its
        # release (the bytes were captured before the save started), and
        # swallowing it would strand _grab.
        import pygame

        if tab.saving and event.type == pygame.MOUSEBUTTONDOWN:
            return
        hovered = self._build_hovered
        if _takes_pointer(self.clay_view, hovered):
            self.clay_view.handle_event(tab.doc, event, hovered)

    def _mason_event(self, event: Any) -> None:
        """Route the mouse to Mason's viewport, on ``_build_event``'s rule."""
        from .. import mason_assets, mason_mode
        from .frame import _takes_pointer

        tab = mason_mode.active(self.app_ctx)
        if tab is None or self.mason_view is None:
            return
        import pygame

        if tab.saving and event.type == pygame.MOUSEBUTTONDOWN:
            return
        hovered = self._mason_hovered
        if _takes_pointer(self.mason_view, hovered):
            source = mason_assets.ensure(self.app_ctx)
            self.mason_view.handle_event(tab.doc, source, event, hovered)

    def _poser_event(self, event: Any) -> None:
        """Route the mouse to Poser's viewer, on the same hover rule as Clay's.

        A drag already in progress ignores the hover, so crossing onto a panel
        mid-orbit does not drop it.
        """
        from .frame import _takes_pointer

        viewer = self.poser_viewer
        if viewer is None:
            return
        if _takes_pointer(viewer, self._poser_hovered):
            viewer.handle_event(event, hovered=self._poser_hovered)

    @staticmethod
    def _passes_text_field(event: Any) -> bool:
        """Whether a key still reaches the shortcuts while a field has focus.

        Modifier chords and the F-keys do; plain keys do not, so typing stays
        typing. The exception list is the one imgui itself owns inside a text
        field -- Ctrl+Z/Y/X/C/V/A are edit-the-text bindings there, and letting
        them through would undo the *document* while you renamed a layer.

        Modifiers come off ``event.mod`` rather than ``pygame.key.get_mods()``,
        which is ``review_mode.handle_key``'s rule and for its reason: ``mod``
        is the state at the moment this key was *pressed*, and ``get_mods()``
        is the state now. Events are drained in a batch after the frame, so a
        modifier released between the press and this call was already read as
        never held -- the shortcut was silently dropped, and only when the
        typist was fast.

        The exception list is Ctrl's alone. Alt and Meta chords were being
        tested against it too, so Alt+C and Alt+V were blocked from the global
        shortcuts on a rationale -- "imgui binds this inside a text field" --
        that is true of neither.
        """
        import pygame

        from ..main import _FUNCTION_KEYS, _TEXT_FIELD_CTRL

        name = pygame.key.name(event.key).lower()
        if name in _FUNCTION_KEYS:
            return True
        mods = event.mod
        if not mods & (pygame.KMOD_CTRL | pygame.KMOD_ALT | pygame.KMOD_META):
            return False
        if mods & pygame.KMOD_CTRL:
            return name not in _TEXT_FIELD_CTRL
        return True

    def _modal_open(self) -> bool:
        """Whether *any* modal is on screen and owns the keyboard."""
        from ..dialogs import modal_open

        return modal_open(self.app_ctx)

    def _note_mode(self, state: Any) -> None:
        """Sample ``mode`` so Esc knows where it came from.

        Once per key event rather than once per frame, and that is the whole
        reason it works: F1 changes the mode from inside this very function, so
        a frame-start sample would still be holding the mode from before it and
        Esc would go two steps back. Sampling here means every change made
        since the previous keypress -- by a landing tile, a library card, a
        drop, or F1 a moment ago -- has already landed.
        """
        if state.mode != state.mode_observed:
            state.previous_mode = state.mode_observed
            state.mode_observed = state.mode

    def _note_last_workspace(self, ctx: Any) -> None:
        """Persist ``ctx.state.mode`` so a later launch can reopen it (W3.1).

        Guarded the same way the crossfade above it is -- ``self._last_mode``
        rather than a per-frame write -- because ``Settings.set`` is a no-op
        on an unchanged value anyway and the point is *when* the value last
        changed being obvious from the diff, not from re-deriving it.
        Whatever ``state.mode`` is when this fires is what "Last workspace"
        means; there is no narrower list to filter it through, because a user
        who quit from Settings or the Library wanted exactly that back too.
        """
        from ..main import LAST_WORKSPACE_SETTING

        if ctx.state.mode != self._last_mode:
            ctx.settings.set(LAST_WORKSPACE_SETTING, ctx.state.mode)

    def _set_mode(self, key: str) -> None:
        """The one way a *shortcut* changes mode, so Home's reset is not a
        second spelling of the switch's.

        The switch itself is :func:`state.set_mode`, which the command palette
        also calls -- the palette used to carry its own copy of these four
        lines, and the copy had already lost the early return.
        """
        from ..state import set_mode

        set_mode(self.app_ctx.state, key)

    def _escape_mode(self) -> None:
        """Esc out of a mode you only pass through, back to the work you left.

        Home is the floor rather than a place you escape from: the app opens on
        it, so there is routinely nothing behind it, and bouncing to a stale
        ``previous_mode`` would be a mode switch nobody asked for.
        """
        from .. import modes

        state = self.app_ctx.state
        if state.mode == "home":
            return
        target = state.previous_mode
        if target == state.mode or target not in modes.KEYS:
            target = "home"
        self._set_mode(target)

    def _shortcut(self, event: Any) -> None:
        import pygame

        from .. import docmodes, modes

        ctx = self.app_ctx
        self._note_mode(ctx.state)
        # Ctrl+K, before everything, because it is the only binding that must
        # work in *every* mode and the workspace modes each consume whatever
        # reaches them. It is also, since the positional Alt+digit bindings went
        # away with the tenth-and-eleventh mode, the only keyboard route to a
        # mode at all. K is bound by neither Inker nor Clay, so no workspace
        # binding is displaced.
        # ``event.mod``, not ``pygame.key.get_mods()`` -- the rule
        # ``_passes_text_field`` states a few hundred lines down and
        # ``review_mode.handle_key`` already follows. ``mod`` is the modifier
        # state at the moment this key was *pressed*; ``get_mods()`` is the
        # state now, and events drain in a batch after the frame. A Ctrl
        # released between the press and this call made ``get_mods()`` lie, so
        # a fast Ctrl+K fell through to bare ``k`` -- which in Inker is the
        # **Rect tool**, so the palette failed to open *and* the active tool
        # changed under the user (UX-12).
        if (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_k
            and event.mod & pygame.KMOD_CTRL
        ):
            from ..panes import palette

            palette.toggle(ctx)
            return
        # **The palette owns the keyboard while it is up**, Esc included: its
        # query box holds the imgui focus and it reads its own Escape, Enter
        # and arrows there (``panes/palette.py``). Only Ctrl+K, above, is
        # exempt, because it is the way out. Without this the chords leaked
        # straight through -- ``palette_open`` was never one of
        # ``modal_open``'s answers, so Ctrl+Enter with the palette open in
        # Create queued a generation behind it.
        if event.type == pygame.KEYDOWN and ctx.state.palette_open:
            return
        # Beside Ctrl+K and for its reason: this is the second binding that
        # has to work in every mode, and the workspace modes below each consume
        # whatever reaches them. Slash rather than a letter because every
        # letter worth having is a tool in Inker or Clay, and Ctrl+/ is what
        # the rest of the world binds this to. It sets a flag; the header
        # consumes it (see ``_mode_switch``), because a key handler is not
        # inside the window the popup is registered in.
        if (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_SLASH
            and event.mod & pygame.KMOD_CTRL
        ):
            ctx.state.shortcuts_requested = True
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
            from ..manual import render as manual_render

            manual_render.toggle(ctx)
            return
        # shell-09 (2026-09-11 audit): F10 is documented in the Ctrl+/ sheet's
        # "Everywhere" section beside Ctrl+K, Ctrl+/, F1 and Esc -- and, like
        # them, it is stateless and has no pane of its own to consume it, so it
        # belongs above the Manual guard rather than being swallowed by it the
        # way every *other* key correctly is.
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F10:
            ctx.state.show_fps = not ctx.state.show_fps
            return
        # **And the Manual owns it too.** It covers the app, and the workspace
        # arms below consume whatever they are handed against a pane the reader
        # cannot see: Delete in Create trashed the selected asset unconfirmed,
        # and a bare tool letter switched Inker's tool under the overlay. Esc
        # passes because the branch immediately below is what answers it, and
        # Ctrl+K/Ctrl+//F1/F10 are above for the reason they always are.
        if (
            event.type == pygame.KEYDOWN
            and ctx.state.manual.open
            and event.key != pygame.K_ESCAPE
        ):
            return
        # Esc closes the Manual before anything else looks at it, and that
        # ordering is the whole of why this sits here rather than in
        # ``_escape_mode``: the workspace modes below consume every key they
        # are handed, so an Esc dispatched to Inker with the overlay up would
        # drop a floating selection and leave the reference open on top of it.
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and ctx.state.manual.open:
            from ..manual import render as manual_render

            manual_render.close(ctx)
            return
        # Then a running tour. Below the Manual because a step's "Read more"
        # raises the Manual over the tour, so the reference is the topmost
        # thing an Esc is about -- and above the workspaces for the same reason
        # the Manual is: Inker would consume the key and drop a floating
        # selection while the tour stayed up.
        if (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_ESCAPE
            and getattr(ctx.state, "tour", None)
            and ctx.state.tour.running
        ):
            from ..panes import tour as tour_pane

            tour_pane.stop(ctx)
            return
        # And the same keys the card reads for its own Back/Next/Finish
        # (tour-01, the 2026-09-07 audit, the other half of the fix beside
        # ``panes/tour.py``'s own focus gate): without this, the mode
        # dispatch below saw the identical press and acted on it too -- an
        # arrow that stepped the tour also moved the Library grid's cursor
        # underneath it. Gated on the card's own focus, not merely on the
        # tour running, so the same press still reaches the mode once the
        # reader has clicked away from the card (into a rename field, say);
        # ``has_focus()`` is one frame stale for the reason ``_viewport_hovered``
        # is, which is not a window a person can feel.
        if (
            event.type == pygame.KEYDOWN
            and event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_RETURN, pygame.K_KP_ENTER)
            and getattr(ctx.state, "tour", None)
            and ctx.state.tour.running
        ):
            from ..panes import tour as tour_pane

            if tour_pane.has_focus():
                return

        if ctx.state.mode not in modes.WORK_MODES:
            # The Manual, Settings and Profiles have no form to submit and no
            # viewport to frame; every one of these would act on a pane that is
            # not on screen. Esc is the one exception, and it is about the mode
            # rather than about anything in it. Home and the Library are lists,
            # and a list the user is looking at takes the arrows and Enter.
            if event.type != pygame.KEYDOWN:
                return
            if event.key == pygame.K_ESCAPE:
                self._escape_mode()
                return
            # Home's Resume list takes the arrows and Enter (M107). Library and
            # Profiles are their own modes now, so there is no sub-view behind
            # which a cursor could move invisibly and then fire on Enter.
            if ctx.state.mode == "home":
                from ..panes import landing

                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    landing.move(ctx, -1 if event.key == pygame.K_UP else 1)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    landing.activate(ctx, ctx.state.home_index)
            elif ctx.state.mode == "library":
                # The Home idiom exactly: the same selection-move the 2D/3D
                # fall-through routes the arrows to, so the library pane has
                # one keyboard whichever mode it is drawn in.
                from ..panes import library

                # A *grid* here, so Up and Down move by a row and Left and
                # Right by one -- the column count is whatever the grid drew
                # last frame. The sidebar keeps ``select_relative`` and its
                # one-card rows; which of the two a key means is a property of
                # the pane it was pressed in.
                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    library.select_grid(ctx, 0, -1 if event.key == pygame.K_UP else 1)
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    library.select_grid(ctx, -1 if event.key == pygame.K_LEFT else 1, 0)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    library.open_selected(ctx)
                elif event.key == pygame.K_DELETE and ctx.state.selected:
                    # The same binding the Create sidebar's library has, and
                    # the same reasoning: delete-to-trash is confirm-free here
                    # because the trash *is* the confirmation. The shortcuts
                    # sheet advertised it in both places and only one had it,
                    # which is a sheet that lies about the mode whose whole
                    # subject is the library.
                    library.delete_asset(ctx, ctx.state.selected)
            return
        if ctx.state.mode == "clay":
            from .. import clay_mode

            # First refusal, and unconditional for the reason Inker's is:
            # handle_key returns False with no document open, and letting that
            # fall through meant F/W/S acted on a viewport Clay has replaced.
            # Every Clay binding is in ``clay_mode.handle_key`` now, F
            # included: it records ``state.frame_pending`` and the viewport
            # consumes it, because framing needs a viewport this module owns
            # and that one may not import (B6).
            clay_mode.handle_key(ctx, event)
            return
        if ctx.state.mode == "poser":
            from .. import poser_mode

            # Unconditional for the workspace-mode reason: handle_key returns
            # False with nothing selected, and letting that fall through would
            # let F/W/S act on the asset viewport Poser has replaced.
            poser_mode.handle_key(ctx, event)
            return
        if ctx.state.mode == "review":
            from .. import review_mode

            # Unconditional for the reason Clay's and Inker's are: handle_key
            # returns False with no sweep run open, and letting that fall
            # through would let A/S/R act on a viewport and forms Review has
            # replaced. Nothing below this line belongs to Review.
            review_mode.handle_key(ctx, event)
            return
        if ctx.state.mode == "inker":
            from .. import inker_mode

            # Unconditionally, whether or not handle_key consumed it: it
            # returns False when no document is open, and letting that fall
            # through meant F/W/S toggled wireframe and turntable and
            # Ctrl+Enter submitted a mesh job -- all against a viewport Inker
            # has replaced. Nothing below this line belongs to Inker.
            inker_mode.handle_key(ctx, event)
            return
        if ctx.state.mode == "plotter":
            from .. import plotter_mode

            # Unconditional for the reason the three above are: handle_key
            # returns False with no map open, and letting that fall through
            # would let F/W/S act on a viewport Plotter has replaced.
            plotter_mode.handle_key(ctx, event)
            return
        if ctx.state.mode == "packwright":
            from .. import packwright_mode

            # Unconditional for the reason the four above are: handle_key
            # returns False for every key it does not bind, and letting that
            # fall through let the shared 2D/3D block below act on a library
            # and a viewport Packwright has replaced -- Delete trashed the
            # selected *library* asset (confirm-free, by that binding's own
            # design) and Ctrl+Enter queued a generation, from the atlas
            # packer. The return was lost when Troupe's branch was spliced in
            # ahead of it; a scan test now pins every workspace mode's arm.
            packwright_mode.handle_key(ctx, event)
            return
        if ctx.state.mode == "muse":
            from .. import muse_mode

            # Unconditional and returning, for the reason every workspace arm
            # here is: ``handle_key`` answers False for every key it does not
            # bind, and letting that fall through would let the shared 2D/3D
            # block act on a library and a viewport Muse has replaced -- Delete
            # would trash the selected *library* asset, from a results tray.
            muse_mode.handle_key(ctx, event)
            return
        if ctx.state.mode == "sirens":
            from .. import sirens_mode

            # Unconditional and returning, for the reason every workspace arm
            # above is: ``handle_key`` answers False for every key it does not
            # bind, and letting that fall through would let the shared 2D/3D
            # block act on a library and a viewport Sirens has replaced --
            # Delete would trash the selected *library* asset, from a tracker.
            # A scan test pins every workspace mode's arm.
            sirens_mode.handle_key(ctx, event)
            return
        if ctx.state.mode == "troupe":
            from .. import troupe_mode

            # Unconditional and returning, for the reason the three above are:
            # ``handle_key`` answers False for every key it does not bind, and
            # letting that fall through would let F/W/S act on a viewport
            # Troupe has replaced with a sprite.
            troupe_mode.handle_key(ctx, event)
            return
        if ctx.state.mode == "mason":
            from .. import mason_mode

            # Unconditional and returning, for site #8's reason (Mason's own
            # sweep row): every workspace arm above returns whether or not its
            # ``handle_key`` consumed the key, because a shared function's
            # tail is what let Packwright's return "get lost when Troupe's
            # branch was spliced in ahead of it" and sent Delete in the atlas
            # packer to the library trash instead. Stage A's ``handle_key``
            # binds nothing and always answers False, but the return still
            # has to be here now -- before a later mode's branch is spliced in
            # above the shared block below, not after.
            mason_mode.handle_key(ctx, event)
            return
        # Both edges reach this function, because Inker's space-to-pan is a
        # hold and needs the release. Nothing below is a hold: every one of
        # these is a toggle or an action, so acting on the release too undoes
        # the toggle the press just made and submits a second job for one
        # Ctrl+Enter.
        if event.type != pygame.KEYDOWN:
            return
        # ``event.mod``, for ``_passes_text_field``'s reason -- the state when
        # the key was pressed, not the state after the batch drained. A fast
        # Ctrl+Enter used to submit nothing at all, silently (UX-12).
        mods = event.mod
        # Before the 2D/3D bindings, because pose mode is drawn *over* them:
        # the inspector's pose editor is the same PoseEditor Poser authors
        # with, and it had no keyboard undo at all while Poser's got one. It
        # consumes only its own three chords and only while the editor is
        # bound, so nothing below moves when pose mode is off.
        if docmodes.pose_undo_key(self.viewer, event):
            return
        if event.key == pygame.K_RETURN and mods & pygame.KMOD_CTRL:
            from .. import create_stages
            from ..panes import settings_2d, settings_3d

            if create_stages.at(ctx.state, "reference"):
                settings_2d.generate(ctx, ctx.state.form_2d)
            else:
                settings_3d.promote(ctx, ctx.cache.get(ctx.state.source_job), ctx.state.form_3d)
        elif event.key == pygame.K_ESCAPE:
            from ..panes import pose_panel

            if ctx.state.comparing:
                ctx.state.comparing = None
                self.viewer.exit_compare()
            elif self.viewer.pose_mode:
                pose_panel.guard(ctx, "leave edit mode", lambda: pose_panel.leave(ctx))
        elif event.key in (pygame.K_UP, pygame.K_DOWN):
            # The library is the sidebar in both generate modes, so the arrows
            # are unambiguous here; Review owns Left/Right for its own list and
            # is returned above. Nothing else in 2D/3D reads an arrow key.
            from ..panes import library

            library.select_relative(ctx, -1 if event.key == pygame.K_UP else 1)
        elif event.key == pygame.K_DELETE and ctx.state.selected:
            # The library keyboard used to stop at navigation: Up/Down/Enter
            # moved and opened, and every action was mouse-only (UX-27).
            #
            # Delete specifically, and unguarded, because delete-to-trash is
            # deliberately confirm-free here -- "the trash *is* the
            # confirmation", which is the reasoning the menu item already
            # stands on. So this binding is exactly as safe as the menu item it
            # mirrors, and no safer or less safe.
            #
            # ``F`` is deliberately not bound to favourite despite the finding
            # offering it: F already frames the viewer a few lines below, and
            # taking a live 3D binding to add a library one would be a trade,
            # not a fix.
            from ..panes import library

            library.delete_asset(ctx, ctx.state.selected)
        elif event.key == pygame.K_f:
            self.viewer.frame()
        elif event.key == pygame.K_w and event.mod & pygame.KMOD_SHIFT:
            # Shift+W: the layout editor (P5.4). Verified free -- plain W is
            # wireframe below, and every mode that takes W takes it before this
            # handler is reached.
            from .. import layout_edit

            layout_edit.toggle(ctx.state)
        elif event.key == pygame.K_w:
            ctx.state.wireframe = not ctx.state.wireframe
            self.viewer.set_wireframe(ctx.state.wireframe)
        elif event.key == pygame.K_s:
            ctx.state.turntable = not ctx.state.turntable
            self.viewer.set_turntable(ctx.state.turntable)

    def _resample_display_scale(self) -> float:
        """Re-read the monitor's scale and rebuild what is baked at it (UX-22).

        DPI was sampled once, at startup, and the module comment said a
        mid-session change "would need a rebuild" as though that were
        unavailable -- but the UI-scale slider has done exactly this rebuild
        since K99, and every piece of it is reusable. Dragging the window from
        a 100% monitor to a 150% one left the whole UI drawn at the old scale
        until the next launch; on the pair of displays this is most likely to
        happen on, that is either a UI two-thirds the size it should be or one
        half again too big.

        -> the scale in force afterwards, so a caller can tell whether
        anything moved.

        The font atlas is *not* rebuilt here. It cannot be: rebuilding
        invalidates every ImFont handle, and those are pushed and popped all
        through ``_build_ui``, so the flag is raised and the frame loop
        consumes it between frames (K99) -- the same route the slider takes.
        """
        import pygame
        from imgui_bundle import imgui

        from .. import dpi, theme, tokens
        from ..main import _min_window_size
        from .frame import _ui_scale

        monitor_scale = dpi.window_scale(pygame)
        if monitor_scale == self._monitor_scale:
            return tokens.SCALE
        # The user's zoom is re-read rather than divided back out of the old
        # product: ``set_scale`` clamps, so on a display scaled past the
        # ceiling the stored zoom and the zoom in force differ, and recovering
        # it by division would bake that clamp in permanently -- each move
        # between two such monitors shrinking the UI again.
        self._monitor_scale = monitor_scale
        self.app_ctx.dpi_scale = monitor_scale
        lo, hi = tokens.ui_scale_bounds(monitor_scale)
        tokens.set_scale(monitor_scale * min(max(_ui_scale(self.app_ctx.settings), lo), hi))
        theme.apply(imgui)
        self.app_ctx.state.fonts_dirty = True
        # The resize floor follows the monitor too, and a stale one is the
        # difference between a window that can be made small enough to fit and
        # one that cannot.
        self._min_size = _min_window_size(monitor_scale)
        log.info("display scale changed to %.2fx; rebuilding style and fonts", monitor_scale)
        return tokens.SCALE

    def _on_drop(self, path: Path) -> None:
        from .. import create_stages
        from ..main import DROP_REFUSALS, DROPPABLE_IMAGES
        from ..panes import settings_3d

        ctx = self.app_ctx
        if ctx.state.mode == "inker":
            from .. import inker_mode

            inker_mode.open_path(ctx, path)
            return
        if ctx.state.mode == "clay":
            from .. import clay_mode, clay_state

            if path.suffix.lower() == clay_state.WBLK_SUFFIX:
                clay_mode.open_path(ctx, path)
            elif path.suffix.lower() == ".glb":
                clay_mode.import_glb_path(ctx, path)
            else:
                ctx.toast("Clay opens .wblk documents and .glb meshes.", "error")
            return
        if ctx.state.mode == "mason":
            from .. import mason_mode, mason_state

            if path.suffix.lower() == mason_state.WSCN_SUFFIX:
                mason_mode.open_path(ctx, path)
            elif path.suffix.lower() == ".glb":
                tab = mason_mode.active(ctx)
                if tab is None:
                    ctx.toast("Open or start a scene first: a mesh is placed into one.", "error")
                else:
                    mason_mode.import_glb_path(ctx, path)
            else:
                ctx.toast("Mason opens .wscn scenes and places .glb meshes.", "error")
            return
        if ctx.state.mode == "plotter":
            from .. import plotter_mode, plotter_state

            suffix = path.suffix.lower()
            if suffix in plotter_state.MAP_SUFFIXES:
                plotter_mode.open_path(ctx, path)
            elif suffix == ".tsx" or suffix in DROPPABLE_IMAGES:
                # An image dropped in Plotter is a *tileset*, not a map -- the
                # 2D pane's reasoning applied here: the refusal and the accept
                # have to say what a drop would have done in this mode.
                plotter_mode.add_tileset_path(ctx, path)
            else:
                ctx.toast(
                    "Plotter opens .wmap, .tmx and .tmj maps, and adds .tsx or "
                    "image files as tilesets.",
                    "error",
                )
            return
        if ctx.state.mode == "packwright":
            from .. import packwright_mode, packwright_state

            suffix = path.suffix.lower()
            if suffix == packwright_state.WPACK_SUFFIX:
                packwright_mode.open_path(ctx, path)
            elif suffix in DROPPABLE_IMAGES:
                packwright_mode.add_source_paths(ctx, [path])
            else:
                ctx.toast("Packwright opens .wpack documents and packs image files.", "error")
            return
        if ctx.state.mode == "sirens":
            from .. import sirens_mode, sirens_state

            suffix = path.suffix.lower()
            if suffix == sirens_state.WSNG_SUFFIX:
                sirens_mode.open_path(ctx, path)
            elif suffix == ".wav":
                # A WAV dropped here is a *sample*, not a song -- the Plotter
                # rule that a refusal and an accept both have to say what a
                # drop would do in this mode. It lands in the open song's sample
                # table; with no song open there is nowhere to put it, and
                # opening one silently to hold a drum hit would be a document
                # the user did not ask for.
                tab = sirens_mode.active(ctx)
                if tab is None:
                    ctx.toast(
                        "Open or start a song first: a sample belongs to one.", "error"
                    )
                else:
                    sirens_mode.import_sample(ctx, tab, path)
            else:
                ctx.toast("Sirens opens .wsng songs and .wav samples.", "error")
            return
        if ctx.state.mode in DROP_REFUSALS:
            # A mode that opens no files says so, before Create's branches
            # below can switch the window out from under the user. See the
            # table for the hole this closes.
            ctx.toast(DROP_REFUSALS[ctx.state.mode], "error")
            return
        if ctx.state.mode in ("home", "library") and path.suffix.lower() == ".glb":
            # Home and Library take a mesh straight into the library, which is
            # the other half of the door ``library.pick_and_import_mesh``
            # opens: a user with a ``.glb`` reaches for a drop before a menu,
            # and until 2026-08-30 the only surface that accepted one was Clay
            # -- which converts it into an editable document and refuses a
            # rigged mesh outright. Create is deliberately *not* here: a mesh
            # dropped mid-generation is ambiguous between "start from this" and
            # "put this in my library", and the branch below already answers
            # that question for images.
            from ..panes import library

            library.import_mesh_path(ctx, path)
            ctx.toast(f"Importing {path.name}...", "info")
            return
        if path.suffix.lower() not in DROPPABLE_IMAGES:
            # The refusal says what a drop would have *done here* (H71). One
            # sentence for both modes was wrong in 2D, where a dropped image is
            # a conditioning reference and never a mesh -- and the sentence is
            # the only thing that teaches the difference, since the two modes
            # accept the same file types.
            ctx.toast(
                "Drop an image to condition this generation on it."
                if create_stages.at(ctx.state, "reference")
                else "Drop an image to start a mesh from it.",
                "error",
            )
            return
        if create_stages.at(ctx.state, "reference"):
            # In the 2D pane a dropped image is a *conditioning reference*, not
            # a mesh to build -- forcing the mode switch here would throw away
            # the prompt the user is composing. One branch, and it is what
            # makes the feature discoverable at all.
            ctx.state.form_2d["ref_path"] = str(path)
            # H70's other half -- a ``widgets.request_open`` for the
            # References block -- is gone, not moved. It was written when that
            # block was a collapsible header that defaulted shut; it is always
            # open now, and no ``persist_key`` by that name is registered
            # anywhere, so the request matched nothing and merely accumulated
            # in ``widgets._OPEN_REQUESTS`` for the life of the process -- with
            # a comment beside it claiming it was why the drop was visible.
            # The flash below is what actually says the drop landed.
            self._flash_drop("2d-ref")
            ctx.toast(f"Using {path.name} as the reference.", "success")
            return
        # A drop is a start: it would otherwise land behind the chooser, with
        # nothing on screen saying anything had happened. ``follow=False``:
        # the file being dropped is the source, so walking the selection onto
        # a mesh the current one already has would describe the wrong asset.
        create_stages.go(ctx, "mesh", follow=False)
        self._flash_drop("3d-source")
        settings_3d.upload(ctx, path)

    def _flash_drop(self, slot: str) -> None:
        """Mark a slot as having just received a drop, for ``widgets.ring``."""
        state = self.app_ctx.state
        state.drop_flash_slot = slot
        state.drop_flash_at = time.monotonic()

    def _toast_action(self, name: str, arg: str | None = None) -> None:
        """What a toast's action button does, kept out of the widget.

        ``widgets.toasts`` knows what to *draw* for an action and nothing about
        what it means, which is what lets state.py carry the name with no
        import of the App and lets a pane raise a toast without either.
        """
        ctx = self.app_ctx
        if name == "log":
            ctx.open_log()
        elif name == "show" and arg:
            from .. import asset_open

            # Through ``asset_open``, which knows that a follow-up row -- a
            # rig, a sheet, a sprite draft -- holds nothing of its own and
            # routes to the asset whose directory its artifacts landed in.
            # Routing by stage sent every one of them to the Mesh stage of a
            # row with no mesh, which is a toast saying "finished" followed by
            # a blank screen.
            asset_open.open_asset(ctx, arg)
            # The row that actually got selected, so the grid scrolls to what
            # is on screen rather than to an invisible follow-up.
            job = ctx.cache.get(arg)
            ctx.state.library_scroll_to = asset_open.route(job).job_id if job is not None else arg
        elif name == "undo" and arg:
            from ..panes import library

            # Through the library's own restore, so the tick set and the
            # selection are handled exactly as they are when the trash view's
            # own Restore button is pressed.
            library.restore_asset(ctx, arg)
        elif name == "unlock" and arg:
            from .. import plotter_mode

            plotter_mode.unlock_layer(ctx, arg)
        elif name == "review":
            # The sweep is not named here: Review rescans on arrival and its
            # run list is the thing that knows which directories exist. Landing
            # on the mode is the whole of what the button promises.
            self._set_mode("review")

    # Every binding the app answers to, in one place the user can find. The
    # tuples are (keys, what), grouped; Inker's letters come from TOOL_KEYS so
    # this list cannot drift from the handler.
    def _shortcuts_popup(self) -> None:
        from imgui_bundle import imgui

        from .. import icons, tokens, widgets
        from ..shortcuts import shortcut_sections
        from ..tokens import sp

        viewport = imgui.get_main_viewport()
        popup_width = min(sp(tokens.SURFACE_W_SHEET), viewport.work_size.x - sp(32))
        popup_height = min(sp(tokens.SURFACE_H_SHEET), viewport.work_size.y - sp(64))
        imgui.set_next_window_pos(
            (
                viewport.work_pos.x + viewport.work_size.x - sp(16),
                viewport.work_pos.y + sp(48),
            ),
            imgui.Cond_.appearing.value,
            (1.0, 0.0),
        )
        imgui.set_next_window_size((popup_width, popup_height))
        alpha, rise = widgets.popup_enter("shortcuts")
        # Translucent (UX.md Phase 5): cleared before ``begin`` paints it,
        # painted back below as a blur of the app or as the solid fill.
        frosted = widgets.frosted()
        if frosted:
            imgui.set_next_window_bg_alpha(0.0)
        imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
        if not imgui.begin_popup("shortcuts"):
            imgui.pop_style_var()
            return
        rounding = imgui.get_style().popup_rounding
        widgets.window_shadow("raised", radius=rounding)
        if frosted:
            widgets.window_backdrop(radius=rounding)
        if rise > 0.0:
            imgui.dummy((0, rise))
        widgets.pane_header(
            "Keyboard shortcuts",
            actions=(("close", f"{icons.X} Close", imgui.close_current_popup),),
        )

        # Collected first, drawn after the box (UX.md Phase 4). The list is ~60
        # rows over eight groups, which is a scroll and a read rather than a
        # lookup -- and the subsequence matcher the command palette already
        # carries is the right instrument, so it is reused rather than
        # reimplemented. The rows themselves are ``shortcut_sections``, which
        # is module-level so the manual can be gated against it.
        sections = shortcut_sections()
        if imgui.begin_child("shortcuts/scroll", (0, 0)):
            self._draw_shortcut_rows(sections)
        imgui.end_child()
        imgui.end_popup()
        imgui.pop_style_var()

    def _draw_shortcut_rows(self, sections: list[tuple[str, list[tuple[str, str]]]]) -> None:
        """The filter box and whatever survives it."""
        from imgui_bundle import imgui

        from .. import widgets
        from ..shortcuts import filter_shortcuts

        imgui.set_next_item_width(-1)
        self._shortcuts_query = widgets.input_text(
            "##shortcuts-filter",
            self._shortcuts_query,
            max_length=60,
            hint="Filter shortcuts...",
        )
        kept = filter_shortcuts(sections, self._shortcuts_query)
        if not kept:
            widgets.muted("No shortcut matches that.")
            return
        for title, rows in kept:
            widgets.section(title)
            if imgui.begin_table(f"keys/{title}", 2):
                for keys, what in rows:
                    imgui.table_next_column()
                    widgets.muted(keys)
                    imgui.table_next_column()
                    imgui.text(what)
                imgui.end_table()

    def _layouts_popup(self, ctx: Any) -> None:
        """The Window menu's layout switcher (P5.3).

        A switcher and nothing else: renaming, duplicating, deleting and
        resetting are Settings -> Advanced, which is **the canonical path**
        because Settings is reachable from the rail in every mode and no
        workspace layout can touch its single-column composition. This popup
        carries a Reset because that is the rung a user reaches for while
        looking at the layout that went wrong.
        """
        from imgui_bundle import imgui

        from .. import controls, widgets

        if not imgui.begin_popup("layouts"):
            return
        widgets.popup_chrome(_imgui=imgui)
        widgets.secondary("Workspace layout")
        imgui.separator()
        for name, layout in sorted(self.layouts.layouts.items()):
            selected = name == self.layouts.active
            label = name if layout.readable else f"{name}  (a newer version)"
            if (
                controls.menu_item(
                    f"{label}##layout/{name}",
                    "",
                    selected,
                    layout.readable,
                    reason=(
                        "This layout was saved by a newer build. It is kept exactly "
                        "as it was found rather than reinterpreted."
                    ),
                )[0]
                and layout.readable
            ):
                self.layouts.set_active(name)
        imgui.separator()
        if controls.menu_item_simple("Reset this layout"):
            self.layouts.reset()
        if controls.menu_item_simple("Manage layouts..."):
            # Settings, rather than a second administration surface here: one
            # place that can rename and delete is one place to look for the
            # thing you deleted.
            from ..state import set_mode

            set_mode(ctx.state, "settings")
        imgui.end_popup()
