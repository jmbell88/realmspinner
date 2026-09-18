"""The frame loop: idle-skip, the per-frame cache tick, and ``_build_ui``.

A **mixin on** :class:`~.app.App`, ``clay_viewport.ClayViewport``'s idiom
restated at shell scale: ``self`` here is the App and every method's body is
unchanged from the line it stood on in ``studio/main.py`` before the P4
restructure split that 5,971-line module into ``shell/{app,frame,events,
tasks,quit}.py``.

Six module-level layout helpers travel with :meth:`FrameMixin._build_ui`
rather than staying behind in ``main.py``: ``_split_column``, ``_right_column``
and ``_column_boundary`` are read by every workspace module (the six
``<mode>_workspace.py`` files, plus ``studio/modes/clay/ui/viewport.py``, ``mason_viewport.py``,
``poser_viewport.py`` and ``review_panes.py``) and not only by this one, so
"only it uses them" was the split plan's guess rather than a fact about the
code -- every one of those callers now reaches this module by name
(``from ..shell.frame import _column_boundary``) instead of the bare name a
shared ``main.py`` namespace used to resolve for free. ``_stage_pane`` and
``_takes_pointer`` really are read only from here (and, for the latter, from
``events.py``'s three event routers). ``_ui_scale`` is read from here and from
``shell/app.py``'s ``setup_window``.

``_stage_rail`` travels here too, beside ``_stage_pane`` it is drawn next to
in ``_build_ui``'s Create branch -- it was not named for either shell module
in the split plan, but it is Create's own breadcrumb and has no workspace
module of its own to go to (Create is not one of the six modes this wave
gives a ``<mode>_workspace.py``; that split is P5's, not this one's).

The shell names this module reaches -- ``main``'s constants and the two
teardown/paint helpers every shell module shares -- are imported *inside* the
methods and module functions that use them, ``studio/modes/clay/ui/viewport.py``'s own rule
restated: ``main`` imports :class:`~.app.App` (which assembles this mixin) to
build the class it re-exports as ``warlock.studio.main.App``, so a module-scope
import back from here would be a cycle.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

from .. import anchors, guard, probe, tokens
from ..modes.create.ui import brief as create_brief

log = logging.getLogger(__name__)


def _split_column(
    ctx: Any,
    lay: Any,
    *,
    split_id: str,
    handle_length: float,
    width: float,
    edge: Any,
    top: tuple[str, Any, Callable[[Any], None]],
    bottom: tuple[str, Any, Callable[[Any], None]],
    before: tuple[str, Any, Callable[[Any], None], float] | None = None,
    middle: tuple[str, Any, Callable[[Any], None], float] | None = None,
    wanted: float | None = None,
    below_floor: float = 0.0,
) -> None:
    """One column of stacked panes with a drag handle between them.

    Every workspace builds the same shape twice -- a pane sized from a share,
    a pane taking what is left -- and each built it by hand. Two consequences
    the one function fixes at the source:

    * **A key per split.** ``split_id`` names *this* column, and the handle's
      id is derived from it (``f"{split_id}-share"``) rather than passed. So
      the two can no longer disagree, and a second column cannot be given the
      first one's key by copying the block.
    * **A handle at all.** Six of the workspaces drew a proportion the
      user could not change, because only the three columns that had a
      ``splitter`` call got one. It is drawn here, so having a split *is*
      having a handle.

    ``avail_y`` is captured before the first sized pane and after ``before``,
    which is the height the shares are really taken out of; measuring it after
    the top pane divides the drag delta by a height that pane already spent,
    and the handle then travels at the wrong rate.

    ``wanted``/``below_floor`` turn the plain proportion into Inker's give-way
    split, where a pane with a known minimum content height wins over the
    stored share and the pane beneath it keeps a floor of its own.
    """
    from imgui_bundle import imgui

    from .. import layout as layout_mod
    from .. import tokens as tokens_mod

    imgui.begin_group()
    if before is not None:
        name, role, draw_fn, height = before
        with layout_mod.pane(name, (width, height), role, edge=edge) as visible:
            if visible:
                draw_fn(ctx)
    avail_y = imgui.get_content_region_avail().y
    if wanted is None:
        top_height = avail_y * lay.share(split_id)
    else:
        top_height = layout_mod.give_way(avail_y, lay.share(split_id), wanted, below_floor)
    with layout_mod.pane(top[0], (width, top_height), top[1], edge=edge) as visible:
        if visible:
            top[2](ctx)
    drag = layout_mod.splitter(f"{split_id}-share", vertical=False, length=handle_length)
    if drag and avail_y > 0:
        if wanted is None:
            share = lay.share(split_id) + drag * tokens_mod.SCALE / avail_y
        else:
            # ``give_way_drag`` leaves the share alone whenever the pane under
            # the handle is pinned by its content and cannot follow the cursor.
            share = layout_mod.give_way_drag(
                avail_y, lay.share(split_id), wanted, below_floor, drag * tokens_mod.SCALE
            )
        previous = lay.share(split_id)
        lay.set_share(split_id, share)
        if lay.share(split_id) != previous:
            lay.save()
    if middle is not None:
        name, role, draw_fn, height = middle
        with layout_mod.pane(name, (width, height), role, edge=edge) as visible:
            if visible:
                draw_fn(ctx)
    with layout_mod.pane(bottom[0], (width, 0), bottom[1], edge=edge) as visible:
        if visible:
            bottom[2](ctx)
    imgui.end_group()


def _right_column(
    ctx: Any,
    lay: Any,
    sidebar_w: float,
    *,
    inspector_draw: Callable[[Any], None],
    library_draw: Callable[[Any], None],
    share_key: str = "create-inspector",
) -> None:
    """The right sidebar: inspector on top, library on bottom.

    Kept as its own name because it is the one column a test drives directly
    -- it is :func:`_split_column` with Create's two panes filled in, and the
    geometry the frame draws is therefore the geometry the test measures.
    """
    from .. import layout as layout_mod

    _split_column(
        ctx,
        lay,
        split_id=share_key,
        handle_length=sidebar_w,
        width=sidebar_w,
        edge=layout_mod.PaneEdge.LEFT,
        top=("inspector", layout_mod.PaneRole.INSPECTOR, inspector_draw),
        bottom=("library", layout_mod.PaneRole.SIDEBAR, library_draw),
    )


def _column_boundary(library: Any, workspace: str, side: str, *, length: float = 0.0) -> None:
    """The draggable boundary between a side column and the centre anchor.

    ``length`` is the handle's height, and forwarding it is not optional for a
    workspace that keeps a row under its columns. Left at 0 the splitter takes
    ``get_content_region_avail().y`` -- **the whole remainder**, not the height
    the columns were given -- so the handle, and not the columns, is what sets
    the row's height. Muse is the workspace that found this: it shortens its
    columns to leave 148 dp for the player strip, the splitter went on claiming
    the full height anyway, and the strip was pushed 8 px past the bottom of
    the content region, where ``begin_child`` returns false and draws nothing.
    """

    from imgui_bundle import imgui

    from .. import layout as layout_mod

    imgui.same_line()
    layout_mod.column_splitter(library, workspace, side, length=length)
    imgui.same_line()


def _stage_pane(ctx: Any) -> None:
    """The settings column's body at the Create stage the user is standing on.

    A module-level function rather than a method for ``_right_column``'s
    reason: it needs no ``self``, and a test that wants to know every stage
    draws should call the dispatch the frame calls rather than a hand-copied
    reimplementation of it -- which is how the fifth stage comes to be missing
    from one of the two.

    The fall-through is Reference: the front of the pipeline, and the only
    stage that says something with nothing selected at all.
    """
    from imgui_bundle import imgui

    from .. import icons, widgets
    from ..modes.create.ui import workspace as generation_workspace
    from ..modes.create.ui.panes import settings_2d, settings_3d
    from ..panes import inspector, pose_panel, stage_rig

    stage = ctx.state.create.stage
    # 2026-09-07 Create review, item 5.7: progress used to be visible only
    # from the Reference stage's canvas tray, so a remesh or a rig bake
    # started from its own stage showed nothing here but the floating card
    # until it finished. Reference keeps its own copy out of this --
    # ``generation_workspace.draw`` no longer draws it either, since
    # Reference already states a running job twice more (the plan block's
    # "Queue: ..." line and the floating card) and did not need a third.
    if stage != "reference" and generation_workspace.progress_row(ctx):
        imgui.separator()
    if stage == "mesh":
        settings_3d.draw(ctx)
    elif stage == "rig":
        stage_rig.draw(ctx)
    elif stage == "pose":
        job = ctx.job()
        if job is None:
            widgets.empty_state(
                icons.PERSON_STANDING, "No mesh selected.", "Pick a rigged mesh to pose it."
            )
        else:
            pose_panel.draw(ctx, job, hosted=True)
    elif stage == "export":
        job = ctx.job()
        if job is None:
            widgets.empty_state(
                icons.DOWNLOAD, "Nothing selected.", "Pick an asset to take files from it."
            )
        else:
            # The inspector's own grid, called rather than copied: it is the
            # one answer to "what can I take away from this", and a second
            # version of it is a second place for an artifact to be missed.
            inspector.downloads(ctx, job)
    else:
        settings_2d.draw(ctx)


def _takes_pointer(target: Any, hovered: bool) -> bool:
    """The one hover/grab rule, for all three viewports.

    A viewport sees the mouse while the pointer is over it, and a gesture
    already in progress keeps it wherever the cursor goes -- so crossing onto
    a panel mid-orbit does not drop the drag. Written three times (the asset
    viewer, Clay's and Poser's) it drifted: only Clay's carried the
    ``tab.saving`` press gate, which is a *different* rule and stays where it
    is, beside the document it is about.

    ``grabbing`` first, falling back to ``dragging``: the 2026-09-11 audit's
    clay-02 renamed ``ClayView``'s "any grab -- orbit, pan, marquee, gizmo,
    keydrag -- is live" property to ``grabbing`` so it stopped shadowing
    ``DragOps.dragging``'s narrower "a transform is running" meaning. This
    rule wants the broad one, which is ``grabbing`` for Clay and, since the
    asset viewer and Poser never had the shadowing property, plain
    ``dragging`` for them.
    """

    if target is None:
        return hovered
    grabbing = getattr(target, "grabbing", None)
    if grabbing is None:
        grabbing = target.dragging
    return bool(hovered or grabbing)


def _ui_scale(settings: Any) -> float:
    """The stored multiplier, snapped to a step. Junk must not brick the window.

    **Snapped rather than clamped**, since the zoom control became a combo of
    named steps: a settings file written by the slider that used to stand there
    carries values like 1.13x, and honouring one would run the app at a size the
    Appearance pane can no longer show, explain, or offer a way back from. The
    monitor's own scale is not known here -- this is read before the window
    exists -- so the snap is against the unbounded step list and the product
    clamp in ``tokens.set_scale`` still has the last word.
    """
    from .. import tokens as tokens_mod

    try:
        value = float(settings.get("ui_scale") or 1.0)
    except (TypeError, ValueError):
        return 1.0
    return tokens_mod.nearest_ui_scale(value)


class FrameMixin:
    """The frame loop and the whole-window UI dispatch, mixed into
    :class:`~.app.App`.

    The shell names it reaches are imported *inside* the methods that use
    them: ``main`` imports :class:`~.app.App` (which assembles this mixin) to
    build the class, so a module-scope import back would be a cycle. Same
    shape as ``clay_viewport.ClayViewport``.
    """

    # -- the loop, one tick of it -------------------------------------------

    def _tick(self) -> float:
        import time

        now = time.perf_counter()
        dt = min(now - self._last_frame, 0.25)
        self._last_frame = now
        self.fps.record(dt)
        self._memory_ticker(now)
        self._health_ticker(now)
        return dt

    def _skip_idle_frame(self) -> bool:
        """Whether this loop pass may go by without a redraw (B11).

        Gates the *whole* frame -- events, cache tick, UI build, render -- on
        whether anything could visibly change. Any pending input renders
        immediately (the events are peeked, never consumed); otherwise a frame
        is due at IDLE_FPS whenever something live is on screen, and the rest
        of the time only at IDLE_FPS anyway -- the conservative shape: being
        wrong about "idle" costs at most 1/IDLE_FPS of latency, never an
        event.
        """
        import time

        from ..main import IDLE_FPS

        if time.perf_counter() - self._last_frame >= 1.0 / IDLE_FPS:
            return False
        return not self._frame_active()

    def _frame_active(self) -> bool:
        """Anything that wants the full TARGET_FPS cadence right now."""
        import pygame
        from imgui_bundle import imgui

        if pygame.event.peek():
            return True
        ctx = self.app_ctx
        if ctx is None:
            return True
        io = imgui.get_io()
        if io.want_text_input:
            return True  # the caret blinks
        # An animation in flight is a reason the screen can change with no
        # input at all -- which is exactly what the rest of this list
        # enumerates. It counts only keys the last frame actually touched, so a
        # widget that left the screen mid-move cannot hold the app at 60 fps
        # (``motion.animating``), and it is constantly false under
        # reduce-motion, where nothing is ever mid-move.
        from .. import motion

        if motion.animating():
            return True
        state = ctx.state
        if state.toasts:
            return True  # TTL fade
        # A job running or queued animates the progress card and its easing.
        if self.runtime.current_job_id is not None or ctx.cache.active is not None:
            return True
        # Any task in flight draws spinners/progress somewhere.
        if ctx.tasks.busy_keys:
            return True
        viewer = self.viewer
        if viewer is not None and (
            viewer.pending is not None
            or viewer.stripping
            or viewer.camera.auto_rotate
            or not viewer.camera.settled()
        ):
            return True
        clay = self.clay_view
        if clay is not None and state.mode == "clay" and not clay.camera.settled():
            return True
        poser = self.poser_viewer
        if poser is not None and state.mode == "poser" and not poser.camera.settled():
            return True
        # Troupe plays its sheet with no input at all, and ``advance`` only
        # runs inside the preview's draw -- so a skipped frame does not advance
        # playback, it *drops* it. Throttled to IDLE_FPS the preview becomes
        # coarse catch-up jumps that can step straight over the frame being
        # judged, which is the one thing the mode exists to make obvious.
        if state.mode == "troupe" and getattr(state.troupe, "playing", False):
            return True
        # Sirens for the same reason, and it was missing: the playhead is drawn
        # from the mixer's clock and nothing else moves, so at IDLE_FPS the row
        # cursor crawled down the pattern at 12 fps while the audio ran at full
        # speed -- the one readout that says *where in the song you are*,
        # visibly disagreeing with what you can hear.
        #
        # **Muse, word for word.** Its player draws a playhead from the same
        # clock across the same kind of picture, and nothing else on that
        # screen moves either. The two audio modes share one predicate because
        # they share one argument.
        if state.mode in ("sirens", "muse"):
            from ..modes.sirens import audio as sirens_audio

            if sirens_audio.playing():
                return True
        inker = state.inker
        tab = None if inker is None else inker.active
        return tab is not None and bool(getattr(tab, "playing", False))

    def _memory_ticker(self, now: float) -> None:
        """Log host memory every MEMORY_TICK_SECONDS.

        The single line that discriminates the two candidate causes of the
        2026-08-03 commit exhaustion. Stage-boundary logging (queue._log_mem)
        only fires when a job runs, so it cannot distinguish "each job leaks a
        little" from "the process grows while sitting idle". This samples
        regardless, so the shape of the curve is in the log either way.

        Cheap enough for the frame loop: two ctypes calls once per 30 s.
        """
        from ... import memlog, winjob
        from ..main import MEMORY_TICK_SECONDS

        if now - self._last_memory_log < MEMORY_TICK_SECONDS:
            return
        self._last_memory_log = now
        # Child commit is included: Warlock's subprocesses are not incidental
        # (the BiRefNet matting worker measured 6.5 GiB of private commit on
        # 2026-08-21), and a line reporting only ``private`` understated this
        # app's charge against the commit limit by 40% on the session that
        # prompted the reading.
        #
        # Over ``measured_pids()`` rather than ``tracked()``: the latter holds
        # the pids ``Popen`` returned, which under a uv venv are trampolines
        # rather than the interpreters holding the weights
        # (dev/measurements/2026-08-22-trampoline-child-pids.md).
        summary = memlog.summary(children=winjob.measured_pids())
        if summary is not None:
            # The frame rate rides along on the same line: a session that dies
            # without unwinding leaves no teardown summary, and memory and
            # smoothness are the two things worth reading against each other.
            rate = f" | {self.fps.fps:.1f} fps" if self.fps.frames else ""
            log.info("host idle-tick: %s%s", summary, rate)

    def _health_ticker(self, now: float) -> None:
        """Keep the header health dot honest after startup.

        The dot reads ``runtime.checks``; before this poller it showed the
        startup snapshot forever -- unplug the disk or orphan the trellis port
        mid-session and the dot stayed green. The probe runs on a task thread
        (it binds a socket and stats a disk), and the submit is paced here so
        a task is not queued sixty times a second; ``cached_checks``' own TTL
        makes a stray extra call cheap rather than harmful.
        """
        from warlock.service import system as svc_system

        if now - self._last_health_poll < svc_system.HEALTH_TTL:
            return
        self._last_health_poll = now
        self.app_ctx.submit("health", svc_system.current_checks, self.app_ctx.svc)

    def frame(self, dt: float) -> None:
        from imgui_bundle import imgui

        from .. import imgui_backend, modes
        from ..main import _background

        self.app_ctx.state.frame_index += 1
        # The caption tracks every unsaved document, not only a pose, so it is
        # sampled per frame rather than pushed from one callback. Cheap: it
        # returns immediately unless the answer changed.
        self._sync_title()
        self.app_ctx.textures.begin_frame()
        # Here rather than in ``_tick``: every path that draws a frame goes
        # through this method, and ``_tick`` belongs to the run loop alone --
        # the screenshot harness calls ``frame`` directly, and a meter that
        # was blank in every shipped picture is how that was found. Gated on
        # the setting, so the opt-out costs nothing at all; the sampler itself
        # is two ctypes calls and a driver ioctl, 0.047 ms measured, behind a
        # one-second cadence (see ``resources.Sampler.sample``).
        if self.app_ctx.state.show_resources:
            # The frame rate is handed over rather than measured there: the
            # meter is the frame loop's, and ``resources`` is pinned free of
            # pygame. ``None`` before the first recorded frame -- which is
            # every screenshot-harness frame, since the harness calls ``frame``
            # directly and never ``_tick`` -- so the segment is simply absent
            # rather than reading a confident 0.
            self.resources.tick(fps=self.fps.fps if self.fps.frames else None)
        self._collect_tasks()
        # Right here and nowhere else: this is the frame thread, the only one
        # allowed to touch a document or the GL context, which is exactly
        # what a queued agent tool call needs to do (``agent_host.AgentHost``'s
        # own module docstring). ``None`` until ``setup_context`` builds the
        # host -- see ``self.agent_host``'s own comment for why teardown and
        # this both guard on it rather than assume it.
        if self.agent_host is not None:
            self.agent_host.pump()
        self._refresh()
        # Before ``_events``, which is where the keys are read: whether the
        # arrows reach imgui at all is a property of the surface they arrive
        # at, so it has to be settled for this frame before any of them is
        # dispatched (UX-02).
        imgui_backend.reserve_nav_keys(self.app_ctx.state.mode in modes.NAV_KEY_MODES)
        self._events()

        import pygame

        io = imgui.get_io()
        io.delta_time = max(dt, 1e-4)
        # Set every frame rather than only on resize: a window that starts
        # minimised, or a display scale change, reaches imgui no other way, and
        # a zero display size is an assertion rather than a blank frame.
        io.display_size = pygame.display.get_window_size()
        io.display_framebuffer_scale = (1.0, 1.0)
        # K99, and the position is the whole of it: rebuilding the atlas
        # invalidates every ImFont handle, and those are pushed and popped all
        # through ``_build_ui``. Between frames is the only safe moment, so the
        # scale slider sets a flag and this consumes it.
        if self.app_ctx.state.fonts_dirty:
            from .. import fonts

            self.app_ctx.state.fonts_dirty = False
            try:
                fonts.reload(imgui)
            except Exception:
                # Mid-session, from the UI-scale slider, and *not* fatal. The
                # atlas either kept the faces it had (the files-missing check
                # runs before ``clear_fonts``) or is empty and imgui falls back
                # to its own default font, which is the state every headless
                # test already runs in. Taking the session down over a type
                # ramp -- with an unsaved document open in every editor -- is
                # the wrong trade by a wide margin.
                log.exception("could not re-bake the font atlas")
                self.app_ctx.toast(
                    "The interface font could not be reloaded at this size.",
                    "error",
                    "log",
                )
        imgui.new_frame()
        self._build_ui()
        imgui.render()
        # After the frame, because ``want_text_input`` is only true once the
        # field that wants it has been drawn (UX-19). SDL emits no TEXTINPUT
        # while text input is stopped, so this is what makes typing work --
        # and stopping it again is what keeps an IME's candidate window off
        # the viewport while nobody is typing.
        imgui_backend.sync_text_input()

        self.ctx.screen.use()
        self.ctx.clear(*_background())
        self.imgui_renderer.render(imgui.get_draw_data())
        # After the render and before the flip: what is on the default
        # framebuffer now *is* the composed frame, which is the whole reason
        # the translucent surfaces sample a captured frame rather than asking
        # for the draw list to be split in two (UX.md Phase 5). It captures
        # nothing on a frame where a floating surface sampled it, so a panel
        # never blurs itself.
        from .. import vibrancy

        vibrancy.capture(self.ctx, io.display_size)
        # Not while a button is held: the debounced flush is a JSON encode of
        # the whole settings document plus an atomic file write, and a splitter
        # drag dirties the layout on every frame it moves -- so the write
        # landed once a second *inside* the drag, on the frame thread, as a
        # hitch under the pointer. Deferred to release, where the same flush
        # happens once. ``flush`` on exit covers a drag that ends the session.
        if not imgui.is_any_mouse_down():
            self.app_ctx.settings.tick()
        # One toast per problem, polled rather than pushed: ``Settings`` is a
        # plain file object with no way to reach the UI, and both of the things
        # it has to report -- a file that could not be read at startup, and one
        # that cannot be written now -- used to be log lines nobody saw while
        # every preference silently reverted or stopped persisting (UX-10).
        notice = self.app_ctx.settings.take_notice()
        if notice is not None:
            self.app_ctx.toast(notice, "error")

    # -- frame steps that are drawing, not landing --------------------------

    def _refresh(self) -> None:
        from ..modes.review import mode as review_mode

        ctx = self.app_ctx

        # A2: the read (one sqlite query plus a per-row ``attach_files`` stat
        # over the whole window) used to run inline here, on the frame thread,
        # every single frame this is called from -- the dominant library cost
        # at a few thousand assets. ``request`` only submits it to a task
        # thread when a refresh is actually due; the result lands later, on
        # whatever frame ``_on_task_done`` sees the "jobs-list" key finish, and
        # is adopted there. Nothing here mutates ``ctx.cache`` at all now.
        ctx.cache.request(ctx.tasks, self._announce_job_transition)
        # Outside the tick: the request may have been made by a verdict on a
        # frame the list did not re-read, and a refused submit has to be
        # retried on some later frame rather than on the next list refresh.
        review_mode.pump_findings(ctx)
        # Same shape, same reason: a burst of image labels must not train once on
        # the set as it stood at the first keypress.
        review_mode.pump_judge(ctx)
        # And once more: scoring is a DINOv2 pass per unit, so it is a task, and
        # the request following a retrain is the one with nothing after it.
        review_mode.pump_scores(ctx)
        self._check_worker()
        # Every mode, not only Inker: a crash while the user is looking at the
        # library still loses the painting. ``submit`` refuses a key already in
        # flight, so a slow encode skips a beat rather than queuing.
        #
        # **This import is eager, and the comment that called it lazy was
        # wrong** (the review's theme T5, settled 2026-09-03 by saying so).
        # ``_refresh`` runs every frame, so every mode module is imported on
        # frame 1 whether or not its mode is ever opened -- and it would be
        # even without this line, because ``journal.snapshot`` below reaches
        # ``ensure_providers``, which imports all six to register their kinds
        # before the first recovery scan. Gating on ``state.inker is not None``
        # would therefore save nothing at all while adding a condition to read.
        # What is genuinely lazy is ``ensure``'s *state*, not this import.
        from .. import journal
        from ..modes.inker import mode as inker_mode

        # Every registered document kind, not only Inker (UX-05). Importing
        # ``inker_mode`` is what registers its provider; the other modes
        # register theirs the same way, lazily, so a session that never opens
        # Clay pays for nothing.
        journal.pump(self.app_ctx)
        # Beside it, and in every mode for the same reason: an export flattens
        # one frame per app frame rather than a whole clip on the frame the
        # button was clicked, and a user who started one and switched to the
        # library must still get their file.
        inker_mode.pump_export(self.app_ctx)
        # And beside it: the history drops its oldest steps when they get too
        # big to hold, and the press that did it is routinely the last thing
        # the user does in Inker before switching away.
        inker_mode.pump_undo_trim(self.app_ctx)
        # Scanned here, on the first frame that has a Ctx, and *offered* by the
        # home screen rather than by a modal in front of it. It has to be here
        # and not in the pane: the autosave directory is also where this
        # session's copies land, so a scan taken any later than the first frame
        # would hand the user their own open documents back. ``snapshot`` is a
        # no-op after the first call, which is what makes calling it per frame
        # correct rather than merely cheap.
        journal.snapshot(ctx)

    # -- the whole-window UI dispatch ----------------------------------------

    def _build_ui(self) -> None:
        from imgui_bundle import imgui

        from .. import layout as layout_mod
        from .. import menus, modes, rail
        from .. import tokens as tokens_mod
        from ..main import _SINGLE_PANE_MODES
        from ..modes.library.ui.panes import library
        from ..modes.settings.ui.panes import app_settings
        from ..panes import bottom_pane, inspector, landing

        ctx = self.app_ctx
        # The rail first of all, because the sidebars are fitted against what
        # is left after it: a rail measured afterwards would leave the columns
        # disagreeing with the window by exactly its own width for one frame
        # every time it was toggled.
        rail.tick(self.layout)
        # Before any pane reads ``layout.SIDEBAR_W``: a width change eases, and
        # a half-eased width read by the left sidebar and the settled one read
        # by the right would be two columns disagreeing about the same frame.
        layout_mod.tick()
        # Straight after, and before any column is drawn: how wide a sidebar
        # can be is a fact about this frame's window, and the left column must
        # not settle it for itself and leave the right one to find out (UX-01).
        mode_for_layout = ctx.state.mode
        self.layout.bind_workspace(self.layouts, mode_for_layout)
        # No ``fixed_left`` any more: it pinned Inker's left column to the
        # toolbox rail's 90 px, and the rail is gone -- both of Inker's columns
        # are ordinary sidebars whose widths are the arrangement's to state.
        # (``fit_widths`` keeps the parameter: it is the general answer for a
        # column that is a fixed size rather than a preference, and deleting it
        # would have to be re-derived by the next workspace that wants one.)
        layout_mod.measure(self.layouts, mode_for_layout)
        # Recomputed every frame by whoever draws the viewport image. Every
        # mode but 3D returns without drawing it, so it stays false there and
        # the viewer gets no events at all.
        self._viewport_hovered = False
        # The dragged asset (I83) mirrors imgui's own drag state rather than
        # being cleared by whoever accepts it: a drag released over nothing
        # accepts nowhere, and a flag only the drop target clears would leave
        # every slot outlined for the rest of the session.
        if imgui.get_drag_drop_payload_py_id() is None:
            ctx.state.dragging_job = None
        # Arriving in a viewport mode is a change the cache will not announce:
        # the job list has not changed, so nothing else would ask the viewer
        # to show what was just picked.
        #
        # A *stage* change is the same event since wave 5, and has to be
        # watched separately: Reference and Mesh are one mode now, so stepping
        # between them moves no mode at all -- and what the viewport should be
        # showing (``input.png`` against ``model.glb``) changed anyway.
        stage_moved = ctx.state.create.stage != self._last_stage
        self._last_stage = ctx.state.create.stage
        if (
            ctx.state.mode != self._last_mode or stage_moved
        ) and ctx.state.mode in modes.VIEWPORT_MODES:
            self._sync_viewer()
        # And on a *selection* change, which is the trigger UX-03 found missing.
        # ``_sync_viewer`` was driven off the cache reread (a 3 s idle timer)
        # and off mode transitions, so clicking a card updated ``state.selected``
        # -- and therefore the inspector -- immediately while the viewport went
        # on showing the previous asset for up to three seconds. The inspector
        # described B and the viewport drew A, with nothing on screen saying so,
        # which makes an export or a compare decision untrustworthy.
        if ctx.state.selected != self._last_selected:
            self._last_selected = ctx.state.selected
            if ctx.state.mode in modes.VIEWPORT_MODES:
                self._sync_viewer()
        if self._last_mode is not None and ctx.state.mode != self._last_mode:
            # The content crossfade (UX.md Phase 1). One place, zero per-pane
            # work: the mode switch's pill already slides, and before this the
            # screen under it teleported. Not on the *first* frame -- there is
            # no previous screen to have come from, and the splash's own fade
            # already owns that moment.
            self._start_transition(tokens_mod.DUR_BASE)
        if ctx.state.mode != self._last_mode and ctx.state.mode == "review":
            # Arriving is the one moment a rescan is certainly wanted, and it
            # is a mode change rather than a job-cache tick, so nothing else
            # would ask. Driven off the change and not off "the list is empty",
            # which would submit a walk of the bench directory every frame on a
            # machine that has never run a sweep.
            from ..modes.review import mode as review_mode

            review_mode.scan(ctx)
        if ctx.state.mode != self._last_mode and ctx.state.mode == "poser":
            # Review's rule: arriving refreshes the library and asks for the
            # armature preview, both cheap on a warm cache.
            from ..modes.poser import mode as poser_mode

            poser_mode.enter(ctx)
        self._note_last_workspace(ctx)
        self._last_mode = ctx.state.mode

        viewport = imgui.get_main_viewport()
        imgui.set_next_window_pos(viewport.work_pos)
        imgui.set_next_window_size(viewport.work_size)
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_bring_to_front_on_focus.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.menu_bar.value
        )
        imgui.begin("##host", None, flags)
        # One frame, one record of what stopped drawing. Above the three clears
        # below rather than beside them, which visibly breaks that block: the
        # menu bar draws first *and* is itself guarded, so its census has to be
        # empty before it runs.
        guard.begin_frame(ctx)
        # One stable command surface in every mode.  The menu rows are adapters
        # over the same command/operation registries used by Ctrl+K and keys.
        guard.run("shell/menus", menus.draw, ctx, self.layout, title="The menu bar")
        # One frame, one record of where every pane ended up -- and one answer
        # to "is the layout editor open", which every splitter reads (P5.4).
        from .. import layout_edit

        layout_mod.begin_frame(layout_edit.ensure(ctx.state).open)
        # And one record of where every *control* that a tour can point at
        # ended up. Cleared here rather than in ``layout`` so the two clears
        # are visibly the same decision, made once, in one place.
        anchors.begin_frame()
        # And -- on a probe run only -- one record of every control the frame
        # submits, for the driver that clicks them. Same clear, same place, for
        # the same reason: a stale census points at whatever took the control's
        # place.
        probe.begin_frame()
        # And -- once every ten seconds rather than once a frame -- the
        # interpolator forgets the keys nothing is asking for any more. Here
        # rather than in ``motion`` itself because this is the one place that
        # knows a frame has started, which is the same reason the four clears
        # above it live here.
        from .. import motion as motion_mod

        motion_mod.sweep()
        # The rail is drawn in every mode, Home included: it is how you leave
        # wherever you are, so a mode that hides it is a dead end.
        guard.run("shell/rail", rail.draw, self, ctx, title="The mode rail")
        # Shell utility popups are opened at host scope. Menu actions can
        # originate in child windows, while imgui resolves a popup in the
        # window that opens it, so they communicate through one-shot requests.
        if rail.take("layouts"):
            imgui.open_popup("layouts")
        # Under ``guard`` like every other surface: these three draw at host
        # scope, so a raise inside one took the whole frame down rather than
        # one pane's worth of it -- and the layouts popup is where finding 1 of
        # this review's section 2 shipped from.
        guard.run(
            "shell/layouts",
            self._layouts_popup,
            ctx,
            title="Workspace layout",
            draw_placeholder=False,
        )
        # Kept behind an explicit developer environment flag; normal installs
        # never gain a design-system destination in their navigation.
        from .. import component_gallery

        guard.run(
            "shell/gallery",
            component_gallery.draw,
            title="Component gallery",
            draw_placeholder=False,
        )
        # Ctrl+/ and the palette's "Keyboard shortcuts" both set this flag,
        # because neither a key handler nor a palette command is inside the
        # window the popup is registered in. It was consumed by the header's
        # ``?`` button; the header is gone, so it is consumed here.
        if ctx.state.shortcuts_requested:
            ctx.state.shortcuts_requested = False
            # Cleared on the way in rather than on the way out: a popup can be
            # dismissed by clicking anywhere, which is not a moment this has a
            # hook in, and reopening onto last time's query would look like a
            # list that had lost most of its rows.
            self._shortcuts_query = ""
            imgui.open_popup("shortcuts")
        guard.run(
            "shell/shortcuts",
            self._shortcuts_popup,
            title="Keyboard shortcuts",
            draw_placeholder=False,
        )
        imgui.same_line()
        # Treat the workspace and its status as one vertical item beside the
        # full-height rail. Without this group imgui advances below the taller
        # rail before drawing the status, clipping it against the host edge.
        imgui.begin_group()
        # A negative child height leaves its magnitude below the child, but
        # the next item also consumes the parent's item spacing. Reserve both
        # so the shared status line is never clipped at the host's lower edge,
        # especially when that spacing is doubled by UI scale.
        status_reserve = tokens_mod.sp(bottom_pane.reserve(ctx)) + imgui.get_style().item_spacing.y
        imgui.begin_child("##content", (0, -status_reserve))
        from ..panes import overlay

        mode = ctx.state.mode
        # Tier two of the same net, and the reason the two tails below became
        # one: a single guarded region needs a single exit, so the duplicated
        # end_child/status/end_group/end/overlays that each branch used to
        # carry is now written once. The mark is taken inside ``##content``,
        # so a failure in the scaffolding *between* panes -- the groups and
        # columns no ``layout.pane`` covers -- costs the workspace and leaves
        # the rail, the menu bar and the status line live. That is the
        # difference between a broken workspace and a dead end.
        with guard.surface("shell/content", title="The workspace") as live:
            if live:
                overlay.doctor_banner(ctx)
                if mode in _SINGLE_PANE_MODES or mode in modes.WORKSPACE_MODES:
                    if mode == "home":
                        landing.draw(ctx)
                    elif mode == "settings":
                        app_settings.draw(ctx)
                    elif mode == "library":
                        # The full-window composition (the UI redesign, wave 4.4), not a
                        # second card list: ``library_full`` draws the *same* filters,
                        # the same cards' actions and the same inspector, arranged for
                        # a window rather than for a 300 px sidebar. The library itself
                        # is still one implementation -- this module composes it.
                        from ..modes.library.ui.panes import full as library_full

                        library_full.draw(ctx)
                    elif mode == "clay":
                        self._clay_workspace()
                    elif mode == "poser":
                        self._poser_workspace()
                    elif mode == "review":
                        self._review_workspace()
                    elif mode == "plotter":
                        self._plotter_workspace()
                    elif mode == "packwright":
                        self._packwright_workspace()
                    elif mode == "muse":
                        self._muse_workspace()
                    elif mode == "sirens":
                        self._sirens_workspace()
                    elif mode == "troupe":
                        self._troupe_workspace()
                    elif mode == "mason":
                        self._mason_workspace()
                    else:
                        self._inker_workspace()
                else:

                    # The library used to share the left sidebar with settings, split by
                    # settings_share; it shares the right sidebar with the inspector now
                    # instead, so the left column is settings alone (nothing left to split
                    # against) and the right column is the two-scroller stack that used to
                    # live on the left.

                    lay = self.layout
                    left_w = layout_mod.sidebar_width("left")
                    right_w = layout_mod.sidebar_width("right")
                    # The rail and the brief share one row now (2026-09-07),
                    # drawn through one pane rather than the rail bare above a
                    # second one: two vertical strips (~90 dp together) for
                    # what a common visit reads as one control bar -- where
                    # this asset is, and what to make next. ``create_brief``'s
                    # own module docstring carries the rest of the argument;
                    # this is only the wiring, and ``create_brief.shows`` no
                    # longer decides whether the pane opens -- the rail is a
                    # breadcrumb for every stage, so it always does, and
                    # ``create_brief.bar_height`` sizes it per stage instead.
                    with layout_mod.pane(
                        "brief",
                        (0, create_brief.bar_height(ctx)),
                        layout_mod.PaneRole.CONTENT,
                        edge=layout_mod.PaneEdge.BOTTOM,
                        title="The brief bar",
                    ) as visible:
                        if visible:
                            create_brief.draw(ctx, self._stage_rail)
                    with layout_mod.pane(
                        "settings",
                        (left_w, 0),
                        layout_mod.PaneRole.SIDEBAR,
                        edge=layout_mod.PaneEdge.RIGHT,
                    ) as visible:
                        if visible:
                            _stage_pane(ctx)

                    _column_boundary(self.layouts, "create", "left")
                    self._viewport_pane()
                    _column_boundary(self.layouts, "create", "right")

                    _right_column(
                        ctx,
                        lay,
                        right_w,
                        inspector_draw=inspector.draw,
                        library_draw=library.draw,
                    )

        imgui.end_child()
        guard.run("shell/status", bottom_pane.draw, ctx, title="The bottom pane")
        imgui.end_group()
        imgui.end()
        self._overlays(viewport)

    def _stage_rail(
        self,
        ctx: Any,
        *,
        max_width: float | None = None,
        row_height: float | None = None,
    ) -> None:
        """Create's breadcrumb, over the three columns.

        The pane dispatch that follows it reads ``state.create.stage``, and
        this is the only control that writes one -- through
        ``create_stages.go``, which is what makes "switching stage may move the
        selection" a rule rather than a thing this happens to remember.

        Handed to ``create_brief.draw`` as a callable (2026-09-07: the rail
        and the brief now share one row, drawn through ``create_brief``) --
        ``max_width`` and ``row_height`` are that row's own give-way ladder
        and vertical alignment, computed there and passed straight through to
        ``create_rail.stage_rail``. This method still owns building ``items``
        and still owns the one call to ``create_stages.go``; nothing about
        *that* moved.

        Not one of the six ``<mode>_workspace.py`` modules P4 split out: it is
        Create's own breadcrumb, and Create is not one of the modes this wave
        gives its own workspace file (P5's split, not this one's). It sits
        here, beside :func:`_stage_pane`, because that is the one other piece
        of Create's three-column layout this module already draws.
        """
        from imgui_bundle import imgui

        from ..modes.create.ui import rail as create_rail
        from ..modes.create.ui import stages as create_stages
        from ..panes import inspector

        job = ctx.job()
        # The two pieces of evidence no job row carries. ``rig_meta`` is the
        # inspector's own mtime-cached read of ``rig.json`` -- the same call it
        # makes for the weighting line, so the rail costs no extra stat --
        # and ``poses`` is what ``_refresh_rig_side_data`` fetched off-thread.
        meta = inspector.rig_meta(ctx, job) if job is not None else None
        poses = ctx.state.preview.get("poses")
        items = [
            (
                stage,
                create_stages.LABELS[stage],
                create_stages.ICONS[stage],
                create_stages.available(stage, job, ctx),
            )
            for stage in create_stages.STAGES
        ]
        picked = create_rail.stage_rail(
            "create-stages",
            items,
            ctx.state.create.stage,
            # A set of ticked segments, not the single furthest one
            # (2026-09-07 Create review, item 3.5): ``reached`` stops at the
            # first stage a job has not got to, which left Rig, Pose and
            # Export dark forever on a finished prop with no rig -- Export
            # sits behind the two it can never pass. ``ticked`` walks the
            # same table but skips a failing *optional* stage instead of
            # stopping on it.
            done=create_stages.ticked(job, meta, poses),
            optional=create_stages.OPTIONAL_HINTS,
            max_width=(imgui.get_content_region_avail().x if max_width is None else max_width),
            row_height=row_height,
        )
        anchors.mark("create/stages")
        if picked != ctx.state.create.stage:
            create_stages.go(ctx, picked)

    def _viewport_pane(self) -> None:
        from imgui_bundle import imgui

        from .. import layout as layout_mod
        from ..modes.create.ui import stages as create_stages
        from ..modes.create.ui import workspace as generation_workspace
        from ..panes import overlay
        from ..tokens import sp

        ctx = self.app_ctx
        # Leave room for the inspector; the progress card floats over the image
        # now, so the full height is the image's.
        width = layout_mod.centre_width()
        # no_scroll_with_mouse: over the viewport the wheel can only mean dolly.
        with layout_mod.pane(
            "viewport",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=imgui.WindowFlags_.no_scroll_with_mouse.value,
        ) as visible:
            if visible:
                overlay.toolbar(ctx)
                avail = imgui.get_content_region_avail()
                height = max(avail.y, 64)
                reference_stage = create_stages.at(ctx.state, "reference")
                # Once a Create run exists, the canvas gains an in-context
                # results tray.  The viewer remains above it, so a reference
                # can still be judged at useful scale while progress and the
                # next variation stay in the same creative loop.
                tray = reference_stage and generation_workspace.should_draw(ctx)
                # The floor is **one whole card**, not a round number: heading,
                # caption, a 72 dp thumbnail and the two rows of actions under
                # it come to a little over 200 dp, and at the old 180 the
                # actions sat below the tray's fold on every card. A button
                # nobody can reach without scrolling a strip they cannot see
                # scrolls is a button that does nothing; ``/exercise-mode
                # create`` reported twelve of them, and no test can, because a
                # clipped button is still drawn.
                tray_height = min(sp(320), max(sp(232), height * 0.36)) if tray else 0.0
                gap = imgui.get_style().item_spacing.y if tray else 0
                canvas_height = max(height - tray_height - gap, sp(64))
                if tray:
                    # ``placeholder`` centres itself by consuming the available
                    # height, so it needs its own top child; otherwise it would
                    # consume the tray's room before the tray is drawn.
                    if imgui.begin_child(
                        "generation-canvas", (0, canvas_height), False,
                        imgui.WindowFlags_.no_scroll_with_mouse.value,
                    ):
                        if self.viewer.reference is not None:
                            self._draw_reference(width, canvas_height)
                        else:
                            overlay.placeholder(ctx)
                    imgui.end_child()
                elif not reference_stage and self.viewer.has_model:
                    self._draw_viewport_image(imgui.get_cursor_screen_pos(), width, height)
                elif reference_stage and self.viewer.reference is not None:
                    self._draw_reference(width, height)
                else:
                    overlay.placeholder(ctx)
                if tray:
                    imgui.separator()
                    generation_workspace.draw(ctx, tray_height)

    def _draw_viewport_image(self, pos: Any, width: float, height: float) -> None:
        from imgui_bundle import imgui

        from .. import widgets

        ctx = self.app_ctx
        # AppState.select clears the flag but cannot reach the viewer, so the
        # split's GPU half is reconciled here -- otherwise a selection change
        # mid-compare leaves the stale mesh rendering a full second scene draw
        # every frame with nothing on screen showing it.
        if not ctx.state.comparing and self.viewer.comparing:
            self.viewer.exit_compare()
        halves = 2 if ctx.state.comparing else 1
        cell = (width - (8 if halves == 2 else 0)) / halves
        texture = self.viewer.render((pos.x, pos.y, cell, height), imgui.get_io().delta_time)
        # UV flipped: GL's origin is bottom-left and imgui's is top-left.
        imgui.image(widgets.texture_ref(texture), (cell, height), (0, 1), (1, 0))
        self._viewport_hovered |= imgui.is_item_hovered()
        if halves == 2 and self.viewer.compare_viewport is not None:
            imgui.same_line()
            imgui.image(
                widgets.texture_ref(self.viewer.compare_viewport.texture),
                (cell, height),
                (0, 1),
                (1, 0),
            )
            self._viewport_hovered |= imgui.is_item_hovered()

    def _draw_reference(self, width: float, height: float) -> None:
        from imgui_bundle import imgui

        from .. import widgets
        from ..panes import overlay

        texture = self.viewer.reference
        # UVs past 1.0 with the sampler set to repeat: one draw call, the
        # inker canvas's checkerboard idiom, rather than N**2 images that would
        # have to be positioned by hand and would show a seam of their own
        # wherever the arithmetic left a sub-pixel gap.
        repeat = 1
        if self.app_ctx.state.create.tile_preview and overlay.shows_tiled(
            self.app_ctx, self.app_ctx.job()
        ):
            repeat = overlay.TILE_REPEAT
        # Set on *both* branches, every frame. Turning the toggle on used to be
        # a one-way door: the sampler was switched to GL_REPEAT and never put
        # back, so the single-tile view that followed sampled a wrapped texture
        # at its own edges -- which is the one place a seamless tile is not
        # seamless, since bilinear filtering there blends the far edge in.
        # Idempotent and cheap: moderngl skips the GL call when the value is
        # already what it is being set to.
        texture.repeat_x = texture.repeat_y = repeat > 1
        scale = min(width / texture.size[0], height / texture.size[1])
        imgui.image(
            widgets.texture_ref(texture),
            (texture.size[0] * scale, texture.size[1] * scale),
            (0.0, 0.0),
            (float(repeat), float(repeat)),
        )

    # -- toasts and modals, and the transition veil over them ----------------

    def _overlays(self, viewport: Any) -> None:
        """Toasts and modals, drawn over whichever layout ran.

        Outside the host window and after it ends, because a modal is its own
        window: the landing screen needs them as much as the workspace does,
        which is why this is not inline in either.
        """
        from .. import widgets
        from ..modes.create.ui.panes import settings_3d
        from ..modes.troupe.ui.panes import send as troupe_send
        from ..panes import first_run, overlay, palette

        ctx = self.app_ctx
        # The layout editor, over the workspace that has just recorded its pane
        # rects and *outside* every pane -- see ``layout_edit``'s docstring for
        # why that is a construction rather than a habit.
        from .. import layout_edit

        over = functools.partial(guard.run, draw_placeholder=False)
        over(
            "overlay/layout-editor",
            layout_edit.draw,
            self,
            ctx,
            viewport,
            title="The layout editor",
        )
        over("overlay/fps", overlay.fps_meter, ctx, self.fps, title="The frame meter")
        if ctx.state.mode != "home":
            over(
                "overlay/progress",
                overlay.progress_card,
                ctx,
                self.eta,
                title="The progress card",
            )
        from ..panes import bottom_pane

        over(
            "overlay/toasts",
            widgets.toasts,
            ctx.state,
            (viewport.work_size.x, viewport.work_size.y),
            on_action=self._toast_action,
            bottom_offset=tokens.sp(bottom_pane.reserve(ctx)),
            title="Notifications",
        )
        # The first-run question owns the screen before any workflow modal.
        # Its two exits close it permanently, then later questions can use the
        # one popup slot on the following frame.
        over("overlay/first-run", first_run.draw, ctx, title="The first-run panel")
        if first_run.is_open(ctx):
            self._transition_overlay(viewport)
            return
        # Before the confirms, because it is the same kind of thing and the
        # earlier one wins the single modal slot imgui gives a frame.
        over("overlay/matte", settings_3d.matte_modal, ctx, title="The cutout dialog")
        # The Manual, over whatever ran above (the UI redesign, wave 3). Before the
        # palette on purpose: Ctrl+K is how you leave anywhere, this included,
        # so it has to float above the reference rather than under it.
        from ..manual import render as manual_render

        over("overlay/manual", manual_render.draw_overlay, ctx, title="The manual")
        # Above the confirms it can raise (Delete asks): the palette closes
        # itself in the same frame it runs a command, so the question it asks
        # takes the modal slot on the frame after, with nothing to contend
        # with.
        # The tour, over the Manual so a "Read more" does not bury the card
        # that offered it, and under the palette for the Manual's own reason:
        # Ctrl+K is how you leave anywhere, this included.
        from ..panes import tour as tour_pane

        over(
            "overlay/tour",
            tour_pane.draw,
            ctx,
            title="The guided tour",
            on_failure=lambda: tour_pane.stop(ctx),
        )
        over("overlay/palette", palette.draw, ctx, title="The command palette")
        # The Send to Troupe question, above the confirms for ``matte``'s
        # reason: it is a modal raised from a context menu, and it must take the
        # single popup slot before any confirm the same frame raises.
        over(
            "overlay/troupe-send",
            troupe_send.draw,
            ctx,
            title="Send to Troupe",
            on_failure=lambda: troupe_send.close(ctx),
        )
        # A queue that stops drawing still reports ``modal_open``, so the
        # keyboard would be owned by a modal nobody can see. Dismissing is the
        # queue's own documented way out, and the reason ``pending`` is
        # read-only.
        over(
            "overlay/confirms",
            ctx.confirms.draw,
            title="A confirmation",
            on_failure=ctx.confirms.dismiss,
        )
        over(
            "overlay/prompts",
            ctx.prompts.draw,
            title="A prompt",
            on_failure=ctx.prompts.dismiss,
        )
        # Last, and on the foreground list, so it covers everything above --
        # including the modals, which are part of the screen being crossfaded.
        self._transition_overlay(viewport)

    # -- transitions -------------------------------------------------------
    #
    # One full-viewport veil in the window's own background colour, fading
    # out. It is not a crossfade between two rendered screens -- imgui has one
    # framebuffer and keeping the previous frame's would be Phase 5's offscreen
    # copy -- it is the cheap half of one, and against a near-black ground the
    # difference is not visible at 200 ms. It paints only; the UI underneath
    # stays live, which is why a transition can never eat a click.

    TRANSITION_KEY = "app/transition"

    def _start_transition(self, duration: float) -> None:
        from .. import motion

        self._transition_duration = duration
        motion.restart(self.TRANSITION_KEY)

    def _transition_overlay(self, viewport: Any) -> None:
        from imgui_bundle import imgui

        from .. import motion, theme

        duration = self._transition_duration
        if duration <= 0.0:
            return
        t = motion.ease(self.TRANSITION_KEY, duration)
        if t >= 1.0:
            # Latched off rather than re-eased every frame for the life of the
            # session: ``ease`` on a finished key is cheap but not free, and a
            # veil at alpha 0 is still a full-viewport quad in the draw list.
            self._transition_duration = 0.0
            return
        low = viewport.pos
        high = (low.x + viewport.size.x, low.y + viewport.size.y)
        imgui.get_foreground_draw_list().add_rect_filled(
            (low.x, low.y), high, imgui.get_color_u32(theme.rgba(theme.BG, 1.0 - t))
        )
