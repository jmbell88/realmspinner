"""The App class: construction, the three setup phases, and the pygame loop.

The P4 restructure split ``studio/main.py`` (5,971 lines) into
``shell/{app,frame,events,tasks,quit}.py``, plus one module per mode for the
six inline ``_*_workspace`` methods; this is the module that assembles the
whole of it. :class:`App` is built from thirteen mixins (fourteen before P9,
2026-09-18 folded Troupe's own ``TroupeWorkspace`` into Poser's stage rather
than a workspace of its own): the four that already lived beside ``main.py``
(``ClayViewport``, ``MasonViewport``, ``PoserViewport``, ``ReviewPanes``, each
"this repository's idiom for a body of drawing that belongs to the shell",
per ``studio/modes/clay/ui/viewport.py``'s own module docstring);
:class:`FrameMixin`, :class:`TasksMixin`, :class:`EventsMixin` and
:class:`QuitMixin`, which ``shell/frame.py``, ``shell/tasks.py``,
``shell/events.py`` and ``shell/quit.py`` define; and the five workspace
mixins this same wave gave their own module -- ``InkerWorkspace``,
``PlotterWorkspace``, ``MuseWorkspace``, ``SirensWorkspace`` and
``PackwrightWorkspace``, one file each, named the way
``ClayViewport``/``MasonViewport`` already were. ``self`` in every mixin's
method is this one assembled class, whichever file the method's body happens
to live in.

``StartupRefused`` lives here rather than in ``main.py``: it is the named
failure ``App.run`` raises and ``_run_locked`` catches by type, so it belongs
beside the class whose setup phases raise it, not beside the process entry
that only re-shows it in a dialog.

The shell names this module reaches are imported *inside* the methods that use
them, ``studio/modes/clay/ui/viewport.py``'s own rule restated: ``main`` imports :class:`App`
from here to build one, so a module-scope import back to ``main`` would be a
cycle. The mixins themselves, and the four pane classes wave one already
split out, are the one exception -- assembling the class needs them at
import time, not at some later call.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any

from .. import fps as fps_mod
from .. import resources
from ..modes.clay.ui.viewport import ClayViewport
from ..modes.inker.ui.workspace import InkerWorkspace
from ..modes.mason.ui.viewport import MasonViewport
from ..modes.muse.ui.workspace import MuseWorkspace
from ..modes.packwright.ui.workspace import PackwrightWorkspace
from ..modes.plotter.ui.workspace import PlotterWorkspace
from ..modes.poser.ui.viewport import PoserViewport
from ..modes.review.ui.workspace import ReviewPanes
from ..modes.sirens.ui.workspace import SirensWorkspace
from .events import EventsMixin
from .frame import FrameMixin
from .quit import QuitMixin
from .tasks import TasksMixin

log = logging.getLogger(__name__)


class StartupRefused(Exception):
    """A named startup failure, with the sentence the user should read.

    ``_run_locked`` shows any exception out of ``App.__init__`` in a native
    dialog, which is already better than the log line it used to be -- but
    "AttributeError: 'NoneType' object has no attribute 'clear'" is a dialog
    the user still cannot act on. The two failures below are recoverable in
    principle and worth explaining, so they carry their own words.
    """

    def __init__(self, title: str, body: str) -> None:
        self.title = title
        self.body = body
        super().__init__(f"{title}: {body}")


class App(
    FrameMixin,
    TasksMixin,
    EventsMixin,
    QuitMixin,
    ClayViewport,
    MasonViewport,
    PoserViewport,
    ReviewPanes,
    InkerWorkspace,
    PlotterWorkspace,
    MuseWorkspace,
    SirensWorkspace,
    PackwrightWorkspace,
):
    def __init__(self, runtime: Any) -> None:
        from ..main import MIN_SIZE

        self.runtime = runtime
        self.svc = None
        self.ctx = None
        self.window = None
        # Read in setup_window (the window size is in it) and consumed in
        # setup_context, which the splash now runs between.
        self.settings = None
        self._monitor_scale = 1.0
        self.imgui_renderer = None
        self.viewer = None
        self.app_ctx = None
        # The MCP listener for driving Clay from an external agent. ``None``
        # until ``setup_context`` builds it (it needs ``app_ctx`` and the GL
        # context to exist first) -- ``frame`` and ``teardown`` both guard on
        # that with ``getattr``/``is not None`` rather than assuming it, for
        # the same reason ``clay_view`` is guarded: teardown also runs after
        # a setup that failed before reaching this line.
        self.agent_host = None
        self.eta = None
        self._running = False
        # H02: set while a quit was asked for during a pack install's commit
        # phase and withheld rather than shown as a confirm dialog -- see
        # ``_ask_quit`` and ``_resume_deferred_quit``. A dialog the user
        # confirms is a promise this code can act on immediately; the commit
        # phase is the one moment that promise cannot be kept, so the ask
        # itself is deferred rather than the answer.
        self._quit_deferred = False
        self._min_size = MIN_SIZE
        self._last_frame = time.perf_counter()
        self._started_at = self._last_frame
        # Seeded to 0.0, not to now: the first tick fires immediately and puts
        # a startup baseline in the log to measure every later sample against.
        self._last_memory_log = 0.0
        self._last_health_poll = 0.0
        # A dead worker is reported once. The banner is dismissible, and
        # re-raising it every frame would make it impossible to dismiss.
        self._fatal_reported = False
        #: The caption's current state, so only a change is sent to the window
        #: manager. ``None`` until the first sync, which therefore always runs.
        self._title_marked: bool | None = None
        # The mode the last frame was built in, so a change into a viewport
        # mode can resync the viewer -- a mode change is not something the
        # job cache announces.
        self._last_mode: str | None = None
        # The Create stage the last frame was built in, for ``_last_mode``'s
        # reason and because the merge removed the mode change that used to
        # stand in for it: Reference -> Mesh is now one mode with a different
        # file under the viewport.
        self._last_stage: str | None = None
        # What the viewport was last asked to show. A selection change is not
        # announced by the cache, so without this the viewer only caught up on
        # the 3 s idle reread and the inspector described one asset while the
        # viewport drew another (UX-03).
        self._last_selected: str | None = None
        # How long the veil over the whole viewport takes to clear, or 0.0 for
        # "there is no transition running". A duration rather than a bool
        # because the two things that raise one want different lengths: a mode
        # switch is DUR_BASE, the splash dissolving into the app is DUR_SLOW.
        self._transition_duration = 0.0
        # Measured every frame, drawn only when state.show_fps is on (F10), and
        # logged once at teardown regardless -- the overlay answers "is it
        # smooth now", the log line is the evidence for "it ran at 60".
        self.fps = fps_mod.FpsMeter()
        # Task keys that have already been reported as arriving with nowhere to
        # go, so the report is once per key rather than once per arrival. See
        # the tail of ``_on_task_done``.
        self._unclaimed: set[str] = set()
        # One sampler for the app, because the CPU figure is a delta between
        # calls: two owners sharing a baseline would each eat the other's
        # interval. Ticked from ``_tick`` at one second, drawn by
        # ``menus._draw_status_group`` (the meter reads it via
        # ``status_bar.resource_item``).
        self.resources = resources.Sampler()
        # Set by _draw_viewport_image, read one frame later by _events. The
        # host window is fullscreen, so io.want_capture_mouse is always true
        # and cannot be the gate; imgui's own hover test on the viewport image
        # is, and it correctly goes false under popups and active widgets.
        self._viewport_hovered = False
        # Clay's own viewport, built on first use for the reason its
        # state is: a session that never opens Clay should not pay for a
        # renderer, a framebuffer and three gizmos.
        self.clay_view = None
        # Which Clay tab the one viewport camera currently belongs to. See
        # ``_clay_viewport``: the camera is per document and the viewport is not.
        self._clay_camera_tab = ""
        # Clay's own hover flag, set by the pane that draws its image, for the
        # reason _viewport_hovered exists: the host window is fullscreen, so
        # io.want_capture_mouse is always true and cannot be the gate.
        self._build_hovered = False
        # Mason's own viewport, its per-tab camera tracking and its hover
        # flag -- ``clay_view``'s own three fields, restated for the reason
        # they are: built on first use, per document rather than per
        # viewport, and read off the render image the pane draws.
        self.mason_view = None
        self._mason_camera_tab = ""
        self._mason_hovered = False
        # Poser's own Viewer and hover flag, for Clay's reasons: built on first
        # entry (a session that never poses pays for no second renderer), and
        # a separate instance so loading the armature preview can never call
        # ``adopt_model``'s unconditional ``exit_pose_mode`` over an inspector
        # pose session on the shared viewer.
        self.poser_viewer = None
        self._poser_hovered = False
        # What is typed into the shortcuts popup's own filter box (UX.md Phase
        # 4). Not persisted and cleared on every open: it is a way through
        # sixty rows, not a preference about them.
        self._shortcuts_query = ""

    # -- setup -------------------------------------------------------------

    # There is deliberately no ``setup()`` composing the three phases. It
    # existed briefly and had no caller: ``run`` drives the phases itself so it
    # can draw a splash over the middle one. What it did have was users in the
    # tests, which stubbed ``app.setup`` to isolate ``run`` -- and once ``run``
    # stopped calling it those stubs silently became no-ops, so ``run`` fell
    # through into the real ``setup_window``, initialised pygame for real, and
    # left a live display behind that broke every GL test that ran after it (74
    # errors, none of them in the code that changed). A convenience method with
    # no caller is not free; this one cost a seam the tests were relying on.

    def setup_window(self, *, size_override: tuple[int, int] | None = None) -> None:
        """Everything that needs the main thread and the one GL context.

        Fast, and first: this used to run *after* ``runtime.start()``, so the
        slow half of startup -- doctor, the worker, the out-of-process bpy
        probe -- happened with no window on screen at all. Splitting it out is
        what lets the splash be drawn over the rest.
        """
        import moderngl
        import pygame
        from imgui_bundle import imgui

        from .. import dpi, fonts, guard, imgui_backend, theme, tokens, widgets
        from .. import layouts as layouts_mod
        from ..layout import Layout
        from ..main import WINDOW_TITLE, _desktop_size, _min_window_size, _window_size
        from ..settings import Settings
        from ..viewer_embed import Viewer
        from .frame import _ui_scale

        # Read before the runtime exists: it is a file under the configured
        # data directory, which the Config already knows, and the window size
        # it carries is needed by set_mode below.
        settings = Settings.load(self.runtime.config.data_dir)
        self.settings = settings

        # Before the window exists: awareness is frozen at window creation.
        dpi.make_process_dpi_aware()

        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        icon_path = Path(__file__).resolve().parent.parent.parent / "assets" / "icon.ico"
        if icon_path.is_file():
            pygame.display.set_icon(pygame.image.load(str(icon_path)))
        # The window is in physical pixels (Per-Monitor-V2): a persisted size
        # is already physical, and the first-run default scales by the primary
        # monitor so 1600x950 means the same amount of screen everywhere.
        first_run_scale = dpi.system_scale()
        size = _window_size(
            settings.get("window_size"),
            override=size_override,
            first_run_scale=first_run_scale,
            desktop=_desktop_size(pygame),
        )
        try:
            self.window = pygame.display.set_mode(
                size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
            )
        except Exception as exc:
            # The GL attributes above ask for a 3.3 core context, and SDL
            # refuses here rather than degrading if the driver cannot give one.
            # It arrived as the generic "could not start" box, which says
            # nothing a user can act on -- and this is one of the few startup
            # failures that genuinely has a remedy.
            raise StartupRefused(
                "Warlock Studio needs OpenGL 3.3",
                f"{exc}\n\nWarlock draws its whole interface on the GPU and "
                "could not create an OpenGL 3.3 window.\n\nThis usually means "
                "the graphics driver is missing or out of date. It can also "
                "happen over Remote Desktop or in a virtual machine, where the "
                "session offers no hardware OpenGL.",
            ) from exc
        if not size_override:
            # ``size`` above is the client area only; the title bar and frame
            # SDL adds on top of it push the outer window further down/right
            # than that clamp can see, which is how Familiar's Build/Send row
            # ended up under the taskbar with a client size that "fit"
            # (2026-09-16). Skipped under ``size_override`` -- the screenshot
            # harness asked for an exact framebuffer, and moving the window
            # would not change that, but a stray SetWindowPos on a headless
            # CI runner is one more thing that could fail for no reason.
            #
            # The resulting VIDEORESIZE is drained by the splash loop, which
            # deliberately does not persist a size, so the fitted size is not
            # stored -- it does not need to be: the fit is re-applied on every
            # launch and is a no-op once the window already fits, and imgui
            # reads ``get_window_size`` fresh each frame. Startup only: a
            # window the user later drags under the taskbar is left alone.
            # Silent on purpose: ``get_wm_info`` has no ``"window"`` off
            # Windows or under a stand-in video driver, and a window left one
            # taskbar too low is the pre-fix state, never worth a startup crash.
            with contextlib.suppress(Exception):
                dpi.fit_window_to_work_area(pygame.display.get_wm_info()["window"])
        pygame.display.set_caption(WINDOW_TITLE)
        # Dropped files are how a reference image gets in without a dialog.
        pygame.event.set_allowed(None)

        # The scale everything is drawn at, before any font or style is built:
        # the monitor's own scale, and the user's multiplier on top of it. The
        # multiplier is folded in *here* rather than applied later so the font
        # atlas is baked at the size it will be drawn at; changing it in the
        # settings pane rescales everything immediately and re-bakes the atlas
        # between frames (K99).
        monitor_scale = dpi.window_scale(pygame)
        tokens.set_scale(monitor_scale * _ui_scale(settings))
        # Before ``theme.apply`` below, which copies the palette into imgui's
        # style (M105). An unknown stored name falls back to dark rather than
        # raising: a settings file written by a build with a third palette must
        # not stop the window opening.
        tokens.set_theme(str(settings.get("theme") or "dark"))
        self._min_size = _min_window_size(monitor_scale)

        try:
            self.ctx = moderngl.create_context()
        except Exception as exc:
            # The window opened and the context still cannot be adopted, which
            # is a narrower fault than the one above -- a driver that advertises
            # 3.3 and does not deliver it. Same remedy, so the same sentence,
            # said separately because the two fail at different lines and a
            # log-reader should be able to tell them apart.
            raise StartupRefused(
                "Warlock Studio needs OpenGL 3.3",
                f"{exc}\n\nWarlock opened a window but could not use the "
                "OpenGL context behind it.\n\nThis usually means the graphics "
                "driver is missing or out of date. It can also happen over "
                "Remote Desktop or in a virtual machine, where the session "
                "offers no hardware OpenGL.",
            ) from exc
        imgui.create_context()
        imgui.get_io().set_ini_filename("")  # imgui's own layout file is not ours to keep
        # Before anything can raise inside a frame. imgui's error-recovery
        # assert defaults *on*, and under imgui-bundle an IM_ASSERT surfaces as
        # a RuntimeError -- so left alone, the unwind that saves a broken pane
        # is itself the exception that ends the session. See ``studio/guard.py``.
        guard.configure()
        # Keyboard navigation, app-wide (UX-02). It was off, and the shortcut
        # sheet made support look broader than it was: Settings, Profiles, the
        # library, the inspector and the mode switch had no focus traversal at
        # all, so a keyboard-only user could reach the two forms ``focus.py``
        # hand-rolls an order for and nothing else.
        #
        # Safe to switch on now only because ``imgui_backend`` arbitrates the
        # arrows and Space, which five surfaces already bind -- see
        # ``_NAV_KEYS`` there for the rule and why it lives at that door.
        imgui.get_io().config_flags |= imgui.ConfigFlags_.nav_enable_keyboard.value
        # Click a drag widget and type into it, with no modifier held.
        # Typed entry has always worked on every slider and drag here --
        # ``controls._clamp_typed_entry`` puts ``clamp_on_input`` on all of
        # them and ``controls.TYPED_ENTRY_HINT`` says so in every one of their
        # tooltips -- but a hover tooltip is a place to *confirm* an
        # affordance, not to discover one, and a user who has not hovered has
        # no reason to guess that Ctrl is the key. This costs the drags
        # nothing: a drag still drags, and only a press that moved no pixels
        # opens the box. Sliders keep Ctrl+click, since a click on a slider
        # already means "jump the handle here" and imgui offers no flag to
        # have it both ways.
        imgui.get_io().config_drag_click_to_input_text = True
        try:
            fonts.load(imgui)
        except fonts.FontsUnavailable as exc:
            # These ship in the wheel -- the offline invariant covers fonts as
            # much as weights -- so this is a partial install, an antivirus
            # quarantine or a half-copied directory, and every one of those is
            # fixed by reinstalling rather than by reading a stack trace.
            raise StartupRefused(
                "Warlock Studio is missing part of its installation",
                f"{exc}\n\nThese files ship with Warlock and are never "
                "downloaded, so one going missing means the installation is "
                "incomplete -- an interrupted install, or an antivirus tool "
                "that quarantined them.\n\nReinstalling Warlock restores them.",
            ) from exc
        theme.apply(imgui)
        self.layout = Layout(settings)
        # Saved arrangements within the fixed three-column skeleton (wave 5).
        # A separate object from ``Layout``, which owns the *proportions*: one
        # is a preference the user drags and the other is a named thing they
        # switch between, and the settings keys are top-level for the reason
        # ``layouts.py`` states.
        self.layouts = layouts_mod.Library(settings)
        widgets.attach_settings(settings)
        self.imgui_renderer = imgui_backend.ImguiRenderer(self.ctx)
        self.viewer = Viewer(self.ctx)
        self._monitor_scale = monitor_scale

    def setup_runtime(self, note: Any = None) -> None:
        """The slow half, run on a plain worker thread behind the splash.

        Nothing here touches GL or imgui: it opens the store, runs the doctor
        checks, starts the worker's loop thread and probes bpy in a
        subprocess. ``Ctx`` is deliberately *not* built here -- it constructs
        textures, and textures belong to the frame thread's one context.

        ``note`` is the splash's line, and it is optional because this is also
        called with nothing listening -- by the tests, and by any caller that
        starts a runtime without a window.
        """
        self.svc = self.runtime.start(note)

    def setup_context(self) -> None:
        """The Ctx and the state it carries. Frame thread only, after both."""
        from .. import motion, textures
        from .. import widgets as widgets_mod
        from ..app_ctx import Ctx
        from ..jobs_cache import JobsCache
        from ..main import AGENT_SERVER_SETTING
        from ..panes import first_run, model_gate
        from ..panes import tour as tour_pane
        from ..settings import as_list, restore_form
        from ..state import (
            DEFAULT_FORM_3D,
            AppState,
            Eta,
            default_form_2d,
            filters_from_stored,
            set_mode,
            set_mode_gate,
            set_mode_leave,
        )
        from .events import _leave_mode_if_needed, initial_mode

        settings = self.settings
        monitor_scale = self._monitor_scale

        state = AppState()
        # No mode restore *here*: ``AppState.mode`` still defaults to Home, so
        # a settings file that never expresses a startup preference (or says
        # "home" outright) opens exactly where it always did. The one case
        # that restores something else -- ``STARTUP_LAST`` -- is applied below,
        # once ``mode_gate`` exists to answer whether the remembered mode's
        # door is still open (W3.1).
        state.show_fps = bool(settings.get("show_fps"))
        # Absent means on: this defaults to shown, so a settings file written
        # before it existed must not read as "the user turned it off".
        stored_resources = settings.get("show_resources")
        state.show_resources = True if stored_resources is None else bool(stored_resources)
        # Both halves, here rather than at the checkbox: the stored value has to
        # reach ``motion.REDUCED`` before the first frame is built, or the app
        # animates its own startup at somebody who asked it not to.
        state.reduce_motion = bool(settings.get("reduce_motion"))
        motion.set_reduced(state.reduce_motion)
        state.form_2d = restore_form(default_form_2d(), settings.get("form_2d"))
        state.form_3d = restore_form(DEFAULT_FORM_3D, settings.get("form_3d"))
        state.history = [str(entry) for entry in as_list(settings.get("history"))]
        # The filter bar, minus the fields that are views rather than filters:
        # the app always opens on the library, never in the trash. See
        # ``state.VOLATILE_FILTERS``.
        state.filters = filters_from_stored(settings.get("filters"))

        self.app_ctx = Ctx(
            svc=self.svc,
            runtime=self.runtime,
            state=state,
            cache=JobsCache(self.svc),
            tasks=self.runtime.tasks,
            settings=settings,
            viewer=self.viewer,
            textures=textures.ThumbnailCache(self.ctx),
        )

        # Sampled once. The marker is intentionally outside studio_settings:
        # resetting preferences must not turn setup into an annual popup.
        self.app_ctx.first_run = first_run.pending(self.svc.config)
        device = getattr(self.runtime, "device_memory", None)
        self.app_ctx.gpu_name = str(getattr(device, "name", "") or "")
        self.app_ctx.dpi_scale = monitor_scale
        # The status bar reads the sampler off the Ctx; the App owns it,
        # because the CPU figure is a delta and one owner has to tick it.
        self.app_ctx.resources = self.resources
        self.app_ctx.layout = self.layout
        self.app_ctx.layouts = self.layouts
        # Every ``widgets.field_error`` call site gets the Install offer at
        # once, without widgets importing a pane. Bound to this Ctx, so a
        # second App in one process replaces it rather than stacking.
        widgets_mod.set_install_offer(lambda field: model_gate.install_offer(self.app_ctx, field))
        # And the mode gate, at the one door every switch already goes through
        # (H14). Bound the same way and replaced the same way; ``state`` itself
        # must not learn what a Ctx is.
        #
        # ``mode_gate`` rather than ``mode_block``: the refusal and the rail's
        # grey-out have to answer the same question, and since F4 that question
        # includes the dependency pack as well as the weights.
        #
        # Named rather than inlined into ``set_mode_gate``'s call below: W3.1's
        # ``initial_mode`` asks the identical question once, after the model
        # and pack snapshots land, and a second lambda here would be a second
        # spelling of the one door.
        mode_available = lambda key: not model_gate.mode_gate(self.app_ctx, key)[0]  # noqa: E731
        set_mode_gate(mode_available)
        # Bound the same way ``mode_available`` is, right above: a closure
        # over this one ``Ctx``, so a second App in one process replaces it
        # rather than stacking. The check itself is a module-level function
        # in ``shell/events.py`` rather than folded into this lambda, so it is
        # reachable from a test with a fake ctx and no App to boot.
        set_mode_leave(lambda old: _leave_mode_if_needed(self.app_ctx, old))
        self.app_ctx.load_presets = self.load_presets
        self.app_ctx.refresh_rig_data = self._refresh_rig_side_data
        self.eta = Eta()
        self._load_static_answers()
        # W3.1: reopen the last workspace, if Settings says to and the door
        # is still open. After ``_load_static_answers`` and not before --
        # that call is what populates ``model_rows``/``pack_rows``, and
        # ``mode_gate`` reads an unpopulated snapshot as "nothing missing"
        # (``model_gate.missing``'s own contract), which would let a genuinely
        # gated mode straight through on the one frame that matters.
        target = initial_mode(settings, mode_available)
        if target != state.mode:
            set_mode(state, target)
        if self.app_ctx.first_run:
            self.app_ctx.first_run_info = first_run.snapshot(self.app_ctx)
        # Off the frame thread (C32): the walk stats every file under every
        # job directory, and nothing on the first frame needs the number --
        # the library's storage line simply appears when the task lands.
        self._request_storage()
        # Opt-in, and off by default: an app that phones home on every launch
        # without being asked is not the offline app this one claims to be.
        self._request_update_check()
        # Which tours have already been finished. Read here rather than
        # defaulted empty, so Home stops offering one the reader has done.
        tour_pane.restore(self.app_ctx)
        self.viewer.on_pose_dirty = self._on_pose_dirty
        # The agent host is constructed unconditionally, not only when the
        # setting is on: the Settings pane calls ``ctx.agent_host.start()``/
        # ``.stop()`` on this same instance at runtime (toggling must take
        # effect immediately, never on next launch), so there has to be one
        # to call before the setting is ever true. ``start()`` itself is the
        # part that is conditional -- it opens the pipe only when asked.
        from ..agent_host import AgentHost

        self.agent_host = AgentHost(self.app_ctx, self.svc.config.home)
        self.app_ctx.agent_host = self.agent_host
        if bool(settings.get(AGENT_SERVER_SETTING, False)):
            # Deliberately unguarded, because ``start()`` is contracted not to
            # raise: this call sits inside the try whose failure message is
            # "Warlock Studio could not start", so a pipe that will not open
            # must leave the feature off and the launch alone. The reason
            # lands on ``agent_host.failure`` for the Settings pane to show.
            self.agent_host.start()
        # T5: one Threads instance for the process, and its TAB_CLOSED
        # listener registered exactly once through the same door a test
        # calls (``familiar_ui.install``) -- see that function's own
        # docstring for why registration through it, rather than a bare
        # ``TAB_CLOSED.append`` here, is what lets a test prove the app
        # itself wires this rather than the test wiring it.
        from ..assistant import ui as familiar_ui

        familiar_ui.install(self.app_ctx)

    def _load_static_answers(self) -> None:
        """Read the things that cannot change without a restart, once."""
        from ...service import rig as svc_rig
        from ...service import sheets as svc_sheets
        from ...service import system as svc_system

        ctx = self.app_ctx
        # Clay's bridge asks the ctx for this rather than importing App: the
        # render it needs is an offscreen GL draw on the frame thread, which is
        # the App's business and not a pane's. Attached here so the button has a
        # handler from the first frame rather than toasting "not wired up yet".
        ctx.clay_send_to_3d = self._clay_send_to_3d
        ctx.ask_quit = self._ask_quit
        ctx.clear_viewport = self._clear_viewport
        ctx.guidance = svc_system.guidance_catalog(self.svc)
        ctx.sheet_options = svc_sheets.sheet_options()
        self._refresh_model_answers()
        # Off the frame thread: rig_templates asks doctor.blender_check, whose
        # first answer is a seconds-long bpy subprocess probe that no longer
        # runs during startup (C30). The rig controls appear when it lands.
        ctx.rig_default = self.runtime.config.rig_template or ""
        ctx.submit("rig-templates", svc_rig.rig_templates, self.svc)
        ctx.export_dir = str(self.runtime.config.export_dir or "") or None
        # The trellis port check is non-fatal -- the app is perfectly usable
        # without ever running trellis -- but a port already held at startup
        # means an orphaned server from a previous crash, and every 3D job will
        # fail (or, worse, be served by the orphan) until it is stopped. That
        # is worth the same banner a fatal check gets, so it joins them here
        # rather than being promoted to fatal in doctor.
        self._report_failed_checks()

    def _report_failed_checks(self) -> None:
        """Every failing fatal row (and the trellis port) onto the banner.

        Also after each health poll: ``note_error`` deduplicates, so a row
        still failing costs nothing, and a row that turned fatal once torch
        imported -- no CUDA -- used to show only as a Home chip.
        """
        ctx = self.app_ctx
        # ``pending_install`` is excluded, and it is the reason this filter
        # exists in this shape: a fresh install has every model row failing,
        # and banner-ing them meant the first thing a new user saw was a red
        # wash listing downloads they had not made yet. Those rows are offered
        # by the first-run panel and by Home's status row instead.
        failed = [
            c
            for c in self.runtime.checks
            if not c.ok
            and not c.pending_install
            and (c.fatal or c.name == "trellis port")
        ]
        for check in failed:
            ctx.state.note_error(f"{check.name}: {check.detail}")

    def _refresh_model_answers(self) -> None:
        """What the app knows about the weights on disk, recomputed from doctor.

        Called at startup and again whenever a download finishes. It used to run
        only once, and the ctx field it writes was documented as immutable --
        which was true only while nothing in the app could make weights appear.
        The generate combos read ``base_models`` every frame, so a model
        downloaded from the Settings pane has to stop saying "weights missing"
        without a restart.
        """
        from ... import fetch, models, vram
        from ...service import downloads as svc_downloads

        ctx = self.app_ctx
        # Marked rather than hidden when weights are absent: the combo listing
        # every registered model regardless meant picking one whose weights
        # were never downloaded and learning at job-failure time, despite
        # doctor knowing at startup.
        # The prefix comes from fetch.CHECK_PREFIXES rather than a literal:
        # doctor composes the row name through the same table, so a prefix
        # spelled twice would go on matching nothing and mark every downloaded
        # base model present forever.
        prefix = fetch.CHECK_PREFIXES["base"]
        missing = {
            check.name.removeprefix(prefix)
            for check in (self.runtime.checks or [])
            if check.name.startswith(prefix) and not check.ok
        }
        # And a second suffix, for the other reason a listed model cannot run.
        # Computed from the spec through ``vram.fits`` rather than by matching
        # more doctor strings: doctor has nothing to say about VRAM per model,
        # and growing the string-matching above into a second question is how
        # the prefix bug this block's comment describes happened the first time.
        # Guarded on the plan, so a host with no resolved budget sees no badge.
        plan_ = getattr(self.svc, "vram_plan", None)

        def _suffix(spec: Any) -> str:
            if spec.label in missing:
                return " - weights missing"
            if plan_ is not None and vram.fits(plan_, spec) == vram.FIT_NO:
                return " - won't fit this GPU"
            return ""

        # The 2-tuple shape is pinned by the smoke tests and read by every
        # combo: label decoration only, never a third element.
        ctx.base_models = [
            (k, f"{spec.label}{_suffix(spec)}") for k, spec in models.BASE_MODELS.items()
        ]
        # Snapshot rather than iterate live: register_imported_loras/
        # remove_imported_lora can mutate this table from another thread
        # between frames (see models.STYLE_LORAS_LOCK).
        ctx.style_loras = [("", "no style LoRA")] + [
            (k, spec.label) for k, spec in models.style_loras_snapshot().items()
        ]
        # The Settings pane draws this and may not ask the service itself: it
        # is a pane, and ``recommended_base`` needs a resolved Plan. Empty when
        # there is no plan, which is the pane's "say nothing" value.
        ctx.recommended_base_label = (
            models.BASE_MODELS[vram.recommended_base(plan_)].label if plan_ is not None else ""
        )
        try:
            ctx.model_rows = svc_downloads.rows(self.svc)
        except Exception:
            # A settings pane that cannot list its rows is not a reason to fail
            # startup, the same posture the rig-template probe below takes.
            log.exception("could not list the downloadable models")
            ctx.model_rows = []
        # The other half of "what is on this machine": the Python distributions
        # that can read the weights. Refreshed here rather than once at startup
        # for this function's own reason -- a pack installed from the Settings
        # pane makes ``bpy`` or ``torch`` importable while the app runs, and
        # every pack answer in the ctx is derived from a ``find_spec`` this
        # process has already cached.
        from ...service import packs as svc_packs

        try:
            ctx.pack_rows = svc_packs.rows(self.svc)
        except Exception:
            log.exception("could not list the dependency packs")
            ctx.pack_rows = []
        try:
            ctx.packs_to_restore = svc_packs.packs_to_restore(self.svc)
        except Exception:
            log.exception("could not check for packs an upgrade removed")
            ctx.packs_to_restore = []

    # -- the loop ----------------------------------------------------------

    def run(self) -> int:
        import pygame

        from ..main import TARGET_FPS

        # Setup is inside the try: it starts the runtime, so a failure past
        # that point used to skip teardown and leave the store, the loop
        # thread and the worker running.
        #
        # The two phases are reported differently on purpose. A window that
        # never appeared and a window that vanished after twenty minutes are
        # different bugs, and the log line was the only thing that could tell
        # them apart -- when there was one at all. All three setup phases are
        # the *first* of those, splash or no splash: the window being up is
        # not the app being up, and a failure to build the Ctx is still a
        # startup failure.
        in_setup = True
        rc = 0
        crashed = False
        try:
            self.setup_window()
            if self._startup_with_splash():
                self.setup_context()
                in_setup = False
                # The splash dissolves into the app rather than cutting to it
                # (UX.md Phase 1). The same veil a mode switch uses, at the
                # longer duration: the splash's last frame and the app's first
                # are both on the window background, so a veil clearing over
                # the app *is* the crossfade between them -- and the cheap half
                # of one is indistinguishable from the whole against near-black.
                from .. import tokens as tokens_mod

                self._start_transition(tokens_mod.DUR_SLOW)
                self._running = True
                clock = pygame.time.Clock()
                while self._running:
                    if self._skip_idle_frame():
                        # Nothing can change on screen and the idle cadence is
                        # not due yet: sleep one 60 Hz tick without drawing.
                        # Events are only *peeked* here, so none is lost -- the
                        # frame that consumes them runs the moment one arrives.
                        clock.tick(TARGET_FPS)
                        continue
                    dt = self._tick()
                    self.frame(dt)
                    pygame.display.flip()
                    clock.tick(TARGET_FPS)
            else:
                # Closed during the splash. The load was waited out rather
                # than abandoned, so teardown unwinds a whole runtime.
                log.info("closed during startup")
        except Exception as exc:
            from ...db import StoreUnreadable

            if isinstance(exc, StoreUnreadable):
                # shell-03 (2026-09-07 audit): this catch-all used to absorb
                # every setup failure, including a corrupt job database, so
                # ``_run_locked``'s own ``except StoreUnreadable`` -- which
                # offers to start over with an empty index -- never got a
                # turn, and the generic "could not start" dialog fired
                # instead. ``teardown()`` below still runs from ``finally``;
                # re-raising past this handler is what lets the offer be
                # reached.
                raise
            rc = 1
            crashed = True
            if in_setup:
                log.exception("Warlock Studio could not start")
            else:
                log.exception(
                    "the frame loop crashed mid-session after %d frames (%.1f s up)",
                    self.fps.frames,
                    time.perf_counter() - self._started_at,
                )
        finally:
            self.teardown()
            if crashed:
                # *After* teardown, which is the whole reason the flag exists
                # rather than the report living in the except block: the
                # journal's last write goes through the task runner, and this
                # sentence counts what is on disk. Reporting first would tell
                # the user about a copy that had not been taken yet.
                self._report_crash(in_setup)
        return rc

    def _report_crash(self, in_setup: bool) -> None:
        """Say the app crashed, and offer the log (UX-06).

        The window has gone by now -- that is what a crash looks like from
        outside, and it is precisely the problem: the app vanished and the one
        artefact that could explain it was in a file the user had no reason to
        know about. A native box, because the GL context and imgui are gone.

        The journal sentence is computed rather than promised: "your work is
        safe" is only worth saying when it is true, and "nothing was waiting"
        is the honest answer the rest of the time.
        """
        from ... import instance
        from .. import journal

        try:
            # ``self.runtime.config`` -- App has no ``config`` of its own, and
            # reading one here was an AttributeError this except swallowed, so
            # the dialog's "Open the log folder?" could never actually open it.
            data_dir = Path(self.runtime.config.data_dir)
        except Exception:  # noqa: BLE001 -- a crash report must not crash
            data_dir = None
        try:
            note = journal.status_line(self.app_ctx) if self.app_ctx else ""
        except Exception:  # noqa: BLE001
            note = ""
        when = "while starting" if in_setup else "and had to close"
        instance.alert_fatal(
            "Warlock Studio",
            f"Warlock Studio ran into a problem {when}.\n\n"
            + (note + "\n\n" if note else "")
            + "The details are in warlock.log. Open the log folder?",
            log_dir=data_dir,
        )

    def _startup_with_splash(self) -> bool:
        """Draw the logo while ``setup_runtime`` runs. -> keep going?

        The window is up by now, which is the point and also the new risk: the
        X button is live, so a quit has to be handled here, before there is a
        ``Ctx``, a job cache or anything else the ordinary quit path talks to.
        It is honoured by *waiting* -- see ``splash.Startup`` -- because
        abandoning a half-started runtime strands whatever it had already
        opened, and then returning False so ``run`` skips straight to teardown.

        A load that raised is re-raised here rather than reported, so it lands
        in ``run``'s existing "could not start" branch with its own traceback.
        """
        import pygame
        from imgui_bundle import imgui

        from .. import imgui_backend, splash
        from ..main import TARGET_FPS, _background

        started = splash.Startup(lambda: self.setup_runtime(started.note))
        started.start()
        splash.begin_fade()
        logo = splash.load_logo(self.ctx)
        clock = pygame.time.Clock()
        try:
            while not started.finished():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        started.request_quit()
                        continue
                    if event.type == pygame.VIDEORESIZE:
                        # Not persisted: settings are written at teardown from
                        # the Ctx that does not exist yet, and a resize during
                        # a three-second splash is not a preference.
                        sized = (
                            max(event.w, self._min_size[0]),
                            max(event.h, self._min_size[1]),
                        )
                        pygame.display.set_mode(
                            sized, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
                        )
                        continue
                    imgui_backend.process_event(event)
                io = imgui.get_io()
                io.delta_time = 1.0 / TARGET_FPS
                size = pygame.display.get_window_size()
                io.display_size = size
                io.display_framebuffer_scale = (1.0, 1.0)
                imgui.new_frame()
                # The load's own words rather than one fixed sentence (UX.md
                # Phase 4): the hold is at least three seconds and up to ten on
                # a cold start, and "Starting Warlock Studio..." spends all of
                # it saying nothing that was not already obvious from the logo.
                splash.draw(logo, size, started.message)
                imgui.render()
                self.ctx.screen.use()
                self.ctx.clear(*_background())
                self.imgui_renderer.render(imgui.get_draw_data())
                pygame.display.flip()
                clock.tick(TARGET_FPS)
        finally:
            # Two megabytes of decoded pixels, and the backend still holds the
            # object under its GL name -- forget it before the release, or the
            # next texture to be handed that name renders as this logo.
            splash.release_logo(logo, self.imgui_renderer)
        started.raise_if_failed()
        return not started.quit_requested
