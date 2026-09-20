"""The process entry, the startup-geometry helpers, and the constants the
whole shell shares.

Until the P4 restructure (``dev/RESTRUCTURE.md``) this module *was* the shell
-- one 5,971-line file holding the window, the frame loop and every mixin it
assembled. That split moved the class itself, and everything that draws
through it, to ``shell/{app,frame,events,tasks,quit}.py`` and to one module
per mode for the six inline ``_*_workspace`` methods (``inker_workspace.py``
and its five siblings). What is left here is the other half of what a
"process entry" module owns: ``run()`` and the single-instance lock it holds
before anything touches the home directory, session-marker bookkeeping so a
crash that killed the process outright still leaves a record of itself, the
window-geometry arithmetic that has to run *before* a window exists (so it
cannot live on the class that needs a window to be built), and the
cross-cutting constants -- the task-key strings ``_on_task_done`` switches on,
the startup-mode settings keys, the drop-refusal table -- that more than one
shell module reads.

Those constants and helpers are not re-exports of something that moved
elsewhere: they were always defined here, and every shell module that needs
one reaches back for it with a local import inside the function that uses
it (``from ..main import VIEWER_KEY``), the same way ``studio/modes/clay/ui/viewport.py``
already reached into this file for ``TARGET_FPS`` since 2026-09-04, long
before this split existed. Importing :class:`~.shell.app.App` here, so
``run()`` can build one, is that same kind of use rather than a shim: it is
what keeps ``realmspinner.studio.main.App`` a real, constructible attribute of
this module for the installer's smoke import (``installer/build.ps1:331``)
and for the tests that stand one up with no window to boot.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import filetypes, viewer_embed

# ``App`` and ``StartupRefused`` are built in ``shell/app.py`` now, out of the
# frame-loop, task-pump, event-router and quit-chain mixins the P4 restructure
# split this module into. Importing them here -- rather than lazily, inside
# ``_run_locked`` -- is what this module's own docstring means by "a use, not
# a shim": ``main.App`` has to be a real module attribute for the installer's
# smoke import and for a test that monkeypatches it in place
# (``monkeypatch.setattr(main, "App", ...)`` requires the attribute already
# exist), and neither of those callers should have to know the class moved.
from .shell.app import App, StartupRefused

log = logging.getLogger(__name__)

WINDOW_TITLE = "Realmspinner"
# How often the frame loop samples host memory. Long enough to be free, short
# enough that a 30-minute idle session yields 60 points to fit a slope through.
MEMORY_TICK_SECONDS = 30.0
# The task key the selection's GLB is parsed under. One key, so a selection
# moving faster than the disk cannot pile up loads: a refused submit is simply
# retried on the next tick, and a landed result is checked against
# ``viewer.pending`` before it is adopted.
#: ``viewer_embed.LOAD_KEY``, named here where the frame loop's three readers
#: already look. It moved so a mode module can ask for a picture without
#: importing the shell (``viewer_embed.request_reference``).
VIEWER_KEY = viewer_embed.LOAD_KEY
REVIEW_MESH_KEY = "viewer-review"
# The post-download re-probe. Its own key rather than "health"'s, so a slow
# forced verification cannot be mistaken for the periodic poll and dropped by
# key-dedupe while the user is watching for it (UX-09).
VERIFY_KEY = "verify-install"

#: Where a character preview records what it put on screen, so ``_sync_viewer``
#: stands down for it. ``(path, selection)``: the selection is half the key so
#: the pin releases by itself the moment the user picks another asset, rather
#: than holding the viewport on a temporary GLB for the rest of the session.
CHARACTER_PIN = "character_preview_pin"
# create-02 (2026-09-11 audit): the parse/adopt split's own key for a
# character preview's model load. Separate from ``VIEWER_KEY`` because the
# two can be in flight at once -- the selection-driven sync and a build the
# user just pressed "Preview character" for are not the same load.
CHARACTER_PREVIEW_LOAD_KEY = "character-preview-load"

#: Task keys whose **success** has nothing to do on the frame thread, listed so
#: that everything else arriving unclaimed can be reported. Each is silent for
#: its own reason, and the reasons are the point of the list:
#:
#: ``open-log``           ``os.startfile``. The outcome is a window the OS
#:                        opened.
#: ``open-folder:``       ``app_ctx.reveal_in_explorer``/``os.startfile`` on a
#:                        directory, keyed per target so two reveals never
#:                        dedupe against each other. Same shape as
#:                        ``open-log`` -- the outcome is an Explorer window
#:                        the OS opened -- but missing here until the
#:                        2026-09-20 audit, finding shell-06: every "Show in
#:                        Folder"/"Reveal in Explorer" press logged the line
#:                        that exists to report a genuine routing bug.
#: ``open-release-notes`` opens a release URL in the user's browser, an
#:                        unrelated process. Split out from ``open-log`` by
#:                        the same audit's finding shell-04 -- see
#:                        ``run-installer`` below.
#: ``run-installer``      hands the downloaded installer to the shell to run.
#:                        Split out from ``open-log`` (shell-04): it shared
#:                        that key with "Release notes" and "Show in Folder",
#:                        so ``TaskRunner.submit``'s per-key dedupe silently
#:                        dropped whichever of the two buttons Settings ->
#:                        Updates draws side by side was pressed second.
#: ``thumb:``             a card image written to disk. ``ThumbnailCache``
#:                        keys on mtime, so the library picks it up without
#:                        being told.
#: ``derive:``            an artifact derived *inside* the job directory.
#:                        Deliberately not ``save:`` -- ``app_ctx.derive_key``
#:                        says why -- because the user chose no destination
#:                        and "Saved to <internal path>" is a sentence about a
#:                        file they cannot find.
#: ``wrap:``              the wrap preview. The pane re-reads the file it
#:                        asked for.
SILENT_TASK_KEYS = (
    "open-log",
    "open-folder:",
    "open-release-notes",
    "run-installer",
    "thumb:",
    "derive:",
    "wrap:",
)


DEFAULT_SIZE = (1600, 950)
MIN_SIZE = (1100, 700)
# Desired pane widths and their frame-local fit live in layout.py; named
# workspace layouts persist explicit horizontal and vertical splitter edits.
TARGET_FPS = 60
# The Ctrl chords a focused text field owns, spelled as ``pygame.key.name``
# gives them. imgui's own input-text widget binds these to editing the text, so
# they are the one class of modifier chord that must *not* reach the global
# shortcuts while a field has focus. See ``shell.events.EventsMixin._passes_text_field``.
_TEXT_FIELD_CTRL = frozenset({"z", "y", "x", "c", "v", "a"})
# Named rather than spelled as ``pygame.K_F*`` because this module (and every
# shell module that reads it) imports pygame lazily, inside a function, and a
# module-level constant table would drag the window library into every import
# of it.
_FUNCTION_KEYS = frozenset(f"f{n}" for n in range(1, 13))
# The redraw rate while nothing on screen can change (B11): no pending input,
# no job, no toast, no task, cameras settled, nothing playing. Fast enough
# that the first frame after a wake-up condition is never far away, slow
# enough that an idle session stops burning a core and the GPU.
IDLE_FPS = 12
# The modes that fill the host window with one pane. Inker, Clay, Review,
# Plotter and Packwright are not here: each fills it with a three-column
# *workspace* instead, which is ``modes.WORKSPACE_MODES``. Those three
# categories partition ``modes.KEYS`` exactly, and the partition is the guard
# on ``shell.frame.FrameMixin._build_ui``'s dispatch.
#
# The Manual left this tuple when it stopped being a mode (the UI redesign,
# wave 3): it is drawn from ``_overlays`` now, so it has no dispatch branch
# to be reached by.
_SINGLE_PANE_MODES = ("home", "settings", "library")


# What a drop onto the window is allowed to be. The refusal message and every
# accept path have to agree about it (H71) -- and so do the file pickers, which
# is why the list itself lives in ``filetypes`` and this is a name for it
# rather than a copy of it.
DROPPABLE_IMAGES = filetypes.IMAGE_SUFFIXES

#: What a drop is told in a mode that opens no files. **A table, not a chain
#: of branches**, because the chain had a hole: Poser and Troupe were given a
#: refusal on 2026-09-04 (a drop there "fell through to Create's branches
#: below, which would either refuse it by describing a generation form that
#: is not on screen or accept it by switching modes out from under the
#: user"), and Muse, Review and Settings were still falling through on
#: 2026-09-05 -- a PNG dropped on a results tray switched the window to
#: Create. Each sentence names what the mode works on instead (H71's rule).
#: A test holds this table and the document branches of ``_on_drop`` against
#: ``modes.KEYS``, so a new mode cannot be forgotten. Home, Library and Create
#: are the modes a drop *starts* something in and are deliberately absent.
#: Troupe's own row folded into Poser's (P9, 2026-09-18) when the mode did.
DROP_REFUSALS: dict[str, str] = {
    "poser": (
        "Poser opens no files: it edits poses on a rig you already have and "
        "plays the character sheets a render has already produced, both "
        "chosen from its own library."
    ),
    "muse": (
        "Muse opens no files: it makes music from a brief, and a take goes to "
        "Sirens from its own card."
    ),
    "review": "Review opens no files: it grades the assets already in the library.",
    "settings": "Settings opens no files. Drop it on the workspace that reads it.",
}


# The two image-labelling passes, named as the questions they are. Wording is the
# feature here: the same PNG is a *product* in 2D mode and a *blank* on the way to
# trellis, and "good" means opposite things -- a dramatic plate with pillars and a
# cast shadow is a better asset and a worse blank. A reviewer who cannot tell
# which question is on screen labels the average of the two.
_LABEL_TITLES = {
    "reference": "Label: good 2D asset?",
    "blank": "Label: good to reconstruct?",
}
_LABEL_QUESTIONS = {
    "reference": "Judge it as the finished picture: composition, style, drama.",
    "blank": "Judge it as input for the mesh: one subject, plain background, neutral pose.",
}


def _min_window_size(monitor_scale: float) -> tuple[int, int]:
    """The resize floor, in physical pixels.

    The *monitor's* scale and nothing else. ``tokens.SCALE`` also carries the
    user's UI-scale preference, and a zoom says nothing about how many pixels
    the screen has -- multiplying it in made a 2x preference demand a window
    larger than a 1080p display and refuse to be shrunk.
    """
    return (int(MIN_SIZE[0] * monitor_scale), int(MIN_SIZE[1] * monitor_scale))


def _desktop_size(pygame: Any) -> tuple[int, int] | None:
    """The area to clamp the startup window's *client* size to.

    Prefers :func:`dpi.work_area` -- the desktop minus the taskbar (and any
    docked toolbars) -- over ``get_desktop_sizes``'s whole-display size,
    because a client size clamped only to the whole display could still fit
    a window whose bottom edge lands under the taskbar: Familiar's Build/Send
    row did exactly that (2026-09-16). ``get_desktop_sizes`` is the fallback
    for whatever isn't Windows, or where the work-area query itself fails --
    a ceiling on what can be *asked for*, not a promise the window won't sit
    under the taskbar, but better than nothing.

    This is still only half the fix: it bounds the client area passed to
    ``set_mode``, not the outer frame (title bar included) the window ends
    up with. See ``setup_window``'s call to ``dpi.fit_window_to_work_area``
    for the other half, applied once the window -- and therefore its real
    frame size -- exists.
    """
    from . import dpi

    area = dpi.work_area()
    if area is not None:
        _, _, width, height = area
        if width >= 1 and height >= 1:
            return (int(width), int(height))
    try:
        sizes = pygame.display.get_desktop_sizes()
    except Exception:  # pragma: no cover - SDL without a display
        return None
    if not sizes:
        return None
    width, height = sizes[0]
    if width < 1 or height < 1:
        return None
    return (int(width), int(height))


def _window_size(
    stored: Any, *, override: Any, first_run_scale: float, desktop: tuple[int, int] | None
) -> tuple[int, int]:
    """The size to open at: validated, then clamped to the screen.

    Two separate bugs, both of which reached ``pygame.display.set_mode``.

    **The stored value was trusted.** ``Settings.load`` discards the whole file
    on a *version* mismatch, but a single malformed ``window_size`` in an
    otherwise-valid file sailed straight through to ``set_mode`` with no shape
    check and no floor -- and ``MIN_SIZE`` was enforced only on the live resize
    path, which is to say after the window already existed. A junk value there
    is not a cosmetic problem: if ``set_mode`` raises, ``run``'s handler reports
    the crash but never rewrites the key, so it recurs on every launch and a
    non-developer has no way back in. ``shell.frame._ui_scale`` states the rule
    this setting was skipping -- *a junk value must not brick the window* -- so
    this is that rule, applied to the other stored geometry.

    **The default was never checked against the screen.** ``DEFAULT_SIZE``
    scaled by the monitor is 2000x1187 at the 125% Windows recommends for many
    1080p laptops, which does not fit a 1920x1080 panel; the unscaled 1600x950
    does not fit a 1366x768 one at all. Clamping is last so it applies to a
    stored size too -- the display a window was closed on may not be the
    display it reopens on.

    ``override`` wins outright and unclamped: it is the screenshot harness
    asking for an exact framebuffer, and a clamp there would silently produce
    shots of a size nothing asked for.
    """
    if override:
        return (int(override[0]), int(override[1]))
    default = (int(DEFAULT_SIZE[0] * first_run_scale), int(DEFAULT_SIZE[1] * first_run_scale))
    size = default
    try:
        width, height = (int(stored[0]), int(stored[1]))  # type: ignore[index]
    except (TypeError, ValueError, IndexError, KeyError):
        pass
    else:
        floor = _min_window_size(first_run_scale)
        if width > 0 and height > 0:
            size = (max(width, floor[0]), max(height, floor[1]))
    if desktop is not None:
        size = (min(size[0], desktop[0]), min(size[1], desktop[1]))
    # Never zero, whatever the display claimed: ``set_mode((0, n))`` is a
    # fullscreen request to SDL, not a small window.
    return (max(size[0], 1), max(size[1], 1))


#: Settings-file keys for W3.1 ("startup can reopen the last workspace").
#:
#: Two keys rather than one, because "what to do" and "what happened last"
#: answer different questions: ``LAST_WORKSPACE_SETTING`` is written on every
#: mode change regardless of ``STARTUP_MODE_SETTING``'s value, so choosing
#: "Last workspace" mid-session has something to read immediately rather than
#: waiting on a mode that was remembered only while the preference was set.
STARTUP_MODE_SETTING = "startup_mode"
STARTUP_HOME = "home"
STARTUP_LAST = "last"
LAST_WORKSPACE_SETTING = "last_workspace"

#: Whether an external MCP agent may build in Clay and make characters for
#: you, through ``agent_host.AgentHost``. The Settings pane and
#: ``shell.app.App.setup_context`` both read the setting through this constant
#: rather than by spelling the string themselves, so the two cannot drift onto
#: different keys and leave the switch reading one name while the listener
#: asks for another.
AGENT_SERVER_SETTING = "agent_server"


def _step(label: str, fn: Any) -> None:
    """Run one teardown stage; a failure is logged and the unwind continues."""
    try:
        fn()
    except Exception:
        log.exception("teardown: %s failed; continuing", label)


def _background() -> tuple[float, float, float, float]:
    from .theme import BG, rgba

    return rgba(BG)


def _setup_logging() -> None:
    """Console logging plus a rotating file log and a native crash log.

    Until this existed the app wrote no log file at all: basicConfig had only
    the default stream handler, so the VRAM instrumentation in queue.py went to
    a console nobody was watching. The 2026-08-03 memory-exhaustion crash left
    no in-app record whatsoever and had to be reconstructed from Windows event
    logs. Both files below exist so the next occurrence is attributable.

    faulthandler covers what logging cannot: a hard crash in native code
    (torch, CUDA, or the allocator giving up under commit exhaustion) never
    unwinds to a Python `except`, but faulthandler's signal handlers still get
    a traceback out to the fd.
    """
    import faulthandler
    from logging.handlers import RotatingFileHandler

    from ..config import get_config

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        data_dir = get_config().data_dir
        handlers.append(
            RotatingFileHandler(
                data_dir / "realmspinner.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
        )
        # Held open for process life on purpose -- faulthandler writes to this
        # fd from a signal handler at crash time, so it must not be a file
        # object that could be closed or garbage-collected first.
        global _crash_log
        _crash_log = (data_dir / "crash.log").open("a", encoding="utf-8")
        # Written before faulthandler is armed, so any dump below it is
        # attributable: crash.log is appended to across runs, and a bare
        # traceback with no session line above it belongs to nobody.
        _crash_log.write(
            f"=== session {_utc_now()} pid={os.getpid()} realmspinner={_version()} ===\n"
        )
        _crash_log.flush()
        faulthandler.enable(file=_crash_log)
    except OSError:
        # A read-only or missing data_dir is not a reason to refuse to start;
        # console logging alone is what we had before.
        logging.getLogger(__name__).warning("file logging unavailable", exc_info=True)

    # force=True is load-bearing, not defensive: cli.main() used to call
    # basicConfig() before dispatching here, which left the root logger with a
    # handler and made this call a silent no-op -- realmspinner.log was created on
    # every launch and never written to on the one path anybody actually uses.
    logging.basicConfig(
        level=os.environ.get("REALMSPINNER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


_crash_log: Any = None

SESSION_MARKER = "session.marker"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _version() -> str:
    """The installed version, falling back to the packaged constant.

    A thin alias now: the implementation is ``realmspinner.installed_version``,
    because Home asked this module for it and a pane has no other business
    importing the frame loop.
    """
    from .. import installed_version

    return installed_version()


def _install_excepthooks() -> None:
    """Route every uncaught exception through logging before the default hook.

    The app path only. An exception escaping the frame loop went to stderr and
    nowhere else, which is precisely why the 2026-08-04 crash left an empty
    realmspinner.log; a daemon thread dying (``realmspinner-loop``, trellis' stdout
    reader) was even quieter, since nothing prints for those at all.
    """

    def _hook(exc_type, exc, tb):  # type: ignore[no-untyped-def]
        log.critical("uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    def _thread_hook(args):  # type: ignore[no-untyped-def]
        if issubclass(args.exc_type, SystemExit):
            return
        log.critical(
            "uncaught exception on thread %s",
            getattr(args.thread, "name", "?"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    def _unraisable(args):  # type: ignore[no-untyped-def]
        log.critical(
            "unraisable exception in %r",
            args.object,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _hook
    threading.excepthook = _thread_hook
    sys.unraisablehook = _unraisable


def _pid_alive(pid: int) -> bool:
    """Whether `pid` names a live process.

    Never ``os.kill(pid, 0)``: on Windows that signature terminates the target
    rather than probing it.
    """
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes
    from ctypes import wintypes

    _QUERY_LIMITED_INFORMATION = 0x1000
    _STILL_ACTIVE = 259
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, AttributeError):
        return False


def _marker_path() -> Path:
    from ..config import get_config

    return get_config().data_dir / SESSION_MARKER


def _note_previous_session() -> None:
    """Say so in the log when the last session did not reach teardown.

    A crash that kills the process outright leaves no record of itself; it
    leaves this file behind instead. Never raises -- an unreadable data_dir is
    already handled by _setup_logging and must not block startup either.
    """
    try:
        raw = _marker_path().read_text(encoding="utf-8")
    except (OSError, ValueError):
        return
    try:
        data = json.loads(raw)
        pid = int(data.get("pid", 0))
    except (ValueError, TypeError, AttributeError):
        log.warning("previous session marker is unreadable: %r", raw[:200])
        return
    if _pid_alive(pid) and pid != os.getpid():
        # Unreachable in an ordinary launch since the single-instance lock went
        # in: a live second instance is refused before this runs. What can still
        # reach it is a *recycled* pid -- the marker names a process that died
        # and whose number the OS handed to something unrelated -- so the
        # sentence says what is actually known rather than asserting a second
        # Realmspinner the lock has already ruled out (RUN-01).
        log.warning(
            "the previous session's marker names pid %d (started %s), which is "
            "alive; the instance lock was free, so that is almost certainly a "
            "recycled pid rather than another Realmspinner",
            pid,
            data.get("started_at", "?"),
        )
        return
    log.warning(
        "the previous session (pid %d, started %s, realmspinner %s) did not shut down "
        "cleanly -- check crash.log and Windows event 2004",
        pid,
        data.get("started_at", "?"),
        data.get("version", "?"),
    )


def _write_session_marker() -> None:
    try:
        _marker_path().write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at": _utc_now(),
                    "version": _version(),
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        log.warning("could not write the session marker", exc_info=True)


def _clear_session_marker() -> None:
    try:
        _marker_path().unlink(missing_ok=True)
    except OSError:
        log.warning("could not clear the session marker", exc_info=True)


def run() -> int:
    """The ``realmspinner`` entry point."""
    _setup_logging()
    _install_excepthooks()
    log.info(
        "Realmspinner %s starting: pid=%d python=%s argv=%s",
        _version(),
        os.getpid(),
        sys.version.split()[0],
        sys.argv[1:],
    )
    # Before ``migrate`` is even imported, because importing it *performs* the
    # one-time move: a second instance starting mid-copy is RUN-02's window, and
    # the whole point of this lock is to be taken before anything touches the
    # home directory. Also before the store is opened and before the runtime
    # claims the engine port.
    from .. import instance
    from ..config import get_config, source_checkout

    if not source_checkout():
        # Refused at the door with a sentence, rather than left to fail as a
        # missing ``trellis-server.exe`` at the first job (DST-01, D4). Every
        # native path resolves against the repository root, which inside a
        # wheel points under the environment -- and ``vendor/`` is not in the
        # wheel, because the binaries are manual downloads. Saying so is the
        # supported answer; pretending otherwise is not.
        log.error("not a source checkout; refusing to start (see DST-01)")
        instance.alert(
            "Realmspinner must be run from its source checkout",
            "This copy of Realmspinner was installed as a package rather than run "
            "from a checkout, and it cannot find the native binaries it needs: "
            "they live in vendor/ beside the source and are downloaded by hand.\n\n"
            "Use the Realmspinner installer, or clone the repository and run:\n\n"
            "    uv sync --extra studio --extra text2image --extra rig\n"
            "    uv run realmspinner",
        )
        return 1
    try:
        config = get_config()
    except Exception as exc:
        # The first thing in this process that touches the disk, and until now
        # the only unguarded one. ``get_config`` runs ``migrate.run`` and then
        # ``mkdir``s four directories, so a home on a disconnected network
        # share, a read-only drive or a path the user has no rights to raises
        # ``OSError`` **here** -- before the window, before GL, before imgui,
        # and (under ``pythonw``) with stderr pointed at the null device. The
        # app simply did not appear, twice in a row, with nothing anywhere but
        # a log file in a directory that is itself the problem.
        #
        # ``instance.alert`` is the right tool and was already used twice in
        # the twenty lines above: it needs no window, no GL context and no
        # imgui, which is exactly the situation this is.
        log.exception("could not prepare the Realmspinner home directory")
        instance.alert(
            "Realmspinner cannot use its home directory",
            f"{exc}\n\nRealmspinner keeps its library, job database and settings in "
            "a home directory it creates on first run, and it could not "
            "prepare that directory.\n\nCheck that the drive is connected and "
            "writable, or point REALMSPINNER_HOME at a directory you own.",
        )
        return 1
    lock = instance.InstanceLocks(instance.lock_paths(config))
    unsafe_lock = os.environ.get("REALMSPINNER_ALLOW_UNSAFE_LOCK") == "1"
    if not lock.acquire(allow_unsafe=unsafe_lock):
        # A dialog, not a log line. The behaviour this replaces wrote a warning
        # into realmspinner.log and carried on, so the second instance went on to
        # share the job database and the engine port with the first -- and
        # could terminate the first's trellis server -- with nothing on screen
        # to say why anything was going wrong (RUN-01).
        if lock.failure:
            log.error("instance locking failed for %s; refusing to start", lock.path)
            instance.alert(
                "Realmspinner cannot protect its data",
                f"{lock.failure}\n\nRealmspinner stopped before opening the library because "
                "running without this protection can corrupt jobs or model files. "
                "Fix the directory permissions and try again. For emergency recovery "
                "only, set REALMSPINNER_ALLOW_UNSAFE_LOCK=1.",
            )
            return 1
        log.error("another Realmspinner instance holds %s; refusing to start", lock.path)
        instance.alert(
            "Realmspinner is already running",
            "Another Realmspinner is using this home, job database, or model "
            "directory.\n\nOnly one can use those resources at a time: sharing "
            "them can corrupt jobs or model installs.\n\nClose the other window "
            "and try again. A second copy needs a different REALMSPINNER_HOME, "
            "REALMSPINNER_DB, and REALMSPINNER_T2I_ROOT.",
        )
        return 1
    try:
        code = _run_locked()
    finally:
        lock.release()
    # The last thing this process does, and deliberately after the ``finally``
    # above: a worker parked on something that never returns -- the native file
    # dialogs block until dismissed, and by now the window they belong to is
    # gone -- is a non-daemon thread, so ``threading._shutdown`` would wait on
    # it forever with nothing on screen to say why. Everything that must happen
    # has happened by this line; what a hard exit skips is only the waiting.
    from .tasks import hard_exit_if_leaked

    return hard_exit_if_leaked(code)


def _offer_store_reset(exc: Any) -> bool:
    """Offer to set a broken job database aside. -> may we start over?

    The library index is a *record of jobs*, not the jobs themselves: the
    assets live in directories under the data dir and survive this untouched.
    What is lost is the history -- prompts, settings, verdicts, favourites --
    which is real and is why this is a question rather than a repair.

    Native, because it runs before the window exists; ``instance.ask`` answers
    No to anything that is not an explicit Yes, which is the right default for
    a button that moves somebody's library index.
    """
    from .. import db, instance

    log.error("the job database could not be opened", exc_info=exc.cause)
    agreed = instance.ask(
        "Realmspinner cannot open its job database",
        f"{exc.path}\n\n{exc.cause}\n\nThis file is the library's index. Your "
        "generated assets are stored as ordinary folders beside it and are not "
        "affected, but the record of them -- prompts, settings, verdicts and "
        "favourites -- is in here.\n\nStart with an empty index? The damaged "
        "file is renamed and kept, not deleted, so it can be examined or "
        "recovered later.\n\nChoosing No leaves everything untouched and "
        "closes Realmspinner.",
    )
    if not agreed:
        return False
    moved = db.set_aside(exc.path)
    if moved is None:
        instance.alert(
            "Realmspinner could not move the damaged database",
            f"{exc.path} could not be renamed, so a new index cannot be "
            "created beside it.\n\nThis usually means the file is open in "
            "another program or the directory is read-only.",
        )
        return False
    log.warning("job database set aside as %s; starting with an empty index", moved)
    return True


def _run_locked() -> int:
    """Everything after the single-instance lock is held."""
    from .. import migrate
    from ..config import get_config
    from ..db import StoreUnreadable
    from .runtime import Runtime

    if migrate.MOVED:
        # Said twice on purpose. The move itself printed to stderr because it
        # happened before this handler existed; the log is where somebody looks
        # a week later to find out why their library is not where they left it.
        log.info("moved into %s: %s", get_config().home, ", ".join(migrate.MOVED))
    _note_previous_session()
    _write_session_marker()
    try:
        try:
            return App(Runtime()).run()
        except StoreUnreadable as exc:
            # The one startup failure with an in-app way out, and it had none.
            # Offered rather than done, and offered exactly once: a second
            # ``StoreUnreadable`` after the rename is a fault in the *new*
            # file, which means the disk or the directory is the problem and
            # making another empty database would be a loop.
            if not _offer_store_reset(exc):
                return 1
            return App(Runtime()).run()
    except Exception as exc:
        # App.run reports and swallows its own failures, so anything arriving
        # here happened before the loop existed -- constructing the Runtime,
        # which opens the job store and claims the engine port.
        log.exception("Realmspinner could not start")
        # And it is said out loud. This branch logged and returned 1, which
        # under ``pythonw`` is a process that starts, writes to a devnull
        # stderr and vanishes: the user double-clicks the icon and nothing at
        # all happens. ``instance.alert`` again -- there is no window to put a
        # dialog in, which is the whole reason that function exists.
        from .. import instance

        # A refusal with words of its own says them; anything else gets the
        # type and the message, which at least distinguishes "the port is in
        # use" from "the database is malformed" without opening the log.
        if isinstance(exc, StartupRefused):
            instance.alert(exc.title, exc.body)
        else:
            instance.alert(
                "Realmspinner could not start",
                f"{type(exc).__name__}: {exc}\n\nRealmspinner stopped before its "
                "window opened. The full details are in realmspinner.log in your "
                "Realmspinner home directory.",
            )
        return 1
    finally:
        _clear_session_marker()


if __name__ == "__main__":
    sys.exit(run())
