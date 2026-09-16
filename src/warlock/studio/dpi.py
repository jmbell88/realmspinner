"""Windows DPI awareness, sampled once at startup.

The app draws in physical pixels: with Per-Monitor-V2 awareness the pygame
window's size *is* the framebuffer size, so ``io.display_framebuffer_scale``
stays (1, 1) and the imgui backend's projection and scissor math never change.
Scaling lives entirely in content -- font pixel size and the style values --
via :data:`tokens.SCALE`.

Failure at any step degrades to 1.0: a wrong scale on an exotic setup is a
small UI, not a crash.
"""

from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)


def make_process_dpi_aware() -> None:
    """Opt in to Per-Monitor-V2 before the window exists.

    Must run before ``pygame.display.set_mode``; once a window is created the
    process awareness is frozen. SDL may have set awareness already, in which
    case the call fails and that is fine -- :func:`window_scale` reads the
    effective DPI regardless of who set it.
    """
    if sys.platform != "win32":
        return
    import ctypes

    try:
        # -4 == DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(-4):
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        log.warning("could not make the process DPI-aware; UI will scale at 1.0")


def system_scale() -> float:
    """The primary monitor's scale, readable before any window exists.

    Used only to size the initial window; :func:`window_scale` is the value
    the UI is actually built at once the window knows its monitor.
    """
    if sys.platform != "win32":
        return 1.0
    import ctypes

    try:
        dpi = ctypes.windll.user32.GetDpiForSystem()
        return (dpi / 96.0) if dpi else 1.0
    except (AttributeError, OSError):
        return 1.0


def window_scale(pygame_module) -> float:
    """The opened window's monitor scale (1.0 = 96 DPI)."""
    if sys.platform != "win32":
        return 1.0
    import ctypes

    try:
        hwnd = pygame_module.display.get_wm_info()["window"]
        dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
        return (dpi / 96.0) if dpi else 1.0
    except (AttributeError, OSError, KeyError):
        return 1.0


def work_area() -> tuple[int, int, int, int] | None:
    """The primary monitor's work area -- (x, y, w, h), physical pixels.

    ``SPI_GETWORKAREA`` is the desktop area minus the taskbar (and any docked
    toolbars), which ``pygame.display.get_desktop_sizes()`` cannot see -- see
    ``main._desktop_size``, where Familiar's Build/Send row sat under the
    taskbar (2026-09-16) because the whole-display size was the only ceiling
    available. Read before any window exists, so this is necessarily the
    *primary* monitor's work area -- there is no HWND yet to ask which
    monitor the window will land on.

    Physical pixels because :func:`make_process_dpi_aware` runs before this
    is ever called. None off Windows, or on any failure: a window sized
    against the whole display is a taskbar-sized annoyance, not a crash, so
    the caller falls back rather than propagating.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    SPI_GETWORKAREA = 0x0030
    try:
        rect = wintypes.RECT()
        if not ctypes.windll.user32.SystemParametersInfoW(
            SPI_GETWORKAREA, 0, ctypes.byref(rect), 0
        ):
            return None
    except (AttributeError, OSError, ValueError):
        return None
    return (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)


def fit_rect(
    outer: tuple[int, int, int, int], work: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """The smallest change that brings ``outer`` fully inside ``work``.

    Both are ``(x, y, w, h)`` in the same pixel space -- a window's outer
    (frame-inclusive) rect and a monitor's work area. None when ``outer``
    already lies entirely inside ``work``, so a caller can skip the
    ``SetWindowPos`` call (and the resize it raises) on the common case of a
    monitor with nothing docked on any edge.

    Size shrinks first, then position is clamped -- clamping position before
    a shrink could still leave an oversized window straddling the far edge
    it was just pushed away from. Clamping each axis to
    ``[work_x, work_x + work_w - w]`` rather than "shift by the overflow"
    is what makes a taskbar on the left or top come out the same as one on
    the right or bottom: ``outer`` can start left of or above ``work``
    (taskbar on the left/top pushes the work origin inward) just as easily
    as it can overhang the far edge, and the one clamp expression corrects
    either direction.
    """
    ox, oy, ow, oh = outer
    wx, wy, ww, wh = work
    if ox >= wx and oy >= wy and ox + ow <= wx + ww and oy + oh <= wy + wh:
        return None
    new_w = min(ow, ww)
    new_h = min(oh, wh)
    new_x = min(max(ox, wx), wx + ww - new_w)
    new_y = min(max(oy, wy), wy + wh - new_h)
    return (new_x, new_y, new_w, new_h)


def fit_window_to_work_area(hwnd: int) -> bool:
    """Move/shrink ``hwnd`` so its outer frame sits inside its monitor's work area.

    ``_window_size`` (main.py) clamps the window's *client* size before
    ``set_mode`` ever runs, but the title bar and frame add pixels a client
    size cannot see -- so a window whose client area fits can still have its
    bottom edge, and Familiar's Build/Send row on it, under the taskbar. By
    the time a HWND exists this can be fixed exactly: the monitor the window
    actually landed on (not :func:`work_area`'s primary-monitor guess, taken
    before the window existed) via ``MonitorFromWindow``/``GetMonitorInfoW``,
    against the window's real outer rect via ``GetWindowRect``.

    Windows only, and every failure is swallowed to ``False``: called right
    after the window opens, where an exception here must never stop the
    window opening -- a window one taskbar too low is recoverable, a startup
    crash is not.
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    MONITOR_DEFAULTTONEAREST = 2
    SWP_NOZORDER = 0x0004
    SWP_NOACTIVATE = 0x0010

    class _MonitorInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    try:
        user32 = ctypes.windll.user32
        user32.MonitorFromWindow.restype = wintypes.HMONITOR
        user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        user32.GetMonitorInfoW.restype = wintypes.BOOL
        user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(_MonitorInfo)]
        user32.GetWindowRect.restype = wintypes.BOOL
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.SetWindowPos.restype = wintypes.BOOL
        user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]

        monitor = user32.MonitorFromWindow(wintypes.HWND(hwnd), MONITOR_DEFAULTTONEAREST)
        if not monitor:
            return False
        info = _MonitorInfo()
        info.cbSize = ctypes.sizeof(_MonitorInfo)
        if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return False
        work = (
            info.rcWork.left,
            info.rcWork.top,
            info.rcWork.right - info.rcWork.left,
            info.rcWork.bottom - info.rcWork.top,
        )
        rect = wintypes.RECT()
        if not user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
            return False
        outer = (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
        fitted = fit_rect(outer, work)
        if fitted is None:
            return False
        x, y, w, h = fitted
        moved = user32.SetWindowPos(
            wintypes.HWND(hwnd), None, x, y, w, h, SWP_NOZORDER | SWP_NOACTIVATE
        )
        return bool(moved)
    except (AttributeError, OSError, ValueError):
        return False
