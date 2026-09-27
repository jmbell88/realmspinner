"""Two Poser findings from the 2026-09-26 audit.

poser-mode-03: ``on_pose_dirty`` was wired only on the shared viewer
(``shell/app.py``), never on Poser's own separate ``Viewer`` -- so the
status-bar ``*``/title mark never followed a Poser edit.

poser-mode-05: ``poser_mode.on_task_failed``'s ``troupe-start``/``troupe-
sheet:``/``troupe-send:`` branch raised a *second* toast, on top of the one
the shell's generic failure path (``shell/tasks.py``'s ``_collect_tasks``)
already shows for every failed task.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from realmspinner.studio.modes.poser import mode as poser_mode
from realmspinner.studio.modes.poser.ui.viewport import PoserViewport

# --- poser-mode-03 --------------------------------------------------------------


class _FakePoserApp(PoserViewport):
    """Just enough of ``App`` for ``_ensure_poser_viewer`` to run."""

    def __init__(self, gl_ctx: Any) -> None:
        self.ctx = gl_ctx
        self.poser_viewer = None
        self.app_ctx = SimpleNamespace()
        self.dirty_calls: list[bool] = []

    def _on_pose_dirty(self, dirty: bool) -> None:
        self.dirty_calls.append(bool(dirty))


def test_poser_viewer_dirty_edits_raise_the_status_bar_document_mark(gl):
    """poser-mode-03, the 2026-09-26 audit: ``shell/app.py`` wires
    ``self.viewer.on_pose_dirty = self._on_pose_dirty`` at startup on the
    *shared* viewer, but Poser edits through its own, separate ``Viewer`` --
    built lazily by ``_ensure_poser_viewer``, long after that assignment ran
    -- so its callback stayed the class default (``None``) and a Poser edit
    never reached ``_on_pose_dirty``.
    """
    app = _FakePoserApp(gl)
    viewer = app._ensure_poser_viewer()
    try:
        assert viewer.on_pose_dirty is not None, "Poser's own viewer never got a dirty callback"
        # ``==`` rather than ``is``: two accesses of a bound method are equal
        # but not identical (a fresh bound-method object each time), which is
        # what a naive ``is`` check against ``app._on_pose_dirty`` would miss.
        assert viewer.on_pose_dirty == app._on_pose_dirty

        viewer.on_pose_dirty(True)
        assert app.dirty_calls == [True]
    finally:
        viewer.release()


# --- poser-mode-05 --------------------------------------------------------------


class _Toasts(list):
    def __call__(self, message: str, kind: str = "info", action: Any = None, **_kw: Any) -> None:
        self.append((message, kind))


def test_a_failed_sheet_task_raises_exactly_one_error_toast():
    """poser-mode-05, the 2026-09-26 audit: the shell's generic failure path
    already toasts ``done.message`` for every failed task before routing to
    the mode; ``poser_mode.on_task_failed``'s ``troupe-start`` branch then
    raised a second toast, with the raw exception object's text rather than
    the curated message.
    """
    from realmspinner.studio import main as main_mod

    app = main_mod.App.__new__(main_mod.App)
    toasts = _Toasts()
    done = SimpleNamespace(
        key="troupe-start",
        ok=False,
        result=None,
        error=RuntimeError("boom"),
        message="Could not start the character sheet.",
        action=None,
        tag=None,
    )
    tasks = SimpleNamespace(poll=lambda: [done])
    app.app_ctx = SimpleNamespace(
        toast=toasts,
        tasks=tasks,
        state=SimpleNamespace(poser=None, preview={}),
    )

    app._collect_tasks()

    assert len(toasts) == 1, f"expected exactly one toast, got {list(toasts)}"
    assert toasts[0] == ("Could not start the character sheet.", "error")


def test_on_task_failed_itself_raises_no_toast_for_a_troupe_start_failure():
    """The unit-level half of the same claim, directly against
    ``poser_mode.on_task_failed`` -- the shell's own toast is exercised above.
    """
    toasts: list[tuple[str, str]] = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(poser=None, preview={}),
        toast=lambda message, kind="info", *_a, **_kw: toasts.append((message, kind)),
    )
    poser_mode.on_task_failed(
        ctx, SimpleNamespace(key="troupe-start", error=RuntimeError("boom"))
    )
    assert toasts == []
