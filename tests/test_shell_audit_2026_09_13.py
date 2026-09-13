"""Regressions for the 2026-09-13 audit's shell-04, shell-05 and shell-06.

Each is a hook that existed for one case and silently missed its siblings:
the mode-leave callback knew only Sirens, the trash measurement wrote UI
state from its own task, and the inspector's unsent-edit flush fired only on
a selection change.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from warlock.studio import main as main_mod
from warlock.studio import state as state_mod
from warlock.studio.panes import inspector, library
from warlock.studio.state import AppState


@pytest.fixture(autouse=True)
def _clean():
    inspector._unsent.clear()
    yield
    inspector._unsent.clear()
    state_mod.set_mode_leave(None)


def _leave(ctx: Any, old: str, new: str) -> None:
    ctx.state.mode = old
    state_mod.set_mode_leave(lambda was: main_mod._leave_mode_if_needed(ctx, was))
    assert state_mod.set_mode(ctx.state, new) is True


# --- shell-04 -----------------------------------------------------------------


def test_leaving_muse_mode_stops_a_sounding_take(monkeypatch):
    from warlock.studio import muse_mode

    stopped: list[Any] = []
    monkeypatch.setattr(muse_mode, "stop", lambda ctx: stopped.append(ctx))
    ctx = SimpleNamespace(state=AppState())

    _leave(ctx, "muse", "home")

    assert stopped == [ctx], "leaving Muse must stop its audition"


def test_leaving_plotter_mode_mid_drag_closes_the_open_edit_session(monkeypatch):
    from warlock.studio import plotter_state

    ended: list[str] = []
    doc = SimpleNamespace(
        end_stroke=lambda: ended.append("stroke"),
        end_object_edit=lambda: ended.append("object"),
        end_tile_meta_edit=lambda: ended.append("tile_meta"),
    )
    monkeypatch.setattr(plotter_state, "active", lambda ctx: SimpleNamespace(doc=doc))
    ctx = SimpleNamespace(state=AppState())

    _leave(ctx, "plotter", "home")

    assert ended == ["stroke", "object", "tile_meta"]


# --- shell-05 -----------------------------------------------------------------


def test_measure_trash_does_not_write_state_from_the_task_thread():
    """The submitted callable used to write ``state.preview`` itself; the
    reading must come back as the task's result, tagged with the trash it
    measured, for ``main`` to adopt on the frame thread."""
    submitted: list[tuple] = []

    def submit(key, fn, *args, **kwargs):
        submitted.append((key, kwargs))
        fn(*args)  # run the task inline, as the task thread would
        return True

    svc = SimpleNamespace()
    ctx = SimpleNamespace(state=AppState(), svc=svc, submit=submit)
    from warlock.service import jobs as svc_jobs

    real = svc_jobs.trash_size
    svc_jobs.trash_size = lambda _svc: {"count": 1, "bytes": 10}
    try:
        library.measure_trash(ctx, [{"id": "a"}])
    finally:
        svc_jobs.trash_size = real

    assert library.TRASH_SIZE_SLOT not in ctx.state.preview
    assert submitted and submitted[0][1].get("tag") == ("a",)


# --- shell-06 -----------------------------------------------------------------


class _Ctx:
    def __init__(self) -> None:
        self.svc = None
        self.state = AppState()
        self.sent: list[tuple[str, dict]] = []

    def submit(self, key: str, _fn: Any, _svc: Any, _job_id: str, payload: dict) -> bool:
        self.sent.append((key, payload))
        return True


def test_an_unsent_edit_is_not_lost_when_the_mode_changes_before_it_lands():
    ctx = _Ctx()
    inspector._unsent["name:abc"] = ("abc", lambda v: {"name": v}, "chest")
    inspector._unsent["tags:xyz"] = ("xyz", lambda v: {"tags": v}, "wood")

    _leave(ctx, "create", "clay")

    assert sorted(ctx.sent) == [("name:abc", {"name": "chest"}), ("tags:xyz", {"tags": "wood"})]
    assert inspector._unsent == {}
