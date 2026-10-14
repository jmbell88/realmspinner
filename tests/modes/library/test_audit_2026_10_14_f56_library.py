"""The 2026-10-14 audit, fixer F56: the trash's bulk purge and Restore.

shell-home-library-02: ``service.jobs.delete_job`` has no trashed check, so the trash's
irreversible button deleted whatever was ticked, live or not.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from realmspinner.studio.modes.library.ui.panes import library
from realmspinner.studio.state import AppState


class _Confirms:
    def __init__(self) -> None:
        self.asked: list[Any] = []

    def ask(self, confirm: Any) -> None:
        self.asked.append(confirm)


def _ctx(rows: list[dict[str, Any]], ticked: list[str]) -> Any:
    by_id = {row["id"]: row for row in rows}
    state = AppState()
    state.checked.update(ticked)
    state.filters.trash = True
    ctx = SimpleNamespace(
        state=state,
        cache=SimpleNamespace(get=by_id.get),
        confirms=_Confirms(),
        svc=SimpleNamespace(),
        submitted=[],
    )

    def submit(key: str, _run: Any, *_args: Any, **_kwargs: Any) -> bool:
        ctx.submitted.append(key)
        return True

    ctx.submit = submit
    return ctx


def test_the_trash_purge_only_deletes_rows_that_are_actually_trashed():
    rows = [
        {"id": "live", "status": "done"},
        {"id": "gone", "status": "done", "deleted_at": 1.0},
    ]
    ctx = _ctx(rows, ["live", "gone", "unknown"])

    library._bulk_action(ctx, "purge", ["gone", "live", "unknown"], 0)

    (confirm,) = ctx.confirms.asked
    assert "1 jobs" in confirm.message
    confirm.on_confirm()
    assert ctx.submitted == ["purge:gone"]


def test_the_trash_purge_with_nothing_trashed_asks_nothing():
    ctx = _ctx([{"id": "live", "status": "done"}], ["live"])
    library._bulk_action(ctx, "purge", ["live"], 0)
    assert ctx.confirms.asked == []
    assert ctx.submitted == []


def test_the_trash_restore_leaves_a_live_row_alone():
    rows = [
        {"id": "live", "status": "done"},
        {"id": "gone", "status": "done", "deleted_at": 1.0},
    ]
    ctx = _ctx(rows, ["live", "gone"])
    library._bulk_action(ctx, "restore", ["gone", "live"], 0)
    assert ctx.submitted == ["restore:gone"]


def test_a_single_purge_refuses_a_live_row():
    ctx = _ctx([{"id": "live", "status": "done"}], ["live"])
    library.purge_asset(ctx, "live")
    assert ctx.submitted == []
