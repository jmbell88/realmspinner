"""Regressions for the 2026-09-26 audit, findings closed against
``studio/palette.py`` -- see ``tests/modes/mason/test_audit_2026_09_26_w1f8.py``
for the rest of this pass's findings (all in the Mason package).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from realmspinner.studio import palette


def _build_ctx() -> Any:
    """Just enough of the App ``ctx`` for ``palette.commands`` to *build* its
    list -- ``tests/studio/test_palette.py``'s own ``_ctx`` helper, since this
    finding is about what one built command's ``run`` does, not about which
    commands get listed."""
    from realmspinner.studio.state import ManualState

    return SimpleNamespace(
        state=SimpleNamespace(
            mode="library",
            previous_mode="library",
            mode_observed="library",
            create=SimpleNamespace(stage="mesh"),
            selected=None,
            source_job=None,
            wireframe=False,
            turntable=False,
            show_fps=False,
            manual=ManualState(),
        ),
        cache=SimpleNamespace(jobs=[], get=lambda _id: None),
        viewer=None,
    )


def _job(job_id: str, **over: Any) -> dict[str, Any]:
    row = {
        "id": job_id,
        "name": "",
        "prompt": "",
        "status": "done",
        "kind": "model",
        "stage": "model",
        "files": [],
    }
    row.update(over)
    return row


class _Confirms:
    def __init__(self) -> None:
        self.asked: list[Any] = []

    def ask(self, confirm: Any) -> None:
        self.asked.append(confirm)


class _State:
    def __init__(self, selected: str | None) -> None:
        self.selected = selected
        self.checked: set[str] = {selected} if selected else set()

    def select(self, value: str | None) -> None:
        self.selected = value


class _Cache:
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self.jobs = jobs
        self._by_id = {job["id"]: job for job in jobs}

    def get(self, job_id: Any) -> Any:
        return self._by_id.get(job_id)


class _Ctx:
    """Just enough of the App ``ctx`` for ``palette.commands``' ``delete`` row
    and ``library.delete_asset`` underneath it."""

    def __init__(self, job_id: str) -> None:
        self.state = _State(job_id)
        self.cache = _Cache([_job(job_id)])
        self.confirms = _Confirms()
        self.svc = None
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []

    def submit(self, key: str, fn: Any, *args: Any) -> bool:
        self.submitted.append(key)
        return True

    def toast(self, message: str, kind: str = "info", *_a: Any, **_k: Any) -> None:
        self.toasts.append((message, kind))


def test_the_palette_delete_command_trashes_without_a_confirm_that_claims_disk_removal():
    """The 2026-09-26 audit's shell-chrome-01: the command palette's "Delete"
    row opened a confirm titled "Delete this asset?" saying the job "is
    removed from disk", but ``on_confirm`` called ``library.delete_asset``,
    which only ever trashes (its own docstring: "It trashes rather than
    deletes... The permanent version is purge_asset, and the only thing that
    reaches it is the trash."). Manual Chapter 36 (The trash): "Nothing is
    removed from disk and no question is asked -- the trash *is* the
    question." ``shell/events.py``'s own Delete-key handler on the same
    library row already skips the confirm for exactly that reason -- its own
    comment states the house rule: "delete-to-trash is confirm-free here
    because the trash *is* the confirmation."

    Against the unfixed command this must fail: ``ctx.confirms.asked`` holds
    one ``Confirm`` and nothing is submitted until it is answered.
    """
    command = next(c for c in palette.commands(_build_ctx()) if c.key == "delete")
    ctx = _Ctx("job-1")

    command.run(ctx)

    assert ctx.confirms.asked == []
    assert ctx.submitted == ["delete:job-1"]
    assert any("trash" in message.lower() for message, _kind in ctx.toasts)
