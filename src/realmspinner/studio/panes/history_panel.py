"""Every mesh a rework replaced, and the one door back to one of them.

``retarget_panel``, ``remesh_panel`` and ``texture_panel`` each publish over
``model.glb`` and each used to say, one way or another, that whatever it was
about to replace was gone for good. ``pipelines.modelhistory`` is what makes
that false now: a rework's own publish (``_q_mesh._publish_model_version``,
``optimize_job``) snapshots the mesh it is about to overwrite before it does,
and this panel is the one place that history is readable and restorable.

Drawn collapsed, and only when there is something in it -- a fresh mesh with
no rework behind it has nothing to show, and a header that always renders
empty is a dead control by another name.
"""

from __future__ import annotations

from typing import Any

from ...pipelines import modelhistory
from ...service import jobs as svc_jobs
from .. import dialogs, theme, widgets
from ..manual import render as manual_render
from . import retarget_panel

#: What each ``kind`` reads as on screen. A key this table does not have
#: (a future rework, or a hand-edited row) falls back to the raw string
#: rather than raising -- an unfamiliar label is still readable.
_KIND_LABELS = {
    "optimize": "Triangle budget",
    "remesh": "Game-ready remesh",
    "retexture": "Surface texture",
    "revert": "Restored mesh",
}


def entries_for(job: Any) -> list[dict[str, Any]]:
    """This job's kept versions, newest first -- the order the panel draws.

    No imgui here, deliberately: ordering is a fact about the data, testable
    without a GL context, and every draw function below is a thin loop over
    what this returns.
    """
    return list(reversed(modelhistory.entries_of(job.get("params") or {})))


def row_label(entry: dict[str, Any]) -> str:
    """What replaced this version -- the panel's first column."""
    kind = entry.get("kind")
    return _KIND_LABELS.get(kind, str(kind or "mesh"))


def row_detail(entry: dict[str, Any]) -> str:
    """The one-line description under a row: what the step recorded, and how
    big the kept file is. Pure, for the same reason :func:`entries_for` is.
    """
    detail = str(entry.get("detail") or "")
    size = _human_bytes(entry.get("bytes"))
    if detail and size:
        return f"{detail} ({size})"
    return detail or size


def _human_bytes(value: Any) -> str:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return ""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _revert_key(job_id: str) -> str:
    return f"model-revert:{job_id}"


def draw(ctx: Any, job: Any) -> None:
    entries = entries_for(job)
    if not entries:
        return
    if not widgets.header("Earlier meshes", default_open=False):
        return
    manual_render.help_button(ctx, "mesh-history")

    job_id = job["id"]
    busy = ctx.busy(_revert_key(job_id))
    dep_reason = None if busy else retarget_panel.dependent_job_reason(ctx.cache.jobs, job_id)
    if dep_reason:
        widgets.text_colored(theme.WARN, dep_reason)
    if busy:
        widgets.busy("Restoring the mesh")

    for entry in entries:
        _row(ctx, job, entry, busy=busy, dep_reason=dep_reason)


def _row(ctx: Any, job: Any, entry: dict[str, Any], *, busy: bool, dep_reason: str | None) -> None:
    n = entry.get("n")
    widgets.muted(f"{row_label(entry)} - {widgets.ago(entry.get('at'))}")
    detail = row_detail(entry)
    if detail:
        widgets.muted(detail)
    if widgets.disabled_button(
        f"Restore##model-history-{n}",
        not busy and not dep_reason,
        (-1, 0),
        reason=(
            "A restore is already running for this asset."
            if busy
            else dep_reason or ""
        ),
    ):
        _confirm_restore(ctx, job, entry)
    widgets.divider()


def _confirm_restore(ctx: Any, job: Any, entry: dict[str, Any]) -> None:
    """Ask before replacing the current mesh, naming what goes stale.

    ``crossed_geometry`` reads the *whole* chain from this version forward,
    not just this entry's own ``geometry`` flag -- restoring a version from
    two reworks back can still be a pure surface change if neither rework
    since touched geometry (see the function's own docstring).
    """
    job_id = job["id"]
    n = entry.get("n")
    raw = modelhistory.entries_of(job.get("params") or {})
    geometry = modelhistory.crossed_geometry(raw, n)
    message = (
        "The rig, its poses and its sheets will describe the old mesh."
        if geometry
        else "Only the surface changes; the rig, its poses and its sheets are unaffected."
    )
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Restore this mesh?",
            message=(
                f"This replaces the current mesh with the one from "
                f"{row_label(entry).lower()}. {message} The mesh you have now "
                f"is kept here too, so this can be undone."
            ),
            confirm_label="Restore",
            cancel_label="Cancel",
            on_confirm=lambda: ctx.submit(
                _revert_key(job_id), svc_jobs.revert_model, ctx.svc, job_id, version=n
            ),
        )
    )
