"""Rebuild a finished mesh at a different triangle budget.

``source.glb`` is the reconstruction and nothing ever overwrites it, which is
the whole reason a retarget is cheap: ``model.glb`` is derived from it by
optimize-then-normalize, so a new budget costs a two-second gltfpack run rather
than another two minutes of trellis. The service has done this correctly for a
long time -- the 409 gating, the derived-artifact sweep, the reapplied grounding
transform, the stale-rig report -- and had no caller once the HTTP API was
removed. This is that caller.

Two things it refuses to hide. Every named tier needs ``vendor/gltfpack``, so
without the binary only ``raw`` is offered and the reason is on screen -- it is
present today, which is what makes this panel the qualification path rather
than a dormant one. And a retarget makes a rig, its poses and its sheets
describe a mesh that no longer exists; the service reports them rather than
deleting them, so the report is shown *before* the button, not after.
"""

from __future__ import annotations

from typing import Any

from ...pipelines import optimize
from ...service import jobs as svc_jobs
from .. import forms, theme, widgets
from ..manual import render as manual_render


def tier_label(key: str, budget: int | None) -> str:
    """What a named tier is called on screen: "Draft (20k)", "Raw (as
    reconstructed)".

    "As reconstructed" rather than "full density": the engine has already
    simplified the mesh to ~300k faces at res 1024 before it reaches gltfpack
    (``config.trellis_decim`` is the flag that lifts that), so raw is "no
    second pass", not "every triangle the reconstruction made".

    Derived rather than written out, which is the same argument the custom
    range already made two functions down: ``optimize.PROFILES`` is the
    authority on the numbers, and a label restating one is a second copy that
    goes wrong silently -- a button offering 50k while gltfpack is asked for
    something else. A budget that is not a round thousand is printed in full
    rather than rounded into a number it is not.
    """
    if budget is None:
        return f"{key.capitalize()} (as reconstructed)"
    if budget % 1000 == 0:
        return f"{key.capitalize()} ({budget // 1000}k)"
    return f"{key.capitalize()} ({budget:,})"


# "raw" first, then the rest of ``optimize.PROFILES``, then the free-form one.
# The *membership* is derived so a tier added to the pipeline appears here
# without an edit; only the position of "raw" is stated, because it is the
# identity -- the full reconstruction density -- and the only entry that needs
# no binary, which is what makes ``TIERS[0]`` the fallback list below.
TIERS: tuple[tuple[str, str], ...] = tuple(
    (key, tier_label(key, optimize.PROFILES[key]))
    for key in ("raw", *(k for k in optimize.PROFILES if k != "raw"))
) + (("custom", "Custom..."),)


def draw(ctx: Any, job: Any) -> None:
    files = job.get("files") or []
    if "source.glb" not in files:
        # No reconstruction to rebuild from. Older jobs and rig jobs are both
        # in this state, and neither can be retargeted at any budget.
        return
    if not widgets.header("Triangle budget", default_open=False):
        return
    manual_render.help_button(ctx, "retarget")

    job_id = job["id"]
    form = _form(ctx, job_id)
    available = _gltfpack_available(ctx)

    options = list(TIERS) if available else [TIERS[0]]
    if not available:
        form["profile"] = "raw"
        widgets.muted_wrapped(
            "Only the engine's own output is available: gltfpack is not installed."
        )
    # Form.help_text renders widgets.help_marker beside the owning label.
    #
    # **The field id is the refusal's address.** ``optimize_job`` (via
    # ``resolve_profile``, ``service/_jobs_create.py``) raises
    # ``field="profile"`` for both an unusable tier and an unusable custom
    # count -- ``remesh_job``'s own door, not this one, is what raises
    # ``field="remesh_profile"``/``field="custom_triangles"``. This panel used
    # to borrow that pair on the theory that it named "the other pane over the
    # same service call", which was wrong on two counts: the calls are
    # different (``optimize_job`` here, ``remesh_job`` there) and neither
    # spelling is what this door actually raises -- so even with ``errors``
    # wired the ring had nothing to land on, the defect this panel's own prior
    # comment claimed to have fixed (the 2026-09-07 audit, finding create-04).
    # ``profile`` and ``custom_triangles`` below are *this* door's own field
    # names, and also its parameter names, so the form dict keys need no
    # second vocabulary -- ``remesh_job`` raising the identical
    # ``"custom_triangles"`` string for its own, unrelated custom count is
    # exactly why ``remesh_panel._FIELD_PREFIX`` relabels that one to
    # ``"remesh_custom_triangles"`` before it ever reaches this shared,
    # unnamespaced dict.
    with forms.Form(
        "retarget-settings",
        errors=ctx.state.field_errors,
        on_edit=ctx.state.clear_field_error,
    ) as form_ui:
        _changed, form["profile"] = form_ui.combo(
            "profile",
            "Budget",
            form["profile"],
            options,
            help_text="Rebuild model.glb from source.glb, the engine's own output.",
            helper=(
                "You can retarget repeatedly without another reconstruction; "
                "trellis does not run again."
            ),
        )

        if form["profile"] == "custom":
            changed, value = form_ui.number(
                "custom_triangles",
                "Triangles",
                int(form["custom_triangles"]),
                helper=f"{optimize.CUSTOM_MIN:,} to {optimize.CUSTOM_MAX:,}",
            )
            if changed:
                form["custom_triangles"] = value

        _warn_stale(ctx, job)
        _submit(ctx, job_id, form)


def _form(ctx: Any, job_id: str) -> dict[str, Any]:
    """Kept on app state, keyed by job id, so a budget typed against one mesh
    is not submitted against another -- and, since ``remesh_panel``'s fix
    (commit 89cb6412), so it survives a glance at another asset too.

    The 2026-09-08 audit, finding create-02: this used to be one shared slot
    compared by ``form.get("job_id") != job_id``, the pattern commit
    89cb6412 fixed on ``remesh_panel``/``sheet_panel``/``sprite_panel`` but
    left standing here -- so looking at a different card in the Library and
    coming back silently reset a chosen custom triangle count to the default
    tier, with no undo.
    """
    forms_by_job = ctx.state.preview.setdefault("retarget_forms", {})
    form = forms_by_job.get(job_id)
    if form is None:
        form = {
            "job_id": job_id,
            "profile": "raw",
            "custom_triangles": optimize.PROFILES["standard"],
        }
        forms_by_job[job_id] = form
    return form


# Whether the vendored binary is on disk, answered once per path. This ran on
# every frame the section was open, for an answer that is fixed for the life of
# the process: the path comes from Config and a vendored binary does not arrive
# while the app runs. Keyed by the path rather than kept as a single flag so a
# second service (a test's, a compare view's) cannot inherit the first one's
# answer. The accepted cost is that installing gltfpack needs a restart before
# the tiers appear -- which is already true of the doctor row that reports it.
_gltfpack_seen: dict[str, bool] = {}


def gltfpack_available(ctx: Any) -> bool:
    try:
        path = ctx.svc.config.gltfpack_exe
    except Exception:  # noqa: BLE001 - a ctx with no config answers "no tiers"
        # Silent on purpose, and it is the same answer as "the tool is not
        # installed": this decides whether *optional* tiers appear, and a
        # panel that refused to draw because it could not ask is worse than
        # one that offers less.
        return False
    key = str(path)
    found = _gltfpack_seen.get(key)
    if found is None:
        try:
            # The 2026-09-05 audit, finding create-06: this used to call
            # ``path.exists()``, so a directory left where the binary should
            # be (a broken extraction, a stray folder) read as present --
            # doctor.py's ``_gltfpack_check`` was already fixed for the same
            # reason (L01). ``is_file()`` is the readiness check; a directory
            # is not a usable binary no matter what its name is.
            found = bool(path.is_file())
        except Exception:  # noqa: BLE001 - an unreachable path is an absent one
            found = False
        _gltfpack_seen[key] = found
    return found


# The private spelling stayed importable, the ``resolve_profile`` pattern
# (``service/_jobs_create.py``): settings_3d._budget (2026-09-23, dev/
# measurements/2026-09-23-default-mesh-budget.md) needs this answer to decide
# whether Create's own Budget combo offers every tier or collapses to Raw, and
# this module's own callers and tests already spell the name with the
# underscore.
_gltfpack_available = gltfpack_available


def _warn_stale(ctx: Any, job: Any) -> None:
    """Name the user work a retarget will invalidate, before it happens.

    The service reports the rig/poses/sheets rather than deleting them -- a
    rig and its poses are minutes of work and must not be destroyed over a
    triangle count -- but reporting after the fact is only half of it.

    The remesh half is not reported as a plain warning any more: ``optimize_job``
    rebuilds model.glb from source.glb, which replaces whatever a prior remesh
    baked onto it and drops the stale ``params["remesh"]`` report as part of
    that same rewrite (the 2026-09-18 audit, finding service-01) -- but since
    ``pipelines.modelhistory`` landed (2026-09-22) that mesh is kept
    under Earlier meshes rather than lost, and a submit here still gets its
    own line before the button, worded for that.
    """
    files = set(job.get("files") or [])
    if "rig.glb" in files:
        widgets.text_colored(
            theme.WARN, "The rig, its poses and its sheets will describe the old mesh."
        )
    if (job.get("params") or {}).get("remesh"):
        widgets.text_colored(
            theme.WARN,
            "The game-ready remesh will be replaced and rebuilt from the raw "
            "reconstruction. It is kept under Earlier meshes below.",
        )
    # create-05 (2026-09-23 audit): the line above only fired when a prior
    # remesh existed, so a retarget with no remesh behind it said nothing at
    # all about the mesh it was about to replace -- but ``optimize_job``
    # snapshots the replaced mesh under Earlier meshes unconditionally
    # (``pipelines.modelhistory``, 2026-09-22), the same guarantee
    # ``remesh_panel`` and ``texture_panel`` already state with no such
    # caveat. Unconditional here too, so the reassurance matches what
    # actually happens on every retarget, not just the ones that follow a
    # remesh.
    widgets.muted_wrapped(
        "The current mesh is kept under Earlier meshes below, so this can be undone."
    )


def dependent_job_reason(jobs: list[dict[str, Any]], job_id: str) -> str | None:
    """Whether an unfinished sibling job still writes into this asset's directory.

    The 2026-09-06 audit, finding create-02: this pane greyed its button only
    on its own client-side ``busy`` flag, so a rig, sheet or sibling rework job
    already queued or running against the same mesh (``_jobs_rework``'s
    ``_require_no_dependents`` / ``_jobs_lifecycle.dependent_jobs``) was
    discovered only after a wasted press, as a ``Conflict`` toast, instead of
    as a reason stated before the button the way the stale-rig warning is.

    A pure function over ``ctx.cache.jobs`` rather than a call to
    ``dependent_jobs(svc, job_id)`` itself: that does a live sqlite query, and
    ``draw`` runs on the frame thread (the frame-thread rule in CLAUDE.md and
    ``tests/test_frame_thread_doors.py``). ``jobs_cache`` already refreshes on
    its own timer for exactly this reason, so this reads the same recent
    window every other pane does.
    """
    count = sum(
        1
        for job in jobs
        if job.get("status") in ("queued", "running")
        and (job.get("params") or {}).get("source_job") == job_id
    )
    if not count:
        return None
    return (
        f"{count} job(s) started from this mesh are still queued or running; "
        "wait for them to finish first."
    )


def _submit(ctx: Any, job_id: str, form: dict[str, Any]) -> None:
    key = f"retarget:{job_id}"
    busy = ctx.busy(key)
    problems = validate(form)
    for problem in problems:
        widgets.muted(problem)
    dep_reason = None if busy else dependent_job_reason(ctx.cache.jobs, job_id)
    if dep_reason:
        widgets.text_colored(theme.WARN, dep_reason)
    if busy:
        widgets.busy("Rebuilding the mesh")
    if widgets.disabled_button(
        "Rebuild mesh",
        not problems and not busy and not dep_reason,
        (-1, 0),
        # ``sheet_panel``'s rule: the problems are listed above the button, so
        # the reason names the other gate and defers to the list otherwise.
        reason=(
            "A rebuild is already running for this asset."
            if busy
            else dep_reason or "; ".join(problems)
        ),
    ):
        # A new submit is judged on its own, so last time's rings go first --
        # ``settings_2d.generate``'s rule, and the half a bare ``errors=``
        # cannot supply.
        ctx.state.clear_field_errors()
        ctx.submit(
            key,
            svc_jobs.optimize_job,
            ctx.svc,
            job_id,
            profile=form["profile"],
            custom_triangles=(
                int(form["custom_triangles"]) if form["profile"] == "custom" else None
            ),
        )


def validate(form: dict[str, Any]) -> list[str]:
    """The refusals stated before the button, matching what the service checks.

    ``optimize.resolve`` is the authority on the range; restating the numbers
    here rather than the rule would let the two drift.
    """
    if form["profile"] != "custom":
        return []
    try:
        optimize.resolve("custom", int(form["custom_triangles"]))
    except (ValueError, TypeError) as exc:
        return [str(exc)]
    return []
