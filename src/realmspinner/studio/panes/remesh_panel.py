"""Remesh a finished mesh to a triangle budget and rebake its surface.

The inspector's third rework panel, between the triangle budget (which keeps
the reconstruction's surface and simplifies it) and the surface texture (which
keeps the geometry and repaints it). This one replaces both: a fresh surface
at a triangle count (quadriflow where the input allows it, a decimate fallback
where it does not -- see ``pipelines.remesh``), a fresh unwrap, and the old
colour, roughness and normals baked onto the new mesh. It is the step that
turns a reconstruction into something an engine budgets for, and it is what
every commercial generator sells as "game-ready".

Two things it does not hide, on ``retarget_panel``'s model. It runs in Blender,
so without the ``rig`` extra the panel says so and offers nothing. And it makes
a rig, its poses and its sheets describe a mesh that no longer exists -- the
service reports them rather than deleting them, and the warning is drawn
*before* the button.
"""

from __future__ import annotations

from typing import Any

from ...pipelines import remesh
from ...service import jobs as svc_jobs
from .. import controls, forms, theme, widgets
from ..manual import render as manual_render

#: ``remesh.TRIANGLE_PROFILES`` first, then the free-form entry -- membership
#: derived, so a budget added to the pipeline appears here without an edit.
PROFILES: tuple[tuple[str, str], ...] = tuple(
    (key, remesh.profile_label(key)) for key in remesh.TRIANGLE_PROFILES
) + (("custom", remesh.profile_label("custom")),)

TEXTURES: tuple[tuple[str, str], ...] = (("", "Match the mesh"),) + tuple(
    (str(s), f"{s} px") for s in remesh.TEXTURE_SIZES
)


# The 2026-09-07 audit, finding create-01: bare "texture_size" is also
# ``texture_panel``'s id in ``ctx.state.field_errors`` -- one flat,
# unnamespaced dict -- so a re-texture refusal about its atlas size rang this
# panel's bake-size control too, whenever the inspector drew both together
# (which is routine). "custom_triangles" is the same hazard against a
# different neighbour: ``retarget_panel`` draws its own "custom_triangles"
# number widget beside this one's, for the gltfpack tier's custom count, and
# ``remesh_job`` raises the identical field name for this panel's own custom
# triangle count. The widgets below are keyed by the prefixed names, and
# ``_remesh_job`` relabels ``remesh_job``'s own refusals to match before they
# ever reach ``ctx.state.field_errors``.
_FIELD_PREFIX = {
    "texture_size": "remesh_texture_size",
    "custom_triangles": "remesh_custom_triangles",
}


def _remesh_job(svc: Any, job_id: str, **kwargs: Any) -> dict[str, Any]:
    """``remesh_job``, with a colliding refusal's address renamed to this
    panel's own control. See ``_FIELD_PREFIX`` above."""
    try:
        return svc_jobs.remesh_job(svc, job_id, **kwargs)
    except Exception as exc:
        renamed = _FIELD_PREFIX.get(getattr(exc, "field", None))
        if renamed:
            exc.field = renamed
        raise


def draw(ctx: Any, job: Any) -> None:
    files = job.get("files") or []
    if "model.glb" not in files:
        return
    if not widgets.header("Game-ready remesh", default_open=False):
        return
    manual_render.help_button(ctx, "remesh")

    if not _blender_available(ctx):
        widgets.muted_wrapped("A remesh runs in Blender, which is not installed (the rig extra).")
        return

    job_id = job["id"]
    form = _form(ctx, job_id)
    # ``on_edit``: this pane read the recorded refusal and never cleared it, so
    # a ring could only be dismissed by a *successful* submit -- the app going
    # on arguing about a value the user had already changed.
    with forms.Form(
        "remesh-settings",
        errors=ctx.state.field_errors,
        on_edit=ctx.state.clear_field_error,
    ) as form_ui:
        _changed, form["remesh_profile"] = form_ui.combo(
            "remesh_profile",
            "Triangles",
            form["remesh_profile"],
            PROFILES,
            help_text="Rebuild the surface at this triangle budget, then bake the "
            "old colour, roughness and normals onto it.",
            helper="A reconstruction is ~300k triangles; a prop ships at 2-20k.",
        )
        if form["remesh_profile"] == "custom":
            changed, value = form_ui.number(
                "remesh_custom_triangles",
                "Triangles",
                int(form["custom_triangles"]),
                helper=f"{remesh.TRIANGLES_MIN:,} to {remesh.TRIANGLES_MAX:,}",
            )
            if changed:
                form["custom_triangles"] = value
        _changed, form["texture_size"] = form_ui.combo(
            "remesh_texture_size",
            "Bake at",
            form["texture_size"],
            TEXTURES,
            help_text="Resolution of the rebaked textures.",
        )
        _changed, form["close_holes"] = controls.checkbox(
            "Close holes first", form["close_holes"]
        )
        widgets.help_marker(
            "A voxel pass before the remesh seals the gaps a reconstruction "
            "leaves, at the cost of slightly rounding sharp edges."
        )
        _warn_stale(ctx, job)
        _submit(ctx, job_id, form)

    line = remesh.report_line((job.get("params") or {}).get("remesh"))
    if line:
        widgets.muted(f"Last remesh: {line}")


def _form(ctx: Any, job_id: str) -> dict[str, Any]:
    """The request, kept on the app state, keyed by job id.

    The 2026-09-07 Create review, item 5.3b: this used to be one form,
    rebuilt whenever the selection moved, so a user who set up a remesh,
    glanced at another asset, and came back found the defaults again rather
    than what they had typed. Keyed by job id instead: nothing here names a
    file or a resource tied to the *previous* selection, so there is nothing
    that must reset on a reselect the way ``sheet_panel``/``sprite_panel``'s
    viewer-bound caches do.
    """
    forms_by_job = ctx.state.preview.setdefault("remesh_forms", {})
    form = forms_by_job.get(job_id)
    if form is None:
        form = {
            "job_id": job_id,
            "remesh_profile": remesh.DEFAULT_TRIANGLE_PROFILE,
            "custom_triangles": remesh.TRIANGLE_PROFILES[remesh.DEFAULT_TRIANGLE_PROFILE],
            "texture_size": "",
            # True: measured 2026-09-23, without the voxel pre-pass the
            # decimate fallback a trellis mesh always takes collapses the
            # mesh rather than producing something usable.
            "close_holes": True,
        }
        forms_by_job[job_id] = form
    return form


def blender_available(ctx: Any) -> bool:
    """The rig door's own answer, unprobed: ``blender_check(probe=False)``
    returns the cached verdict or a pending row, and a pending row draws as
    "not yet" rather than blocking a frame on a subprocess."""
    try:
        from ... import doctor

        return bool(doctor.blender_check(probe=False).ok)
    except Exception:  # noqa: BLE001 - no doctor answer means no panel
        return False


# The private spelling stayed importable, the ``retarget_panel.gltfpack_available``
# pattern: this module's own callers and tests already spell the name with the
# underscore, and settings_3d._budget (2026-09-23) needs this exact answer to
# decide whether Create's own Budget combo offers the Game-ready rungs.
_blender_available = blender_available


def _warn_stale(ctx: Any, job: Any) -> None:
    files = set(job.get("files") or [])
    if "rig.glb" in files:
        widgets.text_colored(
            theme.WARN, "The rig, its poses and its sheets will describe the old mesh."
        )
    # A remesh replaces the whole mesh -- both geometry and skin -- and until
    # pipelines.modelhistory landed (2026-09-22) that replacement was
    # simply gone. It is kept under Earlier meshes now, and a submit here is
    # no longer the one-way trip the panel used to be silent about.
    widgets.muted_wrapped(
        "The current mesh is kept under Earlier meshes below, so this can be undone."
    )


def dependent_job_reason(jobs: list[dict[str, Any]], job_id: str) -> str | None:
    """Whether an unfinished sibling job still writes into this asset's directory.

    The 2026-09-06 audit, finding create-02: matches ``retarget_panel``'s
    function of the same name -- see it for why this reads ``ctx.cache.jobs``
    rather than calling ``_jobs_lifecycle.dependent_jobs`` from the frame
    thread.
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


def validate(form: dict[str, Any]) -> list[str]:
    """The refusals stated before the button; ``remesh.resolve`` is the rule."""
    try:
        remesh.resolve(
            form["remesh_profile"],
            int(form["custom_triangles"]) if form["remesh_profile"] == "custom" else None,
        )
    except (ValueError, TypeError) as exc:
        return [str(exc)]
    return []


def submit_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The keyword arguments ``remesh_job`` is called with, from the form."""
    return {
        "profile": form["remesh_profile"],
        "custom_triangles": (
            int(form["custom_triangles"]) if form["remesh_profile"] == "custom" else None
        ),
        "texture_size": int(form["texture_size"]) if form["texture_size"] else None,
        "close_holes": bool(form["close_holes"]),
    }


def _submit(ctx: Any, job_id: str, form: dict[str, Any]) -> None:
    key = f"remesh:{job_id}"
    busy = ctx.busy(key)
    problems = validate(form)
    for problem in problems:
        widgets.muted(problem)
    dep_reason = None if busy else dependent_job_reason(ctx.cache.jobs, job_id)
    if dep_reason:
        widgets.text_colored(theme.WARN, dep_reason)
    if busy:
        widgets.busy("Queueing the remesh")
    if widgets.disabled_button(
        "Remesh and rebake",
        not problems and not busy and not dep_reason,
        (-1, 0),
        reason=(
            "A remesh is already being queued for this asset."
            if busy
            else dep_reason or "; ".join(problems)
        ),
    ):
        # Last time's rings first: a new submit is judged on its own.
        ctx.state.clear_field_errors()
        ctx.submit(key, _remesh_job, ctx.svc, job_id, **submit_kwargs(form))
