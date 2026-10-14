"""Change a finished asset's size, pivot or front without rebuilding it.

Every other control on the Mesh stage that acts on a finished model replaces it:
a retarget rebuilds from the reconstruction, a remesh and a re-texture bake a new
one. None of them is the answer to "the barrel is the right shape and the wrong
size", "its pivot is in the air" or "it faces the wrong way", and until
``service.jobs.reorient_model`` existed those were fixed by re-importing into a
DCC tool. This panel is that door's caller.

The two presses are separate on purpose. **Apply** is a resize and/or re-ground,
which moves geometry but turns nothing, so it is allowed on a rigged asset and
the service reports the rig as stale. **Make current front the model's front**
rotates the mesh so the front a person set in the viewport becomes +Z, and
that is refused while a rig exists -- the rig and its poses were fitted to the
mesh as it stood. The reason is stated on the disabled button, not discovered
after a press.
"""

from __future__ import annotations

from typing import Any

from ...service import jobs as svc_jobs
from .. import forms, theme, widgets
from ..manual import render as manual_render
from . import remesh_panel

# The same flat, unnamespaced ``ctx.state.field_errors`` the other panels share:
# ``size_m`` is also Create's own Mesh-settings field, and a refusal about this
# panel's resize ringing that control (or the reverse) is the 2026-09-07
# audit's create-01 again. The widget is keyed by the prefixed name and the
# door's refusal is relabelled before it reaches the shared dict.
_FIELD_PREFIX = {"size_m": "reorient_size_m"}

DEFAULT_SIZE_M = 1.0


def _reorient_model(svc: Any, job_id: str, **kwargs: Any) -> dict[str, Any]:
    """``reorient_model``, with a colliding refusal's address renamed to this
    panel's control. See ``_FIELD_PREFIX``."""
    try:
        return svc_jobs.reorient_model(svc, job_id, **kwargs)
    except Exception as exc:
        renamed = _FIELD_PREFIX.get(getattr(exc, "field", None))
        if renamed:
            exc.field = renamed
        raise


def current_size_m(job: Any) -> float:
    """What to put in the size box: the size the mesh was asked for, else the
    size it was measured at, else one metre. A starting point, never a claim."""
    params = job.get("params") or {}
    for value in (params.get("size_m"), (params.get("mesh_report") or {}).get("achieved_size_m")):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return round(float(value), 3)
    return DEFAULT_SIZE_M


def front_reason(job: Any) -> str:
    """Why "Make current front the model's front" is unavailable, or ``""``.

    Mirrors the door's two ``field="front_yaw"`` refusals, so the button says no
    before the press instead of a toast after it.
    """
    front = float((job.get("params") or {}).get("front_yaw") or 0.0)
    if not front:
        return "This asset has no recorded front to turn the mesh by."
    files = set(job.get("files") or [])
    if "rig.glb" in files or "rig.json" in files:
        return (
            "This asset has a rig fitted to its current orientation; turning only the mesh "
            "would leave the rig describing a different one."
        )
    return ""


def draw(ctx: Any, job: Any) -> None:
    files = job.get("files") or []
    if "model.glb" not in files or job.get("status") != "done":
        return
    if not widgets.header("Finished asset", default_open=False):
        return
    manual_render.help_button(ctx, "reorient")

    job_id = job["id"]
    form = _form(ctx, job)
    busy = ctx.busy(f"reorient:{job_id}")
    dep_reason = None if busy else remesh_panel.dependent_job_reason(ctx.cache.jobs, job_id)

    with forms.Form(
        "reorient-settings",
        errors=ctx.state.field_errors,
        on_edit=ctx.state.clear_field_error,
    ) as form_ui:
        changed, value = form_ui.number(
            "reorient_size_m",
            "Size (m)",
            float(form["size_m"]),
            help_text="Scale the model so its longest side is this many metres.",
            helper="The mesh keeps its shape; only its scale changes.",
        )
        if changed:
            form["size_m"] = float(value)
        _changed, form["ground"] = form_ui.switch(
            "reorient_ground",
            "Ground",
            bool(form["ground"]),
            help_text="Centre the model on the origin and put its lowest point on the floor.",
        )

    problems = validate(form, job)
    for problem in problems:
        widgets.muted(problem)
    if "rig.glb" in set(files):
        widgets.text_colored(
            theme.WARN, "The rig, its poses and its sheets will describe the old mesh."
        )
    if dep_reason:
        widgets.text_colored(theme.WARN, dep_reason)
    if busy:
        widgets.busy("Changing the mesh")
    if widgets.disabled_button(
        "Apply size and ground",
        not problems and not busy and not dep_reason,
        (-1, 0),
        reason=(
            "A change is already running for this asset."
            if busy
            else dep_reason or "; ".join(problems)
        ),
    ):
        ctx.state.clear_field_errors()
        ctx.submit(
            f"reorient:{job_id}",
            _reorient_model,
            ctx.svc,
            job_id,
            **submit_kwargs(form, job),
        )

    reason = front_reason(job)
    front = float((job.get("params") or {}).get("front_yaw") or 0.0)
    # Only for an asset that already carries a front: the viewport control that set one left with
    # Poser (the 2026-10-13 Clay rig plan), so a front recorded earlier can still be baked into the
    # mesh but a new one cannot be recorded, and a button that could only ever be greyed would
    # point at a control that no longer exists.
    if front and widgets.disabled_button(
        "Make current front the model's front",
        not reason and not busy and not dep_reason,
        (-1, 0),
        reason=reason
        or ("A change is already running for this asset." if busy else dep_reason or ""),
        tooltip=(
            f"Turns the mesh {-front:.0f} degrees about the vertical so the direction you "
            "set as its front faces +Z, where every engine expects it."
        ),
    ):
        ctx.state.clear_field_errors()
        ctx.submit(
            f"reorient:{job_id}",
            _reorient_model,
            ctx.svc,
            job_id,
            ground=True,
            bake_front=True,
        )
    widgets.muted_wrapped(
        "The current mesh is kept under Earlier meshes below, so this can be undone. "
        "Only the last four changes are kept."
    )


def _form(ctx: Any, job: Any) -> dict[str, Any]:
    forms_by_job = ctx.state.preview.setdefault("reorient_forms", {})
    form = forms_by_job.get(job["id"])
    if form is None:
        form = {"size_m": current_size_m(job), "ground": True}
        forms_by_job[job["id"]] = form
    return form


def validate(form: dict[str, Any], job: Any) -> list[str]:
    """The refusals stated before the button; ``guidance``'s bounds are the rule."""
    from ... import guidance

    try:
        size = float(form["size_m"])
    except (TypeError, ValueError):
        return ["Size must be a number."]
    if not guidance.SIZE_MIN_M <= size <= guidance.SIZE_MAX_M:
        return [f"Size must be between {guidance.SIZE_MIN_M:g} and {guidance.SIZE_MAX_M:g} m."]
    if submit_kwargs(form, job)["size_m"] is None and not form["ground"]:
        return ["Change the size or turn Ground on."]
    return []


def submit_kwargs(form: dict[str, Any], job: Any) -> dict[str, Any]:
    """The keyword arguments the Apply press calls ``reorient_model`` with.

    The size is sent only when it differs from what the mesh already has, so
    pressing Apply to re-ground a model never rescales it by rounding noise.
    """
    size = float(form["size_m"])
    return {
        "size_m": None if abs(size - current_size_m(job)) < 5e-4 else size,
        "ground": bool(form["ground"]),
    }
