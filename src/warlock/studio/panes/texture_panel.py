"""Give a finished mesh a new surface, from a prompt.

The retarget panel's sibling and it is placed beside it deliberately: both are
"an operation on a selected done job", which is the shape the mode list is
closed against. A re-texture is not a mode, it is a button on an asset.

Where the two differ is what they cost and what they invalidate, and the panel
says both before the button rather than after. A retarget is a couple of seconds
of gltfpack; this queues six SDXL passes around two Blender runs, so it is a
job with a place in the queue. And a retarget makes a rig, its poses and its
sheets describe a mesh that no longer exists, where a re-texture makes none of
them stale -- a rig references geometry, not pixels. What it *does* invalidate
is the exports that carry the skin, and the service names them.

The bake's one limitation is stated per run rather than papered over: an
*un-anchored* run has no occlusion test, so an overhang's front-view colours
smear onto whatever hides behind it -- that is the warning, shown exactly when
it is true. The "anchor to geometry" checkbox turns on the depth pass: every
restyle is held to the mesh's own rendered depth through a ControlNet, and the
backprojection depth-tests each texel, so the smear goes and coverage comes
from what the cameras genuinely saw. The report line says which kind of run
produced it, because the two coverage figures do not mean the same thing.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ... import models
from ...pipelines import retexture
from ...service import jobs as svc_jobs
from ...service.validation import MAX_PROMPT
from .. import controls, forms, theme, widgets
from ..manual import render as manual_render

# The 2026-09-07 audit, finding create-01: ``ctx.state.field_errors`` is one
# flat, unnamespaced dict, and this panel's "strength" and "texture_size"
# controls are also ``sheet_panel``'s and ``remesh_panel``'s bare ids for
# their own, unrelated fields -- so a refusal from either sibling door rang
# this panel's control too, whenever the inspector drew them together (which
# is routine). The widgets below are keyed by the prefixed names, and
# ``_retexture_job`` relabels ``retexture_job``'s own refusal to match before
# it ever reaches ``ctx.state.field_errors`` -- the door itself is shared with
# the retired HTTP API and is not this panel's name to change.
#
# The 2026-09-08 audit, finding create-02: "prompt" needed the same
# treatment. It is also ``create_brief.py``'s bare id for the Reference
# stage's main asset prompt, and the id ``settings_2d.validate()`` files an
# empty-prompt refusal under -- so an ordinary "the asset prompt is empty"
# refusal on Create's Reference stage rang this panel's Surface field
# whenever a mesh's inspector happened to be open at the same time, and
# editing either field silently cleared the other's ring
# (``clear_field_error("prompt")``).
_FIELD_PREFIX = {
    "strength": "retexture_strength",
    "texture_size": "retexture_texture_size",
    "prompt": "retexture_prompt",
}


def _retexture_job(svc: Any, job_id: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
    """``retexture_job``, with a colliding refusal's address renamed to this
    panel's own control. See ``_FIELD_PREFIX`` above."""
    try:
        return svc_jobs.retexture_job(svc, job_id, prompt, **kwargs)
    except Exception as exc:
        renamed = _FIELD_PREFIX.get(getattr(exc, "field", None))
        if renamed:
            exc.field = renamed
        raise


def draw(ctx: Any, job: Any) -> None:
    files = job.get("files") or []
    if "model.glb" not in files:
        # Nothing to re-skin. A reference, a tile and a job that never produced
        # a mesh are all in this state.
        return
    if not widgets.header("Surface texture", default_open=False):
        return
    manual_render.help_button(ctx, "retexture")

    job_id = job["id"]
    form = _form(ctx, job_id)

    # The 2026-09-06 audit, finding create-01: this pane wired neither
    # ``errors=`` nor ``on_edit=``, so a ``retexture_job`` refusal recorded
    # against "prompt", "strength", "control" or "control_scale" rang no
    # control here and never cleared -- unlike its two siblings
    # (``remesh_panel``, ``retarget_panel``), which both wire this and clear
    # last time's rings before a fresh submit.
    with forms.Form(
        "retexture-settings",
        errors=ctx.state.field_errors,
        on_edit=ctx.state.clear_field_error,
    ) as form_ui:
        _changed, form["prompt"] = form_ui.multiline(
            # Not the bare "prompt" id: create-02, 2026-09-08 audit -- see
            # ``_FIELD_PREFIX`` above.
            "retexture_prompt",
            "Surface",
            form["prompt"],
            MAX_PROMPT,
            helper="Describe the surface: rusted iron, mossy stone, painted wood.",
        )
        # ``strength`` is a 0..1 denoising fraction drawn as the percentage the
        # reader thinks in -- ``form_ui.slider`` has no ``percent`` mode (only
        # ``widgets.labeled_slider_float`` does), so the scaling is done here
        # by hand and undone on the way back out.
        _changed, shown = form_ui.slider(
            "retexture_strength",
            "Restyle strength",
            float(form["strength"]) * 100.0,
            models.RETEXTURE_STRENGTH_MIN * 100.0,
            # The re-texture's own ceiling, not the sheets': see models.py.
            models.RETEXTURE_STRENGTH_MAX * 100.0,
            fmt="%.0f%%",
            help_text=(
                "How far each rendered view is taken from the mesh's current look. "
                "Low keeps the shapes and recolours them; high reinterprets them. "
                "The top of the range is only worth it with the geometry anchor on."
            ),
        )
        form["strength"] = shown / 100.0
        _changed, form["depth"] = controls.checkbox(
            "Anchor to geometry (depth)",
            bool(form["depth"]),
            tooltip=(
                "Renders the mesh's own depth from every view. Each restyle is "
                "held to it (a depth ControlNet), and colours stop reaching "
                "surfaces hidden behind overhangs (a per-texel depth test). "
                "Needs the depth ControlNet from Settings -> Models."
            ),
        )
        if form["depth"]:
            _changed, form["control_scale"] = form_ui.slider(
                "control_scale",
                "Anchor strength",
                float(form["control_scale"]),
                models.CONTROL_SCALE_MIN,
                models.CONTROL_SCALE_MAX,
                fmt="%.2f",
                help_text=(
                    "How firmly each restyle is held to the rendered depth. The "
                    "default is the model's own."
                ),
            )
        _changed, form["texture_size"] = form_ui.combo(
            "retexture_texture_size",
            "Atlas size",
            form["texture_size"],
            [("", "Match the mesh")]
            + [(str(s), f"{s} px") for s in retexture.TEXTURE_SIZES],
            help_text=(
                "The default keeps the mesh's current atlas resolution. A smaller "
                "one would also shrink the parts no view covers, which keep their "
                "old colour."
            ),
        )

        _warn(ctx, job, form)
        _submit(ctx, job_id, form)
    _report(job)


def _form(ctx: Any, job_id: str) -> dict[str, Any]:
    """Kept on app state, keyed by job id, so a prompt typed against one mesh
    is never submitted against another -- and, since ``remesh_panel``'s fix
    (commit 89cb6412), so it survives a glance at another asset too.

    The 2026-09-08 audit, finding create-02: this used to be one shared slot
    compared by ``form.get("job_id") != job_id``, the pattern commit
    89cb6412 fixed on ``remesh_panel``/``sheet_panel``/``sprite_panel`` but
    left standing here -- so looking at a different card in the Library and
    coming back silently reset a typed surface description to defaults, with
    no undo. A surface description is a sentence somebody wrote, not a tier
    they clicked, which is what made the loss worth a dict instead of a
    comment.
    """
    forms_by_job = ctx.state.preview.setdefault("retexture_forms", {})
    form = forms_by_job.get(job_id)
    if form is None:
        form = {
            "job_id": job_id,
            "prompt": "",
            "strength": models.RETEXTURE_DEFAULT_STRENGTH,
            "texture_size": "",
            # On by the 2026-08-15 retexture-visibility measurement: the
            # anchor took the overhang fixture's hidden-region smear from
            # 99.5% to 0.05%, and ~16 pp of the un-anchored coverage figure
            # was paint on surfaces no camera saw. Unchecking reproduces the
            # baseline arm exactly.
            "depth": True,
            "control_scale": models.CONTROLNETS["depth"].default_scale,
        }
        forms_by_job[job_id] = form
    return form


def occlusion_note(depth_on: bool) -> str | None:
    """The overhang warning, exactly when it is true.

    A property of the run being configured rather than of the feature: an
    anchored run depth-tests every texel, and warning anyway would teach the
    user the checkbox does nothing.
    """
    if depth_on:
        return None
    return (
        "No occlusion test: colours from a facing view can reach surfaces "
        "hidden behind an overhang. Anchoring to geometry removes this."
    )


def _warn(ctx: Any, job: Any, form: dict[str, Any]) -> None:
    """What this will invalidate, and what it deliberately will not.

    The 2026-09-08 audit, finding create-04: this used to call
    ``svc_jobs.stale_surface_artifacts``, three ``Path.exists()`` filesystem
    checks, every frame the "Surface texture" section is open -- disk I/O on
    the frame thread, which ``CLAUDE.md`` names as a hard constraint. Its
    siblings (``remesh_panel._warn_stale``, ``retarget_panel._warn_stale``)
    answer the equivalent question with a plain membership test against
    ``job.get("files")``, the already-cached, jobs_cache-refreshed list; this
    does the same, against the same allowlist ``stale_surface_artifacts``
    checks on disk.
    """
    present = set(job.get("files") or [])
    stale = [name for name in retexture.SURFACE_DERIVED if name in present]
    if stale:
        widgets.text_colored(
            theme.WARN, "These exports will be rebuilt: " + ", ".join(stale)
        )
    if "rig.glb" in set(job.get("files") or []):
        # Said out loud rather than left as an absence: the retarget panel two
        # headers up warns about exactly these, so silence here would read as an
        # oversight rather than as the answer.
        widgets.muted("The rig, its poses and its sheets are unaffected.")
    note = occlusion_note(bool(form["depth"]))
    if note:
        widgets.muted(note)


def submit_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The keyword arguments ``retexture_job`` is called with, from the form.

    A pure function because the conversion had a crash in it nothing could
    see: ``int("")`` raises, and ``""`` is exactly what the default "Match the
    mesh" combo option holds -- so the untouched form blew up in the draw
    frame before ``ctx.submit`` ever ran. The control keys are *absent* rather
    than ``None`` when the anchor is off, matching what the door stores.
    """
    kwargs: dict[str, Any] = {
        "strength": float(form["strength"]),
        "texture_size": int(form["texture_size"]) if form["texture_size"] else None,
    }
    if form.get("depth"):
        kwargs["control"] = "depth"
        kwargs["control_scale"] = float(form["control_scale"])
    return kwargs


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


def _submit(ctx: Any, job_id: str, form: dict[str, Any]) -> None:
    key = f"retexture:{job_id}"
    busy = ctx.busy(key)
    problems = validate(form)
    for problem in problems:
        widgets.muted(problem)
    dep_reason = None if busy else dependent_job_reason(ctx.cache.jobs, job_id)
    if dep_reason:
        widgets.text_colored(theme.WARN, dep_reason)
    if busy:
        widgets.spinner()
        imgui.same_line()
    if widgets.disabled_button(
        "Re-texture mesh",
        not problems and not busy and not dep_reason,
        (-1, 0),
        # ``retarget_panel``'s rule: the problems are listed above, so the
        # reason names the other gate and defers to the list otherwise.
        reason=(
            "A re-texture is already running for this asset."
            if busy
            else dep_reason or "; ".join(problems)
        ),
    ):
        # Last time's rings first: a new submit is judged on its own --
        # ``retarget_panel``/``remesh_panel``'s rule, missing here until the
        # 2026-09-06 audit's finding create-01.
        ctx.state.clear_field_errors()
        ctx.submit(
            key,
            _retexture_job,
            ctx.svc,
            job_id,
            form["prompt"].strip(),
            **submit_kwargs(form),
        )


def report_line(report: Any) -> str | None:
    """One sentence about what the last run measured, or None.

    Anchored and un-anchored coverage do not mean the same thing -- one is
    "what the cameras genuinely saw", the other counts smear -- so the line
    says which kind of run produced it rather than leaving one number to be
    read as the other.
    """
    if not isinstance(report, dict):
        return None
    coverage = report.get("coverage")
    views = report.get("views")
    if coverage is None or views is None:
        return None
    line = (
        f"Last run: {float(coverage) * 100:.0f}% of the atlas covered by "
        f"{int(views)} views"
    )
    effective = report.get("coverage_effective")
    if report.get("occlusion_tested") and effective is not None:
        return f"{line}, {float(effective) * 100:.0f}% repainted (depth-tested)."
    return line + " (no occlusion test)."


def _report(job: Any) -> None:
    """What the last re-texture of this mesh actually managed.

    The coverage figure is the honest half: a low number means most of the
    mesh kept its old skin, which is a thing to be told rather than to
    discover by looking for it.
    """
    line = report_line((job.get("params") or {}).get("retexture"))
    if line:
        widgets.muted(line)


def validate(form: dict[str, Any]) -> list[str]:
    """The refusals stated before the button, matching what the service checks.

    Not the service's numeric ranges: those are enforced by the controls above,
    which cannot express an out-of-range value. What a control *can* express is
    an empty prompt, so that is what is stated.
    """
    if not form["prompt"].strip():
        return ["Describe the surface you want."]
    return []
