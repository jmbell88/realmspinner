"""What a 3D (mesh) recipe means: validation, kwargs, the findings hint.

Split out of ``modes/create/ui/settings_3d.py`` (2026-09-18 restructure, P5).
Smaller than ``recipe.py``'s share of the 2D pane, because most of what that
pane's non-drawing half does elsewhere -- which reference is the source, what
a selected finished mesh's parent is -- reaches into ``modes/create/ui/
stages.py`` for ``create_stages.parent``, and an engine module may not import
a mode's ``ui/`` package even where the function it would otherwise host
draws nothing. That whole cluster (``_source_param``, ``_inherit_label``,
``_platform_options``, ``_bg_options``, ``_selected_mesh``,
``_effective_source``) stays in ``settings_3d.py`` for exactly that reason.

Renamed on the way in, matching ``recipe.py``'s rule: ``_findings_hint`` ->
:func:`findings_hint`, ``_engine_kwargs`` -> :func:`engine_kwargs``,
``_matte_is_clean`` -> :func:`matte_is_clean`, ``_upload_kwargs`` ->
:func:`upload_kwargs`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..... import vectors
from .....bench import findings as findings_lib
from .....service.validation import (
    MAX_MESH_CANDIDATES,
    MAX_TRELLIS_BAND,
    MAX_TRELLIS_TEX_RES,
    MIN_TRELLIS_BAND,
    MIN_TRELLIS_TEX_RES,
    random_seed,
)
from .... import problems


def findings_hint(ctx: Any, param: str, value: Any) -> str | None:
    """Same lookup as the 2D pane's -- see ``recipe.findings_hint``.

    The subject comes from the source asset rather than from a form, because
    this pane owns no prompt controls at all: a 3D job starts from a finished
    2D reference and inherits its prompt, so that reference's prompt *is* the
    subject the mesh will be of. With no source picked yet there is no subject
    to scope by and the pooled corpus answers, unlabelled -- which is honest:
    nothing has been chosen for a hint to be about.
    """
    doc = findings_lib.load(Path(ctx.svc.config.bench_dir) / "findings.json")
    source = ctx.cache.get(ctx.state.source_job)
    subject = vectors.prompt_hash(source.get("prompt")) if source else ""
    return findings_lib.hint(doc, param, value, prompt_hash=subject or None)


def candidate_count(form: dict[str, Any]) -> int:
    """How many meshes this form asks for, clamped to what the service admits.

    Clamped here as well as refused there because the form is persisted: a
    settings file written when the ceiling was higher (or edited by hand) would
    otherwise send a number ``promote_candidates`` refuses, and the refusal
    would arrive as an error toast on a control the user cannot see is wrong.
    """
    try:
        count = int(form.get("candidates", 1))
    except (TypeError, ValueError):
        return 1
    return max(1, min(count, MAX_MESH_CANDIDATES))


def validate(source: dict[str, Any] | None) -> list[problems.Problem]:
    """``recipe.validate``'s counterpart, and its field rule.

    All three of these are about the *reference*, not about a control in this
    form, so none carries a field: they are refusals the library answers, and
    ringing a widget in the promotion form would point at the wrong thing.
    ``note_field_error`` already treats an empty field as "keep going to the
    toast", which is the behaviour that leaves.
    """
    if source is None:
        return [problems.Problem("Choose a reference first.")]
    if source.get("status") != "done":
        return [problems.Problem(f"That reference is {source.get('status')}.")]
    if "input.png" not in (source.get("files") or []):
        return [problems.Problem("That reference has no image.")]
    return []


def engine_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The engine axes as override kwargs, with "still at its sentinel" left
    out entirely -- ``promote_kwargs``'s own rule, restated once and shared
    with :func:`upload_kwargs` rather than duplicated into it: a form field
    honoured for a promoted reference and quietly ignored for a dropped file
    is exactly the bug ``upload_kwargs``'s docstring already names for every
    other field here.
    """
    out: dict[str, Any] = {}
    if int(form["trellis_band"]) > 0:
        out["trellis_band"] = int(form["trellis_band"])
    if int(form["trellis_tex_res"]) > 0:
        out["trellis_tex_res"] = int(form["trellis_tex_res"])
    if float(form["trellis_gss"]) > 0:
        out["trellis_gss"] = float(form["trellis_gss"])
    if float(form["trellis_gsh"]) > 0:
        out["trellis_gsh"] = float(form["trellis_gsh"])
    if int(form["trellis_max_tokens"]) > 0:
        out["trellis_max_tokens"] = int(form["trellis_max_tokens"])
    # >= 0, not > 0: 0 is decimation off, a real value the exe must receive,
    # and only -1 (DEFAULT_FORM_3D's sentinel) means "unset" for this one.
    if int(form["trellis_decim"]) >= 0:
        out["trellis_decim"] = int(form["trellis_decim"])
    if int(form["trellis_atlas"]) > 0:
        out["trellis_atlas"] = int(form["trellis_atlas"])
    return out


def clamp_tex_res(value: int) -> int:
    """The Texture resolution field's value, held inside the door's range.

    0 (and anything below it) stays the "unset" sentinel. Everything else is
    pulled into ``check_trellis_tex_res``'s bounds: the field used to clamp
    only at 0, so a typed 64 was kept, persisted, and refused at every Accept
    with a toast naming a control inside a collapsed header -- a reference
    that could never become a mesh and a log that said nothing (2026-09-18).
    """
    if value <= 0:
        return 0
    return min(max(value, MIN_TRELLIS_TEX_RES), MAX_TRELLIS_TEX_RES)


def clamp_band(value: int) -> int:
    """The Band field's value, held inside ``check_trellis_band``'s range.

    :func:`clamp_tex_res`'s twin, for the same defect: the field clamped only
    at 0, so a typed 65 was kept and refused at every Accept.
    """
    if value <= 0:
        return 0
    return min(max(value, MIN_TRELLIS_BAND), MAX_TRELLIS_BAND)


def promote_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The overrides, with "unset" left out entirely.

    Omitted means "keep what the reference recorded", and that is not the same
    as sending the reference's value back. ``promote_to_model`` drops the
    inherited resolution unconditionally -- re-deriving it from a platform
    override, or from the default 3D platform when there is none -- so this
    pane sends no ``resolution`` at all: an override here would pin a number
    the platform no longer implies.
    """
    out: dict[str, Any] = {}
    if form["platform"]:
        out["platform"] = form["platform"]
    if float(form["size_m"]) > 0:
        out["size_m"] = float(form["size_m"])
    if form["bg_removal"]:
        out["bg_removal"] = form["bg_removal"]
    if form["profile"]:
        out["profile"] = form["profile"]
        if int(form["custom_triangles"]) > 0:
            out["custom_triangles"] = int(form["custom_triangles"])
    if int(form["mesh_seed"]) > 0:
        out["mesh_seed"] = int(form["mesh_seed"])
    # An explicit False, not an omission: it has to clear a rig request the
    # reference inherited, or a reference generated with rigging on would rig
    # every promotion of it whatever this pane says.
    out["rig"] = bool(form["rig"])
    if form["rig"] and form["rig_template"]:
        out["rig_template"] = form["rig_template"]
    # Explicit like rig, and for the same reason: an omission would let the
    # promotion inherit whatever the reference recorded, and this checkbox is
    # the 3D pane's decision, not the reference's.
    out["reference_prep"] = bool(form["reference_prep"])
    out.update(engine_kwargs(form))
    return out


def reroll_mesh_seed(form: dict[str, Any]) -> None:
    """After an *accepted* submit, and only then: the seed the job was made
    with is already in ``kwargs``, so what moves here is the form's next one.
    ``recipe.generate`` does the same for the image seed."""
    if not form.get("mesh_seed_locked", False):
        form["mesh_seed"] = random_seed()


def matte_is_clean(preview: Any) -> bool:
    """Whether ``preview`` is boring enough that the setting may skip it.

    The 2026-09-07 review, item 5.4's bar, verbatim: the composition gate
    raised nothing at all -- neither a hard refusal (``reasons``) nor a soft
    one (``warnings``) -- and the cut itself came from BiRefNet rather than
    the corner-fill fallback (``settings_3d.MATTE_SOURCES``' "the model's
    weights are not installed" case) or an alpha the reference already
    carried. ``approved`` is deliberately not enough on its own: the review
    names the *backend*, and a pre-matted image can carry an edge nobody here
    has ever looked at.
    """
    return preview.source == "birefnet" and not preview.reasons and not preview.warnings


def upload_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The 3D form as create_job keyword arguments.

    Shared by both upload paths so a form field cannot be honoured for a
    dropped file and quietly ignored for a rendered one.
    """
    kwargs: dict[str, Any] = {"kind": "image"}
    if form["platform"]:
        kwargs["guidance_fields"] = {"platform": form["platform"]}
    if float(form["size_m"]) > 0:
        kwargs["size_m"] = float(form["size_m"])
    if form["bg_removal"]:
        kwargs["bg_removal"] = form["bg_removal"]
    if form["profile"]:
        kwargs["profile"] = form["profile"]
        # The 2026-09-06 audit (create2-05): this helper's own docstring states
        # the rule -- a form field cannot be honoured for a dropped file and
        # quietly ignored for a rendered one -- and this line broke it. Without
        # it, an uploaded reference with a custom triangle budget reached
        # ``resolve_profile``/``optimize.resolve`` with ``custom=None`` and was
        # refused, even though ``promote_kwargs`` sends the exact same pair for
        # a promoted reference.
        if int(form["custom_triangles"]) > 0:
            kwargs["custom_triangles"] = int(form["custom_triangles"])
    if int(form["mesh_seed"]) > 0:
        kwargs["mesh_seed"] = int(form["mesh_seed"])
    kwargs["reference_prep"] = bool(form["reference_prep"])
    if form["rig"]:
        kwargs["rig"] = True
        if form["rig_template"]:
            kwargs["rig_template"] = form["rig_template"]
    kwargs.update(engine_kwargs(form))
    return kwargs
