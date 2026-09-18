"""The 3D pane: drawing, and the orchestration of a press.

No prompt controls at all. A 3D job starts from a finished 2D asset (whose
reference is promoted) or from an uploaded image, and everything this pane
holds is an *override* on what that source already recorded -- which is why
every one of them is optional and "unset" is a real value rather than a
default in disguise.

**What a recipe means** -- validation, kwargs, the findings hint -- lives in
``modes/create/engine/mesh.py`` now (2026-09-18 restructure, P5), except the
cluster that reaches ``modes/create/ui/stages.py`` for the parent of a
selected finished mesh (``_source_param``, ``_inherit_label``,
``_platform_options``, ``_bg_options``, ``_selected_mesh``,
``_effective_source``): an engine module may not import a mode's ``ui/``
package, so that half stays here even though none of it draws.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from ...... import guidance, vectors
from ......bench import findings as findings_lib
from ......service import findings as svc_findings
from ......service import jobs as svc_jobs
from ......service import sheets as svc_sheets
from ......service.errors import Invalid
from ......service.validation import MAX_MESH_CANDIDATES, MAX_UPLOAD_BYTES, random_seed
from ..... import controls, dialogs, focus, forms, matte_preview, theme, widgets
from .....formvalues import coerce_form_value
from .....manual import render as manual_render
from .....panes import stage_rig
from .....tokens import sp
from ...engine import mesh as create_mesh
from .. import stages as create_stages

MATTE_TITLE = "Check the cutout"

# The 2026-09-07 review, item 5.4: an opt-in ``ctx.settings`` key (not a
# ``Config`` field -- this is a per-user UI preference, not a process setting,
# and app_settings.py's own toggles read and write ``ctx.settings`` directly
# the same way). Off by default, on purpose: "Put the matte in front of the
# two minutes of GPU" (see ``promote``'s docstring) does not change, and this
# key only ever lets a *clean* result skip the question, never a refused,
# warned or fallback-sourced one.
SKIP_CLEAN_MATTE_SETTING = "skip_clean_matte_preview"

# Where the last matte the setting above skipped past is kept, so the Mesh
# column can still show what Make 3D actually used (the review's ask, done
# with what this pane already owns rather than a new preview pipeline).
# ``state.preview`` because it is exactly this kind of frame-scoped, UI-only
# fact -- the sheet strip cache and the settings category tab already live
# there for the same reason.
_LAST_AUTO_MATTE_SLOT = "mesh_last_auto_matte"

# This pane's key in the focus ring; see ``settings_2d.FOCUS_PANE``.
FOCUS_PANE = "3d"

# What each of ``pipelines/matting``'s three sources is called on screen. The
# distinction matters to the user: the corner fill is a guess a plain
# background makes work, and BiRefNet is a model -- and "this image already
# carries one" is the answer that means their own edit is what will be used.
MATTE_SOURCES = {
    "alpha": "The reference's own alpha",
    "birefnet": "BiRefNet cutout",
    "flood": "Corner fill (BiRefNet's weights are not installed)",
}

# The only tier the UI offers. gltfpack is vendored now, so the named tiers can
# run -- but none of them has been qualified (kept UVs, both PBR maps and
# material assignment on a chest, a sword and a rock), and an unqualified tier
# on a generate form is a button that silently degrades a mesh. The retarget
# control in the inspector is the qualification path: it offers the whole list
# once the binary is present, so a tier can be exercised before it is exposed
# here.
# "As reconstructed", not "no decimation": the engine itself simplifies to
# ~300k faces at res 1024 before Warlock sees the mesh (config.trellis_decim).
PROFILES = [("raw", "Raw (as reconstructed, ~300k faces)")]


def draw(ctx: Any) -> None:
    """Draw the mesh form, including its "Mesh resolution" choice."""

    # Form.errors replaces field_error(ctx.state, "platform") and keeps the
    # service's field key attached to the shared control's ring and error copy.
    with forms.Form("create-3d", errors=ctx.state.field_errors) as form_ui:
        _draw_form(ctx, form_ui, "Mesh resolution")


def _draw_form(
    ctx: Any, form_ui: forms.Form, mesh_resolution_label: str
) -> None:
    state = ctx.state
    form = state.form_3d

    # The keyboard ring (UX.md Phase 3), over this pane's own controls. Shorter
    # than 2D's because the pane is: everything here is an override on what the
    # source reference recorded, so the path to a mesh is pick a source, press
    # the button -- and both ends of that are in the ring.
    focus.pump(state, FOCUS_PANE)
    focus.begin(state, FOCUS_PANE)
    widgets.section("Source")
    manual_render.help_button(ctx, "settings-3d")
    _source(ctx)

    widgets.section("Mesh")
    # Labels above rather than beside: a combo here is drawn at -1 width, and
    # imgui puts a widget's label to its *right* -- so every one of these was
    # a full-width select with its name clipped off the edge of the panel, and
    # "Detail", "Budget" and "Background" were invisible. ``labeled_combo`` is
    # the widget that already answers this, and the 2D pane's guidance grid
    # uses the same small-caps line above each control.
    # "Mesh resolution", not "Detail" (UX.md Phase 3). It and the 2D pane's own
    # ``platform`` control were called "Detail" and "platform detail", two
    # near-identical names for a geometry resolution and a prompt fragment, and
    # the whole of what kept them apart was a tooltip on each apologising for
    # the other. A name that says what the control *is* ends that; the tooltip
    # below keeps only the half that is still worth saying.
    before = form["platform"]
    with focus.item(ctx.state, FOCUS_PANE, "platform"):
        _changed, form["platform"] = form_ui.combo(
            "platform",
            mesh_resolution_label,
            form["platform"],
            _platform_options(ctx),
            help_text=(
                "How much geometry trellis is asked for. Higher costs more GPU "
                "and more triangles."
            ),
        )
    if form["platform"] != before:
        ctx.state.clear_field_error("platform")
    _hint(ctx, form, "platform", form["platform"])
    _budget(ctx, form)

    _size(ctx, form)
    # Deliberately unhinted, unlike every other control here: size_m is
    # continuous, so its buckets are keyed on "0.35" and "0.36" separately and
    # a threshold of five would essentially never be met.
    widgets.help_marker("0 keeps whatever the reference recorded.")

    with focus.item(ctx.state, FOCUS_PANE, "bg_removal"):
        _changed, form["bg_removal"] = form_ui.combo(
            "bg_removal",
            "Background",
            form["bg_removal"],
            _bg_options(ctx),
        )
    _hint(ctx, form, "bg_removal", form["bg_removal"])

    with focus.item(ctx.state, FOCUS_PANE, "mesh_seed"):
        changed, seed = form_ui.number(
            "mesh_seed", "Mesh seed", int(form["mesh_seed"])
        )
    if changed:
        form["mesh_seed"] = max(0, seed)
        ctx.state.clear_field_error("mesh_seed")
    # Rung, the same as the 2D pane's Seed row and for the same reason:
    # ``service.validation.check_seed`` raises ``Invalid(..., field="mesh_seed")``
    # for a seed outside 0..MAX_SEED or not an int, and ``create_job`` calls it
    # as ``check_seed("mesh_seed", mesh_seed)``. This widget's InputInt clamps
    # to a C int32 that happens to coincide with ``MAX_SEED``, which is the only
    # thing that keeps the refusal unreachable through it -- a seed loaded from
    # a hand-edited settings.json has no such ceiling. This pane had no comment
    # at all about the gap; ``settings_2d._seed_row`` had one and it was wrong
    # (the 2026-09-11 audit, finding create-07).
    widgets.field_error(ctx.state, "mesh_seed")
    if controls.button("Reroll##mesh", role=controls.ButtonRole.GHOST):
        form["mesh_seed"] = random_seed()
    # The 2D seed row's Lock, for the 2D seed row's reason: the engine is
    # deterministic in its seed, so two presses of Make 3D on one reference
    # with the seed left alone are the identical mesh twice, and that reads as
    # "the button did nothing". Unlocked, every *accepted* submit rerolls.
    changed, locked = form_ui.switch(
        "mesh_seed_locked",
        "Lock seed",
        bool(form.get("mesh_seed_locked", False)),
        help_text="Reuse this seed on the next Make 3D.",
        helper="Unlocked, every accepted Make 3D draws a fresh one.",
    )
    if changed:
        form["mesh_seed_locked"] = locked

    changed, prep = form_ui.switch(
        "reference_prep",
        "Normalise the reference",
        bool(form["reference_prep"]),
        help_text=(
            "Recentre the subject and scale it to fill the frame before the mesh "
            "engine sees it."
        ),
        helper=(
            "Off by default: the engine does its own cropping, and whether doing "
            "it twice helps has not been measured."
        ),
    )
    if changed:
        form["reference_prep"] = prep
    _hint(ctx, form, "reference_prep", form["reference_prep"])

    _rig(ctx, form)
    _engine(ctx, form, form_ui)
    _turnaround(ctx)
    _reset_row(ctx)
    _submit(ctx, form)


# --- pieces -----------------------------------------------------------------


def _reset_row(ctx: Any) -> None:
    """The 2D pane's *Reset...* had no counterpart here, and the asymmetry was
    the whole of the reason: both panes accumulate overrides across a session
    and only one of them offered a way back.

    Above the submit rather than below it, exactly as 2D places it: a
    destructive control under the primary action is one the hand reaches by
    accident.
    """
    if controls.button("Reset...", role=controls.ButtonRole.GHOST):
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Reset the model settings?",
                message=(
                    "The mesh resolution, size, background removal, seed, "
                    "candidate count and rig controls go back to their "
                    "defaults. The chosen source is kept, and the image form "
                    "is untouched."
                ),
                confirm_label="Reset",
                cancel_label="Cancel",
                on_confirm=lambda: _reset(ctx),
            )
        )


def _reset(ctx: Any) -> None:
    """The 3D form back to first-launch defaults.

    A fresh copy of ``DEFAULT_FORM_3D`` rather than a field-by-field clear, for
    ``settings_2d._reset``'s reason: a field added later is reset by having been
    added rather than by somebody remembering this function. A *copy*, because
    the module-level dict is the default and a form aliased onto it would edit
    it for the rest of the process -- which is the shape ``AppState.form_3d``'s
    own default_factory already takes.

    The source job is deliberately not cleared. It is not part of this form at
    all (it lives on ``state.source_job``), and a reset that silently dropped
    what the user had picked to work from would be a different, larger action
    than the one the button offers.
    """
    from .....state import DEFAULT_FORM_3D

    ctx.state.form_3d = dict(DEFAULT_FORM_3D)
    ctx.toast("The model settings are back to their defaults.")


def _hint(ctx: Any, form: dict[str, Any], param: str, value: Any) -> None:
    """Draw the findings hint for the control just drawn, if there is one,
    plus the offer to jump straight to what the evidence favours.

    This pane used to hint one control out of five, which put the evidence
    furthest from where it applies: an observation measures *geometry* -- hole
    fraction, watertightness, triangle count -- so the settings it can speak
    about most directly are exactly these, and they were the ones showing
    nothing. Every param here is in ``vectors.VECTOR_PARAMS``, so every one of
    them is something a verdict and an observation are already filed against.

    The 2026-09-07 review's ask, "findings become actionable at the control":
    the hint above says what the *current* value scored, and until now that
    was where it stopped -- a user agreeing had to go find the winning value
    and dial it in by hand. ``_best_value_offer`` is the click.
    """
    hint = create_mesh.findings_hint(ctx, param, value)
    if hint is not None:
        widgets.hint_text(hint)
    _best_value_offer(ctx, form, param, value)


def _best_value_offer(ctx: Any, form: dict[str, Any], param: str, value: Any) -> None:
    """"7/8 usable (47%+) · avg +2.9 · this subject" with a button, when the
    evidence favours a value other than the one already set.

    **Offered, never applied** -- ``_size_suggestion``'s shape (below), drawn
    here for every hinted control rather than only Size: a button that
    silently rewrote a slider the moment a sweep tipped the ranking would be
    indistinguishable from the app deciding the setting for the user, which is
    the one thing every findings surface in this app deliberately refuses to
    do. Silent when the current value already leads (``bench.findings.best_value``
    itself answers None then), because a button offering to set what is
    already set is not an offer, it is clutter.
    """
    doc = findings_lib.load(Path(ctx.svc.config.bench_dir) / "findings.json")
    source = ctx.cache.get(ctx.state.source_job)
    subject = vectors.prompt_hash(source.get("prompt")) if source else ""
    found = findings_lib.best_value(
        doc, param, value, min_n=svc_findings.PRESET_MIN_N, prompt_hash=subject or None
    )
    if found is None:
        return
    value_str, entry, scope = found
    widgets.muted(findings_lib.best_value_line(entry, scope))
    imgui.same_line()
    if controls.button(f"Use {value_str}##best-{param}"):
        form[param] = coerce_form_value(form[param], value_str)


# "Size (m)" as a drag rather than a slider (K96), and the *ceiling* is why: a
# slider needs a maximum and there is no largest asset -- a wall section is
# legitimately 8 m. A drag has the same feel, still honours a double-click for
# typed entry, and has no upper bound to be wrong about. The speed is what
# makes it usable: 1 cm per pixel, so the range a prop actually lives in (0.1
# to 2 m) is two hundred pixels of travel rather than four.
SIZE_DRAG_SPEED = 0.01
# ``v_min >= v_max`` is how imgui's drag widgets spell "unbounded". Stated as a
# named pair rather than as a literal ``0.0, 0.0`` so the intent survives
# somebody "fixing" it into a range.
SIZE_NO_BOUND = (0.0, 0.0)


def _size(ctx: Any, form: dict[str, Any]) -> None:
    """Metres, as a drag with the unit *in* the readout.

    "Size (m)" put the unit in the label and the number in the box, so a value
    read at a glance said "0.35" and the label it belonged to was a separate
    thing to look at. The format string carries it now, and 0 says what it
    means rather than showing a measurement of zero metres.
    """
    value = float(form["size_m"])
    fmt = "unset - keeps the reference's" if value <= 0.0 else "%.2f m"
    # Label above, matching the rest of the pane (2026-09-08 consistency
    # pass); the id is kept stable ("Size" visible -> "##Size" hidden) and
    # the unit stays in ``fmt``, which is the whole reason this is a drag and
    # not a slider (see the module comment above).
    widgets.field_label("Size")
    changed, size = controls.drag_float("##Size", value, SIZE_DRAG_SPEED, *SIZE_NO_BOUND, fmt)
    if changed:
        # Floored here rather than by the widget: unbounded means unbounded in
        # both directions, and a negative size is not a smaller asset.
        form["size_m"] = max(0.0, size)
    _size_suggestion(ctx, form)


def _size_suggestion(ctx: Any, form: dict[str, Any]) -> None:
    """"barrel -- usually 0.9 m", with a button, while the size is unset.

    Scale is the first thing an engine import gets wrong, and this control is
    opt-in: leave it alone and every asset lands at whatever the reference
    recorded, so a 0.15 m potion and a 2.1 m door are the same height in the
    scene. The table (``guidance.SIZE_HINTS_M``) is the cheap half of the fix.

    **Offered, never applied.** The unset value means "keeps the reference's"
    and that is a real answer a user may want; a table that filled the field in
    would turn every unconsidered press into a claim about scale nobody made,
    and would do it from a noun match crude enough to be wrong. So this draws
    only while the field is unset, and it takes a press -- the same shape as
    the 2D pane's ``_preflight_fix`` repairs.

    The noun comes from the source reference's prompt, because this pane owns no
    prompt controls at all: the 3D job inherits the 2D asset's words, which is
    the same reasoning ``create_mesh.findings_hint`` picks its subject by.
    """
    if float(form["size_m"]) > 0.0:
        return
    source = ctx.cache.get(ctx.state.source_job)
    hint = guidance.size_hint(source.get("prompt")) if source else None
    if hint is None:
        return
    noun, metres = hint
    widgets.muted(f"{noun} - usually {metres:g} m")
    imgui.same_line()
    if controls.button(f"Use {metres:g} m##size-hint"):
        form["size_m"] = float(metres)


def _budget(ctx: Any, form: dict[str, Any]) -> None:
    """The triangle budget -- drawn only when there is a choice to make.

    While :data:`PROFILES` has one entry there is nothing here a user can do.
    It used to be drawn anyway, disabled, with three lines explaining why: the
    argument was that a combo with a single entry looks broken, so saying
    "unqualified tier, not missing binary" beats saying nothing. But the
    control and its note are five lines of the densest form in the app, spent
    entirely on explaining their own inertness -- and the note's own answer is
    that the *inspector's* retarget control is where a tier gets tried. Send
    the user there by not putting a dead affordance in front of them here.

    The form key is untouched either way, so the door (``profile`` at submit)
    is unchanged: this stops drawing a control, it does not stop sending one.
    """
    if len(PROFILES) == 1:
        return
    form["profile"] = widgets.labeled_combo("Budget", form["profile"], PROFILES)
    widgets.field_error(ctx.state, "profile")
    _hint(ctx, form, "profile", form["profile"])
    if form["profile"] == "custom":
        # The same control the retarget panel draws, appearing under exactly
        # the same condition (K95). It is the widget ``custom_triangles`` never
        # had: the field was submitted, validated and recorded with no way to
        # set it, which is a form field that exists only for the API.
        widgets.field_label("Triangles")
        imgui.set_next_item_width(sp(140))
        changed, value = controls.input_int("##Triangles", int(form["custom_triangles"]), 0, 0)
        if changed:
            form["custom_triangles"] = max(0, value)


def _source_param(ctx: Any, key: str) -> str | None:
    """The reference job's own recorded value for ``key``, if it has one.

    Resolved through :func:`_effective_source`, not a plain read of
    ``ctx.state.source_job`` -- the 2026-09-08 audit, finding create-05: with
    no explicit pick but a finished mesh selected, ``_submit`` inherits from
    that mesh's *parent* reference (``_effective_source``), and its own muted
    line names that reference correctly. This function read ``source_job``
    directly and so named nothing, falling back to the generic "keep the
    reference's" right above a line that had already said what was actually
    being kept.

    The key is the same name a job's ``params`` dict was written under
    (``_q_generate.py``), so this is a plain lookup once the source is
    resolved, rather than a mapping this pane has to keep in sync.
    """
    source = _effective_source(ctx, ctx.cache.get(ctx.state.source_job))
    if source is None:
        return None
    return (source.get("params") or {}).get(key) or None


def _inherit_label(ctx: Any, key: str, label_for: dict[str, str] | None = None) -> str:
    """"keep the reference's" made concrete.

    The generic wording answered "what happens if I leave this alone" with
    "something, unnamed" -- correct, but a user picking a reference with a
    known ``platform``/``bg_removal`` already recorded had no way to see what
    that something *was* without leaving this pane. Falls back to the old
    text when the reference has no value for this key (no reference chosen
    yet, or an older job that predates the field).
    """
    value = _source_param(ctx, key)
    if value is None:
        return "keep the reference's"
    shown = (label_for or {}).get(value, value)
    return f"From reference: {shown}"


def _platform_options(ctx: Any) -> list[tuple[str, str]]:
    entries = (ctx.guidance.get("fields") or {}).get("platform") or []
    label_for = {e["key"]: e["label"] for e in entries}
    return [("", _inherit_label(ctx, "platform", label_for))] + [
        (e["key"], e["label"]) for e in entries
    ]


def _bg_options(ctx: Any) -> list[tuple[str, str]]:
    return [("", _inherit_label(ctx, "bg_removal"))] + [
        (key, key) for key in (ctx.guidance.get("bg_removal") or [])
    ]


def _source(ctx: Any) -> None:
    """The 2D asset this job starts from, or an upload.

    The whole block is one drag-and-drop target (I83): a card dragged out of
    the library lands here. A *group* rather than a child window, because a
    child clips and this content grows a line whenever a source is picked --
    the target has to be exactly the area the user is aiming at, at every
    display scale.
    """
    from .....panes import library

    state = ctx.state
    source = ctx.cache.get(state.source_job)
    mesh = None if source is not None else _selected_mesh(ctx)
    dragging = library.dragged_job(ctx)
    imgui.begin_group()
    origin = imgui.get_cursor_screen_pos()
    if source is not None:
        imgui.text_wrapped(source.get("name") or source.get("prompt") or source["id"])
        widgets.muted(f"reference - {source['id']}")
        if controls.button("Clear"):
            state.source_job = None
        _auto_matte_preview(ctx, source)
    elif mesh is not None:
        # The 2026-09-07 review, item 5.2: a finished mesh selected in the
        # library moves the viewport but never ``state.source_job`` (see
        # ``library.select``), so without this branch the block below read
        # "Pick a finished reference" over a mesh that plainly is one.
        _mesh_source(ctx, mesh)
    elif dragging is not None:
        # The invitation replaces the instruction only while something is in
        # the air: a line about dropping, with nothing to drop, is noise.
        widgets.muted("Drop it here to use it as the source.")
    else:
        widgets.muted("Pick a finished reference in the library, or:")
    busy = ctx.busy("upload")
    if widgets.disabled_button(
        "Open an image...", not busy, reason="A file picker is already open."
    ):
        ctx.submit("upload", dialogs.open_file, "Choose a reference image", dialogs.IMAGE_FILTER)
    widgets.muted("...or drop an image on the window.")
    imgui.end_group()
    end = imgui.get_item_rect_max()
    if dragging is not None:
        # A target the pointer is over says so, and one that is merely
        # *available* says that too but more quietly. Drawn after the group so
        # the outline is not clipped by it.
        hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_blocked_by_active_item.value)
        widgets.ring(
            origin,
            end,
            theme.ACCENT if hovered else theme.MUTED,
            0.9 if hovered else 0.4,
            2.0 if hovered else 1.0,
        )
    else:
        # The same ring, fading, for a file dropped from Explorer (H70): the
        # two arrivals look the same because they are the same event.
        widgets.ring(origin, end, theme.ACCENT, widgets.drop_flash(state, "3d-source"))
    if imgui.begin_drag_drop_target():
        payload = imgui.accept_drag_drop_payload_py_id(library.DRAG_JOB)
        if payload is not None and state.dragging_job:
            # Through ``library.select`` rather than by assigning ``source_job``
            # here: that function is what also moves the selection, so a
            # dropped card is the selected card and the inspector on the right
            # is showing the thing the form now names.
            library.select(ctx, state.dragging_job)
            state.source_job = state.dragging_job
            state.dragging_job = None
        imgui.end_drag_drop_target()


def _selected_mesh(ctx: Any) -> dict[str, Any] | None:
    """The selected job, if it is a finished mesh -- so this column can
    describe it instead of asking for a reference nobody was about to pick.

    The 2026-09-07 review, item 5.2, verified at ``library.select``:
    ``state.source_job`` is only ever set for a *done reference* row, so
    landing on a finished mesh moves the viewport and the selection but
    leaves ``source_job`` pointing at whatever reference (or nothing) had
    been picked before -- which is what read as "Choose a reference first"
    over a mesh that was plainly the thing on screen.

    ``ctx.job`` is asked for rather than assumed, ``create_stages._current``'s
    rule and for its reason: ``promote`` is reachable from the command palette
    and from ``main``, whose ctx objects are not guaranteed to carry a job
    cache -- and a refusal that has nothing to do with this fallback (the
    library's own "Choose a reference first") must not turn into an
    AttributeError on the way to being spoken.
    """
    getter = getattr(ctx, "job", None)
    job = getter() if callable(getter) else None
    if job is None or job.get("stage") != "model" or job.get("status") != "done":
        return None
    return job


def _mesh_source(ctx: Any, mesh: dict[str, Any]) -> None:
    """Describe an already-built mesh instead of drawing the reference picker.

    Drawn only while no explicit ``source_job`` is picked -- an explicit pick
    (a card dragged in, or Clear then a fresh choice) always wins, and this is
    the fallback for the one case that used to read as "nothing is chosen"
    while the viewport disagreed.
    """
    reference = create_stages.parent(ctx, mesh)
    imgui.text_wrapped(mesh.get("name") or mesh.get("prompt") or mesh["id"])
    if reference is not None:
        label = reference.get("name") or reference.get("prompt") or reference["id"]
        widgets.muted(f"mesh - built from {label}")
    else:
        # The parent has scrolled out of the loaded page (create_stages.parent's
        # own caveat) or the mesh predates parent_id being recorded at all.
        widgets.muted(f"mesh - {mesh['id']}")
    widgets.muted_wrapped(
        "This mesh is already built. Make 3D below rebuilds it from that reference."
    )


def _effective_source(ctx: Any, source: dict[str, Any] | None) -> dict[str, Any] | None:
    """``source`` if there is one, or -- item 5.2 -- the reference behind a
    selected finished mesh, so a submit from this column names and uses what
    is actually on screen rather than refusing over a stale or absent
    ``source_job``. ``source_job`` keeps meaning exactly what it always has,
    an explicit pick; this only answers the one case it cannot.
    """
    if source is not None:
        return source
    mesh = _selected_mesh(ctx)
    return create_stages.parent(ctx, mesh) if mesh is not None else None


def _auto_matte_preview(ctx: Any, source: dict[str, Any]) -> None:
    """The cutout Make 3D actually used, when item 5.4's setting skipped the
    modal for it.

    Kept only as long as it is still about the reference on screen: switching
    sources drops it implicitly, since the stored preview's ``job_id`` then no
    longer matches. Nothing is drawn when the setting has never fired, which
    is every session until it is turned on and a matte qualifies.
    """
    preview = ctx.state.preview.get(_LAST_AUTO_MATTE_SLOT)
    if preview is None or getattr(preview, "job_id", None) != source.get("id"):
        return
    widgets.muted("Cutout used for the last Make 3D (the preview was skipped):")
    _matte_image(ctx, preview)


def _rig(ctx: Any, form: dict[str, Any]) -> None:
    if not ctx.rigging_available:
        # Hidden rather than disabled: without bpy the whole feature is absent,
        # and a greyed control implies it could be turned on from here.
        return
    widgets.section("Rig")
    changed, rig = controls.checkbox("Rig when the mesh lands", bool(form["rig"]))
    if changed:
        form["rig"] = rig
    if form["rig"]:
        # The 2026-09-07 review, item 5.1: this combo used to be a second,
        # near-verbatim copy of ``stage_rig._skeleton_picker`` -- same field,
        # same create-05 fix, same comment -- which is exactly the shape that
        # let the two drift apart in the first place. ``stage_rig.py`` already
        # owns the skeleton every rig submission uses (see its module
        # docstring), so the drawing lives there and this just calls it. No
        # ``help_text`` here, same as before: the checkbox row has no room for
        # the longer explanation stage_rig gives on its own stage.
        stage_rig.skeleton_field(ctx, form)


def _engine(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """The seven trellis-server launch flags -- a findings sweep could already
    set every one of these (``service.sweeps.KWARG_AXES``); until this door
    existed no ordinary Create job could.

    Collapsed by default, under "Engine (advanced)": these are not mesh
    settings the way ``platform`` or ``bg_removal`` are, they are how the
    reconstruction *process* is launched, and every one of them restarts
    trellis-server for this job (``service.sweeps.SERVER_AXES`` is exactly
    this list, minus nothing). A pane most people never open should not cost
    the ordinary path a restart it never asked for, which is what a header
    that opens by default would risk the moment somebody brushed a value.

    Each control leaves its field at ``state.DEFAULT_FORM_3D``'s sentinel
    when untouched, and :func:`create_mesh.engine_kwargs` is what turns "still at the
    sentinel" into "omit the kwarg" -- the same shape ``_size`` and
    ``_budget`` already use for ``size_m`` and ``custom_triangles``. Findings
    hints follow the same lookup as every other control here (``_hint``); a
    verdict filed under one of these axes shows up against it exactly as one
    filed under ``resolution`` does.
    """
    opened = controls.collapsing_header("Engine (advanced)##engine")
    if not opened:
        return
    widgets.muted_wrapped(
        "Launch flags for the reconstruction engine itself, not the mesh. "
        "Changing any of these restarts it for this job."
    )
    changed, value = form_ui.number(
        "trellis_band", "Band", int(form["trellis_band"]),
        help_text=(
            "Narrow-band width in voxels for the DC remesh. 0 keeps the "
            "engine's own default."
        ),
    )
    if changed:
        form["trellis_band"] = max(0, int(value))
    _hint(ctx, form, "trellis_band", form["trellis_band"])

    changed, value = form_ui.number(
        "trellis_tex_res", "Texture resolution", int(form["trellis_tex_res"]),
        help_text=(
            "Baked PBR texture edge in px. 0 keeps the engine's own default."
        ),
    )
    if changed:
        form["trellis_tex_res"] = max(0, int(value))
    _hint(ctx, form, "trellis_tex_res", form["trellis_tex_res"])

    changed, value = form_ui.number(
        "trellis_gss", "Sparse-structure guidance", float(form["trellis_gss"]),
        help_text=(
            "Guidance strength for the sparse-structure stage. 0 keeps the "
            "engine's own default."
        ),
    )
    if changed:
        form["trellis_gss"] = max(0.0, float(value))
    _hint(ctx, form, "trellis_gss", form["trellis_gss"])

    changed, value = form_ui.number(
        "trellis_gsh", "Structured-latent guidance", float(form["trellis_gsh"]),
        help_text=(
            "Guidance strength for the structured-latent stage. 0 keeps the "
            "engine's own default."
        ),
    )
    if changed:
        form["trellis_gsh"] = max(0.0, float(value))
    _hint(ctx, form, "trellis_gsh", form["trellis_gsh"])

    changed, value = form_ui.number(
        "trellis_max_tokens", "Token budget", int(form["trellis_max_tokens"]),
        help_text=(
            "The engine's high-resolution token budget (it ships at 49152). "
            "0 keeps the engine's own default."
        ),
    )
    if changed:
        form["trellis_max_tokens"] = max(0, int(value))
    _hint(ctx, form, "trellis_max_tokens", form["trellis_max_tokens"])

    changed, value = form_ui.number(
        "trellis_decim", "Decimation", int(form["trellis_decim"]),
        help_text=(
            "The engine's own decimation. -1 keeps its default (quadric "
            "simplify to ~300k faces at resolution 1024); 0 turns it off and "
            "ships the full reconstruction; a positive grid picks the legacy "
            "cluster-grid pass."
        ),
    )
    if changed:
        form["trellis_decim"] = max(-1, int(value))
    _hint(ctx, form, "trellis_decim", form["trellis_decim"])

    changed, value = form_ui.number(
        "trellis_atlas", "Atlas resolution", int(form["trellis_atlas"]),
        help_text=(
            "UV atlas edge in px for the baked textures. 0 keeps the "
            "engine's own default."
        ),
    )
    if changed:
        form["trellis_atlas"] = max(0, int(value))
    _hint(ctx, form, "trellis_atlas", form["trellis_atlas"])


def _turnaround(ctx: Any) -> None:
    """"Render turnaround": the sprite-sheet control, reached from the mesh
    that already exists rather than from a rig-shaped stage.

    The 2026-09-07 review, item 7.1: ``docs/manual/27-sprite-sheets.md`` is
    plain that a turnaround needs no rig at all, but its only door was a
    collapsed header on the Pose stage (``sheet_panel.py``) -- so a finished,
    unrigged prop needed a stage built around a skeleton it does not have.
    This draws only while the job on screen is itself a finished mesh
    (:func:`_selected_mesh`) and submits through the identical door and key
    ``sheet_panel._submit`` does: ``svc_sheets.create_sheet`` under
    ``f"sheet:{job_id}"``. Same key means a press here and a press on the Pose
    stage's own button are the same in-flight submit as far as
    ``TaskRunner.submit`` is concerned, so a second one anywhere is refused
    exactly as it is there -- nothing here duplicates that door's own submit
    logic, it only reaches it from a second place.
    """
    mesh = _selected_mesh(ctx)
    if mesh is None or "model.glb" not in (mesh.get("files") or []):
        return
    from .....panes import sheet_panel

    job_id = mesh["id"]
    widgets.section("Turnaround")
    if not ctx.rigging_available:
        # Rendering a sheet is Blender out of process the same way rigging is
        # (S138); the pattern every "needs Blender" sentence in the app
        # follows (service/characters.py's own comment states it), applied to
        # this door.
        widgets.muted("Rendering a sheet needs Blender, which is not installed.")
        return
    key = f"sheet:{job_id}"
    busy = ctx.busy(key)
    saved = (ctx.state.preview or {}).get("sheets") or []
    cap_reason = None if busy else sheet_panel.sheet_cap_reason(saved, ctx.cache.jobs, job_id)
    widgets.cost_note(
        "Queued like a generation: the default 8-direction turnaround is "
        "eight Blender renders, run in a separate process."
    )
    if widgets.disabled_button(
        "Render turnaround",
        not busy and not cap_reason,
        reason="A sheet is already rendering for this asset." if busy else (cap_reason or ""),
    ):
        # Last time's rings first, sheet_panel._submit's own rule: a new
        # submit is judged on its own.
        ctx.state.clear_field_errors()
        defaults = (ctx.sheet_options or {}).get("defaults") or {}
        ctx.submit(
            key,
            svc_sheets.create_sheet,
            ctx.svc,
            job_id,
            poses=[],
            elevation=float(defaults.get("elevation") or 0.0),
            frame_size=int(defaults.get("frame_size") or 128),
            lighting=defaults.get("lighting") or "flat",
            name="",
            clip_from=None,
            clip_to=None,
            clip_frames=8,
            yaws=8,
        )


def _submit(ctx: Any, form: dict[str, Any]) -> None:
    imgui.dummy((0, sp(8)))
    widgets.divider()
    state = ctx.state
    explicit = ctx.cache.get(state.source_job)
    source = _effective_source(ctx, explicit)
    problems = create_mesh.validate(source)
    for problem in problems:
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
        imgui.text_wrapped(problem)
        imgui.pop_style_color()
    _candidates(form)
    count = create_mesh.candidate_count(form)
    if explicit is None and source is not None:
        # Item 5.2: naming the reference this button would actually use, since
        # it is not the one the user last explicitly picked -- it is the
        # parent of a selected finished mesh (``_effective_source``).
        label = source.get("name") or source.get("prompt") or source["id"]
        widgets.muted(f"Make 3D uses {label}, this mesh's reference.")
    widgets.muted(
        "Roughly two minutes of GPU."
        if count == 1
        else f"Roughly {count * 2} minutes of GPU - {count} attempts, one queue."
    )
    busy = ctx.busy("submit")
    enabled = not problems and not busy
    with focus.item(ctx.state, FOCUS_PANE, "make3d") as focused:
        pressed = widgets.primary_button(
            "Make 3D",
            (-1, sp(34)),
            enabled=enabled,
            # The 2026-09-05 audit, finding create-08: create_brief._generate
            # states its top refusal as the button's own reason; this button
            # stated nothing (the refusals above it were the only word on it),
            # which is the one thing a hover of a greyed Make 3D could answer.
            reason=str(problems[0]) if problems else "",
        )
        # Enter on the ring's last stop; see ``settings_2d._submit``.
        if focused and enabled and (
            imgui.is_key_pressed(imgui.Key.enter)
            or imgui.is_key_pressed(imgui.Key.keypad_enter)
        ):
            pressed = True
    if pressed:
        promote(ctx, source, form)
    # Gated on ``enabled``, the other half of create-08: unguarded, this fired
    # on hover whether or not Make 3D could be pressed, advertising a shortcut
    # that does nothing while the button is dead.
    if enabled and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
        imgui.set_tooltip("Ctrl+Enter")


def _candidates(form: dict[str, Any]) -> None:
    """The Candidates control: how many attempts one press buys.

    A row of radio-style buttons rather than a combo, because there are three
    values and the number is the label -- and because it sits directly above
    Make 3D, where the cost sentence under it changes with the choice. It is
    the *only* control in this pane that multiplies what the button spends, so
    putting it anywhere else in the form would hide that.
    """
    widgets.field_label("Candidates")
    current = create_mesh.candidate_count(form)
    for count in range(1, MAX_MESH_CANDIDATES + 1):
        if count > 1:
            imgui.same_line()
        # Never drawn past the panel edge: three 40 px buttons and two spacings
        # fit inside the 300 px sidebar with room to spare, and the guard in
        # tests/test_studio_smoke.py measures rather than trusts that.
        if controls.radio_button(f"{count}##candidates", current == count):
            form["candidates"] = count
    widgets.help_marker(
        "Reconstruct the same reference more than once and keep the best. The "
        "engine is deterministic in its seed, so each attempt draws a new one; "
        "the rest are hidden from the library until you keep one."
    )


def promote(ctx: Any, source: dict[str, Any] | None, form: dict[str, Any]) -> None:
    """Put the matte in front of the two minutes of GPU.

    The button no longer submits: it opens the preview, which shows the cutout
    trellis will reconstruct from and offers Accept / Fix matte / Cancel. The
    matte is the single decision that most often turns a good reference into a
    solid slab, and it used to be made inside the exe *after* the user had
    committed. The composition gate's own verdict moves into the same panel for
    the same reason -- one place, before the spend, rather than a confirm here
    and a surprise there.

    ``source`` is resolved through :func:`_effective_source` before anything
    else: ``main.py`` and ``palette.py`` both call this with
    ``ctx.cache.get(ctx.state.source_job)`` verbatim (item 5.2 changes only
    what happens when that is None), so the fallback lives here rather than
    at each call site -- Ctrl+Enter and the palette's promote get the same
    correction ``_submit`` does.
    """
    source = _effective_source(ctx, source)
    problems = create_mesh.validate(source)
    if problems:
        # ``settings_2d.generate``'s reason exactly: Ctrl+Enter in 3D mode and
        # the palette's promote both land here, and this used to return in
        # silence -- so pressing Ctrl+Enter with nothing selected did nothing
        # at all, which reads as a broken shortcut rather than as a refusal.
        from . import settings_2d

        settings_2d.refuse(ctx, problems)
        return
    # ``count`` rides with the overrides because the preview captures the form
    # as it stood when the button was pressed -- the whole point of that
    # capture is that Accept submits what the user pressed with, and how many
    # of it is part of that. ``promote_candidates`` takes it as a kwarg, so
    # nothing downstream has to unpack it back out.
    matte_preview.open_for(
        ctx,
        source["id"],
        {**create_mesh.promote_kwargs(form), "count": create_mesh.candidate_count(form)},
    )


def submit_promotion(ctx: Any, job_id: str, kwargs: dict[str, Any], force: bool) -> None:
    """Queue the mesh job (or the candidate group) the preview was about.

    Always through ``promote_candidates``, count included: at 1 it *is*
    ``promote_to_model`` with no group minted, so there is one call path here
    rather than a branch that could send the two halves different overrides.
    """
    # The rings from the last refusal describe a request that no longer exists;
    # see ``settings_2d.generate``, which clears them for the same reason.
    ctx.state.clear_field_errors()
    if ctx.state.filters.kind not in ("all", "model"):
        # Otherwise a filter left on "reference" (the natural way to find the
        # source image before promoting it) permanently hides the model job
        # this creates.
        ctx.state.filters.kind = "all"
    # The return is checked, and the reason is the shared ``"submit"`` key.
    # ``TaskRunner.submit`` refuses a key that is already in flight -- which is
    # right, because a second click almost always means "I did not see the
    # first one work". But this call arrives from the matte modal's Accept, and
    # the modal closes on its own regardless: an Accept landing while an earlier
    # create was still running was dropped, silently, with the modal closing and
    # looking exactly like success. Nothing queued, nothing said (UX-26).
    if not ctx.submit(
        "submit", svc_jobs.promote_candidates, ctx.svc, job_id, force=force, **kwargs
    ):
        ctx.toast("Still submitting the last one - try again in a moment.")
        return
    create_mesh.reroll_mesh_seed(ctx.state.form_3d)


def _auto_accept(ctx: Any, state: Any) -> None:
    """The Accept button's own path, pressed by the setting instead of a click.

    Through ``matte_preview.accept`` -> ``submit_promotion``, identically: the
    review requires the skipped route to be indistinguishable from pressing
    Accept, and this is how ``_matte_body``'s Accept button does it two
    screens down. ``refused`` is never true here -- ``create_mesh.matte_is_clean`` is the
    gate that got this function called at all, and a refused preview always
    carries a reason.
    """
    # The preview outlives ``accept`` closing the state, so the Mesh column can
    # still show the cutout that was actually used (the review's ask; see
    # ``_LAST_AUTO_MATTE_SLOT``).
    ctx.state.preview[_LAST_AUTO_MATTE_SLOT] = state.preview
    job_id = state.job_id
    matte_preview.accept(
        ctx, lambda kwargs, force: submit_promotion(ctx, job_id, kwargs, force)
    )


def _wants_auto_accept(ctx: Any, state: Any) -> bool:
    """Whether this frame should draw nothing because item 5.4's setting
    applies. True means "handled" -- either the cutout is still being
    computed (wait rather than opening a modal that may turn out to be
    unnecessary) or it just qualified and was submitted without ever being
    shown. False means draw the modal exactly as before, which is also the
    answer whenever the setting is off.
    """
    if not ctx.settings.get(SKIP_CLEAN_MATTE_SETTING, False):
        return False
    if state.preview is None:
        # A failure still has to reach the modal -- its Cancel is the only
        # door back to Fix matte, and the toast alone does not offer it. See
        # dev/INVARIANTS.md on ``_tried_and_failed`` vs. ``failed_stamp``:
        # the stamp alone cannot tell "not tried yet" from "tried and failed".
        return not state._tried_and_failed
    if not create_mesh.matte_is_clean(state.preview):
        return False
    _auto_accept(ctx, state)
    return True


def matte_modal(ctx: Any) -> None:
    """The promote preview. Drawn beside the confirms, because it is a modal.

    Everything expensive happened elsewhere: ``matte_preview.pump`` submits the
    cutout to the TaskRunner and the frame thread only uploads the pixels it
    gets back, through ``ThumbnailCache.from_pixels`` so the texture inherits
    the deferred-release rule every other image in the UI has.
    """
    state = matte_preview.pump(ctx)
    if state is None:
        return
    if not state._open and _wants_auto_accept(ctx, state):
        # Handled without ever drawing a popup: the cutout was clean and the
        # setting is on, or it is still being computed and might yet be --
        # either way there is nothing to show this frame.
        return
    appearing = not state._open
    if appearing:
        imgui.open_popup(MATTE_TITLE)
        state._open = True
    alpha, rise = widgets.popover_enter("matte-preview", appearing)
    frosted = widgets.frosted()
    if frosted:
        imgui.set_next_window_bg_alpha(0.0)
    imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
    radius = widgets.push_surface_rounding()
    widgets.modal_bounds(sp(480))
    opened, _ = imgui.begin_popup_modal(
        MATTE_TITLE, None, imgui.WindowFlags_.always_auto_resize.value
    )
    widgets.pop_surface_rounding()
    if not opened:
        # Escape dismisses a modal without going through any of the buttons,
        # and imgui will not reopen a popup whose id it thinks is already open:
        # without this the modal would vanish once and never come back, with
        # ``job_id`` still set and every later press of Make 3D doing nothing.
        imgui.pop_style_var()
        state._open = False
        matte_preview.close(ctx)
        return
    widgets.window_shadow("overlay", radius=radius)
    if frosted:
        widgets.window_backdrop(radius=radius)
    if rise > 0.0:
        imgui.dummy((0, rise))
    _matte_body(ctx, state)
    imgui.end_popup()
    imgui.pop_style_var()


def _matte_body(ctx: Any, state: Any) -> None:
    preview = state.preview
    # The cutout is as tall as the reference is: a portrait image plus a stack
    # of warnings is exactly the body that used to push Accept off the bottom of
    # a short viewport. It scrolls; the three buttons below do not.
    with widgets.modal_body("matte-body"):
        if preview is None:
            widgets.muted("Cutting the subject out...")
        else:
            _matte_image(ctx, preview)
            widgets.muted(f"{MATTE_SOURCES.get(preview.source, preview.source)} - "
                          f"keeps {preview.coverage * 100:.0f}% of the frame")
            # What Accept actually does, said once. It used to say this only
            # for an already-matted reference -- because that was the only case
            # where the cutout survived. Now it is every case: the pixels on
            # screen are copied into the mesh job and the server is told to keep
            # them rather than cut its own.
            widgets.muted(
                "These are the pixels the 3D engine will rebuild from."
                if not preview.approved
                else "This reference already carries this matte; it will be kept."
            )
            for reason in preview.reasons:
                imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
                imgui.text_wrapped(reason)
                imgui.pop_style_color()
            for warning in preview.warnings:
                imgui.text_wrapped(warning)
    imgui.dummy((0, sp(6)))
    ready = preview is not None
    refused = bool(preview is not None and preview.reasons)
    label = "Build anyway" if refused else "Accept"
    # Both buttons in this popup wait on the same thing, and it is a state the
    # user can see happening -- so the sentence says what is being waited for
    # rather than restating that the button is off.
    preview_why = "The cutout is still being prepared."
    role = controls.ButtonRole.DESTRUCTIVE if refused else controls.ButtonRole.PRIMARY
    if controls.button(
        label,
        (sp(150), 0),
        role=role,
        enabled=ready,
        reason=preview_why,
    ):
        imgui.close_current_popup()
        state._open = False
        # Read *before* ``accept``, which closes the state before it calls
        # back: reading it inside the callback would name the empty string.
        job_id = state.job_id
        matte_preview.accept(
            ctx,
            lambda kwargs, force: submit_promotion(ctx, job_id, kwargs, force or refused),
        )
        return
    imgui.same_line()
    if controls.button(
        "Fix matte", (sp(150), 0), enabled=ready, reason=preview_why
    ):
        imgui.close_current_popup()
        state._open = False
        matte_preview.fix(ctx)
        return
    imgui.same_line()
    if controls.button(
        "Cancel", (sp(100), 0), role=controls.ButtonRole.GHOST
    ):
        imgui.close_current_popup()
        state._open = False
        matte_preview.close(ctx)


def _matte_image(ctx: Any, preview: Any) -> None:
    if ctx.textures is None:
        return
    texture = ctx.textures.from_pixels(
        f"matte:{preview.job_id}",
        float(preview.stamp or 0),
        (preview.width, preview.height),
        preview.rgb,
    )
    if texture is None:
        return
    # Drawn to fit rather than at its pixel size. The reference is whatever the
    # generator made (1024 px square, commonly), and an always-auto-resize modal
    # simply grew to hold it -- which on a 1280x800 viewport at UI scale 2.0 is
    # already taller than the screen before a single warning line is added
    # (2026-09-05). Aspect is preserved; the cap is half the viewport so the
    # buttons and the reasons still have room.
    width = float(preview.width)
    height = float(preview.height)
    # Not the live avail: this is the same shape as ``inspector.py``'s three
    # sites (see ``widgets.stable_width``'s docstring for the feedback chain
    # in full) -- ``layout.pane`` opens every pane as a scrolling child with no
    # ``no_scrollbar`` flag, and an aspect-preserving image sized off this
    # frame's avail feeds its own drawn height back into next frame's
    # scrollbar decision. The ``sp(480)``/``limit`` clamp below is not always
    # the binding constraint -- exactly ``inspector.py``'s own argument about
    # its 192 dp cap -- so a width that only sometimes reaches the threshold
    # is still the bug this pane can reach.
    avail = widgets.stable_content_width()
    if avail <= 1.0:
        avail = sp(480)
    limit = widgets.modal_max_height(float(imgui.get_main_viewport().size.y)) * 0.5
    scale = min(1.0, avail / width, limit / height)
    imgui.image(widgets.texture_ref(texture), (width * scale, height * scale))


def upload_bytes(ctx: Any, data: bytes) -> None:
    """Start a mesh job from pixels that are already in memory.

    The path ``upload`` takes for a file, for a caller that has rendered the
    picture rather than read it -- Clay's "send to 3D", which draws the
    document offscreen on the frame thread and hands the bytes over. The form
    values are read here for the same reason ``upload`` reads them here: they
    are UI state, and the task thread has no business touching them.
    """
    kwargs = create_mesh.upload_kwargs(ctx.state.form_3d)

    def run():
        return svc_jobs.create_job(ctx.svc, image=data, **kwargs)

    from . import settings_2d

    settings_2d.submit_job(ctx, run)


def upload(ctx: Any, path: Path) -> None:
    """Start a mesh job from an image on disk (a picker, or a dropped file)."""
    kwargs = create_mesh.upload_kwargs(ctx.state.form_3d)

    # The form values are read here, on the frame thread, because they are UI
    # state; the *file* is read in the task, because a large one would freeze
    # the window for as long as the disk took. Only MAX_UPLOAD_BYTES + 1 bytes
    # are ever read -- create_job's contract -- so an enormous file is refused
    # rather than allocated.
    def run():
        try:
            with path.open("rb") as fh:
                data = fh.read(MAX_UPLOAD_BYTES + 1)
        except OSError as exc:
            # ``field=`` for ``settings_2d``'s reason: the upload control is
            # what is wrong, and a bare refusal points at nothing.
            raise Invalid(f"could not read {path.name}: {exc}", field="image") from exc
        return svc_jobs.create_job(ctx.svc, image=data, **kwargs)

    from . import settings_2d

    settings_2d.submit_job(ctx, run)
