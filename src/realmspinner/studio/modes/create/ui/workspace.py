"""The in-context generation plan and results tray for Create.

Create used to send people from a form, to a floating progress card, to the
Library.  This small surface keeps the work that a submit starts in the same
place: the plan says what a press costs, and the tray becomes the place to
watch, compare, keep, and vary the result.  It deliberately reads the existing
job cache and services; it does not introduce another generation state.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from imgui_bundle import imgui

from .....service import jobs as svc_jobs
from .....service import sprites as svc_sprites
from .... import asset_open, controls, theme, widgets
from .... import candidates as candidates_mod
from ....manual import render as manual_render
from ....panes import thumbs
from ....tokens import sp
from ..engine import assets as create_assets
from ..engine.plan import Plan

#: How many finished results the tray shows, and the width of its grid. One
#: number because they are one fact: the tray is a fixed-height strip, so the
#: row it can draw whole is the row it should hold.
_RESULT_COLUMNS = 3


def plan_for(form: dict[str, Any], resolved: Any = None) -> Plan:
    """Describe the actual work using the same form values the door receives.

    Kept presentation-only and safe for partially restored forms, so it can be
    called every frame before the service performs its final validation.
    """
    spec = create_assets.selected(form)
    candidates = max(1, _integer(form.get("count"), 1))
    generations = candidates
    duration = "a few seconds"
    stages = "Generate image"
    if spec.key == "3d_model":
        stages = "Generate reference → choose or make 3D"
    elif spec.key == "seamless_material":
        stages = "Generate seamless material → verify wrap"
    elif spec.key == "tileset":
        # The 2026-09-23 audit, finding create-06: this used to hardcode one
        # generation regardless of mode, lines or variants. Read the real
        # count the door and the worker will run through the same helper
        # ``material_lines``/``tile_mode_of`` already use, mirroring the
        # sprite arm below.
        from ..engine import recipe as create_recipe

        candidates = 1
        tile = create_recipe.tileset_generation_plan(form)
        generations = int(tile["generations"])
        duration = svc_sprites.generation_time_phrase(generations)
        stages = "Generate tile sheet → inspect cells"
    elif spec.key == "sprite_sheet":
        # The sprite follow-up has one preliminary character plus one sheet
        # image per planned cell/candidate.
        from ..engine import recipe as create_recipe

        sprite = create_recipe.sprite_plan(form)
        candidates = int(sprite["candidates"])
        generations = 1 + int(sprite["generations"])
        duration = svc_sprites.generation_time_phrase(generations)
        stages = (
            f"1 character reference + {int(sprite['generations'])} sheet generation"
            + ("s" if int(sprite["generations"]) != 1 else "")
        )
    elif spec.key == "character":
        # **Zero image generations, and the plan says so out loud.** Every other
        # arm here counts SDXL passes; a character spends none -- the body comes
        # off the baked registry assets, the rig is Blender and the cells are
        # EEVEE, all of it CPU. A user with a small card is owed that before the
        # press rather than after it, which is why "no GPU needed" is in the
        # line rather than in a tooltip.
        from .....service import characters as svc_characters
        from ..engine import character as character_engine

        candidates = 1
        generations = 0
        cells = character_engine.cell_count(form)
        stages = (
            f"1 character ({_species_label(form)}) → rig → {cells}-cell "
            f"sheet, CPU only, no GPU needed"
        )
        duration = _about_minutes(svc_characters.estimate_minutes(cells))
    recipe = "Automatic recipe"
    if spec.key == "character":
        # Never a checkpoint name. ``_resolved_recipe`` answers for *any* form
        # -- ``generation.legacy_asset_type`` reads an unknown output as a 3D
        # model -- so without this the plan would name an SDXL checkpoint under
        # a press that never loads one.
        recipe = "Character registry - no image model"
    elif resolved is not None:
        recipe = str(getattr(resolved, "base_model", "") or "Automatic recipe")
    return Plan(candidates, generations, duration, stages, recipe)


def plan_footer(
    ctx: Any,
    plan: Plan,
    problems: list[Any],
    repairs: Callable[[Any], None],
    *,
    advisories: list[Any] | None = None,
    advisory_repairs: Callable[[Any], None] | None = None,
) -> None:
    """What a press will cost, and what is stopping it -- one footer, both stages.

    Describes a request and plans nothing: ``plan`` is built by the stage's own
    engine (``plan_for`` here, ``engine.mesh.plan`` for Mesh) and ``problems`` are
    its validators' answer. ``repairs`` draws the one-press fix under a problem,
    if it has one. Reference and Mesh each drew their own version of this, and
    Mesh's was a muted cost line and bare red text with no repair.
    """
    widgets.secondary("Generation plan")
    imgui.text_wrapped(plan.stages)
    if plan.generations > 0:
        widgets.muted(f"{plan.count_line} · {plan.duration}")
    else:
        # A character draws no images at all, and "1 candidate · 0 image
        # generations" is a line that reads as a bug rather than as a fact.
        # The duration still matters -- it is the whole cost of the press.
        widgets.muted(plan.duration)
    if plan.recipe:
        widgets.muted(f"Recipe: {plan.recipe}")
    active = getattr(ctx.cache, "active", None)
    if active is not None:
        position = queue_position(ctx, str(active.get("id") or ""))
        if active.get("status") == "queued":
            widgets.muted(f"Queue: position {position}" if position else "Queue: waiting")
        else:
            widgets.muted("Queue: one local generation is running")
    else:
        widgets.muted("Queue: ready")
    refusal = str(getattr(ctx.state.create, "submit_refusal", "") or "")
    advisories = advisories or []
    if not problems and not refusal:
        # "Ready to generate" is still true with an advisory standing -- that
        # is the whole difference between the two lists -- so it is said, and
        # then the advisory is drawn under it rather than instead of it.
        widgets.muted("Ready to generate.")
        _advisories(advisories, advisory_repairs)
        return
    if refusal:
        # Above the form problems: the form is fine -- this is the *door*
        # saying no, and it is the reason the last press did nothing. It stays
        # until a press is accepted, because a fading toast is what this
        # sentence was already tried as.
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
        imgui.text_wrapped(f"Refused: {refusal}")
        imgui.pop_style_color()
    for problem in problems:
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
        imgui.text_wrapped(f"Needs attention: {problem}")
        imgui.pop_style_color()
        repairs(problem)
    _advisories(advisories, advisory_repairs)


def _advisories(advisories: list[Any], repair: Callable[[Any], None] | None) -> None:
    """The advisories, under the problems, in the warning colour.

    Under, and in a different colour, because the reading order is the order
    they matter in: a problem is why the button is off, and an advisory is
    something to think about while pressing it. "Worth knowing" rather than
    "Needs attention" for the same reason -- nothing here needs anything.
    """
    for advisory in advisories:
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.WARN)))
        imgui.text_wrapped(f"Worth knowing: {advisory}")
        imgui.pop_style_color()
        if repair is not None:
            repair(advisory)


def _species_label(form: dict[str, Any]) -> str:
    """What the plan calls the thing it is about to build.

    Read from the registry rather than from ``character_family`` raw, so the
    line says "ogre" and not "ogre" only by luck of the key matching the label.
    Pure -- ``plan_for`` takes no ``ctx``, and ``characters.family`` imports
    nothing but the standard library, so this drags nothing in behind it.
    """
    from .....characters.family import families

    key = str(form.get("character_family") or "")
    row = families().get(key)
    # "no species yet" rather than an empty pair of brackets: this line is
    # drawn while the brief is still being refused, and a plan that reads
    # "1 character ()" says less than nothing.
    return row.label.lower() if row is not None else "no species yet"


def _about_minutes(minutes: float) -> str:
    """``estimate_minutes`` as a phrase. **"About", always.**

    ``service.characters`` states it: there is no dated document behind
    ``RIG_MINUTES`` and ``SECONDS_PER_CELL``, so nothing may key a decision on
    them and they are only ever rendered as "about N minutes".
    """
    value = max(1, round(float(minutes)))
    return f"about {value} minute" + ("s" if value != 1 else "")


#: The stages whose canvas carries the results tray. Every other stage keeps
#: the progress row in its column (``shell.frame._stage_pane``), so there is
#: exactly one "Working now" per stage and this tuple is the fact that says
#: which of the two draws it.
TRAY_STAGES = ("reference", "mesh")

#: What the progress row adds to the tray's height while a job is on it, in
#: design pixels: a name, a bar, a label and a Cancel button.
_PROGRESS_DP = 104.0


def _in_stage(job: dict[str, Any], stage: str | None) -> bool:
    """Whether ``job`` belongs in ``stage``'s tray. ``None`` means every job.

    A mesh row is Mesh's result and everything else (a reference, a tile, a
    sheet) is Reference's, so neither stage's strip is padded with the other's
    cards and "Rig" is only ever offered on the stage that owns it.
    """
    if stage is None:
        return True
    is_mesh = job.get("stage") == "model"
    return is_mesh if stage == "mesh" else not is_mesh


def tray_extra(ctx: Any) -> float:
    """Extra design pixels the tray needs while its progress row is drawn."""
    return _PROGRESS_DP if getattr(getattr(ctx, "cache", None), "active", None) else 0.0


def should_draw(ctx: Any, stage: str | None = None) -> bool:
    """Whether Create has work worth reserving central space for.

    **The same question :func:`draw` answers**, which is the fix: this asked
    "is there any queued, running or done job in the first twelve rows" while
    the tray showed a running job, a candidate group, or ``_recent_results``
    -- which excludes candidate members. So the two disagreed in both
    directions: a corpus of nothing but candidate rows reserved a strip and
    drew the empty state into it, and the viewer lost ``tray_height`` for a
    tray with nothing in it from the first finished job onward, permanently.

    ``stage`` narrows both questions to that stage's own results; the shell
    always passes it, and ``None`` keeps the stage-blind answer.
    """

    cache = getattr(ctx, "cache", None)
    if cache is None:
        return False
    if getattr(cache, "active", None) is not None:
        return True
    if stage in (None, "mesh") and candidates_mod.pending_cached(cache) is not None:
        return True
    return bool(_recent_results(ctx, stage))


def draw(ctx: Any, height: float = 0.0, stage: str | None = None) -> None:
    """Draw the results tray under the canvas -- Reference and Mesh, one tray.

    **The one place a result is picked between**: thumbnails, then Open, Vary,
    Keep/Discard, Rerun and the next stage's action (Make 3D on an image, Rig
    on a mesh). The inspector used to draw a second candidate picker for Mesh
    with its own Keep; that is gone, and ``panes.candidates_panel`` keeps only
    the keep/discard logic both stages call.

    The "Working now" row is the tray's first line (:func:`progress_row`),
    so a running job is stated once on the two stages that carry the tray. Rig,
    Pose and Export have no tray and keep it in their column instead.
    """
    if height > 0 and not imgui.begin_child("generation-results", (0, height), False):
        imgui.end_child()
        return
    widgets.pane_header("Generations")
    widgets.muted(_brief_caption(ctx))
    active = progress_row(ctx)
    group = candidates_mod.pending_cached(ctx.cache) if stage in (None, "mesh") else None
    if group is not None:
        _candidate_grid(ctx, group)
    else:
        jobs = _recent_results(ctx, stage)
        if jobs:
            _result_grid(ctx, jobs)
        elif not active:
            # Only reachable from a caller that draws the tray without asking
            # ``should_draw`` first; the shell always asks.
            widgets.muted_wrapped(
                "Your completed generations will appear here for comparison "
                "and variation."
            )
    if height > 0:
        imgui.end_child()


def progress_row(ctx: Any) -> bool:
    """The "Working now" narration and Cancel. -> True if a job was drawn.

    Drawn at the top of the results tray on Reference and Mesh, and by
    ``shell.frame._stage_pane`` on the stages that have no tray. The narration
    reads ``ctx.cache.active``, which is not stage-scoped, so a job started
    from Rig reports here while standing on Rig -- and never twice on one
    stage (2026-09-07 review item 5.7 removed the third copy; the Create
    redesign's tray step made Mesh's column copy the fourth and removed it).
    """
    active = getattr(ctx.cache, "active", None)
    if active is None:
        return False
    _progress(ctx, active)
    return True


def _brief_caption(ctx: Any) -> str:
    form = getattr(ctx.state, "form_2d", {})
    prompt = str(form.get("prompt") or "").strip()
    if not prompt:
        return "The current brief stays editable at left."
    return (prompt[:78] + "…") if len(prompt) > 79 else prompt


def _progress(ctx: Any, job: dict[str, Any]) -> None:
    status = str(job.get("status") or "queued")
    name = str(job.get("name") or job.get("prompt") or "Current generation")
    widgets.secondary("Working now")
    imgui.text_wrapped(name)
    if status == "queued":
        position = queue_position(ctx, str(job.get("id") or ""))
        widgets.muted(f"Queued{f' · position {position}' if position else ''}")
        _cancel(ctx, str(job.get("id") or ""))
        return
    progress = ctx.runtime.progress(str(job.get("id") or ""))
    if progress is not None:
        widgets.progress_bar(float(progress.get("percent") or 0.0))
        widgets.muted(str(progress.get("label") or "Generating…"))
    else:
        widgets.muted("Starting generation…")
    _cancel(ctx, str(job.get("id") or ""))


def _cancel(ctx: Any, job_id: str) -> None:
    """The progress card's own Cancel, on the tray's copy of its narration.

    This block says what the floating card says and used to say it without the
    one control the card carries, so the duplicate was strictly worse than the
    thing it duplicated. No confirmation, for the card's reason: the button
    says exactly what it does and sits on the thing it acts on.
    """

    if not job_id:
        return
    busy = ctx.busy(f"cancel:{job_id}")
    if widgets.disabled_button(
        f"Cancel##tray-cancel-{job_id}", not busy, reason="Cancelling..."
    ):
        ctx.submit(f"cancel:{job_id}", svc_jobs.cancel_job, ctx.svc, job_id)


def _candidate_grid(ctx: Any, group: Any) -> None:
    """Every candidate, in a strip that scrolls.

    Unlike the results grid this one cannot be trimmed to a row: a count of 8
    means eight candidates and choosing between them is the entire purpose, so
    the grid is put in a scrolling child rather than being cut short.
    """
    from ....panes import candidates_panel

    widgets.secondary("Compare candidates")
    manual_render.help_button(ctx, "candidates")
    widgets.muted_wrapped(
        "Choose one when every candidate settles. Seeds and scores stay with each result."
    )
    nudge = candidates_panel._nudge_text(group, candidates_panel._grades(ctx, group))
    if nudge is not None:
        widgets.muted(nudge)
    if not imgui.begin_child("generation-candidate-scroll", (0, 0), False):
        imgui.end_child()
        return
    if imgui.begin_table("generation-candidates", 2, imgui.TableFlags_.sizing_stretch_same.value):
        for member in group.members:
            imgui.table_next_column()
            _result_card(ctx, member, group=group)
        imgui.end_table()
    imgui.end_child()


def _result_grid(ctx: Any, jobs: list[dict[str, Any]]) -> None:
    widgets.secondary("Compare and refine")
    if imgui.begin_table(
        "generation-results", _RESULT_COLUMNS, imgui.TableFlags_.sizing_stretch_same.value
    ):
        for job in jobs:
            imgui.table_next_column()
            _result_card(ctx, job)
        imgui.end_table()


def _result_card(ctx: Any, job: dict[str, Any], group: Any = None) -> None:
    job_id = str(job["id"])
    thumbs.job_thumb(ctx, job, sp(72))
    imgui.same_line()
    imgui.begin_group()
    widgets.status_pill(str(job.get("status") or "queued"))
    params = job.get("params") or {}
    seed = params.get("seed", params.get("mesh_seed"))
    if seed is not None:
        widgets.muted(f"seed {seed}")
    rank = params.get("rank")
    score = rank.get("score") if isinstance(rank, dict) else None
    if score is not None:
        # **The ranker, and it says so.** This used to read "judge: N% likely a
        # keeper", which named the wrong instrument twice over: the value is
        # ``rank.score`` (composition, an optional DINOv2 anchor, an optional
        # PickScore blend), written by ``_q_mesh._rank_reference``, and the
        # trained probe -- ``service.judge`` -- is never called from this mode
        # at all. It is only ever scored from Review, and its answer is not
        # persisted onto the row, so there is nothing here to draw even when a
        # probe exists.
        #
        # The mislabel was not merely imprecise. Until the speckle floor was
        # fixed (dev/measurements/2026-09-06-speckle-composition-floor.md) the
        # composition term clamped to zero on every real reference, so an
        # ordinary generation with no anchor rendered "judge: 0% likely a
        # keeper" -- the app asserting, in the trained probe's name, that a
        # picture it had no opinion about was certain to be discarded.
        #
        # "rank" is what it orders and all it claims: where this candidate sits
        # in its strip, not whether it is any good.
        widgets.muted(f"rank {float(score) * 100:.0f}%")
    imgui.end_group()
    # **Two per row, not one per row.** Four full-width buttons stacked under a
    # 72 dp thumbnail make a card taller than the tray that holds it, and the
    # tray is the bottom of a column whose height is a fraction of the window --
    # so the last two fell outside it and could not be pressed at all.
    # ``/exercise-mode create`` reported fifteen clipped controls, every one of
    # them one of these; no test saw it, because a clipped button is drawn.
    half = (_half_width(), 0.0)
    status = str(job.get("status") or "")
    done = status == "done"
    not_ready = _why_not_finished(job, status)

    if controls.button(f"Open##result-open-{job_id}", half):
        # **The one door** (``asset_open.open_asset``), not a bare ``select``:
        # selecting alone leaves ``source_job`` stale on a reference, and a mesh
        # result opened this way showed ``input.png`` on the Reference stage.
        asset_open.open_asset(ctx, job)
    imgui.same_line()
    if widgets.disabled_button(f"Vary##result-vary-{job_id}", done, half, reason=not_ready):
        _vary(ctx, job)

    if group is not None:
        # The 2026-09-14 audit, finding create-04: when every member of the
        # group has failed, Keep's gate (``group.finished and done``) can
        # never open on any card, and nothing here offered a way out of a
        # group the library hides forever. Discard replaces Keep on every
        # card in that state, the same swap ``candidates_panel._member``
        # makes -- a button that can never enable is not a second choice
        # beside it.
        if group.all_failed:
            if controls.button(f"Discard##result-discard-{job_id}", half):
                from ....panes import candidates_panel

                candidates_panel.discard(ctx, group)
        elif (done or not group.finished) and widgets.disabled_button(
            # **One disabled reason.** Keep is only ever greyed for the wait:
            # a settled member that did not finish gets no Keep at all (its
            # status pill and Rerun say what happened), rather than a second
            # sentence for a button that could never open on it. The inspector's
            # picker carried both sentences, and a third of its own.
            f"Keep##result-keep-{job_id}",
            group.finished and done,
            half,
            reason="Wait for every candidate to finish.",
        ):
            from ....panes import candidates_panel

            candidates_panel.keep(ctx, group, job_id)
        # No ``same_line()`` here (the 2026-09-07 audit, finding create-08):
        # five actions do not divide into rows of two, and pairing Keep with
        # Rerun was what pushed the *next* button -- Make 3D, the primary
        # thing a candidate card is for -- onto a row of its own with its
        # other half left blank. Keep is the one left alone instead: Rerun and
        # Make 3D always pair below, on a candidate card and a finished result
        # alike.

    # **Rerun is live on a failure.** ``rerun_job`` needs only the brief and the
    # reference the row already has, and the library card has always offered
    # "Try again" on exactly these rows; disabling it here with "not ready yet"
    # was both the wrong reason and the wrong answer.
    #
    # ``svc_jobs.rerollable``, not a local ``done or status in _FAILED``: the
    # 2026-09-15 audit, finding create-02 -- the local spelling agreed with the
    # service's on *when* a row is finished but not on *what kind of row*, so a
    # finished ``lora_train``, ``separate``, built or hand-made-reference row
    # was offered a Rerun button that pressed straight into an ``Invalid`` at
    # dispatch.
    #
    # ``rerollable_reason`` is imported directly from its module rather than
    # through the ``svc_jobs`` facade: this fix's file list does not include
    # ``service/jobs.py``, and the facade is a re-export list a different
    # change can extend without this one racing it.
    from .....service._jobs_resubmit import rerollable_reason

    can_rerun = svc_jobs.rerollable(job)
    # ``rerollable``'s own status gate, restated: past it, ``rerollable_reason``'s
    # own sentence is the honest reason a greyed row is greyed -- "not ready
    # yet" was always wrong for a finished, built row. Still queued or
    # running, ``not_ready`` is kept instead: it carries the actual failure
    # detail for an error/cancelled row, which ``rerollable_reason``
    # deliberately does not (it is a pure predicate's sentence, not a log
    # line).
    reason = rerollable_reason(job) if status in ("done", "error", "cancelled") else not_ready
    if widgets.disabled_button(f"Rerun##result-rerun-{job_id}", can_rerun, half, reason=reason):
        ctx.submit(f"rerun:{job_id}", svc_jobs.rerun_job, ctx.svc, job_id, mode="reroll")
    imgui.same_line()
    from . import stages as create_stages

    if job.get("stage") == "model":
        # The next stage's action on a mesh card, in the slot Make 3D holds on
        # an image card. Disabled with Blender's own sentence (the rail's
        # segment and the Rig section say the same words), never hidden: a
        # user who never sees Rig concludes the app cannot rig at all.
        blocked = create_stages.blender_reason("rig", ctx)
        ready = done and "model.glb" in (job.get("files") or [])
        if widgets.disabled_button(
            f"Rig##result-rig-{job_id}",
            ready and blocked is None,
            half,
            reason=blocked or "A finished mesh is required.",
        ):
            _rig(ctx, job)
    else:
        is_reference = job.get("stage") == "reference" and "input.png" in (job.get("files") or [])
        if widgets.disabled_button(
            f"Make 3D##result-3d-{job_id}",
            done and is_reference,
            half,
            reason="A finished reference image is required.",
        ):
            _make_3d(ctx, job)


#: Statuses a job can end in without producing artifacts.
_FAILED = frozenset({"error", "cancelled", "failed"})


def _why_not_finished(job: dict[str, Any], status: str) -> str:
    """Why a control that needs a finished result is greyed, for *this* row.

    "This result is not ready yet." is true of a queued job and false of one
    that failed an hour ago; a disabled control that explains itself has to
    tell the two apart.
    """

    if status in ("error", "failed"):
        detail = str(job.get("error") or "").strip()
        return f"This generation failed: {detail}" if detail else "This generation failed."
    if status == "cancelled":
        return "This generation was cancelled."
    return "This result is not ready yet."


def _half_width() -> float:
    """Half the cell, less the gap between the two buttons that share it."""
    return (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) * 0.5


def _make_3d(ctx: Any, job: dict[str, Any]) -> None:
    """Move to Mesh, *then* open the cutout check.

    This used to start the check from wherever the card was drawn, so the
    matte dialog appeared over the Reference stage and the job it queued
    finished on a stage that could not show it. ``follow=False``: a promotion
    carries its own source, and walking the selection onto a mesh this
    reference already has would describe the wrong asset while the form builds
    another (``stages.go``'s own rule for a press that is about to make one).
    """
    from . import stages as create_stages
    from .panes import settings_3d

    create_stages.go(ctx, "mesh", follow=False)
    ctx.state.source_job = str(job["id"])
    settings_3d.promote(ctx, job, ctx.state.form_3d)


def _rig(ctx: Any, job: dict[str, Any]) -> None:
    """Move to the Rig stage with this mesh selected. The stage's own panel
    then owns the skeleton choice and the press; a card never rigs blind."""
    from . import stages as create_stages

    create_stages.go(ctx, "rig", select=str(job["id"]))


def _vary(ctx: Any, job: dict[str, Any]) -> None:
    """Copy a result's recorded brief back to the live form for a controlled edit.

    An image's brief goes to the Reference form; a mesh's own settings go to
    the Mesh form and the reference it was built from becomes the source, so
    the same card verb edits the stage it sits on.
    """
    from ....state import DEFAULT_FORM_3D
    from . import stages as create_stages

    if job.get("stage") == "model":
        params = job.get("params") or {}
        form = ctx.state.form_3d
        for key in DEFAULT_FORM_3D:
            if key in params and key not in ("count", "rig"):
                form[key] = params[key]
        parent = job.get("parent_id")
        if parent:
            ctx.state.source_job = str(parent)
        create_stages.go(ctx, "mesh", follow=False)
        ctx.toast("Loaded these mesh settings. Change one thing, then make it again.")
        return
    from ...library.ui.panes import library

    library.copy_settings(ctx, job)
    create_stages.go(ctx, "reference", follow=False)
    ctx.toast("Loaded this brief. Change one thing, then generate a variation.")


def _recent_results(ctx: Any, stage: str | None = None) -> list[dict[str, Any]]:
    """The most recent finished results. **One row of the grid, not two.**

    Six filled the tray's three columns twice over, and the tray is a
    fixed-height strip -- so the second row's cards were drawn with their
    actions below the fold, where nothing can press them. Three whole cards
    beat six half-drawn ones, and the library beside them holds the rest.

    The 2026-09-26 audit, finding create-workspace-06: this used to build a
    list comprehension over every row in ``ctx.cache.jobs`` -- every job the
    Library has ever cached, not just the three drawn -- every single frame
    this tray is on screen, then threw away everything past the third. A
    generator plus ``islice`` stops walking the cache the moment three
    matches are found, same order, same result.
    """
    matches = (
        job
        for job in ctx.cache.jobs
        if job.get("status") in ("done", "error", "cancelled")
        and not job.get("candidate_group")
        and _in_stage(job, stage)
    )
    return list(itertools.islice(matches, _RESULT_COLUMNS))


#: One memoized ``{job_id: position}`` map, keyed on ``(cache, cache.
#: _generation)`` -- the same shape ``candidates.pending_cached`` and
#: ``candidates_panel._grades`` already use against the identical counter.
#: Module-level for the same reason as its two neighbours: Create only ever
#: shows one cache's queue at a time, so one slot is enough.
_QUEUE_POSITION_CACHE: tuple[Any, dict[str, int]] | None = None


def queue_position(ctx: Any, job_id: str) -> int | None:
    """Where a queued job sits in line, or None if it is not queued.

    Public: the plan footer in ``panes.settings_2d`` asks the same question,
    and was reaching for the private name to do it.

    Memoized on ``ctx.cache``'s generation counter (the 2026-09-20 audit,
    finding create-06): this rebuilt and linearly scanned a filtered list of
    every queued job with no memo key, called every frame from both the
    per-frame progress card and the Reference-stage footer -- the same shape
    ``candidates.pending`` was fixed for one day earlier (2026-09-19,
    finding create-01). A ``None`` generation (a headless stand-in with no
    real cache) never memoizes, since there is nothing behind it that can go
    stale to avoid re-scanning.

    The key holds ``cache`` itself, not ``id(cache)``, for the same reason
    ``candidates.pending_cached`` and ``candidates_panel._grades`` do (the
    2026-09-20 audit, finding create-05): CPython reuses a freed object's
    address, so a bare id can name a cache that no longer exists --
    reproduced in 19,993 of 20,000 create-destroy-create cycles against a
    fresh cache at generation 0. A strong reference to the actual object can
    never be fooled that way.
    """
    global _QUEUE_POSITION_CACHE
    cache = ctx.cache
    generation = getattr(cache, "_generation", None)
    key = (cache, generation)
    if (
        generation is not None
        and _QUEUE_POSITION_CACHE is not None
        and _QUEUE_POSITION_CACHE[0] == key
    ):
        positions = _QUEUE_POSITION_CACHE[1]
    else:
        positions = {
            job["id"]: index
            for index, job in enumerate(
                (job for job in reversed(cache.jobs) if job.get("status") == "queued"),
                start=1,
            )
        }
        if generation is not None:
            _QUEUE_POSITION_CACHE = (key, positions)
    return positions.get(job_id)


def _integer(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
