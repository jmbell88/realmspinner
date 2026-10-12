"""The in-context generation plan and results tray for Create.

Create used to send people from a form, to a floating progress card, to the
Library.  This small surface keeps the work that a submit starts in the same
place: the plan says what a press costs, and the tray becomes the place to
watch, compare, keep, and vary the result.  It deliberately reads the existing
job cache and services; it does not introduce another generation state.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from imgui_bundle import imgui

from .....service import jobs as svc_jobs
from .....service import sprites as svc_sprites
from .... import asset_open, controls, fonts, theme, tokens, widgets
from .... import candidates as candidates_mod
from ....manual import render as manual_render
from ....panes import thumbs
from ....tokens import sp
from ..engine import assets as create_assets
from ..engine import recipe as create_recipe_engine
from ..engine import workspace as families
from ..engine.plan import Plan
from . import session


def plan_for(form: dict[str, Any], resolved: Any = None, *, installed: bool = True) -> Plan:
    """Describe the actual work using the same form values the door receives.

    Kept presentation-only and safe for partially restored forms, so it can be
    called every frame before the service performs its final validation.

    ``installed=False`` is the caller saying Automatic resolved to nothing on
    this host: the recipe line then reads "none installed" rather than the
    "Automatic recipe" that promised a press would run (the 2026-10-11 review:
    the footer said "Recipe: Automatic recipe" and "Queue: ready" on a machine
    that could draw nothing).
    """
    spec = create_assets.selected(form)
    candidates = max(1, _integer(form.get("count"), 1))
    generations = candidates
    # The 2026-10-04 audit, finding create-60: this was the literal "a few
    # seconds" whatever the Count, so eight candidates promised what one does.
    # Derived below from the generation count through the phrase the tile and
    # sprite arms already use, once the arm has settled it.
    duration = ""
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
        stages = f"1 character reference + {int(sprite['generations'])} sheet generation" + (
            "s" if int(sprite["generations"]) != 1 else ""
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
    if not duration:
        duration = svc_sprites.generation_time_phrase(generations)
    recipe = "Automatic recipe"
    if spec.key == "character":
        # Never a checkpoint name. ``_resolved_recipe`` answers for *any* form
        # -- ``generation.legacy_asset_type`` reads an unknown output as a 3D
        # model -- so without this the plan would name an SDXL checkpoint under
        # a press that never loads one.
        recipe = "Character registry - no image model"
    elif not installed:
        recipe = create_recipe_engine.NO_MODEL_RECIPE
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
    # The title is the plan card's own header when this is drawn in one
    # (``plan_card``); only a caller with no card around it needs the line.
    if not widgets.in_group():
        widgets.secondary("Generation plan")
    imgui.text_wrapped(plan.stages)
    if plan.generations > 0:
        widgets.muted_wrapped(f"{plan.count_line} · {plan.duration}")
    else:
        # A character draws no images at all, and "1 candidate · 0 image
        # generations" is a line that reads as a bug rather than as a fact.
        # The duration still matters -- it is the whole cost of the press.
        widgets.muted(plan.duration)
    if plan.recipe:
        widgets.muted_wrapped(f"Recipe: {plan.recipe}")
    refusal = str(getattr(ctx.state.create, "submit_refusal", "") or "")
    active = getattr(ctx.cache, "active", None)
    if active is not None:
        position = queue_position(ctx, str(active.get("id") or ""))
        if active.get("status") == "queued":
            widgets.muted(f"Queue: position {position}" if position else "Queue: waiting")
        else:
            widgets.muted_wrapped("Queue: one local generation is running")
    elif not problems and not refusal:
        # Only while a press would actually be taken: "Queue: ready" under a red
        # problem (or on a host with no image model at all) told the reader the
        # opposite of the line beneath it.
        widgets.muted("Queue: ready")
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
        if is_untouched_prompt_problem(ctx, problem) and not refusal:
            # A form nobody has touched is not in error, it is empty. Red waits
            # for an edit or a press (``CreateState.prompt_touched``).
            widgets.muted(PRISTINE_PROMPT_HINT)
            continue
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
        imgui.text_wrapped(f"Needs attention: {problem}")
        imgui.pop_style_color()
        repairs(problem)
    _advisories(advisories, advisory_repairs)


#: What an empty, untouched brief says instead of the red refusal.
PRISTINE_PROMPT_HINT = "Describe the asset to start"


def touch_prompt(ctx: Any, touched: bool) -> None:
    """Record that the prompt was edited (or a press refused), or that the form
    is fresh again. Tolerates a stand-in ``ctx`` with no Create state."""
    create = getattr(getattr(ctx, "state", None), "create", None)
    if create is not None:
        create.prompt_touched = touched


def is_untouched_prompt_problem(ctx: Any, problem: Any) -> bool:
    """Whether ``problem`` is the empty-prompt refusal on a form no one has edited.

    The 2026-10-11 Create review: a fresh Create opened under a red "Needs
    attention: A prompt is required." -- an error shown to someone who had done
    nothing. The prompt's own message and field are matched together so no
    other refusal is softened, and ``prompt_touched`` is set by the first edit
    of the prompt or the first refused press.
    """
    return (
        getattr(problem, "field", "") == "prompt"
        and str(problem) == "A prompt is required."
        and not getattr(ctx.state.create, "prompt_touched", False)
    )


#: The plan **card's** height before it has been measured, and its floor, in
#: design pixels. The card is the title row (26), the plan's five or six lines,
#: the 40 px Generate button and 12 px of padding all round: about 208 for a
#: ready form, which is what the first frame reserves until the card has been
#: drawn once and has said how tall it really is.
PLAN_DEFAULT_DP = 208.0
#: Never reserve less than the title row, one line, the button and the padding
#: (12 + 26 + 8 + 20 + 8 + 40 + 12), so a stale small measurement cannot hand
#: the form child room the card then overflows into.
PLAN_MIN_DP = 128.0

#: The most of the column the card may be given. 0.7 rather than the 0.55 the
#: old scrolling footer took: that footer scrolled inside its own child, but the
#: card holds the Generate button, and a card taller than its reserve is cut off
#: at the pane edge with the button under the cut.
_PLAN_MAX_FRACTION = 0.7


def plan_height(ctx: Any, key: str) -> float:
    """Pixels to give the pinned plan card of column ``key`` ("2d" or "3d").

    **Its own measured height**, one frame late (:func:`note_plan_height` writes
    the figure from the card's item rect once it has closed), clamped between a
    floor and a share of the column. The column used to reserve a fixed
    ``-sp(220)`` / ``-sp(52)`` split, so a footer with a refusal, a repair
    button and an advisory lost its last line to the clip (the 2026-10-11
    review), and a footer with one line wasted a hundred pixels of form.
    """
    measured = ctx.state.create.plan_heights.get(key)
    height = sp(PLAN_DEFAULT_DP) if measured is None else float(measured)
    ceiling = max(sp(PLAN_MIN_DP), float(imgui.get_content_region_avail().y) * _PLAN_MAX_FRACTION)
    return max(sp(PLAN_MIN_DP), min(height, ceiling))


def plan_reserve(ctx: Any, key: str) -> float:
    """The scrolling form child's height argument: the card's room, negated.

    The card is the *next* item, so imgui puts one ``ItemSpacing.y`` between the
    two children and the reserve is the card plus that gap -- which is what makes
    the form's bottom sit exactly one gutter above the card's top. It used to be
    a guessed ``sp(52)`` for a Generate row drawn outside any child, and a
    different guess on each stage once the footer's own divider was counted.
    """
    return -(plan_height(ctx, key) + imgui.get_style().item_spacing.y)


def note_plan_height(ctx: Any, key: str) -> None:
    """Record how tall the plan card just drawn turned out to be. Call it right
    after the card has closed: the last item is then the card itself, whose rect
    is the whole of it (title, lines, button and padding) and not a cursor
    position inside a child, which left the button and the card's own padding
    out of the figure."""
    ctx.state.create.plan_heights[key] = float(imgui.get_item_rect_size().y)


@contextmanager
def plan_card(ctx: Any, key: str):
    """The pinned "Generation plan" card, the same on both stages.

    The caller draws the plan lines and then the press into it; the card's own
    height is noted when it closes. One card, one title, one place for the
    button: Reference and Mesh used to differ by a divider-versus-gap split above
    the lines and by where Generate sat relative to them, so the button moved
    when the stage changed.
    """
    with widgets.group("Generation plan", id="plan"):
        yield
    note_plan_height(ctx, key)


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


def tray_extra(ctx: Any) -> float:
    """Extra design pixels the tray needs while its progress row is drawn."""
    return _PROGRESS_DP if getattr(getattr(ctx, "cache", None), "active", None) else 0.0


#: An attempt's thumbnail side, in design pixels. 80 rather than the 82 it was:
#: 82 is on no scale, 80 is ten of the 8 px unit.
ATTEMPT_THUMB_DP = 80.0

#: What the tray's card spends around its body, in design pixels: padding above,
#: the 26 px header row, the 8 px to the body, padding below (12 + 26 + 8 + 12).
#: Derived from the card's own constants so a change to the card moves the tray's
#: height with it.
TRAY_CHROME_DP = 2 * tokens.SP_3 + widgets.GROUP_HEADER_H + tokens.SP_2

#: The share of the viewport pane the tray may take. The canvas above it is what
#: the tray is *for*, so the tray gives way before the picture does.
TRAY_MAX_FRACTION = 0.48


def strip_height() -> float:
    """The attempt strip's height in physical px: the tallest cell it can draw.

    A cell is the thumbnail, the compact name button, the status pill and one
    note line ("Chosen source", "Favourite"), each one item spacing apart, and
    the horizontal scrollbar under them. **Explicit, because the card is
    ``auto_resize_y``**: the strip used to be ``(0, 0)`` -- "whatever is left" --
    which inside a child that sizes itself to its content is a circle, and the
    cell's last line was the first thing the circle lost. The note line and the
    scrollbar are reserved whether or not they show, so the card does not change
    height when the first attempt is favourited or the tenth one arrives.
    """
    style = imgui.get_style()
    gap = float(style.item_spacing.y)
    with fonts.small(imgui):
        pill = float(imgui.get_text_line_height()) + 2 * sp(3)
    return (
        sp(ATTEMPT_THUMB_DP)
        + gap
        + sp(tokens.CONTROL_HEIGHT_COMPACT)
        + gap
        + pill
        + gap
        + float(imgui.get_text_line_height())
        + float(style.scrollbar_size)
    )


def tray_height(ctx: Any, height: float) -> float:
    """The tray's height in physical px, inside a viewport pane ``height`` tall.

    The card's chrome plus its body (the attempt strip -- the candidate grid
    takes the same room, and scrolls inside it) plus the progress row while one
    is drawn, **capped at** :data:`TRAY_MAX_FRACTION` **of the pane**. It was a
    guessed ``sp(186)`` with a pane header over it; the card's chrome is 58 dp of
    its own, and a floor that ignores the chrome clips the strip's last line at
    every window size (the 2026-10-11 Create review). Where the cap binds, the
    tray's outer child scrolls rather than the strip losing its bottom edge.
    """
    wanted = sp(TRAY_CHROME_DP) + strip_height() + sp(tray_extra(ctx))
    return min(wanted, height * TRAY_MAX_FRACTION)


def should_draw(ctx: Any, stage: str | None = None) -> bool:
    """Whether Create has work worth reserving central space for.

    **The same question :func:`draw` answers**, which is the fix: this asked
    "is there any queued, running or done job in the first twelve rows" while
    the tray showed a running job, a candidate group, or the family results
    (``families.results`` -- the function :func:`draw` calls, which excludes
    follow-up kinds). So the two disagreed in both
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
    return bool(families.results(session.index(ctx), ctx.state.create.workspace, stage))


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
    top = float(imgui.get_cursor_screen_pos().y)
    # **One card, not a heading over a list.** "Attempts" was a 20 px
    # ``pane_header`` (the one place in Create's centre that spoke in a second
    # register) over bare controls; it is the same ``widgets.group`` every other
    # Create column is made of. The empty-state heroes ("What would you like to
    # make?", "Ready when you are") keep ``pane_header`` on purpose: each is the
    # only thing on the screen, not one group among several.
    with widgets.group("Attempts", id="attempts"):
        active = progress_row(ctx)
        group = candidates_mod.pending_cached(ctx.cache) if stage in (None, "mesh") else None
        if group is not None:
            # What is left of the tray under the lines already drawn, so the
            # scrolling grid ends exactly at the card's bottom padding.
            room = height - (float(imgui.get_cursor_screen_pos().y) - top) - sp(tokens.SP_3)
            _candidate_grid(ctx, group, room if height > 0 else 0.0)
        else:
            jobs = families.results(session.index(ctx), ctx.state.create.workspace, stage)
            if jobs:
                _attempt_strip(ctx, jobs)
            elif not active:
                # Only reachable from a caller that draws the tray without asking
                # ``should_draw`` first; the shell always asks.
                widgets.muted_wrapped(
                    "Your completed generations will appear here for comparison and variation."
                )
    if height > 0:
        imgui.end_child()


def note_submit_landed(ctx: Any) -> None:
    """A press was taken and the shell put the selection on its first row (or
    left it where it was). Remember where, so the finishing images can follow it.
    """
    ctx.state.create.follow_selection = ctx.state.selected


def follow_finished_reference(ctx: Any, job: dict[str, Any]) -> bool:
    """Put the selection on a reference that has just finished. -> whether it did.

    The user's decision: the newest attempt of a press is selected when it
    finishes, so a Count of 4 ends with the sheet showing and the last image
    chosen. **It does not fight a choice**: the selection only moves while it is
    still where the app last put it (:func:`note_submit_landed`, then each
    landing). A click on an earlier sibling mid-batch moves the selection off
    that mark, and nothing after it takes it back. Also only for this creation
    and only while standing on the Reference stage -- a finished image must not
    pull a user who has gone to Mesh, or to another creation, back.
    """
    from ...library.ui.panes import library
    from . import stages

    state = ctx.state
    if job.get("status") != "done" or job.get("stage") != "reference":
        return False
    if job.get("kind") not in ("text", "image") or not stages.at(state, "reference"):
        return False
    if state.create.workspace is None or session.index(ctx).by_job.get(
        str(job["id"])
    ) != state.create.workspace:
        return False
    if state.selected is not None and state.selected != state.create.follow_selection:
        return False
    library.select(ctx, str(job["id"]))
    state.create.follow_selection = str(job["id"])
    return True


def inspect_result(ctx: Any, job: dict[str, Any]) -> None:
    """Inspect without editing the draft or choosing a reconstruction source."""
    asset_open.open_asset(ctx, job, inspect_only=True)


#: Attempts at which the strip starts drawing only what is on screen. A creation
#: of a dozen attempts fits a window and gains nothing from the arithmetic; a
#: long-lived one decoded every thumbnail on the frame thread each frame (the
#: 2026-10-04 audit, finding create-51).
STRIP_CLIP_THRESHOLD = 24

#: Cells drawn past each edge of the viewport, so a scroll tick never shows a hole.
_STRIP_MARGIN = 2


def strip_window(count: int, scroll_x: float, view_w: float, cell_w: float) -> tuple[int, int]:
    """The half-open run of attempts to draw, from the strip's scroll and width. Pure.

    Every cell is the same width (the thumbnail's side plus one item spacing),
    so the visible run is arithmetic, not a measurement. Below
    :data:`STRIP_CLIP_THRESHOLD`, or with no usable viewport, the whole strip.
    """
    if count < STRIP_CLIP_THRESHOLD or view_w <= 0.0 or cell_w <= 0.0:
        return 0, count
    first = max(0, int(scroll_x // cell_w) - _STRIP_MARGIN)
    last = min(count, int((scroll_x + view_w) // cell_w) + 1 + _STRIP_MARGIN)
    return min(first, last), last


def _attempt_strip(ctx: Any, jobs: list[dict[str, Any]]) -> None:
    flags = imgui.WindowFlags_.horizontal_scrollbar.value
    if imgui.begin_child("create-attempt-strip", (0, strip_height()), False, flags):
        side = sp(ATTEMPT_THUMB_DP)
        gap = float(imgui.get_style().item_spacing.x)
        first, last = strip_window(
            len(jobs),
            float(imgui.get_scroll_x()),
            float(imgui.get_window_width()),
            side + gap,
        )
        if first:
            # A dummy of the skipped cells' width, so the scrollbar, the scroll
            # position and every drawn cell's place are what they would have
            # been with all of them drawn.
            imgui.dummy((first * (side + gap) - gap, 1.0))
        for i in range(first, last):
            job = jobs[i]
            if i:
                imgui.same_line()
            imgui.push_id(str(job["id"]))
            imgui.begin_group()
            thumbs.job_thumb(ctx, job, side)
            if imgui.is_item_clicked():
                inspect_result(ctx, job)
            selected = ctx.state.selected == job["id"]
            name = str(job.get("name") or job.get("prompt") or job["id"])
            if controls.button(
                widgets.fit_text(name, side) + "##select",
                (side, 0),
                role=controls.ButtonRole.PRIMARY if selected else controls.ButtonRole.GHOST,
                control_size=controls.ControlSize.COMPACT,
                tooltip=name,
            ):
                inspect_result(ctx, job)
            widgets.status_pill(str(job.get("status") or "queued"))
            if ctx.state.source_job == job["id"]:
                widgets.muted("Chosen source")
            elif ctx.state.create.comparison_pin == job["id"]:
                widgets.muted("Pinned for comparison")
            elif job.get("favorite"):
                widgets.muted("Favourite")
            imgui.end_group()
            imgui.pop_id()
        if last < len(jobs):
            imgui.same_line()
            imgui.dummy(((len(jobs) - last) * (side + gap) - gap, 1.0))
    imgui.end_child()


def next_step(ctx: Any, job: dict[str, Any]) -> tuple[str, bool, str, Callable[[], None]] | None:
    """The one thing most likely to come next for ``job``. **Not drawn.**

    -> ``(label, enabled, why not, press)``, or ``None`` when the stage has no
    next step of its own. A finished reference image offers **Make 3D** -- the
    tray card's verb, through the same :func:`make_3d`, cutout check included --
    a sprite sheet its export, and anything else the first live exit
    ``asset_exits`` names. The old panel drew all of these, plus seven more,
    as full-width buttons of equal weight; the rest are :func:`_more_actions`.
    """
    from .... import asset_exits
    from . import stages

    done = job.get("status") == "done"
    why = _why_not_finished(job, str(job.get("status") or ""))
    kind = create_assets.asset_type_from_params(job.get("params") or {}, stage=job.get("stage", ""))
    if (
        job.get("stage") == "reference"
        and kind in canvas_kinds()
        and "input.png" in (job.get("files") or [])
    ):
        return "Make 3D", done, why, lambda: make_3d(ctx, job)
    if kind == "sprite_sheet":
        return "Export sprite sheet...", done, why, lambda: stages.go(ctx, "export", follow=False)
    exits = [e for e in asset_exits.exits_for(ctx, job) if not e.reason]
    if exits:
        first = exits[0]
        return first.label, True, "", lambda: first.open(ctx, job)
    return None


def canvas_kinds() -> tuple[str, ...]:
    """The asset kinds a finished reference can be reconstructed from."""
    from . import canvas

    return canvas.MAKE_3D_KINDS


def selected_actions(ctx: Any, job: dict[str, Any]) -> None:
    """The inspector's **Next step**: one primary button and a **More** menu.

    It was up to eight full-width buttons of equal weight, two of which did what
    a tray card's button already did under another name (the 2026-10-11 Create
    review). The names are one vocabulary now: **Make 3D**, **Try again**,
    **Load settings**, **Favourite** (the header's) and **Discard**.
    """
    widgets.section("Next step")
    step = next_step(ctx, job)
    # One row, as concept 03 draws it: the primary press takes the width More does
    # not need. Two full-width rows made the menu the same size as the one thing
    # most likely to come next; beside it, More is plainly the lesser control. The
    # primary's width is *derived* (grid width minus More minus the one gap) so the
    # row is flush with the card's padding at any UI scale.
    with fonts.label(imgui):  # the face the button's own label is drawn in
        more_w = widgets.button_width("More")
    if step is not None:
        label, enabled, why, press = step
        gap = float(imgui.get_style().item_spacing.x)
        primary_w = max(imgui.get_content_region_avail().x - more_w - gap, 1.0)
        if widgets.primary_button(label, (primary_w, 0), enabled=enabled, reason=why):
            press()
        imgui.same_line()
    if controls.button(
        "More##next-step-more",
        (more_w if step is not None else -1, 0),
        role=controls.ButtonRole.GHOST,
    ):
        imgui.open_popup("create-next-step-more")
    if imgui.begin_popup("create-next-step-more"):
        _more_actions(ctx, job)
        imgui.end_popup()


def _more_actions(ctx: Any, job: dict[str, Any]) -> None:
    """What the **More** menu holds: everything the primary button is not.

    Load settings, Try again, Reproduce, the pairwise comparison, Export and
    Discard. Favourite is not here: the inspector's header has it, and a flag
    toggled from two places under two names was one of the review's findings.
    """
    from ...library.ui.panes import library
    from . import stages

    job_id = str(job["id"])
    status = str(job.get("status") or "")
    done = status == "done"
    not_ready = _why_not_finished(job, status)
    if widgets.disabled_button(
        "Load settings", done, (-1, 0), reason=not_ready,
        tooltip="Copy this attempt's recipe back into the form, to change one thing.",
    ):
        _vary(ctx, job)
        imgui.close_current_popup()
    if svc_jobs.rerollable(job) and controls.button(
        "Try again",
        (-1, 0),
        tooltip="Reuse this attempt's recipe with a fresh seed. Your draft stays unchanged.",
    ):
        ctx.submit(f"rerun:{job_id}", svc_jobs.rerun_job, ctx.svc, job_id, mode="reroll")
        imgui.close_current_popup()
    # The 2026-10-10 audit, finding create-28: this was offered for every
    # rerollable row, but a rig, retexture, sheet, charsheet or music row
    # records no seed, and ``rerun_job(seed=None)`` draws a random one -- so
    # "Reproduce" ran a different attempt than the one it names. Greyed with
    # the reason instead; the row stays visible so the verb is discoverable.
    reproduce_why = reproduce_reason(job)
    if svc_jobs.rerollable(job) and widgets.disabled_button(
        "Reproduce recorded attempt",
        not reproduce_why,
        (-1, 0),
        reason=reproduce_why,
        tooltip="Reuse this attempt's recorded recipe and seed. Results can vary by runtime.",
    ):
        seed = recorded_seed(job)
        ctx.submit(f"rerun:{job_id}", svc_jobs.rerun_job, ctx.svc, job_id, mode="reroll", seed=seed)
        imgui.close_current_popup()
    pin = ctx.state.create.comparison_pin
    if controls.button("Unpin comparison" if pin == job_id else "Pin for comparison", (-1, 0)):
        ctx.state.create.comparison_pin = None if pin == job_id else job_id
        ctx.state.create.image_comparing = None
    baseline = ctx.cache.get(pin) if pin and pin != job_id else None
    if baseline is not None:
        same_kind = (baseline.get("stage") == "model") == (job.get("stage") == "model")
        ready = done and baseline.get("status") == "done" and same_kind
        if widgets.disabled_button(
            "Compare with pinned",
            ready,
            (-1, 0),
            reason="Choose two finished images or two finished meshes.",
        ):
            if job.get("stage") == "model":
                ctx.state.compare_baseline = None
                library.compare(ctx, str(baseline["id"]))
            else:
                ctx.state.create.image_comparing = str(baseline["id"])
            imgui.close_current_popup()
    if widgets.disabled_button(
        "Export...",
        stages.available("export", job, ctx) is None,
        (-1, 0),
        reason=stages.available("export", job, ctx) or "",
    ):
        stages.go(ctx, "export", follow=False)
        imgui.close_current_popup()
    widgets.divider()
    if controls.button(
        "Discard",
        (-1, 0),
        tooltip="Move this attempt to the trash. The toast that follows can undo it.",
    ):
        discard(ctx, job)
        imgui.close_current_popup()


def discard(ctx: Any, job: dict[str, Any]) -> None:
    """Move one attempt to the trash -- the Library's own door, with its Undo.

    ``library.delete_asset`` is what the Delete key and the Library's card menu
    call: it clears the tick and the selection, submits ``trash_job`` (a soft
    delete; the row can be restored) and raises the "Moved to trash" toast with
    its Undo action. One path, so a discarded attempt comes back the same way
    from either place.
    """
    from ...library.ui.panes import library

    library.delete_asset(ctx, str(job["id"]))


def empty_canvas(ctx: Any, stage: str = "reference") -> None:
    """The canvas with nothing on it. Which empty state is the stage's: see
    :func:`canvas.empty` (the setup card, the Mesh drop zone, or the type cards
    -- :func:`type_cards` -- while the brief is still blank)."""
    from . import canvas

    canvas.empty(ctx, stage)


def type_cards(ctx: Any) -> None:
    """The six output cards. Drawn only while the Reference form is pristine --
    no prompt typed, nothing loaded -- because the column's "What are you
    making?" combo is the same choice and the cards were a second control for it
    (the 2026-10-11 review)."""
    from . import brief

    # Empty-state hero: the only thing on the screen, not one group among several, so it
    # keeps the pane heading rather than becoming a card.
    widgets.pane_header("What would you like to make?")
    widgets.muted_wrapped(
        "Choose an output, describe it in your brief, or drop an image to build a 3D model."
    )
    columns = 2 if imgui.get_content_region_avail().x < sp(760) else 3
    if imgui.begin_table(
        "create-output-cards", columns, imgui.TableFlags_.sizing_stretch_same.value
    ):
        for key, label in create_assets.ASSET_TYPE_OPTIONS:
            imgui.table_next_column()
            current = create_assets.selected(ctx.state.form_2d).key == key
            if controls.button(
                label + "##output-card",
                (-1, sp(56)),
                role=controls.ButtonRole.PRIMARY if current else controls.ButtonRole.SECONDARY,
            ):
                form = ctx.state.form_2d
                form["asset_type"] = form["generation_type"] = key
                create_assets.sync_legacy_fields(form)
                ctx.state.clear_field_errors()
            widgets.muted_wrapped(brief._TYPE_HINTS[key])
        imgui.end_table()
    imgui.spacing()
    widgets.secondary("How it works")
    widgets.muted_wrapped(plan_for(ctx.state.form_2d).stages)


def image_comparison(ctx: Any, width: float, height: float) -> bool:
    pinned = ctx.state.create.image_comparing
    if not pinned:
        return False
    pair = [ctx.job(), ctx.cache.get(pinned)]
    if any(
        j is None or j.get("deleted_at") or "input.png" not in (j.get("files") or []) for j in pair
    ):
        ctx.state.create.image_comparing = None
        return False
    if controls.button("Exit image comparison"):
        ctx.state.create.image_comparing = None
        return False
    cell = max(1, (width - imgui.get_style().item_spacing.x) / 2)
    for i, job in enumerate(pair):
        if i:
            imgui.same_line()
        imgui.begin_group()
        name = str(job.get("name") or job.get("prompt") or job["id"])
        widgets.secondary(widgets.fit_text(name, cell))
        texture = (
            ctx.textures.get(
                f"create-compare:{job['id']}",
                ctx.job_dir(job["id"]) / "input.png",
                max_side=1600,
                background=True,
            )
            if ctx.textures
            else None
        )
        if texture is None:
            widgets.muted("Loading image...")
            imgui.dummy((cell, max(1, height - sp(80))))
        else:
            factor = min(cell / texture.size[0], max(1, height - sp(80)) / texture.size[1])
            imgui.image(
                widgets.texture_ref(texture), (texture.size[0] * factor, texture.size[1] * factor)
            )
        imgui.end_group()
    return True


def progress_row(ctx: Any) -> bool:
    """The "Working now" narration and Cancel. -> True if a job was drawn.

    Drawn at the top of the results tray on Reference and Mesh, and by
    ``shell.frame._stage_pane`` on the stages that have no tray. The narration
    reads ``ctx.cache.active``, which is not stage-scoped, so a job started
    from Rig reports here while standing on Rig -- and never twice on one
    stage (2026-09-07 review item 5.7 removed the third copy; the Create
    redesign's tray step made Mesh's column copy the fourth and removed it).

    **Where it draws decides its surface.** Inside the Attempts card it is the
    card's first lines; on a stage with no tray (Rig, Pose, Export) there is no
    card around it, so it opens a header-less one of its own -- before the cards
    existed it was bare lines and a divider, the one group in that column that
    was not a card. Nothing is opened when no job is running, so an idle column
    does not start with an empty slab.
    """
    active = getattr(ctx.cache, "active", None)
    if active is None:
        return False
    if widgets.in_group():
        _progress(ctx, active)
    else:
        with widgets.group(id="progress"):
            _progress(ctx, active)
    return True


def _progress(ctx: Any, job: dict[str, Any]) -> None:
    status = str(job.get("status") or "queued")
    name = str(job.get("name") or job.get("prompt") or "Current generation")
    widgets.field_label("Working now")
    imgui.text_wrapped(name)
    if status == "queued":
        position = queue_position(ctx, str(job.get("id") or ""))
        widgets.muted(f"Queued{f' · position {position}' if position else ''}")
        _cancel(ctx, str(job.get("id") or ""))
        return
    progress = ctx.runtime.progress(str(job.get("id") or ""))
    left = eta_phrase(job, time.time())
    if progress is not None:
        widgets.progress_bar(float(progress.get("percent") or 0.0))
        label = str(progress.get("label") or "Generating…")
        widgets.muted(f"{label} · {left}" if left else label)
    else:
        widgets.muted(f"Starting generation… · {left}" if left else "Starting generation…")
    _cancel(ctx, str(job.get("id") or ""))


#: Seconds one running job is expected to take, by what it is. Reference images
#: are one SDXL pass (``svc_sprites.SECONDS_PER_GENERATION``, the figure the
#: plan footer already promises); a mesh is the "about 2 minutes of GPU" the Mesh
#: plan states. Anything else has no honest estimate, and gets none.
MESH_SECONDS = 120.0


def estimated_seconds(job: dict[str, Any]) -> float | None:
    """How long this job is expected to run, or ``None`` when nothing says. **Pure.**"""
    if job.get("kind") not in ("text", "image"):
        return None
    params = job.get("params") or {}
    if job.get("stage") == "model":
        return MESH_SECONDS
    if job.get("stage") == "reference" and not (params.get("sprite_sheet") or params.get("troupe")):
        return float(svc_sprites.SECONDS_PER_GENERATION)
    return None


def eta_phrase(job: dict[str, Any], now: float) -> str:
    """"about 20 seconds left", or the time so far. **Pure.**

    The plan's estimate less the job's own elapsed time. Never negative and
    never a guess past the estimate: once the job has run longer than it was
    expected to (or the clock is wrong, or there is no estimate) the phrase is
    the elapsed time alone, which is a fact rather than a promise. ``""`` while
    the job has not started.
    """
    try:
        started = float(job.get("started_at") or 0.0)
    except (TypeError, ValueError):
        return ""
    if started <= 0.0:
        return ""
    elapsed = now - started
    if elapsed < 0.0 or elapsed > 24 * 3600:
        return ""
    expected = estimated_seconds(job)
    remaining = None if expected is None else expected - elapsed
    if remaining is not None and remaining >= 5.0:
        if remaining < 90.0:
            return f"about {int(round(remaining / 5.0) * 5)} seconds left"
        minutes = round(remaining / 60.0)
        return f"about {minutes} minute{'s' if minutes != 1 else ''} left"
    if elapsed < 90.0:
        return f"{int(elapsed)} s so far"
    return f"{int(elapsed // 60)} min {int(elapsed % 60)} s so far"


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
    if widgets.disabled_button(f"Cancel##tray-cancel-{job_id}", not busy, reason="Cancelling..."):
        ctx.submit(f"cancel:{job_id}", svc_jobs.cancel_job, ctx.svc, job_id)


def _candidate_grid(ctx: Any, group: Any, room: float = 0.0) -> None:
    """Every candidate, in a strip that scrolls.

    Unlike the results grid this one cannot be trimmed to a row: a count of 8
    means eight candidates and choosing between them is the entire purpose, so
    the grid is put in a scrolling child rather than being cut short.

    ``room`` is the height the scrolling child gets, from the tray (what is left
    of it under the lines above). **Explicit**, because the card around it is
    ``auto_resize_y`` and a ``(0, 0)`` child inside a window that sizes itself to
    its content is "as tall as I already am" -- never taller, so the grid could
    not grow into room the tray had. Floored at a few rows' worth.
    """
    from ....panes import candidates_panel

    widgets.field_label("Compare candidates")
    manual_render.help_button(ctx, "candidates")
    widgets.muted_wrapped(
        "Choose one when every candidate settles. Seeds and scores stay with each result."
    )
    nudge = candidates_panel.nudge_text(group, candidates_panel.grades_for(ctx, group))
    if nudge is not None:
        widgets.muted(nudge)
    floor = sp(tokens.SP_6 * 4)
    if not imgui.begin_child("generation-candidate-scroll", (0, max(room, floor)), False):
        imgui.end_child()
        return
    if imgui.begin_table("generation-candidates", 2, imgui.TableFlags_.sizing_stretch_same.value):
        for member in group.members:
            imgui.table_next_column()
            _result_card(ctx, member, group=group)
        imgui.end_table()
    imgui.end_child()


def recorded_seed(job: dict[str, Any]) -> Any:
    """The seed that reproduces what is on disk for this attempt (create-06).

    ``params["seed"]`` is the draw the attempt asked for; when the reference
    reroll budget redrew, ``reference_seed`` was rewritten to the one that
    shipped ``input.png`` and ``seed`` still names the refused first draw. A
    mesh is reproduced by ``mesh_seed`` alone.
    """
    params = job.get("params") or {}
    if job.get("stage") == "model":
        return params.get("mesh_seed", params.get("seed"))
    return params.get("reference_seed", params.get("seed", params.get("mesh_seed")))


def reproduce_reason(job: dict[str, Any]) -> str:
    """Why "Reproduce recorded attempt" is greyed for this row, or "". **Pure.**

    Beside :func:`recorded_seed` because it is that function's question: a seed
    of ``0`` is a legal draw, so the test is ``is None``, never falsiness.
    """
    if recorded_seed(job) is None:
        return "This attempt recorded no seed."
    return ""


def _result_card(ctx: Any, job: dict[str, Any], group: Any = None) -> None:
    job_id = str(job["id"])
    thumbs.job_thumb(ctx, job, sp(72))
    imgui.same_line()
    imgui.begin_group()
    widgets.status_pill(str(job.get("status") or "queued"))
    params = job.get("params") or {}
    seed = recorded_seed(job)
    if seed is not None:
        widgets.muted(f"seed {seed}")
    rank = params.get("rank")
    score = rank.get("score") if isinstance(rank, dict) else None
    order_hint = ""
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
        # Not drawn as a number on the card: "rank 37%" read as a quality score
        # (the 2026-10-11 review), and it is only where this candidate sits in
        # its strip. It is the card's hover, named for what it is.
        order_hint = f"Order hint {float(score) * 100:.0f}% - a sorting aid, not a quality score."
    imgui.end_group()
    if order_hint and imgui.is_item_hovered():
        imgui.set_tooltip(order_hint)
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
    if widgets.disabled_button(
        f"Load settings##result-vary-{job_id}", done, half, reason=not_ready
    ):
        _vary(ctx, job)

    if group is not None:
        # The 2026-09-14 audit, finding create-04: when every member of the
        # group has failed, Keep's gate (``group.finished and done``) can
        # never open on any card, and nothing here offered a way out of a
        # group the library hides forever. Discard replaces Keep on every
        # card in that state, the same swap the retired inspector picker
        # made -- a button that can never enable is not a second choice
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
    if widgets.disabled_button(
        f"Try again##result-rerun-{job_id}", can_rerun, half, reason=reason
    ):
        ctx.submit(f"rerun:{job_id}", svc_jobs.rerun_job, ctx.svc, job_id, mode="reroll")
    imgui.same_line()
    from . import stages as create_stages

    if job.get("stage") == "model":
        # The next stage's action on a mesh card, in the slot Make 3D holds on
        # an image card. Disabled with Blender's own sentence (the rail's
        # segment and the Rig section say the same words), never hidden: a
        # user who never sees Rig concludes the app cannot rig at all.
        blocked = create_stages.blender_reason("rig", ctx)
        rig_blocked = rig_reason(job, blocked)
        if widgets.disabled_button(
            f"Rig##result-rig-{job_id}",
            rig_blocked is None,
            half,
            reason=rig_blocked or "",
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
            make_3d(ctx, job)


#: Statuses a job can end in without producing artifacts.
_FAILED = frozenset({"error", "cancelled", "failed"})


def rig_reason(job: dict[str, Any], blocked: str | None) -> str | None:
    """Why a mesh card's Rig is greyed, or None when it is live. Pure.

    The mesh's own state first, Blender second: the 2026-10-04 audit, finding
    create-38, found the reason was ``blocked or "A finished mesh is required."``,
    so with Blender missing a failed or still-running mesh said "needs Blender"
    -- true, and not what stood in the way. Blender's sentence is the honest one
    only once there is a finished mesh to rig.
    """
    status = str(job.get("status") or "queued")
    if status != "done":
        return _why_not_finished(job, status)
    if "model.glb" not in (job.get("files") or []):
        return "A finished mesh is required."
    return blocked


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


def make_3d(ctx: Any, job: dict[str, Any]) -> None:
    """**Make 3D**, everywhere it is offered: the tray card, the canvas, the
    contact sheet, the inspector and Enter. Move to Mesh, *then* open the cutout
    check.

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


_make_3d = make_3d  # the name tests and older callers used


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
        # The 2026-10-03 audit, finding create-21: only the keys the job
        # recorded were copied, and ``promote_to_model`` records the engine
        # values (trellis_band, size_m...) only when they were set -- so every
        # key it left unset kept whatever the live form held, and "Change one
        # thing, then make it again" re-ran a recipe that differed from the card
        # in ways nothing showed. Start from the defaults, as ``session.resume``
        # does; ``attempts`` and ``rig`` are not part of a recipe and stay as the
        # user has them.
        keep = {key: form[key] for key in ("attempts", "rig") if key in form}
        form.clear()
        form.update(DEFAULT_FORM_3D)
        form.update(keep)
        for key in DEFAULT_FORM_3D:
            if key in params and key not in ("attempts", "rig"):
                form[key] = params[key]
        parent = job.get("parent_id")
        if parent:
            ctx.state.source_job = str(parent)
        # The 2026-10-10 audit, finding create-31: one rule for both Vary verbs.
        # The recorded seed is what makes "change one thing" a controlled
        # change, and Generate rolls the unlocked seed before it builds the
        # request -- so a restored but unlocked seed was spent on the first
        # press. The 2D half is in ``library.copy_settings``.
        locked = form.get("mesh_seed") is not None
        if locked:
            form["mesh_seed_locked"] = True
        create_stages.go(ctx, "mesh", follow=False)
        ctx.toast(
            "Loaded these mesh settings and locked their seed. Change one thing, "
            "then make it again."
            if locked
            else "Loaded these mesh settings. Change one thing, then make it again."
        )
        return
    from ...library.ui.panes import library

    library.copy_settings(ctx, job, announce=False, follow=False)
    ctx.toast(
        "Loaded this brief and locked its seed. Change one thing, then generate a variation."
        if ctx.state.form_2d.get("seed_locked")
        else "Loaded this brief. Change one thing, then generate a variation."
    )


#: One memoized ``{job_id: position}`` map, keyed on ``(cache, cache.
#: _generation)`` -- the same shape ``candidates.pending_cached`` and
#: ``candidates_panel.grades_for`` already use against the identical counter.
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
    ``candidates.pending_cached`` and ``candidates_panel.grades_for`` do (the
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
