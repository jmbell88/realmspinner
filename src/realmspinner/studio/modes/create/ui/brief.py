"""Create's navigation header and reusable draft controls.

The header holds the asset journey, New and Inspector. The settings pane owns
the brief and candidate count, with one persistent submit action below its plan.
Inspecting an attempt does not load its settings or replace the chosen source.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from imgui_bundle import imgui

from .... import anchors, controls, dialogs, focus, icons, theme, widgets
from ....tokens import sp
from ..engine import assets as create_assets
from ..engine import mesh as create_mesh
from ..engine import recipe as create_recipe

#: The header pane's height in design pixels: the stage rail alone, on every
#: stage. Measured rather than derived -- the naive arithmetic fell short of
#: imgui's own trailing ``item_spacing`` after the row's last item, which a bare
#: guess has no way to know to add (the same way Muse's own ``BAR_H`` was wrong
#: before it grew a version of this test). ``test_the_bar_fits_the_height_it_declares``
#: draws the row for real and this is what it measured. The bar used to carry a
#: taller press row on Reference and Mesh; the press moved into the settings
#: column (:func:`inputs`, :func:`submit_control`), so there is one height now.
RAIL_ONLY_H = 72.0

#: The prompt field's own height. Two lines rather than one, because
#: ``MAX_PROMPT`` is a thousand characters and a single-line input for a
#: paragraph shows the tail and hides the subject. It scrolls past two.
PROMPT_H = 40.0

#: Fixed widths for the controls the column draws at a set size. Design pixels;
#: ``sp`` scales them. ``COUNT_W`` is what ``imgui.push_item_width`` reserves
#: for the count pills' fields (the pills themselves are buttons, which the
#: push does nothing to). ``TYPE_W`` and ``GENERATE_W`` are only the fallbacks
#: for a caller that passes no width: :func:`inputs` and :func:`submit_control`
#: both hand the column's full width.
TYPE_W = 138.0
COUNT_W = 124.0
GENERATE_W = 158.0

#: The Reference stage's count values. 8 is ``validation.MAX_REFERENCE_COUNT``.
_COUNTS: tuple[int, ...] = (1, 2, 4, 8)

#: The Mesh stage's, from the service's own ceiling (``MAX_MESH_CANDIDATES``,
#: 3): the pills are the *same control*, and only their range is the stage's.
_MESH_COUNTS: tuple[int, ...] = tuple(range(1, create_mesh.MAX_MESH_CANDIDATES + 1))

#: The visible label above the pills (:func:`inputs` draws it), on both stages --
#: four bare digits said nothing about what they choose.
COUNT_LABEL = "Candidates"

#: Per-pill hover text for the count control, the same shape as
#: ``_TYPE_HINTS`` beside it. Four bare pills ("1 2 4 8") carried no label and
#: no tooltip -- unlike the Type combo, which names each value on hover -- so
#: a first-time user had nothing on screen or on hover saying they choose how
#: many candidates one press draws. The 2026-09-05 audit, finding create-11.
_COUNT_HINTS: dict[str, str] = {
    "1": "Draw one candidate.",
    "2": "Draw two candidates to compare.",
    "4": "Draw four candidates to compare.",
    "8": "Draw eight, the most one press can generate.",
}

#: The same shape for Mesh, whose count is attempts on one reference.
_MESH_COUNT_HINTS: dict[str, str] = {
    "1": "Reconstruct the reference once.",
    "2": "Reconstruct it twice, each from a new seed, and keep the better one.",
    "3": "Reconstruct it three times, each from a new seed -- the most one press buys.",
}

#: The Create column's one key in the focus ring, shared by ``settings_2d`` and
#: ``settings_3d``. The brief's controls are drawn *inside* those panes' columns,
#: so a ring of their own meant two rings pumped in one frame: one Tab moved both
#: cursors, drew two rings, and the single ``focus_moved`` flag went to whichever
#: focused control was drawn first (the 2026-10-03 audit, finding create-15).
#: The host pane pumps and begins it; :func:`inputs` and :func:`submit_control`
#: only record into it, so the order is the order the column is drawn in.
FOCUS_PANE = "brief"


def bar_height(ctx: Any) -> float:
    """This frame's header height: the rail's own band, on every stage. An
    inert strip under a bare rail would be the same complaint the module
    docstring makes about a bar with dead controls, so nothing more is reserved.
    """
    return sp(RAIL_ONLY_H)


def draw(ctx: Any, rail: Callable[..., None]) -> None:
    """The header. Called from ``main._build_ui`` for the one pane that holds
    the stage rail, New and Inspector.

    ``rail`` is ``App._stage_rail``, bound -- see the module docstring for why
    it is handed in rather than called through an import here.
    """

    from . import session

    session.sync(ctx)
    state = ctx.state
    form_2d = state.form_2d
    # The same synchronisation the settings column runs, and for the same
    # reason: the five derived door fields are a function of the type, and the
    # type is edited *here* now, on the Reference stage. Running it before the
    # early return keeps every stage's rail agreeing with the column about
    # what the asset currently is, not only Reference's.
    if "asset_type" not in form_2d:
        form_2d["asset_type"] = create_assets.legacy_asset_type(form_2d)
    create_assets.sync_legacy_fields(form_2d)
    reserve = widgets.button_width("New...") + widgets.button_width("Inspector") + sp(24)
    rail(ctx, max_width=max(sp(160), imgui.get_content_region_avail().x - reserve))
    imgui.same_line()
    if controls.button("New...", role=controls.ButtonRole.GHOST):
        imgui.open_popup("create-new")
    if imgui.begin_popup("create-new"):
        if controls.button("New creation"):
            session.new(ctx)
            imgui.close_current_popup()
        if controls.button("New with previous settings"):
            session.new(ctx, reuse=True)
            imgui.close_current_popup()
        imgui.end_popup()
    imgui.same_line()
    if controls.button(
        "Inspector",
        role=controls.ButtonRole.GHOST,
        tooltip="Show or hide details for the selected result.",
    ):
        state.create.inspector_open = not state.create.inspector_open


def inputs(ctx: Any, *, mesh: bool = False) -> None:
    """The draft's essential controls, above artistic and technical settings."""
    form = ctx.state.form_3d if mesh else ctx.state.form_2d
    # No ``focus.pump``/``begin`` here: the settings pane that hosts this
    # column already did, on the same key (see ``FOCUS_PANE``).
    widgets.pane_header("Build from a reference" if mesh else "Your brief")
    if mesh:
        _source_chip(ctx, imgui.get_content_region_avail().x)
    else:
        widgets.secondary("What are you making?")
        _type(ctx, form, width=imgui.get_content_region_avail().x)
        widgets.muted_wrapped(_TYPE_HINTS.get(create_assets.selected(form).key, ""))
        widgets.secondary(
            "Describe the asset" if form.get("asset_type") != "tileset" else "Shared style"
        )
        _prompt(ctx, form, imgui.get_content_region_avail().x, height=104)
        widgets.field_error(ctx.state, "prompt")
    if mesh or form.get("output") not in ("sheet", "character"):
        widgets.secondary(COUNT_LABEL)
        _count(
            ctx,
            form,
            _MESH_COUNTS if mesh else _COUNTS,
            create_mesh.candidate_count(form) if mesh else int(form["count"]),
        )
    if controls.button("Reset settings...", role=controls.ButtonRole.GHOST):
        from .panes import settings_2d, settings_3d

        ctx.confirms.ask(
            dialogs.Confirm(
                title=_reset_title("mesh" if mesh else "image"),
                message=_RESET_MESH_CONFIRM_MESSAGE if mesh else _RESET_CONFIRM_MESSAGE,
                confirm_label="Reset",
                cancel_label="Cancel",
                on_confirm=(lambda: settings_3d._reset(ctx))
                if mesh
                else (lambda: settings_2d._reset(ctx)),
            )
        )


def submit_control(ctx: Any, *, mesh: bool = False) -> None:
    """One persistent action directly below the scrollable generation plan."""
    from .panes import settings_3d

    form = ctx.state.form_3d if mesh else ctx.state.form_2d
    if not mesh:
        create_assets.sync_legacy_fields(form)
    source = settings_3d.bar_source(ctx) if mesh else None
    problems = (
        settings_3d.problems(ctx, source)
        if mesh
        else _with_pending_candidates_problem(ctx, create_recipe.problems_for(ctx, form))
    )
    _generate(
        ctx,
        form,
        label="Make 3D" if mesh else create_assets.selected(form).create_label,
        enabled=not problems and not ctx.busy("submit"),
        busy=ctx.busy("submit"),
        problems=problems,
        show_count=True,
        count=int(form.get("count", 1)),
        press=(lambda: settings_3d.promote(ctx, source, form)) if mesh else None,
        width=imgui.get_content_region_avail().x,
    )


#: The refusal's sentence, which names the stage that draws the picker.
PENDING_CANDIDATES_REFUSAL = (
    "Decide the pending candidates first -- they are on the Mesh stage."
)


def _with_pending_candidates_problem(ctx: Any, problems: list[Any]) -> list[Any]:
    """Add a refusal while an undecided candidate group exists. -> ``problems``.

    The 2026-09-18 audit, finding create-01: a second candidate-producing
    brief used to hide the *older* undecided group with nothing on screen --
    ``candidates.pending`` only ever offers the newest group, and
    ``state.Filters.matches`` hides every ``candidate_group`` row from the
    library regardless of age, so a third submission before the second is
    decided orphaned the first forever. Refusing a new submission while any
    group -- decided-looking or not -- is still pending means there is never
    a second group to hide the first behind.
    """
    from .... import candidates as candidates_mod

    cache = getattr(ctx, "cache", None)
    jobs = getattr(cache, "jobs", None) if cache is not None else None
    if jobs is None or candidates_mod.pending_cached(cache) is None:
        return problems
    from ....problems import Problem

    # The picker is drawn only on the Mesh stage's tray (``workspace.draw``
    # offers a pending group when ``stage in (None, "mesh")``), and this
    # refusal is raised only on Reference: the 2026-10-03 audit, finding
    # create-45, found a user holding an old undecided batch blocked here with
    # a sentence that named no destination.
    return [*problems, Problem(PENDING_CANDIDATES_REFUSAL)]


def _type(ctx: Any, form: dict[str, Any], *, width: float | None = None) -> None:
    """What to make. The one choice that decides what everything else means."""
    before = create_assets.selected(form).key
    with focus.item(ctx.state, FOCUS_PANE, "asset_type"):
        picked = widgets.combo(
            "##generation-type",
            before,
            list(create_assets.ASSET_TYPE_OPTIONS),
            width=sp(TYPE_W) if width is None else width,
            tooltip=_TYPE_HINTS.get(before, ""),
        )
        # The tour's character-type step rings this combo by its caption (the
        # 2026-10-03 audit's tour-09: it named "Generation type", which no
        # control on screen says, and pointed at nothing).
        anchors.mark("create/type")
    form["asset_type"] = picked if picked in create_assets.ASSET_TYPES else before
    form["generation_type"] = form["asset_type"]
    if create_assets.sync_legacy_fields(form).key != before:
        ctx.state.clear_field_error("asset_type")
    _ring(ctx, "asset_type")


def _prompt(ctx: Any, form: dict[str, Any], width: float, *, height: float = PROMPT_H) -> None:
    """The words. The field this whole rearrangement is about.

    An explicit ``width``, which is the one thing that matters here:
    ``widgets.multiline`` defaults to -1, meaning *fill the row*. In a column
    that is what you want; on a row it takes the width the controls after it
    were going to use, and ``same_line`` past the pane edge draws a control
    nowhere -- so everything after it vanished off the right-hand side
    entirely.
    """
    before = form["prompt"]
    with focus.item(ctx.state, FOCUS_PANE, "prompt"):
        form["prompt"] = widgets.multiline(
            "##brief-prompt", before, sp(height), _max_prompt(), width=width
        )
        anchors.mark("create/prompt")
        widgets.char_count(form["prompt"], _max_prompt())
    if form["prompt"] != before:
        ctx.state.clear_field_error("prompt")
    if imgui.is_item_hovered() and not str(form["prompt"]).strip():
        imgui.set_tooltip("Describe one subject -- a prop, a character, a surface.")
    _ring(ctx, "prompt")


def _count(
    ctx: Any,
    form: dict[str, Any],
    counts: tuple[int, ...],
    current: int,
) -> None:
    """How many alternatives one press should draw -- Candidates, on both stages.

    Not drawn at all for a sheet: both sheet doors refuse a batch and say why,
    so four radios of which three are refusals would be a control offering
    what the thing behind it will not do. ``sync_legacy_fields`` has already
    written ``count = 1`` for those. ``counts`` is the stage's range and
    ``current`` the value it reads back (Mesh clamps a stale stored one).

    The caption is the caller's (:func:`inputs` draws ``COUNT_LABEL`` as the
    column's own ``secondary`` line above the pills). This used to centre an
    optional beside-the-pills label and the pills themselves inside the old
    one-row command bar with spacer dummies; the bar is the rail alone now, the
    pills sit on their own line in the column, and the 2026-10-04 audit
    (finding create-61) found the parameter always ``False`` and the spacers
    centring in a row that no longer exists.
    """
    hints = _MESH_COUNT_HINTS if counts is _MESH_COUNTS else _COUNT_HINTS
    imgui.begin_group()
    imgui.push_item_width(sp(COUNT_W))
    with focus.item(ctx.state, FOCUS_PANE, "count") as focused:
        changed, picked = controls.segmented_choice(
            "brief-count",
            tuple((str(n), str(n)) for n in counts),
            str(current),
            compact=True,
            tooltips=hints,
        )
        if changed:
            form["count"] = int(picked)
            ctx.state.clear_field_error("count")
        # Hand-answered, as it was in the column: a row of radios is one
        # control to the keyboard even though it is four items to imgui.
        if focused:
            here = counts.index(current) if current in counts else 0
            before_arrow = form["count"]
            if imgui.is_key_pressed(imgui.Key.left_arrow):
                form["count"] = counts[(here - 1) % len(counts)]
            if imgui.is_key_pressed(imgui.Key.right_arrow):
                form["count"] = counts[(here + 1) % len(counts)]
            # The click branch above clears the ring on a change; this
            # hand-rolled branch edits ``form["count"]`` the same way and must
            # clear the same error, or a user who fixes an invalid count with
            # the keyboard instead of a click keeps a ring pointing at a value
            # that is no longer wrong. The 2026-09-05 audit, finding create-07.
            if form["count"] != before_arrow:
                ctx.state.clear_field_error("count")
    imgui.pop_item_width()
    _ring(ctx, "count")
    imgui.end_group()


def _generate(
    ctx: Any,
    form: dict[str, Any],
    *,
    label: str,
    enabled: bool,
    problems: list[Any],
    show_count: bool,
    count: int,
    press: Callable[[], None] | None = None,
    width: float | None = None,
    busy: bool = False,
) -> None:
    """The press. Always visible, which is the point of the bar.

    ``label`` is the stage's own (the asset type's on Reference, "Make 3D" on
    Mesh) and ``press`` the stage's door -- ``None`` means Reference's
    ``settings_2d.generate``. ``width`` is the column's full available width
    (:func:`submit_control`, the only caller, always passes it); :data:`GENERATE_W`
    is only the fallback for a caller that passes none.

    The *reason* it is disabled is the button's own tooltip, and the pinned
    plan footer under the stage's column lists every problem and offers the
    one-click repairs: a bar has no room for a list, and a button that says
    nothing about why it is dead is the complaint this redesign started from.

    ``show_count`` says whether the count pills are on screen: they carry
    their own value, so restating it here as well would be a second control
    saying the same number an inch away. It is only appended when they are not.
    """
    from .panes import settings_2d

    with focus.item(ctx.state, FOCUS_PANE, "generate") as focused:
        pressed = widgets.primary_button(
            label,
            (sp(GENERATE_W) if width is None else width, sp(PROMPT_H)),
            enabled=enabled,
            # ``Problem`` is a str subclass -- the message *is* the object.
            reason=disabled_reason(problems, busy),
            tooltip=_generate_tooltip(show_count, count),
        )
        anchors.mark("create/generate")
        if focused and enabled and _enter_pressed():
            pressed = True
    if pressed:
        if press is not None:
            press()
        else:
            settings_2d.generate(ctx, form)


#: What a Generate held off by a task still at the door says on hover.
SUBMIT_BUSY_REASON = "A request is still being sent to the queue; try again in a moment."


def disabled_reason(problems: list[Any], busy: bool) -> str:
    """The tooltip a disabled Generate / Make 3D wears. -> ``""`` when live.

    The first problem wins -- the plan footer lists the rest. With no problem
    but a submit task still in flight (``ctx.busy("submit")``), the button is
    disabled all the same, and the 2026-10-03 audit, finding create-32, found
    it greyed with no sentence on hover for the whole length of the task.
    """
    if problems:
        # ``Problem`` is a str subclass -- the message *is* the object.
        return str(problems[0])
    return SUBMIT_BUSY_REASON if busy else ""


def _generate_tooltip(show_count: bool, count: int) -> str:
    """Ctrl+Enter, plus the count once its own pills are off screen.

    The shortcut stays first because that is what the tooltip is mainly for;
    the count is appended rather than replacing it, separated the same way a
    plan block reads a list of facts. Silent while the pills are visible --
    a tooltip repeating a control drawn an inch to its left is noise, not help.
    """
    if show_count:
        return "Ctrl+Enter"
    noun = "candidate" if count == 1 else "candidates"
    return f"Ctrl+Enter · {count} {noun}"


# The Reset confirm's own words, module-level so a regression test can read
# them without an imgui context (``_reset`` below draws a button first, which
# needs one). The 2026-09-08 audit, finding create-03: this text used to name
# only six things -- the prompt, the negative prompt, the model, the LoRA,
# the reference and the run controls -- while ``settings_2d._reset`` actually
# replaces the *whole* form with ``default_form_2d()``, silently discarding
# the asset type (the whole Image/3D Model/Seamless Material/Tileset/Sprite
# Sheet/Character selection) and every Tileset/Sprite/Character field the
# user had typed in (materials, variants, style_lock, seam_erase, palette,
# cell_size, and the rest) -- none of which the old text named. The fix
# broadens the *text* to match what Reset has always discarded, per the
# 2026-09-08 audit's own call: narrowing ``_reset`` instead would make "another
# like this" lose the asset type on every rerun, which is a bigger behaviour
# change than a confirm dialog earns on its own.
_RESET_CONFIRM_MESSAGE = (
    "The prompt, the negative prompt, the model, the LoRA, the reference and "
    "the run controls go back to their defaults, with a freshly rolled seed "
    "-- and so does everything else on this form, including the asset type "
    "(Image, 3D Model, Seamless Material, Tileset, Sprite Sheet or Character) "
    "and any Tileset, Sprite Sheet or Character fields you have filled in. "
    "The 3D form is untouched."
)


#: The Mesh stage's confirm, the same pattern as the image one above: it says
#: what *its* Reset replaces (the whole of ``DEFAULT_FORM_3D``) and what it
#: keeps. The seed goes back to *unset*, not to a rolled number.
_RESET_MESH_CONFIRM_MESSAGE = (
    "Every mesh setting -- the resolution, budget, size, background, seed, "
    "candidate count, rig and engine controls -- goes back to its default, "
    "with the seed unset. The chosen source is kept, and the image settings "
    "are untouched."
)


def _reset_title(noun: str) -> str:
    """One pattern for both stages: "Reset the image settings?" / "... mesh ...?"."""
    return f"Reset the {noun} settings?"


def _source_chip(ctx: Any, width: float) -> None:
    """The Mesh stage's "what": the reference this press reconstructs.

    A thumbnail and the reference's name, or "Choose an image..." with a drop
    hint when there is none. **It is a real button drawn at ``width`` with its
    picture laid over it**, so it has a hit rect, a keyboard stop and a tooltip
    like any control, and it is a drop target for a library card dragged out of
    the Library and for a file dropped on the window (``events`` flashes it).
    Clicking it opens the file picker -- whose result is still *uploaded* as a
    new reference, today's behaviour, not a pick of an existing one.

    The name shown is the *effective* source (``settings_3d.bar_source``): the
    explicit pick, else the reference behind a selected finished mesh -- what
    the press will actually use, which is the whole job of the line it replaced.
    """
    from ....panes import thumbs
    from ...library.ui.panes import library
    from .panes import settings_3d

    state = ctx.state
    height = sp(PROMPT_H)
    source = settings_3d.bar_source(ctx)
    derived = source is not None and ctx.cache.get(state.source_job) is None
    dragging = library.dragged_job(ctx)
    picker_open = ctx.busy("upload")
    if source is None:
        title, sub = "Choose an image...", "or drop a card or a file here"
        tip = "Click to choose an image, or drop a library card or a file here."
    else:
        title = str(source.get("name") or source.get("prompt") or source["id"])
        sub = "this mesh's reference" if derived else f"reference - {source['id']}"
        tip = f"{title}\nClick to upload a different image, or drop a library card here."
    if dragging is not None:
        sub = "drop it here to use it"

    with focus.item(state, FOCUS_PANE, "source") as focused:
        clicked = controls.button(
            "##mesh-source",
            (width, height),
            role=controls.ButtonRole.GHOST,
            enabled=not picker_open,
            reason="A file picker is already open.",
            tooltip=tip,
        )
        if focused and not picker_open and _enter_pressed():
            clicked = True
    low = imgui.get_item_rect_min()
    high = imgui.get_item_rect_max()
    if clicked:
        ctx.submit("upload", dialogs.open_file, "Choose a reference image", dialogs.IMAGE_FILTER)
    if imgui.begin_drag_drop_target():
        payload = imgui.accept_drag_drop_payload_py_id(library.DRAG_JOB)
        if payload is not None and state.dragging_job:
            # **Names the source and does nothing else.** This used to go
            # through ``library.select`` so the inspector showed the dropped
            # card, but a card from another creation then made
            # ``session.sync`` resume *that* creation next frame and replace
            # the tuned Mesh form with its draft or the defaults (the
            # 2026-10-03 audit, finding create-07). The chip already names the
            # reference, which is what the drop was for; ``can_drag`` is the
            # same predicate the card lifted by, so a card that has since
            # stopped being a finished reference is refused rather than named.
            dropped = ctx.cache.get(state.dragging_job)
            if dropped is not None and library.can_drag(dropped):
                state.source_job = state.dragging_job
            state.dragging_job = None
        imgui.end_drag_drop_target()
    if dragging is not None:
        hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_blocked_by_active_item.value)
        widgets.ring(
            low,
            high,
            theme.ACCENT if hovered else theme.MUTED,
            0.9 if hovered else 0.4,
            2.0 if hovered else 1.0,
        )
    else:
        # The same ring, fading, for a file dropped from Explorer (H70): the
        # two arrivals look the same because they are the same event.
        widgets.ring(low, high, theme.ACCENT, widgets.drop_flash(state, "3d-source"))

    # The picture laid over the button. Drawn after the drop target above,
    # which must attach to the *button* item, and never interactive itself.
    pad = sp(4.0)
    side = max(1.0, height - 2 * pad)
    imgui.set_cursor_screen_pos((low.x + pad, low.y + pad))
    if source is not None:
        thumbs.job_thumb(ctx, source, side)
    else:
        widgets.thumb_placeholder(side, icons.IMAGE)
    text_w = max(1.0, width - side - 3 * pad)
    text_h = imgui.get_text_line_height() * 2 + imgui.get_style().item_spacing.y
    imgui.set_cursor_screen_pos((low.x + 2 * pad + side, low.y + max(0.0, (height - text_h) * 0.5)))
    imgui.begin_group()
    imgui.text(widgets.fit_text(title, text_w))
    widgets.muted(widgets.fit_text(sub, text_w))
    imgui.end_group()
    # Back to the row: the next control continues from the chip's right edge,
    # not from wherever the overlay left the cursor.
    imgui.set_cursor_screen_pos((high.x, low.y))
    imgui.dummy((0.0, 0.0))


def _ring(ctx: Any, field: str) -> bool:
    """Mark the control just drawn if a refusal named it. -> whether it did.

    ``widgets.field_error`` is the column's version and draws the message and
    any install offer *below* the control, which in a one-row bar would push
    the row apart. The ring alone here; the words are in the plan block.
    """
    if not (getattr(ctx.state, "field_errors", None) or {}).get(field):
        return False
    widgets.ring(imgui.get_item_rect_min(), imgui.get_item_rect_max(), theme.ERR, 0.9, thick=1.5)
    return True


def _max_prompt() -> int:
    """``MAX_PROMPT``, imported lazily so this module stays cheap to import."""
    from .....service.validation import MAX_PROMPT

    return MAX_PROMPT


def _enter_pressed() -> bool:
    return imgui.is_key_pressed(imgui.Key.enter) or imgui.is_key_pressed(imgui.Key.keypad_enter)


def _species_count() -> int:
    """How many species the character registry ships. Counted, never typed.

    ``characters.family`` imports nothing but the standard library, so reading
    it at import time here costs a dict copy and drags nothing in behind it.
    """
    from .....characters.family import families

    return len(families())


#: One line per type, on the combo's tooltip. The five-row descriptive block
#: that used to sit under the selector is gone with the column it sat in; this
#: is what survives of it, which is the orientation and not the prose.
_TYPE_HINTS: dict[str, str] = {
    "image": "A standalone 2D image.",
    "3d_model": "A reference image you can turn into a 3D model.",
    "seamless_material": "A seamless surface texture.",
    "tileset": "A coherent pixel-art tile sheet.",
    "sprite_sheet": "A character in several frames and directions.",
    # The real scope, in one line, because every other hint here describes an
    # SDXL request and this one describes the opposite: a body built from the
    # species registry, rigged and rendered on the CPU. "No GPU needed" is the
    # half a user with a small card most needs to read.
    #
    # The species count is *counted*, never typed: a sibling adding a row to
    # ``characters.family._FAMILIES`` must not leave a tooltip promising the
    # number the registry held last month.
    "character": (
        f"{_species_count()} species across four body plans, built and animated "
        f"into a sprite sheet. No GPU needed."
    ),
}

__all__ = [
    "FOCUS_PANE",
    "RAIL_ONLY_H",
    "bar_height",
    "draw",
]
