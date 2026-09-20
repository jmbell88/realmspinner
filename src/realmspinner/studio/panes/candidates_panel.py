"""The mesh-candidate picker: three attempts at one asset, and which to keep.

Drawn at the top of the 3D inspector, above the identity header, because it is
a question the user is being asked rather than a description of what they are
looking at -- and because the rows it lists are hidden from the library, so
this is the only way back to them.

Two rules are worth stating here rather than in a comment at the call site.

**Selecting a candidate changes the selection and nothing else.** It calls
``state.select``, and the frame loop's ``_sync_viewer`` does the rest -- which
is what keeps the pose-mode guard, the ``viewer.pending`` drop rule and the
off-thread parse applying to a candidate exactly as they apply to any other
asset. A shortcut into ``viewer.load_model`` from here would have to reproduce
all three, and would get one of them wrong.

**Keep never deletes.** It settles the group (every member becomes an ordinary
asset) and then *offers* the losers to the ordinary delete path behind the
ordinary confirm. Deleting two meshes on a single click, because the user
pressed the button that means "I like this one", is not a trade anybody agreed
to.

**A5: grading feeds the corpus, and the nudge is the whole intervention.**
Keeping a candidate used to be a decision that never reached a verdict --
nothing here ever showed, or asked for, a grade, so a kept mesh taught
findings nothing about which settings won. `_grades` reads every member's
latest verdict in one call, memoized on the job cache's own generation
counter the way `panes.landing.rows` already is, so it costs one query per
*refresh* of the group rather than one per member or one per frame; a grade
just filed reaches it because `panes.inspector.record_verdict` already calls
`ctx.cache.invalidate()`. What is drawn from it is deliberately thin: a
grade beside the candidate that has one, and a muted line while a finished
attempt does not. No ordering, no pre-selection, no filtering -- the judge's
own doctrine (`dev/INVARIANTS.md`, "advisory... sorts and never filters")
applies here even though nothing here is the judge, because the failure mode
is the same one: a picker that reordered or hid a candidate on the strength
of a grade would be making the keep decision instead of nudging it.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ...service import jobs as svc_jobs
from .. import candidates as candidates_mod
from .. import controls, dialogs, widgets
from ..manual import render as manual_render
from ..modes.library.ui.panes import library
from ..modes.review import mode as review_mode
from ..tokens import sp

#: The nudge, drawn once per group while some finished attempt has no grade.
#: Basic-Latin only (imgui's default atlas), so " - " and not an em dash --
#: the same rule ``review_mode``'s own UI strings state for themselves.
_NUDGE = "Grade each attempt before you keep one - they feed What works."

#: One memoized answer: ``(key, {job_id: grade})``, where ``key`` names the
#: cache generation, group and member set it was read for. Module-level like
#: ``panes.landing._ROWS_CACHE``, for the same reason -- there is exactly one
#: candidate group offered at a time (``candidates.pending`` says so), so one
#: slot is the whole cache rather than something keyed per group.
_GRADES_CACHE: tuple[Any, dict[str, int | None]] | None = None


def draw(ctx: Any) -> None:
    """Draw the picker for the newest undecided group, if there is one."""
    group = candidates_mod.pending_cached(ctx.cache)
    if group is None:
        return
    widgets.section("Candidates")
    manual_render.help_button(ctx, "candidates")
    if group.finished:
        if group.all_failed:
            # The 2026-09-14 audit, finding create-04: this caption used to
            # say "Keep one" even when every attempt had failed, over a
            # picker whose Keep button can never open for a single member --
            # there is nothing to keep, and the caption said otherwise.
            widgets.muted(f"None of the {len(group.members)} attempts finished. Discard them?")
        else:
            widgets.muted(f"{len(group.members)} meshes from one reference. Keep one.")
    else:
        widgets.muted(
            f"{group.done_count} of {len(group.members)} finished. "
            "Keep becomes available once they all have."
        )
    selected = ctx.state.selected
    grades = _grades(ctx, group)
    for member in group.members:
        _member(ctx, group, member, selected == member["id"], grades)
    nudge = _nudge_text(group, grades)
    if nudge is not None:
        widgets.muted(nudge)
    widgets.divider()


def _grades(ctx: Any, group: Any) -> dict[str, int | None]:
    """``{job_id: grade}`` for every member of ``group``. -> One ``verdicts_for``
    read per group, not one per member and not one per frame.

    Memoized on the job cache's own generation counter -- ``jobs_cache.visible``
    and ``panes.landing.rows`` are already memoized the identical way, against
    the identical counter, and it only moves when a fresh read actually lands.
    That is what makes this cheap to call from every frame's ``draw``: a grade
    just filed shows up the next time the cache refreshes, which
    ``panes.inspector.record_verdict``'s own ``ctx.cache.invalidate()`` call is
    what schedules.

    A ``None`` generation -- a headless ``ctx`` with no real cache, as every
    test here builds -- never memoizes: nothing behind it can go stale to
    avoid re-reading, and a test asking "did this cost one query" builds a
    real ``JobsCache`` to get an answer that means anything.

    The key holds ``cache`` itself rather than ``id(cache)``, for the same
    reason ``candidates.pending_cached`` does (the 2026-09-20 audit, finding
    create-05): a bare id can be a freed object's address handed to a new
    one, and a strong reference to the real object can never collide that
    way.
    """
    global _GRADES_CACHE
    member_ids = [m["id"] for m in group.members]
    cache = getattr(ctx, "cache", None)
    generation = getattr(cache, "_generation", None)
    key = (cache, generation, group.group, tuple(member_ids))
    if generation is not None and _GRADES_CACHE is not None and _GRADES_CACHE[0] == key:
        return _GRADES_CACHE[1]
    try:
        recorded = ctx.svc.store.verdicts_for(member_ids, source=review_mode.SOURCE, stage="model")
    except Exception:
        # Never fail a frame over a grade nobody asked for explicitly --
        # ``panes.inspector.is_graded`` takes the same stance for the same
        # reason. The picker itself still works with no grades in hand.
        return {}
    grades = {job_id: verdict.get("grade") for (job_id, _source), verdict in recorded.items()}
    if generation is not None:
        _GRADES_CACHE = (key, grades)
    return grades


def _status_text(member: dict[str, Any], grades: dict[str, int | None]) -> str:
    """The status line, with the recorded grade appended if there is one.

    A plain string, on purpose: it is what makes the grade assertable without
    a GL context, and it is what ``_member`` hands straight to ``widgets.muted``
    rather than composing on two lines that would need a second ``same_line``.
    """
    status = candidates_mod.status_line(member)
    grade = grades.get(member["id"])
    if grade is None:
        return status
    text = review_mode.grade_text(grade)
    return f"{status} · {text}" if text else status


def _nudge_text(group: Any, grades: dict[str, int | None]) -> str | None:
    """``_NUDGE``, or ``None`` while nothing in ``group`` needs it.

    Only a ``done`` candidate can carry a grade at all -- an errored or
    cancelled attempt never reaches ``panes.inspector``'s verdict section, so
    counting it as "settled and ungraded" would leave the sentence on screen
    forever for a mesh nobody could ever grade. That is what "settled" means
    here, narrower than ``candidates.Group.finished``'s own (which also
    counts a failure as settled, because *that* question is "is there
    anything left to wait for").
    """
    if any(m.get("status") == "done" and grades.get(m["id"]) is None for m in group.members):
        return _NUDGE
    return None


#: How wide a candidate's "A"/"B" picker button is, in design pixels. Wide
#: enough for two characters and the frame padding, narrow enough that the
#: status line beside it still fits a sidebar.
_PICKER_BUTTON = 44.0


def _member(
    ctx: Any, group: Any, member: dict[str, Any], current: bool, grades: dict[str, int | None]
) -> None:
    job_id = member["id"]
    label = candidates_mod.label(member)
    # ``sp``, not raw pixels: this is a design measurement like every other
    # size in the app, and unscaled it shrinks against a 150%-scaled sidebar.
    width = (sp(_PICKER_BUTTON), 0.0)
    if current:
        if widgets.primary_button(f"{label}##candidate-{job_id}", width):
            select(ctx, job_id)
    elif controls.button(f"{label}##candidate-{job_id}", width):
        select(ctx, job_id)
    # Safe against the pane edge: a 44 dp button plus one item spacing inside a
    # 300 dp sidebar leaves most of the line. The smoke-test guard measures it.
    imgui.same_line()
    widgets.muted(_status_text(member, grades))
    # Keep is offered on the selected candidate only, and only once every
    # member has settled: keeping one dissolves the group, so a member still
    # queued would quietly become an asset nobody chose.
    if current:
        # The 2026-09-14 audit, finding create-04: when every member has
        # failed, Keep's gate (``group.finished and member done``) can never
        # open -- no member is ever ``done`` -- and nothing offered a way out
        # of a group ``Filters.matches`` hides from the library forever.
        # Discard takes Keep's place here rather than sitting beside it: a
        # button that can never become enabled is not a second option, it is
        # dead weight next to the one that works.
        if group.all_failed:
            if controls.button(f"Discard all##discard-{group.group}", (-1, 0)):
                discard(ctx, group)
        else:
            ready = group.finished and member.get("status") == "done"
            if widgets.disabled_button(
                f"Keep this one##keep-{job_id}",
                ready,
                (-1, 0),
                # Keeping one dissolves the group, so a member still queued would
                # quietly become an asset nobody chose -- which is why the gate is
                # about the *group* even though the button is on one candidate.
                reason="The other attempts have not finished yet."
                if not group.finished
                else "This one did not finish, so there is nothing to keep.",
            ):
                keep(ctx, group, job_id)
            if not ready and member.get("status") != "done":
                widgets.hint_text("This one did not finish; keep another.")


def select(ctx: Any, job_id: str) -> None:
    """Show a candidate. The viewer follows the selection, not this click."""
    ctx.state.select(job_id)


def keep(ctx: Any, group: Any, job_id: str) -> None:
    """Settle the group, then offer the losers to the delete path."""
    losers = group.losers(job_id)
    try:
        svc_jobs.keep_candidate(ctx.svc, job_id)
    except Exception as exc:
        # Inline rather than on a task thread: it is one UPDATE against the
        # store the frame loop already reads directly, and the answer decides
        # what the confirm below says.
        ctx.toast(f"That candidate could not be kept: {exc}", "error", action="log")
        return
    ctx.cache.invalidate()
    ctx.toast("Kept. The other attempts are in the library now.")
    if not losers:
        return
    ctx.confirms.ask(
        dialogs.Confirm(
            title=f"Delete the other {len(losers)}?",
            message=(
                "The attempts you did not keep are ordinary assets now. "
                "Deleting removes them and everything derived from them."
            ),
            confirm_label="Delete",
            cancel_label="Keep them",
            # The library's own delete, so a loser goes by the one path that
            # clears the selection and the tick set as well as the row -- the
            # *batch* spelling, so seven losers are one toast with one Undo
            # rather than seven of each.
            on_confirm=lambda: library.delete_assets(ctx, list(losers)),
        )
    )


def discard(ctx: Any, group: Any) -> None:
    """Dissolve an all-failed group, then *offer* to trash every member.

    The 2026-09-14 audit, finding create-04: no door existed to clear a
    group where every attempt errored or was cancelled, so ``N`` failed rows
    sat hidden from the library forever (``Filters.matches`` hides any row
    still carrying ``candidate_group``) with a picker that could never open
    Keep for any of them. ``keep_candidate`` is still the only door that
    clears ``candidate_group`` (through ``store.resolve_candidates``, one
    statement for the whole group) -- called on an arbitrary member it
    settles the group exactly as :func:`keep` does.

    **Reopened the same day**: this used to trash every member outright, on
    the reasoning that an undoable trash was gentle enough. It is not --
    ``docs/manual/23-generating-meshes.md`` promises for this exact picker
    that "only then are you *asked* whether to delete the ones you did not
    keep. Nothing is ever deleted on your behalf", and an Undo toast is still
    a deletion the user did not ask for. :func:`keep`'s shape is the fix
    already proven for a mixed group -- settle first, *then* confirm -- so
    Discard now takes it verbatim, offering every member rather than only
    the losers because Discard has no winner to spare.
    """
    member_ids = [m["id"] for m in group.members]
    if not member_ids:
        return
    try:
        svc_jobs.keep_candidate(ctx.svc, member_ids[0])
    except Exception as exc:
        ctx.toast(f"That group could not be discarded: {exc}", "error", action="log")
        return
    ctx.cache.invalidate()
    ctx.toast("None of the attempts finished. They are in the library now.")
    ctx.confirms.ask(
        dialogs.Confirm(
            title=f"Delete all {len(member_ids)}?",
            message=(
                "These attempts are ordinary assets now, and none of them "
                "finished. Deleting removes them and everything derived "
                "from them."
            ),
            confirm_label="Delete",
            cancel_label="Keep them",
            # keep()'s own reasoning, applied to every member rather than
            # only the losers: the library's one path that clears the
            # selection and the tick set as well as the row, one toast and
            # one Undo for the whole batch.
            on_confirm=lambda: library.delete_assets(ctx, list(member_ids)),
        )
    )
