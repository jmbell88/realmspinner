"""The inspector's element state: what Points, Edges and Faces show in the right column.

Replaces the old Object tab's body (and the Adjust card that floated over the
viewport) with four things, top to bottom, all of them drawn from tables that already
exist rather than listed here:

* **The Select row** -- Select all / None / Invert / Linked, the ops of
  ``menutree.GROUPS["select"]`` that apply in the mode (so a ``Grow`` registered
  tomorrow is a button on the day it lands in the group), greyed with the registry's
  own sentence ("Select linked needs a selection").
* **The element line** -- what is picked, in words ("3 faces selected"), or, with
  nothing picked, what to click; then the median as three typed numbers
  (``props._element_median`` / ``commit_median``) and the measurement.
* **Shape it** -- the mode's mesh ops, one ``disabled_button(reason=)`` per op
  (``inspector_actions.draw_group``): the groups in :data:`VISIBLE_GROUPS` in the card.
* **Last action** -- the op just run and its live numbers (``recent_op``), only while
  the document is exactly as that op left it, right under Shape it. A change re-runs the
  op from the state it started in, so editing a number is one undo step equal to having
  run the op at the final value. It scrolls itself into view once, when it appears.
* **The folds** -- every other group (Set Origin, Shading, Separate) behind a closed
  fold of its own, below. A closed fold never asks an op's ``enabled``.

**Layout does not depend on what is picked.** The median is always drawn (greyed with a
reason when nothing is picked) and the sentence above it always reserves the height of
the longest thing it can say, so a pick that arrives between a press and its release
cannot move the button under the pointer (``tests/modes/clay/test_inspector_elements.py``
compares the rects).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from imgui_bundle import imgui

from .... import theme, tokens, widgets
from ....tokens import sp
from .. import menutree, recent_op
from .. import mode as clay_mode
from .. import ops as clay_ops
from . import inspector_actions
from .panes import menu as clay_menu
from .panes import props

#: The groups of ``menutree.BARS[mode]`` that sit in the open on the Shape it card.
#: A policy about *groups*, never a list of ops: what is in each group is the
#: registry's, so an op added to Mesh or Delete shows up here with no edit. Every
#: other group (Set Origin, Shading, Separate...) is a closed fold named by its
#: group.
VISIBLE_GROUPS = ("mesh", "delete")

#: ``element mode -> (singular, plural, article)`` for the sentence. "Point" for
#: ``vertex``, because Points is what the mode is called everywhere a person reads.
NOUNS = {
    "vertex": ("point", "points", "a"),
    "edge": ("edge", "edges", "an"),
    "face": ("face", "faces", "a"),
}


def select_ops(mode: str) -> tuple[clay_ops.Op, ...]:
    """The Select group's ops that apply in *mode*, in the group's order."""
    return menutree.grouped_ops("select", mode)


def short_label(op: clay_ops.Op) -> str:
    """"Select All" -> "All", "Invert Selection" -> "Invert": the row's own words.

    The ops' labels are the menus' and say "Select" because a menu line has no row
    heading over it; here the row is headed "Select", so repeating it on every button
    is a third of the width gone. The full label stays in the tooltip.
    """
    text = op.label.removeprefix("Select ").removesuffix(" Selection")
    return text or op.label


def element_counts(doc: Any) -> tuple[int, int]:
    """``(elements, shapes)`` picked in *doc*'s element mode.

    Hidden shapes are not counted: a drag, the gizmo and every element door skip
    them (``selection._element_pickable``), so "N selected" must not promise elements
    the next move would leave where they were.
    """
    # Memoised on the readout key, as the median line beside it is: the
    # 2026-10-09 audit's clay-23 found this scanning ``doc.objects`` once per
    # picked shape on every frame (61 ms at 2,000 picked), though nothing it reads
    # moves without ``rev`` moving.
    return props._memoised(doc, "element_counts", None, lambda: _count_elements(doc))


def _count_elements(doc: Any) -> tuple[int, int]:
    from .....kernels.mesh.selection import _element_pickable

    # One pass for a uid index: ``ClayDoc.by_uid`` is a linear scan.
    by_uid = {obj.uid: obj for obj in doc.objects}
    total = shapes = 0
    for uid, sel in doc.element_sel.items():
        obj = by_uid.get(uid)
        if obj is None or not _element_pickable(obj):
            continue
        count = sel.count(doc.element_mode)
        if count:
            total += count
            shapes += 1
    return total, shapes


def empty_text(mode: str) -> str:
    """What to do when nothing is picked, in the mode's own noun."""
    singular, _plural, article = NOUNS.get(mode, ("part", "parts", "a"))
    return (
        f"Nothing selected. Click {article} {singular}, or hold Shift and drag to select several."
    )


def element_line(doc: Any) -> str:
    """The sentence under the Select row: the count, or what to click."""
    total, shapes = element_counts(doc)
    if total == 0:
        return empty_text(doc.element_mode)
    singular, plural, _article = NOUNS[doc.element_mode]
    text = f"{total} {singular if total == 1 else plural} selected"
    if shapes > 1:
        text += f" across {shapes} shapes"
    return text


# --- the Select row -------------------------------------------------------------


def draw_select_row(ctx: Any, state: Any, tab: Any, doc: Any) -> None:
    """Select all / None / Invert / Linked, from the registry's Select group."""
    ops = select_ops(doc.element_mode)
    if not ops:
        return
    reasons = inspector_actions.status(doc, ops, bool(tab.saving))
    widgets.field_label("select")
    labels = [short_label(op) for op in ops]
    equal = widgets.grid_width(len(ops))
    # Equal columns while the shortest word still fits its column, else a wrapping
    # run of natural widths: a 200 px sidebar must not clip "Linked".
    fits = equal >= max(widgets.button_width(label) for label in labels)
    for index, (op, label) in enumerate(zip(ops, labels, strict=True)):
        title = f"{label}##insp-select/{op.name}"
        if index:
            widgets.same_line_or_wrap(equal if fits else widgets.button_width(title))
        reason = reasons.get(op.name, "")
        if widgets.disabled_button(
            title,
            not reason,
            (equal, 0) if fits else (0, 0),
            reason=reason,
            tooltip=inspector_actions.tooltip_for(op),
        ):
            clay_mode.fire_op(ctx, doc, op)


# --- the element line -----------------------------------------------------------


def draw_element_line(ctx: Any, state: Any, doc: Any) -> None:
    """The count sentence, the median as typed numbers, and the measurement."""
    text = element_line(doc)
    # Reserve the height of the longest thing the sentence can say (the empty-state
    # instruction), so the count that replaces it does not pull everything below up.
    wrap = imgui.get_content_region_avail().x
    longest = max(
        imgui.calc_text_size(candidate, None, False, wrap).y
        for candidate in (text, empty_text(doc.element_mode))
    )
    top = imgui.get_cursor_pos_y()
    widgets.muted_wrapped(text)
    spent = imgui.get_cursor_pos_y() - top
    if longest > spent:
        imgui.dummy((0, longest - spent))
    imgui.dummy((0, sp(tokens.SP_1)))
    props._element_median(doc, state, ctx)
    if doc.element_mode in ("edge", "face"):
        line = props.element_measure_text(doc, props._unit_of(state))
        # A blank line of the same height when there is nothing to measure, so the
        # card below does not move when the first pick lands.
        widgets.muted(line or " ")


# --- Shape it -------------------------------------------------------------------


def shape_groups(mode: str) -> tuple[list[Any], list[Any]]:
    """``(open groups, folded groups)`` for *mode*: ``action_groups`` split by
    :data:`VISIBLE_GROUPS`, in the menu's order."""
    groups = inspector_actions.action_groups(mode)
    return (
        [group for group in groups if group[0] in VISIBLE_GROUPS],
        [group for group in groups if group[0] not in VISIBLE_GROUPS],
    )


def draw_shape_it(ctx: Any, state: Any, tab: Any, doc: Any) -> None:
    """The card: the open groups of the mode's ops, asked once per revision through
    ``inspector_actions.status``."""
    visible, _folded = shape_groups(doc.element_mode)
    widgets.section("Shape it")
    ops = tuple(op for _key, _label, group in visible for op in group)
    reasons = inspector_actions.status(doc, ops, bool(tab.saving))
    for key, label, group in visible:
        inspector_actions.draw_group(
            ctx, state, tab, doc, key, label, group, reasons, heading=False
        )
    widgets.end_section()
    imgui.dummy((0, sp(tokens.SP_2)))


def draw_folds(ctx: Any, state: Any, tab: Any, doc: Any) -> None:
    """Every other group of the mode's ops, each behind a closed fold named by it.

    A fold's body -- and every ``op.enabled`` under it -- runs only while it is open
    (``tests/modes/clay/test_inspector_elements.py`` spies on ``Op.enabled``).
    """
    _visible, folded = shape_groups(doc.element_mode)
    saving = bool(tab.saving)
    for key, label, group in folded:
        if widgets.header(label, default_open=False, persist_key=f"clay/insp/el/{key}"):
            inspector_actions.draw_group(
                ctx,
                state,
                tab,
                doc,
                key,
                label,
                group,
                inspector_actions.status(doc, group, saving),
                heading=False,
            )


# --- Last action ----------------------------------------------------------------


def draw_last_action(ctx: Any, state: Any, tab: Any, doc: Any) -> None:
    """The op just run, its numbers still in reach -- while the model is as it left it.

    Gated by ``recent_op.RecentOp.live``, exactly as the card over the viewport was:
    any later edit, undo or selection change hides it, because "change the number"
    stops meaning "re-run that op" the moment something else has happened. The
    record is only made for a parameterised op that started in an element mode, so
    in practice this belongs to the element state alone; a record whose op has since
    been removed draws nothing.

    The same ``Param`` widgets the dialog and the inline forms draw
    (``menu.param_widget``). A change goes to ``recent_op.adjust`` -- undo the op's
    step, put its starting picks back, run it again -- so a drag over a field is
    one undo step, and a refusal keeps the last good result and prints its reason
    here (``ClayState.last_action_message``) rather than toasting per keystroke.
    """
    recent = doc.recent_op
    if recent is None or not recent.live(doc):
        state.last_action_message = ""
        state.last_action_open = False
        return
    try:
        op = clay_ops.get(recent.op_name)
    except KeyError:  # pragma: no cover - a record of an op since removed
        return
    widgets.section(f"Last action: {op.label.rstrip('.')}")
    if not state.last_action_open:
        # The op has just been run, and the card is the thing to do next -- but with the
        # All shapes dock open the body is short and the card sits below its fold. Once,
        # on the frame it appears (not on each adjust, which keeps it live), bring it
        # into view.
        imgui.set_scroll_here_y(0.25)
    state.last_action_open = True
    imgui.begin_disabled(bool(tab.saving))
    change: dict[str, float] = {}
    for param in op.params:
        changed, value = clay_menu.param_widget(
            f"last-{op.name}", param, recent.params.get(param.name, param.default)
        )
        if changed:
            change[param.name] = value
    imgui.end_disabled()
    if state.last_action_message:
        widgets.text_colored(theme.WARN, state.last_action_message)
    if change:
        result = recent_op.adjust(ctx, doc, **change)
        state.last_action_message = result.message
    widgets.end_section()
    imgui.dummy((0, sp(tokens.SP_2)))


def draw(
    ctx: Any, state: Any, tab: Any, doc: Any, between: Callable[[], None] | None = None
) -> None:
    """The whole element state under its title: Select row and element line in the
    title's block, then Shape it, Last action (right under the op buttons it belongs to)
    and the folds.

    *between* draws a card of the caller's between Last action and the folds -- the
    Inspector's Colour card, which is how a face is painted. Above the folds because
    they are closed disclosures nobody scrolls past on the way to a colour."""
    imgui.begin_disabled(bool(tab.saving))
    draw_select_row(ctx, state, tab, doc)
    imgui.dummy((0, sp(tokens.SP_1)))
    draw_element_line(ctx, state, doc)
    imgui.end_disabled()
    widgets.end_section()
    imgui.dummy((0, sp(tokens.SP_2)))
    imgui.begin_disabled(bool(tab.saving))
    draw_shape_it(ctx, state, tab, doc)
    imgui.end_disabled()
    draw_last_action(ctx, state, tab, doc)
    if between is not None:
        between()
    imgui.begin_disabled(bool(tab.saving))
    draw_folds(ctx, state, tab, doc)
    imgui.end_disabled()
