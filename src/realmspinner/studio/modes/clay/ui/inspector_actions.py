"""The inspector's "More actions": every object-level op, drawn from the registry.

**Generated, never hand-listed.** The groups come from ``menutree.BARS["object"]``
(the same table the right-click menu reads, so the two cannot disagree about what
Shape mode offers) and each button from an ``ops.Op``: its label, its hint and key
as the tooltip, and -- when ``op.enabled(doc)`` says no -- greyed with the plain
sentence ``ops.reason_for`` gives. An op registered tomorrow is a button here the
day it lands in a ``menutree`` group; ``tests/modes/clay/test_inspector.py`` checks
the two directions.

What is *not* here is on purpose: Duplicate and Delete have a row of their own
under the cards (``PRIMARY``), and the Select group is Ctrl+A / Ctrl+D / Ctrl+Shift+I
and the Edit menu's, not a panel's.

**An op with numbers opens its fields inline** -- :func:`menu.param_widget`, the
widget the right-click dialog and the adjust card draw, so a checkbox is a
checkbox and a choice a combo in all three -- with an Apply under them. The values
are remembered per op on ``ClayState.op_params``, the dict the dialog reads and
writes, so a width typed here is the width the dialog opens on. The popup door
(``mode.fire_op``) stays the keyboard's and the menus'; a click in this panel does
not open a dialog over the panel it is beside.

**A closed disclosure costs nothing.** ``op.enabled`` is a predicate over the
document -- several walk every object -- and the panel is drawn every frame, so
:func:`status` is only called from inside an open More actions, and memoises on the
document's ``rev`` (a selection change touches it) so an open one pays once per
change rather than once per frame. ``test_closed_disclosures_do_not_evaluate_op_enabled``
spies on ``Op.enabled`` to hold that.
"""

from __future__ import annotations

import weakref
from typing import Any

from imgui_bundle import imgui

from .... import controls, icons, widgets
from .. import menutree as clay_menutree
from .. import mode as clay_mode
from .. import ops as clay_ops
from .panes import menu as clay_menu

#: The ops with a row of their own under the cards, so they are not repeated here.
PRIMARY = frozenset({"duplicate", "delete"})

#: Groups the panel leaves to the keyboard and the menus (see the docstring).
SKIPPED_GROUPS = frozenset({"select"})

#: The group whose heading also carries the "Texture mapping" link to the Unwrap
#: page -- the panel's own door to the UV canvas, which no op opens.
UV_GROUP = "uv"

_SAVING = "This document is being written; the actions come back when it lands."


def action_groups(mode: str = "object") -> list[tuple[str, str, tuple[clay_ops.Op, ...]]]:
    """``[(group key, heading, ops)]`` for *mode*, in the right-click menu's order.

    Pure and imgui-free, so "generated from the registry" is a claim a test can
    check without a frame. Empty groups (everything in them is in :data:`PRIMARY`
    or another mode's) are dropped, and an op named by two bar entries appears
    once, at its first.
    """
    seen: set[str] = set(PRIMARY)
    out: list[tuple[str, str, tuple[clay_ops.Op, ...]]] = []
    for menu in clay_menutree.BARS[mode]:
        for entry in menu.entries:
            if entry.kind == "add" or entry.name in SKIPPED_GROUPS:
                continue
            ops = tuple(
                op for op in clay_menutree.grouped_ops(entry.name, mode) if op.name not in seen
            )
            if not ops:
                continue
            seen.update(op.name for op in ops)
            out.append((entry.name, clay_menutree.GROUPS[entry.name].label, ops))
    return out


#: ``doc -> ((rev, element mode, saving), {op name: reason})``. Weak, so a closed
#: document takes its entry with it (``props._MATERIAL_USERS_CACHE``'s own shape).
_STATUS: weakref.WeakKeyDictionary[Any, tuple[tuple[Any, ...], dict[str, str]]] = (
    weakref.WeakKeyDictionary()
)


def status(doc: Any, ops: tuple[clay_ops.Op, ...], saving: bool) -> dict[str, str]:
    """``{op name: reason}`` -- ``""`` for an op that can run right now.

    Memoised on ``doc.rev``: every change that could flip an op's gate (a
    selection, an edit, an undo) touches it. A saving document greys every op for
    the one shared reason, so no predicate is asked at all.
    """
    if saving:
        return {op.name: _SAVING for op in ops}
    key = (getattr(doc, "rev", None), doc.element_mode, False)
    cached = _STATUS.get(doc)
    if cached is not None and cached[0] == key and key[0] is not None:
        known = cached[1]
        if all(op.name in known for op in ops):
            return known
    reasons: dict[str, str] = {}
    for op in ops:
        enabled = op.enabled(doc)
        reasons[op.name] = "" if enabled else (clay_ops.reason_for(op, doc) or "Not available.")
    _STATUS[doc] = (key, reasons)
    return reasons


def tooltip_for(op: clay_ops.Op) -> str:
    """What hovering the button says: the op's own sentence and its key."""
    text = op.hint or op.label.rstrip(".")
    return f"{text}  ({op.key})" if op.key else text


def draw(ctx: Any, state: Any, tab: Any, doc: Any) -> None:
    """Every group's ops as wrapped buttons, a form under the one being filled in.

    Meant to be called only from inside an open disclosure; it does no gating of
    its own about *whether* to draw, only about whether each button may be pressed.
    """
    groups = action_groups("object")
    every = tuple(op for _key, _label, ops in groups for op in ops)
    reasons = status(doc, every, bool(tab.saving))
    for key, label, ops in groups:
        widgets.field_label(label)
        first = True
        for op in ops:
            title = f"{op.label}##insp-op/{op.name}"
            if not first:
                widgets.same_line_or_wrap(widgets.button_width(title))
            first = False
            reason = reasons.get(op.name, "")
            opened = state.inspector_form == op.name
            if widgets.disabled_button(
                title,
                not reason,
                reason=reason,
                tooltip=tooltip_for(op),
            ):
                if op.params:
                    state.inspector_form = "" if opened else op.name
                    state.op_params.setdefault(op.name, clay_ops.defaults_for(op))
                else:
                    clay_mode.fire_op(ctx, doc, op)
        if key == UV_GROUP:
            page_label = f"Texture mapping {icons.CHEVRON_RIGHT}##insp-uv-page"
            widgets.same_line_or_wrap(widgets.button_width(page_label))
            if controls.button(
                page_label,
                tooltip="Lay the faces out flat on the picture sheet: islands, "
                "move, turn, scale, pack.",
            ):
                state.inspector_page = "unwrap"
        for op in ops:
            if state.inspector_form == op.name:
                _form(ctx, state, tab, doc, op, reasons.get(op.name, ""))


def _form(ctx: Any, state: Any, tab: Any, doc: Any, op: clay_ops.Op, reason: str) -> None:
    """*op*'s numbers and an Apply, inline under its group.

    Closed by itself when the op stops being runnable (the selection moved on):
    a form whose Apply would be greyed on arrival is a dead end with a heading.
    """
    if reason:
        state.inspector_form = ""
        return
    widgets.muted_wrapped(op.hint or op.label)
    values = state.op_params.setdefault(op.name, clay_ops.defaults_for(op))
    for param in op.params:
        current = values.get(param.name, param.default)
        changed, value = clay_menu.param_widget(op.name, param, current)
        if changed:
            values[param.name] = value
    if widgets.primary_button(
        f"Apply##insp-apply/{op.name}",
        enabled=not tab.saving,
        reason=widgets.DOCUMENT_SAVING_WHY,
    ):
        clay_ops.run(ctx, doc, op, **values)
        state.inspector_form = ""
    imgui.same_line()
    if controls.button(f"Cancel##insp-cancel/{op.name}"):
        state.inspector_form = ""
