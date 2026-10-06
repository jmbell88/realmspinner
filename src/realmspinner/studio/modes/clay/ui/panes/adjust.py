"""The adjust card: the op you just ran, its numbers still in reach.

Drawn over the viewport's bottom-left corner while the document is **exactly as
the op left it** (``recent_op.RecentOp.live``): the same ``Param`` widgets the
dialog uses (``menu.param_widget``), and a change re-runs the op from the state
it started in (``recent_op.adjust``) -- one undo step, equal to having run the
op at the final value in the first place. Any later edit, undo or selection
change hides it, because "change the number" stops meaning "re-run that op"
the moment something else has happened.

A corner overlay rather than a pane for ``hud.axis_widget``'s reason: it is
something you reach for without looking away from the model you just changed.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ..... import theme, widgets
from .....tokens import sp
from ... import mode as clay_mode
from ... import ops as clay_ops
from ... import recent_op
from .menu import param_widget

#: Design pixels. The card is a fixed size for its op rather than auto-sized
#: (``begin_child`` has no portable auto-resize across the imgui versions this
#: app has shipped with); a parameter is a label line and a field line.
CARD_W = 232.0
HEAD_H = 34.0
PARAM_H = 58.0
WARN_H = 18.0
INSET = 8.0


def draw(ctx: Any, rect: tuple[float, float, float, float]) -> bool:
    """Draw the card if the recent op is live. -> whether the pointer is over it.

    The return value is ``hud.axis_widget``'s: the viewport records its hover off
    the render image, which cannot know a card was put on top of it since, so a
    press on the card would otherwise also reach the mesh behind it.
    """
    state = clay_mode.ensure(ctx)
    tab = state.active
    if tab is None:
        return False
    doc = tab.doc
    recent = doc.recent_op
    if recent is None or not recent.live(doc):
        state.adjust_message = ""
        return False
    try:
        op = clay_ops.get(recent.op_name)
    except KeyError:  # pragma: no cover - a record of an op since removed
        return False

    message = state.adjust_message
    height = sp(HEAD_H + PARAM_H * len(op.params) + (WARN_H if message else 0.0))
    imgui.set_cursor_screen_pos(
        (rect[0] + sp(INSET), rect[1] + rect[3] - sp(INSET) - height)
    )
    imgui.push_style_color(imgui.Col_.child_bg, theme.rgba(theme.ELEV_2, 0.94))
    hovered = False
    if imgui.begin_child(
        "##clay-adjust",
        (sp(CARD_W), height),
        imgui.ChildFlags_.borders.value | imgui.ChildFlags_.always_use_window_padding.value,
        imgui.WindowFlags_.no_scrollbar.value,
    ):
        widgets.field_label(f"Adjust {op.label.rstrip('.')}")
        imgui.begin_disabled(tab.saving)
        change: dict[str, float] = {}
        for param in op.params:
            changed, value = param_widget(
                f"adjust-{op.name}", param, recent.params.get(param.name, param.default)
            )
            if changed:
                change[param.name] = value
        imgui.end_disabled()
        if message:
            widgets.text_colored(theme.WARN, message)
        hovered = imgui.is_window_hovered(imgui.HoveredFlags_.root_and_child_windows.value)
        if change:
            result = recent_op.adjust(ctx, doc, **change)
            state.adjust_message = result.message
    imgui.end_child()
    imgui.pop_style_color()
    return hovered
