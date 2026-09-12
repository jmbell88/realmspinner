"""What Mason's viewport says without being asked: the hint line, and the
placed-count corner readout.

Chrome drawn *under* the viewport, ``clay_hud``'s own placement rule: a line
over the model competes with what you are looking at, and a line under it is
read when you are stuck. Muted, for the same reason -- it has to be legible
and must not compete with the render.

**No help button, on purpose.** This is chrome with no heading to hang one
beside -- ``clay_hud``'s own exemption, recorded here in
``tests/manual/test_coverage.py`` beside it for the identical reason: a (?)
on a corner readout would be a second help button over the one surface the
mode is about.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import mason_mode, theme, widgets
from ..tokens import sp

#: How tall the hint line is, for the viewport's reservation -- ``clay_hud``'s
#: own constant, matched so the two editors reserve the same strip.
HINT_H = 20.0

INSET = 8.0


def hint_line(ctx: Any) -> None:
    """One line of what the mouse and keyboard do right now."""
    state = mason_mode.ensure(ctx)
    tab = mason_mode.active(ctx)
    if tab is None:
        return
    line = _hint(state)
    widgets.muted(line)


def _hint(state: Any) -> str:
    """Pure: no imgui, no document -- the same split ``clay_hints`` makes, so
    "does the hint change with the tool" is a headless assertion rather than
    one that needs a GL context."""
    if state.place_kind:
        kind = state.place_kind.replace("_", " ")
        return f"Click in the viewport to place {kind} -- Esc to cancel"
    label = {
        "select": "Click to select -- Shift extends, Ctrl toggles",
        "move": "Drag to move the selection -- type a number, or X/Y/Z to lock an axis",
        "rotate": "Drag to rotate the selection about its pivot",
        "scale": "Drag to scale the selection about its pivot",
    }.get(state.tool, "")
    return label


def stats_overlay(ctx: Any, view: Any, rect: tuple[float, float, float, float]) -> None:
    """The placed count, and past the warning threshold, why it matters --
    read from ``mason.scene.PLACED_WARN_THRESHOLD`` through
    ``mason_mode.scene_stats`` rather than a literal, ``mason_bridge``'s own
    rule restated: the corner readout and the document pane's own warning
    must name the same number, and a copy of it here is a second place that
    can drift from the measurement it is about."""
    tab = mason_mode.active(ctx)
    if tab is None:
        return
    stats = mason_mode.scene_stats(ctx, tab)
    draw_list = imgui.get_window_draw_list()
    inset = sp(INSET)
    line = f"{stats['placed']:,} placed"
    if stats["missing"]:
        line += f"  -  {stats['missing']} missing"
    draw_list.add_text(
        (rect[0] + inset, rect[1] + inset),
        imgui.get_color_u32(theme.rgba(theme.MUTED, 0.9)),
        line,
    )
    if stats["warn"]:
        draw_list.add_text(
            (rect[0] + inset, rect[1] + inset + imgui.get_text_line_height_with_spacing()),
            imgui.get_color_u32(theme.rgba(theme.ACCENT, 0.9)),
            f"over {stats['threshold']:,} placed -- this scene may cost frame rate",
        )
    del view
