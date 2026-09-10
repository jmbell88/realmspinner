"""The uniform icon grid, for the panel-grammar pass Clay's Tools panel piloted
(2026-09-08).

Clay's own panel had drawn three different grammars stacked in one sidebar --
primitives as an unlabelled icon grid, figures as one full-width text button
per row, and the ops as a ragged two-column grid with a hand-rolled Delete --
for the same reason every one of them existed: each grew from its own call
site rather than from one idea of what "pick a tool" looks like. Inker's
toolbox and Plotter's tool rail are the two other known callers of that same
idea; neither is migrated here, because each already carries its own gesture
on top of its grid (Inker's press-and-slide flyout groups in particular), and
adopting this is a decision for whoever owns those files next, not a rewrite
bundled with Clay's own redesign.

Kept to the one piece that is genuinely generic across all three: equal-sized,
tooltip-named buttons wrapped at a fixed column count, with a visible selected
state and a width asked from :func:`widgets.grid_width` every frame rather
than assumed. See that function's own docstring for the incident an assumed
width caused -- an unscaled ``8`` px gap literal, right at UI scale 1.0 and
short by 4.8px per gap at 1.5x, which is exactly what cost Clay's own tool row
its fourth button and the Inker toolbox its fifth column.

Everything past "which key was clicked" -- what a grid's selection means,
what runs on a click, what options a selection shows -- stays the caller's:
that answer is different for every one of the three palettes, and folding it
in here would be the same "one grammar drawn three ragged ways" mistake this
module exists to end, just moved into the shared file instead of out of it.
"""

from __future__ import annotations

from collections.abc import Sequence

from imgui_bundle import imgui

from . import controls, widgets
from .tokens import sp


def icon_grid(
    items: Sequence[tuple[str, str, str]],
    columns: int,
    selected: str | None,
    *,
    id_prefix: str,
    height: float = 28.0,
) -> str | None:
    """Equal-sized icon buttons wrapped at ``columns`` across.

    ``items`` is ``(key, icon, tooltip)``. -> the key clicked this frame, or
    ``None`` -- at most one button can report a click in one frame, so the
    caller never has to pick between two.

    The width is asked fresh from :func:`widgets.grid_width` on every call,
    never cached: both the sidebar's own drag and the UI scale slider change
    what it answers, and a grid that measured itself once at startup is a
    grid that is wrong the first time either moves.
    """
    width = widgets.grid_width(columns)
    clicked: str | None = None
    for index, (key, icon, tooltip) in enumerate(items):
        if controls.button(
            f"{icon}##{id_prefix}{key}", (width, sp(height)), selected=selected == key
        ):
            clicked = key
        if tooltip and imgui.is_item_hovered():
            imgui.set_tooltip(tooltip)
        if index % columns != columns - 1:
            imgui.same_line()
    imgui.new_line()
    return clicked
