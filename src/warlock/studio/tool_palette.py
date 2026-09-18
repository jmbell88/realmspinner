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

from . import controls, icons, widgets
from .tokens import sp

# One icon per mesh primitive generator: the items Clay's Tools pane and
# Mason's palette both hand :func:`icon_grid`. It lived in Clay's pane until
# restructure P6, and Mason's palette importing it from there was a
# sibling-mode reach (``tests/test_layering.py``). Not in ``icons.py``, whose
# every upper-case name is a glyph string and is swept as one.
#
# One icon per generator in the registry. Strict at test time, graceful at
# runtime -- the same pair Clay's ``_sections`` states for ``CATEGORIES``:
# ``tests/modes/clay/test_clay_wiring.py`` holds this table *bijective* against
# ``primitives.GENERATORS``, so a sixteenth shape is a red test here rather
# than a glyph nobody chose, while ``PRIMITIVE_ICONS.get(name, icons.BOX)``
# at the draw site still gives that shape a button on the day it is written.
# This comment used to promise only the second half, and read as though
# forgetting a row here were free; it is not, and the pin is why.
PRIMITIVE_ICONS = {
    "box": icons.BOX,
    "plane": icons.RECTANGLE,
    "cylinder": icons.CROP,
    "cone": icons.TRIANGLE_ALERT,
    "uv_sphere": icons.CIRCLE,
    "torus": icons.CIRCLE,
    "grid": icons.GRID,
    "capsule": icons.EGG,
    "icosphere": icons.STAR,
    # The first four structures. The icon set is strained by now -- ``cone`` borrows
    # triangle-alert, and ``uv_sphere`` and ``torus`` are both a circle -- so
    # these are the nearest silhouettes rather than the right glyphs: a magnet
    # is a horseshoe, which is the arch, and a ruler is the tallest thing in
    # the set. ``lathe`` gets the spline glyph -- a profile revolved about an
    # axis is quite literally a spline, and it is otherwise unclaimed. The
    # tooltip carries the name.
    "pyramid": icons.PENTAGON,
    "arch": icons.MAGNET,
    "column": icons.RULER,
    "lathe": icons.SPLINE,
    # A sweep is an extrusion of stacked cross-sections, and layers is the
    # nearest silhouette this set has for that -- the same "strained by now"
    # trade-off the comment above already makes for the rest of this group.
    "sweep": icons.LAYERS,
    # Waypoints along a route is the nearest silhouette to a path with rings
    # threaded along it, and it is otherwise unclaimed here -- Inker's own
    # polyline tool uses the same glyph, which is fine: the two panes are
    # never on screen at once, and this set is strained enough already.
    "tube": icons.WAYPOINTS,
}



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
