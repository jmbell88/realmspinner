"""The row across the top of Mason's viewport: how the scene is drawn.

``clay_header``'s argument, one dimension over: shading, the grid, the wire
overlay, X-ray and the pivot are settings changed *between* clicks in the
viewport, not while looking at a sidebar on the far side of the window from
the model -- so they live in a strip over the render rather than in the Tools
pane, which keeps what actually wants a sidebar's height (the placement ops)
uncrowded by four switches flicked once a session.

**No mode field and no tool field here**, unlike Clay's header -- Mason's
transform tool has no per-element mode to pair it with (there is no vertex,
edge or face here, only nodes), so a single field would sit alone where
Clay's bar has two. It stays on ``mason_tools`` instead: Clay's own header
docstring reserves this bar for settings that change *between clicks*, and a
scene editor's transform tool is exactly that, so ``TOOL_ICONS``/``TOOLS`` are
drawn once, on the sidebar, rather than duplicated onto a bar that would have
to agree with it every frame.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ..... import controls, toolbar, widgets
from .....manual import render as manual_render
from ... import mode as mason_mode
from ... import state as mason_state

BAR = "mason-header"
HEADER_H = 34.0

#: Solid / Material / Wire, the flat albedo, the lit render, and the edges
#: alone -- ``clay_header.SHADING``'s own three, restated for the identical
#: reason a scene editor wants them: Blender's own order, glyph-free because
#: lucide has nothing for a shading mode and ``icons.py`` forbids inventing a
#: codepoint.
SHADING: tuple[tuple[str, str, str], ...] = (
    ("solid", "Solid", "S"),
    ("material", "Material", "M"),
    ("wireframe", "Wire", "W"),
)


def draw(ctx: Any, view: Any = None) -> None:
    state = mason_mode.ensure(ctx)
    tab = mason_mode.active(ctx)
    if tab is None:
        return
    toolbar.toolbar(BAR, [], fields=[_pivot_field(state)], trailing=_trailing(ctx, state, view))
    widgets.divider()


def _pivot_field(state: Any) -> Any:
    def draw_it(_compact: bool) -> None:
        options = [(key, key.title()) for key in mason_state.PIVOTS]
        changed, picked = controls.segmented_choice(
            "mason-header-pivot", options, state.pivot, compact=True
        )
        if changed:
            state.pivot = picked

    return toolbar.Field("pivot", "Pivot", draw_it, width=140.0, compact=140.0)


def _trailing(ctx: Any, state: Any, view: Any) -> Any:
    style_gap = 6.0

    def draw_it() -> None:
        changed, picked = controls.segmented_choice(
            "mason-shading",
            [(key, short) for key, _label, short in SHADING],
            state.shading,
            tooltips={key: label for key, label, _short in SHADING},
            compact=True,
        )
        if changed:
            state.shading = picked
        imgui.same_line()
        changed, value = widgets.toggle("Grid", state.grid, tag="mason-grid")
        if changed:
            state.grid = value
        imgui.same_line()
        changed, value = widgets.toggle("Wire", state.overlays.get("wire", False), tag="mason-wire")
        if changed:
            state.overlays["wire"] = value
        imgui.same_line()
        if controls.button(
            f"X-ray##{BAR}/xray",
            role=controls.ButtonRole.GHOST,
            control_size=controls.ControlSize.COMPACT,
            selected=bool(state.xray),
            tooltip="See through the surface, so a node behind it can be picked",
        ):
            state.xray = not state.xray
        imgui.same_line()
        manual_render.help_button_inline(ctx, "mason-header")

    width = sum(
        imgui.calc_text_size(short).x + 16.0 for _key, _label, short in SHADING
    ) + imgui.calc_text_size("Grid").x + imgui.calc_text_size("Wire").x + imgui.calc_text_size(
        "X-ray"
    ).x + 80.0 + imgui.get_frame_height()
    del style_gap, view
    return (width, draw_it)
