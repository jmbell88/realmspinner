"""Packwright's right-top pane: what the last pack actually produced.

A pack that could not fit says so *here* rather than showing an empty list,
which is the one outcome that reads as success when it is the opposite. The
error text comes from the engine -- ``layout`` raises with the number and the
remedy -- and is carried through ``PackTab.pack_error`` by the mode's
``on_task_failed``.

The selection is the same ``PackwrightState.selected`` the sources list and the
preview use: one answer to "which sprite are we talking about", rather than
three that have to be kept in step.
"""

from __future__ import annotations

from typing import Any

from .. import controls, icons, packwright_mode, theme, tokens, widgets
from ..manual import render as manual_render
from ..tokens import sp


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = packwright_mode.ensure(ctx)
    tab = state.active
    widgets.section("Packed")
    manual_render.help_button(ctx, "packwright-items")

    if tab is None:
        # The heading and nothing else. One voice for one empty state:
        # the canvas's ``nothing_open`` is it, and four panels each
        # repeating it reads as four separate problems.
        return

    if tab.pack_error:
        widgets.text_colored(theme.ERR, f"{icons.TRIANGLE_ALERT} {tab.pack_error}")
        imgui.dummy((0, sp(tokens.SP_1)))
        widgets.muted_wrapped(
            "Raise the max size, turn trimming on, or split this into two atlases."
        )
        return

    layout = tab.layout
    if layout is None:
        widgets.muted(
            "Packing..." if tab.packing else "Add a sprite -- packing runs by itself."
        )
        return

    widgets.muted(
        f"{len(layout.frames)} sprite(s) in {layout.width} x {layout.height} px "
        f"-- {_coverage_pct(tab)}% covered"
    )
    if layout.is_grid:
        widgets.muted(
            f"grid: {layout.columns} x {layout.rows} cells of "
            f"{layout.cell_w} x {layout.cell_h}"
        )
    if tab.packing:
        widgets.busy("Repacking")

    imgui.dummy((0, sp(tokens.SP_2)))
    # ``packwright_mode.source_index``: the 2026-09-07 audit's packwright-07
    # found this dict rebuilt from every source on every single frame this
    # pane draws, though it only has to change when ``pack_generation`` does
    # -- the same counter the preview's outline keys on.
    by_key = packwright_mode.source_index(tab)
    # Clipped for ``packwright_sources``' reason: a packed atlas has one row per
    # sprite and a thousand-sprite atlas is ordinary, while the pane shows
    # twenty. Every row here *is* the same height, so this is the
    # straightforward case.
    clipper = imgui.ListClipper()
    clipper.begin(len(layout.frames))
    while clipper.step():
        for index in range(clipper.display_start, clipper.display_end):
            _item_row(state, layout.frames[index], by_key)
    clipper.end()


def _coverage_pct(tab: Any) -> int:
    """The "-- N% covered" figure, cached on the tab.

    The 2026-09-16 audit (packwright-08) found this summed over every packed
    frame on every single frame this pane draws, with no memoisation keyed on
    ``pack_generation`` -- the same shape ``packwright_mode.source_index``
    (packwright-07, 2026-09-07) was fixed for. An atlas near ``MAX_SPRITES``
    paid a full Python-level sum sixty times a second merely by having this
    pane on screen. ``getattr`` on both ends rather than a hard read, so a
    minimal test double with no ``pack_generation`` just never hits the cache
    rather than raising.
    """
    layout = tab.layout
    generation = getattr(tab, "pack_generation", None)
    cached = getattr(tab, "_pw_coverage_pct", None)
    if cached is not None and cached[0] == generation and generation is not None:
        return cached[1]
    used = sum(frame.w * frame.h for frame in layout.frames)
    total = max(layout.width * layout.height, 1)
    pct = 100 * used // total
    tab._pw_coverage_pct = (generation, pct)
    return pct


def _item_row(state: Any, frame: Any, by_key: dict) -> None:
    """One packed frame's row. Lifted out of the loop so it can be clipped."""
    from imgui_bundle import imgui

    uid = by_key.get(frame.key)
    selected = uid is not None and state.selected == uid
    imgui.push_id(frame.key)
    if controls.selectable(f"{frame.name}##item", selected)[0] and uid is not None:
        state.selected = None if selected else uid
    if imgui.is_item_hovered():
        trimmed = " (trimmed)" if frame.trimmed else ""
        imgui.set_tooltip(
            f"{frame.w} x {frame.h} at {frame.x}, {frame.y}{trimmed}\n"
            f"source {frame.source_w} x {frame.source_h}"
        )
    if frame.empty:
        imgui.same_line()
        widgets.muted("(blank)")
    imgui.pop_id()
