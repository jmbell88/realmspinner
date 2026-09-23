"""Regressions for the 2026-09-23 audit, findings shell-05, shell-06, shell-11.

Three unrelated shell-widget gaps, closed together because each owns the same
small set of files (``toolbar.py``, ``widgets.py``, ``rail.py``,
``tool_palette.py``, ``probe.py``) and the audit's fixer pass keeps one owner
per file rather than three overlapping patches.
"""

from __future__ import annotations

from _ui_context import imgui_context

from realmspinner.studio import controls, probe, rail, tool_palette, widgets


def test_an_icon_tier_toolbar_item_still_shows_its_disabled_reason(monkeypatch):
    """shell-05: a reasoned item collapsed to ICON tier used to drop its
    reason on the floor -- ``icon_button`` took no ``reason`` parameter at
    all, so a toolbar item disabled with an explanation (Inker's "Delete
    frame") became, once compacted to a glyph, indistinguishable from one
    disabled for no reason. ``toolbar.py`` never called ``probe.record`` for
    the ICON tier either, so the disabled reason had nowhere to land even if
    it had been threaded through.
    """
    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.begin("test")
        probe.begin_frame()
        widgets.icon_button(
            f"{chr(0xE000)}##toolbar/delete-frame",
            "Delete frame",
            enabled=False,
            reason="No frame to delete.",
        )
        imgui.end()
        imgui.render()

    recorded = [c for c in probe.FRAME_CONTROLS if "delete-frame" in c.label]
    assert recorded, "icon_button must hand its press to the control census"
    assert recorded[0].reason == "No frame to delete."
    assert recorded[0].enabled is False


def test_rail_items_are_visible_to_the_control_probe_or_counted_as_a_pinned_blind_spot(
    monkeypatch,
):
    """shell-06: a rail row is drawn with a raw ``imgui.invisible_button`` that
    records nothing -- absent from the per-frame census *and* uncounted by
    ``test_probe.RAW_IMGUI_CONTROLS``'s pinned blind-spot scan (that scan
    only watches the widget names in ``_RAW_WIDGETS``, which does not
    include ``invisible_button``). A driver pressing every control the probe
    can see never presses a mode's own rail entry.
    """
    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.begin("test")
        probe.begin_frame()
        rail._item(
            "inker", "Inker", "I", 48.0, selected=False, tooltip="Pixel art"
        )
        imgui.end()
        imgui.render()

    assert probe.FRAME_CONTROLS, (
        "a rail row must either call probe.record or be an accounted-for "
        "raw control -- neither was true before this fix"
    )


def test_icon_grid_does_not_add_a_blank_line_after_a_full_last_row(monkeypatch):
    """shell-11: ``icon_grid`` calls ``imgui.new_line()`` unconditionally
    after its loop. When the last row is full (``len(items)`` is a multiple
    of ``columns``), the final item's own ``same_line`` was never called, so
    the cursor already sits on the next line by the time the loop ends --
    and the unconditional ``new_line()`` then opens a second, blank one
    beneath it. A grid whose item count isn't a clean multiple of the column
    count never shows the gap, which is why it survived unnoticed.
    """
    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.begin("test")
        width = widgets.grid_width(2)
        y0 = imgui.get_cursor_pos().y
        controls.button("ref##audit-2026-09-23/btn", (width, 28.0))
        y_single_row_advance = imgui.get_cursor_pos().y - y0

        imgui.dummy((1.0, 20.0))
        y1 = imgui.get_cursor_pos().y
        tool_palette.icon_grid(
            [("a", "A", "A"), ("b", "B", "B")],
            2,
            None,
            id_prefix="audit-2026-09-23/grid/",
        )
        y_grid_advance = imgui.get_cursor_pos().y - y1
        imgui.end()
        imgui.render()

    assert y_grid_advance == y_single_row_advance, (
        "a full last row (2 items, 2 columns) must advance the cursor by "
        "exactly one row, not one row plus a blank new_line()"
    )
