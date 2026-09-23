"""Regression for the 2026-09-23 audit, finding create-08.

``stage_rail`` draws each segment with a raw ``imgui.invisible_button`` and
never calls ``probe.record`` -- the same shell-06 gap ``studio/rail.py`` fixed
for the shell's own rail, left open here because Create's breadcrumb is a
separate module.
"""

from __future__ import annotations

from _ui_context import imgui_context

from realmspinner.studio import probe
from realmspinner.studio.modes.create.ui import rail as create_rail


def test_create_stage_rail_items_are_visible_to_the_control_probe(monkeypatch):
    items = [
        ("reference", "Reference", "R", None),
        ("mesh", "Mesh", "M", None),
    ]
    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.begin("test")
        probe.begin_frame()
        create_rail.stage_rail("audit-2026-09-23/rail", items, "reference")
        imgui.end()
        imgui.render()

    assert probe.FRAME_CONTROLS, (
        "stage_rail's segments must call probe.record -- neither a raw "
        "control census entry nor a probe record existed before this fix"
    )
    recorded = [c for c in probe.FRAME_CONTROLS if c.kind == "rail_item"]
    assert recorded, "stage_rail must record its segments as rail_item controls"
