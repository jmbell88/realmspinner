"""The 2026-10-10 audit's clay-28, the half that lives in the UV pane itself.

``material_for_pane`` learned to follow the tab's edit slot, but the pane has to
hand it the tab: without that the Ctrl+clicked slot is still not the one drawn
behind the islands, and the Inspector note says the UV view shows another slot.
"""

from __future__ import annotations

from typing import Any

import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import state as clay_state
from realmspinner.studio.modes.clay.ui import _uv_texture
from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def test_the_uv_pane_draws_the_slot_a_ctrl_click_chose_when_no_face_is_selected(
    ui, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    doc.add_material()
    doc.add_material()
    doc.paint_faces(obj.uid, list(range(len(obj.mesh.starts) - 1)), 2)
    doc.add_texture(1, 32)
    doc.add_texture(2, 32)
    obj = doc.by_uid(obj.uid)

    tab = clay_state.ClayTab.__new__(clay_state.ClayTab)
    tab.edit_slots = {}
    tab.edit_bases = {}
    tab.uid = "bd1"
    tab.saving = False
    tab.doc = doc
    view_state = clay_uv.UvPaneState(for_uid=obj.uid)
    tab.uv_view = view_state
    tab.choose_edit_slot(doc, obj, 1)
    assert tab.edit_slot(doc, obj) == 1, "the Ctrl+click took"

    drawn: list[Any] = []
    real = _uv_texture.pane_texture

    def spy(ctx: Any, uid: str, material: Any) -> Any:
        drawn.append(material)
        return real(ctx, uid, material)

    monkeypatch.setattr(clay_uv._uv_texture, "pane_texture", spy)

    ui.new_frame()
    ui.begin("##host")
    try:
        clay_uv._canvas(ctx=None, tab=tab, doc=doc, obj=obj, view_state=view_state)
    finally:
        ui.end()
        ui.end_frame()

    assert drawn, "the canvas resolved a texture this frame"
    assert drawn[-1] is doc.materials[1], (
        "the pane follows the slot the page edits, not the one every face wears"
    )
