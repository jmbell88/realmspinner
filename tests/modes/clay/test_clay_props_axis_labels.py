"""Clay's transform panel names each box it draws (2026-09-12 pass).

``_transform`` used to hand ``input_float3``/``input_float4`` a bare list of
numbers with nothing over the boxes to say which was X, which was Y, and
(for rotation) which was the quaternion's W -- a user had to already know the
order to read it. The fix routes every vector field through
``controls.input_vec``, which draws one axis letter per box; the user's own
call on rotation was then to keep it a quaternion. **Reversed with the
Blender-Lite plan's Phase A**: the app now has one rotation order, the MCP
surface's Euler XYZ, so rotation is three degrees (``rotation (deg)``) and the
panel shows what an agent's ``clay_transform`` takes.

Driven through ``controls.input_vec`` itself (monkeypatched to record what it
was called with) rather than a live frame's draw output, because it is the
*axes tuple each field is labelled with* that is the claim -- the same reason
``test_clay_props_undo.py`` reads ``_transform``'s source rather than pressing
a real field.
"""

from __future__ import annotations

import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay.ui.panes import props as clay_props


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _spy_input_vec(monkeypatch):
    calls: dict[str, tuple] = {}

    def spy(label, values, axes, **kwargs):
        calls[label] = tuple(axes)
        return False, list(values)

    monkeypatch.setattr(clay_props.controls, "input_vec", spy)
    return calls


def test_transform_labels_position_scale_and_rotation_xyz_in_degrees(ui, monkeypatch) -> None:
    """Fails against the quaternion row: ``rotation##br`` was XYZW and carried no
    unit, so neither the label nor the three-letter axes below existed."""
    calls = _spy_input_vec(monkeypatch)
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))

    ui.new_frame()
    ui.begin("##host")
    try:
        clay_props._transform(doc, obj)
    finally:
        ui.end()
        ui.end_frame()

    assert calls["position##bt"] == ("X", "Y", "Z")
    assert calls["scale##bs"] == ("X", "Y", "Z")
    # The load-bearing assertion: rotation is three Euler degrees, not a
    # quaternion with a fourth W.
    assert calls["rotation (deg)##br"] == ("X", "Y", "Z")
    assert "rotation##br" not in calls
