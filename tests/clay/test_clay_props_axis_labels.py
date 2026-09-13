"""Clay's transform panel names each box it draws (2026-09-12 pass).

``_transform`` used to hand ``input_float3``/``input_float4`` a bare list of
numbers with nothing over the boxes to say which was X, which was Y, and
(for rotation) which was the quaternion's W -- a user had to already know the
order to read it. The fix routes every vector field through
``controls.input_vec``, which draws one axis letter per box; the user's own
call on rotation was to keep it a quaternion rather than switch to Euler, so
its fourth letter is W, not a fourth XYZ triple or a dropped component.

Driven through ``controls.input_vec`` itself (monkeypatched to record what it
was called with) rather than a live frame's draw output, because it is the
*axes tuple each field is labelled with* that is the claim -- the same reason
``test_clay_props_undo.py`` reads ``_transform``'s source rather than pressing
a real field.
"""

from __future__ import annotations

import pytest
from _ui_context import imgui_context

from warlock.studio.clay import document as bd
from warlock.studio.clay import primitives as bp
from warlock.studio.panes import clay_props


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


def test_transform_labels_position_and_scale_xyz_and_rotation_xyzw(ui, monkeypatch) -> None:
    """Fails against the pre-fix ``_transform``, which calls
    ``controls.input_float3``/``input_float4`` directly -- ``controls.input_vec``
    is never called at all, so ``calls`` comes back empty."""
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
    # The load-bearing assertion: rotation keeps its fourth component, and it
    # is named W (a quaternion), not Z-again or dropped to a triple.
    assert calls["rotation##br"] == ("X", "Y", "Z", "W")
