"""Properties' friendly transform controls (Blender-Lite plan, Phase A).

Rotation is Euler degrees in the app's one order, size is editable, and
position/size can be shown in another unit. Three claims need proving and none
needs a frame: a typed angle is not rewritten while it is being typed, a size
edit lands on the scale that reproduces it, and a unit is only ever a lens on
metres.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.geom3d import units
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import transform_edit as te
from realmspinner.studio.modes.clay.state import ClayState
from realmspinner.studio.modes.clay.ui.panes import props as clay_props


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _doc(mesh=None, **kwargs):
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box() if mesh is None else mesh, **kwargs)
    )
    return doc, obj


def _draw(ui, fn, *args, **kwargs) -> None:
    ui.new_frame()
    ui.begin("##host")
    try:
        fn(*args, **kwargs)
    finally:
        ui.end()
        ui.end_frame()


# --- the Euler helpers moved down a layer -------------------------------------


@pytest.mark.parametrize("angles", [(0, 0, 0), (30, -45, 80), (-170, 20, 10), (90, 0, 0)])
def test_euler_round_trips_through_math3d(angles) -> None:
    q = m3.quat_from_euler_xyz(angles)
    back = m3.euler_xyz_from_quat(q)
    q2 = m3.quat_from_euler_xyz(back)
    # The claim is about the rotation, not the three numbers: q and -q agree.
    assert abs(float(np.dot(q, q2))) == pytest.approx(1.0, abs=1e-9)


def test_the_agent_surface_uses_the_one_implementation() -> None:
    from realmspinner.studio.modes.clay.agent import validate

    assert validate._quat_from_euler_xyz is m3.quat_from_euler_xyz
    assert validate._euler_xyz_from_quat is m3.euler_xyz_from_quat


# --- a typed angle is not rewritten while it is being typed -------------------


def test_entering_190_degrees_is_not_rewritten_to_minus_170() -> None:
    cache: te.EulerCache = {}
    q = m3.quat_from_euler_xyz((0.0, 0.0, 190.0))
    # What the decomposition would read back -- the unstable half.
    assert m3.euler_xyz_from_quat(q)[2] == pytest.approx(-170.0)
    te.remember_euler(cache, 1, q, (0.0, 0.0, 190.0))
    # The next frame, same quaternion: the typed angles stay.
    assert te.displayed_euler(cache, 1, q)[2] == 190.0


def test_the_cache_refreshes_when_the_quaternion_moves_from_outside() -> None:
    cache: te.EulerCache = {}
    q = m3.quat_from_euler_xyz((0.0, 0.0, 190.0))
    te.remember_euler(cache, 1, q, (0.0, 0.0, 190.0))
    moved = m3.quat_from_euler_xyz((0.0, 0.0, 45.0))  # a gizmo drag, an undo
    assert te.displayed_euler(cache, 1, moved)[2] == pytest.approx(45.0)


def test_typing_a_rotation_through_the_panel_sets_the_quaternion_and_keeps_the_angle(
    ui, monkeypatch
) -> None:
    doc, obj = _doc()
    state = ClayState()
    shown: list[list[float]] = []

    def spy(label, values, axes, **kwargs):
        if label == "rotation (deg)##br":
            shown.append(list(values))
            if len(shown) == 1:
                return True, [0.0, 0.0, 190.0]
        return False, list(values)

    monkeypatch.setattr(clay_props.controls, "input_vec", spy)
    _draw(ui, clay_props._transform, doc, doc.by_uid(obj.uid), state=state)
    assert np.allclose(
        doc.by_uid(obj.uid).rotation, m3.quat_from_euler_xyz((0, 0, 190)), atol=1e-9
    )
    _draw(ui, clay_props._transform, doc, doc.by_uid(obj.uid), state=state)
    assert shown[1] == [0.0, 0.0, 190.0], "the second frame must show what was typed"


# --- units --------------------------------------------------------------------


@pytest.mark.parametrize("unit", [key for key, _ in units.LENGTH_UNITS])
def test_a_unit_conversion_round_trips(unit) -> None:
    for metres in (0.0, 0.125, 1.0, -3.5, 123.456):
        assert units.from_display(units.to_display(metres, unit), unit) == pytest.approx(metres)


def test_the_import_scale_combo_derives_from_the_unit_table() -> None:
    from realmspinner.studio.modes.clay.ui.panes import bridge

    assert [label for _, label in bridge.IMPORT_SCALE_OPTIONS] == [k for k, _ in units.LENGTH_UNITS]
    assert dict(bridge.IMPORT_SCALE_OPTIONS)["0.0254"] == "in"


def test_an_unknown_unit_reads_as_metres() -> None:
    assert units.to_display(2.0, "furlong") == 2.0


def test_a_position_typed_in_centimetres_is_stored_in_metres(ui, monkeypatch) -> None:
    doc, obj = _doc()
    state = ClayState(length_unit="cm")

    def spy(label, values, axes, **kwargs):
        if label == "position##bt":
            return True, [150.0, values[1], values[2]]
        return False, list(values)

    monkeypatch.setattr(clay_props.controls, "input_vec", spy)
    _draw(ui, clay_props._transform, doc, doc.by_uid(obj.uid), state=state)
    assert list(doc.by_uid(obj.uid).translation) == [1.5, 0.0, 0.0]


# --- size ---------------------------------------------------------------------


def test_a_size_edit_sets_the_scale_that_reproduces_it() -> None:
    extent = np.array([1.0, 2.0, 0.5])
    scale = np.array([1.0, 1.0, 1.0])
    out = te.resized_scale(extent, scale, 1, 5.0)
    assert out is not None
    assert te.size_of(extent, out)[1] == pytest.approx(5.0)
    assert list(out[[0, 2]]) == [1.0, 1.0], "the other axes do not move without the lock"


def test_lock_aspect_scales_the_other_axes_by_the_same_ratio() -> None:
    extent = np.array([1.0, 2.0, 0.5])
    scale = np.array([2.0, 1.0, 4.0])
    out = te.resized_scale(extent, scale, 1, 4.0, lock_aspect=True)  # 2 m -> 4 m: x2
    assert out is not None
    assert list(out) == pytest.approx([4.0, 2.0, 8.0])


def test_a_size_edit_keeps_a_negative_scales_sign() -> None:
    out = te.resized_scale(np.ones(3), np.array([-1.0, 1.0, 1.0]), 0, 3.0)
    assert out is not None and out[0] == -3.0


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_a_nonsense_size_is_a_no_op(bad) -> None:
    assert te.resized_scale(np.ones(3), np.ones(3), 0, bad) is None


def test_a_zero_extent_axis_is_refused_with_a_reason() -> None:
    extent = np.array([1.0, 0.0, 1.0])
    reason = te.axis_refusal(extent, 1)
    assert reason is not None and "height" in reason
    assert te.axis_refusal(extent, 0) is None
    assert te.resized_scale(extent, np.ones(3), 1, 2.0) is None


def test_editing_size_through_the_panel_is_one_undo_step_and_rescales(ui, monkeypatch) -> None:
    doc, obj = _doc(scale=np.array([2.0, 2.0, 2.0]))
    extent = te.local_extent(obj.mesh)
    state = ClayState(size_lock_aspect=True)
    steps = len(doc.history)

    def spy(label, values, axes, **kwargs):
        if label == "size##bz":
            return True, [values[0] * 2.0, values[1], values[2]]
        return False, list(values)

    monkeypatch.setattr(clay_props.controls, "input_vec", spy)
    _draw(ui, clay_props._dimensions, doc, doc.by_uid(obj.uid), state=state)
    new = doc.by_uid(obj.uid)
    assert list(new.scale) == pytest.approx([4.0, 4.0, 4.0]), "locked: every axis follows"
    assert te.size_of(extent, new.scale)[0] == pytest.approx(extent[0] * 4.0)
    assert len(doc.history) == steps + 1
    assert doc.undo()
    assert list(doc.by_uid(obj.uid).scale) == [2.0, 2.0, 2.0]


def test_editing_a_flat_axis_through_the_panel_changes_nothing(ui, monkeypatch) -> None:
    doc, obj = _doc(mesh=bp.plane())
    extent = te.local_extent(obj.mesh)
    axis = next(i for i in range(3) if extent[i] <= te.FLAT_EXTENT)
    state = ClayState()
    steps = len(doc.history)

    def spy(label, values, axes, **kwargs):
        if label == "size##bz":
            out = list(values)
            out[axis] = 3.0
            return True, out
        return False, list(values)

    monkeypatch.setattr(clay_props.controls, "input_vec", spy)
    _draw(ui, clay_props._dimensions, doc, doc.by_uid(obj.uid), state=state)
    assert len(doc.history) == steps
