"""The curve editor (Blender-Lite plan, Phase D, editor half).

``curve_edit`` decides what a press, a drag and a release do; ``curve_editor``
sends the result through the generic generator door. So the claims are:

* the model's edits keep the curve's **shape** (insert) and its **alignment**
  (delete takes the handle row with the point);
* a **drag, an insert and a delete are each one undo step**, and the mesh follows
  the curve live (every frame of a drag rebuilds it);
* a figure-eight outline is **found**, a simple one is not;
* after a drag the **stored, clamped** params are what is read back;
* the pane **draws** for all three generators without raising.
"""

from __future__ import annotations

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import curve_edit as ce
from realmspinner.studio.modes.clay.state import ClayState
from realmspinner.studio.modes.clay.ui.panes import curve_editor
from realmspinner.studio.modes.clay.ui.panes import props as clay_props

AXES = (0, 1)


# --- the model ------------------------------------------------------------------------


def _wave() -> ce.Curve:
    return ce.Curve.from_params(
        [[0.0, 0.0], [1.0, 0.0]],
        [[[0, 0], [0.0, 0.5]], [[0.0, 0.5], [0, 0]]],
    )


def test_inserting_a_point_does_not_change_the_curves_shape() -> None:
    curve = _wave()
    before = np.array(curve.flat(tol=1e-4))
    after = ce.insert_point(curve, 0, 0.37)
    assert len(after.points) == 3
    flat = np.array(after.flat(tol=1e-4))
    # Every point of the old curve still lies on the new one, and vice versa.
    for point in before[:: max(len(before) // 40, 1)]:
        assert min(np.linalg.norm(flat - point, axis=1)) < 2e-3
    for point in flat[:: max(len(flat) // 40, 1)]:
        assert min(np.linalg.norm(before - point, axis=1)) < 2e-3


def test_inserting_into_a_straight_segment_leaves_two_straight_ones() -> None:
    straight = ce.Curve.from_params([[0.0, 0.0], [2.0, 0.0]], [])
    out = ce.insert_point(straight, 0, 0.25)
    assert out.points[1] == [0.5, 0.0], "a straight segment splits at the linear position"
    assert out.stored_handles() == [], "and stays straight: no handles appear"


def test_inserting_into_a_closed_curves_wrap_segment_appends() -> None:
    square = ce.Curve.from_params([[0, 0], [1, 0], [1, 1], [0, 1]], [], closed=True)
    out = ce.insert_point(square, 3, 0.5)  # the segment from the last point back to the first
    assert len(out.points) == 5 and out.points[-1] == [0.0, 0.5]


def test_deleting_a_point_takes_its_handles_with_it() -> None:
    curve = ce.Curve.from_params(
        [[0, 0], [1, 1], [2, 0]],
        [[[0, 0], [0, 0.1]], [[0.2, 0], [0.3, 0]], [[0, -0.1], [0, 0]]],
    )
    out = ce.delete_point(curve, 1)
    assert out.points == [[0.0, 0.0], [2.0, 0.0]]
    assert len(out.handles) == 2 and out.handles[0] == [[0.0, 0.0], [0.0, 0.1]]


def test_a_symmetric_pull_makes_a_smooth_pair() -> None:
    curve = ce.Curve.from_params([[0, 0], [1, 0]], [])
    out = ce.pull_symmetric(curve, 0, [0.3, 0.2], AXES)
    assert out.handles[0] == [[-0.3, -0.2], [0.3, 0.2]]


def test_a_handle_drag_in_a_plane_leaves_the_third_coordinate_alone() -> None:
    curve = ce.Curve.from_params([[0, 0, 0], [1, 0, 0]], [[[0, 0, 0.7], [0, 0, 0.7]]] * 2)
    out = ce.set_handle(curve, 0, 1, [0.5, 0.25], (0, 1))
    assert out.handles[0][1] == [0.5, 0.25, 0.7]


def test_untouched_curves_store_no_handles() -> None:
    assert ce.Curve.from_params([[0, 0], [1, 1]], []).stored_handles() == []
    assert ce.Curve.from_params([[0, 0], [1, 1]], [[[0, 0], [0, 0]]] * 2).stored_handles() == []


def test_a_figure_eight_is_found_and_a_square_is_not() -> None:
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert ce.crossings(square, closed=True) == set()
    bowtie = [[0, 0], [1, 1], [1, 0], [0, 1]]
    assert ce.crossings(bowtie, closed=True) == {0, 2}
    assert ce.crossings([[0, 0], [1, 0], [2, 0]], closed=False) == set()


def test_the_nearest_point_on_a_curve_is_reported_as_segment_and_parameter() -> None:
    curve = ce.Curve.from_params([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], [])
    segment, t, dist = ce.nearest_on_curve(curve, (1.5, 0.05), AXES)
    assert segment == 1 and t == pytest.approx(0.5, abs=0.05) and dist == pytest.approx(0.05)


# --- the gesture state machine -----------------------------------------------------------


def _pointer(x: float, y: float, **kw) -> ce.Pointer:
    return ce.Pointer(pos=(x, y), per_pixel=0.01, **kw)


def test_a_press_on_an_anchor_selects_and_begins_a_drag_that_moves_it() -> None:
    curve = ce.Curve.from_params([[0.0, 0.0], [1.0, 0.0]], [])
    ui = ce.CurveUi()
    _, events = ce.step(ui, curve, "outline", _pointer(1.0, 0.0, pressed=True, down=True), AXES)
    assert events == ["begin"] and ui.selected == 1 and ui.drag == ("anchor", 1, 0)
    moved, events = ce.step(ui, curve, "outline", _pointer(1.0, 0.5, down=True), AXES)
    assert events == ["change"] and moved.points[1] == [1.0, 0.5]
    _, events = ce.step(ui, moved, "outline", _pointer(1.0, 0.5, down=False), AXES)
    assert events == ["end"] and ui.drag is None


def test_alt_drag_pulls_handles_out_of_an_anchor() -> None:
    curve = ce.Curve.from_params([[0.0, 0.0], [1.0, 0.0]], [])
    ui = ce.CurveUi()
    ce.step(ui, curve, "path", _pointer(0.0, 0.0, pressed=True, down=True, alt=True), AXES)
    assert ui.drag == ("pull", 0, 0)
    pulled, _ = ce.step(ui, curve, "path", _pointer(0.0, 0.4, down=True, alt=True), AXES)
    assert pulled.handles[0] == [[0.0, -0.4], [0.0, 0.4]]


def test_a_lathe_radius_cannot_be_dragged_past_the_axis() -> None:
    curve = ce.Curve.from_params([[0.2, 0.0], [0.3, 1.0]], [])
    ui = ce.CurveUi()
    ce.step(ui, curve, "profile", _pointer(0.2, 0.0, pressed=True, down=True), AXES)
    moved, _ = ce.step(ui, curve, "profile", _pointer(-0.5, 0.1, down=True), AXES)
    assert moved.points[0][0] == 0.0


def test_a_press_on_the_line_inserts_a_point_and_starts_dragging_it() -> None:
    curve = ce.Curve.from_params([[0.0, 0.0], [1.0, 0.0]], [])
    ui = ce.CurveUi()
    new, events = ce.step(ui, curve, "path", _pointer(0.5, 0.01, pressed=True, down=True), AXES)
    assert events == ["begin", "change"]
    assert len(new.points) == 3 and ui.selected == 1 and ui.drag == ("anchor", 1, 0)


def test_a_press_in_empty_space_deselects() -> None:
    curve = ce.Curve.from_params([[0.0, 0.0], [1.0, 0.0]], [])
    ui = ce.CurveUi(selected=1)
    _, events = ce.step(ui, curve, "path", _pointer(0.5, 5.0, pressed=True, down=True), AXES)
    assert events == [] and ui.selected == -1


def test_delete_removes_the_selected_point_but_never_below_the_minimum() -> None:
    curve = ce.Curve.from_params([[0, 0], [1, 0], [2, 0]], [])
    ui = ce.CurveUi(selected=1)
    new, events = ce.step(ui, curve, "path", _pointer(0, 0, delete=True), AXES)
    assert events == ["begin", "change", "end"] and len(new.points) == 2
    ui2 = ce.CurveUi(selected=0)
    same, events = ce.step(ui2, new, "path", _pointer(0, 0, delete=True), AXES)
    assert events == [] and len(same.points) == 2, "a path keeps two points"
    tri = ce.Curve.from_params([[0, 0], [1, 0], [0, 1]], [], closed=True)
    _, events = ce.step(ce.CurveUi(selected=0), tri, "outline", _pointer(0, 0, delete=True), AXES)
    assert events == [], "an outline keeps three corners"


def test_a_right_click_on_a_point_deletes_it() -> None:
    curve = ce.Curve.from_params([[0, 0], [1, 0], [2, 0]], [])
    new, events = ce.step(
        ce.CurveUi(), curve, "path", _pointer(1.0, 0.0, right_pressed=True), AXES
    )
    assert "change" in events and new.points == [[0.0, 0.0], [2.0, 0.0]]


# --- the document: one undo step each -----------------------------------------------------


def _lathe() -> tuple[bd.ClayDoc, bd.Obj]:
    doc = bd.ClayDoc()
    defaults, build = bp.GENERATORS["lathe"]
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Vase",
            mesh=build(**defaults),
            generator="lathe",
            params=dict(defaults),
        )
    )
    return doc, obj


def _drive(doc, obj, key, ui, curve, pointers, axes=AXES):
    """The pane's once-a-frame glue, minus imgui: ``step`` then ``apply_events``."""
    for pointer in pointers:
        updated, events = ce.step(ui, curve, key, pointer, axes)
        curve = curve_editor.apply_events(
            doc, obj, key, ui, curve, updated, events, clay_props.apply_generator_params
        )
    return curve


def _stored(doc, obj, key="profile") -> ce.Curve:
    current = doc.by_uid(obj.uid)
    return ce.Curve.from_params(current.params[key], current.params["profile_handles"])


def test_a_drag_is_one_undo_step_and_the_mesh_follows_live() -> None:
    doc, obj = _lathe()
    start = _stored(doc, obj)
    anchor = start.points[4]
    steps = len(doc.history)
    ui = ce.CurveUi()
    seen = []
    pointers = [
        _pointer(*anchor, pressed=True, down=True),
        *[_pointer(anchor[0] + 0.02 * k, anchor[1], down=True) for k in range(1, 6)],
        _pointer(anchor[0] + 0.1, anchor[1], down=False),
    ]
    curve = start
    for pointer in pointers:
        updated, events = ce.step(ui, curve, "profile", pointer, AXES)
        curve = curve_editor.apply_events(
            doc, obj, "profile", ui, curve, updated, events, clay_props.apply_generator_params
        )
        seen.append(len(doc.history))

    assert max(seen) > steps + 1, "every frame of the drag rebuilt the mesh"
    assert len(doc.history) == steps + 1, "and the whole drag folded into one step"
    assert doc.history.top.label == "Edit profile"
    assert _stored(doc, obj).points[4][0] == pytest.approx(anchor[0] + 0.1)
    assert doc.undo()
    assert _stored(doc, obj).points[4][0] == pytest.approx(anchor[0])
    assert ui.working is None and ui.mark == -1


def test_inserting_a_point_is_one_undo_step() -> None:
    doc, obj = _lathe()
    start = _stored(doc, obj)
    a, b = np.array(start.points[1]), np.array(start.points[2])
    middle = (a + b) / 2
    steps = len(doc.history)
    ui = ce.CurveUi()
    _drive(
        doc,
        obj,
        "profile",
        ui,
        start,
        [_pointer(*middle, pressed=True, down=True), _pointer(*middle, down=False)],
    )
    assert len(_stored(doc, obj).points) == len(start.points) + 1
    assert len(doc.history) == steps + 1
    assert doc.undo() and len(_stored(doc, obj).points) == len(start.points)


def test_deleting_a_point_is_one_undo_step() -> None:
    doc, obj = _lathe()
    start = _stored(doc, obj)
    steps = len(doc.history)
    ui = ce.CurveUi(selected=2)
    _drive(doc, obj, "profile", ui, start, [_pointer(0, 0, delete=True)])
    assert len(_stored(doc, obj).points) == len(start.points) - 1
    assert len(doc.history) == steps + 1
    assert doc.undo() and len(_stored(doc, obj).points) == len(start.points)


def test_a_handle_drag_stores_handles_and_smooths_the_mesh() -> None:
    doc, obj = _lathe()
    start = _stored(doc, obj)
    vertices = len(doc.by_uid(obj.uid).mesh.positions)
    anchor = start.points[4]
    ui = ce.CurveUi()
    _drive(
        doc,
        obj,
        "profile",
        ui,
        start,
        [
            _pointer(*anchor, pressed=True, down=True, alt=True),
            _pointer(anchor[0], anchor[1] + 0.12, down=True, alt=True),
            _pointer(anchor[0], anchor[1] + 0.12, down=False),
        ],
    )
    params = doc.by_uid(obj.uid).params
    assert any(any(side) for row in params["profile_handles"] for side in row)
    assert len(doc.by_uid(obj.uid).mesh.positions) > vertices


def test_the_stored_params_are_read_back_after_the_clamp_reorders_them() -> None:
    """A drag that pushes a station past its neighbour in y is clamped by the
    generator (the plain clamp raises it to its predecessor's). The editor drew
    the working curve during the drag and reads the *stored* one after."""
    doc, obj = _lathe()
    start = _stored(doc, obj)
    top = start.points[-1]
    ui = ce.CurveUi()
    _drive(
        doc,
        obj,
        "profile",
        ui,
        start,
        [
            _pointer(*top, pressed=True, down=True),
            _pointer(top[0], top[1] - 5.0, down=True),
            _pointer(top[0], top[1] - 5.0, down=False),
        ],
    )
    stored = _stored(doc, obj)
    ys = [p[1] for p in stored.points]
    assert ys == sorted(ys), "the clamp kept the profile non-decreasing in y"
    assert ui.working is None


# --- the pane draws ----------------------------------------------------------------------------


@pytest.fixture
def ui_ctx(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


@pytest.mark.parametrize("generator", ["lathe", "sweep", "tube"])
def test_the_pane_draws_the_editor_for_each_curve_generator(ui_ctx, generator) -> None:
    doc = bd.ClayDoc()
    defaults, build = bp.GENERATORS[generator]
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name=generator,
            mesh=build(**defaults),
            generator=generator,
            params=dict(defaults),
        )
    )
    state = ClayState()
    for _ in range(2):  # the first frame fits the view; the second draws with it
        ui_ctx.new_frame()
        ui_ctx.begin("##host")
        try:
            clay_props._generator(doc, doc.by_uid(obj.uid), state=state)
        finally:
            ui_ctx.end()
            ui_ctx.end_frame()
    assert obj.uid in state.curve_ui


def test_the_generic_loop_no_longer_prints_the_curve_parameters_as_text(
    ui_ctx, monkeypatch
) -> None:
    doc, obj = _lathe()
    lines: list[str] = []
    monkeypatch.setattr(clay_props.widgets, "secondary", lambda text: lines.append(text))
    ui_ctx.new_frame()
    ui_ctx.begin("##host")
    try:
        clay_props._generator(doc, doc.by_uid(obj.uid), state=ClayState())
    finally:
        ui_ctx.end()
        ui_ctx.end_frame()
    assert not any(line.startswith(("profile:", "profile handles:")) for line in lines)
