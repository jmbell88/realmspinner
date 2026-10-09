"""The human Add door: what a click on a shape now gets that the agent's call does not.

``place_primitive`` puts the shape on the ground beside what is there, paints it
with the colour last picked and asks the camera to frame it -- all before
``add_object``, so the add stays one undo step. ``add_primitive`` stays the pure,
origin-placed door the agent calls. Each test's name is the claim it makes; the
changed-behaviour ones fail against the code before the first-click pass on the
assertion they name (a shape at ``y = 0``, a tab marked fitted while empty, a
remembered slot that was never kept).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import ops as mesh_ops
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.ui.panes import swatches
from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools
from realmspinner.studio.viewer.camera import Camera


class _Settings:
    def get(self, key, default=None):
        return default

    def set(self, key, value):
        pass


def _ctx(view=None):
    ctx = SimpleNamespace(
        state=SimpleNamespace(mode="clay", clay=None, preview={}),
        settings=_Settings(),
        toasts=[],
        cache=SimpleNamespace(invalidate=lambda: None),
    )
    ctx.toast = lambda message, kind="info", *a: ctx.toasts.append((message, kind))
    ctx.clay_view = view
    return ctx


def _world_lo(doc, obj):
    return mesh_ops.world_box(obj, world=doc.world_matrix(obj.uid))[0]


# -- the ground, beside, one step ----------------------------------------------------


def test_a_placed_shape_stands_on_the_ground_not_half_below_it():
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    obj = clay_tools.place_primitive(ctx, tab.doc, "box")
    assert _world_lo(tab.doc, obj)[1] == 0.0
    assert tab.doc.selection == {obj.uid}


@pytest.mark.parametrize("name", sorted(bp.CLAY_GENERATOR_NAMES))
def test_every_clay_shape_the_click_door_places_has_its_lowest_point_on_the_ground(name):
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    obj = clay_tools.place_primitive(ctx, tab.doc, name)
    assert _world_lo(tab.doc, obj)[1] == 0.0, name


def test_a_second_placed_shape_lands_beside_the_first_not_inside_it():
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    first = clay_tools.place_primitive(ctx, tab.doc, "box")
    second = clay_tools.place_primitive(ctx, tab.doc, "box")
    a = mesh_ops.world_box(first)
    b = mesh_ops.world_box(second)
    apart = b[0][0] >= a[1][0] or b[1][0] <= a[0][0] or b[0][2] >= a[1][2] or b[1][2] <= a[0][2]
    assert apart, "the second box was placed overlapping the first"
    assert b[0][1] == 0.0


def test_a_shape_added_while_one_is_selected_lands_next_to_the_selected_one():
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    doc = tab.doc
    far = clay_tools.place_primitive(ctx, doc, "box")
    doc.set_transform(far.uid, translation=(30.0, 0.5, 0.0))
    doc.select([far.uid])
    new = clay_tools.place_primitive(ctx, doc, "box")
    assert abs(new.translation[0] - 30.0) < 3.0, new.translation


def test_placing_a_shape_is_one_undo_step_and_one_undo_removes_it_entirely():
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    doc = tab.doc
    before = len(doc.history)
    clay_tools.place_primitive(ctx, doc, "box")
    assert len(doc.history) == before + 1, "grounding the shape pushed a second step"
    assert doc.undo()
    assert doc.objects == []


def test_the_agent_door_still_places_a_box_at_the_origin_for_its_own_translation_to_move():
    """The frozen Familiar card says a box of height h stands at [0, h/2, 0]
    *after the agent's own translation*; that only holds if the pure door does
    not ground or space the shape itself."""
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    clay_tools.add_primitive(ctx, tab.doc, "box")
    second = clay_tools.add_primitive(ctx, tab.doc, "box")
    assert np.allclose(second.translation, 0.0)


# -- refusals are unchanged ------------------------------------------------------------


def test_a_refused_click_asks_nothing_of_the_camera_and_remembers_nothing_new(monkeypatch):
    from realmspinner.kernels.mesh import glbimport

    monkeypatch.setattr(glbimport, "MAX_OBJECTS", 1)
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    clay_tools.place_primitive(ctx, tab.doc, "box")
    state = clay_mode.ensure(ctx)
    state.frame_pending = False
    steps = len(tab.doc.history)

    refused = clay_tools.place_primitive(ctx, tab.doc, "box")

    assert refused is None
    assert len(ctx.toasts) == 1 and ctx.toasts[0][1] == "error"
    assert len(tab.doc.history) == steps
    assert tab.doc.history._open_gestures == 0
    assert not state.frame_pending


# -- the colour last picked ---------------------------------------------------------------


def _with_palette(ctx, slots=3):
    tab = clay_mode.new_document(ctx)
    for _ in range(slots - 1):
        tab.doc.add_material()
    return tab


def test_a_shape_is_painted_with_the_slot_the_user_last_picked_in_one_undo_step():
    ctx = _ctx()
    tab = _with_palette(ctx)
    clay_mode.remember_slot(ctx, tab.doc, 2)
    before = len(tab.doc.history)

    obj = clay_tools.place_primitive(ctx, tab.doc, "box")

    assert obj.material == 2
    assert set(obj.mesh.material.tolist()) == {2}, "the faces carry the slot that renders"
    assert len(tab.doc.history) == before + 1, "the paint became a second undo step"


def test_a_shape_is_unpainted_before_any_colour_has_been_picked():
    ctx = _ctx()
    tab = _with_palette(ctx)
    obj = clay_tools.place_primitive(ctx, tab.doc, "box")
    assert obj.material == 0 and set(obj.mesh.material.tolist()) == {0}
    assert tab.last_slot is None


def test_a_remembered_slot_past_the_end_of_a_shrunk_palette_is_clamped_at_use():
    ctx = _ctx()
    tab = _with_palette(ctx, slots=3)
    clay_mode.remember_slot(ctx, tab.doc, 2)
    assert tab.doc.remove_material(2)
    assert len(tab.doc.materials) == 2

    obj = clay_tools.place_primitive(ctx, tab.doc, "box")

    assert obj.material == 1, "the remembered slot named a colour that is no longer there"
    assert set(obj.mesh.material.tolist()) == {1}


def test_a_remembered_slot_in_a_palette_reduced_to_one_places_the_default():
    ctx = _ctx()
    tab = _with_palette(ctx, slots=2)
    clay_mode.remember_slot(ctx, tab.doc, 1)
    assert tab.doc.remove_material(1)
    obj = clay_tools.place_primitive(ctx, tab.doc, "box")
    assert obj.material == 0


def test_the_remembered_colour_belongs_to_its_own_document():
    ctx = _ctx()
    a = _with_palette(ctx)
    b = _with_palette(ctx)
    clay_mode.remember_slot(ctx, a.doc, 2)
    assert clay_tools.place_primitive(ctx, b.doc, "box").material == 0
    assert clay_tools.place_primitive(ctx, a.doc, "box").material == 2


def test_a_plain_swatch_click_is_remembered_and_a_ctrl_click_is_not(monkeypatch):
    ctx = _ctx()
    tab = _with_palette(ctx)
    held = {"ctrl": True}
    monkeypatch.setattr(swatches, "_ctrl_held", lambda: held["ctrl"])
    swatches.click_slot(ctx, tab.doc, 1)
    assert tab.last_slot is None, "Ctrl+click only opens a slot to edit"
    held["ctrl"] = False
    swatches.click_slot(ctx, tab.doc, 2)
    assert tab.last_slot == 2


def test_a_swatch_click_with_an_object_selected_paints_it_and_is_remembered(monkeypatch):
    ctx = _ctx()
    tab = _with_palette(ctx)
    monkeypatch.setattr(swatches, "_ctrl_held", lambda: False)
    obj = clay_tools.place_primitive(ctx, tab.doc, "box")
    swatches.click_slot(ctx, tab.doc, 1)
    assert obj.material == 1 and tab.last_slot == 1
    assert clay_tools.place_primitive(ctx, tab.doc, "box").material == 1


def test_a_context_with_no_clay_state_places_the_shape_in_the_default_colour():
    """The Add buttons' tests hand the door a bare toast context."""

    class Bare:
        def toast(self, *a, **k):  # pragma: no cover - never refused here
            pass

    doc = bd.ClayDoc()
    obj = clay_tools.place_primitive(Bare(), doc, "box")
    assert obj.material == 0
    assert _world_lo(doc, obj)[1] == 0.0


# -- the camera ------------------------------------------------------------------------------


def test_the_first_shape_asks_the_viewport_to_frame_it():
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    state = clay_mode.ensure(ctx)
    assert not state.frame_pending
    clay_tools.place_primitive(ctx, tab.doc, "box")
    assert state.frame_pending and not state.frame_whole


def test_a_later_shape_that_lands_in_view_does_not_move_the_camera():
    camera = Camera(aspect=1.5)
    camera.distance = 14.0  # a view that has room, as one framed on a larger model has
    ctx = _ctx(SimpleNamespace(camera=camera))
    tab = clay_mode.new_document(ctx)
    clay_tools.place_primitive(ctx, tab.doc, "box")
    state = clay_mode.ensure(ctx)
    state.frame_pending = False

    clay_tools.place_primitive(ctx, tab.doc, "box")

    assert not state.frame_pending


def test_a_later_shape_that_lands_off_screen_asks_for_the_whole_model_to_be_framed():
    camera = Camera(aspect=1.5)
    ctx = _ctx(SimpleNamespace(camera=camera))
    tab = clay_mode.new_document(ctx)
    doc = tab.doc
    far = clay_tools.place_primitive(ctx, doc, "box")
    doc.set_transform(far.uid, translation=(60.0, 0.5, 0.0))
    doc.select([far.uid])
    state = clay_mode.ensure(ctx)
    state.frame_pending = False

    clay_tools.place_primitive(ctx, doc, "box")

    assert state.frame_pending and state.frame_whole, (
        "framing only the new shape would push the rest of the model out of view"
    )


def test_the_agent_door_never_asks_for_a_frame():
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    clay_tools.add_primitive(ctx, tab.doc, "box")
    assert not clay_mode.ensure(ctx).frame_pending


# -- apply_camera -----------------------------------------------------------------------------


class _FakeView:
    def __init__(self):
        self.camera = Camera()
        self.framed = 0

    def frame_selection(self, doc):
        self.framed += 1
        self.camera.distance = self.camera._goal_distance = 9.0


def test_an_empty_document_is_not_marked_fitted_by_the_camera_hand_off():
    view = _FakeView()
    ctx = _ctx(view)
    tab = clay_mode.new_document(ctx)
    clay_mode.apply_camera(ctx, tab)
    assert not tab.view.fitted, "an empty model was declared framed, so its first shape never was"
    assert view.camera.distance == 3.0, "an empty document moved the camera"


def test_the_first_hand_off_after_a_shape_arrives_frames_and_fits_the_tab():
    view = _FakeView()
    ctx = _ctx(view)
    tab = clay_mode.new_document(ctx)
    clay_mode.apply_camera(ctx, tab)
    clay_tools.add_primitive(ctx, tab.doc, "box")

    clay_mode.apply_camera(ctx, tab)

    assert tab.view.fitted and view.framed == 1
    assert tab.view.distance == 9.0


# -- the other human door -------------------------------------------------------------------


def test_the_empty_viewports_add_button_places_on_the_ground_like_the_rail_does():
    """``overlay._clay_box`` called the pure agent door directly, so the one button
    under an empty viewport -- the first thing a new user presses -- sank the box."""
    from realmspinner.studio.panes import overlay

    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    overlay._clay_box(ctx)
    (obj,) = tab.doc.objects
    assert _world_lo(tab.doc, obj)[1] == 0.0
    assert clay_mode.ensure(ctx).frame_pending, "the first shape was not framed"


def test_the_empty_viewports_add_button_toasts_a_refusal_instead_of_raising(monkeypatch):
    from realmspinner.kernels.mesh import glbimport
    from realmspinner.studio.panes import overlay

    monkeypatch.setattr(glbimport, "MAX_OBJECTS", 1)
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    overlay._clay_box(ctx)
    overlay._clay_box(ctx)  # the document is full: this used to raise out of the draw
    assert len(tab.doc.objects) == 1
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
