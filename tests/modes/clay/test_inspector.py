"""The Clay Inspector: one panel whose body follows what is selected.

Each test's name is the claim it makes. The panel is driven through a real (GL-less)
imgui context by the same input queue a person uses (``_ui_driver``), so a control
that is wired to nothing, clipped past its pane or greyed with no reason fails here
the way it would on screen.

The tests for *changed behaviour* name the old code's failure beside them:

* a taller height used to sink a grounded shape (``props._dimensions`` scaled about the
  pivot, so a 1 m box grew to 2 m with its lowest point at y = -0.5) -- the assertion on
  the lowest point fails on it;
* the Size field used to apply on every keystroke (one undo step per digit) -- the
  one-undo-step assertion fails on it;
* the right column's lower slot was ``clay-props`` with a tab strip -- the slot-id
  assertions fail on it.
"""

from __future__ import annotations

import statistics
import time
from types import SimpleNamespace

import numpy as np
import pytest
from _ui_context import imgui_context
from _ui_driver import UiDriver

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import ops as mesh_ops
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import anchors, icons, widgets
from realmspinner.studio.modes.clay import doc_io, menutree, starter_colours
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.ui import inspector_actions, inspector_size
from realmspinner.studio.modes.clay.ui.panes import inspector as clay_inspector
from realmspinner.studio.modes.clay.ui.panes import props as clay_props
from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools
from realmspinner.studio.viewer.camera import Camera

# -- a rig -----------------------------------------------------------------------------


class _Settings:
    def get(self, key, default=None):
        return default

    def set(self, key, value):
        pass


def _ctx(view=None):
    ctx = SimpleNamespace(
        state=SimpleNamespace(mode="clay", clay=None, preview={}, list_filters={}),
        settings=_Settings(),
        toasts=[],
        cache=SimpleNamespace(invalidate=lambda: None),
        busy=lambda key: False,
        submit=lambda *a, **k: True,
    )
    ctx.toast = lambda message, kind="info", *a: ctx.toasts.append((message, kind))
    ctx.clay_view = view
    return ctx


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(clay_mode, "recent_paths", lambda ctx: [])
    widgets._OPEN_REQUESTS.clear()
    yield
    widgets._OPEN_REQUESTS.clear()


@pytest.fixture
def imgui_ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _document(shapes: int = 1, *, view=None):
    """A ctx and a Clay document holding *shapes* boxes placed through the human door
    (on the ground, beside each other); the last one placed is selected."""
    ctx = _ctx(view)
    tab = clay_mode.new_document(ctx)
    for _ in range(shapes):
        clay_tools.place_primitive(ctx, tab.doc, "box")
    return ctx, tab, tab.doc


class Rig:
    """The inspector drawn in a 300 px column, with a driver that clicks it."""

    def __init__(self, imgui, ctx, *, height: float = 4000.0, draw=None):
        self.imgui = imgui
        self.ctx = ctx
        self.height = height
        self.draw = draw or (lambda: clay_inspector.draw(ctx))
        # A control below the display's edge is clipped away, and a test that reads
        # the whole column needs the whole column on screen.
        imgui.get_io().display_size = (1600.0, max(float(height), 950.0))
        self.driver = UiDriver(imgui, self._frame)

    def _frame(self):
        imgui = self.imgui
        flags = (
            imgui.WindowFlags_.no_title_bar.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_resize.value
            | imgui.WindowFlags_.no_saved_settings.value
        )
        anchors.begin_frame()
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((300, self.height))
        imgui.begin("##inspector-host", None, flags)
        self.draw()
        imgui.end()
        imgui.end_frame()

    def settle(self, frames: int = 3):
        return self.driver.settle(frames)

    @property
    def labels(self) -> list[str]:
        return [c.label for c in self.driver.controls if c.visible]

    def find(self, **kw):
        return self.driver.find(**kw)

    def click(self, **kw):
        self.driver.click(self.driver.find(**kw))
        self.settle(2)


def _rig(imgui_ui, ctx, **kw):
    rig = Rig(imgui_ui, ctx, **kw)
    rig.settle()
    return rig


def _state(ctx):
    return clay_mode.ensure(ctx)


# -- the slot ----------------------------------------------------------------------------


def test_the_inspector_is_the_slot_that_replaced_clay_props_and_has_no_tab_strip():
    """The tab strip itself is gone from ``props`` too: ``test_inspector_elements.py``."""
    from realmspinner.studio import skeletons

    ids = [slot.id for slot in skeletons.clay(SimpleNamespace())["right"].slots]
    assert ids == ["clay-inspector"], "the outliner is docked inside it, not a slot above it"
    assert not hasattr(clay_inspector, "TABS"), "the inspector is a switch on the selection"


# -- the four situations -------------------------------------------------------------------


def test_nothing_selected_shows_the_four_shape_tiles_and_the_model_card(imgui_ui):
    ctx, tab, doc = _document(1)
    doc.select([])
    rig = _rig(imgui_ui, ctx)
    labels = rig.labels
    for name in clay_tools.QUICK_SHAPES:
        assert f"##insp-tile/{name}" in labels
    assert f"{icons.PLUS} New" in labels and f"{icons.SAVE} Save (Ctrl+S)" in labels
    assert f"{icons.DOWNLOAD} Export to the library" in labels
    assert any("##insp-history" in label for label in labels)
    assert "##size##insp" not in labels, "no Size card without a shape"


def test_a_tile_places_its_shape_through_the_human_door(imgui_ui):
    ctx, tab, doc = _document(0)
    rig = _rig(imgui_ui, ctx)
    rig.click(label="##insp-tile/cylinder")
    assert [o.generator for o in doc.objects] == ["cylinder"]
    lowest = mesh_ops.world_box(doc.objects[0], world=doc.world_matrix(doc.objects[0].uid))[0][1]
    assert lowest == 0.0, "it stands on the ground, as the rail's button places it"
    assert doc.selection == {doc.objects[0].uid}


def test_one_shape_selected_shows_size_colour_duplicate_and_delete_and_closes_every_fold(
    imgui_ui,
):
    ctx, tab, doc = _document(1)
    rig = _rig(imgui_ui, ctx)
    labels = rig.labels
    assert "##size##insp" in labels
    swatches = [label for label in labels if label.startswith("##inspsw/")]
    assert swatches[:8] == [f"##inspsw/{c.name}" for c in starter_colours.STARTER_COLOURS]
    assert swatches[8:] == ["##inspsw/custom"], "the default grey is none of the eight"
    assert "Duplicate##insp-duplicate" in labels and "Delete##insp-delete" in labels
    # The three folds are closed, so nothing that lives inside them is drawn.
    assert not any(label.startswith(("position##", "rotation (deg)##")) for label in labels)
    assert not any("##insp-op/" in label for label in labels)
    assert not any("parent" in label for label in labels if "clearparent" in label)
    # And the old tab strip is gone.
    assert "clay-props-tab" not in " ".join(labels)


def test_a_box_has_no_shape_settings_card_but_a_cylinder_does(imgui_ui):
    ctx, tab, doc = _document(0)
    clay_tools.place_primitive(ctx, doc, "box")
    rig = _rig(imgui_ui, ctx)
    assert not any("##gen" in label for label in rig.labels), (
        "a box's only parameter is its size, which the Size card already says"
    )
    clay_tools.place_primitive(ctx, doc, "cylinder")
    rig.settle()
    assert any(label.endswith("##genradius") for label in rig.labels)
    assert any(label.endswith("##gensegments") for label in rig.labels)


def test_several_shapes_selected_hide_the_size_card_and_open_the_group_actions(imgui_ui):
    ctx, tab, doc = _document(3)
    doc.select([o.uid for o in doc.objects])
    rig = _rig(imgui_ui, ctx)
    labels = rig.labels
    assert "##size##insp" not in labels, "no Size card for several shapes"
    assert "Duplicate##insp-duplicate" in labels and "Delete##insp-delete" in labels
    assert any(label.startswith("##inspsw/") for label in labels)
    for name in ("group", "join", "parent-to-last", "mirror-x"):
        assert f"{clay_ops.get(name).label}##insp-op/{name}" in labels, name


def test_an_element_mode_shows_the_element_state_under_a_done_button_that_returns_to_shape_mode(
    imgui_ui,
):
    """Old code: the Object tab's body (``position##bt``, ``size##bz``) under Done. The
    element state's own claims are in ``test_inspector_elements.py``."""
    ctx, tab, doc = _document(1)
    doc.set_element_mode("face")
    rig = _rig(imgui_ui, ctx)
    labels = rig.labels
    assert any("##insp-done" in label for label in labels)
    assert any("##insp-select/" in label for label in labels), "the Select row"
    assert "Extrude##insp-op/extrude" in labels, "Shape it"
    assert "position##bt" not in labels and "size##bz" not in labels, "no legacy Object body"
    assert "##size##insp" not in labels
    rig.click(contains="##insp-done")
    assert doc.element_mode == "object"
    assert doc.selection, "the shape stays selected, as the selection rule says"


@pytest.mark.parametrize("count", [2, 3, 16])
def test_the_inspector_says_how_many_shapes_are_selected_instead_of_nothing_selected(
    imgui_ui, monkeypatch, count
):
    """The 2026-09 UX pass's claim, moved with the pane that made it: with sixteen shapes lit
    up the Object tab said "Nothing selected", which the viewport plainly contradicts."""
    ctx, tab, doc = _document(count)
    doc.select([o.uid for o in doc.objects])
    seen: list[str] = []
    real = widgets.section
    monkeypatch.setattr(widgets, "section", lambda label: (seen.append(label), real(label))[1])
    _rig(imgui_ui, ctx)
    assert f"{count} shapes selected" in seen
    assert "Nothing selected" not in seen


def test_the_inspector_with_no_shape_selected_does_not_say_a_shape_is_selected(
    imgui_ui, monkeypatch
):
    ctx, tab, doc = _document(1)
    doc.select([])
    seen: list[str] = []
    real = widgets.section
    monkeypatch.setattr(widgets, "section", lambda label: (seen.append(label), real(label))[1])
    _rig(imgui_ui, ctx)
    assert not any("selected" in label for label in seen)


def test_a_page_is_reachable_from_an_element_mode(imgui_ui):
    ctx, tab, doc = _document(1)
    doc.set_element_mode("face")
    rig = _rig(imgui_ui, ctx)
    rig.click(contains="##insp-page/colours")
    assert _state(ctx).inspector_page == "colours"
    assert any("##insp-back" in label for label in rig.labels)


# -- starter colours -------------------------------------------------------------------------


def test_the_eight_starter_colours_are_named_and_are_ui_only():
    assert [c.name for c in starter_colours.STARTER_COLOURS] == [
        "Stone", "Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Charcoal",
    ]  # fmt: skip
    assert starter_colours.by_name("Blue").hex == "#4A7BEA"
    fresh = bd.ClayDoc()
    assert len(fresh.materials) == 1, "a fresh document has its one grey slot and nothing more"
    assert fresh.materials[0].name != "Blue"


def test_drawing_the_inspector_never_changes_a_documents_palette_until_a_colour_is_clicked(
    imgui_ui,
):
    ctx, tab, doc = _document(2)
    before = list(doc.materials)
    steps = len(doc.history)
    rig = _rig(imgui_ui, ctx)
    for _ in range(5):
        rig.settle(1)
    assert doc.materials == before and len(before) == 1
    assert len(doc.history) == steps, "drawing is not an edit"


def test_clicking_a_starter_colour_adds_its_slot_and_paints_the_shape_in_one_undo_step(imgui_ui):
    ctx, tab, doc = _document(1)
    obj = doc.objects[0]
    steps = len(doc.history)
    rig = _rig(imgui_ui, ctx)
    rig.click(label="##inspsw/Blue")
    assert [m.name for m in doc.materials] == [doc.materials[0].name, "Blue"]
    blue = doc.materials[1]
    assert tuple(round(c, 3) for c in blue.base_color_factor) == pytest.approx(
        starter_colours.by_name("Blue").factor, abs=1e-3
    )
    assert obj.material == 1 and set(obj.mesh.material.tolist()) == {1}
    assert len(doc.history) == steps + 1, "the new slot and the paint are one step"
    assert doc.undo()
    assert len(doc.materials) == 1 and obj.material == 0
    assert set(obj.mesh.material.tolist()) == {0}
    assert clay_mode.remembered_slot(ctx, doc) is None, "slot 0 is the default, so nothing to wear"


def test_clicking_a_starter_colour_the_palette_already_holds_finds_that_slot(imgui_ui):
    ctx, tab, doc = _document(2)
    first, second = doc.objects
    doc.select([first.uid])
    rig = _rig(imgui_ui, ctx)
    rig.click(label="##inspsw/Red")
    doc.select([second.uid])
    rig.settle()
    rig.click(label="##inspsw/Red")
    assert len(doc.materials) == 2, "the second click found the slot the first one made"
    assert first.material == second.material == 1


def test_a_starter_colour_paints_every_selected_shape_in_one_undo_step(imgui_ui):
    ctx, tab, doc = _document(3)
    doc.select([o.uid for o in doc.objects])
    steps = len(doc.history)
    rig = _rig(imgui_ui, ctx)
    rig.click(label="##inspsw/Green")
    assert all(set(o.mesh.material.tolist()) == {1} for o in doc.objects)
    assert len(doc.history) == steps + 1
    assert doc.undo()
    assert all(set(o.mesh.material.tolist()) == {0} for o in doc.objects)
    assert len(doc.materials) == 1


def test_the_swatch_a_shape_wears_is_outlined_and_a_custom_colour_gets_its_own_ring(imgui_ui):
    ctx, tab, doc = _document(1)
    rig = _rig(imgui_ui, ctx)
    selected = [c.label for c in rig.driver.controls if c.kind == "swatch" and c.selected]
    assert selected == ["##inspsw/custom"], "the default grey is none of the eight"
    rig.click(label="##inspsw/Purple")
    selected = [c.label for c in rig.driver.controls if c.kind == "swatch" and c.selected]
    assert selected == ["##inspsw/Purple"]


def test_starter_matching_ignores_textured_and_see_through_slots():
    from dataclasses import replace

    doc = bd.ClayDoc()
    blue = starter_colours.by_name("Blue")
    flat = replace(doc.materials[0], base_color_factor=blue.factor)
    assert starter_colours.holds(flat, blue)
    assert not starter_colours.holds(replace(flat, base_color=(1, 1, 1, b"\xff" * 4)), blue)
    assert not starter_colours.holds(replace(flat, alpha_mode="BLEND"), blue)
    doc.materials[0] = flat
    assert starter_colours.matching(doc, 0) is blue
    assert starter_colours.matching(doc, 5) is None


# -- size --------------------------------------------------------------------------------


def _grounded_box(imgui_ui=None):
    ctx, tab, doc = _document(1)
    return ctx, tab, doc, doc.objects[0]


def _lowest(doc, obj) -> float:
    return float(mesh_ops.world_box(obj, world=doc.world_matrix(obj.uid))[0][1])


def test_a_taller_height_keeps_the_lowest_point_fixed_in_one_undo_step():
    """Old code: ``props._dimensions`` scaled about the pivot, so the 1 m box grew to
    2 m with its lowest point at -0.5 and the undo step count was right but the box
    was through the floor."""
    ctx, tab, doc, obj = _grounded_box()
    assert _lowest(doc, obj) == 0.0
    steps = len(doc.history)
    was = tuple(v.copy() for v in obj.trs())

    assert inspector_size.resize(ctx, doc, obj, np.array([1.0, 2.0, 1.0]), was)

    assert obj.scale[1] == pytest.approx(2.0)
    assert _lowest(doc, obj) == pytest.approx(0.0, abs=1e-9), "the box sank through the ground"
    assert len(doc.history) == steps + 1, "the size and the lift are one step"
    assert doc.undo()
    assert obj.scale[1] == pytest.approx(1.0)
    assert obj.translation[1] == pytest.approx(0.5)


def test_a_width_or_depth_change_keeps_the_middle_where_it_was():
    ctx, tab, doc, obj = _grounded_box()
    centre = obj.translation.copy()
    inspector_size.resize(ctx, doc, obj, np.array([3.0, 1.0, 2.0]))
    assert obj.translation == pytest.approx(centre)
    assert _lowest(doc, obj) == pytest.approx(0.0, abs=1e-9)


def test_a_height_change_under_lock_aspect_still_stands_on_the_ground():
    ctx, tab, doc, obj = _grounded_box()
    inspector_size.resize(ctx, doc, obj, np.array([2.0, 2.0, 2.0]))
    assert _lowest(doc, obj) == pytest.approx(0.0, abs=1e-9)


def test_a_child_of_a_scaled_parent_is_anchored_on_the_world_floor():
    ctx, tab, doc, parent = _grounded_box()
    child = clay_tools.place_primitive(ctx, doc, "box")
    doc.set_parent(child.uid, parent.uid, keep_world=True)
    doc.set_transform(parent.uid, scale=[1.0, 3.0, 1.0])
    floor = _lowest(doc, child)
    new_scale = np.array(child.scale) * [1.0, 2.0, 1.0]
    inspector_size.resize(ctx, doc, child, new_scale)
    assert _lowest(doc, child) == pytest.approx(floor, abs=1e-9)


def test_typing_a_height_is_applied_once_when_the_field_is_left_as_one_undo_step(imgui_ui):
    """Old code: the size boxes applied per keystroke, one undo step per digit."""
    ctx, tab, doc = _document(1)
    obj = doc.objects[0]
    steps = len(doc.history)
    rig = _rig(imgui_ui, ctx)
    rig.driver.click(rig.find(label="##size##insp"))
    rig.driver.type_text("25")
    assert obj.scale[1] == pytest.approx(1.0), "typing alone changes nothing yet"
    rig.driver.press_key(imgui_ui.Key.enter)
    rig.settle()
    assert obj.scale[1] == pytest.approx(25.0)
    assert len(doc.history) == steps + 1, "one step for '25', not one for '2' and one for '25'"
    assert _lowest(doc, obj) == pytest.approx(0.0, abs=1e-9)


def test_a_size_that_leaves_the_view_asks_the_camera_to_frame_the_model():
    camera = Camera(aspect=1.5)
    camera.distance = 6.0
    ctx, tab, doc = _document(1, view=SimpleNamespace(camera=camera))
    obj = doc.objects[0]
    state = _state(ctx)
    state.frame_pending = False
    inspector_size.resize(ctx, doc, obj, np.array([1.0, 1.2, 1.0]))
    assert not state.frame_pending, "still in view: the camera is left alone"
    inspector_size.resize(ctx, doc, obj, np.array([400.0, 1.2, 1.0]))
    assert state.frame_pending and state.frame_whole


def test_the_bounds_line_is_in_the_display_unit_and_free_of_float_dust():
    box = (np.array([-0.5, 0.0, -0.5]), np.array([0.5, 3.0000000001, 0.5]))
    assert inspector_size.bounds_line(box, "m") == (
        "Spans x -0.5 to 0.5 · y 0 to 3 · z -0.5 to 0.5 m"
    )
    assert "cm" in inspector_size.bounds_line(box, "cm")


# -- the folds and More actions -------------------------------------------------------------


def test_position_and_turn_holds_position_and_rotation_and_scale_hides_behind_advanced(imgui_ui):
    ctx, tab, doc = _document(1)
    widgets.request_open("clay/insp/position")
    rig = _rig(imgui_ui, ctx)
    labels = rig.labels
    assert "position##bt" in labels and "rotation (deg)##br" in labels
    assert "scale##bs" not in labels and "size##bz" not in labels, (
        "Size is the one sizing control, and Scale is behind Advanced"
    )
    assert labels.count("##clay-length-unit") == 1, "one unit chooser, on the Size card"
    widgets.request_open("clay/insp/advanced")
    rig.settle()
    assert "scale##bs" in rig.labels


def test_group_and_parent_holds_the_parent_combo_and_clear_parent(imgui_ui):
    ctx, tab, doc = _document(2)
    widgets.request_open("clay/insp/group")
    rig = _rig(imgui_ui, ctx)
    assert "Clear parent##clearparent" in rig.labels


def test_more_actions_are_generated_from_the_registry_not_listed_by_hand():
    expected = {
        op.name
        for run in menutree.flat("object")
        for op in run
        if op.name not in inspector_actions.PRIMARY
        and op.name not in menutree.GROUPS["select"].ops
        and op.name != "frame"
    }
    drawn = {op.name for _k, _l, ops in inspector_actions.action_groups("object") for op in ops}
    assert drawn == expected
    assert "duplicate" not in drawn and "delete" not in drawn, "they have a row of their own"


def test_every_more_actions_button_is_greyed_with_the_registrys_own_reason(imgui_ui):
    ctx, tab, doc = _document(1)
    widgets.request_open("clay/insp/more")
    rig = _rig(imgui_ui, ctx)
    by_label = {c.label: c for c in rig.driver.controls if c.visible}
    grey = 0
    for _key, _label, ops in inspector_actions.action_groups("object"):
        for op in ops:
            control = by_label[f"{op.label}##insp-op/{op.name}"]
            assert control.enabled == op.enabled(doc), op.name
            if not control.enabled:
                grey += 1
                assert control.reason and control.reason == clay_ops.reason_for(op, doc), op.name
    assert grey, "one box leaves Merge Objects and Parent to Last greyed"
    assert by_label[f"{clay_ops.get('join').label}##insp-op/join"].reason


def test_closed_disclosures_do_not_evaluate_op_enabled(imgui_ui):
    ctx, tab, doc = _document(2)
    calls: list[str] = []
    ops = [op for op in clay_ops.OPS if op.name not in inspector_actions.PRIMARY]
    originals = {op.name: op.enabled for op in ops}

    def spy_for(name):
        def spy(d):
            calls.append(name)
            return originals[name](d)

        return spy

    for op in ops:
        object.__setattr__(op, "enabled", spy_for(op.name))
    try:
        rig = _rig(imgui_ui, ctx)
        for _ in range(4):
            rig.settle(1)
        assert calls == [], f"a closed More actions asked {sorted(set(calls))}"
        widgets.request_open("clay/insp/more")
        rig.settle(2)
        assert calls, "open, it does ask"
    finally:
        for op in ops:
            object.__setattr__(op, "enabled", originals[op.name])


def test_the_gates_are_asked_once_per_change_not_once_per_frame(imgui_ui):
    ctx, tab, doc = _document(2)
    widgets.request_open("clay/insp/more")
    calls: list[str] = []
    ops = [op for op in clay_ops.OPS if op.name not in inspector_actions.PRIMARY]
    originals = {op.name: op.enabled for op in ops}
    for op in ops:
        object.__setattr__(
            op, "enabled", lambda d, n=op.name: (calls.append(n), originals[n](d))[1]
        )
    try:
        rig = _rig(imgui_ui, ctx)
        calls.clear()
        for _ in range(5):
            rig.settle(1)
        assert calls == [], "an open More actions on an unchanged document asks nothing"
    finally:
        for op in ops:
            object.__setattr__(op, "enabled", originals[op.name])


def test_an_op_with_numbers_opens_inline_fields_and_apply_runs_it_with_them(imgui_ui):
    ctx, tab, doc = _document(1)
    obj = doc.objects[0]
    doc.set_transform(obj.uid, translation=[0.3, 0.7, 0.1])
    widgets.request_open("clay/insp/more")
    rig = _rig(imgui_ui, ctx)
    state = _state(ctx)
    snap = clay_ops.get("snap-to-grid")
    rig.click(label=f"{snap.label}##insp-op/{snap.name}")
    assert state.inspector_form == snap.name
    assert any(f"##insp-apply/{snap.name}" in label for label in rig.labels), "the form is inline"
    state.op_params[snap.name]["step"] = 0.5
    steps = len(doc.history)
    rig.click(contains=f"##insp-apply/{snap.name}")
    assert list(obj.translation) == pytest.approx([0.5, 0.5, 0.0])
    assert len(doc.history) == steps + 1
    assert state.inspector_form == ""
    assert state.op_params[snap.name]["step"] == 0.5, "remembered for next time, as the dialog does"


def test_a_bare_op_button_runs_the_op_through_the_menus_door(imgui_ui):
    ctx, tab, doc = _document(1)
    obj = doc.objects[0]
    doc.set_transform(obj.uid, translation=[0.0, 4.0, 0.0])
    widgets.request_open("clay/insp/more")
    rig = _rig(imgui_ui, ctx)
    drop = clay_ops.get("drop-to-ground")
    rig.click(label=f"{drop.label}##insp-op/{drop.name}")
    assert _lowest(doc, obj) == pytest.approx(0.0, abs=1e-9)


# -- duplicate and delete -----------------------------------------------------------------


def test_duplicate_and_delete_run_the_registrys_ops_and_are_one_step_each(imgui_ui):
    ctx, tab, doc = _document(1)
    steps = len(doc.history)
    rig = _rig(imgui_ui, ctx)
    rig.click(label="Duplicate##insp-duplicate")
    assert len(doc.objects) == 2 and len(doc.history) == steps + 1
    rig.click(label="Delete##insp-delete")
    assert len(doc.objects) == 1 and len(doc.history) == steps + 2


def test_delete_with_nothing_to_delete_is_not_offered_because_the_state_is_nothing_selected(
    imgui_ui,
):
    ctx, tab, doc = _document(1)
    doc.select([])
    rig = _rig(imgui_ui, ctx)
    assert "Delete##insp-delete" not in rig.labels


def test_every_greyed_control_in_the_inspector_says_why(imgui_ui):
    ctx, tab, doc = _document(1)
    widgets.request_open("clay/insp/more")
    rig = _rig(imgui_ui, ctx)
    greyed = [c for c in rig.driver.controls if c.visible and not c.enabled]
    assert greyed
    assert [c.label for c in greyed if not c.reason] == []


def test_a_save_in_flight_greys_the_actions_with_one_shared_sentence(imgui_ui):
    ctx, tab, doc = _document(1)
    tab.saving = True
    widgets.request_open("clay/insp/more")
    rig = _rig(imgui_ui, ctx)
    actions = [c for c in rig.driver.controls if "##insp-op/" in c.label]
    assert actions and all(not c.enabled for c in actions)
    assert len({c.reason for c in actions}) == 1 and actions[0].reason
    dup = rig.find(label="Duplicate##insp-duplicate")
    assert not dup.enabled and dup.reason


# -- pages ---------------------------------------------------------------------------------


def test_edit_colour_pushes_the_colours_page_and_back_pops_it(imgui_ui):
    ctx, tab, doc = _document(1)
    rig = _rig(imgui_ui, ctx)
    state = _state(ctx)
    assert state.inspector_page == ""
    rig.click(contains="##insp-edit-colour")
    assert state.inspector_page == "colours"
    assert any("##insp-back" in label for label in rig.labels)
    assert "##size##insp" not in rig.labels, "the page replaces the body"
    rig.click(contains="##insp-back")
    assert state.inspector_page == ""
    assert "##size##insp" in rig.labels


def test_the_history_row_opens_model_and_history_and_the_texture_link_opens_unwrap(imgui_ui):
    ctx, tab, doc = _document(1)
    doc.select([])
    rig = _rig(imgui_ui, ctx)
    rig.click(contains="##insp-history")
    assert _state(ctx).inspector_page == "model"
    assert any("##clay-import-scale" in label for label in rig.labels), "the import settings"
    rig.click(contains="##insp-back")
    doc.select([doc.objects[0].uid])
    widgets.request_open("clay/insp/more")
    rig.settle()
    rig.click(contains="##insp-uv-page")
    assert _state(ctx).inspector_page == "unwrap"


def test_escape_closes_a_page_first_then_the_picks_then_the_mode_then_the_shape():
    ctx, tab, doc = _document(1)
    state = _state(ctx)
    uid = doc.objects[0].uid
    doc.set_element_mode("face")
    doc.set_element_sel(uid, bd.el.ElementSel(faces=[0]))
    state.inspector_page = "colours"

    clay_mode._escape(state, tab, doc)
    assert state.inspector_page == "" and doc.element_sel and doc.element_mode == "face"
    clay_mode._escape(state, tab, doc)
    assert not doc.element_sel and doc.element_mode == "face"
    clay_mode._escape(state, tab, doc)
    assert doc.element_mode == "object" and doc.selection == {uid}
    clay_mode._escape(state, tab, doc)
    assert doc.selection == set()


def test_escape_closes_a_page_even_while_a_save_is_in_flight():
    ctx, tab, doc = _document(1)
    state = _state(ctx)
    tab.saving = True
    state.inspector_page = "unwrap"
    clay_mode._escape(state, tab, doc)
    assert state.inspector_page == ""
    assert doc.selection, "and the selection is untouched"


def test_a_page_is_dropped_when_no_document_is_open_and_an_unknown_page_reads_as_none(imgui_ui):
    ctx, tab, doc = _document(1)
    state = _state(ctx)
    state.inspector_page = "retired-page"
    rig = _rig(imgui_ui, ctx)
    assert state.inspector_page == "" and "##size##insp" in rig.labels
    state.inspector_page = "colours"
    state.close(tab.uid)
    rig.settle()
    assert state.inspector_page == ""


# -- the tour --------------------------------------------------------------------------------


def test_the_tour_anchors_are_marked_where_the_tour_expects_them(imgui_ui):
    ctx, tab, doc = _document(1)
    rig = _rig(imgui_ui, ctx)
    assert {"clay/inspector", "clay/props/size"} <= anchors.marked()
    assert "clay/props/save" not in anchors.marked(), "the Model card is the nothing-selected state"
    doc.select([])
    rig.settle()
    assert {"clay/inspector", "clay/props/save", "clay/props/export"} <= anchors.marked()


def test_save_and_export_are_there_to_press_on_the_model_card(imgui_ui):
    """The tour's Save and Export steps wait on ``clay_saved`` and ``clay_exported``;
    both buttons are drawn, enabled, with nothing selected."""
    ctx, tab, doc = _document(1)
    doc.select([])
    rig = _rig(imgui_ui, ctx)
    save = rig.find(label=f"{icons.SAVE} Save (Ctrl+S)")
    export = rig.find(label=f"{icons.DOWNLOAD} Export to the library")
    assert save.enabled and export.enabled


# -- the model card ------------------------------------------------------------------------


def test_the_counts_line_names_shapes_colours_vertices_faces_and_triangles():
    ctx, tab, doc = _document(2)
    line = doc_io.model_counts_line(doc)
    assert line == "2 shapes · 1 colour · 16 vertices · 12 faces · 24 triangles"
    doc.set_props(doc.objects[0].uid, visible=False)
    assert doc_io.model_counts_line(doc).startswith("1 shape (1 hidden) ·")
    assert doc_io.model_counts_line(bd.ClayDoc()) == "No shapes yet."


def test_the_counts_line_is_memoised_on_the_documents_revision(monkeypatch):
    ctx, tab, doc = _document(2)
    calls: list[int] = []
    real = doc_io.model_counts_line
    monkeypatch.setattr(doc_io, "model_counts_line", lambda d: (calls.append(1), real(d))[1])
    first = clay_inspector.counts_line(doc)
    for _ in range(5):
        assert clay_inspector.counts_line(doc) == first
    assert len(calls) == 1
    clay_tools.place_primitive(ctx, doc, "box")
    assert clay_inspector.counts_line(doc) != first and len(calls) == 2


# -- layout stability ------------------------------------------------------------------------
#
# The outliner is the Inspector's docked "All shapes" section now, so the press-and-release
# stability of its rows (the 2026-10-07 audit's layout jump) is pinned in
# ``test_inspector_dock.py``, next to every other claim about the dock.


# -- cost ------------------------------------------------------------------------------------


def _big_document(shapes: int):
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    doc = tab.doc
    objects = [
        bd.Obj(
            uid=bd.new_uid(),
            name=f"Box {i}",
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
            translation=np.array([float(i % 64), 0.5, float(i // 64)]),
        )
        for i in range(shapes)
    ]
    doc.add_objects(objects)
    return ctx, tab, doc


@pytest.mark.parametrize("shapes", [1, 100, 4096])
def test_the_inspector_does_no_per_shape_work_while_nothing_changes(imgui_ui, shapes, monkeypatch):
    """At 4,096 shapes a panel that walks the document every frame is the frame.
    Nothing selected (counts and the model card), one selected (the body) and all
    selected (the several-shapes body) each draw with the heavy readouts memoised."""
    ctx, tab, doc = _big_document(shapes)
    calls: list[int] = []
    real = doc_io.model_counts_line
    monkeypatch.setattr(doc_io, "model_counts_line", lambda d: (calls.append(1), real(d))[1])
    for selection in ([], [doc.objects[0].uid], [o.uid for o in doc.objects]):
        doc.select(selection)
        calls.clear()
        rig = _rig(imgui_ui, ctx, height=1800.0)
        for _ in range(4):
            rig.settle(1)
        assert len(calls) <= 1, f"{len(selection)} selected: the counts were rebuilt per frame"


@pytest.mark.perf
@pytest.mark.parametrize("shapes", [1, 100, 4096])
def test_the_inspector_frame_stays_cheap_at_any_document_size(imgui_ui, shapes):
    ctx, tab, doc = _big_document(shapes)
    # Measured 2026-10-09: 0.3 ms to 0.7 ms a frame in every state, at every size.
    budget = {1: 0.005, 100: 0.005, 4096: 0.010}[shapes]
    for selection in ([], [doc.objects[0].uid], [o.uid for o in doc.objects]):
        doc.select(selection)
        rig = _rig(imgui_ui, ctx, height=1800.0)
        samples = []
        for _ in range(12):
            start = time.perf_counter()
            rig.settle(1)
            samples.append(time.perf_counter() - start)
        median = statistics.median(samples)
        where = f"{shapes} shapes, {len(selection)} selected"
        assert median < budget, f"{where}: {median * 1000:.1f} ms"


def test_the_inspector_frame_is_bounded_even_under_contention(imgui_ui):
    """The perf lane's claim with a ceiling loose enough for a busy machine: 4,096
    shapes must not turn a frame into a second."""
    ctx, tab, doc = _big_document(4096)
    doc.select([doc.objects[0].uid])
    rig = _rig(imgui_ui, ctx, height=1800.0)
    start = time.perf_counter()
    for _ in range(5):
        rig.settle(1)
    assert (time.perf_counter() - start) / 5 < 0.5


# -- the retired pane's functions ------------------------------------------------------------


def test_properties_keeps_its_functions_so_the_old_doors_still_answer(imgui_ui):
    """25 test files and the pages call these by name; the tab strip and the legacy
    Object body are the part that went (``test_inspector_elements.py``)."""
    for name in (
        "_transform", "_dimensions", "_palette_row", "_material", "_document", "_outputs",
        "_generator", "_relations", "commit_median", "scale_fields", "_element_median",
    ):  # fmt: skip
        assert callable(getattr(clay_props, name)), name
