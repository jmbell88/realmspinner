"""The Clay Inspector's element state: Points, Edges and Faces.

Each test's name is the claim it makes. The panel is drawn through a real (GL-less) imgui
context and pressed by the same input queue a person uses (``_ui_driver``), so a button
wired to nothing, a control clipped past the column or a row that moves between a press
and its release fails here the way it would on screen.

What the old code did, test by test (the element state replaced the Object tab's body,
which a Done button sat over, and the Adjust card that floated over the viewport):

* the old body had no Select row, no Shape it card and no op buttons at all -- every
  ``##insp-select/`` / ``##insp-op/`` lookup fails on it, and so does the registry check;
* it drew the one-object Object tab (``position##bt``, ``size##bz``) in an element mode --
  the "no legacy body" assertions fail on it;
* the median row vanished with nothing picked and the cards under it moved up -- the
  stable-layout and greyed-median assertions fail on it;
* ``props.draw``, ``TABS``, ``tab_key``, ``legacy_object_body`` and ``ClayState.props_tab``
  existed -- the "retired" assertions fail on it;
* the live op's numbers were a card over the viewport (``panes/adjust.py``), not drawn by
  the inspector -- every ``last-inset`` lookup fails on it.
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
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import anchors, widgets
from realmspinner.studio.modes.clay import element_move, menutree
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay import state as clay_state
from realmspinner.studio.modes.clay.ui import inspector_actions
from realmspinner.studio.modes.clay.ui import inspector_elements as clay_elements
from realmspinner.studio.modes.clay.ui.panes import inspector as clay_inspector
from realmspinner.studio.modes.clay.ui.panes import props as clay_props
from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools

MODES = ("vertex", "edge", "face")

#: ``mode -> a one-pick selection of the first box``.
PICKS = {
    "vertex": lambda: el.ElementSel(verts=[0]),
    "edge": lambda: el.ElementSel(edges=[(0, 1)]),
    "face": lambda: el.ElementSel(faces=[0]),
}


class _Settings:
    def get(self, key, default=None):
        return default

    def set(self, key, value):
        pass


def _ctx():
    ctx = SimpleNamespace(
        state=SimpleNamespace(mode="clay", clay=None, preview={}, list_filters={}),
        settings=_Settings(),
        toasts=[],
        cache=SimpleNamespace(invalidate=lambda: None),
        busy=lambda key: False,
        submit=lambda *a, **k: True,
    )
    ctx.toast = lambda message, kind="info", *a: ctx.toasts.append((message, kind))
    ctx.clay_view = None
    return ctx


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(clay_mode, "recent_paths", lambda ctx: [])
    monkeypatch.setattr(widgets, "_SETTINGS", None)
    widgets._OPEN_REQUESTS.clear()
    yield
    widgets._OPEN_REQUESTS.clear()


@pytest.fixture
def imgui_ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _document(mode: str | None = "face", *, shapes: int = 1, picked: bool = False):
    """A ctx and a document of *shapes* grounded boxes, in *mode* with the last shape
    kept selected (the selection rule) and, with *picked*, its first element picked."""
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    for _ in range(shapes):
        clay_tools.place_primitive(ctx, tab.doc, "box")
    doc = tab.doc
    if mode is not None:
        doc.set_element_mode(mode)
        if picked:
            doc.set_element_sel(doc.objects[-1].uid, PICKS[mode]())
    return ctx, tab, doc


class Rig:
    """The Inspector drawn in a column of *width* x *height*, with a driver that presses it."""

    def __init__(self, imgui, ctx, *, width: float = 300.0, height: float = 4000.0):
        self.imgui = imgui
        self.ctx = ctx
        self.width = width
        self.height = height
        imgui.get_io().display_size = (1600.0, max(float(height), 950.0))
        self.driver = UiDriver(imgui, self._frame)
        self.draw = lambda: clay_inspector.draw(ctx)
        self.driver.frame()
        self.settle(3)

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
        imgui.set_next_window_size((self.width, self.height))
        imgui.begin("##inspector-host", None, flags)
        self.draw()
        imgui.end()
        imgui.end_frame()

    def settle(self, frames: int = 2):
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
    return Rig(imgui_ui, ctx, **kw)


def _state(ctx):
    return clay_mode.ensure(ctx)


def _shape_ops(mode: str) -> set[str]:
    """What Shape it must show in *mode*: the registry's ops for the mode minus the Select
    group (the Select row's) and Frame Selection (the header's View menu's)."""
    return {
        op.name
        for op in clay_ops.menu(mode)
        if op.name not in menutree.GROUPS["select"].ops and op.name != "frame"
    }


def _open_every_fold(mode: str) -> None:
    for key, _label, _ops in clay_elements.shape_groups(mode)[1]:
        widgets.request_open(f"clay/insp/el/{key}")


# -- the legacy is gone ---------------------------------------------------------------------


def test_the_tab_strip_and_the_old_object_body_are_gone_from_properties():
    """Old code: ``props.draw``, ``TABS``, ``tab_key``, ``legacy_object_body`` and the
    helpers only they used were still there (and still tested), drawn by nothing."""
    for name in (
        "draw", "TABS", "tab_key", "DEFAULT_TAB", "PROPS_FLOOR", "_body", "_no_single_object",
        "legacy_object_body", "_nothing_selected_text", "_element_summary",
        "element_summary_text", "_selected", "_identity",
    ):  # fmt: skip
        assert not hasattr(clay_props, name), name
    assert "props_tab" not in {f for f in clay_state.ClayState.__dataclass_fields__}


def test_the_adjust_card_pane_is_gone_and_the_viewport_no_longer_draws_it():
    """Old code: ``ui/panes/adjust.py`` drew the card over the viewport's corner."""
    import importlib.util
    from pathlib import Path

    from realmspinner.studio.modes.clay.ui import viewport

    assert importlib.util.find_spec("realmspinner.studio.modes.clay.ui.panes.adjust") is None
    assert "clay_adjust" not in Path(viewport.__file__).read_text(encoding="utf-8")


def test_properties_keeps_the_functions_the_element_state_and_the_pages_call():
    for name in (
        "_transform", "_dimensions", "_palette_row", "_material", "_document", "_outputs",
        "_generator", "_relations", "commit_median", "scale_fields", "_element_median",
        "element_measure_text", "element_median_cached", "_texture_slots", "_take_back_row",
        "_history", "_import_settings",
    ):  # fmt: skip
        assert callable(getattr(clay_props, name)), name


# -- the state itself --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_the_element_state_has_a_select_row_an_element_line_shape_it_and_the_page_links(
    imgui_ui, mode
):
    ctx, tab, doc = _document(mode)
    rig = _rig(imgui_ui, ctx)
    labels = rig.labels
    for op in clay_elements.select_ops(mode):
        assert f"{clay_elements.short_label(op)}##insp-select/{op.name}" in labels, op.name
    assert any("##insp-done" in label for label in labels)
    assert "##median##bmedian" in labels, "the median row is drawn even with nothing picked"
    assert "Extrude##insp-op/extrude" in labels
    assert any("##insp-page/colours" in label for label in labels)
    assert any("##insp-page/unwrap" in label for label in labels)
    # The old Object tab's body is not here.
    assert "position##bt" not in labels and "size##bz" not in labels
    assert "name##buildname" not in labels


@pytest.mark.parametrize(
    ("mode", "word"), [("vertex", "Points"), ("edge", "Edges"), ("face", "Faces")]
)
def test_the_title_names_the_mode_in_the_header_toolbars_words(
    imgui_ui, monkeypatch, mode, word
):
    ctx, tab, doc = _document(mode)
    seen: list[str] = []
    real = widgets.section
    monkeypatch.setattr(widgets, "section", lambda label: (seen.append(label), real(label))[1])
    _rig(imgui_ui, ctx)
    assert f"Editing {word.lower()}" in seen


def test_done_returns_to_shape_mode_and_the_shape_stays_selected(imgui_ui):
    ctx, tab, doc = _document("face", picked=True)
    uid = doc.objects[0].uid
    rig = _rig(imgui_ui, ctx)
    rig.click(contains="##insp-done")
    assert doc.element_mode == "object"
    assert doc.selection == {uid}, "the shape stays selected, as the selection rule says"
    assert doc.element_sel == {}
    assert "##size##insp" in rig.labels, "and the Shape state is what shows"


def test_done_calls_the_door_the_headers_shape_mode_uses(imgui_ui, monkeypatch):
    ctx, tab, doc = _document("edge")
    calls: list[str] = []
    real = doc.set_element_mode
    monkeypatch.setattr(doc, "set_element_mode", lambda mode: (calls.append(mode), real(mode))[1])
    rig = _rig(imgui_ui, ctx)
    rig.click(contains="##insp-done")
    assert calls == ["object"]


# -- the Select row ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_the_select_row_is_the_registrys_select_group_not_a_list_written_here(mode):
    ops = clay_elements.select_ops(mode)
    assert [op.name for op in ops] == [
        name for name in menutree.GROUPS["select"].ops if mode in clay_ops.get(name).modes
    ]
    assert [clay_elements.short_label(op) for op in ops] == ["All", "None", "Invert", "Linked"]


def test_select_all_and_invert_and_linked_run_the_registrys_ops(imgui_ui):
    ctx, tab, doc = _document("face", picked=True)
    obj = doc.objects[0]
    rig = _rig(imgui_ui, ctx)
    rig.click(label="All##insp-select/select-all")
    assert len(doc.element_sel[obj.uid].faces) == 6, "a box has six faces"
    rig.click(label="None##insp-select/select-none")
    assert doc.element_sel == {}
    rig.click(label="Invert##insp-select/select-invert")
    assert len(doc.element_sel[obj.uid].faces) == 6, "inverting nothing picks every face"


def test_a_select_button_with_nothing_to_act_on_is_greyed_with_the_registrys_sentence(imgui_ui):
    ctx, tab, doc = _document("face")
    rig = _rig(imgui_ui, ctx)
    linked = rig.find(label="Linked##insp-select/select-linked")
    assert not linked.enabled
    assert linked.reason == clay_ops.reason_for(clay_ops.get("select-linked"), doc)
    assert linked.reason, "greyed with a sentence"


# -- the element line -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "text"),
    [
        ("vertex", "Nothing selected. Click a point, or hold Shift and drag to select several."),
        ("edge", "Nothing selected. Click an edge, or hold Shift and drag to select several."),
        ("face", "Nothing selected. Click a face, or hold Shift and drag to select several."),
    ],
)
def test_with_nothing_picked_the_line_says_what_to_click_in_the_modes_own_noun(mode, text):
    ctx, tab, doc = _document(mode)
    assert clay_elements.element_line(doc) == text
    assert clay_elements.empty_text(mode) == text


def test_the_line_counts_what_is_picked_and_names_the_shapes_when_there_are_several():
    ctx, tab, doc = _document("face", shapes=2)
    first, second = doc.objects
    doc.set_element_sel(first.uid, el.ElementSel(faces=[0]))
    assert clay_elements.element_line(doc) == "1 face selected"
    doc.set_element_sel(first.uid, el.ElementSel(faces=[0, 1, 2]))
    assert clay_elements.element_line(doc) == "3 faces selected"
    doc.set_element_sel(second.uid, el.ElementSel(faces=[4]))
    assert clay_elements.element_line(doc) == "4 faces selected across 2 shapes"
    doc.set_props(second.uid, visible=False)
    assert clay_elements.element_line(doc) == "3 faces selected", "a hidden shape is not counted"


def test_the_drawn_line_is_the_count_and_the_empty_sentence_when_nothing_is_picked(
    imgui_ui, monkeypatch
):
    ctx, tab, doc = _document("face")
    drawn: list[str] = []
    real = widgets.muted_wrapped
    monkeypatch.setattr(widgets, "muted_wrapped", lambda text: (drawn.append(text), real(text))[1])
    rig = _rig(imgui_ui, ctx)
    assert clay_elements.empty_text("face") in drawn
    doc.set_element_sel(doc.objects[0].uid, el.ElementSel(faces=[0, 1, 2]))
    rig.settle(2)
    assert "3 faces selected" in drawn


def test_the_median_is_greyed_with_a_reason_when_nothing_is_picked_and_live_when_something_is(
    imgui_ui,
):
    ctx, tab, doc = _document("face")
    rig = _rig(imgui_ui, ctx)
    median = rig.find(label="##median##bmedian")
    assert not median.enabled and median.reason == clay_props.NO_MEDIAN_WHY
    doc.set_element_sel(doc.objects[0].uid, el.ElementSel(faces=[0]))
    rig.settle(2)
    median = rig.find(label="##median##bmedian")
    assert median.enabled


def test_typing_a_median_coordinate_moves_the_picks_in_one_undo_step(imgui_ui):
    ctx, tab, doc = _document("face", picked=True)
    obj = doc.objects[0]
    before = element_move.element_median_world(doc)
    positions = np.array(obj.mesh.positions)
    steps = len(doc.history)
    rig = _rig(imgui_ui, ctx)
    # The centre of a three-box field is the middle box: Y.
    rig.driver.click(rig.find(label="##median##bmedian"))
    rig.driver.type_text("5")
    assert len(doc.history) == steps, "typing alone changes nothing yet"
    rig.driver.press_key(imgui_ui.Key.enter)
    rig.settle(2)
    after = element_move.element_median_world(doc)
    assert float(after[1]) == pytest.approx(5.0, abs=1e-6), "the median is where it was typed"
    assert float(before[1]) != pytest.approx(5.0)
    assert float(after[0]) == pytest.approx(float(before[0])), "and X and Z were left alone"
    assert len(doc.history) == steps + 1, "one step for '5', not one per keystroke"
    assert doc.undo()
    assert np.array_equal(np.array(obj.mesh.positions), positions), "one Ctrl+Z puts it all back"


# -- Shape it ---------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_shape_it_shows_exactly_the_registrys_ops_for_the_mode_minus_the_select_group(
    imgui_ui, mode
):
    """Generated, not hand-listed: with every fold open, the op buttons drawn are
    ``menu(mode)`` less the Select group (and Frame Selection, the header's)."""
    ctx, tab, doc = _document(mode, picked=True)
    _open_every_fold(mode)
    rig = _rig(imgui_ui, ctx)
    drawn = {label.split("##insp-op/")[1] for label in rig.labels if "##insp-op/" in label}
    assert drawn == _shape_ops(mode)
    pure = {op.name for _k, _l, ops in inspector_actions.action_groups(mode) for op in ops}
    assert pure == _shape_ops(mode)


def test_the_faces_mode_op_set_is_not_the_object_modes():
    """The three modes differ: Faces has Inset and Subdivide, Points has Weld, Edges neither."""
    assert {"inset", "subdivide", "flip"} <= _shape_ops("face")
    assert "weld" in _shape_ops("vertex") and "weld" not in _shape_ops("face")
    assert "inset" not in _shape_ops("edge") and "inset" not in _shape_ops("vertex")


@pytest.mark.parametrize("mode", MODES)
def test_every_op_button_is_greyed_with_the_registrys_own_reason_for_each_mode(imgui_ui, mode):
    ctx, tab, doc = _document(mode)  # nothing picked: most are refused
    _open_every_fold(mode)
    rig = _rig(imgui_ui, ctx)
    by_label = {c.label: c for c in rig.driver.controls if c.visible}
    grey = 0
    for name in _shape_ops(mode):
        op = clay_ops.get(name)
        control = by_label[f"{op.label}##insp-op/{op.name}"]
        assert control.enabled == op.enabled(doc), name
        if not control.enabled:
            grey += 1
            assert control.reason and control.reason == clay_ops.reason_for(op, doc), name
    assert grey, "with nothing picked some ops are refused"
    delete = by_label[f"{clay_ops.get('delete').label}##insp-op/delete"]
    assert not delete.enabled and delete.reason, "Delete says why when nothing is picked"


def test_delete_in_an_element_mode_is_the_registrys_op_and_one_undo_step(imgui_ui):
    ctx, tab, doc = _document("face", picked=True)
    obj = doc.objects[0]
    steps = len(doc.history)
    before = len(obj.mesh.starts)
    rig = _rig(imgui_ui, ctx)
    rig.click(label="Delete##insp-op/delete")
    assert len(obj.mesh.starts) < before, "the picked face is gone"
    assert len(doc.history) == steps + 1


def test_a_bare_op_button_runs_the_op_through_the_menus_door(imgui_ui):
    ctx, tab, doc = _document("face", picked=True)
    obj = doc.objects[0]
    before = len(obj.mesh.starts)
    rig = _rig(imgui_ui, ctx)
    rig.click(label="Extrude##insp-op/extrude")
    assert len(obj.mesh.starts) > before, "extruding a box face adds the side faces"


def test_an_op_with_numbers_opens_inline_fields_and_apply_commits_it_in_one_undo_step(imgui_ui):
    ctx, tab, doc = _document("face", picked=True)
    obj = doc.objects[0]
    state = _state(ctx)
    steps = len(doc.history)
    before = len(obj.mesh.starts)
    rig = _rig(imgui_ui, ctx)
    rig.click(label=f"{clay_ops.get('inset').label}##insp-op/inset")
    assert state.inspector_form == "inset"
    assert any("##insp-apply/inset" in label for label in rig.labels), "the form is inline"
    assert len(doc.history) == steps, "opening the form pushes nothing"
    state.op_params["inset"]["thickness"] = 0.2
    rig.click(contains="##insp-apply/inset")
    assert len(obj.mesh.starts) > before
    assert len(doc.history) == steps + 1, "the whole op is one step"
    assert state.inspector_form == ""
    assert doc.recent_op is not None and doc.recent_op.params["thickness"] == 0.2


def test_a_form_that_stops_being_runnable_closes_itself(imgui_ui):
    ctx, tab, doc = _document("face", picked=True)
    state = _state(ctx)
    rig = _rig(imgui_ui, ctx)
    rig.click(label=f"{clay_ops.get('inset').label}##insp-op/inset")
    assert state.inspector_form == "inset"
    doc.clear_element_sel()
    rig.settle(2)
    assert state.inspector_form == ""


def test_closed_folds_do_not_evaluate_op_enabled_and_an_open_one_does(imgui_ui):
    ctx, tab, doc = _document("face", picked=True)
    visible, folded = clay_elements.shape_groups("face")
    folded_ops = {op.name for _k, _l, ops in folded for op in ops}
    shown_ops = {op.name for _k, _l, ops in visible for op in ops}
    assert folded_ops, "Faces has folded groups (Set Origin, Shading, Separate)"
    calls: list[str] = []
    originals = {op.name: op.enabled for op in clay_ops.OPS}

    def spy_for(name):
        def spy(d):
            calls.append(name)
            return originals[name](d)

        return spy

    for op in clay_ops.OPS:
        object.__setattr__(op, "enabled", spy_for(op.name))
    try:
        rig = _rig(imgui_ui, ctx)
        for _ in range(3):
            rig.settle(1)
        asked = set(calls) & folded_ops
        assert not asked, f"closed folds asked {sorted(asked)}"
        assert shown_ops <= set(calls), "the open groups are asked"
        widgets.request_open("clay/insp/el/separate")
        calls.clear()
        rig.settle(2)
        assert "separate-selection" in calls, "open, a fold asks"
        assert "shade-smooth" not in calls, "and only that fold"
    finally:
        for op in clay_ops.OPS:
            object.__setattr__(op, "enabled", originals[op.name])


def test_an_open_fold_and_the_card_above_it_share_one_cached_answer_per_revision(imgui_ui):
    ctx, tab, doc = _document("face", picked=True)
    _open_every_fold("face")
    rig = _rig(imgui_ui, ctx)
    calls: list[str] = []
    originals = {op.name: op.enabled for op in clay_ops.OPS}
    for op in clay_ops.OPS:
        object.__setattr__(
            op, "enabled", lambda d, n=op.name: (calls.append(n), originals[n](d))[1]
        )
    try:
        for _ in range(5):
            rig.settle(1)
        assert calls == [], "an unchanged document asks nothing, open folds included"
    finally:
        for op in clay_ops.OPS:
            object.__setattr__(op, "enabled", originals[op.name])


def test_a_save_in_flight_greys_every_op_with_one_shared_sentence(imgui_ui):
    ctx, tab, doc = _document("face", picked=True)
    tab.saving = True
    _open_every_fold("face")
    rig = _rig(imgui_ui, ctx)
    ops = [c for c in rig.driver.controls if "##insp-op/" in c.label or "##insp-select/" in c.label]
    assert ops and all(not c.enabled for c in ops)
    assert all(c.reason for c in ops)


def test_every_greyed_control_in_the_element_state_says_why(imgui_ui):
    ctx, tab, doc = _document("face")
    _open_every_fold("face")
    rig = _rig(imgui_ui, ctx)
    greyed = [c for c in rig.driver.controls if c.visible and not c.enabled]
    assert greyed
    assert [c.label for c in greyed if not c.reason] == []


# -- the Last action card -----------------------------------------------------------------------


def _inset(ctx, doc, thickness=0.1):
    return clay_ops.run(ctx, doc, clay_ops.get("inset"), thickness=thickness)


def test_the_last_action_card_shows_the_live_ops_own_fields_and_only_while_it_is_live(imgui_ui):
    ctx, tab, doc = _document("face", picked=True)
    rig = _rig(imgui_ui, ctx)
    assert not any("last-inset" in label for label in rig.labels), "nothing run yet"
    assert _inset(ctx, doc)
    rig.settle(2)
    shown = [label for label in rig.labels if "last-inset" in label]
    assert len(shown) >= 2, shown  # thickness, depth (and the region checkbox)
    assert any(label.endswith("last-inset-thickness") for label in shown)
    doc.set_props(doc.objects[0].uid, name="Renamed")
    rig.settle(2)
    assert not any("last-inset" in label for label in rig.labels), "a later edit hides it"


def test_editing_a_last_action_number_reruns_the_op_as_one_undo_step(imgui_ui):
    ctx, tab, doc = _document("face", picked=True)
    obj = doc.objects[0]
    base = len(doc.history)
    assert _inset(ctx, doc, 0.1)
    rig = _rig(imgui_ui, ctx)
    field = rig.find(contains="last-inset-thickness")
    rig.driver.click(field)
    rig.driver.type_text("0.3")
    rig.driver.press_key(rig.imgui.Key.enter)
    rig.settle(3)
    assert doc.recent_op.params["thickness"] == pytest.approx(0.3)
    assert _state(ctx).last_action_message == ""
    assert len(doc.history) == base + 1, "an adjust replaces the step, it does not add one"

    # Equal to having run the op at the final value in the first place.
    direct_ctx, _t, direct = _document("face", picked=True)
    final = doc.recent_op.params["thickness"]
    _inset(direct_ctx, direct, final)
    assert np.array_equal(
        np.array(obj.mesh.positions), np.array(direct.objects[0].mesh.positions)
    )
    assert doc.undo()
    assert np.array_equal(np.array(obj.mesh.positions), np.array(bp.box().positions))


def test_a_refused_adjust_prints_its_reason_in_the_card_and_keeps_the_last_good_result(
    imgui_ui, monkeypatch
):
    from realmspinner.kernels.mesh.elements import OpError

    index = next(i for i, op in enumerate(clay_ops.OPS) if op.name == "inset")
    real = clay_ops.OPS[index]

    def picky(ctx, doc, **params):
        if params["thickness"] > 0.5:
            raise OpError("Too thick for this face.")
        return real.run(ctx, doc, **params)

    import dataclasses

    patched = list(clay_ops.OPS)
    patched[index] = dataclasses.replace(real, run=picky)
    monkeypatch.setattr(clay_ops, "OPS", patched)

    ctx, tab, doc = _document("face", picked=True)
    obj = doc.objects[0]
    assert _inset(ctx, doc, 0.1)
    kept = np.array(obj.mesh.positions)
    steps = len(doc.history)
    rig = _rig(imgui_ui, ctx)
    field = rig.find(contains="last-inset-thickness")
    rig.driver.click(field)
    rig.driver.type_text("9")
    rig.driver.press_key(rig.imgui.Key.enter)
    rig.settle(3)
    assert _state(ctx).last_action_message == "Too thick for this face."
    assert np.array_equal(np.array(obj.mesh.positions), kept)
    assert len(doc.history) == steps
    assert ctx.toasts == [], "the reason goes to the card, not to a toast"


def test_the_card_scrolls_itself_into_view_when_the_dock_leaves_the_body_short(imgui_ui):
    """With the All shapes dock open the body is short and the card sits under its fold, so
    the op you just ran would be run with its numbers out of sight. Old code: the card was
    a viewport overlay and always in view."""
    ctx, tab, doc = _document("face", picked=True, shapes=3)
    widgets.request_open(clay_inspector.DOCK_KEY)
    rig = _rig(imgui_ui, ctx, height=700.0)
    assert not any("last-inset" in label for label in rig.labels)
    assert _inset(ctx, doc)
    rig.settle(4)
    assert any(label.endswith("last-inset-thickness") for label in rig.labels), rig.labels
    # And an adjust, which keeps the card live, does not yank the scroll position again.
    field = rig.find(contains="last-inset-thickness")
    top = field.rect[1]
    rig.driver.click(field)
    rig.driver.type_text("2")
    rig.settle(3)
    assert rig.find(contains="last-inset-thickness").rect[1] == pytest.approx(top, abs=1.0)


def test_the_card_is_gone_in_shape_mode_because_a_mode_change_ends_the_live_op(imgui_ui):
    ctx, tab, doc = _document("face", picked=True)
    assert _inset(ctx, doc)
    rig = _rig(imgui_ui, ctx)
    assert any("last-inset" in label for label in rig.labels)
    rig.click(contains="##insp-done")
    assert not any("last-inset" in label for label in rig.labels)
    assert _state(ctx).last_action_message == ""


# -- layout ----------------------------------------------------------------------------------


def test_a_controls_rect_does_not_change_when_the_picks_do(imgui_ui):
    """Old code: the median row only existed with something picked, so every card under
    it moved by a row the moment the first pick landed."""
    ctx, tab, doc = _document("face")
    _open_every_fold("face")
    rig = _rig(imgui_ui, ctx)
    watched = ("Extrude##insp-op/extrude", "Delete##insp-op/delete", "##median##bmedian")
    empty = {label: rig.find(label=label).rect for label in watched}
    obj = doc.objects[0]
    for picks in (
        el.ElementSel(faces=[0]),
        el.ElementSel(faces=[0, 1, 2, 3]),
        el.ElementSel(faces=[]),
    ):
        doc.set_element_sel(obj.uid, picks) if len(picks.faces) else doc.clear_element_sel()
        rig.settle(3)
        for label in watched:
            assert rig.find(label=label).rect == empty[label], (label, picks)


def test_a_pick_between_press_and_release_does_not_move_the_button_under_the_pointer(imgui_ui):
    ctx, tab, doc = _document("face", picked=True)
    obj = doc.objects[0]
    rig = _rig(imgui_ui, ctx)
    button = rig.find(label="Extrude##insp-op/extrude")
    where = button.centre
    driver = rig.driver
    driver.frame(where, down=False)
    pressed = rig.find(label="Extrude##insp-op/extrude").rect
    driver.frame(where, down=True)
    doc.set_element_sel(obj.uid, el.ElementSel(faces=[0, 1]))  # picks change mid-press
    before = len(obj.mesh.starts)
    driver.frame(where, down=True)
    held = rig.find(label="Extrude##insp-op/extrude").rect
    driver.frame(where, down=False)
    assert held == pressed, "the button moved under the pointer"
    assert len(obj.mesh.starts) > before, "and the release still pressed it"


@pytest.mark.parametrize("width", [300.0, 200.0])
@pytest.mark.parametrize("mode", MODES)
def test_the_element_state_fits_the_column_at_the_sidebars_width_and_its_floor(
    imgui_ui, mode, width
):
    """Nothing is drawn past the column's right edge, with every fold open and a live
    op's card showing, at 300 px and at the 200 px a squeezed window gives it."""
    ctx, tab, doc = _document(mode, picked=True)
    if mode == "face":
        assert _inset(ctx, doc)
    _open_every_fold(mode)
    rig = _rig(imgui_ui, ctx, width=width)
    over = [
        (c.label, round(c.rect[0] + c.rect[2], 1))
        for c in rig.driver.controls
        if c.visible and c.rect[0] + c.rect[2] > width + 1.0
    ]
    assert not over, over


@pytest.mark.parametrize("dock", [False, True])
def test_the_element_state_fits_a_950_px_window_with_the_dock_closed_and_open(imgui_ui, dock):
    """The folds a closed dock leaves room for are all reachable: the pane itself never
    scrolls (the body does, in its own child) and the dock keeps its header on screen."""
    ctx, tab, doc = _document("face", picked=True, shapes=3)
    if dock:
        widgets.request_open(clay_inspector.DOCK_KEY)
    rig = _rig(imgui_ui, ctx, height=830.0)
    drawn = {c.label: c for c in rig.driver.controls if c.visible}
    assert "Extrude##insp-op/extrude" in drawn
    for control in drawn.values():
        assert control.rect[1] + control.rect[3] <= 830.0 + 1.0, control.label


# -- the generated-not-hand-listed gate -------------------------------------------------------


def test_an_op_registered_into_a_menutree_group_shows_up_in_shape_it_with_no_edit_here(
    monkeypatch,
):
    fake = clay_ops.Op(
        name="zz-fake",
        label="Zz Fake",
        modes=("face",),
        run=lambda ctx, doc: True,
    )
    monkeypatch.setattr(clay_ops, "OPS", [*clay_ops.OPS, fake])
    mesh = menutree.GROUPS["mesh"]
    monkeypatch.setitem(
        menutree.GROUPS, "mesh", menutree.Group(mesh.label, (*mesh.ops, "zz-fake"))
    )
    visible, _folded = clay_elements.shape_groups("face")
    assert "zz-fake" in {op.name for _k, _l, ops in visible for op in ops}


# -- cost -------------------------------------------------------------------------------------


def _big_document(shapes: int):
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    doc = tab.doc
    doc.add_objects(
        [
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
    )
    doc.select([doc.objects[0].uid])
    doc.set_element_mode("face")
    doc.set_element_sel(doc.objects[0].uid, el.ElementSel(faces=[0]))
    return ctx, tab, doc


@pytest.mark.parametrize("shapes", [1, 100, 4096])
def test_the_element_state_does_no_per_shape_work_while_nothing_changes(
    imgui_ui, shapes, monkeypatch
):
    """At 4,096 shapes a panel that walks the document every frame is the frame: the
    median and the measure are memoised on ``rev``, the gates are asked once per change."""
    ctx, tab, doc = _big_document(shapes)
    median_calls: list[int] = []
    real = element_move.element_median_world
    monkeypatch.setattr(
        element_move, "element_median_world", lambda d: (median_calls.append(1), real(d))[1]
    )
    rig = _rig(imgui_ui, ctx, height=1800.0)
    median_calls.clear()
    for _ in range(4):
        rig.settle(1)
    assert median_calls == [], "the median is read from the memo while nothing changes"


@pytest.mark.perf
@pytest.mark.parametrize("shapes", [1, 100, 4096])
def test_the_element_state_frame_stays_cheap_at_any_document_size(imgui_ui, shapes):
    ctx, tab, doc = _big_document(shapes)
    # Measured 2026-10-09: 0.36 to 0.38 ms a frame at 1, 100 and 4,096 shapes, face mode with
    # one face picked; the budgets are the Inspector's own (test_inspector.py) with headroom.
    budget = {1: 0.006, 100: 0.006, 4096: 0.012}[shapes]
    rig = _rig(imgui_ui, ctx, height=1800.0)
    samples = []
    for _ in range(12):
        start = time.perf_counter()
        rig.settle(1)
        samples.append(time.perf_counter() - start)
    median = statistics.median(samples)
    assert median < budget, f"{shapes} shapes: {median * 1000:.1f} ms"


def test_the_element_state_frame_is_bounded_even_under_contention(imgui_ui):
    ctx, tab, doc = _big_document(4096)
    rig = _rig(imgui_ui, ctx, height=1800.0)
    start = time.perf_counter()
    for _ in range(5):
        rig.settle(1)
    assert (time.perf_counter() - start) / 5 < 0.5


# -- Esc ---------------------------------------------------------------------------------------


def test_escape_stages_page_then_picks_then_mode_then_shapes_with_the_element_state_up(imgui_ui):
    ctx, tab, doc = _document("face", picked=True)
    state = _state(ctx)
    uid = doc.objects[0].uid
    state.inspector_page = "colours"
    clay_mode._escape(state, tab, doc)
    assert state.inspector_page == "" and doc.element_sel and doc.element_mode == "face"
    clay_mode._escape(state, tab, doc)
    assert not doc.element_sel and doc.element_mode == "face"
    rig = _rig(imgui_ui, ctx)
    assert any("##insp-done" in label for label in rig.labels), "still the element state"
    clay_mode._escape(state, tab, doc)
    assert doc.element_mode == "object" and doc.selection == {uid}
    clay_mode._escape(state, tab, doc)
    assert doc.selection == set()


# -- Bevel Edges, restored in Landing 12 -------------------------------------------------------
#
# All four fail on the tree before the restore: ``clay_ops.get("bevel")`` raises KeyError and
# the Edges ``Shape it`` card has no ``##insp-op/bevel`` button.


def test_edges_mode_shape_it_has_a_bevel_button_with_its_key_and_no_other_mode_does(imgui_ui):
    assert "bevel" in _shape_ops("edge")
    assert "bevel" not in _shape_ops("face") and "bevel" not in _shape_ops("vertex")
    ctx, tab, doc = _document("edge", picked=True)
    rig = _rig(imgui_ui, ctx)
    control = rig.find(label="Bevel Edges...##insp-op/bevel")
    assert control.enabled, "an edge is picked, so Bevel is live"
    tip = inspector_actions.tooltip_for(clay_ops.get("bevel"))
    assert "Ctrl+B" in tip, "the key is in the tooltip"


def test_bevel_is_greyed_with_a_plain_reason_when_no_edge_is_picked(imgui_ui):
    ctx, tab, doc = _document("edge")  # edge mode, nothing picked
    rig = _rig(imgui_ui, ctx)
    control = rig.find(label="Bevel Edges...##insp-op/bevel")
    bevel = clay_ops.get("bevel")
    assert not control.enabled
    assert control.reason and control.reason == clay_ops.reason_for(bevel, doc)


def test_the_bevel_form_opens_inline_with_a_width_field_and_apply_is_one_undo_step(imgui_ui):
    ctx, tab, doc = _document("edge", picked=True)
    obj = doc.objects[0]
    state = _state(ctx)
    steps = len(doc.history)
    faces = len(obj.mesh.starts)
    rig = _rig(imgui_ui, ctx)
    rig.click(label="Bevel Edges...##insp-op/bevel")
    assert state.inspector_form == "bevel"
    assert any("width" in label and "bevel" in label for label in rig.labels), rig.labels
    assert len(doc.history) == steps, "opening the form pushes nothing"
    state.op_params["bevel"]["width"] = 0.1
    rig.click(contains="##insp-apply/bevel")
    assert len(obj.mesh.starts) > faces, "the bevel adds its strip"
    assert len(doc.history) == steps + 1, "the whole op is one step"
    assert doc.recent_op is not None and doc.recent_op.params["width"] == 0.1
    assert doc.undo()
    assert len(obj.mesh.starts) == faces


# -- Dissolve, Bridge Loops, Fill Hole and Collapse, restored 2026-10-13 -----------------------
#
# They sit in a closed Topology fold, not the open Shape it card: the card is the one the
# 1600x950 dock test measures, and four more buttons pushed Delete 17 px past it.


def test_the_four_topology_ops_are_in_a_closed_fold_and_not_on_the_shape_it_card():
    for mode in ("vertex", "edge", "face"):
        visible, folded_groups = clay_elements.shape_groups(mode)
        open_ops = {op.name for _k, _l, ops in visible for op in ops}
        folded = {op.name for key, _l, ops in folded_groups for op in ops if key == "topology"}
        assert not open_ops & {"dissolve", "bridge", "fill-hole", "collapse"}, mode
        assert folded, mode
    assert {"dissolve", "bridge", "fill-hole", "collapse"} == {
        op.name for _k, _l, ops in clay_elements.shape_groups("edge")[1] for op in ops
    } & {"dissolve", "bridge", "fill-hole", "collapse"}
