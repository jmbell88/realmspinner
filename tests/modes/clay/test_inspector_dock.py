"""The outliner lives inside the Clay Inspector, as the "All shapes (N)" dock.

Each test's name is the claim it makes. The dock is driven through a real (GL-less)
imgui context by the same input queue a person uses (``_ui_driver``), so a row wired
to nothing, a drag that lands nowhere or a region that moves between a press and its
release fails here the way it would on screen.

The tests for *changed behaviour* name what the old code did beside them:

* the right column was two slots, ``clay-outliner`` over ``clay-inspector``, and the
  outliner had its own ``draw`` -- the slot-id and ``hasattr`` assertions fail on it;
* every row carried a trash can (``##del``) that deleted without a word -- the
  "no trash" assertions fail on it;
* a closed dock draws no rows -- the old pane drew its whole list whatever else was
  on screen, so the ``_row`` spy assertions fail on it.

Behaviour that did not change (rename, drag-drop, solo, filter, range select) is ported
here from the pane's old tests to run through the embedded draw: the pure halves
(``_range``, ``_apply_drop``, ``_visible_rows``) stay pinned by ``test_clay_outliner``,
``test_outliner_tree`` and the audit files, which call the same functions.
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
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import anchors, icons, widgets
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.ui.panes import inspector as clay_inspector
from realmspinner.studio.modes.clay.ui.panes import outliner as clay_outliner
from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools

HOST_H = 830.0  # about what a 950 px window leaves the right column


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


def _document(shapes: int = 3):
    """A ctx and a document of *shapes* boxes; the last one placed is selected."""
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    for _ in range(shapes):
        clay_tools.place_primitive(ctx, tab.doc, "box")
    return ctx, tab, tab.doc


def _big_document(shapes: int):
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    tab.doc.add_objects(
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
    return ctx, tab, tab.doc


class Rig:
    """The Inspector drawn in a 300 px column, with a driver that presses it."""

    def __init__(self, imgui, ctx, *, height: float = HOST_H):
        self.imgui = imgui
        self.ctx = ctx
        self.height = height
        imgui.get_io().display_size = (1600.0, max(height, 950.0))
        self.driver = UiDriver(imgui, self._frame)
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
        imgui.set_next_window_size((300, self.height))
        imgui.begin("##inspector-host", None, flags)
        clay_inspector.draw(self.ctx)
        # Read before ``end``: how far the *pane itself* could scroll. It never should --
        # the body and the dock add up to the pane, and each scrolls on its own.
        self.pane_scroll_max = imgui.get_scroll_max_y()
        imgui.end()
        imgui.end_frame()

    def settle(self, frames: int = 2):
        return self.driver.settle(frames)

    @property
    def rows(self):
        found = [c for c in self.driver.controls if c.label.endswith("##row") and c.visible]
        return sorted(found, key=lambda c: c.rect[1])

    def row(self, name: str):
        return next(c for c in self.rows if c.label == f"{name}##row")

    def named(self, label: str):
        return [c for c in self.driver.controls if c.label == label and c.visible]

    def click(self, control):
        self.driver.click(control)
        self.settle(2)

    def hold(self, key, down: bool):
        self.imgui.get_io().add_key_event(key, down)

    def drag(self, source, to):
        sx, sy = source.centre
        dx, dy = to
        driver = self.driver
        driver.frame((sx, sy), down=False)
        driver.frame((sx, sy), down=True)
        for step in (0.25, 0.5, 0.75, 1.0):
            driver.frame((sx + (dx - sx) * step, sy + (dy - sy) * step), down=True)
        driver.frame((dx, dy), down=True)
        driver.frame((dx, dy), down=False)
        self.settle(2)

    def snapshot(self):
        return anchors.rect("clay/outliner"), [c.rect for c in self.rows]


def _open_dock(ctx):
    widgets.request_open(clay_inspector.DOCK_KEY)


def _state(ctx):
    return clay_mode.ensure(ctx)


# -- one column, one pane ---------------------------------------------------------------


def test_the_outliner_lives_inside_the_inspector_and_is_no_longer_a_slot_or_a_pane():
    """Old code: the right column was ``["clay-outliner", "clay-inspector"]`` and the
    outliner module had a ``draw`` the skeleton called."""
    from realmspinner.studio import skeletons

    ids = [slot.id for slot in skeletons.clay(SimpleNamespace())["right"].slots]
    assert ids == ["clay-inspector"]
    assert not hasattr(clay_outliner, "draw"), "only draw_embedded is left"
    assert callable(clay_outliner.draw_embedded)


def test_the_all_shapes_header_counts_the_shapes_and_keeps_its_id_as_the_count_changes(
    imgui_ui, monkeypatch
):
    """The tree node's open state is keyed by its label; a count in the visible text
    with no ``##`` id would shut the dock every time a shape was added."""
    seen: list[str] = []
    real = imgui_ui.collapsing_header

    def spy(label, *args):
        seen.append(label)
        return real(label, *args)

    monkeypatch.setattr(imgui_ui, "collapsing_header", spy)
    ctx, tab, doc = _document(3)
    rig = Rig(imgui_ui, ctx)
    assert "All shapes (3)##clay-all-shapes" in seen
    clay_tools.place_primitive(ctx, doc, "box")
    rig.settle(2)
    assert "All shapes (4)##clay-all-shapes" in seen
    assert {s.split("##")[1] for s in seen if s.startswith("All shapes")} == {"clay-all-shapes"}


# -- closed and open ----------------------------------------------------------------------


def test_a_closed_dock_is_one_header_row_and_runs_no_row_drawing(imgui_ui, monkeypatch):
    """Old code: the pane drew every row of the list whatever else was on screen. A shape
    is selected, so the dock defaults closed."""
    calls: list[str] = []
    monkeypatch.setattr(clay_outliner, "_body", lambda *a, **k: calls.append("body"))
    monkeypatch.setattr(clay_outliner, "_row", lambda *a, **k: calls.append("row"))
    ctx, tab, doc = _document(3)
    assert len(doc.selection) == 1
    rig = Rig(imgui_ui, ctx)
    for _ in range(4):
        rig.settle(1)
    assert calls == [], "a closed header must not run the outliner"
    assert rig.rows == []
    assert anchors.rect("clay/outliner") is not None, "the header is marked for the tour"
    assert _state(ctx).dock_open is False


def test_an_open_dock_shows_a_row_for_every_shape_in_document_order(imgui_ui):
    ctx, tab, doc = _document(3)
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx)
    assert [c.label for c in rig.rows] == [f"{o.name}##row" for o in doc.objects]
    assert any(c.label == "Solo##claysolo" or "claysolo" in c.label for c in rig.driver.controls)
    assert rig.named(f"{icons.EYE} Show all##clayshowall"), "the label the icon test pins"


def test_the_dock_defaults_open_with_nothing_selected_and_closed_with_a_shape_selected(imgui_ui):
    ctx, tab, doc = _document(3)
    doc.select([])
    assert Rig(imgui_ui, ctx).rows, "nothing selected: the list is the useful thing"
    assert _state(ctx).dock_open is True


def test_an_empty_document_opens_the_dock_to_one_line_not_a_pinned_empty_list(imgui_ui):
    ctx, tab, doc = _document(0)
    rig = Rig(imgui_ui, ctx)
    assert rig.rows == []
    assert _state(ctx).dock_open is True
    assert anchors.rect("clay/outliner") is not None, "the tour points at it from an empty model"
    compact = clay_inspector.dock_region_height(HOST_H, 30.0, 70.0, 28.0, 180.0, compact=True)
    assert compact == 28.0


def test_the_open_or_closed_choice_is_remembered_across_selections(imgui_ui):
    ctx, tab, doc = _document(3)
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx)
    assert rig.rows
    doc.select([])
    rig.settle(2)
    doc.select([doc.objects[0].uid])
    rig.settle(2)
    assert rig.rows, "selecting did not close the dock the user opened"


def test_clicking_the_header_opens_and_closes_the_dock(imgui_ui):
    ctx, tab, doc = _document(3)
    rig = Rig(imgui_ui, ctx)
    assert not rig.rows
    x, y, w, h = anchors.rect("clay/outliner")
    rig.driver.frame((x + 40.0, y + h * 0.5), down=False)
    rig.driver.frame((x + 40.0, y + h * 0.5), down=True)
    rig.driver.frame((x + 40.0, y + h * 0.5), down=False)
    rig.driver.frame()
    rig.settle(3)
    assert rig.rows, "one click opened it"


def test_the_dock_header_has_a_help_button_onto_the_outliner_section(imgui_ui):
    ctx, tab, doc = _document(3)
    rig = Rig(imgui_ui, ctx)
    help_button = rig.named(f"{icons.INFO}##help-clay-outliner")
    assert len(help_button) == 1
    header = anchors.rect("clay/outliner")
    assert header[1] <= help_button[0].rect[1] <= header[1] + header[3], "on the header row"


# -- the region is pinned -------------------------------------------------------------------


def test_the_open_region_is_a_pinned_share_of_the_pane_with_at_least_five_rows():
    row, header, bars, floor = 28.0, 34.0, 70.0, 180.0
    tall = clay_inspector.dock_region_height(800.0, header, bars, row, floor, compact=False)
    assert tall == pytest.approx(800.0 * clay_inspector.DOCK_SHARE), "the share of the pane"
    assert 0.25 <= clay_inspector.DOCK_SHARE <= 0.35, (
        "retuned from 0.40 in Landing 9: the element state's Shape it card is 580 px, and "
        "an open dock at 40% cut it across its second row at 1600x950"
    )
    mid = clay_inspector.dock_region_height(500.0, header, bars, row, floor, compact=False)
    assert mid >= bars + 5 * row, "never fewer than five rows under the bars"
    assert mid <= 500.0 - header - floor + 1e-9, "and the cards above keep their floor"
    short = clay_inspector.dock_region_height(260.0, header, bars, row, floor, compact=False)
    assert short == bars + 2 * row, "a pane too short for both keeps two rows; the cards scroll"


def test_the_pinned_height_depends_on_the_pane_alone_not_on_anything_the_selection_changes():
    args = (800.0, 34.0, 70.0, 28.0, 180.0)
    assert clay_inspector.dock_region_height(*args, compact=False) == (
        clay_inspector.dock_region_height(*args, compact=False)
    )
    import inspect

    params = set(inspect.signature(clay_inspector.dock_region_height).parameters)
    assert params == {"avail", "header", "bars", "row", "body_floor", "compact"}


def test_the_header_and_the_rows_stay_put_when_the_selection_count_changes(imgui_ui):
    """Old code: the outliner was its own pane and the inspector's state change could
    not move it -- but the new dock shares a pane with the morphing body, so this is
    the claim that has to hold: the 2026-10-07 audit's layout jump swallowed Ctrl and
    Shift clicks."""
    ctx, tab, doc = _document(5)
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx)
    baseline = rig.snapshot()
    assert baseline[0] is not None and baseline[1]
    uids = [o.uid for o in doc.objects]
    for selection in ([], uids[:1], uids[:3], uids, uids[1:2]):
        doc.select(selection)
        rig.settle(2)
        assert rig.snapshot() == baseline, f"moved with {len(selection)} selected"


def test_the_header_and_the_rows_stay_put_when_the_inspector_changes_state(imgui_ui):
    ctx, tab, doc = _document(4)
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx)
    baseline = rig.snapshot()
    doc.set_element_mode("face")
    rig.settle(2)
    assert rig.snapshot() == baseline, "an element mode swaps the whole body above"
    doc.set_element_mode("object")
    doc.select([doc.objects[0].uid])
    rig.settle(2)
    assert rig.snapshot() == baseline


def test_a_row_does_not_move_between_the_press_and_the_release_when_the_selection_changes(
    imgui_ui,
):
    """Ported from the two-pane version of this test (the outliner host above the
    inspector host): the same press, the same selection change between press and
    release, now through the one pane."""
    ctx, tab, doc = _document(3)
    _open_dock(ctx)
    doc.select([doc.objects[0].uid])
    rig = Rig(imgui_ui, ctx)
    driver = rig.driver
    target = rig.rows[-1]
    x, y = target.centre
    driver.frame((x, y), down=False)
    driver.frame((x, y), down=True)
    press = next(c.rect for c in driver.controls if c.label == target.label)
    doc.select([o.uid for o in doc.objects])  # an agent, a key
    driver.frame((x, y), down=False)
    release = next(c.rect for c in driver.controls if c.label == target.label)
    assert press == release, "the row moved under the pointer when the inspector changed state"
    doc.select([])
    driver.settle(2)
    assert next(c.rect for c in driver.controls if c.label == target.label) == press


def test_a_page_takes_the_whole_pane_and_the_dock_returns_with_the_main_view(imgui_ui):
    ctx, tab, doc = _document(3)
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx)
    assert rig.rows
    _state(ctx).inspector_page = "colours"
    rig.settle(2)
    assert rig.rows == [] and anchors.rect("clay/outliner") is None, (
        "the Unwrap canvas wants all of the height, so no page shares it with a list"
    )
    _state(ctx).inspector_page = ""
    rig.settle(2)
    assert rig.rows and anchors.rect("clay/outliner") is not None


def test_the_body_above_the_dock_scrolls_in_its_own_child_and_the_dock_does_not(imgui_ui):
    ctx, tab, doc = _document(3)
    doc.select([])
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx, height=330.0)  # far too short for the whole nothing-selected body
    assert rig.rows, "the dock still shows rows"
    header = anchors.rect("clay/outliner")
    assert header[1] + header[3] <= 330.0, "and the header is inside the pane, not scrolled off"


@pytest.mark.parametrize("height", [330.0, 600.0, HOST_H, 1400.0])
@pytest.mark.parametrize("opened", [False, True])
def test_the_pane_itself_never_scrolls_at_any_height_open_or_closed(imgui_ui, height, opened):
    """The body child and the dock must add up to the pane, to the pixel, or the pane
    grows a second scrollbar next to the body's own."""
    ctx, tab, doc = _document(5)
    if opened:
        _open_dock(ctx)
    for selection in ([], [doc.objects[0].uid], [o.uid for o in doc.objects]):
        doc.select(selection)
        rig = Rig(imgui_ui, ctx, height=height)
        assert rig.pane_scroll_max == 0.0, f"{len(selection)} selected, open={opened}"


# -- every old behaviour, through the embedded draw ------------------------------------------


def test_a_row_click_selects_ctrl_toggles_and_shift_extends_a_range(imgui_ui):
    ctx, tab, doc = _document(4)
    _open_dock(ctx)
    doc.select([])
    rig = Rig(imgui_ui, ctx)
    uids = [o.uid for o in doc.objects]
    names = [o.name for o in doc.objects]
    rig.click(rig.row(names[0]))
    assert doc.selection == {uids[0]}
    rig.hold(imgui_ui.Key.mod_ctrl, True)
    rig.click(rig.row(names[2]))
    rig.hold(imgui_ui.Key.mod_ctrl, False)
    assert doc.selection == {uids[0], uids[2]}, "Ctrl toggles"
    rig.click(rig.row(names[1]))
    rig.hold(imgui_ui.Key.mod_shift, True)
    rig.click(rig.row(names[3]))
    rig.hold(imgui_ui.Key.mod_shift, False)
    assert doc.selection == {uids[1], uids[2], uids[3]}, "Shift extends from the anchor"


def test_clicking_a_row_in_an_element_mode_returns_to_object_mode(imgui_ui):
    ctx, tab, doc = _document(3)
    _open_dock(ctx)
    doc.set_element_mode("face")
    rig = Rig(imgui_ui, ctx)
    other = doc.objects[0]
    rig.click(rig.row(other.name))
    assert doc.element_mode == "object"
    assert doc.selection == {other.uid}


def test_double_clicking_a_row_renames_it_as_one_undo_step(imgui_ui):
    ctx, tab, doc = _document(3)
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx)
    obj = doc.objects[0]
    steps = len(doc.history)
    target = rig.row(obj.name)
    rig.driver.click(target)
    rig.driver.click(target)
    rig.settle(2)
    assert clay_outliner.renaming_uid(_state(ctx), tab.uid) == obj.uid
    rig.driver.type_text("Door")
    rig.driver.press_key(imgui_ui.Key.enter)
    rig.settle(2)
    assert "Door" in obj.name
    assert len(doc.history) == steps + 1, "one step for the name, not one per letter"
    assert clay_outliner.renaming_uid(_state(ctx), tab.uid) == 0


def test_dragging_a_row_onto_the_middle_of_another_reparents_it(imgui_ui):
    ctx, tab, doc = _document(3)
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx)
    child, parent = doc.objects[0], doc.objects[2]
    steps = len(doc.history)
    target = rig.row(parent.name)
    rig.drag(rig.row(child.name), target.centre)
    assert doc.by_uid(child.uid).parent == parent.uid
    assert len(doc.history) == steps + 1


def test_dragging_a_row_onto_the_top_quarter_of_another_reorders_it_beside_it(imgui_ui):
    ctx, tab, doc = _document(3)
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx)
    first, last = doc.objects[0], doc.objects[2]
    target = rig.row(first.name)
    x, y, w, h = target.rect
    rig.drag(rig.row(last.name), (x + w * 0.5, y + h * 0.1))
    assert [o.uid for o in doc.objects][0] == last.uid
    assert doc.by_uid(last.uid).parent is None


def test_dragging_is_refused_while_a_save_is_in_flight(imgui_ui):
    ctx, tab, doc = _document(3)
    _open_dock(ctx)
    tab.saving = True
    rig = Rig(imgui_ui, ctx)
    child, parent = doc.objects[0], doc.objects[2]
    rig.drag(rig.row(child.name), rig.row(parent.name).centre)
    assert doc.by_uid(child.uid).parent is None


def test_solo_hides_every_other_shape_and_show_all_brings_them_back(imgui_ui):
    ctx, tab, doc = _document(3)
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx)
    keep = doc.objects[1]
    doc.select([keep.uid])
    rig.settle(2)
    rig.click(rig.named(f"{icons.EYE} Solo##claysolo")[0])
    assert [o.visible for o in doc.objects] == [False, True, False]
    rig.click(rig.named(f"{icons.EYE} Show all##clayshowall")[0])
    assert all(o.visible for o in doc.objects)


def test_the_eye_toggles_visibility_as_one_undo_step(imgui_ui):
    ctx, tab, doc = _document(3)
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx)
    steps = len(doc.history)
    eyes = rig.named(f"{icons.EYE}##vis")
    assert len(eyes) == 3
    rig.click(eyes[1])
    assert [o.visible for o in doc.objects] == [True, False, True]
    assert len(doc.history) == steps + 1


def test_the_find_box_narrows_the_list_by_name_and_says_so_when_nothing_matches(imgui_ui):
    ctx, tab, doc = _document(0)
    for name in ("Wall", "Roof", "Door", "Window", "Chimney", "Step", "Fence", "Gate"):
        obj = clay_tools.place_primitive(ctx, doc, "box")
        doc.set_props(obj.uid, name=name)
    _open_dock(ctx)
    # Tall enough that all eight rows are on screen whatever share of the pane the open
    # dock takes (it was 40% and is 28% since the element state; the claim is the filter's).
    rig = Rig(imgui_ui, ctx, height=1600.0)
    assert len(rig.rows) == 8
    # ``widgets.input_text`` is not in the control census, so the box is found by where
    # the dock puts it: the first thing under the header.
    hx, hy, hw, hh = anchors.rect("clay/outliner")
    imgui = imgui_ui
    box_y = hy + hh + imgui.get_style().item_spacing.y + imgui.get_frame_height() * 0.5
    for down in (False, True, False):
        rig.driver.frame((hx + 60.0, box_y), down=down)
    rig.settle(1)
    rig.driver.type_text("oor")
    rig.settle(2)
    assert [c.label for c in rig.rows] == ["Door##row"]
    rig.driver.type_text("zzz")
    rig.settle(2)
    assert rig.rows == []


def test_the_row_menu_opens_from_the_more_button_and_deletes_the_shape(imgui_ui):
    """Old code: a trash can on every row deleted without a menu. The can is gone;
    the menu it duplicated is one click away."""
    ctx, tab, doc = _document(3)
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx)
    assert not [c for c in rig.driver.controls if c.label.endswith("##del")], "no trash can"
    assert not any(icons.TRASH in c.label for c in rig.rows)
    steps = len(doc.history)
    victim = doc.objects[1]
    more = sorted(rig.named(f"{icons.ELLIPSIS}##more"), key=lambda c: c.rect[1])
    assert len(more) == 3
    rig.click(more[1])
    items = {c.label: c for c in rig.driver.controls if c.kind == "menu_item"}
    assert f"{icons.PENCIL} Rename" in items and f"{icons.COPY} Duplicate" in items
    assert f"{icons.EYE} Solo" in items
    delete = next(c for label, c in items.items() if label.endswith(" Delete"))
    rig.click(delete)
    assert [o.uid for o in doc.objects if o.uid == victim.uid] == []
    assert len(doc.objects) == 2
    assert len(doc.history) == steps + 1, "one undo step brings it back"


def test_the_more_button_is_greyed_with_its_reason_while_a_row_is_being_renamed(imgui_ui):
    ctx, tab, doc = _document(3)
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx)
    clay_outliner.begin_rename(_state(ctx), tab.uid, doc.objects[0].uid)
    rig.settle(2)
    greyed = [c for c in rig.named(f"{icons.ELLIPSIS}##more") if not c.enabled]
    assert len(greyed) == 1 and greyed[0].reason


def test_the_inspectors_delete_button_still_deletes_the_selected_shape(imgui_ui):
    ctx, tab, doc = _document(3)
    rig = Rig(imgui_ui, ctx)
    rig.click(rig.named("Delete##insp-delete")[0])
    assert len(doc.objects) == 2


# -- the drag auto-scroll ---------------------------------------------------------------------


def test_a_drag_near_the_top_or_bottom_edge_scrolls_that_way_and_the_middle_does_not():
    f = clay_outliner.autoscroll_delta
    assert f(100.0, 100.0, 300.0, 28.0, 14.0) == -14.0
    assert f(300.0, 100.0, 300.0, 28.0, 14.0) == 14.0
    assert f(200.0, 100.0, 300.0, 28.0, 14.0) == 0.0
    assert -14.0 < f(114.0, 100.0, 300.0, 28.0, 14.0) < 0.0, "ramps up toward the edge"
    assert f(400.0, 100.0, 300.0, 28.0, 14.0) == 14.0, "past the edge is still full speed"
    assert f(101.0, 100.0, 110.0, 28.0, 14.0) < 0.0, "a short list still has an edge to scroll at"
    assert f(100.0, 100.0, 100.0, 28.0, 14.0) == 0.0, "a zero-height list scrolls nothing"


def test_a_long_list_scrolls_while_a_row_is_held_at_its_bottom_edge(imgui_ui):
    ctx, tab, doc = _big_document(200)
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx)
    first = rig.rows[0]
    last_visible = rig.rows[-1]
    sx, sy = first.centre
    # Inside the rows' window, just above its bottom edge.
    ex, ey = last_visible.centre
    driver = rig.driver
    driver.frame((sx, sy), down=False)
    driver.frame((sx, sy), down=True)
    for step in (0.3, 0.6, 1.0):
        driver.frame((sx + (ex - sx) * step, sy + (ey - sy) * step), down=True)
    for _ in range(40):
        driver.frame((ex, ey), down=True)
    names_during = [c.label for c in rig.rows]
    assert first.label not in names_during, "the list scrolled under the held row"
    assert _state(ctx).outliner_drag == doc.objects[0].uid, "and the row is still held"
    driver.frame((ex, ey), down=False)
    driver.settle(2)
    assert _state(ctx).outliner_drag == 0, "released: nothing stays armed"


# -- cost -------------------------------------------------------------------------------------


@pytest.mark.parametrize("shapes", [1, 100, 4096])
def test_an_open_dock_submits_only_the_rows_it_can_show_at_any_document_size(
    imgui_ui, shapes, monkeypatch
):
    ctx, tab, doc = _big_document(shapes)
    drawn: list[int] = []
    real = clay_outliner._row
    monkeypatch.setattr(
        clay_outliner, "_row", lambda *a, **k: (drawn.append(a[3].uid), real(*a, **k))[1]
    )
    doc.select([])
    rig = Rig(imgui_ui, ctx)
    drawn.clear()
    rig.settle(2)
    assert 0 < len(drawn) // 2 <= min(shapes, 40), f"{len(drawn) // 2} rows for {shapes} shapes"


@pytest.mark.parametrize("shapes", [1, 100, 4096])
def test_a_closed_dock_costs_no_row_work_at_any_document_size(imgui_ui, shapes, monkeypatch):
    ctx, tab, doc = _big_document(shapes)
    calls: list[int] = []
    monkeypatch.setattr(clay_outliner, "_visible_rows", lambda *a, **k: calls.append(1) or [])
    doc.select([doc.objects[0].uid])
    rig = Rig(imgui_ui, ctx)
    for _ in range(4):
        rig.settle(1)
    assert calls == []


@pytest.mark.perf
@pytest.mark.parametrize("shapes", [1, 100, 4096])
def test_the_inspector_frame_with_the_dock_open_stays_cheap_at_any_document_size(
    imgui_ui, shapes
):
    ctx, tab, doc = _big_document(shapes)
    doc.select([])
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx)
    budget = {1: 0.005, 100: 0.006, 4096: 0.020}[shapes]
    samples = []
    for _ in range(12):
        start = time.perf_counter()
        rig.settle(1)
        samples.append(time.perf_counter() - start)
    median = statistics.median(samples)
    assert median < budget, f"{shapes} shapes: {median * 1000:.1f} ms"


def test_the_inspector_frame_with_the_dock_open_is_bounded_even_under_contention(imgui_ui):
    ctx, tab, doc = _big_document(4096)
    doc.select([])
    _open_dock(ctx)
    rig = Rig(imgui_ui, ctx)
    start = time.perf_counter()
    for _ in range(5):
        rig.settle(1)
    assert (time.perf_counter() - start) / 5 < 0.5


# -- the settings helpers the dock leans on ----------------------------------------------------


def test_persisted_open_reads_the_stored_choice_then_the_default(monkeypatch):
    class Store:
        def __init__(self, data):
            self.data = data

        def get(self, key, default=None):
            return self.data.get(key, default)

    monkeypatch.setattr(widgets, "_SETTINGS", None)
    assert widgets.persisted_open("k/a", True) is True
    assert widgets.persisted_open("k/a", False) is False
    monkeypatch.setattr(widgets, "_SETTINGS", Store({"panels_open": {"k/a": False, "k/b": True}}))
    assert widgets.persisted_open("k/a", True) is False, "the stored choice wins"
    assert widgets.persisted_open("k/b", False) is True
    assert widgets.persisted_open("k/c", True) is True, "unstored: the default"
    widgets.request_open("k/a")
    assert widgets.open_requested("k/a") and widgets.persisted_open("k/a", False) is True
