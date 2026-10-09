"""Does Clay's viewport header fit, and does it hold what it claims to.

The one question a header can silently get wrong is whether it fits at the
window the app opens at. ``toolbar`` degrades rather than clips, so a bar that
does not fit does not *look* broken -- it looks like a bar whose every label is
a hover away, on the machine everybody uses. That is the failure this file
exists for, and it is measured rather than eyeballed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from _ui_context import imgui_context
from _ui_driver import UiDriver

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import icons, toolbar
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay import state as clay_state
from realmspinner.studio.modes.clay.ui.panes import header as clay_header
from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools

# --- the tables ---------------------------------------------------------------


def test_every_element_mode_has_a_short_label():
    """The compact tier swaps the words for these, so a mode missing one would
    be a segment drawn with nothing in it."""
    for mode, _label, _key in clay_tools.MODE_BUTTONS:
        assert clay_header.MODE_SHORT.get(mode), mode
    assert set(clay_header.MODE_SHORT) == {
        mode for mode, _l, _k in clay_tools.MODE_BUTTONS
    }


def test_the_modes_the_header_offers_are_the_modes_the_ops_registry_knows():
    """A mode on the bar that no op declares is a mode with nothing in it."""
    assert {mode for mode, _l, _k in clay_tools.MODE_BUTTONS} == set(clay_ops.ALL_MODES)


def test_every_tool_has_a_glyph_or_falls_back_to_its_initial():
    for key, label, _shortcut in clay_state.TOOLS:
        drawn = clay_tools.TOOL_ICONS.get(key) or label[:1]
        assert drawn, key


def test_the_overlay_rows_name_real_switches():
    """``grid`` is a field and the rest live in ``state.overlays``; the two
    accessors are what let the popover be a loop over one table."""
    state = clay_state.ClayState()
    for key, label, tip in clay_header.OVERLAY_ROWS:
        assert label and tip
        before = clay_header.overlay_value(state, key)
        clay_header.set_overlay(state, key, not before)
        assert clay_header.overlay_value(state, key) is (not before)


def test_the_grid_keeps_its_own_field_rather_than_moving_into_the_dict():
    """One switch, one home. It is wired straight to ``ClayView.show_grid`` and
    a second copy in the dict is two places that can disagree."""
    state = clay_state.ClayState()
    clay_header.set_overlay(state, "grid", False)
    assert state.grid is False
    assert "grid" not in state.overlays


def test_god_light_also_keeps_its_own_field_rather_than_moving_into_the_dict():
    """Task C's row: a ``ClayState`` field like ``grid``, not a third home in
    ``state.overlays`` -- the same reason ``grid`` gets one."""
    state = clay_state.ClayState()
    clay_header.set_overlay(state, "god_light", True)
    assert state.god_light is True
    assert "god_light" not in state.overlays


def test_the_overlay_popup_has_a_grid_size_field():
    """Task A: the size field sits under the Grid row, not as a fifth item in
    ``OVERLAY_ROWS`` -- it is not a switch, it is the number the Grid switch
    governs, the same shape ``_snap_popup``'s grid step has."""
    assert hasattr(clay_header, "_grid_size_field")
    keys = [key for key, _label, _tip in clay_header.OVERLAY_ROWS]
    assert keys.index("grid") < keys.index("god_light")


def test_the_grid_tooltip_says_1m_cells_not_the_snap_size():
    """The row used to say "at the snap size", which stopped being true the
    moment the grid got its own ``grid_size`` field independent of Snap."""
    tips = {key: tip for key, _label, tip in clay_header.OVERLAY_ROWS}
    assert "1 m" in tips["grid"]
    assert "snap" not in tips["grid"].lower()


def test_every_axis_row_names_a_view_the_camera_has():
    from realmspinner.studio.viewer.camera import Camera

    for name, label, chord in clay_header.AXIS_ROWS:
        assert name in Camera.AXIS_VIEWS, name
        assert label and chord.startswith("Ctrl+")
    assert {name for name, _l, _c in clay_header.AXIS_ROWS} == set(Camera.AXIS_VIEWS), (
        "all six, not the three that had buttons -- the backs were reachable "
        "only by holding Shift, which nothing said"
    )


# The fit itself is measured against the real font atlas, which needs the GL-backed
# imgui context ``tests/studio/test_studio_smoke.py`` already owns (this file's
# contexts are the headless kind, with imgui's default font, and are torn down with
# each test). The width tests -- the default window, the 1100x700 floor at three
# scales and the order the row gives things up in -- live there.


@pytest.mark.parametrize("popup", ["SNAP_POPUP", "VIEW_POPUP", "HISTORY_POPUP", "EXPORT_POPUP"])
def test_every_popup_has_its_own_name(popup):
    """Two popups sharing a name is one popup that opens when either is asked
    for, which imgui reports as neither working. (The Overlays popup is gone:
    View holds what it held.)"""
    names = {
        getattr(clay_header, key)
        for key in ("SNAP_POPUP", "VIEW_POPUP", "HISTORY_POPUP", "EXPORT_POPUP")
    }
    assert len(names) == 4
    assert getattr(clay_header, popup) in names
    assert not hasattr(clay_header, "OVERLAYS_POPUP")


def _tab(doc=None):
    from types import SimpleNamespace

    from realmspinner.kernels.mesh import document as bd

    return SimpleNamespace(doc=doc or bd.ClayDoc(), saving=False)


def test_the_mode_pill_is_a_pinned_field_rather_than_an_item(ui):
    """Items collapse into the overflow menu first, and a mode picker in a menu
    is a mode picker nobody can see the state of -- which is the one thing a
    mode picker is for. Pinned, so no width ever puts it there."""
    state = clay_state.ClayState()
    tab = _tab()
    keys = {item.key for item in clay_header._items(state, tab)}
    assert "mode" not in keys
    # Measuring a font needs a frame in progress (no frame, no baked font).
    ui.new_frame()
    field = clay_header._mode_field(tab)
    ui.end_frame()
    assert isinstance(field, toolbar.Field)
    assert field.priority == 0
    assert field.pinned, "the one control whose state must always be visible"


def test_the_tool_pill_left_the_header_for_the_rail():
    """Select / Move / Rotate / Scale are the tool rail's now. A second pill for
    the same ``state.tool`` was two controls for one state, and its width is
    what Undo and Redo took."""
    assert not hasattr(clay_header, "_tool_field")
    assert "tool" not in {item.key for item in clay_header._items(clay_state.ClayState(), _tab())}


def test_the_mode_switch_reads_shape_points_edges_faces_and_its_keys_are_unchanged():
    """The words changed; the stored modes and keys 1-4 did not. Old code read
    Object / Verts / Edges / Faces, led with Vertex and put "Object" last."""
    assert [(mode, label) for mode, label, _k in clay_tools.MODE_BUTTONS] == [
        ("object", "Shape"),
        ("vertex", "Points"),
        ("edge", "Edges"),
        ("face", "Faces"),
    ]
    assert clay_mode.ELEMENT_KEYS == {"1": "vertex", "2": "edge", "3": "face", "4": "object"}
    for mode, _label, key in clay_tools.MODE_BUTTONS:
        assert clay_mode.ELEMENT_KEYS[key] == mode, "the button's tooltip names the wrong key"


def test_the_shading_switch_reads_flat_lit_wire_over_the_unchanged_stored_values():
    """Labels only: ``state.shading`` is still solid / material / wireframe (the
    renderer, saved layouts and the agent read those), and each tooltip says what
    the word means."""
    assert [(key, label) for key, label, _s, _t in clay_header.SHADING] == [
        ("solid", "Flat"),
        ("material", "Lit"),
        ("wireframe", "Wire"),
    ]
    assert {key: tip for key, _l, _s, tip in clay_header.SHADING} == {
        "solid": "Flat colours, no lighting",
        "material": "Lit surface",
        "wireframe": "Edges only",
    }
    assert [short for _k, _l, short, _t in clay_header.SHADING] == ["F", "L", "W"]


def test_the_header_has_undo_and_redo_enabled_from_the_documents_history():
    """The history lived three panes away in the Document tab, so the most-used
    control in a modeller was reachable by a chord alone. The items read
    ``tab.doc.history`` -- the stack the chord drives -- so their state cannot
    disagree with what Ctrl+Z would do."""
    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import primitives as bp

    state = clay_state.ClayState()
    doc = bd.ClayDoc()
    tab = _tab(doc)

    def items():
        return {item.key: item for item in clay_header._items(state, tab)}

    assert "undo" in items() and "redo" in items()
    assert not items()["undo"].enabled and items()["undo"].reason
    assert not items()["redo"].enabled and items()["redo"].reason

    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    assert items()["undo"].enabled and not items()["redo"].enabled
    assert doc.undo()
    assert not items()["undo"].enabled and items()["redo"].enabled

    doc.redo()
    tab.saving = True
    assert not items()["undo"].enabled and "written" in items()["undo"].reason, (
        "a click must not mutate the stack under a running encode"
    )


def test_the_header_items_are_undo_redo_and_a_history_chevron_all_drawn_as_glyphs():
    """Three squares and no words; Undo and Redo pinned so no width hides them."""
    items = {item.key: item for item in clay_header._items(clay_state.ClayState(), _tab())}
    assert list(items) == ["undo", "redo", "history"]
    assert all(item.icon_only and item.icon for item in items.values())
    assert items["undo"].pinned and items["redo"].pinned
    assert items["history"].icon == icons.CHEVRON_DOWN
    assert not items["history"].enabled and items["history"].reason, "nothing to list yet"


# --- driving the real row ----------------------------------------------------------


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.settings = _Settings()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))

    def busy(self, key: str) -> bool:
        return False


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    # The recents read a store.
    monkeypatch.setattr(clay_mode, "recent_paths", lambda ctx: [])


def _open(boxes: int):
    ctx = _Ctx()
    doc = bd.ClayDoc()
    for index in range(boxes):
        doc.add_object(bd.Obj(uid=bd.new_uid(), name=f"B{index}", mesh=bp.box()))
    tab = clay_mode.adopt(ctx, doc, title="Scene")
    return ctx, tab


def _driver(ui, ctx, view=None) -> UiDriver:
    def draw() -> None:
        ui.new_frame()
        ui.set_next_window_pos((0.0, 0.0))
        ui.set_next_window_size((1500.0, 400.0))
        ui.begin("##host", None, ui.WindowFlags_.no_decoration.value)
        clay_header.draw(ctx, view)
        ui.end()
        ui.end_frame()

    driver = UiDriver(ui, draw)
    driver.settle(3)
    return driver


EXPORT = f"{icons.DOWNLOAD} Export##clay-header/export"
CHEVRON = f"{icons.CHEVRON_DOWN}##clay-header/history"
EXPORT_MORE = f"{icons.CHEVRON_DOWN}##clay-header/export-more"


def _history_rows(d: UiDriver) -> list[Any]:
    return [c for c in d.controls if "##clay-header-undo" in c.label and c.visible]


def test_the_history_chevron_lists_every_step_and_a_click_jumps_through_the_modes_door(
    monkeypatch, ui
):
    """Old code had no history control in the header at all (the list was three
    panes away in Properties' Document tab), so ``HISTORY_POPUP`` and the
    ``history`` item did not exist. The jump goes through ``clay_mode.step_history``
    -- the door that refuses under a live gesture -- never ``doc.step_history``."""
    ctx, tab = _open(3)
    jumps: list[int] = []
    real = clay_mode.step_history

    def door(ctx_, tab_, index):
        jumps.append(index)
        return real(ctx_, tab_, index)

    monkeypatch.setattr(clay_mode, "step_history", door)
    d = _driver(ui, ctx)
    assert len(tab.doc.history) == 3

    d.click(d.find(label=CHEVRON))
    d.settle(3)
    rows = _history_rows(d)
    assert len(rows) == 4, [c.label for c in rows]  # the document as opened + 3 steps

    d.click(rows[1])
    d.settle(2)
    assert jumps == [1], "the click did not go through clay_mode.step_history"
    assert len(tab.doc.history) == 1, "the document did not move to the clicked step"


def test_a_history_jump_is_refused_under_a_live_gesture(monkeypatch, ui):
    """The door is the point: a chevron that called ``doc.step_history`` would
    undo the thing being dragged."""
    ctx, tab = _open(2)
    monkeypatch.setattr(clay_mode, "_gesture_live", lambda ctx_, tab_: True)
    d = _driver(ui, ctx)
    d.click(d.find(label=CHEVRON))
    d.settle(3)
    d.click(_history_rows(d)[0])
    d.settle(2)
    assert len(tab.doc.history) == 2


def test_export_refuses_in_place_on_an_empty_scene_and_says_why(monkeypatch, ui):
    """The main click is the library export; with nothing to send it is greyed
    with ``doc_io.outputs_why`` rather than toasting after the press."""
    calls: list[str] = []
    monkeypatch.setattr(clay_mode, "export_asset", lambda ctx_, tab_=None: calls.append("lib"))
    ctx, _tab_ = _open(0)
    d = _driver(ui, ctx)
    button = d.find(label=EXPORT)
    assert not button.enabled
    assert button.reason == "Nothing visible to send -- every object is hidden."
    d.click(button)
    d.settle(2)
    assert calls == [], "a refused export ran"


def test_export_main_click_runs_the_library_export_and_the_chevron_lists_every_way_out(
    monkeypatch, ui
):
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(clay_mode, "export_asset", lambda ctx_, tab_=None: calls.append(("lib",)))
    monkeypatch.setattr(
        clay_mode, "export_mesh_file", lambda ctx_, tab_, kind: calls.append(("file", kind))
    )
    monkeypatch.setattr(
        clay_mode, "save_screenshot", lambda ctx_, tab_=None: calls.append(("shot",))
    )
    ctx, _tab_ = _open(1)
    d = _driver(ui, ctx)

    button = d.find(label=EXPORT)
    assert button.enabled
    d.click(button)
    d.settle(2)
    assert calls == [("lib",)]

    # Every way out is under the chevron, each through the same door the
    # Document tab's button uses.
    for key, expected in (
        ("library", ("lib",)),
        ("glb", ("file", "glb")),
        ("obj", ("file", "obj")),
        ("screenshot", ("shot",)),
    ):
        calls.clear()
        d.click(d.find(label=EXPORT_MORE))
        d.settle(3)
        row = next(c for c in d.controls if c.label.endswith(f"##clay-header/export/{key}"))
        assert row.enabled, key
        d.click(row)
        d.settle(2)
        assert calls == [expected], key


def test_the_export_chevron_rows_are_greyed_with_the_reason_on_an_empty_scene(ui):
    ctx, _tab_ = _open(0)
    d = _driver(ui, ctx)
    d.click(d.find(label=EXPORT_MORE))
    d.settle(3)
    rows = [c for c in d.controls if "##clay-header/export/" in c.label]
    assert len(rows) == 4
    assert all(not c.enabled and c.reason for c in rows), [
        (c.label, c.enabled, c.reason) for c in rows
    ]


def test_export_greys_while_a_save_is_in_flight(ui):
    ctx, tab = _open(1)
    tab.saving = True
    d = _driver(ui, ctx)
    button = d.find(label=EXPORT)
    assert not button.enabled and button.reason == "Saving..."


class _Camera:
    orthographic = False

    def __init__(self) -> None:
        self.looked: list[str] = []

    def look_along(self, name: str) -> None:
        self.looked.append(name)


#: Set by :func:`_press_in_view_popup` for the last frame only: a click is one
#: event, and the popup is drawn four times before it is open and sized.
_ARMED = {"on": False}


def _view_popup_frames(ui, ctx, state, view) -> dict[str, Any]:
    """Draw the View popover for the four frames a popup needs to open and size
    itself; -> the last frame's controls by label."""
    from realmspinner.studio import probe

    found: dict[str, Any] = {}
    for frame in range(4):
        _ARMED["on"] = frame == 3
        probe.begin_frame()
        ui.new_frame()
        ui.set_next_window_size((400.0, 600.0))
        ui.begin("##host")
        if not ui.is_popup_open(clay_header.VIEW_POPUP):
            ui.open_popup(clay_header.VIEW_POPUP)
        clay_header._view_popup(ctx, state, view)
        ui.end()
        ui.end_frame()
        found = {c.label: c for c in probe.FRAME_CONTROLS}
    return found


def _press_in_view_popup(monkeypatch, ui, wanted: str, state, view, ctx) -> None:
    """The View popover with the one row ending in *wanted* reporting a click."""
    real = clay_header.controls.menu_item

    def press(label, *args, **kwargs):
        shown = real(label, *args, **kwargs)
        return (True, True) if _ARMED["on"] and label.endswith(wanted) else shown

    monkeypatch.setattr(clay_header.controls, "menu_item", press)
    try:
        _view_popup_frames(ui, ctx, state, view)
    finally:
        _ARMED["on"] = False


def test_the_view_popover_holds_every_overlay_view_ortho_and_frame_row(ui):
    """One popover where there were two (Overlays and View). Every row it must
    hold is drawn, with the key beside each view."""
    ctx, _tab_ = _open(1)
    state = clay_mode.ensure(ctx)
    labels = set(_view_popup_frames(ui, ctx, state, SimpleNamespace(camera=_Camera())))
    for key, label, _tip in clay_header.OVERLAY_ROWS:
        assert f"{label}##clay-header/overlay/{key}" in labels, key
    for name, label, _chord in clay_header.AXIS_ROWS:
        assert f"{label}##clay-header/axis/{name}" in labels, name
    assert "Orthographic##clay-header/ortho" in labels
    assert "Frame the selection##clay-header/frame" in labels
    assert "Wireframe over surface##clay-header/overlay/wire" in labels
    assert [name for name, _l, _c in clay_header.AXIS_ROWS] == [
        "front", "back", "left", "right", "top", "bottom",
    ]


@pytest.mark.parametrize(
    "row",
    ["grid", "wire", "stats", "god_light"],
)
def test_each_view_overlay_row_changes_the_state_it_names_and_persists_the_view_block(
    monkeypatch, ui, row
):
    ctx, _tab_ = _open(1)
    state = clay_mode.ensure(ctx)
    before = clay_header.overlay_value(state, row)
    view = SimpleNamespace(camera=_Camera())
    _press_in_view_popup(monkeypatch, ui, f"##clay-header/overlay/{row}", state, view, ctx)
    assert clay_header.overlay_value(state, row) is (not before), row
    stored = ctx.settings.get("clay")["view"]
    assert stored["grid"] is bool(state.grid)
    assert stored["god_light"] is bool(state.god_light), (
        "the grid and the god light are the persisted pair and are written on a change"
    )


@pytest.mark.parametrize("name", [name for name, _l, _c in clay_header.AXIS_ROWS])
def test_each_view_row_turns_the_camera_to_that_axis(monkeypatch, ui, name):
    ctx, _tab_ = _open(1)
    state = clay_mode.ensure(ctx)
    camera = _Camera()
    view = SimpleNamespace(camera=camera)
    _press_in_view_popup(monkeypatch, ui, f"##clay-header/axis/{name}", state, view, ctx)
    assert camera.looked == [name]


def test_the_view_popover_toggles_orthographic_and_asks_for_a_frame(monkeypatch, ui):
    ctx, _tab_ = _open(1)
    state = clay_mode.ensure(ctx)
    camera = _Camera()
    view = SimpleNamespace(camera=camera)
    _press_in_view_popup(monkeypatch, ui, "##clay-header/ortho", state, view, ctx)
    assert camera.orthographic is True
    assert state.frame_pending is False
    _press_in_view_popup(monkeypatch, ui, "##clay-header/frame", state, view, ctx)
    assert state.frame_pending is True


def test_the_view_rows_that_need_the_viewport_are_greyed_with_a_reason_before_it_exists(ui):
    """``view`` is None on the first frame: the rows wait rather than raise."""
    ctx, _tab_ = _open(1)
    state = clay_mode.ensure(ctx)
    found = _view_popup_frames(ui, ctx, state, None)
    rows = [c for label, c in found.items() if "##clay-header/axis/" in label]
    assert len(rows) == 6 and all(not c.enabled and c.reason for c in rows)


def test_the_shading_switch_draws_the_words_and_writes_the_stored_value(ui):
    ctx, _tab_ = _open(1)
    state = clay_mode.ensure(ctx)
    d = _driver(ui, ctx)
    drawn = [c.label for c in d.controls if "##clay-shading/" in c.label]
    assert drawn == [
        "Flat##clay-shading/solid",
        "Lit##clay-shading/material",
        "Wire##clay-shading/wireframe",
    ]
    d.click(d.find(label="Wire##clay-shading/wireframe"))
    d.settle(2)
    assert state.shading == "wireframe"


def test_the_mode_switch_draws_shape_points_edges_faces_and_keeps_the_internal_modes(ui):
    ctx, tab = _open(1)
    d = _driver(ui, ctx)
    drawn = [c.label for c in d.controls if "##clay-mode/" in c.label]
    assert drawn == [
        "Shape##clay-mode/object",
        "Points##clay-mode/vertex",
        "Edges##clay-mode/edge",
        "Faces##clay-mode/face",
    ]
    d.click(d.find(label="Points##clay-mode/vertex"))
    d.settle(2)
    assert tab.doc.element_mode == "vertex"
