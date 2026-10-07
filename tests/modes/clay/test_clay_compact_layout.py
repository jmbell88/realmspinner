"""Clay's picoCAD-compact layout (2026-10-07): the claims the change makes.

The right column is the outliner over a tabbed Inspector, the tools are a rail
inside the centre, the palette is a strip under the render, undo and redo are on
the header, and the file verbs are in the File menu. Each test's name is the
claim it makes, and each fails against the tree as it stood before the change
(the names it imports did not exist, or the old arrangement answers
differently).

Panes are driven headless the way ``test_material_swatches.py`` does it: a real
imgui frame, with the one control that would take a press monkeypatched to
report one.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import layout_skeleton, menus, skeletons
from realmspinner.studio.modes.clay import element_move
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import state as clay_state
from realmspinner.studio.modes.clay.ui.panes import palette_strip as clay_palette
from realmspinner.studio.modes.clay.ui.panes import props as clay_props
from realmspinner.studio.modes.clay.ui.panes import rail as clay_rail
from realmspinner.studio.modes.clay.ui.panes import swatches as clay_swatches
from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools


class _Ctx:
    def __init__(self) -> None:
        self.info: list[str] = []
        self.errors: list[str] = []

    def toast(self, message: str, level: str = "info") -> None:
        (self.errors if level == "error" else self.info).append(message)

    def busy(self, key: str) -> bool:
        return False


def _tab(doc: bd.ClayDoc, *, saving: bool = False) -> Any:
    return SimpleNamespace(doc=doc, saving=saving, uid="t1", job_id="")


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


@pytest.fixture(autouse=True)
def _no_recents(monkeypatch):
    """The recent list reads the settings store; nothing here is about it, and a
    bare ``SimpleNamespace`` ctx has no store to write a migrated list into."""
    monkeypatch.setattr(clay_mode, "recent_paths", lambda ctx: [])


def _frame(ui, draw) -> None:
    ui.new_frame()
    ui.begin("##host")
    draw()
    ui.end()
    ui.end_frame()


def _doc() -> tuple[bd.ClayDoc, int]:
    """One box and a two-entry palette (slot 1 is red)."""
    doc = bd.ClayDoc()
    uid = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Box",
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
        )
    ).uid
    doc.add_material(replace(bd.default_material("red"), base_color_factor=(1.0, 0.0, 0.0, 1.0)))
    return doc, uid


def _select_faces(doc: bd.ClayDoc, uid: int, *faces: int) -> None:
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=list(faces)))


# --- the skeleton -------------------------------------------------------------


def test_the_clay_skeleton_is_an_empty_left_column_and_a_right_column_of_outliner_and_props():
    columns = skeletons.clay(SimpleNamespace())
    assert columns["left"].slots == (), "the Add sidebar is the rail inside the centre now"
    right = columns["right"].slots
    assert [slot.id for slot in right] == ["clay-outliner", "clay-props"]
    outliner, props = right
    assert outliner.sizing == layout_skeleton.SHARE and outliner.share_key == "clay-outliner"
    assert props.sizing == layout_skeleton.FILL and props.floor > 0, (
        "the Inspector is the fill, with a floor so a short window keeps its tab strip"
    )


def test_a_layout_saved_with_the_retired_clay_panes_loads_onto_the_two_pane_column():
    """No ``VERSION`` bump is owed: ``reconcile`` drops an id the built-in column
    no longer names, so a saved right column that still lists the UV and Document
    panes lands on the new one."""
    builtin = [slot.id for slot in skeletons.clay(SimpleNamespace())["right"].slots]
    saved = ["clay-outliner", "clay-uv", "clay-props", "clay-bridge"]
    assert layout_skeleton.reconcile(builtin, saved) == ["clay-outliner", "clay-props"]
    assert layout_skeleton.reconcile([], ["clay-tools"]) == [], "and the empty left column"


# --- the palette strip --------------------------------------------------------


def test_a_strip_click_paints_the_selected_faces_as_one_undo_step(monkeypatch):
    doc, uid = _doc()
    _select_faces(doc, uid, 2, 3)
    monkeypatch.setattr(clay_swatches, "_ctrl_held", lambda: False)
    ctx = _Ctx()
    before = len(doc.history)

    clay_swatches.click_slot(ctx, doc, 1)

    assert list(doc.by_uid(uid).mesh.material) == [0, 0, 1, 1, 0, 0]
    assert len(doc.history) == before + 1
    assert doc.undo()
    assert not doc.by_uid(uid).mesh.material.any()


def test_a_ctrl_click_on_the_strip_only_retargets_the_slot(monkeypatch):
    """Ctrl+click is how a slot's colour and texture are reached from a face
    selection. It must not paint: the user is going to edit the slot, not wear
    it."""
    doc, uid = _doc()
    _select_faces(doc, uid, 2, 3)
    monkeypatch.setattr(clay_swatches, "_ctrl_held", lambda: True)
    before = len(doc.history)

    clay_swatches.click_slot(_Ctx(), doc, 1)

    obj = doc.by_uid(uid)
    assert obj.material == 1, "now the slot the Material tab edits"
    assert not obj.mesh.material.any(), "and no face was painted"
    assert len(doc.history) == before + 1


def test_a_ctrl_click_in_object_mode_retargets_instead_of_repainting(monkeypatch):
    doc, uid = _doc()
    doc.select([uid])
    monkeypatch.setattr(clay_swatches, "_ctrl_held", lambda: True)

    clay_swatches.click_slot(_Ctx(), doc, 1)

    obj = doc.by_uid(uid)
    assert obj.material == 1 and not obj.mesh.material.any()


def test_a_strip_click_in_object_mode_repaints_every_selected_object_as_one_step(monkeypatch):
    doc, first = _doc()
    second = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box())).uid
    doc.select([first, second])
    monkeypatch.setattr(clay_swatches, "_ctrl_held", lambda: False)
    before = len(doc.history)

    clay_swatches.click_slot(_Ctx(), doc, 1)

    assert (doc.by_uid(first).mesh.material == 1).all()
    assert (doc.by_uid(second).mesh.material == 1).all()
    assert len(doc.history) == before + 1, "one Ctrl+Z puts the whole selection back"


def test_the_strip_draws_the_documents_slots_and_never_seeds_new_ones(monkeypatch, ui):
    doc, _uid = _doc()
    before = list(doc.materials)
    drawn: list[str] = []

    def fake(label, colour, side, **kw):
        drawn.append(label)
        return False

    monkeypatch.setattr(clay_swatches, "_swatch", fake)
    _frame(ui, lambda: clay_palette._body(_Ctx(), _tab(doc)))

    assert drawn == [f"##palsw{i}" for i in range(len(before))]
    assert doc.materials == before, "drawing is not an edit"


def test_the_strips_plus_appends_a_slot_as_one_undo_step(monkeypatch, ui):
    doc, _uid = _doc()
    monkeypatch.setattr(clay_swatches, "_swatch", lambda *a, **k: False)
    monkeypatch.setattr(
        clay_palette.controls, "small_button", lambda label, **kw: "palette/add" in label
    )
    slots, steps = len(doc.materials), len(doc.history)

    _frame(ui, lambda: clay_palette._body(_Ctx(), _tab(doc)))

    assert len(doc.materials) == slots + 1
    assert len(doc.history) == steps + 1
    assert doc.undo() and len(doc.materials) == slots


def test_the_strip_ignores_a_press_while_a_save_is_in_flight(monkeypatch, ui):
    doc, uid = _doc()
    doc.select([uid])
    monkeypatch.setattr(clay_swatches, "_swatch", lambda label, *a, **k: label == "##palsw1")
    steps = len(doc.history)

    _frame(ui, lambda: clay_palette._body(_Ctx(), _tab(doc, saving=True)))

    assert len(doc.history) == steps and not doc.by_uid(uid).mesh.material.any()


def test_the_strips_height_is_fixed_at_52_design_pixels():
    assert clay_palette.STRIP_H == 52.0


def test_the_click_hint_names_what_a_click_does_in_each_mode():
    doc, uid = _doc()
    assert "Select" in clay_palette.click_hint(doc)
    doc.select([uid])
    assert "repaints" in clay_palette.click_hint(doc)
    _select_faces(doc, uid, 0)
    assert "paints the selected faces" in clay_palette.click_hint(doc)
    doc.set_element_mode("vertex")
    assert "face mode" in clay_palette.click_hint(doc)


# --- the rail -----------------------------------------------------------------


def test_the_rail_tooltips_quote_the_keys_the_state_table_declares():
    """The rail reads its letters off ``clay_state.TOOLS`` so a rebinding is one
    edit and the rail cannot quote a key that no longer works."""
    for key, label, shortcut in clay_state.TOOLS:
        assert clay_rail.tool_tooltip(key, label, shortcut) == f"{label}  ({shortcut})"


def test_pressing_a_rail_tool_sets_the_tool_and_a_quick_shape_places_it(monkeypatch, ui):
    ctx = _Ctx()
    ctx.state = SimpleNamespace(clay=None)
    state = clay_mode.ensure(ctx)
    doc = bd.ClayDoc()
    tab = _tab(doc)
    pressed = {"##clay-rail/tool/rotate", f"##clay-rail/add/{clay_tools.QUICK_SHAPES[0]}"}
    monkeypatch.setattr(
        clay_rail.controls,
        "button",
        lambda label, *a, **k: "#" + label.split("#", 1)[1] in pressed,
    )
    _frame(ui, lambda: clay_rail._body(ctx, state, tab))

    assert state.tool == "rotate"
    assert len(doc.objects) == 1 and doc.objects[0].generator == clay_tools.QUICK_SHAPES[0]
    assert state.generator == clay_tools.QUICK_SHAPES[0]


def test_every_quick_shape_button_is_disabled_with_a_reason_without_a_document(monkeypatch, ui):
    ctx = _Ctx()
    ctx.state = SimpleNamespace(clay=None)
    state = clay_mode.ensure(ctx)
    seen: list[dict[str, Any]] = []

    def spy(label, size=(0, 0), **kw):
        seen.append({"label": label, **kw})
        return False

    monkeypatch.setattr(clay_rail.controls, "button", spy)
    _frame(ui, lambda: clay_rail._body(ctx, state, None))

    shapes = [one for one in seen if "/add/" in one["label"]]
    assert len(shapes) == len(clay_tools.QUICK_SHAPES)
    assert all(not one["enabled"] and one["reason"] for one in shapes)
    tools = [one for one in seen if "/tool/" in one["label"]]
    assert len(tools) == 4 and all(one.get("enabled", True) for one in tools), (
        "choosing a tool needs no document"
    )


# --- the Inspector's element rows ---------------------------------------------


def test_a_typed_median_moves_the_selected_elements_in_one_undo_step(monkeypatch, ui):
    doc, uid = _doc()
    _select_faces(doc, uid, 5)  # a +Y face of the 1 m box
    median = element_move.element_median_world(doc)
    assert median is not None
    state = clay_state.ClayState()

    def typed(label, values, axes, **kw):
        if label.startswith("##median"):
            assert kw.get("commit") is True, "one step per commit, not per keystroke"
            return True, [values[0] + 2.0, values[1], values[2] - 0.5]
        return False, list(values)

    monkeypatch.setattr(clay_props.controls, "input_vec", typed)
    before = len(doc.history)
    start = np.array(doc.by_uid(uid).mesh.positions)

    _frame(ui, lambda: clay_props._element_median(doc, state))

    moved = element_move.element_median_world(doc)
    assert moved == pytest.approx(median + np.array([2.0, 0.0, -0.5]), abs=1e-5)
    assert len(doc.history) == before + 1
    assert doc.undo()
    assert np.allclose(doc.by_uid(uid).mesh.positions, start)


def test_a_median_typed_in_centimetres_moves_by_metres_and_only_on_the_edited_axis():
    doc, uid = _doc()
    _select_faces(doc, uid, 5)
    median = element_move.element_median_world(doc)
    shown = np.array([100.0 * v for v in median])
    typed = [shown[0], shown[1] + 50.0, shown[2]]

    assert clay_props.commit_median(doc, median, shown, typed, "cm") is True

    assert element_move.element_median_world(doc) == pytest.approx(
        median + np.array([0.0, 0.5, 0.0]), abs=1e-5
    )


def test_an_unchanged_median_records_no_step():
    doc, uid = _doc()
    _select_faces(doc, uid, 5)
    median = element_move.element_median_world(doc)
    shown = [float(v) for v in median]
    before = len(doc.history)

    assert clay_props.commit_median(doc, median, shown, list(shown), "m") is False

    assert len(doc.history) == before


def test_the_median_row_sits_outside_the_one_object_gate():
    """An element selection spanning two objects has no single object to edit,
    and the median of it is still a fact the user can type."""
    doc, first = _doc()
    second = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box())).uid
    doc.set_element_mode("face")
    doc.set_element_sel(first, el.ElementSel(faces=[0]))
    doc.set_element_sel(second, el.ElementSel(faces=[0]))
    assert clay_props._selected(doc) is None, "two objects: the object gate is shut"
    assert element_move.element_median_world(doc) is not None
    assert clay_props.element_summary_text(doc) == "face mode -- 2 faces across 2 objects"


def test_the_element_measure_line_reports_area_and_edge_length():
    doc, uid = _doc()
    _select_faces(doc, uid, 5)
    assert clay_props.element_measure_text(doc) == "area  1.0000 m\u00b2"
    doc.set_element_mode("edge")
    doc.set_element_sel(uid, el.ElementSel(edges=[(0, 1)]))
    assert clay_props.element_measure_text(doc, "m") == "length  1.0000 m"
    assert clay_props.element_measure_text(doc, "cm") == "length  100.0000 cm"
    doc.set_element_mode("object")
    assert clay_props.element_measure_text(doc) is None


# --- the File menu ------------------------------------------------------------


def _clay_menu_ctx(tab: Any | None = None, *, recent: tuple[str, ...] = ()) -> Any:
    state = SimpleNamespace(active=tab)
    return SimpleNamespace(
        state=SimpleNamespace(
            mode="clay",
            clay=state,
            selected=None,
            wireframe=False,
            turntable=False,
            show_fps=False,
            filters=SimpleNamespace(trash=False),
            errors=[],
        ),
        cache=SimpleNamespace(get=lambda _key: None, jobs=[]),
        runtime=SimpleNamespace(checks=[]),
        settings=SimpleNamespace(get=lambda key, default=None: default),
        viewer=None,
    )


def test_the_file_menu_has_the_clay_rows_while_clay_is_active(monkeypatch):
    monkeypatch.setattr(clay_mode, "recent_paths", lambda ctx: ["C:/work/barrel.rblk"])
    doc, _uid = _doc()
    rows = {row.identity: row for row in menus.specs(_clay_menu_ctx(_tab(doc)))}

    for identity in ("clay:open", "clay:export-glb", "clay:export-obj", "clay:screenshot"):
        assert identity in rows, identity
        assert rows[identity].path == ("File",)
        assert rows[identity].enabled, identity
    recent = rows["clay:recent:C:/work/barrel.rblk"]
    assert recent.path == ("File", menus.CLAY_RECENT_MENU) and recent.label == "barrel.rblk"
    assert rows["clay:open"].label == "Open..."
    # Not duplicated: New, Save, Save As, Export to the library, Undo and Redo
    # are the palette's generic commands, already File and Edit rows.
    assert not [i for i in rows if i.startswith("clay:") and "save" in i and "screenshot" not in i]


def test_the_clay_file_rows_are_absent_in_every_other_mode():
    for mode in ("home", "library", "inker", "mason"):
        ctx = _clay_menu_ctx()
        ctx.state.mode = mode
        ctx.state.inker = None
        assert not [r for r in menus._clay_file_specs(ctx, [])], mode


def test_the_clay_exports_are_greyed_with_a_reason_when_there_is_nothing_to_send():
    rows = {row.identity: row for row in menus._clay_file_specs(_clay_menu_ctx(None), [])}
    for identity in ("clay:export-glb", "clay:export-obj", "clay:screenshot"):
        assert not rows[identity].enabled and rows[identity].disabled_reason, identity
    assert rows["clay:open"].enabled, "opening needs no document"
    doc, uid = _doc()
    doc.set_visibility({uid: False})
    hidden = {row.identity: row for row in menus._clay_file_specs(_clay_menu_ctx(_tab(doc)), [])}
    assert "hidden" in hidden["clay:export-glb"].disabled_reason


def test_a_clay_file_row_calls_the_modes_own_door(monkeypatch):
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(clay_mode, "ask_open", lambda ctx: calls.append(("open",)))
    monkeypatch.setattr(
        clay_mode, "export_mesh_file", lambda ctx, tab, kind: calls.append(("export", kind))
    )
    monkeypatch.setattr(clay_mode, "save_screenshot", lambda ctx, tab: calls.append(("shot",)))
    doc, _uid = _doc()
    rows = {row.identity: row for row in menus._clay_file_specs(_clay_menu_ctx(_tab(doc)), [])}
    for identity in ("clay:open", "clay:export-glb", "clay:export-obj", "clay:screenshot"):
        rows[identity].callback()
    assert calls == [("open",), ("export", "glb"), ("export", "obj"), ("shot",)]


def test_a_second_path_segment_is_a_submenu_of_its_root():
    """``Open Recent`` is a row with a two-segment path; ``roots`` still keys on
    the first, so the submenu does not spawn a root of its own."""
    rows = menus._clay_file_specs(_clay_menu_ctx(None), [])
    sub = [row for row in rows if len(row.path) > 1]
    assert sub and {row.path[0] for row in sub} == {"File"}
    assert menus.roots(rows).count("File") == 1


def test_the_rails_plus_opens_a_flyout_naming_every_clay_shape(monkeypatch, ui):
    """``+`` is the way to the shapes the four quick buttons leave out: pressed
    once, the popup lists all fifteen by name (the flyout body is
    ``tools.draw_add_menu``; this proves the rail opens it, in the rail's own
    id stack)."""
    from realmspinner.studio import probe

    ctx = _Ctx()
    ctx.state = SimpleNamespace(clay=None)
    state = clay_mode.ensure(ctx)
    tab = _tab(bd.ClayDoc())
    presses = {"more": 1}

    def button(label, *args, **kwargs):
        if label.endswith("##clay-rail/more") and presses["more"]:
            presses["more"] -= 1
            return True
        return False

    monkeypatch.setattr(clay_rail.controls, "button", button)
    seen: set[str] = set()
    for _ in range(4):  # the popup sizes itself hidden for a frame before it draws
        probe.begin_frame()
        _frame(ui, lambda: clay_rail._body(ctx, state, tab))
        seen |= {c.label for c in probe.FRAME_CONTROLS}

    for _section, names in clay_tools.sections():
        for name in names:
            assert any(label.endswith(f"##clay-add/{name}") for label in seen), name
