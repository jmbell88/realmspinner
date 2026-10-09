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


def _frame(ui, draw, size: tuple[float, float] | None = None) -> None:
    ui.new_frame()
    if size is not None:
        ui.set_next_window_size(size)
    ui.begin("##host")
    draw()
    ui.end()
    ui.end_frame()


#: A host tall enough for the whole rail: its four quick shapes are drawn only
#: when they fit, so a test that presses one must give them the room.
TALL = (300.0, 1400.0)


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


def test_the_clay_skeleton_is_an_empty_left_column_and_a_right_column_of_the_inspector_alone():
    """Old code: the right column was ``clay-outliner`` (a SHARE slot) over the
    Inspector, so the id list was two long and the first slot had a share key."""
    columns = skeletons.clay(SimpleNamespace())
    assert columns["left"].slots == (), "the Add sidebar is the rail inside the centre now"
    right = columns["right"].slots
    assert [slot.id for slot in right] == ["clay-inspector"]
    (inspector,) = right
    assert inspector.sizing == layout_skeleton.FILL and inspector.floor > 0, (
        "the Inspector is the fill, with a floor so a short window keeps its first card"
    )
    assert inspector.share_key == "", "one slot has no split, so no handle and no share to store"


def test_a_layout_saved_with_the_retired_clay_panes_loads_onto_the_one_slot_column():
    """No ``VERSION`` bump is owed: ``reconcile`` drops an id the built-in column
    no longer names, so a saved right column that still lists the outliner, UV,
    Properties and Document panes lands on the one-slot column -- and the Inspector,
    a new id, takes it. Old code kept ``clay-outliner`` in the result."""
    builtin = [slot.id for slot in skeletons.clay(SimpleNamespace())["right"].slots]
    saved = ["clay-outliner", "clay-uv", "clay-props", "clay-bridge"]
    assert layout_skeleton.reconcile(builtin, saved) == ["clay-inspector"]
    assert layout_skeleton.reconcile(builtin, ["clay-outliner", "clay-inspector"]) == [
        "clay-inspector"
    ], "a layout saved by the last build, which had both"
    assert layout_skeleton.reconcile([], ["clay-tools"]) == [], "and the empty left column"


def test_the_retired_properties_id_is_never_reused_by_the_inspector():
    """A slot is hideable and a saved layout may have hidden ``clay-props``: the
    Inspector must not answer to that id, or the layout that hid the old pane would
    hide the new one without anybody having asked."""
    columns = skeletons.clay(SimpleNamespace()).values()
    ids = {slot.id for column in columns for slot in column.slots}
    assert "clay-props" not in ids and "clay-inspector" in ids


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
    _frame(ui, lambda: clay_rail._body(ctx, state, tab), TALL)

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
    _frame(ui, lambda: clay_rail._body(ctx, state, None), TALL)

    shapes = [one for one in seen if "/add/" in one["label"]]
    assert len(shapes) == len(clay_tools.QUICK_SHAPES)
    assert all(not one["enabled"] and one["reason"] for one in shapes)
    tools = [one for one in seen if "/tool/" in one["label"]]
    assert len(tools) == 4 and all(one.get("enabled", True) for one in tools), (
        "choosing a tool needs no document"
    )


def _rail_cells(monkeypatch, ui, height: float, *, tab=None) -> list[tuple[str, str]]:
    """``(ident, word)`` for every cell the rail draws in a child *height* tall."""
    ctx = _Ctx()
    ctx.state = SimpleNamespace(clay=None)
    state = clay_mode.ensure(ctx)
    seen: list[tuple[str, str]] = []
    real = clay_rail._cell

    def spy(glyph, ident, word, **kw):
        seen.append((ident, word))
        return real(glyph, ident, word, **kw)

    monkeypatch.setattr(clay_rail, "_cell", spy)
    _frame(ui, lambda: clay_rail._body(ctx, state, tab or _tab(bd.ClayDoc())), (300.0, height))
    return seen


def test_every_rail_tool_is_named_with_a_plain_word_under_its_glyph(monkeypatch, ui):
    """Before, the rail was four unlabelled glyphs and a column of shape icons.
    Old code drew no word at all: it had no ``_cell`` and no ``TOOL_WORDS``."""
    cells = dict(_rail_cells(monkeypatch, ui, 1400.0))
    assert [cells[f"clay-rail/tool/{key}"] for key, _l, _k in clay_state.TOOLS] == [
        "Select",
        "Move",
        "Turn",
        "Resize",
    ]
    assert set(clay_rail.TOOL_WORDS) == {key for key, _l, _k in clay_state.TOOLS}, (
        "a tool with no word would be drawn with its stored name"
    )
    assert [cells[f"clay-rail/add/{name}"] for name in clay_tools.QUICK_SHAPES] == [
        "Box",
        "Cylinder",
        "Sphere",
        "Cone",
    ]
    assert cells["clay-rail/more"] == "More"


def test_a_rail_tool_tooltip_carries_the_word_on_the_button_and_its_key(monkeypatch, ui):
    """The tooltip says the same word the button does ("Turn", not "Rotate") and
    the key, so what is read and what is pressed agree."""
    ctx = _Ctx()
    ctx.state = SimpleNamespace(clay=None)
    state = clay_mode.ensure(ctx)
    tips: dict[str, str] = {}

    def spy(label, size=(0, 0), **kw):
        tips[label.split("##", 1)[1]] = kw.get("tooltip", "")
        return False

    monkeypatch.setattr(clay_rail.controls, "button", spy)
    _frame(ui, lambda: clay_rail._body(ctx, state, _tab(bd.ClayDoc())), TALL)
    assert tips["clay-rail/tool/select"] == "Select  (Q)"
    assert tips["clay-rail/tool/rotate"] == "Turn  (R)"
    assert tips["clay-rail/tool/scale"] == "Resize  (S)"


def test_the_quick_shapes_collapse_into_the_plus_flyout_when_the_window_is_short(
    monkeypatch, ui
):
    """Old code drew the four quick shapes at any height and clipped the ``+``
    off the bottom of a short rail. Now they are drawn only when all four fit,
    the tools and the ``+`` always are, and the ``+`` says "Add" when it is the
    only way to a shape."""
    tall = [ident for ident, _w in _rail_cells(monkeypatch, ui, 1400.0)]
    assert [i for i in tall if "/add/" in i], "a tall rail keeps its quick shapes"

    short = dict(_rail_cells(monkeypatch, ui, 300.0))
    assert not [i for i in short if "/add/" in i], "a short rail must not clip them"
    assert len([i for i in short if "/tool/" in i]) == 4
    assert short["clay-rail/more"] == "Add"


def test_the_rail_fit_rule_is_pure_and_all_or_nothing():
    cell, spacing = 46.0, 8.0
    need = clay_rail.stack_height(len(clay_tools.QUICK_SHAPES), cell, spacing)
    assert clay_rail.quick_shapes_fit(need, cell, spacing)
    assert not clay_rail.quick_shapes_fit(need - 1.0, cell, spacing)
    # The rail with its quick shapes dropped is shorter by exactly their cells.
    bare = clay_rail.stack_height(0, cell, spacing)
    assert need - bare == len(clay_tools.QUICK_SHAPES) * (cell + spacing)


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
def test_the_rail_stays_inside_the_body_the_resize_floor_leaves_at_every_scale(
    monkeypatch, ui, scale
):
    """Clay's own tool rail -- not the shell's navigation rail, whose floor test is
    ``test_studio_smoke.py::test_the_rail_fits_the_resize_floor_at_every_scale``.

    The body the viewport leaves at 700 px is the window minus its chrome (tab
    bar, menu strip, toolbar, palette strip, hint line and the shell's margins).
    Every cell the rail draws must be inside the child it was given, whether or
    not the quick shapes were dropped, at 1.0, 1.5 and 2.0 -- 2.0 is where a
    labelled 46 px cell stops being small."""
    from realmspinner.studio import tokens
    from realmspinner.studio.modes.clay.ui.panes import header as clay_header
    from realmspinner.studio.modes.clay.ui.panes import hud as clay_hud

    old = tokens.SCALE
    tokens.set_scale(scale)
    try:
        chrome = (
            tokens.sp(clay_header.HEADER_H) * 3.0
            + clay_palette.height()
            + tokens.sp(clay_hud.HINT_H)
            + tokens.sp(60.0)
        )
        body = 700.0 * scale - chrome
        ctx = _Ctx()
        ctx.state = SimpleNamespace(clay=None)
        clay_mode.ensure(ctx)
        boxes: list[tuple[str, float, float]] = []
        real = clay_rail._cell

        def spy(glyph, ident, word, **kw):
            result = real(glyph, ident, word, **kw)
            lo, hi = ui.get_item_rect_min(), ui.get_item_rect_max()
            boxes.append((ident, lo.y, hi.y))
            return result

        monkeypatch.setattr(clay_rail, "_cell", spy)
        tops: list[float] = []

        def draw():
            tops.append(ui.get_cursor_screen_pos().y)
            clay_rail.draw(ctx, body)

        ui.new_frame()
        ui.set_next_window_size((300.0 * scale, body + 40.0 * scale))
        ui.begin("##host", None, ui.WindowFlags_.no_decoration.value)
        draw()
        ui.end()
        ui.end_frame()
    finally:
        tokens.set_scale(old)

    idents = [ident for ident, _lo, _hi in boxes]
    assert all(f"clay-rail/tool/{key}" in idents for key, _l, _k in clay_state.TOOLS)
    assert "clay-rail/more" in idents, "the way to every shape must always be reachable"
    bottom = tops[0] + body
    for ident, _lo, hi in boxes:
        assert hi <= bottom + 1.0, (
            f"{ident} ends at {hi:.0f}, below the rail's {bottom:.0f} at scale {scale}"
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
