"""The 2026-10-11 audit's Clay pane findings: clay-16, -17, -18, -38 and -42.

Each test's name is the claim, and each one fails against the pane code as it stood
before the fix (the failing assertion is quoted in the fixer's report).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _ui_context import imgui_context
from _ui_driver import UiDriver

from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import anchors, widgets
from realmspinner.studio.modes.clay import doc_io
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.ui import inspector_size
from realmspinner.studio.modes.clay.ui.panes import inspector as clay_inspector
from realmspinner.studio.modes.clay.ui.panes import outliner as clay_outliner
from realmspinner.studio.modes.clay.ui.panes import props as clay_props
from realmspinner.studio.modes.clay.ui.panes import swatches
from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools
from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv

# -- a rig -----------------------------------------------------------------------------


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


def _document(shapes: int = 1):
    """A ctx and a Clay document holding *shapes* boxes; the last one placed is selected."""
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    for _ in range(shapes):
        clay_tools.place_primitive(ctx, tab.doc, "box")
    return ctx, tab, tab.doc


def _drag(ctx: Any) -> None:
    """A live G/R/S or gizmo drag, as ``ops.gesture_live`` reads it."""
    ctx.clay_view = SimpleNamespace(dragging=True)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(clay_mode, "recent_paths", lambda ctx: [])
    monkeypatch.setattr(swatches, "_ctrl_held", lambda: False)
    widgets._OPEN_REQUESTS.clear()
    yield
    widgets._OPEN_REQUESTS.clear()


@pytest.fixture
def imgui_ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


class _Rig:
    def __init__(self, imgui: Any, draw: Any, *, height: float = 3000.0) -> None:
        self.imgui = imgui
        self.draw = draw
        self.height = height
        imgui.get_io().display_size = (1600.0, max(height, 950.0))
        self.driver = UiDriver(imgui, self._frame)
        self.driver.settle(3)

    def _frame(self) -> None:
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
        imgui.begin("##host", None, flags)
        self.draw()
        imgui.end()
        imgui.end_frame()

    def click(self, **kw: Any) -> None:
        self.driver.click(self.driver.find(**kw))
        self.driver.settle(2)


# --- clay-16: the swatch and the + under a live drag ---------------------------------


def test_swatch_click_and_plus_under_a_live_drag_add_no_slot_and_push_no_step() -> None:
    ctx, tab, doc = _document(1)
    doc.add_material()
    obj = doc.objects[0]
    _drag(ctx)
    head, slots, material = doc.history.head, len(doc.materials), obj.material

    swatches.click_slot(ctx, doc, 1)

    assert doc.history.head == head, "a swatch pressed mid-drag pushed a step the commit reverts"
    assert doc.objects[0].material == material
    assert ctx.toasts and ctx.toasts[-1] == (doc_io.GESTURE_WHY, "error")

    ctx.toasts.clear()
    swatches.add_slot_and_paint(ctx, doc)

    assert len(doc.materials) == slots, "the + left a stray slot behind"
    assert doc.history.head == head
    assert ctx.toasts == [(doc_io.GESTURE_WHY, "error")]


def test_the_plus_is_greyed_with_the_gesture_sentence_under_a_live_drag() -> None:
    assert swatches.add_reason(saving=False, gesture=True) == doc_io.GESTURE_WHY
    assert swatches.add_reason(saving=False, gesture=False) == ""
    assert swatches.add_reason(saving=True, gesture=True), "a save is still a reason"


def test_a_swatch_pressed_with_no_drag_still_paints_the_object() -> None:
    ctx, tab, doc = _document(1)
    doc.add_material()
    swatches.click_slot(ctx, doc, 1)
    assert doc.objects[0].material == 1 and not ctx.toasts


# --- clay-17: every human Add door under a live drag ---------------------------------


def test_place_primitive_under_a_live_drag_adds_nothing_and_says_why() -> None:
    ctx, tab, doc = _document(1)
    _drag(ctx)
    head, count = doc.history.head, len(doc.objects)

    placed = clay_tools.place_primitive(ctx, doc, "box")

    assert placed is None
    assert len(doc.objects) == count and doc.history.head == head
    assert ctx.toasts == [(doc_io.GESTURE_WHY, "error")]


def test_place_primitive_after_the_drag_ends_places_again() -> None:
    ctx, tab, doc = _document(1)
    _drag(ctx)
    ctx.clay_view.dragging = False
    assert clay_tools.place_primitive(ctx, doc, "box") is not None


# --- clay-18: the outliner and the transform fields under a live drag ----------------


def test_outliner_and_transform_fields_are_greyed_under_a_live_drag(imgui_ui) -> None:
    ctx, tab, doc = _document(2)
    rig = _Rig(imgui_ui, lambda: clay_outliner.draw_embedded(ctx, 400.0))
    head = doc.history.head
    _drag(ctx)
    rig.driver.settle(2)

    rig.click(contains="##claysolo")

    assert doc.history.head == head, "Solo wrote through the outliner mid-drag"
    assert all(o.visible for o in doc.objects)

    # And the same pane with the drag over: the control is live again.
    ctx.clay_view.dragging = False
    rig.driver.settle(2)
    rig.click(contains="##claysolo")
    assert doc.history.head == head + 1


def test_the_inspector_size_and_position_fields_are_dead_under_a_live_drag(imgui_ui) -> None:
    ctx, tab, doc = _document(1)
    obj = doc.objects[0]
    widgets.request_open("clay/insp/position")
    rig = _Rig(imgui_ui, lambda: clay_inspector.draw(ctx), height=4000.0)
    _drag(ctx)
    rig.driver.settle(2)
    head, scale = doc.history.head, tuple(float(v) for v in obj.scale)
    where = tuple(float(v) for v in obj.translation)

    rig.driver.click(rig.driver.find(label="##size##insp"))
    rig.driver.type_text("25")
    rig.driver.press_key(imgui_ui.Key.enter)
    rig.driver.settle(2)
    rig.driver.click(rig.driver.find(label="position##bt"))
    rig.driver.type_text("7")
    rig.driver.press_key(imgui_ui.Key.enter)
    rig.driver.settle(2)

    assert tuple(float(v) for v in obj.scale) == scale, "Size landed mid-drag"
    assert tuple(float(v) for v in obj.translation) == where, "Position landed mid-drag"
    assert doc.history.head == head, "a field pushed a step the drag's commit reverts"


def test_the_transform_door_refuses_under_a_live_drag() -> None:
    """The door under the greyed fields: ``_apply_transform`` serves Position, Rotation,
    Scale and Size, and ``inspector_size.resize`` goes through it."""
    ctx, tab, doc = _document(1)
    obj = doc.objects[0]
    _drag(ctx)
    head = doc.history.head

    assert not clay_props._apply_transform(doc, obj, ctx, translation=[3.0, 0.0, 0.0])
    assert not inspector_size.resize(ctx, doc, obj, np.array([2.0, 2.0, 2.0]))

    assert doc.history.head == head
    assert np.allclose(doc.objects[0].translation, obj.translation)
    ctx.clay_view.dragging = False
    assert clay_props._apply_transform(doc, obj, ctx, translation=[3.0, 0.0, 0.0])


def test_every_pane_door_that_writes_history_is_gated_on_a_live_gesture() -> None:
    """SWEEP. The class the 2026-10-07 audit's clay-12 and the 2026-10-09 second audit's
    clay-61 closed for ``run`` and undo, found again by the 2026-10-11 audit at the panes
    (clay-16, -17, -18). Reads the sources, because a drag cannot be armed through a real
    pointer here; each reading names the gate its door must keep."""
    gates = {
        "outliner body": (clay_outliner._body, "gesture_live"),
        "inspector one-shape card": (clay_inspector._one, "gesture_live"),
        "transform door": (clay_props._apply_transform, "gesture_live"),
        "swatch click": (swatches.click_slot, "refused_by_drag"),
        "the shared refusal": (swatches.refused_by_drag, "gesture_live"),
        "swatch plus": (swatches.add_slot_and_paint, "refused_by_drag"),
        "place_primitive": (clay_tools.place_primitive, "refused_by_drag"),
    }
    missing = [
        name for name, (fn, needle) in gates.items() if needle not in inspect.getsource(fn)
    ]
    assert missing == [], f"doors with no live-gesture gate: {missing}"


# --- clay-38: the UV canvas fills a concave face as triangles ------------------------


def _l_shaped_face() -> bm.Mesh:
    positions = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ]
    uv = [[(x / 2.0, y / 2.0) for x, y, _z in positions]]
    return bm.from_faces(positions, [[0, 1, 2, 3, 4, 5]], uv)


class _Draw:
    def __init__(self) -> None:
        self.polys: list[list[tuple[float, float]]] = []

    def add_convex_poly_filled(self, points: Any, colour: int) -> None:
        self.polys.append(list(points))

    def add_triangle_filled(self, a: Any, b: Any, c: Any, colour: int) -> None:
        self.polys.append([tuple(a), tuple(b), tuple(c)])

    def add_polyline(self, *args: Any) -> None:
        pass

    def add_line(self, *args: Any) -> None:
        pass


def _area(points: Any) -> float:
    xs, ys = zip(*points, strict=True)
    return 0.5 * abs(
        sum(xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i] for i in range(len(xs)))
    )


def test_a_concave_face_is_filled_as_triangles_not_one_convex_polygon(imgui_ui) -> None:
    from realmspinner.studio.shell import paintview

    mesh = _l_shaped_face()
    view = paintview.PaintView()
    origin = (10.0, 20.0)
    geo: dict[str, Any] = {}
    draw = _Draw()

    clay_uv._faces(draw, view, origin, mesh, None, geo)

    assert len(draw.polys) == 4, f"a hexagon is four triangles, not {len(draw.polys)} fill(s)"
    assert all(len(p) == 3 for p in draw.polys), "a concave face went to the convex fill whole"
    outline = clay_uv._screen_corners(geo, view, origin, mesh)
    assert sum(_area(p) for p in draw.polys) == pytest.approx(_area(outline), rel=1e-9), (
        "the triangles must cover the L exactly: no wedge over the notch"
    )

    # Memoised with the other screen-space geometry: a second frame does not rebuild it.
    again = _Draw()
    clay_uv._faces(again, view, origin, mesh, None, geo)
    assert again.polys == draw.polys


def test_convex_faces_are_still_one_fill_each(imgui_ui) -> None:
    from realmspinner.studio.shell import paintview

    mesh = bp.box()
    draw = _Draw()
    clay_uv._faces(draw, paintview.PaintView(), (0.0, 0.0), mesh, None, {})
    assert len(draw.polys) == len(mesh.starts) - 1


# --- clay-42: the Colours & textures page names the shape it edits -------------------


def test_colours_page_names_the_shape_it_edits_when_several_are_selected() -> None:
    ctx, tab, doc = _document(3)
    one = doc.objects[0]
    doc.select([o.uid for o in doc.objects])

    note = clay_props._edit_slot_note(doc, one, 0, tab)

    assert one.name in note, note
    assert "3" in note, "it should say how many are selected"


def test_colours_page_note_is_unchanged_for_one_shape() -> None:
    ctx, tab, doc = _document(2)
    obj = doc.objects[1]
    doc.select([obj.uid])
    note = clay_props._edit_slot_note(doc, obj, 0, tab)
    assert note.startswith("Editing slot 0")
    assert obj.name not in note
