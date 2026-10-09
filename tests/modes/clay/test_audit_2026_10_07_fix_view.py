"""The 2026-10-07 audit's Clay viewport findings: clay-06, 11, 12, 21, 22, 83-87.

The viewport owns a renderer, so most of these ride the session ``gl`` fixture
and skip where there is no GPU, like every sibling in this package.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest
from PIL import Image

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import viewport_hints as clay_hints
from realmspinner.studio.modes.clay.ui import _view_overlay
from realmspinner.studio.modes.clay.ui import view as clay_view

RECT = (0.0, 0.0, 200.0, 150.0)


class _State:
    def __init__(self, tool: str = "select") -> None:
        self.tool = tool
        self.snap = False
        self.snap_translate = 0.125
        self.snap_rotate = 15.0


class _Ctx:
    def __init__(self, tool: str = "select") -> None:
        self.state = type("S", (), {"clay": _State(tool)})()


@pytest.fixture(autouse=True)
def _headless_mods(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0, raising=False)


@pytest.fixture
def view(gl):
    v = clay_view.ClayView(gl, _Ctx("select"))
    yield v
    v.release()


def _doc(count: int = 1, mesh: Any = None) -> bd.ClayDoc:
    doc = bd.ClayDoc()
    for i in range(count):
        doc.add_object(
            bd.Obj(
                uid=bd.new_uid(),
                name=f"o{i}",
                mesh=bp.box() if mesh is None else mesh,
                translation=m3.vec3(float(i) * 3.0, 0.0, 0.0),
            )
        )
    return doc


def _event(kind: int, **kwargs: Any) -> Any:
    import pygame

    return pygame.event.Event(kind, **kwargs)


def _count_index_of(monkeypatch) -> dict[str, int]:
    scans = {"n": 0}
    real = bd.ClayDoc.index_of

    def counting(self: Any, uid: int) -> int:
        scans["n"] += 1
        return real(self, uid)

    monkeypatch.setattr(bd.ClayDoc, "index_of", counting)
    return scans  # the caller zeroes ``n`` before each call it measures


# --- clay-06: a selected-but-hidden object is not dragged ---------------------


def test_an_object_drag_leaves_a_selected_hidden_object_where_it_is(view) -> None:
    doc = _doc(count=2)
    a, b = doc.objects
    doc.select([a.uid, b.uid])
    doc.set_props(b.uid, visible=False)  # the outliner's eye keeps it selected
    assert b.uid in doc.selection
    view.draw(doc, RECT, 0.016)
    view._last_mouse = (100.0, 75.0)
    before = np.array(b.translation, copy=True)

    assert view.begin_keyboard_drag(doc, "move")
    view._drag_keyboard(doc, (160.0, 75.0))

    assert list(view._drag_start) == [a.uid], "the hidden object joined the drag set"
    assert np.array_equal(b.translation, before), "a hidden object was moved unseen"
    view.cancel_drag(doc)


def test_a_drag_with_only_hidden_objects_selected_does_not_start_and_leaves_no_gizmo(view) -> None:
    doc = _doc(count=1)
    only = doc.objects[0]
    doc.select([only.uid])
    doc.set_props(only.uid, visible=False)
    view.draw(doc, RECT, 0.016)

    assert view.active_gizmo(doc) is None, "a gizmo is offered for nothing visible"
    assert not view._begin_gizmo_drag(doc)
    assert not view.dragging
    view._last_mouse = (100.0, 75.0)
    assert not view.begin_keyboard_drag(doc, "move")


def test_a_visible_child_of_a_hidden_selected_parent_still_moves(view) -> None:
    """The ancestor rule drops a child whose ancestor the drag carries; a hidden
    parent is not carried, so its visible child has to move itself."""
    doc = _doc(count=2)
    parent, child = doc.objects
    child.parent = parent.uid
    doc.select([parent.uid, child.uid])
    doc.set_props(parent.uid, visible=False)
    view.draw(doc, RECT, 0.016)
    view._last_mouse = (100.0, 75.0)
    assert view.begin_keyboard_drag(doc, "move")
    assert list(view._drag_start) == [child.uid]
    view.cancel_drag(doc)


# --- clay-11 (the view's half): the extrude normal skips a hidden object ------


def test_the_extrude_normal_ignores_a_hidden_objects_stale_selection(view) -> None:
    doc = _doc(count=2)
    visible, hidden = doc.objects
    doc.set_element_mode("face")
    normals = bm.face_normals(visible.mesh)
    up = int(np.argmax(normals[:, 1]))
    right = int(np.argmax(normals[:, 0]))
    from realmspinner.kernels.mesh import elements as el

    doc.set_element_sel(visible.uid, el.ElementSel(faces=[up]))
    doc.set_element_sel(hidden.uid, el.ElementSel(faces=[right]))
    doc.set_props(hidden.uid, visible=False)

    normal = view._selection_normal(doc)

    assert normal is not None
    assert np.allclose(normal, [0.0, 1.0, 0.0], atol=1e-6), normal


# --- clay-12 (the view's half): a press outside the viewport ends a key drag --


def _start_key_drag(view: Any, doc: Any) -> Any:
    obj = doc.objects[0]
    doc.select([obj.uid])
    view.draw(doc, RECT, 0.016)
    view._last_mouse = (100.0, 75.0)
    assert view.begin_keyboard_drag(doc, "move")
    view._drag_keyboard(doc, (150.0, 75.0))
    assert not np.array_equal(obj.translation, [0, 0, 0])
    return obj


def test_a_left_press_outside_the_viewport_commits_a_keyboard_drag(view) -> None:
    import pygame

    doc = _doc(count=1)
    obj = _start_key_drag(view, doc)
    moved = np.array(obj.translation, copy=True)

    consumed = view.handle_event(
        doc, _event(pygame.MOUSEBUTTONDOWN, button=1, pos=(900, 40)), False
    )

    assert consumed
    assert not view.dragging, "a press on a panel left the keyboard drag live"
    assert np.allclose(obj.translation, moved), "the commit lost the displacement"
    assert doc.history.top is not None and doc.history.top.label == "Move"


def test_a_right_press_outside_the_viewport_cancels_a_keyboard_drag(view) -> None:
    import pygame

    doc = _doc(count=1)
    obj = _start_key_drag(view, doc)
    top_before = doc.history.top

    view.handle_event(doc, _event(pygame.MOUSEBUTTONDOWN, button=3, pos=(900, 40)), False)

    assert not view.dragging
    assert np.allclose(obj.translation, [0.0, 0.0, 0.0]), "the cancel left the object moved"
    assert doc.history.top is top_before, "a cancel recorded a step"


def test_a_press_outside_the_viewport_with_no_keyboard_drag_is_still_ignored(view) -> None:
    import pygame

    doc = _doc(count=1)
    view.draw(doc, RECT, 0.016)
    assert not view.handle_event(
        doc, _event(pygame.MOUSEBUTTONDOWN, button=1, pos=(900, 40)), False
    )
    assert view._grab is None


# --- clay-21: pick, overlays and drag motion do not scan per object -----------


def _big_doc(count: int = 4096) -> bd.ClayDoc:
    mesh = bp.box()  # one frozen Mesh shared, so the BVH is built once
    doc = _doc(count=count, mesh=mesh)
    return doc


def test_pick_and_a_drag_motion_over_four_thousand_objects_do_not_scan_the_document_per_object(
    view, monkeypatch
) -> None:
    doc = _big_doc()
    view._rect = RECT
    view.camera.aspect = RECT[2] / RECT[3]
    scans = _count_index_of(monkeypatch)

    def measured(label: str, call: Any) -> None:
        scans["n"] = 0
        call()
        assert scans["n"] == 0, f"{label} scanned the document {scans['n']} times"

    measured("pick_face", lambda: view.pick_face(doc, (100.0, 75.0)))
    doc.set_element_mode("vertex")
    measured("pick_element", lambda: view.pick_element(doc, (100.0, 75.0)))
    measured("world_bounds", lambda: view.world_bounds(doc))

    def sweep() -> None:
        # A marquee release projects every object (screen_of) once each.
        view._grab = "marquee"
        view.marquee = (0.0, 0.0, 200.0, 150.0)
        view._marquee_from = (0.0, 0.0)
        view._marquee_add = "replace"
        view._release_drag(doc, 1)

    measured("a marquee release", sweep)
    doc.clear_element_sel()
    doc.set_element_mode("object")

    doc.select([o.uid for o in doc.objects])
    view.frame_selection(doc)  # the pivot has to be in front of the camera
    view._last_mouse = (100.0, 75.0)
    measured("world_bounds of the selection", lambda: view.world_bounds(doc, selected_only=True))
    measured("the press of a drag over every object", lambda: view.begin_keyboard_drag(doc, "move"))
    assert view.dragging
    measured("a keyboard drag motion", lambda: view._drag_keyboard(doc, (140.0, 75.0)))
    # ``_drag_gizmo`` asks the gizmo for its delta; hand it one so the loop runs.
    monkeypatch.setattr(view.translate_gizmo, "update", lambda o, d: np.array([1.0, 0.0, 0.0]))
    measured("a gizmo drag motion", lambda: view._drag_gizmo(doc, (140.0, 75.0)))
    view.cancel_drag(doc)


def test_the_element_overlays_over_four_thousand_objects_do_not_scan_the_document(
    view, monkeypatch
) -> None:
    class _Stub:
        def __init__(self, ctx: Any, program: Any, key: Any, positions: Any) -> None:
            self.key = key
            self.pins = None
            self.specs: list[Any] = []
            self.hover = -1
            self.hover_specs: list[Any] = []

        def release(self) -> None:
            pass

    monkeypatch.setattr(_view_overlay, "_SelOverlay", _Stub)
    doc = _big_doc()
    doc.set_element_mode("vertex")
    scans = _count_index_of(monkeypatch)
    view._element_overlays(doc)
    assert scans["n"] == 0, f"_element_overlays scanned the document {scans['n']} times"
    doc.set_element_mode("object")
    view._object_overlays(doc)
    assert scans["n"] == 0, f"_object_overlays scanned the document {scans['n']} times"


def test_the_indexed_drag_writes_what_the_documents_own_drag_wrote(view) -> None:
    """The indexed ``_apply`` is the same arithmetic, parented or not."""
    doc = _doc(count=3)
    a, b, c = doc.objects
    b.parent, c.parent = a.uid, b.uid
    a.rotation = m3.quat_from_axis_angle(m3.vec3(0, 1, 0), 0.4)
    a.scale = m3.vec3(2.0, 2.0, 2.0)
    state = _State("move")
    view._drag_origin = np.zeros(3)
    delta = np.array([1.0, 0.5, -0.25])
    index = {o.uid: o for o in doc.objects}
    for obj in doc.objects:
        was = tuple(np.array(v, copy=True) for v in obj.trs())
        view._apply(doc, obj, was, delta, state)
        plain = tuple(np.array(v, copy=True) for v in obj.trs())
        obj.translation, obj.rotation, obj.scale = (np.array(v, copy=True) for v in was)
        view._apply(doc, obj, was, delta, state, index)
        for x, y in zip(plain, obj.trs(), strict=True):
            assert np.allclose(x, y)
        obj.translation, obj.rotation, obj.scale = (np.array(v, copy=True) for v in was)


# --- clay-22: object-id colours are collision-free ----------------------------


def test_object_id_colours_are_distinct_for_four_thousand_objects() -> None:
    uids = list(range(1, 4097))
    colors = clay_view._id_colors(uids)
    assert len(colors) == 4096
    assert len(set(colors.values())) == 4096, "two objects share an id colour"
    assert (255, 255, 255) not in colors.values(), "an id collides with the clear colour"
    back = clay_view._uid_of_colors(colors)
    assert all(back[color] == uid for uid, color in colors.items()), "colour -> uid round trip"
    # Single colours for 4,096 consecutive indices are distinct too.
    assert len({clay_view._id_color(i) for i in range(1, 4097)}) == 4096


def test_object_id_colours_are_distinct_for_uids_the_hue_walk_used_to_merge(view) -> None:
    """uids 1 and 611 shared (36, 96, 242) under the golden-ratio hue step, so
    ``clay_render``'s table reported one hex for two objects and summed their
    pixels."""
    doc = bd.ClayDoc()
    left = bd.Obj(uid=1, name="l", mesh=bp.box(), translation=m3.vec3(-1.5, 0, 0))
    right = bd.Obj(uid=611, name="r", mesh=bp.box(), translation=m3.vec3(1.5, 0, 0))
    doc.add_object(left)
    doc.add_object(right)
    png, rows = view.render_ids(doc, size=96, view="front")
    by_uid = {uid: (hexcolor, px) for uid, hexcolor, px in rows}
    assert by_uid[1][0] != by_uid[611][0], "two objects reported one colour"
    assert by_uid[1][1] > 0 and by_uid[611][1] > 0
    pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).reshape(-1, 3)
    for hexcolor, px in by_uid.values():
        color = tuple(int(hexcolor[i : i + 2], 16) for i in (1, 3, 5))
        assert int((pixels == color).all(axis=1).sum()) == px


# --- clay-83: the hover highlight does not survive an element-mode switch -----


def test_switching_element_mode_clears_the_hover_highlight(view) -> None:
    doc = _doc(count=1)
    obj = doc.objects[0]
    doc.set_element_mode("vertex")
    view.draw(doc, (0, 0, 128, 96), 0.016)
    view.hover_element = (obj.uid, 5)
    view.draw(doc, (0, 0, 128, 96), 0.016)

    doc.set_element_mode("face")  # the 3 key, with the mouse still
    view.draw(doc, (0, 0, 128, 96), 0.016)

    assert view.hover_element is None
    assert view._overlays[obj.uid].hover == -1


# --- clay-84: an offscreen render does not evict the displayed cache ----------


def test_rendering_another_document_does_not_evict_the_displayed_documents_gpu_cache(view) -> None:
    shown, agent = _doc(count=50), _doc(count=1)
    view.draw(shown, (0, 0, 128, 96), 0.016)
    assert len(view._cache) == 50
    view.render_png(agent, size=64)
    view.render_ids(agent, size=64)
    assert len(view._cache) == 50, "the agent's render replaced the displayed cache"

    redraw_from = view.rebuilds
    view._render_dirty = True
    view.draw(shown, (0, 0, 128, 96), 0.016)
    assert view.rebuilds == redraw_from, (
        f"the displayed document re-uploaded {view.rebuilds - redraw_from} objects "
        "after another was rendered"
    )
    # And the render still draws the other document's objects.
    pixels = np.asarray(
        Image.open(io.BytesIO(view.render_png(agent, size=64, view="front"))).convert("RGB")
    )
    assert not (pixels == 255).all()


def test_the_private_offscreen_caches_are_bounded_and_released(view) -> None:
    shown = _doc(count=2)
    view.draw(shown, (0, 0, 64, 48), 0.016)
    others = [_doc(count=1) for _ in range(6)]
    for other in others:
        view.render_png(other, size=32)
    assert len(view._offscreen_caches) <= 2
    view.clear()
    assert not view._offscreen_caches


# --- clay-85: a live marquee does not survive a tab switch --------------------


def _start_marquee(view: Any, doc: Any) -> None:
    """Begin a box select the only way left: a Shift press on empty space, then
    movement. (A plain press there orbits; see ``test_clay_input_model``.)"""
    import pygame

    doc.set_element_mode("face")
    view.draw(doc, RECT, 0.0)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_SHIFT, raising=False)
        view.handle_event(doc, _event(pygame.MOUSEBUTTONDOWN, button=1, pos=(1, 1)), True)
    view.handle_event(doc, _event(pygame.MOUSEMOTION, pos=(190, 140)), True)
    assert view._grab == "marquee"


def test_a_tab_switch_mid_marquee_does_not_select_in_the_new_document(view) -> None:
    import pygame

    old, new = _doc(count=1), _doc(count=1)
    _start_marquee(view, old)

    assert view.cancel_marquee()  # what the tab switch calls

    assert view.marquee is None and view._marquee_from is None and view._grab is None
    new.set_element_mode("face")
    view.draw(new, RECT, 0.0)
    view.handle_event(new, _event(pygame.MOUSEBUTTONUP, button=1, pos=(190, 140)), True)
    assert not new.element_sel, "the old document's rectangle selected in the new one"
    assert not view.cancel_marquee()


def test_settling_a_drag_also_drops_a_live_marquee(view) -> None:
    doc = _doc(count=1)
    _start_marquee(view, doc)
    assert view.settle_drag(doc)
    assert view.marquee is None and view._grab is None


# --- clay-86: the hint promises only the gestures the tool has ----------------
# It used to confine the marquee clause to the Select tool, the one tool where a
# plain drag swept a box. The box is Shift's and Ctrl's under every tool now (a
# plain drag orbits everywhere), so the honest line names it under all four.


@pytest.mark.parametrize("mode", ["vertex", "edge", "face"])
@pytest.mark.parametrize("tool", ["select", "move", "rotate", "scale"])
def test_the_hint_names_the_box_select_under_every_tool_and_never_a_plain_drag_sweep(
    mode, tool
) -> None:
    line = clay_hints.hint(mode, tool)
    assert "hold Shift and drag to select several" in line, f"{mode}/{tool}: {line!r}"
    assert "drag marquee" not in line, "a plain drag orbits now; it sweeps nothing"
    assert "L linked" in line and "4 object" in line


# --- clay-87: the extrude drag's normal lock is not snapped (documented) -------


def test_an_extrude_drag_is_not_snapped_and_a_plain_move_is(view) -> None:
    """The normal lock counts as an explicit axis, so Snap stands down for it --
    which the Snap section of the Clay chapter now says. This pins the other
    half: the same drag without the lock *is* snapped."""
    from realmspinner.kernels.mesh import drag as bdrag

    state = view.state
    state.snap = True
    state.snap_translate = 0.5
    view._element_centre = np.zeros(3)
    delta = np.array([0.0, 0.3, 0.0])

    view.drag_input = bdrag.DragInput()
    assert view._element_snap_step(delta, state) == 0.5

    view.drag_input = bdrag.DragInput(axis="normal", normal=np.array([0.0, 1.0, 0.0]))
    assert view._element_snap_step(delta, state) == 0.0
    moved = view._narrow(_doc(count=1), np.array([0.0, 0.3, 0.0]), state, (0.0, 0.0))
    assert np.allclose(moved, [0.0, 0.3, 0.0]), "the normal-locked distance was quantised"

