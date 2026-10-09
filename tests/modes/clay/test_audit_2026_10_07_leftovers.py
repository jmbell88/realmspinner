"""Regressions for the narrow leftovers the 2026-10-07 Clay audit's fix pass listed.

One block per finding id: clay-11 (the three element doors that skipped the
hidden-object check), clay-06 (hiding deselects), clay-14 (the typed median's
silent refusal), clay-16 (a refused UV move ends the drag), clay-83 (hover
clears when the pointer leaves the viewport), clay-03 (the box's X walls stood
on their side) and clay-35 (``round``'s digit count).
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import uv as bmuv
from realmspinner.studio.modes.clay import element_move
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.agent import program as ap
from realmspinner.studio.modes.clay.ui import view as clay_view
from realmspinner.studio.modes.clay.ui.panes import props as clay_props


class _Ctx:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.info: list[str] = []

    def toast(self, message: str, level: str = "info") -> None:
        (self.errors if level == "error" else self.info).append(message)


def _box(doc: bd.ClayDoc, name: str = "Box") -> bd.Obj:
    return doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name=name,
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
        )
    )


def _two_boxes_selected(mode: str) -> tuple[bd.ClayDoc, bd.Obj, bd.Obj]:
    """Both boxes carry a *mode* selection, the second then hidden **under** it.

    ``visible`` is flipped on the object rather than through ``set_visibility``:
    that door deselects (clay-06), and the stale selection these tests are about
    is the one an undone Show leaves behind.
    """
    doc = bd.ClayDoc()
    a, b = _box(doc, "a"), _box(doc, "b")
    doc.set_element_mode(mode)
    for obj in (a, b):
        doc.set_element_sel(obj.uid, el.select_all(obj.mesh, mode))
    b.visible = False
    return doc, a, b


# --- clay-11: the three doors that did not ask whether the object was visible ---


def test_assign_material_leaves_a_hidden_objects_faces_alone() -> None:
    doc, a, b = _two_boxes_selected("face")
    slot = doc.add_material()
    hidden_mesh = b.mesh
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("assign-material"), index=slot)
    assert doc.by_uid(b.uid).mesh is hidden_mesh, "a hidden object's faces were repainted"
    assert np.all(doc.by_uid(a.uid).mesh.material == slot), "the visible one was painted"


def test_shade_in_face_mode_leaves_a_hidden_object_alone() -> None:
    doc, a, b = _two_boxes_selected("face")
    hidden_mesh = b.mesh
    assert not np.any(a.mesh.smooth), "the premise: a box starts flat"
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("shade-smooth")) is not False
    assert doc.by_uid(b.uid).mesh is hidden_mesh, "a hidden object was re-shaded"
    assert np.all(doc.by_uid(a.uid).mesh.smooth), "the visible one was shaded"


def test_a_selection_verb_does_not_rewrite_a_hidden_objects_selection() -> None:
    doc = bd.ClayDoc()
    a, b = _box(doc, "a"), _box(doc, "b")
    doc.set_element_mode("vertex")
    one = replace(el.empty(), verts=np.array([0], dtype="i4"))
    doc.set_element_sel(a.uid, one)
    doc.set_element_sel(b.uid, one)
    b.visible = False
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("select-linked"))
    assert len(doc.element_sel_of(a.uid).verts) == 8, "the visible object grew to its whole mesh"
    assert len(doc.element_sel_of(b.uid).verts) == 1, "a hidden object's selection was rewritten"


# --- clay-06: hiding deselects -----------------------------------------------------


def test_hiding_an_object_deselects_it() -> None:
    doc = bd.ClayDoc()
    a, b = _box(doc, "a"), _box(doc, "b")
    doc.select([a.uid, b.uid])
    assert doc.set_visibility({b.uid: False})
    assert doc.selection == {a.uid}


def test_hiding_an_object_drops_its_element_selection_too() -> None:
    doc = bd.ClayDoc()
    a, b = _box(doc, "a"), _box(doc, "b")
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, el.select_all(a.mesh, "face"))
    doc.set_element_sel(b.uid, el.select_all(b.mesh, "face"))
    doc.set_visibility({b.uid: False})
    assert set(doc.element_sel) == {a.uid}
    assert doc.selection == {a.uid}, "the hidden shape is no longer one being edited"
    assert doc.selection_problems() == []


def test_isolating_deselects_everything_it_hid_and_keeps_what_it_kept() -> None:
    doc = bd.ClayDoc()
    a, b = _box(doc, "a"), _box(doc, "b")
    doc.select([a.uid, b.uid])
    assert doc.isolate([a.uid])
    assert doc.selection == {a.uid}


def test_showing_an_object_does_not_select_it() -> None:
    doc = bd.ClayDoc()
    a = _box(doc, "a")
    doc.set_visibility({a.uid: False})
    doc.set_visibility({a.uid: True})
    assert doc.selection == set()


# --- clay-14: the typed median's refusal reaches the caller -------------------------


def test_commit_median_reports_the_object_the_document_refused() -> None:
    doc = bd.ClayDoc()
    a = _box(doc, "a")
    doc.set_element_mode("vertex")
    doc.set_element_sel(a.uid, el.select_all(a.mesh, "vertex"))
    median = element_move.element_median_world(doc)
    shown = [float(v) for v in median]
    typed = [1.0e300, shown[1], shown[2]]
    refused: list[Exception] = []
    before = a.mesh
    assert clay_props.commit_median(doc, median, shown, typed, "m", refused) is False
    assert len(refused) == 1, "the overflowing move was refused with nothing to say so"
    assert str(refused[0])
    assert doc.by_uid(a.uid).mesh is before


def test_move_elements_without_a_refusal_list_still_returns_quietly() -> None:
    doc = bd.ClayDoc()
    a = _box(doc, "a")
    doc.set_element_mode("vertex")
    doc.set_element_sel(a.uid, el.select_all(a.mesh, "vertex"))
    assert element_move.move_elements(doc, np.array([1.0e300, 0.0, 0.0])) is False


# --- clay-83: hover clears when the pointer is not over the viewport -----------------


def _hover_view(**over: Any) -> Any:
    gizmo = SimpleNamespace(hover=over.pop("gizmo_hover", None))
    base: dict[str, Any] = {
        "_grab": None,
        "hover_element": None,
        "hover_object": None,
        "_hover_pick_at": None,
        "_render_dirty": False,
        "active_gizmo": lambda doc: gizmo,
        "gizmo": gizmo,
    }
    base.update(over)
    return SimpleNamespace(**base)


def test_clear_hover_drops_a_lit_element_an_object_and_a_gizmo_arm() -> None:
    view = _hover_view(hover_element=(1, 5), hover_object=2, gizmo_hover="x", _hover_pick_at=(4, 4))
    assert clay_view.ClayView.clear_hover(view, None) is True
    assert view.hover_element is None and view.hover_object is None
    assert view.gizmo.hover is None and view._hover_pick_at is None
    assert view._render_dirty, "the cleared highlight must be redrawn"


def test_clear_hover_with_nothing_lit_asks_for_no_redraw() -> None:
    view = _hover_view()
    assert clay_view.ClayView.clear_hover(view, None) is False
    assert not view._render_dirty


def test_clear_hover_leaves_a_live_grab_alone() -> None:
    view = _hover_view(hover_element=(1, 5), _grab="orbit")
    assert clay_view.ClayView.clear_hover(view, None) is False
    assert view.hover_element == (1, 5)


# --- clay-03: the box's X walls are upright, not on their side -----------------------


def _wall(mesh: Any, axis: int, sign: float) -> list[int]:
    """The corner indices of the faces of *mesh* whose normal points *sign* along *axis*."""
    from realmspinner.kernels.mesh.mesh import face_normals

    normals = face_normals(mesh)
    faces = [f for f in range(len(normals)) if normals[f, axis] * sign > 0.5]
    out: list[int] = []
    for f in faces:
        out.extend(range(int(mesh.starts[f]), int(mesh.starts[f + 1])))
    return out


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_the_x_walls_run_v_down_the_height_not_along_z(sign: float) -> None:
    mesh = bmuv.box_unwrap(bp.box())
    corners = _wall(mesh, 0, sign)
    pos = mesh.positions[mesh.loops[corners]]
    uv = np.asarray(mesh.uv)[corners]
    # v is image-down: the higher the corner, the smaller its v, and nothing
    # about v depends on Z.
    assert np.corrcoef(pos[:, 1], uv[:, 1])[0, 1] < -0.99, "v does not follow height"
    assert np.ptp(uv[:, 1]) > 0.5, "v spans the wall's height"


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_the_x_walls_read_unmirrored_seen_from_outside(sign: float) -> None:
    mesh = bmuv.box_unwrap(bp.box())
    corners = _wall(mesh, 0, sign)
    pos = mesh.positions[mesh.loops[corners]]
    uv = np.asarray(mesh.uv)[corners]
    # Seen from +X the viewer's right is -Z; from -X it is +Z.
    right = -sign * pos[:, 2]
    assert np.corrcoef(right, uv[:, 0])[0, 1] > 0.99, "u runs against the viewer's right"


def test_every_box_wall_has_the_upright_uv_sign() -> None:
    """The one sign the module docstring names: negative in plain (u, v)."""
    mesh = bmuv.box_unwrap(bp.box())
    uv = np.asarray(mesh.uv, dtype="f8")
    for f in range(len(mesh.starts) - 1):
        q = uv[int(mesh.starts[f]) : int(mesh.starts[f + 1])]
        area = 0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])
        assert area < 0.0, f"face {f} reads mirrored"


# --- clay-35: round's digit count must be whole ---------------------------------------


def _compile_translation(expr: str) -> Any:
    return ap.compile_program(
        {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"transform": {"uid": "a", "translation": [expr, 0, 0]}},
            ]
        }
    )


def test_round_refuses_a_fractional_digit_count() -> None:
    with pytest.raises(ap.ProgramError):
        _compile_translation("round(1.2345, 2.7)")


def test_round_still_takes_a_whole_digit_count() -> None:
    compiled = _compile_translation("round(1.2345, 2)")
    (call,) = [c for c in compiled.calls if c[0] == "clay_transform"]
    assert call[1]["translation"][0] == pytest.approx(1.23)


# --- clay-16: a refused UV move ends the drag instead of toasting every frame ---------


def test_a_refused_uv_move_toasts_once_and_ends_the_gesture(monkeypatch) -> None:
    from _ui_context import imgui_context

    from realmspinner.kernels.mesh import mesh as bm
    from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv

    positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    mesh = bm.from_faces(
        positions, [[0, 1, 2, 3]], uv=[[[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]]]
    )
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=mesh))
    obj = doc.by_uid(obj.uid)
    tab = SimpleNamespace(doc=doc, uv_view=clay_uv.UvPaneState(), saving=False, uid="t")
    view_state = tab.uv_view
    view_state.drag_mode = "move"
    view_state.drag_last = (-9.0, -9.0)
    view_state.selected_islands = frozenset({0})

    calls: list[int] = []

    def refuse(*_a: Any, **_k: Any) -> bool:
        calls.append(1)
        raise el.OpError("past the island ceiling")

    ctx = _Ctx()
    with imgui_context(monkeypatch) as ui:
        monkeypatch.setattr(clay_uv, "apply_translate", refuse)
        monkeypatch.setattr(ui, "is_item_active", lambda: True)
        monkeypatch.setattr(ui, "is_mouse_dragging", lambda *_a: True)
        for _ in range(3):
            ui.new_frame()
            ui.begin("##host")
            try:
                clay_uv._canvas(ctx=ctx, tab=tab, doc=doc, obj=obj, view_state=view_state)
            finally:
                ui.end()
                ui.end_frame()
            view_state.drag_last = (-9.0, -9.0)

    assert len(calls) == 1, "the drag kept asking for a move that had been refused"
    assert ctx.errors == ["past the island ceiling"], "one toast, not one per frame"
    assert view_state.drag_mode == "refused"
    assert view_state.selected_islands == frozenset({0}), "the box select took the drag over"


def test_the_viewport_pane_clears_hover_on_a_frame_the_image_is_not_hovered() -> None:
    import inspect

    from realmspinner.studio.modes.clay.ui import viewport

    source = inspect.getsource(viewport)
    assert "view.clear_hover(tab.doc)" in source
    assert source.index("if not self._build_hovered") < source.index("view.clear_hover(")
