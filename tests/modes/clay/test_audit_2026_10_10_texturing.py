"""The 2026-10-10 audit's texturing findings (clay-13, clay-14, clay-42, clay-43).

Each test's name is the claim, and each failed against the code it was written for.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import uv as bmuv
from realmspinner.kernels.mesh import uvtools
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay import texture_link as tl
from realmspinner.studio.modes.clay.state import ClayTab
from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv
from realmspinner.studio.modes.inker import opening as inker_open


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


# --- clay-13 ------------------------------------------------------------------


def test_touched_islands_for_a_face_selection_ignores_faces_that_only_share_a_vertex() -> None:
    mesh = bp.box()
    ids = uvtools.islands(mesh)
    assert len(set(ids.tolist())) == 6, "a box unwraps to six islands"
    for face in range(6):
        sel = el.ElementSel(faces=np.array([face], dtype="i4"))
        assert clay_uv.touched_islands(mesh, ids, sel) == {int(ids[face])}, face


def test_touched_islands_for_an_edge_selection_names_only_the_faces_that_own_the_edge() -> None:
    mesh = bp.box()
    ids = uvtools.islands(mesh)
    start = int(mesh.starts[2])
    a, b = int(mesh.loops[start]), int(mesh.loops[start + 1])  # one side of face 2
    sel = el.ElementSel(edges=np.array([[a, b]], dtype="i4"))
    touched = clay_uv.touched_islands(mesh, ids, sel)
    # A box edge is shared by exactly two faces; the vertex union would name
    # every face round both endpoints (five of the six).
    assert int(ids[2]) in touched
    assert len(touched) == 2, touched


def test_touched_islands_for_a_vertex_selection_is_still_the_union() -> None:
    mesh = bp.box()
    ids = uvtools.islands(mesh)
    sel = el.ElementSel(verts=np.array([int(mesh.loops[0])], dtype="i4"))
    assert len(clay_uv.touched_islands(mesh, ids, sel)) == 3  # a box corner meets three faces


# --- clay-14 ------------------------------------------------------------------


def test_pack_islands_is_greyed_with_a_reason_outside_object_mode() -> None:
    pack = clay_ops.get("pack-uv")
    why = clay_uv.unwrap_op_reason(pack, "face", saving=False)
    # The greyed sentence speaks the header's words ("Shape", not "object"):
    # ``ops.MODE_WORDS``.
    assert why and "Shape" in why
    assert clay_uv.unwrap_op_reason(pack, "object", saving=False) == ""
    assert clay_uv.unwrap_op_reason(pack, "object", saving=True) == "Saving..."
    box = clay_ops.get("unwrap")
    assert clay_uv.unwrap_op_reason(box, "vertex", saving=False)
    assert clay_uv.unwrap_op_reason(box, "object", saving=True) == "Saving..."


def _toolbar_buttons(ui, monkeypatch, *, element_mode: str, saving: bool):
    doc = bd.ClayDoc()
    mesh = bmuv.box_unwrap(bp.box())
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=mesh))
    doc.select([obj.uid])
    if element_mode != "object":
        doc.set_element_mode(element_mode)
    view_state = clay_uv.UvPaneState(for_uid=obj.uid)
    tab = SimpleNamespace(doc=doc, uv_view=view_state, saving=saving, uid="bd1")
    seen: dict[str, tuple[bool, str]] = {}
    real = clay_uv.controls.small_button

    def spy(label, *a, enabled=True, reason="", **k):
        seen[label] = (bool(enabled), reason)
        return False

    monkeypatch.setattr(clay_uv.controls, "small_button", spy)
    ui.new_frame()
    ui.begin("##host")
    try:
        clay_uv._toolbar(ctx=None, tab=tab, doc=doc, obj=doc.by_uid(obj.uid), view_state=view_state)
    finally:
        ui.end()
        ui.end_frame()
    monkeypatch.setattr(clay_uv.controls, "small_button", real)
    return seen


def test_the_unwrap_page_buttons_carry_a_reason_in_an_element_mode_and_while_saving(
    ui, monkeypatch
) -> None:
    seen = _toolbar_buttons(ui, monkeypatch, element_mode="face", saving=False)
    for suffix in ("##uvpack", "##uvunwrap-unwrap"):
        enabled, reason = next((v for k, v in seen.items() if k.endswith(suffix)), (None, None))
        assert enabled is False and "Shape" in reason, (suffix, seen)

    seen = _toolbar_buttons(ui, monkeypatch, element_mode="object", saving=True)
    for suffix in ("##uvpack", "##uvunwrap-unwrap"):
        enabled, reason = next((v for k, v in seen.items() if k.endswith(suffix)), (None, None))
        assert enabled is False and reason == "Saving...", (suffix, seen)

    seen = _toolbar_buttons(ui, monkeypatch, element_mode="object", saving=False)
    for suffix in ("##uvpack", "##uvunwrap-unwrap"):
        enabled, reason = next((v for k, v in seen.items() if k.endswith(suffix)), (None, None))
        assert enabled is True, (suffix, seen)


# --- clay-42 ------------------------------------------------------------------


def test_box_unwrap_of_an_already_unwrapped_mesh_returns_the_same_mesh() -> None:
    once = bmuv.box_unwrap(bp.box())
    assert bmuv.box_unwrap(once) is once


def test_pack_islands_of_an_already_packed_mesh_returns_the_same_mesh() -> None:
    packed = uvtools.pack_islands(bmuv.box_unwrap(bp.box()))
    assert uvtools.pack_islands(packed) is packed


class _Ctx:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, text: str, level: str = "info", **_kw: Any) -> None:
        self.toasts.append((text, level))

    def busy(self, kind: str) -> bool:
        return False


def test_box_unwrap_of_an_already_unwrapped_object_pushes_no_undo_step() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(), name="B", mesh=bp.box(), generator="box", params={"size": [1, 1, 1]}
        )
    )
    doc.select([obj.uid])
    ctx = _Ctx()
    unwrap = clay_ops.get("unwrap")
    clay_ops.run(ctx, doc, unwrap)
    before = len(doc.history)
    clay_ops.run(ctx, doc, unwrap)
    assert len(doc.history) == before, "a second unwrap changed nothing and pushed a step"


# --- clay-43 ------------------------------------------------------------------


def test_on_open_does_not_land_a_pull_while_the_clay_tab_is_saving(monkeypatch) -> None:
    from realmspinner.kernels.pixel import Document
    from realmspinner.studio.modes.clay.state import ClayState
    from realmspinner.studio.state import AppState

    ctx = _Ctx()
    ctx.state = AppState()
    ctx.state.clay = ClayState()
    ctx.state.inker = SimpleNamespace(docs=[], activate=lambda uid: None)
    doc = bd.ClayDoc()
    assert doc.add_texture(0, 32)
    tab = ClayTab(doc=doc, title="Crate")
    ctx.state.clay.add(tab)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(inker_open, "open_pixels", lambda *a, **k: calls.append(k))
    tl.edit_in_inker(ctx, tab, 0)

    px = np.zeros((32, 32, 4), np.uint8)
    px[..., 3] = 255
    inker_tab = SimpleNamespace(uid="opened", title="t", doc=Document.from_pixels(px, name="T"))
    ctx.state.inker.docs.append(inker_tab)

    tab.saving = True
    before = len(doc.history)
    calls[0]["on_open"](inker_tab)
    assert len(doc.history) == before, "the PICO-8 snap landed on a document mid-save"
    (link,) = tab.inker_links
    assert link.pulled_rev == tl.UNLANDED, "the link keeps its old revision so the pull still lands"

    tab.saving = False
    tl.pull_all(ctx)
    assert len(doc.history) == before + 1, "and lands on the first frame after the save"


def test_on_open_does_not_land_a_pull_inside_a_live_gesture(monkeypatch) -> None:
    """``on_open`` gated only on ``saving``; ``pull_all`` also skips a live G/R/S
    or UV gesture (the 2026-10-11 audit's clay-15), so the snap landed inside a
    drag a second before the gate that exists for it would have held it."""
    from realmspinner.kernels.pixel import Document
    from realmspinner.studio.modes.clay.state import ClayState
    from realmspinner.studio.state import AppState

    ctx = _Ctx()
    ctx.state = AppState()
    ctx.state.clay = ClayState()
    ctx.state.inker = SimpleNamespace(docs=[], activate=lambda uid: None)
    ctx.clay_view = SimpleNamespace(dragging=False)
    doc = bd.ClayDoc()
    assert doc.add_texture(0, 32)
    tab = ClayTab(doc=doc, title="Crate")
    ctx.state.clay.add(tab)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(inker_open, "open_pixels", lambda *a, **k: calls.append(k))
    tl.edit_in_inker(ctx, tab, 0)

    px = np.zeros((32, 32, 4), np.uint8)
    px[..., 3] = 255
    inker_tab = SimpleNamespace(uid="opened", title="t", doc=Document.from_pixels(px, name="T"))
    ctx.state.inker.docs.append(inker_tab)

    ctx.clay_view.dragging = True
    before = len(doc.history)
    calls[0]["on_open"](inker_tab)
    assert len(doc.history) == before, "the PICO-8 snap landed inside a live drag"
    (link,) = tab.inker_links
    assert link.pulled_rev == tl.UNLANDED

    ctx.clay_view.dragging = False
    tl.pull_all(ctx)
    assert len(doc.history) == before + 1, "and lands on the first frame after the gesture"
