"""Bevel Edges, restored by Landing 12 of the 2026-10 Clay UI rebuild.

112c69e2 cut the op with the Blender-shaped long tail; the kernel (``ops_bevel``) stayed
because the ``.rblk`` migration replays it, and the user asked for the op back. Every claim
here is about the registry row and the doors that are generated from it, and each fails on the
tree before the restore (``clay_ops.get("bevel")`` raises ``KeyError``; the palette has no
``clay-op:bevel``; the shortcut sheet has no Ctrl+B).
"""

from __future__ import annotations

import pygame
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_bevel
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import menutree
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.shortcuts import shortcut_sections

from .test_clay_mode import FakeCtx
from .test_clay_palette import _clay
from .test_clay_palette import _ctx as palette_ctx
from .test_clay_palette import _doc as palette_doc


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


def _edge_doc(mode: str = "edge", *, picked: bool = True) -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    uid = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box())).uid
    doc.select([uid])
    doc.set_element_mode(mode)
    if picked and mode == "edge":
        faces = el.convert(doc.by_uid(uid).mesh, el.ElementSel(faces=[0]), "edge")
        doc.set_element_sel(uid, el.ElementSel(edges=[tuple(faces.edges[0])]))
    return doc, uid


class _Ctx:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def test_bevel_is_an_edge_mode_row_with_a_width_param_and_a_drag() -> None:
    bevel = clay_ops.get("bevel")
    assert bevel.label == "Bevel Edges..." and bevel.modes == ("edge",)
    assert bevel.key == "Ctrl+B"
    assert [p.name for p in bevel.params] == ["width"]
    assert bevel.drag is not None and bevel.drag.param == "width"


def test_bevel_is_in_the_mesh_group_next_to_inset_and_in_the_edge_menu_only() -> None:
    ops = menutree.GROUPS["mesh"].ops
    assert ops.index("bevel") == ops.index("inset") + 1
    flat = {
        m: [op.name for run in menutree.flat(m) for op in run]
        for m in ("vertex", "edge", "face", "object")
    }
    assert "bevel" in flat["edge"], "the right-click menu in edge mode lists it"
    assert all("bevel" not in flat[m] for m in ("vertex", "face", "object"))


def test_bevel_is_enabled_in_edge_mode_with_a_pick_and_says_why_otherwise() -> None:
    bevel = clay_ops.get("bevel")
    doc, _ = _edge_doc()
    assert bevel.enabled(doc) and clay_ops.reason_for(bevel, doc) == ""

    nothing, _ = _edge_doc(picked=False)
    assert not bevel.enabled(nothing)
    assert clay_ops.reason_for(bevel, nothing), "an unpicked edge mode still says what to do"

    for mode in ("object", "face", "vertex"):
        other, _ = _edge_doc(mode)
        assert not bevel.enabled(other), mode
        assert "edge" in clay_ops.reason_for(bevel, other).lower(), mode


def test_bevel_at_005_on_a_box_edge_adds_faces_and_is_one_undo_step() -> None:
    doc, uid = _edge_doc()
    before = bm.face_count(doc.by_uid(uid).mesh)
    steps = len(doc.history)
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("bevel"), width=0.05) is True
    assert ctx.toasts == []
    assert bm.face_count(doc.by_uid(uid).mesh) > before
    assert len(doc.history) == steps + 1
    assert doc.undo()
    assert bm.face_count(doc.by_uid(uid).mesh) == before


def test_bevel_with_a_width_of_zero_is_clamped_by_the_op_and_refused_by_the_kernel() -> None:
    # The 2026-10-09 audit's clay-50: the op's width Param now floors at 0.001, so
    # ``run`` clamps a typed zero into a real bevel; the kernel still refuses a
    # zero width handed to it directly (the 2026-10-03 audit's clay-93).
    doc, uid = _edge_doc()
    before = bm.face_count(doc.by_uid(uid).mesh)
    mesh, sel = doc.by_uid(uid).mesh, doc.element_sel_of(uid)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("bevel"), width=0.0)
    assert bm.face_count(doc.by_uid(uid).mesh) > before
    with pytest.raises(el.OpError, match="width"):
        ops_bevel.bevel_edges(mesh, sel, width=0.0)


def test_ctrl_b_in_edge_mode_reaches_bevel_and_opens_its_dialog_without_a_viewport() -> None:
    ctx = FakeCtx()
    doc, uid = _edge_doc()
    clay_mode.adopt(ctx, doc, title="Scene")
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=pygame.KMOD_CTRL)
    assert clay_mode.handle_key(ctx, event) is True
    state = clay_mode.ensure(ctx)
    assert state.pending_op == "bevel", "no viewport, so the dialog stands in for the drag"


def test_ctrl_b_in_another_mode_does_nothing_to_the_mesh() -> None:
    ctx = FakeCtx()
    doc, uid = _edge_doc("face")
    clay_mode.adopt(ctx, doc, title="Scene")
    before, steps = bm.face_count(doc.by_uid(uid).mesh), len(doc.history)
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=pygame.KMOD_CTRL)
    clay_mode.handle_key(ctx, event)
    assert bm.face_count(doc.by_uid(uid).mesh) == before and len(doc.history) == steps
    assert clay_mode.ensure(ctx).pending_op != "bevel"


def test_ctrl_k_lists_bevel_greyed_with_a_reason_outside_edge_mode_and_live_in_it() -> None:
    ctx = palette_ctx(objects=1)
    doc = palette_doc(ctx)
    row = _clay(ctx)["clay-op:bevel"]
    assert row.label == "Bevel Edges..." and row.hint == "Ctrl+B"
    assert not row.enabled(ctx) and row.why

    doc.set_element_mode("edge")
    uid = doc.objects[0].uid
    faces = el.convert(doc.by_uid(uid).mesh, el.ElementSel(faces=[0]), "edge")
    doc.set_element_sel(uid, el.ElementSel(edges=[tuple(faces.edges[0])]))
    row = _clay(ctx)["clay-op:bevel"]
    assert row.enabled(ctx)
    row.run(ctx)
    assert ctx.state.clay.pending_op == "bevel", "an op with a number asks for its dialog"


def test_the_shortcut_sheet_lists_ctrl_b_under_clay() -> None:
    clay = dict(shortcut_sections())["Clay"]
    rows = {keys: what for keys, what in clay}
    assert "Ctrl+B" in rows and "Bevel" in rows["Ctrl+B"]


def test_the_keys_of_the_registry_stay_unique_per_mode() -> None:
    for mode in ("object", "vertex", "edge", "face"):
        keys = [op.key for op in clay_ops.menu(mode) if op.key]
        assert len(keys) == len(set(keys)), (mode, keys)
