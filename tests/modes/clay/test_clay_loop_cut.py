"""Loop Cut, restored 2026-10-13 (Tranche 1 landing 3 of the Clay rig plan).

The kernel (``ops_bevel.loop_cut``) came back from 112c69e2^ with its own claims in
``test_ops_bevel.py``; every claim here is about the registry row and the doors generated
from it, and each fails on the tree before the restore: ``clay_ops.get("loop-cut")`` raises
``KeyError``, the palette has no ``clay-op:loop-cut`` and the shortcut sheet has no Ctrl+R.
"""

from __future__ import annotations

import pygame
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
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


class _Ctx:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _edge_doc(
    mode: str = "edge", *, picked: bool = True, edges: int = 1
) -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    uid = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box())).uid
    doc.select([uid])
    doc.set_element_mode(mode)
    if picked and mode == "edge":
        found = el.convert(doc.by_uid(uid).mesh, el.ElementSel(faces=[0]), "edge")
        doc.set_element_sel(uid, el.ElementSel(edges=[tuple(e) for e in found.edges[:edges]]))
    return doc, uid


def test_loop_cut_is_an_edge_mode_row_with_a_position_param_and_a_fraction_drag() -> None:
    cut = clay_ops.get("loop-cut")
    assert cut.label == "Loop Cut..." and cut.modes == ("edge",) and cut.key == "Ctrl+R"
    [t] = cut.params
    assert (t.name, t.default, t.low, t.high) == ("t", 0.5, 0.01, 0.99)
    assert cut.drag is not None and (cut.drag.param, cut.drag.kind) == ("t", "fraction")


def test_loop_cut_is_in_the_mesh_group_after_bevel_and_in_the_edge_menu_only() -> None:
    ops = menutree.GROUPS["mesh"].ops
    assert ops.index("loop-cut") == ops.index("bevel") + 1
    flat = {
        m: [op.name for run in menutree.flat(m) for op in run]
        for m in ("vertex", "edge", "face", "object")
    }
    assert "loop-cut" in flat["edge"]
    assert all("loop-cut" not in flat[m] for m in ("vertex", "face", "object"))


def test_loop_cut_is_greyed_with_a_reason_outside_edge_mode_and_with_nothing_picked() -> None:
    cut = clay_ops.get("loop-cut")
    doc, _ = _edge_doc()
    assert cut.enabled(doc) and clay_ops.reason_for(cut, doc) == ""
    nothing, _ = _edge_doc(picked=False)
    assert not cut.enabled(nothing) and clay_ops.reason_for(cut, nothing)
    for mode in ("object", "face", "vertex"):
        other, _ = _edge_doc(mode)
        assert not cut.enabled(other), mode
        assert "edge" in clay_ops.reason_for(cut, other).lower(), mode


def test_a_loop_cut_through_a_box_edge_splits_a_strip_in_one_undo_step() -> None:
    doc, uid = _edge_doc()
    before = bm.face_count(doc.by_uid(uid).mesh)
    steps, ctx = len(doc.history), _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("loop-cut"), t=0.5) is True
    assert ctx.toasts == []
    assert bm.face_count(doc.by_uid(uid).mesh) == before + 4, "the four faces of the ring split"
    assert len(doc.history) == steps + 1
    assert doc.undo()
    assert bm.face_count(doc.by_uid(uid).mesh) == before


def test_two_picked_edges_are_a_refusal_toast_that_changes_nothing() -> None:
    doc, uid = _edge_doc(edges=2)
    steps, ctx = len(doc.history), _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("loop-cut"), t=0.5) is False
    assert any("exactly one edge" in message for message, _ in ctx.toasts)
    assert len(doc.history) == steps


def test_the_dialog_clamps_a_position_of_zero_into_the_param_range() -> None:
    # t = 0 would only duplicate a point (the 2026-10-03 audit's clay-93): the Param floors it.
    doc, uid = _edge_doc()
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("loop-cut"), t=0.0) is True
    assert len(doc.by_uid(uid).mesh.positions) > 8


def test_ctrl_r_in_edge_mode_reaches_loop_cut_and_opens_its_dialog_without_a_viewport() -> None:
    ctx = FakeCtx()
    doc, uid = _edge_doc()
    clay_mode.adopt(ctx, doc, title="Scene")
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, mod=pygame.KMOD_CTRL)
    assert clay_mode.handle_key(ctx, event) is True
    assert clay_mode.ensure(ctx).pending_op == "loop-cut"


def test_ctrl_r_in_another_mode_does_nothing_to_the_mesh() -> None:
    ctx = FakeCtx()
    doc, uid = _edge_doc("face")
    clay_mode.adopt(ctx, doc, title="Scene")
    before, steps = bm.face_count(doc.by_uid(uid).mesh), len(doc.history)
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, mod=pygame.KMOD_CTRL)
    clay_mode.handle_key(ctx, event)
    assert bm.face_count(doc.by_uid(uid).mesh) == before and len(doc.history) == steps
    assert clay_mode.ensure(ctx).pending_op != "loop-cut"


def test_ctrl_k_lists_loop_cut_greyed_outside_edge_mode_and_live_in_it() -> None:
    ctx = palette_ctx(objects=1)
    doc = palette_doc(ctx)
    row = _clay(ctx)["clay-op:loop-cut"]
    assert row.label == "Loop Cut..." and row.hint == "Ctrl+R"
    assert not row.enabled(ctx) and row.why

    doc.set_element_mode("edge")
    uid = doc.objects[0].uid
    found = el.convert(doc.by_uid(uid).mesh, el.ElementSel(faces=[0]), "edge")
    doc.set_element_sel(uid, el.ElementSel(edges=[tuple(found.edges[0])]))
    row = _clay(ctx)["clay-op:loop-cut"]
    assert row.enabled(ctx)
    row.run(ctx)
    assert ctx.state.clay.pending_op == "loop-cut"


def test_the_shortcut_sheet_lists_ctrl_r_under_clay() -> None:
    rows = dict(dict(shortcut_sections())["Clay"])
    assert "Ctrl+R" in rows and "Loop cut" in rows["Ctrl+R"]
