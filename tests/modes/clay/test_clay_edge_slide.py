"""Edge Slide, restored 2026-10-13 (Tranche 1 landing 4 of the Clay rig plan).

The kernel (``ops_slide.edge_slide``) has its own claims in ``test_ops_slide.py``; every claim
here is about the registry row and the doors generated from it, and each fails on the tree
before the restore: ``clay_ops.get("edge-slide")`` raises ``KeyError``, the palette has no
``clay-op:edge-slide`` and the shortcut sheet has no Ctrl+Shift+R.
"""

from __future__ import annotations

import numpy as np
import pygame
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.mesh import from_faces
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


def _strip() -> bm.Mesh:
    """Three quads side by side, vertices at x = 0..3 on two rows; the rung at x = 1 is (2, 3)."""
    positions = [[x, 0, z] for x in range(4) for z in (0, 1)]
    return from_faces(positions, [[0, 2, 3, 1], [2, 4, 5, 3], [4, 6, 7, 5]])


def _doc(mode: str = "edge", *, picked: bool = True) -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    uid = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Strip", mesh=_strip())).uid
    doc.select([uid])
    doc.set_element_mode(mode)
    if picked and mode == "edge":
        doc.set_element_sel(uid, el.ElementSel(edges=np.array([[2, 3]], dtype="i4")))
    return doc, uid


def test_edge_slide_is_an_edge_mode_row_with_a_signed_slide_param_and_a_fraction_drag() -> None:
    slide = clay_ops.get("edge-slide")
    assert slide.label == "Edge Slide..." and slide.modes == ("edge",)
    assert slide.key == "Ctrl+Shift+R"
    [t] = slide.params
    assert (t.name, t.default, t.low, t.high) == ("t", 0.0, -0.99, 0.99)
    assert slide.drag is not None and (slide.drag.param, slide.drag.kind) == ("t", "fraction")


def test_ctrl_shift_e_the_app_wide_export_chord_is_not_taken() -> None:
    assert clay_ops.by_key("edge", "Ctrl+Shift+E") is None


def test_edge_slide_is_in_the_mesh_group_after_loop_cut_and_in_the_edge_menu_only() -> None:
    ops = menutree.GROUPS["mesh"].ops
    assert ops.index("edge-slide") == ops.index("loop-cut") + 1
    flat = {
        m: [op.name for run in menutree.flat(m) for op in run]
        for m in ("vertex", "edge", "face", "object")
    }
    assert "edge-slide" in flat["edge"]
    assert all("edge-slide" not in flat[m] for m in ("vertex", "face", "object"))


def test_edge_slide_is_greyed_with_a_reason_outside_edge_mode_and_with_nothing_picked() -> None:
    slide = clay_ops.get("edge-slide")
    doc, _ = _doc()
    assert slide.enabled(doc) and clay_ops.reason_for(slide, doc) == ""
    nothing, _ = _doc(picked=False)
    assert not slide.enabled(nothing) and clay_ops.reason_for(slide, nothing)
    for mode in ("object", "face", "vertex"):
        other, _ = _doc(mode)
        assert not slide.enabled(other), mode
        assert "edge" in clay_ops.reason_for(slide, other).lower(), mode


def test_sliding_a_rung_moves_it_along_the_strip_as_one_undo_step() -> None:
    doc, uid = _doc()
    before = np.array(doc.by_uid(uid).mesh.positions)
    steps, ctx = len(doc.history), _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("edge-slide"), t=0.5) is True
    assert ctx.toasts == []
    after = np.array(doc.by_uid(uid).mesh.positions)
    assert after[2, 0] == pytest.approx(after[3, 0]) and after[2, 0] != pytest.approx(1.0)
    assert len(doc.history) == steps + 1
    assert doc.undo()
    assert np.allclose(doc.by_uid(uid).mesh.positions, before)


def test_the_dialog_clamps_a_slide_of_one_short_of_the_neighbour() -> None:
    # t = 1 would put the rung on its neighbour and duplicate two vertices; the Param caps
    # it at 0.99 so the dialog, the drag, the palette and the agent all stay clear of it.
    doc, uid = _doc()
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("edge-slide"), t=1.0) is True
    x = float(doc.by_uid(uid).mesh.positions[2, 0])
    assert x in (pytest.approx(0.01, abs=1e-4), pytest.approx(1.99, abs=1e-4))


def test_a_loop_with_no_sideways_edge_is_a_refusal_toast_that_changes_nothing() -> None:
    doc = bd.ClayDoc()
    quad = from_faces([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], [[0, 1, 2, 3]])
    uid = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Quad", mesh=quad)).uid
    doc.select([uid])
    doc.set_element_mode("edge")
    doc.set_element_sel(uid, el.ElementSel(edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0]])))
    steps, ctx = len(doc.history), _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("edge-slide"), t=0.5) is False
    assert any("no edge to slide along" in message for message, _ in ctx.toasts)
    assert len(doc.history) == steps


def test_ctrl_shift_r_in_edge_mode_reaches_edge_slide_without_a_viewport() -> None:
    ctx = FakeCtx()
    doc, uid = _doc()
    clay_mode.adopt(ctx, doc, title="Scene")
    event = pygame.event.Event(
        pygame.KEYDOWN, key=pygame.K_r, mod=pygame.KMOD_CTRL | pygame.KMOD_SHIFT
    )
    assert clay_mode.handle_key(ctx, event) is True
    assert clay_mode.ensure(ctx).pending_op == "edge-slide"


def test_ctrl_r_still_reaches_loop_cut_not_edge_slide() -> None:
    ctx = FakeCtx()
    doc, uid = _doc()
    clay_mode.adopt(ctx, doc, title="Scene")
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, mod=pygame.KMOD_CTRL)
    clay_mode.handle_key(ctx, event)
    assert clay_mode.ensure(ctx).pending_op != "edge-slide"


def test_ctrl_k_lists_edge_slide_greyed_outside_edge_mode_and_live_in_it() -> None:
    ctx = palette_ctx(objects=1)
    doc = palette_doc(ctx)
    row = _clay(ctx)["clay-op:edge-slide"]
    assert row.label == "Edge Slide..." and row.hint == "Ctrl+Shift+R"
    assert not row.enabled(ctx) and row.why

    doc.set_element_mode("edge")
    uid = doc.objects[0].uid
    found = el.convert(doc.by_uid(uid).mesh, el.ElementSel(faces=[0]), "edge")
    doc.set_element_sel(uid, el.ElementSel(edges=[tuple(found.edges[0])]))
    row = _clay(ctx)["clay-op:edge-slide"]
    assert row.enabled(ctx)
    row.run(ctx)
    assert ctx.state.clay.pending_op == "edge-slide"


def test_the_shortcut_sheet_lists_ctrl_shift_r_under_clay() -> None:
    rows = dict(dict(shortcut_sections())["Clay"])
    assert "Ctrl+Shift+R" in rows and "Edge slide" in rows["Ctrl+Shift+R"]


def test_a_box_is_a_valid_target_too() -> None:
    doc = bd.ClayDoc()
    uid = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box())).uid
    doc.select([uid])
    doc.set_element_mode("edge")
    found = el.convert(doc.by_uid(uid).mesh, el.ElementSel(faces=[0]), "edge")
    doc.set_element_sel(uid, el.ElementSel(edges=[tuple(found.edges[0])]))
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("edge-slide"), t=0.3) is True
    bm.validate(doc.by_uid(uid).mesh)
