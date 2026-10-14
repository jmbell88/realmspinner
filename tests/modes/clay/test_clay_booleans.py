"""Union, Difference and Intersect Objects, restored 2026-10-13 (Tranche 1 landing 5).

``ops_boolean`` (on ``manifold3d``) never left -- the character generators call it -- so these
claims are about the three registry rows, the chord, and the honesty the plan asked for: the
survivor and the minuend are the topmost selected object in the outliner, the result is
triangles without UVs, an open input is refused with the kernel's remedy, and an empty answer
is a refusal. Each fails on the tree before the restore: ``clay_ops.get("union")`` raises
``KeyError`` and ``Ctrl+Shift+M`` does what ``Ctrl+M`` does.
"""

from __future__ import annotations

import numpy as np
import pygame
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_boolean, topo
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import menutree
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.shortcuts import shortcut_sections

from .test_clay_mode import FakeCtx
from .test_clay_palette import _clay
from .test_clay_palette import _ctx as palette_ctx

KINDS = ("union", "difference", "intersection")


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


class _Ctx:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _two_boxes(
    offset: tuple[float, float, float] = (0.5, 0.0, 0.0),
    second: bm.Mesh | None = None,
) -> tuple[bd.ClayDoc, int, int]:
    """Box ``A`` at the origin, then ``B`` (a box by default) *offset* from it; both selected."""
    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box())).uid
    b = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="B",
            mesh=second if second is not None else bp.box(),
            translation=offset,
        )
    ).uid
    doc.select([a, b])
    return doc, a, b


def test_the_three_rows_are_object_mode_with_union_alone_on_a_key() -> None:
    assert [clay_ops.get(k).label for k in KINDS] == [
        "Union Objects",
        "Difference Objects",
        "Intersect Objects",
    ]
    assert all(clay_ops.get(k).modes == ("object",) for k in KINDS)
    assert clay_ops.get("union").key == "Ctrl+Shift+M"
    assert clay_ops.get("difference").key == "" and clay_ops.get("intersection").key == ""
    assert all(not clay_ops.get(k).params for k in KINDS), "no dialog, so no ellipsis"


def test_the_join_group_lists_merge_then_the_three_booleans_in_the_object_menu_only() -> None:
    assert menutree.GROUPS["join"].ops == ("join", *KINDS)
    for mode in ("vertex", "edge", "face"):
        names = [op.name for run in menutree.flat(mode) for op in run]
        assert not set(KINDS) & set(names), mode
    names = [op.name for run in menutree.flat("object") for op in run]
    assert set(KINDS) <= set(names)


def test_merges_hint_names_union_and_its_key_and_each_hint_states_the_costs_in_order() -> None:
    assert "Union Objects (Ctrl+Shift+M)" in clay_ops.get("join").hint
    for kind in KINDS:
        hint = clay_ops.get(kind).hint
        order = [hint.index(word) for word in ("triangles", "texture coordinates", "closed solid")]
        assert order == sorted(order), kind


@pytest.mark.parametrize("kind", KINDS)
def test_each_row_needs_two_visible_selected_objects_and_says_so(kind: str) -> None:
    op = clay_ops.get(kind)
    doc, a, b = _two_boxes()
    assert op.enabled(doc) and clay_ops.reason_for(op, doc) == ""
    doc.select([a])
    assert not op.enabled(doc)
    assert "two visible objects" in clay_ops.reason_for(op, doc)


def test_union_of_two_overlapping_boxes_is_one_object_of_triangles_in_one_undo_step() -> None:
    doc, a, b = _two_boxes()
    steps, ctx = len(doc.history), _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("union")) is True
    assert ctx.toasts == []
    assert [o.uid for o in doc.objects] == [a], "the topmost selected object survives"
    mesh = doc.by_uid(a).mesh
    assert set(np.diff(mesh.starts).tolist()) == {3}, "the kernel works in triangles"
    assert mesh.uv is None, "a boolean recuts every face, so UVs are dropped"
    assert doc.selection == {a}
    assert len(doc.history) == steps + 1
    assert doc.undo()
    assert {o.uid for o in doc.objects} == {a, b}
    assert bm.face_count(doc.by_uid(a).mesh) == 6


def test_difference_cuts_the_others_out_of_the_topmost_object_and_the_order_decides_which() -> None:
    doc, a, b = _two_boxes()
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("difference")) is True
    lo, hi = bm.bounds(doc.by_uid(a).mesh)
    # A (x -0.5..0.5) minus B (x 0..1 in A's frame): what is left is the left half of A.
    assert lo[0] == pytest.approx(-0.5, abs=1e-4) and hi[0] == pytest.approx(0.0, abs=1e-4)

    # Swap the outliner order and the *other* box is the minuend: B minus A is B's right half,
    # reported in B's own frame.
    doc, a, b = _two_boxes()
    doc.objects[0], doc.objects[1] = doc.objects[1], doc.objects[0]
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("difference")) is True
    assert [o.uid for o in doc.objects] == [b]
    lo, hi = bm.bounds(doc.by_uid(b).mesh)
    assert lo[0] == pytest.approx(0.0, abs=1e-4) and hi[0] == pytest.approx(0.5, abs=1e-4)


def test_intersection_keeps_only_the_shared_volume() -> None:
    doc, a, b = _two_boxes()
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("intersection")) is True
    lo, hi = bm.bounds(doc.by_uid(a).mesh)
    assert lo[0] == pytest.approx(0.0, abs=1e-4) and hi[0] == pytest.approx(0.5, abs=1e-4)


def test_an_intersection_with_nothing_in_common_is_refused_and_changes_nothing() -> None:
    doc, a, b = _two_boxes(offset=(5.0, 0.0, 0.0))
    steps, ctx = len(doc.history), _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("intersection")) is False
    assert any("came out empty" in message for message, _ in ctx.toasts)
    assert len(doc.history) == steps and {o.uid for o in doc.objects} == {a, b}


def test_an_open_input_is_refused_with_the_kernels_remedy_and_changes_nothing() -> None:
    box = bp.box()
    open_box = topo.compact_vertices(topo.take_faces(box, np.arange(1, bm.face_count(box))))[0]
    doc, a, b = _two_boxes(second=open_box)
    steps, ctx = len(doc.history), _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("union")) is False
    assert any("closed solid" in message for message, _ in ctx.toasts), ctx.toasts
    assert len(doc.history) == steps and {o.uid for o in doc.objects} == {a, b}


def test_the_triangle_budget_is_summed_over_every_input_before_the_kernel_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two boxes are 12 triangles each; a budget of 20 is crossed only by the pair together.
    monkeypatch.setattr(ops_boolean, "MAX_BOOLEAN_TRIANGLES", 20)
    doc, a, b = _two_boxes()
    steps, ctx = len(doc.history), _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("union")) is False
    assert any("24 triangles" in message for message, _ in ctx.toasts), ctx.toasts
    assert len(doc.history) == steps and {o.uid for o in doc.objects} == {a, b}


def test_a_hidden_selected_object_is_not_part_of_the_boolean() -> None:
    doc, a, b = _two_boxes()
    doc.by_uid(b).visible = False
    assert not clay_ops.get("union").enabled(doc), "only one visible object is selected"


def _ctrl(ctx, doc, *, shift: bool) -> None:
    clay_mode.adopt(ctx, doc, title="Scene")
    mod = pygame.KMOD_CTRL | (pygame.KMOD_SHIFT if shift else 0)
    clay_mode.handle_key(ctx, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m, mod=mod))


def test_ctrl_shift_m_reaches_union_and_ctrl_m_still_reaches_merge() -> None:
    ctx = FakeCtx()
    doc, a, b = _two_boxes()
    _ctrl(ctx, doc, shift=True)
    mesh = doc.by_uid(a).mesh
    assert set(np.diff(mesh.starts).tolist()) == {3} and len(doc.objects) == 1, "a union"

    ctx = FakeCtx()
    doc, a, b = _two_boxes()
    _ctrl(ctx, doc, shift=False)
    # Merge asks for its weld distance first, so it is a dialog and not a union.
    assert clay_mode.ensure(ctx).pending_op == "join"
    assert len(doc.objects) == 2, "nothing ran yet"


def test_ctrl_k_lists_the_three_booleans_and_union_runs() -> None:
    ctx = palette_ctx(objects=2)
    commands = _clay(ctx)
    for kind in KINDS:
        assert commands[f"clay-op:{kind}"].label == clay_ops.get(kind).label
    assert commands["clay-op:union"].hint == "Ctrl+Shift+M"


def test_the_shortcut_sheet_lists_ctrl_shift_m_under_clay() -> None:
    rows = dict(dict(shortcut_sections())["Clay"])
    assert "Ctrl+Shift+M" in rows and "Union" in rows["Ctrl+Shift+M"]
