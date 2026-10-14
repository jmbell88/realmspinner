"""Smooth..., Catmull-Clark returning to Clay (Tranche 1 landing 2 of the Clay rig plan).

The kernel (``ops_subdiv.catmull_clark``) never left -- ``characters/`` and the ``.rblk``
migration call it -- so these are about the registry row, the per-press ceiling and the
all-or-nothing write. Each fails on the tree before the restore: ``clay_ops.get("smooth")``
raises ``KeyError``.
"""

from __future__ import annotations

import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import glbimport, ops_subdiv
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import menutree
from realmspinner.studio.modes.clay import ops as clay_ops

from .test_clay_palette import _clay
from .test_clay_palette import _ctx as palette_ctx


class _Ctx:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _doc(n: int = 1) -> tuple[bd.ClayDoc, list[int]]:
    doc = bd.ClayDoc()
    uids = [
        doc.add_object(bd.Obj(uid=bd.new_uid(), name=f"Box {i}", mesh=bp.box())).uid
        for i in range(n)
    ]
    doc.select(uids)
    return doc, uids


def test_smooth_is_an_object_mode_row_with_a_levels_field_from_one_to_four() -> None:
    smooth = clay_ops.get("smooth")
    assert smooth.label == "Smooth..." and smooth.modes == ("object",)
    [levels] = smooth.params
    assert (levels.name, levels.default, levels.low, levels.high, levels.integer) == (
        "levels",
        1.0,
        1.0,
        4.0,
        True,
    )


def test_smooth_is_in_the_object_menu_after_clean_up_and_in_no_element_menu() -> None:
    assert menutree.GROUPS["smooth"].ops == ("smooth",)
    names = [e.name for e in menutree.BARS["object"][1].entries]
    assert names.index("smooth") == names.index("cleanup") + 1
    for mode in ("vertex", "edge", "face"):
        assert "smooth" not in [op.name for run in menutree.flat(mode) for op in run], mode
    assert "smooth" in [op.name for run in menutree.flat("object") for op in run]


def test_smooth_is_greyed_with_a_reason_when_nothing_is_selected() -> None:
    smooth = clay_ops.get("smooth")
    doc, _ = _doc()
    doc.select([])
    assert not smooth.enabled(doc) and clay_ops.reason_for(smooth, doc)


@pytest.mark.parametrize(("levels", "faces"), [(1, 24), (2, 96), (3, 384)])
def test_smoothing_a_box_n_levels_builds_four_times_the_faces_per_level(
    levels: int, faces: int
) -> None:
    doc, [uid] = _doc()
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("smooth"), levels=levels) is True
    assert ctx.toasts == []
    assert bm.face_count(doc.by_uid(uid).mesh) == faces


def test_smoothing_rounds_the_shape_rather_than_just_splitting_it() -> None:
    doc, [uid] = _doc()
    before = bm.bounds(doc.by_uid(uid).mesh)
    clay_ops.run(_Ctx(), doc, clay_ops.get("smooth"), levels=2)
    lo, hi = bm.bounds(doc.by_uid(uid).mesh)
    assert (hi - lo).max() < (before[1] - before[0]).max(), "the corners pull in"


def test_smoothing_several_objects_is_one_undo_step_and_freezes_their_generators() -> None:
    doc, uids = _doc(3)
    steps = len(doc.history)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("smooth"), levels=1) is True
    assert len(doc.history) == steps + 1
    assert all(bm.face_count(doc.by_uid(u).mesh) == 24 for u in uids)
    assert all(doc.by_uid(u).generator is None for u in uids)
    assert doc.undo()
    assert all(bm.face_count(doc.by_uid(u).mesh) == 6 for u in uids)


def test_the_faces_a_press_builds_are_summed_over_the_selection_not_judged_per_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One box builds 24 faces and each is inside the ceiling; three together (72) are not.
    monkeypatch.setattr(ops_subdiv, "MAX_SUBDIVIDED_FACES", 40)
    one, [only] = _doc(1)
    assert clay_ops.run(_Ctx(), one, clay_ops.get("smooth"), levels=1) is True
    assert bm.face_count(one.by_uid(only).mesh) == 24

    three, uids = _doc(3)
    steps, ctx = len(three.history), _Ctx()
    assert clay_ops.run(ctx, three, clay_ops.get("smooth"), levels=1) is False
    assert ctx.toasts and "72" in ctx.toasts[0][0] and "faces" in ctx.toasts[0][0]
    assert len(three.history) == steps, "a refused press pushes no step"
    assert all(bm.face_count(three.by_uid(u).mesh) == 6 for u in uids)


def test_the_levels_multiply_into_the_per_press_sum() -> None:
    # Level four on one box is 24 * 4**3 = 1,536 faces; a ceiling between that and the
    # level-three figure (384) must refuse four levels and allow three.
    import realmspinner.kernels.mesh.ops_subdiv as sub

    saved = sub.MAX_SUBDIVIDED_FACES
    sub.MAX_SUBDIVIDED_FACES = 1000
    try:
        doc, [uid] = _doc()
        assert clay_ops.run(_Ctx(), doc, clay_ops.get("smooth"), levels=3) is True
        doc, [uid] = _doc()
        ctx = _Ctx()
        assert clay_ops.run(ctx, doc, clay_ops.get("smooth"), levels=4) is False
        assert "1,536" in ctx.toasts[0][0]
        assert bm.face_count(doc.by_uid(uid).mesh) == 6
    finally:
        sub.MAX_SUBDIVIDED_FACES = saved


def test_a_run_that_crosses_the_triangle_ceiling_writes_no_object_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two boxes are 24 triangles; smoothed once they are 96. A ceiling of 60 is crossed
    # only by the pair together, and nothing may be written before the check.
    monkeypatch.setattr(glbimport, "MAX_TRIANGLES", 60)
    doc, uids = _doc(2)
    steps, ctx = len(doc.history), _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("smooth"), levels=1) is False
    assert ctx.toasts
    assert len(doc.history) == steps
    assert all(bm.face_count(doc.by_uid(u).mesh) == 6 for u in uids)


def test_a_group_on_its_own_is_refused_with_the_no_geometry_sentence() -> None:
    doc, uids = _doc(2)
    group = doc.group(uids)
    doc.select([group.uid])
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("smooth")) is False
    assert ctx.toasts and ctx.toasts[0][0] == clay_ops.GROUP_HAS_NO_GEOMETRY


def test_ctrl_k_lists_smooth_and_it_opens_its_dialog() -> None:
    ctx = palette_ctx(objects=1)
    row = _clay(ctx)["clay-op:smooth"]
    assert row.label == "Smooth..."
    assert row.enabled(ctx)
    row.run(ctx)
    assert ctx.state.clay.pending_op == "smooth", "an op with a number asks for its dialog"
