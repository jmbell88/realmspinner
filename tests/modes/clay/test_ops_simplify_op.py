"""``simplify``: the Clean Up op that merges flat faces and then collapses edges.

The geometry is ``kernels/mesh/ops_simplify``'s own tests; what is held here is
the op around it -- the gate, the refusals, the one undo step, the generator
freeze, the hidden-object rule and the sentence a press ends with -- and the
kernel is reached only through the op.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_subdiv, topo
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import menutree
from realmspinner.studio.modes.clay import ops as clay_ops


class _Ctx:
    """Records every toast with the level it was shown at."""

    def __init__(self) -> None:
        self.toasted: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasted.append((message, level))

    @property
    def messages(self) -> list[str]:
        return [message for message, _ in self.toasted]

    @property
    def errors(self) -> list[str]:
        return [message for message, level in self.toasted if level == "error"]


def _subdivided_box(levels: int = 4, *, material_per_face: bool = False) -> bm.Mesh:
    """A box split ``levels`` times into flat quads: ``6 * 4**levels`` faces that
    are still, to the eye, six."""
    mesh = bp.box()
    if material_per_face:
        mesh = topo.rebuild(
            mesh.positions,
            mesh.loops,
            mesh.starts,
            np.arange(bm.face_count(mesh)),
            mesh.smooth,
            uv=None,
        )
    for _ in range(levels):
        mesh, _sel = ops_subdiv.subdivide(mesh, el.ElementSel())
    return mesh


def _doc_with(mesh: bm.Mesh, *, generator: str | None = None) -> tuple[Any, Any]:
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Box", mesh=mesh, generator=generator)
    )
    doc.select([obj.uid])
    return doc, obj


# --- registration -----------------------------------------------------------


def test_simplify_is_an_object_mode_op_with_one_target_parameter() -> None:
    op = clay_ops.get("simplify")
    assert op.modes == ("object",)
    assert op.label == "Simplify..."
    (param,) = op.params
    assert (param.name, param.default, param.low) == ("target", 100.0, 4.0)
    assert param.integer and param.refuse_below


def test_simplify_sits_in_the_clean_up_group_beside_recalculate_normals() -> None:
    assert menutree.GROUPS["cleanup"].ops == ("recalc-normals", "simplify")
    assert "simplify" in [op.name for op in clay_ops.menu("object")]


def test_simplify_is_greyed_with_a_reason_on_an_empty_document() -> None:
    doc = bd.ClayDoc()
    op = clay_ops.get("simplify")
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc)


def test_simplify_is_refused_outside_object_mode() -> None:
    doc, obj = _doc_with(_subdivided_box(1))
    doc.set_element_mode("face")
    ctx = _Ctx()
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("simplify")) is False

    assert ctx.errors, "a wrong mode is a refusal the user can read"
    assert len(doc.history) == depth
    assert bm.face_count(doc.by_uid(obj.uid).mesh) == 24


def test_a_target_under_four_is_refused_not_raised_to_four() -> None:
    doc, obj = _doc_with(_subdivided_box(2))
    ctx = _Ctx()
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("simplify"), target=3) is False

    assert any("at least 4" in message for message in ctx.errors)
    assert len(doc.history) == depth
    assert bm.face_count(doc.by_uid(obj.uid).mesh) == 96


# --- running ----------------------------------------------------------------


def test_a_subdivided_box_comes_back_as_six_faces_in_one_undo_step() -> None:
    mesh = _subdivided_box(4)
    assert bm.face_count(mesh) == 1536
    doc, obj = _doc_with(mesh)
    depth = len(doc.history)
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("simplify"), target=100) is True

    assert bm.face_count(doc.by_uid(obj.uid).mesh) == 6
    assert len(doc.history) == depth + 1, "one press, one Ctrl+Z"
    assert doc.undo() is True
    assert doc.by_uid(obj.uid).mesh is mesh
    assert len(doc.history) == depth


def test_the_toast_carries_the_real_before_and_after_counts() -> None:
    doc, _ = _doc_with(_subdivided_box(4))
    ctx = _Ctx()

    clay_ops.run(ctx, doc, clay_ops.get("simplify"), target=100)

    assert ctx.toasted == [("Simplified 1,536 faces to 6.", "info")]


def test_a_target_the_seams_forbid_is_named_in_the_sentence() -> None:
    """Six materials, one per side of the box: a merge never crosses one and the
    collapse stage locks the border, so nothing can go below six and the toast
    says why the four asked for was not reached."""
    doc, obj = _doc_with(_subdivided_box(3, material_per_face=True))
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("simplify"), target=4) is True

    assert bm.face_count(doc.by_uid(obj.uid).mesh) > 4
    (message, level), = ctx.toasted
    assert level == "info"
    assert message.startswith("Simplified 384 faces to ")
    assert "4 was not reachable without breaking seams or materials" in message


def test_simplify_freezes_the_generator() -> None:
    doc, obj = _doc_with(_subdivided_box(2), generator="box")
    assert doc.by_uid(obj.uid).generator == "box"

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("simplify"), target=10) is True

    assert doc.by_uid(obj.uid).generator is None
    assert doc.undo() is True
    assert doc.by_uid(obj.uid).generator == "box", "the freeze is part of the same step"


def test_a_target_above_the_face_count_pushes_nothing_and_says_so() -> None:
    mesh = _subdivided_box(1)
    doc, obj = _doc_with(mesh, generator="box")
    depth = len(doc.history)
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("simplify"), target=500) is False

    assert len(doc.history) == depth
    assert doc.by_uid(obj.uid).mesh is mesh
    assert doc.by_uid(obj.uid).generator == "box"
    assert ctx.toasted == [("Already at or under 500 faces; nothing to simplify.", "info")]


def test_a_hidden_object_is_left_alone() -> None:
    doc = bd.ClayDoc()
    shown = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Shown", mesh=_subdivided_box(2)))
    hidden_mesh = _subdivided_box(2)
    hidden = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hidden", mesh=hidden_mesh))
    doc.select([shown.uid, hidden.uid])
    doc.set_visibility({hidden.uid: False})
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("simplify"), target=10) is True

    assert bm.face_count(doc.by_uid(shown.uid).mesh) == 6
    assert doc.by_uid(hidden.uid).mesh is hidden_mesh
    assert ctx.messages == ["Simplified 96 faces to 6."]


def test_hiding_the_only_selected_object_greys_simplify_rather_than_running() -> None:
    """Hiding deselects, so there is nothing left for the op to rewrite."""
    doc, obj = _doc_with(_subdivided_box(2))
    doc.set_visibility({obj.uid: False})
    op = clay_ops.get("simplify")
    depth = len(doc.history)
    ctx = _Ctx()

    assert not op.enabled(doc)
    assert clay_ops.run(ctx, doc, op, target=10) is False
    assert len(doc.history) == depth
    assert bm.face_count(doc.by_uid(obj.uid).mesh) == 96


def test_several_objects_fold_into_one_step_and_one_sentence() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=_subdivided_box(2)))
    b = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=_subdivided_box(3)))
    doc.select([a.uid, b.uid])
    depth = len(doc.history)
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("simplify"), target=10) is True

    assert len(doc.history) == depth + 1
    # The target is per object, so the sentence says so (the 2026-10-11 audit's
    # clay-25): a bare "to 12" beside a typed 10 read as the target being missed.
    assert ctx.messages == ["Simplified 480 faces to 12 across 2 objects, each toward 10."]
