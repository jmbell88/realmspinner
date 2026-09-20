"""Clay tranche 2's two ``ops.py`` changes: Apply Modifiers, and "merging
applies every stack".

Apply Modifiers is a plain object-level op, tested the same shape every
sibling in ``test_clay_ops.py`` already is: registered once, greyed with a
stated reason when its gate fails, one undo step per press.

The merge ops (Join/Union/Difference/Intersection) changed what mesh they
hand the geometry kernel -- the object's *evaluated* mesh rather than its
base -- so a target carrying a modifier merges what was on screen, not the
raw generator output underneath it. ``test_every_merge_op_consumes_the_
evaluated_mesh_not_the_base`` is the bidirectional-style gate across all
four (a source check, the same shape ``test_clay_props_regen.py``'s
``test_both_generator_rebuild_doors_go_through_clay_regen`` already uses for
its own two doors); ``test_join_of_an_object_with_a_mirror_modifier_...`` and
its Union sibling are the behavioural pins for two of the four -- Difference
and Intersection share the exact one-line change and the source gate already
covers them, so a full CSG fixture for each would be pinning the same line
twice more.
"""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import modifiers as mods
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops


class _Toasts:
    def __init__(self) -> None:
        self.errors: list[str] = []


class _Ctx:
    def __init__(self) -> None:
        self.toasts = _Toasts()

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.toasts.errors.append(message)


def _offset_box() -> bm.Mesh:
    """A unit box moved off its own local origin -- so mirroring it about the
    local X plane produces a *second*, disjoint box rather than an exact
    self-overlap. ``bp.box()`` is centred on the origin (the module docstring:
    "every generator is centred on its own origin"), and mirroring a shape
    that already straddles the mirror plane reflects it onto itself.
    """
    base = bp.box()
    return replace(base, positions=base.positions + np.array([2.0, 0.0, 0.0], dtype="f8"))


def _doc_with_two_boxes(*, mirror: bool) -> tuple[bd.ClayDoc, int, int]:
    """Object A (optionally carrying a mirror modifier) and a second, distant
    box B -- both selected and visible, the shape every merge op reads its
    input from."""
    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=_offset_box()))
    if mirror:
        doc.set_modifiers(a.uid, (mods.make("mirror", {"axis": "X", "weld": 0.0}, id=1),))
    b = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    doc.set_transform(b.uid, translation=[20.0, 0.0, 0.0])
    doc.select([a.uid, b.uid])
    return doc, a.uid, b.uid


# --- Apply Modifiers ----------------------------------------------------------


def test_apply_modifiers_appears_in_menu_object() -> None:
    assert "apply-modifiers" in [op.name for op in clay_ops.menu("object")]


def test_apply_modifiers_is_greyed_with_a_reason_when_no_stack() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    op = clay_ops.get("apply-modifiers")

    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc) == "Select an object first."

    doc.select([obj.uid])
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc) != ""

    doc.set_modifiers(obj.uid, (mods.make("mirror", id=1),))
    assert op.enabled(doc)
    assert clay_ops.reason_for(op, doc) == ""


def test_apply_modifiers_is_one_undo_step_and_undoes_the_whole_bake() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="A",
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
        )
    )
    doc.set_modifiers(obj.uid, (mods.make("mirror", id=1),))
    doc.select([obj.uid])
    base_faces = bm.face_count(obj.mesh)
    depth = len(doc.history)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("apply-modifiers")) is True
    assert len(doc.history) == depth + 1, "one press, one Ctrl+Z"

    baked = doc.by_uid(obj.uid)
    assert baked.modifiers == ()
    assert baked.generator is None, "baked geometry is no longer what the generator would build"
    assert bm.face_count(baked.mesh) == 2 * base_faces

    assert doc.undo() is True
    restored = doc.by_uid(obj.uid)
    assert bm.face_count(restored.mesh) == base_faces
    assert restored.modifiers and restored.modifiers[0].kind == "mirror"
    assert restored.generator == "box"


def test_apply_modifiers_toasts_and_leaves_the_rest_of_the_selection_alone() -> None:
    """A modifier that currently refuses (a boolean with no target) refuses
    the whole apply for *that* object -- ``ClayDoc.apply_modifiers``'s own
    contract -- and the rest of the selection still bakes, the same
    refusal-tolerant shape every other object-level op in this registry
    already follows (``run_object_op``'s own docstring).
    """
    doc = bd.ClayDoc()
    broken = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Broken", mesh=bp.box()))
    doc.set_modifiers(broken.uid, (mods.make("boolean", id=1),))  # target=0: refuses
    fine = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Fine", mesh=bp.box()))
    doc.set_modifiers(fine.uid, (mods.make("mirror", id=1),))
    doc.select([broken.uid, fine.uid])
    ctx = _Ctx()

    clay_ops.run(ctx, doc, clay_ops.get("apply-modifiers"))

    assert ctx.toasts.errors, "the broken object's refusal must be toasted"
    assert doc.by_uid(broken.uid).modifiers, "the broken object's stack is untouched"
    assert doc.by_uid(fine.uid).modifiers == (), "the rest of the selection still bakes"


# --- merging applies every stack ---------------------------------------------


def test_every_merge_op_consumes_the_evaluated_mesh_not_the_base() -> None:
    for name in ("_join", "_union", "_difference", "_intersection"):
        source = inspect.getsource(getattr(clay_ops, name))
        assert "doc.evaluated(" in source, f"{name} does not read the evaluated mesh"


def test_join_of_an_object_with_a_mirror_modifier_yields_the_mirrored_geometry() -> None:
    doc, a_uid, b_uid = _doc_with_two_boxes(mirror=True)
    base_faces = bm.face_count(bp.box())

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("join")) is True

    merged = doc.by_uid(a_uid)
    assert b_uid not in {o.uid for o in doc.objects}, "the absorbed object is gone"
    assert merged.modifiers == (), "the target's stack is baked into the merge"
    # A merged with its own mirrored reflection (2x) plus B (1x), and A and B
    # sit far enough apart that ``join``'s own weld (its docstring: applied to
    # the whole merged result) touches nothing.
    assert bm.face_count(merged.mesh) == 3 * base_faces


def test_union_of_an_object_with_a_mirror_modifier_yields_the_mirrored_geometry() -> None:
    mirrored_doc, mirrored_a, _ = _doc_with_two_boxes(mirror=True)
    plain_doc, plain_a, _ = _doc_with_two_boxes(mirror=False)

    assert clay_ops.run(_Ctx(), mirrored_doc, clay_ops.get("union")) is True
    assert clay_ops.run(_Ctx(), plain_doc, clay_ops.get("union")) is True

    mirrored_result = mirrored_doc.by_uid(mirrored_a)
    plain_result = plain_doc.by_uid(plain_a)
    assert mirrored_result.modifiers == (), "the target's stack is baked into the merge"
    assert bm.face_count(mirrored_result.mesh) > bm.face_count(plain_result.mesh), (
        "the union must have merged the mirrored (doubled) shape, not the single base box"
    )


@pytest.mark.parametrize("op_name", ["join", "union"])
def test_a_merge_op_clears_the_targets_stack_even_with_nothing_to_bake(op_name: str) -> None:
    """A target with an *empty* stack is unaffected -- ``join_objects``'s own
    ``clear_modifiers=True`` default is a no-op when there is nothing to
    clear, and every merge op must still run cleanly for the ordinary case of
    two plain objects. Join and Union only: A and B are disjoint (deliberately,
    so the mirror fixture above never welds by accident), and disjoint solids
    have no intersection or difference for ``ops_boolean`` to compute --
    exactly the geometry question this test is not about.
    """
    doc, a_uid, _b_uid = _doc_with_two_boxes(mirror=False)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get(op_name)) is True
    assert doc.by_uid(a_uid).modifiers == ()
