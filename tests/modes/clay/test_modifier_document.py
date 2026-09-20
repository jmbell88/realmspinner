"""The document half of the modifier stack: undo steps, apply, join, to_model.

The geometry itself is :mod:`test_modifiers`'s job; this file pins the shape
of the *edits* -- one step per change, ``apply_modifiers`` as a single
compound an undo restores whole, ``set_mesh``'s "freeze the generator, keep
the stack" rule, and the merging ops' "merging ops consume evaluated meshes"
rule for :meth:`~.document.ClayDoc.join_objects`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from warlock.core.undo import CompoundEdit
from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import elements as el
from warlock.kernels.mesh import mesh as bm
from warlock.kernels.mesh import modifiers as mod
from warlock.kernels.mesh import ops as clay_ops_geom
from warlock.kernels.mesh import primitives as bp


def _obj(name: str, mesh: bm.Mesh | None = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box() if mesh is None else mesh, **kwargs)


def _last_edit(doc: bd.ClayDoc) -> Any:
    return doc.history._done[-1]


# --- set_modifiers -------------------------------------------------------


def test_set_modifiers_pushes_one_step() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A"))
    depth = len(doc.history)

    stack = (mod.make("triangulate", {}, id=1),)
    assert doc.set_modifiers(obj.uid, stack) is True
    assert len(doc.history) == depth + 1
    assert obj.modifiers == stack


def test_set_modifiers_to_the_same_stack_pushes_nothing() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A"))
    stack = (mod.make("triangulate", {}, id=1),)
    doc.set_modifiers(obj.uid, stack)
    depth = len(doc.history)

    assert doc.set_modifiers(obj.uid, stack) is False
    assert len(doc.history) == depth


def test_set_modifiers_refuses_an_unknown_kind_and_pushes_nothing() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A"))
    depth = len(doc.history)
    bad = mod.Modifier(id=1, kind="not-a-kind", params=())

    with pytest.raises(el.OpError):
        doc.set_modifiers(obj.uid, (bad,))
    assert len(doc.history) == depth
    assert obj.modifiers == ()


def test_set_modifiers_refuses_a_cycle_and_pushes_nothing() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    doc.set_modifiers(b.uid, (mod.make("boolean", {"target": a.uid}, id=1),))
    depth = len(doc.history)

    with pytest.raises(el.OpError, match="cycle"):
        doc.set_modifiers(a.uid, (mod.make("boolean", {"target": b.uid}, id=1),))
    assert len(doc.history) == depth
    assert a.modifiers == ()


def test_set_modifiers_refuses_a_self_target_and_pushes_nothing() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    depth = len(doc.history)

    with pytest.raises(el.OpError, match="cycle"):
        doc.set_modifiers(a.uid, (mod.make("boolean", {"target": a.uid}, id=1),))
    assert len(doc.history) == depth


def test_undo_restores_the_previous_modifier_stack() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A"))
    doc.set_modifiers(obj.uid, (mod.make("weld", id=1),))
    doc.set_modifiers(obj.uid, (mod.make("weld", id=1), mod.make("triangulate", {}, id=2)))

    assert doc.undo() is True
    assert [m.kind for m in obj.modifiers] == ["weld"]
    assert doc.undo() is True
    assert obj.modifiers == ()


# --- set_mesh / set_generator_params keep the stack -----------------------


def test_set_mesh_freezes_the_generator_and_keeps_the_stack() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A", generator="box", params={"size": [1.0, 1.0, 1.0]}))
    stack = (mod.make("triangulate", {}, id=1),)
    doc.set_modifiers(obj.uid, stack)

    doc.set_mesh(obj.uid, bp.cone())
    assert obj.generator is None  # the existing freeze rule, untouched
    assert obj.modifiers == stack  # and the stack survives it


def test_set_generator_params_keeps_the_stack() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A", generator="box", params={"size": [1.0, 1.0, 1.0]}))
    stack = (mod.make("mirror", id=1),)
    doc.set_modifiers(obj.uid, stack)

    doc.set_generator_params(
        obj.uid, {"size": [2.0, 1.0, 1.0]}, bp.box(size=(2.0, 1.0, 1.0)), was={}
    )
    assert obj.modifiers == stack


# --- apply_modifiers -------------------------------------------------------


def test_apply_modifiers_bakes_the_whole_stack_by_default() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A"))
    before_mesh = obj.mesh
    doc.set_modifiers(obj.uid, (mod.make("triangulate", {}, id=1),))

    assert doc.apply_modifiers(obj.uid) is True
    assert obj.modifiers == ()
    assert obj.mesh is not before_mesh
    assert np.all(np.diff(obj.mesh.starts) == 3)


def test_apply_modifiers_through_an_id_bakes_only_the_prefix() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A"))
    doc.set_modifiers(
        obj.uid,
        (
            mod.make("triangulate", {}, id=1),
            mod.make(
                "array",
                {"count": 2, "offset_x": 5.0, "offset_y": 0.0, "offset_z": 0.0, "weld": 0.0},
                id=2,
            ),
        ),
    )

    assert doc.apply_modifiers(obj.uid, through_id=1) is True
    assert [m.id for m in obj.modifiers] == [2]  # only the array remains
    assert np.all(np.diff(obj.mesh.starts) == 3)  # the baked triangulate shows


def test_apply_modifiers_is_one_compound_edit_and_undo_restores_it_whole() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A", generator="box", params={"size": [1.0, 1.0, 1.0]}))
    before_mesh = obj.mesh
    doc.set_modifiers(obj.uid, (mod.make("triangulate", {}, id=1),))
    depth = len(doc.history)

    doc.apply_modifiers(obj.uid)
    assert len(doc.history) == depth + 1
    assert isinstance(_last_edit(doc), CompoundEdit)

    assert doc.undo() is True
    assert obj.mesh is before_mesh
    assert obj.generator == "box"
    assert [m.kind for m in obj.modifiers] == ["triangulate"]


def test_apply_modifiers_drops_a_disabled_modifier_in_the_prefix_without_applying_it() -> None:
    from dataclasses import replace as _replace

    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A"))
    disabled_weld = _replace(mod.make("weld", {"distance": 0.5}, id=1), enabled=False)
    doc.set_modifiers(obj.uid, (disabled_weld,))

    before_mesh = obj.mesh
    assert doc.apply_modifiers(obj.uid) is True
    assert obj.modifiers == ()
    # A disabled modifier contributed nothing to what was on screen, so baking
    # it must not change the mesh at all.
    assert obj.mesh is before_mesh


def test_apply_modifiers_refuses_the_whole_apply_when_the_prefix_currently_errors() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A"))
    doc.set_modifiers(obj.uid, (mod.make("boolean", {"target": 0}, id=1),))  # always refuses
    depth = len(doc.history)
    before_mesh = obj.mesh
    before_stack = obj.modifiers

    with pytest.raises(el.OpError):
        doc.apply_modifiers(obj.uid)
    assert len(doc.history) == depth
    assert obj.mesh is before_mesh
    assert obj.modifiers == before_stack


def test_apply_modifiers_on_an_empty_stack_is_a_no_op() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A"))
    depth = len(doc.history)
    assert doc.apply_modifiers(obj.uid) is False
    assert len(doc.history) == depth


def test_apply_modifiers_refuses_an_unknown_through_id() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A"))
    doc.set_modifiers(obj.uid, (mod.make("weld", id=1),))
    with pytest.raises(el.OpError):
        doc.apply_modifiers(obj.uid, through_id=99)


# --- join_objects: merging ops consume evaluated meshes --------------------


def test_join_objects_clears_the_targets_stack_by_default() -> None:
    doc = bd.ClayDoc()
    target = doc.add_object(_obj("T"))
    other = doc.add_object(_obj("O", translation=(3.0, 0.0, 0.0)))
    doc.set_modifiers(target.uid, (mod.make("triangulate", {}, id=1),))

    merged = doc.evaluated(target.uid)
    assert doc.join_objects(target.uid, merged, [other.uid]) is True
    assert target.modifiers == ()
    assert target.mesh is merged


def test_join_objects_keeps_the_stack_when_asked() -> None:
    doc = bd.ClayDoc()
    target = doc.add_object(_obj("T"))
    other = doc.add_object(_obj("O", translation=(3.0, 0.0, 0.0)))
    stack = (mod.make("triangulate", {}, id=1),)
    doc.set_modifiers(target.uid, stack)

    doc.join_objects(target.uid, bp.box(), [other.uid], clear_modifiers=False)
    assert target.modifiers == stack


def test_join_objects_stack_clear_is_inside_the_one_compound_step() -> None:
    doc = bd.ClayDoc()
    target = doc.add_object(_obj("T"))
    other = doc.add_object(_obj("O", translation=(3.0, 0.0, 0.0)))
    stack = (mod.make("triangulate", {}, id=1),)
    doc.set_modifiers(target.uid, stack)
    merged = doc.evaluated(target.uid)
    before_mesh = target.mesh
    depth = len(doc.history)

    doc.join_objects(target.uid, merged, [other.uid])
    assert len(doc.history) == depth + 1

    assert doc.undo() is True
    assert target.mesh is before_mesh
    assert target.modifiers == stack
    assert doc.by_uid(other.uid) is other


# --- to_model draws the evaluated mesh -------------------------------------


def test_to_model_draws_the_evaluated_mesh_not_the_base() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A"))
    doc.set_modifiers(obj.uid, (mod.make("triangulate", {}, id=1),))

    model = bd.to_model(doc)
    assert len(model.meshes) == 1
    total_indices = sum(len(p.indices) for p in model.meshes[0])
    # Triangulated: 12 triangles * 3 indices, not the 6-quad base's own count.
    assert total_indices == 12 * 3


def test_to_primitives_defaults_to_the_base_mesh_with_no_doc_in_hand() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A"))
    doc.set_modifiers(obj.uid, (mod.make("triangulate", {}, id=1),))

    # No ``mesh=`` passed: a caller with no document reaches for the base,
    # exactly as before modifiers existed.
    prims = bd.to_primitives(obj, doc.materials)
    total_indices = sum(len(p.indices) for p in prims)
    assert total_indices == 6 * 2 * 3  # the box's own quads, fanned to triangles


# --- duplicate keeps the stack ---------------------------------------------


def test_duplicate_keeps_the_modifier_stack_by_sharing_the_tuple() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A"))
    stack = (mod.make("mirror", id=1),)
    doc.set_modifiers(obj.uid, stack)

    dup = clay_ops_geom.duplicate(obj, bd.new_uid())
    assert dup.modifiers == stack
    # A tuple is immutable, so dataclasses.replace sharing the same object is
    # safe -- and worth pinning, because it is what makes this free.
    assert dup.modifiers is obj.modifiers


# --- remove_object pops the evaluation cache -------------------------------


def test_remove_and_undo_still_evaluates_correctly() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("A"))
    doc.set_modifiers(obj.uid, (mod.make("triangulate", {}, id=1),))
    doc.evaluated(obj.uid)  # populate the cache before it is removed

    doc.remove_object(obj.uid)
    assert doc.undo() is True
    restored = doc.by_uid(obj.uid)
    assert restored is obj
    ev = doc.evaluation(obj.uid)
    assert np.all(np.diff(ev.mesh.starts) == 3)
