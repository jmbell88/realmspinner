"""The 2026-10-09 audit's document-layer findings: clay-01, 02, 08, 17, 18, 46.

Each test name is the claim, and each failed against the code it was written for.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import glbimport, ops
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

from .test_agent_clay import _Ctx, _payload


def _add(doc: bd.ClayDoc, name: str, **kwargs: object) -> bd.Obj:
    obj = bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box(), **kwargs)  # type: ignore[arg-type]
    doc.add_object(obj)
    return obj


def test_set_generator_params_refuses_a_mesh_whose_positions_overflow_float32() -> None:
    doc = bd.ClayDoc()
    defaults, build = bp.GENERATORS["torus"]
    obj = bd.Obj(
        uid=bd.new_uid(), name="t", mesh=build(**defaults), generator="torus", params=dict(defaults)
    )
    doc.add_object(obj)
    edited = bp.clamp_params("torus", {**defaults, "radius": 3.0e38, "tube": 3.0e38})
    with np.errstate(all="ignore"):
        mesh = build(**edited)
    assert not np.isfinite(mesh.positions).all()
    steps, before = len(doc.history), obj.mesh
    with pytest.raises(el.OpError):
        doc.set_generator_params(obj.uid, edited, mesh, was={"params": dict(defaults)})
    assert obj.mesh is before
    assert obj.params == defaults
    assert len(doc.history) == steps


def test_set_generator_params_refuses_a_rebuild_past_the_triangle_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = bd.ClayDoc()
    defaults, build = bp.GENERATORS["torus"]
    obj = bd.Obj(
        uid=bd.new_uid(), name="t", mesh=build(**defaults), generator="torus", params=dict(defaults)
    )
    doc.add_object(obj)
    big = bp.clamp_params("torus", {**defaults, "segments": 64, "sides": 64})
    mesh = build(**big)
    monkeypatch.setattr(glbimport, "MAX_TRIANGLES", bd._triangles_of(obj.mesh) + 10)
    before, steps = obj.mesh, len(doc.history)
    with pytest.raises(el.OpError, match="triangles"):
        doc.set_generator_params(obj.uid, big, mesh, was={"params": dict(defaults)})
    assert obj.mesh is before and len(doc.history) == steps


def test_set_mesh_refuses_a_bigger_mesh_past_the_triangle_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = bd.ClayDoc()
    obj = _add(doc, "a")
    defaults, build = bp.GENERATORS["uv_sphere"]
    bigger = build(**defaults)
    assert bd._triangles_of(bigger) > bd._triangles_of(obj.mesh)
    monkeypatch.setattr(glbimport, "MAX_TRIANGLES", bd._triangles_of(obj.mesh) + 1)
    before, steps = obj.mesh, len(doc.history)
    with pytest.raises(el.OpError, match="triangles"):
        doc.set_mesh(obj.uid, bigger)
    assert obj.mesh is before and len(doc.history) == steps


def test_join_objects_refuses_a_merged_mesh_that_is_not_finite() -> None:
    doc = bd.ClayDoc()
    a, b = _add(doc, "a"), _add(doc, "b")
    doc.set_transform(a.uid, scale=(1e-20,) * 3)
    doc.set_transform(b.uid, scale=(1e20,) * 3)
    with np.errstate(all="ignore"):
        mesh = ops.join(
            [a, b], eps=0.0, world=[doc.world_matrix(a.uid), doc.world_matrix(b.uid)]
        )
    assert not np.isfinite(mesh.positions).all()
    before, steps, count = a.mesh, len(doc.history), len(doc.objects)
    with pytest.raises(el.OpError):
        doc.join_objects(a.uid, mesh, [b.uid])
    assert a.mesh is before and len(doc.objects) == count and len(doc.history) == steps


def test_set_origin_refuses_a_translation_that_is_not_finite_on_a_vertexless_object() -> None:
    doc = bd.ClayDoc()
    a = _add(doc, "a")
    g = doc.group([a.uid])
    par = _add(doc, "par")
    doc.set_parent(g.uid, par.uid)
    doc.remove_object(a.uid)
    doc.set_transform(par.uid, scale=(1e-320,) * 3)
    steps, before = len(doc.history), np.array(doc.by_uid(g.uid).translation, copy=True)
    with np.errstate(all="ignore"), pytest.raises(el.OpError):
        doc.set_origin(g.uid, (1.0, 2.0, 3.0))
    assert np.array_equal(doc.by_uid(g.uid).translation, before)
    assert len(doc.history) == steps


def _agent_doc() -> tuple[_Ctx, agent_clay.Session, bd.ClayDoc, int, int]:
    ctx, session = _Ctx(), agent_clay.Session()
    a = _payload(agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"}))["uid"]
    b = _payload(
        agent_clay.call(
            ctx, session, "clay_add_primitive", {"generator": "box", "translation": [3, 0, 0]}
        )
    )["uid"]
    doc = clay_mode.ensure(ctx).get(session.tab_uid).doc
    doc.set_visibility({b: False})
    return ctx, session, doc, a, b


def test_clay_select_refuses_a_hidden_object_and_leaves_selection_problems_empty() -> None:
    ctx, session, doc, a, b = _agent_doc()
    agent_clay.call(ctx, session, "clay_select", {"uids": [a, b]})
    assert b not in doc.selection
    assert doc.selection_problems() == []
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    agent_clay.call(ctx, session, "clay_op", {"name": "select-all"})
    assert doc.selection_problems() == []


def test_select_and_set_element_sel_skip_a_hidden_object() -> None:
    doc = bd.ClayDoc()
    a, b = _add(doc, "a"), _add(doc, "b")
    doc.set_visibility({b.uid: False})
    doc.select([a.uid, b.uid])
    assert doc.selection == {a.uid}
    doc.set_element_mode("face")
    doc.set_element_sel(b.uid, el.select_all(b.mesh, "face"))
    assert b.uid not in doc.element_sel and b.uid not in doc.selection
    assert doc.selection_problems() == []


def test_set_element_mode_docstring_says_it_adds_converted_picks() -> None:
    doc = (bd.ClayDoc.set_element_mode.__doc__ or "").lower()
    assert "never touches" not in doc
