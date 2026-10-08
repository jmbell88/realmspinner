"""The 2026-10-07 audit's document-model findings: clay-02, 05, 13, 14, 56, 68, 69, 71, 72.

Each test name is the claim; each one failed against the code it was written
for. Everything here is headless: a ``ClayDoc`` and the kernels beside it.
"""

from __future__ import annotations

import dataclasses
import random
import re
from pathlib import Path

import numpy as np
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import glbimport, ops_subdiv, serialize
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops as mesh_ops
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import separate as sep


def _obj(name: str = "A", **kwargs: object) -> bd.Obj:
    kwargs.setdefault("mesh", bp.box((1, 1, 1)))
    return bd.Obj(uid=bd.new_uid(), name=name, **kwargs)  # type: ignore[arg-type]


def _slots(doc: bd.ClayDoc) -> list[tuple[str, list[int], int]]:
    return [(o.name, sorted(set(o.mesh.material.tolist())), o.material) for o in doc.objects]


# --- clay-02 -------------------------------------------------------------------


def test_an_object_added_after_a_palette_change_keeps_its_slot_through_undo_and_redo() -> None:
    # (a) an append: a duplicate of a painted object, added after the palette add.
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    slot = doc.add_material()
    doc.paint_faces(a.uid, range(6), slot)
    doc.add_objects([_obj("A.copy", mesh=a.mesh)])
    expected = _slots(doc)
    for _ in range(3):
        assert doc.undo()
    for _ in range(3):
        assert doc.redo()
    assert _slots(doc) == expected

    # (b) a removal of the lowest slot, then an object added after it.
    doc = bd.ClayDoc()
    slot = doc.add_material()
    a = doc.add_object(_obj("A"))
    doc.paint_faces(a.uid, range(6), slot)
    assert doc.remove_material(0)
    doc.add_object(_obj("B"))
    expected = _slots(doc)
    while doc.undo():
        pass
    while doc.redo():
        pass
    assert _slots(doc) == expected
    assert all(min(slots) >= 0 for _name, slots, _default in _slots(doc))


def test_separate_pieces_keep_their_slot_through_undo_all_and_redo_all() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B", translation=[5, 0, 0]))
    joined = mesh_ops.join(
        [a, b], eps=1e-4, world=[doc.world_matrix(a.uid), doc.world_matrix(b.uid)]
    )
    doc.join_objects(a.uid, joined, [b.uid])
    slot = doc.add_material()
    doc.paint_faces(a.uid, range(len(doc.by_uid(a.uid).mesh.material)), slot)
    doc.separate(a.uid, sep.by_loose_parts(doc.by_uid(a.uid).mesh))
    expected = _slots(doc)
    while doc.undo():
        pass
    while doc.redo():
        pass
    assert _slots(doc) == expected


def _snapshot(doc: bd.ClayDoc) -> tuple:
    def mesh_key(m: bm.Mesh) -> tuple:
        return tuple(
            np.ascontiguousarray(getattr(m, n)).tobytes()
            for n in ("positions", "loops", "starts", "material", "smooth")
        )

    return (
        tuple(
            (o.uid, o.name, o.parent, o.material, o.generator, o.visible, mesh_key(o.mesh))
            for o in doc.objects
        ),
        tuple((m.name, m.base_color is not None) for m in doc.materials),
    )


def _fuzz_once(seed: int) -> list[str]:
    """Random edits including palette steps, then undo-all / redo-all with every
    intermediate state compared against the snapshot taken when it was made."""
    rnd = random.Random(seed)
    doc = bd.ClayDoc()
    states = [_snapshot(doc)]
    for step in range(40):
        head = doc.history.head
        uids = [o.uid for o in doc.objects]
        kind = rnd.choice(
            ["add", "add", "remove", "addmat", "rmmat", "rmmat2", "paint", "repaint",
             "dup", "separate", "join", "mesh", "tex"]
        )  # fmt: skip
        try:
            if kind == "add" or not uids:
                doc.add_object(_obj(f"o{step}", translation=[rnd.uniform(-3, 3), 0, 0]))
            elif kind == "remove":
                doc.remove_object(rnd.choice(uids))
            elif kind == "addmat":
                doc.add_material()
            elif kind == "rmmat":
                doc.remove_material(rnd.randrange(len(doc.materials)))
            elif kind == "rmmat2":
                slot = rnd.randrange(len(doc.materials))
                doc.remove_material_and_reassign(rnd.choice(uids), slot)
            elif kind == "paint":
                o = doc.by_uid(rnd.choice(uids))
                doc.paint_faces(o.uid, [0, 1], rnd.randrange(len(doc.materials)))
            elif kind == "repaint":
                doc.repaint_object(rnd.choice(uids), rnd.randrange(len(doc.materials)))
            elif kind == "dup":
                src = doc.by_uid(rnd.choice(uids))
                doc.add_objects([_obj(f"{src.name}.d{step}", mesh=src.mesh, material=src.material)])
            elif kind == "separate":
                o = doc.by_uid(rnd.choice(uids))
                doc.separate(o.uid, sep.by_loose_parts(o.mesh))
            elif kind == "join" and len(uids) >= 2:
                sel = rnd.sample(uids, 2)
                objs = [doc.by_uid(u) for u in sel]
                worlds = [doc.world_matrix(u) for u in sel]
                doc.join_objects(sel[0], mesh_ops.join(objs, eps=1e-4, world=worlds), sel[1:])
            elif kind == "mesh":
                o = doc.by_uid(rnd.choice(uids))
                doc.set_mesh(o.uid, bm.transformed(o.mesh, m3.translation(m3.vec3(1, 0, 0))))
            elif kind == "tex":
                doc.add_texture(rnd.randrange(len(doc.materials)), 32)
        except (el.OpError, KeyError, ValueError):
            pass
        if doc.history.head != head:
            states.append(_snapshot(doc))
    problems: list[str] = []
    n = len(states) - 1
    for i in range(n, 0, -1):
        assert doc.undo()
        if _snapshot(doc) != states[i - 1]:
            problems.append(f"seed {seed}: undo to state {i - 1} differs")
            return problems
    for i in range(1, n + 1):
        assert doc.redo()
        if _snapshot(doc) != states[i]:
            problems.append(f"seed {seed}: redo to state {i} differs")
            return problems
    return problems


def test_undo_all_then_redo_all_restores_every_state_across_palette_steps() -> None:
    problems: list[str] = []
    for seed in range(60):
        problems.extend(_fuzz_once(seed))
    assert not problems, problems[:3]


# --- clay-05 -------------------------------------------------------------------


def test_a_rolled_back_batch_leaves_no_element_selection_on_the_restored_mesh() -> None:
    doc = bd.ClayDoc()
    o = doc.add_object(_obj("A"))
    doc.set_element_mode("face")
    doc.set_element_sel(o.uid, el.ElementSel(faces=np.arange(6)))
    new_mesh, new_sel = ops_subdiv.subdivide(o.mesh, doc.element_sel_of(o.uid))
    doc.set_mesh(o.uid, new_mesh, select=new_sel)
    steps = len(doc.history)
    redo_before = doc.history.can_redo

    assert doc.undo_unrecorded()

    assert len(doc.history) == steps - 1
    assert doc.history.can_redo == redo_before  # forgotten, not redoable
    assert bm.face_count(o.mesh) == 6
    assert el.is_empty(doc.element_sel_of(o.uid))
    assert o.uid not in doc.selection  # the derived-selection invariant holds


def test_undo_unrecorded_with_nothing_to_undo_is_false() -> None:
    assert bd.ClayDoc().undo_unrecorded() is False


# --- clay-13 -------------------------------------------------------------------


def test_the_document_ceilings_are_the_ones_the_writer_and_reader_enforce() -> None:
    assert sep.MAX_SEPARATE_PIECES == glbimport.MAX_OBJECTS


def test_add_object_refuses_to_grow_the_document_past_what_it_can_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(glbimport, "MAX_OBJECTS", 3)
    doc = bd.ClayDoc()
    for i in range(3):
        doc.add_object(_obj(f"o{i}"))
    steps = len(doc.history)
    with pytest.raises(el.OpError, match=r"4|3"):
        doc.add_object(_obj("one too many"))
    assert len(doc.objects) == 3
    assert len(doc.history) == steps
    with pytest.raises(el.OpError, match="3"):
        doc.add_objects([_obj("x")])
    assert len(doc.objects) == 3 and len(doc.history) == steps
    with pytest.raises(el.OpError, match="3"):
        doc.group([doc.objects[0].uid])
    assert len(doc.objects) == 3 and len(doc.history) == steps


def test_mirror_copy_refuses_to_grow_the_document_past_what_it_can_save() -> None:
    """The real ceiling, no monkeypatch: a document that holds exactly
    ``MAX_OBJECTS`` objects refuses the next one, and one at the ceiling saves
    and reopens."""
    box = bp.box((1, 1, 1))
    doc = bd.ClayDoc([bd.Obj(uid=bd.new_uid(), name=f"o{i}", mesh=box) for i in range(4_095)])
    doc.add_object(_obj("last one that fits"))
    assert len(doc.objects) == glbimport.MAX_OBJECTS
    with pytest.raises(el.OpError, match=f"{glbimport.MAX_OBJECTS:,}"):
        doc.add_objects([_obj("a"), _obj("b")])
    with pytest.raises(el.OpError, match=f"{glbimport.MAX_OBJECTS:,}"):
        doc.add_object(_obj("c"))
    assert len(doc.objects) == glbimport.MAX_OBJECTS
    reopened = serialize.read_rblk(serialize.rblk_bytes(doc))
    assert len(reopened.objects) == glbimport.MAX_OBJECTS


def test_add_objects_refuses_past_the_triangle_ceiling_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(glbimport, "MAX_TRIANGLES", 30)  # a box is 12
    doc = bd.ClayDoc()
    doc.add_object(_obj("a"))
    doc.add_object(_obj("b"))
    with pytest.raises(el.OpError, match="30") as caught:
        doc.add_object(_obj("c"))
    assert "triangle" in str(caught.value)
    assert len(doc.objects) == 2
    with pytest.raises(el.OpError, match="30"):
        doc.add_objects([_obj("c")])
    # An empty object (Group's) costs no triangles and still fits.
    doc.add_object(_obj("empty", mesh=bd._empty_mesh()))


def test_a_split_past_the_object_ceiling_is_refused_by_name_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B", translation=[5, 0, 0]))
    joined = mesh_ops.join(
        [a, b], eps=1e-4, world=[doc.world_matrix(a.uid), doc.world_matrix(b.uid)]
    )
    doc.join_objects(a.uid, joined, [b.uid])
    doc.add_object(_obj("C"))
    pieces = sep.by_loose_parts(doc.by_uid(a.uid).mesh)
    assert len(pieces) == 2
    # Two objects now; the split would make three -- one more than the ceiling.
    monkeypatch.setattr(glbimport, "MAX_OBJECTS", 2)
    steps = len(doc.history)
    with pytest.raises(el.OpError, match="2"):
        doc.separate(a.uid, pieces)
    assert [o.name for o in doc.objects] == ["A", "C"]
    assert len(doc.history) == steps


# --- clay-14 -------------------------------------------------------------------


def test_a_typed_move_too_large_for_float32_is_refused_and_leaves_the_mesh_finite() -> None:
    doc = bd.ClayDoc()
    o = doc.add_object(_obj("A"))
    before = o.mesh
    steps = len(doc.history)
    with np.errstate(over="ignore"):
        huge = bm.transformed(before, m3.translation(m3.vec3(1e40, 0, 0)))
    assert not np.isfinite(huge.positions).all()  # the premise: f4 overflows to inf
    with pytest.raises(el.OpError):
        doc.set_mesh(o.uid, huge)
    assert o.mesh is before
    assert len(doc.history) == steps
    nan_uv = dataclasses.replace(
        before, uv=np.full((len(before.loops), 2), np.nan, dtype="f4")
    )
    with pytest.raises(el.OpError):
        doc.set_mesh(o.uid, nan_uv)
    assert o.mesh is before and len(doc.history) == steps
    serialize.rblk_bytes(doc)  # still saves


# --- clay-56 -------------------------------------------------------------------


def test_reduce_material_keeps_or_documents_the_alpha_cutoff() -> None:
    cut = dataclasses.replace(bd.default_material("leaf"), alpha_mode="MASK", alpha_cutoff=0.2)
    reduced = bd.reduce_material(cut)
    assert reduced.alpha_mode == "MASK"
    assert reduced.alpha_cutoff == pytest.approx(0.2)
    assert bd.reduce_material(reduced) is reduced  # still idempotent
    opaque = bd.reduce_material(dataclasses.replace(cut, alpha_mode="BLEND"))
    assert opaque.alpha_mode == "OPAQUE"
    assert opaque.alpha_cutoff == pytest.approx(0.5)  # meaningless there: the spec default


# --- clay-68 -------------------------------------------------------------------


def test_set_origin_refuses_a_shift_that_makes_the_mesh_non_finite() -> None:
    doc = bd.ClayDoc()
    o = doc.add_object(_obj("A"))
    assert doc.set_transform(o.uid, scale=[1e-40, 1, 1])
    steps = len(doc.history)
    before_mesh, before_t = o.mesh, np.array(o.translation, copy=True)
    with pytest.raises(el.OpError):
        doc.set_origin(o.uid, [1.0, 0, 0])
    assert o.mesh is before_mesh
    assert np.array_equal(o.translation, before_t)
    assert len(doc.history) == steps
    assert np.isfinite(o.mesh.positions).all()


# --- clay-69 -------------------------------------------------------------------


def test_a_non_unit_rotation_is_normalised_on_set() -> None:
    doc = bd.ClayDoc()
    o = doc.add_object(_obj("A"))
    assert doc.set_transform(o.uid, rotation=[0.0, 0.5, 0.0, 0.5])
    assert np.linalg.norm(o.rotation) == pytest.approx(1.0)
    assert o.rotation[1] == pytest.approx(np.sqrt(0.5))
    columns = np.linalg.norm(doc.world_matrix(o.uid)[:3, :3], axis=0)
    assert columns == pytest.approx([1.0, 1.0, 1.0])
    # Undo restores the identity rotation, redo the normalised one.
    doc.undo()
    assert np.array_equal(o.rotation, [0, 0, 0, 1])
    doc.redo()
    assert np.linalg.norm(o.rotation) == pytest.approx(1.0)
    # A huge but finite quaternion must not overflow while normalising.
    assert doc.set_transform(o.uid, rotation=[1e200, 0, 0, 1e200])
    assert np.isfinite(o.rotation).all()
    assert np.linalg.norm(o.rotation) == pytest.approx(1.0)
    with pytest.raises(el.OpError):
        doc.set_transform(o.uid, rotation=[0, 0, 0, 0])


def test_setting_an_already_unit_rotation_is_still_a_no_op() -> None:
    doc = bd.ClayDoc()
    o = doc.add_object(_obj("A"))
    q = m3.quat_from_euler_xyz([10.0, 20.0, 30.0])
    assert doc.set_transform(o.uid, rotation=q)
    steps = len(doc.history)
    assert doc.set_transform(o.uid, rotation=o.rotation) is False
    assert len(doc.history) == steps


# --- clay-71 -------------------------------------------------------------------


def _chain(n: int) -> tuple[bd.ClayDoc, list[bd.Obj]]:
    box = bp.box((1, 1, 1))
    objs: list[bd.Obj] = []
    for i in range(n):
        objs.append(
            bd.Obj(
                uid=bd.new_uid(),
                name=f"o{i}",
                mesh=box,
                translation=[0.01, 0, 0],
                parent=objs[-1].uid if objs else None,
            )
        )
    return bd.ClayDoc(objs), objs


def test_world_matrix_on_a_two_thousand_deep_chain_stays_under_a_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, objs = _chain(2_000)
    calls = {"index_of": 0, "children_of": 0}
    real_index_of, real_children_of = bd.ClayDoc.index_of, bd.ClayDoc.children_of

    def counted_index_of(self: bd.ClayDoc, uid: int) -> int:
        calls["index_of"] += 1
        return real_index_of(self, uid)

    def counted_children_of(self: bd.ClayDoc, uid: int) -> list[int]:
        calls["children_of"] += 1
        return real_children_of(self, uid)

    monkeypatch.setattr(bd.ClayDoc, "index_of", counted_index_of)
    monkeypatch.setattr(bd.ClayDoc, "children_of", counted_children_of)

    world = doc.world_matrix(objs[-1].uid)
    assert world[0, 3] == pytest.approx(0.01 * 2_000)
    ancestors = doc.ancestors(objs[-1].uid)
    assert ancestors == [o.uid for o in reversed(objs[:-1])]
    descendants = doc.descendants(objs[0].uid)
    assert descendants == [o.uid for o in objs[1:]]
    # One map per call, not a linear scan per hop: a handful of lookups, where
    # the scan was ~2,000 per call (O(depth x objects)).
    assert calls["index_of"] <= 10
    assert calls["children_of"] <= 10


def test_set_parent_cycle_check_is_not_quadratic_on_a_deep_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, objs = _chain(1_500)
    calls = {"children_of": 0}
    real = bd.ClayDoc.children_of

    def counted(self: bd.ClayDoc, uid: int) -> list[int]:
        calls["children_of"] += 1
        return real(self, uid)

    monkeypatch.setattr(bd.ClayDoc, "children_of", counted)
    with pytest.raises(el.OpError):
        doc.set_parent(objs[0].uid, objs[-1].uid)
    assert calls["children_of"] <= 10


def test_descendants_and_ancestors_keep_their_answers_on_a_branching_tree() -> None:
    doc = bd.ClayDoc()
    root = doc.add_object(_obj("root"))
    a = doc.add_object(_obj("a", parent=root.uid))
    b = doc.add_object(_obj("b", parent=root.uid))
    a1 = doc.add_object(_obj("a1", parent=a.uid))
    b1 = doc.add_object(_obj("b1", parent=b.uid))
    assert doc.descendants(root.uid) == [a.uid, a1.uid, b.uid, b1.uid]
    assert doc.ancestors(b1.uid) == [b.uid, root.uid]
    assert doc.ancestors(root.uid) == []
    with pytest.raises(KeyError):
        doc.ancestors(10**9)
    with pytest.raises(KeyError):
        doc.world_matrix(10**9)


# --- clay-72 -------------------------------------------------------------------


def test_the_stale_comments_name_their_real_owner() -> None:
    root = Path(bd.__file__).parent
    document = (root / "document.py").read_text(encoding="utf-8")
    edits = (root / "edits.py").read_text(encoding="utf-8")
    assert "``clay_ops`` does that in one place for every op" not in document
    assert not re.search(r"itself never fills one in", edits)
