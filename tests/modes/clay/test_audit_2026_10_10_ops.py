"""Regression tests for the 2026-10-10 Clay audit's op-registry findings
(clay-22, 23, 51, 52, 53, 54). Each test name is the claim, and each was run
against the unfixed tree first."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import glbimport
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops as geom_ops
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops


class _Ctx:
    """Only what the real ``Ctx`` offers: ``toast(text, level)``."""

    def __init__(self) -> None:
        self.msgs: list[str] = []
        self.clay_view: Any = None
        self.state: Any = None

    def toast(self, message: str, level: str = "info") -> None:
        self.msgs.append(message)


def _box(doc: bd.ClayDoc, name: str = "Box", mesh: bm.Mesh | None = None, **kw: Any) -> int:
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name=name,
            mesh=bp.box() if mesh is None else mesh,
            generator="box" if mesh is None else None,
            params={"size": (1.0, 1.0, 1.0)} if mesh is None else {},
            **kw,
        )
    )
    return obj.uid


def _strip(quads: int) -> bm.Mesh:
    """A valid mesh of *quads* unit quads in a row (2 triangles each)."""
    positions = []
    faces = []
    for i in range(quads):
        base = len(positions)
        positions += [(i, 0, 0), (i + 1, 0, 0), (i + 1, 1, 0), (i, 1, 0)]
        faces.append([base, base + 1, base + 2, base + 3])
    return bm.from_faces(positions, faces)


def _copy_of(mesh: bm.Mesh) -> bm.Mesh:
    return bm.Mesh(
        positions=mesh.positions.copy(),
        loops=mesh.loops.copy(),
        starts=mesh.starts.copy(),
        material=mesh.material.copy(),
        smooth=mesh.smooth.copy(),
        uv=None if mesh.uv is None else mesh.uv.copy(),
    )


def _invalid_like(mesh: bm.Mesh) -> bm.Mesh:
    """A mesh whose CSR is not valid: ``starts`` and ``material`` disagree."""
    return bm.Mesh(
        positions=mesh.positions.copy(),
        loops=mesh.loops.copy(),
        starts=np.array([0, 2, mesh.starts[-1]], dtype=mesh.starts.dtype),
        material=mesh.material[:2].copy(),
        smooth=mesh.smooth[:2].copy(),
        uv=None,
    )


def _three_boxes_in_face_mode() -> tuple[bd.ClayDoc, list[int]]:
    doc = bd.ClayDoc()
    uids = [_box(doc, name) for name in ("A", "B", "C")]
    doc.set_element_mode("face")
    for uid in uids:
        doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    return doc, uids


# --- clay-22: a refused multi-object element op never half-applies -------------


def test_a_multi_object_element_op_whose_second_result_is_invalid_never_half_applies() -> None:
    """The unfixed ``run_mesh_op`` checked only ``_refuse_unstorable`` in phase
    one, so the second object's invalid CSR raised from ``set_mesh`` in phase
    two *after* the first was rewritten: ``run`` answered False with one object
    changed and a step in history. An invalid result is now that object's own
    refusal (toasted, skipped), and ``run``'s answer agrees with the document."""
    doc, uids = _three_boxes_in_face_mode()
    before = {u: doc.by_uid(u).mesh for u in uids}
    calls = {"n": 0}

    def kernel(mesh: bm.Mesh, sel: Any, **_: Any) -> Any:
        calls["n"] += 1
        # The first object gets a valid (new) mesh, the second an invalid one.
        return (_copy_of(mesh) if calls["n"] != 2 else _invalid_like(mesh)), sel

    op = clay_ops.Op(
        name="probe",
        label="Probe",
        modes=("face",),
        run=lambda ctx, doc, **kw: clay_ops.run_mesh_op(ctx, doc, kernel, **kw),
    )
    ctx = _Ctx()
    head = doc.history.head

    ran = clay_ops.run(ctx, doc, op)

    changed = [u for u in uids if doc.by_uid(u).mesh is not before[u]]
    pushed = doc.history.head != head
    # Whatever it answers, the answer must be the truth about the document.
    assert ran == bool(changed) == pushed, (ran, changed, pushed, ctx.msgs)
    assert before[uids[1]] is doc.by_uid(uids[1]).mesh, "the invalid object was written"
    assert any("'B'" in m and "not valid" in m for m in ctx.msgs), ctx.msgs
    if ran:
        doc.undo()
        assert all(doc.by_uid(u).mesh is before[u] for u in uids), "one Ctrl+Z restores all"


def test_a_multi_object_op_whose_every_result_is_invalid_answers_false_and_pushes_nothing() -> None:
    doc, uids = _three_boxes_in_face_mode()
    before = {u: doc.by_uid(u).mesh for u in uids}

    def kernel(mesh: bm.Mesh, sel: Any, **_: Any) -> Any:
        return _invalid_like(mesh), sel

    op = clay_ops.Op(
        name="probe",
        label="Probe",
        modes=("face",),
        run=lambda ctx, doc, **kw: clay_ops.run_mesh_op(ctx, doc, kernel, **kw),
    )
    head = doc.history.head
    ran = clay_ops.run(_Ctx(), doc, op)
    assert ran is False
    assert doc.history.head == head
    assert all(doc.by_uid(u).mesh is before[u] for u in uids)


def test_a_multi_object_element_op_writes_the_shrinking_object_before_the_growing_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = bd.ClayDoc()
    small = _box(doc, "Small", mesh=_strip(1))
    big = _box(doc, "Big", mesh=_strip(6))
    doc.set_element_mode("face")
    for uid in (small, big):
        doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    grown, shrunk = _strip(6), _strip(1)

    def kernel(mesh: bm.Mesh, sel: Any, **_: Any) -> Any:
        return (grown if len(mesh.positions) == 4 else shrunk), sel

    total = sum(bd._triangles_of(o.mesh) for o in doc.objects)
    # Room for the end state only if the shrink lands first: growing 'Small'
    # while 'Big' still holds its triangles would take the document one over.
    monkeypatch.setattr(glbimport, "MAX_TRIANGLES", total)
    op = clay_ops.Op(
        name="probe",
        label="Probe",
        modes=("face",),
        run=lambda ctx, doc, **kw: clay_ops.run_mesh_op(ctx, doc, kernel, **kw),
    )
    ctx = _Ctx()
    ran = clay_ops.run(ctx, doc, op)
    assert ran, ctx.msgs
    assert doc.by_uid(small).mesh is grown and doc.by_uid(big).mesh is shrunk


# --- clay-23: whole-selection object ops build the uid map once ----------------


class _Counter:
    """Counts the doors that rebuild the whole uid map, on one document class.

    ``index_of`` is not counted: it is a bare scan, and ``set_transform`` /
    ``set_mesh`` / ``set_origin`` each make one of their own on every call, which
    is the document's cost and not the op's.
    """

    NAMES = ("_uid_map", "world_matrix", "ancestors", "descendants")

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.counts = dict.fromkeys(self.NAMES, 0)
        for name in self.NAMES:
            original = getattr(bd.ClayDoc, name)

            def wrapped(self_: Any, *a: Any, _o: Any = original, _n: str = name, **k: Any) -> Any:
                self.counts[_n] += 1
                return _o(self_, *a, **k)

            monkeypatch.setattr(bd.ClayDoc, name, wrapped)

    def reset(self) -> None:
        for name in self.NAMES:
            self.counts[name] = 0

    def total(self) -> int:
        return sum(self.counts.values())


def _many(n: int, parented: bool = False) -> bd.ClayDoc:
    doc = bd.ClayDoc()
    objs = [
        bd.Obj(
            uid=bd.new_uid(),
            name=f"B{i}",
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
            translation=np.array([i * 2.0 + 0.3, 0.37, 0.0]),
        )
        for i in range(n)
    ]
    if parented:
        for child in objs[n // 2 :]:
            child.parent = objs[(objs.index(child) - n // 2) % (n // 2)].uid
    doc.add_objects(objs)
    doc.select([o.uid for o in doc.objects])
    return doc


@pytest.mark.parametrize("op_name", ["drop-to-ground", "snap-to-grid", "mirror-copy"])
@pytest.mark.parametrize("parented", [False, True])
def test_whole_selection_object_ops_scale_linearly_with_the_object_count(
    monkeypatch: pytest.MonkeyPatch, op_name: str, parented: bool
) -> None:
    """The unfixed ops asked ``world_matrix``/``ancestors``/``descendants``/
    ``index_of`` once per selected object, each of which rebuilds the uid map
    (at 4,000 objects Drop to Ground took 3.1 s). Counted rather than timed: the
    number of those O(objects) doors an op calls must not grow with the
    selection."""
    counter = _Counter(monkeypatch)
    totals = []
    for n in (24, 96):
        doc = _many(n, parented=parented)
        counter.reset()
        params = {"step": 0.5} if op_name == "snap-to-grid" else {}
        ctx = _Ctx()
        assert clay_ops.run(ctx, doc, clay_ops.get(op_name), **params), ctx.msgs
        totals.append(counter.total())
    assert totals[1] <= totals[0] + 6, (
        f"{op_name} (parented={parented}) rebuilt the uid map {totals[0]}x for 24 "
        f"objects and {totals[1]}x for 96: {counter.counts}"
    )


def test_mirror_copy_past_the_object_ceiling_builds_no_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ceiling refusal came from ``add_objects``, after every copy had been
    built (3.8 s at 4,000 objects); it is taken before the first one now."""
    doc = _many(6)
    monkeypatch.setattr(glbimport, "MAX_OBJECTS", 7)  # six objects, three copies asked
    doc.select([o.uid for o in doc.objects[:3]])
    built = {"n": 0}
    real = geom_ops.duplicate

    def counting(*a: Any, **k: Any) -> Any:
        built["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(geom_ops, "duplicate", counting)
    ctx = _Ctx()
    head = doc.history.head
    assert clay_ops.run(ctx, doc, clay_ops.get("mirror-copy")) is False
    assert built["n"] == 0, f"{built['n']} copies were built before the refusal"
    assert doc.history.head == head and len(doc.objects) == 6
    assert any("objects" in m for m in ctx.msgs), ctx.msgs


def test_duplicate_past_the_object_ceiling_builds_no_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = _many(6)
    monkeypatch.setattr(glbimport, "MAX_OBJECTS", 7)
    doc.select([o.uid for o in doc.objects[:3]])
    built = {"n": 0}
    real = geom_ops.duplicate

    def counting(*a: Any, **k: Any) -> Any:
        built["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(geom_ops, "duplicate", counting)
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("duplicate")) is False
    assert built["n"] == 0, f"{built['n']} copies were built before the refusal"
    assert len(doc.objects) == 6
    assert any("objects" in m for m in ctx.msgs), ctx.msgs


# --- clay-51: hidden-only selections are not "ran" ------------------------------


def test_duplicate_and_delete_with_only_hidden_objects_selected_answer_not_ran() -> None:
    for name in ("duplicate", "delete"):
        doc = bd.ClayDoc()
        uid = _box(doc)
        doc.select([uid])
        doc.by_uid(uid).visible = False
        op = clay_ops.get(name)
        head = doc.history.head
        ctx = _Ctx()

        assert not op.enabled(doc), f"{name} is enabled with only a hidden object selected"
        assert clay_ops.reason_for(op, doc), f"{name} is greyed with no reason"
        # The door a caller with no menu (the agent, a key chord) uses.
        func = clay_ops._duplicate if name == "duplicate" else clay_ops._delete
        assert func(ctx, doc) is False, f"{name} answered 'ran' for a press that did nothing"
        assert doc.history.head == head and len(doc.objects) == 1


def test_duplicate_and_delete_still_run_with_a_visible_object_beside_a_hidden_one() -> None:
    doc = bd.ClayDoc()
    shown, hidden = _box(doc, "Shown"), _box(doc, "Hidden")
    doc.by_uid(hidden).visible = False
    doc.select([shown, hidden])
    assert clay_ops.get("duplicate").enabled(doc) and clay_ops.get("delete").enabled(doc)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("duplicate"))
    assert len(doc.objects) == 3


# --- clay-52: a group has no geometry, and Pack Islands / Separate Loose say so -


@pytest.mark.parametrize("op_name", ["pack-uv", "separate-loose"])
def test_pack_islands_and_separate_loose_on_a_group_say_a_group_has_no_geometry(
    op_name: str,
) -> None:
    doc = bd.ClayDoc()
    a, b = _box(doc, "A"), _box(doc, "B")
    group = doc.group([a, b])
    doc.select([group.uid])
    ctx = _Ctx()
    head = doc.history.head

    assert clay_ops.run(ctx, doc, clay_ops.get(op_name)) is False
    assert doc.history.head == head
    assert ctx.msgs == [clay_ops.GROUP_HAS_NO_GEOMETRY], ctx.msgs


# --- clay-53: METRE_PARAMS names only what an op declares -----------------------


def test_every_metre_param_named_in_metre_params_is_declared_by_an_op() -> None:
    """An entry for a param no op has is a dead row that the manual (and a
    comment) then describe as a field: "Extrude-offset" never existed."""
    for kernel, names in clay_ops.METRE_PARAMS.items():
        declared: set[str] = set()
        for op in clay_ops.OPS:
            kernels = {getattr(op.run, "kernel", None), getattr(op.drag, "kernel", None)}
            if kernel in kernels:
                declared.update(p.name for p in op.params)
        missing = [n for n in names if n not in declared]
        assert not missing, f"METRE_PARAMS[{kernel!r}] names {missing}, which no op declares"


# --- clay-54: Repeat Last keeps the repeated operation's name ------------------


def test_repeat_last_leaves_the_repeated_operations_name_on_the_undo_step() -> None:
    doc = bd.ClayDoc()
    uid = _box(doc)
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("inset"), thickness=0.1), ctx.msgs
    assert doc.history.top.label == "Inset Faces"
    doc.set_element_sel(uid, el.ElementSel(faces=[1]))
    head = doc.history.head

    assert clay_ops.run(ctx, doc, clay_ops.get("repeat-last")), ctx.msgs

    assert doc.history.head != head
    assert doc.history.top.label == "Inset Faces", doc.history.top.label
