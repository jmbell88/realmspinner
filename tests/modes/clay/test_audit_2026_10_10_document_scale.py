"""Regression tests for the document half of the 2026-10-10 Clay audit's
clay-23: ``ClayDoc.group``, ``set_parent`` and ``set_origin`` rebuilt the whole
uid map (and ``set_parent`` the whole parent -> children map) several times per
call, so a whole-selection op over N objects did N full scans. At 4,000 objects
Group Selected took 5.8 s, Parent to Last 3.0 s, Origin to World 2.6 s and
Origin to Bounds 1.3 s on the frame thread.

Counted rather than timed, as the ops-side test is: a wall-clock bound flakes
under a loaded machine and the defect is the shape. Each test runs the real
``ops`` entry point at 24 and at 96 objects and requires the number of
O(objects) rebuilds not to grow with the selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops


class _Ctx:
    def __init__(self) -> None:
        self.msgs: list[str] = []
        self.clay_view: Any = None
        self.state: Any = None

    def toast(self, message: str, level: str = "info") -> None:
        self.msgs.append(message)


class _Counter:
    """Counts the doors that walk the whole document, on the document class:
    the uid-map rebuild, the descendants walk (its own parent -> children map)
    and the position index the lookups share. ``children_of`` is not counted:
    it is the one bare scan ``set_origin`` still makes per object."""

    NAMES = ("_uid_map", "descendants", "_index_positions")

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.counts = dict.fromkeys(self.NAMES, 0)
        for name in self.NAMES:
            original = getattr(bd.ClayDoc, name, None)
            if original is None:  # the unfixed class has no index to count
                continue

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
        for i, child in enumerate(objs[n // 2 :]):
            child.parent = objs[i].uid
    doc.add_objects(objs)
    doc.select([o.uid for o in doc.objects])
    return doc


def _assert_flat(monkeypatch: pytest.MonkeyPatch, op_name: str, parented: bool) -> None:
    counter = _Counter(monkeypatch)
    totals = []
    for n in (24, 96):
        doc = _many(n, parented=parented)
        counter.reset()
        ctx = _Ctx()
        assert clay_ops.run(ctx, doc, clay_ops.get(op_name)), ctx.msgs
        totals.append(counter.total())
    assert totals[1] <= totals[0] + 6, (
        f"{op_name} (parented={parented}) walked the whole document {totals[0]}x for 24 "
        f"objects and {totals[1]}x for 96: {counter.counts}"
    )


@pytest.mark.parametrize("parented", [False, True])
def test_group_selected_walks_the_document_a_constant_number_of_times(
    monkeypatch: pytest.MonkeyPatch, parented: bool
) -> None:
    _assert_flat(monkeypatch, "group", parented)


@pytest.mark.parametrize("parented", [False, True])
def test_parent_to_last_walks_the_document_a_constant_number_of_times(
    monkeypatch: pytest.MonkeyPatch, parented: bool
) -> None:
    _assert_flat(monkeypatch, "parent-to-last", parented)


@pytest.mark.parametrize("parented", [False, True])
def test_duplicate_walks_the_document_a_constant_number_of_times(
    monkeypatch: pytest.MonkeyPatch, parented: bool
) -> None:
    """The 2026-10-11 audit's clay-21: ``selection.with_descendants`` called
    ``doc.descendants`` once per selected uid, each rebuilding the whole
    parent -> children map."""
    _assert_flat(monkeypatch, "duplicate", parented)


def test_with_descendants_walks_the_document_a_constant_number_of_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from realmspinner.kernels.mesh import selection

    counter = _Counter(monkeypatch)
    totals = []
    for n in (24, 96):
        doc = _many(n, parented=True)
        counter.reset()
        selection.with_descendants(doc, [o.uid for o in doc.objects])
        totals.append(counter.total())
    assert totals[1] <= totals[0] + 1, (
        f"with_descendants walked the whole document {totals[0]}x for 24 objects "
        f"and {totals[1]}x for 96: {counter.counts}"
    )


@pytest.mark.parametrize("op_name", ["origin-to-world", "origin-to-bounds"])
@pytest.mark.parametrize("parented", [False, True])
def test_origin_ops_walk_the_document_a_constant_number_of_times(
    monkeypatch: pytest.MonkeyPatch, op_name: str, parented: bool
) -> None:
    _assert_flat(monkeypatch, op_name, parented)


# --- the lookups behind that must never answer from a stale position ----------


def test_a_lookup_follows_an_object_that_moved_in_the_list() -> None:
    doc = _many(6)
    uids = [o.uid for o in doc.objects]
    assert doc.by_uid(uids[4]).name == "B4"  # warms whatever the lookups remember
    doc.objects.insert(0, doc.objects.pop(4))
    assert doc.index_of(uids[4]) == 0
    assert doc.by_uid(uids[4]).name == "B4"
    assert doc.by_uid(uids[0]).name == "B0"
    del doc.objects[0]
    with pytest.raises(KeyError):
        doc.by_uid(uids[4])
    assert doc.by_uid(uids[5]).name == "B5"


def test_a_world_matrix_follows_a_parent_that_was_deleted_and_an_undo_that_restored_it() -> None:
    doc = bd.ClayDoc()
    top = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="top", mesh=bp.box(), translation=np.array([1.0, 0, 0]))
    )
    leaf = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="leaf",
            mesh=bp.box(),
            parent=top.uid,
            translation=np.array([0, 2.0, 0]),
        )
    )
    assert doc.world_matrix(leaf.uid)[:3, 3].tolist() == [1.0, 2.0, 0.0]
    doc.remove_object(top.uid)
    assert doc.world_matrix(leaf.uid)[:3, 3].tolist() == [1.0, 2.0, 0.0]  # re-parented, kept
    assert doc.by_uid(leaf.uid).parent is None
    assert doc.undo()
    assert doc.by_uid(leaf.uid).parent == top.uid
    assert doc.world_matrix(leaf.uid)[:3, 3].tolist() == [1.0, 2.0, 0.0]
    assert doc.ancestors(leaf.uid) == [top.uid]


def test_set_parent_still_refuses_a_cycle_without_walking_the_descendants() -> None:
    doc = _many(5)
    a, b, c = (doc.objects[i].uid for i in range(3))
    doc.set_parent(b, a)
    doc.set_parent(c, b)
    steps = len(doc.history)
    for uid, parent in ((a, c), (a, b), (b, b)):
        with pytest.raises(bd.el.OpError):
            doc.set_parent(uid, parent)
    assert len(doc.history) == steps
    with pytest.raises(KeyError):
        doc.set_parent(a, 10**9)
