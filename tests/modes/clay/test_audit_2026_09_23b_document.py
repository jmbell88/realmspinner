"""The 2026-09-23 (second run) audit's clay-01 and clay-02: two doors that
delete an object without re-parenting its children.

``remove_object`` (``document.py``) re-parents a doomed object's children onto
its own parent before popping it, keeping world placement, in the same
compound edit -- the pattern this file pins for the two other doors that
delete an object and used to skip it: ``join_objects``'s per-absorbed-object
loop and ``separate``'s removal of the source. Left unfixed, a child's
``parent`` names a uid the document no longer carries: it jumps in world
space with no undo step, and ``serialize._validate_hierarchy`` refuses to
reload the saved file at all -- the whole document lost at the next open or
crash recovery.
"""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops as clay_ops_geom
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser


def _obj(name: str, mesh: bm.Mesh | None = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=bp.box() if mesh is None else mesh,
        **kwargs,  # type: ignore[arg-type]
    )


def _merged(doc: bd.ClayDoc, target: int, others: list[int]) -> bm.Mesh:
    return clay_ops_geom.join([doc.by_uid(u) for u in [target, *others]], eps=0.0)


# --- clay-01: join_objects orphaned an absorbed object's children ------------


def test_join_reparents_the_absorbed_objects_children_instead_of_orphaning_them() -> None:
    grandparent = _obj("GP", translation=(10.0, 0.0, 0.0))
    a = _obj("A")
    b = _obj("B", translation=(5.0, 0.0, 0.0))
    c = _obj("C", translation=(0.5, 0.0, 0.0))
    doc = bd.ClayDoc([grandparent, a, b, c])
    doc.set_parent(b.uid, grandparent.uid)  # B's own parent, which C must land on.
    doc.set_parent(c.uid, b.uid)  # C is a child of B, which Join is about to absorb.
    world_before = doc.world_matrix(c.uid).copy()

    merged = _merged(doc, a.uid, [b.uid])
    assert doc.join_objects(a.uid, merged, [b.uid]) is True

    # B is gone; C must not still name B's uid as its parent.
    assert [o.uid for o in doc.objects] == [grandparent.uid, a.uid, c.uid]
    child = doc.by_uid(c.uid)
    assert child.parent == grandparent.uid, "C should be re-parented onto B's own parent (GP)"
    assert np.allclose(doc.world_matrix(c.uid), world_before), "world placement must be kept"

    # A save/reload round-trip must not choke on a dangling parent uid.
    reloaded = ser.read_rblk(ser.rblk_bytes(doc))
    assert reloaded.by_uid(c.uid).parent == grandparent.uid

    # Undo restores the original parenting (C back under B, B back in the doc).
    assert doc.undo() is True
    assert [o.uid for o in doc.objects] == [grandparent.uid, a.uid, b.uid, c.uid]
    assert doc.by_uid(c.uid).parent == b.uid
    assert np.allclose(doc.world_matrix(c.uid), world_before)


def test_join_reparents_children_of_every_absorbed_object_not_just_the_first() -> None:
    """Two absorbed objects, each with its own child -- both children must
    land on their own object's parent, not on the target or on each other."""
    a = _obj("A")
    b = _obj("B", translation=(5.0, 0.0, 0.0))
    d = _obj("D", translation=(9.0, 0.0, 0.0))
    b_child = _obj("Bc", translation=(0.5, 0.0, 0.0))
    d_child = _obj("Dc", translation=(0.25, 0.0, 0.0))
    doc = bd.ClayDoc([a, b, d, b_child, d_child])
    doc.set_parent(b_child.uid, b.uid)
    doc.set_parent(d_child.uid, d.uid)
    bc_world = doc.world_matrix(b_child.uid).copy()
    dc_world = doc.world_matrix(d_child.uid).copy()

    merged = _merged(doc, a.uid, [b.uid, d.uid])
    doc.join_objects(a.uid, merged, [b.uid, d.uid])

    assert doc.by_uid(b_child.uid).parent is None  # B was a root; its own parent was None.
    assert doc.by_uid(d_child.uid).parent is None
    assert np.allclose(doc.world_matrix(b_child.uid), bc_world)
    assert np.allclose(doc.world_matrix(d_child.uid), dc_world)

    reloaded = ser.read_rblk(ser.rblk_bytes(doc))
    assert reloaded.by_uid(b_child.uid).parent is None
    assert reloaded.by_uid(d_child.uid).parent is None


# --- clay-02: separate orphaned the source's children -------------------------


def test_separate_reparents_the_sources_children_instead_of_orphaning_them() -> None:
    parent = _obj("Parent")
    source = _obj("Source", translation=(5.0, 0.0, 0.0))
    child = _obj("Child", translation=(0.5, 0.0, 0.0))
    doc = bd.ClayDoc([parent, source, child])
    doc.set_parent(source.uid, parent.uid)
    doc.set_parent(child.uid, source.uid)
    world_before = doc.world_matrix(child.uid).copy()

    pieces = [bp.box(), bp.box()]
    new_objs = doc.separate(source.uid, pieces)

    assert source.uid not in [o.uid for o in doc.objects]
    kept_child = doc.by_uid(child.uid)
    assert kept_child.parent == parent.uid, "Child should land on Source's own parent (Parent)"
    assert np.allclose(doc.world_matrix(child.uid), world_before)

    reloaded = ser.read_rblk(ser.rblk_bytes(doc))
    assert reloaded.by_uid(child.uid).parent == parent.uid

    assert doc.undo() is True
    assert [o.uid for o in doc.objects] == [parent.uid, source.uid, child.uid]
    assert doc.by_uid(child.uid).parent == source.uid
    assert np.allclose(doc.world_matrix(child.uid), world_before)
    for piece in new_objs:
        assert piece.uid not in [o.uid for o in doc.objects]


def test_separate_of_a_rootless_source_reparents_its_child_to_none() -> None:
    source = _obj("Source")
    child = _obj("Child", translation=(1.0, 0.0, 0.0))
    doc = bd.ClayDoc([source, child])
    doc.set_parent(child.uid, source.uid)
    world_before = doc.world_matrix(child.uid).copy()

    pieces = [bp.box(), bp.box()]
    doc.separate(source.uid, pieces)

    assert doc.by_uid(child.uid).parent is None
    assert np.allclose(doc.world_matrix(child.uid), world_before)

    reloaded = ser.read_rblk(ser.rblk_bytes(doc))
    assert reloaded.by_uid(child.uid).parent is None
