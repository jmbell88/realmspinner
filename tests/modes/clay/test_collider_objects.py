"""Tranche 7 (integration half): a collider is an ordinary object with a role.

``colliders.py``'s own pure-kernel fits (box/sphere/capsule/convex/compound)
are pinned in ``test_colliders.py``; this file is the document-level door
that turns one of those fits into a live ``Obj`` -- ``ClayDoc.add_collider``
-- and the ``.wblk`` v3 round trip for ``Obj.role``/``Obj.collider_kind``.
See ``document.py``'s own module docstring (tranche 7 paragraph) and
``add_collider``'s own docstring for the design this holds it to: parenting
places it (a collider's mesh is already expressed in the source's own local
frame, so the child's own local TRS is the identity, not a reparent that
tries to preserve some prior world placement), the role/kind pair is the one
thing validated, and adding a collider is not a locking door.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from warlock.kernels.mesh import colliders as cl
from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import mesh as bm
from warlock.kernels.mesh import primitives as bp
from warlock.kernels.mesh.elements import OpError


def _obj(name: str, mesh: bm.Mesh | None = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=bp.box() if mesh is None else mesh,
        **kwargs,  # type: ignore[arg-type]
    )


# --- add_collider: placement and identity -----------------------------------


def test_add_collider_parents_onto_the_source_with_visible_role_and_kind() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("Crate"))
    new = doc.add_collider(a.uid, cl.fit_box(a.mesh))

    assert new.parent == a.uid
    assert new.role == "collider"
    assert new.collider_kind == "box"
    assert new.visible is True
    assert new.uid in {o.uid for o in doc.objects}


def test_add_collider_lands_at_the_sources_own_world_place() -> None:
    """``Collider.mesh`` is already expressed in the source's own local
    frame (``colliders.py``'s own module docstring), so the correct child
    placement is a local identity under the source, not a world-preserving
    reparent -- this is the claim that distinguishes ``add_collider`` from a
    plain ``add_object`` + ``set_parent(keep_world=True)``."""
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("Crate", translation=(3.0, 1.0, -2.0)))
    new = doc.add_collider(a.uid, cl.fit_box(a.mesh))
    assert np.allclose(doc.world_matrix(new.uid), doc.world_matrix(a.uid))


def test_add_collider_names_it_source_plus_kind_label() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("Crate"))
    new = doc.add_collider(a.uid, cl.fit_sphere(a.mesh))
    assert new.name == "Crate Sphere"


def test_add_collider_disambiguates_a_second_collider_of_the_same_kind() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("Crate"))
    first = doc.add_collider(a.uid, cl.fit_box(a.mesh))
    second = doc.add_collider(a.uid, cl.fit_box(a.mesh))
    assert first.name == "Crate Box"
    assert second.name != first.name
    assert {o.name for o in doc.objects} == {"Crate", "Crate Box", second.name}


# --- validation ---------------------------------------------------------


def test_add_collider_refuses_an_unknown_kind_and_pushes_nothing() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("Crate"))
    before = len(doc.objects)
    head = doc.history.head
    fake = cl.Collider(kind="not-a-real-kind", mesh=a.mesh, params={})
    with pytest.raises(OpError, match="not-a-real-kind"):
        doc.add_collider(a.uid, fake)
    assert len(doc.objects) == before
    assert doc.history.head == head


def test_add_collider_refuses_an_unknown_source_uid() -> None:
    doc = bd.ClayDoc()
    with pytest.raises(KeyError):
        doc.add_collider(999999, cl.Collider(kind="box", mesh=bp.box(), params={}))


# --- not a locking door ---------------------------------------------------


def test_add_collider_works_on_a_locked_source() -> None:
    """Adding a collider changes nothing about the source object itself --
    its mesh, transform and every other field are untouched -- the same
    "attaching a new child" exemption ``set_parent``/``group`` already have
    (see ``add_collider``'s own docstring)."""
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("Crate", locked=True))
    new = doc.add_collider(a.uid, cl.fit_box(a.mesh))
    assert new.role == "collider"
    assert doc.by_uid(a.uid).locked is True  # the source itself is untouched


# --- one undo step -----------------------------------------------------


def test_add_collider_is_one_undo_step() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("Crate"))
    head = doc.history.head
    new = doc.add_collider(a.uid, cl.fit_box(a.mesh))
    assert doc.history.head == head + 1
    assert doc.undo()
    assert new.uid not in {o.uid for o in doc.objects}
    assert doc.history.head == head


# --- .wblk v3: role / collider_kind -----------------------------------------


def test_wblk_round_trips_role_and_collider_kind() -> None:
    from warlock.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    a = doc.add_object(_obj("Crate"))
    new = doc.add_collider(a.uid, cl.fit_capsule(a.mesh))

    out = ser.read_wblk(ser.wblk_bytes(doc))
    restored = out.by_uid(new.uid)
    assert restored.role == "collider"
    assert restored.collider_kind == "capsule"
    assert restored.parent == a.uid


def test_an_ordinary_mesh_object_writes_no_role_or_kind_keys() -> None:
    from warlock.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    doc.add_object(_obj("A"))
    entry = json.loads(ser.scene_json(doc))["objects"][0]
    assert "role" not in entry
    assert "collider_kind" not in entry


def test_a_document_with_a_collider_is_still_byte_identical_when_repeated() -> None:
    from warlock.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    a = doc.add_object(_obj("Crate"))
    doc.add_collider(a.uid, cl.fit_box(a.mesh))
    assert ser.wblk_bytes(doc) == ser.wblk_bytes(doc)


def _hand_edit_scene(data: bytes, mutate) -> bytes:
    import zipfile
    from io import BytesIO

    from warlock.kernels.mesh import serialize as ser

    out = BytesIO()
    with zipfile.ZipFile(BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            if name == ser.SCENE:
                scene = json.loads(src.read(name))
                mutate(scene)
                dst.writestr(name, json.dumps(scene))
            else:
                dst.writestr(name, src.read(name))
    return out.getvalue()


def test_wblk_refuses_an_unknown_role() -> None:
    from warlock.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    doc.add_object(_obj("A"))
    data = ser.wblk_bytes(doc)
    bad = _hand_edit_scene(data, lambda s: s["objects"][0].__setitem__("role", "giraffe"))
    with pytest.raises(ValueError, match="role"):
        ser.read_wblk(bad)


def test_wblk_refuses_a_collider_kind_that_names_nothing() -> None:
    from warlock.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    doc.add_object(_obj("A", role="collider", collider_kind="box"))
    data = ser.wblk_bytes(doc)
    bad = _hand_edit_scene(
        data, lambda s: s["objects"][0].__setitem__("collider_kind", "not-a-kind")
    )
    with pytest.raises(ValueError, match="collider kind"):
        ser.read_wblk(bad)


def test_wblk_refuses_a_collider_kind_on_a_non_collider_object() -> None:
    from warlock.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    doc.add_object(_obj("A"))
    data = ser.wblk_bytes(doc)
    bad = _hand_edit_scene(data, lambda s: s["objects"][0].__setitem__("collider_kind", "box"))
    with pytest.raises(ValueError, match="collider_kind"):
        ser.read_wblk(bad)
