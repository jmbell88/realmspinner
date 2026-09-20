"""Tranche 3: scene structure -- parenting, groups and the world-space glTF
export they change the shape of.

A document with no parenting must behave exactly as it did before this
tranche: that is the backstop every test in the first section pins. The rest
exercise the actual feature -- world placement kept across a reparent or a
deleted parent, the cycle refusal, three levels of composition, and the real
glTF hierarchy :func:`~realmspinner.kernels.mesh.document.to_model` now emits.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite, gltf
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import modifiers as mod
from realmspinner.kernels.mesh import ops_boolean
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.elements import OpError


def _obj(name: str, mesh: bm.Mesh | None = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=bp.box() if mesh is None else mesh,
        **kwargs,  # type: ignore[arg-type]
    )


# --- a document with no parenting behaves exactly as before ------------------


def test_roots_of_an_unparented_document_is_every_object() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    assert doc.roots() == [a.uid, b.uid]
    assert doc.children_of(a.uid) == []
    assert doc.ancestors(a.uid) == []
    assert doc.descendants(a.uid) == []


def test_world_matrix_of_a_root_is_its_own_local_trs() -> None:
    from realmspinner.kernels.geom3d import math3d as m3

    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", translation=(1.0, 2.0, 3.0), scale=(2.0, 2.0, 2.0)))
    expected = m3.compose(a.translation, a.rotation, a.scale)
    assert np.allclose(doc.world_matrix(a.uid), expected)


def test_to_model_on_an_unparented_document_is_unchanged() -> None:
    """Pins the exact shape the pre-tranche-3 tests already asserted:
    ``model.roots == list(range(len(nodes)))``, every node a root, no
    ``children``."""
    doc = bd.ClayDoc()
    doc.add_object(_obj("A", translation=(1.0, 2.0, 3.0)))
    doc.add_object(_obj("B"))
    model = bd.to_model(doc)
    assert [n.name for n in model.nodes] == ["A", "B"]
    assert model.roots == [0, 1]
    assert [n.children for n in model.nodes] == [[], []]
    assert [n.mesh for n in model.nodes] == [0, 1]


# --- set_parent ---------------------------------------------------------------


def test_reparenting_keeps_world_placement_nothing_moves() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", translation=(5.0, 0.0, 0.0)))
    b = doc.add_object(_obj("B", translation=(3.0, 0.0, 0.0)))
    world_before = doc.world_matrix(b.uid).copy()

    assert doc.set_parent(b.uid, a.uid) is True
    assert doc.by_uid(b.uid).parent == a.uid
    assert np.allclose(doc.world_matrix(b.uid), world_before)
    # And the local translation was recomputed relative to A -- it is no
    # longer (3, 0, 0), it is (3, 0, 0) - (5, 0, 0) = (-2, 0, 0).
    assert np.allclose(doc.by_uid(b.uid).translation, [-2.0, 0.0, 0.0])


def test_reparent_and_parent_change_are_one_undo_step() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", translation=(5.0, 0.0, 0.0)))
    b = doc.add_object(_obj("B", translation=(3.0, 0.0, 0.0)))
    before_head = doc.history.head

    doc.set_parent(b.uid, a.uid)
    assert doc.history.head != before_head

    assert doc.undo() is True
    assert doc.by_uid(b.uid).parent is None
    assert np.allclose(doc.by_uid(b.uid).translation, [3.0, 0.0, 0.0])
    assert doc.history.head == before_head


def test_set_parent_to_none_clears_the_parent() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B", translation=(1.0, 0.0, 0.0)))
    doc.set_parent(b.uid, a.uid)
    world = doc.world_matrix(b.uid).copy()

    assert doc.set_parent(b.uid, None) is True
    assert doc.by_uid(b.uid).parent is None
    assert np.allclose(doc.world_matrix(b.uid), world)


def test_set_parent_is_a_no_op_when_nothing_changes() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    before = doc.history.head
    assert doc.set_parent(a.uid, None) is False
    assert doc.history.head == before


def test_set_parent_refuses_parenting_to_self() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    with pytest.raises(OpError, match="itself"):
        doc.set_parent(a.uid, a.uid)


def test_set_parent_refuses_a_cycle_through_a_descendant() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    c = doc.add_object(_obj("C"))
    doc.set_parent(b.uid, a.uid)
    doc.set_parent(c.uid, b.uid)  # A -> B -> C

    with pytest.raises(OpError, match="descendants"):
        doc.set_parent(a.uid, c.uid)
    # Refused before anything was pushed.
    assert doc.by_uid(a.uid).parent is None


def test_set_parent_refuses_an_unknown_parent_uid() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    with pytest.raises(KeyError):
        doc.set_parent(a.uid, 999999)


def test_set_parent_without_keep_world_leaves_local_trs_untouched() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", translation=(5.0, 0.0, 0.0)))
    b = doc.add_object(_obj("B", translation=(3.0, 0.0, 0.0)))
    doc.set_parent(b.uid, a.uid, keep_world=False)
    assert np.allclose(doc.by_uid(b.uid).translation, [3.0, 0.0, 0.0])
    # World placement moved, because the local numbers were kept as-is.
    assert np.allclose(doc.world_matrix(b.uid)[:3, 3], [8.0, 0.0, 0.0])


# --- remove_object: deleting a parent keeps grandchildren in place -----------


def test_deleting_a_parent_keeps_grandchildren_in_place() -> None:
    doc = bd.ClayDoc()
    grandparent = doc.add_object(_obj("GP", translation=(10.0, 0.0, 0.0)))
    parent = doc.add_object(_obj("P", translation=(1.0, 0.0, 0.0)))
    child = doc.add_object(_obj("C", translation=(0.5, 0.0, 0.0)))
    doc.set_parent(parent.uid, grandparent.uid)
    doc.set_parent(child.uid, parent.uid)
    world_before = doc.world_matrix(child.uid).copy()

    assert doc.remove_object(parent.uid) is True
    assert doc.by_uid(child.uid).parent == grandparent.uid
    assert np.allclose(doc.world_matrix(child.uid), world_before)


def test_deleting_a_root_re_parents_its_children_to_none() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B", translation=(1.0, 0.0, 0.0)))
    doc.set_parent(b.uid, a.uid)
    world_before = doc.world_matrix(b.uid).copy()

    doc.remove_object(a.uid)
    assert doc.by_uid(b.uid).parent is None
    assert np.allclose(doc.world_matrix(b.uid), world_before)


def test_remove_object_with_reparenting_is_one_undo_step() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", translation=(10.0, 0.0, 0.0)))
    b = doc.add_object(_obj("B", translation=(1.0, 0.0, 0.0)))
    doc.set_parent(b.uid, a.uid)
    before_head = doc.history.head

    world_before_remove = doc.world_matrix(b.uid).copy()
    doc.remove_object(a.uid)
    assert doc.undo() is True
    assert [o.uid for o in doc.objects] == [a.uid, b.uid]
    assert doc.by_uid(b.uid).parent == a.uid
    assert np.allclose(doc.world_matrix(b.uid), world_before_remove)
    assert doc.history.head == before_head


# --- world_matrix composes through the whole ancestor chain ------------------


def test_world_matrix_composes_three_levels() -> None:
    doc = bd.ClayDoc()
    gp = doc.add_object(_obj("GP", translation=(1.0, 0.0, 0.0)))
    p = doc.add_object(_obj("P", translation=(0.0, 2.0, 0.0)))
    c = doc.add_object(_obj("C", translation=(0.0, 0.0, 3.0)))
    doc.set_parent(p.uid, gp.uid, keep_world=False)
    doc.set_parent(c.uid, p.uid, keep_world=False)

    assert doc.ancestors(c.uid) == [p.uid, gp.uid]
    assert np.allclose(doc.world_matrix(c.uid)[:3, 3], [1.0, 2.0, 3.0])


def test_descendants_are_document_order_depth_first() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    c = doc.add_object(_obj("C"))
    d = doc.add_object(_obj("D"))
    doc.set_parent(b.uid, a.uid, keep_world=False)
    doc.set_parent(c.uid, a.uid, keep_world=False)
    doc.set_parent(d.uid, b.uid, keep_world=False)
    assert doc.descendants(a.uid) == [b.uid, d.uid, c.uid]


def test_local_from_world_round_trips_through_a_parent() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", translation=(5.0, 0.0, 0.0), scale=(2.0, 2.0, 2.0)))
    b = doc.add_object(_obj("B"))
    doc.set_parent(b.uid, a.uid, keep_world=False)

    wanted_world = doc.world_matrix(a.uid).copy()
    wanted_world[:3, 3] = [9.0, 0.0, 0.0]
    t, r, s = doc.local_from_world(b.uid, wanted_world)
    doc.set_transform(b.uid, translation=t, rotation=r, scale=s)
    assert np.allclose(doc.world_matrix(b.uid)[:3, 3], [9.0, 0.0, 0.0])


# --- group() -------------------------------------------------------------


def test_group_makes_a_mesh_less_empty_at_the_bounds_centre() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", translation=(0.0, 0.0, 0.0)))  # box: -0.5..0.5
    b = doc.add_object(_obj("B", translation=(2.0, 0.0, 0.0)))  # box: 1.5..2.5

    empty = doc.group([a.uid, b.uid])
    assert bm.face_count(empty.mesh) == 0
    assert np.allclose(empty.translation, [1.0, 0.0, 0.0])
    assert doc.children_of(empty.uid) == [a.uid, b.uid]
    assert doc.by_uid(a.uid).parent == empty.uid
    assert doc.by_uid(b.uid).parent == empty.uid


def test_group_is_one_undo_step() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B", translation=(2.0, 0.0, 0.0)))
    before_head = doc.history.head

    doc.group([a.uid, b.uid])
    assert doc.undo() is True
    assert len(doc.objects) == 2
    assert doc.by_uid(a.uid).parent is None
    assert doc.by_uid(b.uid).parent is None
    assert doc.history.head == before_head


def test_group_refuses_an_empty_selection() -> None:
    doc = bd.ClayDoc()
    with pytest.raises(OpError, match="Select at least one"):
        doc.group([])


def test_group_places_the_empty_at_the_origin_for_empty_geometry() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", mesh=bd._empty_mesh(), translation=(1.0, 1.0, 1.0)))
    empty = doc.group([a.uid])
    assert np.allclose(empty.translation, [0.0, 0.0, 0.0])


# --- to_model hierarchy and hidden objects ------------------------------------


def test_to_model_wires_children_from_parent() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    doc.set_parent(b.uid, a.uid, keep_world=False)

    model = bd.to_model(doc)
    assert model.roots == [0]
    assert model.nodes[0].children == [1]


def test_a_hidden_parent_still_exports_its_visible_child_meshless() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", visible=False, translation=(5.0, 0.0, 0.0)))
    b = doc.add_object(_obj("B", translation=(1.0, 0.0, 0.0)))
    doc.set_parent(b.uid, a.uid, keep_world=False)

    model = bd.to_model(doc)
    assert len(model.nodes) == 2
    assert model.nodes[0].name == "A"
    assert model.nodes[0].mesh is None  # carries the frame, draws nothing
    assert model.nodes[1].mesh is not None
    assert model.roots == [0]
    assert model.nodes[0].children == [1]

    model.update_world()
    # B's world position is A's translation + B's local translation, exactly
    # as it would be if A were visible -- hiding a parent never moves its
    # children's frame.
    assert np.allclose(model.nodes[1].world[:3, 3], [6.0, 0.0, 0.0])


def test_a_hidden_object_with_no_visible_descendant_is_omitted_entirely() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", visible=False))
    b = doc.add_object(_obj("B", visible=False))
    doc.set_parent(b.uid, a.uid, keep_world=False)

    model = bd.to_model(doc)
    assert model.nodes == []
    assert model.roots == []


def test_export_hierarchy_round_trips_through_glbwrite_and_gltf() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", translation=(3.0, 0.0, 0.0)))
    b = doc.add_object(_obj("B", translation=(1.0, 0.0, 0.0)))
    doc.set_parent(b.uid, a.uid, keep_world=False)

    glb = glbwrite.write_glb(bd.to_model(doc))
    from realmspinner.kernels.geom3d import glbio

    json_doc, _bin = glbio.read_glb(glb)
    nodes = json_doc["nodes"]
    a_index = next(i for i, n in enumerate(nodes) if n.get("name") == "A")
    b_index = next(i for i, n in enumerate(nodes) if n.get("name") == "B")
    assert nodes[a_index]["children"] == [b_index]
    assert a_index in json_doc["scenes"][json_doc["scene"]]["nodes"]
    assert b_index not in json_doc["scenes"][json_doc["scene"]]["nodes"]

    # And the world position glTF's own node-graph composition gives is the
    # sum of the two local translations, matching Model.update_world.
    model = gltf.Model(
        [
            gltf.Node(
                name=n.get("name", ""),
                translation=np.array(n.get("translation", (0, 0, 0)), dtype="f8"),
                rotation=np.array(n.get("rotation", (0, 0, 0, 1)), dtype="f8"),
                scale=np.array(n.get("scale", (1, 1, 1)), dtype="f8"),
                children=list(n.get("children", [])),
            )
            for n in nodes
        ],
        list(json_doc["scenes"][json_doc["scene"]]["nodes"]),
        [[] for _ in nodes],
        [],
    )
    assert np.allclose(model.nodes[b_index].world[:3, 3], [4.0, 0.0, 0.0])


# --- set_origin ----------------------------------------------------------


def _world_positions(doc: bd.ClayDoc, uid: int) -> np.ndarray:
    matrix = doc.world_matrix(uid)
    pts = np.asarray(doc.evaluated(uid).positions, dtype="f8")
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
    return (matrix @ homogeneous.T).T[:, :3]


def test_set_origin_leaves_geometry_and_children_in_place() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", translation=(0.0, 0.0, 0.0)))
    b = doc.add_object(_obj("B", translation=(2.0, 0.0, 0.0)))
    doc.set_parent(b.uid, a.uid)
    positions_before = _world_positions(doc, a.uid)
    child_world_before = doc.world_matrix(b.uid).copy()

    assert doc.set_origin(a.uid, (0.5, 0.0, 0.0)) is True
    assert np.allclose(doc.by_uid(a.uid).translation, [0.5, 0.0, 0.0])
    # The mesh moved by the inverse delta, so its *world* positions --
    # and every child's -- are exactly unchanged.
    assert np.allclose(_world_positions(doc, a.uid), positions_before)
    assert np.allclose(doc.world_matrix(b.uid), child_world_before)


def test_set_origin_freezes_the_generator() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    a.generator = "box"
    a.params = {"size": 1.0}
    doc.set_origin(a.uid, (0.25, 0.0, 0.0))
    assert doc.by_uid(a.uid).generator is None


def test_set_origin_is_one_undo_step_and_undo_restores_everything() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B", translation=(1.0, 0.0, 0.0)))
    doc.set_parent(b.uid, a.uid)
    mesh_before = doc.by_uid(a.uid).mesh
    translation_before = doc.by_uid(a.uid).translation.copy()
    child_translation_before = doc.by_uid(b.uid).translation.copy()
    before_head = doc.history.head

    doc.set_origin(a.uid, (0.5, 0.0, 0.0))
    assert doc.undo() is True
    assert doc.by_uid(a.uid).mesh is mesh_before
    assert np.allclose(doc.by_uid(a.uid).translation, translation_before)
    assert np.allclose(doc.by_uid(b.uid).translation, child_translation_before)
    assert doc.history.head == before_head


def test_set_origin_is_a_no_op_at_the_current_origin() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    before = doc.history.head
    assert doc.set_origin(a.uid, (0.0, 0.0, 0.0)) is False
    assert doc.history.head == before


# --- boolean modifier world-matrix cache (a pure ancestor move) --------------

pytest.importorskip("manifold3d")


def test_a_pure_ancestor_move_invalidates_a_boolean_modifiers_cache() -> None:
    """The regression the tranche 3 spec names by hand: the boolean cache
    used to key on the two objects' own *local* TRS, so moving an ancestor
    -- neither object's own local TRS changes at all -- served a stale
    result."""
    doc = bd.ClayDoc()
    cutter = doc.add_object(_obj("Cutter", translation=(0.3, 0.0, 0.0)))
    target = doc.add_object(_obj("Target"))
    parent = doc.add_object(_obj("Parent"))
    doc.set_parent(target.uid, parent.uid, keep_world=False)

    stack = (mod.make("boolean", {"target": cutter.uid, "operation": "difference"}, id=1),)
    doc.set_modifiers(target.uid, stack)
    first = doc.evaluated(target.uid)

    # Move the parent -- the target's own local TRS is untouched.
    doc.set_transform(parent.uid, translation=(5.0, 0.0, 0.0))
    second = doc.evaluated(target.uid)
    assert second is not first

    # Correctness, not only cache invalidation: recompute the same boolean
    # directly, from the *current* world matrices, and check the modifier's
    # cached-or-recomputed result agrees. A stale cache keyed on local TRS
    # would still report ``second is not first`` here only by coincidence of
    # some *other* change; this is the actual number the spec's own
    # "or a pure ancestor move serves a stale boolean" warns about.
    from dataclasses import replace


    expected = ops_boolean.boolean(
        [replace(doc.by_uid(target.uid)), replace(doc.by_uid(cutter.uid))],
        "difference",
        world=[doc.world_matrix(target.uid), doc.world_matrix(cutter.uid)],
    )
    assert np.allclose(
        np.sort(second.positions, axis=0), np.sort(expected.positions, axis=0), atol=1e-4
    )


# --- .rblk v3: parent/locked/tags -------------------------------------------


def test_rblk_round_trips_parent_locked_and_tags() -> None:
    from realmspinner.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B", translation=(1.0, 0.0, 0.0)))
    doc.set_parent(b.uid, a.uid)
    doc.set_props(b.uid, locked=True, tags=("Prop", "hero"))

    out = ser.read_rblk(ser.rblk_bytes(doc))
    restored = out.by_uid(b.uid)
    assert restored.parent == a.uid
    assert restored.locked is True
    assert restored.tags == ("hero", "prop")  # sorted, lower-cased
    assert np.allclose(doc.world_matrix(b.uid), out.world_matrix(b.uid))


def test_rblk_round_trips_a_colliders_parent_alongside_its_role() -> None:
    """Tranche 7: ``ClayDoc.add_collider`` parents the collider onto its
    source (see that method's own docstring) -- this is the one point where
    tranche 3's own hierarchy and tranche 7's role/kind fields have to agree
    about the same object at once, which neither ``test_rblk_round_trips_
    parent_locked_and_tags`` above nor ``test_collider_objects.py``'s own
    role/kind-focused round trip exercises together."""
    from realmspinner.kernels.mesh import colliders as cl
    from realmspinner.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    collider = doc.add_collider(a.uid, cl.fit_box(a.mesh))

    out = ser.read_rblk(ser.rblk_bytes(doc))
    restored = out.by_uid(collider.uid)
    assert restored.parent == a.uid
    assert restored.role == "collider"
    assert restored.collider_kind == "box"
    assert np.allclose(doc.world_matrix(collider.uid), out.world_matrix(collider.uid))


def test_an_object_at_every_default_writes_no_hierarchy_keys() -> None:
    import json

    from realmspinner.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    doc.add_object(_obj("A"))
    entry = json.loads(ser.scene_json(doc))["objects"][0]
    assert "parent" not in entry
    assert "locked" not in entry
    assert "tags" not in entry


def test_a_document_with_parenting_is_still_byte_identical_when_repeated() -> None:
    from realmspinner.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    doc.set_parent(b.uid, a.uid)
    doc.set_props(b.uid, locked=True, tags=("hero",))
    assert ser.rblk_bytes(doc) == ser.rblk_bytes(doc)


def test_rblk_refuses_a_parent_naming_an_absent_uid() -> None:
    import json
    import zipfile
    from io import BytesIO

    from realmspinner.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    doc.add_object(_obj("A"))
    data = ser.rblk_bytes(doc)
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            if name == ser.SCENE:
                scene = json.loads(src.read(name))
                scene["objects"][0]["parent"] = 999999
                dst.writestr(name, json.dumps(scene))
            else:
                dst.writestr(name, src.read(name))

    with pytest.raises(ValueError, match="does not carry"):
        ser.read_rblk(out.getvalue())


def test_rblk_refuses_a_cycle_in_the_file() -> None:
    import json
    import zipfile
    from io import BytesIO

    from realmspinner.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    doc.set_parent(b.uid, a.uid, keep_world=False)
    data = ser.rblk_bytes(doc)
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            if name == ser.SCENE:
                scene = json.loads(src.read(name))
                # A already names B as absent (A is a root); make B name A's
                # parent too -- a naive per-entry check would miss this,
                # since each entry's own ``parent`` is individually valid.
                for entry in scene["objects"]:
                    if entry["uid"] == a.uid:
                        entry["parent"] = b.uid
                dst.writestr(name, json.dumps(scene))
            else:
                dst.writestr(name, src.read(name))

    with pytest.raises(ValueError, match="cycle"):
        ser.read_rblk(out.getvalue())
