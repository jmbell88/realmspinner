"""``mason/scene.py``: the one resolver -- what a node adds up to once its
whole ancestry has had its say.

Every test name is a claim, written to fail against the version of
``scene.py`` it would have caught: a transform that forgets its parent, a
hidden ancestor whose flag leaks onto the leaf it hid, a lock the resolver
enforces instead of merely reporting, a prefab instance whose second copy
reuses the first's world, an ``owner`` that names the template leaf instead
of the instance, a hand-edited prefab cycle that never returns.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.mason import document as doc
from warlock.studio.mason import nodes as nd
from warlock.studio.mason import refs, scene
from warlock.studio.viewer import math3d as m3
from warlock.studio.viewer.gltf import Material


def _box_ref(**params) -> refs.PrimitiveRef:
    return refs.primitive_ref("box", params)


def _collect(d: doc.MasonDoc, **kwargs) -> list[scene.Placed]:
    """Every node ``scene.walk`` crosses, groups and hidden nodes included --
    what ``resolve`` itself is built on before it filters either away."""
    out: list[scene.Placed] = []

    def visit(node, path, owner, world, visible, locked, static, ref, material, prefab, dangling):
        out.append(
            scene.Placed(
                node, path, owner, world, visible, locked, static, ref, material, prefab, dangling
            )
        )

    scene.walk(d, visit, **kwargs)
    return out


class _FakeSource:
    """A minimal ``GeometrySource``: a fixed table of boxes, keyed by ``ref_key``."""

    def __init__(self, boxes: dict) -> None:
        self._boxes = boxes
        self.rev = 0

    def primitives(self, ref):  # pragma: no cover - not exercised by these tests
        return []

    def box(self, ref):
        return self._boxes.get(refs.ref_key(ref))


# --- the five combination rules --------------------------------------------


def test_transform_composes_parent_world_with_child_local():
    d = doc.MasonDoc()
    parent = nd.GroupNode(uid=nd.new_uid(), translation=m3.vec3(10.0, 0.0, 0.0))
    child = nd.MeshNode(
        uid=nd.new_uid(), ref=_box_ref(), translation=m3.vec3(1.0, 0.0, 0.0)
    )
    parent.children.append(child)
    d.add_node(parent)

    placed = scene.resolve(d)[0]
    assert placed.world[:3, 3] == pytest.approx([11.0, 0.0, 0.0])


def test_a_hidden_ancestor_hides_a_visible_leaf_and_unhiding_restores_it_exactly():
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), visible=True)
    leaf = nd.MeshNode(uid=nd.new_uid(), ref=_box_ref(), visible=True)
    group.children.append(leaf)
    d.add_node(group)

    assert [p.node.uid for p in scene.resolve(d)] == [leaf.uid]

    d.set_props(group.uid, visible=False)
    assert scene.resolve(d) == []
    assert leaf.visible is True  # the leaf's own flag never moved

    d.set_props(group.uid, visible=True)
    assert [p.node.uid for p in scene.resolve(d)] == [leaf.uid]


def test_include_hidden_reaches_ancestors_too_not_only_leaves():
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), visible=False)
    leaf = nd.MeshNode(uid=nd.new_uid(), ref=_box_ref(), visible=True)
    group.children.append(leaf)
    d.add_node(group)

    assert scene.resolve(d) == []  # the cheap default: not even reported
    placed = scene.resolve(d, include_hidden=True)
    assert [p.node.uid for p in placed] == [leaf.uid]
    # The leaf's own flag says "visible", but what it inherited says
    # otherwise -- combined visibility is what include_hidden lets through.
    assert placed[0].visible is False


def test_locked_ors_up_the_tree_and_is_reported_never_enforced():
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), locked=True)
    leaf = nd.MeshNode(uid=nd.new_uid(), ref=_box_ref(), locked=False)
    group.children.append(leaf)
    d.add_node(group)

    placed = scene.resolve(d)[0]
    assert placed.locked is True
    assert leaf.locked is False  # the leaf's own flag never moved either
    # Reported, not enforced: the resolver says nothing about what the
    # document itself will still let happen.
    assert d.set_transform(leaf.uid, translation=[1.0, 0.0, 0.0]) is True


def test_static_ors_up_the_tree():
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), static=True)
    leaf = nd.MeshNode(uid=nd.new_uid(), ref=_box_ref(), static=False)
    group.children.append(leaf)
    d.add_node(group)

    assert scene.resolve(d)[0].static is True


def test_material_override_wins_nearest_ancestor():
    ancestor_mat = Material(name="ancestor")
    leaf_mat = Material(name="own")
    ancestor = nd.MeshNode(uid=nd.new_uid(), material=ancestor_mat)
    plain_leaf = nd.MeshNode(uid=nd.new_uid(), ref=_box_ref())
    overridden_leaf = nd.MeshNode(uid=nd.new_uid(), ref=_box_ref(), material=leaf_mat)
    ancestor.children.extend([plain_leaf, overridden_leaf])
    d = doc.MasonDoc()
    d.add_node(ancestor)

    by_uid = {p.node.uid: p for p in scene.resolve(d)}
    assert by_uid[ancestor.uid].material is ancestor_mat
    assert by_uid[plain_leaf.uid].material is ancestor_mat
    assert by_uid[overridden_leaf.uid].material is leaf_mat


# --- groups -------------------------------------------------------------


def test_groups_are_absent_from_resolve_but_present_in_a_bare_walk():
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), name="G")
    leaf = nd.MeshNode(uid=nd.new_uid(), ref=_box_ref())
    group.children.append(leaf)
    d.add_node(group)

    assert [p.node.uid for p in scene.resolve(d)] == [leaf.uid]
    assert {p.node.uid for p in _collect(d)} == {group.uid, leaf.uid}


# --- prefabs and instancing -------------------------------------------------


def _template() -> tuple[nd.Node, nd.Node, nd.Node]:
    """A two-leaf template: a group holding two mesh leaves."""
    root = nd.GroupNode(uid=nd.new_uid(), name="template_root")
    leaf_a = nd.MeshNode(uid=nd.new_uid(), name="leafA", ref=_box_ref(size=(1.0, 1.0, 1.0)))
    leaf_b = nd.MeshNode(uid=nd.new_uid(), name="leafB", ref=_box_ref(size=(2.0, 2.0, 2.0)))
    root.children.extend([leaf_a, leaf_b])
    return root, leaf_a, leaf_b


def test_instance_expansion_yields_one_placed_per_template_leaf_per_instance_with_distinct_paths():
    d = doc.MasonDoc()
    template, leaf_a, leaf_b = _template()
    d.prefabs["Prop"] = template
    inst1 = nd.PrefabNode(uid=nd.new_uid(), template="Prop", translation=m3.vec3(1.0, 0.0, 0.0))
    inst2 = nd.PrefabNode(uid=nd.new_uid(), template="Prop", translation=m3.vec3(5.0, 0.0, 0.0))
    d.add_nodes([inst1, inst2])

    placed = scene.resolve(d)
    assert len(placed) == 4  # two leaves x two instances
    paths = {p.path for p in placed}
    assert len(paths) == 4  # every path distinct


def test_two_instances_of_one_template_give_different_worlds():
    d = doc.MasonDoc()
    template, leaf_a, leaf_b = _template()
    d.prefabs["Prop"] = template
    inst1 = nd.PrefabNode(uid=nd.new_uid(), template="Prop", translation=m3.vec3(1.0, 0.0, 0.0))
    inst2 = nd.PrefabNode(uid=nd.new_uid(), template="Prop", translation=m3.vec3(5.0, 0.0, 0.0))
    d.add_nodes([inst1, inst2])

    worlds = [p.world[:3, 3].tolist() for p in scene.resolve(d) if p.node.uid == leaf_a.uid]
    assert len(worlds) == 2
    assert worlds[0] != worlds[1]


def test_owner_is_the_instance_for_a_node_inside_one_and_the_leaf_itself_otherwise():
    d = doc.MasonDoc()
    template, leaf_a, leaf_b = _template()
    d.prefabs["Prop"] = template
    inst = nd.PrefabNode(uid=nd.new_uid(), template="Prop")
    d.add_node(inst)
    ordinary = nd.MeshNode(uid=nd.new_uid(), ref=_box_ref())
    d.add_node(ordinary)

    placed = scene.resolve(d)
    by_uid = {p.node.uid: p for p in placed}

    assert by_uid[ordinary.uid].owner == ordinary.uid
    assert by_uid[leaf_a.uid].owner == inst.uid
    assert by_uid[leaf_b.uid].owner == inst.uid


def test_a_prefab_node_naming_a_missing_template_yields_nothing_rather_than_raising():
    d = doc.MasonDoc()
    inst = nd.PrefabNode(uid=nd.new_uid(), template="ghost")
    d.add_node(inst)

    assert scene.resolve(d) == []


def test_a_prefab_cycle_costs_its_own_branch_and_terminates():
    """``define_prefab`` refuses this at the door; a hand-edited ``.wscn``
    bypasses the door the same way it bypasses every other one, so the
    resolver has to survive a template graph that names itself indirectly."""
    d = doc.MasonDoc()
    a_template = nd.GroupNode(uid=nd.new_uid(), name="A")
    a_template.children.append(nd.PrefabNode(uid=nd.new_uid(), template="B"))
    b_template = nd.GroupNode(uid=nd.new_uid(), name="B")
    b_template.children.append(nd.PrefabNode(uid=nd.new_uid(), template="A"))
    d.prefabs["A"] = a_template
    d.prefabs["B"] = b_template

    cyclical = nd.PrefabNode(uid=nd.new_uid(), template="A")
    sound = nd.MeshNode(uid=nd.new_uid(), ref=_box_ref())
    d.add_nodes([cyclical, sound])

    placed = scene.resolve(d)  # must terminate, not recurse forever

    assert [p.node.uid for p in placed] == [sound.uid]


def test_expand_prefabs_false_yields_the_instance_node_itself():
    d = doc.MasonDoc()
    template, leaf_a, leaf_b = _template()
    d.prefabs["Prop"] = template
    inst = nd.PrefabNode(uid=nd.new_uid(), template="Prop")
    d.add_node(inst)

    unexpanded = _collect(d, expand_prefabs=False)
    assert [p.node.uid for p in unexpanded if isinstance(p.node, nd.PrefabNode)] == [inst.uid]
    # None of the template's own leaves appear when the instance is left folded.
    assert leaf_a.uid not in {p.node.uid for p in unexpanded}


# --- dangling ----------------------------------------------------------


def test_dangling_follows_doc_missing():
    d = doc.MasonDoc()
    ref = refs.LibraryRef(job_id="abc123", name="Barrel")
    leaf = nd.MeshNode(uid=nd.new_uid(), ref=ref)
    d.add_node(leaf)

    assert scene.resolve(d)[0].dangling is False

    d.missing.add(refs.ref_key(ref))
    assert scene.resolve(d)[0].dangling is True


# --- resolved_for --------------------------------------------------------


def test_resolved_for_answers_about_a_hidden_node_and_about_a_group():
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), visible=False)
    leaf = nd.MeshNode(uid=nd.new_uid(), ref=_box_ref(), visible=True)
    group.children.append(leaf)
    d.add_node(group)

    assert scene.resolve(d) == []  # both skipped by resolve

    group_placed = scene.resolved_for(d, group.uid)
    assert group_placed is not None
    assert group_placed.node is group

    leaf_placed = scene.resolved_for(d, leaf.uid)
    assert leaf_placed is not None
    assert leaf_placed.node is leaf
    assert leaf_placed.visible is False  # combined with its hidden ancestor


def test_resolved_for_answers_none_for_an_unknown_uid():
    d = doc.MasonDoc()
    assert scene.resolved_for(d, 999_999) is None


# --- world_bounds --------------------------------------------------------


def test_world_bounds_over_a_subset():
    d = doc.MasonDoc()
    ref_a = _box_ref(size=(1.0, 1.0, 1.0))
    ref_b = _box_ref(size=(2.0, 2.0, 2.0))
    a = nd.MeshNode(uid=nd.new_uid(), ref=ref_a, translation=m3.vec3(10.0, 0.0, 0.0))
    b = nd.MeshNode(uid=nd.new_uid(), ref=ref_b, translation=m3.vec3(-10.0, 0.0, 0.0))
    d.add_nodes([a, b])

    box = (np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5]))
    source = _FakeSource({refs.ref_key(ref_a): box, refs.ref_key(ref_b): box})

    lo, hi = scene.world_bounds(d, source, uids={a.uid})
    assert lo == pytest.approx([9.5, -0.5, -0.5])
    assert hi == pytest.approx([10.5, 0.5, 0.5])

    lo_all, hi_all = scene.world_bounds(d, source)
    assert lo_all[0] == pytest.approx(-10.5)
    assert hi_all[0] == pytest.approx(10.5)


def test_world_bounds_returns_none_when_the_source_has_no_boxes_yet():
    d = doc.MasonDoc()
    a = nd.MeshNode(uid=nd.new_uid(), ref=_box_ref())
    d.add_node(a)
    source = _FakeSource({})  # nothing has resolved yet

    assert scene.world_bounds(d, source) is None


# --- max_items -----------------------------------------------------------


def test_max_items_refuses_with_a_real_sentence_rather_than_truncating():
    d = doc.MasonDoc()
    for _ in range(3):
        d.add_node(nd.MeshNode(uid=nd.new_uid(), ref=_box_ref()))

    with pytest.raises(ValueError, match="more than 2"):
        scene.resolve(d, max_items=2)


def test_the_soft_warning_threshold_sits_below_the_hard_refusal_and_both_are_positive():
    """Pins the *relationship* rather than either number: a constant asserted
    against a literal would just restate itself and pass right through the
    edit that raises one past the other. What must never happen is a scene
    that warns only after it would already have been refused."""
    assert 0 < scene.PLACED_WARN_THRESHOLD < scene.MAX_PLACED


def test_a_path_names_every_uid_from_the_root_and_not_only_the_segment_it_is_in():
    """``Placed.path`` is "uids root-first", and for a node inside a group it
    was not.

    This used to be ``parent_path + (node.uid,)`` computed off the *segment's*
    base rather than off the node's own parent, so a mesh three groups down
    came back as a one-element path. Every Stage C consumer wanted the path
    only as a unique key -- and one uid already is unique within a segment --
    so nothing caught it. What it broke is the instruction this module's own
    docstring gives a structural exporter, "hang it under ``path[:-1]``'s
    node": with the ancestry missing, every node in the file lands at the
    root and the exported scene is a flattening of exactly the hierarchy the
    export exists to keep. ``gltfout.py`` found it in Stage D, which is the
    consumer that sentence was written for.
    """
    d = doc.MasonDoc()
    outer = nd.GroupNode(uid=nd.new_uid())
    inner = nd.GroupNode(uid=nd.new_uid())
    leaf = nd.MeshNode(uid=nd.new_uid(), ref=_box_ref())
    inner.children.append(leaf)
    outer.children.append(inner)
    d.add_node(outer)

    by_uid = {p.node.uid: p.path for p in _collect(d)}
    assert by_uid[outer.uid] == (outer.uid,)
    assert by_uid[inner.uid] == (outer.uid, inner.uid)
    assert by_uid[leaf.uid] == (outer.uid, inner.uid, leaf.uid)


def test_a_path_crosses_a_prefab_boundary_carrying_both_halves_of_the_ancestry():
    """The instance's own path, then the template's own nesting under it --
    which is what distinguishes one instance's copy of a template leaf from
    another's, and what lets an exporter rebuild the tree on either side of
    the boundary from the path alone."""
    d = doc.MasonDoc()
    template_root = nd.GroupNode(uid=nd.new_uid())
    template_leaf = nd.MeshNode(uid=nd.new_uid(), ref=_box_ref())
    template_root.children.append(template_leaf)
    d.define_prefab("crate", template_root)

    holder = nd.GroupNode(uid=nd.new_uid())
    instance = nd.PrefabNode(uid=nd.new_uid(), template="crate")
    holder.children.append(instance)
    d.add_node(holder)

    leaf = scene.resolve(d)[0]
    assert leaf.path == (
        holder.uid,
        instance.uid,
        template_root.uid,
        template_leaf.uid,
    )
    assert leaf.owner == instance.uid
