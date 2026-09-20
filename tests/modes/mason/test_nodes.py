"""``mason/nodes.py``: the six node classes, the uid machinery, and the walk.

Every test name is a claim, and each one is written to fail against the
version of ``nodes.py`` it would have caught -- an unowned transform array, a
WXYZ rotation default, a uid that moves backwards, a walk that spins forever
on a hand-made cycle.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.studio.modes.mason.engine import nodes as nd


def test_a_node_owns_its_translation_array_a_caller_mutating_it_after_does_not_reach_in() -> None:
    handed_over = np.array([1.0, 2.0, 3.0])
    node = nd.GroupNode(uid=nd.new_uid(), translation=handed_over)
    handed_over[0] = 999.0
    assert node.translation[0] == 1.0


def test_a_node_owns_its_rotation_and_scale_arrays_too() -> None:
    rot = np.array([0.1, 0.2, 0.3, 0.9])
    scl = np.array([2.0, 2.0, 2.0])
    node = nd.GroupNode(uid=nd.new_uid(), rotation=rot, scale=scl)
    rot[0] = -1.0
    scl[0] = -1.0
    assert node.rotation[0] == 0.1
    assert node.scale[0] == 2.0


def test_rotation_defaults_to_the_xyzw_identity_not_wxyz() -> None:
    """A WXYZ default here would be a bug that renders as a plausible-looking
    rotation, which is exactly why this gets its own test."""
    node = nd.GroupNode(uid=nd.new_uid())
    assert node.rotation.tolist() == [0.0, 0.0, 0.0, 1.0]


def _count_compose_calls(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Patch ``m3.compose`` with a counting wrapper; the returned list's
    length is how many times it actually ran.

    ``local()`` always hands back a *copy* (see its own docstring), so two
    calls that hit the memo and two calls that both missed produce
    indistinguishable output -- equal matrices, distinct objects, either way.
    The only way to tell a hit from a recompute is to watch the thing a hit
    skips, which is this. Shared by every test below that needs to observe
    the memo rather than infer it from a result nothing here can tell apart.
    """
    calls: list[object] = []
    real_compose = nd.m3.compose

    def counting_compose(t: np.ndarray, r: np.ndarray, s: np.ndarray) -> np.ndarray:
        calls.append(None)
        return real_compose(t, r, s)

    monkeypatch.setattr(nd.m3, "compose", counting_compose)
    return calls


def test_local_composes_once_for_repeated_calls_and_again_after_a_rebind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The memo actually hitting, pinned on the thing a hit skips rather than
    on ``local()``'s output: a copy of an equal matrix is what *both* a hit
    and a miss look like from the outside (see :func:`_count_compose_calls`),
    so only counting the underlying ``compose`` calls can tell them apart."""
    calls = _count_compose_calls(monkeypatch)
    node = nd.GroupNode(uid=nd.new_uid(), translation=np.array([1.0, 2.0, 3.0]))

    node.local()
    node.local()
    assert len(calls) == 1  # the second call was a hit, not a second compose

    node.translation = np.array([9.0, 9.0, 9.0])  # a rebind, not a write-through
    node.local()
    assert len(calls) == 2  # the rebind forced a real recompute


def test_local_returns_a_fresh_copy_equal_in_value_on_repeated_calls() -> None:
    """Whether a call hits the memo or misses it, the caller gets an object
    it is free to write into: ``local()`` never hands back the cache entry
    itself. (Whether the *second* call in this test was a hit or a fresh
    compose is not something equal, distinct copies can answer -- that claim
    belongs to ``test_local_composes_once_for_repeated_calls...`` above.)"""
    node = nd.GroupNode(uid=nd.new_uid(), translation=np.array([1.0, 2.0, 3.0]))
    first = node.local()
    second = node.local()
    assert np.array_equal(first, second)
    assert first is not second


def test_local_reflects_a_rebound_arrays_new_value() -> None:
    """Correctness after a rebind, independent of caching strategy: this
    would hold even with no memo at all, and it is not a claim about the memo
    invalidating -- see ``test_local_composes_once_for_repeated_calls...``
    for the test that actually counts whether a recompute happened."""
    node = nd.GroupNode(uid=nd.new_uid(), translation=np.array([1.0, 2.0, 3.0]))
    node.local()
    node.translation = np.array([9.0, 9.0, 9.0])  # a rebind, not a write-through
    after = node.local()
    assert after[:3, 3].tolist() == [9.0, 9.0, 9.0]


def test_two_nodes_with_equal_transforms_do_not_share_a_cache_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The memo lives on the instance, not on a table keyed by the arrays'
    *values* or by a plain ``id()`` two distinct objects could share once one
    is freed -- two nodes built with equal-valued, distinct transform arrays
    must each pay for (and cache) their own compose, and a rebind on one must
    never make the other look stale or force it to recompute."""
    calls = _count_compose_calls(monkeypatch)
    a = nd.GroupNode(uid=nd.new_uid(), translation=np.array([1.0, 2.0, 3.0]))
    b = nd.GroupNode(uid=nd.new_uid(), translation=np.array([1.0, 2.0, 3.0]))

    a.local()
    b.local()
    assert len(calls) == 2  # each node's first call is its own miss

    a.translation = np.array([5.0, 5.0, 5.0])
    changed = a.local()
    unaffected = b.local()
    assert len(calls) == 3  # only a's rebind forced a recompute
    assert changed[:3, 3].tolist() == [5.0, 5.0, 5.0]
    assert unaffected[:3, 3].tolist() == [1.0, 2.0, 3.0]


def test_writing_through_a_transform_array_instead_of_rebinding_is_the_documented_trap() -> None:
    """Pins the *stated* behaviour rather than fixing it: ``local()`` keys on
    which object ``self.translation`` names, not on the numbers inside it, so
    a write-through is invisible to the memo. This is the surprise the class
    docstring names -- the node draws stale while the properties panel, which
    reads the array directly, reports the new numbers -- and it must stay
    reproducible so nobody "fixes" the memo into something slower on the
    strength of a surprise nobody wrote down."""
    node = nd.GroupNode(uid=nd.new_uid(), translation=np.array([1.0, 2.0, 3.0]))
    node.local()  # populate the cache against the array node.translation names
    node.translation[:] = [9.0, 9.0, 9.0]  # a write-through, the documented trap
    stale = node.local()
    assert stale[:3, 3].tolist() == [1.0, 2.0, 3.0]  # the memo did not notice
    assert node.translation.tolist() == [9.0, 9.0, 9.0]  # but the array really changed


def test_properties_dict_is_copied_not_aliased() -> None:
    props = {"tag": "crate"}
    node = nd.GroupNode(uid=nd.new_uid(), properties=props)
    props["tag"] = "changed"
    assert node.properties["tag"] == "crate"


def test_a_light_refuses_an_unknown_kind_naming_what_it_got() -> None:
    with pytest.raises(ValueError, match="bogus"):
        nd.LightNode(uid=nd.new_uid(), kind="bogus")


def test_a_mesh_node_defaults_to_no_ref_and_no_material_override() -> None:
    node = nd.MeshNode(uid=nd.new_uid())
    assert node.ref is None
    assert node.material is None


def test_a_prefab_node_defaults_to_no_template() -> None:
    node = nd.PrefabNode(uid=nd.new_uid())
    assert node.template == ""


# --- walk --------------------------------------------------------------------


def _tree() -> tuple[nd.Node, nd.Node, nd.Node, nd.Node]:
    """root -> [A -> [B], C], a three-level tree with a sibling at the top."""
    root = nd.GroupNode(uid=nd.new_uid(), name="root")
    a = nd.GroupNode(uid=nd.new_uid(), name="A")
    b = nd.GroupNode(uid=nd.new_uid(), name="B")
    c = nd.GroupNode(uid=nd.new_uid(), name="C")
    a.children.append(b)
    root.children.extend([a, c])
    return root, a, b, c


def test_walk_visits_a_three_level_tree_depth_first_pre_order() -> None:
    root, a, b, c = _tree()
    names = [node.name for node, _parent, _index, _depth in nd.walk([root])]
    assert names == ["root", "A", "B", "C"]


def test_walk_reports_the_object_parent_of_each_node() -> None:
    root, a, b, c = _tree()
    by_name = {node.name: parent for node, parent, _index, _depth in nd.walk([root])}
    assert by_name["root"] is None
    assert by_name["A"] is root
    assert by_name["B"] is a
    assert by_name["C"] is root


def test_walk_reports_each_nodes_index_among_its_own_siblings() -> None:
    root, a, b, c = _tree()
    by_name = {node.name: index for node, _parent, index, _depth in nd.walk([root])}
    assert by_name["root"] == 0
    assert by_name["A"] == 0
    assert by_name["B"] == 0
    assert by_name["C"] == 1  # C is root's second child


def test_walk_reports_each_nodes_depth() -> None:
    root, a, b, c = _tree()
    by_name = {node.name: depth for node, _parent, _index, depth in nd.walk([root])}
    assert by_name == {"root": 0, "A": 1, "B": 2, "C": 1}


def test_a_cycle_costs_only_the_branch_it_is_in() -> None:
    """``walk`` runs on the frame thread every time the resolver draws the
    scene, so a hand-edited ``.rscn`` with a cycle must not take the whole
    draw down -- only the branch that is actually broken is skipped, and an
    unrelated sibling still walks."""
    root = nd.GroupNode(uid=nd.new_uid(), name="root")
    x = nd.GroupNode(uid=nd.new_uid(), name="x")
    y = nd.GroupNode(uid=nd.new_uid(), name="y")
    sound = nd.GroupNode(uid=nd.new_uid(), name="sound")
    x.children.append(y)
    y.children.append(x)  # x is its own grandchild -- the corrupt branch
    root.children.extend([x, sound])

    results = list(nd.walk([root]))  # must terminate at all, not spin forever

    names = [node.name for node, _parent, _index, _depth in results]
    assert names.count("x") == 1  # seen once, not revisited through the cycle
    assert "y" in names
    assert "sound" in names  # the unrelated sibling branch is untouched


def test_a_tree_deeper_than_the_ceiling_still_walks_what_is_above_it() -> None:
    root = nd.GroupNode(uid=nd.new_uid())
    cursor = root
    for _ in range(nd.MAX_DEPTH + 5):
        child = nd.GroupNode(uid=nd.new_uid())
        cursor.children.append(child)
        cursor = child

    results = list(nd.walk([root]))  # must terminate, not raise

    depths = [depth for _node, _parent, _index, depth in results]
    # Every depth from the root down to the ceiling is walked, and nothing
    # past it -- the over-deep tail is cut, not the whole chain.
    assert depths == list(range(nd.MAX_DEPTH + 1))


# --- contains ------------------------------------------------------------


def test_contains_is_false_for_the_node_itself() -> None:
    root, a, b, c = _tree()
    assert nd.contains(root, root.uid) is False


def test_contains_is_true_for_a_grandchild() -> None:
    root, a, b, c = _tree()
    assert nd.contains(root, b.uid) is True


def test_contains_is_false_for_a_uid_nowhere_in_the_subtree() -> None:
    root, a, b, c = _tree()
    assert nd.contains(root, 999_999) is False


# --- copy_subtree ----------------------------------------------------------


def _mesh_subtree() -> tuple[nd.Node, gltf.Material]:
    material = gltf.Material(name="shared")
    leaf = nd.MeshNode(uid=nd.new_uid(), name="leaf", material=material)
    root = nd.GroupNode(uid=nd.new_uid(), name="root")
    root.children.append(leaf)
    return root, material


def test_copy_subtree_with_fresh_uids_gives_every_node_a_new_uid() -> None:
    root, _material = _mesh_subtree()
    copy = nd.copy_subtree(root, fresh_uids=True)
    assert copy.uid != root.uid
    assert copy.children[0].uid != root.children[0].uid


def test_copy_subtree_with_fresh_uids_gives_every_node_its_own_arrays() -> None:
    root, _material = _mesh_subtree()
    copy = nd.copy_subtree(root, fresh_uids=True)
    assert copy.translation is not root.translation
    assert copy.children[0].translation is not root.children[0].translation
    assert copy.translation.tolist() == root.translation.tolist()


def test_copy_subtree_shares_the_material_object_by_identity() -> None:
    """``GpuMaterial`` and the GLB writer both de-duplicate by ``id()``, so a
    copied material would be a second upload for a material nobody changed."""
    root, material = _mesh_subtree()
    copy = nd.copy_subtree(root, fresh_uids=True)
    assert copy.children[0].material is material


def test_copy_subtree_without_fresh_uids_keeps_every_uid() -> None:
    root, _material = _mesh_subtree()
    copy = nd.copy_subtree(root, fresh_uids=False)
    assert copy.uid == root.uid
    assert copy.children[0].uid == root.children[0].uid


def test_copy_subtree_preserves_the_shape_of_the_tree() -> None:
    root, a, b, c = _tree()
    copy = nd.copy_subtree(root, fresh_uids=True)
    names = [node.name for node, _parent, _index, _depth in nd.walk([copy])]
    assert names == ["root", "A", "B", "C"]


# --- subtree_bytes -----------------------------------------------------------


def test_subtree_bytes_is_non_zero_for_a_single_node() -> None:
    node = nd.GroupNode(uid=nd.new_uid())
    assert nd.subtree_bytes(node) > 0


def test_subtree_bytes_grows_as_children_are_added() -> None:
    root = nd.GroupNode(uid=nd.new_uid())
    before = nd.subtree_bytes(root)
    root.children.append(nd.GroupNode(uid=nd.new_uid()))
    after = nd.subtree_bytes(root)
    assert after > before


def test_subtree_bytes_counts_a_large_properties_value() -> None:
    small = nd.GroupNode(uid=nd.new_uid())
    large = nd.GroupNode(uid=nd.new_uid(), properties={"blob": "x" * 10_000})
    assert nd.subtree_bytes(large) > nd.subtree_bytes(small)


# --- uid machinery -----------------------------------------------------------


def test_new_uid_never_repeats() -> None:
    seen = {nd.new_uid() for _ in range(50)}
    assert len(seen) == 50


def test_reserve_uid_raises_the_floor_so_the_next_new_uid_is_past_it() -> None:
    high = nd.new_uid()
    nd.reserve_uid(high + 1000)
    assert nd.new_uid() == high + 1001


def test_reserve_uid_reserving_a_low_value_after_a_high_one_does_not_move_the_floor_back() -> None:
    high = nd.new_uid()
    nd.reserve_uid(high + 1000)
    minted = nd.new_uid()  # consumes high + 1001, raising the live counter to high + 1002
    nd.reserve_uid(high + 1)  # far below the current floor -- must be a no-op
    assert nd.new_uid() > minted
