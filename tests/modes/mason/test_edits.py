"""``mason/edits.py``: the ten undo steps -- what each one costs, what each
one owns, and which document hook each one calls back into.

A ``_Recorder`` double stands in for ``MasonDoc`` in most tests here: the
edits' whole job is to translate "undo" and "redo" into exactly one call on
one document hook, and that translation is what these tests pin, deliberately
without a real tree, a real terrain or a real history stack behind it --
``document.py``'s own tests are where the hooks' *behaviour* is proved.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np

from realmspinner.kernels.geom3d.gltf import Material
from realmspinner.studio.modes.mason.engine import document as docmod
from realmspinner.studio.modes.mason.engine import edits as ed
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import refs
from realmspinner.studio.modes.mason.engine import terrain as tr


class _Recorder:
    """Stands in for ``MasonDoc``: every hook an edit can call, recorded."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def _attach_node(self, node, parent_uid, index) -> None:
        self.calls.append(("attach", node.uid, parent_uid, index))

    def _detach_node(self, uid) -> None:
        self.calls.append(("detach", uid))

    def _relocate(self, uid, target) -> None:
        self.calls.append(("relocate", uid, target))

    def _apply_transform(self, uid, trs) -> None:
        self.calls.append(("transform", uid, tuple(v.tolist() for v in trs)))

    def _apply_props(self, uid, props) -> None:
        self.calls.append(("props", uid, dict(props)))

    def _apply_ref(self, uid, ref) -> None:
        self.calls.append(("ref", uid, ref))

    def _blit_terrain(self, rect, sub) -> None:
        self.calls.append(("blit", rect, sub.copy()))

    def _apply_terrain(self, terrain) -> None:
        self.calls.append(("terrain", terrain))

    def _apply_terrain_config(self, values) -> None:
        self.calls.append(("terrain_config", dict(values)))

    def _apply_prefab(self, name, node) -> None:
        self.calls.append(("prefab", name, node))


# --- NodeAddEdit / NodeRemoveEdit --------------------------------------------


def test_node_add_edit_undo_detaches_by_uid_and_redo_attaches_at_the_recorded_spot():
    node = nd.GroupNode(uid=nd.new_uid())
    edit = ed.NodeAddEdit(parent_uid=7, index=2, node=node)
    doc = _Recorder()
    edit.undo(doc)
    edit.redo(doc)
    assert doc.calls == [("detach", node.uid), ("attach", node.uid, 7, 2)]


def test_node_remove_edit_undo_reattaches_and_redo_detaches_by_uid():
    node = nd.GroupNode(uid=nd.new_uid())
    edit = ed.NodeRemoveEdit(parent_uid=None, index=0, node=node)
    doc = _Recorder()
    edit.undo(doc)
    edit.redo(doc)
    assert doc.calls == [("attach", node.uid, None, 0), ("detach", node.uid)]


def test_node_add_and_remove_edits_cost_the_whole_subtree_they_hold():
    leaf = nd.GroupNode(uid=nd.new_uid())
    root = nd.GroupNode(uid=nd.new_uid())
    root.children.append(leaf)
    solo_cost = ed.NodeAddEdit(None, 0, nd.GroupNode(uid=nd.new_uid())).cost
    subtree_cost = ed.NodeAddEdit(None, 0, root).cost
    assert subtree_cost > solo_cost
    assert subtree_cost == nd.subtree_bytes(root)
    assert ed.NodeRemoveEdit(None, 0, root).cost == subtree_cost


# --- NodeMoveEdit -------------------------------------------------------------


def test_node_move_edit_undo_and_redo_relocate_by_uid_costing_nothing():
    edit = ed.NodeMoveEdit(node_uid=5, before=(None, 0), after=(9, 2))
    doc = _Recorder()
    edit.redo(doc)
    edit.undo(doc)
    assert doc.calls == [("relocate", 5, (9, 2)), ("relocate", 5, (None, 0))]
    assert edit.cost == 0


# --- TransformEdit -------------------------------------------------------------


def test_transform_edit_owns_copies_a_caller_mutating_the_original_after_does_not_reach_in():
    before = (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), np.ones(3))
    after = (np.ones(3), np.array([0.0, 0.0, 0.0, 1.0]), np.ones(3) * 2)
    edit = ed.TransformEdit(node_uid=1, before=before, after=after)
    before[0][0] = 999.0
    after[0][0] = -999.0
    assert edit.before[0][0] == 0.0
    assert edit.after[0][0] == 1.0


def test_transform_edit_costs_all_six_arrays_and_calls_apply_transform():
    before = (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), np.ones(3))
    after = (np.ones(3), np.array([0.0, 0.0, 0.0, 1.0]), np.ones(3) * 2)
    edit = ed.TransformEdit(node_uid=1, before=before, after=after)
    assert edit.cost == sum(v.nbytes for v in (*before, *after))
    doc = _Recorder()
    edit.undo(doc)
    edit.redo(doc)
    assert doc.calls[0][0] == "transform" and doc.calls[0][1] == 1
    assert doc.calls[0][2] == tuple(v.tolist() for v in before)
    assert doc.calls[1][2] == tuple(v.tolist() for v in after)


# --- NodePropsEdit -------------------------------------------------------------


def test_node_props_edit_costs_more_for_a_larger_properties_value():
    small = ed.NodePropsEdit(1, {"name": "a"}, {"name": "b"})
    large = ed.NodePropsEdit(1, {"properties": {"x": "y" * 10_000}}, {"properties": {}})
    assert large.cost > small.cost


def test_node_props_edit_undo_and_redo_apply_props_by_uid():
    edit = ed.NodePropsEdit(3, {"name": "old"}, {"name": "new"})
    doc = _Recorder()
    edit.undo(doc)
    edit.redo(doc)
    assert doc.calls == [("props", 3, {"name": "old"}), ("props", 3, {"name": "new"})]


# --- RefEdit -------------------------------------------------------------------


def test_ref_edit_costs_nothing_for_two_none_refs():
    edit = ed.RefEdit(1, None, None)
    assert edit.cost == 0


def test_ref_edit_costs_something_for_a_real_ref():
    ref = refs.primitive_ref("box", {"size": (1.0, 1.0, 1.0)})
    edit = ed.RefEdit(1, None, ref)
    assert edit.cost > 0


def test_ref_edit_undo_and_redo_apply_ref_by_uid():
    ref = refs.primitive_ref("box", {"size": (1.0, 1.0, 1.0)})
    edit = ed.RefEdit(4, None, ref)
    doc = _Recorder()
    edit.undo(doc)
    edit.redo(doc)
    assert doc.calls == [("ref", 4, None), ("ref", 4, ref)]


def test_ref_edit_and_subtree_bytes_charge_a_primitive_refs_large_nested_params_not_just_sys_getsizeof():  # noqa: E501
    """The 2026-09-20 audit's mason-01: ``_ref_bytes`` priced a
    ``PrimitiveRef`` with a bare ``sys.getsizeof``, which only sees the
    dataclass's own three pointers and never what its ``params`` tuple
    holds -- a lathe or sweep profile of thousands of ``(x, y)`` points was
    charged the same handful of bytes as an empty one (an ~11,700x
    undercount against the real bytes, per the audit's own measurement).
    ``nd.subtree_bytes`` compounded it by never charging a ``MeshNode``'s
    ``ref`` at all. Against the unfixed code, ``big_cost`` is barely bigger
    than ``small_cost`` and ``subtree_bytes`` does not move when the same
    big ref is attached to a mesh node.
    """
    small_ref = refs.primitive_ref("lathe", {"profile": [[0.0, 0.0], [1.0, 1.0]]})
    big_profile = [[float(i), float(i) * 0.5] for i in range(5_000)]
    big_ref = refs.primitive_ref("lathe", {"profile": big_profile})

    small_cost = ed.RefEdit(1, None, small_ref).cost
    big_cost = ed.RefEdit(1, None, big_ref).cost
    # ~5,000 (x, y) float pairs is on the order of hundreds of KB; two
    # orders of magnitude of headroom over the small ref is well clear of
    # dataclass/tuple overhead noise and still catches the reported bug.
    assert big_cost > small_cost * 100

    mesh_no_ref = nd.MeshNode(uid=nd.new_uid())
    mesh_with_ref = nd.MeshNode(uid=nd.new_uid(), ref=big_ref)
    assert nd.subtree_bytes(mesh_with_ref) > nd.subtree_bytes(mesh_no_ref) + 100_000


# --- TerrainEdit ---------------------------------------------------------------


def test_terrain_edit_owns_copies_of_its_sub_arrays():
    before = np.zeros((2, 2), dtype="f4")
    after = np.ones((2, 2), dtype="f4")
    edit = ed.TerrainEdit((0, 0, 2, 2), before, after)
    before[0, 0] = 999.0
    after[0, 0] = -999.0
    assert edit.before[0, 0] == 0.0
    assert edit.after[0, 0] == 1.0


def test_terrain_edit_costs_exactly_the_two_sub_arrays_not_a_whole_height_field():
    """The plan's own claim: a step over a rect must not be billed as if it
    held the entire terrain -- see ``edits.TerrainEdit``'s docstring for why
    that would make an hour of sculpting evictable by eight small dabs."""
    small_before = np.zeros((3, 3), dtype="f4")
    small_after = np.ones((3, 3), dtype="f4")
    edit = ed.TerrainEdit((0, 0, 3, 3), small_before, small_after)
    assert edit.cost == small_before.nbytes + small_after.nbytes
    # A whole 256-side height field is roughly 256 KB; this step's own
    # tiny rect must be nowhere near that, regardless of the terrain it was
    # cut from.
    assert edit.cost < 1024


def test_terrain_edit_undo_and_redo_blit_the_recorded_sub_array():
    before = np.zeros((2, 2), dtype="f4")
    after = np.ones((2, 2), dtype="f4")
    edit = ed.TerrainEdit((1, 1, 3, 3), before, after)
    doc = _Recorder()
    edit.undo(doc)
    edit.redo(doc)
    assert doc.calls[0][:2] == ("blit", (1, 1, 3, 3))
    np.testing.assert_array_equal(doc.calls[0][2], before)
    np.testing.assert_array_equal(doc.calls[1][2], after)


# --- TerrainSwapEdit -----------------------------------------------------------


def _small_terrain() -> tr.Terrain:
    heights = np.zeros((3, 3), dtype="f4")
    return tr.Terrain(heights=heights, size_x=4.0, size_z=4.0, material=Material())


def test_terrain_swap_edit_costs_nothing_for_two_none_sides():
    assert ed.TerrainSwapEdit(None, None).cost == 0


def test_terrain_swap_edit_costs_the_whole_height_field_of_whichever_side_exists():
    """Unlike ``TerrainEdit``, a swap is billed for the *entire* array: the
    whole field arrives or leaves the document in one step, so the whole
    field is what an undo of that step has to be able to restore."""
    terrain = _small_terrain()
    installed = ed.TerrainSwapEdit(None, terrain)
    removed = ed.TerrainSwapEdit(terrain, None)
    assert installed.cost == terrain.heights.nbytes
    assert removed.cost == terrain.heights.nbytes


def test_terrain_swap_edit_holds_the_terrain_objects_themselves_not_a_copy():
    """No extra copy is taken -- ``Terrain.__post_init__`` already owns the
    array, so this step just holds the instance, the same argument
    ``NodeAddEdit`` makes for a node."""
    terrain = _small_terrain()
    edit = ed.TerrainSwapEdit(None, terrain)
    assert edit.after is terrain


def test_terrain_swap_edit_undo_and_redo_apply_terrain_by_object():
    terrain = _small_terrain()
    edit = ed.TerrainSwapEdit(None, terrain)
    doc = _Recorder()
    edit.undo(doc)
    edit.redo(doc)
    assert doc.calls == [("terrain", None), ("terrain", terrain)]


# --- TerrainConfigEdit ---------------------------------------------------------


def test_terrain_config_edit_is_small_even_carrying_a_material():
    edit = ed.TerrainConfigEdit({"size_x": 8.0}, {"size_x": 16.0, "material": Material()})
    assert edit.cost < 4096


def test_terrain_config_edit_undo_and_redo_apply_terrain_config():
    edit = ed.TerrainConfigEdit({"size_x": 8.0}, {"size_x": 16.0})
    doc = _Recorder()
    edit.undo(doc)
    edit.redo(doc)
    assert doc.calls == [
        ("terrain_config", {"size_x": 8.0}),
        ("terrain_config", {"size_x": 16.0}),
    ]


# --- PrefabEdit ------------------------------------------------------------


def test_prefab_edit_costs_nothing_for_two_none_sides():
    assert ed.PrefabEdit("thing", None, None).cost == 0


def test_prefab_edit_costs_the_subtree_of_whichever_side_exists():
    template = nd.GroupNode(uid=nd.new_uid())
    template.children.append(nd.GroupNode(uid=nd.new_uid()))
    edit = ed.PrefabEdit("thing", None, template)
    assert edit.cost == nd.subtree_bytes(template)


def test_prefab_edit_undo_and_redo_apply_prefab_by_name():
    template = nd.GroupNode(uid=nd.new_uid())
    edit = ed.PrefabEdit("thing", None, template)
    doc = _Recorder()
    edit.undo(doc)
    edit.redo(doc)
    assert doc.calls == [("prefab", "thing", None), ("prefab", "thing", template)]


def test_every_edit_type_subclasses_the_shared_undo_engines_edit():
    """A sanity check on the sweep the document-level tests parametrize over:
    if this ever went to zero, that sweep would be silently testing nothing."""
    from realmspinner.core.undo import Edit

    names = [
        name
        for name, obj in vars(ed).items()
        if isinstance(obj, type) and issubclass(obj, Edit) and obj is not Edit
    ]
    assert len(names) == 10


# --- the docstring's hook list, gated both ways -------------------------------
#
# The module docstring names the document hooks in prose, and a prose list is
# exactly the shape this repo's own ``PUBLISHERS`` note warns about: it fails
# open, silently, the moment one side drifts from the other. It already did,
# in this very package -- ``_apply_terrain`` was added alongside
# ``TerrainSwapEdit`` and the docstring's list was not updated in the same
# change, so it went on naming nine hooks when ten were being called. Fixing
# that one line does not fix the *gate*: nothing stopped it from happening and
# nothing would stop it from happening again to the next hook. So this is
# derived, not re-read by eye -- the same argument ``_pure_packages.py`` makes
# for a sibling ban, one file over.


def _called_document_hooks() -> set[str]:
    """Every ``doc._foo(...)`` call ``edits.py`` actually makes, found by
    parsing its own source rather than hand-listed. ``doc`` is the parameter
    name every ``undo``/``redo`` method in this module uses for the document
    it is handed, so a call on it whose attribute starts with ``_`` is, by
    construction, one of the hooks this file's docstring promises to name.
    """
    tree = ast.parse(Path(ed.__file__).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "doc"
            and node.func.attr.startswith("_")
        ):
            found.add(node.func.attr)
    return found


def test_every_document_hook_an_edit_calls_is_named_in_the_docstring():
    """The direction that actually drifted: a hook some edit's ``undo``/
    ``redo`` calls, that the reader teaching themselves the rule from the
    docstring alone would never learn exists."""
    called = _called_document_hooks()
    # A parse that silently matched nothing would make the loop below
    # vacuously pass -- guard that this derivation is finding real calls,
    # not quietly finding none.
    assert len(called) >= 10, f"the AST walk found suspiciously few hooks: {sorted(called)}"
    missing = {name for name in called if f"``{name}``" not in ed.__doc__}
    assert not missing, (
        f"called from edits.py but not named in its module docstring: {sorted(missing)}"
    )


def test_every_hook_the_docstring_names_exists_on_the_document():
    """The other direction: a hook renamed (or never written) in
    ``document.py`` must fail here, at the docstring that claims it, rather
    than at the first undo a user actually presses."""
    # Scoped to the one sentence that lists the hooks, not the whole
    # docstring -- a later paragraph names ``_apply_layer_props`` too, but
    # that is ``plotter/_map_layers.py``'s own hook, cited by analogy, and it
    # does not exist on ``MasonDoc``.
    start = ed.__doc__.index("document hooks, never by reaching into")
    end = ed.__doc__.index("are ``MasonDoc``'s own methods", start)
    segment = ed.__doc__[start:end]
    named = set(re.findall(r"``(_[a-zA-Z][a-zA-Z0-9_]*)``", segment))
    assert len(named) >= 10, f"the docstring segment parse found too few names: {sorted(named)}"
    for name in sorted(named):
        hook = getattr(docmod.MasonDoc, name, None)
        assert callable(hook), (
            f"{name!r} is named in edits.py's docstring as a document hook "
            "but does not resolve to a callable on MasonDoc"
        )
