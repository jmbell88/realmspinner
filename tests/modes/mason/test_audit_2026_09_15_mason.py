"""Regression tests for the 2026-09-15 audit's Mason findings.

- mason-01 (`src/warlock/studio/modes/mason/engine/document.py`): the 2026-09-14 audit's
  mason-01 put a `scene.MAX_PLACED` check in front of `MasonDoc.add_nodes`
  only. `add_node` -- every single placement, looped by
  `mason_mode.duplicate_selected` -- and `unpack_instance` -- which can
  attach a whole template subtree, with `define_prefab` putting no ceiling
  of its own on how big a template may be -- still attached unchecked. Past
  the ceiling, every `scene.walk()`/`resolve()` refuses for good: the scene
  can no longer be drawn or exported. Fixed by routing all three attach
  points through one shared `MasonDoc._check_max_placed`.
- mason-03 (`src/warlock/studio/modes/mason/ui/panes/menu.py`,
  `src/warlock/studio/modes/mason/ui/panes/outliner.py`): the viewport context menu
  enabled "Ungroup" for any selection, while the outliner gated the same
  action on `_groupish` and the shared handler just returned silently for a
  non-group. Fixed by sharing the outliner's predicate.
- mason-04 (`src/warlock/studio/modes/mason/ui/view.py`): `MasonView._terrain` was
  annotated as a 3-tuple but stored 4 fields. Comment/annotation-only, no
  regression test is possible for it; see this module's own note below.
"""

from __future__ import annotations

import pytest

from warlock.studio.modes.mason.engine import document as doc
from warlock.studio.modes.mason.engine import nodes as nd
from warlock.studio.modes.mason.engine import scene as sc

# --- mason-01 ----------------------------------------------------------------


def test_add_node_refuses_before_building_past_max_placed_rather_than_after(monkeypatch):
    """The 2026-09-15 audit's mason-01: unlike ``add_nodes``, ``add_node`` --
    every single placement, looped once per copy by
    ``mason_mode.duplicate_selected`` -- attached with no ``MAX_PLACED``
    check at all. Against the unfixed ``add_node`` this raises nothing and
    both calls below succeed, growing ``d.roots`` to 6 past a ceiling of 5.
    """
    monkeypatch.setattr(sc, "MAX_PLACED", 5)
    d = doc.MasonDoc()
    for _ in range(5):
        d.add_node(nd.GroupNode(uid=nd.new_uid()))
    assert len(d.roots) == 5
    with pytest.raises(ValueError, match="MAX_PLACED"):
        d.add_node(nd.GroupNode(uid=nd.new_uid()))
    # Refused before attaching: the tree and the history did not grow.
    assert len(d.roots) == 5
    assert len(d.history) == 5


def test_add_node_counts_a_duplicated_groups_whole_subtree_not_just_the_group():
    """A single ``add_node`` call can carry a whole subtree (a duplicated
    group with children), so the ceiling has to count the subtree, not just
    the one node handed in -- otherwise a single "duplicate" of a
    thousand-child group would count as 1 against the ceiling.
    """
    from warlock.studio.modes.mason.engine import scene as sc

    monkeypatch_value = sc.MAX_PLACED
    try:
        sc.MAX_PLACED = 3
        d = doc.MasonDoc()
        group = nd.GroupNode(uid=nd.new_uid())
        group.children.append(nd.GroupNode(uid=nd.new_uid()))
        group.children.append(nd.GroupNode(uid=nd.new_uid()))
        group.children.append(nd.GroupNode(uid=nd.new_uid()))
        # group + 3 children == 4 nodes, past a ceiling of 3.
        with pytest.raises(ValueError, match="MAX_PLACED"):
            d.add_node(group)
        assert d.roots == []
    finally:
        sc.MAX_PLACED = monkeypatch_value


def test_unpack_instance_refuses_before_building_past_max_placed_rather_than_after(
    monkeypatch,
):
    """The 2026-09-15 audit's mason-01: ``unpack_instance`` can attach a
    whole template subtree, and ``define_prefab`` puts no ceiling on how big
    a template may be -- so an oversized template met the scene tree
    unchecked. Against the unfixed ``unpack_instance`` this raises nothing
    and the instance is replaced by an oversized copy.
    """
    d = doc.MasonDoc()
    template = nd.GroupNode(uid=nd.new_uid(), name="template")
    template.children.append(nd.GroupNode(uid=nd.new_uid()))
    template.children.append(nd.GroupNode(uid=nd.new_uid()))
    template.children.append(nd.GroupNode(uid=nd.new_uid()))
    d.define_prefab("big", template)  # 4 nodes total, defined with no ceiling check

    instance = nd.PrefabNode(uid=nd.new_uid(), template="big")
    monkeypatch.setattr(sc, "MAX_PLACED", 5)
    d.add_node(instance)  # 1 node placed; template unpack would bring it to 4

    monkeypatch.setattr(sc, "MAX_PLACED", 3)  # now past the ceiling once unpacked
    with pytest.raises(ValueError, match="MAX_PLACED"):
        d.unpack_instance(instance.uid)
    # Refused before attaching: the instance is still there, untouched.
    assert d.node(instance.uid) is instance


def test_unpack_instance_replacing_an_instance_with_a_same_sized_copy_never_refuses(
    monkeypatch,
):
    """A same-size unpack (net growth of zero) must not be caught by the
    ceiling check just because the check is now present -- only *growth*
    past ``MAX_PLACED`` refuses.
    """
    d = doc.MasonDoc()
    template = nd.GroupNode(uid=nd.new_uid(), name="template")
    template.children.append(nd.GroupNode(uid=nd.new_uid()))
    d.define_prefab("thing", template)

    instance = nd.PrefabNode(uid=nd.new_uid(), template="thing")
    d.add_node(instance)

    monkeypatch.setattr(sc, "MAX_PLACED", len(d.all_nodes()) + 1)  # already right at it
    unpacked = d.unpack_instance(instance.uid)  # net growth is 0: 2 nodes -> 2 nodes
    assert unpacked is not None


# --- mason-03 ------------------------------------------------------------------


def _ungroup_enabled_flag(monkeypatch, doc) -> bool:
    """Draw ``mason_menu``'s context menu rows and capture the ``enabled``
    flag its "Ungroup" row passes to ``controls.menu_item`` -- the same
    intercept shape ``test_plotter_mode.py``'s own ``_menu_rows`` uses:
    ``controls.menu_item`` is what a row *is*, so replacing it is how a test
    sees the enabled gate without needing a live imgui frame.
    """
    from types import SimpleNamespace

    from warlock.studio import controls
    from warlock.studio.modes.mason.ui.panes import menu as mason_menu

    captured: dict[str, bool] = {}

    def fake_menu_item(label, shortcut="", selected=False, enabled=True, **kwargs):
        if label == "Ungroup":
            captured["enabled"] = bool(enabled)
        return (False, selected)

    monkeypatch.setattr(controls, "menu_item", fake_menu_item)
    monkeypatch.setattr(controls, "menu_separator", lambda: None)

    tab = SimpleNamespace(doc=doc, saving=False)
    mason_menu._rows(ctx=None, tab=tab)
    assert "enabled" in captured, "no row named 'Ungroup' was drawn"
    return captured["enabled"]


def test_viewport_context_menu_ungroup_is_disabled_without_a_group_selected(monkeypatch):
    """The 2026-09-15 audit's mason-03: the viewport context menu
    (``mason_menu.py``) enabled "Ungroup" off the bare ``selected`` flag --
    true for *any* non-empty selection -- while the handler,
    ``mason_mode.ungroup_selected``, only acts on a group with children and
    returns silently otherwise. Against the unfixed row this returns
    ``True`` for a selection with no group in it at all.
    """
    d = doc.MasonDoc()
    leaf = nd.GroupNode(uid=nd.new_uid())  # a "group" with no children: not groupish
    d.add_node(leaf)
    d.select([leaf.uid])

    assert _ungroup_enabled_flag(monkeypatch, d) is False


def test_viewport_context_menu_ungroup_is_enabled_for_a_group_with_children(monkeypatch):
    """The positive case for the same row: once ``mason_outliner.groupish``
    is true, the viewport menu's row must still enable, not just the
    outliner's own copy of this predicate."""
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid())
    group.children.append(nd.GroupNode(uid=nd.new_uid()))
    d.add_node(group)
    d.select([group.uid])

    assert _ungroup_enabled_flag(monkeypatch, d) is True


# --- mason-04 ------------------------------------------------------------------

# mason-04 (``MasonView._terrain``, ``src/warlock/studio/mason_view.py``) is
# comment/annotation-only: the slot was annotated as a 3-tuple but always
# stored 4 fields, and nothing type-checks a runtime annotation string, so no
# regression test can fail against the unfixed code. Fixed by correcting the
# annotation text to match what is actually stored; see the diff in
# ``mason_view.py`` instead of a test here.
