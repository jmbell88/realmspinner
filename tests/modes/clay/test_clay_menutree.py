"""The Clay menu tree: every op has one home, and every surface reads it.

``ops.OPS`` says what Clay can do; ``menutree`` says how it groups. The risk is
the one the registry exists to remove -- a second list that nobody updates -- so
the claims here are about *completeness in both directions*: no registered op
missing from the tree, no name in the tree that is not an op, and no mode whose
menu strip leaves an op unreachable.
"""

from __future__ import annotations

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import menutree
from realmspinner.studio.modes.clay import ops as clay_ops


class _Ctx:
    def toast(self, message: str, level: str = "info") -> None:  # pragma: no cover
        raise AssertionError(message)


def _doc(count: int = 3) -> bd.ClayDoc:
    doc = bd.ClayDoc()
    for index in range(count):
        doc.add_object(
            bd.Obj(
                uid=bd.new_uid(),
                name=f"Box{index}",
                mesh=bp.box(),
                generator="box",
                params={"size": (1.0, 1.0, 1.0)},
            )
        )
    return doc


def test_every_registered_op_is_in_exactly_one_group() -> None:
    placed = [name for group in menutree.GROUPS.values() for name in group.ops]
    assert len(placed) == len(set(placed)), "an op is listed in two groups"
    assert set(placed) == {op.name for op in clay_ops.OPS}


def test_every_bar_entry_names_a_real_group() -> None:
    for mode, menus in menutree.BARS.items():
        assert mode in clay_ops.ALL_MODES
        for menu in menus:
            for entry in menu.entries:
                if entry.kind != "add":
                    assert entry.name in menutree.GROUPS, (mode, menu.title, entry)


def test_every_op_is_reachable_from_the_strip_or_an_off_bar_group_in_each_mode() -> None:
    for mode in clay_ops.ALL_MODES:
        shown = {
            op.name
            for menu in menutree.resolve(mode)
            for section in menu.sections
            for op in section.ops
        }
        off_bar = {
            op.name
            for group in menutree.OFF_BAR_GROUPS
            for op in menutree.grouped_ops(group, mode)
        }
        expected = {op.name for op in clay_ops.menu(mode)}
        assert shown | off_bar == expected, mode
        assert not shown & off_bar


def test_the_flat_context_list_is_exactly_the_modes_ops_with_delete_last() -> None:
    for mode in clay_ops.ALL_MODES:
        names = [op.name for run in menutree.flat(mode) for op in run]
        assert sorted(names) == sorted(op.name for op in clay_ops.menu(mode)), mode
        assert len(names) == len(set(names)), "an op was drawn twice"
        assert names[-1] == "delete"
        assert all(run for run in menutree.flat(mode)), "an empty run draws a bare rule"


def test_no_menu_in_a_resolved_strip_is_empty() -> None:
    for mode in clay_ops.ALL_MODES:
        for menu in menutree.resolve(mode):
            assert menu.sections, (mode, menu.title)


def test_object_mode_gets_a_select_menu_and_vertex_mode_has_no_uv_menu() -> None:
    object_titles = [menu.title for menu in menutree.resolve("object")]
    assert object_titles[:2] == ["Select", "Add"]
    assert "Object" in object_titles and "UV" in object_titles
    vertex_titles = [menu.title for menu in menutree.resolve("vertex")]
    assert "UV" not in vertex_titles, "vertex mode has no UV op to hold"
    assert {"Mesh", "Vertex"} <= set(vertex_titles)
    assert "Object" not in vertex_titles


def test_each_element_mode_gets_its_own_menu_and_not_the_others() -> None:
    for mode, own in (("vertex", "Vertex"), ("edge", "Edge"), ("face", "Face")):
        titles = {menu.title for menu in menutree.resolve(mode)}
        assert own in titles
        assert titles.isdisjoint({"Vertex", "Edge", "Face"} - {own}), mode


def test_shading_is_under_object_and_under_face_never_next_to_duplicate() -> None:
    """The row this regrouping exists for: Shade Smooth was ``separator_before``
    under Duplicate, which is a position, not a reason."""
    basic = menutree.GROUPS["basic"].ops
    assert "shade-smooth" not in basic
    assert menutree.GROUPS["shading"].ops[:2] == ("shade-smooth", "shade-flat")
    face = {
        s.label: s
        for menu in menutree.resolve("face")
        for s in menu.sections
    }
    assert "Shading" in face


def test_the_three_unwraps_share_the_uv_group_ahead_of_packing() -> None:
    """Smart Unwrap was seven rows from Box Unwrap, past Decimate and Retopologize."""
    uv = menutree.GROUPS["uv"].ops
    unwraps = ("unwrap", "unwrap-seams", "smart-unwrap")
    assert all(name in uv for name in unwraps)
    assert max(uv.index(name) for name in unwraps) < uv.index("pack-uv")


def test_colliders_are_a_named_collider_submenu_in_the_add_menu() -> None:
    add = next(m for m in menutree.resolve("object") if m.title == "Add")
    collider = next(s for s in add.sections if s.label == "Collider")
    assert collider.submenu
    assert [op.name for op in collider.ops] == list(menutree.GROUPS["collider"].ops)


def test_select_all_none_and_invert_now_run_in_object_mode() -> None:
    for name in ("select-all", "select-none", "select-invert"):
        assert "object" in clay_ops.get(name).modes, name
    # The element-only verbs stay element-only: there is no object "linked".
    for name in ("select-linked", "select-more", "select-less", "select-boundary"):
        assert "object" not in clay_ops.get(name).modes, name


def test_select_none_in_object_mode_clears_the_object_selection() -> None:
    doc = _doc()
    doc.select([obj.uid for obj in doc.objects])
    none = clay_ops.get("select-none")
    assert none.enabled(doc), "greyed although three objects are selected"
    clay_ops.run(_Ctx(), doc, none)
    assert not doc.selection
    assert not none.enabled(doc)
    assert none.reason(doc)


def test_select_all_and_invert_in_object_mode_take_the_visible_objects() -> None:
    doc = _doc()
    clay_ops.run(_Ctx(), doc, clay_ops.get("select-all"))
    assert doc.selection == {obj.uid for obj in doc.objects}
    clay_ops.run(_Ctx(), doc, clay_ops.get("select-invert"))
    assert not doc.selection


def test_ops_no_longer_carry_a_positional_separator_flag() -> None:
    assert not hasattr(clay_ops.get("duplicate"), "separator_before")
