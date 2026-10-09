"""The Clay menu tree: every op has one home, and every surface reads it.

``ops.OPS`` says what Clay can do; ``menutree`` says how it groups. The risk is
the one the registry exists to remove -- a second list that nobody updates -- so
the claims here are about *completeness in both directions*: no registered op
missing from the tree, no name in the tree that is not an op, and no mode whose
menu leaves an op unreachable. (The Select | Add | Object | Mesh | UV strip that
drew the bars is gone; the right-click menu, the Inspector and Ctrl+K read them now,
and ``test_clay_no_menu_strip.py`` checks every op is reachable from them.)
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


def _bar_ops(mode: str) -> set[str]:
    """Every op name the bar for *mode* names, as the Select/Object/Mesh/UV menus did."""
    return {
        op.name
        for menu in menutree.BARS[mode]
        for entry in menu.entries
        for op in menutree.grouped_ops(entry.name, mode)
    }


def test_every_bar_entry_names_a_real_group() -> None:
    for mode, menus in menutree.BARS.items():
        assert mode in clay_ops.ALL_MODES
        for menu in menus:
            for entry in menu.entries:
                assert entry.name in menutree.GROUPS, (mode, menu.title, entry)
                assert entry.kind in ("sub", "inline"), (mode, menu.title, entry)


def test_every_op_is_reachable_from_the_bar_or_an_off_bar_group_in_each_mode() -> None:
    for mode in clay_ops.ALL_MODES:
        shown = _bar_ops(mode)
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


def test_no_menu_in_a_bar_is_empty_for_its_mode() -> None:
    """A menu the bar names holds at least one op that works in that mode, so a
    surface drawing the bar never draws a bare heading (the UV menu is simply not in
    the element modes' bars, since both UV ops are object-level)."""
    for mode, menus in menutree.BARS.items():
        for menu in menus:
            held = [
                op
                for entry in menu.entries
                for op in menutree.grouped_ops(entry.name, mode)
            ]
            assert held, (mode, menu.title)


def test_object_mode_has_select_object_and_uv_and_vertex_mode_has_no_uv() -> None:
    object_titles = [menu.title for menu in menutree.BARS["object"]]
    assert object_titles == ["Select", "Object", "UV"]
    vertex_titles = [menu.title for menu in menutree.BARS["vertex"]]
    assert "UV" not in vertex_titles, "vertex mode has no UV op to hold"
    assert "Mesh" in vertex_titles
    assert "Object" not in vertex_titles


def test_the_three_element_modes_share_one_mesh_menu_with_no_vertex_edge_or_face_menu() -> None:
    for mode in ("vertex", "edge", "face"):
        titles = [menu.title for menu in menutree.BARS[mode]]
        assert titles == ["Select", "Mesh"], mode


def test_the_add_menu_and_its_resolver_left_with_the_strip() -> None:
    """Old code: ``BARS`` carried an Add menu whose ``Entry("add", ...)`` rows only the
    deleted strip drew, and ``resolve`` existed to hand that strip its sections."""
    assert not hasattr(menutree, "resolve") and not hasattr(menutree, "ADD_SOURCES")
    for mode, menus in menutree.BARS.items():
        assert "Add" not in [menu.title for menu in menus], mode


def test_shading_is_under_object_and_under_face_never_next_to_duplicate() -> None:
    """The row this regrouping exists for: Shade Smooth was ``separator_before``
    under Duplicate, which is a position, not a reason."""
    basic = menutree.GROUPS["basic"].ops
    assert "shade-smooth" not in basic
    assert menutree.GROUPS["shading"].ops[:2] == ("shade-smooth", "shade-flat")
    face = {entry.name for menu in menutree.BARS["face"] for entry in menu.entries}
    assert "shading" in face


def test_box_unwrap_shares_the_uv_group_ahead_of_packing() -> None:
    uv = menutree.GROUPS["uv"].ops
    assert uv.index("unwrap") < uv.index("pack-uv")


def test_select_all_none_and_invert_now_run_in_object_mode() -> None:
    for name in ("select-all", "select-none", "select-invert"):
        assert "object" in clay_ops.get(name).modes, name
    # The element-only verbs stay element-only: there is no object "linked".
    assert "object" not in clay_ops.get("select-linked").modes


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
