"""Does Clay's viewport header fit, and does it hold what it claims to.

The one question a header can silently get wrong is whether it fits at the
window the app opens at. ``toolbar`` degrades rather than clips, so a bar that
does not fit does not *look* broken -- it looks like a bar whose every label is
a hover away, on the machine everybody uses. That is the failure this file
exists for, and it is measured rather than eyeballed.
"""

from __future__ import annotations

import pytest

from realmspinner.studio import toolbar
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay import state as clay_state
from realmspinner.studio.modes.clay.ui.panes import header as clay_header
from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools

# --- the tables ---------------------------------------------------------------


def test_every_element_mode_has_a_short_label():
    """The compact tier swaps the words for these, so a mode missing one would
    be a segment drawn with nothing in it."""
    for mode, _label, _key in clay_tools.MODE_BUTTONS:
        assert clay_header.MODE_SHORT.get(mode), mode
    assert set(clay_header.MODE_SHORT) == {
        mode for mode, _l, _k in clay_tools.MODE_BUTTONS
    }


def test_the_modes_the_header_offers_are_the_modes_the_ops_registry_knows():
    """A mode on the bar that no op declares is a mode with nothing in it."""
    assert {mode for mode, _l, _k in clay_tools.MODE_BUTTONS} == set(clay_ops.ALL_MODES)


def test_every_tool_has_a_glyph_or_falls_back_to_its_initial():
    for key, label, _shortcut in clay_state.TOOLS:
        drawn = clay_tools.TOOL_ICONS.get(key) or label[:1]
        assert drawn, key


def test_the_overlay_rows_name_real_switches():
    """``grid`` is a field and the rest live in ``state.overlays``; the two
    accessors are what let the popover be a loop over one table."""
    state = clay_state.ClayState()
    for key, label, tip in clay_header.OVERLAY_ROWS:
        assert label and tip
        before = clay_header.overlay_value(state, key)
        clay_header.set_overlay(state, key, not before)
        assert clay_header.overlay_value(state, key) is (not before)


def test_the_grid_keeps_its_own_field_rather_than_moving_into_the_dict():
    """One switch, one home. It is wired straight to ``ClayView.show_grid`` and
    a second copy in the dict is two places that can disagree."""
    state = clay_state.ClayState()
    clay_header.set_overlay(state, "grid", False)
    assert state.grid is False
    assert "grid" not in state.overlays


def test_god_light_also_keeps_its_own_field_rather_than_moving_into_the_dict():
    """Task C's row: a ``ClayState`` field like ``grid``, not a third home in
    ``state.overlays`` -- the same reason ``grid`` gets one."""
    state = clay_state.ClayState()
    clay_header.set_overlay(state, "god_light", True)
    assert state.god_light is True
    assert "god_light" not in state.overlays


def test_the_overlay_popup_has_a_grid_size_field():
    """Task A: the size field sits under the Grid row, not as a fifth item in
    ``OVERLAY_ROWS`` -- it is not a switch, it is the number the Grid switch
    governs, the same shape ``_snap_popup``'s grid step has."""
    assert hasattr(clay_header, "_grid_size_field")
    keys = [key for key, _label, _tip in clay_header.OVERLAY_ROWS]
    assert keys.index("grid") < keys.index("god_light")


def test_the_grid_tooltip_says_1m_cells_not_the_snap_size():
    """The row used to say "at the snap size", which stopped being true the
    moment the grid got its own ``grid_size`` field independent of Snap."""
    tips = {key: tip for key, _label, tip in clay_header.OVERLAY_ROWS}
    assert "1 m" in tips["grid"]
    assert "snap" not in tips["grid"].lower()


def test_every_axis_row_names_a_view_the_camera_has():
    from realmspinner.studio.viewer.camera import Camera

    for name, label, chord in clay_header.AXIS_ROWS:
        assert name in Camera.AXIS_VIEWS, name
        assert label and chord.startswith("Ctrl+")
    assert {name for name, _l, _c in clay_header.AXIS_ROWS} == set(Camera.AXIS_VIEWS), (
        "all six, not the three that had buttons -- the backs were reachable "
        "only by holding Shift, which nothing said"
    )


# The fit itself needs a live imgui context to measure a font in, and this
# file must not build one: two imgui contexts over the one GL context crash
# the process when they overlap, which is why every context in the suite is
# per-file and torn down with it. The three width tests live in
# ``tests/studio/test_studio_smoke.py``, which already owns one.


@pytest.mark.parametrize("popup", ["SNAP_POPUP", "OVERLAYS_POPUP", "VIEW_POPUP"])
def test_every_popup_has_its_own_name(popup):
    """Two popups sharing a name is one popup that opens when either is asked
    for, which imgui reports as neither working."""
    names = {
        getattr(clay_header, key)
        for key in ("SNAP_POPUP", "OVERLAYS_POPUP", "VIEW_POPUP")
    }
    assert len(names) == 3
    assert getattr(clay_header, popup) in names


def _tab(doc=None):
    from types import SimpleNamespace

    from realmspinner.kernels.mesh import document as bd

    return SimpleNamespace(doc=doc or bd.ClayDoc(), saving=False)


def test_the_mode_pill_is_a_field_rather_than_an_item():
    """Items collapse into the overflow menu first, and a mode picker in a menu
    is a mode picker nobody can see the state of -- which is the one thing a
    mode picker is for."""
    state = clay_state.ClayState()
    tab = _tab()
    keys = {item.key for item in clay_header._items(state, tab)}
    assert "mode" not in keys
    field = clay_header._mode_field(tab)
    assert isinstance(field, toolbar.Field)
    assert field.priority == 0


def test_the_tool_pill_left_the_header_for_the_rail():
    """Select / Move / Rotate / Scale are the tool rail's now. A second pill for
    the same ``state.tool`` was two controls for one state, and its width is
    what Undo and Redo took."""
    assert not hasattr(clay_header, "_tool_field")
    assert "tool" not in {item.key for item in clay_header._items(clay_state.ClayState(), _tab())}


def test_the_mode_pill_is_in_the_order_of_its_keys():
    """Vertex, Edge, Face, Object are keys 1 2 3 4. The pill led with Object, so
    counting its segments left to right read "4 1 2 3"."""
    keys = [key for _mode, _label, key in clay_tools.MODE_BUTTONS]
    assert keys == ["1", "2", "3", "4"]
    assert [mode for mode, _l, _k in clay_tools.MODE_BUTTONS] == [
        "vertex", "edge", "face", "object",
    ]


def test_the_header_has_undo_and_redo_enabled_from_the_documents_history():
    """The history lived three panes away in the Document tab, so the most-used
    control in a modeller was reachable by a chord alone. The items read
    ``tab.doc.history`` -- the stack the chord drives -- so their state cannot
    disagree with what Ctrl+Z would do."""
    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import primitives as bp

    state = clay_state.ClayState()
    doc = bd.ClayDoc()
    tab = _tab(doc)

    def items():
        return {item.key: item for item in clay_header._items(state, tab)}

    assert "undo" in items() and "redo" in items()
    assert not items()["undo"].enabled and items()["undo"].reason
    assert not items()["redo"].enabled and items()["redo"].reason

    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    assert items()["undo"].enabled and not items()["redo"].enabled
    assert doc.undo()
    assert not items()["undo"].enabled and items()["redo"].enabled

    doc.redo()
    tab.saving = True
    assert not items()["undo"].enabled and "written" in items()["undo"].reason, (
        "a click must not mutate the stack under a running encode"
    )
