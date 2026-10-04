"""The 2026-10-03 audit's Low findings inker-99 and inker-101.

Each test's name is the claim. inker-99 was already fixed in
``Frame.__setattr__`` (the test pins it); inker-101 is an evidence gap (pure
pane decisions no test named), so those tests are the fix and pass from the
start.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels.pixel import animation as anim_mod
from realmspinner.kernels.pixel.document import Document
from realmspinner.kernels.pixel.sheetin import document_from_sheet
from realmspinner.studio.modes.inker import state as inker_state
from realmspinner.studio.modes.inker.ui.panes import canvas as inker_canvas
from realmspinner.studio.modes.inker.ui.panes import context as inker_context
from realmspinner.studio.modes.inker.ui.panes import slices as inker_slices
from realmspinner.studio.modes.inker.ui.panes import tools as inker_tools
from realmspinner.studio.modes.inker.ui.panes import walk as inker_walk
from realmspinner.studio.shell import paintview
from realmspinner.studio.tokens import sp

# --- inker-99 ----------------------------------------------------------------


def _sheet(durations):
    cell = 4
    atlas = np.zeros((cell, cell * 3, 4), dtype=np.uint8)
    atlas[..., 3] = 255
    cells = [{"x": i * cell, "y": 0, "w": cell, "h": cell} for i in range(3)]
    frames = [{"cell_index": i, "duration_ms": d} for i, d in enumerate(durations)]
    anim = {"tags": [{"name": "walk_front", "start": 0, "end": 2, "loop": True}], "frames": frames}
    return document_from_sheet(atlas, cells, anim)


def test_document_from_sheet_clamps_a_sidecar_duration():
    """A hand-edited or foreign sidecar's duration must not install a value past
    the MIN/MAX range. Already held by ``Frame.__setattr__`` (which clamps every
    write), so this pins the claim rather than failing against old code."""
    doc = _sheet([10**9, 80, 3])
    got = [frame.duration_ms for frame in doc.anim.frames]
    assert got[0] == anim_mod.MAX_DURATION_MS
    assert got[1] == 80
    assert got[2] == 3
    assert all(anim_mod.MIN_DURATION_MS <= d <= anim_mod.MAX_DURATION_MS for d in got)


def test_document_from_sheet_keeps_the_default_for_a_missing_or_zero_duration():
    doc = _sheet([0, None, 40])
    # Zero and None were skipped before the fix and must still be; the clamp
    # must not turn "no duration" into the one-millisecond floor.
    assert doc.anim.frames[0].duration_ms == anim_mod.DEFAULT_DURATION_MS
    assert doc.anim.frames[1].duration_ms == anim_mod.DEFAULT_DURATION_MS


# --- inker-101: slices._nudged / _slice_grab ----------------------------------


def test_slice_nudge_moves_only_the_named_corner():
    rect = (10, 20, 30, 40)
    assert inker_slices._nudged(rect, "nw", 1, 2) == (11, 22, 30, 40)
    assert inker_slices._nudged(rect, "ne", 1, 2) == (10, 22, 31, 40)
    assert inker_slices._nudged(rect, "sw", 1, 2) == (11, 20, 30, 42)
    assert inker_slices._nudged(rect, "se", 1, 2) == (10, 20, 31, 42)
    # The corner is allowed to cross the far one: ordering is release's job.
    assert inker_slices._nudged(rect, "nw", 50, 0) == (60, 20, 30, 40)
    assert rect == (10, 20, 30, 40), "the input is not mutated"


def _grab_tab(**view):
    doc = Document.blank(64, 64)
    tab = inker_state.InkerDoc(doc=doc, views=[paintview.PaintView(**view)])
    state = inker_state.InkerState(tool="slice")
    state.add(tab)
    return state, tab


def _mouse(monkeypatch, x, y):
    monkeypatch.setattr(
        inker_slices.imgui, "get_mouse_pos", lambda: SimpleNamespace(x=x, y=y)
    )


ORIGIN = (0.0, 0.0)


def test_slice_grab_hit_order_is_corner_then_pivot_then_centre_then_body(monkeypatch):
    state, tab = _grab_tab(zoom=10.0, pan=(0.0, 0.0))
    entry = tab.doc.add_slice((10, 10, 30, 30), pivot=(0.0, 0.0), center=(5, 5, 15, 15))
    state.slice_uid = entry.uid

    # Pivot parked exactly on the nw corner: resizing wins at a coincident grab.
    _mouse(monkeypatch, 100.0, 100.0)
    assert inker_slices._slice_grab(state, tab, ORIGIN, (10, 10)) == ("slice-resize", "nw")

    # Move the pivot away: the same press now only reaches the pivot where it is.
    entry.pivot = (10.0, 10.0)
    _mouse(monkeypatch, 200.0, 200.0)
    assert inker_slices._slice_grab(state, tab, ORIGIN, (20, 20)) == ("slice-pivot", "")

    # The nine-slice centre's corner, well away from a bounds corner and the pivot.
    _mouse(monkeypatch, 150.0, 150.0)
    assert inker_slices._slice_grab(state, tab, ORIGIN, (15, 15)) == ("slice-center", "nw")

    # Inside the body, away from every handle: a move, selecting what was hit.
    _mouse(monkeypatch, 280.0, 160.0)
    assert inker_slices._slice_grab(state, tab, ORIGIN, (28, 16)) == ("slice-move", "")
    assert state.slice_uid == entry.uid

    # Empty canvas starts a new slice.
    _mouse(monkeypatch, 500.0, 500.0)
    assert inker_slices._slice_grab(state, tab, ORIGIN, (50, 50)) == ("slice-new", "")


def test_slice_grab_body_hit_goes_to_the_slice_drawn_on_top(monkeypatch):
    state, tab = _grab_tab(zoom=10.0, pan=(0.0, 0.0))
    below = tab.doc.add_slice((0, 0, 40, 40))
    above = tab.doc.add_slice((10, 10, 30, 30))
    state.slice_uid = 0  # nothing selected, so no handle can win
    _mouse(monkeypatch, 200.0, 200.0)
    assert inker_slices._slice_grab(state, tab, ORIGIN, (20, 20)) == ("slice-move", "")
    assert state.slice_uid == above.uid
    # Outside the upper one, only the lower one is under the cursor.
    _mouse(monkeypatch, 50.0, 50.0)
    assert inker_slices._slice_grab(state, tab, ORIGIN, (5, 5)) == ("slice-move", "")
    assert state.slice_uid == below.uid


def test_slice_grab_radius_is_in_screen_space_not_image_space(monkeypatch):
    """At 100x a corner's grab radius is still ``SLICE_HANDLE * SLICE_GRAB``
    screen pixels, not 25 image pixels."""
    state, tab = _grab_tab(zoom=100.0, pan=(0.0, 0.0))
    entry = tab.doc.add_slice((1, 1, 20, 20))
    state.slice_uid = entry.uid
    radius = sp(inker_slices.SLICE_HANDLE) * inker_slices.SLICE_GRAB
    _mouse(monkeypatch, 100.0 + radius - 1.0, 100.0)
    assert inker_slices._slice_grab(state, tab, ORIGIN, (1, 1))[0] == "slice-resize"
    _mouse(monkeypatch, 100.0 + radius + 1.0, 100.0)
    assert inker_slices._slice_grab(state, tab, ORIGIN, (1, 1))[0] != "slice-resize"


# --- inker-101: canvas._group_shown -------------------------------------------


def _stack(fold, active):
    return SimpleNamespace(stack=SimpleNamespace(group_fold=fold, active_index=active))


def test_group_shown_reads_the_compositors_fold_not_the_layers_own_eye():
    assert inker_canvas._group_shown(_stack(None, 0)) is True
    assert inker_canvas._group_shown(_stack([(True, 1.0), (False, 1.0)], 0)) is True
    assert inker_canvas._group_shown(_stack([(True, 1.0), (False, 1.0)], 1)) is False


def test_group_shown_treats_a_fold_caught_mid_rebuild_as_shown():
    assert inker_canvas._group_shown(_stack([(False, 1.0)], 3)) is True


# --- inker-101: canvas._footprint_box -----------------------------------------


def _brush_state(*, size, nib, tip=None):
    return SimpleNamespace(
        tool="pencil", brush_size=size, nib=nib, tip_for=lambda tool: tip
    )


def test_footprint_box_is_none_for_a_one_pixel_brush():
    assert inker_canvas._footprint_box(_brush_state(size=1, nib="pixel"), None, (5, 5)) is None


def test_footprint_box_odd_pixel_nib_is_centred_and_even_grows_down_and_right():
    odd = inker_canvas._footprint_box(_brush_state(size=3, nib="pixel"), None, (5, 5))
    assert odd == (4, 4, 7, 7)
    even = inker_canvas._footprint_box(_brush_state(size=2, nib="pixel"), None, (5, 5))
    assert even == (5, 5, 7, 7)


def test_footprint_box_soft_nib_uses_the_rounding_anchor():
    box = inker_canvas._footprint_box(_brush_state(size=4, nib="soft"), None, (5, 5))
    # point = (5.5, 5.5); radius 2 -> floor(5.5 - 2 + 0.5) = 4.
    assert box == (4, 4, 8, 8)


def test_footprint_box_for_an_image_tip_is_the_tips_own_size():
    tip = SimpleNamespace(size=(3, 5))
    box = inker_canvas._footprint_box(_brush_state(size=9, nib="soft", tip=tip), None, (10, 10))
    assert box == (9, 8, 12, 13)


# --- inker-101: canvas.symmetry_axes ------------------------------------------


def test_symmetry_axes_resolves_legacy_and_composed_spellings():
    assert inker_canvas.symmetry_axes(SimpleNamespace(symmetry="none")) == ()
    assert inker_canvas.symmetry_axes(SimpleNamespace(symmetry="xy")) == ("x", "y")
    assert inker_canvas.symmetry_axes(SimpleNamespace(symmetry="diag+x")) == ("x", "diag")
    assert inker_canvas.symmetry_axes(SimpleNamespace(symmetry="x+bogus")) == ("x",)


# --- inker-101: tools._group_tool ---------------------------------------------


def test_group_tool_prefers_the_tool_in_hand_then_the_remembered_then_the_first():
    members = ("pencil", "pen", "brush")
    held = SimpleNamespace(tool="pen", group_tool={"draw": "brush"})
    assert inker_tools._group_tool(held, "draw", members) == "pen"
    remembered = SimpleNamespace(tool="fill", group_tool={"draw": "brush"})
    assert inker_tools._group_tool(remembered, "draw", members) == "brush"
    stale = SimpleNamespace(tool="fill", group_tool={"draw": "gone"})
    assert inker_tools._group_tool(stale, "draw", members) == "pencil"
    empty = SimpleNamespace(tool="fill", group_tool={})
    assert inker_tools._group_tool(empty, "draw", members) == "pencil"


# --- inker-101: walk._layer_options -------------------------------------------


def test_walk_layer_options_lists_the_two_specials_then_layers_top_first():
    layers = [
        SimpleNamespace(uid=1, name="bottom"),
        SimpleNamespace(uid=2, name="middle"),
        SimpleNamespace(uid=3, name="top"),
    ]
    tab = SimpleNamespace(doc=SimpleNamespace(stack=SimpleNamespace(layers=layers)))
    assert inker_walk._layer_options(tab) == [
        ("", "Not assigned"),
        ("selection", "From selection"),
        ("3", "top"),
        ("2", "middle"),
        ("1", "bottom"),
    ]


# --- inker-101: context._view_dirty -------------------------------------------

_VIEW_AIDS = ("grid", "grid_snap", "pixel_grid", "layer_edges", "tile_numbers")


@pytest.mark.parametrize("aid", _VIEW_AIDS)
def test_view_dirty_is_true_for_each_view_aid_alone(aid):
    state = SimpleNamespace(**{name: False for name in _VIEW_AIDS})
    assert inker_context._view_dirty(state) is False
    setattr(state, aid, True)
    assert inker_context._view_dirty(state) is True
