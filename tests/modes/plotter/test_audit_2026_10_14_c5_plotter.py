"""The 2026-10-14 audit's Plotter cleanup round (fixer C5): Esc, the object ceiling, the Layer menu.

One test per claim. Each was run against the pre-fix source loaded as a
throwaway module, or fails by construction where the symbol did not exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.grid2d import gid
from realmspinner.studio.modes.plotter import mode as plotter_mode
from realmspinner.studio.modes.plotter.engine import _map_objects, rmap
from realmspinner.studio.modes.plotter.engine.tilemap import MapDoc, MapObject, new_uid
from realmspinner.studio.modes.plotter.ui.panes import menu as plotter_menu

from .test_plotter_mode import FakeCtx, _key, _stamping, _tab

# --- plotter-32: Esc abandons a drag, it does not commit it ------------------------


def test_escape_during_a_paint_drag_leaves_no_step_behind():
    ctx = FakeCtx()
    tab, state, layer = _stamping(ctx)
    before = np.array(layer.data)
    depth = len(tab.doc.history)

    tab.doc.begin_stroke(layer.uid)
    tab.doc.stroke_write(0, 0, np.array([[1, 1], [1, 1]], gid.DTYPE))
    state.drag_kind = "paint"

    assert plotter_mode.handle_key(ctx, _key("ESCAPE")) is True

    assert not tab.doc.stroking
    assert np.array_equal(layer.data, before), "the half stroke stayed in the layer"
    # What the canvas's idle-frame recovery would have committed.
    assert tab.doc.end_stroke() is False
    assert len(tab.doc.history) == depth


def test_escape_during_an_object_drag_puts_the_object_back_and_pushes_nothing():
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    layer = tab.doc.add_object_layer()
    obj = tab.doc.add_object(layer.uid, MapObject(uid=new_uid(), name="a", x=4.0, y=6.0))
    depth = len(tab.doc.history)

    tab.doc.begin_object_edit(layer.uid, obj.uid)
    tab.doc.place_object(x=50.0, y=60.0)
    state.drag_kind = "object-move"

    plotter_mode.handle_key(ctx, _key("ESCAPE"))

    assert (obj.x, obj.y) == (4.0, 6.0)
    assert not tab.doc.editing_object
    assert len(tab.doc.history) == depth


def test_escape_during_a_group_drag_puts_every_member_back_and_pushes_nothing():
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    layer = tab.doc.add_object_layer()
    first = tab.doc.add_object(layer.uid, MapObject(uid=new_uid(), name="a", x=1.0, y=2.0))
    second = tab.doc.add_object(layer.uid, MapObject(uid=new_uid(), name="b", x=3.0, y=4.0))
    depth = len(tab.doc.history)

    tab.doc.begin_group_edit(layer.uid, [first.uid, second.uid])
    tab.doc.move_group(10.0, 10.0)
    state.drag_kind = "group"

    plotter_mode.handle_key(ctx, _key("ESCAPE"))

    assert [(o.x, o.y) for o in (first, second)] == [(1.0, 2.0), (3.0, 4.0)]
    assert not tab.doc.editing_group
    assert len(tab.doc.history) == depth


# --- plotter-11: one object ceiling, read by every door ----------------------------


def test_the_reader_and_the_editing_doors_share_one_object_ceiling():
    assert rmap.MAX_OBJECTS is _map_objects.MAX_OBJECTS


def test_add_object_refuses_past_the_ceiling_read_rmap_reopens(monkeypatch):
    monkeypatch.setattr(_map_objects, "MAX_OBJECTS", 3)
    doc = MapDoc(2, 2, 16, 16)
    layer = doc.add_object_layer()
    for _ in range(3):
        doc.add_object(layer.uid, MapObject(uid=new_uid(), name="o"))
    depth = len(doc.history)
    next_id = doc.next_object_id

    with pytest.raises(ValueError, match="3 is the most"):
        doc.add_object(layer.uid, MapObject(uid=new_uid(), name="one too many"))

    assert len(layer.objects) == 3
    assert len(doc.history) == depth, "a refusal must not leave a step"
    assert doc.next_object_id == next_id, "a refusal must not burn an id"


def test_the_ceiling_counts_objects_across_every_object_layer(monkeypatch):
    monkeypatch.setattr(_map_objects, "MAX_OBJECTS", 2)
    doc = MapDoc(2, 2, 16, 16)
    one, two = doc.add_object_layer(), doc.add_object_layer()
    doc.add_object(one.uid, MapObject(uid=new_uid(), name="a"))
    doc.add_object(two.uid, MapObject(uid=new_uid(), name="b"))
    with pytest.raises(ValueError):
        doc.add_object(one.uid, MapObject(uid=new_uid(), name="c"))


def test_pasting_an_object_past_the_ceiling_toasts_and_adds_nothing(monkeypatch):
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    layer = tab.doc.add_object_layer()
    tab.doc.set_active_layer(layer.uid)
    source = tab.doc.add_object(layer.uid, MapObject(uid=new_uid(), name="a"))
    monkeypatch.setattr(_map_objects, "MAX_OBJECTS", 1)
    state.clipboard = type("Clip", (), {"obj": source})()
    state.clipboard_doc = tab.uid

    plotter_mode._paste_object(ctx, state, tab)

    assert len(layer.objects) == 1
    assert ctx.toasts and "most this build keeps" in ctx.toasts[-1][0]


# --- plotter-34: the Layer menu's Delete reads the whole tree ----------------------


def test_the_layer_menu_delete_row_is_not_refused_for_a_layer_in_the_only_group(monkeypatch):
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    group = doc.add_group_layer()
    inner = doc.add_tile_layer(parent_uid=group.uid)
    doc.set_active_layer(inner.uid)
    # The map holds the group's layers and the starting one; remove the root
    # sibling so the group is the only root entry, which is what the root-only
    # count mistook for "the last layer".
    for entry in list(doc.layers):
        if entry.uid != group.uid:
            doc.remove_layer(entry.uid)
    assert len(doc.layers) == 1 and len(doc.all_layers()) > 1

    rows: dict[str, dict] = {}

    def fake_row(label, shortcut=None, **kwargs):
        rows[label] = kwargs
        return False

    monkeypatch.setattr(plotter_menu, "_row", fake_row)
    monkeypatch.setattr(plotter_menu.controls, "menu_separator", lambda: None)
    plotter_menu._layer_rows(ctx, plotter_mode.ensure(ctx), tab)

    assert rows["Delete layer"]["enabled"] is True
    assert rows["Delete layer"]["reason"] == ""
