"""The 2026-10-14 audit's residue round (fixer R), finding plotter-11.

A layer copy mints an object id per object and does not come through ``add_object``, so
duplicating a layer or a group past ``MAX_OBJECTS`` made a map that saved and could not
be reopened. The ceiling is asked before the copy is made, beside the layer-count one.
"""

from __future__ import annotations

import pytest

from realmspinner.studio.modes.plotter.engine import _map_objects
from realmspinner.studio.modes.plotter.engine.tilemap import MapDoc, MapObject, new_uid


def _layer_with(doc: MapDoc, count: int, *, parent_uid=None):
    layer = doc.add_object_layer(parent_uid=parent_uid)
    for _ in range(count):
        doc.add_object(layer.uid, MapObject(uid=new_uid(), name="o"))
    return layer


def test_duplicating_an_object_layer_past_the_object_ceiling_is_refused_before_the_copy(
    monkeypatch,
):
    monkeypatch.setattr(_map_objects, "MAX_OBJECTS", 3)
    doc = MapDoc(2, 2, 16, 16)
    layer = _layer_with(doc, 2)
    layers = len(doc.all_layers())
    depth = len(doc.history)
    next_object_id = doc.next_object_id

    with pytest.raises(ValueError, match="3 is the most"):
        doc.duplicate_layer(layer.uid)

    assert len(doc.all_layers()) == layers
    assert len(doc.history) == depth, "a refusal must not leave a step"
    assert doc.next_object_id == next_object_id, "a refusal must not burn object ids"


def test_duplicating_a_group_counts_the_objects_of_every_layer_in_its_subtree(monkeypatch):
    monkeypatch.setattr(_map_objects, "MAX_OBJECTS", 5)
    doc = MapDoc(2, 2, 16, 16)
    group = doc.add_group_layer()
    _layer_with(doc, 2, parent_uid=group.uid)
    _layer_with(doc, 1, parent_uid=group.uid)
    # Three objects held, three more in the copy: one past the ceiling.
    with pytest.raises(ValueError, match="5 is the most"):
        doc.duplicate_layer(group.uid)


def test_duplicating_within_the_object_ceiling_still_copies(monkeypatch):
    monkeypatch.setattr(_map_objects, "MAX_OBJECTS", 4)
    doc = MapDoc(2, 2, 16, 16)
    layer = _layer_with(doc, 2)

    copy = doc.duplicate_layer(layer.uid)

    assert len(copy.objects) == 2
    assert doc.object_total() == 4
