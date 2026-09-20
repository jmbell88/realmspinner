"""Tranche 3 (scene structure): the element HUD's measure readout.

``viewport_hints.measure_line`` is the pure half (``hud.hint_line`` only
draws it); every test here is against a unit box (``bp.box()``, extents
1x1x1, centred on the origin), whose numbers are exact by construction --
adjacent vertices are 1 m apart, three of them meet at a right angle, one
face is 1 m^2 and the whole box is 1 m^3 -- so every assertion below is an
equality, not a tolerance.
"""

from __future__ import annotations

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import viewport_hints as clay_hints


def _doc_with_box() -> tuple[bd.ClayDoc, bd.Obj]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    return doc, obj


def test_object_mode_with_nothing_selected_shows_nothing():
    doc, _obj = _doc_with_box()
    assert clay_hints.measure_line(doc) == ""


def test_vertex_mode_with_one_vertex_shows_nothing():
    """Only exactly two or exactly three is a shape this readout answers."""
    doc, obj = _doc_with_box()
    doc.set_element_mode("vertex")
    doc.set_element_sel(obj.uid, el.ElementSel(verts=[0]))
    assert clay_hints.measure_line(doc) == ""


def test_two_selected_vertices_show_distance():
    doc, obj = _doc_with_box()
    doc.set_element_mode("vertex")
    doc.set_element_sel(obj.uid, el.ElementSel(verts=[0, 1]))

    assert clay_hints.measure_line(doc) == "distance  1.0000 m"


def test_three_selected_vertices_show_the_angle_at_the_middle_one():
    """Vertices 0, 1, 2 of ``primitives.box()`` meet at a right angle: 0->1 is
    +X, 1->2 is +Z, so the angle *at* vertex 1 is 90 degrees."""
    doc, obj = _doc_with_box()
    doc.set_element_mode("vertex")
    doc.set_element_sel(obj.uid, el.ElementSel(verts=[0, 1, 2]))

    assert clay_hints.measure_line(doc) == "angle  90.00°"


def test_a_face_selection_shows_area():
    doc, obj = _doc_with_box()
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=[0]))

    assert clay_hints.measure_line(doc) == "area  1.0000 m²"


def test_an_object_selection_shows_volume():
    doc, obj = _doc_with_box()
    doc.select([obj.uid])

    assert clay_hints.measure_line(doc) == "volume  1.0000 m³"


def test_distance_is_measured_in_world_space_through_a_parent():
    """Tranche 3: scene structure. A scaled *parent* must change the
    reported distance between two of the child's vertices -- proving this
    reads ``doc.world_matrix`` and not the child's own local mesh alone.

    The child's own local edge is 1 m (an unscaled unit box, vertices 0 and
    1); it is reparented onto an identity parent with ``keep_world=True``, so
    its local ``scale`` of 2 survives untouched, and the parent is *then*
    scaled by 3 -- world scale 6, so the same local 1 m edge measures 6 m.
    """
    doc = bd.ClayDoc()
    parent = doc.add_object(bd.Obj(uid=bd.new_uid(), name="P", mesh=bp.box()))
    child = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="C", mesh=bp.box(), scale=m3.vec3(2.0, 2.0, 2.0))
    )
    doc.set_parent(child.uid, parent.uid, keep_world=True)
    doc.set_transform(parent.uid, scale=(3.0, 3.0, 3.0))

    doc.set_element_mode("vertex")
    doc.set_element_sel(child.uid, el.ElementSel(verts=[0, 1]))

    assert clay_hints.measure_line(doc) == "distance  6.0000 m"


def test_volume_sums_every_selected_object():
    doc, first = _doc_with_box()
    second = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    doc.select([first.uid, second.uid])

    assert clay_hints.measure_line(doc) == "volume  2.0000 m³"
