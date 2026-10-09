"""Clicks and sweeps in an element mode, under the selection rule.

The shape being edited stays selected, so a click on empty space or a bare
surface clears *picks* and never ends the edit, a plain click on an element of
another shape makes that shape the one being edited, and Shift adds a shape to
the edit. Rides the session ``gl`` fixture like ``test_clay_input_model``.
"""

from __future__ import annotations

import pygame
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay.ui import view as clay_view

from .test_clay_input_model import CORNER, RECT, _Ctx, _hold, _send


@pytest.fixture
def _mods(monkeypatch):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0, raising=False)
    return monkeypatch


@pytest.fixture
def view(gl):
    v = clay_view.ClayView(gl, _Ctx())
    yield v
    v.release()


def _two(mode: str = "face") -> tuple[bd.ClayDoc, bd.Obj, bd.Obj]:
    doc = bd.ClayDoc()
    a = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box(), translation=m3.vec3(-1.2, 0, 0))
    )
    b = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box(), translation=m3.vec3(1.2, 0, 0))
    )
    doc.select([a.uid])
    doc.set_element_mode(mode)
    return doc, a, b


def _ready(view, doc) -> None:
    view.frame_whole(doc)
    view.draw(doc, RECT, 0.0)


def _spot_on(view, doc, uid: int) -> tuple[int, int]:
    """A pixel whose pick lands on an element of *uid* (the nearest hit)."""
    for y in range(8, int(RECT[3]) - 8, 3):
        for x in range(8, int(RECT[2]) - 8, 3):
            hit = view.pick_face(doc, (x, y))
            if hit is not None and hit.uid == uid:
                picked = view.pick_element(doc, (x, y), hit)
                if picked is not None:
                    return (x, y)
    raise AssertionError(f"no pixel picks an element of {uid}")


def _click(view, doc, pos) -> None:
    _send(view, doc, "down", pos)
    _send(view, doc, "up", pos)


def test_a_replace_click_on_an_element_of_another_shape_makes_it_the_one_being_edited(view) -> None:
    doc, a, b = _two()
    doc.set_element_sel(a.uid, el.ElementSel(faces=[0]))
    _ready(view, doc)

    _click(view, doc, _spot_on(view, doc, b.uid))

    assert doc.selection == {b.uid}
    assert set(doc.element_sel) == {b.uid}, "the picks on the shape it replaced went with it"
    assert doc.selection_problems() == []


def test_shift_click_on_an_element_of_another_shape_adds_it_to_the_edit(view, _mods) -> None:
    doc, a, b = _two()
    doc.set_element_sel(a.uid, el.ElementSel(faces=[0]))
    _ready(view, doc)

    _hold(_mods, pygame.KMOD_SHIFT)
    _click(view, doc, _spot_on(view, doc, b.uid))

    assert doc.selection == {a.uid, b.uid}
    assert set(doc.element_sel) == {a.uid, b.uid}


def test_ctrl_click_that_empties_a_shapes_picks_does_not_end_the_edit(view, _mods) -> None:
    doc, a, b = _two()
    _ready(view, doc)
    spot = _spot_on(view, doc, a.uid)
    _click(view, doc, spot)
    assert len(doc.element_sel_of(a.uid).faces) == 1

    _hold(_mods, pygame.KMOD_CTRL)
    _click(view, doc, spot)

    assert doc.element_sel == {}
    assert doc.selection == {a.uid}, "Ctrl never removes a shape from the edit"


def test_a_click_on_empty_space_clears_the_picks_and_keeps_the_shape_selected(view) -> None:
    doc, a, _b = _two()
    doc.set_element_sel(a.uid, el.ElementSel(faces=[0]))
    _ready(view, doc)

    _click(view, doc, CORNER)

    assert doc.element_sel == {}
    assert doc.selection == {a.uid}
    assert doc.element_mode == "face"


def test_a_replace_sweep_makes_the_shapes_it_caught_the_ones_being_edited(view) -> None:
    doc, a, b = _two()
    _ready(view, doc)
    view._grab = "marquee"
    view._marquee_add = "replace"
    # The right half of the view: B's faces and none of A's (they sit left of centre).
    view.marquee = (RECT[2] * 0.5 + 2, 2.0, RECT[2] - 2.0, RECT[3] - 2.0)

    view._release_drag(doc, 1)

    assert set(doc.element_sel) == {b.uid}
    assert doc.selection == {b.uid}, "the shapes that received picks are the edit"
    assert doc.selection_problems() == []


def test_a_replace_sweep_over_nothing_leaves_the_edited_set_alone(view) -> None:
    doc, a, _b = _two()
    _ready(view, doc)
    view._grab = "marquee"
    view._marquee_add = "replace"
    view.marquee = (0.0, 0.0, 6.0, 6.0)  # the corner of the viewport: no geometry

    view._release_drag(doc, 1)

    assert doc.selection == {a.uid}
    assert doc.element_sel == {}


def test_a_gizmo_drag_in_an_element_mode_moves_only_the_shapes_with_picks(view) -> None:
    doc, a, b = _two()
    doc.select([a.uid, b.uid])
    doc.set_element_sel(a.uid, el.ElementSel(faces=[0]))
    _ready(view, doc)

    assert view._begin_gizmo_drag(doc)

    assert list(view._drag_start) == [a.uid]
    view.cancel_drag(doc)
