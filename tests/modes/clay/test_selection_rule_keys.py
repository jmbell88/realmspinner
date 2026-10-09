"""Esc and H in an element mode, under the selection rule.

The shape being edited stays selected in Points/Edges/Faces, so Esc has three
stages where it used to have two (picks, then the mode, then the shape) and H
hides the shape being edited whether or not anything is picked inside it.
Neither was covered in an element mode before.
"""

from __future__ import annotations

import pygame

from realmspinner.kernels.mesh import elements as el
from realmspinner.studio.modes.clay import mode as clay_mode

from .test_clay_interaction import _adopt, _key


def _editing_a_face():
    ctx, tab, doc = _adopt()
    a, b = doc.objects
    doc.select([a.uid])
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, el.ElementSel(faces=[0, 1]))
    return ctx, tab, doc, a, b


def test_escape_stages_picks_then_mode_then_shape() -> None:
    ctx, _tab, doc, a, _b = _editing_a_face()
    esc = _key(pygame.K_ESCAPE)

    assert clay_mode.handle_key(ctx, esc) is True
    assert doc.element_sel == {}, "the first Esc drops the picks"
    assert doc.element_mode == "face" and doc.selection == {a.uid}

    assert clay_mode.handle_key(ctx, esc) is True
    assert doc.element_mode == "object", "the second Esc leaves the mode"
    assert doc.selection == {a.uid}, "with the shape still selected"

    assert clay_mode.handle_key(ctx, esc) is True
    assert doc.selection == set(), "and the third deselects it"
    assert doc.selection_problems() == []


def test_escape_with_a_shape_edited_but_nothing_picked_leaves_the_mode_first() -> None:
    ctx, _tab, doc, a, _b = _editing_a_face()
    doc.clear_element_sel()
    clay_mode.handle_key(ctx, _key(pygame.K_ESCAPE))
    assert doc.element_mode == "object"
    assert doc.selection == {a.uid}


def test_h_in_an_element_mode_hides_the_edited_shape_and_keeps_the_mode() -> None:
    ctx, _tab, doc, a, b = _editing_a_face()
    depth = len(doc.history)

    assert clay_mode.handle_key(ctx, _key(pygame.K_h)) is True

    assert not a.visible and b.visible
    assert doc.element_mode == "face"
    assert doc.selection == set() and doc.element_sel == {}
    assert len(doc.history) == depth + 1, "hiding is one undo step"
    assert doc.selection_problems() == []


def test_h_in_an_element_mode_hides_a_shape_with_nothing_picked_in_it() -> None:
    ctx, _tab, doc, a, b = _editing_a_face()
    doc.clear_element_sel()
    clay_mode.handle_key(ctx, _key(pygame.K_h))
    assert not a.visible and b.visible
    assert doc.selection == set()


def test_shift_h_in_an_element_mode_isolates_the_edited_shape_and_keeps_editing_it() -> None:
    ctx, _tab, doc, a, b = _editing_a_face()
    clay_mode.handle_key(ctx, _key(pygame.K_h, pygame.KMOD_SHIFT))
    assert a.visible and not b.visible
    assert doc.selection == {a.uid}
    assert doc.element_sel_of(a.uid).faces.tolist() == [0, 1]
