"""Tranche 3 (scene structure): the outliner as a tree.

No imgui harness runs full rows here -- ``test_clay_outliner.py`` already
established the pattern of testing this pane's *decidable* logic directly
(``_range``) rather than a live frame, and every claim this file makes is
exactly that kind: what order the tree walks in, which band of a row's rect a
drop reads as which gesture, and what a drop actually does to the document.
"""

from __future__ import annotations

import numpy as np

from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import primitives as bp
from warlock.studio.modes.clay.ui.panes import outliner as clay_outliner


class _Toasts:
    def __init__(self) -> None:
        self.errors: list[str] = []


class _Ctx:
    """``test_modifier_props.py``'s own double, verbatim: only what ``Ctx``
    really offers (``ctx.toast``)."""

    def __init__(self) -> None:
        self.toasts = _Toasts()

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.toasts.errors.append(message)


def _obj(name: str) -> bd.Obj:
    return bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box())


def _three_level_doc() -> tuple[bd.ClayDoc, bd.Obj, bd.Obj, bd.Obj, bd.Obj]:
    """A -> B -> C (three levels), plus a second root D -- exactly the shape
    the tranche 3 brief's "three-level tree" test asks for, with a sibling
    root so document order among roots is also exercised."""
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    c = doc.add_object(_obj("C"))
    d = doc.add_object(_obj("D"))
    doc.set_parent(b.uid, a.uid, keep_world=True)
    doc.set_parent(c.uid, b.uid, keep_world=True)
    return doc, a, b, c, d


def test_a_three_level_tree_walks_depth_first_in_document_order():
    """Fails against the unfixed pane: ``_tree_rows`` does not exist before
    this tranche, and the old flat walk (``range(len(doc.objects) - 1, -1,
    -1)``) shows the *newest* object first with no notion of depth at all --
    it would report D, C, B, A with every depth at 0.
    """
    doc, a, b, c, d = _three_level_doc()

    rows = clay_outliner._tree_rows(doc)

    assert [(obj.uid, depth, has_children) for obj, depth, has_children in rows] == [
        (a.uid, 0, True),
        (b.uid, 1, True),
        (c.uid, 2, False),
        (d.uid, 0, False),
    ]


def test_tree_rows_ignores_collapse_so_a_filter_can_still_find_a_match():
    """``_tree_rows`` itself never hides a row -- that is ``_body``'s own
    depth-skip loop's job, over this function's full answer -- so a filter
    can find a match nested under a collapsed ancestor."""
    doc, a, b, c, _d = _three_level_doc()

    rows = clay_outliner._tree_rows(doc)

    assert {obj.uid for obj, _depth, _has in rows} == {a.uid, b.uid, c.uid, _d.uid}


def test_drop_zone_reads_the_top_and_bottom_quarters_as_between():
    row_min, row_max = 100.0, 200.0  # a 100px row
    assert clay_outliner._drop_zone(105.0, row_min, row_max) == "before"
    assert clay_outliner._drop_zone(150.0, row_min, row_max) == "onto"
    assert clay_outliner._drop_zone(196.0, row_min, row_max) == "after"


def test_drop_zone_on_a_degenerate_row_reads_as_onto():
    assert clay_outliner._drop_zone(50.0, 50.0, 50.0) == "onto"


def test_a_drop_onto_a_row_reparents_and_keeps_world_place():
    doc = bd.ClayDoc()
    parent = doc.add_object(_obj("Parent"))
    parent.translation[:] = (5.0, 0.0, 0.0)
    child = doc.add_object(_obj("Child"))
    child.translation[:] = (1.0, 0.0, 0.0)
    world_before = doc.world_matrix(child.uid).copy()

    clay_outliner._apply_drop(_Ctx(), doc, child.uid, parent, doc.index_of(parent.uid), "onto")

    assert doc.by_uid(child.uid).parent == parent.uid
    assert np.allclose(doc.world_matrix(child.uid), world_before, atol=1e-9)


def test_a_drop_between_rows_reorders_without_touching_parent():
    doc, a, b, c, d = _three_level_doc()
    before_parent = doc.by_uid(d.uid).parent

    clay_outliner._apply_drop(_Ctx(), doc, d.uid, a, doc.index_of(a.uid), "before")

    assert doc.by_uid(d.uid).parent == before_parent
    assert doc.objects.index(doc.by_uid(d.uid)) == 0
    assert doc.objects[0].uid == d.uid


def test_a_drop_after_a_row_places_the_object_just_past_it():
    doc, a, b, c, d = _three_level_doc()

    clay_outliner._apply_drop(_Ctx(), doc, d.uid, a, doc.index_of(a.uid), "after")

    assert doc.objects.index(doc.by_uid(d.uid)) == 1


def test_a_cycle_drop_is_refused_and_toasted():
    """Dropping an ancestor onto its own descendant: ``set_parent`` refuses
    a cycle by name, and the pane must show that refusal rather than raise
    on the frame thread."""
    doc, a, b, c, _d = _three_level_doc()
    ctx = _Ctx()

    clay_outliner._apply_drop(ctx, doc, a.uid, c, doc.index_of(c.uid), "onto")

    assert doc.by_uid(a.uid).parent is None, "the refused reparent must not have landed"
    assert ctx.toasts.errors, "the refusal was never shown"


def test_a_drop_onto_itself_is_a_no_op():
    doc, a, _b, _c, _d = _three_level_doc()
    ctx = _Ctx()

    clay_outliner._apply_drop(ctx, doc, a.uid, a, doc.index_of(a.uid), "onto")

    assert doc.by_uid(a.uid).parent is None
    assert not ctx.toasts.errors
