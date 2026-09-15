"""Regression tests for the 2026-09-15 audit's ink1 fixer batch, for the
findings that have no existing owned test module of their own to sit beside.

inker-01, inker-02 (``asein.py``'s decode-pixel budget) live in
``tests/test_untrusted_input.py``, beside the sibling ceilings they extend.
inker-04 (``pixelsheet.reduced``) lives in ``tests/test_pixelsheet.py``.
inker-08 (the blend-mode table) lives in ``tests/inker/test_composite.py``.
This file is what is left: inker-03, on ``_doc_layers.move_into_group``,
which has no owned test module of its own.
"""

from __future__ import annotations

from warlock.studio.inker import groups as gp
from warlock.studio.inker.document import Document


def _doc(layers: int = 4) -> Document:
    """The same shape ``tests/inker/test_groups.py`` builds its fixtures
    from -- not imported from there because that module is not one of this
    batch's owned files."""
    doc = Document.blank(8, 8)
    doc.stack[0].name = "L0"
    for i in range(1, layers):
        doc.add_layer(f"L{i}")
    doc.invalidate_all()
    return doc


def _ok(doc: Document) -> None:
    gp.check(doc.groups, doc.group_of, doc.member_uids())


def test_move_into_group_at_top_lands_above_the_group_when_the_row_starts_above_it():
    """``move_into_group(index, group, at_top=True)`` used to target
    ``max(leaves)`` whichever side ``index`` started on. That is the right
    slot for a row starting *below* the span: popping it shifts every leaf
    down by one, so the span's own old top index becomes the slot directly
    above its post-pop position. A row starting *above* the span leaves the
    leaves' positions untouched when popped, so that same index landed
    *inside* the span, under its top member, instead of adjacent above it.
    The 2026-09-15 audit, finding inker-03.
    """
    doc = _doc(4)
    node = doc.group_layers([0, 1], name="Bottom")
    moved = doc.stack[3].uid
    top_of_group = doc.stack[1].uid

    assert doc.move_into_group(3, node.uid)  # at_top=True, the default

    _ok(doc)
    assert doc.group_of[moved] == node.uid
    # Landed directly above the group's own top member -- not sandwiched
    # under it, which is what the un-branched formula did.
    order = doc.member_uids()
    assert order.index(moved) == order.index(top_of_group) + 1


def test_move_into_group_not_at_top_lands_below_the_group_when_the_row_starts_below_it():
    """The symmetric case the same bug caused for ``at_top=False``: a row
    dragged from below a group and dropped at its bottom edge landed inside
    the span, above its bottom member, instead of below it."""
    doc = _doc(4)
    node = doc.group_layers([2, 3], name="Top")
    moved = doc.stack[0].uid
    bottom_of_group = doc.stack[2].uid

    assert doc.move_into_group(0, node.uid, at_top=False)

    _ok(doc)
    assert doc.group_of[moved] == node.uid
    order = doc.member_uids()
    assert order.index(moved) == order.index(bottom_of_group) - 1
