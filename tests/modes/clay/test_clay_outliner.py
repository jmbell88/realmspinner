"""The Clay outliner pane: the decidable half of what ``outliner.py`` does.

No imgui harness exists in this suite, so a draw function is covered by the
smoke pass and its decisions are covered here -- the pattern
``tests/test_findings_blind_spots.py`` already uses for every other pane.
"""

from __future__ import annotations

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay.ui.panes import outliner as clay_outliner


def _doc_with_three_boxes() -> tuple[bd.ClayDoc, int, int, int]:
    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box())).uid
    b = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box())).uid
    c = doc.add_object(bd.Obj(uid=bd.new_uid(), name="C", mesh=bp.box())).uid
    return doc, a, b, c


# --- clay-25 (2026-09-19 audit): a collider gets its own row icon ----------


def test_row_label_gives_a_collider_a_distinct_icon_ordinary_rows_lack():
    """clay-25: the outliner used to draw a collider as an ordinary row with
    ordinary icons -- the eye and the lock are the only two any row ever
    gets -- so the auto-generated name was the only thing in the whole UI
    saying an object was one, and a rename could erase it silently. Pinned
    against ``icons.SQUARE_DASHED`` by name, not by codepoint, so a future
    icon-atlas change that keeps the *meaning* still passes.
    """
    from realmspinner.studio import icons

    doc = bd.ClayDoc()
    mesh_obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Crate", mesh=bp.box()))
    from realmspinner.kernels.mesh import colliders as cl

    collider = doc.add_collider(mesh_obj.uid, cl.fit_box(mesh_obj.mesh))

    assert clay_outliner.row_label(mesh_obj) == "Crate"
    collider_label = clay_outliner.row_label(collider)
    assert collider_label != collider.name, "a collider row must not read exactly like its name"
    assert collider_label == f"{icons.SQUARE_DASHED} {collider.name}"


def test_row_label_falls_back_to_a_placeholder_for_an_unnamed_object():
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="", mesh=bp.box()))
    assert clay_outliner.row_label(obj) == f"object {obj.uid}"


def test_visibility_toggle_pushes_one_history_step():
    """The 2026-09-18 audit's clay-03: ``_body``'s comment used to say
    visibility does not change the document's history, but the per-row eye
    toggle calls ``doc.set_props(uid, visible=...)``, and ``set_props`` is an
    unconditional ``history.push`` like every other caller -- renaming
    included. This is a documentation fix, not a behaviour fix: the code was
    already right, so this test passes before and after the comment change,
    and it is the comment's new claim it exists to hold true.
    """
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    before = doc.history.head

    doc.set_props(obj.uid, visible=not obj.visible)

    assert doc.history.head == before + 1
    assert obj.visible is False


def test_range_orders_an_ascending_pair():
    doc, a, b, c = _doc_with_three_boxes()
    assert clay_outliner._range(doc, a, c) == [a, b, c]


def test_range_orders_a_descending_pair():
    doc, a, b, c = _doc_with_three_boxes()
    # Clicked backwards from the anchor -- the range still reads top to
    # bottom in document order, not in click order.
    assert clay_outliner._range(doc, c, a) == [a, b, c]


def test_range_falls_back_to_the_clicked_row_when_its_anchor_is_gone():
    """The 2026-09-18 audit's clay-04: ``_range``'s ``ValueError`` fallback
    (the anchor was deleted since it was set) had no test anywhere in the
    tree. Falling back to just the clicked row is the same outcome a plain
    click without Shift produces, so a stale anchor degrades gracefully
    instead of raising out of a Shift+click.
    """
    doc, a, b, c = _doc_with_three_boxes()
    doc.remove_object(b)
    deleted_anchor = b

    assert clay_outliner._range(doc, deleted_anchor, c) == [c]
