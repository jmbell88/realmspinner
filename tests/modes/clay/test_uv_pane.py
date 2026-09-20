"""Clay's UV pane (``studio/modes/clay/ui/panes/uv.py``, tranche 6).

The island hit-testing and box-select maths (:func:`uv.islands_in_rect`,
:func:`uv.touched_islands`) are pure numpy over a :class:`~.mesh.Mesh` and are
tested directly, with no imgui at all -- the brief's own instruction. The
undo-step doors (:func:`uv.apply_translate` and friends) are driven straight
against a :class:`~.document.ClayDoc`, the same shape
``test_clay_props_undo.py`` already uses for a non-imgui door. Only the empty
states are driven through a real, headless imgui frame (``_ui_context``),
the way every other pane test in this package does.

**The live rotate/scale gesture** (:func:`uv.begin_live_transform` and
friends) gets the same non-imgui treatment as the undo-step doors above --
every function it is built from takes a document and a plain
:class:`uv.UvPaneState`, never imgui, so a whole drag (several frames of
:func:`uv.update_live_transform`, then a commit or a cancel) is driven
directly with no headless frame at all.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import uv as bmuv
from realmspinner.kernels.mesh import uvtools
from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _two_island_mesh() -> bm.Mesh:
    """Two disjoint quads, each its own island, at known, non-overlapping uv
    squares -- deliberately not sharing a single vertex, so
    ``uvtools.islands`` cannot read them as anything but two components."""
    positions = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0],
    ]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7]]
    uv = [
        [[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]],  # island 0
        [[0.6, 0.6], [1.0, 0.6], [1.0, 1.0], [0.6, 1.0]],  # island 1
    ]
    return bm.from_faces(positions, faces, uv=uv)


def _doc_with_two_islands() -> tuple[bd.ClayDoc, bd.Obj]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=_two_island_mesh()))
    return doc, doc.by_uid(obj.uid)


def _asymmetric_island_mesh() -> bm.Mesh:
    """One quad island, deliberately not a square: its own axis-aligned uv
    bbox centre shifts under a *partial* turn in a way a *full* turn from
    the original shape would not reproduce, which is exactly the pivot
    drift a live rotate/scale that measured from "wherever the last frame
    left it" would compound and a from-the-drag-start one does not
    (``uvtools.transform_islands`` turns each island about its own
    bbox-centre, freshly, every call)."""
    positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    faces = [[0, 1, 2, 3]]
    uv = [[[0.1, 0.1], [0.5, 0.12], [0.42, 0.38], [0.15, 0.3]]]
    return bm.from_faces(positions, faces, uv=uv)


# --- islands_in_rect: pure maths ---------------------------------------------


def test_a_box_around_one_island_catches_only_that_island() -> None:
    """A box catches an island by its *corners*, the way a UV editor's box
    select catches vertices -- not by area overlap, so the box has to reach
    at least one of the island's own uv corners (here, island 0's (0, 0) and
    island 1's (1, 1))."""
    mesh = _two_island_mesh()
    ids = np.array([0, 1], dtype="i4")
    assert clay_uv.islands_in_rect(mesh, ids, (-0.05, -0.05, 0.1, 0.1)) == {0}
    assert clay_uv.islands_in_rect(mesh, ids, (0.9, 0.9, 1.05, 1.05)) == {1}


def test_a_box_covering_both_islands_catches_both() -> None:
    mesh = _two_island_mesh()
    ids = np.array([0, 1], dtype="i4")
    assert clay_uv.islands_in_rect(mesh, ids, (0.0, 0.0, 1.0, 1.0)) == {0, 1}


def test_a_box_touching_neither_island_is_empty() -> None:
    mesh = _two_island_mesh()
    ids = np.array([0, 1], dtype="i4")
    assert clay_uv.islands_in_rect(mesh, ids, (0.45, 0.45, 0.5, 0.5)) == set()


def test_a_box_dragged_backwards_reads_the_same_as_forwards() -> None:
    """The corners need not be ordered -- a marquee dragged up-and-left of
    its start is as valid as one dragged down-and-right."""
    mesh = _two_island_mesh()
    ids = np.array([0, 1], dtype="i4")
    forward = clay_uv.islands_in_rect(mesh, ids, (-0.05, -0.05, 0.1, 0.1))
    backward = clay_uv.islands_in_rect(mesh, ids, (0.1, 0.1, -0.05, -0.05))
    assert forward == backward == {0}


def test_a_partial_overlap_still_catches_the_island() -> None:
    """"At least one corner", not "every corner"."""
    mesh = _two_island_mesh()
    ids = np.array([0, 1], dtype="i4")
    # This box only reaches island 0's own top-right corner (0.4, 0.4).
    assert clay_uv.islands_in_rect(mesh, ids, (0.3, 0.3, 0.5, 0.5)) == {0}


def test_a_click_in_the_middle_of_a_face_hits_it_with_no_corner_caught() -> None:
    """A degenerate rect is a point test that falls back to point-in-face
    when no corner is caught -- clicking the *middle* of an island, nowhere
    near a corner, must still select it."""
    mesh = _two_island_mesh()
    ids = np.array([0, 1], dtype="i4")
    centre = (0.2, 0.2)  # island 0's own centre; not a corner
    assert clay_uv.islands_in_rect(mesh, ids, (*centre, *centre)) == {0}


def test_a_click_outside_every_face_hits_nothing() -> None:
    mesh = _two_island_mesh()
    ids = np.array([0, 1], dtype="i4")
    assert clay_uv.islands_in_rect(mesh, ids, (0.5, 0.5, 0.5, 0.5)) == set()


def _no_uv_mesh() -> bm.Mesh:
    """A box() carries a default cube-projection uv (its own canonical
    unwrap) -- a hand-built quad with no ``uv`` argument at all is what an
    object with no texture coordinates actually looks like."""
    positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    return bm.from_faces(positions, [[0, 1, 2, 3]])


def test_islands_in_rect_on_a_mesh_with_no_uv_is_empty() -> None:
    mesh = _no_uv_mesh()
    assert mesh.uv is None
    ids = np.zeros(len(mesh.starts) - 1, dtype="i4")
    assert clay_uv.islands_in_rect(mesh, ids, (0.0, 0.0, 1.0, 1.0)) == set()


# --- touched_islands: pure maths ---------------------------------------------


def test_touched_islands_is_a_union_one_selected_vertex_is_enough() -> None:
    """``elements.convert``'s up-conversion would need *every* corner of a
    face selected before it counted as touched; this is the union test the
    pane actually wants -- one vertex is enough."""
    mesh = _two_island_mesh()
    ids = np.array([0, 1], dtype="i4")
    sel = el.ElementSel(verts=np.array([0], dtype="i4"))  # one corner of island 0
    assert clay_uv.touched_islands(mesh, ids, sel) == {0}


def test_touched_islands_of_an_empty_selection_is_empty() -> None:
    mesh = _two_island_mesh()
    ids = np.array([0, 1], dtype="i4")
    assert clay_uv.touched_islands(mesh, ids, None) == set()
    assert clay_uv.touched_islands(mesh, ids, el.empty()) == set()


def test_touched_islands_face_selection_names_the_right_island() -> None:
    mesh = _two_island_mesh()
    ids = np.array([0, 1], dtype="i4")
    sel = el.ElementSel(faces=np.array([1], dtype="i4"))
    assert clay_uv.touched_islands(mesh, ids, sel) == {1}


# --- one undo step, generator kept -------------------------------------------


def test_apply_translate_is_one_undo_step_and_keeps_the_generator() -> None:
    doc = bd.ClayDoc()
    mesh = bmuv.box_unwrap(bp.box())
    obj = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Box", mesh=mesh, generator="box", params={"size": (1, 1, 1)})
    )
    ids = {int(i) for i in np.unique(uvtools.islands(mesh))}
    before_uv = np.array(obj.mesh.uv, copy=True)
    before_history = len(doc.history)

    changed = clay_uv.apply_translate(doc, obj.uid, ids, (0.1, 0.05))

    assert changed is True
    updated = doc.by_uid(obj.uid)
    assert len(doc.history) == before_history + 1, "one call, one undo step"
    assert updated.generator == "box", "a uv edit is not geometry -- the generator survives"
    assert not np.allclose(updated.mesh.uv, before_uv), "the uv actually moved"
    assert doc.undo()
    assert np.allclose(doc.by_uid(obj.uid).mesh.uv, before_uv)


def test_apply_rotate_is_one_undo_step() -> None:
    doc, obj = _doc_with_two_islands()
    before_history = len(doc.history)
    changed = clay_uv.apply_rotate(doc, obj.uid, {0}, 90.0)
    assert changed is True
    assert len(doc.history) == before_history + 1


def test_apply_scale_is_one_undo_step() -> None:
    doc, obj = _doc_with_two_islands()
    before_history = len(doc.history)
    changed = clay_uv.apply_scale(doc, obj.uid, {0, 1}, 0.5)
    assert changed is True
    assert len(doc.history) == before_history + 1


def test_apply_pack_is_one_undo_step() -> None:
    doc, obj = _doc_with_two_islands()
    before_history = len(doc.history)
    changed = clay_uv.apply_pack(doc, obj.uid)
    assert changed is True
    assert len(doc.history) == before_history + 1
    assert doc.undo()


def test_apply_translate_with_no_islands_selected_is_a_no_op() -> None:
    doc, obj = _doc_with_two_islands()
    before_history = len(doc.history)
    changed = clay_uv.apply_translate(doc, obj.uid, set(), (1.0, 1.0))
    assert changed is False
    assert len(doc.history) == before_history


# --- the live rotate/scale gesture: pure maths -------------------------------


def test_selection_pivot_is_the_bbox_centre_of_every_selected_corner() -> None:
    mesh = _two_island_mesh()
    ids = np.array([0, 1], dtype="i4")
    # Island 0's own uv corners span (0, 0) to (0.4, 0.4).
    assert clay_uv.selection_pivot(mesh, ids, {0}) == pytest.approx((0.2, 0.2))


def test_selection_pivot_of_an_empty_selection_is_none() -> None:
    mesh = _two_island_mesh()
    ids = np.array([0, 1], dtype="i4")
    assert clay_uv.selection_pivot(mesh, ids, set()) is None


def test_selection_pivot_of_an_island_id_that_does_not_exist_is_none() -> None:
    mesh = _two_island_mesh()
    ids = np.array([0, 1], dtype="i4")
    assert clay_uv.selection_pivot(mesh, ids, {7}) is None


def test_drag_angle_of_a_quarter_turn_is_ninety_degrees() -> None:
    """u toward v, matching ``uvtools.transform_islands``'s own rotation
    matrix (``uv.drag_angle``'s own docstring)."""
    assert clay_uv.drag_angle((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0)


def test_drag_angle_with_either_arm_on_the_pivot_is_zero() -> None:
    assert clay_uv.drag_angle((0.5, 0.5), (0.5, 0.5), (0.9, 0.9)) == 0.0
    assert clay_uv.drag_angle((0.5, 0.5), (0.9, 0.9), (0.5, 0.5)) == 0.0


def test_drag_scale_of_double_the_distance_is_a_factor_of_two() -> None:
    assert clay_uv.drag_scale((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)) == pytest.approx(2.0)


def test_drag_scale_with_the_anchor_on_the_pivot_is_one() -> None:
    assert clay_uv.drag_scale((0.5, 0.5), (0.5, 0.5), (0.9, 0.1)) == 1.0


# --- the live rotate/scale gesture: the doors --------------------------------


def test_begin_live_transform_refuses_when_the_selection_touches_no_uv_corner() -> None:
    mesh = _two_island_mesh()
    ids = np.array([0, 1], dtype="i4")
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=mesh))
    view_state = clay_uv.UvPaneState(selected_islands=frozenset({7}))  # no such island
    armed = clay_uv.begin_live_transform(doc, view_state, "rotate", mesh, ids, (0.0, 0.0))
    assert armed is False
    assert view_state.drag_mode == ""
    assert view_state.drag_base is None


def test_live_rotate_lands_exactly_where_the_typed_field_lands() -> None:
    """The pivot-drift claim, proven: several frames of an armed live rotate
    -- each recomputed from the drag's own start against the frozen base
    mesh, an asymmetric island whose bbox centre would otherwise creep --
    must land bit-for-bit where a single one-shot ``apply_rotate`` at the
    *same final angle* lands.
    """
    base = _asymmetric_island_mesh()
    ids = np.array([0], dtype="i4")
    pivot = clay_uv.selection_pivot(base, ids, {0})
    anchor = (0.6, 0.05)
    # An arbitrary drag path -- several different intermediate positions
    # before the pointer settles, the sequence a real drag produces.
    path = [(0.5, 0.2), (0.3, 0.5), (0.05, 0.55), (0.2, -0.1)]

    doc_live = bd.ClayDoc()
    obj_live = doc_live.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=base))
    view_state = clay_uv.UvPaneState(selected_islands=frozenset({0}))
    assert clay_uv.begin_live_transform(doc_live, view_state, "rotate", base, ids, anchor)
    for now in path:
        assert clay_uv.update_live_transform(doc_live, obj_live.uid, view_state, now)
    clay_uv.commit_live_transform(doc_live, view_state)

    final_degrees = clay_uv.drag_angle(pivot, anchor, path[-1])
    doc_typed = bd.ClayDoc()
    obj_typed = doc_typed.add_object(
        bd.Obj(uid=bd.new_uid(), name="A", mesh=_asymmetric_island_mesh())
    )
    clay_uv.apply_rotate(doc_typed, obj_typed.uid, {0}, final_degrees)

    live_uv = doc_live.by_uid(obj_live.uid).mesh.uv
    typed_uv = doc_typed.by_uid(obj_typed.uid).mesh.uv
    assert np.array_equal(live_uv, typed_uv)
    assert not np.allclose(live_uv, base.uv), "the drag actually moved something"


def test_live_scale_lands_exactly_where_the_typed_field_lands() -> None:
    """:func:`test_live_rotate_lands_exactly_where_the_typed_field_lands`'s
    twin, for scale."""
    base = _asymmetric_island_mesh()
    ids = np.array([0], dtype="i4")
    pivot = clay_uv.selection_pivot(base, ids, {0})
    anchor = (0.2, 0.15)
    path = [(0.25, 0.2), (0.32, 0.28), (0.05, 0.02)]

    doc_live = bd.ClayDoc()
    obj_live = doc_live.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=base))
    view_state = clay_uv.UvPaneState(selected_islands=frozenset({0}))
    assert clay_uv.begin_live_transform(doc_live, view_state, "scale", base, ids, anchor)
    for now in path:
        assert clay_uv.update_live_transform(doc_live, obj_live.uid, view_state, now)
    clay_uv.commit_live_transform(doc_live, view_state)

    final_factor = clay_uv.drag_scale(pivot, anchor, path[-1])
    doc_typed = bd.ClayDoc()
    obj_typed = doc_typed.add_object(
        bd.Obj(uid=bd.new_uid(), name="A", mesh=_asymmetric_island_mesh())
    )
    clay_uv.apply_scale(doc_typed, obj_typed.uid, {0}, final_factor)

    live_uv = doc_live.by_uid(obj_live.uid).mesh.uv
    typed_uv = doc_typed.by_uid(obj_typed.uid).mesh.uv
    assert np.array_equal(live_uv, typed_uv)
    assert not np.allclose(live_uv, base.uv), "the drag actually moved something"


def test_a_naive_per_frame_apply_drifts_but_the_live_path_does_not() -> None:
    """The pivot-drift bug, made concrete: applying the same three
    intermediate deltas as *independent* one-shot calls -- each rotating
    about whatever the previous call left behind, the bug this feature
    exists to fix -- lands somewhere else than a live drag measuring the
    same nominal total as one absolute angle from a fixed start. If the two
    ever agreed, the fix would not be proving anything.
    """
    base = _asymmetric_island_mesh()
    ids = np.array([0], dtype="i4")

    doc_naive = bd.ClayDoc()
    obj_naive = doc_naive.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=base))
    for delta in (10.0, 15.0, 12.0):  # sums to 37 degrees, applied incrementally
        clay_uv.apply_rotate(doc_naive, obj_naive.uid, {0}, delta)

    pivot = clay_uv.selection_pivot(base, ids, {0})
    anchor = (0.6, 0.05)
    theta = math.radians(37.0)
    ax, ay = anchor[0] - pivot[0], anchor[1] - pivot[1]
    now = (
        pivot[0] + ax * math.cos(theta) - ay * math.sin(theta),
        pivot[1] + ax * math.sin(theta) + ay * math.cos(theta),
    )
    doc_live = bd.ClayDoc()
    obj_live = doc_live.add_object(
        bd.Obj(uid=bd.new_uid(), name="A", mesh=_asymmetric_island_mesh())
    )
    view_state = clay_uv.UvPaneState(selected_islands=frozenset({0}))
    assert clay_uv.begin_live_transform(doc_live, view_state, "rotate", base, ids, anchor)
    clay_uv.update_live_transform(doc_live, obj_live.uid, view_state, now)
    clay_uv.commit_live_transform(doc_live, view_state)

    naive_uv = doc_naive.by_uid(obj_naive.uid).mesh.uv
    live_uv = doc_live.by_uid(obj_live.uid).mesh.uv
    assert not np.allclose(naive_uv, live_uv)


def test_a_whole_live_rotate_drag_is_one_undo_step_and_undoes_to_the_exact_starting_mesh() -> None:
    doc, obj = _doc_with_two_islands()
    base = obj.mesh
    ids = np.array([0, 1], dtype="i4")
    view_state = clay_uv.UvPaneState(selected_islands=frozenset({0, 1}))
    before_history = len(doc.history)

    assert clay_uv.begin_live_transform(doc, view_state, "rotate", base, ids, (0.05, 0.05))
    for now in [(0.1, 0.2), (0.3, 0.1), (0.5, 0.5)]:
        clay_uv.update_live_transform(doc, obj.uid, view_state, now)
    clay_uv.commit_live_transform(doc, view_state)

    assert len(doc.history) == before_history + 1, "the whole drag is one undo step"
    assert doc.by_uid(obj.uid).mesh is not base, "the commit actually left something changed"
    assert doc.undo()
    assert doc.by_uid(obj.uid).mesh is base, (
        "undo restores the exact starting mesh object, not a value-equal copy"
    )


def test_a_whole_live_scale_drag_is_one_undo_step_and_undoes_to_the_exact_starting_mesh() -> None:
    doc, obj = _doc_with_two_islands()
    base = obj.mesh
    ids = np.array([0, 1], dtype="i4")
    view_state = clay_uv.UvPaneState(selected_islands=frozenset({0, 1}))
    before_history = len(doc.history)

    assert clay_uv.begin_live_transform(doc, view_state, "scale", base, ids, (0.05, 0.05))
    for now in [(0.1, 0.1), (0.3, 0.3)]:
        clay_uv.update_live_transform(doc, obj.uid, view_state, now)
    clay_uv.commit_live_transform(doc, view_state)

    assert len(doc.history) == before_history + 1
    assert doc.undo()
    assert doc.by_uid(obj.uid).mesh is base


def test_escape_mid_drag_leaves_the_document_untouched_and_pushes_nothing() -> None:
    doc, obj = _doc_with_two_islands()
    base = obj.mesh
    ids = np.array([0, 1], dtype="i4")
    view_state = clay_uv.UvPaneState(selected_islands=frozenset({0, 1}))
    before_history = len(doc.history)

    assert clay_uv.begin_live_transform(doc, view_state, "scale", base, ids, (0.2, 0.2))
    for now in [(0.25, 0.25), (0.4, 0.4)]:
        clay_uv.update_live_transform(doc, obj.uid, view_state, now)
    # The drag really did write something live -- a no-op cancel would prove
    # nothing about restoring it.
    assert doc.by_uid(obj.uid).mesh is not base
    assert len(doc.history) > before_history

    clay_uv.cancel_live_transform(doc, view_state)

    assert len(doc.history) == before_history, "cancel pushes nothing that survives it"
    assert doc.by_uid(obj.uid).mesh is base, "cancel restores the exact pre-drag mesh, by identity"
    assert view_state.drag_mode == ""
    assert not doc.history.can_redo, "nothing is left on the stack to redo back to"


def test_a_cancel_with_no_motion_at_all_is_also_a_no_op() -> None:
    """Armed, then cancelled before a single frame ever ran -- the
    zero-frame edge case :func:`cancel_live_transform`'s own mark-moved
    guard exists for."""
    doc, obj = _doc_with_two_islands()
    base = obj.mesh
    ids = np.array([0, 1], dtype="i4")
    view_state = clay_uv.UvPaneState(selected_islands=frozenset({0, 1}))
    before_history = len(doc.history)

    assert clay_uv.begin_live_transform(doc, view_state, "rotate", base, ids, (0.2, 0.2))
    clay_uv.cancel_live_transform(doc, view_state)

    assert len(doc.history) == before_history
    assert doc.by_uid(obj.uid).mesh is base


def test_uv_pane_switching_the_selected_object_mid_live_transform_closes_the_open_gesture(
    ui, monkeypatch
) -> None:
    """The 2026-09-19 audit's clay-16: ``_body``'s own per-object reset used
    to clear ``drag_mode`` without closing the ``UndoStack.mark()`` gesture
    ``begin_live_transform`` opened -- only ``collapse_since`` (via a commit
    or a cancel) decrements ``UndoStack._open_gestures``, and while any
    gesture is open the stack's deferred-eviction rule switches off depth-
    and byte-budget trimming for the rest of the document's life. Asserts
    both halves the audit measured: the several per-frame steps fold into
    one undo step, and ``_open_gestures`` actually returns to zero."""
    doc, obj_a = _doc_with_two_islands()
    obj_b = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=_two_island_mesh()))
    doc.select([obj_a.uid])
    view_state = clay_uv.UvPaneState(for_uid=obj_a.uid, selected_islands=frozenset({0, 1}))
    ids = np.array([0, 1], dtype="i4")
    before_history = len(doc.history)

    assert clay_uv.begin_live_transform(doc, view_state, "rotate", obj_a.mesh, ids, (0.05, 0.05))
    clay_uv.update_live_transform(doc, obj_a.uid, view_state, (0.3, 0.1))
    clay_uv.update_live_transform(doc, obj_a.uid, view_state, (0.1, 0.4))
    assert len(doc.history) > before_history + 1, "the drag pushed several frames to fold"
    assert doc.history._open_gestures == 1, "begin_live_transform's mark() is still open"

    # The properties panel (or the outliner) moves the selection with no
    # release and no Escape to route through -- _body's own per-object reset
    # is the only place left that can see this happened.
    doc.select([obj_b.uid])

    class _Tab:
        def __init__(self, doc, view_state) -> None:
            self.doc = doc
            self.uv_view = view_state

    class _State:
        def __init__(self, tab) -> None:
            self.active = tab

    monkeypatch.setattr(clay_uv.clay_mode, "ensure", lambda ctx: _State(_Tab(doc, view_state)))
    ui.new_frame()
    ui.begin("##host")
    try:
        clay_uv._body(ctx=None)
    finally:
        ui.end()
        ui.end_frame()

    assert len(doc.history) == before_history + 1, "the abandoned gesture folds into one undo step"
    assert doc.history._open_gestures == 0, "the mark begin_live_transform opened must close"
    assert view_state.drag_mode == ""
    assert view_state.for_uid == obj_b.uid


def test_uv_pane_apply_buttons_at_their_own_identity_value_push_no_undo_step(
    ui, monkeypatch
) -> None:
    """The 2026-09-19 audit's clay-34: ``transform_islands`` always returns a
    freshly-built ``Mesh``, even for a 0-degree rotate or an x1 scale, so
    ``set_mesh``'s own identity check can never see that nothing changed --
    the Apply buttons used to fire regardless. Forces a click every call
    (the same ``monkeypatch`` shape ``test_modifier_props.py`` uses for its
    own small-button doors) so the guard under test is the value check, not
    imgui's own enabled/disabled gating."""
    doc, obj = _doc_with_two_islands()
    view_state = clay_uv.UvPaneState(selected_islands=frozenset({0, 1}))
    monkeypatch.setattr(clay_uv.widgets, "disabled_button", lambda *a, **kw: True)

    def click() -> None:
        ui.new_frame()
        ui.begin("##host")
        try:
            clay_uv._toolbar(ctx=None, doc=doc, obj=obj, view_state=view_state)
        finally:
            ui.end()
            ui.end_frame()

    before = len(doc.history)
    click()  # pending_rotate=0.0, pending_scale=1.0: both at their own resting value
    assert len(doc.history) == before, "Apply at 0 degrees / x1 scale pushed no undo step"

    view_state.pending_rotate = 45.0
    click()
    assert len(doc.history) == before + 1, "a real rotate still applies and pushes one step"
    assert view_state.pending_rotate == 0.0, "Apply resets the field back to identity"

    view_state.pending_scale = 2.0
    click()
    assert len(doc.history) == before + 2, "a real scale still applies and pushes one step"
    assert view_state.pending_scale == 1.0


def test_uv_pane_measurements_are_memoised_on_the_mesh_and_not_recomputed_every_frame(
    monkeypatch,
) -> None:
    """The 2026-09-19 audit's clay-12: ``_measurements`` used to call
    ``uvtools.overlap_faces``/``stretch`` fresh on every single frame the
    pane was open, including a frame where nothing about the mesh had
    changed at all (panning, zooming, hovering) -- only an actual edit
    replaces ``obj.mesh`` with a new object."""
    mesh = _two_island_mesh()
    calls = []
    original = uvtools.overlap_faces

    def counting(m):
        calls.append(1)
        return original(m)

    monkeypatch.setattr(clay_uv.uvtools, "overlap_faces", counting)
    view_state = clay_uv.UvPaneState()

    clay_uv._measurements(view_state, mesh)
    clay_uv._measurements(view_state, mesh)
    clay_uv._measurements(view_state, mesh)
    assert len(calls) == 1, "the same mesh object must be measured once, not every call"

    other = _two_island_mesh()
    clay_uv._measurements(view_state, other)
    assert len(calls) == 2, "a genuinely different mesh still gets measured"


def test_live_rotate_keeps_the_generator() -> None:
    doc = bd.ClayDoc()
    mesh = bmuv.box_unwrap(bp.box())
    obj = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Box", mesh=mesh, generator="box", params={"size": (1, 1, 1)})
    )
    ids = uvtools.islands(mesh)
    all_ids = {int(i) for i in np.unique(ids)}
    view_state = clay_uv.UvPaneState(selected_islands=frozenset(all_ids))

    assert clay_uv.begin_live_transform(doc, view_state, "rotate", mesh, ids, (0.1, 0.1))
    clay_uv.update_live_transform(doc, obj.uid, view_state, (0.3, 0.4))
    clay_uv.commit_live_transform(doc, view_state)

    assert doc.by_uid(obj.uid).generator == "box", (
        "a uv edit is not geometry -- the generator survives"
    )


# --- the empty states, driven through a real headless frame ------------------


class _FakeTab:
    """The two attributes ``_body`` reads off the active tab: the document,
    and this pane's own per-tab view state."""

    def __init__(self, doc: bd.ClayDoc) -> None:
        self.doc = doc
        self.uv_view = clay_uv.UvPaneState()


class _FakeState:
    def __init__(self, doc: bd.ClayDoc) -> None:
        self.active = _FakeTab(doc)


def _draw_body(monkeypatch, ui, doc: bd.ClayDoc) -> None:
    """Drive ``clay_uv._body`` inside a real, headless imgui frame, with
    ``clay_mode.ensure`` monkeypatched to a minimal stand-in -- the same
    "model only what is read" shape ``test_modifier_props.py``'s own ``_Ctx``
    double uses, rather than assembling a whole live ``ClayState``/``App``
    for a pane that only ever reads ``state.active``.
    """
    monkeypatch.setattr(clay_uv.clay_mode, "ensure", lambda ctx: _FakeState(doc))
    ui.new_frame()
    ui.begin("##host")
    try:
        clay_uv._body(ctx=None)
    finally:
        ui.end()
        ui.end_frame()


def test_draw_offers_nothing_for_an_object_with_no_uv(ui, monkeypatch) -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Plain", mesh=_no_uv_mesh()))
    assert obj.mesh.uv is None
    doc.select([obj.uid])

    # The claim is that this does not raise, and stops right after the empty
    # state -- ``_body`` returns the line after ``widgets.empty_state`` for
    # exactly this branch, so a clean run through a real frame is the proof
    # a screenshot-free test can offer that nothing past it (the toolbar, the
    # canvas) was reached with a mesh that has no ``.uv`` to read.
    _draw_body(monkeypatch, ui, doc)


def test_draw_offers_nothing_with_no_object_selected(ui, monkeypatch) -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    _draw_body(monkeypatch, ui, doc)


def test_draw_with_a_uv_object_selected_reaches_the_canvas(ui, monkeypatch) -> None:
    """The positive case: a selected object that does have uv runs the whole
    body -- toolbar, canvas, legend -- with no exception, which is what
    proves the empty-state branch above is actually the exception rather
    than the rule the rest of this file tests headlessly."""
    doc, obj = _doc_with_two_islands()
    doc.select([obj.uid])
    _draw_body(monkeypatch, ui, doc)
