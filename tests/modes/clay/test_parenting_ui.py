"""Tranche 3 (scene structure): the viewport's half of parenting.

Three claims, each one that fails against the unfixed view: the world memo
has to invalidate when an ancestor moves (``ClayView._world`` used to key on
only the object's own three arrays), a drag on a parented object has to write
*local* TRS (the gizmo/keyboard drag used to write straight to
``obj.translation``/``rotation``/``scale`` as if they were world values), and
a locked object has to refuse a drag and refuse to be picked while the
outliner can still reach it.
"""

from __future__ import annotations

import numpy as np
import pytest
from _ui_context import imgui_context

from warlock.kernels.geom3d import math3d as m3
from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import primitives as bp
from warlock.studio.modes.clay.ui import view as clay_view
from warlock.studio.modes.clay.ui.panes import props as clay_props


class _State:
    def __init__(self, tool: str = "select") -> None:
        self.tool = tool
        self.snap = False
        self.snap_translate = 0.125
        self.snap_rotate = 15.0
        self.snap_vertex = False
        self.snap_edge = False
        self.snap_face = False


class _Ctx:
    """The *app* ctx -- ``ClayView``'s own read of it (the tool) plus a
    ``toast`` recorder, ``test_modifier_props.py``'s own double's shape."""

    def __init__(self, tool: str = "select") -> None:
        self.state = type("S", (), {"clay": _State(tool)})()
        self.toasted: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasted.append((message, level))


@pytest.fixture
def view(gl):
    v = clay_view.ClayView(gl, _Ctx())
    yield v
    v.release()


RECT = (0.0, 0.0, 128.0, 96.0)


def _parent_and_child() -> tuple[bd.ClayDoc, bd.Obj, bd.Obj]:
    doc = bd.ClayDoc()
    parent = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="P", mesh=bp.box(), translation=m3.vec3(3.0, 0.0, 0.0))
    )
    child = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="C", mesh=bp.box(), translation=m3.vec3(1.0, 0.0, 0.0))
    )
    doc.set_parent(child.uid, parent.uid, keep_world=True)
    return doc, parent, child


# --- the world memo -----------------------------------------------------------


def test_the_world_memo_invalidates_when_an_ancestor_moves(view) -> None:
    """Moving the *parent* leaves the child's own three arrays untouched --
    they are local to it -- so a memo keyed on only the child's own arrays
    would keep matching after the parent moved and would go on serving the
    parent's *old* placement composed with the child's current local one.

    The expected shift is derived rather than hard-coded: a pure translation
    change on the parent (same rotation, same scale) moves every descendant's
    *world* translation by exactly that same delta, whatever the child's own
    local offset happens to be -- see ``document.py``'s ``world_matrix`` for
    why (composing ``T(delta)`` in front of the parent's matrix adds ``delta``
    to every descendant's world translation and leaves the rest of each
    matrix alone).
    """
    doc, parent, child = _parent_and_child()

    first = view._world(doc, child)[:3, 3].copy()

    # A real edit: ``set_transform`` rebinds the array, which is what the
    # memo's identity key is supposed to notice.
    delta = np.array([27.0, 0.0, 0.0])
    doc.set_transform(parent.uid, translation=tuple(np.array(parent.translation) + delta))

    second = view._world(doc, child)[:3, 3]
    assert np.allclose(second, first + delta), (
        "the cache served the parent's old placement composed with the "
        "child's current local one"
    )


def test_the_world_memo_still_serves_a_root_from_one_compose(view) -> None:
    """A document with no parenting behaves exactly as it always did: the
    memo costs one ``compose`` and the world matrix is the object's own TRS."""
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box(), translation=m3.vec3(2.0, 0.0, 0.0))
    )
    assert np.allclose(view._world(doc, obj)[:3, 3], [2.0, 0.0, 0.0])


# --- the drag writes local ----------------------------------------------------


def test_dragging_a_child_writes_local_trs_so_the_world_position_matches_the_gizmo(
    view,
) -> None:
    """The gizmo's delta is a *world* displacement; a parented object's
    ``translation`` is local. Fails against the unfixed ``_apply``, which
    wrote the world delta straight onto ``obj.translation`` -- for a child
    that is not the same number, and the object would land at its parent's
    world position plus only its own *old* local offset, ignoring the drag.

    ``_drag_origin`` is set to the gizmo's own pivot at the press -- the
    child's world position -- and ``delta`` to the new world position a
    translate gizmo's own ``update`` would report, the same two numbers
    ``_drag_gizmo`` hands ``_apply`` for real; deriving both from
    ``doc.world_matrix`` rather than a hand-computed constant keeps the test
    honest about what "the world position matches the gizmo" means without
    assuming anything about the parent's own placement.
    """
    doc, parent, child = _parent_and_child()
    doc.select([child.uid])
    view.app_ctx.state.clay.tool = "move"

    view._begin_gizmo_drag(doc)
    world_before = doc.world_matrix(child.uid)[:3, 3].copy()
    view._drag_origin = world_before.copy()
    was = view._drag_start[child.uid]
    displacement = np.array([5.0, 0.0, 0.0])
    delta = world_before + displacement  # the gizmo's new world position

    view._apply(doc, doc.by_uid(child.uid), was, delta, view.state)

    world_after = doc.world_matrix(child.uid)[:3, 3]
    assert np.allclose(world_after, world_before + displacement), (
        "the world position must match where the gizmo was dragged to"
    )
    # And it is genuinely local: the stored translation is relative to the
    # parent, not the world position the gizmo put the object at.
    assert not np.allclose(doc.by_uid(child.uid).translation, world_after)


def test_dragging_a_root_is_unchanged_by_the_local_conversion(view) -> None:
    """A root's local TRS *is* its world TRS, so the conversion is a
    no-op -- the same drag maths a document with no parenting always had."""
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    doc.select([obj.uid])
    view.app_ctx.state.clay.tool = "move"

    view._begin_gizmo_drag(doc)
    was = view._drag_start[obj.uid]
    view._apply(doc, doc.by_uid(obj.uid), was, np.array([2.0, 3.0, 4.0]), view.state)

    assert np.allclose(doc.by_uid(obj.uid).translation, [2.0, 3.0, 4.0])


# --- locking -------------------------------------------------------------------


def test_a_locked_object_cannot_be_dragged(view) -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box(), locked=True))
    doc.select([obj.uid])
    view.app_ctx.state.clay.tool = "move"
    before = np.array(obj.translation, copy=True)

    started = view._begin_gizmo_drag(doc)

    assert started is False
    assert view._grab is None
    assert np.allclose(obj.translation, before)
    assert view.app_ctx.toasted, "the refusal was never shown"
    assert "locked" in view.app_ctx.toasted[0][0]


def test_a_locked_descendant_of_a_locked_group_also_refuses_the_object_drag(view) -> None:
    doc, parent, child = _parent_and_child()
    doc.set_props(parent.uid, locked=True)
    doc.select([child.uid])
    view.app_ctx.state.clay.tool = "move"

    started = view._begin_gizmo_drag(doc)

    assert started is False
    assert view.app_ctx.toasted


def test_a_locked_object_cannot_be_picked(view) -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box(), locked=True))
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    centre = (RECT[2] * 0.5, RECT[3] * 0.5)
    assert view.pick(doc, centre) is None


def test_a_locked_object_can_still_be_selected_and_unlocked_by_the_outliner(view) -> None:
    """The outliner never picks through the viewport -- it addresses the
    object by uid directly -- so ``doc.select`` and the lock toggle
    (``set_props``, not a locking door) must keep working on it."""
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box(), locked=True))

    doc.select([obj.uid])
    assert doc.selection == {obj.uid}

    doc.set_props(obj.uid, locked=False)
    assert doc.by_uid(obj.uid).locked is False

    # And now that it is unlocked, the viewport can pick it and drag it.
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    centre = (RECT[2] * 0.5, RECT[3] * 0.5)
    assert view.pick(doc, centre) == obj.uid

    view.app_ctx.state.clay.tool = "move"
    assert view._begin_gizmo_drag(doc) is True


# --- the properties panel's parent combo --------------------------------------


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


class _PropsCtx:
    def toast(self, message: str, level: str = "info") -> None:
        del message, level


def test_the_props_parent_combo_never_offers_a_descendant(ui, monkeypatch) -> None:
    """A -> B -> C, plus a sibling D. Editing B's parent must offer A and D,
    never C (B's own descendant) or B itself -- offering C would let a click
    ask ``set_parent`` for a cycle it refuses anyway, and a control that
    visibly offers a choice it is about to refuse is worse than one that
    never shows it."""
    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    b = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    c = doc.add_object(bd.Obj(uid=bd.new_uid(), name="C", mesh=bp.box()))
    d = doc.add_object(bd.Obj(uid=bd.new_uid(), name="D", mesh=bp.box()))
    doc.set_parent(b.uid, a.uid, keep_world=True)
    doc.set_parent(c.uid, b.uid, keep_world=True)

    seen: dict = {}

    def spy(label, value, options, *a_, **kw):
        seen["options"] = options
        return value

    monkeypatch.setattr(clay_props.widgets, "labeled_combo", spy)

    ui.new_frame()
    ui.begin("##host")
    try:
        clay_props._relations(_PropsCtx(), doc, doc.by_uid(b.uid))
    finally:
        ui.end()
        ui.end_frame()

    offered = {key for key, _label in seen["options"]}
    assert offered == {"0", str(a.uid), str(d.uid)}, offered


def test_clearing_the_parent_through_the_button_makes_the_object_a_root(ui, monkeypatch) -> None:
    doc, parent, child = _parent_and_child()
    monkeypatch.setattr(clay_props.widgets, "disabled_button", lambda *a, **kw: True)

    ui.new_frame()
    ui.begin("##host")
    try:
        clay_props._relations(_PropsCtx(), doc, doc.by_uid(child.uid))
    finally:
        ui.end()
        ui.end_frame()

    assert doc.by_uid(child.uid).parent is None
