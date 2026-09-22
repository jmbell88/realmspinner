"""Tranche 3: locking -- "cannot be changed", not "cannot be seen".

Every door the module docstring names is exercised here, refusing before
anything is pushed; :meth:`~realmspinner.kernels.mesh.document.ClayDoc.set_props`
is exercised for the opposite reason -- it is *not* a locking door, and the
fields the spec says still get through it while locked (name, visibility,
tags, locked itself, material) are pinned one by one. Undo/redo apply
``Edit`` objects directly, never through a door, so a locked object's own
history still moves.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import modifiers as mod
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.elements import OpError


def _obj(name: str, mesh: bm.Mesh | None = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=bp.box() if mesh is None else mesh,
        **kwargs,  # type: ignore[arg-type]
    )


def _locked(doc: bd.ClayDoc, name: str = "A", **kwargs: object) -> bd.Obj:
    obj = doc.add_object(_obj(name, **kwargs))
    doc.set_props(obj.uid, locked=True)
    return obj


# --- the six locking doors -----------------------------------------------


def test_set_mesh_refuses_a_locked_object() -> None:
    doc = bd.ClayDoc()
    a = _locked(doc)
    with pytest.raises(OpError, match="locked"):
        doc.set_mesh(a.uid, bp.cylinder())
    assert doc.by_uid(a.uid).mesh is not None
    assert doc.rev  # touched by add_object/set_props, not by the refusal
    head = doc.history.head
    with pytest.raises(OpError):
        doc.set_mesh(a.uid, bp.cylinder())
    assert doc.history.head == head  # nothing pushed


def test_set_generator_params_refuses_a_locked_object() -> None:
    doc = bd.ClayDoc()
    a = _locked(doc)
    with pytest.raises(OpError, match="locked"):
        doc.set_generator_params(a.uid, {"size": 2.0}, bp.box(), was={"params": {}})


def test_set_transform_refuses_a_locked_object() -> None:
    doc = bd.ClayDoc()
    a = _locked(doc)
    with pytest.raises(OpError, match="locked"):
        doc.set_transform(a.uid, translation=(1.0, 0.0, 0.0))


def test_set_transform_refuses_when_an_ancestor_is_locked() -> None:
    doc = bd.ClayDoc()
    parent = doc.add_object(_obj("Parent"))
    child = doc.add_object(_obj("Child"))
    doc.set_parent(child.uid, parent.uid)
    doc.set_props(parent.uid, locked=True)

    with pytest.raises(OpError, match="locked"):
        doc.set_transform(child.uid, translation=(1.0, 0.0, 0.0))
    # The child itself is not locked -- only its ancestor is.
    assert doc.by_uid(child.uid).locked is False


def test_set_modifiers_refuses_a_locked_object() -> None:
    doc = bd.ClayDoc()
    a = _locked(doc)
    with pytest.raises(OpError, match="locked"):
        doc.set_modifiers(a.uid, (mod.make("weld", id=1),))


def test_apply_modifiers_refuses_a_locked_object() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_modifiers(a.uid, (mod.make("weld", id=1),))
    doc.set_props(a.uid, locked=True)
    with pytest.raises(OpError, match="locked"):
        doc.apply_modifiers(a.uid)


def test_remove_object_refuses_a_locked_object() -> None:
    doc = bd.ClayDoc()
    a = _locked(doc)
    with pytest.raises(OpError, match="locked"):
        doc.remove_object(a.uid)
    assert a.uid in [o.uid for o in doc.objects]


# --- set_props is not a locking door --------------------------------------


def test_set_props_allows_renaming_a_locked_object() -> None:
    doc = bd.ClayDoc()
    a = _locked(doc)
    assert doc.set_props(a.uid, name="Renamed") is True
    assert doc.by_uid(a.uid).name == "Renamed"


def test_set_props_allows_hiding_a_locked_object() -> None:
    doc = bd.ClayDoc()
    a = _locked(doc)
    assert doc.set_props(a.uid, visible=False) is True
    assert doc.by_uid(a.uid).visible is False


def test_set_props_allows_tagging_a_locked_object() -> None:
    doc = bd.ClayDoc()
    a = _locked(doc)
    assert doc.set_props(a.uid, tags=("Prop",)) is True
    assert doc.by_uid(a.uid).tags == ("prop",)  # normalized on the way in


def test_set_props_allows_re_pointing_the_default_material_while_locked() -> None:
    doc = bd.ClayDoc()
    a = _locked(doc)
    doc.add_material()
    assert doc.set_props(a.uid, material=1) is True
    assert doc.by_uid(a.uid).material == 1


def test_set_props_allows_unlocking() -> None:
    doc = bd.ClayDoc()
    a = _locked(doc)
    assert doc.set_props(a.uid, locked=False) is True
    assert doc.by_uid(a.uid).locked is False
    # And it is genuinely unlocked -- a door refused a moment ago now works.
    assert doc.set_transform(a.uid, translation=(1.0, 0.0, 0.0)) is True


def test_set_props_refuses_changing_the_parent() -> None:
    """``parent`` has its own door, ``set_parent``, which is the only one
    that checks for a cycle -- this generic one must not be a back door
    around it, locked or not."""
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    with pytest.raises(OpError, match="set_parent"):
        doc.set_props(b.uid, parent=a.uid)
    assert doc.by_uid(b.uid).parent is None


# --- undo/redo bypass every door, on purpose --------------------------------


def test_undo_still_works_on_a_locked_object() -> None:
    """Locked directly on the live object, not through ``set_props`` -- a
    step recorded *for* the lock would itself become the thing ``undo()``
    reverses first, which is a different claim (and its own test, above)
    from the one this pins: an *earlier* edit still undoes while the object
    stays locked throughout."""
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_transform(a.uid, translation=(1.0, 0.0, 0.0))
    doc.by_uid(a.uid).locked = True

    assert doc.undo() is True
    assert np.allclose(doc.by_uid(a.uid).translation, [0.0, 0.0, 0.0])
    assert doc.by_uid(a.uid).locked is True
    with pytest.raises(OpError, match="locked"):
        doc.set_transform(a.uid, translation=(2.0, 0.0, 0.0))


def test_redo_still_works_on_a_locked_object() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_transform(a.uid, translation=(1.0, 0.0, 0.0))
    doc.undo()
    doc.by_uid(a.uid).locked = True

    assert doc.redo() is True
    assert np.allclose(doc.by_uid(a.uid).translation, [1.0, 0.0, 0.0])
    assert doc.by_uid(a.uid).locked is True


# --- set_parent and set_origin are deliberately not locking doors ----------


def test_set_parent_is_not_gated_by_a_lock() -> None:
    """Decision 3/6 in the tranche 3 spec: reparenting with ``keep_world``
    moves nothing on screen, so it is a re-framing, not a change to what the
    object looks like -- unlike the six doors above."""
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = _locked(doc, "B")
    assert doc.set_parent(b.uid, a.uid) is True
    assert doc.by_uid(b.uid).parent == a.uid
    assert doc.by_uid(b.uid).locked is True


def test_set_parent_with_keep_world_false_refuses_a_locked_object_because_it_visibly_moves() -> None:  # noqa: E501
    """The 2026-09-22 audit's clay-04: ``keep_world=False`` does not hold
    the "moves nothing on screen" reasoning the test right above this one
    exercises -- the object's local TRS is left untouched while its parent
    changes, so it jumps to wherever the new parent's frame puts it.
    ``clay_parent`` exposes ``keep_world`` to an agent, and used to move a
    locked object with no refusal at all.
    """
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", translation=(10.0, 0.0, 0.0)))
    b = _locked(doc, "B")
    with pytest.raises(OpError):
        doc.set_parent(b.uid, a.uid, keep_world=False)
    assert doc.by_uid(b.uid).parent is None
    assert doc.by_uid(b.uid).locked is True


def test_set_origin_is_not_gated_by_a_lock() -> None:
    doc = bd.ClayDoc()
    a = _locked(doc)
    assert doc.set_origin(a.uid, (0.5, 0.0, 0.0)) is True
    assert doc.by_uid(a.uid).locked is True


# --- separate refuses a locked source ---------------------------------------


def test_separate_refuses_a_locked_object() -> None:
    from realmspinner.kernels.mesh import separate

    doc = bd.ClayDoc()
    two_boxes = bm.Mesh(
        positions=np.concatenate([bp.box().positions, bp.box().positions + 5.0]),
        loops=np.concatenate([bp.box().loops, bp.box().loops + len(bp.box().positions)]),
        starts=np.concatenate([bp.box().starts, bp.box().starts[1:] + len(bp.box().loops)]),
        material=np.concatenate([bp.box().material, bp.box().material]),
        smooth=np.concatenate([bp.box().smooth, bp.box().smooth]),
    )
    a = _locked(doc, "A", mesh=two_boxes)
    pieces = separate.by_loose_parts(two_boxes)
    with pytest.raises(OpError, match="locked"):
        doc.separate(a.uid, pieces)


# --- 2026-09-22 audit, finding clay-02: element picking is a locking door too


def test_pick_element_never_returns_an_index_into_a_locked_object(gl) -> None:
    """``pick_face`` (object mode) has always skipped a locked object --
    "viewport clicks pass through it" -- but ``pick_element`` did not, so a
    click in face mode over a locked box's only object on screen still
    selected one of its faces, the hole that let ``delete_selected`` reach a
    locked object without going through the outliner at all.
    """
    from realmspinner.kernels.geom3d import math3d as m3
    from realmspinner.studio.modes.clay.ui import view as clay_view

    class _State:
        def __init__(self) -> None:
            self.tool = "select"
            self.snap = False
            self.snap_translate = 0.125
            self.snap_rotate = 15.0
            self.snap_vertex = False

    class _Ctx:
        def __init__(self) -> None:
            self.state = type("S", (), {"clay": _State()})()

    doc = bd.ClayDoc()
    _locked(doc, "A")
    doc.set_element_mode("face")

    view = clay_view.ClayView(gl, _Ctx())
    try:
        view._rect = (0.0, 0.0, 128.0, 96.0)
        view.camera.set_target(m3.vec3())
        view.camera.set_position(m3.vec3(0.0, 0.0, 8.0))
        view.camera.aspect = view._rect[2] / view._rect[3]
        centre = (view._rect[2] * 0.5, view._rect[3] * 0.5)

        assert view.pick_face(doc, centre) is None, "object picking already skips it"
        assert view.pick_element(doc, centre) is None
    finally:
        view.release()


def test_commit_marquee_never_selects_elements_on_a_locked_object(gl) -> None:
    """Same hole as :func:`test_pick_element_never_returns_an_index_into_a_locked_object`,
    reached through a drag rather than a click: a marquee that sweeps the
    whole viewport used to add a locked box's faces to ``doc.element_sel``
    just the same."""
    from realmspinner.kernels.geom3d import math3d as m3
    from realmspinner.studio.modes.clay.ui import view as clay_view

    class _State:
        def __init__(self) -> None:
            self.tool = "select"
            self.snap = False
            self.snap_translate = 0.125
            self.snap_rotate = 15.0
            self.snap_vertex = False

    class _Ctx:
        def __init__(self) -> None:
            self.state = type("S", (), {"clay": _State()})()

    doc = bd.ClayDoc()
    locked = _locked(doc, "A")
    doc.set_element_mode("face")

    view = clay_view.ClayView(gl, _Ctx())
    try:
        view._rect = (0.0, 0.0, 128.0, 96.0)
        view.camera.set_target(m3.vec3())
        view.camera.set_position(m3.vec3(0.0, 0.0, 8.0))
        view.camera.aspect = view._rect[2] / view._rect[3]

        view.marquee = (0.0, 0.0, view._rect[2], view._rect[3])
        view._marquee_from = (0.0, 0.0)
        view._marquee_add = "replace"
        view._commit_marquee(doc)

        from realmspinner.kernels.mesh import elements as el

        assert el.is_empty(doc.element_sel_of(locked.uid))
    finally:
        view.release()
