"""The 2026-09-23 (second run) audit's "Clay locked-object family" findings.

One fix shape across five sites, all sharing the same root cause: a locked
object's own document-level refusal (``kernels.mesh.document.ClayDoc``'s six
locking doors, each raising ``OpError`` with nothing pushed) reached a pane
or an ``ops.py`` batch loop uncaught.

* clay-03 (``ui/panes/props.py``'s ``_transform``/``_generator``): typing
  into Position/Rotation/Scale or a generator field of a locked object let
  ``OpError`` fly out of the properties pane's ``draw`` call, which the
  pane's own guard catches by replacing Properties with a "stopped drawing"
  placeholder for the rest of the frame.
* clay-04 (``ui/panes/uv.py``'s four apply doors and the E/R live-drag arm):
  the UV pane never read ``obj.locked`` anywhere, so dragging, Apply
  rotate/scale, Pack islands or an armed live drag on a locked object all
  raised the same way, and an armed drag re-raised on every retry frame.
* clay-06 (``ops.py``'s ``_apply_deltas``, under Align/Distribute/Drop to
  ground): one locked object in a multi-object selection aborted the
  ancestor-ordered write loop, abandoning every object still queued behind
  it.
* clay-07 (``ops.py``'s ``run_mesh_op``): the kernel call is guarded but
  ``doc.set_mesh`` was not, so a locked object abandoned every later object
  in the same element-mode batch -- contradicting the function's own
  docstring ("a refusal on one object does not abandon the others").
* clay-08 (``ops.py``'s ``_knife``): the same gap in the hand-written loop
  ``run_mesh_op`` shares the shape of but does not share the code of.
* clay-09 (``ops.py``'s ``_decimate_apply``/``_retopo_apply``/
  ``_unwrap_apply``): a locked object skipped by the 2026-09-22 audit's
  clay-03 fix (which only stopped the undo-stack leak) was reported with the
  same sentence a *stamp mismatch* gets -- "it changed while decimating"/
  "it changed while running" -- which is false; it never changed at all.

Every "before" claim below was checked by loading ``git show HEAD:<path>``'s
own version of the relevant function into a throwaway namespace (never by
writing it into the tree) and calling it the same way the test below does;
see the fix's own return message for the transcripts.
"""

from __future__ import annotations

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import uv as bmuv
from realmspinner.kernels.mesh import uvtools
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.ui.panes import props as clay_props
from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv


class _Ctx:
    """Only what ``Ctx`` really offers -- ``test_clay_ops.py``'s own double,
    repeated here rather than imported: a ``ctx.toasts`` attribute the real
    ``Ctx`` has never had is exactly how a refusal used to vanish silently in
    the running app and only ever surface in a test with the wrong double."""

    def __init__(self) -> None:
        self.toasted: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasted.append((message, level))


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


# --- clay-03: the properties pane's transform and generator fields ----------


def test_editing_a_locked_objects_transform_field_toasts_instead_of_crashing(
    ui, monkeypatch
) -> None:
    """Fails against the unfixed ``_transform``: with no ``try/except``
    around ``doc.set_transform``, the object's own ``OpError`` ("'A' is
    locked.") flies straight out of this call, which is exactly what the
    properties pane's ``_body`` guard catches by replacing the whole pane
    with "stopped drawing" for the rest of the frame."""
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    doc.set_props(obj.uid, locked=True)

    def spy(label, values, axes, **kwargs):
        if label == "position##bt":
            return True, [9.0, 9.0, 9.0]
        return False, list(values)

    monkeypatch.setattr(clay_props.controls, "input_vec", spy)
    ctx = _Ctx()

    ui.new_frame()
    ui.begin("##host")
    try:
        clay_props._transform(doc, doc.by_uid(obj.uid), ctx=ctx)
    finally:
        ui.end()
        ui.end_frame()

    assert list(doc.by_uid(obj.uid).translation) == [0.0, 0.0, 0.0], "the edit did not apply"
    assert ctx.toasted, "the refusal is toasted, not raised"
    assert "locked" in ctx.toasted[0][0]


def test_editing_a_locked_objects_generator_field_toasts_instead_of_crashing(
    ui, monkeypatch
) -> None:
    """``_generator``'s own copy of the same gap: ``set_generator_params``
    raises the identical way ``set_transform`` does, and nothing in this
    function caught it either."""
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Box",
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
        )
    )
    doc.set_props(obj.uid, locked=True)

    def fake_widget(key, value, default):
        if key == "size":
            return (2.0, 1.0, 1.0), True
        return value, False

    monkeypatch.setattr(clay_props, "_widget", fake_widget)
    ctx = _Ctx()

    ui.new_frame()
    ui.begin("##host")
    try:
        clay_props._generator(doc, doc.by_uid(obj.uid), ctx=ctx)
    finally:
        ui.end()
        ui.end_frame()

    assert doc.by_uid(obj.uid).params["size"] == (1.0, 1.0, 1.0), "the edit did not apply"
    assert ctx.toasted
    assert "locked" in ctx.toasted[0][0]


# --- clay-04: the UV pane's four apply doors and the live-drag arm ----------


def test_dragging_a_uv_island_on_a_locked_object_toasts_instead_of_crashing() -> None:
    """``apply_translate`` (the move-drag door ``_canvas`` calls every
    frame of a box/move gesture) reaches ``doc.set_mesh``, which refuses a
    locked object. Fails against the unfixed door: it took no ``ctx`` at all
    and had nothing to catch the refusal with, so it flew straight out to
    the canvas's own frame loop.
    """
    doc = bd.ClayDoc()
    mesh = bmuv.box_unwrap(bp.box())
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(), name="Box", mesh=mesh, generator="box", params={"size": (1, 1, 1)}
        )
    )
    doc.set_props(obj.uid, locked=True)
    ids = {int(i) for i in np.unique(uvtools.islands(mesh))}
    before_uv = np.array(doc.by_uid(obj.uid).mesh.uv, copy=True)
    before_history = len(doc.history)
    ctx = _Ctx()

    changed = clay_uv.apply_translate(doc, obj.uid, ids, (0.1, 0.05), ctx=ctx)

    assert changed is False
    assert len(doc.history) == before_history, "nothing pushed"
    assert np.allclose(doc.by_uid(obj.uid).mesh.uv, before_uv), "the uv did not move"
    assert ctx.toasted
    assert "locked" in ctx.toasted[0][0]


def test_uv_apply_rotate_scale_and_pack_all_toast_a_locked_object_too() -> None:
    """The other three of the "four apply doors" the finding names --
    :func:`apply_rotate`, :func:`apply_scale`, :func:`apply_pack`."""
    doc = bd.ClayDoc()
    mesh = bmuv.box_unwrap(bp.box())
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(), name="Box", mesh=mesh, generator="box", params={"size": (1, 1, 1)}
        )
    )
    doc.set_props(obj.uid, locked=True)
    ids = {int(i) for i in np.unique(uvtools.islands(mesh))}

    for door, args in (
        (clay_uv.apply_rotate, (obj.uid, ids, 90.0)),
        (clay_uv.apply_scale, (obj.uid, ids, 0.5)),
        (clay_uv.apply_pack, (obj.uid,)),
    ):
        ctx = _Ctx()
        before_history = len(doc.history)
        result = door(doc, *args, ctx=ctx)
        assert result is False, door.__name__
        assert len(doc.history) == before_history, door.__name__
        assert ctx.toasted, door.__name__
        assert "locked" in ctx.toasted[0][0], door.__name__


def test_canvas_refuses_to_arm_a_live_drag_on_a_locked_object() -> None:
    """The arm side of clay-04, checked by source rather than a live frame --
    the same choice ``test_clay_props_undo.py`` makes for
    ``clay_props._transform``/``_generator``, and for the identical reason:
    ``_canvas`` draws pan/zoom, box/move *and* the E/R arm in one function
    with no single imgui item a headless frame's "one active item" model can
    attribute a key press to.

    Before the fix, E/R armed the live rotate/scale gesture on a locked
    object exactly as freely as an unlocked one -- only the eventual
    ``update_live_transform`` -> ``apply_rotate``/``apply_scale`` ->
    ``doc.set_mesh`` refused, and it refused on *every* mouse-move frame
    until Escape, since nothing ever un-armed the drag. Refusing to arm at
    all (an explicit ``not obj.locked`` beside the existing
    ``view_state.selected_islands`` check) means there is no live gesture
    left to keep re-raising.
    """
    import inspect

    source = inspect.getsource(clay_uv._canvas)
    arm_block = source.split("if imgui.is_key_pressed(imgui.Key.e):", 1)[0]
    # The gate block ends where the E-key check begins; the guard must sit in
    # the ``if (...)`` above it, alongside the existing selection check.
    gate = arm_block.rsplit("if (", 1)[1]
    assert "obj.locked" in gate, "the E/R arm gate never reads obj.locked"


# --- clay-06: Align/Distribute/Drop to ground do not abort the whole batch --


def test_align_does_not_silently_refuse_the_whole_batch_when_one_selected_object_is_locked() -> (
    None
):
    """Fails against the unfixed ``_apply_deltas``: its loop calls
    ``doc.set_transform`` with no ``try/except`` at all, so the first locked
    object it reaches (ancestor-first order; both are roots here, so
    ``A`` sorts before ``B`` by insertion) raises straight out of ``_align``
    and ``B`` -- which has nothing to do with the lock -- never moves.
    """
    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    b = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    doc.set_transform(b.uid, translation=(5.0, 0.0, 0.0))
    doc.set_props(a.uid, locked=True)
    doc.select([a.uid, b.uid])

    ctx = _Ctx()
    b_before = np.array(doc.by_uid(b.uid).translation, copy=True)

    ran = clay_ops.run(ctx, doc, clay_ops.get("align"), axis=0.0, mode=1.0)

    assert ran is True, "B's own move still counts as the batch having run"
    b_after = np.array(doc.by_uid(b.uid).translation, copy=True)
    assert not np.array_equal(b_after, b_before), "B must still move despite A being locked"
    assert list(doc.by_uid(a.uid).translation) == [0.0, 0.0, 0.0], "A itself never moves"
    assert any("locked" in message for message, _ in ctx.toasted), ctx.toasted


# --- clay-07: run_mesh_op guards set_mesh, not only the kernel call ---------


def _identity_mesh_op(mesh: bm.Mesh, sel, **params):
    """A trivial ``(mesh, sel) -> (mesh, sel)`` op that never raises and
    always hands back a *new* ``Mesh`` object -- ``doc.set_mesh`` decides
    "did anything happen" by identity (its own docstring), so returning the
    same object back would never register as a change to guard against."""
    new_mesh = bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
    )
    return new_mesh, sel


def test_run_mesh_op_does_not_abandon_a_later_object_when_an_earlier_selected_one_is_locked() -> (
    None
):
    """Fails against the unfixed loop: the kernel call (``func(...)``) is
    guarded, but the ``doc.set_mesh`` call right after it is not, so ``A``'s
    lock refusal flies straight out of ``run_mesh_op`` and ``B`` -- next in
    ``doc.element_sel``'s own iteration order -- is never reached at all,
    contradicting this function's own docstring ("a refusal on one object
    does not abandon the others").
    """
    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    b = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    doc.set_props(a.uid, locked=True)
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, el.ElementSel(faces=[0]))
    doc.set_element_sel(b.uid, el.ElementSel(faces=[0]))

    ctx = _Ctx()
    b_before = doc.by_uid(b.uid).mesh

    ran = clay_ops.run_mesh_op(ctx, doc, _identity_mesh_op)

    assert ran is True, "B's own edit still counts as something having run"
    assert doc.by_uid(b.uid).mesh is not b_before, "B must still be edited despite A being locked"
    assert any("locked" in message for message, _ in ctx.toasted), ctx.toasted


# --- clay-08: the knife's hand-written loop has the identical gap ----------


def test_knife_does_not_abandon_a_later_object_when_an_earlier_selected_one_is_locked() -> None:
    """Fails against the unfixed ``_knife``: same shape as ``run_mesh_op``'s
    own gap -- ``ops_model.knife`` is guarded, ``doc.set_mesh`` is not."""
    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    b = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    doc.set_props(a.uid, locked=True)
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, el.ElementSel(faces=list(range(6))))
    doc.set_element_sel(b.uid, el.ElementSel(faces=list(range(6))))

    ctx = _Ctx()
    b_before = doc.by_uid(b.uid).mesh

    ran = clay_ops._knife(ctx, doc, point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))

    assert ran is True
    assert doc.by_uid(b.uid).mesh is not b_before, "B must still be cut despite A being locked"
    assert bm.face_count(doc.by_uid(b.uid).mesh) == 10, "B's cut actually landed"
    assert any("locked" in message for message, _ in ctx.toasted), ctx.toasted


# --- clay-09: Decimate/Retopologize/Smart Unwrap's own toast sentence ------


def test_decimate_apply_toast_does_not_claim_a_locked_skip_changed_while_decimating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails against the unfixed ``_decimate_apply``: a locked object's
    ``set_mesh`` refusal landed in the same ``skipped`` list a stamp
    mismatch does, so ``_decimate_report`` said "it changed while
    decimating" for an object that never changed at all -- it was simply
    locked.
    """
    monkeypatch.setattr(clay_ops, "_decimate_mesh_from_glb", lambda data, material: bp.box())

    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    doc.set_props(obj.uid, locked=True)

    result = {
        "items": [
            {
                "uid": obj.uid,
                "name": obj.name,
                "stamp": doc.mesh_stamp(obj.uid),
                "glb_out": doc.by_uid(obj.uid).mesh,
                "material": 0,
                "before": 12,
            }
        ],
        "ratio": 0.5,
    }

    ctx = _Ctx()
    clay_ops._decimate_apply(ctx, doc, result)

    assert ctx.toasted
    message = ctx.toasted[0][0]
    assert "changed while decimating" not in message, message
    assert "it is locked" in message, message
