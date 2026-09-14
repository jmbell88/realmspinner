"""``familiar.apply``: turning a preview into a real edit, and refusing to
when the document has moved out from under it.

See ``apply.py``'s own module docstring for the refusal rule and why "no tab
open" is not one of the cases it covers.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.studio import clay_mode, familiar
from warlock.studio.clay import document as bd
from warlock.studio.clay import primitives as bp
from warlock.studio.clay import scratch as clay_scratch
from warlock.studio.clay import serialize


class _FakeCtx:
    def __init__(self, doc: bd.ClayDoc, *, grabbing: bool = False) -> None:
        tab = clay_mode.ClayTab(doc=doc)
        self.state = SimpleNamespace(clay=clay_mode.ClayState(docs=[tab], active_uid=tab.uid))
        self.settings = SimpleNamespace()
        self._tab = tab
        self.clay_view = SimpleNamespace(grabbing=grabbing, cleared=0)
        self.clay_view.clear_preview = lambda: setattr(
            self.clay_view, "cleared", self.clay_view.cleared + 1
        )

    @property
    def tab(self):
        return self._tab


def _seeded_doc() -> bd.ClayDoc:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box()))
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="b", mesh=bp.box()))
    return doc


def _preview_that_adds_and_moves(doc: bd.ClayDoc):
    scratch = clay_scratch.clone(doc)
    kept_uid = scratch.objects[0].uid
    scratch.set_transform(kept_uid, translation=[4.0, 0.0, 0.0])
    added = scratch.add_objects(
        [bd.Obj(uid=bd.new_uid(), name="agent_added", mesh=bp.box())]
    )[0]
    diff = clay_scratch.diff(doc, scratch)
    return scratch, diff, kept_uid, added.uid


def test_apply_transplants_exactly_what_the_preview_showed():
    doc = _seeded_doc()
    ctx = _FakeCtx(doc)
    scratch, diff, kept_uid, added_uid = _preview_that_adds_and_moves(doc)

    result = familiar.apply(ctx, ctx.tab.uid, diff, scratch)

    assert result["ok"] is True
    assert (doc.by_uid(kept_uid).translation == [4.0, 0.0, 0.0]).all()
    assert any(o.uid == added_uid for o in doc.objects)
    assert ctx.clay_view.cleared == 1


def test_apply_is_one_undo_step_and_undo_restores_the_base():
    doc = _seeded_doc()
    ctx = _FakeCtx(doc)
    before_head = doc.history.head
    before_uids = {o.uid for o in doc.objects}
    scratch, diff, kept_uid, added_uid = _preview_that_adds_and_moves(doc)

    result = familiar.apply(ctx, ctx.tab.uid, diff, scratch)
    assert result["ok"] is True
    assert doc.history.head != before_head

    assert doc.undo()
    assert doc.history.head == before_head
    assert {o.uid for o in doc.objects} == before_uids
    assert (doc.by_uid(kept_uid).translation == [0.0, 0.0, 0.0]).all()


def test_apply_refuses_when_the_user_edited_undid_or_is_dragging():
    doc = _seeded_doc()
    ctx = _FakeCtx(doc)
    scratch, diff, kept_uid, added_uid = _preview_that_adds_and_moves(doc)

    # The user made a real edit in between: the head has moved.
    doc.set_transform(doc.objects[1].uid, translation=[1.0, 0.0, 0.0])
    result = familiar.apply(ctx, ctx.tab.uid, diff, scratch)
    assert result["ok"] is False
    assert "preview again" in result["message"]
    assert not any(o.uid == added_uid for o in doc.objects)

    doc.undo()  # back to the base head
    ctx2 = _FakeCtx(doc, grabbing=True)
    dragging_result = familiar.apply(ctx2, ctx2.tab.uid, diff, scratch)
    assert dragging_result["ok"] is False

    ctx3 = _FakeCtx(doc)
    ctx3.tab.saving = True
    saving_result = familiar.apply(ctx3, ctx3.tab.uid, diff, scratch)
    assert saving_result["ok"] is False

    ctx4 = _FakeCtx(doc)
    doc.select([doc.objects[0].uid])  # selection differs from the base snapshot
    selection_result = familiar.apply(ctx4, ctx4.tab.uid, diff, scratch)
    assert selection_result["ok"] is False


def test_apply_with_material_removal_keeps_face_indices_right():
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="multi", mesh=bp.box()))
    extra = doc.add_material()
    third_material = doc.materials[doc.add_material()]
    from warlock.studio.clay import mesh as bm

    faces = obj.mesh.material.copy()
    third_index = next(i for i, m in enumerate(doc.materials) if m is third_material)
    faces[: len(faces) // 2] = third_index
    repainted = bm.Mesh(
        positions=obj.mesh.positions, loops=obj.mesh.loops, starts=obj.mesh.starts,
        smooth=obj.mesh.smooth, material=faces,
    )
    doc.set_mesh(obj.uid, repainted, keep_generator=True)

    ctx = _FakeCtx(doc)
    scratch = clay_scratch.clone(doc)
    assert scratch.remove_material(extra)
    diff = clay_scratch.diff(doc, scratch)

    result = familiar.apply(ctx, ctx.tab.uid, diff, scratch)
    assert result["ok"] is True
    assert len(doc.materials) == 2
    new_index = next(i for i, m in enumerate(doc.materials) if m is third_material)
    live = doc.by_uid(obj.uid).mesh.material
    assert (live[: len(live) // 2] == new_index).all()


def test_apply_with_no_tab_open_mints_exactly_one_document():
    # A scratch run started against an empty base -- nothing the interactive
    # UI ever had open -- and added one object.
    base = bd.ClayDoc()
    scratch = clay_scratch.clone(base)
    added = scratch.add_objects([bd.Obj(uid=bd.new_uid(), name="agent_added", mesh=bp.box())])[0]
    diff = clay_scratch.diff(base, scratch)

    ctx = SimpleNamespace(state=SimpleNamespace(clay=None), settings=SimpleNamespace())

    result = familiar.apply(ctx, "", diff, scratch)
    assert result["ok"] is True
    state = clay_mode.ensure(ctx)
    assert len(state.docs) == 1
    assert any(o.uid == added.uid for o in state.docs[0].doc.objects)


def test_discard_leaves_the_document_byte_identical():
    doc = _seeded_doc()
    before_bytes = serialize.wblk_bytes(doc)
    ctx = _FakeCtx(doc)
    scratch, diff, kept_uid, added_uid = _preview_that_adds_and_moves(doc)

    familiar.discard(ctx, ctx.tab.uid)

    assert serialize.wblk_bytes(doc) == before_bytes
    assert ctx.clay_view.cleared == 1
