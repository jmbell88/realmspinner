"""F20-8: ``assistant/ui.py`` tagged a build with a bare ``id(tab.doc)``, the
clay-40 hole (a freed document's address handed to its replacement). The tag now
holds a weakref and compares identity through it, as ``PreviewDiff.is_base`` does."""

from __future__ import annotations

from realmspinner.kernels.mesh import document as bd
from realmspinner.studio.assistant import ui as familiar_ui
from realmspinner.studio.tasks import Done

from .test_familiar_ui import _canned_calls, _FakeCtx, _run_land


def test_a_replacement_document_at_the_same_address_is_not_taken_for_the_base(monkeypatch):
    # Address reuse is the allocator's choice, so simulate it: every object
    # "has" the same id. An id-keyed tag cannot tell the replacement from the base.
    monkeypatch.setattr(familiar_ui, "id", lambda _obj: 1, raising=False)
    ctx = _FakeCtx(bd.ClayDoc(), mode="clay")
    done = Done(
        key=familiar_ui.BUILD_KEY,
        result=_canned_calls(),
        tag={"thread_key": ("clay", ctx.tab.uid), "tab_uid": ctx.tab.uid},
    )
    familiar_ui.on_task_done(ctx, done)  # phase one: the base is snapshotted

    ctx.tab.doc = bd.ClayDoc()  # a revert / reload swapped in a fresh document

    _run_land(ctx)

    ui = familiar_ui.ensure(ctx)
    assert ui.preview_calls is None
    assert ui.message is not None and "preview again" in ui.message
