"""``familiar_preview``'s scratch ctx: an agent tool call run against a
scratch clone, never the real document.

Lives under ``tests/familiar/`` even though the code moved to
``studio/assistant/preview.py`` (2026-09-14, so ``studio/familiar/`` could be
made genuinely headless) -- see ``test_apply.py``'s own docstring for why.

See that module's own docstring for why the sandbox is a whole private
``ClayState`` rather than the real ``ctx`` with a tab swapped in.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import primitives as bp
from warlock.studio.assistant import preview as familiar_preview
from warlock.studio.modes.clay import mode as clay_mode
from warlock.studio.modes.clay.agent import dispatch as agent_clay


def _payload(result: dict):
    return json.loads(result["content"][0]["text"])


class _RealCtx:
    """The real app ``ctx``, holding the user's own document -- exactly what
    a scratch run must never see or touch."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.settings = SimpleNamespace()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _real_doc_with_one_box() -> tuple[_RealCtx, bd.ClayDoc]:
    ctx = _RealCtx()
    session = agent_clay.Session()
    result = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    assert result["isError"] is False, result
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    return ctx, tab.doc


def test_a_scratch_run_never_touches_the_real_clay_state():
    real_ctx, real_doc = _real_doc_with_one_box()
    real_state = clay_mode.ensure(real_ctx)
    real_history_len = len(real_doc.history.history())
    real_object_count = len(real_doc.objects)

    scratch_ctx = familiar_preview.build(real_doc)
    result = familiar_preview.run_scratch(
        scratch_ctx, "clay_add_primitive", {"generator": "uv_sphere"}
    )
    assert result["isError"] is False, result

    # The real document is untouched, by every measure.
    assert len(real_doc.history.history()) == real_history_len
    assert len(real_doc.objects) == real_object_count
    # And the real app's ClayState never gained a second tab.
    assert clay_mode.ensure(real_ctx) is real_state
    assert len(real_state.docs) == 1


def test_a_scratch_run_never_mints_a_tab():
    """Building a scratch ctx pins ``Session.tab_uid`` up front, so a
    ``create=True`` tool (``clay_add_primitive`` among them) can never reach
    ``clay_mode.new_document`` -- there is nothing in ``familiar_preview``
    that ever calls it."""
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="seed", mesh=bp.box()))

    scratch_ctx = familiar_preview.build(doc)
    state = scratch_ctx.state.clay
    assert len(state.docs) == 1
    before_uid = state.docs[0].uid

    result = familiar_preview.run_scratch(scratch_ctx, "clay_add_primitive", {"generator": "box"})
    assert result["isError"] is False, result

    # Still exactly the one tab this scratch ctx was built with.
    assert len(state.docs) == 1
    assert state.docs[0].uid == before_uid


@pytest.mark.parametrize("name", sorted(familiar_preview.PREVIEW_EXCLUDED))
def test_preview_excluded_tools_are_refused_before_the_door(name: str):
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="seed", mesh=bp.box()))
    scratch_ctx = familiar_preview.build(doc)
    history_len = len(scratch_ctx.state.clay.docs[0].doc.history.history())

    result = familiar_preview.run_scratch(scratch_ctx, name, {})

    assert result["isError"] is True
    # Refused before agent_clay.call ever ran a handler: the scratch document
    # is untouched, whatever the tool's arguments would otherwise have done.
    assert len(scratch_ctx.state.clay.docs[0].doc.history.history()) == history_len


def test_a_non_excluded_tool_reaches_the_scratch_document():
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="seed", mesh=bp.box()))
    scratch_ctx = familiar_preview.build(doc)

    result = familiar_preview.run_scratch(scratch_ctx, "clay_add_primitive", {"generator": "box"})
    assert result["isError"] is False, result
    payload = _payload(result)
    assert "uid" in payload
    assert len(scratch_ctx.state.clay.docs[0].doc.objects) == 2


@pytest.mark.parametrize(
    "name", sorted(agent_clay.REFERENCE_TOOLS - {"clay_reference_get"})
)
def test_a_familiar_build_batch_cannot_nest_a_preview_excluded_reference_tool(
    name: str,
) -> None:
    """Regression, the 2026-09-18 audit's familiar-04: ``PREVIEW_EXCLUDED``
    refuses all four reference tools when ``run_scratch`` is handed one
    directly, but a Familiar build never is -- ``_submit_build_preview``
    (``studio/assistant/ui.py``) always calls ``run_scratch(scratch_ctx,
    "clay_batch", {"calls": [...]})``, and at HEAD nothing inside
    ``run_scratch`` ever looks past that outer ``"clay_batch"`` name at what
    the batch itself nests. A first attempt fixed this by folding
    ``clay_reference_add``/``_list``/``_remove`` into
    ``agent_clay_schema.BATCH_EXCLUDED`` -- reworked out again (see
    ``REFERENCE_TOOLS``'s own docstring there) because that constant is
    published in ``clay_batch``'s own description, so widening it changed the
    live tool catalogue and broke a pinned training-dataset hash
    (``dev/tests/familiar/test_contract.py``) and silently made these three
    tools unbatchable for every non-Familiar MCP caller too. The real fix is
    ``run_scratch`` itself walking a ``clay_batch`` call's own ``calls`` and
    refusing any nested name in ``PREVIEW_EXCLUDED``
    (:func:`~.preview._batch_nests_a_preview_excluded_tool`), a preview-only
    door invisible to the published catalogue.

    Fails against HEAD: with no such walk and the original, narrower
    ``BATCH_EXCLUDED``, a batch nesting one of these three either ran the
    handler (``clay_reference_list`` takes no required arguments and would
    answer ``isError: False``) or refused for a wholly different reason (a
    missing required argument, never mentioning "excluded"), so the message
    assertion below distinguishes "refused because excluded" from any other
    refusal shape.
    """
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="seed", mesh=bp.box()))
    scratch_ctx = familiar_preview.build(doc)
    history_len = len(scratch_ctx.state.clay.docs[0].doc.history.history())

    # No arguments given: the nested-name check runs before anything about
    # the entry's own arguments is inspected, so what this proves is that
    # the *name* is refused for being preview-excluded, not that this
    # particular call would have succeeded with real arguments.
    result = familiar_preview.run_scratch(
        scratch_ctx, "clay_batch", {"calls": [{"name": name}]}
    )

    assert result["isError"] is True
    assert "excluded from Familiar scratch runs" in result["content"][0]["text"]
    # Refused before agent_clay.call ever ran the batch: the scratch
    # document and this session's own references are both untouched.
    assert len(scratch_ctx.state.clay.docs[0].doc.history.history()) == history_len
