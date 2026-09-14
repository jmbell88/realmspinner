"""T5's service door onto Familiar: ``service/familiar.py``.

``llama_client.chat`` is monkeypatched module-wide rather than driven through
a real ``httpx`` transport here -- what is under test is the door's own
refusal mapping and gating, which ``tests/familiar/test_llama_client.py``
already covers at the HTTP layer. Building a real ``WarlockService`` is
unnecessary too: both service functions only ever touch ``svc.call_on_loop``
and ``svc.worker.familiar``, so a bare fake stands in for it.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from warlock import models
from warlock.service import familiar as svc_familiar
from warlock.service.familiar import FamiliarRefusal
from warlock.studio.familiar import contract


class _FakeSvc:
    """Just enough of ``WarlockService`` for the two doors under test:
    ``svc.worker.familiar`` (never actually touched, since ``llama_client.
    chat`` is monkeypatched below) and a synchronous ``call_on_loop``."""

    def __init__(self) -> None:
        self.worker = type("Worker", (), {"familiar": object()})()

    def call_on_loop(self, coro_factory, timeout: float = 30.0):
        return asyncio.run(coro_factory())


def test_the_lease_refusal_reaches_the_caller_as_a_lease_reason(monkeypatch):
    """``LlamaServer.ensure_started``'s exact lease sentence must survive the
    door as a ``FamiliarRefusal`` whose ``reason`` is ``"lease"`` -- the one
    refusal the T5 brief says keeps its exact wording, because the pane shows
    it verbatim rather than a generic "Familiar is busy" line."""

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, transport=None):
        raise RuntimeError(
            "Familiar cannot start while a GPU job holds the card -- "
            "it will restart on your next message."
        )

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.chat_reply(_FakeSvc(), "hello")

    assert excinfo.value.reason == "lease"
    assert excinfo.value.message == (
        "Familiar cannot start while a GPU job holds the card -- "
        "it will restart on your next message."
    )


def test_clay_build_on_the_testing_pin_refuses_before_any_request(monkeypatch):
    """The testing pin's ``card_shas`` is empty, so a Clay build must refuse
    with reason ``"card"`` *before* ``llama_client.chat`` is ever called --
    not after a wasted round trip that would fail anyway (run A's own
    measurement: base Gemma scores 0% door acceptance on Clay builds)."""
    called = []

    async def fake_chat(*args, **kwargs):
        called.append(True)
        return "should never run"

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)
    assert models.FAMILIAR_MODELS["familiar_gguf"].card_shas == ()

    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.clay_build(_FakeSvc(), "build a box", {"objects": []})

    assert excinfo.value.reason == "card"
    assert "familiar_v1.0" in excinfo.value.message
    assert called == [], "clay_build must gate before any request, not after"


def test_plain_chat_starts_on_the_testing_pin(monkeypatch):
    """Plain chat must pass ``expected_card_sha=None`` and ``skill=None`` --
    the "no card in play" shape ``LlamaServer._check_card_sha`` never refuses
    -- so it works on the base testing pin in every mode, unlike Clay."""
    seen = {}

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, transport=None):
        seen["skill"] = skill
        seen["expected_card_sha"] = expected_card_sha
        seen["messages"] = messages
        return "hi there"

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    reply = svc_familiar.chat_reply(_FakeSvc(), "hello")

    assert reply == "hi there"
    assert seen["skill"] is None
    assert seen["expected_card_sha"] is None
    assert seen["messages"][-1] == {"role": "user", "content": "hello"}


def test_chat_with_no_worker_is_refused_rather_than_answering_nothing():
    """``call_on_loop`` returns ``None`` when the service has no worker, so a
    door that trusted it would hand the pane an empty "reply"."""
    svc = _FakeSvc()
    svc.worker = None

    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.chat_reply(svc, "hello")

    assert excinfo.value.reason == "missing"


def test_the_door_waits_out_a_cold_start_plus_a_full_reply():
    """A cold ``ensure_started`` can take ``STARTUP_TIMEOUT`` before the chat
    round trip's own ``CHAT_TIMEOUT`` starts; a loop timeout equal to the chat
    timeout alone gave up on every slow cold start."""
    from warlock.pipelines import llama, llama_client

    assert svc_familiar.LOOP_TIMEOUT > llama.STARTUP_TIMEOUT + llama_client.CHAT_TIMEOUT


def test_a_loop_timeout_is_a_refusal_not_an_unmapped_error(monkeypatch):
    class _SlowSvc(_FakeSvc):
        def call_on_loop(self, coro_factory, timeout: float = 30.0):
            raise TimeoutError

    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.chat_reply(_SlowSvc(), "hello")

    assert excinfo.value.reason == "unhealthy"


def test_an_unparseable_reply_is_a_parse_refusal(monkeypatch):
    """A reply with no fenced ``{"calls": [...]}}`` JSON must surface as a
    ``FamiliarRefusal`` with reason ``"parse"`` -- ``contract.parse_calls``'s
    own three failure modes, never an uncaught ``KeyError``/``TypeError``
    reaching the pane."""

    async def fake_chat(*args, **kwargs):
        return "sure, here you go: no tool call at all"

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)
    fine_tuned = dataclasses.replace(
        models.FAMILIAR_MODELS["familiar_gguf"], card_shas=(contract.card_sha("clay"),)
    )
    monkeypatch.setitem(models.FAMILIAR_MODELS, "familiar_gguf", fine_tuned)

    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.clay_build(_FakeSvc(), "build a box", {"objects": []})

    assert excinfo.value.reason == "parse"
