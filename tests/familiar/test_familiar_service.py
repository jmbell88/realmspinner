"""T5's service door onto Familiar: ``service/familiar.py``.

``llama_client.chat`` is monkeypatched module-wide rather than driven through
a real ``httpx`` transport here -- what is under test is the door's own
refusal mapping and gating, which ``tests/familiar/test_llama_client.py``
already covers at the HTTP layer. Building a real ``RealmspinnerService`` is
unnecessary too: both service functions only ever touch ``svc.call_on_loop``
and ``svc.worker.familiar``, so a bare fake stands in for it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from types import SimpleNamespace

import httpx
import pytest

from realmspinner import models
from realmspinner.familiar import contract, retrieval, router
from realmspinner.service import characters as svc_characters
from realmspinner.service import familiar as svc_familiar
from realmspinner.service.core import RealmspinnerService
from realmspinner.service.errors import Invalid
from realmspinner.service.familiar import FamiliarRefusal


class _FakeSvc:
    """Just enough of ``RealmspinnerService`` for the two doors under test:
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
                         expected_card_sha=None, response_format=None, transport=None):
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
    measurement on the previous, Gemma 4 E2B pin: base Gemma scores 0% door
    acceptance on Clay builds; the untuned base is no more trusted on the
    current Qwen3-VL-4B-Instruct pin, which has no eval corpus of its own
    yet either)."""
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


def test_clay_build_docstring_cites_the_document_with_the_door_acceptance_figures():
    """The 2026-09-16 audit (familiar-02): ``clay_build``'s docstring
    attributed the 0 % -> 74 % door-acceptance figures to
    ``dev/measurements/2026-09-14-clay-assistant-ablation.md``, but that
    document contains neither number -- the table is actually in
    ``dev/measurements/2026-09-12-clay-assistant-run-A.md`` (lines 42, 82),
    corroborated in ``dev/measurements/2026-09-14-familiar-base-vram.md``
    (86-88). A reader checking the evidence behind the ``reason="card"``
    refusal must be sent to the document that actually holds the number."""
    doc = svc_familiar.clay_build.__doc__
    assert "2026-09-12-clay-assistant-run-A.md" in doc
    assert "2026-09-14-clay-assistant-ablation.md" not in doc


def test_plain_chat_starts_on_the_testing_pin(monkeypatch):
    """Plain chat must pass ``expected_card_sha=None`` and ``skill=None`` --
    the "no card in play" shape ``LlamaServer._check_card_sha`` never refuses
    -- so it works on the base testing pin in every mode, unlike Clay."""
    seen = {}

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
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
    from realmspinner.familiar import llama_client
    from realmspinner.pipelines import llama

    assert svc_familiar.LOOP_TIMEOUT > llama.STARTUP_TIMEOUT + llama_client.CHAT_TIMEOUT


def test_a_loop_timeout_is_a_refusal_not_an_unmapped_error(monkeypatch):
    class _SlowSvc(_FakeSvc):
        def call_on_loop(self, coro_factory, timeout: float = 30.0):
            raise TimeoutError

    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.chat_reply(_SlowSvc(), "hello")

    assert excinfo.value.reason == "unhealthy"


class _LoopSvc:
    """Enough of ``RealmspinnerService`` to reach the real ``call_on_loop`` --
    unlike ``_FakeSvc`` above (a synchronous stand-in that never goes near
    ``run_coroutine_threadsafe``), this is what the primitive actually does
    on a loop thread, since familiar-05 is a bug in that primitive itself,
    not in ``_call``'s own except clause."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.worker = type("Worker", (), {"familiar": object()})()
        self.loop = loop

    call_on_loop = RealmspinnerService.call_on_loop


def test_a_loop_timeout_cancels_the_abandoned_chat_request_instead_of_leaving_it_running(
    monkeypatch,
):
    """The 2026-09-17 audit (familiar-05): ``call_on_loop``'s
    ``fut.result(timeout)`` never cancelled the ``run_coroutine_threadsafe``
    future it gave up waiting on, so a timed-out chat request kept running on
    the ``realmspinner-loop`` thread -- holding its llama-server slot with nothing
    to reclaim it, and its eventual (late) end logged nowhere. A retry then
    queued behind a request nobody was still waiting for. Proven with a real
    event loop on its own thread, not the synchronous ``_FakeSvc`` stand-in
    the rest of this module uses, because the bug is specifically in what
    happens to the *abandoned coroutine*, which a synchronous fake never has."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    started = threading.Event()
    cancelled = threading.Event()

    async def fake_chat(*args, **kwargs):
        started.set()
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "too slow to matter"  # pragma: no cover -- never reached

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)
    monkeypatch.setattr(svc_familiar, "LOOP_TIMEOUT", 0.05)

    try:
        with pytest.raises(FamiliarRefusal) as excinfo:
            svc_familiar.chat_reply(_LoopSvc(loop), "hello")
        assert excinfo.value.reason == "unhealthy"
        assert started.wait(1), "the fake chat coroutine never ran at all"
        assert cancelled.wait(1), (
            "a timed-out call_on_loop must cancel the abandoned coroutine, "
            "not leave it running on the loop"
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


def test_an_httpx_transport_timeout_is_a_familiar_refusal_not_an_unmapped_exception(
    monkeypatch,
):
    """The 2026-09-16 audit (familiar-01): ``_call``'s except clauses named
    ``TimeoutError``/``concurrent.futures.TimeoutError`` (the *outer*
    ``LOOP_TIMEOUT`` bound) but never ``httpx``'s own transport exceptions --
    so the *inner* timeout, ``llama_client.CHAT_TIMEOUT``, which always fires
    first (``LOOP_TIMEOUT = STARTUP_TIMEOUT + CHAT_TIMEOUT + 30``), escaped
    ``_call`` as a raw ``httpx.ReadTimeout`` instead of the "Familiar did not
    answer in time -- try again" refusal LOOP_TIMEOUT's own commentary is
    built to produce."""

    async def fake_chat(*args, **kwargs):
        raise httpx.ReadTimeout("the read operation timed out")

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.chat_reply(_FakeSvc(), "hello")

    assert excinfo.value.reason == "unhealthy"


def test_a_key_file_deleted_mid_request_is_a_familiar_refusal_not_an_unclassified_oserror(
    monkeypatch,
):
    """familiar-01 (2026-09-18 audit, second run): a chat racing
    ``LlamaServer.stop()`` can land between ``llama_client._headers`` reading
    the key file and ``pipelines.llama._release_key_file`` unlinking it --
    a raw ``FileNotFoundError`` (an ``OSError``), which ``_call``'s except
    clauses (``TimeoutError``, ``ValueError``, ``httpx.HTTPError``,
    ``RuntimeError``) did not name, so it used to escape unclassified
    instead of the "did not answer in time" refusal every other
    stopped-server failure produces."""

    async def fake_chat(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.chat_reply(_FakeSvc(), "hello")

    assert excinfo.value.reason == "unhealthy"


def test_a_key_file_absent_when_the_request_starts_is_an_unhealthy_refusal_not_an_http_one(
    monkeypatch,
):
    """The 2026-09-20 audit (familiar-02): ``llama_client._headers`` raises a
    plain ``RuntimeError("llama-server has no key file -- it is not
    running")`` when ``server.key_path`` is already ``None`` at the start of
    a request -- a stopped server, the same story as the sibling race the
    test above covers (a raw ``FileNotFoundError`` when the key *file*
    vanishes mid-read). That sibling was mapped to ``"unhealthy"`` by the
    2026-09-18 familiar-01 fix, but ``_reason_for`` had no branch matching
    this RuntimeError's own sentence, so it fell through to the catch-all
    ``"http"`` bucket instead -- the same stopped-server failure classified
    two different ways depending on which side of the race a request landed
    on, and the pane picks its icon/action off ``reason``."""

    async def fake_chat(*args, **kwargs):
        raise RuntimeError("llama-server has no key file -- it is not running")

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.chat_reply(_FakeSvc(), "hello")

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


def test_clay_build_refuses_a_call_naming_a_tool_outside_the_frozen_cards_allowed_calls(
    monkeypatch,
):
    """The 2026-09-18 audit (familiar-05): ``contract.allowed_calls`` is
    parsed from the frozen card's own ``clay_batch`` schema (what the model
    was actually trained to see) but was never called at runtime, so
    ``clay_build`` returned a reply naming a tool outside that vocabulary
    unchecked. ``clay_undo`` is real (an MCP tool) but is not one of the
    names in ``cards/clay-1.txt``'s own ``clay_batch`` enum, so it stands in
    for a decoding fluke or a card/weights-pin mismatch here."""

    async def fake_chat(*args, **kwargs):
        return (
            '```json\n{"calls": [{"name": "clay_undo", "arguments": {}}]}\n```'
        )

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)
    fine_tuned = dataclasses.replace(
        models.FAMILIAR_MODELS["familiar_gguf"], card_shas=(contract.card_sha("clay"),)
    )
    monkeypatch.setitem(models.FAMILIAR_MODELS, "familiar_gguf", fine_tuned)
    assert "clay_undo" not in contract.allowed_calls("clay")

    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.clay_build(_FakeSvc(), "undo my last change", {"objects": []})

    assert excinfo.value.reason == "parse"
    assert "clay_undo" in excinfo.value.message


# ---------------------------------------------------------------------------
# T6: ``ask`` -- the router, and the Manual answer path. ``svc_familiar.ask``
# and ``svc_familiar._manual_index`` do not exist on the pre-T6 tree, so
# every test below fails with an ``AttributeError`` before its first
# assertion runs against the unmodified code.
# ---------------------------------------------------------------------------


def _fake_citation(n: int) -> retrieval.Citation:
    return retrieval.Citation(
        n=n,
        chapter="12-plotter",
        anchor="exporting",
        title_path="12 Plotter > Exporting",
        text="Export a map with File > Export > Tiled.",
    )


def test_a_manual_question_is_answered_from_retrieved_sections_with_their_citations(monkeypatch):
    """A ``manual``-routed question must retrieve first, then send the
    excerpts to the model, then keep only the citations the reply actually
    names -- proven end to end: the fake router sends ``skill: manual``, the
    fake index hands back one citation, the fake answer cites it, and the
    second request's own messages must carry that citation's text."""
    citation = _fake_citation(1)
    calls: list[dict] = []

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
        calls.append({"messages": messages, "skill": skill, "slot": slot})
        if skill == "router":
            return '{"skill": "manual"}'
        return "Use File > Export > Tiled [1]."

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)
    fake_index = SimpleNamespace(search=lambda prompt, **kw: [citation])
    monkeypatch.setattr(svc_familiar, "_manual_index", lambda: fake_index)

    answer = svc_familiar.ask(_FakeSvc(), "how do I export a map?", mode="home", history=())

    assert answer.skill == "manual"
    assert answer.text == "Use File > Export > Tiled [1]."
    assert answer.citations == (citation,)
    assert len(calls) == 2
    answer_request = calls[1]
    assert citation.title_path in answer_request["messages"][1]["content"]
    assert citation.text in answer_request["messages"][1]["content"]
    assert answer_request["slot"] == router.SKILL_SLOT


def test_a_manual_question_the_manual_does_not_cover_makes_no_answer_request(monkeypatch):
    """No citations retrieved must mean no second model call at all -- an
    honest "the Manual doesn't cover that" is cheaper and no less true than
    an ungrounded chat answer the base model was never given the Manual to
    write."""
    calls: list[str | None] = []

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
        calls.append(skill)
        return '{"skill": "manual"}'

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)
    monkeypatch.setattr(
        svc_familiar, "_manual_index", lambda: SimpleNamespace(search=lambda prompt, **kw: [])
    )

    answer = svc_familiar.ask(_FakeSvc(), "does Realmspinner support VR?", mode="home", history=())

    assert answer.text == "The Manual doesn't cover that."
    assert answer.citations == ()
    assert calls == ["router"], "a router reply with no citations must not make an answer request"


def test_the_router_runs_on_slot_zero_with_the_schema(monkeypatch):
    """The routing request itself must run on ``router.ROUTER_SLOT``, with
    ``response_format`` set to the router's own JSON schema and no card
    gate -- the router is prompt-engineered, not trained, so it must work on
    the base testing pin exactly like plain chat does."""
    captured: dict = {}

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
        if skill == "router":
            captured.update(
                slot=slot, sampling=sampling, response_format=response_format,
                expected_card_sha=expected_card_sha,
            )
            return '{"skill": "other"}'
        return "hi there"

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    svc_familiar.ask(_FakeSvc(), "hello", mode="home", history=())

    assert captured["slot"] == router.ROUTER_SLOT
    assert captured["sampling"] == contract.SAMPLING["router"]
    assert captured["response_format"] == {
        "type": "json_schema", "json_schema": {"schema": router.ROUTE_SCHEMA}
    }
    assert captured["expected_card_sha"] is None


def test_an_unbuilt_skill_falls_back_to_chat_and_still_reports_the_route(monkeypatch):
    """character/create/navigate all need their own list/options handed in
    by the caller (T7/T8) before ``ask`` will route to them at all -- with
    none offered (this call's own default, empty/``None``), a route to any
    of them falls back to plain chat exactly like ``other`` does, and the
    router's own decision must still land in ``Answer.skill`` even though
    ``chat_reply`` is what actually answered, so a caller can tell "the
    router picked X and nothing was offered for X" from "the router picked
    other"."""

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
        if skill == "router":
            return '{"skill": "character"}'
        return "sure, here's how to make a character"

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    answer = svc_familiar.ask(_FakeSvc(), "make me a goblin", mode="home", history=())

    assert answer.skill == "character"
    assert answer.text == "sure, here's how to make a character"
    assert answer.calls is None


def test_a_build_routed_in_clay_on_the_testing_pin_is_refused_by_the_card_gate(monkeypatch):
    """Routed to ``clay_build``, in Clay, with a scene -- ``ask`` must
    delegate to the real ``clay_build``, card gate included, not answer as
    chat just because it went through the router first."""
    assert models.FAMILIAR_MODELS["familiar_gguf"].card_shas == ()

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
        if skill == "router":
            return '{"skill": "clay_build"}'
        raise AssertionError("clay_build must refuse before any request on the testing pin")

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.ask(_FakeSvc(), "build a box", mode="clay", history=(), scene={"objects": []})

    assert excinfo.value.reason == "card"


def test_a_build_routed_outside_clay_is_answered_as_chat(monkeypatch):
    """The router naming ``clay_build`` from a mode other than Clay (or with
    no scene captured) must fall back to plain chat rather than reach
    ``clay_build`` with nothing to build against."""

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
        if skill == "router":
            return '{"skill": "clay_build"}'
        return "I can only build inside Clay."

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    answer = svc_familiar.ask(_FakeSvc(), "build a box", mode="home", history=())

    assert answer.skill == "clay_build"
    assert answer.text == "I can only build inside Clay."
    assert answer.calls is None


def test_a_refusal_during_routing_is_not_swallowed(monkeypatch):
    """A lease refusal raised while the *router itself* is answering must
    reach the caller as a ``FamiliarRefusal`` -- routing is one more request
    to the same door, not a special case that swallows what that door
    raises."""

    async def fake_chat(*args, **kwargs):
        raise RuntimeError(
            "Familiar cannot start while a GPU job holds the card -- "
            "it will restart on your next message."
        )

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.ask(_FakeSvc(), "hello", mode="home", history=())

    assert excinfo.value.reason == "lease"


# ---------------------------------------------------------------------------
# T8: ``navigate``/``create`` routes. ``ask`` takes no ``destinations``/
# ``asset_types`` keyword on the pre-T8 tree, so every test below fails with
# a ``TypeError`` before its first assertion runs against the unmodified
# code.
# ---------------------------------------------------------------------------


def test_a_navigate_route_asks_for_a_target_among_the_offered_destinations(monkeypatch):
    """The navigate request itself must run on the skill slot (never the
    router's own slot 0 -- that already answered "navigate"), with
    ``response_format`` constrained to exactly the offered destinations plus
    "none"."""
    from realmspinner.familiar import doors as doors_mod

    destinations = (
        doors_mod.Destination(key="go:clay", label="Go to Clay"),
        doors_mod.Destination(key="manual", label="Open the manual"),
    )
    captured: dict = {}

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
        if skill == "router":
            return '{"skill": "navigate"}'
        captured.update(slot=slot, sampling=sampling, response_format=response_format)
        return '{"target": "go:clay"}'

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    answer = svc_familiar.ask(
        _FakeSvc(), "open clay", mode="home", history=(), destinations=destinations
    )

    assert answer.skill == "navigate"
    assert answer.action == {"kind": "navigate", "target": "go:clay"}
    assert answer.text is None
    assert captured["slot"] == router.SKILL_SLOT
    assert captured["sampling"] == contract.SAMPLING["navigate"]
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"schema": doors_mod.navigate_schema(("go:clay", "manual"))},
    }


def test_a_navigate_route_with_no_usable_target_falls_back_to_chat(monkeypatch):
    """The model answering "none" (or garbage) must fall back to a plain
    chat reply -- the same "don't act, just answer" contract an unbuilt
    skill already keeps."""
    from realmspinner.familiar import doors as doors_mod

    destinations = (doors_mod.Destination(key="go:clay", label="Go to Clay"),)

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
        if skill == "router":
            return '{"skill": "navigate"}'
        if response_format is not None:
            # The navigate request itself -- the model found nothing to name.
            return '{"target": "none"}'
        return "I'm not sure where that is."

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    answer = svc_familiar.ask(
        _FakeSvc(), "take me somewhere", mode="home", history=(), destinations=destinations
    )

    assert answer.skill == "navigate"
    assert answer.action is None
    assert answer.text == "I'm not sure where that is."


def test_a_create_route_returns_a_draft_action_not_a_submission(monkeypatch):
    """A ``create`` route must hand back an action for the caller to draft,
    never anything that looks like a submission (no ``calls``, no queued
    job) -- drafting and submitting are two different presses."""
    asset_types = (("image", "Image"), ("3d_model", "3D Model"))

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
        if skill == "router":
            return '{"skill": "create"}'
        return '{"asset_type": "image", "prompt": "a lantern"}'

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    answer = svc_familiar.ask(
        _FakeSvc(), "make a lantern picture", mode="home", history=(), asset_types=asset_types
    )

    assert answer.skill == "create"
    assert answer.action == {"kind": "draft", "asset_type": "image", "prompt": "a lantern"}
    assert answer.calls is None
    assert answer.text is None


def test_a_refusal_while_choosing_a_destination_is_not_swallowed(monkeypatch):
    """A lease refusal raised while the *navigate* request itself is
    answering must reach the caller as a ``FamiliarRefusal``, the same
    contract the router's own request already keeps."""
    from realmspinner.familiar import doors as doors_mod

    destinations = (doors_mod.Destination(key="go:clay", label="Go to Clay"),)

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
        if skill == "router":
            return '{"skill": "navigate"}'
        raise RuntimeError(
            "Familiar cannot start while a GPU job holds the card -- "
            "it will restart on your next message."
        )

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.ask(
            _FakeSvc(), "open clay", mode="home", history=(), destinations=destinations
        )

    assert excinfo.value.reason == "lease"


# --- T7: the character route ------------------------------------------------


def _character_options() -> dict:
    return {
        "families": [
            {"key": "goblin", "label": "Goblin", "themes": ["swamp", "cave"]},
        ],
        "movements": ["idle", "walk", "run"],
        "directions": [1, 4, 8, 16],
        "size_range": (8, 256),
    }


def test_a_character_route_returns_a_plan_after_a_dry_run_and_mints_nothing(monkeypatch):
    """A ``character`` route must ask ``recipe_from_prompt`` -- a dry run --
    for the plan's own estimate, and must never reach ``create_character``:
    T7's whole point is a plan sits waiting for a press, nothing is minted
    by the chat turn that proposed it."""
    calls: dict = {}

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
        if skill == "router":
            return '{"skill": "character"}'
        return '{"family": "goblin", "theme": "swamp", "movements": ["walk"]}'

    def fake_recipe_from_prompt(svc, prompt, *, overrides=None):
        calls["overrides"] = overrides
        return {
            "recipe": {"family": "goblin"},
            "resolution": {},
            "ignored": [],
            "cells": 24,
            "estimate_minutes": 3.5,
        }

    def fake_create_character(*args, **kwargs):
        raise AssertionError("a character plan must not mint anything")

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)
    monkeypatch.setattr(svc_characters, "recipe_from_prompt", fake_recipe_from_prompt)
    monkeypatch.setattr(svc_characters, "create_character", fake_create_character)

    answer = svc_familiar.ask(
        _FakeSvc(),
        "make me a goblin in the swamp",
        mode="home",
        history=(),
        character_options=_character_options(),
    )

    assert answer.skill == "character"
    assert answer.text is None
    assert answer.action["kind"] == "character_plan"
    assert answer.action["plan"]["family"] == "goblin"
    assert calls["overrides"] == {
        "family": "goblin", "theme": "swamp", "animations": {"walk": None},
    }
    assert answer.action["overrides"] == calls["overrides"]
    assert answer.action["summary"]["species"] == "Goblin"
    assert answer.action["summary"]["cells"] == 24
    assert answer.action["summary"]["estimate_minutes"] == 3.5


def test_a_character_route_the_recipe_refuses_answers_with_the_refusal_sentence(monkeypatch):
    """A refusal out of ``recipe_from_prompt`` (an unbuilt theme, an
    unknown clip, ...) must answer as ordinary chat text -- it is a
    conversational answer the user can act on, not a
    :class:`FamiliarRefusal`: Familiar itself answered fine; it is the plan
    the recipe would not build."""

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
        if skill == "router":
            return '{"skill": "character"}'
        return '{"family": "goblin"}'

    def fake_recipe_from_prompt(svc, prompt, *, overrides=None):
        raise Invalid("Goblin has no 'gilded' look; it offers swamp, cave.", field="theme")

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)
    monkeypatch.setattr(svc_characters, "recipe_from_prompt", fake_recipe_from_prompt)

    answer = svc_familiar.ask(
        _FakeSvc(), "make a gilded goblin", mode="home", history=(),
        character_options=_character_options(),
    )

    assert answer.skill == "character"
    assert answer.action is None
    assert answer.text == "Goblin has no 'gilded' look; it offers swamp, cave."


def test_a_character_route_with_no_usable_plan_falls_back_to_chat(monkeypatch):
    """The model naming no species this build offers must fall back to a
    plain chat reply -- the same "don't act, just answer" contract
    ``_ask_navigate``/``_ask_create`` already keep."""

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
        if skill == "router":
            return '{"skill": "character"}'
        if response_format is not None:
            # The character request itself -- the model named no species.
            return '{"family": "none"}'
        return "Realmspinner builds goblins and knights -- which would you like?"

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    answer = svc_familiar.ask(
        _FakeSvc(), "make me something cool", mode="home", history=(),
        character_options=_character_options(),
    )

    assert answer.skill == "character"
    assert answer.action is None
    assert answer.text == "Realmspinner builds goblins and knights -- which would you like?"


def test_a_movement_or_theme_the_plan_itself_dropped_is_surfaced_in_the_character_plan_summary_not_only_ones_recipe_from_prompt_rejects(  # noqa: E501
    monkeypatch,
):
    """The 2026-09-18 audit (familiar-02):
    ``docs/manual/20-overview.md`` promises a word the plan could not act on
    "is named under the plan rather than silently dropped" -- but
    ``character_plan.parse_plan`` drops an unknown movement or an unoffered
    theme with no trace of its own, and ``_ask_character`` only ever
    forwarded ``recipe_from_prompt``'s own ``ignored`` list. Here the model
    names one movement ``recipe_from_prompt`` would happily accept (``walk``)
    and one ``parse_plan`` itself drops before the recipe ever sees it
    (``fly``, not in ``character_options["movements"]`` at all) -- the plan
    summary's own ``ignored`` list must still name ``fly``."""

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
        if skill == "router":
            return '{"skill": "character"}'
        return '{"family": "goblin", "movements": ["walk", "fly"]}'

    def fake_recipe_from_prompt(svc, prompt, *, overrides=None):
        # recipe_from_prompt never even sees "fly" -- parse_plan already
        # dropped it -- so its own ``ignored`` is empty here on purpose:
        # this proves the summary's "fly" entry can only have come from
        # parse_plan's own dropped list, not a pass-through of this one.
        assert overrides == {"family": "goblin", "animations": {"walk": None}}
        return {
            "recipe": {"family": "goblin"},
            "resolution": {},
            "ignored": [],
            "cells": 24,
            "estimate_minutes": 3.5,
        }

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)
    monkeypatch.setattr(svc_characters, "recipe_from_prompt", fake_recipe_from_prompt)

    answer = svc_familiar.ask(
        _FakeSvc(),
        "make me a goblin that can walk and fly",
        mode="home",
        history=(),
        character_options=_character_options(),
    )

    ignored = answer.action["summary"]["ignored"]
    assert any(item.get("text") == "fly" for item in ignored), (
        f"a word the plan itself dropped ('fly') never reached the summary: {ignored!r}"
    )


def test_creating_a_planned_character_re_runs_the_recipe_before_minting(monkeypatch):
    """``create_planned_character`` must re-resolve the recipe rather than
    trust the plan's own moment -- the world may have changed since (a
    download finished, a species that no longer resolves the same way)."""
    calls: list[str] = []

    def fake_recipe_from_prompt(svc, prompt, *, overrides=None):
        calls.append("recipe")
        assert overrides == {"family": "goblin"}
        return {"recipe": {"family": "goblin", "seed": 7}, "resolution": {"family": "goblin"}}

    def fake_create_character(svc, recipe, *, name=None, prompt="", resolution=None):
        calls.append("create")
        assert calls == ["recipe", "create"], "the recipe must be re-run before minting"
        assert recipe == {"family": "goblin", "seed": 7}
        assert resolution == {"family": "goblin"}
        assert name == "Grubnak"
        return {"id": "abc123", "rig": "def456", "kind": "character"}

    monkeypatch.setattr(svc_characters, "recipe_from_prompt", fake_recipe_from_prompt)
    monkeypatch.setattr(svc_characters, "create_character", fake_create_character)

    result = svc_familiar.create_planned_character(
        _FakeSvc(), "make me a goblin", {"family": "goblin"}, "Grubnak"
    )

    assert result == {"id": "abc123", "rig": "def456", "kind": "character"}
    assert calls == ["recipe", "create"]
