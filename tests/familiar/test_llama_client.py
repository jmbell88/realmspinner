"""T5's chat client: ``familiar/llama_client.py``.

Everything here runs against ``httpx.MockTransport`` and a fake server object
duck-typed to ``LlamaServer``'s own ``ensure_started``/``touch``/``key_path``/
``base_url`` -- no real subprocess, no real network, per the T5 brief's own
rule that a probe touching Familiar must never spawn a real ``llama-server``.
"""

from __future__ import annotations

import json

import httpx
import pytest

from warlock.familiar import contract, llama_client


class _FakeServer:
    """Duck-types ``pipelines.llama.LlamaServer`` for :func:`llama_client.chat`."""

    def __init__(self, key_path):
        self._key_path = key_path
        self.touches = 0
        self.ensure_started_calls: list[str | None] = []

    @property
    def base_url(self) -> str:
        return "http://127.0.0.1:9999"

    @property
    def key_path(self):
        return self._key_path

    async def ensure_started(self, *, expected_card_sha=None) -> None:
        self.ensure_started_calls.append(expected_card_sha)

    def touch(self) -> None:
        self.touches += 1


def _server(tmp_path, key: str = "s3cr3t"):
    key_path = tmp_path / "familiar-9999.key"
    key_path.write_text(key + "\n", encoding="utf-8")
    return _FakeServer(key_path)


def _chat_handler(*, reply: str = "hi", tokens: int = 10):
    """A ``httpx.MockTransport`` handler answering ``/tokenize`` and
    ``/v1/chat/completions``, recording every request it saw."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(tokens))})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200, json={"choices": [{"message": {"content": reply}}]}
            )
        return httpx.Response(404, text="not found")

    return handler, requests


async def test_every_request_touches_the_server_so_the_idle_clock_restarts(tmp_path):
    """``touch()`` must fire at least once before and once after the network
    round trip -- without it a live conversation's server is evicted mid-reply
    (``tests/familiar/test_familiar.py::
    test_touch_resets_the_idle_clock_so_a_live_conversation_is_not_evicted``
    is the door's own half of this claim; this is the client's)."""
    server = _server(tmp_path)
    handler, _ = _chat_handler()
    transport = httpx.MockTransport(handler)

    await llama_client.chat(
        server,
        [{"role": "user", "content": "hello"}],
        slot=1,
        sampling=contract.SAMPLING["chat"],
        transport=transport,
    )

    assert server.touches >= 2


async def test_the_api_key_is_read_from_the_key_file_not_argv(tmp_path):
    """The Authorization header must carry whatever is in the key file, and
    nothing about the fake server exposes the key any other way -- there is
    no ``_api_key`` attribute on the duck-typed server at all, so this can
    only pass if the client actually reads the file."""
    server = _server(tmp_path, key="the-real-key")
    handler, requests = _chat_handler()
    transport = httpx.MockTransport(handler)

    await llama_client.chat(
        server,
        [{"role": "user", "content": "hello"}],
        slot=1,
        sampling=contract.SAMPLING["chat"],
        transport=transport,
    )

    completion_requests = [r for r in requests if r.url.path == "/v1/chat/completions"]
    assert completion_requests
    assert completion_requests[0].headers["authorization"] == "Bearer the-real-key"


async def test_a_clay_request_sizes_max_tokens_from_tokenize_and_output_budget(tmp_path):
    """A skill with a frozen card (Clay) must size ``max_tokens`` off
    ``/tokenize``'s own count plus ``contract.output_budget`` -- not the flat
    ``SAMPLING["clay"]["max_tokens"]`` -- because a flat 4,096 overruns the
    8,192-token trained window once the prompt itself is a few thousand
    tokens (the largest recorded val prompt is ~4,931)."""
    server = _server(tmp_path)
    n_tokens = 5000
    handler, requests = _chat_handler(tokens=n_tokens)
    transport = httpx.MockTransport(handler)

    await llama_client.chat(
        server,
        [{"role": "system", "content": "card"}, {"role": "user", "content": "build a box"}],
        slot=1,
        sampling=contract.SAMPLING["clay"],
        skill="clay",
        expected_card_sha=contract.card_sha("clay"),
        transport=transport,
    )

    completion = next(r for r in requests if r.url.path == "/v1/chat/completions")
    body = json.loads(completion.content)
    # The raw-text count misses the chat template's own tokens, so the budget
    # must be taken against the count *plus* the margin -- sizing it off the
    # bare count over-sizes the reply and overruns the slot.
    expected = contract.output_budget("clay", n_tokens + llama_client.TEMPLATE_MARGIN_TOKENS)
    assert body["max_tokens"] == expected
    assert body["max_tokens"] < contract.output_budget("clay", n_tokens)
    assert expected != contract.SAMPLING["clay"]["max_tokens"]


async def test_a_prompt_too_large_for_the_trained_window_is_refused_not_truncated(tmp_path):
    """A prompt whose own token count leaves less than
    ``contract.MIN_REPLY_TOKENS`` of the trained window must raise
    ``ValueError`` straight out of ``contract.output_budget`` -- the client
    must never silently clamp ``max_tokens`` to whatever is left instead."""
    server = _server(tmp_path)
    too_many = contract.TRAINED_WINDOW  # leaves 0 tokens of reply room
    handler, _ = _chat_handler(tokens=too_many)
    transport = httpx.MockTransport(handler)

    with pytest.raises(ValueError, match="floor"):
        await llama_client.chat(
            server,
            [{"role": "system", "content": "card"}, {"role": "user", "content": "x"}],
            slot=1,
            sampling=contract.SAMPLING["clay"],
            skill="clay",
            transport=transport,
        )


async def test_every_request_turns_the_models_thinking_off(tmp_path):
    """Base Gemma 4, the previous pin, had a chat template that opened a
    reasoning channel unless told not to. On the first real-card run
    (2026-09-14) the router's 16-token budget went entirely into
    reasoning_content and the reply's content was '', so every
    schema-constrained call failed to decode. The guard stays on the current
    (Qwen3-VL-4B-Instruct) pin, harmlessly: every request, chat and skill
    alike, must ask the template for no thinking."""
    for sampling, skill in ((contract.SAMPLING["chat"], None), (contract.SAMPLING["router"], None)):
        server = _server(tmp_path)
        handler, requests = _chat_handler()
        await llama_client.chat(
            server,
            [{"role": "user", "content": "hello"}],
            slot=0,
            sampling=sampling,
            skill=skill,
            transport=httpx.MockTransport(handler),
        )
        completion = next(r for r in requests if r.url.path == "/v1/chat/completions")
        body = json.loads(completion.content)
        assert body.get("chat_template_kwargs") == {"enable_thinking": False}


async def test_requests_go_to_the_skill_slot(tmp_path):
    """``id_slot`` in the completion payload must be exactly the slot the
    caller passed -- Familiar's router (slot 0) and its routed skill (slot 1)
    must never collide in the same slot's request queue."""
    server = _server(tmp_path)
    handler, requests = _chat_handler()
    transport = httpx.MockTransport(handler)

    await llama_client.chat(
        server,
        [{"role": "user", "content": "hello"}],
        slot=1,
        sampling=contract.SAMPLING["chat"],
        transport=transport,
    )

    completion = next(r for r in requests if r.url.path == "/v1/chat/completions")
    body = json.loads(completion.content)
    assert body["id_slot"] == 1


async def test_a_response_format_is_forwarded_to_the_server(tmp_path):
    """T6's router constrains the model's output via ``response_format`` --
    the completion payload must carry it verbatim when a caller passes one,
    and omit the key entirely when it does not (a caller with no schema to
    enforce must get llama-server's ordinary free-text decoding, not an
    explicit ``null`` it has to special-case)."""
    server = _server(tmp_path)
    handler, requests = _chat_handler()
    transport = httpx.MockTransport(handler)
    schema = {"type": "object", "properties": {"skill": {"enum": ["a", "b"]}}}

    await llama_client.chat(
        server,
        [{"role": "user", "content": "hello"}],
        slot=0,
        sampling=contract.SAMPLING["router"],
        response_format={"type": "json_schema", "json_schema": {"schema": schema}},
        transport=transport,
    )

    completion = next(r for r in requests if r.url.path == "/v1/chat/completions")
    body = json.loads(completion.content)
    assert body["response_format"] == {"type": "json_schema", "json_schema": {"schema": schema}}

    requests.clear()
    await llama_client.chat(
        server,
        [{"role": "user", "content": "hello"}],
        slot=1,
        sampling=contract.SAMPLING["chat"],
        transport=transport,
    )
    completion = next(r for r in requests if r.url.path == "/v1/chat/completions")
    assert "response_format" not in json.loads(completion.content)


async def test_a_router_request_is_not_sized_as_a_skill_reply(tmp_path):
    """The router's card is frozen (``contract.CARDS["router"]``) but its
    reply is a fixed handful of tokens with no trained window to overrun --
    sizing must be keyed on ``contract.SIZED_SKILLS`` (today, just
    ``"clay"``), not on ``CARDS`` membership. A ``skill="router"`` request
    must use its flat ``SAMPLING["router"]["max_tokens"]`` verbatim and must
    never hit ``/tokenize`` at all -- paying for that round trip before every
    routing decision would slow down the one request Familiar most wants to
    feel instant."""
    server = _server(tmp_path)
    handler, requests = _chat_handler(tokens=5000)  # would drastically resize a sized skill
    transport = httpx.MockTransport(handler)
    assert "router" in contract.CARDS  # the router card really is frozen

    await llama_client.chat(
        server,
        [{"role": "system", "content": "router card"}, {"role": "user", "content": "hi"}],
        slot=0,
        sampling=contract.SAMPLING["router"],
        skill="router",
        transport=transport,
    )

    assert not any(r.url.path == "/tokenize" for r in requests)
    completion = next(r for r in requests if r.url.path == "/v1/chat/completions")
    body = json.loads(completion.content)
    assert body["max_tokens"] == contract.SAMPLING["router"]["max_tokens"]
