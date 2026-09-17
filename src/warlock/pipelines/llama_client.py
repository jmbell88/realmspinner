"""A chat-completions client for Familiar's resident ``llama-server``.

Lives here, next to ``llama.py``, rather than under ``studio/familiar/``,
because that package's own import pin
(``tests/familiar/test_familiar_imports.py``) bans ``httpx`` anywhere in it,
even inside a function body -- ``studio/familiar/`` must stay importable by a
training script that never touches the network. ``pipelines/`` already
depends on httpx (``trellis.py``, this module's own sibling), so this is
where a real HTTP call to the child belongs.

Two request shapes, chosen by whether *skill* is one of
:data:`contract.SIZED_SKILLS` -- **not** whether it has a frozen card
(``contract.CARDS``). T6 added a second frozen card, the router's, whose
reply is a fixed handful of tokens with no trained window to overrun; keying
sizing on ``CARDS`` membership would pay for a ``/tokenize`` round trip
before every routing decision for no reason.

* **Plain chat, or a card with no trained window to fit** (``skill=None`` or
  ``skill`` not in ``SIZED_SKILLS``): the caller already sized ``sampling``
  (e.g. ``contract.SAMPLING["chat"]``/``["router"]``), so :func:`chat` sends
  ``max_tokens`` unchanged.
* **A sized skill** (``skill="clay"``): run A was trained on an 8,192-token
  window per slot (``contract.TRAINED_WINDOW``), and a flat ``max_tokens``
  can overrun it once the prompt itself is long -- the largest recorded val
  prompt is ~4,931 tokens against a flat 4,096-token reply budget.
  :func:`chat` tokenizes the rendered prompt with llama-server's own
  ``/tokenize`` endpoint first, then asks :func:`contract.output_budget` for
  the reply budget that actually fits what is left of the window, letting its
  ``ValueError`` (too little room for a real reply) propagate rather than
  silently truncating -- a refusal a caller can show, not a reply cut off
  mid-JSON.

**Constrained decoding (T6).** *response_format* is forwarded to the server
verbatim -- the router's own request sends
``{"type": "json_schema", "json_schema": {"schema": router.ROUTE_SCHEMA}}``,
which is llama.cpp's OpenAI-compatible ``response_format`` shape. Both
``response_format`` and ``json_schema`` are present in the installed
``b10948`` build's own ``llama-server-impl.dll`` (checked 2026-09-14,
``grep -a -c``: 2 and 3 occurrences respectively; ``/apply-template`` is not,
see below), so this build accepts it.

**Why ``/tokenize`` on the concatenated message text, not ``/apply-template``
then ``/tokenize``.** llama.cpp's server has carried ``/tokenize`` since the
very first ``server`` example; ``/apply-template`` is newer and this
programme has no vendored llama.cpp checkout in-tree to confirm it ships in
the exact ``b10948`` build ``models.py`` pins (``docs/MODELS.md``). Rendering
the chat template ourselves would risk disagreeing with whatever Jinja
template the GGUF actually carries (``--jinja`` is passed at spawn, per
``llama.py``'s own argv). Concatenating the raw message contents
undercounts the template's own role/turn tokens by a small, roughly constant
amount, and an undercount is the unsafe direction (it over-sizes the reply),
so :data:`TEMPLATE_MARGIN_TOKENS` is added back before the budget is taken.
The installed build was checked and has no ``/apply-template`` at all.

**Every request touches the server.** ``server.touch()`` is called both
before and after the network round trip: before, so a slow tokenize/generate
pair does not itself look idle to the eviction sweep while it is still
running; after, so the reply landing resets the clock for the *next* one.
``queue.Worker._maybe_evict_idle`` only ever reads ``last_used`` -- nothing
else writes it once the server is up (see ``LlamaServer.touch``'s own
docstring) -- so a client that forgot this would have its server evicted out
from under a long conversation.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..familiar import contract

#: How long a chat round trip may take before this gives up. Generous:
#: Familiar runs on whatever GPU is in the machine, `-ngl 999` with no
#: batching guarantee, and a slow reply is still better than a client that
#: gives up on a real answer -- there is no user-facing "retry" for a chat
#: message today, so a low timeout would just be a harder failure.
CHAT_TIMEOUT = 180.0

#: Byte ceiling on any one response body (``/tokenize`` or the completion
#: itself) -- the same defence-in-depth ``pipelines/trellis.py``'s
#: ``generate`` applies to its own error bodies (MDL-13): a wedged or
#: compromised local server must not be able to exhaust the host by handing
#: back an unbounded body just because the peer is loopback.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

#: Tokens added to ``/tokenize``'s count of the raw message text before
#: :func:`contract.output_budget` sizes the reply. The chat template wraps
#: every turn in role and turn-boundary tokens that the raw text does not
#: carry, and the installed ``b10948`` build has no ``/apply-template`` to
#: count them (checked 2026-09-14: the string is absent from
#: ``llama-server-impl.dll``). Uncorrected, the count comes out *low*, which
#: is the unsafe direction -- a budget sized off it overruns the 8,192-token
#: slot by exactly the template's overhead. 32 is a ceiling, not the
#: measurement: on Qwen3-VL-4B's ChatML template the GPU lane measured the
#: real overhead of a two-turn Clay request at 13 tokens
#: (dev/measurements/2026-09-16-familiar-qwen-vram.md), and it sat inside
#: 32 on the previous Gemma 4 E2B pin too. Shrinking it would buy 19 tokens
#: of an 8,192-token slot and lose the room for a template that spends more.
TEMPLATE_MARGIN_TOKENS = 32


def _headers(server: Any) -> dict[str, str]:
    """``Authorization: Bearer <key>``, read from the key *file* -- never
    from an ``_api_key`` attribute or (worse) argv. See ``LlamaServer.
    key_path``'s own docstring for why the file is the one thing a client
    outside ``pipelines/llama.py`` is allowed to touch."""
    key_path = server.key_path
    if key_path is None:
        raise RuntimeError("llama-server has no key file -- it is not running")
    key = key_path.read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


async def _post_capped(
    client: httpx.AsyncClient, path: str, headers: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    """POST *payload* to *path*, streamed, aborting past
    :data:`MAX_RESPONSE_BYTES` instead of buffering the whole reply first.

    The 2026-09-15 audit (pipelines-03): this module's own docstring already
    claimed parity with ``trellis.generate``, which streams for exactly this
    reason (MDL-13, a wedged or compromised local server exhausting the host
    with an unbounded body). But both call sites here used ``client.post``,
    which reads the entire response into ``response.content`` before either
    of them ever looked at its length -- the ceiling was only ever checked
    after the damage it exists to prevent had already happened.
    """
    received = bytearray()
    async with client.stream("POST", path, json=payload, headers=headers) as r:
        if r.status_code != 200:
            # Bounded exactly like the success body below -- an error page
            # from a wedged server is not exempt from the same risk.
            error_bytes = bytearray()
            async for chunk in r.aiter_bytes():
                error_bytes.extend(chunk)
                if len(error_bytes) >= 500:
                    break
            text = bytes(error_bytes).decode("utf-8", "replace")
            raise RuntimeError(f"llama-server {r.status_code}: {text[:500]}")
        async for chunk in r.aiter_bytes():
            received.extend(chunk)
            if len(received) > MAX_RESPONSE_BYTES:
                raise RuntimeError(
                    f"llama-server's reply was {len(received)} bytes, over the "
                    f"{MAX_RESPONSE_BYTES}-byte ceiling, before it finished "
                    "arriving -- refusing to use it"
                )
    return json.loads(bytes(received).decode("utf-8"))


async def _tokenize(
    client: httpx.AsyncClient, headers: dict[str, str], text: str
) -> int:
    body = await _post_capped(client, "/tokenize", headers, {"content": text})
    tokens = body.get("tokens", [])
    return len(tokens)


async def chat(
    server: Any,
    messages: list[dict[str, str]],
    *,
    slot: int,
    sampling: dict[str, float | int],
    skill: str | None = None,
    expected_card_sha: str | None = None,
    response_format: dict[str, Any] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """Send *messages* to *server*'s chat-completions endpoint on *slot*,
    return the reply text.

    *sampling* is the caller's own ``contract.SAMPLING[...]`` row; its
    ``max_tokens`` is used verbatim except for a skill in
    :data:`contract.SIZED_SKILLS`, re-sized by :func:`contract.output_budget`
    off a real ``/tokenize`` count (T6: keyed on ``SIZED_SKILLS``, not
    ``contract.CARDS`` membership -- the router card is frozen too, but its
    reply is a fixed handful of tokens with no trained window to overrun, so
    paying for a ``/tokenize`` round trip before every routing decision
    would slow down the one request that most wants to feel instant).
    *response_format* (T6) is forwarded to the server verbatim when given --
    the router's own constrained-decoding schema (see ``studio.familiar.
    router.ROUTE_SCHEMA``); every other caller omits it and gets llama-
    server's default free-text decoding. *server* is anything shaped like
    ``pipelines.llama.LlamaServer`` (an ``ensure_started``/``touch``/
    ``key_path``/``base_url``, duck-typed so a test can hand in a lighter
    fake). *transport* is for tests (``httpx.MockTransport``); production
    callers never pass it.
    """
    await server.ensure_started(expected_card_sha=expected_card_sha)
    server.touch()
    headers = _headers(server)
    async with httpx.AsyncClient(
        base_url=server.base_url, timeout=CHAT_TIMEOUT, transport=transport
    ) as client:
        max_tokens = sampling["max_tokens"]
        if skill is not None and skill in contract.SIZED_SKILLS:
            prompt_text = "\n\n".join(m["content"] for m in messages)
            n_tokens = await _tokenize(client, headers, prompt_text)
            # Propagates ValueError as-is: a prompt that leaves no room for a
            # real reply is a refusal the caller must show, not something
            # this client papers over by truncating.
            max_tokens = contract.output_budget(skill, n_tokens + TEMPLATE_MARGIN_TOKENS)

        payload = {
            "model": "familiar",
            "messages": messages,
            "id_slot": slot,
            "temperature": sampling["temperature"],
            "top_k": sampling["top_k"],
            "top_p": sampling["top_p"],
            "max_tokens": max_tokens,
            "stream": False,
            # Thinking off, asked for on every request too. Added for the
            # previous pin, Gemma 4: its chat template (the server runs
            # --jinja) opened a reasoning channel by default, and the first
            # real-card run (2026-09-14) spent the router's whole budget on
            # "Thinking Process: ..." in reasoning_content and returned
            # content=''. This field alone did not hold on every prompt, so
            # the real switch is llama.py's --reasoning off
            # --reasoning-budget 0; this stays as the request's own statement
            # of the same intent, for a server started some other way, and is
            # harmless on Qwen3-VL-4B-Instruct, which doesn't open a
            # reasoning channel by default.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if response_format is not None:
            payload["response_format"] = response_format
        body = await _post_capped(client, "/v1/chat/completions", headers, payload)
        server.touch()
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"llama-server reply had no message content: {body!r}") from exc
