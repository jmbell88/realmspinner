"""A chat-completions client for Familiar's resident ``llama-server``.

Moved here from ``pipelines/llama_client.py`` in the 2026-09-17 restructure.
This module needs :mod:`.contract` (``SIZED_SKILLS``, ``SAMPLING``,
``TRAINED_WINDOW``, :func:`contract.output_budget`) for its reply-sizing
logic, and ``contract`` is layer 3 (``contract.derive_clay_card`` builds the
Clay training card from the *live* ``agent_clay`` tool surface, so the module
as a whole depends on layer 5) -- so promoting ``contract`` down to a kernel
so ``pipelines/llama_client.py`` could keep importing it was never available.
The edge had to close from the other side: this module comes down to sit
beside what it is a client of.

That leaves one property this package's own import pin
(``tests/familiar/test_familiar_imports.py``) exists to protect: everything
else under ``realmspinner/familiar/`` must stay importable by a training script
that never touches the network, which is why the pin bans ``httpx`` anywhere
in the package, even inside a function body. This module is the **one
recorded exception** to that ban (see ``HTTPX_ALLOWED`` in the test) --
its entire job is the real HTTP call to the resident child, exactly the
reason ``pipelines/trellis.py`` depends on httpx too, and there is no way to
make a network client that does not import a network library.

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
``response_format`` and ``json_schema`` are accepted by the pinned
``b11457`` build (a schema-constrained navigate reply was measured on it
2026-10-06, and came back with no reasoning at all).

**Why ``/tokenize`` on the concatenated message text, not ``/apply-template``
then ``/tokenize``.** llama.cpp's server has carried ``/tokenize`` since the
very first ``server`` example. ``/apply-template`` does exist in the pinned
``b11457`` build (the 2026-10-06 measurement used it), but it costs a second
round trip before every sized request, and rendering the template is the
server's job, not ours. Concatenating the raw message contents undercounts
the template's own role/turn tokens, and an undercount is the unsafe
direction (it over-sizes the reply), so :func:`template_margin` is added
back before the budget is taken. **That undercount is not constant:** on
Gemma 4 12B it grows by about five tokens per message (7 + 5 per message,
measured at 2 to 26 messages), which a flat margin would stop covering past
five messages -- a Clay build with two repairs is already eight.

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

from . import contract

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
#: :func:`contract.output_budget` sizes the reply, as a fixed part plus a
#: per-message part (see :func:`template_margin`). The chat template wraps
#: every turn in role and turn-boundary tokens that the raw text does not
#: carry. Uncorrected, the count comes out *low*, which is the unsafe
#: direction -- a budget sized off it overruns the 8,192-token slot by
#: exactly the template's overhead.
#:
#: **Measured on Gemma 4 12B QAT, b11457, 2026-10-06**
#: (dev/measurements/2026-10-06-familiar-gemma4-12b.md): the overhead of
#: ``/apply-template`` + ``/tokenize`` over the raw text is 7 + 5 per message
#: -- 17, 27, 37, 47, 57, 77, 97, 137 tokens at 2, 4, 6, 8, 10, 14, 18, 26
#: messages. The constants below are ceilings over that line, rounded up
#: rather than fitted: 16 over its 7, and 8 over its 5 per message, so two
#: messages come to the same 32 this constant used to be (Qwen3-VL-4B measured
#: 13 there). The previous flat 32 would have undercounted from the sixth
#: message on.
TEMPLATE_MARGIN_TOKENS = 16

#: The per-message half of :func:`template_margin`; see
#: :data:`TEMPLATE_MARGIN_TOKENS` for the measurement it is a ceiling over.
TEMPLATE_MESSAGE_TOKENS = 8


def template_margin(messages: Any) -> int:
    """The tokens to add back to a raw ``/tokenize`` count of *messages* for
    the chat template's own role/turn tokens: a fixed part plus a per-message
    part, because the template's overhead grows with the message count."""
    return TEMPLATE_MARGIN_TOKENS + TEMPLATE_MESSAGE_TOKENS * len(messages)

#: A conservative per-image token charge for :func:`contract.output_budget`'s
#: sizing, added once for every ``image_url`` content part in a request --
#: ``/tokenize`` only ever counts the raw *text* of a message (see this
#: module's own "why /tokenize on the concatenated message text" note above),
#: so an image's own cost has nowhere else to come from and would otherwise
#: silently count as zero, which is the unsafe direction (INVARIANTS: a token
#: count that feeds a budget must err high).
#:
#: **Measured against the real b11457 server on Gemma 4 12B QAT** (2026-10-06,
#: dev/measurements/2026-10-06-familiar-gemma4-12b.md): a 512x512 PNG -- the
#: exact size Clay's own ghost render ships (``ClayView.render_png``'s
#: ``three_quarter`` view) -- sent as one ``image_url`` part cost 123 prompt
#: tokens above the same request's own text-only prompt, and a flat red
#: square, uniform noise and a two-axis gradient all cost exactly 123, so the
#: cost is a property of the image's size, not its content. 150 is a ceiling
#: over that, the same "round up, never estimate down" shape
#: :data:`MIN_REPLY_TOKENS`'s own docstring already uses. (Qwen3-VL-4B, the
#: previous pin, tiled dynamically and measured 258, which is why this was 300.)
#:
#: **This constant is only a valid ceiling because every image is capped at
#: 512px on its longer side before it ever reaches this module (the
#: orchestrator's 2026-09-24 review, second finding), and measured, the cap
#: is load-bearing: an uncapped 1024x1024 noise image cost 443 tokens.**
#: ``studio.assistant.ui.normalize_attachment_image``/``VISION_MAX_SIDE`` is
#: the one door both a user-typed attach and Clay's own ghost render go
#: through. This module has no way to enforce that cap itself (it is not
#: where an image first arrives), so it is stated here as a precondition
#: rather than checked here.
IMAGE_TOKEN_COST = 150


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
    text = bytes(received).decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # The 2026-09-18 audit (familiar-07): this used to let
        # ``json.JSONDecodeError`` (a ``ValueError`` subclass) escape
        # unchanged. ``service.familiar._call``'s ``except ValueError``
        # clause exists for exactly one thing -- ``contract.output_budget``
        # refusing a prompt that leaves no room for a real reply -- and
        # classifies *every* ``ValueError`` as that same ``"too_large"``
        # reason. A malformed 200 body (a non-JSON page from a wedged or
        # misbehaving server) is a different failure with nothing to do with
        # the reply budget, so it is re-raised as a ``RuntimeError`` here,
        # the same exception type every other ``_post_capped`` failure above
        # already uses, and lets ``_call``'s ``except RuntimeError``/
        # ``_reason_for`` classify it (falling through to ``"http"``)
        # instead of the wrong bucket.
        raise RuntimeError(
            f"llama-server sent a non-JSON body: {text[:200]!r}"
        ) from exc


async def _tokenize(
    client: httpx.AsyncClient, headers: dict[str, str], text: str
) -> int:
    body = await _post_capped(client, "/tokenize", headers, {"content": text})
    tokens = body.get("tokens", [])
    return len(tokens)


def _message_text(content: Any) -> str:
    """*content* as plain text for ``/tokenize`` -- a bare string (every
    caller before vision existed) unchanged, or the joined ``text`` parts of
    an OpenAI-style content-part list (vision: ``[{"type": "text", ...},
    {"type": "image_url", ...}]``). ``/tokenize`` only ever counts text --
    an ``image_url`` part's own cost is :data:`IMAGE_TOKEN_COST`, added by
    :func:`_image_count`/:func:`chat` instead, never by handing the data URI
    itself to ``/tokenize``."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _image_count(content: Any) -> int:
    """How many ``image_url`` parts *content* carries -- 0 for a bare string
    (every caller before vision existed)."""
    if not isinstance(content, list):
        return 0
    return sum(
        1 for part in content if isinstance(part, dict) and part.get("type") == "image_url"
    )


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
            prompt_text = "\n\n".join(_message_text(m["content"]) for m in messages)
            n_tokens = await _tokenize(client, headers, prompt_text)
            n_images = sum(_image_count(m["content"]) for m in messages)
            # Propagates ValueError as-is: a prompt that leaves no room for a
            # real reply is a refusal the caller must show, not something
            # this client papers over by truncating. Every image content part
            # adds IMAGE_TOKEN_COST -- see its own docstring for why
            # ``/tokenize``'s text-only count would otherwise silently charge
            # zero for an image actually sent.
            max_tokens = contract.output_budget(
                skill, n_tokens + template_margin(messages) + n_images * IMAGE_TOKEN_COST
            )

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
