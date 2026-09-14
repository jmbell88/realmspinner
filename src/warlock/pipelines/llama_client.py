"""A chat-completions client for Familiar's resident ``llama-server``.

Lives here, next to ``llama.py``, rather than under ``studio/familiar/``,
because that package's own import pin
(``tests/familiar/test_familiar_imports.py``) bans ``httpx`` anywhere in it,
even inside a function body -- ``studio/familiar/`` must stay importable by a
training script that never touches the network. ``pipelines/`` already
depends on httpx (``trellis.py``, this module's own sibling), so this is
where a real HTTP call to the child belongs.

Two request shapes, chosen by whether *skill* names a frozen card
(``contract.CARDS``):

* **Plain chat** (``skill=None``): the caller already sized ``sampling``
  (``contract.SAMPLING["chat"]``), so :func:`chat` sends it unchanged.
* **A skill with a card** (``skill="clay"``): run A was trained on an
  8,192-token window per slot (``contract.TRAINED_WINDOW``), and a flat
  ``max_tokens`` can overrun it once the prompt itself is long -- the largest
  recorded val prompt is ~4,931 tokens against a flat 4,096-token reply
  budget. :func:`chat` tokenizes the rendered prompt with llama-server's own
  ``/tokenize`` endpoint first, then asks :func:`contract.output_budget` for
  the reply budget that actually fits what is left of the window, letting its
  ``ValueError`` (too little room for a real reply) propagate rather than
  silently truncating -- a refusal a caller can show, not a reply cut off
  mid-JSON.

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

from typing import Any

import httpx

from ..studio.familiar import contract

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
#: slot by exactly the template's overhead. Gemma's template spends a
#: handful of tokens per turn plus BOS and the generation prompt; a Clay
#: request has two turns, so 32 is a generous ceiling, not a measurement.
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


def _check_size(response: httpx.Response) -> None:
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise RuntimeError(
            f"llama-server's reply was {len(response.content)} bytes, over the "
            f"{MAX_RESPONSE_BYTES}-byte ceiling -- refusing to use it"
        )


async def _tokenize(
    client: httpx.AsyncClient, headers: dict[str, str], text: str
) -> int:
    r = await client.post("/tokenize", json={"content": text}, headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f"llama-server {r.status_code}: {r.text[:500]}")
    _check_size(r)
    tokens = r.json().get("tokens", [])
    return len(tokens)


async def chat(
    server: Any,
    messages: list[dict[str, str]],
    *,
    slot: int,
    sampling: dict[str, float | int],
    skill: str | None = None,
    expected_card_sha: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """Send *messages* to *server*'s chat-completions endpoint on *slot*,
    return the reply text.

    *sampling* is the caller's own ``contract.SAMPLING[...]`` row; its
    ``max_tokens`` is used verbatim for plain chat and re-sized by
    :func:`contract.output_budget` when *skill* names a frozen card. *server*
    is anything shaped like ``pipelines.llama.LlamaServer`` (an
    ``ensure_started``/``touch``/``key_path``/``base_url``, duck-typed so a
    test can hand in a lighter fake). *transport* is for tests
    (``httpx.MockTransport``); production callers never pass it.
    """
    await server.ensure_started(expected_card_sha=expected_card_sha)
    server.touch()
    headers = _headers(server)
    async with httpx.AsyncClient(
        base_url=server.base_url, timeout=CHAT_TIMEOUT, transport=transport
    ) as client:
        max_tokens = sampling["max_tokens"]
        if skill is not None and skill in contract.CARDS:
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
        }
        r = await client.post("/v1/chat/completions", json=payload, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"llama-server {r.status_code}: {r.text[:500]}")
        _check_size(r)
        body = r.json()
        server.touch()
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"llama-server reply had no message content: {body!r}") from exc
