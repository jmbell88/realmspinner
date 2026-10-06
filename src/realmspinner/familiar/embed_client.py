"""The HTTP side of Familiar's dense retrieval: texts in, unit vectors out.

Talks to the second ``llama-server`` child (EmbeddingGemma 2, ``--embeddings
--pooling mean``, ``pipelines/llama.py``) over its OpenAI-compatible
``/v1/embeddings``. Conventions follow :mod:`.llama_client` exactly: *server* is
duck-typed (``ensure_started``/``touch``/``key_path``/``base_url``), the API key
is read from the key *file*, the response body is size-capped while it streams,
and *transport* exists so a test can hand in ``httpx.MockTransport``. Like
``llama_client`` this is a recorded exception to the package's httpx ban
(``HTTPX_ALLOWED`` in ``tests/familiar/test_familiar_imports.py``).

Differences, each deliberate:

* **``trust_env=False``.** The embedder is always loopback; a configured
  ``HTTP(S)_PROXY`` without a ``NO_PROXY`` entry would hand every Manual chunk
  and Library prompt to the proxy (the TRELLIS-clients invariant).
* **The model card's prompts are added here, not by callers.** EmbeddingGemma
  is trained with task prefixes and retrieval quality drops when one is
  forgotten, so :func:`embed` takes *raw* text and a *role* and applies
  ``task: search result | query: {text}`` or ``title: {title} | text: {text}``
  itself. A caller cannot get this wrong.
* **Errors never carry text.** What is being embedded is the user's own prompt
  or Library row; no exception message, and no log line, contains it -- nor the
  server's error body, which could echo it. A refusal reports a status code
  only.

Every failure that means "no dense vectors right now" is an
:class:`EmbedUnavailable`, so the one ``except`` a caller needs is the one that
falls back to BM25 (or LIKE).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np

from .embed_index import NATIVE_DIM

#: The ``model`` field sent with every request. llama-server serves one model
#: and ignores the name, but the OpenAI schema requires one.
MODEL_NAME = "embeddinggemma"

#: Texts per request. Measured 2026-10-06 (dev/measurements/2026-10-06-familiar-
#: gemma4-12b.md): 910 Manual chunks in 165 s on the CPU in batches of 16.
BATCH_SIZE = 16

#: Per-request timeout. A batch of the longest chunks is a few seconds on the
#: CPU; this is generous without being unbounded.
EMBED_TIMEOUT = 120.0

#: Same defence-in-depth ceiling as :data:`llama_client.MAX_RESPONSE_BYTES`.
#: 16 vectors of 768 floats as JSON is a few hundred KiB.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

ROLES = ("query", "document")


class EmbedUnavailable(RuntimeError):
    """The embedder cannot give vectors right now -- not started, refused,
    unreachable, or it answered with something unusable. Callers fall back to
    BM25/LIKE. The message never contains the text that was being embedded."""


class EmbedBadResponse(EmbedUnavailable):
    """The server answered 200 but the body was the wrong count, the wrong
    width, malformed or not finite. A subclass so one ``except
    EmbedUnavailable`` covers it, kept separate so a test (or a doctor line)
    can tell "down" from "up but wrong"."""


def format_query(text: str) -> str:
    """The model card's retrieval-query prompt for *text*."""
    return f"task: search result | query: {text}"


def format_document(item: str | tuple[str, str]) -> str:
    """The model card's document prompt: *item* is ``(title, text)`` or a plain
    string (title ``none``, the card's own placeholder for an untitled text).
    A blank title is also ``none``."""
    if isinstance(item, str):
        title, text = "", item
    else:
        title, text = item
    return f"title: {title.strip() or 'none'} | text: {text}"


def _headers(server: Any) -> dict[str, str]:
    """``Authorization: Bearer <key>`` from the key *file* (see
    ``llama_client._headers``)."""
    key_path = server.key_path
    if key_path is None:
        raise EmbedUnavailable("the embedding server has no key file -- it is not running")
    key = key_path.read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


async def _post_capped(
    client: httpx.AsyncClient, headers: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    """POST *payload* to ``/v1/embeddings``, streamed and aborted past
    :data:`MAX_RESPONSE_BYTES`. A non-200 raises with its status code only --
    the body is never read, because it could echo the input."""
    received = bytearray()
    async with client.stream("POST", "/v1/embeddings", json=payload, headers=headers) as r:
        if r.status_code != 200:
            raise EmbedUnavailable(f"the embedding server refused the request ({r.status_code})")
        async for chunk in r.aiter_bytes():
            received.extend(chunk)
            if len(received) > MAX_RESPONSE_BYTES:
                raise EmbedBadResponse(
                    f"the embedding server's reply passed {MAX_RESPONSE_BYTES} bytes "
                    "before it finished arriving -- refusing to use it"
                )
    try:
        body = json.loads(bytes(received).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EmbedBadResponse("the embedding server sent a non-JSON body") from exc
    if not isinstance(body, dict):
        raise EmbedBadResponse("the embedding server sent a JSON body that is not an object")
    return body


def _vectors(body: dict[str, Any], expected: int) -> np.ndarray:
    """The ``(expected, 768)`` float32 matrix in *body*, ordered by each item's
    ``index`` (the OpenAI schema does not promise array order), rows
    renormalised to unit length. Anything else is an :class:`EmbedBadResponse`."""
    items = body.get("data")
    if not isinstance(items, list) or len(items) != expected:
        got = len(items) if isinstance(items, list) else "no"
        raise EmbedBadResponse(f"asked for {expected} embeddings, the server returned {got}")
    slots: list[Any] = [None] * expected
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise EmbedBadResponse("an embedding item is not an object")
        index = item.get("index", position)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < expected:
            raise EmbedBadResponse(f"an embedding item has an out-of-range index {index!r}")
        if slots[index] is not None:
            raise EmbedBadResponse(f"two embedding items claim index {index}")
        slots[index] = item.get("embedding")
    rows = np.empty((expected, NATIVE_DIM), dtype=np.float32)
    for index, vector in enumerate(slots):
        try:
            row = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise EmbedBadResponse("an embedding is not a list of numbers") from exc
        if row.shape != (NATIVE_DIM,):
            raise EmbedBadResponse(
                f"expected {NATIVE_DIM}-d embeddings, the server returned {tuple(row.shape)}"
            )
        if not np.isfinite(row).all():
            raise EmbedBadResponse("an embedding contains a non-finite value")
        norm = float(np.linalg.norm(row))
        if norm == 0.0:
            raise EmbedBadResponse("an embedding is all zeros")
        rows[index] = row / norm
    return rows


async def embed(
    server: Any,
    texts: Sequence[str | tuple[str, str]],
    *,
    role: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> np.ndarray:
    """Embed *texts* and return an ``(n, 768)`` float32 array, rows unit-norm.

    *role* is ``"query"`` (each text a plain string) or ``"document"`` (each a
    plain string, or a ``(title, text)`` pair); the model card's prompt for
    that role is added here (:func:`format_query`/:func:`format_document`).
    Truncate to a Matryoshka width afterwards with ``embed_index.truncate``.
    An empty *texts* returns a ``(0, 768)`` array without starting the server.

    Requests go out in batches of :data:`BATCH_SIZE`. ``server.touch()`` fires
    around every batch so a 165 s Manual index does not look idle to the
    eviction sweep (see ``llama_client``'s own note). Anything that means "no
    vectors" -- spawn failure, refusal, a dropped connection, a malformed or
    wrong-shaped reply -- raises :class:`EmbedUnavailable`. *server* is
    anything shaped like ``pipelines.llama.LlamaServer``; *transport* is for
    tests.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    prompts: list[str] = []
    for item in texts:
        if role == "query":
            if not isinstance(item, str):
                raise TypeError("a query is a plain string; (title, text) is for documents")
            prompts.append(format_query(item))
        else:
            prompts.append(format_document(item))
    if not prompts:
        return np.empty((0, NATIVE_DIM), dtype=np.float32)

    try:
        await server.ensure_started()
        server.touch()
        headers = _headers(server)
        out = np.empty((len(prompts), NATIVE_DIM), dtype=np.float32)
        async with httpx.AsyncClient(
            base_url=server.base_url,
            timeout=EMBED_TIMEOUT,
            transport=transport,
            trust_env=False,
        ) as client:
            for start in range(0, len(prompts), BATCH_SIZE):
                batch = prompts[start : start + BATCH_SIZE]
                body = await _post_capped(client, headers, {"input": batch, "model": MODEL_NAME})
                server.touch()
                out[start : start + len(batch)] = _vectors(body, len(batch))
        return out
    except EmbedUnavailable:
        raise
    except httpx.HTTPError as exc:
        # The class name only: an httpx message can quote the request.
        raise EmbedUnavailable(
            f"the embedding server could not be reached ({type(exc).__name__})"
        ) from None
    except (RuntimeError, OSError) as exc:
        # ensure_started failing to spawn or find the weights, or a missing key
        # file. Its message is kept (doctor and the log want "weights not
        # installed"): it is raised before any text is sent and never sees one.
        raise EmbedUnavailable(
            f"the embedding server could not be started ({type(exc).__name__}: {exc})"
        ) from exc
