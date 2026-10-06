"""The Library's meaning index: what a finished job is *about*, as one vector.

Library search has always been a substring match over name, prompt, tags and id.
With the optional retrieval row installed (EmbeddingGemma 2, ``models.
FAMILIAR_MODELS["familiar_embed"]``, the CPU-only second ``llama-server`` child)
a search also finds a job by what it was about -- "weapon for a swordsman" finds
the job whose prompt says "rusty iron longsword" -- and fuses that ranking with
the substring one (``embed_index.rrf``). **With the row absent nothing here
runs**: no table write, no child, no toast, and ``search`` is exactly
``JobStore.search_ids``.

This is the service layer's because it reaches the worker's embedder child
(``svc.worker.familiar_embed``) and the config, which ``db.py`` (layer 0) and
``familiar/`` may not; ``db.py`` stores and ranks the vectors it is handed and
never touches the network. The scheduling half -- when to index, in what order,
how to back off -- is ``studio/embed_indexer.py``, which calls the blocking
functions below from ``TaskRunner`` tasks and never from the frame thread.

**What is embedded** (:func:`job_text`): the job's name as the document title,
its prompt, and a short facet line -- what kind of thing it is in words a person
would search for ("3D model", "sprite sheet", "music track"), plus the handful of
``params`` that name a subject or a look (``asset_type``, ``style_lora``,
``family``, ``theme``, ``appearance``, the base model's name...). Never a seed, a
path, a size or a count: nobody searches by those, and they would only pull
unrelated rows together. Never tags either -- ``tag:`` is already an exact field
search, and a tag edit would otherwise re-embed the row.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import hashlib
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import fetch, models
from ..familiar import embed_client, embed_index
from ..pipelines import llama

log = logging.getLogger(__name__)

#: The embedder's weights, as every stored vector's provenance. A different pin
#: is a different space: its rows are *stale*, replaced rather than ranked.
MODEL_SHA = models.FAMILIAR_EMBED_GGUF_SHA256

#: The Matryoshka width the Library keeps (``embed_index.LIBRARY_DIM``): one
#: kilobyte a row, and a row's text is a sentence or two.
DIM = embed_index.LIBRARY_DIM

#: Bumped whenever :func:`job_text` changes what it says about a job, so every
#: stored vector built from the old wording is noticed by the verify pass and
#: re-embedded instead of being ranked against queries it was never meant for.
TEXT_VERSION = 1

#: A search shorter than this many non-space characters is never embedded: one
#: letter has no meaning to look for, and the substring search already answers it.
MIN_QUERY_CHARS = 2

#: What a row must clear to be added to a search by its meaning, and how many may
#: be. *Measured* 2026-10-06 with the staged EmbeddingGemma 2 Q8_0 at 256-d on 39
#: Library-shaped rows (28 queries a row answers, 18 that mean nothing here): an
#: absolute cosine cannot separate them -- the right row scored 0.64-0.78, the
#: best row for nonsense 0.51-0.73 -- but measured against the query's own median
#: and spread over the library the right row stood 2.4-9.7 robust sigma out (26
#: of 28 at 3.0 or more) and nonsense mostly below 3 (3 of 15 still produced a
#: row, never more than 4). See ``JobStore.semantic_ranked`` for why the spread
#: is a median/MAD one. A z-floor passes a sliver of any library for any query,
#: so :data:`SEMANTIC_CAP` is what bounds a large one: at most 20 rows are added,
#: best first, and the rest are dropped. A library of fewer than
#: :data:`SEMANTIC_MIN_ROWS` vectors has no baseline and is not searched by
#: meaning at all.
SEMANTIC_Z_FLOOR = 3.0
SEMANTIC_CAP = 20
SEMANTIC_MIN_ROWS = 20

#: Rows embedded per background task: two ``embed_client`` batches. A task is one
#: pool slot for as long as it runs, and a backfill must never hold one of the
#: pool's four for minutes.
BATCH = 32

#: What one embed call may wait on the loop: a cold start plus two batches.
CALL_TIMEOUT = llama.STARTUP_TIMEOUT + 2 * embed_client.EMBED_TIMEOUT + 30.0

#: A search's own query vector. Short: a search that cannot get one falls back
#: to the substring result it already has, so a hung embedder must cost seconds.
QUERY_TIMEOUT = 20.0

#: Free text longer than this is cut before it is embedded. A typed query is a
#: few words; a pasted paragraph is not worth a slower request.
MAX_QUERY_CHARS = 300

#: The embedder's two rows: the shared ``llama-server`` runtime and the weights.
_ROWS = ("familiar_runtime", "familiar_embed")

# --------------------------------------------------------------------------
# What a job says about itself
# --------------------------------------------------------------------------

#: How long a prompt / a facet value is kept. The embedder's context is 8192
#: tokens, so this is not a limit of the model's: it is what keeps one pasted
#: essay from outweighing the name and the facets in a row's vector.
MAX_PROMPT_CHARS = 600
MAX_NAME_CHARS = 120
MAX_FACET_CHARS = 80

#: ``job["kind"]`` values that are not "a model made from a prompt", in the words
#: a person would use for the result. Everything else is a 3D model, a reference
#: image or a seamless tile, by ``stage`` (see ``state.card_kind``, which this
#: deliberately does not import: it is a studio module and this is the service).
_KIND_WORDS = {
    "rig": "rigged character",
    "sheet": "sprite sheet",
    "charsheet": "character sprite sheet",
    "music": "music track",
    "separate": "separated music stems",
    "sprite_synthesis": "pixel sprite",
    "tile_sheet": "tile sheet",
    "remesh": "remeshed 3D model",
    "retexture": "retextured 3D model",
}
_STAGE_WORDS = {"reference": "reference image", "tile": "seamless tile"}

#: ``params`` keys that name a subject or a look, in the order they are said, and
#: the word each is introduced by. Curated: a key not listed here is never
#: embedded, so a new seed-like field cannot leak into every vector.
_FACETS: tuple[tuple[str, str], ...] = (
    ("generation_type", "type"),
    ("asset_type", "asset"),
    ("asset_intent", "intent"),
    ("family", "family"),
    ("theme", "theme"),
    ("appearance", "look"),
    ("style_lora", "style"),
    ("style", "style"),
    ("base_model", "model"),
)

_SPACE = re.compile(r"\s+")


def _clean(value: Any, limit: int) -> str:
    """*value* as one line of text no longer than *limit*; ``""`` for anything
    that is not a string. Control characters and runs of whitespace become one
    space, so the same words always hash the same."""
    if not isinstance(value, str):
        return ""
    return _SPACE.sub(" ", value).strip()[:limit].strip()


def _humanise(value: str) -> str:
    """``juggernaut-xl_v9`` -> ``juggernaut xl v9``: a key reads as words."""
    return _SPACE.sub(" ", value.replace("_", " ").replace("-", " ")).strip()


def kind_word(kind: Any, stage: Any) -> str:
    """What this row is, in the words a person would search for it by."""
    if isinstance(kind, str) and kind in _KIND_WORDS:
        return _KIND_WORDS[kind]
    if isinstance(stage, str) and stage in _STAGE_WORDS:
        return _STAGE_WORDS[stage]
    return "3D model"


def job_text(job: dict[str, Any]) -> tuple[str, str] | None:
    """``(text, text_hash)`` for a job, or ``None`` when it has nothing to say.

    *text* is ``"{title}\\n{body}"``: the name as the document title (the model
    card's ``title: ... | text: ...`` prompt wants one; a blank title becomes the
    card's own ``none``), then the prompt, then one ``kind: ...; style: ...``
    facet line. ``None`` when both name and prompt are blank -- a row with only
    facets would be a vector of the word "3D model", which would sit near every
    other such row.

    Deterministic (same row, same text, same hash, on any machine), which is the
    whole point of ``text_hash``: it is what a rename, an edited facet set or a
    bumped :data:`TEXT_VERSION` changes, and so what makes the verify pass
    re-embed exactly the rows whose meaning moved.
    """
    name = _clean(job.get("name"), MAX_NAME_CHARS)
    prompt = _clean(job.get("prompt"), MAX_PROMPT_CHARS)
    if not name and not prompt:
        return None
    params = job.get("params")
    if not isinstance(params, dict):
        params = {}
    facets = [f"kind: {kind_word(job.get('kind'), job.get('stage'))}"]
    for key, label in _FACETS:
        value = _humanise(_clean(params.get(key), MAX_FACET_CHARS))
        if value:
            facets.append(f"{label}: {value}")
    body = [] if prompt == name else [prompt]
    body.append("; ".join(facets))
    text = name + "\n" + "\n".join(line for line in body if line)
    digest = hashlib.sha256(f"v{TEXT_VERSION}\n{text}".encode()).hexdigest()
    return text, digest


def split_text(text: str) -> tuple[str, str]:
    """The ``(title, body)`` pair :func:`embed_client.embed` takes as a
    document: the inverse of how :func:`job_text` joins them."""
    title, _, body = text.partition("\n")
    return title, body


def eligible_query(text: str) -> bool:
    """Whether a search's free text is worth embedding: at least
    :data:`MIN_QUERY_CHARS` characters that are not whitespace."""
    return sum(1 for c in text if not c.isspace()) >= MIN_QUERY_CHARS


# --------------------------------------------------------------------------
# Is there an embedder
# --------------------------------------------------------------------------


def installed(svc: Any) -> bool:
    """Whether the retrieval weights *and* the runtime that serves them are on
    disk and there is a worker to spawn the child. A few ``stat`` calls: cheap,
    but still disk, so the frame thread asks through a memo
    (``embed_indexer.LibraryIndexer.available``) and never per frame."""
    config = getattr(svc, "config", None)
    worker = getattr(svc, "worker", None)
    if config is None or getattr(worker, "familiar_embed", None) is None:
        return False
    try:
        for key in _ROWS:
            spec = models.FAMILIAR_MODELS[key]
            if not fetch.present(config, "familiar", spec):
                return False
            # A stand-in config answers every ``is_file`` truthily; a real one
            # resolves to a path. Refusing anything else keeps a test double
            # from switching the whole feature on.
            if not isinstance(fetch.familiar_dir(config, spec), Path):
                return False
    except Exception:
        return False
    return True


def _stopping(svc: Any) -> bool:
    """Whether the worker's loop has gone (``Runtime.shutdown`` ends it first):
    nothing left to embed with, so a background pass should simply stop."""
    loop = getattr(svc, "loop", None)
    return loop is not None and not loop.is_running()


def _on_loop(svc: Any, factory: Any, timeout: float) -> np.ndarray:
    """Run an ``embed_client.embed`` coroutine on the worker's loop and wait.

    Every way of not getting vectors is an :class:`embed_client.EmbedUnavailable`
    -- the one thing callers catch. ``call_on_loop`` returns ``None`` with no
    worker and raises ``TimeoutError`` past *timeout* (cancelling the coroutine
    itself); a closing loop surfaces as ``RuntimeError``/``CancelledError``.
    The message never carries the text being embedded.
    """
    try:
        out = svc.call_on_loop(factory, timeout=timeout)
    except embed_client.EmbedUnavailable:
        raise
    except (
        TimeoutError,
        concurrent.futures.TimeoutError,
        concurrent.futures.CancelledError,
        RuntimeError,
        OSError,
    ) as exc:
        raise embed_client.EmbedUnavailable(
            f"the embedder did not answer ({type(exc).__name__})"
        ) from None
    if out is None:
        raise embed_client.EmbedUnavailable("there is no worker to embed with")
    return out


def embed_documents(svc: Any, texts: Sequence[str]) -> np.ndarray:
    """``(n, DIM)`` unit vectors for *texts* (each a :func:`job_text` string), or
    :class:`embed_client.EmbedUnavailable`. Blocking: a ``TaskRunner`` task's
    business, never the frame thread's."""
    if not texts:
        return np.empty((0, DIM), dtype=np.float32)
    pairs = [split_text(t) for t in texts]
    out = _on_loop(
        svc,
        lambda: embed_client.embed(svc.worker.familiar_embed, pairs, role="document"),
        CALL_TIMEOUT,
    )
    if len(out) != len(pairs):
        raise embed_client.EmbedUnavailable("the embedder returned the wrong number of vectors")
    return embed_index.truncate(out, DIM)


_warming: list[Any] = [None]


def _warm(svc: Any, server: Any) -> None:
    """Start the embedder child without waiting for it. A search that finds it
    stopped (the idle sweep stops it after five minutes) answers from the
    substring match at once and asks it to come up; the next refresh -- never
    more than a few seconds away -- finds it running."""
    loop = getattr(svc, "loop", None)
    previous = _warming[0]
    if loop is None or not loop.is_running() or (previous is not None and not previous.done()):
        return
    try:
        future = asyncio.run_coroutine_threadsafe(server.ensure_started(), loop)
    except RuntimeError:
        return

    def _quiet(done: Any) -> None:
        # A failed start is the server's own backoff and log line; reading the
        # exception here is only so asyncio does not report it as never retrieved.
        try:
            done.exception()
        except BaseException:
            return

    future.add_done_callback(_quiet)
    _warming[0] = future


def embed_query(svc: Any, text: str) -> np.ndarray | None:
    """The ``(DIM,)`` query vector for a search's free text, or ``None`` for any
    reason it cannot be had *right now* (not installed, not running yet, slow,
    refused). ``None`` is never an error: the caller already holds the
    substring answer. Blocking, off the frame thread only."""
    server = getattr(getattr(svc, "worker", None), "familiar_embed", None)
    if server is None:
        return None
    # A cold child takes seconds to load, and the substring result must not
    # wait for it: start it and answer without meaning this once. (With no
    # loop thread -- a script, a test -- there is nothing to start it *on*, so
    # the embed below does it inline.)
    if getattr(svc, "loop", None) is not None and not getattr(server, "running", True):
        _warm(svc, server)
        return None
    try:
        out = _on_loop(
            svc,
            lambda: embed_client.embed(server, [text[:MAX_QUERY_CHARS]], role="query"),
            QUERY_TIMEOUT,
        )
        if len(out) != 1:
            return None
        return embed_index.truncate(out[0], DIM)
    except embed_client.EmbedUnavailable as exc:
        log.info("no query embedding, searching by text alone (%s)", exc)
        return None


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class IndexPass:
    """What one bounded indexing pass did. *more* is whether it stopped because
    the batch was full rather than because nothing was left."""

    indexed: int
    more: bool


def _row_text(row: dict[str, Any]) -> tuple[str, str] | None:
    return job_text(row)


def index_pending(svc: Any, limit: int = BATCH, *, verify: bool = False) -> IndexPass:
    """Embed up to *limit* finished jobs whose vector is missing or stale and
    store them. Raises :class:`embed_client.EmbedUnavailable` (nothing written)
    when the embedder cannot answer -- the scheduler's cue to back off.

    ``verify=True`` also re-reads every job that *has* a vector and re-embeds
    the ones whose text no longer hashes to the stored ``text_hash`` (a rename,
    a changed facet set); see ``JobStore.pending_embeddings``.
    """
    store = svc.store
    if _stopping(svc):
        return IndexPass(0, False)
    pending = store.pending_embeddings(MODEL_SHA, DIM, _row_text, limit, verify=verify)
    if not pending:
        return IndexPass(0, False)
    vectors = embed_documents(svc, [text for _id, text, _hash in pending])
    written = store.upsert_embeddings(
        MODEL_SHA,
        DIM,
        [
            (job_id, text_hash, vec)
            for (job_id, _text, text_hash), vec in zip(pending, vectors, strict=True)
        ],
    )
    return IndexPass(written, len(pending) >= limit)


def index_jobs(svc: Any, job_ids: Sequence[str]) -> int:
    """Embed exactly these jobs now (a job that just finished, a row just
    renamed). -> how many vectors were written; a job that is not finished, is a
    sweep unit, has nothing to say or already has a current vector costs
    nothing. Raises :class:`embed_client.EmbedUnavailable` like
    :func:`index_pending`."""
    store = svc.store
    ids = list(dict.fromkeys(job_ids))
    if not ids or _stopping(svc):
        return 0
    stamps = store.embedding_stamps(ids)
    items: list[tuple[str, str, str]] = []
    for job_id in ids:
        job = store.get(job_id)
        if job is None or job.get("status") != "done" or job.get("sweep_id"):
            continue
        built = job_text(job)
        if built is None:
            continue
        text, text_hash = built
        if stamps.get(job_id) == (MODEL_SHA, DIM, text_hash):
            continue
        items.append((job_id, text, text_hash))
    if not items:
        return 0
    vectors = embed_documents(svc, [text for _id, text, _hash in items])
    return int(
        store.upsert_embeddings(
            MODEL_SHA,
            DIM,
            [
                (job_id, text_hash, vec)
                for (job_id, _t, text_hash), vec in zip(items, vectors, strict=True)
            ],
        )
    )


# --------------------------------------------------------------------------
# Searching
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Hits:
    """A search's answer. *ids* is the fused ranking, best first. *semantic* is
    the subset that is there **because of its meaning** -- the rows whose text
    does not necessarily contain what was typed, and which the Library's filter
    must therefore let past its free-text clause (and only that clause)."""

    ids: list[str]
    semantic: frozenset[str] = frozenset()


def search(
    svc: Any,
    text: str,
    *,
    limit: int,
    tags: Sequence[str] = (),
    names: Sequence[str] = (),
    status: str | None = None,
    favorite: bool | None = None,
    trash: bool = False,
    query_vec: Any = None,
) -> Hits:
    """``JobStore.search_ids`` fused with the meaning ranking of *query_vec*.

    With no *query_vec* this is exactly ``search_ids`` and ``semantic`` is empty
    -- what a machine without the retrieval row, a too-short query, or an
    embedder that did not answer gets. With one, the two rankings are fused by
    reciprocal rank: the substring ranking is newest-first, the meaning ranking
    is by cosine (above :data:`SEMANTIC_Z_FLOOR`, at most :data:`SEMANTIC_CAP`),
    and a row that is in both outranks a row that is in either. The store
    scopes both identically (trash, status, favourites, ``tag:``/``name:``), so
    meaning widens only the free-text clause. Never touches the network:
    *query_vec* was computed by the caller.
    """
    like = svc.store.search_ids(
        text,
        limit=limit,
        tags=tags,
        names=names,
        status=status,
        favorite=favorite,
        trash=trash,
    )
    if query_vec is None:
        return Hits(like)
    ranked = svc.store.semantic_ranked(
        query_vec,
        model_sha=MODEL_SHA,
        dim=DIM,
        z_floor=SEMANTIC_Z_FLOOR,
        cap=SEMANTIC_CAP,
        min_rows=SEMANTIC_MIN_ROWS,
        tags=tags,
        names=names,
        status=status,
        favorite=favorite,
        trash=trash,
    )
    meaning = [job_id for job_id, _score in ranked]
    fused = [job_id for job_id, _ in embed_index.rrf([like, meaning], limit=limit)]
    return Hits(fused, frozenset(meaning) & frozenset(fused))
