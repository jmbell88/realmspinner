"""The Manual's meaning index: the dense half of Familiar's Manual retrieval.

BM25 (``familiar/retrieval.py``) finds the sections that share a *word* with a
question; the embedder (EmbeddingGemma 2, the optional ``familiar_embed`` row,
``pipelines/llama.py``'s second child) finds the ones that share its *meaning*
("undo a mistake while painting pixels" -> ``05 Drawing > Undo``). This module
holds everything that joins the two: where the Manual's vectors live on disk,
how they get built, and how a question's two rankings are fused. It is the
service layer's, not ``familiar/``'s, because it reaches the worker's embedder
child (``svc.worker.familiar_embed``) and the config -- ``familiar/`` may not
import ``service``.

**Nothing here may block ``ask`` and nothing here runs on the frame thread.**
The matrix is 910 chunks and took 165 s to build on the CPU (measured
2026-10-06, ``dev/measurements/2026-10-06-familiar-gemma4-12b.md``), so a cache
miss answers from BM25 at once and starts *one* background build; the answer
that arrives while it runs is exactly the one a machine without the retrieval
row gets. Every failure to get a vector -- a missing row, a server that will
not start, a timeout -- is "use BM25", never an error the user sees.

The build runs on its own daemon thread rather than a ``TaskRunner`` slot: it
outlives any one question, it must not hold one of the pool's four workers for
three minutes (a thumbnail decode or an export would queue behind it), and a
daemon thread cannot keep the process alive at exit the way a pool thread
parked in an HTTP call can. It checks a stop flag -- and whether the worker's
loop is still running, which is what ``Runtime.shutdown`` ends first -- between
batches, so quitting mid-build leaves no half-written file (the matrix is
written only once, temp-then-replace) and no thread to wait for.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .. import fetch, models
from ..familiar import embed_client, embed_index
from ..pipelines import llama

log = logging.getLogger(__name__)

#: How deep each ranking is cut before fusion. ``Index.citations`` only ever
#: reads the first ``limit * 4`` (24) fused ids, so this is the headroom a
#: chunk needs to be ranked high by one side and absent from the other's top.
CANDIDATES = 40

#: A build is started at most this many times per cache key in one session: the
#: first try plus one retry. A failed build (the embedder would not start, the
#: card is held, a transient reset) is worth one more attempt on the next
#: question; a persistent one must not become a retry loop that respawns a
#: server and burns three minutes of CPU every time somebody asks something.
MAX_BUILD_ATTEMPTS = 2

#: Texts per ``embed`` call during a build -- the client's own batch, so each
#: call is exactly one HTTP request and the stop flag is checked between them.
BUILD_BATCH = embed_client.BATCH_SIZE

#: One batch of 16 of the longest chunks is a few seconds on the CPU; this
#: bounds the wait on the loop for a cold start plus one batch.
BUILD_CALL_TIMEOUT = llama.STARTUP_TIMEOUT + embed_client.EMBED_TIMEOUT + 30.0

#: A question's own embedding: 33 ms warm (2026-10-06), a couple of seconds if
#: the idle sweep had stopped the server. Short on purpose -- ``ask`` is
#: waiting, and a hung embedder must cost it seconds, not minutes.
QUERY_TIMEOUT = 45.0

#: The embedder's two rows: the shared ``llama-server`` runtime and the weights.
_ROWS = ("familiar_runtime", "familiar_embed")

_lock = threading.Lock()
#: Matrices already in memory, by cache key: a process loads the file once.
_matrices: dict[str, np.ndarray] = {}
#: Keys with a build in flight, and how many builds each has had.
_building: set[str] = set()
_attempts: dict[str, int] = {}
#: Set to end a build between batches (:func:`stop_builds`).
_stop = threading.Event()
#: ``(index, key)`` of the last Manual index asked about: ``tree_sha`` walks
#: ~1 MB of chunk text, and ``ask`` calls it on every question.
_key_memo: list[Any] = [None, ""]


def reset() -> None:
    """Forget every in-memory matrix, build record and the stop flag. For
    tests, and for a process that wants the next question to start clean."""
    with _lock:
        _matrices.clear()
        _building.clear()
        _attempts.clear()
        _key_memo[0], _key_memo[1] = None, ""
    _stop.clear()


def stop_builds() -> None:
    """Ask any running build to stop at its next batch boundary."""
    _stop.set()


def _spawn(target: Callable[[], None]) -> None:
    """Run *target* on a new daemon thread. One seam, so a test can capture the
    build instead of racing a real thread."""
    threading.Thread(target=target, name="realmspinner-manual-embed", daemon=True).start()


# --------------------------------------------------------------------------
# Is there an embedder, and where do the vectors live
# --------------------------------------------------------------------------


def installed(svc: Any) -> bool:
    """Whether the retrieval row *and* the runtime that serves it are on disk,
    and there is a worker to spawn it. Cheap (a few ``stat`` calls), but it is
    still disk: the frame thread asks through a memo (``studio/manual/semantic``).
    """
    config = getattr(svc, "config", None)
    worker = getattr(svc, "worker", None)
    if config is None or getattr(worker, "familiar_embed", None) is None:
        return False
    try:
        return all(fetch.present(config, "familiar", models.FAMILIAR_MODELS[k]) for k in _ROWS)
    except Exception:
        return False


def cache_dir(svc: Any) -> Path:
    """``<home>/cache``. Under ``config.home`` so ``REALMSPINNER_HOME`` moves it
    with everything else the app owns -- never a bare ``Path.home()``."""
    return Path(svc.config.home) / "cache"


def manual_key(index: Any) -> str:
    """The cache key for *index*'s Manual: its text, the embedder's weights and
    the width (:func:`embed_index.cache_key`). Editing one sentence of one
    chapter, or a new model pin, is a different key and so a miss."""
    with _lock:
        if _key_memo[0] is index:
            return str(_key_memo[1])
    key = embed_index.cache_key(
        embed_index.tree_sha(index.chunks),
        models.FAMILIAR_EMBED_GGUF_SHA256,
        embed_index.MANUAL_DIM,
    )
    with _lock:
        _key_memo[0], _key_memo[1] = index, key
    return key


# --------------------------------------------------------------------------
# The matrix: memory, then disk, else one background build
# --------------------------------------------------------------------------


def matrix(svc: Any, index: Any, *, build: bool = True) -> np.ndarray | None:
    """The Manual's ``(chunks, 768)`` matrix if it is available *now*, else
    ``None`` -- after starting the one background build a miss calls for (unless
    *build* is false, the embedder is not installed, the key already has one in
    flight, or it has used up :data:`MAX_BUILD_ATTEMPTS`).

    Never builds inline. A file that is corrupt, truncated, the wrong width or
    the wrong row count is a miss (``embed_index.load_matrix``), which is also
    what makes a Manual edited since the file was written a rebuild.
    """
    key = manual_key(index)
    with _lock:
        found = _matrices.get(key)
    if found is not None:
        return found
    loaded = embed_index.load_matrix(cache_dir(svc), key, rows=len(index.chunks))
    if loaded is not None:
        with _lock:
            _matrices[key] = loaded
        return loaded
    if build:
        _start_build(svc, index, key)
    return None


def build_state(index: Any) -> str:
    """``"ready"``, ``"building"`` or ``"unavailable"`` (no build running and
    none will be started: not installed, or the retry was spent)."""
    key = manual_key(index)
    with _lock:
        if key in _matrices:
            return "ready"
        if key in _building:
            return "building"
    return "unavailable"


def _start_build(svc: Any, index: Any, key: str) -> bool:
    if _stop.is_set() or not installed(svc) or not index.chunks:
        return False
    with _lock:
        if key in _building or _attempts.get(key, 0) >= MAX_BUILD_ATTEMPTS:
            return False
        _building.add(key)
        _attempts[key] = _attempts.get(key, 0) + 1
    try:
        _spawn(lambda: _build(svc, index, key))
    except Exception:
        # A thread that could not start must not leave the key "building" for
        # the rest of the session.
        with _lock:
            _building.discard(key)
        log.exception("could not start the Manual's meaning-index build")
        return False
    return True


def _stopping(svc: Any) -> bool:
    if _stop.is_set():
        return True
    loop = getattr(svc, "loop", None)
    # ``Runtime.shutdown`` stops the worker's loop before anything else; a
    # build whose loop has gone has nothing left to call.
    return loop is not None and not loop.is_running()


def _build(svc: Any, index: Any, key: str) -> None:
    """The build body, on its own thread. Embeds ``(title_path, text)`` pairs
    as documents, a batch per call, then saves and prunes -- in that order, so
    a build that fails or is stopped never deletes the last good file."""
    try:
        chunks = index.chunks
        parts: list[np.ndarray] = []
        for start in range(0, len(chunks), BUILD_BATCH):
            if _stopping(svc):
                log.info("the Manual's meaning-index build stopped at chunk %d", start)
                return
            batch = [(c.title_path, c.text) for c in chunks[start : start + BUILD_BATCH]]
            part = svc.call_on_loop(
                lambda b=batch: embed_client.embed(
                    svc.worker.familiar_embed, b, role="document"
                ),
                timeout=BUILD_CALL_TIMEOUT,
            )
            if part is None or len(part) != len(batch):
                raise embed_client.EmbedUnavailable("the embedding server gave no vectors")
            parts.append(part)
        built = embed_index.truncate(np.vstack(parts), embed_index.MANUAL_DIM)
        directory = cache_dir(svc)
        embed_index.save_matrix(directory, key, built)
        embed_index.prune_superseded(directory, key)
        with _lock:
            _matrices[key] = built
        log.info("the Manual's meaning index is built (%d chunks)", len(chunks))
    except Exception as exc:
        # The text being embedded never appears in these messages
        # (``embed_client``'s own rule), so the message is safe to log.
        log.warning("the Manual's meaning index could not be built: %s", exc)
    finally:
        with _lock:
            _building.discard(key)


# --------------------------------------------------------------------------
# A question: the query vector and the fused ranking
# --------------------------------------------------------------------------


def query_vector(svc: Any, text: str) -> np.ndarray | None:
    """*text*'s 768-d query vector, or ``None`` for any reason it cannot be had.
    Runs on the worker loop via ``call_on_loop``, so the caller must itself be
    a thread that may block (an ``ask`` worker, a pane task) -- never the frame."""
    try:
        out = svc.call_on_loop(
            lambda: embed_client.embed(svc.worker.familiar_embed, [text], role="query"),
            timeout=QUERY_TIMEOUT,
        )
        if out is None or len(out) != 1:
            return None
        return embed_index.truncate(out[0], embed_index.MANUAL_DIM)
    except Exception as exc:
        # ``EmbedUnavailable`` is the expected one; a timeout or a closing loop
        # is the same answer. Class name only: nothing here may echo the text.
        log.info("no query embedding, falling back to BM25 (%s)", type(exc).__name__)
        return None


def fused_chunks(svc: Any, index: Any, query: str, depth: int = CANDIDATES) -> list[int] | None:
    """Chunk indices for *query*, best first, by reciprocal-rank fusion of the
    BM25 and cosine rankings -- or ``None`` meaning "use BM25 alone" (the
    embedder is not installed, the matrix is not ready, the query could not be
    embedded). The caller turns either answer into citations the same way."""
    if not installed(svc):
        return None
    vectors = matrix(svc, index)
    if vectors is None:
        return None
    vector = query_vector(svc, query)
    if vector is None:
        return None
    try:
        dense = embed_index.top_k(vector, vectors, depth)
    except ValueError:
        return None
    bm25 = index.rank(query, depth)
    fused = embed_index.rrf(
        [[i for i, _ in bm25], [i for i, _ in dense]], limit=depth
    )
    return [int(i) for i, _ in fused]


@dataclasses.dataclass(frozen=True)
class ManualHits:
    """What the Manual pane's semantic search gets back.

    *state* is ``"ready"`` (``sections`` is the answer), ``"building"`` (the
    meaning index is not built yet: the pane says so and keeps its text
    results) or ``"unavailable"`` (fall back silently). *sections* are
    ``(chapter_key, anchor_or_None, title_path)``, best first, one per section.
    """

    state: str
    sections: tuple[tuple[str, str | None, str], ...] = ()


def sections_for(svc: Any, index: Any, query: str, limit: int = 12) -> ManualHits:
    """The best *limit* distinct Manual sections for *query* under the fused
    ranking. A section whose long text was split into several chunks appears
    once, at its best chunk's place -- the same regrouping ``Index.citations``
    makes."""
    if not installed(svc):
        return ManualHits("unavailable")
    if matrix(svc, index) is None:
        return ManualHits(build_state(index))
    order = fused_chunks(svc, index, query)
    if order is None:
        return ManualHits("unavailable")
    seen: set[tuple[str, str | None]] = set()
    rows: list[tuple[str, str | None, str]] = []
    for i in order:
        chunk = index.chunks[i]
        key = (chunk.chapter, chunk.anchor)
        if key in seen:
            continue
        seen.add(key)
        rows.append((chunk.chapter, chunk.anchor, chunk.title_path))
        if len(rows) >= limit:
            break
    return ManualHits("ready", tuple(rows))

