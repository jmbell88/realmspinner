"""The pure half of Familiar's dense retrieval: vectors in, rankings and a cache out.

Phase 2 adds EmbeddingGemma 2 to the Manual search and the Library. The model
itself is a second ``llama-server`` child (``pipelines/llama.py``) reached over
HTTP by :mod:`.embed_client`; everything that does not need the network lives
here, so it can be tested without a server and imported by a script that never
touches one. **numpy, the stdlib and ``core/safeio`` only** -- ``core/`` is
layer 0, so this stays inside what ``tests/familiar/test_familiar_imports.py``
pins. Chunks are *duck-typed* (:func:`tree_sha` reads four attributes), so this
module never imports :mod:`.retrieval`.

What is here, in the order a retrieval turn uses it:

* :func:`tree_sha` and :func:`cache_key` -- what the Manual's vectors were built
  *from*, so a stale file can never be mistaken for a fresh one.
* :func:`load_matrix` / :func:`save_matrix` / :func:`prune_superseded` -- the
  on-disk cache (the caller passes ``~/.realmspinner/cache/``). Indexing 910
  Manual chunks takes ~165 s on the CPU (dev/measurements/2026-10-06-familiar-
  gemma4-12b.md), which is acceptable exactly once, as a background task, and
  never again for the same tree.
* :func:`truncate` -- Matryoshka: EmbeddingGemma 2 is trained so the first
  ``dim`` components of its 768-d vector are themselves a good embedding once
  renormalised. 768 for the Manual, 256 for the Library (a Library row's text
  is short and there can be many rows, so the smaller vector is the right
  trade).
* :func:`top_k` -- cosine ranking in numpy.
* :func:`rrf` -- reciprocal-rank fusion of that ranking with the BM25
  ``retrieval.Index`` ranking, so a dense miss that BM25 gets right (and the
  reverse) both survive. Nothing is score-calibrated against anything else.

Vectors are float32 throughout, L2-normalised, so cosine is a dot product.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections.abc import Hashable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from realmspinner.core.safeio import atomic, npyguard

#: The model's native width.
NATIVE_DIM = 768
#: The Manual keeps the full vector: 910 chunks, quality over size.
MANUAL_DIM = 768
#: The Library keeps the 256-d Matryoshka prefix: one row per job, many jobs.
LIBRARY_DIM = 256
#: Every width the model's card lists as supported (native + Matryoshka).
SUPPORTED_DIMS = (768, 512, 256, 128)

#: Standard reciprocal-rank-fusion constant (Cormack et al. 2009).
RRF_K = 60

#: A cache file past this is never read: a corrupt or hostile file in the
#: user's cache directory must not become a huge allocation. Equal to
#: ``npyguard.MAX_ARRAY_BYTES`` (which re-checks the header's declared size);
#: the Library at 256-d float32 is 1 KiB a row, so this is ~260k rows.
MAX_CACHE_BYTES = 1 << 28

#: Cache keys are file names. Anything outside this alphabet could climb out of
#: the cache directory (``..``, a separator) or collide with another key's
#: parse, so it is refused rather than escaped.
_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_KEY = re.compile(
    r"(?P<kind>[a-z][a-z0-9_]*)-(?P<dim>[0-9]+)-(?P<tree>[A-Za-z0-9_]+)-(?P<model>[A-Za-z0-9_]+)"
)

#: How many hex characters of each sha go into a key. 64 bits apiece is far
#: past any collision this cache can see and keeps file names readable.
_SHA_CHARS = 16


def check_dim(dim: int) -> int:
    """*dim* if the model supports it, else a ``ValueError`` naming it and the
    supported set -- a typo'd width must fail here, not as a silently wrong
    shape three layers down."""
    if isinstance(dim, bool) or not isinstance(dim, int) or dim not in SUPPORTED_DIMS:
        raise ValueError(
            f"EmbeddingGemma does not support an embedding dimension of {dim!r}; "
            f"supported: {', '.join(str(d) for d in SUPPORTED_DIMS)}"
        )
    return dim


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """Each row of a 2-D float array divided by its L2 norm. A zero row stays
    zero rather than becoming NaN (it then scores 0 against everything)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def truncate(vectors: Any, dim: int) -> np.ndarray:
    """The first *dim* components of each row, renormalised to unit length.

    Accepts a 2-D ``(n, >=dim)`` array or a single 1-D vector (returned 1-D).
    float32 out. *dim* must be one of :data:`SUPPORTED_DIMS`; a vector narrower
    than *dim* is refused (a 256-d vector cannot be "truncated" to 768).
    Slicing without renormalising is the classic mistake: the prefix of a unit
    vector is shorter than 1, so cosine scores would shrink with the width.
    """
    check_dim(dim)
    arr = np.asarray(vectors, dtype=np.float32)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array of vectors, got {arr.ndim}-D")
    if arr.shape[1] < dim:
        raise ValueError(f"cannot truncate {arr.shape[1]}-d vectors to {dim}")
    out = _normalise(np.ascontiguousarray(arr[:, :dim], dtype=np.float32)).astype(
        np.float32, copy=False
    )
    return out[0] if single else out


def top_k(query: Any, matrix: Any, k: int) -> list[tuple[int, float]]:
    """The *k* rows of *matrix* most cosine-similar to *query*, best first, as
    ``[(row_index, score)]``.

    Ties on score break to the lower row index, so the same inputs always give
    the same ranking (a flapping order would make the citation numbers a
    Familiar reply carries change between identical questions). An empty
    matrix, ``k <= 0`` or ``k`` larger than the row count are all fine and
    return what exists. Vectors are expected unit-length; any row or query
    that is not (norm off by more than 1e-3) is renormalised here rather than
    trusted, because a stale or hand-built matrix would otherwise rank by
    magnitude. A width mismatch between query and matrix is a ``ValueError``.
    """
    q = np.asarray(query, dtype=np.float32)
    if q.ndim != 1:
        raise ValueError("top_k takes one query vector")
    m = np.asarray(matrix, dtype=np.float32)
    if m.size == 0 or k <= 0:
        return []
    if m.ndim != 2:
        raise ValueError("top_k takes a 2-D matrix")
    if m.shape[1] != q.shape[0]:
        raise ValueError(f"query is {q.shape[0]}-d but the matrix rows are {m.shape[1]}-d")
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return []
    if abs(q_norm - 1.0) > 1e-3:
        q = q / q_norm
    norms = np.linalg.norm(m, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-3):
        m = _normalise(m)
    scores = np.nan_to_num(m @ q, nan=-1.0, posinf=-1.0, neginf=-1.0)
    # lexsort sorts by its LAST key first: score descending, then index ascending.
    order = np.lexsort((np.arange(scores.shape[0]), -scores))[: min(k, scores.shape[0])]
    return [(int(i), float(scores[i])) for i in order]


def rrf(
    rankings: Iterable[Sequence[Hashable]], k: int = RRF_K, limit: int | None = None
) -> list[tuple[Any, float]]:
    """Reciprocal-rank fusion: each id scores ``sum(1 / (k + rank))`` over every
    ranking it appears in (rank is 1-based; an id missing from a ranking adds
    nothing from it, and an id repeated inside one ranking counts once, at its
    first place). Returns ``[(id, fused_score)]`` best first, cut to *limit*.

    An id high in both lists beats an id first in only one, which is the whole
    point of fusing a dense ranking with BM25: neither scale is comparable to
    the other, only the ranks are. Ties break by the best (lowest) rank the id
    reached in any list, then by id itself, so the order is stable; ids must
    therefore be mutually orderable (all ints or all strings). Scores are
    summed with :func:`math.fsum`, so they do not depend on list order.
    """
    if k < 0:
        raise ValueError("rrf's k must be >= 0")
    terms: dict[Hashable, list[float]] = {}
    best_rank: dict[Hashable, int] = {}
    for ranking in rankings:
        seen: set[Hashable] = set()
        for position, ident in enumerate(ranking, start=1):
            if ident in seen:
                continue
            seen.add(ident)
            terms.setdefault(ident, []).append(1.0 / (k + position))
            if position < best_rank.get(ident, position + 1):
                best_rank[ident] = position
    fused = [(ident, math.fsum(t)) for ident, t in terms.items()]
    fused.sort(key=lambda pair: (-pair[1], best_rank[pair[0]], pair[0]))
    return fused if limit is None else fused[: max(limit, 0)]


# --------------------------------------------------------------------------
# The Manual's identity and the on-disk cache
# --------------------------------------------------------------------------


def tree_sha(chunks: Iterable[Any]) -> str:
    """A stable sha256 (hex) over the Manual chunks, in order.

    Each chunk is read as ``(chapter, anchor, title_path, text)`` -- the four
    things a vector depends on or is looked up by -- duck-typed, so any object
    with those attributes works (``retrieval.Chunk`` is the real one). Encoded
    as JSON per chunk so ``None`` and ``""`` differ and no field can bleed into
    its neighbour. Editing one sentence of one chapter changes this, which is
    what invalidates the cached matrix; chunk objects that merely moved
    in memory do not.
    """
    h = hashlib.sha256(b"familiar-embed-tree-v1\n")
    for chunk in chunks:
        record = [chunk.chapter, chunk.anchor, chunk.title_path, chunk.text]
        h.update(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def cache_key(tree_sha: str, model_sha: str, dim: int, kind: str = "manual") -> str:
    """The cache file stem for vectors of *kind* built from *tree_sha* by the
    model whose weights hash to *model_sha*, at width *dim*.

    All three identities are in the name because any one changing makes the old
    matrix wrong: new text, new model weights, or a different Matryoshka width.
    *kind* (``"manual"`` or ``"library"``) keeps the two corpora from pruning
    each other. Shas are shortened to their first 16 characters.
    """
    check_dim(dim)
    for label, token in (("tree_sha", tree_sha), ("model_sha", model_sha)):
        if not isinstance(token, str) or not _TOKEN.fullmatch(token):
            raise ValueError(f"{label} must be a non-empty alphanumeric string, got {token!r}")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", kind or ""):
        raise ValueError(
            f"cache kind must be lowercase letters, digits and underscores, got {kind!r}"
        )
    return f"{kind}-{dim}-{tree_sha[:_SHA_CHARS]}-{model_sha[:_SHA_CHARS]}"


def _parse_key(key: str) -> re.Match[str]:
    match = _KEY.fullmatch(key)
    if match is None:
        raise ValueError(f"not a cache key made by cache_key(): {key!r}")
    return match


def _path(cache_dir: Path | str, key: str) -> Path:
    _parse_key(key)
    return Path(cache_dir) / f"{key}.npy"


def save_matrix(cache_dir: Path | str, key: str, matrix: Any) -> Path:
    """Write *matrix* (float32, ``(n, dim)`` with ``dim`` taken from *key*) as
    ``<cache_dir>/<key>.npy`` and return the path.

    Staged and ``os.replace``d via ``core/safeio/atomic`` like every served
    file: a crash, a full disk or a second app instance mid-write leaves
    either the previous file or none under the served name, never half of one.
    (:func:`load_matrix` would read a half file as a miss anyway; this is so
    there is never one to read.) *cache_dir* is created if missing.
    """
    path = _path(cache_dir, key)
    dim = int(_parse_key(key)["dim"])
    arr = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"key {key!r} is {dim}-d but the matrix is shaped {arr.shape}")
    buf = io.BytesIO()
    np.lib.format.write_array(buf, arr, allow_pickle=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic.write_bytes(path, buf.getvalue())
    return path


def load_matrix(cache_dir: Path | str, key: str, *, rows: int | None = None) -> np.ndarray | None:
    """The cached matrix for *key*, or ``None`` -- a miss -- for *any* reason:
    no file, unreadable, truncated, not float32, not 2-D, a width that is not
    the one the key names, or (when *rows* is given) a row count that is not
    the corpus's. Never raises for a bad file: a cache that can take the
    search down is worse than no cache, and the caller's answer to ``None`` is
    always "rebuild it". A malformed *key* is the caller's bug and does raise.
    """
    path = _path(cache_dir, key)
    dim = int(_parse_key(key)["dim"])
    try:
        if path.stat().st_size > MAX_CACHE_BYTES:
            return None
        arr = npyguard.read_array(path.read_bytes(), "the embedding cache")
    except (OSError, ValueError, EOFError, MemoryError):
        return None
    if arr.dtype != np.float32 or arr.ndim != 2 or arr.shape[1] != dim:
        return None
    if rows is not None and arr.shape[0] != rows:
        return None
    if not np.isfinite(arr).all():
        return None
    return np.ascontiguousarray(arr)


def prune_superseded(cache_dir: Path | str, keep_key: str) -> list[str]:
    """Delete every cache file of the same kind and width as *keep_key* except
    *keep_key*'s own, and return the names removed.

    Each Manual edit or model update writes a new key; without this the
    directory would gain a 3 MiB matrix per release for good. Other kinds and
    other widths are untouched (the Library's 256-d file is not the Manual's
    768-d one), as is anything that does not parse as a key. Call it after a
    successful :func:`save_matrix`, so a failed rebuild never deletes the last
    good file.
    """
    keep = _parse_key(keep_key)
    directory = Path(cache_dir)
    removed: list[str] = []
    try:
        entries = sorted(directory.glob("*.npy"))
    except OSError:
        return removed
    for entry in entries:
        match = _KEY.fullmatch(entry.stem)
        if match is None or entry.stem == keep_key:
            continue
        if match["kind"] != keep["kind"] or match["dim"] != keep["dim"]:
            continue
        try:
            entry.unlink()
        except OSError:
            continue
        removed.append(entry.name)
    return removed
