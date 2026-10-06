"""Phase 2: Familiar's Manual answers fuse BM25 with the embedder's ranking.

``service/familiar_manual.py`` is what joins the two. Everything here runs with
no server: ``embed_client.embed`` is replaced by a fake that gives every Manual
chunk a deterministic unit vector, so *which section is dense-nearest to a
query* is something a test chooses, and ``familiar_manual._spawn`` is replaced
by a capture so a test can see that a cache miss started a build without
racing a real thread -- and run the build itself on the test's own thread.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner import fetch, models
from realmspinner.config import Config
from realmspinner.familiar import embed_client, embed_index, retrieval
from realmspinner.service import familiar as svc_familiar
from realmspinner.service import familiar_manual

#: Words no chapter contains: BM25 finds nothing for it, so any citation for
#: this query came from the dense side alone.
ONLY_MEANING = "zxqvj wkpt"
#: A question BM25 answers on its own.
LEXICAL = "how do I export a GLB"


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(embed_index.NATIVE_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture(scope="module")
def index() -> retrieval.Index:
    return retrieval.Index.build()


@pytest.fixture(autouse=True)
def _clean_module_state():
    familiar_manual.reset()
    yield
    familiar_manual.reset()


@pytest.fixture
def target(index) -> int:
    """A chunk the dense ranking will be told is nearest to :data:`ONLY_MEANING`."""
    return next(i for i, c in enumerate(index.chunks) if i > 100 and c.anchor)


class _Embedder:
    """A stand-in for ``embed_client.embed``: chunk *i* is ``_unit(i)``, and the
    query :data:`ONLY_MEANING` is exactly the *target* chunk's vector."""

    def __init__(self, index, target: int) -> None:
        self.by_text = {(c.title_path, c.text): i for i, c in enumerate(index.chunks)}
        self.target = target
        self.calls: list[tuple[str, int]] = []
        self.fail: Exception | None = None
        self.after_call = None

    async def __call__(self, server, texts, *, role, transport=None):
        self.calls.append((role, len(texts)))
        if self.fail is not None:
            raise self.fail
        if role == "query":
            out = np.stack([_unit(self.target)] * len(texts))
        else:
            out = np.stack([_unit(self.by_text[t]) for t in texts])
        if self.after_call is not None:
            self.after_call()
        return out.astype(np.float32)


def _config(tmp_path: Path) -> Config:
    return Config(
        home=tmp_path / "home",
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models",
        familiar_runtime_dir=tmp_path / "engine" / "llama",
        familiar_models_dir=tmp_path / "models" / "familiar",
    )


class _Svc:
    def __init__(self, tmp_path: Path) -> None:
        self.config = _config(tmp_path)
        self.worker = SimpleNamespace(familiar=object(), familiar_embed=object())
        self.loop = None

    def call_on_loop(self, factory, timeout: float = 30.0):
        return asyncio.run(factory())


@pytest.fixture
def svc(tmp_path) -> _Svc:
    return _Svc(tmp_path)


@pytest.fixture
def embedder(monkeypatch, index, target) -> _Embedder:
    fake = _Embedder(index, target)
    monkeypatch.setattr(embed_client, "embed", fake)
    return fake


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setattr(familiar_manual, "installed", lambda svc: True)


@pytest.fixture
def spawned(monkeypatch) -> list:
    """Builds that were started and not run: the capture replaces the thread."""
    out: list = []
    monkeypatch.setattr(familiar_manual, "_spawn", out.append)
    return out


@pytest.fixture
def pin_index(monkeypatch, index):
    monkeypatch.setattr(svc_familiar, "_manual_index", lambda: index)


def _ask(monkeypatch, svc, prompt: str):
    """``ask`` with a router that always says ``manual`` and an answer that cites
    ``[1]``, so the citations are exactly what retrieval handed the model."""

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                        expected_card_sha=None, response_format=None, transport=None):
        return '{"skill": "manual"}' if skill == "router" else "See the Manual [1]."

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)
    return svc_familiar.ask(svc, prompt, mode="home", history=())


def _build_now(svc, index) -> None:
    key = familiar_manual.manual_key(index)
    with familiar_manual._lock:
        familiar_manual._building.add(key)
    familiar_manual._build(svc, index, key)


# --- the fused ranking ------------------------------------------------------


def test_a_section_only_the_meaning_ranking_finds_is_cited_once_the_matrix_is_ready(
    monkeypatch, svc, index, embedder, installed, pin_index, target
):
    assert index.search(ONLY_MEANING) == [], "the query must be invisible to BM25"
    _build_now(svc, index)

    answer = _ask(monkeypatch, svc, ONLY_MEANING)

    chunk = index.chunks[target]
    assert (chunk.chapter, chunk.anchor) in {(c.chapter, c.anchor) for c in answer.citations}
    assert answer.text == "See the Manual [1]."


def test_a_section_both_rankings_like_is_lifted_above_the_one_only_bm25_ranks_first(
    svc, index, embedder, installed
):
    bm25 = [i for i, _ in index.rank(LEXICAL, 40)]
    embedder.target = bm25[3]  # the embedder's favourite is BM25's fourth
    _build_now(svc, index)

    fused = familiar_manual.fused_chunks(svc, index, LEXICAL)

    assert fused is not None
    assert fused[0] == bm25[3], "two votes beat BM25's lone first place"
    assert fused.index(bm25[0]) > 0


def test_with_the_embedder_absent_the_citations_are_exactly_the_bm25_ones(
    monkeypatch, svc, index, pin_index
):
    # No ``installed`` fixture: the rows are really absent from the tmp models dir.
    assert not familiar_manual.installed(svc)

    answer = _ask(monkeypatch, svc, LEXICAL)

    assert answer.citations == (index.search(LEXICAL)[0],)
    assert not (svc.config.home / "cache").exists(), "nothing is written without the row"


def test_an_embedder_that_cannot_answer_falls_back_to_bm25_without_an_error(
    monkeypatch, svc, index, embedder, installed, pin_index
):
    _build_now(svc, index)
    embedder.fail = embed_client.EmbedUnavailable("the embedding server could not be started")

    answer = _ask(monkeypatch, svc, LEXICAL)

    assert answer.citations == (index.search(LEXICAL)[0],)
    assert answer.text == "See the Manual [1]."
    # The query was tried (and failed); the user saw an ordinary answer.
    assert ("query", 1) in embedder.calls


# --- the cache miss ---------------------------------------------------------


def test_a_cache_miss_answers_from_bm25_at_once_and_starts_exactly_one_build(
    monkeypatch, svc, index, embedder, installed, spawned, pin_index
):
    first = _ask(monkeypatch, svc, LEXICAL)
    second = _ask(monkeypatch, svc, "undo")

    assert first.citations == (index.search(LEXICAL)[0],)
    assert second.citations, "BM25 still answers while the build is pending"
    assert len(spawned) == 1, "a second question must not start a second build"
    assert embedder.calls == [], "nothing was embedded on the asking thread"
    assert familiar_manual.build_state(index) == "building"


def test_the_pending_build_is_the_one_that_runs_and_it_lands_the_matrix(
    svc, index, embedder, installed, spawned
):
    assert familiar_manual.matrix(svc, index) is None
    (build,) = spawned
    build()

    built = familiar_manual.matrix(svc, index)
    assert built is not None and built.shape == (len(index.chunks), embed_index.MANUAL_DIM)
    assert {role for role, _ in embedder.calls} == {"document"}
    assert familiar_manual.build_state(index) == "ready"
    assert len(spawned) == 1


def test_a_built_matrix_is_saved_and_the_next_process_loads_it_without_building(
    svc, index, embedder, installed, spawned
):
    _build_now(svc, index)
    key = familiar_manual.manual_key(index)
    saved = svc.config.home / "cache" / f"{key}.npy"
    assert saved.is_file(), "the matrix is written under <home>/cache"

    familiar_manual.reset()  # a new process: nothing in memory
    embedder.calls.clear()
    loaded = familiar_manual.matrix(svc, index)

    assert loaded is not None and loaded.shape[0] == len(index.chunks)
    assert embedder.calls == [] and spawned == []


def test_a_changed_manual_is_a_cache_miss_and_its_build_replaces_the_old_file(
    svc, index, embedder, installed, spawned
):
    _build_now(svc, index)
    old_key = familiar_manual.manual_key(index)

    edited = list(index.chunks)
    edited[0] = dataclasses.replace(edited[0], text=edited[0].text + " One more sentence.")
    changed = dataclasses.replace(index, chunks=edited)
    new_key = familiar_manual.manual_key(changed)
    assert new_key != old_key

    assert familiar_manual.matrix(svc, changed) is None, "the stale file must not be served"
    assert len(spawned) == 1

    # The new text must be embedded, so teach the fake the edited chunk.
    embedder.by_text[(edited[0].title_path, edited[0].text)] = 0
    spawned[0]()
    cache = svc.config.home / "cache"
    assert (cache / f"{new_key}.npy").is_file()
    assert not (cache / f"{old_key}.npy").exists(), "a superseded matrix is pruned after the save"


def test_a_corrupt_cache_file_is_a_miss_that_rebuilds(
    svc, index, embedder, installed, spawned
):
    key = familiar_manual.manual_key(index)
    cache = svc.config.home / "cache"
    cache.mkdir(parents=True)
    (cache / f"{key}.npy").write_bytes(b"not a numpy file at all")

    assert familiar_manual.matrix(svc, index) is None
    assert len(spawned) == 1


def test_a_matrix_with_the_wrong_row_count_is_a_miss(svc, index, embedder, installed, spawned):
    key = familiar_manual.manual_key(index)
    embed_index.save_matrix(
        svc.config.home / "cache", key, np.stack([_unit(0)] * (len(index.chunks) - 1))
    )
    assert familiar_manual.matrix(svc, index) is None
    assert len(spawned) == 1


# --- the build itself -------------------------------------------------------


def test_the_stop_flag_ends_a_build_between_batches_and_saves_nothing(
    svc, index, embedder, installed
):
    embedder.after_call = familiar_manual.stop_builds  # the app quits mid-build
    _build_now(svc, index)

    assert len(embedder.calls) == 1, "one batch ran, then the flag was seen"
    assert not (svc.config.home / "cache").exists()
    assert familiar_manual.build_state(index) == "unavailable"


def test_a_build_whose_loop_has_stopped_ends_between_batches(svc, index, embedder, installed):
    state = {"running": True}
    svc.loop = SimpleNamespace(is_running=lambda: state["running"])
    embedder.after_call = lambda: state.update(running=False)  # Runtime.shutdown

    _build_now(svc, index)

    assert len(embedder.calls) == 1
    assert not (svc.config.home / "cache").exists()


def test_a_build_embeds_title_and_text_pairs_as_documents_in_client_sized_batches(
    svc, index, embedder, installed
):
    _build_now(svc, index)

    assert {role for role, _ in embedder.calls} == {"document"}
    sizes = [n for _, n in embedder.calls]
    assert max(sizes) == familiar_manual.BUILD_BATCH
    assert sum(sizes) == len(index.chunks)


def test_a_failed_build_is_retried_once_per_session_and_then_left_alone(
    monkeypatch, svc, index, embedder, installed, spawned, pin_index
):
    embedder.fail = embed_client.EmbedUnavailable("the embedding server could not be started")

    for expected_builds in (1, 2, 2, 2):
        _ask(monkeypatch, svc, LEXICAL)
        assert len(spawned) == expected_builds
        if spawned and familiar_manual.build_state(index) == "building":
            spawned[-1]()

    assert familiar_manual.build_state(index) == "unavailable"
    assert not (svc.config.home / "cache").exists(), "a failed build writes nothing"


def test_the_cache_lives_under_the_configured_home_never_the_real_one(svc):
    cache = familiar_manual.cache_dir(svc)
    assert cache == svc.config.home / "cache"
    assert Path.home() / ".realmspinner" not in cache.parents


# --- is the embedder installed ---------------------------------------------


def test_the_embedder_counts_as_installed_only_with_its_weights_and_the_runtime(svc):
    assert not familiar_manual.installed(svc)
    for key in ("familiar_runtime", "familiar_embed"):
        spec = models.FAMILIAR_MODELS[key]
        base = fetch.familiar_dir(svc.config, spec)
        base.mkdir(parents=True, exist_ok=True)
        if key == "familiar_embed":
            assert not familiar_manual.installed(svc), "the runtime alone is not enough"
        for name in spec.probe:
            (base / name).write_bytes(b"x")
    assert familiar_manual.installed(svc)


def test_a_service_with_no_config_or_no_embedder_child_is_not_installed():
    assert not familiar_manual.installed(SimpleNamespace(worker=SimpleNamespace(familiar=1)))
    assert not familiar_manual.installed(SimpleNamespace(config=object(), worker=None))
