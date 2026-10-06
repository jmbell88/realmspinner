"""Does meaning-based retrieval earn its keep on the real Manual? The gate the
retrieval design was accepted against (2026-10-06): the fused BM25 + dense
ranking must beat BM25 alone, on a fixed question set, with the real
EmbeddingGemma 2 child embedding the real Manual.

``data/manual_questions.json`` is 80 questions the stock Gemma 4 12B wrote from
sampled Manual passages (one short question a reader who had never seen the
chapter might type), each with the chapter and section its passage came from.
The set is *synthetic*: paraphrased questions favour a meaning-based ranker over
a keyword one, so the margin below is the margin on that set, not on what
people type. A question whose section heading no longer exists in the Manual
(the Manual is edited constantly) is dropped rather than failed, and the test
refuses to pass on fewer than 60 survivors, so the set has to be regenerated
before it quietly stops measuring anything.

Measured when the set was made (2026-10-06, 80 questions, top 5, section level):
BM25 0.512, dense 0.700, fused 0.637; section MRR 0.333 / 0.579 / 0.479. Dense
alone beat the fusion on this set; the fusion is kept because BM25 is what finds
an exact identifier or environment-variable name, which a paraphrased set does
not ask for.

Run with: uv run pytest tests/familiar/test_manual_hitrate_gpu.py -m gpu -n 0
Takes ~3-4 minutes (the whole Manual is embedded on the CPU). Skips when the
runtime or the retrieval row is not on disk.
"""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path

import pytest

from realmspinner import fetch, models
from realmspinner.config import Config
from realmspinner.familiar import embed_client, embed_index, retrieval
from realmspinner.pipelines.llama import EMBED_PROFILE, LlamaServer

pytestmark = [pytest.mark.gpu, pytest.mark.timeout(1200)]

_QUESTIONS = Path(__file__).parent / "data" / "manual_questions.json"
_K = 5
_MIN_QUESTIONS = 60


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_fused_ranking_beats_bm25_alone_on_the_fixed_manual_questions(tmp_path):
    config = Config()
    spec = models.FAMILIAR_MODELS["familiar_embed"]
    for row in (models.FAMILIAR_MODELS["familiar_runtime"], spec):
        if not fetch.present(config, "familiar", row):
            pytest.skip(f"{row.label} not downloaded")

    index = retrieval.Index.build()
    chunks = index.chunks
    sections = {(c.chapter, c.title_path) for c in chunks}
    questions = [
        q
        for q in json.loads(_QUESTIONS.read_text(encoding="utf-8"))
        if (q["chapter"], q["section"]) in sections
    ]
    assert len(questions) >= _MIN_QUESTIONS, (
        f"only {len(questions)} of the fixed questions still name a section that exists: "
        "regenerate tests/familiar/data/manual_questions.json"
    )

    srv = LlamaServer(
        lambda: config.familiar_runtime_dir / "llama-server.exe",
        lambda: config.familiar_models_dir / models.FAMILIAR_EMBED_GGUF_FILE,
        _free_port(),
        key_dir=tmp_path / "keys",
        log_path=tmp_path / EMBED_PROFILE.log_name,
        idle_timeout=3600.0,
        profile=EMBED_PROFILE,
    )
    try:
        docs = [(c.title_path, c.text) for c in chunks]
        matrix = asyncio.run(embed_client.embed(srv, docs, role="document"))
        qvecs = asyncio.run(embed_client.embed(srv, [q["q"] for q in questions], role="query"))
    finally:
        srv.stop()

    def section_hit(ranked: list[int], q: dict) -> bool:
        return any((chunks[i].chapter, chunks[i].title_path) == (q["chapter"], q["section"])
                   for i in ranked[:_K])

    bm25_hits = dense_hits = fused_hits = 0
    for q, qv in zip(questions, qvecs, strict=True):
        bm = [i for i, _ in index.rank(q["q"], 50)]
        de = [i for i, _ in embed_index.top_k(qv, matrix, 50)]
        fu = [i for i, _ in embed_index.rrf([bm, de], limit=50)]
        bm25_hits += section_hit(bm, q)
        dense_hits += section_hit(de, q)
        fused_hits += section_hit(fu, q)

    n = len(questions)
    print(
        f"MANUAL-HITRATE: n={n} section@{_K} bm25={bm25_hits / n:.3f} "
        f"dense={dense_hits / n:.3f} fused={fused_hits / n:.3f}"
    )
    assert fused_hits > bm25_hits, (
        f"the fusion found the right section {fused_hits}/{n} times, BM25 alone {bm25_hits}/{n}: "
        "meaning-based retrieval is not earning its place"
    )
