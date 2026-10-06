"""A stand-in for the retrieval child, for the tests of the Library's meaning index.

Never the real ``llama-server``: the vectors come from a tiny concept table, so
"blade" lands next to "a rusty iron longsword" without sharing a character of it,
and everything else is seeded noise. Patching goes through ``embed_client.embed``
(the one function ``service.library_index`` calls), so the whole path from a job
row to a stored vector and back to a ranked id runs for real except the HTTP hop.
"""

from __future__ import annotations

import re
import threading
import time
import zlib
from types import SimpleNamespace
from typing import Any

import numpy as np

from realmspinner.familiar import embed_client, embed_index
from realmspinner.service import library_index

#: concept -> the words that mean it. A text containing any of them carries that
#: concept's direction; nothing else in the fake knows what a word means.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "weapon": ("sword", "longsword", "blade", "dagger", "axe", "weapon"),
    "container": ("chest", "barrel", "crate", "box", "container"),
    "creature": ("dragon", "wolf", "monster", "creature"),
    "building": ("cottage", "castle", "house", "tower", "building"),
}


def _seeded(label: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(label.encode()))
    v = rng.standard_normal(embed_index.NATIVE_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def vector_for(text: str) -> np.ndarray:
    """The fake's 768-d unit vector for *text*."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    v = 0.25 * _seeded("noise:" + text)
    for concept, vocab in CONCEPTS.items():
        if words & set(vocab):
            v = v + _seeded("concept:" + concept)
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeEmbedServer:
    """The duck-typed ``LlamaServer`` surface ``embed_client`` touches."""

    def __init__(self) -> None:
        self.running = True
        self.key_path = None
        self.base_url = "http://127.0.0.1:0"
        self.started = 0

    async def ensure_started(self) -> None:
        self.started += 1
        self.running = True

    def touch(self) -> None:
        pass


class FakeEmbedder:
    """Records what was asked of the embedder and answers from :func:`vector_for`."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = False
        self.server = FakeEmbedServer()

    async def embed(self, server, texts, *, role, transport=None):
        self.calls.append(
            {
                "role": role,
                "texts": list(texts),
                "thread": threading.current_thread().name,
            }
        )
        if self.fail:
            raise embed_client.EmbedUnavailable("the embedding server could not be started")
        rows = []
        for item in texts:
            rows.append(vector_for(" ".join(item) if isinstance(item, tuple) else item))
        return np.stack(rows).astype(np.float32) if rows else np.empty((0, 768), np.float32)

    @property
    def document_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["role"] == "document"]

    @property
    def query_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["role"] == "query"]


def install(monkeypatch, svc) -> FakeEmbedder:
    """Give *svc* a worker whose embedder is the fake, make ``installed`` say
    yes, and route ``embed_client.embed`` to it. Returns the recorder."""
    fake = FakeEmbedder()
    svc.worker = SimpleNamespace(
        familiar_embed=fake.server,
        progress=SimpleNamespace(snapshot=lambda _job_id: None),
        current_job_id=None,
        wake=lambda: None,
    )
    monkeypatch.setattr(embed_client, "embed", fake.embed)
    monkeypatch.setattr(library_index, "installed", lambda _svc: True)
    return fake


def wait_done(runner, until=bool, timeout: float = 10.0) -> list[Any]:
    """Poll a real ``TaskRunner`` until ``until(done_so_far)`` holds, sleeping
    between empty polls so the worker thread gets the interpreter."""
    done: list[Any] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not until(done):
        got = runner.poll()
        done += got
        if not got:
            time.sleep(0.005)
    return done
