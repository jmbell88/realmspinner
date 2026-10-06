"""What only a real ``llama-server.exe`` and the real EmbeddingGemma 2 file can
answer about Familiar's retrieval child: that the second ``LlamaServer`` (the
embed profile) really starts on its own loopback port, that ``--device none``
really keeps it off the card, and that ``/v1/embeddings`` really returns a
768-d unit-norm vector for a prefixed query.

The VRAM claim is the reason this file exists. ``-ngl 0`` is *not* CPU-only:
measured 2026-10-06 (``dev/measurements/2026-10-06-familiar-gemma4-12b.md``) it
still took +1,458 MiB of the card, which the GPU lease would then have had to
evict, against +17 MiB with ``--device none``. A profile that quietly lost that
flag would pass every argv-shaped unit test in ``test_familiar_embed.py`` that
did not name it, and only a card-wide reading proves what the child costs.

Run with: uv run pytest tests/familiar/test_embed_gpu.py -m gpu -n 0

The runtime and weights come from ``Config()`` -- whatever
``REALMSPINNER_FAMILIAR_RUNTIME`` and ``REALMSPINNER_FAMILIAR_MODELS`` name --
and *not* from ``get_config()``, which would run the home migration and create
directories under ``~/.realmspinner``. Key, owner and log files go to a
throwaway directory. Skips (not fails) when the retrieval row or the runtime is
not on disk.
"""

from __future__ import annotations

import asyncio
import math
import socket
import threading
import time

import httpx
import pytest

from realmspinner import fetch, models, vram
from realmspinner.config import Config
from realmspinner.pipelines.llama import EMBED_PROFILE, LlamaServer

pytestmark = [pytest.mark.gpu, pytest.mark.timeout(600)]

#: The model card's text-only query prompt, the one the 910-chunk measurement
#: used.
_QUERY = "task: search result | query: how do I export a model as a GLB"

#: The ceiling the retrieval child must stay under. Measured +17 MiB; the chat
#: child is ~9 GiB and ``-ngl 0`` alone was 1,458 MiB, so 100 MiB separates
#: "not on the card" from both without sitting on the noise of a desktop GPU.
_MAX_DELTA_MIB = 100.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _used_mib() -> float:
    mem = vram.live_memory()
    assert mem is not None, "NVML unavailable on this host -- nothing to measure against"
    return (mem.total_gib - mem.free_gib) * 1024.0


def _sample(stop: threading.Event, samples: list[float]) -> None:
    while not stop.is_set():
        mem = vram.live_memory()
        if mem is not None:
            samples.append((mem.total_gib - mem.free_gib) * 1024.0)
        time.sleep(0.1)


def test_the_embed_child_stays_off_the_card_and_returns_a_unit_norm_768d_vector(tmp_path):
    config = Config()
    spec = models.FAMILIAR_MODELS["familiar_embed"]
    for runtime in (models.FAMILIAR_MODELS["familiar_runtime"], spec):
        if not fetch.present(config, "familiar", runtime):
            pytest.skip(f"{runtime.label} not downloaded")

    srv = LlamaServer(
        lambda: config.familiar_runtime_dir / "llama-server.exe",
        lambda: config.familiar_models_dir / models.FAMILIAR_EMBED_GGUF_FILE,
        _free_port(),
        key_dir=tmp_path / "keys",
        log_path=tmp_path / EMBED_PROFILE.log_name,
        idle_timeout=3600.0,
        profile=EMBED_PROFILE,
    )

    before = _used_mib()
    samples: list[float] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_sample, args=(stop, samples), daemon=True)
    sampler.start()
    try:
        asyncio.run(srv.ensure_started())
        assert srv.running
        key = srv.key_path.read_text(encoding="utf-8").strip()
        with httpx.Client(trust_env=False, timeout=120.0) as client:
            reply = client.post(
                f"{srv.base_url}/v1/embeddings",
                headers={"Authorization": f"Bearer {key}"},
                json={"input": _QUERY, "model": "familiar-embed"},
            )
        reply.raise_for_status()
        vector = reply.json()["data"][0]["embedding"]
        time.sleep(0.5)  # let the sampler see the post-request steady state
    finally:
        stop.set()
        sampler.join(timeout=5)
        srv.stop()

    peak = max(samples) if samples else before
    delta = peak - before
    norm = math.sqrt(sum(x * x for x in vector))
    print(f"EMBED-GPU: vram_before_mib={before:.0f} peak_mib={peak:.0f} delta_mib={delta:.0f}")
    print(f"EMBED-GPU: dim={len(vector)} norm={norm:.6f}")

    assert len(vector) == models.FAMILIAR_EMBED_DIM == 768
    assert norm == pytest.approx(1.0, abs=1e-3)
    assert delta < _MAX_DELTA_MIB, (
        f"the embed child took {delta:.0f} MiB of the card (limit {_MAX_DELTA_MIB:.0f}): "
        "--device none is the CPU switch, -ngl 0 alone took 1,458 MiB when measured"
    )
    assert srv.key_path is None, "stop() must remove the per-spawn key file"
