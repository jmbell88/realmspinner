"""The 2026-10-14 audit, fixer F04: the publish journal's backup tree, the
palette suffix the worker re-resolves, the music job's image-pipe credit, a quit
during a t2i stage, and the NVML first-init race.

One file at the top of ``tests/`` rather than one per mirror directory:
``tests/service`` carries no ``__init__.py``, so a second file sharing this
basename would collide under pytest's default import mode.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from realmspinner import publish, vram
from realmspinner import queue as queue_mod
from realmspinner.config import Config
from realmspinner.db import JobStore
from realmspinner.service import downloads, palettes
from realmspinner.service.errors import Invalid

# --- service-assets-01 / -02: the journal protects its backup tree and its own marker


def test_the_sweep_spares_the_backup_tree_an_open_journal_needs(svc):
    """A recovery that failed on a locked file keeps its journal so a later run
    can put the *original* files back from ``.fetch.bak``. Both sweeps matched
    that suffix and deleted the only copy of the original weights."""
    from realmspinner import fetch

    root = svc.config.t2i_model_root
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "sdxl-base-1.0"
    staging = root / ".sdxl-base-1.0.fetch.part"
    staging.mkdir()
    (staging / "new.bin").write_bytes(b"new")
    backup = publish.backup_dir(dest, staging)
    backup.mkdir()
    (backup / "original.bin").write_bytes(b"original")
    publish.begin(
        root,
        [{"repo_id": "a/b", "staging": str(staging), "dest": str(dest), "published": ["new.bin"]}],
    )

    assert str(backup) in publish.staged_dirs(root)

    job = fetch.Job(repo_id="a/b", dest=dest)
    downloads._sweep_staging([job], spare=publish.staged_dirs(root))
    assert (backup / "original.bin").exists()

    # The Models pane's own sweep takes the same journal roots.
    downloads.sweep_staging(svc)
    assert (backup / "original.bin").exists()
    assert (staging / "new.bin").exists()

    # With the journal gone both are litter again, which is still right.
    publish.finish(root)
    downloads._sweep_staging([job], spare=publish.staged_dirs(root))
    assert not backup.exists()
    assert not staging.exists()


def test_a_download_refuses_to_begin_over_a_journal_a_failed_recovery_retained(
    svc, monkeypatch
):
    """``publish.begin`` overwrites the journal and ``finish`` then deletes it, so
    pressing Install over a journal an incomplete recovery retained ended the
    only retry for a half-published model. The download refuses by name, before
    it spends a minute of bandwidth, and leaves the journal as it found it."""
    root = Path(svc.config.home)
    root.mkdir(parents=True, exist_ok=True)
    old = {
        "staging": str(root / ".old.fetch.part"),
        "dest": str(root / "old"),
        "published": ["w.bin"],
    }
    publish.begin(root, [old])
    # The recovery cannot finish: a file at the destination is locked.
    monkeypatch.setattr(publish, "undo_into", lambda *_a, **_k: False)

    def _no_download(job, **_kw):  # pragma: no cover - must not be reached
        raise AssertionError("the download ran over a retained journal")

    monkeypatch.setattr(downloads, "_run_worker", _no_download)
    with pytest.raises(Invalid) as caught:
        downloads.download(svc, ["base:sdxl_cfg"])

    assert "interrupted" in caught.value.message
    kept = publish.read_json(publish.journal_path(root))
    assert kept is not None
    assert [entry["dest"] for entry in kept["entries"]] == [old["dest"]]


# --- service-queue-04: the worker resolves every suffix the service loads --------

_SAMPLES = {
    ".hex": "FF0000\n00FF00\n",
    ".gpl": "GIMP Palette\n255 0 0 red\n0 255 0 green\n",
    ".pal": "JASC-PAL\n0100\n2\n255 0 0\n0 255 0\n",
    ".txt": "FFFF0000\nFF00FF00\n",
}


def test_the_worker_resolves_every_palette_suffix_the_service_loads(svc, tmp_path):
    directory = tmp_path / "palettes"
    directory.mkdir(exist_ok=True)
    svc.config.palette_dir = directory
    assert set(_SAMPLES) == set(palettes.SUFFIXES)
    for suffix, text in _SAMPLES.items():
        (directory / f"pal{suffix[1:]}{suffix}").write_text(text, "utf-8")
    for suffix in palettes.SUFFIXES:
        name = f"pal{suffix[1:]}"
        service_colours = palettes.load(svc.config, name)[1]
        assert len(service_colours) == 2
        assert queue_mod._palette_entries(svc.config, name) == service_colours, suffix


# --- service-queue-01: a music job is not credited for an image pipe ------------


def _music_worker(tmp_path, monkeypatch) -> queue_mod.Worker:
    monkeypatch.setattr(queue_mod, "commit_fraction", lambda: None)
    monkeypatch.setattr(queue_mod, "vram_gib", lambda: (0.0, 0.0))
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )
    worker = queue_mod.Worker(config, JobStore(config.db_path))
    worker.trellis = SimpleNamespace(running=True, last_used=0.0)
    return worker


def test_a_music_job_is_not_credited_for_a_resident_image_pipe_it_does_not_reuse(
    tmp_path, fake_pipelines, monkeypatch
):
    monkeypatch.setattr(vram, "device_memory", lambda: vram.DeviceMemory(32.0, 4.0))
    worker = _music_worker(tmp_path, monkeypatch)
    try:
        job = {"kind": "music", "stage": "music", "params": {}}
        # With nothing resident the job is refused: 4 GiB free plus a trellis
        # credit is not enough for the music checkpoint.
        worker._text2image = None
        with pytest.raises(RuntimeError):
            worker._check_resources(job)
        # A trimmed SDXL pipe left loaded by a failed text job is memory the
        # music job neither reuses nor evicts: the answer must not change.
        worker._text2image = SimpleNamespace(loaded=True)
        worker._t2i_key = "sdxl_cfg"
        with pytest.raises(RuntimeError):
            worker._check_resources(job)
    finally:
        worker.store.close()


# --- service-queue-06: a quit during a t2i stage is a cancel, not an error ------


class _KillFirstClient:
    """Shaped like ``t2i_client``'s documented ``close``: kill the child, then
    wait for the in-flight request to release its lock."""

    loaded = True
    last_used = 0.0

    def __init__(self) -> None:
        self.killed = threading.Event()

    def generate(self) -> None:
        self.killed.wait(10)
        raise RuntimeError("ChildFailed: the image worker exited")

    def close(self) -> None:
        self.killed.set()
        time.sleep(0.2)

    def unload(self) -> None:
        pass

    def trim(self) -> None:
        pass


async def test_quitting_during_a_t2i_stage_records_the_job_cancelled_and_leaves_the_sweep_alone(
    tmp_path, fake_pipelines, monkeypatch
):
    monkeypatch.setattr(queue_mod, "commit_fraction", lambda: None)
    monkeypatch.setattr(vram, "device_memory", lambda: None)
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )
    store = JobStore(config.db_path)
    try:
        sweep = store.create_sweep("s", "a chest", {})
        running = store.create("text", "a chest", {"seed": 1}, sweep_id=sweep, sweep_unit="u1")
        worker = queue_mod.Worker(config, store)
        client = _KillFirstClient()
        worker._text2image = client
        failed: list[str] = []
        worker.on_job_failed = lambda job: failed.append(job["id"])

        async def fake_generate(job):
            await asyncio.to_thread(client.generate)

        worker._generate = fake_generate
        worker.start()
        for _ in range(200):
            if store.get(running)["status"] == "running":
                break
            await asyncio.sleep(0.05)
        assert store.get(running)["status"] == "running"
        await worker.shutdown()
        row = store.get(running)
        assert row["status"] == "cancelled", row["error"]
        assert failed == []
    finally:
        store.close()


# --- service-queue-02: the NVML first-init race ---------------------------------


def test_a_thread_racing_the_first_nvml_init_gets_the_session_not_the_negative_cache(
    monkeypatch,
):
    monkeypatch.setattr(vram, "_NVML_SESSION", vram._UNSET)
    monkeypatch.setattr(vram.sys, "platform", "win32")

    class _FakeNvml:
        def nvmlInit_v2(self):
            time.sleep(0.3)
            return 0

        def nvmlDeviceGetHandleByIndex_v2(self, idx, ref):
            return 0

        def nvmlDeviceGetName(self, handle, buf, n):
            buf.value = b"Fake RTX"
            return 0

    monkeypatch.setattr(vram.ctypes, "CDLL", lambda name: _FakeNvml())

    results: dict[str, object] = {}

    def call(tag: str) -> None:
        results[tag] = vram._nvml()

    first = threading.Thread(target=call, args=("first",))
    second = threading.Thread(target=call, args=("second",))
    first.start()
    time.sleep(0.05)  # the second thread arrives while the first is inside nvmlInit_v2
    second.start()
    first.join(10)
    second.join(10)

    assert results["first"] is not None
    assert results["second"] is not None, "told there is no NVIDIA driver mid-init"
    assert results["second"] is results["first"]
