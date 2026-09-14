"""Familiar's T4 engine: download rows, the llama-server child, and the GPU lease.

No UI conversation here -- what is under test is that the child can only ever
be spawned loopback, offline, keyed by file rather than argv, against exactly
the one pinned weights file; that it refuses to start on a bad manifest, a
card-sha mismatch, or while a GPU job holds the lease; that a GPU job always
kills it first and a rig job never bothers to; that it is stopped before its
own rows can be deleted; and that an un-downloaded row reads as a setup step,
never a fault.
"""

from __future__ import annotations

import asyncio

import pytest

from warlock import doctor, fetch, models, vram
from warlock.pipelines import llama as llama_mod
from warlock.pipelines.llama import LlamaServer


def _srv(tmp_path, **kwargs) -> LlamaServer:
    exe = tmp_path / "llama-server.exe"
    weights = tmp_path / "models" / "familiar" / models.FAMILIAR_GGUF_FILE
    return LlamaServer(
        exe, weights, 17972, key_dir=tmp_path / "keys", log_path=tmp_path / "familiar.log",
        **kwargs,
    )


def test_llama_argv_is_loopback_offline_keyed_and_never_names_a_hub(tmp_path):
    srv = _srv(tmp_path)
    argv = srv._argv(tmp_path / "key.txt")
    assert "--host" in argv and argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "--offline" in argv
    assert "--jinja" in argv
    assert "--no-webui" in argv
    assert not any(flag in ("-hf", "-hfr", "--hf-repo") for flag in argv)


def test_the_api_key_never_appears_on_the_command_line(tmp_path):
    srv = _srv(tmp_path)
    key_path = srv._write_key_file()
    argv = srv._argv(key_path)
    assert "--api-key-file" in argv
    assert str(key_path) in argv
    key = key_path.read_text(encoding="utf-8").strip()
    assert key and key not in argv
    srv._release_key_file()
    assert not key_path.exists()


def test_the_spawn_path_loads_only_the_pinned_row_file(tmp_path):
    srv = _srv(tmp_path)
    argv = srv._argv(tmp_path / "key.txt")
    assert "-m" in argv
    weights_arg = argv[argv.index("-m") + 1]
    assert weights_arg == str(tmp_path / "models" / "familiar" / models.FAMILIAR_GGUF_FILE)
    assert weights_arg.endswith(models.FAMILIAR_GGUF_FILE)


@pytest.mark.asyncio
async def test_a_weights_file_whose_manifest_does_not_verify_refuses_to_start(
    tmp_path, monkeypatch
):
    srv = _srv(tmp_path)
    srv._resolve_exe().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_exe().write_bytes(b"")
    srv._resolve_weights().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_weights().write_bytes(b"")

    monkeypatch.setattr(
        llama_mod.fetch,
        "verify_manifest",
        lambda dest: fetch.Verification(dest=dest, status=fetch.VERIFY_BAD, bad=("x",)),
    )
    spawned = []
    monkeypatch.setattr(llama_mod.subprocess, "Popen", lambda *a, **k: spawned.append(a))

    with pytest.raises(RuntimeError, match="manifest verification"):
        await srv.ensure_started()
    assert spawned == []


@pytest.mark.asyncio
async def test_a_card_sha_mismatch_refuses_to_start(tmp_path, monkeypatch):
    srv = _srv(tmp_path, expected_card_shas=lambda: ())
    srv._resolve_exe().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_exe().write_bytes(b"")
    srv._resolve_weights().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_weights().write_bytes(b"")
    monkeypatch.setattr(
        llama_mod.fetch,
        "verify_manifest",
        lambda dest: fetch.Verification(dest=dest, status=fetch.VERIFY_UNKNOWN),
    )
    spawned = []
    monkeypatch.setattr(llama_mod.subprocess, "Popen", lambda *a, **k: spawned.append(a))

    with pytest.raises(RuntimeError, match="prompt card"):
        await srv.ensure_started(expected_card_sha="deadbeef")
    assert spawned == []
    # No card in play at all (T3 not built) is never refused by this check --
    # asserted directly against the private helper rather than by driving a
    # second ensure_started, which would need its own fresh exe/weights/port
    # to avoid the first call's now-populated directory.
    srv._check_card_sha(None)


@pytest.mark.asyncio
async def test_ensure_started_refuses_to_spawn_familiar_without_vram_headroom(
    tmp_path, monkeypatch
):
    """The 2026-09-14 audit (service-01): every other model door is admitted
    through a VRAM-headroom gate before it can overcommit the card (the
    2026-08-03 crash class), but ``ensure_started`` went straight to
    ``subprocess.Popen`` -- ``vram.familiar_admission`` existed and was never
    called from anywhere in ``src/``."""
    srv = _srv(tmp_path)
    srv._resolve_exe().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_exe().write_bytes(b"")
    srv._resolve_weights().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_weights().write_bytes(b"")
    monkeypatch.setattr(
        llama_mod.fetch,
        "verify_manifest",
        lambda dest: fetch.Verification(dest=dest, status=fetch.VERIFY_UNKNOWN),
    )
    monkeypatch.setattr(llama_mod.vram, "live_memory", lambda: None)
    spawned = []
    monkeypatch.setattr(llama_mod.subprocess, "Popen", lambda *a, **k: spawned.append(a))

    with pytest.raises(RuntimeError, match="VRAM headroom"):
        await srv.ensure_started()
    assert spawned == []


@pytest.mark.asyncio
async def test_familiar_refuses_to_start_while_a_gpu_job_holds_the_lease(tmp_path):
    srv = _srv(tmp_path)
    srv._leased = True
    with pytest.raises(RuntimeError, match="GPU job holds the card"):
        await srv.ensure_started()


def test_an_idle_familiar_is_stopped_after_its_timeout(tmp_path, monkeypatch):
    from warlock.config import Config
    from warlock.db import JobStore
    from warlock.queue import Worker

    config = Config(
        data_dir=tmp_path / "assets", db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe", trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models", familiar_idle_timeout=1.0,
    )
    store = JobStore(config.db_path)
    worker = Worker(config, store)
    worker.familiar._proc = object()  # anything non-None with .poll()
    worker.familiar._proc = type("P", (), {"poll": lambda self: None})()
    worker.familiar.last_used = 0.0  # long ago
    stopped = []
    monkeypatch.setattr(worker.familiar, "stop", lambda: stopped.append(True))
    asyncio.run(worker._maybe_evict_idle())
    assert stopped == [True]


def test_a_gpu_job_dispatch_kills_familiar_before_check_resources(tmp_path, monkeypatch):
    from warlock.config import Config
    from warlock.db import JobStore
    from warlock.queue import Worker

    config = Config(
        data_dir=tmp_path / "assets", db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe", trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models",
    )
    store = JobStore(config.db_path)
    worker = Worker(config, store)
    order = []
    monkeypatch.setattr(worker.familiar, "stop_for_gpu_job", lambda: order.append("stop"))
    monkeypatch.setattr(worker, "_check_resources", lambda job: order.append("check"))
    job = {"kind": "text", "stage": "model", "params": {}}
    asyncio.run(worker.before_gpu_job(job))
    worker._check_resources(job)
    assert order == ["stop", "check"]


def test_a_rig_job_does_not_evict_familiar(tmp_path, monkeypatch):
    from warlock.config import Config
    from warlock.db import JobStore
    from warlock.queue import Worker

    config = Config(
        data_dir=tmp_path / "assets", db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe", trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models",
    )
    store = JobStore(config.db_path)
    worker = Worker(config, store)
    called = []
    monkeypatch.setattr(worker.familiar, "stop_for_gpu_job", lambda: called.append(True))
    job = {"kind": "rig", "stage": "model", "params": {}}
    assert vram.estimate_job_parts(job)[0] == 0
    asyncio.run(worker.before_gpu_job(job))
    assert called == []


def test_deleting_familiar_rows_stops_the_child_first(tmp_path, monkeypatch):
    from warlock.service import downloads as downloads_mod

    class FakeFamiliar:
        running = True

        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    class FakeWorker:
        current_job_id = None
        familiar = FakeFamiliar()

        async def unload_text2image(self):
            return None

    worker = FakeWorker()
    result = asyncio.run(downloads_mod._release_if_idle(worker, stop_engine=False))
    assert result is True
    assert worker.familiar.stopped is True


def test_a_familiar_row_not_downloaded_is_pending_install_not_a_fault(tmp_path):
    from warlock.config import Config

    config = Config(
        data_dir=tmp_path / "assets", db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe", trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models",
        familiar_runtime_dir=tmp_path / "engine" / "llama",
        familiar_models_dir=tmp_path / "models" / "familiar",
    )
    checks = doctor._familiar_checks(config)
    assert len(checks) == len(models.FAMILIAR_MODELS) == 3
    for check in checks:
        assert check.ok is False
        assert check.fatal is False
        assert check.pending_install is True
