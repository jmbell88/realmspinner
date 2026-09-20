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
import os
import threading

import pytest

from realmspinner import doctor, fetch, models, vram
from realmspinner.pipelines import llama as llama_mod
from realmspinner.pipelines.llama import LlamaServer


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


def test_llama_argv_turns_the_models_reasoning_off_on_the_server(tmp_path):
    """Base Gemma 4 kept reasoning even with enable_thinking=false on the
    request, and a reasoning channel eats the whole max_tokens of a
    schema-constrained reply (content='' on the 2026-09-14 real-card run).
    The server must be started with reasoning off and a zero budget."""
    argv = _srv(tmp_path)._argv(tmp_path / "key.txt")
    assert "--reasoning" in argv and argv[argv.index("--reasoning") + 1] == "off"
    assert "--reasoning-budget" in argv and argv[argv.index("--reasoning-budget") + 1] == "0"


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


def test_a_row_served_name_reaches_llama_server_as_its_alias(tmp_path):
    srv = _srv(tmp_path, served_name=lambda: models.FAMILIAR_V1_NAME)
    argv = srv._argv(tmp_path / "key.txt")
    assert "--alias" in argv
    assert argv[argv.index("--alias") + 1] == models.FAMILIAR_V1_NAME


def test_the_testing_pin_is_never_served_as_familiar_v1(tmp_path):
    # The testing row itself must not carry T10's name...
    assert models.FAMILIAR_MODELS["familiar_gguf"].served_name != models.FAMILIAR_V1_NAME
    # ...and the argv built from that real row must pass no --alias at all,
    # so llama-server falls back to whatever general.name the GGUF carries
    # rather than reporting our model's name for weights that are not ours.
    srv = _srv(
        tmp_path,
        served_name=lambda: models.FAMILIAR_MODELS["familiar_gguf"].served_name,
    )
    argv = srv._argv(tmp_path / "key.txt")
    assert "--alias" not in argv


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
async def test_ensure_started_runs_manifest_verification_off_the_loop_thread(
    tmp_path, monkeypatch
):
    """The 2026-09-18 audit (familiar-01): ``_check_manifest`` SHA-256-hashes
    the runtime and weights directories (about 4.9 GB) synchronously inside
    ``ensure_started``, on the ``realmspinner-loop`` thread -- ``before_gpu_job``
    stops Familiar ahead of every GPU job, so a chat message right after
    generating an asset stalled job dispatch, progress and cancel for as
    long as the hash took (263 ms for a 315 MB directory alone, the
    2026-09-18 probe familiar-engine-02.py). Proved by recording which
    thread ``fetch.verify_manifest`` actually runs on."""
    srv = _srv(tmp_path)
    srv._resolve_exe().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_exe().write_bytes(b"")
    srv._resolve_weights().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_weights().write_bytes(b"")

    calling_thread_ids: list[int] = []

    def _fake_verify(dest):
        calling_thread_ids.append(threading.get_ident())
        return fetch.Verification(dest=dest, status=fetch.VERIFY_BAD, bad=("x",))

    monkeypatch.setattr(llama_mod.fetch, "verify_manifest", _fake_verify)

    loop_thread_id = threading.get_ident()
    with pytest.raises(RuntimeError, match="manifest verification"):
        await srv.ensure_started()

    assert calling_thread_ids, "verify_manifest was never called"
    assert loop_thread_id not in calling_thread_ids, (
        "manifest verification ran on the calling/loop thread instead of "
        "being offloaded with asyncio.to_thread"
    )


def test_ensure_started_does_not_rehash_an_unchanged_manifest(tmp_path, monkeypatch):
    """Part of the familiar-01 fix: offloading the hash to a thread makes it
    non-blocking but does not make it free, and ``ensure_started`` runs on
    every restart because ``before_gpu_job`` stops Familiar before every GPU
    job. The fingerprint cache in ``_verify_manifest_cached`` must skip the
    re-hash entirely when the manifest file's mtime/size have not moved
    since the last check."""
    srv = _srv(tmp_path)
    srv._resolve_exe().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_exe().write_bytes(b"")
    srv._resolve_weights().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_weights().write_bytes(b"")

    call_count = 0

    def _fake_verify(dest):
        nonlocal call_count
        call_count += 1
        return fetch.Verification(dest=dest, status=fetch.VERIFY_UNKNOWN)

    monkeypatch.setattr(llama_mod.fetch, "verify_manifest", _fake_verify)

    # _check_manifest walks two directories (exe parent, weights parent),
    # so a cold call verifies both once each...
    srv._check_manifest()
    assert call_count == 2
    # ...and a repeat with nothing on disk touched must not re-hash either.
    srv._check_manifest()
    assert call_count == 2


def test_a_repeated_check_rehashes_when_a_payload_file_changes_without_touching_the_manifest(
    tmp_path, monkeypatch
):
    """The fingerprint cache first keyed only on ``manifest.json``'s own
    mtime/size -- a weights or runtime file corrupted, overwritten by hand,
    or left half-written by a partial copy after the first verification
    never touches ``manifest.json``, so that fingerprint stayed unchanged
    and the stale ``VERIFY_OK``/``VERIFY_UNKNOWN`` verdict was trusted for
    the rest of the process. That over-trusts a directory that has
    genuinely changed, contradicting ``_manifest_fingerprint``'s own "can
    only ever under-trust" claim. The fingerprint must cover every regular
    file under the directory, not just the manifest, so a changed payload
    file forces a re-verify even though the manifest itself never moved."""
    srv = _srv(tmp_path)
    srv._resolve_exe().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_exe().write_bytes(b"")
    weights_dir = srv._resolve_weights().parent
    weights_dir.mkdir(parents=True, exist_ok=True)
    srv._resolve_weights().write_bytes(b"good weight bytes")
    # A real install's manifest.json, present but never touched by the
    # corruption below -- exactly the case a manifest-only fingerprint
    # cannot see.
    fetch.manifest_path(weights_dir).write_text("{}", encoding="utf-8")

    call_count = 0

    def _fake_verify(dest):
        nonlocal call_count
        call_count += 1
        return fetch.Verification(dest=dest, status=fetch.VERIFY_UNKNOWN)

    monkeypatch.setattr(llama_mod.fetch, "verify_manifest", _fake_verify)

    srv._check_manifest()
    assert call_count == 2

    # Corrupt the weights payload only -- new bytes, a fresh mtime -- and
    # leave manifest.json exactly as it was.
    weights = srv._resolve_weights()
    before = weights.stat()
    weights.write_bytes(b"corrupted -- different size and a bumped mtime")
    os.utime(weights, ns=(before.st_mtime_ns + 5_000_000_000, before.st_mtime_ns + 5_000_000_000))

    srv._check_manifest()
    assert call_count == 4, (
        "a changed payload file must force a re-verify even though "
        "manifest.json itself never changed"
    )


def test_a_transient_fingerprint_failure_does_not_poison_the_cache_with_none(
    tmp_path, monkeypatch
):
    """The 2026-09-20 audit (familiar-04): ``_manifest_fingerprint`` returns
    ``None`` on any ``OSError`` (a file briefly locked, a race with an
    in-progress copy), and ``_verify_manifest_cached`` used to store that
    ``None`` verbatim as the cached fingerprint. A ``None`` fingerprint can
    never equal a real one, so that poisoned the comparison twice over: the
    call that hit the transient failure always had to re-hash (unavoidable
    -- there is nothing to compare a ``None`` fingerprint against), but the
    *next*, fully healthy call also re-hashed, because the freshly computed
    real fingerprint could never equal the ``None`` now sitting in the
    cache. One hiccup cost two several-hundred-millisecond-to-second stalls
    against Familiar's ~4.9 GB runtime+weights pair instead of one. The fix
    only ever overwrites the cache with a real fingerprint, keeping the
    last-known-good one across a transient failure so the very next healthy
    call can still hit the cache."""
    srv = _srv(tmp_path)
    srv._resolve_exe().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_exe().write_bytes(b"")
    dest = srv._resolve_exe().parent

    call_count = 0

    def _fake_verify(d):
        nonlocal call_count
        call_count += 1
        return fetch.Verification(dest=d, status=fetch.VERIFY_UNKNOWN)

    monkeypatch.setattr(llama_mod.fetch, "verify_manifest", _fake_verify)

    real_fingerprint = LlamaServer._manifest_fingerprint(dest)
    # First call sees the real fingerprint (populating the cache), second
    # call simulates the transient OSError, third call is back to normal --
    # the same directory, genuinely unchanged, on either side of the hiccup.
    fingerprints = iter([real_fingerprint, None, real_fingerprint])
    monkeypatch.setattr(srv, "_manifest_fingerprint", lambda d: next(fingerprints))

    srv._verify_manifest_cached(dest)
    assert call_count == 1

    # The transient failure: nothing to compare against, so this re-hash is
    # unavoidable...
    srv._verify_manifest_cached(dest)
    assert call_count == 2

    # ...but the directory never actually changed, so the very next call
    # (back to the same real fingerprint as before the hiccup) must hit the
    # cache rather than pay for a second unnecessary re-hash.
    srv._verify_manifest_cached(dest)
    assert call_count == 2, (
        "a transient fingerprint failure must not poison the cache with "
        "None -- the next healthy call should still hit the last-known-"
        "good fingerprint"
    )


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


@pytest.mark.asyncio
async def test_a_gpu_lease_taken_while_familiar_is_still_spawning_still_prevents_the_spawn(
    tmp_path, monkeypatch
):
    """The 2026-09-17 audit (familiar-02): ``ensure_started`` read ``_leased``
    once at the top and never again, so a lease taken on another thread after
    that read but before ``subprocess.Popen`` -- while the server was not yet
    ``running``, so ``stop_for_gpu_job`` had nothing to kill and returned at
    once -- went unnoticed, and the spawn went ahead anyway: ``llama-server
    -ngl 999`` started beside the GPU job that had just taken the card, the
    overcommit the lease exists to prevent (the 2026-08-03 crash class).

    Drives a real second thread through ``stop_for_gpu_job`` from inside a
    monkeypatched ``_check_vram``, which lands it deterministically in that
    window -- no sleeps, no guessing at timing.
    """
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
    monkeypatch.setattr(llama_mod, "_port_in_use", lambda port: False)
    monkeypatch.setattr(llama_mod.winjob, "assign", lambda pid: None)
    monkeypatch.setattr(llama_mod.winjob, "track", lambda pid, name: None)
    monkeypatch.setattr(srv, "_claim_port", lambda pid: None)
    monkeypatch.setattr(srv, "_pump", lambda: None)
    fake_proc = type("P", (), {"pid": 4242, "returncode": None, "poll": lambda self: None})()
    monkeypatch.setattr(llama_mod.subprocess, "Popen", lambda *a, **k: fake_proc)

    def _lease_mid_start() -> None:
        # ``_check_vram`` runs after the early ``_leased`` check and before
        # the spawn -- exactly the window familiar-02 found unguarded.
        thread = threading.Thread(target=srv.stop_for_gpu_job)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive(), "stop_for_gpu_job did not return"

    monkeypatch.setattr(srv, "_check_vram", _lease_mid_start)

    import httpx

    async def fake_get(self, url, **kwargs):
        return httpx.Response(200)

    monkeypatch.setattr(llama_mod.httpx.AsyncClient, "get", fake_get)

    with pytest.raises(RuntimeError, match="GPU job holds the card"):
        await srv.ensure_started()

    assert srv._proc is None, "the lease landed mid-start but the child spawned anyway"
    assert not srv.running


def test_an_idle_familiar_is_stopped_after_its_timeout(tmp_path, monkeypatch):
    from realmspinner.config import Config
    from realmspinner.db import JobStore
    from realmspinner.queue import Worker

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
    from realmspinner.config import Config
    from realmspinner.db import JobStore
    from realmspinner.queue import Worker

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
    from realmspinner.config import Config
    from realmspinner.db import JobStore
    from realmspinner.queue import Worker

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
    from realmspinner.service import downloads as downloads_mod

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


def test_touch_resets_the_idle_clock_so_a_live_conversation_is_not_evicted(tmp_path, monkeypatch):
    """No client exists yet (T5's ``llama_client`` is unbuilt), so nothing
    calls ``touch()`` -- this proves the door itself works: without it a
    long-running conversation's server would be evicted mid-reply because
    ``last_used`` is otherwise only written at spawn and on the health poll."""
    from realmspinner.config import Config
    from realmspinner.db import JobStore
    from realmspinner.queue import Worker

    config = Config(
        data_dir=tmp_path / "assets", db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe", trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models", familiar_idle_timeout=1.0,
    )
    store = JobStore(config.db_path)
    worker = Worker(config, store)
    worker.familiar._proc = type("P", (), {"poll": lambda self: None})()
    worker.familiar.last_used = 0.0  # long ago -- would be evicted without touch()
    stopped = []
    monkeypatch.setattr(worker.familiar, "stop", lambda: stopped.append(True))
    worker.familiar.touch()
    asyncio.run(worker._maybe_evict_idle())
    assert stopped == []


@pytest.mark.asyncio
async def test_a_cold_start_still_loading_its_weights_is_not_evicted_as_idle(
    tmp_path, monkeypatch
):
    """The first real run in the app, 2026-09-14: llama-server spawned at
    21:18:10.652, answered /health with 503 while it loaded the Q8_0 weights,
    and ``Worker._maybe_evict_idle`` stopped it at 21:18:13.399. ``running`` is
    true from the moment the child exists, but ``last_used`` was only written
    once /health answered 200 -- so it still held the constructor's 0.0 and a
    server mid-load read as idle since boot. The idle sweep here runs at
    exactly that moment, between a 503 and the 200 that would follow."""
    import httpx

    from realmspinner.config import Config
    from realmspinner.db import JobStore
    from realmspinner.queue import Worker

    srv = _srv(tmp_path, idle_timeout=300.0)
    srv._resolve_exe().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_exe().write_bytes(b"")
    srv._resolve_weights().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_weights().write_bytes(b"")
    monkeypatch.setattr(
        llama_mod.fetch,
        "verify_manifest",
        lambda dest: fetch.Verification(dest=dest, status=fetch.VERIFY_UNKNOWN),
    )
    monkeypatch.setattr(srv, "_check_vram", lambda: None)
    monkeypatch.setattr(llama_mod, "_port_in_use", lambda port: False)
    monkeypatch.setattr(llama_mod.winjob, "assign", lambda pid: None)
    monkeypatch.setattr(llama_mod.winjob, "track", lambda pid, name: None)
    monkeypatch.setattr(srv, "_claim_port", lambda pid: None)
    monkeypatch.setattr(srv, "_pump", lambda: None)
    fake_proc = type("P", (), {"pid": 4242, "returncode": None, "poll": lambda self: None})()
    monkeypatch.setattr(llama_mod.subprocess, "Popen", lambda *a, **k: fake_proc)

    config = Config(
        data_dir=tmp_path / "assets", db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe", trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models",
    )
    worker = Worker(config, JobStore(config.db_path))
    worker.familiar = srv
    stopped = []
    monkeypatch.setattr(srv, "stop", lambda: stopped.append(True))

    health_calls = []

    async def fake_get(self, url, **kwargs):
        health_calls.append(url)
        if len(health_calls) == 1:
            # Still loading: the sweep runs now, the way the worker's own
            # idle tick did in the app.
            await worker._maybe_evict_idle()
            return httpx.Response(503)
        return httpx.Response(200)

    monkeypatch.setattr(llama_mod.httpx.AsyncClient, "get", fake_get)

    await srv.ensure_started()

    assert len(health_calls) >= 2
    assert stopped == [], "a server still loading its weights was evicted as idle"


def test_shutdown_stops_familiar_and_removes_its_key_and_owner_files(tmp_path, monkeypatch):
    """A clean app exit used to stop trellis and unload SDXL but never touch
    ``self.familiar`` -- only idle eviction or row deletion did. ``stop()`` is
    what deletes ``familiar-<port>.key`` and ``familiar-<port>.owner``, so a
    normal exit left both behind and the next ``ensure_started`` walked the
    orphaned-llama-server reclaim path even though nothing had crashed."""
    from realmspinner.config import Config
    from realmspinner.db import JobStore
    from realmspinner.queue import Worker

    config = Config(
        data_dir=tmp_path / "assets", db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe", trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models",
    )
    store = JobStore(config.db_path)
    worker = Worker(config, store)

    stopped = []
    monkeypatch.setattr(worker.familiar, "stop", lambda: stopped.append(True))
    monkeypatch.setattr(worker.trellis, "stop", lambda: None)
    monkeypatch.setattr(worker, "_unload_under_lease", lambda: None)
    worker._task = None
    worker.current_job_id = None
    worker._text2image = None
    worker._music_client = None

    asyncio.run(worker.shutdown())
    assert stopped == [True]


def test_a_familiar_row_not_downloaded_is_pending_install_not_a_fault(tmp_path):
    from realmspinner.config import Config

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
