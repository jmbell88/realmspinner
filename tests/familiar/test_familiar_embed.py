"""Familiar's second child: EmbeddingGemma 2 for Manual and Library retrieval.

The retrieval model is the same ``llama-server`` binary as Familiar's chat
child under a different :class:`~realmspinner.pipelines.llama.ServerProfile`.
What is under test is what makes it a *different* child -- a pinned optional
registry row, an argv with ``--device none`` and none of the chat flags, no
VRAM admission and no GPU lease, its own key and port-owner files and port --
and that it still gets everything every tracked child gets: the kill-on-close
job, its own manifest check, idle eviction, shutdown, and a stop before any
Familiar row is removed. The real-card half (the +17 MiB claim itself) is
``test_embed_gpu.py``.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from realmspinner import config as config_mod
from realmspinner import doctor, fetch, models
from realmspinner.config import Config
from realmspinner.db import JobStore
from realmspinner.pipelines import llama as llama_mod
from realmspinner.pipelines.llama import CHAT_PROFILE, EMBED_PROFILE, LlamaServer
from realmspinner.queue import Worker

_FILE = models.FAMILIAR_EMBED_GGUF_FILE


def _embed(tmp_path, **kwargs) -> LlamaServer:
    return LlamaServer(
        tmp_path / "engine" / "llama-server.exe",
        tmp_path / "models" / "familiar" / _FILE,
        17973,
        key_dir=tmp_path / "keys",
        log_path=tmp_path / EMBED_PROFILE.log_name,
        profile=EMBED_PROFILE,
        **kwargs,
    )


def _chat(tmp_path, **kwargs) -> LlamaServer:
    return LlamaServer(
        tmp_path / "engine" / "llama-server.exe",
        tmp_path / "models" / "familiar" / models.FAMILIAR_GGUF_FILE,
        17972,
        key_dir=tmp_path / "keys",
        log_path=tmp_path / CHAT_PROFILE.log_name,
        **kwargs,
    )


def _config(tmp_path) -> Config:
    return Config(
        data_dir=tmp_path / "assets", db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe", trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models",
        familiar_runtime_dir=tmp_path / "engine" / "llama",
        familiar_models_dir=tmp_path / "models" / "familiar",
        familiar_idle_timeout=1.0,
    )


# --- the registry row ----------------------------------------------------


def test_the_retrieval_row_is_pinned_to_the_measured_embeddinggemma_file():
    spec = models.FAMILIAR_MODELS["familiar_embed"]
    (one,) = spec.fetch
    assert spec.label == "Familiar retrieval (EmbeddingGemma 2)"
    assert one.repo_id == "ggml-org/embeddinggemma-2-GGUF" == models.FAMILIAR_EMBED_GGUF_REPO
    assert one.revision == "bfcd298762cc34d0357ece5ebdd31791a3a374d8"
    assert one.revision == models.FAMILIAR_EMBED_GGUF_REVISION
    assert one.filenames == ("embeddinggemma-2-Q8_0.gguf",) == (models.FAMILIAR_EMBED_GGUF_FILE,)
    sha = "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135"
    assert sha == models.FAMILIAR_EMBED_GGUF_SHA256
    assert spec.digests == ((models.FAMILIAR_EMBED_GGUF_FILE, sha),)
    # 309,855,456 B measured; the figure only has to be the right size class.
    assert one.size_gib == pytest.approx(309_855_456 / 2**30, abs=0.01)
    assert models.FAMILIAR_EMBED_DIM == 768


def test_the_retrieval_row_is_optional_and_never_a_runtime_row():
    spec = models.FAMILIAR_MODELS["familiar_embed"]
    assert spec.optional is True
    assert spec.runtime is False
    # Its description is the first thing Settings -> Models shows beside the row.
    assert len(spec.description.splitlines()[0]) < 90
    # ...and it must not read as a vision row: the loss is its own sentence.
    assert "text-only" not in spec.without
    assert models.FAMILIAR_MODELS["familiar_mmproj"].without


def test_the_retrieval_row_is_a_familiar_row_that_lands_in_the_models_directory(tmp_path):
    entry = fetch.find("familiar:familiar_embed")
    assert entry is not None and entry.kind == "familiar"
    config = _config(tmp_path)
    (one,) = entry.spec.fetch
    assert fetch.destination(config, entry, one) == config.familiar_models_dir
    assert fetch.claims(config, entry) == (config.familiar_models_dir,)
    assert fetch.present(config, "familiar", entry.spec) is False
    config.familiar_models_dir.mkdir(parents=True)
    (config.familiar_models_dir / _FILE).write_bytes(b"x")
    assert fetch.present(config, "familiar", entry.spec) is True


def test_a_zero_byte_retrieval_file_is_suspect_like_every_familiar_weight(tmp_path):
    config = _config(tmp_path)
    config.familiar_models_dir.mkdir(parents=True)
    (config.familiar_models_dir / _FILE).write_bytes(b"")
    spec = models.FAMILIAR_MODELS["familiar_embed"]
    assert fetch.suspect_files(config, "familiar", spec) == [
        str(config.familiar_models_dir / _FILE)
    ]


def test_the_embed_port_is_a_gated_setting_with_its_own_default_and_variable(monkeypatch):
    assert ("familiar_embed_port", "REALMSPINNER_FAMILIAR_EMBED_PORT") in config_mod.SETTINGS
    monkeypatch.delenv("REALMSPINNER_FAMILIAR_EMBED_PORT", raising=False)
    monkeypatch.delenv("REALMSPINNER_FAMILIAR_PORT", raising=False)
    assert Config().familiar_embed_port == 17973
    assert Config().familiar_embed_port != Config().familiar_port
    monkeypatch.setenv("REALMSPINNER_FAMILIAR_EMBED_PORT", "18123")
    assert Config().familiar_embed_port == 18123
    assert Config().familiar_port == 17972


# --- the spawn ---------------------------------------------------------------


def test_the_embed_argv_is_cpu_only_embedding_and_none_of_the_chat_flags(tmp_path):
    argv = _embed(tmp_path)._argv(tmp_path / "key.txt")
    assert argv[argv.index("--device") + 1] == "none"
    assert "--embeddings" in argv
    assert argv[argv.index("--pooling") + 1] == "mean"
    # One number covers context, batch and micro-batch: a non-causal model
    # refuses an input longer than -ub (measured: the longest chunk is 1,524 words).
    for flag in ("-c", "-b", "-ub"):
        assert argv[argv.index(flag) + 1] == "8192"
    assert argv[argv.index("--parallel") + 1] == "1"
    for flag in ("--jinja", "--mmproj", "-ngl", "--ngl", "--reasoning", "--reasoning-budget",
                 "--alias", "--ctx-size"):
        assert flag not in argv, flag
    # Everything every tracked child shares survives the profile.
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "17973"
    assert "--offline" in argv and "--no-webui" in argv
    assert argv[argv.index("--api-key-file") + 1] == str(tmp_path / "key.txt")
    assert argv[argv.index("-m") + 1].endswith(_FILE)


def test_the_embed_child_is_never_handed_a_projector_or_an_alias(tmp_path):
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"x")
    srv = _embed(tmp_path, mmproj_path=mmproj, served_name="familiar_v1.0")
    argv = srv._argv(tmp_path / "key.txt")
    assert "--mmproj" not in argv and "--alias" not in argv


def test_the_chat_argv_is_exactly_what_it_was_before_the_profile_existed(tmp_path):
    key = tmp_path / "key.txt"
    argv = _chat(tmp_path)._argv(key)
    assert argv == [
        str(tmp_path / "engine" / "llama-server.exe"),
        "-m", str(tmp_path / "models" / "familiar" / models.FAMILIAR_GGUF_FILE),
        "--host", "127.0.0.1",
        "--port", "17972",
        "--api-key-file", str(key),
        "--offline",
        "--jinja",
        "--no-webui",
        "-ngl", "999",
        "--parallel", "2",
        "--ctx-size", "16384",
        "--reasoning", "off",
        "--reasoning-budget", "0",
    ]


def test_the_embed_child_has_its_own_key_and_owner_files_and_log(tmp_path):
    chat, embed = _chat(tmp_path), _embed(tmp_path)
    chat_key, embed_key = chat._write_key_file(), embed._write_key_file()
    assert chat_key != embed_key and chat_key.exists() and embed_key.exists()
    assert embed_key.name == "familiar-embed-17973.key"
    assert chat_key.name == "familiar-17972.key"
    assert chat._owner_path != embed._owner_path
    assert embed._owner_path.name == "familiar-embed-17973.owner"
    assert llama_mod.EMBED_PROFILE.log_name != llama_mod.CHAT_PROFILE.log_name
    # Same port by misconfiguration must still not share a claim file.
    clash = LlamaServer(
        tmp_path / "e.exe", tmp_path / "w.gguf", 17972,
        key_dir=tmp_path / "keys", log_path=tmp_path / "x.log", profile=EMBED_PROFILE,
    )
    assert clash._owner_path != chat._owner_path
    assert clash._write_key_file() != chat_key
    # Stopping one never removes the other's key.
    embed._release_key_file()
    assert chat_key.exists() and not embed_key.exists()


def _arm(srv, tmp_path, monkeypatch, *, spawned=None):
    """Make ``ensure_started`` reach its spawn against fakes."""
    srv._resolve_exe().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_exe().write_bytes(b"")
    srv._resolve_weights().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_weights().write_bytes(b"")
    monkeypatch.setattr(
        llama_mod.fetch,
        "verify_manifest",
        lambda dest, **kw: fetch.Verification(dest=dest, status=fetch.VERIFY_UNKNOWN),
    )
    monkeypatch.setattr(llama_mod, "_port_in_use", lambda port: False)
    sink = spawned if spawned is not None else []
    monkeypatch.setattr(llama_mod.winjob, "assign", lambda pid: sink.append(("assign", pid)))
    monkeypatch.setattr(llama_mod.winjob, "track", lambda pid, name: sink.append(("track", name)))
    monkeypatch.setattr(llama_mod.winjob, "untrack", lambda pid: None)
    monkeypatch.setattr(srv, "_pump", lambda: None)
    fake = type("P", (), {"pid": 4343, "returncode": None, "poll": lambda self: None})()
    monkeypatch.setattr(llama_mod.subprocess, "Popen", lambda *a, **k: fake)

    async def ok(self, url, **kwargs):
        return httpx.Response(200)

    monkeypatch.setattr(llama_mod.httpx.AsyncClient, "get", ok)


@pytest.mark.asyncio
async def test_the_embed_child_is_not_priced_at_the_vram_door(tmp_path, monkeypatch):
    """Not a GPU tenant: with the card reported full the chat child refuses
    to start and the embed child (``--device none``, +17 MiB measured) does."""
    monkeypatch.setattr(llama_mod.vram, "live_memory", lambda: None)
    monkeypatch.setattr(llama_mod.vram, "familiar_admission", lambda mem: False)
    embed = _embed(tmp_path)
    _arm(embed, tmp_path, monkeypatch)
    await embed.ensure_started()
    assert embed.running

    chat = _chat(tmp_path)
    _arm(chat, tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="not enough VRAM"):
        await chat.ensure_started()


@pytest.mark.asyncio
async def test_the_embed_child_is_assigned_to_the_kill_on_close_job_and_tracked(
    tmp_path, monkeypatch
):
    events: list = []
    embed = _embed(tmp_path)
    _arm(embed, tmp_path, monkeypatch, spawned=events)
    await embed.ensure_started()
    assert ("assign", 4343) in events
    assert ("track", "llama-server-embed") in events
    assert embed._owner_path.exists(), "the port-owner claim is what a reclaim reads"
    assert json.loads(embed._owner_path.read_text(encoding="utf-8"))["server_pid"] == 4343


def test_the_gpu_lease_is_a_no_op_for_the_embed_child(tmp_path):
    embed = _embed(tmp_path)
    stopped: list = []
    embed._proc = type("P", (), {"poll": lambda self: None})()
    embed.stop = lambda: stopped.append(True)
    embed.stop_for_gpu_job()
    assert stopped == [], "a GPU job must never evict the retrieval child"
    assert embed.leased is False
    embed.release_lease()
    assert embed.leased is False
    # ...and the chat child still takes it and stops (the contrast).
    chat = _chat(tmp_path)
    chat._proc = type("P", (), {"poll": lambda self: None})()
    chat.stop = lambda: stopped.append("chat")
    chat.stop_for_gpu_job()
    assert stopped == ["chat"] and chat.leased is True


@pytest.mark.asyncio
async def test_a_gpu_lease_on_the_chat_child_never_blocks_the_embed_child(tmp_path, monkeypatch):
    chat, embed = _chat(tmp_path), _embed(tmp_path)
    chat._leased = True
    _arm(embed, tmp_path, monkeypatch)
    await embed.ensure_started()
    assert embed.running


@pytest.mark.asyncio
async def test_the_embed_child_verifies_only_its_own_weights_file(tmp_path, monkeypatch):
    """It shares ``models/familiar/`` with 6.5 GiB of chat weights: its start
    must not re-hash them. The runtime directory is checked whole."""
    calls: list = []
    embed = _embed(tmp_path)
    _arm(embed, tmp_path, monkeypatch)

    def record(dest, **kw):
        calls.append((dest, kw.get("only")))
        return fetch.Verification(dest=dest, status=fetch.VERIFY_UNKNOWN)

    monkeypatch.setattr(llama_mod.fetch, "verify_manifest", record)
    await embed.ensure_started()
    assert (embed._resolve_exe().parent, None) in calls
    assert (embed._resolve_weights().parent, (_FILE,)) in calls

    calls.clear()
    chat = _chat(tmp_path)
    _arm(chat, tmp_path, monkeypatch)
    monkeypatch.setattr(llama_mod.fetch, "verify_manifest", record)
    monkeypatch.setattr(chat, "_check_vram", lambda: None)
    await chat.ensure_started()
    assert all(only is None for _dest, only in calls), "the chat child checks the whole directory"


@pytest.mark.asyncio
async def test_a_corrupt_retrieval_file_refuses_to_start_and_names_retrieval(tmp_path, monkeypatch):
    embed = _embed(tmp_path)
    _arm(embed, tmp_path, monkeypatch)
    monkeypatch.setattr(
        llama_mod.fetch,
        "verify_manifest",
        lambda dest, **kw: fetch.Verification(dest=dest, status=fetch.VERIFY_BAD, bad=(_FILE,)),
    )
    with pytest.raises(RuntimeError, match="retrieval"):
        await embed.ensure_started()


def test_verify_manifest_only_narrows_the_check_to_the_named_files(tmp_path):
    from realmspinner import hashes

    (tmp_path / "a.gguf").write_bytes(b"alpha")
    (tmp_path / "b.gguf").write_bytes(b"beta")
    digests = {
        "a.gguf": {"sha256": hashes.sha256(tmp_path / "a.gguf")},
        "b.gguf": {"sha256": "0" * 64},  # b is "corrupt"
    }
    fetch.manifest_path(tmp_path).write_text(
        json.dumps({"repos": {"r": {"digests": digests}}}), encoding="utf-8"
    )
    assert fetch.verify_manifest(tmp_path).status == fetch.VERIFY_BAD
    only_a = fetch.verify_manifest(tmp_path, only=("a.gguf",))
    assert only_a.status == fetch.VERIFY_OK and only_a.checked == 1
    assert fetch.verify_manifest(tmp_path, only=("b.gguf",)).status == fetch.VERIFY_BAD
    assert fetch.verify_manifest(tmp_path, only=("nothing.gguf",)).status == fetch.VERIFY_UNKNOWN


# --- the queue ----------------------------------------------------------------


def test_the_worker_builds_the_embed_child_on_its_own_port_beside_the_chat_one(tmp_path):
    config = _config(tmp_path)
    worker = Worker(config, JobStore(config.db_path))
    assert worker.familiar_embed is not worker.familiar
    assert worker.familiar_embed.profile is EMBED_PROFILE
    assert worker.familiar.profile is CHAT_PROFILE
    assert worker.familiar_embed.base_url.endswith(f":{config.familiar_embed_port}")
    assert worker.familiar_embed.base_url != worker.familiar.base_url
    assert worker.familiar_embed._resolve_exe() == worker.familiar._resolve_exe()
    assert worker.familiar_embed._resolve_weights() == config.familiar_models_dir / _FILE


def test_a_gpu_job_stops_familiar_and_leaves_the_retrieval_child_alone(tmp_path, monkeypatch):
    config = _config(tmp_path)
    worker = Worker(config, JobStore(config.db_path))
    called: list = []
    monkeypatch.setattr(worker.familiar, "stop_for_gpu_job", lambda: called.append("chat"))
    monkeypatch.setattr(worker.familiar_embed, "stop_for_gpu_job", lambda: called.append("embed"))
    monkeypatch.setattr(worker.familiar_embed, "stop", lambda: called.append("embed-stop"))
    monkeypatch.setattr(worker.familiar_embed, "release_lease", lambda: called.append("embed-rel"))
    job = {"kind": "text", "stage": "model", "params": {}}
    asyncio.run(worker.before_gpu_job(job))
    asyncio.run(worker.after_gpu_job(job))
    assert called == ["chat"]


def test_an_idle_retrieval_child_is_stopped_on_its_own_clock(tmp_path, monkeypatch):
    config = _config(tmp_path)
    worker = Worker(config, JobStore(config.db_path))
    worker.familiar_embed._proc = type("P", (), {"poll": lambda self: None})()
    worker.familiar_embed.last_used = 0.0
    stopped: list = []
    monkeypatch.setattr(worker.familiar_embed, "stop", lambda: stopped.append("embed"))
    monkeypatch.setattr(worker.familiar, "stop", lambda: stopped.append("chat"))
    asyncio.run(worker._maybe_evict_idle())
    assert stopped == ["embed"], "the chat child was not running and must not be touched"
    # A touch (every request does one) resets the clock.
    stopped.clear()
    worker.familiar_embed.touch()
    asyncio.run(worker._maybe_evict_idle())
    assert stopped == []


def test_shutdown_stops_the_retrieval_child_and_removes_its_key_and_owner_files(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    worker = Worker(config, JobStore(config.db_path))
    stopped: list = []
    monkeypatch.setattr(worker.familiar, "stop", lambda: stopped.append("chat"))
    monkeypatch.setattr(worker.familiar_embed, "stop", lambda: stopped.append("embed"))
    monkeypatch.setattr(worker.trellis, "stop", lambda: None)
    monkeypatch.setattr(worker, "_unload_under_lease", lambda: None)
    worker._task = None
    worker.current_job_id = None
    worker._text2image = None
    worker._music_client = None
    asyncio.run(worker.shutdown())
    assert stopped == ["chat", "embed"]


def test_removing_a_familiar_row_stops_the_retrieval_child_as_well_as_the_chat_one():
    from realmspinner.service import downloads as downloads_mod

    class Child:
        running = True

        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    class FakeWorker:
        current_job_id = None
        familiar = Child()
        familiar_embed = Child()

        async def unload_text2image(self):
            return None

    worker = FakeWorker()
    assert asyncio.run(downloads_mod._release_if_idle(worker)) is True
    assert worker.familiar.stopped is True
    assert worker.familiar_embed.stopped is True


def test_a_retrieval_child_that_is_not_running_is_not_stopped_on_removal():
    from realmspinner.service import downloads as downloads_mod

    class Child:
        def __init__(self, running):
            self.running = running
            self.stopped = False

        def stop(self):
            self.stopped = True

    class FakeWorker:
        current_job_id = None

        async def unload_text2image(self):
            return None

    worker = FakeWorker()
    worker.familiar, worker.familiar_embed = Child(False), Child(False)
    asyncio.run(downloads_mod._release_if_idle(worker))
    assert not worker.familiar_embed.stopped


# --- doctor ------------------------------------------------------------------


def test_an_absent_retrieval_row_is_pending_install_and_does_not_fail_doctor(
    tmp_path, monkeypatch, capsys
):
    config = _config(tmp_path)
    # Everything Familiar needs is on disk; only the optional rows are absent.
    for key in ("familiar_runtime", "familiar_runtime_cudart", "familiar_gguf"):
        spec = models.FAMILIAR_MODELS[key]
        base = fetch.familiar_dir(config, spec)
        base.mkdir(parents=True, exist_ok=True)
        for name in spec.probe:
            (base / name).write_bytes(b"x")
    checks = {c.name: c for c in doctor._familiar_checks(config)}
    row = checks[fetch.check_name("familiar", models.FAMILIAR_MODELS["familiar_embed"].label)]
    assert row.ok is False and row.fatal is False and row.pending_install is True
    assert row.detail.startswith("optional --")
    assert "text-only" not in row.detail
    assert "uvx hf download ggml-org/embeddinggemma-2-GGUF" in row.detail
    assert models.FAMILIAR_EMBED_GGUF_REVISION in row.detail

    # The exit code is the CLI's own predicate: only a fatal, failing row counts.
    from realmspinner import cli

    monkeypatch.setattr(doctor, "run_checks", lambda cfg, **kw: list(checks.values()))
    monkeypatch.setattr(config_mod, "get_config", lambda: config)
    cli._run_doctor()  # must not raise SystemExit
    out = capsys.readouterr().out
    assert "[SETUP]" in out and "retrieval" in out.lower()


def test_the_llama_server_health_poll_ignores_proxy_environment_variables():
    """Both llama-server children (chat and embed) are polled on loopback by
    ``LlamaServer.ensure_started``: an ``AsyncClient`` that honours
    HTTP(S)_PROXY or the Windows system proxy hands that poll to the proxy, so a
    machine behind one never sees its own server come up (the same defect the
    TRELLIS clients had, 2026-10-03 audit). Every ``AsyncClient`` in
    ``pipelines/llama.py`` must set ``trust_env=False`` explicitly."""
    import ast
    import inspect

    from realmspinner.pipelines import llama as llama_mod

    tree = ast.parse(inspect.getsource(llama_mod))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "AsyncClient"
    ]
    assert calls, "the scan found no AsyncClient to guard"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "trust_env" in kw, f"line {call.lineno}: AsyncClient trusts the proxy environment"
        assert isinstance(kw["trust_env"], ast.Constant) and kw["trust_env"].value is False
