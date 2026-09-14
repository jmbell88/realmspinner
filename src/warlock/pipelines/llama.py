"""Manages a resident llama-server.exe subprocess for Familiar, Warlock's in-app assistant.

Familiar is one pinned model (models.FAMILIAR_MODELS["familiar_gguf"] --
Unsloth's Q8_0 requantization of the base Gemma 4 E2B instruct model, a
testing pin; T10's Clay-assistant fine-tune replaces it as the shipped pin)
served
by llama.cpp's own HTTP server, on loopback, with no webui and offline mode
forced. It never coexists with a GPU job: ``Worker.before_gpu_job`` stops it
before ``_check_resources`` runs, and ``ensure_started`` refuses to spawn it
while a GPU job holds the lease that action takes. There is no picker and no
path override -- the weights row is pinned by revision and digest in
``models.py``, exactly as trellis' GGUF weights are.

Structurally this mirrors ``pipelines/trellis.py``'s ``TrellisServer`` closely
-- the backoff, the port-claim/reclaim dance (RUN-01), the winjob assign/track
pair, the health poll -- and reuses its two free helper functions
(``_pid_alive``, ``_port_in_use``) rather than re-deriving them, since neither
is trellis-specific. A full extraction of the shared port-claim machinery into
``pipelines/local_server.py`` (as the original brief for this tranche asked
for) was left undone to keep this change from touching ``trellis.py`` and
risking ``tests/test_trellis.py`` -- see the tranche report.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import stat
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from .. import fetch, vram, winjob
from ..progress import pump
from .trellis import _pid_alive, _port_in_use

log = logging.getLogger(__name__)

STARTUP_TIMEOUT = 120.0
LOG_MAX_BYTES = 5 * 1024 * 1024
BACKOFF_BASE = 5.0
BACKOFF_MAX = 300.0
BACKOFF_GIVE_UP = 5
RECLAIM_TIMEOUT = 5.0
KILL_TIMEOUT = 5.0
#: How long ``stop_for_gpu_job`` waits for the driver to report the freed
#: memory before giving up and letting the caller's own admission check take
#: over -- a slow reclaim is not a reason to refuse the GPU job that asked for it.
GPU_YIELD_TIMEOUT = 3.0

#: Context size, parallel slots and offload -- fixed, not configurable: there
#: is no Settings control for these because there is no picker at all for
#: Familiar, only a pinned model and a pinned launch.
CTX_SIZE = 16384
PARALLEL_SLOTS = 2
GPU_LAYERS = 999


class LlamaServer:
    """Familiar's llama-server.exe child: start, watch, and never coexist with GPU work."""

    def __init__(
        self,
        exe: Path | Callable[[], Path],
        weights_path: Path | Callable[[], Path],
        port: int,
        *,
        key_dir: Path,
        log_path: Path | None = None,
        idle_timeout: float = 300.0,
        expected_card_shas: Callable[[], tuple[str, ...]] | None = None,
    ) -> None:
        self._exe = exe
        self._weights_path = weights_path
        self._port = port
        self._key_dir = key_dir
        self._log_path = log_path
        self.idle_timeout = idle_timeout
        # Returns the pinned row's card_shas -- injected rather than imported,
        # because T3 (familiar/contract.py, the frozen prompt cards) is not
        # built yet. Once it exists, the caller wires this to
        # ``contract.card_sha`` instead of the registry's own (empty) tuple.
        self._expected_card_shas = expected_card_shas or (lambda: ())
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock_asyncio = None  # set lazily; see _lock property
        self._stop_lock = threading.Lock()
        self.last_used = 0.0
        self.on_line: Callable[[str], None] | None = None
        self._reader: threading.Thread | None = None
        self._logfh = None
        self._spawned_at: float | None = None
        self._start_failures = 0
        self._backoff_until = 0.0
        self._api_key: str | None = None
        self._key_path: Path | None = None
        # Whether a queued GPU job currently holds the lease -- ensure_started
        # refuses while this is True, and only ``release_lease`` clears it.
        # Read on the loop thread inside ``ensure_started`` (under ``_lock``,
        # an asyncio lock) but written from a ``to_thread`` pool thread by
        # ``stop_for_gpu_job``/``release_lease``. Safe with no lock of its
        # own only because a Python bool assignment is atomic under the GIL,
        # and the only ordering that matters is "set before stop()" --
        # ``stop_for_gpu_job`` already sets ``_leased = True`` before it
        # calls ``self.stop()``, so a concurrent ``ensure_started`` always
        # sees the lease before the kill it would otherwise race.
        self._leased = False

    @property
    def _lock(self):
        if self._lock_asyncio is None:
            self._lock_asyncio = asyncio.Lock()
        return self._lock_asyncio

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    @property
    def leased(self) -> bool:
        return self._leased

    def _resolve_exe(self) -> Path:
        return self._exe() if callable(self._exe) else self._exe

    def _resolve_weights(self) -> Path:
        return self._weights_path() if callable(self._weights_path) else self._weights_path

    # --- the API key file -----------------------------------------------

    def _write_key_file(self) -> Path:
        """A fresh key, in a file llama-server reads with ``--api-key-file``.

        Never on argv: a command line is visible to every other process on
        the machine (``tasklist``, Task Manager's command-line column, any
        process-listing tool an agent or another user's session might run),
        and an API key is the one thing standing between loopback and an
        unauthenticated local model. The key rotates every spawn -- there is
        no reason for it to outlive one child.

        Best-effort owner-only permissions on Windows: ``os.chmod`` with
        ``stat.S_IRUSR | stat.S_IWUSR`` clears the group/other bits Python
        tracks, which is the ACL-free half of "not world-readable" available
        without pywin32's ``SetNamedSecurityInfo``. It does not remove
        inherited ACEs a parent directory may grant (a real ACL edit would),
        so this is stated as what it is -- a best effort -- not a guarantee.
        """
        self._key_dir.mkdir(parents=True, exist_ok=True)
        key = secrets.token_hex(32)
        path = self._key_dir / f"familiar-{self._port}.key"
        path.write_text(key + "\n", encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        self._api_key = key
        self._key_path = path
        return path

    def _release_key_file(self) -> None:
        if self._key_path is not None:
            with contextlib.suppress(OSError):
                self._key_path.unlink(missing_ok=True)
            self._key_path = None
        self._api_key = None

    # --- argv --------------------------------------------------------------

    def _argv(self, key_path: Path) -> list[str]:
        return [
            str(self._resolve_exe()),
            "-m", str(self._resolve_weights()),
            "--host", "127.0.0.1",
            "--port", str(self._port),
            "--api-key-file", str(key_path),
            "--offline",
            "--jinja",
            "--no-webui",
            "-ngl", str(GPU_LAYERS),
            "--parallel", str(PARALLEL_SLOTS),
            "--ctx-size", str(CTX_SIZE),
        ]

    # --- spawn ---------------------------------------------------------

    def _check_backoff(self) -> None:
        remaining = self._backoff_until - time.monotonic()
        if remaining > 0:
            raise RuntimeError(
                f"llama-server failed to start {self._start_failures} time(s); "
                f"refusing to respawn for another {remaining:.0f} s -- see familiar.log"
            )

    def _note_start_failure(self) -> None:
        self._start_failures += 1
        delay = min(BACKOFF_BASE * 2 ** (self._start_failures - 1), BACKOFF_MAX)
        self._backoff_until = time.monotonic() + delay
        if self._start_failures >= BACKOFF_GIVE_UP:
            log.critical(
                "llama-server has failed to start %d times in a row; Familiar "
                "will keep failing until it is fixed -- see familiar.log",
                self._start_failures,
            )
        else:
            log.warning(
                "llama-server start failed (%d in a row); next attempt in %.0f s",
                self._start_failures, delay,
            )

    @property
    def _owner_path(self) -> Path | None:
        if self._log_path is None:
            return None
        return self._log_path.parent / f"familiar-{self._port}.owner"

    def _claim_port(self, pid: int) -> None:
        path = self._owner_path
        if path is None:
            return
        with contextlib.suppress(OSError):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"owner_pid": os.getpid(), "server_pid": pid}),
                encoding="utf-8",
            )

    def _release_port_claim(self) -> None:
        path = self._owner_path
        if path is not None:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)

    def _recorded_owner(self) -> int | None:
        path = self._owner_path
        if path is None:
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return int(data.get("owner_pid") or 0) or None
        except (OSError, ValueError, TypeError):
            return None

    async def _reclaim_port(self) -> None:
        """Same RUN-01 shape as ``TrellisServer._reclaim_port``: kill a
        provably-ours orphan, or refuse to guess."""
        pid = winjob.listener_pid(self._port)
        if pid is None:
            raise RuntimeError(
                f"port {self._port} is already in use, probably by an orphaned "
                "llama-server.exe left behind by a previous crash. Run "
                "`Get-Process llama-server` and stop it before retrying."
            )
        path = winjob.image_path(pid)
        ours = self._resolve_exe().resolve()
        if path is None or os.path.normcase(path) != os.path.normcase(str(ours)):
            raise RuntimeError(
                f"port {self._port} is held by pid {pid} ({path or 'unknown program'}), "
                f"which is not this Warlock's llama-server ({ours}). Stop it or "
                "change WARLOCK_FAMILIAR_PORT before retrying."
            )
        owner = self._recorded_owner()
        if owner is None:
            raise RuntimeError(
                f"port {self._port} is held by a llama-server (pid {pid}) that this "
                f"Warlock did not start. Stop it, or change WARLOCK_FAMILIAR_PORT, "
                f"before retrying."
            )
        if owner != os.getpid() and _pid_alive(owner):
            raise RuntimeError(
                f"port {self._port} is held by a llama-server started by a Warlock "
                f"that is still running (pid {owner}). Close it, or give this one "
                f"its own WARLOCK_FAMILIAR_PORT, before retrying."
            )
        log.warning(
            "port %d is held by an orphaned llama-server (pid %d) from a previous "
            "crash; terminating it", self._port, pid,
        )
        winjob.terminate(pid)
        deadline = time.monotonic() + RECLAIM_TIMEOUT
        while time.monotonic() < deadline:
            if not _port_in_use(self._port):
                return
            await asyncio.sleep(0.05)
        self._note_start_failure()
        raise RuntimeError(f"port {self._port} is still held after terminating pid {pid}")

    def _check_card_sha(self, expected_card_sha: str | None) -> None:
        """Refuse to start when the caller's expected prompt-card hash isn't
        one this weights pin was validated against.

        T3 (``familiar/contract.py``) is what will supply a real
        ``expected_card_sha`` and populate ``FamiliarModel.card_shas``; until
        it lands, the registry's ``card_shas`` is empty for the base pin, so
        any caller that *does* pass an expected sha is refused -- there is
        nothing yet to match. A caller that passes ``None`` (no card in play)
        is never refused here.
        """
        if expected_card_sha is None:
            return
        if expected_card_sha not in self._expected_card_shas():
            raise RuntimeError(
                f"Familiar's prompt card ({expected_card_sha}) does not match "
                "any card this weights pin was validated against -- refusing "
                "to start rather than run an unvalidated prompt."
            )

    def _check_manifest(self) -> None:
        """Refuse to start when either downloaded directory fails its own
        digest verification -- corrupt bytes must not reach a running server."""
        for dest in (self._resolve_exe().parent, self._resolve_weights().parent):
            verification = fetch.verify_manifest(dest)
            if verification.status == fetch.VERIFY_BAD:
                raise RuntimeError(
                    f"{dest} failed manifest verification ({verification.detail}); "
                    "remove and reinstall before starting Familiar."
                )

    def _check_vram(self) -> None:
        """Refuse to spawn ``-ngl 999`` without VRAM headroom.

        The 2026-09-14 audit (service-01) found ``vram.familiar_admission``
        called from nowhere in ``src/`` -- every other model door is admitted
        at the door before it can overcommit the card (the 2026-08-03 crash
        class), but this one went straight to ``subprocess.Popen`` because
        Familiar starts off a chat message rather than a queued job with a
        params dict for ``service.validation.check_vram`` to price. Reads
        ``live_memory()``, not ``device_memory()``, per
        ``familiar_admission``'s own docstring: a chat message has no stale
        published reading from a text2image child to fall back on, only
        whatever NVML reports right now.
        """
        if not vram.familiar_admission(vram.live_memory()):
            raise RuntimeError(
                "not enough VRAM headroom to start Familiar right now -- "
                "close whatever else is using the card and try again"
            )

    async def ensure_started(self, *, expected_card_sha: str | None = None) -> None:
        async with self._lock:
            self._reap_if_dead()
            if self.running:
                return
            if self._leased:
                raise RuntimeError(
                    "Familiar cannot start while a GPU job holds the card -- "
                    "it will restart on your next message."
                )
            self._check_backoff()
            self._check_card_sha(expected_card_sha)
            self._check_manifest()
            exe = self._resolve_exe()
            if not exe.is_file():
                raise RuntimeError(f"llama-server not found at {exe}")
            weights = self._resolve_weights()
            if not weights.is_file():
                raise RuntimeError(f"Familiar weights not found at {weights}")
            self._check_vram()
            if _port_in_use(self._port):
                await self._reclaim_port()
            log.info("starting llama-server on port %d", self._port)
            self._open_log()
            key_path = self._write_key_file()
            self._proc = subprocess.Popen(
                self._argv(key_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            winjob.assign(self._proc.pid)
            winjob.track(self._proc.pid, "llama-server")
            self._claim_port(self._proc.pid)
            self._spawned_at = time.monotonic()
            log.info("llama-server spawned as pid %d", self._proc.pid)
            self._reader = threading.Thread(
                target=self._pump, name="llama-server-stdout", daemon=True
            )
            self._reader.start()
            deadline = time.monotonic() + STARTUP_TIMEOUT
            async with httpx.AsyncClient() as client:
                while time.monotonic() < deadline:
                    proc = self._proc
                    if proc is None:
                        raise RuntimeError("llama-server was stopped during startup")
                    if proc.poll() is not None:
                        self._note_start_failure()
                        raise RuntimeError(
                            f"llama-server exited during startup (code {proc.returncode})"
                        )
                    with contextlib.suppress(httpx.TransportError):
                        r = await client.get(
                            f"{self.base_url}/health",
                            timeout=2.0,
                            headers={"Authorization": f"Bearer {self._api_key}"},
                        )
                        if r.status_code == 200:
                            log.info("llama-server ready")
                            self._start_failures = 0
                            self._backoff_until = 0.0
                            self.last_used = time.monotonic()
                            return
                    await asyncio.sleep(0.1)
            with contextlib.suppress(RuntimeError):
                await asyncio.to_thread(self.stop)
            self._note_start_failure()
            raise RuntimeError("llama-server did not become healthy in time")

    def _reap_if_dead(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is not None:
            lifetime = (
                time.monotonic() - self._spawned_at
                if self._spawned_at is not None
                else float("nan")
            )
            log.warning(
                "llama-server pid %s exited with code %s after %.1f s; reaping",
                proc.pid, proc.returncode, lifetime,
            )
            self.stop()

    # --- stdout plumbing -------------------------------------------------

    def _open_log(self) -> None:
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            if self._log_path.stat().st_size > LOG_MAX_BYTES:
                self._log_path.unlink()
        self._logfh = self._log_path.open("ab")

    def _write_log(self, chunk: bytes) -> None:
        if self._logfh is not None:
            self._logfh.write(chunk)
            self._logfh.flush()

    def _dispatch(self, line: str) -> None:
        if self.on_line is not None:
            self.on_line(line)

    def _pump(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        pump(proc.stdout, self._write_log, self._dispatch)

    def stop(self) -> None:
        """Kill the child and release its handles. Idempotent and thread-safe.

        Same contract as ``TrellisServer.stop``: raises if the process is
        still alive after the kill, because callers (the GPU-job lease, the
        idle timeout) treat a returned ``stop()`` as proof the VRAM is back.
        """
        with self._stop_lock:
            if self._proc is None and self._reader is None and self._logfh is None:
                self._release_key_file()
                return
            proc = self._proc
            if proc is not None and proc.poll() is None:
                log.info("stopping llama-server pid %d", proc.pid)
                proc.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=15)
                if proc.poll() is None:
                    proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=KILL_TIMEOUT)
                if proc.poll() is None:
                    log.critical(
                        "llama-server pid %d survived kill(); VRAM is still held",
                        proc.pid,
                    )
                    raise RuntimeError(
                        f"llama-server pid {proc.pid} did not exit after "
                        f"{KILL_TIMEOUT:.0f}s; its GPU memory is still held"
                    )
            if self._reader is not None:
                self._reader.join(timeout=5)
                self._reader = None
            if proc is not None and proc.stdout is not None:
                with contextlib.suppress(OSError):
                    proc.stdout.close()
            if self._logfh is not None:
                with contextlib.suppress(OSError):
                    self._logfh.close()
                self._logfh = None
            if proc is not None:
                winjob.untrack(proc.pid)
            self._proc = None
            self._spawned_at = None
            self._release_port_claim()
            self._release_key_file()

    # --- the GPU lease ----------------------------------------------------

    def stop_for_gpu_job(self) -> None:
        """Yield the card to a queued GPU job: take the lease, kill the child.

        Blocking, like ``stop()`` -- ``Worker.before_gpu_job`` dispatches this
        through ``asyncio.to_thread``. Takes the lease *before* stopping so a
        concurrent ``ensure_started`` (a chat message arriving in the same
        instant) sees ``leased`` and refuses rather than racing the kill.
        Waits up to ``GPU_YIELD_TIMEOUT`` for the driver to report the memory
        back, best-effort: a slow reclaim is not a reason to block the GPU job
        that asked for the card, which has its own admission check regardless.
        """
        self._leased = True
        if not self.running:
            return
        with contextlib.suppress(RuntimeError):
            self.stop()
        deadline = time.monotonic() + GPU_YIELD_TIMEOUT
        before = vram.live_memory()
        if before is None:
            return
        while time.monotonic() < deadline:
            after = vram.live_memory()
            if after is not None and after.free_gib > before.free_gib:
                return
            time.sleep(0.1)

    def release_lease(self) -> None:
        """Give the card back: the next chat message may start Familiar again."""
        self._leased = False

    def touch(self) -> None:
        """Mark the server as just-used, resetting the idle-eviction clock.

        ``last_used`` is otherwise only written at spawn and on the health
        poll (construction, ``ensure_started``), so idle eviction counts
        from *startup*, not last use -- a five-minute conversation would get
        its server evicted mid-reply. No client exists yet (T5's
        ``llama_client``); this is the door it must call once per request.
        """
        self.last_used = time.monotonic()
