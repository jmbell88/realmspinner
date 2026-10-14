"""The transport under MCP: a named pipe (Windows) or Unix socket, with a token.

`multiprocessing.connection` gives us a length-framed, authenticated
byte channel for free -- `Listener`/`Client` already do the HMAC challenge
against `authkey`, so nothing here reimplements it. The one hard rule is
**never `send`/`recv`**: those pickle the object, and a pickle deserialised
off a local pipe is an arbitrary-code door -- the token proves the peer knows
the secret, but it must not be the only thing standing between a connection
and `eval`-by-another-name. `send_bytes`/`recv_bytes` move plain bytes and
leave the JSON-RPC framing to `protocol.py`, which is the whole point of
splitting the two modules.

**One session at a time is the v1 decision, not an oversight.** An agent
session is exclusive use of Realmspinner it is driving -- there is no sensible
way to interleave two agents' tool calls against one `ClayTab`. The pipe
itself authenticates every peer that connects and keeps accepting; it is the
host (`AgentHost._listen`/`_admit`) that admits one session and answers every
later bridge `busy` on its first frame (`_refuse_busy`) and closes it, rather
than leaving it waiting for the first to disconnect. Multiplexing, if it is
ever needed, is a v2 problem.

**The challenge is run here, on a clock, rather than inside
`Listener.accept()`.** `Listener(authkey=...)` does the HMAC exchange for you,
but it does it with an unbounded `recv_bytes` -- so a peer that opens the pipe
and then says nothing parks the accept loop forever. Measured, three things
followed from that and every one of them was reachable by any local process:
the listener never served anyone again; `Server.close()` did **not** free it,
so `AgentHost.stop()` joined for two seconds and left the thread parked; and
the handle that thread still held made the *next* `start()` fail with
`PermissionError`, because CPython opens the Windows pipe with
`FILE_FLAG_FIRST_PIPE_INSTANCE` and a leaked instance is still an instance.
That last one is what turned a wedged pipe into an app that would not launch,
since the Settings switch persists before `start()` runs.

So the `Listener` here is built with **no** authkey and :meth:`Server._handshake`
performs the exchange itself, on a helper thread it is willing to abandon. It
still calls `multiprocessing.connection`'s own `deliver_challenge` /
`answer_challenge` rather than reimplementing them -- the rule at the top of
this docstring is about not hand-rolling the crypto, and this does not; it only
sequences it. The order mirrors `Listener.accept` exactly (server delivers then
answers; the client answers then delivers), and the round-trip test is the gate
that says so: if a future CPython changes the exchange, a real `Client` stops
authenticating and that test fails loudly rather than this drifting quietly.
Closing the connection on timeout is the part that actually matters -- it
unblocks the abandoned thread, which is what stops the handle leaking into the
next `start()`.

**The pipe is not owner-only, and nothing here pretends it is.** CPython creates
the Windows pipe with a default security descriptor that grants Everyone and
Anonymous read access (the 2026-10-14 audit, finding agents-protocol-01, read it
back: `D:(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;<user>)(A;;FR;;;WD)(A;;FR;;;AN)`).
Such a peer can open the pipe but holds no write access, so it can never answer
the challenge and never gets a connection; what it *could* do was occupy the
serial accept loop. The defence is therefore the token plus concurrent,
bounded handshakes (:class:`Server`), not the DACL.
"""

from __future__ import annotations

import contextlib
import hashlib
import multiprocessing.connection as mpconn
import os
import queue
import secrets
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

_FAMILY = "AF_PIPE" if sys.platform == "win32" else "AF_UNIX"

HANDSHAKE_TIMEOUT = 5.0
"""How long a connecting peer gets to finish the HMAC exchange before it is
dropped and the next one is served.

Generous for what it covers: the exchange is four small frames over a local
pipe, which a live peer completes in microseconds, so this is "that process is
never going to answer" territory rather than "the machine is briefly busy".
The cost of it being wrong in the slow direction is only that one legitimate
bridge on a badly loaded box has to dial again; the cost of having no bound at
all was an agent server any local process could switch off for good."""

MAX_PENDING_HANDSHAKES = 16
"""How many peers may be mid-handshake at once; past it the *oldest* is dropped.

The 2026-10-14 audit (finding agents-protocol-01) reproduced that two peers
holding only the read access CPython's default pipe security descriptor grants
Everyone could starve the serial accept loop for ten seconds. Handshakes now
run side by side, and this bound keeps a flood of silent peers from becoming a
thread-per-connection leak. Oldest-first because a live peer answers in
microseconds, so the longest-waiting one is the likeliest never to."""

ACCEPT_RETRY_DELAY = 0.05
ACCEPT_RETRY_CAP = 1.0
"""Backoff, in seconds, between accept attempts after a transient failure from a
live listener (the 2026-10-14 audit, finding agents-protocol-06)."""

CLOSE_WAKE_TIMEOUT = 1.5
"""The most `Server.close()` spends waiting for the acceptor thread to leave
`Listener.accept()`. It runs on the frame thread, so it is a deadline and never
a join; a healthy close is over in a few milliseconds."""


def address_for(home: Path) -> str:
    """The pipe/socket address for a given `REALMSPINNER_HOME`.

    Derived from `home` rather than fixed, because more than one `REALMSPINNER_HOME`
    can be live on one machine (tests pin a throwaway one; `REALMSPINNER_TRELLIS_EXE`-
    style overrides exist precisely so two checkouts can coexist) and two
    Realmspinners listening on the same pipe name would race each other's
    `accept()`.
    Hashed rather than used verbatim because a Windows pipe name has to be a
    single path segment -- `home` itself may contain `\\`, `:`, anything a
    filesystem allows.
    """
    if sys.platform == "win32":
        digest = hashlib.sha256(str(home).lower().encode("utf-8")).hexdigest()[:16]
        return "\\\\.\\pipe\\realmspinner-mcp-" + digest
    return str(home / "mcp.sock")


def token_path(home: Path) -> Path:
    return home / "mcp.token"


def write_token(home: Path) -> bytes:
    """Generate a fresh 32-byte authkey and publish it for the bridge to read.

    Staged beside the destination and landed with `os.replace` -- this repo's
    rule for every write onto a name another process reads, here because a
    bridge that opens `mcp.token` mid-write must never see a truncated or
    half-hex-encoded secret.

    **The restrictive mode is baked into the file's creation, not applied
    after it.** The 2026-09-26 audit (agents-protocol-03) found this used to
    `tmp.write_text(...)` -- which creates the file at the process's default,
    umask-derived mode (typically world- or group-readable on a POSIX box) --
    and only *then* `os.chmod` it to `0o600`, leaving the 32-byte secret
    sitting on disk at a wider mode for the gap between those two calls.
    `os.open` with `O_CREAT | O_EXCL` and the mode passed to the *creation*
    call itself closes that gap: the file never exists at a mode wider than
    `0o600`, not even for one syscall. The follow-up `os.chmod` stays as
    belt-and-braces (and is still what actually does the work on a `tmp` that
    somehow already existed at a wider mode); `os.chmod` failing outright is
    still not fatal on Windows, since the authkey challenge inside
    `Listener`/`Client` is what actually gates the connection -- file
    permissions are defence in depth, not the mechanism.
    """
    home.mkdir(parents=True, exist_ok=True)
    token = secrets.token_bytes(32)
    dest = token_path(home)
    tmp = dest.with_name(f".{dest.name}.{secrets.token_hex(4)}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="ascii") as fh:
            fh.write(token.hex())
        with contextlib.suppress(OSError):
            os.chmod(tmp, 0o600)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return token


def read_token(home: Path) -> bytes:
    """The authkey `write_token` published, or a plain-sentence `FileNotFoundError`."""
    try:
        raw = token_path(home).read_text(encoding="ascii")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "No agent session is open: Realmspinner has not started an MCP "
            "server for this home directory."
        ) from exc
    return bytes.fromhex(raw.strip())


def clear_token(home: Path) -> None:
    token_path(home).unlink(missing_ok=True)


def _clear_stale_socket(home: Path) -> None:
    """Remove a Unix socket left behind by a Realmspinner that did not close it.

    Windows named pipes die with the process that made them; an `AF_UNIX`
    socket is a file that outlives one, and `bind` refuses an address whose
    file already exists. So on POSIX a hard kill (or any exit that skipped
    `Server.close`) left `mcp.sock` on disk and every later `start()` raised
    `EADDRINUSE` -- permanently, since nothing in the app ever removed it. The
    agent server could not be switched on again for the life of that home
    without someone deleting the file by hand.

    Unlinking it is safe because of `instance.py`: one Realmspinner holds the lock
    for a home at a time, so a socket file here cannot belong to a live
    Realmspinner. It is a leftover by construction, and treating it as one is what
    makes the feature recoverable rather than one crash away from dead. No-op
    on Windows, where the address is a pipe name and not a path at all.
    """
    if sys.platform == "win32":
        return
    with contextlib.suppress(OSError):
        Path(address_for(home)).unlink(missing_ok=True)


class Server:
    """The app's side of the pipe: one listener, which hands back each
    authenticated connection in turn. Admitting only one *session* of them is
    the host's job (see the module docstring), not this class's.

    **The accepting is done on a thread this class owns, and the handshakes
    run concurrently.** The 2026-10-14 audit (finding agents-protocol-01)
    reproduced that the pipe grants Everyone and Anonymous read access by
    CPython's default security descriptor, and that the old accept loop vetted
    one peer at a time with a five-second budget each: two peers that could
    open the pipe and never answer starved every legitimate bridge for the sum
    of their timeouts. Now the acceptor thread only *accepts* -- it hands each
    raw connection to its own vetting thread and goes straight back to the
    next -- and :meth:`accept` waits on a queue of the connections that passed.
    The set of vetting threads in flight is bounded (:data:`MAX_PENDING_HANDSHAKES`),
    and the **oldest** is dropped when it is full, because a legitimate peer
    answers in microseconds: the oldest in-flight peer is the likeliest to be
    one that never will."""

    def __init__(self, home: Path) -> None:
        self._home = home
        self._listener: mpconn.Listener | None = None
        # Held rather than handed to the Listener: this class runs the
        # challenge itself, on a clock -- see the module docstring.
        self._authkey: bytes | None = None
        # Everything below is guarded by `_lock`. It is never held across a
        # blocking call: the acceptor parks in `Listener.accept` outside it.
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._closed.set()  # not started is closed
        self._ready: queue.SimpleQueue[mpconn.Connection | None] = queue.SimpleQueue()
        self._pending: OrderedDict[int, mpconn.Connection] = OrderedDict()
        self._acceptor: threading.Thread | None = None

    def start(self) -> None:
        authkey = write_token(self._home)
        _clear_stale_socket(self._home)
        try:
            listener = mpconn.Listener(address_for(self._home), family=_FAMILY)
        except Exception:
            # The 2026-09-26 audit (agents-protocol-04): the token above is
            # already published by the time construction can fail (a busy
            # address, a permissions problem) -- leaving it in place would
            # advertise a working secret for a pipe nobody is listening on.
            # `AgentHost.start`'s own except clause happens to call
            # `close()` after a failed `start()`, which would clean this up
            # too, but that is that caller's own discipline, not a contract
            # this method makes; a future or a test-only caller with no such
            # cleanup would otherwise leave a stale `mcp.token` sitting there
            # until the next successful `start()` overwrites it.
            clear_token(self._home)
            raise
        with self._lock:
            self._authkey = authkey
            self._listener = listener
            self._closed.clear()
            self._ready = queue.SimpleQueue()
            self._pending.clear()
            acceptor = threading.Thread(
                target=self._accept_loop,
                args=(listener,),
                name="realmspinner-mcp-accept",
                daemon=True,
            )
            self._acceptor = acceptor
        acceptor.start()

    def close(self) -> None:
        with self._lock:
            listener = self._listener
            # Cleared here, *before* the self-dial below, not after it. The
            # 2026-09-26 audit (agents-protocol-05) found the old code set
            # this after the dial, while the comment beside the dial already
            # claimed it was "already cleared" by then -- a real race: a
            # vetting thread reads `self._authkey` and only takes its fast
            # "key is None: closed underneath us" reject path once this is
            # genuinely `None`. Cleared after, it could walk the self-dial
            # through the full crypto exchange, which the self-dial's own
            # bare `Client` call never answers.
            self._listener = None
            self._authkey = None
            self._closed.set()
            # Everything still being vetted, and everything vetted but not yet
            # collected, belongs to a server that no longer exists.
            doomed = list(self._pending.values())
            self._pending.clear()
            while True:
                try:
                    stranded = self._ready.get_nowait()
                except queue.Empty:
                    break
                if stranded is not None:
                    doomed.append(stranded)
            acceptor = self._acceptor
            # Wakes every caller blocked in `accept()`, now and later.
            self._ready.put(None)
        if listener is not None:
            for conn in doomed:
                with contextlib.suppress(OSError):
                    conn.close()
            if sys.platform == "win32":
                self._wake_acceptor(listener.address, acceptor)
            # Last, not first: see `_wake_acceptor` for why the stdlib
            # listener must outlive the thread that is calling into it.
            listener.close()
        # Cleared even if start() was never called: idempotent close is what
        # lets AgentHost.stop() be idempotent too, per its own contract.
        clear_token(self._home)

    @staticmethod
    def _wake_acceptor(address: str, acceptor: threading.Thread | None) -> None:
        """Get the acceptor thread out of `Listener.accept()` and wait for it to leave.

        A thread parked in `ConnectNamedPipe` is not reliably released by
        closing the handle from another thread (measured: 5/5 trials left it
        alive past `stop()`'s own join, holding the pipe's
        `FILE_FLAG_FIRST_PIPE_INSTANCE` handle, so the next `start()` failed
        with `PermissionError`). Dialling in ourselves completes the pending
        connect, and the acceptor -- which sees the server closed -- drops the
        connection and exits.

        **Three properties the audit asked for, each from a measured incident.**
        The dial is bounded: `mpconn.Client` retries a busy pipe for twenty
        seconds, and `close()` runs on the frame thread, so the dial is made
        from a helper we are willing to abandon (agents-protocol-02: 4.5 s of
        frozen frames when the old code dialled while a peer was mid-handshake
        and no instance was free). It is repeated until the acceptor is really
        gone, because a `close()` can land before the thread has reached
        `ConnectNamedPipe` and the first dial then finds nothing to connect to.
        And the stdlib listener is closed only *afterwards* (agents-protocol-05,
        reproduced as a restart that raised `PermissionError`): its finalizer
        closes the handles in its queue exactly once, so a thread still inside
        `PipeListener.accept()` when that ran appended a fresh pipe instance
        that nothing would ever close, and CPython opens the next `start()`'s
        pipe with FILE_FLAG_FIRST_PIPE_INSTANCE.
        """
        deadline = time.monotonic() + CLOSE_WAKE_TIMEOUT
        first = True
        while first or (
            acceptor is not None and acceptor.is_alive() and time.monotonic() < deadline
        ):
            first = False
            dial = threading.Thread(
                target=Server._self_dial,
                args=(address,),
                name="realmspinner-mcp-wake",
                daemon=True,
            )
            dial.start()
            dial.join(0.25)
            if acceptor is not None:
                acceptor.join(0.05)

    @staticmethod
    def _self_dial(address: str) -> None:
        with contextlib.suppress(Exception):
            mpconn.Client(address, family=_FAMILY).close()

    def _accept_loop(self, listener: mpconn.Listener) -> None:
        """The acceptor thread: take raw connections and hand each to a vetting thread.

        **`listener` is the one reference this thread works from.** The
        2026-09-26 audit (agents-protocol-02) found the old loop reading
        `self._listener` twice, so a `close()` between the reads made it call
        `None.accept()`. And stdlib's own `Listener.accept` reads its inner
        listener twice, so any failure here is caught as a whole rather than
        by type.

        **A failure from a live listener is retried, not read as shutdown.**
        The 2026-10-14 audit (finding agents-protocol-06) injected
        ECONNABORTED and found every `OSError` mapped to "closed", which the
        host reads as "stop listening" -- one failed accept (EMFILE on POSIX,
        a failed `CreateNamedPipe` on Windows) switched the feature off while
        Settings still showed the pipe address. Only `close()` ends this
        thread; anything else backs off and tries again.
        """
        delay = ACCEPT_RETRY_DELAY
        while True:
            with self._lock:
                if self._listener is not listener:
                    return
            try:
                conn = listener.accept()
            except Exception:  # noqa: BLE001 -- see the docstring
                with self._lock:
                    live = self._listener is listener
                if not live or self._closed.wait(delay):
                    return
                delay = min(delay * 2, ACCEPT_RETRY_CAP)
                continue
            delay = ACCEPT_RETRY_DELAY
            self._begin_vetting(conn, listener)

    def _begin_vetting(self, conn: mpconn.Connection, listener: mpconn.Listener) -> None:
        evicted: list[mpconn.Connection] = []
        with self._lock:
            live = self._listener is listener
            if live:
                while len(self._pending) >= MAX_PENDING_HANDSHAKES:
                    evicted.append(self._pending.popitem(last=False)[1])
                self._pending[id(conn)] = conn
        # Closing an evicted peer is what releases its vetting thread.
        for old in evicted:
            with contextlib.suppress(OSError):
                old.close()
        if not live:  # closed while this connection was being accepted
            with contextlib.suppress(OSError):
                conn.close()
            return
        threading.Thread(
            target=self._vet,
            args=(conn, listener),
            name="realmspinner-mcp-vet",
            daemon=True,
        ).start()

    def _vet(self, conn: mpconn.Connection, listener: mpconn.Listener) -> None:
        try:
            passed = self._handshake(conn)
        finally:
            with self._lock:
                self._pending.pop(id(conn), None)
        if not passed:
            return  # `_handshake` already closed it
        with self._lock:
            live = self._listener is listener
            if live:
                self._ready.put(conn)
        if not live:
            with contextlib.suppress(OSError):
                conn.close()

    def accept(self) -> mpconn.Connection | None:
        """The next *authenticated* bridge connection, or `None` once
        `close()` has run (or if the server was never started).

        **A peer that fails the challenge is refused, not fatal.**
        `AuthenticationError` does not inherit from `OSError` -- it is a
        `ProcessError` -- so the first process to dial this pipe with a stale
        or wrong token once raised straight out through the host's accept loop
        and took the listener down for the session; and the exchange also
        raises `EOFError` when a peer disconnects part-way through, which is
        what an MCP client does every time it is killed while its bridge is
        still dialling in. :meth:`_handshake` owns every failure the exchange
        can produce, so a refused peer never reaches this queue at all.
        """
        ready = self._ready
        if self._listener is None or self._acceptor is None:
            return None
        conn = ready.get()
        if conn is None:
            ready.put(None)  # leave the sentinel for any other waiter
        return conn

    def _handshake(self, conn: mpconn.Connection) -> bool:
        """Prove the peer holds the token, within `HANDSHAKE_TIMEOUT`.

        `True` if it did; otherwise the connection is closed and `False` is
        returned. Every failure ends up here rather than at the call site:
        a wrong key (`AuthenticationError`), a peer that vanished mid-exchange
        (`EOFError`), a broken handle (`OSError`), and the silence that has no
        exception at all (the thread is simply still parked).

        **Closing the connection is the whole point of the timeout branch, not
        tidiness.** The abandoned thread is blocked in `recv_bytes` on this
        handle; closing it is what makes that call raise so the thread exits
        and the handle goes with it. Leave it open and the thread lives on
        holding a pipe instance, and the next `start()` fails with
        `PermissionError` -- see the module docstring for the chain that turns
        into an app that will not launch.
        """
        key = self._authkey
        if key is None:  # closed underneath us between accept and here
            with contextlib.suppress(OSError):
                conn.close()
            return False

        done: list[bool] = []

        def exchange() -> None:
            try:
                # The same two calls, in the same order, that
                # `Listener.accept` makes for an authkey-bearing listener;
                # the client half of `connect()` mirrors them. Sequenced here
                # rather than reimplemented -- the HMAC is still stdlib's.
                mpconn.deliver_challenge(conn, key)
                mpconn.answer_challenge(conn, key)
            except BaseException:  # noqa: BLE001 -- a refusal, whatever shape
                # it arrived in; the caller gets one bool and the connection
                # is closed below either way.
                done.append(False)
            else:
                done.append(True)

        # Deliberately not logged, at any level: on a shared machine an
        # unauthenticated probe is noise, and a listener that writes a line
        # per probe is a log-volume lever for anyone who can reach the pipe.
        worker = threading.Thread(target=exchange, name="realmspinner-mcp-handshake", daemon=True)
        worker.start()
        worker.join(HANDSHAKE_TIMEOUT)
        if done and done[0]:
            return True
        with contextlib.suppress(OSError):
            conn.close()
        # Bounded, like every other join in this feature: the close above is
        # what releases the thread, and if some platform ever fails to honour
        # it, a daemon thread that outlives this call is still better than an
        # accept loop that never returns.
        worker.join(HANDSHAKE_TIMEOUT)
        return False

    @property
    def address(self) -> str:
        listener = self._listener
        return listener.address if listener is not None else ""


def connect(home: Path) -> mpconn.Connection:
    """The bridge's side: read the token the app published and dial in,
    bounded by :data:`HANDSHAKE_TIMEOUT`.

    `mpconn.Client`'s own authentication (`answer_challenge`/`deliver_
    challenge`, plus, on Windows, its own retry loop against a busy pipe)
    has no deadline of its own -- unlike the server half, which
    `Server._handshake` already bounds for exactly this reason (see that
    method's docstring). A bridge dialling a Realmspinner that accepted the
    connection but is stuck mid-handshake (or a pipe address that exists
    but nothing is really listening behind any more) would otherwise block
    here forever, with nothing above it (`bridge.main`) able to tell "still
    connecting" from "hung".

    Run on a worker thread and joined with a timeout, the same shape
    `Server._handshake` already uses for its own half of this exchange --
    not reimplemented, mirrored. If the worker has not finished by then,
    there is no connection object to close (`mpconn.Client` has not
    returned one), so this raises `TimeoutError` rather than trying to
    abandon a handle the way the server side does; the worker thread itself
    is left as a daemon, since a `Client()` call this codebase has already
    given up on has nothing further for the caller to wait on.
    """
    token = read_token(home)
    address = address_for(home)

    result: list[mpconn.Connection] = []
    error: list[BaseException] = []

    def dial() -> None:
        try:
            result.append(mpconn.Client(address, family=_FAMILY, authkey=token))
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the caller's
            # own thread below; this thread only carries it across.
            error.append(exc)

    worker = threading.Thread(target=dial, name="realmspinner-mcp-connect", daemon=True)
    worker.start()
    worker.join(HANDSHAKE_TIMEOUT)
    if result:
        return result[0]
    if error:
        raise error[0]
    raise TimeoutError(
        f"Realmspinner did not complete the agent connection handshake within "
        f"{HANDSHAKE_TIMEOUT:.0f}s."
    )
