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

**One connection at a time is the v1 decision, not an oversight.** An agent
session is exclusive use of the Studio it is driving -- there is no sensible
way to interleave two agents' tool calls against one `ClayTab` -- so a second
`connect()` simply waits inside the app's `accept()` until the first bridge
disconnects. Multiplexing, if it is ever needed, is a v2 problem.

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
"""

from __future__ import annotations

import contextlib
import hashlib
import multiprocessing.connection as mpconn
import os
import secrets
import sys
import threading
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


def address_for(home: Path) -> str:
    """The pipe/socket address for a given `WARLOCK_HOME`.

    Derived from `home` rather than fixed, because more than one `WARLOCK_HOME`
    can be live on one machine (tests pin a throwaway one; `WARLOCK_TRELLIS_EXE`-
    style overrides exist precisely so two checkouts can coexist) and two
    Studios listening on the same pipe name would race each other's `accept()`.
    Hashed rather than used verbatim because a Windows pipe name has to be a
    single path segment -- `home` itself may contain `\\`, `:`, anything a
    filesystem allows.
    """
    if sys.platform == "win32":
        digest = hashlib.sha256(str(home).lower().encode("utf-8")).hexdigest()[:16]
        return "\\\\.\\pipe\\warlock-mcp-" + digest
    return str(home / "mcp.sock")


def token_path(home: Path) -> Path:
    return home / "mcp.token"


def write_token(home: Path) -> bytes:
    """Generate a fresh 32-byte authkey and publish it for the bridge to read.

    Staged beside the destination and landed with `os.replace` -- this repo's
    rule for every write onto a name another process reads, here because a
    bridge that opens `mcp.token` mid-write must never see a truncated or
    half-hex-encoded secret. The permission bit is set on the *staging* file
    before the rename, not after, so the final name is never briefly
    world-readable; `os.chmod` failing (there is no POSIX-style mode bit to
    set on Windows) is not fatal, since the authkey challenge inside
    `Listener`/`Client` is what actually gates the connection -- file
    permissions are defence in depth, not the mechanism.
    """
    home.mkdir(parents=True, exist_ok=True)
    token = secrets.token_bytes(32)
    dest = token_path(home)
    tmp = dest.with_name(f".{dest.name}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_text(token.hex(), encoding="ascii")
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
            "No agent session is open: Warlock Studio has not started an MCP "
            "server for this home directory."
        ) from exc
    return bytes.fromhex(raw.strip())


def clear_token(home: Path) -> None:
    token_path(home).unlink(missing_ok=True)


def _clear_stale_socket(home: Path) -> None:
    """Remove a Unix socket left behind by a Warlock that did not close it.

    Windows named pipes die with the process that made them; an `AF_UNIX`
    socket is a file that outlives one, and `bind` refuses an address whose
    file already exists. So on POSIX a hard kill (or any exit that skipped
    `Server.close`) left `mcp.sock` on disk and every later `start()` raised
    `EADDRINUSE` -- permanently, since nothing in the app ever removed it. The
    agent server could not be switched on again for the life of that home
    without someone deleting the file by hand.

    Unlinking it is safe because of `instance.py`: one Warlock holds the lock
    for a home at a time, so a socket file here cannot belong to a live
    Studio. It is a leftover by construction, and treating it as one is what
    makes the feature recoverable rather than one crash away from dead. No-op
    on Windows, where the address is a pipe name and not a path at all.
    """
    if sys.platform == "win32":
        return
    with contextlib.suppress(OSError):
        Path(address_for(home)).unlink(missing_ok=True)


class Server:
    """The app's side of the pipe: one listener, one connection at a time."""

    def __init__(self, home: Path) -> None:
        self._home = home
        self._listener: mpconn.Listener | None = None
        # Held rather than handed to the Listener: this class runs the
        # challenge itself, on a clock -- see the module docstring.
        self._authkey: bytes | None = None

    def start(self) -> None:
        authkey = write_token(self._home)
        _clear_stale_socket(self._home)
        self._authkey = authkey
        self._listener = mpconn.Listener(address_for(self._home), family=_FAMILY)

    def close(self) -> None:
        if self._listener is not None:
            address = self._listener.address
            self._listener.close()
            self._listener = None
            if sys.platform == "win32":
                # The mid-handshake peer (module docstring above) is not the
                # only way `accept()` gets stuck: if *no* peer has ever
                # dialled in, the listener thread is parked inside
                # `ConnectNamedPipe`'s wait, and closing the handle from this
                # thread does not reliably release that wait the way it does
                # once a first connection has already been accepted once.
                # Measured: 5/5 trials left the thread alive past `stop()`'s
                # own join, holding the pipe's `FILE_FLAG_FIRST_PIPE_INSTANCE`
                # handle, so the next `start()` failed with `PermissionError`
                # -- the same failure shape the docstring above describes,
                # from a trigger it does not cover. Dialling in ourselves
                # completes the pending `ConnectNamedPipe`, so the blocked
                # `accept()` call returns instead of hanging until the
                # process exits. `self._authkey` is already cleared below by
                # the time this connection reaches `_handshake` in the
                # listener thread, so it is rejected there and never mistaken
                # for a real bridge.
                with contextlib.suppress(Exception):
                    mpconn.Client(address, family=_FAMILY).close()
        self._authkey = None
        # Cleared even if start() was never called: idempotent close is what
        # lets AgentHost.stop() be idempotent too, per its own contract.
        clear_token(self._home)

    def accept(self) -> mpconn.Connection | None:
        """The next *authenticated* bridge connection, or `None` once
        `close()` has run.

        A `Listener.accept()` blocked in another thread raises `OSError` (not
        some clean sentinel) when `close()` tears down the underlying handle
        from under it, on both platforms -- so `None` here covers the normal
        "we are shutting down" case as well as "never started".

        **A peer that fails the challenge is refused, not fatal, and the loop
        is what makes that true.** `AuthenticationError` does not inherit from
        `OSError` -- it is a `ProcessError` -- so an `except OSError` around
        `accept()` does not catch it, and the first process to dial this pipe
        with a stale or wrong token would otherwise have raised straight out
        through the host's accept loop and taken the listener down for the
        session. One bad guess must not be a way to switch the feature off:
        a rejected peer is dropped and the next `accept()` is issued here,
        which is also why this returns a connection rather than a status --
        the caller has nothing to decide.

        **That guard was written for this hazard and named only two of its
        shapes.** The exchange also raises `EOFError` -- neither an
        `OSError` nor a `ProcessError` -- when a peer disconnects part-way
        through it, which is what an MCP client does every time it is killed
        or restarted while its bridge is still dialling in. Measured, that
        escaped here and ended the listener thread, leaving Settings still
        showing the pipe address for a server that had stopped answering.
        :meth:`_handshake` now owns every failure the exchange can produce,
        so this loop sees one bool and cannot be surprised by a third
        exception type the next CPython invents.
        """
        while self._listener is not None:
            try:
                conn = self._listener.accept()
            except OSError:
                return None
            if self._handshake(conn):
                return conn
            # Refused, and already closed -- take the next peer.

        return None

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
        worker = threading.Thread(target=exchange, name="warlock-mcp-handshake", daemon=True)
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
        return self._listener.address if self._listener is not None else ""


def connect(home: Path) -> mpconn.Connection:
    """The bridge's side: read the token the app published and dial in."""
    token = read_token(home)
    return mpconn.Client(address_for(home), family=_FAMILY, authkey=token)
