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
"""

from __future__ import annotations

import contextlib
import hashlib
import multiprocessing.connection as mpconn
import os
import secrets
import sys
from pathlib import Path

_FAMILY = "AF_PIPE" if sys.platform == "win32" else "AF_UNIX"


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


class Server:
    """The app's side of the pipe: one listener, one connection at a time."""

    def __init__(self, home: Path) -> None:
        self._home = home
        self._listener: mpconn.Listener | None = None

    def start(self) -> None:
        authkey = write_token(self._home)
        self._listener = mpconn.Listener(address_for(self._home), family=_FAMILY, authkey=authkey)

    def close(self) -> None:
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        # Cleared even if start() was never called: idempotent close is what
        # lets AgentHost.stop() be idempotent too, per its own contract.
        clear_token(self._home)

    def accept(self) -> mpconn.Connection | None:
        """The next bridge connection, or `None` once `close()` has run.

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
        """
        while self._listener is not None:
            try:
                return self._listener.accept()
            except mpconn.AuthenticationError:
                # Deliberately not logged at warning: on a shared machine an
                # unauthenticated probe is noise, and a listener that writes a
                # line per probe is a log-volume lever for anyone who can
                # reach the pipe.
                continue
            except OSError:
                return None
        return None

    @property
    def address(self) -> str:
        return self._listener.address if self._listener is not None else ""


def connect(home: Path) -> mpconn.Connection:
    """The bridge's side: read the token the app published and dial in."""
    token = read_token(home)
    return mpconn.Client(address_for(home), family=_FAMILY, authkey=token)
