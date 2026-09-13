"""`warlock mcp` -- the real MCP server, speaking Warlock's private RPC v1 to
Studio and dual-era MCP (`protocol.bridge_dispatch`) to whatever client
dialled this process's stdio.

It is no longer a dumb relay: Studio keeps all per-call state (dedup,
replay, `warlock_status`, transcript, timeouts) behind its RPC v1 `call` op
(`rpc.py`), and this process's job is to negotiate an MCP era with its
stdio peer, fetch and cache the tool catalogue, and translate one MCP
`tools/call` into one RPC v1 `call` -- splicing the raw result bytes back
into an MCP envelope without ever re-parsing them (see
`protocol.splice_tool_result`). `WARLOCK_MCP_RELAY=1` keeps the old dumb
byte-relay behaviour available as an escape hatch (`_relay_main`, below),
in case a client ever depended on it.

**Main thread only, and that is not a style preference.**
`pipelines/_workerio.py` documents a measured Windows deadlock: a daemon
thread parked in a blocking read on an inherited stdin pipe stops the main
thread's very next native-extension import cold, indefinitely. This process
does nothing after start-up that needs a second thread -- it is one blocking
read, one write, repeat -- so the fix is simply to never introduce the thread
that could trip over that landmine in the first place.

**The bridge can outlive Studio, and can outlast Studio not yet having
started at all.** `studio/agent_host.py::_write_catalogue_snapshot` writes
`<home>/mcp.catalogue.json` -- the `catalogue` op's own reply header --
every time Studio's agent server starts, so a bridge dialled before the app
is running, or dialled after the app has closed, can still answer
`initialize`/`server/discover`/`tools/list` from that snapshot rather than
refusing to start. A `tools/call`, though, always needs Studio: `_Session`
tries to (re)connect lazily, on the call that needs it, rather than at
start-up, and a call made while nothing answers gets an `isError` result
naming the same Settings toggle this process has always named on stderr --
never a crash, since the bridge itself is still alive and useful for
discovery. There is still exactly one case that is fatal at start-up: no
snapshot and no reachable Studio, in which case there is nothing to serve
at all and the old exit(1) behaviour stands.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import pipe, protocol, rpc

RPC_VERSIONS = [1]
"""What this bridge sends `hello` -- the RPC integers it understands. See
`rpc.SUPPORTED_RPC_VERSIONS` for Studio's own side of the same negotiation."""

NOT_ACCEPTING_MESSAGE = (
    "Warlock Studio is not accepting agent connections. Open the app "
    "and switch on Settings -> Advanced -> Allow AI agents to drive "
    "the Studio."
)
"""The one sentence this module has always printed to stderr when nothing
answered the pipe at start-up. Reused verbatim as the text of an `isError`
`tools/call` reply once the bridge can stay alive with no Studio reachable
-- same wording, same remedy, whichever surface reports it."""


def _connect(home) -> Any:
    """Dial Studio's pipe, or `None` if nothing answered."""
    try:
        return pipe.connect(home)
    except OSError:
        return None


def _load_snapshot(home: Path) -> dict[str, Any] | None:
    """`<home>/mcp.catalogue.json`, parsed, or `None` if it does not exist or
    does not parse -- the same tolerance `_write_catalogue_snapshot` itself
    already gives a write that fails: a missing or corrupt snapshot is a
    reason to fall through to "no Studio, no snapshot, exit 1", not a reason
    to raise out of `main`."""
    try:
        raw = home.joinpath("mcp.catalogue.json").read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _hello(conn: Any) -> dict[str, Any] | None:
    """Send `hello`, return its header, or `None` (already reported) on a
    version mismatch."""
    request = rpc.encode_request(
        "hello", versions=RPC_VERSIONS, bridge_version=protocol.SERVER_VERSION
    )
    conn.send_bytes(request)
    header, _body = rpc.split_reply(conn.recv_bytes())
    if "error" in header:
        err = header["error"]
        if err.get("code") == "rpc_version":
            print(
                "Warlock Studio speaks a different agent RPC version than this bridge "
                f"understands (bridge: {RPC_VERSIONS}, Studio supports: {err.get('supported')}). "
                "Update Warlock Studio or this bridge so the two match.",
                file=sys.stderr,
            )
        else:
            print(f"Warlock Studio refused the agent connection: {err}", file=sys.stderr)
        return None
    return header


def _fetch_catalogue(conn: Any) -> dict[str, Any]:
    request = rpc.encode_request("catalogue")
    conn.send_bytes(request)
    header, _body = rpc.split_reply(conn.recv_bytes())
    return header


class _Session:
    """Everything one bridge process needs across the whole MCP connection:
    the pipe to Studio (when there is one), the negotiated MCP era, the
    cached catalogue, and Studio's advertised per-call timeout.

    `conn` is `None` whenever this process is not currently attached to
    Studio -- at start-up, serving from the snapshot (:meth:`from_snapshot`),
    or after a disconnect mid-call (:meth:`_disconnect`) -- and every place
    that needs Studio calls :meth:`_ensure_connected` first rather than
    assuming `conn` is live. There is deliberately no background reconnect
    thread: the next `tools/call` from the MCP client is what triggers a
    fresh attempt, and a fresh attempt opens a *new* Studio agent session
    (a new tab, per `AgentHost._serve`), never a resume of whatever the old
    connection was doing.
    """

    def __init__(
        self,
        home: Any,
        conn: Any,
        hello_header: dict[str, Any] | None,
        catalogue: dict[str, Any],
    ) -> None:
        self.home = home
        self.conn = conn
        self.call_timeout = float((hello_header or {}).get("call_timeout", 30.0))
        self.era = protocol.BridgeEra()
        self.catalogue = catalogue
        #: `notifications/tools/list_changed` frames queued by a catalogue
        #: refresh, drained by `main`'s loop after each reply -- legacy era
        #: only (modern clients re-poll `tools/list` themselves, per its own
        #: `ttlMs` cache hint, rather than being pushed a notification).
        self.pending_notifications: list[bytes] = []

    @classmethod
    def connected(cls, home: Any, conn: Any, hello_header: dict[str, Any]) -> _Session:
        """Built from a `hello` that already succeeded -- the ordinary
        start-up path, unchanged."""
        return cls(home, conn, hello_header, _fetch_catalogue(conn))

    @classmethod
    def from_snapshot(cls, home: Any, snapshot: dict[str, Any]) -> _Session:
        """Built with no Studio connection at all, from
        `<home>/mcp.catalogue.json`'s last-saved catalogue -- `initialize`,
        `server/discover` and `tools/list` can all be answered from this
        alone; only a `tools/call` ever needs :meth:`_ensure_connected` to
        actually dial in."""
        return cls(home, None, None, snapshot)

    def _ensure_connected(self) -> bool:
        """Dial Studio now if this session is not already attached. `True`
        if a live connection exists by the time this returns (whether it was
        already there or was just made); `False` if Studio still cannot be
        reached, in which case `conn` stays `None`.

        On a fresh connection, `hello`'s `catalogue_hash` is compared
        against what this session last served: a mismatch means Studio's
        catalogue moved on since the snapshot (or since the previous
        connection) and :meth:`_maybe_refresh_catalogue` re-fetches it and,
        for a legacy-era client, queues the notification that tells it so.
        """
        if self.conn is not None:
            return True
        conn = _connect(self.home)
        if conn is None:
            return False
        header = _hello(conn)
        if header is None:
            with contextlib.suppress(OSError):
                conn.close()
            return False
        self.conn = conn
        self.call_timeout = float(header.get("call_timeout", 30.0))
        self._maybe_refresh_catalogue(header.get("catalogue_hash", ""))
        return True

    def _disconnect(self) -> None:
        """Close and drop the connection to Studio, if there is one. Leaves
        this session serving discovery from whatever catalogue it last
        held; the next `tools/call` is what tries to reconnect."""
        if self.conn is not None:
            with contextlib.suppress(OSError):
                self.conn.close()
        self.conn = None

    def call_tool(self, name: str, arguments: dict[str, Any]) -> bytes:
        """The `call_tool` callback `protocol.bridge_dispatch` invokes for a
        `tools/call`. Three ways this can end without ever reaching Studio's
        real answer, and each gets its own honest `isError` text rather than
        a dropped connection or a silent retry:

        * Studio cannot be reached at all (nothing was listening, and a
          fresh :meth:`_ensure_connected` attempt failed too) -- the same
          "not accepting agent connections" remedy this module has always
          printed to stderr at start-up.
        * The reply outran `call_timeout + 5s` on `conn.poll` -- Studio may
          or may not have run the call; lockstep framing means this process
          cannot tell which without reading a reply that might never come.
        * The pipe raised `EOFError`/`OSError` while sending the request or
          waiting for/reading the reply -- Studio's process (or the pipe
          itself) went away mid-call, same "may or may not have run"
          uncertainty.

        All three close/discard the connection (nothing left to trust it
        with) and leave this session ready to reconnect lazily on the next
        call -- this process never exits over a lost or timed-out call.
        """
        if not self._ensure_connected():
            result = protocol.fail(NOT_ACCEPTING_MESSAGE, recovery="read_scene")
            return json.dumps(result, separators=(",", ":")).encode("utf-8")

        request = rpc.encode_request("call", tool=name, args=arguments)
        try:
            self.conn.send_bytes(request)
            if not self.conn.poll(self.call_timeout + 5.0):
                result = protocol.fail(
                    "Warlock Studio did not answer this call in time. It may or may not "
                    "have run -- re-read the scene (e.g. clay_scene) before trying again "
                    "rather than assuming either way. The next call will open a new "
                    "Studio agent session, not resume this one.",
                    recovery="read_scene",
                )
                self._disconnect()
                return json.dumps(result, separators=(",", ":")).encode("utf-8")
            header, body = rpc.split_reply(self.conn.recv_bytes())
        except (EOFError, OSError):
            self._disconnect()
            result = protocol.fail(
                "Warlock Studio's connection to this bridge was lost while this call "
                "was in flight. It may or may not have run on Studio's side -- re-read "
                "the scene (e.g. clay_scene) before retrying rather than assuming "
                "either way. The next call will open a new Studio agent session, not "
                "resume this one.",
                recovery="read_scene",
            )
            return json.dumps(result, separators=(",", ":")).encode("utf-8")

        new_hash = header.get("hash", "")
        self._maybe_refresh_catalogue(new_hash)
        return body

    def _maybe_refresh_catalogue(self, hash_: str) -> None:
        if not hash_ or hash_ == self.catalogue.get("hash"):
            return
        self.catalogue = _fetch_catalogue(self.conn)
        if self.era.era == "legacy":
            self.pending_notifications.append(_notify_tools_changed())


def _notify_tools_changed() -> bytes:
    return protocol.encode({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})


def main(argv: Sequence[str] | None = None) -> int:
    """The MCP server: negotiate an era with stdin/stdout, translate every
    request into Studio's RPC v1 and back. `argv` is accepted and ignored --
    `warlock mcp` takes no flags today, and the parameter exists so a test
    can call this the same way `cli.main` does, argv and all.
    """
    if os.environ.get("WARLOCK_MCP_RELAY") == "1":
        return _relay_main()

    from ..config import get_config

    home = get_config().home
    conn = _connect(home)
    if conn is not None:
        header = _hello(conn)
        if header is None:
            # A version mismatch: _hello has already printed the specific
            # reason to stderr, and unlike "nothing was listening" this is a
            # real disagreement a snapshot cannot paper over.
            conn.close()
            return 1
        protocol.SERVER_VERSION = header.get("studio_version", protocol.SERVER_VERSION)
        session = _Session.connected(home, conn, header)
    else:
        snapshot = _load_snapshot(home)
        if snapshot is None:
            print(NOT_ACCEPTING_MESSAGE, file=sys.stderr)
            return 1
        session = _Session.from_snapshot(home, snapshot)

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    try:
        while True:
            line = stdin.readline(protocol.MAX_FRAME + 1)
            if not line:
                break
            if len(line) > protocol.MAX_FRAME:
                # An over-long line: refuse it, then discard whatever is left
                # of it up to the next '\n' (bounded the same way, so a
                # confused or hostile peer cannot force an unbounded read
                # here either) or EOF, so the next readline() starts at the
                # next real line rather than mid-way through this one.
                while line and not line.endswith(b"\n"):
                    line = stdin.readline(protocol.MAX_FRAME + 1)
                stdout.write(
                    protocol.encode(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32600, "message": "request exceeds MAX_FRAME"},
                        }
                    )
                )
                stdout.flush()
                continue
            reply = protocol.bridge_dispatch(
                line, session.era, catalogue=session.catalogue, call_tool=session.call_tool
            )
            if reply is not None:
                stdout.write(reply)
            for notification in session.pending_notifications:
                stdout.write(notification)
            session.pending_notifications.clear()
            stdout.flush()
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        session._disconnect()
    return 0


def _relay_main() -> int:
    """The pre-RPC behaviour, kept verbatim as an escape hatch
    (`WARLOCK_MCP_RELAY=1`): every byte read from stdin goes straight to
    Studio's pipe via `send_bytes`, and every reply comes straight back to
    stdout, with no JSON parsed and no method names known here. This only
    works against a Studio old enough to still answer bare MCP JSON-RPC on
    the pipe directly, without RPC v1's `hello`/`catalogue`/`call` envelope."""
    from ..config import get_config

    home = get_config().home
    try:
        conn = pipe.connect(home)
    except OSError:
        print(
            "Warlock Studio is not accepting agent connections. Open the app "
            "and switch on Settings -> Advanced -> Allow AI agents to drive "
            "the Studio.",
            file=sys.stderr,
        )
        return 1

    try:
        stdin = sys.stdin.buffer
        stdout = sys.stdout.buffer
        while True:
            line = stdin.readline()
            if not line:
                break
            conn.send_bytes(line)
            reply = conn.recv_bytes()
            if not reply:
                continue
            stdout.write(reply)
            stdout.flush()
    except EOFError:
        pass
    finally:
        conn.close()
    return 0
