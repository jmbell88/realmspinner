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
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from typing import Any

from . import pipe, protocol, rpc

RPC_VERSIONS = [1]
"""What this bridge sends `hello` -- the RPC integers it understands. See
`rpc.SUPPORTED_RPC_VERSIONS` for Studio's own side of the same negotiation."""


def _connect(home) -> Any:
    """Dial Studio's pipe, or print the same "not accepting connections"
    message this module has always printed and let the caller exit(1).

    Serving straight from `<home>/mcp.catalogue.json` when nothing answers
    is a later change (the snapshot already exists for it -- see
    `studio/agent_host.py::_write_catalogue_snapshot` -- but this bridge does
    not read it yet); today, as before, no app means no server.
    """
    try:
        return pipe.connect(home)
    except OSError:
        return None


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
    the pipe to Studio, the negotiated MCP era, the cached catalogue, and
    Studio's advertised per-call timeout."""

    def __init__(self, conn: Any, hello_header: dict[str, Any]) -> None:
        self.conn = conn
        self.call_timeout = float(hello_header.get("call_timeout", 30.0))
        self.era = protocol.BridgeEra()
        self.catalogue = _fetch_catalogue(conn)
        #: `notifications/tools/list_changed` frames queued by a catalogue
        #: refresh, drained by `main`'s loop after each reply -- legacy era
        #: only (modern clients re-poll `tools/list` themselves, per its own
        #: `ttlMs` cache hint, rather than being pushed a notification).
        self.pending_notifications: list[bytes] = []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> bytes:
        """The `call_tool` callback `protocol.bridge_dispatch` invokes for a
        `tools/call`. Backstopped by `call_timeout + 5s` on the wait for
        Studio's reply (`conn.poll`, not an unbounded `recv_bytes`): a call
        that outran that window may or may not have run on Studio's side --
        lockstep framing means this process cannot tell which without
        reading a reply that might never come -- so the honest answer is an
        `isError` tool result saying so, and then the connection to Studio
        is closed and this process exits, since the two sides can no longer
        agree whose turn it is to speak.
        """
        request = rpc.encode_request("call", tool=name, args=arguments)
        self.conn.send_bytes(request)
        if not self.conn.poll(self.call_timeout + 5.0):
            result = protocol.fail(
                "Warlock Studio did not answer this call in time. It may or may not have run -- "
                "re-read the scene before trying again rather than assuming either way.",
                recovery="read_scene",
            )
            body = json.dumps(result, separators=(",", ":")).encode("utf-8")
            self._give_up()
            return body
        header, body = rpc.split_reply(self.conn.recv_bytes())
        new_hash = header.get("hash", "")
        self._maybe_refresh_catalogue(new_hash)
        return body

    def _maybe_refresh_catalogue(self, hash_: str) -> None:
        if not hash_ or hash_ == self.catalogue.get("hash"):
            return
        self.catalogue = _fetch_catalogue(self.conn)
        if self.era.era == "legacy":
            self.pending_notifications.append(_notify_tools_changed())

    def _give_up(self) -> None:
        """A timed-out call to Studio leaves this process unable to trust
        the lockstep framing any further -- the reply it gave up waiting on
        may still arrive and be misread as the answer to whatever request
        comes next. Closing the connection and exiting is the only response
        that cannot desynchronise the two sides."""
        import contextlib

        with contextlib.suppress(OSError):
            self.conn.close()
        sys.exit(1)


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
    if conn is None:
        print(
            "Warlock Studio is not accepting agent connections. Open the app "
            "and switch on Settings -> Advanced -> Allow AI agents to drive "
            "the Studio.",
            file=sys.stderr,
        )
        return 1

    header = _hello(conn)
    if header is None:
        conn.close()
        return 1
    protocol.SERVER_VERSION = header.get("studio_version", protocol.SERVER_VERSION)

    session = _Session(conn, header)

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
        conn.close()
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
