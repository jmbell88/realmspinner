"""Regression tests for the 2026-09-26 audit's `mcp/` findings owned by wave
3, fixer 8: `agents-host-03` (`bridge.py`), `agents-protocol-01` (`protocol.py`),
and `agents-protocol-02`/`03`/`04`/`05` (`pipe.py`).
"""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from realmspinner.mcp import bridge, pipe, protocol, rpc

# =============================================================================
# agents-host-03: bridge.py -- a catalogue-refresh failure must not discard a
# real, already-successful tool result.
# =============================================================================


class _ConnDiesAfterFirstReply:
    """A duck-typed `Connection` stand-in that answers exactly one
    request/reply round trip cleanly and then raises `OSError` on every
    `send_bytes`/`recv_bytes` call after -- the pipe dying right after the
    real tool call's own reply arrived, before the catalogue re-fetch that
    reply's changed hash triggers gets a chance to run."""

    def __init__(self, first_reply: bytes) -> None:
        self._first_reply = first_reply
        self._replied = False
        self.sent: list[bytes] = []

    def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)
        if self._replied:
            raise OSError("pipe gone (simulated): the catalogue refresh's own send")

    def poll(self, timeout: float | None = None) -> bool:
        return True

    def recv_bytes(self, maxlength=None) -> bytes:  # noqa: ARG002
        if not self._replied:
            self._replied = True
            return self._first_reply
        raise OSError("pipe gone (simulated): the catalogue refresh's own recv")

    def close(self) -> None:
        pass


def test_call_tool_returns_the_real_result_even_when_the_catalogue_refresh_dies(
    tmp_path,
) -> None:
    """The 2026-09-26 audit (agents-host-03): `call_tool` used to run
    `_maybe_refresh_catalogue` outside its own try, *after* the real call had
    already succeeded -- so a pipe failure inside that refresh (its own
    `_fetch_catalogue` doing an unguarded `send_bytes`/`recv_bytes`) raised
    straight out of `call_tool`, discarding the already-successful `body` in
    favour of whatever a bare, uncaught exception turns into further up the
    stack, instead of the honest result the real call produced.
    """
    reply = rpc.encode_reply({"hash": "new-hash"}, b'{"isError":false,"content":[]}')
    conn = _ConnDiesAfterFirstReply(reply)
    session = bridge._Session(tmp_path, conn, {"call_timeout": 5.0}, {"hash": "old-hash"})

    body = session.call_tool("some_tool", {})

    assert body == b'{"isError":false,"content":[]}', (
        "the real, already-successful tool result was discarded when the "
        "catalogue refresh that followed it failed"
    )
    # The dead connection is dropped so the next call reconnects -- the same
    # response `_maybe_refresh_catalogue` already gives a timed-out refresh.
    assert session.conn is None


# =============================================================================
# agents-protocol-01: protocol.py -- an id of NaN/Infinity must not come back
# out as invalid JSON.
# =============================================================================


@pytest.mark.parametrize("bad_id", [float("nan"), float("inf"), float("-inf")])
def test_a_request_id_of_nan_or_infinity_is_not_echoed_back_as_invalid_json(bad_id) -> None:
    """The 2026-09-26 audit (agents-protocol-01): `json.loads` accepts the
    bare `NaN`/`Infinity`/`-Infinity` tokens as a Python-specific extension,
    and `json.dumps` writes them straight back out by default -- so a
    request whose `id` was one of those parsed clean and then went right
    back onto the wire the same way, in `_error_bytes`'s `{"id": msg_id,
    ...}` literal. That is not valid JSON per RFC 8259 (which defines
    neither token), and a client with a standards-strict parser could not
    even read the reply naming its own request back to it.
    """
    state = protocol.BridgeEra()
    reply = protocol._dispatch_one(
        {"jsonrpc": "2.0", "id": bad_id, "method": "", "params": {}},
        state,
        catalogue={"hash": "h", "tools": []},
        call_tool=lambda name, args: b"{}",  # noqa: ARG005
    )
    assert reply is not None
    text = reply.decode("utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    # And it really is standards-valid JSON: strict `json.loads` (no
    # `parse_constant` hook installed) must not choke on it, whereas it
    # happily accepts the very tokens this reply must no longer contain.
    parsed = json.loads(text)
    assert parsed["error"]["code"] == -32600


# =============================================================================
# agents-protocol-02: pipe.py -- accept() must not re-read a cleared listener.
# =============================================================================


class _FakeConn:
    def close(self) -> None:
        pass


class _StubListener:
    """Stands in for the real `mpconn.Listener`. `.accept()` must only ever
    be reached by code that captured `self._listener` into a local variable
    *before* calling it -- see `_RacyListenerServer`'s own docstring for
    exactly what race this reproduces."""

    def __init__(self) -> None:
        self.accept_calls = 0

    def accept(self) -> _FakeConn:
        self.accept_calls += 1
        return _FakeConn()


class _RacyListenerServer(pipe.Server):
    """A `Server` whose `_listener` attribute answers its first read with the
    real (stub) listener and every read after with `None` -- reproducing,
    deterministically and with no real thread timing, `close()` landing in
    the gap between `accept()`'s own `while self._listener is not None:`
    guard (read #1) and the very next statement's separate, second read of
    the same name to call `.accept()` on it (read #2). The 2026-09-26 audit
    (agents-protocol-02) found that gap real: a GIL switch between two
    statements is not contrived, and a `close()` running in it sets
    `self._listener = None` right after the guard already passed, so the
    old code's second read saw `None` and `None.accept()` raised
    `AttributeError` -- not the `OSError` the next line is written to catch
    -- ending the accept loop's thread outright.
    """

    def __init__(self, home) -> None:
        super().__init__(home)
        self._racy_reads = 0

    @property
    def _listener(self):
        self._racy_reads += 1
        return self._racy_stub if self._racy_reads == 1 else None

    @_listener.setter
    def _listener(self, value) -> None:
        self._racy_stub = value


def test_accept_survives_the_listener_being_cleared_between_its_guard_and_its_dereference(
    tmp_path,
) -> None:
    server = _RacyListenerServer(tmp_path)
    stub = _StubListener()
    server._listener = stub
    # The fast `_handshake` reject path (`key is None`), so the stub
    # connection this test hands back is dropped cleanly rather than this
    # test also having to fake out a real crypto exchange.
    server._authkey = None

    try:
        result = server.accept()
    except AttributeError as exc:  # pragma: no cover -- the bug this pins
        pytest.fail(
            f"accept() re-read self._listener after it was cleared and "
            f"crashed calling .accept() on None: {exc!r}"
        )

    assert result is None
    assert stub.accept_calls == 1, (
        "accept() must read self._listener exactly once per iteration and "
        "call .accept() on that same reference, never re-read the attribute"
    )


# =============================================================================
# agents-protocol-03: pipe.py -- the token file must never exist at a wider
# mode than 0o600, not even for one syscall.
# =============================================================================


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file mode bits do not apply on Windows")
def test_write_token_never_creates_the_secret_file_group_or_other_readable(
    tmp_path, monkeypatch
) -> None:
    """The 2026-09-26 audit (agents-protocol-03): `write_token` used to
    `tmp.write_text(...)` -- creating the file at the process's default,
    umask-derived mode (typically group/other readable) -- and only *then*
    `os.chmod` it down to `0o600`, leaving the 32-byte secret sitting on disk
    at a wider mode for the gap between those two calls. Caught here by
    wrapping `os.open` (what `write_text` uses under the hood, and what the
    fix now calls directly) to record the mode the OS actually applied at
    *creation* time, before any later `chmod` in the same function gets a
    chance to run.
    """
    observed_modes: list[int] = []
    real_open = os.open

    def spying_open(path, flags, mode=0o777):
        fd = real_open(path, flags, mode)
        if flags & os.O_CREAT:
            observed_modes.append(stat.S_IMODE(os.fstat(fd).st_mode))
        return fd

    monkeypatch.setattr(pipe.os, "open", spying_open)

    pipe.write_token(tmp_path)

    assert observed_modes, "write_token() never created the token file through os.open"
    for mode in observed_modes:
        assert mode & 0o077 == 0, (
            f"the secret file existed at mode {oct(mode)} right after creation -- "
            "group/other bits must never be set, not even for one syscall"
        )


# =============================================================================
# agents-protocol-04: pipe.py -- a failed Listener construction must not
# leave a stale token file behind.
# =============================================================================


def test_a_listener_construction_failure_does_not_leave_a_stale_token(
    tmp_path, monkeypatch
) -> None:
    """The 2026-09-26 audit (agents-protocol-04): `Server.start` wrote
    `mcp.token` *before* constructing the `Listener`; a construction that
    then failed (a busy address, a permissions problem) left that token
    file behind, advertising a working secret for a pipe nobody was
    listening on.
    """

    def boom(*_a, **_kw):
        raise OSError("address already in use (simulated)")

    monkeypatch.setattr(pipe.mpconn, "Listener", boom)
    server = pipe.Server(tmp_path)

    with pytest.raises(OSError):
        server.start()

    assert not pipe.token_path(tmp_path).exists(), (
        "start() published mcp.token before constructing the Listener and "
        "never cleaned it up when construction failed"
    )


# =============================================================================
# agents-protocol-05: pipe.py -- close() must clear _authkey before the
# win32 self-dial, not after it.
# =============================================================================


def test_close_clears_authkey_before_the_win32_self_dial_not_after(
    tmp_path, monkeypatch
) -> None:
    """The 2026-09-26 audit (agents-protocol-05): the comment beside the
    win32 self-dial claimed `self._authkey` "is already cleared below by the
    time this connection reaches `_handshake`" -- but the assignment that
    clears it ran *after* the dial, not before. `_handshake`'s fast reject
    path ("key is None: closed underneath us") only fires once this is
    genuinely `None`; cleared after, the listener thread's own concurrent
    `_handshake` call could still read the live key and attempt the full
    crypto exchange against a self-dial that never answers it. Forced onto
    the win32 branch via `monkeypatch` so this reproduces on any platform.
    """
    monkeypatch.setattr(pipe.sys, "platform", "win32")

    class _FakeListener:
        address = "fake-pipe-address"

        def close(self) -> None:
            pass

    server = pipe.Server(tmp_path)
    server._listener = _FakeListener()
    server._authkey = b"\x01" * 32

    seen_authkey_at_dial: list[bytes | None] = []

    class _FakeSelfDialConn:
        def close(self) -> None:
            pass

    def fake_client(address, family=None):  # noqa: ARG001
        seen_authkey_at_dial.append(server._authkey)
        return _FakeSelfDialConn()

    monkeypatch.setattr(pipe.mpconn, "Client", fake_client)

    server.close()

    assert seen_authkey_at_dial == [None], (
        f"close() dialled itself while self._authkey was {seen_authkey_at_dial!r}, "
        "not None -- the comment claims the key 'is already cleared' at the "
        "self-dial, but the assignment used to run after it"
    )
