"""`warlock mcp` -- `bridge.main`'s dumb byte relay between an agent's stdio
and the app's pipe.

Nothing here parses JSON: `bridge.py`'s own docstring is explicit that all the
MCP semantics live in `protocol.py`, which runs inside the app. What is
pinned here is the relay's two visible behaviours -- a readable remedy on
stderr when nothing is listening, and an exact byte round trip when something
is -- plus the one asymmetry that matters: a zero-length reply (the
notification case) is never forwarded to stdout, because JSON-RPC forbids a
reply to a notification and writing an empty line would corrupt the framing
for whatever reads stdout next.
"""

from __future__ import annotations

import io
import sys
import threading
from types import SimpleNamespace

import pytest

from warlock.mcp import bridge, pipe, protocol


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway `WARLOCK_HOME` this file's `get_config().home` resolves to.

    `bridge.main` reads ``home`` off ``config.get_config()``, so the session's
    own `WARLOCK_HOME` pin (see ``tests/conftest.py``) is not enough on its
    own here -- each test in this file wants its *own* pipe address, not the
    one every other test in the suite already shares.
    """
    import warlock.config as config_mod

    monkeypatch.setenv("WARLOCK_HOME", str(tmp_path))
    monkeypatch.setattr(config_mod, "_config", None)
    return tmp_path


# --- nothing listening -----------------------------------------------------------


def test_with_nothing_listening_main_returns_1_and_names_the_settings_toggle(
    home, capsys
) -> None:
    """This sentence is the entire remedy a user gets, so it has to name the
    exact control that fixes it -- not just say "connection failed"."""
    assert bridge.main([]) == 1
    err = capsys.readouterr().err
    assert "Settings" in err
    assert "Advanced" in err
    assert "Allow AI agents to drive the Studio" in err


def test_with_a_token_but_no_listener_main_still_returns_1(home, capsys) -> None:
    """`read_token` succeeding is not enough either: a session that started and
    then quit leaves the token file behind (nothing clears it but `Server.close`
    running to completion), and `mpconn.Client` against a dead address is the
    same "not accepting connections" outcome as no token at all."""
    pipe.write_token(home)
    assert bridge.main([]) == 1
    assert "not accepting agent connections" in capsys.readouterr().err


# --- a real round trip -----------------------------------------------------------


def _serve_one_message(server: pipe.Server, *, reply: bytes, request_box: dict) -> threading.Thread:
    """Accept one connection, capture the one request it sends, answer it, and
    close -- the app side of the relay, standing in for `agent_host.py`."""

    def run() -> None:
        conn = server.accept()
        assert conn is not None
        assert conn.poll(10)
        request_box["request"] = conn.recv_bytes()
        conn.send_bytes(reply)
        conn.close()

    thread = threading.Thread(target=run)
    thread.start()
    return thread


def _patch_stdio(monkeypatch, stdin_bytes: bytes) -> io.BytesIO:
    """`bridge.main` only ever touches `.buffer`, so a `SimpleNamespace`
    standing in for the whole stream is enough -- and safer than patching the
    real `sys.stdin`'s `.buffer`, which is a read-only attribute on a real
    `TextIOWrapper`."""
    stdin = io.BytesIO(stdin_bytes)
    stdout = io.BytesIO()
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=stdin))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=stdout))
    return stdout


def test_a_request_goes_out_and_the_reply_comes_back_on_stdout(home, monkeypatch) -> None:
    request = protocol.encode({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    canned_reply = protocol.encode({"jsonrpc": "2.0", "id": 1, "result": {}})

    server = pipe.Server(home)
    server.start()
    request_box: dict = {}
    server_thread = _serve_one_message(server, reply=canned_reply, request_box=request_box)
    stdout = _patch_stdio(monkeypatch, request)
    try:
        assert bridge.main([]) == 0
    finally:
        server_thread.join(timeout=10)
        server.close()

    assert request_box["request"] == request, "the exact bytes read from stdin went out unmodified"
    assert stdout.getvalue() == canned_reply, "the exact bytes back from the pipe landed on stdout"


def test_a_zero_length_reply_writes_nothing_to_stdout(home, monkeypatch) -> None:
    """The notification case: the app answers with an empty frame and the
    bridge must not turn that into a blank line on stdout."""
    request = protocol.encode({"jsonrpc": "2.0", "method": "notifications/initialized"})

    server = pipe.Server(home)
    server.start()
    request_box: dict = {}
    server_thread = _serve_one_message(server, reply=b"", request_box=request_box)
    stdout = _patch_stdio(monkeypatch, request)
    try:
        assert bridge.main([]) == 0
    finally:
        server_thread.join(timeout=10)
        server.close()

    assert stdout.getvalue() == b""
