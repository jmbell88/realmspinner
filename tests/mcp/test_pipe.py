"""The transport under MCP: a token-gated named pipe (Windows) / Unix socket.

`pipe.py`'s own docstring names the one rule everything else here defends:
never `send`/`recv` (those pickle, and a pickle off a local channel is an
arbitrary-code door) and never let a connection through without proving the
peer holds the token `write_token` published. The tests below pin the token
lifecycle, the address derivation, a real byte round trip over the platform's
actual pipe, and -- the security claim -- that a client dialling in with the
wrong authkey is refused rather than let through.
"""

from __future__ import annotations

import contextlib
import multiprocessing.connection as mpconn
import stat
import sys
import threading
from typing import Any

import pytest

from warlock.mcp import pipe

# --- the token -----------------------------------------------------------------


def test_write_token_creates_the_file_and_a_subsequent_read_gets_the_same_bytes(
    tmp_path,
) -> None:
    authkey = pipe.write_token(tmp_path)
    assert pipe.token_path(tmp_path).exists()
    assert pipe.read_token(tmp_path) == authkey


def test_write_token_returns_thirty_two_bytes(tmp_path) -> None:
    assert len(pipe.write_token(tmp_path)) == 32


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file mode bits do not apply on Windows")
def test_the_token_file_is_written_owner_read_write_only_on_posix(tmp_path) -> None:
    pipe.write_token(tmp_path)
    mode = stat.S_IMODE(pipe.token_path(tmp_path).stat().st_mode)
    assert mode == 0o600


def test_read_token_on_a_home_with_no_token_raises_a_readable_sentence(tmp_path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        pipe.read_token(tmp_path)
    message = str(excinfo.value)
    # A readable sentence, not a bare "[Errno 2] No such file or directory: ..."
    # traceback fragment -- this is the message a user-facing caller like
    # bridge.main would otherwise have to translate itself.
    assert message.endswith(".")
    assert "No agent session is open" in message


def test_clear_token_removes_it_and_is_idempotent(tmp_path) -> None:
    pipe.write_token(tmp_path)
    pipe.clear_token(tmp_path)
    assert not pipe.token_path(tmp_path).exists()
    pipe.clear_token(tmp_path)  # a second call on an already-clear home does not raise


# --- address derivation ----------------------------------------------------------


def test_address_for_is_stable_for_one_home(tmp_path) -> None:
    assert pipe.address_for(tmp_path) == pipe.address_for(tmp_path)


def test_address_for_differs_between_two_homes(tmp_path_factory) -> None:
    a = tmp_path_factory.mktemp("home-a")
    b = tmp_path_factory.mktemp("home-b")
    assert pipe.address_for(a) != pipe.address_for(b)


# --- a real round trip over the platform's actual pipe --------------------------


def test_a_server_and_a_connect_client_exchange_bytes_both_ways(tmp_path) -> None:
    """Not a mock of `multiprocessing.connection` -- a real `Listener`/`Client`
    pair over the platform's actual named pipe (Windows) or Unix socket.
    `poll(timeout)` throughout, so a transport regression hangs this one test
    for a few seconds and fails it, rather than blocking the whole suite (the
    project pins `--timeout 120` as its hang net; this stays well under it)."""
    server = pipe.Server(tmp_path)
    server.start()
    outcome: dict[str, bytes] = {}

    def run_client() -> None:
        conn = pipe.connect(tmp_path)
        conn.send_bytes(b"hello from the bridge")
        if conn.poll(10):
            outcome["reply"] = conn.recv_bytes()
        conn.close()

    client_thread = threading.Thread(target=run_client)
    client_thread.start()
    try:
        server_conn = server.accept()
        assert server_conn is not None
        assert server_conn.poll(10), "the client's first message never arrived"
        assert server_conn.recv_bytes() == b"hello from the bridge"
        server_conn.send_bytes(b"reply from the app")
        server_conn.close()
    finally:
        client_thread.join(timeout=10)
        server.close()

    assert outcome.get("reply") == b"reply from the app"


def test_accept_returns_none_after_close(tmp_path) -> None:
    server = pipe.Server(tmp_path)
    server.start()
    server.close()
    assert server.accept() is None


def test_a_client_with_the_wrong_authkey_is_refused(tmp_path) -> None:
    """The security claim: knowing the address is not enough.

    `Listener`/`Client`'s own HMAC challenge is what gates the connection, so
    the token file is doing real work and not decorating an open pipe. Pinned
    from the *client* side, which is the side an attacker is on: a wrong key
    raises rather than yielding a usable connection.
    """
    server = pipe.Server(tmp_path)
    server.start()

    accepted: list[Any] = []
    accepting = threading.Thread(target=lambda: accepted.append(server.accept()), daemon=True)
    accepting.start()
    try:
        with pytest.raises(mpconn.AuthenticationError):
            mpconn.Client(pipe.address_for(tmp_path), family=pipe._FAMILY, authkey=b"\x00" * 32)
    finally:
        server.close()
        accepting.join(timeout=10)


def test_a_rejected_peer_does_not_take_the_listener_down(tmp_path) -> None:
    """One wrong guess must not be a way to switch the feature off.

    `AuthenticationError` is a `ProcessError`, **not** an `OSError`, so the
    `except OSError` that catches a torn-down listener does not catch a failed
    challenge. Before `Server.accept` looped, the first process to dial the
    pipe with a stale token raised straight out through the host's accept loop
    and ended the agent session -- an unauthenticated caller could stop the
    listener for everyone. This asserts the loop: a rejected peer is dropped
    and the *next* client is served normally.

    Every wait is bounded and every connect happens on a thread, deliberately.
    Against the unfixed code the listener is gone, so a `Client` on the main
    thread would block until pytest's 120 s timeout killed the run -- a
    regression test that hangs instead of failing tells you far less, and
    costs two minutes to say it.
    """
    server = pipe.Server(tmp_path)
    server.start()
    token = pipe.read_token(tmp_path)
    address = pipe.address_for(tmp_path)

    accepted: list[Any] = []
    accepting = threading.Thread(target=lambda: accepted.append(server.accept()), daemon=True)
    accepting.start()

    def dial(key: bytes, into: list[Any]) -> None:
        with contextlib.suppress(Exception):
            into.append(mpconn.Client(address, family=pipe._FAMILY, authkey=key))

    try:
        # The probe, with a key that is 32 zero bytes and therefore wrong.
        probe: list[Any] = []
        prober = threading.Thread(target=dial, args=(bytes(32), probe), daemon=True)
        prober.start()
        prober.join(timeout=5)
        assert not probe, "a client with the wrong token was handed a connection"

        # The real bridge, arriving after it, is still served.
        good: list[Any] = []
        dialer = threading.Thread(target=dial, args=(token, good), daemon=True)
        dialer.start()
        dialer.join(timeout=5)
        accepting.join(timeout=5)

        assert accepted and accepted[0] is not None, "the listener died on a bad-token probe"
        assert good, "the listener never served the client that had the right token"
        good[0].send_bytes(b"after the probe")
        assert accepted[0].recv_bytes() == b"after the probe"
        good[0].close()
        accepted[0].close()
    finally:
        server.close()
        accepting.join(timeout=5)
