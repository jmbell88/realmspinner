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
from pathlib import Path
from typing import Any

import pytest

from realmspinner.mcp import pipe

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


# --- the handshake is bounded, and owns every way it can fail -------------------


def _dial_raw(address: str) -> Any:
    """Open the pipe without answering the challenge, the way a hostile or
    merely broken peer does. Returns something to close, or ``None`` on a
    platform this probe does not cover.

    Deliberately *not* `mpconn.Client`: that answers the challenge, which is
    the one thing these tests need a peer to refuse to do.
    """
    if sys.platform == "win32":
        import _winapi

        return _winapi.CreateFile(
            address,
            _winapi.GENERIC_READ | _winapi.GENERIC_WRITE,
            0,
            _winapi.NULL,
            _winapi.OPEN_EXISTING,
            0,
            _winapi.NULL,
        )
    import socket

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(address)
    return sock


def _close_raw(handle: Any) -> None:
    if sys.platform == "win32":
        import _winapi

        _winapi.CloseHandle(handle)
    else:
        handle.close()


def test_a_peer_that_never_answers_the_challenge_does_not_wedge_the_listener(
    tmp_path, monkeypatch
) -> None:
    """Silence must not be a way to switch the agent server off for good.

    `Listener(authkey=...)` runs the HMAC exchange inside `accept()` with an
    unbounded `recv_bytes`, so a peer that opened the pipe and then said
    nothing parked the accept loop forever -- and, measured, `Server.close()`
    did not free it either, so `AgentHost.stop()` joined for two seconds and
    left the thread holding a pipe instance. Any local process could end an
    agent session that way, and the real bridge behind it blocked in its own
    `connect()` rather than getting the readable refusal `bridge.main` prints.

    Asserts the bound: the silent peer is dropped and the client that follows
    it, holding the right token, is served.
    """
    # Shortened for the suite's sake: ``_handshake`` reads this module global
    # on every call, so the path under test is exactly the shipped one -- only
    # the wait is smaller. Left at the real value the test would cost the
    # default lane ten seconds to say the same thing.
    monkeypatch.setattr(pipe, "HANDSHAKE_TIMEOUT", 0.5)
    server = pipe.Server(tmp_path)
    server.start()
    token = pipe.read_token(tmp_path)
    address = pipe.address_for(tmp_path)

    accepted: list[Any] = []
    accepting = threading.Thread(target=lambda: accepted.append(server.accept()), daemon=True)
    accepting.start()

    rude = _dial_raw(address)
    try:
        good: list[Any] = []

        def dial() -> None:
            with contextlib.suppress(Exception):
                good.append(mpconn.Client(address, family=pipe._FAMILY, authkey=token))

        dialer = threading.Thread(target=dial, daemon=True)
        dialer.start()
        # Comfortably past HANDSHAKE_TIMEOUT: the silent peer has to be given
        # up on before the honest one behind it can be served.
        dialer.join(timeout=pipe.HANDSHAKE_TIMEOUT + 15)
        accepting.join(timeout=10)

        assert good, "the listener never served a client holding the right token"
        assert accepted and accepted[0] is not None, "the listener was wedged by a silent peer"
        good[0].send_bytes(b"after the silence")
        assert accepted[0].recv_bytes() == b"after the silence"
        good[0].close()
        accepted[0].close()
    finally:
        _close_raw(rude)
        server.close()
        accepting.join(timeout=5)


def test_a_peer_that_vanishes_mid_handshake_does_not_kill_the_listener(
    tmp_path, monkeypatch
) -> None:
    """The `EOFError` the old guard did not name.

    `Server.accept` caught `AuthenticationError` and `OSError` because those
    are the two shapes a *rejected* peer arrives in. A peer that disconnects
    part-way through the exchange raises neither: `deliver_challenge`'s
    `recv_bytes` raises `EOFError`, which went straight out through the accept
    loop and ended `AgentHost._listen` -- and an MCP client killed or
    restarted while its bridge is dialling in does exactly this. The agent
    server then stopped answering while Settings still showed the pipe
    address, with no way back but toggling the switch.
    """
    # Shortened for the suite's sake: ``_handshake`` reads this module global
    # on every call, so the path under test is exactly the shipped one -- only
    # the wait is smaller. Left at the real value the test would cost the
    # default lane ten seconds to say the same thing.
    monkeypatch.setattr(pipe, "HANDSHAKE_TIMEOUT", 0.5)
    server = pipe.Server(tmp_path)
    server.start()
    token = pipe.read_token(tmp_path)
    address = pipe.address_for(tmp_path)

    accepted: list[Any] = []
    accepting = threading.Thread(target=lambda: accepted.append(server.accept()), daemon=True)
    accepting.start()

    # Connect, then vanish before answering anything.
    _close_raw(_dial_raw(address))

    try:
        good: list[Any] = []

        def dial() -> None:
            with contextlib.suppress(Exception):
                good.append(mpconn.Client(address, family=pipe._FAMILY, authkey=token))

        dialer = threading.Thread(target=dial, daemon=True)
        dialer.start()
        dialer.join(timeout=pipe.HANDSHAKE_TIMEOUT + 15)
        accepting.join(timeout=10)

        assert accepted, "accept() never returned: the listener died on a half-open peer"
        assert accepted[0] is not None, "the listener was torn down by a half-open peer"
        assert good, "the listener never served the client that had the right token"
        good[0].close()
        accepted[0].close()
    finally:
        server.close()
        accepting.join(timeout=5)


def test_a_refused_peer_leaves_no_handle_that_blocks_the_next_start(tmp_path, monkeypatch) -> None:
    """The chain that turned a wedged pipe into an app that would not launch.

    CPython opens the Windows pipe with `FILE_FLAG_FIRST_PIPE_INSTANCE`, so a
    handle still held by a thread abandoned mid-handshake is enough to make
    the *next* `Server.start()` raise `PermissionError` -- and since the
    Settings switch persists before it calls `start()`, that failure used to
    follow the app into `main.setup_context` on every subsequent launch.
    Closing the connection on timeout is what releases the abandoned thread;
    this asserts the consequence rather than the mechanism, so it holds on
    whichever platform the test runs.

    Scoped to the handshake deliberately. The accept loop is driven all the
    way to a *served* connection before anything is closed, so the listener is
    not sitting in a pending `accept()` of its own -- whatever still holds a
    pipe instance at the end can only have come from the peer that was
    refused, which is the thing this fix is responsible for.
    """
    # Shortened for the suite's sake: ``_handshake`` reads this module global
    # on every call, so the path under test is exactly the shipped one -- only
    # the wait is smaller. Left at the real value the test would cost the
    # default lane ten seconds to say the same thing.
    monkeypatch.setattr(pipe, "HANDSHAKE_TIMEOUT", 0.5)
    server = pipe.Server(tmp_path)
    server.start()
    token = pipe.read_token(tmp_path)
    address = pipe.address_for(tmp_path)

    accepted: list[Any] = []
    accepting = threading.Thread(target=lambda: accepted.append(server.accept()), daemon=True)
    accepting.start()

    rude = _dial_raw(address)
    good: list[Any] = []

    def dial() -> None:
        with contextlib.suppress(Exception):
            good.append(mpconn.Client(address, family=pipe._FAMILY, authkey=token))

    dialer = threading.Thread(target=dial, daemon=True)
    dialer.start()
    dialer.join(timeout=pipe.HANDSHAKE_TIMEOUT + 15)
    accepting.join(timeout=10)
    _close_raw(rude)
    assert accepted and accepted[0] is not None, "the listener never got past the refused peer"
    for conn in (*good, *accepted):
        conn.close()
    server.close()

    again = pipe.Server(tmp_path)
    # The claim: a refused peer costs nothing that outlives it.
    again.start()
    again.close()


@pytest.mark.skipif(sys.platform == "win32", reason="a pipe name is not a file on Windows")
def test_a_socket_file_left_by_a_crash_does_not_disable_the_server_for_good(
    tmp_path,
) -> None:
    """POSIX only, and permanent before this: a Unix socket outlives the
    process that made it, and `bind` refuses an address whose file exists. A
    Realmspinner killed hard left `mcp.sock` behind and every later `start()`
    raised `EADDRINUSE` -- for the life of that home, since nothing removed
    it. `instance.py` guarantees one Realmspinner per home, so a socket file here
    cannot belong to a live Realmspinner and is a leftover by construction.
    """
    stale = Path(pipe.address_for(tmp_path))
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"")
    assert stale.exists()

    server = pipe.Server(tmp_path)
    server.start()
    try:
        assert server.address
    finally:
        server.close()
