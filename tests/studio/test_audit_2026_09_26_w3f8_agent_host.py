"""Regression tests for the 2026-09-26 audit's ``agent_host.py`` findings
owned by wave 3, fixer 8 (``agents-host-01``; ``agents-host-02``, about the character
surface's transcript rows, went with that surface).

Kept in its own file rather than appended to ``tests/studio/test_agent_host.py``
because that file is not one of the files this fixer's brief lists as owned
for editing -- these tests stand alone, with their own minimal harness
mirroring (not importing, since ``tests/`` carries no ``__init__.py`` and
cross-test-module imports are not this codebase's convention) the small
helpers ``test_agent_host.py`` already uses for the same job.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from types import SimpleNamespace

from realmspinner.studio import agent_host


class _Ctx:
    """Just enough of the app's ``Ctx`` for ``agent_clay._tab`` to mint a tab
    and for ``AgentHost._toast`` to have somewhere to put a message -- see
    ``tests/studio/test_agent_host.py``'s own ``_Ctx`` for the fuller
    reasoning; this is the same minimal shape."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.settings = SimpleNamespace()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))

    def submit(self, key: str, fn, *args, tag=None, **kwargs) -> bool:
        threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()
        return True


def _bare_host() -> agent_host.AgentHost:
    """A host with a queue but no listener thread -- for ``_serve``/``_call``
    called directly, on the test's own thread."""
    host = agent_host.AgentHost(_Ctx(), Path("unused-for-these-tests"))
    host._queue = queue.Queue()
    return host


# --- agents-host-01: a stale _serve() must only clear its own connection ----


def test_a_stale_listener_finishing_after_restart_does_not_clear_the_new_connection_state(
    monkeypatch,
) -> None:
    """The 2026-09-26 audit (agents-host-01): ``AgentHost.stop()``'s own
    ``self._thread.join(timeout=STOP_JOIN_TIMEOUT)`` gives up after 2s, but
    ``_serve`` can still be blocked in its own ``conn.recv_bytes()`` well
    past that -- so a ``start()`` that follows can accept a brand new
    connection, set ``self._active_conn``/``self._connected`` for it, and
    then have *this stale call's own* ``finally`` run afterwards and wipe
    that live state anyway, because the old code cleared both fields
    unconditionally rather than only when they still named its own
    connection.

    Reproduced with no real thread timing: the stale call's own stub
    ``recv_bytes`` installs a second, unrelated "new" connection into
    ``host._active_conn`` on its first read (standing in for the fresh
    ``start()``+connection that raced ahead of this stale ``_serve`` while
    it sat blocked here) and then raises the ``EOFError`` a real closed pipe
    would, driving ``_serve`` straight to its own ``finally``.
    """
    host = _bare_host()
    # ``_serve`` blocks in ``_run_on_frame`` for the tab-open job until
    # ``pump()`` claims it (up to ``CALL_TIMEOUT``) -- stubbed to return
    # immediately, the same way ``tests/studio/test_agent_host.py``'s own
    # ``test_a_pipe_peer_cannot_force_an_unbounded_recv_bytes_allocation``
    # does, since this test has no pump loop running and does not care about
    # that call's outcome.
    monkeypatch.setattr(
        host, "_run_on_frame", lambda run, timeout=None, owner=None: (None, None, "done")
    )

    new_conn = object()

    class _StaleConn:
        def __init__(self) -> None:
            self._read = False

        def recv_bytes(self, maxlength=None):  # noqa: ARG002
            if not self._read:
                self._read = True
                # A fresh start() + connection landed while this stale
                # _serve call was still blocked in this very read.
                host._active_conn = new_conn
                host._connected = True
            raise EOFError()

        def close(self) -> None:
            pass

    host._serve(_StaleConn())

    assert host._active_conn is new_conn, (
        "the stale listener's own finally cleared the NEW connection's "
        "self._active_conn -- it must only clear state that is still its own"
    )
    assert host._connected is True, (
        "the stale listener's own finally cleared self._connected for a "
        "connection it does not own"
    )
