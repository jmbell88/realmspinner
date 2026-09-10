"""`warlock mcp` -- a dumb relay between an agent's stdio and the app's pipe.

Deliberately dumb: every byte this reads from `stdin` goes straight to the
pipe via `send_bytes`, and every reply comes straight back to `stdout`. No
JSON is parsed here, no method names are known here -- all of that is
`protocol.py`, which runs *inside the app* (see `studio/agent_host.py`)
because the app is the only side that can act on a tool call. Keeping the
untestable half (a real child process talking to a real MCP client over real
pipes) this thin is what makes the semantics testable without either.

**Main thread only, and that is not a style preference.**
`pipelines/_workerio.py` documents a measured Windows deadlock: a daemon
thread parked in a blocking read on an inherited stdin pipe stops the main
thread's very next native-extension import cold, indefinitely. This process
does nothing after start-up that needs a second thread -- it is one blocking
read, one write, repeat -- so the fix is simply to never introduce the thread
that could trip over that landmine in the first place.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from . import pipe


def main(argv: Sequence[str] | None = None) -> int:
    """Relay stdio JSON-RPC to the app's pipe until either side closes.

    `argv` is accepted and ignored -- `warlock mcp` takes no flags today, and
    the parameter exists so a test can call this the same way `cli.main`
    does, argv and all.
    """
    # warlock.config is the one warlock import this package may make: it is
    # display-free (no pygame, no moderngl, no torch), unlike almost anything
    # else under warlock/, which is exactly what lets `mcp/` stay importable
    # on a machine with no GPU and no window.
    from ..config import get_config

    home = get_config().home
    try:
        conn = pipe.connect(home)
    except OSError:
        # Covers both "nothing is listening" (a Windows pipe or Unix socket
        # with no server, an OSError subtype each) and pipe.read_token's own
        # FileNotFoundError (also an OSError) when the app has never started
        # a session for this home at all. One remedy either way.
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
                # A zero-length frame is the app's signal that the message it
                # just handled was a notification -- JSON-RPC forbids a reply
                # to one, so there is nothing to forward.
                continue
            stdout.write(reply)
            stdout.flush()
    except EOFError:
        # The app's end of the pipe closed -- Studio quit, or the agent
        # toggle was switched off mid-session. Same outcome as the client's
        # own stdin closing: end quietly rather than raising a traceback an
        # MCP tool runner has no use for.
        pass
    finally:
        conn.close()
    return 0
