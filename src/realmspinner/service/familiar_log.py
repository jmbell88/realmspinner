"""Dev-only recorder for every Familiar round trip.

**Why this exists.** User, 2026-09-16: "all chats and builds recorded for
later review" -- a jsonl trail of what was sent, what came back and how the
pane landed it, for a human to read after the fact. **Dev only, never
ships**: the user's own words, so this is gated by an env var
(:data:`ENV_KEY`) the same way ``studio.component_gallery.enabled`` gates
the developer catalogue -- off unless a developer sets it, undocumented in
the Manual or Settings, and free (one ``os.environ.get`` and a boolean
return) on every call site when it is off.

**One file per process session.** Interleaving two sessions' turns into one
growing log would make "what happened in this run" a grep exercise instead
of "open the file"; a fresh ``<timestamp>-<pid>.jsonl`` under
``<REALMSPINNER_HOME>/familiar-log/`` is opened lazily, on the first record this
process actually makes, so a session that never touches Familiar creates
nothing. **Rotated at :data:`LOG_MAX_BYTES`** (``pipelines.llama``'s own
ceiling for the sibling llama-server log, imported rather than duplicated) --
the 2026-09-20 audit (familiar-05) found this file appended to for a whole
process's life, recording full prompts and replies with no ceiling at all,
while the log next to it already rotated. :func:`_rotate_if_over_ceiling`
truncates the same session file in place rather than starting a second one,
so this paragraph's "one file per process session" still holds.

**The directory is resolved through ``config._home()``, not a value cached
at import time** -- the same reason every root in ``config.py`` resolves
``REALMSPINNER_HOME`` independently (see that module's own docstring): a test
pins ``REALMSPINNER_HOME`` per test via ``monkeypatch``, and a value read once at
import time would still point at whatever the first test to import this
module happened to see.

**A logger must never break a chat.** :func:`record` swallows ``OSError``
(an unwritable directory, a full disk), ``TypeError``/``ValueError`` (a
field ``json.dumps`` cannot serialise even with ``default=str``) -- the
whole point of a dev-only observability hook is that turning it on must
never be the reason a real chat fails.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import config as _config
from ..pipelines.llama import LOG_MAX_BYTES

#: Same truthy set ``studio.component_gallery.enabled`` uses -- one
#: convention for "an env var toggles a dev-only surface" across the app.
ENV_KEY = "REALMSPINNER_FAMILIAR_LOG"
_TRUTHY = {"1", "true", "yes", "on"}


def enabled(environ: dict[str, str] | None = None) -> bool:
    """Whether this process records Familiar round trips at all."""
    source = os.environ if environ is None else environ
    return str(source.get(ENV_KEY, "")).strip().lower() in _TRUTHY


# One session file per process, opened lazily and kept open (append + flush
# per line, never reopened) for as long as the process runs. Guarded by
# ``_lock`` because ``record`` is called from both the frame thread (a
# submit) and a ``TaskRunner`` worker thread (a request/outcome) for the
# same session file.
_lock = threading.Lock()
_path: Path | None = None
_file: Any = None

# The exchange id records on the *current thread* carry -- a chat's worker
# thread and the frame thread that submitted it are different threads, so
# each has to set its own (``exchange``/``submit_chat`` et al set it on the
# worker via the closure ``run()`` runs inside; a submit record wraps the one
# frame-thread call in the same context manager to tag it with the same id).
_local = threading.local()


def _dir() -> Path:
    """``<REALMSPINNER_HOME>/familiar-log`` -- a function, not a cached value, so
    a test's ``REALMSPINNER_HOME`` pin (or a test that monkeypatches this
    function itself, to prove an unwritable directory never raises) is
    picked up on the next record rather than the first one this process
    happened to make."""
    return _config._home() / "familiar-log"


def _session_path() -> Path:
    global _path
    if _path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        _path = _dir() / f"{stamp}-{os.getpid()}.jsonl"
    return _path


def reset() -> None:
    """Forget the open session file. For tests: the next :func:`record`
    reopens, lazily, against whatever ``REALMSPINNER_HOME`` the next test pinned
    -- without this, a session file opened by an earlier test would keep
    being appended to under a directory that no longer exists."""
    global _path, _file
    with _lock:
        if _file is not None:
            with contextlib.suppress(OSError):
                _file.close()
        _file = None
        _path = None


def new_exchange_id() -> str:
    """A fresh id for one user-initiated round trip (a Send, a Build, a
    Create-on-a-plan) -- minted on the frame thread and threaded through
    ``exchange`` so every record the round trip produces, on whichever
    thread it runs on, can be grouped back together."""
    return uuid.uuid4().hex


@contextlib.contextmanager
def exchange(exchange_id: str | None):
    """Tag every :func:`record` call made on *this thread*, for the
    duration of the ``with`` block, with *exchange_id``."""
    previous = getattr(_local, "id", None)
    _local.id = exchange_id
    try:
        yield
    finally:
        _local.id = previous


def _current_exchange() -> str | None:
    return getattr(_local, "id", None)


def _rotate_if_over_ceiling() -> None:
    """Truncate the open session file once it passes :data:`LOG_MAX_BYTES`
    -- must be called with :data:`_lock` already held, and only once
    :data:`_file` is known to be open.

    The 2026-09-20 audit (familiar-05): this file records full prompts and
    replies for as long as the process runs, with no ceiling at all, while
    ``pipelines.llama``'s own log for the same feature (the llama-server
    child's stdout) rotates at this exact byte count (``_open_log``). Mirrors
    that check-then-unlink shape rather than a new timestamped file, so
    :func:`_session_path`'s "one file per process session" contract still
    holds -- a rotation drops old lines the way ``_open_log`` does, it does
    not start a second file a reader would have to know to go looking for.
    """
    global _file
    if _path is None:
        return
    with contextlib.suppress(OSError):
        if _path.stat().st_size > LOG_MAX_BYTES:
            _file.close()
            _path.unlink()
            _file = _path.open("a", encoding="utf-8")


def record(kind: str, **fields: Any) -> None:
    """Append one JSON line: ``{"ts", "kind", "exchange", **fields}``. A
    no-op, with nothing opened or written, unless :func:`enabled`. Never
    raises -- see the module docstring's "a logger must never break a
    chat"."""
    if not enabled():
        return
    try:
        line = json.dumps(
            {
                "ts": datetime.now(UTC).isoformat(),
                "kind": kind,
                "exchange": _current_exchange(),
                **fields,
            },
            default=str,
            ensure_ascii=False,
        )
        global _file
        with _lock:
            if _file is None:
                path = _session_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                _file = path.open("a", encoding="utf-8")
            else:
                _rotate_if_over_ceiling()
            _file.write(line + "\n")
            _file.flush()
    except (OSError, TypeError, ValueError):
        # A logger must never break a chat -- an unwritable directory, a
        # full disk, or a field that ``default=str`` still could not stand
        # in for is dropped rather than raised.
        pass
