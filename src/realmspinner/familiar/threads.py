"""Familiar's per-tab conversation memory, for display only.

**One thread per document tab**, keyed by ``(mode, tab_uid)`` -- a Clay tab
and a Mason tab each keep their own turn history, because the user's fixed
design decision is that switching tabs should feel like switching
conversations, the same way switching tabs already swaps the undo stack and
the camera. **Non-document modes share one "Studio" thread** (:data:`STUDIO`),
because Home, Settings and the rest have no tab to key a thread on.

**Session only.** Nothing here is persisted; a restart starts empty, the same
way the in-memory undo stacks do.

**Display and pending-preview refinement, never model context.** The model
still sees only the current request plus the scene -- that is the shape
Familiar's model was trained on -- so a thread here is not conversation
history fed back into a prompt. It is what the bottom pane renders, and it is
what a "no, make it taller" follow-up reads to know which pending preview it
is refining. Conflating the two would grow the prompt without bound and feed
the model a shape it was never trained to use.

Stdlib only: no imgui, moderngl, pygame, ``service``, ``queue`` or httpx --
see ``router.py``'s docstring for why.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

#: The shared thread key for every mode with no document tabs of its own
#: (Home, Library, Settings, ...). ``tab_uid`` is the empty string so a
#: falsy uid always resolves here -- see :meth:`Threads.key_for`.
STUDIO: tuple[str, str] = ("studio", "")

ThreadKey = tuple[str, str]


@dataclass(frozen=True)
class Turn:
    """One line of a thread: who said it, what, and (T6) which Manual
    sections it cited."""

    role: str  # "user" | "familiar"
    text: str
    #: The ``retrieval.Citation`` rows a Manual answer actually named (see
    #: ``contract.cited``) -- always empty for a user turn, a plain-chat
    #: reply, or a routed build. Defaulted so every pre-T6 ``Turn(role,
    #: text)`` call site keeps working unchanged.
    citations: tuple = ()


class Threads:
    """Turn history keyed by ``(mode, tab_uid)``.

    Guarded by a lock because the future ``realmspinner-familiar`` thread (T4/T5)
    appends turns as a skill's answer streams in while the frame thread reads
    a snapshot to draw the bottom pane -- the same cross-thread shape as
    every other piece of Familiar state, and a lock is cheaper than teaching
    every reader to tolerate a torn read.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: dict[ThreadKey, list[Turn]] = {}

    @staticmethod
    def key_for(mode: str, tab_uid: str) -> ThreadKey:
        """The key for ``mode``/``tab_uid`` -- :data:`STUDIO` when
        ``tab_uid`` is falsy, because a mode with no tabs (or a tab not yet
        minted a uid) shares the one Studio thread rather than getting a new
        thread per empty string."""
        if not tab_uid:
            return STUDIO
        return (mode, tab_uid)

    def append(self, key: ThreadKey, turn: Turn) -> None:
        with self._lock:
            self._threads.setdefault(key, []).append(turn)

    def get(self, key: ThreadKey) -> tuple[Turn, ...]:
        """The thread at ``key``, or an empty tuple for one never started --
        never a ``KeyError``, since "no turns yet" and "no thread yet" are
        the same fact to every reader of this."""
        with self._lock:
            return tuple(self._threads.get(key, ()))

    def drop(self, mode: str, uid: str) -> None:
        """Forget one tab's thread. Registered as a ``docmodes.TAB_CLOSED``
        listener (T5): a closed tab's conversation ends with it, the same way
        its undo stack and journal copy already do."""
        key = (mode, uid)
        with self._lock:
            self._threads.pop(key, None)

    def clear(self) -> None:
        """Forget every thread -- a fresh session, or a test's teardown."""
        with self._lock:
            self._threads.clear()
