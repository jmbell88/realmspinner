"""The Manual pane's optional "Semantic" search: everything but the drawing.

The pane's own search is a substring match (``kernels/manual/loader``) and stays
the default. With the retrieval row installed (``familiar_embed``, EmbeddingGemma 2)
a **Semantic** toggle appears beside it, and with it on the typed query is ranked
by the same BM25 + meaning fusion Familiar answers Manual questions with
(``service/familiar_manual.py``) -- so "undo a mistake while painting pixels"
finds ``Drawing > Undo`` without the word "mistake" being in it.

This module is imgui-free so the decisions are testable: :func:`view` is called
once per frame by ``render._draw_toc`` and answers *what to draw*; it never
embeds anything itself. The one blocking step -- embedding the query, and
loading or kicking off the matrix -- runs in a ``TaskRunner`` task under
:data:`KEY` (``ctx.submit``), submitted only once the typed text has settled
for :data:`DEBOUNCE` seconds, so typing never queues one request per keystroke
and the frame thread never touches the embed client.

The task hands its answer back through a small locked holder instead of through
``main._on_task_done``: the Manual is an overlay, not a mode, and no mode owns a
key prefix for it. The task catches everything, so a failure is a result of
"unavailable" and the pane shows its substring results, with no toast.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any

#: The one task key. ``familiar/``-prefixed so the shell routes its landing to
#: the Familiar dock's handler, which has nothing to do for it (the result is
#: read from the holder below); one key also means ``TaskRunner`` refuses a
#: second concurrent search, so a fast typist cannot pile requests up.
KEY = "familiar/manual-semantic"

#: How long the typed text must stay unchanged before a query is embedded.
DEBOUNCE = 0.4

#: While the meaning index is still being built, how often to ask again whether
#: it is ready. The ask is a memo lookup plus a ``stat`` -- no embedding -- so
#: this is a poll, not a request storm.
RETRY_BUILDING = 4.0

#: How long the "is the retrieval row installed" answer is reused. It is a few
#: ``stat`` calls and the toggle is drawn every frame the Manual is open.
INSTALLED_TTL = 2.0

#: Fewest characters worth embedding.
MIN_QUERY = 2

BUILDING_NOTE = "The meaning index is still being built; showing text matches."


@dataclasses.dataclass(frozen=True)
class Row:
    """One semantic result, drawn like a substring match: a section to open."""

    chapter: str
    anchor: str | None
    label: str


@dataclasses.dataclass(frozen=True)
class View:
    """What ``render._draw_toc`` draws this frame.

    *rows* non-empty: draw them instead of the chapter list. *note*: one dim
    line above whatever else is drawn. Neither: the ordinary substring results.
    """

    show_toggle: bool
    rows: tuple[Row, ...] = ()
    note: str = ""


class _Holder:
    """The task's answer and the debounce clock. ``result``/``building_since``
    are written by the task thread and read by the frame thread, hence the lock;
    everything else is frame-thread only."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.result: tuple[str, Any] | None = None  # (query, ManualHits | None)
        self.typed = ""
        self.typed_at = 0.0
        self.asked = ""
        self.asked_at = 0.0
        self.installed: tuple[float, bool] | None = None


_state = _Holder()


def reset() -> None:
    """Forget every result and clock (tests, and a Manual that was closed)."""
    with _state.lock:
        _state.reset()


def available(ctx: Any, now: float | None = None) -> bool:
    """Whether the Semantic toggle is offered: the retrieval row is installed.
    Memoised for :data:`INSTALLED_TTL` so the frame loop does not ``stat`` the
    models directory sixty times a second."""
    now = time.monotonic() if now is None else now
    cached = _state.installed
    if cached is not None and now - cached[0] < INSTALLED_TTL:
        return cached[1]
    try:
        from ...service import familiar as svc_familiar

        ok = bool(svc_familiar.manual_semantic_available(ctx.svc))
    except Exception:
        # Called every frame (memoised): the offer is optional, so any failure here
        # reads as 'not offered' rather than a toast or a crashed Manual pane.
        ok = False
    _state.installed = (now, ok)
    return ok


def _run(svc: Any, query: str) -> None:
    """The task body (a ``TaskRunner`` worker thread): embed *query* and rank.
    Stores ``(query, hits)``; ``None`` for hits on any failure."""
    hits: Any = None
    try:
        from ...service import familiar as svc_familiar

        hits = svc_familiar.manual_sections(svc, query)
    except Exception:
        # The embedder is optional and every way it can fail means "use the
        # substring results"; nothing here is the user's to be told about.
        hits = None
    with _state.lock:
        _state.result = (query, hits)


def view(ctx: Any, ms: Any, needle: str, now: float | None = None) -> View:
    """One frame's decision. Cheap and non-blocking: it may *submit* the query
    task (never run it) and reads the last result the task stored.

    *needle* is the stripped, lower-cased text in the search box. The toggle's
    own value is ``ms.semantic``; the renderer draws the checkbox and writes it.
    """
    now = time.monotonic() if now is None else now
    if not available(ctx, now):
        return View(show_toggle=False)
    if not getattr(ms, "semantic", False):
        return View(show_toggle=True)
    query = ms.search.strip()
    if len(query) < MIN_QUERY:
        return View(show_toggle=True)

    if query != _state.typed:
        _state.typed = query
        _state.typed_at = now
    with _state.lock:
        result = _state.result
    answered = result is not None and result[0] == query
    hits = result[1] if answered else None

    settled = now - _state.typed_at >= DEBOUNCE
    # Ask when the text has settled and there is no answer for it yet -- or the
    # answer was "still building" and it is time to look again. A failure
    # (``hits is None``) is not retried until the text changes: a dead embedder
    # must cost one request per query, not one per frame.
    wants = (not answered) or (
        hits is not None
        and hits.state == "building"
        and now - _state.asked_at >= RETRY_BUILDING
    )
    if settled and wants and not ctx.busy(KEY) and ctx.submit(KEY, _run, ctx.svc, query):
        _state.asked = query
        _state.asked_at = now

    if hits is None:
        return View(show_toggle=True)
    if hits.state == "building":
        return View(show_toggle=True, note=BUILDING_NOTE)
    if hits.state == "ready" and hits.sections:
        rows = tuple(Row(chapter, anchor, label) for chapter, anchor, label in hits.sections)
        return View(show_toggle=True, rows=rows)
    return View(show_toggle=True)
