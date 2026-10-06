"""When the Library's meaning index gets written: quietly, off the frame thread.

``service.library_index`` has the blocking functions (embed some jobs, store the
vectors); this decides *when* to call them and never calls them itself.
:class:`JobsCache` owns one :class:`LibraryIndexer`, tells it about finished and
renamed jobs from ``adopt`` and pumps it once a frame from ``request`` -- both
frame-thread calls that do a dict operation and at most one ``TaskRunner.submit``.

**Two jobs, two task keys, both bounded.**

* ``library-embed:jobs`` embeds the handful of jobs that just finished (or were
  renamed): one batch, usually one row.
* ``library-embed:backfill`` indexes the history that was there before the
  retrieval row was installed, :data:`~service.library_index.BATCH` rows at a
  time, and *reschedules itself* -- every batch ends by leaving the state
  machine where the next frame's pump finds it -- until nothing is missing
  (``missing``), then makes one pass over the rows that already have a vector to
  catch a rename or a changed text recipe (``verify``), then stops (``done``)
  until the retrieval row is installed again after being absent.

**It only runs while the embedder is installed**, and asks that with a memo
(:meth:`available`, a few ``stat`` calls every :data:`AVAILABLE_SECONDS`) so the
frame loop never pays disk for it. With the row absent :meth:`pump` returns on
its first line and :meth:`note` drops what it is told: no table writes, no child
process, no log line, no error shown.

**It does not fight the card's jobs for the CPU.** The backfill pauses while a
job is queued or running (:meth:`set_live`); finishing a job still indexes that
one row, which is tiny. Neither touches the GPU -- the embedder is ``--device
none``.

**A failure is never a toast and never a tight loop.** ``EmbedUnavailable`` (the
embedder will not start, or would not answer in time) means "try later": the
batch writes nothing, the ids go back in the queue, and every pass is refused
for ``BACKOFF_BASE * 2**(failures - 1)`` seconds, capped at :data:`BACKOFF_MAX`;
a success resets it. Task bodies catch everything and return a small dict --
``TaskRunner`` turns an uncaught exception into a toast, and an index the user
never asked to see must not toast.

State written by the task threads (the phase, the failure count, the retry
time, the queue of ids) is this object's own and sits behind its lock; nothing
here writes ``JobsCache``'s published rows.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from ..familiar.embed_client import EmbedUnavailable
from ..service import library_index

log = logging.getLogger(__name__)

#: ``TaskRunner`` keys. A prefix of its own so the shell can claim both silently
#: (``main.SILENT_TASK_KEYS``) -- nothing is delivered from them.
ONE_KEY = "library-embed:jobs"
BACKFILL_KEY = "library-embed:backfill"
KEY_PREFIX = "library-embed:"

#: How long an "is the retrieval row installed" answer is kept. The check is a
#: handful of ``stat`` calls and the answer changes only when somebody finishes
#: a download or removes a row in Settings, so half a minute is invisible.
AVAILABLE_SECONDS = 30.0

#: A pause between two backfill batches, so the history is indexed in a trickle
#: and not as fast as the CPU allows while the user is working.
BATCH_GAP_SECONDS = 0.5

#: The first retry after a failure, doubling to the ceiling.
BACKOFF_BASE = 30.0
BACKOFF_MAX = 900.0

#: Ids waiting to be indexed past this are dropped: a long outage with a busy
#: queue must not grow without bound, and the backfill's ``missing`` phase finds
#: whatever was dropped.
MAX_PENDING = 1000

# Backfill phases.
MISSING = "missing"
VERIFY = "verify"
DONE = "done"


class LibraryIndexer:
    """The scheduler. Frame-thread methods: :meth:`available`, :meth:`note`,
    :meth:`set_live`, :meth:`pump`, :meth:`stop`. Task-thread methods (private):
    ``_run_ids`` and ``_run_backfill``."""

    def __init__(
        self,
        svc: Any,
        *,
        clock: Callable[[], float] = time.monotonic,
        batch: int = library_index.BATCH,
    ) -> None:
        self.svc = svc
        self._clock = clock
        self._batch = batch
        self._lock = threading.Lock()
        self._stopped = False
        self._live = False
        self._available = False
        self._available_until = float("-inf")
        self._pending: dict[str, None] = {}
        self._phase = MISSING
        self._pruned = False
        self._failures = 0
        self._retry_at = 0.0
        self._next_batch_at = 0.0

    # -- frame thread ----------------------------------------------------

    def available(self) -> bool:
        """Whether the retrieval row (weights and runtime) is installed and
        there is a worker to run it, memoised for :data:`AVAILABLE_SECONDS`.

        A flip from absent to installed restarts the backfill from ``missing``:
        the rows finished before the download are exactly what it is for."""
        now = self._clock()
        if now < self._available_until:
            return self._available
        found = False
        if not self._stopped:
            try:
                found = library_index.installed(self.svc)
            except Exception:
                # A few ``stat`` calls: any fault reads as "not installed" and
                # the memo asks again in half a minute. Nothing here is the
                # user's to fix, so it is neither toasted nor logged per frame.
                found = False
        with self._lock:
            if found and not self._available:
                self._phase = MISSING
                self._pruned = False
                self._failures = 0
                self._retry_at = 0.0
            self._available = found
            self._available_until = now + AVAILABLE_SECONDS
        return found

    def note(self, job_id: str) -> None:
        """A job just finished, or was renamed: index it soon. A no-op without
        the retrieval row (nothing is queued for a feature that is off)."""
        if self._stopped or not job_id or not self.available():
            return
        with self._lock:
            if len(self._pending) < MAX_PENDING:
                self._pending[job_id] = None

    def set_live(self, live: bool) -> None:
        """Whether a job is queued or running, which pauses the backfill."""
        self._live = bool(live)

    def stop(self) -> None:
        """Shutdown: no further task is submitted, and a batch already running
        checks before it embeds (``library_index`` stops once the worker's loop
        has gone), so quitting waits for nothing this started."""
        self._stopped = True
        with self._lock:
            self._pending.clear()

    def pump(self, runner: Any) -> None:
        """Submit whatever is due. Called once a frame; cheap when idle.

        At most one task per key is ever in flight (``TaskRunner.submit``
        refuses a key that is), so a slow batch makes later pumps do nothing
        rather than queue behind it."""
        if self._stopped or not self.available():
            return
        now = self._clock()
        with self._lock:
            if now < self._retry_at:
                return
            ids = tuple(self._pending)
            backfill_due = (
                self._phase != DONE and not self._live and now >= self._next_batch_at
            )
        if ids and runner.submit(ONE_KEY, self._run_ids, ids):
            with self._lock:
                for job_id in ids:
                    self._pending.pop(job_id, None)
        if backfill_due:
            runner.submit(BACKFILL_KEY, self._run_backfill)

    # -- the state the tests (and a status line, one day) read ---------------

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    @property
    def pending(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._pending)

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    # -- task threads ----------------------------------------------------------

    def _backoff(self) -> float:
        with self._lock:
            self._failures += 1
            delay = min(BACKOFF_MAX, BACKOFF_BASE * 2 ** (self._failures - 1))
            self._retry_at = self._clock() + delay
        return delay

    def _succeeded(self) -> None:
        with self._lock:
            self._failures = 0
            self._retry_at = 0.0

    def _run_ids(self, ids: tuple[str, ...]) -> dict[str, Any]:
        """``library-embed:jobs``: embed exactly these jobs."""
        if self._stopped:
            return {"indexed": 0}
        try:
            written = library_index.index_jobs(self.svc, ids)
        except EmbedUnavailable as exc:
            delay = self._backoff()
            log.info("Library index: embedder unavailable (%s); retrying in %.0f s", exc, delay)
            with self._lock:
                for job_id in ids:
                    if len(self._pending) < MAX_PENDING and not self._stopped:
                        self._pending[job_id] = None
            return {"indexed": 0, "retry": True}
        except Exception:
            # A store or text-builder fault: log it, drop the ids (the verify
            # pass of the next session catches them) and back off.
            log.exception("Library index: could not index %d finished job(s)", len(ids))
            self._backoff()
            return {"indexed": 0, "error": True}
        self._succeeded()
        return {"indexed": written}

    def _run_backfill(self) -> dict[str, Any]:
        """``library-embed:backfill``: one bounded batch of the history, then
        leave the phase where the next pump finds it."""
        if self._stopped:
            return {"indexed": 0}
        with self._lock:
            phase = self._phase
            first = not self._pruned
        if phase == DONE:
            return {"indexed": 0}
        try:
            if first:
                self.svc.store.prune_embeddings()
                with self._lock:
                    self._pruned = True
            result = library_index.index_pending(
                self.svc, self._batch, verify=(phase == VERIFY)
            )
        except EmbedUnavailable as exc:
            delay = self._backoff()
            log.info("Library index: embedder unavailable (%s); retrying in %.0f s", exc, delay)
            return {"indexed": 0, "retry": True}
        except Exception:
            log.exception("Library index: a backfill batch failed")
            self._backoff()
            return {"indexed": 0, "error": True}
        self._succeeded()
        with self._lock:
            if not result.more:
                # Nothing (more) missing: check the rows that have a vector for
                # drift once, then finish. A verify pass that is also not full
                # is the end of the run.
                self._phase = VERIFY if self._phase == MISSING else DONE
            self._next_batch_at = self._clock() + BATCH_GAP_SECONDS
            finished = self._phase == DONE
        if finished:
            log.info("Library index: the history is indexed")
        return {"indexed": result.indexed, "phase": self._phase}
