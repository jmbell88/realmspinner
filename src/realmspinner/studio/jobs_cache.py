"""The job list the UI reads, refreshed on a timer rather than per frame.

``JobStore.list`` is a real sqlite query behind a lock, and at 60 fps calling
it every frame would put the single connection under 60 reads a second for
data that changes when a job changes status. Every 500 ms, or immediately when
something the UI did makes it stale, is the same tradeoff the browser made with
its poll -- minus the HTTP.

Status transitions are diffed here rather than watched for elsewhere: this is
the one place that sees both the old list and the new one, which makes it the
only place that can tell "finished" from "was already finished".
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from .. import followups
from ..service import jobs as svc_jobs
from ..service.files import dir_size

log = logging.getLogger(__name__)

REFRESH_SECONDS = 0.5
# The poll while nothing is queued or running (L102): the list can only change
# through this UI, and everything the UI does calls ``invalidate`` -- so the
# slow tick is a backstop against a missed invalidation, not the signal path.
IDLE_REFRESH_SECONDS = 3.0
# How often COUNT(*) is re-run while the page is full (A3). When the page is
# not full the count is exact for free (total == len(jobs)).
COUNT_SECONDS = 5.0
LIST_LIMIT = 200
# How many ids a search widens the window by. Small on purpose: it only needs
# to find candidates the loaded window is missing, not to become a second
# pager -- "Load older" still exists for that.
SEARCH_LIMIT = 50
#: The task key :meth:`request_widen` submits under, so :meth:`adopt_widen`
#: can be found from ``TaskRunner``'s result the way ``"jobs-list"`` already
#: is for :meth:`read`/:meth:`adopt`.
SEARCH_KEY = "jobs-search"
#: The widest the window may get, whatever "Load older" is pressed. It is the
#: service's own ``MAX_LIST_LIMIT``, read lazily below rather than imported so
#: a test that lowers the ceiling lowers this too -- and it is a *local* cap
#: rather than a reliance on the service's clamp, which is what the code did
#: before and which had a visible cost: ``limit`` went on growing past 5000
#: forever, so "Load older" kept offering to widen a window that could not
#: widen and the press did nothing with nothing said about it. Every row in the
#: window also costs an ``attach_files`` stat per tick on the frame thread, so
#: the ceiling is a frame-time bound as much as a query one.


class JobsCache:
    """A recent job list plus what changed since the frame before."""

    def __init__(self, svc: Any, limit: int = LIST_LIMIT) -> None:
        self.svc = svc
        self.limit = limit
        self.jobs: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}
        self.storage: dict[str, Any] = {}
        # How many jobs exist at all, so the library can say "showing newest N
        # of M" rather than silently presenting a truncated history as whole.
        self.total = 0
        self.error: str | None = None
        # Why ``total`` is not to be trusted, when it is not (E43). A failed
        # COUNT(*) falls back to ``len(jobs)``, which on a full page is exactly
        # the value that means "this is the whole history" -- so the fallback
        # silently retracts "Load older" and presents a truncated window as
        # complete. The library reads this and says so instead.
        self.count_error: str | None = None
        # Likewise for the storage walk (E45). Set on the task thread and read
        # on the frame thread, as ``_dir_sizes`` beside it already is: a plain
        # attribute write is atomic enough for a flag nothing branches twice on.
        self.storage_error: str | None = None
        self._last_status: dict[str, str] = {}
        self._next_refresh = 0.0
        self._next_count = 0.0
        self._dirty = True
        # Bumped every time ``jobs`` is replaced -- the key the per-generation
        # memos below (visible, failures) hang off.
        self._generation = 0
        self._visible_memo: tuple[Any, list[dict[str, Any]]] | None = None
        self._failures_memo: tuple[Any, int] | None = None
        # {job dir name: bytes} from the last full walk, so a finished job can
        # fold its own directory back in without re-walking everything (C33).
        self._dir_sizes: dict[str, int] = {}
        # Bumped whenever ``_dir_sizes`` is replaced or amended, so the
        # ``visible`` memo notices a measurement landing (J85's size sort).
        self._sizes_generation = 0
        # {job_id: ((status, dir mtime), names)} for attach_files -- the frame
        # loop's largest syscall cost, and the one that grew without limit as
        # "load more" widened the window. Pruned to the page below, so it can
        # never outgrow what is being shown.
        self._files: dict[str, tuple[tuple[Any, int], list[str]]] = {}
        # Rows beyond the newest page (A2/O119): ``read`` only ever refreshes
        # the top ``LIST_LIMIT`` rows on an ordinary tick, so anything "Load
        # older" has widened the window with lives here, untouched, until
        # ``reset_window`` drops it or a wider window asks ``read`` for more of
        # it than is already held. Re-reading the whole window on every tick
        # was the defect: every row past the first page paid its
        # ``attach_files`` stat again, forever, for rows nothing had changed.
        self._old_rows: list[dict[str, Any]] = []
        # What ``_dirty`` was the moment the in-flight read was started, for
        # the COUNT(*) cadence in :meth:`adopt` -- ``was_dirty`` used to be a
        # local in ``tick``; now the read and the adopt are different calls,
        # so it has to survive between them.
        self._read_was_dirty = False
        # The last (generation, filter fields) a widen ran for, so it is not
        # re-run every frame draw() calls it on -- only when a widenable field
        # changes or ``tick`` has replaced ``jobs`` and thrown the merge away.
        self._search_key: tuple[Any, ...] | None = None

    def invalidate(self) -> None:
        """Refresh on the next tick. Called after anything the UI did that
        changes the list -- a submit, a delete, a rename."""
        self._dirty = True

    def max_limit(self) -> int:
        """The service's ceiling, read at call time.

        Through the facade rather than from ``validation`` directly, for
        ``_jobs_list.list_jobs``' reason exactly: the ceiling is patched on
        ``service.jobs`` by tests, and this is another reader of it.
        """
        from ..service import jobs as _facade

        return int(_facade.MAX_LIST_LIMIT)

    def can_load_more(self) -> bool:
        """Whether widening the window would do anything. -> for the button."""
        return self.limit < self.max_limit()

    def load_more(self) -> None:
        """Widen the window by one page, up to the ceiling.

        A bigger single read rather than a merge of pages: tick() is one
        list_jobs call by design, and the per-row attach_files cost only grows
        when the user asks to see further back.

        **Clamped here rather than left to the service.** ``list_jobs`` clamps
        a read at ``MAX_LIST_LIMIT``, which made the query safe and left this
        counter growing without bound -- so past the ceiling every press moved
        ``limit`` and changed nothing else, and "Load older" stayed on screen
        offering to do it again.
        """
        self.limit = min(self.limit + LIST_LIMIT, self.max_limit())
        self.invalidate()

    def reset_window(self) -> None:
        """Back to the newest page (O119).

        The window only ever grew, and after a few presses of "Load older" the
        list is thousands of rows the user has to scroll back through -- with
        restarting the app as the only way to the top. Every per-row cost the
        widening bought (``attach_files``, the thumbnail cache) shrinks with
        it, which is the other half of why it is worth having.
        """
        if self.limit == LIST_LIMIT:
            return
        self.limit = LIST_LIMIT
        self.invalidate()

    def _due(self) -> bool:
        now = time.monotonic()
        return self._dirty or now >= self._next_refresh

    def read(self, files_snapshot: dict[str, Any]) -> dict[str, Any]:
        """The blocking half: one sqlite read plus ``attach_files``' stat walk.

        **Off the frame thread only** -- this is the read A2 exists to move
        there. Touches no attribute on ``self`` besides reading ``self.limit``
        and ``self._old_rows`` (never mutated here, only by :meth:`adopt`), so
        it is safe to hand to :class:`TaskRunner`; the caller must hand it a
        *copy* of ``self._files`` (``files_snapshot``), because this mutates
        that dict in place exactly as ``attach_files`` always has, and the
        frame thread must not see those writes until :meth:`adopt` publishes
        them.

        Only the newest ``LIST_LIMIT`` rows are re-read here -- "Load older"
        rows already held in ``self._old_rows`` are reused rather than
        re-fetched, and only the delta between what is held and what
        ``self.limit`` now asks for is pulled in, by keyset cursor, page by
        page. That is what stops "Load older" from turning into a re-read (and
        a re-stat) of the whole growing window on every tick (O119/A2).

        -> ``{"jobs": [...], "old": [...], "files": files_snapshot}`` or
        ``{"error": str}`` for :meth:`adopt` to publish.
        """
        try:
            top_size = min(self.limit, LIST_LIMIT)
            top = svc_jobs.list_jobs(self.svc, top_size, files_cache=files_snapshot)
            target_old = max(0, self.limit - LIST_LIMIT)
            old = list(self._old_rows[:target_old])
            while len(old) < target_old:
                tail = old[-1] if old else (top[-1] if top else None)
                if tail is None:
                    break
                before = (tail.get("created_at") or 0.0, tail.get("id") or "")
                page = svc_jobs.list_jobs(
                    self.svc,
                    min(LIST_LIMIT, target_old - len(old)),
                    before=before,
                    files_cache=files_snapshot,
                )
                if not page:
                    break
                old.extend(page)
                if len(page) < LIST_LIMIT:
                    break
        except Exception as exc:  # a locked DB, a vanished file
            log.exception("could not read the job list")
            return {"error": str(exc)}
        return {"jobs": top, "old": old, "files": files_snapshot}

    def adopt(
        self,
        reading: dict[str, Any],
        on_transition: Callable[[dict[str, Any], str | None], None] | None = None,
    ) -> bool:
        """Frame-thread half of :meth:`read` -- publish a reading. -> whether
        it landed (a ``{"error": ...}`` reading, or a stale one from a task
        started before the last :meth:`reset_window`, does not).

        This is the one place that ever assigns ``jobs``, ``by_id`` and
        ``_last_status``, and the one place that ever fires ``on_transition``
        -- exactly the property :meth:`tick` had, moved here so a call
        submitted through :class:`TaskRunner` can share it.
        """
        if not isinstance(reading, dict):
            return False
        error = reading.get("error")
        if error:
            self.error = str(error)
            return False
        top = reading.get("jobs")
        if top is None:
            return False
        old = reading.get("old") or []
        self._old_rows = old
        jobs = top + old
        self.error = None
        self.jobs = jobs
        self._files = reading.get("files", self._files)
        self._generation += 1
        now = time.monotonic()
        # Adaptive cadence (L102): fast only while a job is live -- that is the
        # only time a row can change without the UI having called invalidate.
        live = any(j.get("status") in ("queued", "running") for j in jobs)
        self._next_refresh = now + (REFRESH_SECONDS if live else IDLE_REFRESH_SECONDS)
        # Whatever fell off the page cannot be asked for again without a
        # re-read, so its entry is dead weight.
        if len(self._files) > len(jobs):
            live_ids = {j["id"] for j in jobs}
            self._files = {k: v for k, v in self._files.items() if k in live_ids}
        # COUNT(*) only when it can disagree with len(jobs) (A3): a page that
        # is not full *is* the whole history. A full page re-counts on a longer
        # cadence, and immediately after anything the UI did (invalidate).
        if len(jobs) < self.limit:
            self.total = len(jobs)
            self.count_error = None
        elif self._read_was_dirty or now >= self._next_count:
            self._next_count = now + COUNT_SECONDS
            try:
                self.total = self.svc.store.count()
            except Exception as exc:  # a count is not worth failing the refresh over
                log.exception("could not count the job list")
                self.total = len(jobs)
                self.count_error = str(exc)
            else:
                self.count_error = None
        self.by_id = {j["id"]: j for j in jobs}
        if on_transition is not None:
            for job in jobs:
                previous = self._last_status.get(job["id"])
                if previous is not None and previous != job["status"]:
                    on_transition(job, previous)
        self._last_status = {j["id"]: j["status"] for j in jobs}
        return True

    def request(
        self,
        runner: Any,
        on_transition: Callable[[dict[str, Any], str | None], None] | None = None,
    ) -> bool:
        """Submit :meth:`read` to ``runner`` if a refresh is due. -> whether a
        read was submitted this frame -- not whether new data landed, which
        only :meth:`adopt` (called from wherever ``runner``'s result is
        collected) can say.

        **Frame thread.** This, not :meth:`tick`, is what the app's own frame
        loop calls: the read is what A2 moves off this thread, via
        ``TaskRunner.submit``, which refuses a key already in flight -- so if
        the previous read has not landed yet, this frame's request is simply
        skipped rather than queued, and the next due frame tries again.
        ``on_transition`` is not used here; the caller passes the same
        callback to :meth:`adopt` once the task's result comes back.
        """
        if not self._due():
            return False
        self._read_was_dirty = self._dirty
        self._dirty = False
        return bool(runner.submit("jobs-list", self.read, dict(self._files)))

    def tick(self, on_transition: Callable[[dict[str, Any], str | None], None] | None = None):
        """The synchronous form of :meth:`request` + :meth:`adopt`, for a
        caller with no frame loop to route a task result through -- a script,
        a test, or a headless harness. Blocks, exactly like
        :meth:`refresh_storage` beside it and for the same reason; the app
        itself calls :meth:`request` instead. -> whether the list was re-read.
        """
        if not self._due():
            return False
        self._read_was_dirty = self._dirty
        self._dirty = False
        reading = self.read(dict(self._files))
        if reading.get("error"):
            self.error = str(reading["error"])
            return False
        return self.adopt(reading, on_transition)

    def refresh_storage(self) -> None:
        """Measure the data directory now, and publish the reading.

        **Blocking** -- callers off the frame thread only.

        **The app deliberately does not call this, and that is the design
        rather than an oversight.** ``main.App._request_storage`` submits
        :meth:`measure`, which returns the reading instead of assigning it, so
        the frame thread adopts it when the task result comes back; two tests
        pin exactly that split (``test_measuring_storage_is_reachable_without
        _touching_the_cache`` and ``test_requesting_storage_submits_the_non
        _publishing_measurement``), and a third pins that ``_refresh`` never
        reaches this method on the frame thread. Wiring this into the task
        would publish ``storage`` from the measuring thread and contradict all
        three.

        What it survives as is the measure-and-adopt form for a caller that has
        no frame loop to route a result through -- a script, a test, or a
        headless harness. It is not dead code looking for a consumer; it is the
        synchronous half of a split whose asynchronous half the app uses.
        """
        self.adopt_storage(self.measure())

    def measure(self) -> dict[str, Any]:
        """The full measurement, safe to call from a task thread.

        **A reading, not a publication.** Nothing on this object is touched:
        the walk runs on a task and ``_dir_sizes``, ``_sizes_generation`` and
        ``storage_error`` are read on the frame thread, so the amendment is
        :meth:`adopt_storage`'s (the review's theme T3). Decomposed per job
        directory so :meth:`measure_one` can fold a single finished job back in
        without re-walking the whole tree.
        """
        try:
            sizes = svc_jobs.storage_sizes(self.svc)
        except Exception as exc:
            log.exception("could not measure storage")
            return {"error": str(exc)}
        return {"sizes": sizes}

    def measure_one(self, job_id: str) -> dict[str, Any]:
        """Incremental storage accounting (C33): re-measure one job directory.

        Sound because job directories are disjoint: only this job's entry can
        have changed when this job finished. Falls back to the full walk when
        no baseline exists yet -- adding one directory to a total that was
        never measured would present a single job as the whole workshop.

        A reading, like :meth:`measure`: the fold it describes is applied by
        :meth:`adopt_storage`, on the frame thread that reads the totals.
        """
        if not self._dir_sizes:
            return self.measure()
        try:
            path = self.svc.job_dir(job_id)
            size = dir_size(path)
        except Exception as exc:
            log.exception("could not measure job dir %s", job_id)
            return {"error": str(exc)}
        return {"fold": (path.name, size if (size or path.exists()) else None)}

    def adopt_storage(self, reading: Any) -> None:
        """Publish a reading from :meth:`measure` / :meth:`measure_one`.

        **Frame thread only**, which is the whole point: the sizes back the
        library's size sort and ``_sizes_generation`` is in the ``visible``
        memo's key, so a task amending them mid-frame is a list re-sorted under
        a reader. A failed walk keeps the last good figure and says why -- a
        stale number beats a blank one, but not silently (E45).
        """
        if not isinstance(reading, dict):
            return
        error = reading.get("error")
        if error:
            self.storage_error = str(error)
            return
        self.storage_error = None
        if "sizes" in reading:
            self._dir_sizes = dict(reading["sizes"])
        elif "fold" in reading:
            name, size = reading["fold"]
            if size is None:
                self._dir_sizes.pop(name, None)
            else:
                self._dir_sizes[name] = int(size)
        else:
            return
        self._sizes_generation += 1
        self.storage = {
            "job_dirs": len(self._dir_sizes),
            "bytes": sum(self._dir_sizes.values()),
        }

    # -- queries -----------------------------------------------------------

    def get(self, job_id: str | None) -> dict[str, Any] | None:
        return None if job_id is None else self.by_id.get(job_id)

    def request_widen(self, filters: Any, runner: Any) -> bool:
        """W2.1, widened for A3: pull in matches the loaded window does not cover.

        Filtering only ever ran over ``self.jobs`` -- the newest page the
        cache happened to have loaded -- so searching for a job the pager had
        not reached yet found nothing, and "Load older" was the only way to
        it. This asks the store for ids the *window itself* would never have
        surfaced (by free text, ``tag:``/``name:`` field terms, status,
        favourites and the trash/workshop split -- everything ``search_ids``
        can turn into a real column predicate) and merges their rows in;
        ``Filters.matches`` still decides whether any of them actually match
        -- this only widens what it is asked about, never more permissively.

        Called from both ``panes/library.py`` and ``panes/library_full.py``:
        the two views share one ``Filters`` and must never disagree about
        what a search finds.

        **Frame thread.** This used to call ``self.svc.store.search_ids``
        inline -- a real sqlite query behind ``JobStore``'s shared RLock, run
        synchronously here on essentially every keystroke in the filter box,
        the exact stall :meth:`read`/:meth:`request`/:meth:`adopt`'s split
        exists to prevent for the ordinary poll (the 2026-09-08 audit, finding
        shell-01). The query now goes through :meth:`_search` via
        ``runner.submit``, the same door ``request`` sends the list poll
        through; the caller collects the result and hands it to
        :meth:`adopt_widen` when it lands, exactly as ``main._on_task_done``
        already does for ``"jobs-list"``. -> whether a search was submitted
        this frame.

        Skipped once per (list generation, filter fields that reach the
        store): ``tick`` replaces ``self.jobs`` wholesale on every refresh,
        which throws any previous merge away, so a changed generation is
        exactly when this needs to run again -- and unchanged, running it
        every frame ``draw`` calls this on would be a LIKE scan per frame for
        nothing new. The key is only recorded once ``runner.submit`` actually
        accepts the search: a refusal (one already in flight) must be retried
        on a later frame rather than being mistaken for "already handled".
        """
        from .state import parse_query

        text = (filters.text or "").strip()
        terms, fields = parse_query(text) if text else ([], [])
        tags = tuple(v for f, v in fields if f == "tag")
        names = tuple(v for f, v in fields if f == "name")
        # No field terms: the raw text is the whole free-text query, exactly
        # as it always was. With field terms present, only the plain words
        # are still a name/prompt substring search -- ``tag:wood`` itself is
        # not a word to LIKE against ``name``/``prompt``.
        free_text = text if not fields else " ".join(terms)
        status = None if filters.status == "all" else filters.status
        favorite = filters.favorites_only or None
        active = bool(free_text or tags or names or status or favorite)
        key = (self._generation, free_text, tags, names, status, favorite, filters.trash)
        if not active:
            self._search_key = None
            return False
        if key == self._search_key:
            return False
        # The 2026-09-23 audit's shell-04: this used to be ``if not
        # can_load_more()``, which also skips the widen once ``limit`` hits
        # ``MAX_LIST_LIMIT`` (5000) -- but a store past that ceiling still has
        # rows outside the window, and ``can_load_more`` cannot tell "the
        # window covers the whole store" from "the window is as wide as it is
        # ever allowed to get". ``len(self.jobs) >= self.total`` is the real
        # question: whether anything is left outside the window at all.
        # ``total`` is 0 until the first read lands, which says nothing about
        # the store yet, so only a known total may skip the widen.
        if self.total and len(self.jobs) >= self.total:
            # The window already holds everything the store has -- there is
            # nothing outside it left to widen with.
            self._search_key = key
            return False
        submitted = bool(
            runner.submit(
                SEARCH_KEY,
                self._search,
                free_text,
                tags,
                names,
                status,
                favorite,
                filters.trash,
            )
        )
        if submitted:
            self._search_key = key
        return submitted

    def _search(
        self,
        free_text: str,
        tags: tuple[str, ...],
        names: tuple[str, ...],
        status: str | None,
        favorite: bool | None,
        trash: bool,
    ) -> dict[str, Any]:
        """The blocking half of :meth:`request_widen` -- one ``search_ids``
        call, off the frame thread. -> ``{"ids": [...]}`` or ``{"error": str}``
        for :meth:`adopt_widen` to publish.
        """
        try:
            ids = self.svc.store.search_ids(
                free_text,
                limit=SEARCH_LIMIT,
                tags=tags,
                names=names,
                status=status,
                favorite=favorite,
                trash=trash,
            )
        except Exception as exc:
            log.exception("could not search the job list")
            return {"error": str(exc)}
        return {"ids": ids}

    def adopt_widen(self, reading: Any) -> None:
        """Frame-thread half of :meth:`request_widen` -- merge a
        :meth:`_search` reading into ``self.jobs``.

        Called from wherever ``runner``'s result is collected, keyed on
        :data:`SEARCH_KEY` -- ``main._on_task_done`` does that for the app.
        """
        if not isinstance(reading, dict):
            return
        ids = reading.get("ids")
        if ids is None:
            return
        missing = [i for i in ids if i not in self.by_id]
        if not missing:
            return
        for job_id in missing:
            try:
                job = svc_jobs.get_job(self.svc, job_id)
            except Exception:
                log.exception("could not load search match %s", job_id)
                continue
            self.jobs.append(job)
            self.by_id[job_id] = job
        self.jobs.sort(key=lambda j: (j.get("created_at") or 0.0, j.get("id") or ""), reverse=True)
        # The shape of ``self.jobs`` changed under whatever ``visible``/
        # ``failures`` last memoized -- invalidate directly rather than
        # bumping ``_generation``, which would immediately fail the ``key ==
        # self._search_key`` check in :meth:`request_widen` and re-run the
        # search next frame.
        self._visible_memo = None
        self._failures_memo = None

    def _filters_key(self, filters: Any) -> Any:
        """A hashable snapshot: the generation plus every filter field. The
        fields are strings and bools, so ``vars`` is the whole state.

        ``_sizes_generation`` joins it because sort-by-size reads the storage
        walk (J85), which lands on a task thread long after the list did:
        without it the first measurement would never reorder anything.
        """
        return (
            self._generation,
            self._sizes_generation,
            tuple(sorted(vars(filters).items())),
        )

    def visible(self, filters: Any) -> list[dict[str, Any]]:
        """The filtered, ordered list -- memoized per (list generation,
        filter state), because the library recomputes it every frame (B19)."""
        key = self._filters_key(filters)
        memo = self._visible_memo
        if memo is not None and memo[0] == key:
            return memo[1]
        out = filters.order(
            [j for j in self.jobs if filters.matches(j)], sizes=self._dir_sizes
        )
        self._visible_memo = (key, out)
        return out

    def failures(self, filters: Any) -> int:
        """``filters.failures`` over the loaded page, memoized like
        :meth:`visible` and for the same reason."""
        key = self._filters_key(filters)
        memo = self._failures_memo
        if memo is not None and memo[0] == key:
            return memo[1]
        count = filters.failures(self.jobs)
        self._failures_memo = (key, count)
        return count

    @property
    def active(self) -> dict[str, Any] | None:
        """The job worth narrating: whatever is running, else whatever is queued."""
        for status in ("running", "queued"):
            for job in self.jobs:
                if job["status"] == status:
                    return job
        return None


def transition_message(job: dict[str, Any], previous: str | None) -> tuple[str, str] | None:
    """-> (text, level) for a status change worth a toast, or None.

    Only terminal transitions: a queued job becoming running is what the
    progress card is for, and a toast for it would fire on every submit.
    """
    name = job.get("name") or job.get("prompt") or job["id"]
    name = name if len(name) <= 40 else name[:37] + "..."
    if job["status"] == "done":
        # ``success`` rather than ``info`` (H68): a finished job is the one
        # unambiguously good thing this function reports, and it spent its
        # whole life in the same neutral grey as "settings copied to the form".
        #
        # The noun comes from ``followups.PRODUCTS`` where there is one: a
        # follow-up row is minted with its *source's* prompt, so "fire guardian
        # finished." fired twice for one character with the same name both
        # times and said nothing about which half had landed -- while the
        # ``Show`` beside it opened two entirely different places.
        product = followups.PRODUCTS.get(str(job.get("kind") or ""))
        if product:
            return f"{product} for {name} is ready.", "success"
        return f"{name} finished.", "success"
    if job["status"] == "error":
        return f"{name} failed: {job.get('error') or 'unknown error'}", "error"
    if job["status"] == "cancelled" and previous == "running":
        return f"{name} cancelled.", "info"
    return None


def sweep_summary(jobs: list[dict[str, Any]], sweep_id: str) -> tuple[str, str] | None:
    """-> (text, level) once every loaded unit of a sweep has finished (N109).

    ``None`` while any of them is still queued or running, which is what makes
    one call per finished unit collapse into exactly one toast at the end.

    Judged against the *loaded* window, and that is a real limitation rather
    than an oversight: the list is the newest N of M, so a sweep longer than
    the window could be declared finished early. It is the honest trade -- the
    alternative is a ``COUNT`` on the frame thread, which the single-connection
    rule forbids -- and it fails in the safe direction: an early summary is a
    slightly wrong number, where a missed one is silence at the end of an
    overnight run.
    """
    units = [job for job in jobs if job.get("sweep_id") == sweep_id]
    if not units or any(job.get("status") in ("queued", "running") for job in units):
        return None
    done = sum(1 for job in units if job.get("status") == "done")
    failed = sum(1 for job in units if job.get("status") == "error")
    cancelled = len(units) - done - failed
    parts = [f"{done} done"]
    if failed:
        # *Refused* rather than *failed*: the dominant error here is the
        # composition gate declining the reference, which is a measurement
        # rather than a fault, and the word decides whether the user goes
        # looking for a bug.
        parts.append(f"{failed} refused")
    if cancelled:
        parts.append(f"{cancelled} cancelled")
    level = "warn" if failed and not done else "success"
    return f"Sweep finished - {', '.join(parts)}.", level
