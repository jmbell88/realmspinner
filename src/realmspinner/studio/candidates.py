"""Which mesh candidates are still undecided, and what the picker says about them.

The ``clay_mode``/``review_mode``/``matte_preview`` shape: no imgui here at
all, so every rule about *which* group is offered and *when* it is finished is
assertable headlessly. ``panes/candidates_panel`` draws it.

The one design decision worth stating is why nothing is remembered. A group
could have been recorded on ``AppState`` when it was submitted -- but the
library hides an undecided candidate (``Filters.matches``, the same rule sweep
units follow), so a remembered group that was forgotten by a restart, or by the
user clicking something else, would leave a handful of rows nobody could reach
and nothing could offer. Deriving the group from the rows themselves makes that
impossible: while an undecided candidate exists it is offered, and once the
group is decided there is nothing left to derive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Terminal statuses, i.e. the ones a candidate can be judged from. ``cancelled``
# counts: the user stopped that one, and waiting for it is waiting forever.
_SETTLED = ("done", "error", "cancelled")


@dataclass(frozen=True)
class Group:
    """One undecided candidate group, in the order the seeds were drawn."""

    group: str
    members: list[dict[str, Any]]

    @property
    def done_count(self) -> int:
        return sum(1 for m in self.members if m.get("status") in _SETTLED)

    @property
    def finished(self) -> bool:
        """Whether every member has reached a terminal status.

        The picker is drawn either way -- watching two candidates arrive is the
        point of asking for two -- but Keep is only offered once the group has
        settled, because keeping one dissolves the group and a member still
        queued would become an ordinary asset nobody chose.
        """
        return bool(self.members) and self.done_count == len(self.members)

    def losers(self, keep_id: str) -> list[str]:
        return [m["id"] for m in self.members if m["id"] != keep_id]

    @property
    def all_failed(self) -> bool:
        """Whether every member has settled and none reached ``done``.

        The 2026-09-14 audit, finding create-04: ``finished`` alone is true
        of this group too -- a mix of done and failed settles it exactly as
        completely as an all-done one -- so the Keep gate (``finished and
        member done``) never opens for a single member here, and nothing
        else offered a way out: no dismiss on either picker, and
        ``state.Filters.matches`` hides every row carrying
        ``candidate_group`` from the library regardless. This is the
        question both pickers ask instead, to offer Discard in Keep's place.
        """
        return self.finished and not any(m.get("status") == "done" for m in self.members)


def pending(jobs: list[dict[str, Any]]) -> Group | None:
    """The newest undecided group among ``jobs``, or None.

    Newest by the group's own newest row, so a group submitted while an older
    one is still undecided is the one on screen -- the same "most recent thing
    you asked for" rule the selection follows. Only one group is ever offered:
    two pickers stacked in a sidebar is a choice about which choice to make.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        group = job.get("candidate_group")
        if group:
            groups.setdefault(str(group), []).append(job)
    if not groups:
        return None
    newest = max(
        groups.items(),
        key=lambda item: max((j.get("created_at") or 0.0, j["id"]) for j in item[1]),
    )
    members = sorted(newest[1], key=lambda j: (int(j.get("candidate_index") or 0), j["id"]))
    return Group(newest[0], members)


#: One memoized answer: ``(key, Group | None)`` for :func:`pending_cached`.
#: Module-level, like ``panes.candidates_panel._GRADES_CACHE`` beside it --
#: Create only ever shows one cache's tray at a time, so one slot is enough.
_PENDING_CACHE: tuple[Any, Group | None] | None = None


def pending_cached(cache: Any) -> Group | None:
    """Memoized :func:`pending`, keyed on ``cache``'s generation counter.

    The 2026-09-19 audit, finding create-01: ``pending`` did a full linear
    scan of ``cache.jobs`` (up to ``MAX_LIST_LIMIT`` = 5000 rows once "Load
    older" has widened the window), building a fresh ``groups`` dict and
    sorting every group's members, with no memo at all -- and every one of
    its call sites (``workspace.should_draw``, ``workspace.draw``,
    ``brief._with_pending_candidates_problem`` and ``candidates_panel.draw``)
    runs every frame the Create canvas is visible, so one frame paid the
    scan three or four times over though nothing had changed since the last
    one. Keyed on ``cache._generation`` the way
    ``candidates_panel._grades`` already keys its own memo against the
    identical counter -- it only moves when ``JobsCache.adopt`` actually
    publishes a fresh read, never on a frame that changed nothing.

    ``pending`` itself stays pure and untouched (it is tested directly in
    ``tests/studio/test_candidates.py``); this only wraps it. A ``None``
    generation -- a headless stand-in with no real cache, as most tests here
    build -- never memoizes, since there is nothing behind it that can go
    stale to avoid re-scanning.

    The key holds ``cache`` itself, not ``id(cache)``. The 2026-09-20 audit,
    finding create-05: CPython reuses a freed object's address, so a bare id
    can name a cache that no longer exists -- reproduced in 19,993 of 20,000
    create-destroy-create cycles against a fresh cache at generation 0. A
    strong reference to the actual object can never be fooled that way, and
    it costs nothing extra here: this module already keeps the one cache
    Create ever shows alive for as long as the app runs.
    """
    global _PENDING_CACHE
    generation = getattr(cache, "_generation", None)
    key = (cache, generation)
    if generation is not None and _PENDING_CACHE is not None and _PENDING_CACHE[0] == key:
        return _PENDING_CACHE[1]
    result = pending(cache.jobs)
    if generation is not None:
        _PENDING_CACHE = (key, result)
    return result


def label(member: dict[str, Any]) -> str:
    """What one candidate's button says. Latin-1 only, like every other UI
    string here -- imgui's default atlas has nothing above it."""
    index = int(member.get("candidate_index") or 0)
    return f"#{index + 1}"


def status_line(member: dict[str, Any]) -> str:
    status = str(member.get("status") or "")
    seed = (member.get("params") or {}).get("mesh_seed")
    if seed is None:
        return status
    return f"{status} · seed {seed}"
