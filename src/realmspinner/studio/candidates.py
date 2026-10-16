"""Which mesh candidates are still undecided, and what the picker says about them.

The ``clay_mode``/``review_mode``/``matte_preview`` shape: no imgui here at
all, so every rule about *which* group is offered and *when* it is finished is
assertable headlessly. Create's results tray (``modes/create/ui/workspace.py``)
draws it, and ``panes/candidates_panel`` holds the keep/discard decisions.

Candidates are ordered best silhouette match first (``order_key``, from
``params["mesh_score"]``), ties and unscored ones by seed order -- a sort key, not
a verdict; ``candidates_panel`` says why that is allowed now.

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

from ..kernels.mesh import score as mesh_score

# Terminal statuses, i.e. the ones a candidate can be judged from. ``cancelled``
# counts: the user stopped that one, and waiting for it is waiting forever.
_SETTLED = ("done", "error", "cancelled")


def order_key(member: dict[str, Any]) -> tuple[Any, ...]:
    """Where one candidate sits in its strip: best match first, ties by seed order.

    ``params["mesh_score"]`` (``kernels.mesh.score``) is a sort key and never a verdict:
    this orders the strip and changes nothing else. Keep stays a click, no candidate is
    hidden, and an unscored one (still running, no reference to compare, a measurement
    that broke) sorts after every scored one -- so while attempts are still arriving the
    strip fills in at the back and the order only changes as scores land.
    """
    params = member.get("params") or {}
    return (
        mesh_score.rank_key(params.get("mesh_score")),
        int(member.get("candidate_index") or 0),
        str(member.get("id")),
    )


@dataclass(frozen=True)
class Group:
    """One undecided candidate group, best match first (see :func:`order_key`)."""

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
        if group and not (job.get("params") or {}).get("create_workspace"):
            groups.setdefault(str(group), []).append(job)
    if not groups:
        return None
    newest = max(
        groups.items(),
        key=lambda item: max((j.get("created_at") or 0.0, j["id"]) for j in item[1]),
    )
    return Group(newest[0], sorted(newest[1], key=order_key))


def batch(jobs: list[dict[str, Any]], group_id: str) -> Group | None:
    """Every row of one candidate group, ranked -- *including* a Create workspace's.

    ``pending`` deliberately skips a workspace's batch (it is ordinary work and gets no
    Keep picker), but its attempts are still N seeds of one reference and still worth
    ordering and pre-selecting by the same score. This is the group without that filter.
    """
    members = [j for j in jobs if str(j.get("candidate_group") or "") == str(group_id)]
    return Group(str(group_id), sorted(members, key=order_key)) if members else None


def rank_batches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``rows`` with each candidate group's members put best-match first, in place.

    A group's members take the slots its members already occupied, so a creation's
    strip stays newest-batch-first and only the order *inside* a batch changes; rows
    that belong to no group are untouched. Pure, and a no-op for a list with no group.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = row.get("candidate_group")
        if group:
            groups.setdefault(str(group), []).append(row)
    if not groups:
        return list(rows)
    ranked = {g: iter(sorted(members, key=order_key)) for g, members in groups.items()}
    return [
        next(ranked[str(row["candidate_group"])]) if row.get("candidate_group") else row
        for row in rows
    ]


def best(group: Group) -> dict[str, Any] | None:
    """The top-ranked finished candidate, or None when there is no basis to prefer one.

    Only a *finished* member with a real score qualifies: with nothing scored the order is
    just the seed order, and "pre-select the best" would then mean "pre-select #1", which
    is a pick the app has no grounds for.
    """
    for member in group.members:
        score = (member.get("params") or {}).get("mesh_score")
        if member.get("status") == "done" and mesh_score.rank_key(score)[0] == 0:
            return member
    return None


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
    ``brief.with_pending_candidates_problem`` and the tray's ``_candidate_grid``)
    runs every frame the Create canvas is visible, so one frame paid the
    scan three or four times over though nothing had changed since the last
    one. Keyed on ``cache._generation`` the way
    ``candidates_panel.grades_for`` already keys its own memo against the
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
