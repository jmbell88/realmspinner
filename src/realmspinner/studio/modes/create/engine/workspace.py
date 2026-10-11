"""Creation families derived from persisted provenance, including legacy jobs.

Workspace identity groups attempts; it never changes pipeline stage or ownership
of an artifact. Missing ancestors remain graph nodes so a paged history still
groups siblings, and cycles cannot hang a frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WORKSPACE_PARAM = "create_workspace"
LEGACY_ROOTS = ("job:", "group:")
ASSET_KINDS = frozenset({"text", "image", "tile_sheet", "character"})
FOLLOWUP_KINDS = frozenset(
    {"rig", "sheet", "pixel_sheet", "sprite_synthesis", "charsheet", "retexture", "remesh"}
)


@dataclass
class Index:
    by_job: dict[str, str]
    families: dict[str, list[dict[str, Any]]]


def build_index(jobs: list[dict[str, Any]]) -> Index:
    parents: dict[str, str] = {}
    #: Nodes a row's ``create_workspace`` stamp names. A stamp is the key the
    #: creation was filed under when the row was submitted, so it is the one
    #: identity that must survive the family growing.
    stamped: set[str] = set()

    def rank(key: str) -> tuple[bool, str]:
        return (key not in stamped, key)

    def root(key: str) -> str:
        parents.setdefault(key, key)
        cursor = key
        while parents[cursor] != cursor:
            cursor = parents[cursor]
        while parents[key] != key:
            key, parents[key] = parents[key], cursor
        return cursor

    def join(a: str, b: str) -> None:
        left, right = root(a), root(b)
        if left != right:
            # Explicit workspace IDs sort before lineage nodes, and a node some
            # row is *stamped* with outranks any that none is: the 2026-10-04
            # audit, finding create-30. A legacy family's key is a job id, and
            # job ids are random hex, so when a new attempt stamped "job:bbb"
            # had an id that sorted below it ("job:aaa") the plain minimum moved
            # the root to the new row and the tray, which still held
            # "job:bbb", listed no attempts and filed its drafts under a dead key.
            parents[max(left, right, key=rank)] = min(left, right, key=rank)

    rows = [
        j
        for j in jobs
        if not j.get("deleted_at")
        and (j.get("kind") in ASSET_KINDS | FOLLOWUP_KINDS or (j.get("params") or {}).get("built"))
    ]
    for job in rows:
        workspace = (job.get("params") or {}).get(WORKSPACE_PARAM)
        if isinstance(workspace, str) and workspace:
            stamped.add(
                workspace if workspace.startswith(LEGACY_ROOTS) else "creation:" + workspace
            )
    for job in rows:
        key = "job:" + str(job["id"])
        root(key)
        params = job.get("params") or {}
        workspace = params.get(WORKSPACE_PARAM)
        if isinstance(workspace, str) and workspace:
            # A legacy family's root is itself a graph node ("job:<id>" or
            # "group:<id>"); a row stamped with that spelling joins the node
            # the family already hangs from rather than a fresh "creation:"
            # island nothing else is attached to (create-05).
            join(key, workspace if workspace.startswith(LEGACY_ROOTS) else "creation:" + workspace)
        for ancestor in (
            job.get("parent_id"),
            params.get("rerun_of"),
            params.get("source_job") if job.get("kind") in FOLLOWUP_KINDS else None,
        ):
            if ancestor:
                join(key, "job:" + str(ancestor))
        if job.get("candidate_group"):
            join(key, "group:" + str(job["candidate_group"]))
    by_job = {str(j["id"]): root("job:" + str(j["id"])) for j in rows}
    families: dict[str, list[dict[str, Any]]] = {}
    for job in rows:
        families.setdefault(by_job[str(job["id"])], []).append(job)
    for family in families.values():
        family.sort(key=lambda j: (j.get("created_at") or 0, str(j["id"])), reverse=True)
    return Index(by_job, families)


def results(index: Index, key: str | None, stage: str | None = None) -> list[dict[str, Any]]:
    return [
        j
        for j in index.families.get(key or "", [])
        if j.get("kind") not in FOLLOWUP_KINDS
        and (stage is None or (j.get("stage") == "model") == (stage == "mesh"))
    ]


#: How far apart, in seconds, two reference rows of one press can be created. A
#: Count of 8 inserts its rows inside one transaction (``_jobs_create``), so
#: they land milliseconds apart; a person pressing Generate again is seconds at
#: the very least. Two seconds is wide enough for a slow disk and narrow enough
#: that the next press is never folded in.
BATCH_WINDOW_S = 2.0

#: The largest batch ``MAX_REFERENCE_COUNT`` allows, which the grid is laid out for.
GRID_MAX = 8


def reference_batch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The newest press's reference images, in the order they were drawn. **Pure.**

    ``rows`` is one creation's results (``results(..., "reference")``, newest
    first). A press with Count above 1 writes N independent rows and no
    ``candidate_group`` (that column is the *mesh* picker's), so the submit door
    stamps ``params["press_id"]`` on every image of the press and a batch is the
    rows sharing it. Rows from before that fall back to the clock: the newest row
    and every image-kind sibling made within :data:`BATCH_WINDOW_S` of it from the
    same prompt. A creation's older
    presses, a rerun and a hand-made reference are never folded in.
    Returns ``[]`` when there is nothing to group, and a single row for an
    ordinary press: the caller decides that one image is not a contact sheet.
    """
    members = [
        j for j in rows if j.get("stage") == "reference" and j.get("kind") in ("text", "image")
    ]
    if not members:
        return []
    newest = members[0]
    newest_params = newest.get("params") or {}
    press = newest_params.get("press_id")
    if press and not newest_params.get("rerun_of"):
        # The submit door stamped one id on every image of the press: exact, and
        # immune to a slow submit spreading the rows past the window.
        batch = [
            j
            for j in members
            if (j.get("params") or {}).get("press_id") == press
            and not (j.get("params") or {}).get("rerun_of")
        ]
        batch.sort(key=lambda j: (j.get("created_at") or 0, str(j["id"])))
        return batch[:GRID_MAX]
    stamp = float(newest.get("created_at") or 0.0)
    batch = [
        j
        for j in members
        if abs(stamp - float(j.get("created_at") or 0.0)) <= BATCH_WINDOW_S
        and (j.get("prompt") or "") == (newest.get("prompt") or "")
        and not (j.get("params") or {}).get("rerun_of")
        # A row of some other press is never folded into an older one by
        # proximity alone; only rows from before ``press_id`` rely on the clock.
        and not (j.get("params") or {}).get("press_id")
    ]
    if not batch:
        batch = [newest]
    batch.sort(key=lambda j: (j.get("created_at") or 0, str(j["id"])))
    return batch[:GRID_MAX]


def batch_settled(batch: list[dict[str, Any]]) -> bool:
    """Whether every member has reached a terminal status. **Pure.**"""
    return bool(batch) and all(j.get("status") in ("done", "error", "cancelled") for j in batch)


def grid_shape(count: int) -> tuple[int, int]:
    """``(rows, columns)`` of the contact sheet for ``count`` images. **Pure.**

    Two sit side by side, three or four make a square, five to eight two rows
    of four -- the shapes the review asked for, chosen so a landscape canvas
    keeps its images large. Anything outside 2..8 is clamped into that range.
    """
    n = max(2, min(int(count), GRID_MAX))
    if n == 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    return 2, 4


def grid_step(index: int, key: str, count: int) -> int:
    """Where an arrow key moves the grid's selection. **Pure.**

    Clamped, not wrapped: the grid's edges mean something and a wrap would
    jump the eye across the sheet. Up and Down move by a row, so they only
    change anything when the shape has two.
    """
    rows, cols = grid_shape(count)
    step = {"left": -1, "right": 1, "up": -cols, "down": cols}.get(key, 0)
    if rows == 1 and key in ("up", "down"):
        return index
    return max(0, min(index + step, count - 1))


#: The five stages in rail order. ``ui/stages.STAGES`` is the same tuple; the
#: engine may not import ``ui``, so :func:`journey` keeps its own copy to place a
#: stage it adds. ``tests/modes/create/test_audit_2026_10_04_workspace.py``
#: asserts the two are equal (the 2026-10-04 audit, finding create-46): a rail
#: stage added to one and not the other would otherwise be silently missing
#: from every journey.
STAGE_ORDER = ("reference", "mesh", "rig", "pose", "export")


def journey(
    asset_type: str, *, has_mesh: bool = False, current: str | None = None
) -> tuple[str, ...]:
    """The rail's stages for this asset.

    *current* is the stage the user is standing on and is always in the answer:
    Home's "New 3D model", a file drop and Library's Make 3D all ``go`` to Mesh
    without touching an Image/Material/Tileset/Sprite form, whose own journey
    has no Mesh segment, and the rail was drawn without it and parked its pill
    under Reference (the 2026-10-03 audit, finding create-16).
    """
    if has_mesh or asset_type in ("3d_model", "character"):
        stages = set(STAGE_ORDER)
    else:
        stages = {"reference", "export"}
    if current in STAGE_ORDER:
        stages.add(current)
    return tuple(stage for stage in STAGE_ORDER if stage in stages)
