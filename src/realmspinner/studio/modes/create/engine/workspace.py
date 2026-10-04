"""Creation families derived from persisted provenance, including legacy jobs.

Workspace identity groups attempts; it never changes pipeline stage or ownership
of an artifact. Missing ancestors remain graph nodes so a paged history still
groups siblings, and cycles cannot hang a frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    #: :func:`ordered_creations`' answer, filled on first ask. It lives on the
    #: index because the index is exactly as long-lived as the cache generation
    #: it was built from (``session.index`` rebuilds it when jobs change), so
    #: the list is rebuilt on the same key and never on a frame that changed
    #: nothing (the 2026-10-03 audit, finding create-44).
    _ordered: list[tuple[str, list[dict[str, Any]]]] | None = field(
        default=None, repr=False, compare=False
    )


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


def ordered_creations(index: Index) -> list[tuple[str, list[dict[str, Any]]]]:
    """Every creation that has a result, newest activity first, as
    ``(family key, its results)``. Memoised on the index; do not mutate."""
    if index._ordered is None:
        ordered = sorted(
            index.families.items(),
            key=lambda pair: max(j.get("created_at") or 0 for j in pair[1]),
            reverse=True,
        )
        index._ordered = [
            (key, assets) for key, _family in ordered if (assets := results(index, key))
        ]
    return index._ordered


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
