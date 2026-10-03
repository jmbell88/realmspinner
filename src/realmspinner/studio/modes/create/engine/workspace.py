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
            # Explicit workspace IDs sort before lineage nodes.
            parents[max(left, right)] = min(left, right)

    rows = [
        j
        for j in jobs
        if not j.get("deleted_at")
        and (j.get("kind") in ASSET_KINDS | FOLLOWUP_KINDS or (j.get("params") or {}).get("built"))
    ]
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


def journey(asset_type: str, *, has_mesh: bool = False) -> tuple[str, ...]:
    if has_mesh or asset_type in ("3d_model", "character"):
        return ("reference", "mesh", "rig", "pose", "export")
    return ("reference", "export")
