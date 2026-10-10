"""What is left before this asset is ready to hand to a game engine.

The 2026-09-07 Create review, item 3.3: the Export stage's inspector showed
only the reference thumbnails and the generation settings, while every fact a
person actually wants standing there -- is the triangle count in budget, does
it seal, does it have the maps an importer needs, did a rebuild silently lose
its grounding, does the rig on this asset still describe the mesh under it --
was either buried on the Mesh stage (:mod:`.panes.inspector`'s ``_quality``)
or recorded nowhere a person could read it at all.

This module does not measure anything. Every row is read off evidence another
part of the app already produced -- :mod:`realmspinner.meshreport` (triangle
count, watertightness, UVs, materials, pivot, achieved size),
``params["degraded"]`` (:mod:`realmspinner.service.validation`'s record of a
swallowed normalize/optimize failure) and the same fact
``panes.retarget_panel`` warns about before a retarget: a rig beside a mesh
whose measurements were never retaken. Nothing here loads a GLB, and nothing
here refuses anything -- ``meshreport``'s own docstring is explicit that it is
advisory, and this checklist inherits that stance rather than becoming the
first thing in the app that blocks an export.

Pure and headless: no imgui, no moderngl, no pygame. :mod:`.panes.inspector`
draws the rows this returns and wires the two repairs that already exist
elsewhere in the app (see the ``REPAIR_*`` constants).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .. import guidance, meshreport
from ..pipelines import modelhistory

# What a row's repair button does, named rather than a callable: this module
# draws nothing and calls no service, so a row can only *point at* an action
# the caller already owns. Wiring only what already exists is the review's own
# rule for this item -- a row with neither identifier below carries no button.
REPAIR_GO_TO_RIG = "go_to_rig"
# The Mesh stage is where retarget, remesh, retexture, Earlier meshes and the
# Finished-asset panel live (they used to sit behind the Rig stage, which a prop
# never visits). Two names because they answer two questions -- "rebuild this
# mesh" and "change this finished asset's size or pivot" -- even though both
# land on one stage today.
REPAIR_GO_TO_MESH = "go_to_mesh"
REPAIR_FIX_ASSET = "fix_asset"


@dataclass(frozen=True, slots=True)
class Row:
    """One line of the checklist: a fact, its verdict, and maybe a fix.

    ``state`` is ``"ok"`` or ``"attention"`` -- two words, not a colour, so the
    verdict survives a screenshot taken without one (and reads the same to
    someone who cannot see the tint at all). ``repair`` is one of the
    ``REPAIR_*`` constants or ``None``.
    """

    label: str
    state: str
    detail: str
    repair: str | None = None


def rows_for(job: Any, rig_mtime: float | None = None) -> list[Row]:
    """The checklist for ``job``, or ``[]`` when there is nothing to check.

    ``rig_mtime`` is when ``rig.json`` was last written, read by the one caller
    that may touch the disk; this module stays pure. ``None`` means "not known",
    and the stale-rig row then falls back to its older, weaker evidence.

    Empty rather than a checklist of assumptions: a reference at the Export
    stage (legal -- Export shows whatever the asset has) carries no
    ``mesh_report``, no ``degraded`` note and no rig, so every row below finds
    nothing to say and the list comes back empty. Drawing an all-green
    checklist in that case would claim a measurement that never ran, which is
    exactly the overclaim ``meshreport`` itself is written not to make.
    """
    if not isinstance(job, dict):
        return []
    params = job.get("params")
    params = params if isinstance(params, dict) else {}
    files = set(job.get("files") or [])

    rows: list[Row] = []

    report = params.get("mesh_report")
    if isinstance(report, dict):
        rows.extend(_report_rows(report, params, job.get("prompt")))

    degraded = params.get("degraded")
    if isinstance(degraded, dict) and degraded:
        rows.append(_degraded_row(degraded))

    stale = _stale_rig_row(params, files, rig_mtime)
    if stale is not None:
        rows.append(stale)

    return rows


def _effective_triangle_budget(report: dict[str, Any], params: dict[str, Any]) -> int:
    """The triangle ceiling this particular mesh was actually built against.

    The 2026-09-23 audit, finding pipelines-01: ``meshreport.TRIANGLE_BUDGET``
    (150k) is below the custom budgets the retarget and remesh panels accept
    (``pipelines.optimize.CUSTOM_MAX`` 250k, ``pipelines.remesh.TRIANGLES_MAX``
    200k), so a mesh optimized to its own accepted budget was flagged "over
    budget" forever, with this row's own repair button pointing back at the
    very panel that had already done exactly what it asked. The budget a
    mesh was actually built to is recorded on the job that built it --
    ``optimize.run``'s ``"requested"`` or ``remesh``'s ``"target_triangles"`` --
    so that is read first; ``report["triangle_budget"]`` (set by a caller of
    ``meshreport.build`` that already knew the budget) is the next fallback,
    and the module default is last, for a raw reconstruction that was never
    retargeted or a report recorded before either field existed.

    The 2026-09-23 (second run) audit, finding pipelines-01: a remesh
    record's ``target_faces`` is Blender's own quad count, not a triangle
    budget -- ``pipelines.remesh.target_faces`` halves the triangle budget
    before handing it to quadriflow, and ``_q_mesh.py`` records both:
    ``target_faces`` always, ``target_triangles`` (the real budget) only when
    the job was queued through the modern ``target_triangles`` ladder rather
    than a legacy ``target_faces`` request. Reading ``target_faces`` as the
    triangle budget flagged a remesh that landed exactly on its 5,000
    default -- "4,996 triangles is above the 2,500 budget". A legacy row
    that has only ``target_faces`` is doubled back into a triangle count
    instead, since that field was never anything but a quad count.
    """
    optimize_record = params.get("optimize")
    if isinstance(optimize_record, dict):
        requested = optimize_record.get("requested")
        if isinstance(requested, int) and requested > 0:
            return requested
    remesh_record = params.get("remesh")
    if isinstance(remesh_record, dict):
        triangles = remesh_record.get("target_triangles")
        if isinstance(triangles, int) and triangles > 0:
            return triangles
        faces = remesh_record.get("target_faces")
        if isinstance(faces, int) and faces > 0:
            return faces * 2
    stored = report.get("triangle_budget")
    if isinstance(stored, int) and stored > 0:
        return stored
    return meshreport.TRIANGLE_BUDGET


def _report_rows(report: dict[str, Any], params: dict[str, Any], prompt: Any) -> list[Row]:
    rows: list[Row] = []

    triangles = report.get("triangles")
    if isinstance(triangles, int):
        budget = _effective_triangle_budget(report, params)
        over = triangles > budget
        rows.append(
            Row(
                "Triangles",
                "attention" if over else "ok",
                (
                    f"{triangles:,} triangles is above the {budget:,} budget"
                    if over
                    else f"{triangles:,} of the {budget:,} triangle budget"
                ),
                # The over-budget mesh is exactly what a remesh (or a cheaper
                # retarget) exists to fix -- ``panes.remesh_panel`` and
                # ``panes.retarget_panel`` both live on the Mesh stage.
                repair=REPAIR_GO_TO_MESH if over else None,
            )
        )

    # ``welded_watertight`` is the reading that means what a person means by
    # "does it seal" (see ``inspector.watertight_value``'s own docstring for
    # why the raw flag is not that); the fallback keeps a report recorded
    # before that field existed readable.
    watertight = report.get("welded_watertight", report.get("watertight"))
    if isinstance(watertight, bool):
        rows.append(
            Row(
                "Watertight",
                "ok" if watertight else "attention",
                "sealed, no boundary edges" if watertight else _watertight_detail(report),
            )
        )

    if "has_uvs" in report:
        has_uvs = bool(report.get("has_uvs"))
        rows.append(
            Row(
                "UV coordinates",
                "ok" if has_uvs else "attention",
                "present" if has_uvs else "no UV coordinates -- an importer sees a blank skin",
            )
        )

    textures = report.get("textures")
    if isinstance(textures, dict):
        missing = [
            label
            for key, label in (
                ("base_color", "base colour"),
                ("metallic_roughness", "metallic/roughness"),
            )
            if not textures.get(key)
        ]
        rows.append(
            Row(
                "Textures",
                "ok" if not missing else "attention",
                "base colour and metallic/roughness maps both present"
                if not missing
                else f"no {' or '.join(missing)} texture",
            )
        )

    if "grounded" in report:
        grounded = bool(report.get("grounded"))
        rows.append(
            Row(
                "Pivot on floor",
                "ok" if grounded else "attention",
                "the pivot sits on the floor" if grounded else "the pivot is off the floor",
                repair=None if grounded else REPAIR_FIX_ASSET,
            )
        )

    rows.append(_size_row(report, params, prompt))
    return rows


def _watertight_detail(report: dict[str, Any]) -> str:
    edges = report.get("welded_boundary_edges", report.get("boundary_edges"))
    components = report.get("welded_components", report.get("components"))
    if isinstance(edges, int) and isinstance(components, int):
        return f"{edges} boundary edge(s) in {components} component(s), after welding"
    return "not watertight"


def _size_row(report: dict[str, Any], params: dict[str, Any], prompt: Any) -> Row:
    """The size row: whether one was ever asked for, and if so whether it landed.

    **Stated here, repaired on the Mesh stage**, and the first draft of this row
    got that wrong in both of the ways it could (the 2026-09-07 Create review,
    item 3.3). ``guidance.SIZE_HINTS_M`` is a suggestion table nothing applies
    on its own (``guidance.size_hint``'s own docstring), and writing ``size_m``
    onto a done job without touching the geometry would re-key that job's
    stored evidence (it is in ``vectors.VECTOR_PARAMS``) and claim a size the
    mesh does not have. So this row never applies anything itself. What changed
    afterwards is that a press which *is* honest now exists:
    ``service.jobs.reorient_model`` rescales ``model.glb`` and records the new
    ``size_m`` in the same step, under Earlier meshes, so the row's button
    points at the Finished-asset panel that calls it.
    """
    target = params.get("size_m")
    if not target:
        hint = guidance.size_hint(prompt if isinstance(prompt, str) else None)
        if hint is None:
            return Row(
                "Size", "attention", "no target size was requested", repair=REPAIR_FIX_ASSET
            )
        noun, metres = hint
        return Row(
            "Size",
            "attention",
            f"no target size was requested; a {noun} is usually {metres:g} m",
            repair=REPAIR_FIX_ASSET,
        )

    try:
        target = float(target)
    except (TypeError, ValueError):
        # The 2026-09-18 audit, finding create-02: a non-numeric ``size_m``
        # in a job row (a hand-edited job, or a future writer's bug) raised
        # straight out of this draw, which ``panes/inspector.py`` calls on
        # the frame thread on every Export-stage frame -- a session-ending
        # ``ValueError`` in a loop that runs sixty times a second.
        return Row("Size", "attention", "size target could not be read")
    if not math.isfinite(target) or target <= 0:
        # The 2026-10-04 audit, finding create-08: the same unvalidated input
        # as above, one step later. ``"0"`` is truthy, so it passed the
        # ``not target`` gate and then divided by zero on the frame thread; a
        # negative target reported "matching the -1.000 m requested". A size
        # that cannot be a size is unreadable, not a measurement to compare.
        return Row("Size", "attention", "size target could not be read")
    achieved = report.get("achieved_size_m")
    if not isinstance(achieved, (int, float)) or achieved <= 0:
        return Row("Size", "ok", f"asked for {target:.3f} m; achieved size not measured")
    mismatch = abs(float(achieved) - target) / target > meshreport.SIZE_TOLERANCE
    return Row(
        "Size",
        "attention" if mismatch else "ok",
        f"{float(achieved):.3f} m, asked for {target:.3f} m"
        if mismatch
        else f"{float(achieved):.3f} m, matching the {target:.3f} m requested",
        repair=REPAIR_FIX_ASSET if mismatch else None,
    )


def _degraded_row(degraded: dict[str, Any]) -> Row:
    """The one row for every step ``validation.note_degraded`` ever recorded.

    A normalize failure is the one this review names a repair for: the
    rebuild door is exactly ``panes.retarget_panel``'s "Rebuild mesh" (on the
    Mesh stage), which
    re-runs ``postprocess.normalize_glb`` and clears the note on success (see
    ``_jobs_rework.optimize_job``). A degraded ``report`` step has no such
    door -- nothing re-measures a mesh on its own -- so it is stated with no
    button rather than one that would not do anything.
    """
    steps = sorted(degraded)
    detail = "; ".join(f"{step}: {degraded[step]}" for step in steps)
    return Row(
        "Processing",
        "attention",
        detail or "a post-processing step did not run",
        repair=REPAIR_GO_TO_MESH if "normalize" in degraded else None,
    )


def _stale_rig_row(
    params: dict[str, Any], files: set[str], rig_mtime: float | None = None
) -> Row | None:
    """Whether the rig on this asset predates its last known geometry change.

    The evidence, strongest first. **A geometry step newer than the rig:**
    every rework that changes geometry (retarget, remesh, a resize or re-ground
    through ``reorient_model``, a restore across one) keeps the mesh it replaces
    under ``model_history`` with the time it ran and a ``geometry`` flag, and
    ``rig.json`` is written last when a rig lands -- so a geometry entry newer
    than ``rig.json``'s mtime is a mesh changed since it was rigged, whichever
    rework did it. Without a ``rig_mtime`` (the caller could not stat the file)
    this falls back to the older signal, which only a retarget leaves:
    ``optimize_job`` drops ``mesh_report`` when it rebuilds ``model.glb`` and
    nothing recomputes it afterwards, so a rig beside a mesh with no report and
    a recorded ``optimize`` run has been rebuilt since anything last measured
    it.
    """
    if "rig.glb" not in files:
        return None
    changed = rig_mtime is not None and any(
        e.get("geometry", True)
        and isinstance(e.get("at"), (int, float))
        and e["at"] > rig_mtime
        for e in modelhistory.entries_of(params)
    )
    if not changed and (params.get("mesh_report") or not params.get("optimize")):
        return None
    return Row(
        "Rig",
        "attention",
        "the mesh was changed after it was rigged; the rig, its "
        "poses and its sheets may still describe the old geometry",
        repair=REPAIR_GO_TO_RIG,
    )


__all__ = [
    "REPAIR_FIX_ASSET",
    "REPAIR_GO_TO_MESH",
    "REPAIR_GO_TO_RIG",
    "Row",
    "rows_for",
]
