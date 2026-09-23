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

from dataclasses import dataclass
from typing import Any

from .. import guidance, meshreport

# What a row's repair button does, named rather than a callable: this module
# draws nothing and calls no service, so a row can only *point at* an action
# the caller already owns. Wiring only what already exists is the review's own
# rule for this item -- a row with neither identifier below carries no button.
REPAIR_GO_TO_RIG = "go_to_rig"


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


def rows_for(job: Any) -> list[Row]:
    """The checklist for ``job``, or ``[]`` when there is nothing to check.

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

    stale = _stale_rig_row(params, files)
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
    ``optimize.run``'s ``"requested"`` or ``remesh``'s ``"target_faces"`` --
    so that is read first; ``report["triangle_budget"]`` (set by a caller of
    ``meshreport.build`` that already knew the budget) is the next fallback,
    and the module default is last, for a raw reconstruction that was never
    retargeted or a report recorded before either field existed.
    """
    for record, key in (
        (params.get("optimize"), "requested"),
        (params.get("remesh"), "target_faces"),
    ):
        if isinstance(record, dict) and isinstance(record.get(key), int) and record[key] > 0:
            return record[key]
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
                # The over-budget mesh is exactly what a Rig-stage remesh (or a
                # cheaper retarget) exists to fix -- ``panes.remesh_panel`` and
                # ``panes.retarget_panel`` both live there.
                repair=REPAIR_GO_TO_RIG if over else None,
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

    **Stated, never repaired here**, and the first draft of this row got that
    wrong in both of the ways it could (the 2026-09-07 Create review, item
    3.3). ``guidance.SIZE_HINTS_M`` is a suggestion table nothing applies on
    its own (``guidance.size_hint``'s own docstring), and the settings
    column's "Use N m" applies it to a *form*, before a mesh exists. There is
    no such press for a finished mesh, because there is nothing honest for it
    to do: ``size_m`` is a recorded **input** in ``vectors.VECTOR_PARAMS``, so
    writing one onto a done job would re-key that job's stored evidence and
    claim a size the geometry does not have -- and it would not resize
    anything, since the scale was composed into the mesh by
    ``postprocess.normalize_glb`` two minutes of GPU ago. The hint is worth
    saying, so it is said; the way to act on it is a rebuild, which the
    Processing and Triangles rows already point at.
    """
    target = params.get("size_m")
    if not target:
        hint = guidance.size_hint(prompt if isinstance(prompt, str) else None)
        if hint is None:
            return Row("Size", "attention", "no target size was requested")
        noun, metres = hint
        return Row(
            "Size",
            "attention",
            f"no target size was requested; a {noun} is usually {metres:g} m",
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
    )


def _degraded_row(degraded: dict[str, Any]) -> Row:
    """The one row for every step ``validation.note_degraded`` ever recorded.

    A normalize failure is the one this review names a repair for: the
    rebuild door is exactly ``panes.retarget_panel``'s "Rebuild mesh", which
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
        repair=REPAIR_GO_TO_RIG if "normalize" in degraded else None,
    )


def _stale_rig_row(params: dict[str, Any], files: set[str]) -> Row | None:
    """Whether the rig on this asset predates its last known geometry change.

    Not a fresh measurement -- there is no stored timestamp pairing a rig to
    the mesh it was fitted to, so this reads the same signal
    ``panes.retarget_panel``'s own warning is built from (a rig existing at
    all is what a retarget invalidates) plus one more fact that only a
    retarget leaves behind: ``optimize_job`` drops ``mesh_report`` when it
    rebuilds ``model.glb`` and nothing recomputes it afterwards (unlike a
    remesh, which re-measures through ``_audit_published``). A rig beside a
    mesh with no report and a recorded ``optimize`` run is a mesh that has
    been rebuilt since anything last measured it -- which is exactly the
    asset the retarget panel's warning describes, seen after the fact instead
    of before it.
    """
    if "rig.glb" not in files:
        return None
    if params.get("mesh_report"):
        return None
    if not params.get("optimize"):
        return None
    return Row(
        "Rig",
        "attention",
        "the mesh was retargeted after it was last measured; the rig, its "
        "poses and its sheets may still describe the old geometry",
        repair=REPAIR_GO_TO_RIG,
    )


__all__ = [
    "REPAIR_GO_TO_RIG",
    "Row",
    "rows_for",
]
