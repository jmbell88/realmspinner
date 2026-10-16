"""Game-ready remesh: the host-side half, and every number the panel and the
door agree on.

A TRELLIS reconstruction is a triangle soup at ~300k faces with xatlas islands
-- fine to look at, wrong to ship: no edge loops, a UV layout no artist can
paint, and holes wherever the narrow band gave up. Every commercial pipeline
(Meshy's Remesh, Tripo's quad remesher, Rodin's quad tiers) closes exactly that
gap with one step: **remesh to a face budget, unwrap the new surface, bake the
old surface's colour, roughness and normals onto it.** This module is the
arithmetic and the vocabulary of that step; the Blender half is
``blender_worker.op_remesh``, and the two never import each other.

Unlike a gltfpack retarget (``pipelines.optimize``), which *simplifies* the
reconstruction and keeps its UVs, a remesh produces a **new** surface -- so it
bakes rather than preserves, and ``tiercheck.compare`` against the mesh it
replaced is what says whether anything a tier must keep was lost.

Pure: stdlib only, no bpy, no service, no queue.
"""

from __future__ import annotations

from typing import Any

#: Triangle budgets, keyed by the word the panel and Create's Budget combo
#: show. Triangles, not quads: measured 2026-09-23 on the raccoon
#: (`dev/measurements/2026-09-23-default-mesh-budget.md` [lost; re-measure to change]) -- the voxel
#: pass forces quadriflow to refuse ("consistent normals") on every trellis mesh, so the decimate
#: fallback is what actually runs, and decimate's own budget is a triangle count. A quad-labelled
#: ladder was therefore promising a unit the worker could not deliver. ``custom`` takes a number in
#: ``TRIANGLES_MIN..TRIANGLES_MAX``.
#:
#: 5k is the default: Crash-Bandicoot-range fidelity on a whole-character
#: reconstruction, per the same measurement (4,996 triangles in ~6 s with good
#: fidelity on the raccoon). The other three rungs bracket it rather than
#: being independently qualified -- the same weaker footing the old quad
#: ladder's own comment named (2026-09-11 audit, finding pipelines-08), carried
#: forward rather than resolved: re-deriving the ladder is still a sitting
#: with a card and eyes, and it is still owed rather than done.
TRIANGLE_PROFILES: dict[str, int] = {
    "2k": 2_000,
    "5k": 5_000,
    "10k": 10_000,
    "20k": 20_000,
}
DEFAULT_TRIANGLE_PROFILE = "5k"
TRIANGLES_MIN = 1_000
TRIANGLES_MAX = 200_000

#: How far over its budget a finished mesh may land and still count as meeting it.
#: Blender's repair path lands within 0.5% of any budget and gltfpack and the
#: preserve path land far above one (``dev/measurements/2026-10-15-lowpoly-silhouette-gate.md``);
#: the engine's own QEM target is not measured yet (an open sitting). Also the
#: line ``_q_mesh._lowpoly`` uses to decide an engine that was asked for ``faces``
#: ignored the field (an old exe returns its built-in ~300k) and Blender must finish.
BUDGET_SLACK = 1.25
#: Past this many times the request a mesh is ``over_budget``, never ``complete``: the
#: user asked for 5,000 and must not be told a 13,813 mesh is done.
OVER_BUDGET_RATIO = 1.5
#: The shape gate. A finished mesh may lose at most this share of its source's
#: silhouette in its worst audit view, and no view's IoU may fall below the floor.
#: **Provisional**: M0 measured two reconstructions (30 finisher runs): every mesh
#: that met its budget lost 0.8-4.4% except one at 5.2%, and every deformed one lost
#: 7.3-26.8%, so 0.05 sits in the gap between 0.052 and 0.073 with one row on the
#: wrong side of it. A human decides 0.05 vs 0.06 after looking at those meshes.
LOWPOLY_LOST_MAX = 0.05
LOWPOLY_IOU_MIN = 0.90
#: A rework budget is refused when it asks for more than this many times the
#: triangles ``source.glb`` holds (``params["source_faces"]``, recorded when the
#: engine simplified before baking). The margin is only so asking for the very
#: number a job was made at -- which the engine lands a few triangles under --
#: is not refused as "above" itself.
SOURCE_FACES_TOLERANCE = 1.05

#: Bake resolutions. ``None`` at the door means "match the mesh's own atlas",
#: resolved by the worker against the file for ``retexture``'s reason.
TEXTURE_SIZES = (512, 1024, 2048)
DEFAULT_TEXTURE_PX = 1024

#: Voxel size, as a fraction of the mesh's bounding diagonal, for the
#: hole-closing pre-pass. One number rather than a slider: at 1/200 of the
#: diagonal a 1 m prop remeshes at 5 mm voxels, which closes the plate-crust
#: gaps ``meshaudit`` flags without rounding off a sword's edge; finer than
#: that is minutes in Blender for no visible change.
#:
#: That sentence reads as measured and is not: no `dev/measurements/` document
#: backs it, and the 2026-09-11 audit (finding pipelines-09) flagged exactly
#: that ambiguity -- a reader could not tell a measured number from one chosen
#: by feel and written up confidently. It is chosen by feel. Nothing in the
#: corpus is keyed on it; a re-derivation is a sitting with a card and eyes.
VOXEL_FRACTION = 0.005

#: The seed quadriflow takes. Fixed so two remeshes of one mesh at one budget
#: are the same mesh -- a reroll is a different job, not a different seed.
QUADRIFLOW_SEED = 0

#: Every export a remesh invalidates: the whole of ``service.files.DERIVED``,
#: restated here because the queue may not import ``service``. Geometry *and*
#: skin change, so this is the superset of ``retexture.SURFACE_DERIVED``;
#: ``tests/test_remesh.py`` asserts the two spellings agree.
GEOMETRY_DERIVED = (
    "model.stl",
    "model_obj.zip",
    "collision.glb",
    "textures.zip",
    "model.fbx",
    "model_lods.zip",
)

#: The margin (texels) the bake grows past every island edge, so bilinear
#: filtering and the first two mips never read the background. Eight texels is
#: the conventional figure for exactly that pair of readers and is not measured
#: here; no `dev/measurements/` document backs it (the 2026-09-11 audit,
#: finding pipelines-09), and nothing in the corpus is keyed on it.
BAKE_MARGIN_PX = 8


def resolve(profile: str, custom: int | None = None) -> int:
    """The triangle budget a profile names, or a refusal that names the range.

    ``ValueError`` rather than a service error: this is the one implementation
    the door *and* the panel's pre-flight call, exactly as ``optimize.resolve``
    is for the gltfpack tiers, so both refuse in the same words.
    """
    if profile == "custom":
        if custom is None:
            raise ValueError("a custom remesh needs a triangle count")
        try:
            value = int(custom)
        except (TypeError, ValueError) as exc:
            raise ValueError("triangle count must be a whole number") from exc
        if not TRIANGLES_MIN <= value <= TRIANGLES_MAX:
            raise ValueError(
                f"triangle count must be between {TRIANGLES_MIN:,} and {TRIANGLES_MAX:,}"
            )
        return value
    if profile not in TRIANGLE_PROFILES:
        raise ValueError(
            f"unknown remesh profile {profile!r}; one of "
            f"{sorted(TRIANGLE_PROFILES)} or 'custom'"
        )
    return TRIANGLE_PROFILES[profile]


def target_faces(triangles: int) -> int:
    """A triangle budget -> Blender's own ``target_faces``.

    ``quadriflow_remesh``'s ``target_faces`` counts quads, and after the
    always-on voxel pre-pass every polygon in the decimate fallback's input is
    one -- so half the triangle budget is what the decimate ratio in
    ``blender_worker._remesh_object`` actually converges on (its own comment
    states the arithmetic: ``(target_faces * 2) / tris``). At least 1, so a
    degenerate custom budget never asks Blender for zero faces.
    """
    return max(int(triangles) // 2, 1)



def gate(requested: int, achieved: int | None, geometry: Any) -> tuple[float, list[str]]:
    """How far a finished mesh is from passing the Game-ready gate.

    -> ``(miss, reasons)``. ``miss`` is the largest of the mesh's excesses, each as
    a multiple of its own limit -- triangles over ``requested * BUDGET_SLACK``,
    worst lost coverage over ``LOWPOLY_LOST_MAX``, worst IoU's shortfall over
    ``1 - LOWPOLY_IOU_MIN`` -- so ``miss <= 1.0`` passes and a larger number is a
    worse mesh in whichever way it is worst. That is what "the closest result"
    means when no finisher passes: not the least silhouette loss (a 54,000-triangle
    gltfpack mesh loses 2.5% and is ten times the budget), not the fewest
    triangles (the preserve floor is 13,800 and loses 20%), but the smallest worst
    miss. ``reasons`` names each failed limit in words.

    Unmeasured geometry (``compare_geometry`` could not load a mesh) contributes
    nothing: a silhouette that was not measured is not a silhouette that failed,
    and the budget half of the gate still applies.
    """
    miss = 0.0
    reasons: list[str] = []
    if achieved is not None and requested > 0:
        over = achieved / (requested * BUDGET_SLACK)
        miss = max(miss, over)
        if over > 1.0:
            reasons.append(
                f"{achieved:,} triangles is {achieved / requested:.1f}x the {requested:,} requested"
            )
    if isinstance(geometry, dict) and geometry.get("measured"):
        lost = float(geometry.get("worst_lost_coverage", 0.0))
        miss = max(miss, lost / LOWPOLY_LOST_MAX)
        if lost > LOWPOLY_LOST_MAX:
            reasons.append(
                f"lost {lost:.1%} of the silhouette (limit {LOWPOLY_LOST_MAX:.0%})"
            )
        ious = [float(v["iou"]) for v in geometry.get("views") or () if "iou" in v]
        if ious:
            worst = min(ious)
            miss = max(miss, (1.0 - worst) / (1.0 - LOWPOLY_IOU_MIN))
            if worst < LOWPOLY_IOU_MIN:
                reasons.append(
                    f"a view's silhouette IoU fell to {worst:.2f} (floor {LOWPOLY_IOU_MIN:.2f})"
                )
    return miss, reasons


#: The Blender rungs in the order they are tried. ``engine`` is not here: it is
#: the step *before* these, and falls into the first one.
BLENDER_LADDER = ("preserve_shape", "repair")


def finishing_ladder(finishing: str) -> tuple[str, ...]:
    """The finishers to try, in order, starting at ``finishing``.

    ``engine`` (which reaches this only because the engine could not or did not
    finish the budget) starts at the first Blender rung; ``repair`` starts at
    itself and has only gltfpack after it. ``gltfpack`` is always last: it is the
    one finisher that cannot hold a small budget (it floors at ~46-54k triangles,
    ``dev/measurements/2026-10-15-lowpoly-silhouette-gate.md``), so it is a
    fallback for budgets that large and otherwise only ever the "closest" of
    meshes that all missed.
    """
    start = BLENDER_LADDER.index(finishing) if finishing in BLENDER_LADDER else 0
    return (*BLENDER_LADDER[start:], "gltfpack")


def compare_geometry(source: Any, finished: Any, resolution: int = 256) -> dict[str, Any]:
    """Shared-frame silhouettes, including openings, from the mesh audit views.

    The silhouette numbers feed ``gate`` (the Game-ready shape gate); they are not
    a technical-readiness grade, and identical bounds alone cannot detect a lost
    handle or a closed opening -- mask differences can. Both meshes must be in the
    same frame: ``model.glb`` is grounded and scaled after the gate runs, so this
    is only meaningful on the finisher's own output.
    """
    from .finishing import compare_geometry as measure

    return measure(source, finished, resolution)


def finishing_lines(report: Any) -> list[str]:
    """Describe the result without treating absent measurements as success."""
    if not isinstance(report, dict):
        return []
    if report.get("status") in ("failed", "cancelled"):
        return [f"Finishing {report['status']}: {report.get('error', 'reconstruction retained')}",
                "The reconstruction is available as source.glb."]
    triangles = report.get("triangles", report.get("achieved"))
    count = f"{triangles:,}" if isinstance(triangles, int) else "unknown"
    lines = [f"Finished mesh: {count} triangles; requested {report.get('requested', 'unknown')}."]
    if report.get("method") == "engine":
        # The engine simplified before it baked, so there is no higher-resolution
        # mesh to measure a silhouette against and nothing was rebaked from one.
        lines.append(
            "Simplified by the engine before it textured the mesh; no silhouette "
            "comparison was made (there is no higher-resolution mesh to compare to)."
        )
        return lines
    geometry = report.get("geometry") or {}
    if geometry.get("measured"):
        lines.append(
            f"Silhouette coverage lost: {geometry['worst_lost_coverage']:.1%}; "
            f"added: {geometry['worst_added_coverage']:.1%}."
        )
        closures = [v["closed_openings"] for v in geometry["views"]
                    if v.get("closed_openings") is not None]
        lines.append(f"Source opening pixels filled: {max(closures):.1%}." if closures
                     else "Opening preservation: unknown (no source opening pixels measured).")
    else:
        lines.append("Shape preservation: unknown (measurement unavailable).")
    verdict = report.get("tiercheck") or {}
    lines.extend(f"Material/UV change: {failure}" for failure in verdict.get("failures", ()))
    if not verdict:
        lines.append("Material preservation: unknown.")
    lines.append(
        "Silhouettes do not establish prompt fidelity or asset readiness. source.glb is retained."
    )
    return lines


def profile_label(key: str, triangles: int | None = None) -> str:
    """"5k (5,000 triangles)" -- derived from the table so a label cannot
    disagree with the number the worker is asked for."""
    if key == "custom":
        return "Custom..."
    triangles = TRIANGLE_PROFILES[key] if triangles is None else triangles
    return f"{key} ({triangles:,} triangles)"


def report_line(report: Any) -> str | None:
    """One sentence about what the last remesh produced, or None.

    A decimated mesh is the *expected* path on a trellis reconstruction now
    (2026-09-23: the voxel pre-pass that closes plate-crust holes leaves every
    surface with "inconsistent normals" as far as quadriflow is concerned, so
    it refuses and the decimate fallback is what actually runs) -- so this no
    longer reads a decimated result as a failure of the quad path. It still
    says which path ran, because a quadriflow surface and a decimated one are
    different UV layouts and a reader comparing two remeshes should be able to
    tell them apart.
    """
    if not isinstance(report, dict):
        return None
    faces = report.get("faces")
    if faces is None:
        return None
    quads = report.get("quads")
    method = report.get("method")
    size = report.get("texture_size")
    if method == "decimate":
        head = f"{faces:,} triangles (remeshed and re-baked)"
    elif quads is not None:
        head = f"{faces:,} faces, {quads * 100:.0f}% quads"
    else:
        head = f"{faces:,} faces"
    tail = f", baked at {size} px" if size else ""
    verdict = report.get("tiercheck") or {}
    if verdict.get("ok") is False:
        # The 2026-09-07 audit found this unparenthesised: `+` binds tighter
        # than `or`, so "; lost: " + "".join(()) was the non-empty string
        # "; lost: " and the "or" fallback could never fire, even with an
        # empty or missing failures list. *Reproduced* (test_remesh.py).
        failures = verdict.get("failures") or ()
        tail += ("; lost: " + ", ".join(failures)) if failures else "; lost something"
    return head + tail
