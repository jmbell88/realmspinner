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
#: (`dev/measurements/2026-09-23-default-mesh-budget.md`) -- the voxel pass
#: forces quadriflow to refuse ("consistent normals") on every trellis mesh,
#: so the decimate fallback is what actually runs, and decimate's own budget
#: is a triangle count. A quad-labelled ladder was therefore promising a unit
#: the worker could not deliver. ``custom`` takes a number in
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
GEOMETRY_DERIVED = ("model.stl", "model_obj.zip", "collision.glb", "textures.zip", "model.fbx")

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
