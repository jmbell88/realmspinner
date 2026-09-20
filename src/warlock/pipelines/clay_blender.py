"""The host side of Clay's three mesh-cleanup Blender ops (retopologise,
unwrap, bake high to low) -- ``dev/CLAY-PLAN.md`` tranche 4.

Every function here takes and returns raw GLB bytes: Clay holds its document
in memory, not on disk, so the door between "a Clay object" and "a Blender
worker spec" is a temp file this module owns the whole lifetime of. The
mirror of ``pipelines.remesh``'s split from ``blender_worker.op_remesh`` --
this is the *host* half, pure orchestration around ``blender_run.run_worker``;
the ``bpy`` half is ``blender_worker.op_clay_retopo``/``op_clay_unwrap``/
``op_clay_bake``, and the two never import each other.

Synchronous and blocking, like every ``blender_run.run_worker`` caller in
this codebase -- the Clay background op that calls one of these three
dispatches it through ``asyncio.to_thread``, never the frame thread.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..kernels.rig import blender_spec
from . import blender_run


class ClayBlenderError(RuntimeError):
    """A Clay background Blender op failed, timed out, or bpy is missing.

    One name for all three failure modes ``blender_run.BlenderError`` already
    carries (non-zero exit, no result file, an unreadable one, a timeout) --
    named for the caller's world rather than the worker-process plumbing's,
    the same way ``pipelines.remesh``'s door never lets a raw
    ``BlenderError`` reach a job's ``error`` column.
    """


def available() -> tuple[bool, str]:
    """Can a Clay background op reach Blender right now? -> (yes/no, why not).

    Wraps ``doctor.blender_check(probe=False)`` -- the same cached-or-pending
    answer ``studio.panes.remesh_panel._blender_available`` reads, unprobed:
    deciding whether to offer the menu item must not block a frame on a bpy
    subprocess doctor has already run (or is already running in the
    background) to answer this exact question.
    """
    from .. import doctor

    check = doctor.blender_check(probe=False)
    if check.ok:
        return True, ""
    return False, "Needs Blender, which is not installed (the rig extra)."


def _run(spec: dict[str, Any], *, timeout: float, name: str) -> dict[str, Any]:
    try:
        return blender_run.run_worker(spec, timeout=timeout, name=name)
    except blender_run.BlenderError as exc:
        raise ClayBlenderError(str(exc)) from exc


def retopo_bytes(
    glb: bytes,
    *,
    target_faces: int,
    close_holes: bool = False,
    seed: int = 0,
    keep_uvs: bool = False,
    timeout: float = blender_run.BLENDER_TIMEOUT,
) -> tuple[bytes, dict[str, Any]]:
    """Retopologise every mesh object in ``glb``, independently.

    -> (the retopologised GLB's bytes, ``op_clay_retopo``'s per-object
    report: ``{"ok": True, "objects": [{"name", "method", "faces_before",
    "faces", "quads"}, ...]}``).
    """
    with tempfile.TemporaryDirectory(prefix="wl-clay-retopo-") as tmp:
        work = Path(tmp)
        source = work / "source.glb"
        source.write_bytes(glb)
        out = work / "out.glb"
        spec = blender_spec.clay_retopo_spec(
            source,
            out,
            work,
            target_faces=target_faces,
            close_holes=close_holes,
            seed=seed,
            keep_uvs=keep_uvs,
        )
        result = _run(spec, timeout=timeout, name="Clay retopo")
        return out.read_bytes(), result


def unwrap_bytes(
    glb: bytes,
    *,
    angle_limit: float = 66.0,
    island_margin: float = 0.003,
    timeout: float = blender_run.BLENDER_TIMEOUT,
) -> tuple[bytes, dict[str, Any]]:
    """Smart-UV-Project every mesh object in ``glb``. Geometry untouched.

    -> (the unwrapped GLB's bytes, ``op_clay_unwrap``'s per-object report:
    ``{"ok": True, "objects": [{"name", "islands"}, ...]}``).
    """
    with tempfile.TemporaryDirectory(prefix="wl-clay-unwrap-") as tmp:
        work = Path(tmp)
        source = work / "source.glb"
        source.write_bytes(glb)
        out = work / "out.glb"
        spec = blender_spec.clay_unwrap_spec(
            source, out, work, angle_limit=angle_limit, island_margin=island_margin
        )
        result = _run(spec, timeout=timeout, name="Clay unwrap")
        return out.read_bytes(), result


def bake_bytes(
    high_glb: bytes,
    low_glb: bytes,
    *,
    texture_size: int,
    cage_extrusion: float,
    maps: Sequence[str] = blender_spec.CLAY_BAKE_MAPS,
    timeout: float = blender_run.BLENDER_TIMEOUT,
) -> tuple[bytes, dict[str, Any]]:
    """Selected-to-active bake from ``high_glb`` onto ``low_glb``'s UVs.

    ``low_glb`` must already carry a UV layer (``op_clay_unwrap``, or the
    mesh's own); the worker refuses by name rather than baking a blank atlas.

    -> (the low mesh's GLB bytes with the bake packed in, ``op_clay_bake``'s
    report: ``{"ok": True, "maps", "texture_size", "metallic"}``).
    """
    with tempfile.TemporaryDirectory(prefix="wl-clay-bake-") as tmp:
        work = Path(tmp)
        high = work / "high.glb"
        low = work / "low.glb"
        high.write_bytes(high_glb)
        low.write_bytes(low_glb)
        out = work / "out.glb"
        spec = blender_spec.clay_bake_spec(
            high,
            low,
            out,
            work,
            texture_size=texture_size,
            cage_extrusion=cage_extrusion,
            maps=maps,
        )
        result = _run(spec, timeout=timeout, name="Clay bake")
        return out.read_bytes(), result
