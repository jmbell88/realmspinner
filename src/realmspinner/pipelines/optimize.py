"""Retarget a reconstruction to a triangle budget with a vendored gltfpack.

The trellis response is ~290k triangles and 22 MB, which is a source mesh, not a
game asset. gltfpack simplifies it without re-running the reconstruction, which
is the whole point: a re-target is a two-second subprocess, and a trellis run is
two minutes of GPU.

The flags are not negotiable and each earns its place:

* ``-si <ratio>`` -- the simplification ratio. gltfpack takes a ratio, not a
  triangle count, so the caller's budget is divided by the source count here.
* ``-noq`` -- no quantisation. Quantised attributes need KHR_mesh_quantization,
  which some importers list as required and refuse the file over.
* ``-ke`` / ``-km`` -- keep extras and materials. Without them the material
  assignment (and therefore both PBR textures) can be dropped on merge.

Like ``trellis-server.exe`` the binary is vendored and pinned; nothing here
downloads anything. Missing it is not fatal -- the ``raw`` profile is always
available and is what every job did before this existed.

**Every gltfpack pass is checked before it is published.** Until
2026-09-23 a named tier stayed out of the generate forms until a one-off
corpus run (``dev/measurements/2026-08-13-tier-qualification.md``) had shown
it kept UVs, both PBR maps and material assignment -- and that run never
finished. ``dev/measurements/2026-09-23-default-mesh-budget.md`` replaces the
sample with a check on every job: :func:`run` compares the tier's own output to
its own source with ``tiercheck.compare`` before ``tmp`` ever replaces
``dest``, and a verdict that lost UVs, a material assignment or a PBR texture
is refused rather than published -- ``dest`` is left exactly as it was. That is
strictly stronger than the old sample, since it cannot pass on a sword and then
fail unseen on a chest. What it still cannot see is a mangled silhouette; the
eye catches that, and Retarget -> Raw rebuilds from ``source.glb``, which this
module never touches.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .. import tiercheck, winjob

log = logging.getLogger(__name__)

# Named budgets. None means "ship what the engine returned" -- which is not
# the untouched reconstruction: trellis-server quadric-simplifies to ~300K
# faces at res 1024 before writing source.glb unless it is launched with
# --decim 0 (Config.trellis_decim). "raw" here means no *second* pass.
PROFILES: dict[str, int | None] = {
    "draft": 20_000,
    "standard": 50_000,
    "detailed": 100_000,
    "raw": None,
}

CUSTOM_MIN = 5_000
# Lowered from 2M to 250k on 2026-09-06. The 2M ceiling was raised on
# 2026-09-03 so a gltfpack budget could ask for a million faces once
# ``--decim 0`` stopped the exe throwing them away. The detail-060 sweep then
# retired that axis: every ``decim0`` rung failed on the first subject at ~29
# minutes each, so ``trellis_decim`` stays ``None`` and the exe's own quadric
# simplify always runs. That is the condition this ceiling depends on --
# *with decimation on*, source.glb lands at ~300k faces at res 1024, so a
# budget above that asks gltfpack to remove nothing. (Undecimated the mesh is
# ~35M faces and a 300k budget would be a real reduction; it is unreachable,
# because the same run showed the pipeline cannot carry an undecimated mesh
# at all.) 250k therefore sits just under the landing point, where a custom
# budget is always a genuine reduction of something that exists.
# dev/measurements/2026-09-03-trellis-detail-sweep.md has the evidence.
CUSTOM_MAX = 250_000

DEFAULT_TIMEOUT = 300.0


class OptimizeError(RuntimeError):
    """gltfpack was missing, failed, timed out, or produced an unusable file."""


def staged_copy(source: Path, dest: Path) -> None:
    """Copy ``source`` onto ``dest`` via a temp file and an atomic rename.

    ``dest`` here is model.glb, which the file route serves on mere existence
    once a job is done -- and POST /optimize runs on done jobs. A plain
    copyfile truncates ``dest`` before writing, so a concurrent reader could
    observe a half-written file; this way it sees the old file or the new
    one, never a mixture. Same idiom as postprocess._staged, kept local so
    this module never has to import trimesh-heavy postprocess.
    """
    fd, raw = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(raw)
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, dest)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def run(
    source: Path,
    dest: Path,
    *,
    target_triangles: int | None,
    exe: Path,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Write an optimized copy of ``source`` to ``dest``.

    ``dest`` is only created on success -- a half-written or rejected output must
    never end up as the model the user downloads. Success now also means
    ``tiercheck.compare`` accepted the pass: a gltfpack run that dropped UVs, a
    material assignment or a PBR texture raises :class:`OptimizeError` just
    like a non-zero exit does, and ``dest`` is left untouched either way (see
    the module docstring and dev/measurements/2026-09-23-default-mesh-budget.md).
    """
    if target_triangles is None:
        # No budget asked for, so nothing here needs to know the count: this is
        # a copy either way. Counting meant a full trimesh load of a ~22 MB,
        # ~290k-face GLB, serially on the queue, purely to record a number the
        # mesh report measures again a step later -- and `raw` is the profile
        # every job runs today. None, not a guess: the field says "not
        # measured" rather than claiming a figure nothing produced.
        staged_copy(source, dest)
        return {
            "requested": None,
            "achieved": None,
            "source_triangles": None,
            "bytes": dest.stat().st_size,
        }
    source_triangles = _triangles(source)
    if source_triangles <= target_triangles:
        # Already inside the budget. Copying is honest: running the simplifier
        # to a ratio above 1.0 is a no-op that still re-encodes the file.
        staged_copy(source, dest)
        return {
            "requested": target_triangles,
            # A byte-for-byte copy has exactly the source's count; loading the
            # copy through trimesh just to re-measure it is wasted work.
            "achieved": source_triangles,
            "source_triangles": source_triangles,
            "bytes": dest.stat().st_size,
        }
    # ``.is_file()``, not ``.exists()`` (the 2026-09-08 audit, pipelines-04):
    # the same defect already fixed at ``doctor._gltfpack_check`` and
    # ``retarget_panel._gltfpack_available``, and this is the site that
    # actually decides whether a retarget job runs. A directory left where
    # ``gltfpack.exe`` should be used to read as "present" and reach
    # ``winjob.run([str(exe), ...])`` below, where ``subprocess.Popen`` raises
    # an uncaught OSError instead of this module's own ``OptimizeError``.
    if not exe.is_file():
        raise OptimizeError(
            f"gltfpack not found at {exe}; use the 'raw' profile or set REALMSPINNER_GLTFPACK"
        )

    ratio = max(min(target_triangles / max(source_triangles, 1), 1.0), 0.0)
    # staged_copy's dotfile spelling, not a visible sibling: a staging file is
    # a dotfile (the staged-writes rule), and this one spent a while as a bare
    # model.glb.opt.tmp beside the served model. A dotfile also can never
    # collide with anything in files.LISTED, which is a list of plain names.
    # It must still *end* in ``.glb``: gltfpack picks its writer from the
    # output extension and refuses anything else ("unsupported output
    # extension '.tmp'", exit 4). The old ``.model.glb.opt.tmp`` spelling made
    # every real retarget fail that way -- unseen, because the only default was
    # ``raw`` and the tests stub gltfpack -- until the 2026-09-23 switch to a
    # ``standard`` default ran one against a real reconstruction.
    tmp = dest.with_name(f".{dest.stem}.opt{dest.suffix}")
    argv = _argv(exe, source, tmp, ratio)
    try:
        # winjob.run (via _invoke) rather than subprocess.run, for the same
        # reason every other child is in the job object: a hard kill of the
        # app must not leave a gltfpack behind holding a half-written
        # .opt.glb staging file.
        proc = _invoke(argv, timeout=timeout)
    except OptimizeError:
        tmp.unlink(missing_ok=True)
        raise
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise OptimizeError(
            f"gltfpack exited {proc.returncode}: {(proc.stderr or proc.stdout)[:500]}"
        )

    # Every other exit from this function unlinks the staging file; the tail
    # did not, so a _triangles() that raised (a mesh trimesh cannot parse) left
    # an .opt.glb staging file beside the served model for the next reader to find.
    try:
        achieved = _triangles(tmp)
        if achieved <= 0:
            raise OptimizeError("gltfpack produced a mesh with no triangles")
        # dev/measurements/2026-09-23-default-mesh-budget.md: the per-tier
        # corpus qualification this used to wait on never finished, so every
        # pass now checks itself instead of trusting a sample run on other
        # meshes. Before the replace, not after -- a losing verdict must leave
        # `dest` exactly as it was; the `finally` below still unlinks `tmp`
        # either way, which is what makes "unlink and raise" free here.
        verdict = tiercheck.compare(tiercheck.survey(source), tiercheck.survey(tmp))
        if not verdict.ok:
            raise OptimizeError(
                f"the {target_triangles:,}-triangle budget lost what it must "
                "keep: " + "; ".join(verdict.failures)
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    log.info(
        "optimized %s: %d -> %d triangles (asked %d)",
        source.name, source_triangles, achieved, target_triangles,
    )
    return {
        "requested": target_triangles,
        "achieved": achieved,
        "source_triangles": source_triangles,
        "bytes": dest.stat().st_size,
        "tiercheck": {"ok": True, "notes": list(verdict.notes)},
    }


def simplify_bytes(
    data: bytes,
    *,
    ratio: float,
    exe: Path,
    timeout: float = DEFAULT_TIMEOUT,
    lock_border: bool = False,
    aggressive: bool = False,
) -> bytes:
    """Simplify one in-memory GLB and return the simplified bytes.

    For Clay's decimate operator, which has no ``source.glb`` on disk to point
    ``run`` at -- it serialises the object being edited and wants the result
    back the same way. Same flags and the same ``OptimizeError`` discipline as
    ``run``, minus the triangle-budget/staged-copy bookkeeping that only makes
    sense for a Library job with a ``dest`` other readers may be watching.

    ``-slb`` locks vertices on an open boundary (a UV seam, a mesh border) so
    they do not wander during simplification; ``-sa`` lets gltfpack change
    topology when the plain simplifier cannot reach ``ratio`` otherwise. Both
    flags are in the vendored gltfpack 1.2's own ``-h`` output (checked
    2026-09-19), under "Simplification" alongside ``-si``.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio!r}")
    if ratio == 1.0:
        # No reduction asked for. Spawning gltfpack for a ratio of 1.0 would
        # still re-encode the file for no gain -- the same reasoning `run`
        # uses when the source is already inside budget.
        return data
    if not exe.is_file():
        raise OptimizeError(
            f"gltfpack not found at {exe}; use the 'raw' profile or set REALMSPINNER_GLTFPACK"
        )
    # TemporaryDirectory's own __exit__ removes tmpdir on every way out of this
    # block -- the return below, an OptimizeError raised inside it, or
    # anything else -- so there is nothing left for this function to clean up.
    with tempfile.TemporaryDirectory(prefix="realmspinner-simplify-") as tmpdir:
        src = Path(tmpdir) / "in.glb"
        dst = Path(tmpdir) / "out.glb"
        src.write_bytes(data)
        argv = _argv(exe, src, dst, ratio, lock_border=lock_border, aggressive=aggressive)
        proc = _invoke(argv, timeout=timeout)
        if proc.returncode != 0:
            raise OptimizeError(
                f"gltfpack exited {proc.returncode}: {(proc.stderr or proc.stdout)[:500]}"
            )
        if not dst.is_file():
            raise OptimizeError("gltfpack produced no output")
        out = dst.read_bytes()
        if not out:
            raise OptimizeError("gltfpack produced an empty output")
        return out


def _argv(
    exe: Path,
    src: Path,
    dst: Path,
    ratio: float,
    *,
    lock_border: bool = False,
    aggressive: bool = False,
) -> list[str]:
    """The gltfpack invocation shared by ``run`` and ``simplify_bytes``.

    ``-noq``/``-ke``/``-km`` are not negotiable (see the module docstring);
    ``-slb``/``-sa`` are opt-in because they change *how* geometry is allowed
    to move, which only ``simplify_bytes``'s caller (Clay, editing one object
    interactively) has an opinion about -- a Library retarget always wants the
    plain simplifier.
    """
    argv = [
        str(exe),
        "-i", str(src),
        "-o", str(dst),
        "-si", f"{ratio:g}",
        "-noq",
        "-ke",
        "-km",
    ]
    if lock_border:
        argv.append("-slb")
    if aggressive:
        argv.append("-sa")
    return argv


def _invoke(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    """Spawn gltfpack inside the kill-on-close job; map a timeout to ``OptimizeError``.

    Exit-code and output-file checking stay with the caller: ``run`` and
    ``simplify_bytes`` stage their output differently (a named ``.opt.glb``
    beside ``dest`` vs. a throwaway tempdir), so each decides for itself what
    "produced no usable output" means and what, if anything, it must clean up
    before raising.
    """
    try:
        return winjob.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise OptimizeError(f"gltfpack timed out after {timeout:.0f}s") from exc


def resolve(profile: str, custom: int | None = None) -> int | None:
    """A profile name (or 'custom' plus a count) -> a triangle budget."""
    if profile == "custom":
        if custom is None or not CUSTOM_MIN <= custom <= CUSTOM_MAX:
            raise ValueError(f"custom triangles must be {CUSTOM_MIN}-{CUSTOM_MAX}")
        return custom
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}")
    return PROFILES[profile]


def _triangles(path: Path) -> int:
    import trimesh

    loaded = trimesh.load(path, process=False)
    mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
    return int(len(mesh.faces))
