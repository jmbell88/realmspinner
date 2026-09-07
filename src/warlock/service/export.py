"""Bulk export: many jobs' artifacts into one zip, or into a project folder."""

from __future__ import annotations

import contextlib
import os
import secrets
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import files
from .core import WarlockService
from .errors import Invalid, NotFound
from .files import MEDIA
from .validation import ARTIFACT_HEALTH, check_job_id


def export_names(names_wanted: list[str] | None) -> list[str]:
    """The requested artifact names, defaulting to the GLB.

    The allowlist *is* files.MEDIA: the point is that a caller-supplied name
    never becomes a path component without passing through it first -- which is
    also why the parameter is not called ``files``, the name of the module that
    allowlist comes out of. ``field="files"`` on the refusal is unchanged: that
    is the name of the *control*, which is what an error has to point at.
    """
    names = [f for f in (names_wanted or []) if f] or ["model.glb"]
    unknown = [n for n in names if n not in MEDIA]
    if unknown:
        raise Invalid(f"unknown file(s): {sorted(unknown)}", field="files")
    return names


def collect(svc: WarlockService, ids: list[str], names: list[str]) -> list[tuple[str, Path]]:
    """-> (arcname, path) for every requested file that is ready to serve.

    Silently skips what is missing rather than failing the batch: a selection
    of ten jobs where one never produced an OBJ should still deliver nine, and
    the zip's contents say which. Readiness, not existence -- exporting a
    running job's model.glb would zip a file the worker is still writing.
    """
    out: list[tuple[str, Path]] = []
    for job_id in ids:
        check_job_id(job_id)
        job = svc.store.get(job_id)
        if job is None:
            continue
        job_dir = svc.job_dir(job_id)
        for name in names:
            path = job_dir / name
            # fresh_2d as well as ready: this is a serving path that never
            # derives, so without it a batch would zip an icon left over from
            # a reference the user has since edited. It answers True for every
            # name that is not a 2D export, so nothing else changes. The pixel
            # palette knob is deliberately not consulted here -- a palette
            # mismatch is a preference, not staleness, so a batch ships the
            # last-derived palette.
            if path.exists() and files.ready(job, job_dir, name) and files.fresh_2d(job_dir, name):
                out.append((f"{job_id}/{name}", path))
    return out


def bulk_export(
    svc: WarlockService,
    ids: list[str],
    names_wanted: list[str] | None,
    dest_zip: Path,
) -> dict[str, Any]:
    """Zip the named artifacts of several jobs into ``dest_zip``.

    Derived artifacts are *not* generated on demand here -- a batch export
    should not be able to kick off twenty Blender subprocesses.

    The selection parameter is ``names_wanted`` and not ``files`` because this
    module imports a module called ``files`` and reads it in ``collect`` a few
    lines up: the shorter name shadowed it for the length of both functions, so
    one added line reaching for ``files.ready`` here would have got a list of
    strings and an AttributeError.
    """
    names = export_names(names_wanted)
    members = collect(svc, ids, names)
    if not members:
        raise NotFound("nothing to export")
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    # Through a temp sibling, like ``staged_copy`` below and for its reason:
    # the destination is wherever the user pointed the save dialog -- possibly
    # a watched project folder -- and writing the zip in place would leave a
    # torn archive there for the length of the build (SVC-06).
    tmp = dest_zip.with_name(f".{dest_zip.name}.{secrets.token_hex(4)}.tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for arcname, path in members:
                zf.write(path, arcname)
        os.replace(tmp, dest_zip)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
    return {"path": str(dest_zip), "files": len(members)}


def export_to_folder(
    svc: WarlockService, ids: list[str], names_wanted: list[str] | None
) -> dict[str, Any]:
    """Copy the same selection into WARLOCK_EXPORT_DIR."""
    if svc.config.export_dir is None:
        raise NotFound("no export folder configured (set WARLOCK_EXPORT_DIR)")
    names = export_names(names_wanted)
    members = collect(svc, ids, names)
    if not members:
        raise NotFound("nothing to export")
    copied = 0
    for arcname, path in members:
        dest = svc.config.export_dir / arcname
        dest.parent.mkdir(parents=True, exist_ok=True)
        staged_copy(path, dest)
        copied += 1
    return {
        "copied": copied,
        "dir": str(svc.config.export_dir),
        # Named rather than counted: "3 of 12 assets are degraded" is not
        # something a user can act on, and which ones is (ART-01).
        "degraded": degraded_ids(svc, ids),
    }


@dataclass(frozen=True)
class ExportJob:
    """What an export has been asked to write, before any of it happens.

    The same three arguments ``bulk_export``/``export_to_folder`` already take,
    bundled so :func:`plan_export` can answer "what would this write" as one
    pure call (W2.2) -- a pane shows the answer *before* either of those two
    functions is ever reached, rather than after.
    """

    svc: WarlockService
    ids: list[str]
    names_wanted: list[str] | None
    #: Whether the destination is a single zip file or a folder that receives
    #: one entry per planned file. The two write functions differ on exactly
    #: this, so the plan has to know it too.
    as_zip: bool = True


@dataclass(frozen=True)
class PlannedFile:
    """One file an export would write: its name, the path it would land at,
    and whether that path already has something at it."""

    name: str
    dest: Path
    exists: bool


@dataclass(frozen=True)
class ExportPlan:
    """The answer to "what would this export write, and what would it clobber".

    Pure data -- :func:`plan_export` is the only producer and never writes a
    byte, which is what lets a pane show this *before* committing to anything
    on disk.
    """

    files: tuple[PlannedFile, ...]

    @property
    def existing(self) -> tuple[PlannedFile, ...]:
        """The subset already on disk -- what Replace would overwrite."""
        return tuple(f for f in self.files if f.exists)


def plan_export(job: ExportJob, dest: Path) -> ExportPlan:
    """Which files ``job`` would write, and which destinations already exist.

    Pure: the only disk access is ``Path.exists()`` on each destination (and,
    through :func:`collect`, the source-readiness checks ``bulk_export``/
    ``export_to_folder`` already do before writing) -- nothing is written or
    copied. ``dest`` is the zip file's own path when ``job.as_zip``, or the
    folder the export would land its files under otherwise.
    """
    names = export_names(job.names_wanted)
    if job.as_zip:
        return ExportPlan(files=(PlannedFile(dest.name, dest, dest.exists()),))
    members = collect(job.svc, job.ids, names)
    files = tuple(
        PlannedFile(arcname, dest / arcname, (dest / arcname).exists())
        for arcname, _path in members
    )
    return ExportPlan(files=files)


def keep_both(plan: ExportPlan) -> ExportPlan:
    """"Keep both", applied to a plan: every destination suffixed by the same
    number, so the export never overwrites part of an existing set while
    renaming the rest of it.

    Picks the smallest N >= 2 for which *none* of the N-suffixed destinations
    already exist -- an ``-2`` left by an earlier "Keep both" is not
    overwritten either, the export lands on ``-3``. Pure: only stats the
    candidate destinations.
    """
    n = 2
    while True:
        candidates = [_suffixed_path(f.dest, n) for f in plan.files]
        if not any(c.exists() for c in candidates):
            break
        n += 1
    return ExportPlan(
        files=tuple(
            PlannedFile(_suffixed_name(f.name, n), c, False)
            for f, c in zip(plan.files, candidates, strict=True)
        )
    )


def _suffixed_path(path: Path, n: int) -> Path:
    return path.with_name(f"{path.stem}-{n}{path.suffix}")


def _suffixed_name(name: str, n: int) -> str:
    """:func:`keep_both`'s suffix, applied to a *name* rather than a real
    filesystem path -- a folder export's name carries a ``job_id/`` head that
    ``Path.with_name`` has no reason to know is a forward slash rather than
    whatever separator the platform prefers."""
    head, _, tail = name.rpartition("/")
    stem, dot, ext = tail.rpartition(".")
    suffixed = f"{stem}-{n}.{ext}" if dot else f"{tail}-{n}"
    return f"{head}/{suffixed}" if head else suffixed


def export_planned_to_folder(
    svc: WarlockService,
    ids: list[str],
    names_wanted: list[str] | None,
    plan: ExportPlan,
) -> dict[str, Any]:
    """``export_to_folder``'s body, generalised over *where* each file lands.

    Additive rather than a change to ``export_to_folder``: that function's
    ``export_dir / arcname`` destination is still exactly right for "Replace",
    which keeps calling it unchanged. This is the door "Keep both" goes
    through instead, writing to ``plan``'s (by then suffixed) destinations --
    ``plan`` is expected to have come from :func:`plan_export` for the same
    ``ids``/``names_wanted``, so ``collect`` returns the same members in the
    same order and the zip below lines each source up with the planned
    destination for its name.
    """
    if svc.config.export_dir is None:
        raise NotFound("no export folder configured (set WARLOCK_EXPORT_DIR)")
    names = export_names(names_wanted)
    members = collect(svc, ids, names)
    if not members:
        raise NotFound("nothing to export")
    pairs = [
        (path, target.dest) for (_arcname, path), target in zip(members, plan.files, strict=True)
    ]
    for _source, dest in pairs:
        dest.parent.mkdir(parents=True, exist_ok=True)
    staged_copy_all(pairs)
    return {
        "copied": len(pairs),
        "dir": str(svc.config.export_dir),
        "degraded": degraded_ids(svc, ids),
    }


def degraded_ids(svc: WarlockService, ids: list[str]) -> list[str]:
    """Which of these jobs had a canonical post-processing step fail.

    Normalization and the mesh report are non-fatal by design -- see
    ``validation.ARTIFACT_HEALTH`` -- so a job can be ``done`` and still carry a
    ``model.glb`` whose pivot and scale are the engine's rather than this
    project's. An export is the moment that stops being an internal detail: the
    file is on its way into a game project, where a wrong pivot is a manual
    fixup on every import and nothing anywhere would say why.
    """
    out: list[str] = []
    for job_id in ids:
        job = svc.store.get(job_id)
        if job and (job.get("params") or {}).get(ARTIFACT_HEALTH):
            out.append(job_id)
    return out


def staged_copy(source: Path, dest: Path) -> None:
    """Copy through a temp sibling and rename. Never truncates the destination.

    ``WARLOCK_EXPORT_DIR`` exists to be *watched* -- it is a game project's
    assets folder -- and ``shutil.copyfile`` truncates its target before writing
    a byte, so a hot-reloading engine could read a torn GLB for the length of
    the copy. Outside the letter of the staged-writes invariant, which is about
    files this app serves, and squarely inside its reasoning (SVC-06).

    Public since 2026-09-05: ``service.characters.export_package`` writes a
    sheet and its sidecar into the same watched folder and needs exactly this
    guarantee, and a second private copy of it in that module would be one edit
    away from a plain ``copyfile`` landing in a project's assets directory.
    """
    staged_copy_all([(source, dest)])


def staged_copy_all(pairs: list[tuple[Path, Path]]) -> None:
    """Stage *every* copy first, then rename them all. Both land or neither.

    The pair case is why this exists rather than a loop over :func:`staged_copy`
    at each call site. A character sheet is a PNG **and** its JSON sidecar, and
    a reader that finds one without the other has an asset it cannot interpret
    -- ``rigging.list_sheets`` states the same rule for the directory this app
    serves, where the sidecar is written last as the completion marker. Copying
    one at a time puts a disk-full between them; staging both and then replacing
    both narrows the window to two renames.

    Not atomic, because no filesystem offers that across two names. What it is
    is *all-or-nothing about the expensive half*: nothing appears at either
    destination until both temps are fully written.
    """
    tmps = [
        (source, dest, dest.with_name(f".{dest.name}.{secrets.token_hex(4)}.tmp"))
        for source, dest in pairs
    ]
    try:
        for source, _dest, tmp in tmps:
            shutil.copyfile(source, tmp)
        for _source, dest, tmp in tmps:
            os.replace(tmp, dest)
    finally:
        for _source, _dest, tmp in tmps:
            with contextlib.suppress(OSError):
                tmp.unlink()
