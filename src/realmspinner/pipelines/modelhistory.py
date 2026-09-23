"""Every mesh a rework replaced, kept until the cap, so a restore has
something to restore.

A retarget, a remesh and a re-texture all publish over the *same* served
name, ``model.glb``, in the *source* job's directory (``_q_mesh._remesh``,
``_q_sprite._retexture``, ``service._jobs_rework.optimize_job``) -- and until
this module existed, each one simply discarded whatever the last one built.
Remesh-then-retexture lost the remesh the moment the retexture published;
a retarget rebuilt ``model.glb`` from ``source.glb`` and silently threw away
both. The reconstruction was never at risk -- ``source.glb`` is untouched by
every one of these -- but the *work someone asked for on top of it* was, and
this is what stops that.

``<job_dir>/versions/<n>.model.glb`` plus ``<n>.json`` (a snapshot of the
params that describe that file) is the storage; ``params["model_history"]``
is the index, newest last, and it lives in a job's ordinary params blob like
any other derived record. Nothing here is in ``service/files.py``'s
``MEDIA``/``LISTED`` -- the same precedent as a Clay op's ``build.rblk``:
these are recovery data for a restore door, never a download.

**Why this module and not a queue-private or service-private one.** A
version is pushed by two different callers on two different layers: the
worker's own mesh-post mixin (``_q_mesh.MeshPostOps._publish_model_version``,
layer 3, "jobs") ahead of every remesh/re-texture publish, and
``service._jobs_rework.optimize_job`` and ``revert_model`` (also layer 3,
"service") for a retarget and a restore. Neither layer may import the other
(``tests/test_layering.py``), so the one piece both need to agree on -- what
a version *is*, where it lives, and when the cap evicts one -- has to sit
below both of them. ``pipelines/`` is layer 2, importable from either, and
holds no service or queue state of its own: stdlib only, exactly like
``pipelines/optimize.py`` beside it, whose ``staged_copy`` idiom this reuses
for the same reason optimize.py gives -- ``model.glb`` is served on mere
existence once a job is done, so a plain ``copyfile`` would let a concurrent
reader see a half-written file.

**The lock.** Every writer of ``model.glb`` takes ``MODEL_LOCK`` --
``"optimize"``, the name ``optimize_job`` already took before this module
existed -- through ``svc.convert_lock``/the worker's injected
``artifact_lock`` (``studio.runtime`` wires ``worker.artifact_lock =
svc.convert_lock``, so the two names are one lock, not two). A version pushed
outside that lock could race a concurrent writer's own read-modify-write of
``model_history`` and drop an entry silently; every caller here is expected
to already hold it.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

#: The directory, inside a job's own directory, that holds every kept version.
VERSIONS_DIR = "versions"

#: How many earlier meshes a job keeps at once. The oldest is evicted -- file
#: and index entry both -- the moment a fifth would be kept; a mesh's history
#: is a safety net for the last few reworks, not an unbounded archive.
MAX_MODEL_VERSIONS = 4

#: Every params key that together describes what ``model.glb`` currently *is*
#: -- read into a version's sidecar at ``keep()`` time and restored verbatim
#: by ``revert_model``, because a restore has to put the mesh's *description*
#: back, not only its bytes. ``profile`` matters most: it is in
#: ``vectors.VECTOR_PARAMS``, so a restore that left the wrong tier on the row
#: would key a verdict in the findings corpus to a triangle budget that is no
#: longer on disk. Every key here is verified against the code that writes
#: it: ``profile``/``custom_triangles`` (``optimize_job``), ``optimize``
#: (``_q_mesh._optimize``, ``optimize_job``), ``remesh`` (``_q_mesh._remesh``),
#: ``retexture`` (``_q_sprite._retexture``), ``transform``/``scale_factor``
#: (``_q_mesh._apply_scale``, ``_q_mesh._remesh``, ``optimize_job``),
#: ``mesh_audit``/``mesh_report`` (``_q_mesh._audit_mesh``,
#: ``_q_mesh._audit_published``), ``degraded``
#: (``_q_mesh._note_degraded``/``validation.note_degraded``, spelled as
#: ``ARTIFACT_HEALTH`` in both).
MODEL_PARAMS: tuple[str, ...] = (
    "profile",
    "custom_triangles",
    "optimize",
    "remesh",
    "retexture",
    "transform",
    "scale_factor",
    "mesh_audit",
    "mesh_report",
    "degraded",
)

#: The lock name every ``model.glb`` writer takes -- ``optimize_job``'s own
#: name, restated here so a version push and a retarget can never disagree
#: about which lock protects the file.
MODEL_LOCK = "optimize"


def version_path(job_dir: Path, n: int) -> Path:
    """Where kept version ``n``'s bytes live."""
    return job_dir / VERSIONS_DIR / f"{n}.model.glb"


def meta_path(job_dir: Path, n: int) -> Path:
    """Where kept version ``n``'s params snapshot lives."""
    return job_dir / VERSIONS_DIR / f"{n}.json"


def entries_of(params: dict[str, Any]) -> list[dict[str, Any]]:
    """This job's own ``model_history``, read defensively.

    A hand-edited row, or a test fixture that fills every
    ``service.validation.DERIVED_PARAMS`` key with a marker string (this repo
    has one -- ``tests/service/test_service.py::_finished_job``), can leave
    ``params["model_history"]`` holding anything at all. ``note_degraded``'s
    guard is the precedent: starting fresh beats raising over a value nothing
    here wrote. Every caller that reads ``model_history`` off a live row goes
    through this rather than a bare ``.get(...) or []``, so the one place that
    decides what counts as a valid entry cannot drift from the three call
    sites that trust it.
    """
    raw = params.get("model_history")
    if not isinstance(raw, list):
        return []
    return [e for e in raw if isinstance(e, dict) and isinstance(e.get("n"), int)]


def snapshot(params: dict[str, Any]) -> dict[str, Any]:
    """The subset of ``params`` that describes ``model.glb`` right now.

    Only :data:`MODEL_PARAMS`, and only the keys actually present -- a
    version kept before a key existed (or on a run that never set it, such as
    a remesh whose normalize step failed) must not claim a value it never
    had.
    """
    return {k: params[k] for k in MODEL_PARAMS if k in params}


def stage(
    job_dir: Path,
    entries: list[dict[str, Any]],
    params: dict[str, Any],
    *,
    kind: str,
    geometry: bool,
    detail: str,
    now: float,
) -> list[dict[str, Any]]:
    """Snapshot the ``model.glb`` on disk right now, before it is replaced.

    Stages a copy into ``versions/<n>.model.glb`` and its params sidecar into
    ``versions/<n>.json``, appends the index entry *last* -- both files exist
    before anything claims they do. **Evicts nothing.** The 2026-09-23 audit
    (finding service-01) found the old single-function ``keep`` evicting the
    oldest kept version unconditionally, before the caller's own write had
    even been attempted -- so a failed retarget at the version cap permanently
    lost the oldest recoverable mesh even though nothing on disk actually
    changed. Splitting the stage from the eviction lets a caller try its own
    write first and only :func:`commit` -- which is where eviction lives --
    once that write has actually succeeded; on failure it calls
    :func:`discard_last` instead, which undoes exactly this call and nothing
    more, because no eviction has happened yet to undo.

    ``entries`` is rebound rather than mutated in place, returning a new
    list -- ``_q_mesh._note_degraded``'s rule, and for the same reason: a
    caller holding a shallow ``dict(params)`` copy elsewhere must not see this
    call's append through a reference it never asked to share.

    If there is no ``model.glb`` to keep, ``entries`` comes back unchanged --
    a version cannot describe a file that was never on disk (a job whose mesh
    failed to build at all, or a first-ever optimize on a row nothing has
    reworked yet).
    """
    model_glb = job_dir / "model.glb"
    if not model_glb.exists():
        return list(entries)
    versions_dir = job_dir / VERSIONS_DIR
    versions_dir.mkdir(parents=True, exist_ok=True)
    n = (entries[-1]["n"] + 1) if entries else 1
    dest = version_path(job_dir, n)
    meta = meta_path(job_dir, n)
    tmp_glb = dest.with_name(f".{dest.name}.tmp")
    tmp_meta = meta.with_name(f".{meta.name}.tmp")
    try:
        shutil.copyfile(model_glb, tmp_glb)
        os.replace(tmp_glb, dest)
        tmp_meta.write_text(
            json.dumps(snapshot(params), indent=2), encoding="utf-8"
        )
        os.replace(tmp_meta, meta)
    finally:
        with contextlib.suppress(OSError):
            tmp_glb.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            tmp_meta.unlink(missing_ok=True)
    entry = {
        "n": n,
        "kind": kind,
        "at": now,
        "detail": detail,
        "geometry": bool(geometry),
        "bytes": dest.stat().st_size,
    }
    return [*entries, entry]


def commit(job_dir: Path, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evict past :data:`MAX_MODEL_VERSIONS` and sweep orphaned version files.

    Call this only once the write :func:`stage` was guarding has actually
    succeeded -- eviction deletes a kept mesh's bytes, and that is only safe
    to do once the caller no longer needs to be able to back out via
    :func:`discard_last`. Sweeps anything in ``versions/`` that is not a live
    entry's pair (a dotfile stranded by a crash between the two stages in
    :func:`stage`, or a file an eviction could not remove because the process
    died first).
    """
    out = list(entries)
    while len(out) > MAX_MODEL_VERSIONS:
        evicted = out.pop(0)
        _delete_version_files(job_dir, evicted["n"])
    _sweep_orphans(job_dir, out)
    return out


def keep(
    job_dir: Path,
    entries: list[dict[str, Any]],
    params: dict[str, Any],
    *,
    kind: str,
    geometry: bool,
    detail: str,
    now: float,
) -> list[dict[str, Any]]:
    """:func:`stage` then :func:`commit` in one call.

    For a caller with no failure path of its own to guard -- ``revert_model``
    only reaches this after the version being restored is already known good,
    so there is nothing here for it to back out of on failure the way
    ``optimize_job`` and ``_publish_model_version`` must (2026-09-23 audit,
    findings service-01 and service-03): see :func:`stage` for why those two
    call ``stage``/``commit`` directly instead of this.
    """
    return commit(job_dir, stage(
        job_dir, entries, params, kind=kind, geometry=geometry, detail=detail, now=now
    ))


def discard_last(job_dir: Path, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Undo the most recent :func:`stage` (or :func:`keep`) -- for a caller
    whose own write then failed, so the version it just pushed must not claim
    to describe a replacement that never happened.

    Meant to be called on a *staged, not yet committed* list -- before
    :func:`commit` has run -- so undoing it is exactly undoing the one
    ``stage`` call and never an eviction, which is the fix for the 2026-09-23
    audit's finding service-01: the old combined ``keep`` evicted the oldest
    kept version unconditionally, before this function ever ran, so calling
    this after a failure only ever un-pushed the entry just added and never
    restored what eviction had already deleted.

    ``optimize_job``'s use: it stages a version *before* calling
    ``optimize.run`` (the old ``model.glb`` has to be snapshotted while it is
    still the file on disk), and a raised ``OptimizeError`` means the row's
    ``model.glb`` is untouched -- so the version just staged describes nothing
    that changed and has to come back off the index, files and all.
    """
    if not entries:
        return entries
    out = list(entries)
    dropped = out.pop()
    _delete_version_files(job_dir, dropped["n"])
    return out


def crossed_geometry(entries: list[dict[str, Any]], n: int) -> bool:
    """Whether restoring version ``n`` would change geometry back.

    Entry ``k``'s ``geometry`` flag describes the step that *replaced*
    version ``k`` -- the transition from ``k``'s saved bytes to whatever came
    next (the next entry's bytes, or the live ``model.glb`` for the newest
    entry). Restoring version ``n`` undoes every one of those transitions
    from ``n`` up to the live mesh, so this is true when any of them changed
    geometry -- not only the one named ``n``: a version two reworks back can
    still be a pure surface restore if neither of the reworks since touched
    geometry, and a version from just before a re-texture can still cross
    geometry if a remesh landed after it.
    """
    return any(e["geometry"] for e in entries if e["n"] >= n)


def _delete_version_files(job_dir: Path, n: int) -> None:
    with contextlib.suppress(OSError):
        version_path(job_dir, n).unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        meta_path(job_dir, n).unlink(missing_ok=True)


def _sweep_orphans(job_dir: Path, entries: list[dict[str, Any]]) -> None:
    """Delete anything in ``versions/`` that is not a live entry's pair.

    The eviction loop above already deletes what it evicts; this catches what
    a crash between a stage-write and the index append could strand -- a
    ``.n.model.glb.tmp`` dotfile nothing ever swept, or a version whose files
    survived an eviction that itself died mid-delete. Cheap: at most
    ``MAX_MODEL_VERSIONS`` live pairs plus stragglers, never a job's whole
    directory.
    """
    versions_dir = job_dir / VERSIONS_DIR
    if not versions_dir.is_dir():
        return
    live = set()
    for e in entries:
        live.add(version_path(job_dir, e["n"]).name)
        live.add(meta_path(job_dir, e["n"]).name)
    for path in versions_dir.iterdir():
        if path.is_file() and path.name not in live:
            with contextlib.suppress(OSError):
                path.unlink()
