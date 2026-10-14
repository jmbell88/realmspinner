"""Everything about a rig, a pose or a sprite draft that is a path
or a record on disk.

Split out of the former ``rigging.py`` (P4 of the 2026-09 restructure): this is
the storage half -- resource ids, the rig's own temp/served-name convention,
and one file per pose/sprite-draft under a job directory, with no
skeleton math and no template registry in it. Deliberately a leaf inside
``kernels/rig/``: :mod:`.templates` and :mod:`.skeleton` both import
:class:`RigError` from here, so this module may not import either of them
back, or the package would cycle.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Same shape and generator as a job id (uuid4().hex[:12]), and validated for the
# same reason: config.job_dir() and every path built under it do no sanitisation.
# ``\Z``, not ``$``, for the reason ``service.validation.JOB_ID_RE`` gives: ``$``
# matches before a trailing newline (the 2026-10-03 audit, finding service-21).
RESOURCE_ID_RE = re.compile(r"^[0-9a-f]{12}\Z")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def is_valid_id(value: str) -> bool:
    return bool(RESOURCE_ID_RE.match(value))


class RigError(ValueError):
    """A rig.json this reads cannot answer for, naming the field.

    ``poselib.RecordError``'s shape, for a rig record instead of a pose one:
    a ``ValueError`` subclass so nothing that already catches ``ValueError``
    changes, with ``field`` carried so ``service.errors.invalid_from`` can
    point the refusal at a control instead of the message arriving addressed
    to nothing.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


# --- the rig's own temp/served names -----------------------------------------
#
# The worker writes to the *_TMP names; the queue renames them onto the served
# names on success (finalize_rig). Two reasons, both load-bearing: Blender's
# GLB export writes in place over seconds, and rig.json -- the completion gate
# _attach_files and the file route key on -- is already satisfied by an
# earlier rig during a re-rig, so pointing the worker at the served names
# would let a concurrent GET read a truncated rig.glb. It also makes
# cancellation naturally non-destructive: discarding a half-written re-rig
# deletes the temps, never the previous successful rig's artifacts.
# The suffixes are load-bearing: Blender's glTF exporter appends ".glb" to a
# filepath that does not already end in it, so a ".tmp" suffix would make the
# worker write ".rig.tmp.glb.glb" and finalize_rig find nothing.
RIG_GLB_TMP = ".rig.tmp.glb"
RIG_JSON_TMP = ".rig.tmp.json"

# Where a re-texture assembles its replacement mesh before it becomes the one
# being served. Named here beside the rig temps because it is the same rule for
# the same reason -- a job writing into a *different* job's directory publishes
# by rename, so a cancel deletes a temp and never a finished artifact -- and
# because both the worker that writes it and ``_discard_artifacts`` need the
# one spelling. Nothing goes near Blender's exporter here, so the suffix is
# free; it keeps the rig temps' shape anyway.
RETEXTURE_GLB_TMP = ".retexture.tmp.glb"

# The remesh's staging name, beside the ``model.glb`` it is a candidate
# replacement for -- ``RETEXTURE_GLB_TMP``'s rule exactly: published by
# rename, discarded by a cancel, one spelling for the writer and the sweeper.
REMESH_GLB_TMP = ".remesh.tmp.glb"


def finalize_rig(job_dir: Path) -> None:
    """Rename the worker's temp artifacts onto the served names.

    GLB first, json second: rig.json is the completion marker, so at the
    moment it lands the rig.glb beside it is already the matching one.

    The renames retry briefly: on Windows, replacing rig.glb while a
    FileResponse is mid-stream from it raises PermissionError (files open
    without FILE_SHARE_DELETE), and failing an otherwise successful re-rig
    at the last step over a transient reader is the wrong trade.

    The pair is not atomic, and the failure that matters is the second half:
    once the GLB has landed, a JSON rename that exhausts its retries would leave
    the *new* rig.glb beside the *old* rig.json -- a completion marker
    advertising a skeleton the mesh no longer has. So that failure takes the
    stale marker with it: the directory then reads as "not rigged", which is
    true, and the caller's ``discard_rig_temps`` removes the unpublished JSON.

    A pose baked under the *previous* skeleton (``poses/<id>.glb``) or an
    ``animated.glb`` baked from the previous rig.glb's clips depicts joints
    this rig no longer has -- existence is each artifact's whole freshness
    test (``derive.get_file``'s ``animated.glb`` branch, ``rig.posed_model``),
    so left alone either would go on being served, silently wrong, forever.
    Removed here, before the renames below land the new rig.json: a crash in
    between costs a rebake under the *old* rig on next request, never a stale
    one served under the new rig.json's authority. The pose *records*
    (``poses/<id>.json``) are untouched -- a pose is still the same rotation
    request, only its cached bake is invalidated.
    """
    _unlink_stale(job_dir / "animated.glb")
    # The previous rig's deformation review (poser-09, 2026-10-03): the QA tail
    # only replaces these when it renders, so a re-rig whose QA is skipped
    # (no battery for the template, kill switch, cancel) or fails would serve
    # the old skeleton's sheet beside the new one. JSON first, PNG second, so a
    # reader never finds a verdict without its picture's absence meaning "none".
    _unlink_stale(rig_qa_path(job_dir))
    _unlink_stale(rig_qa_png_path(job_dir))
    _sweep_pose_bakes(job_dir)

    for src, dest in ((RIG_GLB_TMP, "rig.glb"), (RIG_JSON_TMP, "rig.json")):
        for attempt in range(10):
            try:
                os.replace(job_dir / src, job_dir / dest)
                break
            except PermissionError:
                if attempt == 9:
                    if dest == "rig.json":
                        with contextlib.suppress(OSError):
                            (job_dir / "rig.json").unlink()
                    raise
                time.sleep(0.5)
    # Again, now that the new rig.json is the one served: a bake that began under
    # the old skeleton and published between the sweep above and this point is
    # exactly the stale pose the first sweep could not see yet (the 2026-10-03
    # audit, finding service-09). Anything baked after this line reads the new
    # rig, and ``service.rig.posed_model`` refuses to publish a bake whose
    # rig.json moved underneath it.
    _sweep_pose_bakes(job_dir)


def _unlink_stale(path: Path, *, attempts: int = 4) -> bool:
    """Delete one stale derived artifact, riding out a transient reader.

    The 2026-10-03 audit, finding poser-33: these unlinks sat outside the
    PermissionError retry that wraps the renames, so an export or Save GLB copy
    holding ``animated.glb`` (or a pose bake) open on Windows made the publish
    raise before any rename and threw a minutes-long Blender solve away -- the
    trade this module's own docstring calls wrong. A file that is still held
    after the retries is left for the next sweep (``finalize_rig`` runs two,
    and a stale bake is only ever a rebake away) rather than failing the job;
    the return value says whether it went."""
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return True
        except PermissionError:
            if attempt == attempts - 1:
                log.warning(
                    "could not remove stale %s; a reader still holds it open", path
                )
                return False
            time.sleep(0.5)
    return False


def _sweep_pose_bakes(job_dir: Path) -> None:
    """Delete every published pose bake, and nothing a bake is still writing.

    Dot-prefixed files are ``posed_model``'s staging names
    (``.<id>.tmp.glb``): ``glob("*.glb")`` matches them, and deleting one under
    a running Blender made the bake's own rename fail with only "could not bake
    this pose" (the 2026-10-03 audit, finding service-09). The bake publishes or
    discards its own staging file.
    """
    poses_dir = job_dir / POSE_DIR_NAME
    if poses_dir.is_dir():
        for stale in poses_dir.glob("*.glb"):
            if not stale.name.startswith("."):
                _unlink_stale(stale)


def discard_rig_temps(job_dir: Path) -> None:
    """Remove whatever a failed or cancelled rig run left behind. Idempotent."""
    (job_dir / RIG_GLB_TMP).unlink(missing_ok=True)
    (job_dir / RIG_JSON_TMP).unlink(missing_ok=True)


def read_rig(job_dir: Path) -> dict[str, Any] | None:
    return read_record(job_dir / "rig.json", "rig.json")


def validate_rig_bones(bones: Any) -> list[str]:
    """A rig.json's ``bones`` list, name-checked. Raises :class:`RigError`.

    dev/INVARIANTS.md states a pose *or rig* JSON is validated at the read
    door, not only at the write door -- but until the 2026-09-08 audit
    (poser-02) that was only ever built for pose records: a rig.json passed
    ``read_record``'s three file-level guards (valid JSON, valid dict, under
    the byte ceiling) and then a bare ``[b["name"] for b in bones]`` crashed
    on a bone with no ``name`` key with an uncaught ``KeyError``, one field
    deeper than the case the invariant already names as fixed. This is that
    same door for rig.json's bone list, tolerant like ``poselib.validate_
    record``'s own bone check: a bone entry that is not a dict, or whose name
    is not a non-empty string, is refused by name rather than by traceback.
    """
    if not isinstance(bones, list):
        raise RigError("rig.json's bones must be a list", field="bones")
    names: list[str] = []
    for entry in bones:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name:
            raise RigError("rig.json has a bone with no name", field="bones")
        names.append(name)
    return names


def rig_bone_names(job_dir: Path) -> list[str] | None:
    rig = read_rig(job_dir)
    if rig is None:
        return None
    return validate_rig_bones(rig.get("bones", []))


# --- pose storage -----------------------------------------------------------
#
# One file per pose under <job_dir>/poses/, not a single poses.json. A pose is
# saved from the browser while the queue may be writing that same job's
# directory, and per-file writes need no read-modify-write and so no lock; the
# derived <pose_id>.glb sits beside its json for the same reason.

POSE_DIR_NAME = "poses"

# A ceiling so a scripted client can't turn a job directory into a million
# files. Far above any plausible hand-authored set.
MAX_POSES = 500

# A rig or pose record is a few KB of bone names and quaternions; a megabyte is
# already three orders of magnitude of headroom. Checked by stat *before* the
# read so a blob dropped into a job directory costs a log line rather than the
# RAM to parse it. Shared with poselib, which stores the same shape of record.
MAX_RECORD_BYTES = 1 << 20


def read_record(path: Path, what: str) -> dict[str, Any] | None:
    """One JSON sidecar, or ``None`` -- with the three guards every one needs.

    Written once because the sidecar readers each had a two-line version of
    this, and each was missing all three guards.

    - **The ceiling is checked by ``stat`` before the read.** A blob dropped
      into a job directory should cost a log line, not the RAM to parse it.
    - **``ValueError``, not ``JSONDecodeError``.** ``UnicodeDecodeError`` is a
      ``ValueError`` and is *not* a ``JSONDecodeError``, so a sidecar holding
      bytes that are not UTF-8 raised straight out of the reader, through the
      listing, and into the pane -- instead of costing one record.
    - **The document has to be an object.** A valid-JSON array passes the
      parse and then fails at the first ``record.get(...)``, somewhere else
      entirely.
    """
    if not path.exists():
        return None
    try:
        size = path.stat().st_size
        if size > MAX_RECORD_BYTES:
            log.warning("ignoring %s at %s: %d bytes, over the ceiling", what, path, size)
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.exception("unreadable %s at %s", what, path)
        return None
    if not isinstance(record, dict):
        log.warning("ignoring %s at %s: the document is not an object", what, path)
        return None
    return record


def write_json_staged(path: Path, payload: dict[str, Any], *, prefix: str) -> None:
    """Write one JSON document by staging it beside its destination.

    The Settings.flush rule, held in one place because three callers had a
    byte-identical copy of it: an overwrite that died mid-write used to leave
    the existing record truncated, and a pose file is the only copy of its
    rotations. ``prefix`` names the temp after what it will become, so a
    stranded one is attributable; ``BaseException`` rather than ``Exception``
    because a KeyboardInterrupt between the write and the rename must still
    take the dotfile with it.
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def pose_dir(job_dir: Path) -> Path:
    return job_dir / POSE_DIR_NAME


def pose_path(job_dir: Path, pose_id: str) -> Path:
    """The record for one pose. Raises ValueError on an id that isn't ours.

    The guard is the point: this is the only place a caller-supplied pose id is
    turned into a path, and ``job_dir / pose_id`` does no sanitising of its own.
    """
    if not is_valid_id(pose_id):
        raise ValueError(f"malformed pose id {pose_id!r}")
    return pose_dir(job_dir) / f"{pose_id}.json"


def pose_glb_path(job_dir: Path, pose_id: str) -> Path:
    return pose_path(job_dir, pose_id).with_suffix(".glb")


def save_pose(
    job_dir: Path,
    pose: dict[str, Any],
    pose_id: str | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a validated pose payload and return the stored record.

    Passing an existing ``pose_id`` overwrites it in place -- that is how the
    editor saves an edit rather than accumulating near-duplicates. The derived
    GLB is dropped on overwrite, since it no longer depicts the pose.

    ``extra`` is merged into the record before the atomic write -- how a
    library-pose snapshot carries ``root_translation`` and ``source_pose``
    without every plain save learning those fields exist. It may not touch the
    keys this function owns: an ``extra`` that renamed the pose or swapped its
    bones would be a second writer of the same fact.
    """
    pose_id = pose_id or new_id()
    path = pose_path(job_dir, pose_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reuse the existing record's ``created`` on overwrite, as ``poselib.
    # save_record`` already does -- the 2026-09-20 audit's poser-03 found this
    # door stamping a fresh one on every edit instead, which jumps an edited
    # pose to the end of a list sorted oldest-first (``list_poses`` sorts on
    # this field) purely because it was re-saved, not because it is new.
    existing = read_record(path, "pose")
    record = {
        "id": pose_id,
        "name": pose["name"],
        "bones": pose["bones"],
        "created": existing["created"] if existing and "created" in existing else time.time(),
    }
    if extra:
        protected = {"id", "name", "bones", "created"} & set(extra)
        if protected:
            raise ValueError(f"extra may not override {sorted(protected)}")
        record.update(extra)
    # The GLB is dropped *before* the JSON is staged in, not after. The two
    # statements are not atomic together, and a crash between them is real:
    # the 2026-09-08 audit's poser-04 found the old order -- write the JSON,
    # then drop the GLB -- left a crash between them pairing the *new* pose
    # record with the *old* baked GLB on disk, which posed_model's
    # ``if not path.exists(): ... bake ...`` then served as fresh forever,
    # with no error or staleness signal. This order's worst case is a crash
    # leaving the *old* JSON with no cached GLB -- costing only a redundant
    # rebake next time the pose is fetched, never stale content.
    pose_glb_path(job_dir, pose_id).unlink(missing_ok=True)
    write_json_staged(path, record, prefix=f".{pose_id}.")
    return record


def read_pose(job_dir: Path, pose_id: str) -> dict[str, Any] | None:
    return read_record(pose_path(job_dir, pose_id), "pose")


def _created_key(record: dict[str, Any]) -> float:
    """Sort key for ``created``, tolerant of a hand-edited file.

    A string timestamp sorted against floats raises and costs the whole list;
    an unusable one sorts as 0.0 instead, and a stable sort leaves equal keys
    in the filename order they were globbed in.
    """
    value = record.get("created", 0.0)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def list_poses(job_dir: Path) -> list[dict[str, Any]]:
    """Every saved pose, oldest first. A corrupt file costs itself, not the list."""
    directory = pose_dir(job_dir)
    if not directory.is_dir():
        return []
    poses = []
    for path in sorted(directory.glob("*.json")):
        if not is_valid_id(path.stem):
            continue
        record = read_pose(job_dir, path.stem)
        if record is None:
            continue
        if record.get("id") != path.stem:
            # The filename is the address every caller uses -- read, delete and
            # overwrite all go through pose_path(stem). A record whose internal
            # id disagrees is listed under its stem so it stays reachable; the
            # next save writes the reconciled id back.
            log.warning(
                "pose %s records id %r; listing it under its filename", path, record.get("id")
            )
            record = dict(record, id=path.stem)
        poses.append(record)
    poses.sort(key=_created_key)
    return poses


def delete_pose(job_dir: Path, pose_id: str) -> bool:
    path = pose_path(job_dir, pose_id)
    # ``is_file``, not ``exists``: the 2026-10-03 audit (poser-32) -- a directory
    # squatting at ``poses/<id>.json`` read as found and ``unlink`` raised
    # PermissionError on Windows, the failure ``delete_sprite_draft`` was
    # already fixed against (2026-09-18, poser-02).
    if not path.is_file():
        return False
    # The derived GLB goes before the source JSON it depends on, matching
    # save_pose's ordering above and for the same reason (poser-05, the
    # 2026-09-08 audit): the old order -- delete the JSON, then the GLB --
    # left a crash between them stranding an orphaned <pose_id>.glb that
    # nothing lists, sweeps, or ever deletes, since posed_model raises
    # NotFound on the missing JSON before it would ever look for the GLB.
    # This order's worst case is a crash that leaves the JSON undeleted
    # (delete_pose raises, the pose still "exists") with no orphan behind it.
    pose_glb_path(job_dir, pose_id).unlink(missing_ok=True)
    path.unlink()
    return True


# --- the deformation QA sheet -----------------------------------------------
#
# One per rig, overwritten by the next rig: a review artifact about the rig, not
# something the user asked for. It lands in the source job's directory beside
# rig.glb for the reason the rig itself does -- it describes that mesh. (A
# directory the retired character-sheet kinds once wrote is not touched by this
# and nothing lists or deletes it any more -- ``retired.py``.)

RIG_QA_PNG = "rig_qa.png"
RIG_QA_JSON = "rig_qa.json"


def rig_qa_png_path(job_dir: Path) -> Path:
    return job_dir / RIG_QA_PNG


def rig_qa_path(job_dir: Path) -> Path:
    return job_dir / RIG_QA_JSON


# --- sprite sheet drafts ----------------------------------------------------
#
# Same storage shape as poses, and for the same reasons -- but with one rule the
# others do not have: a draft is **write-once**. Every generate run mints a new
# id, so no draft is ever rewritten in place, and the pane can therefore cache
# this listing behind the directory's mtime without a stale record surviving an
# edit that never happens. Deleting is the only mutation.
#
# A draft is a *pair* of candidates from one run, so it is three files: the two
# PNGs, then the sidecar last, which is the completion marker.

SPRITE_DIR_NAME = "sprites"

# Two files per draft and a thumbnail per candidate in the sidebar; well under
# MAX_SHEETS because these accumulate per *attempt* rather than per keeper.
MAX_SPRITE_DRAFTS = 50

SPRITE_CANDIDATES = ("a", "b")


def sprite_dir(job_dir: Path) -> Path:
    return job_dir / SPRITE_DIR_NAME


def sprite_draft_path(job_dir: Path, draft_id: str) -> Path:
    if not is_valid_id(draft_id):
        raise ValueError(f"malformed sprite draft id {draft_id!r}")
    return sprite_dir(job_dir) / f"{draft_id}.json"


def sprite_draft_png_path(job_dir: Path, draft_id: str, candidate: str) -> Path:
    """``<id>.a.png`` / ``<id>.b.png``, beside the sidecar.

    The candidate letter is checked here rather than trusted: it arrives from a
    UI route, and ``<id>.<anything>.png`` would otherwise be a path a caller
    chooses. ``<id>.a`` also fails ``is_valid_id``, so ``list_sprite_drafts``
    skips the PNGs for free without knowing this naming exists.
    """
    if candidate not in SPRITE_CANDIDATES:
        raise ValueError(f"unknown sprite candidate {candidate!r}")
    return sprite_draft_path(job_dir, draft_id).with_suffix(f".{candidate}.png")


def read_sprite_draft(job_dir: Path, draft_id: str) -> dict[str, Any] | None:
    return read_record(sprite_draft_path(job_dir, draft_id), "sprite draft")


def list_sprite_drafts(job_dir: Path) -> list[dict[str, Any]]:
    """Every finished draft, oldest first. A draft missing a PNG is not one.

    Missing *which* PNG is the sidecar's answer and not this function's, which
    is the one thing here that is not obvious. It used to demand both letters of
    :data:`SPRITE_CANDIDATES`, and that was a hidden cap: a big sheet is drawn
    as a single candidate (``spritesynth.default_candidates``), so every
    eight-direction draft ever made would have been complete on disk, correct in
    its sidecar, and invisible in the pane. The record is read first and its own
    ``candidates`` list says which images it claims -- which is a stricter check
    than the old one as well as a truer one, since it also catches a draft
    claiming a candidate whose PNG never landed.
    """
    directory = sprite_dir(job_dir)
    if not directory.is_dir():
        return []
    drafts = []
    for path in sorted(directory.glob("*.json")):
        if not is_valid_id(path.stem):
            continue
        record = read_sprite_draft(job_dir, path.stem)
        if record is None:
            continue
        claimed = record.get("candidates")
        # A record with no usable list falls back to demanding both, which is
        # what every draft written before the count was a choice carries anyway
        # -- untrusted JSON does not get to shrink the check to nothing.
        letters = (
            SPRITE_CANDIDATES[: len(claimed)]
            if isinstance(claimed, list) and claimed
            else SPRITE_CANDIDATES
        )
        # .is_file(), not .exists(): a directory at ``<id>.a.png`` would
        # otherwise read as a ready candidate (poser-02, the 2026-09-18
        # audit).
        if not all(path.with_suffix(f".{c}.png").is_file() for c in letters):
            continue
        drafts.append(record)
    # Same tolerance as ``list_poses``, same reason (poser-rig-04): a
    # string-typed ``created`` on one sidecar must cost that field, not the
    # whole listing.
    drafts.sort(key=_created_key)
    return drafts


def delete_sprite_draft(job_dir: Path, draft_id: str) -> bool:
    """The trio. A draft is one run, so its two candidates go together."""
    paths = [sprite_draft_path(job_dir, draft_id)] + [
        sprite_draft_png_path(job_dir, draft_id, c) for c in SPRITE_CANDIDATES
    ]
    # .is_file(), not .exists(): a directory squatting at one of these names
    # would fail ``unlink`` (poser-02, the 2026-09-18 audit).
    if not any(p.is_file() for p in paths):
        return False
    for path in paths:
        path.unlink(missing_ok=True)
    return True
