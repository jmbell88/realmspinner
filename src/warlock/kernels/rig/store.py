"""Everything about a rig, a pose, a sheet or a sprite draft that is a path
or a record on disk.

Split out of the former ``rigging.py`` (P4 of ``dev/RESTRUCTURE.md``): this is
the storage half -- resource ids, the rig's own temp/served-name convention,
and one file per pose/sheet/sprite-draft under a job directory, with no
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
RESOURCE_ID_RE = re.compile(r"^[0-9a-f]{12}$")


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
    (job_dir / "animated.glb").unlink(missing_ok=True)
    poses_dir = job_dir / POSE_DIR_NAME
    if poses_dir.is_dir():
        for stale in poses_dir.glob("*.glb"):
            stale.unlink(missing_ok=True)

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

    Written once because it was written four times and skipped three: the sheet,
    the pixel-sheet and the sprite-draft readers each had the two-line version
    of this, and each was missing all three guards.

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
    record = {
        "id": pose_id,
        "name": pose["name"],
        "bones": pose["bones"],
        "created": time.time(),
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
    if not path.exists():
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


# --- sprite sheets ----------------------------------------------------------
#
# Same storage shape as poses, and beside them, for the same reasons: a sheet
# belongs to the mesh it depicts, one file per sheet needs no locking, and the
# id guard is the only thing between a caller and an arbitrary path.

SHEET_DIR_NAME = "sheets"

MAX_SHEETS = 200

# Same cap as MAX_POSE_NAME (poses.py), for the same reason: a label the UI
# has to render.
MAX_SHEET_NAME = 64


def sheet_dir(job_dir: Path) -> Path:
    return job_dir / SHEET_DIR_NAME


# --- the deformation QA sheet -----------------------------------------------
#
# One per rig, overwritten by the next rig, and deliberately *not* in sheets/:
# it is a review artifact about the rig, not a sprite sheet the user asked for,
# so it must not appear in list_sheets, count against MAX_SHEETS or be deleted
# by delete_sheet. It lands in the source job's directory beside rig.glb for
# the reason the rig itself does -- it describes that mesh.

RIG_QA_PNG = "rig_qa.png"
RIG_QA_JSON = "rig_qa.json"


def rig_qa_png_path(job_dir: Path) -> Path:
    return job_dir / RIG_QA_PNG


def rig_qa_path(job_dir: Path) -> Path:
    return job_dir / RIG_QA_JSON


def sheet_path(job_dir: Path, sheet_id: str) -> Path:
    if not is_valid_id(sheet_id):
        raise ValueError(f"malformed sheet id {sheet_id!r}")
    return sheet_dir(job_dir) / f"{sheet_id}.json"


def sheet_png_path(job_dir: Path, sheet_id: str) -> Path:
    return sheet_path(job_dir, sheet_id).with_suffix(".png")


def sheet_pixel_path(job_dir: Path, sheet_id: str) -> Path:
    """The pixel restyle's sidecar, beside the render's.

    ``<id>.pixel.json`` rather than a second directory, and it is invisible to
    ``list_sheets`` for free: ``<id>.pixel`` fails ``is_valid_id``, so the
    listing skips it without needing to know this feature exists.
    """
    return sheet_path(job_dir, sheet_id).with_suffix(".pixel.json")


def sheet_pixel_png_path(job_dir: Path, sheet_id: str) -> Path:
    return sheet_path(job_dir, sheet_id).with_suffix(".pixel.png")


def read_sheet_pixel(job_dir: Path, sheet_id: str) -> dict[str, Any] | None:
    return read_record(sheet_pixel_path(job_dir, sheet_id), "pixel sheet")


def read_sheet(job_dir: Path, sheet_id: str) -> dict[str, Any] | None:
    return read_record(sheet_path(job_dir, sheet_id), "sheet")


def list_sheets(job_dir: Path) -> list[dict[str, Any]]:
    """Every finished sheet, oldest first.

    A sidecar with no PNG beside it is skipped: the queue writes the PNG first
    and the sidecar last, so the sidecar is the completion marker -- but a
    half-cleaned directory should not advertise an image that isn't there.
    """
    directory = sheet_dir(job_dir)
    if not directory.is_dir():
        return []
    sheets = []
    for path in sorted(directory.glob("*.json")):
        # .is_file(), not .exists(): the 2026-09-07/2026-09-08 audits fixed the
        # identical presence check at every other site in this area (sheet.pack,
        # pixelize.reduce_frames, troupe_mode.scores/atlas_texture) because
        # .exists() is also True for a directory, which a completed sheet's PNG
        # name never is but a hand-dropped one could be -- and this reader was
        # the one site the 2026-09-11 audit (troupe-06) found still unfixed.
        if not is_valid_id(path.stem) or not path.with_suffix(".png").is_file():
            continue
        record = read_sheet(job_dir, path.stem)
        if record is not None:
            sheets.append(record)
    sheets.sort(key=lambda s: s.get("created", 0.0))
    return sheets


def delete_sheet(job_dir: Path, sheet_id: str) -> bool:
    """Both files, and the pixel restyle's pair if one was ever made.

    The restyle is derived from this render and depicts nothing else, so
    leaving it behind would leave a sprite sheet of a sheet that is gone.
    """
    paths = [
        sheet_path(job_dir, sheet_id),
        sheet_png_path(job_dir, sheet_id),
        sheet_pixel_path(job_dir, sheet_id),
        sheet_pixel_png_path(job_dir, sheet_id),
    ]
    # .is_file(), not .exists(): the same swap list_sheets got below (the
    # 2026-09-07/2026-09-08 audits) -- a directory squatting at one of these
    # names would otherwise read as "found" and then fail ``unlink`` (raises
    # ``IsADirectoryError`` on POSIX, ``PermissionError`` on Windows) instead
    # of the clean "nothing to delete" this returns (poser-02, the 2026-09-18
    # audit).
    if not any(p.is_file() for p in paths):
        return False
    for path in paths:
        path.unlink(missing_ok=True)
    return True


# --- sprite sheet drafts ----------------------------------------------------
#
# Same storage shape again, and for the same reasons -- but with one rule the
# others do not have: a draft is **write-once**. Every generate run mints a new
# id, so no draft is ever rewritten in place, and the pane can therefore cache
# this listing behind the directory's mtime without a stale record surviving an
# edit that never happens. Deleting is the only mutation.
#
# A draft is a *pair* of candidates from one run, so it is three files: the two
# PNGs, then the sidecar last, which is the completion marker -- the same order
# and the same reason as sheets.

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
        # otherwise read as a ready candidate the way an empty one could at
        # ``list_sheets`` before the 2026-09-07/2026-09-08 audits fixed it
        # there (poser-02, the 2026-09-18 audit).
        if not all(path.with_suffix(f".{c}.png").is_file() for c in letters):
            continue
        drafts.append(record)
    drafts.sort(key=lambda d: d.get("created", 0.0))
    return drafts


def delete_sprite_draft(job_dir: Path, draft_id: str) -> bool:
    """The trio. A draft is one run, so its two candidates go together."""
    paths = [sprite_draft_path(job_dir, draft_id)] + [
        sprite_draft_png_path(job_dir, draft_id, c) for c in SPRITE_CANDIDATES
    ]
    # .is_file(), not .exists(): same reason as delete_sheet above
    # (poser-02, the 2026-09-18 audit).
    if not any(p.is_file() for p in paths):
        return False
    for path in paths:
        path.unlink(missing_ok=True)
    return True
