"""What a bulk delete keeps, because finishing a grading pass was destroying it.

**The incident, and it is measured.** ``docs/measurements/2026-09-07-mesh-probe-
preregistration.md`` opened by counting what it had to fit against: nine
model-stage human verdicts on the machine, and **zero** of them still carrying a
``source.glb``. Every one had been graded to completion inside a sweep, and
``review_mode.JudgingPass``' cleanup licence -- "a sweep every unit of which has
a verdict has its assets removed, with a toast and no dialog" -- is what took
them. The loop consumed its own corpus: the act of finishing the judgement was
the act of deleting the evidence for it. The probe is still unfitted, and
``scripts/qualify_tiers.py`` warns about the same shortage from the other end.

``jobs.retained_job_ids`` already solves half of this and its docstring argues
the half it solves: an *accept* is kept **in place**, because ``tiercheck`` and
the probe both read accepted meshes. What it deliberately does not keep is a
**reject**, and the 2026-08-09 note beside it says why -- a model-stage reject is
carried by its row, "the finding *is* 'this vector produced a bad mesh'". That is
true of that finding and false of every *later* question: a regression probe over
grades needs both ends of the scale, and a re-analysis a month on needs the
pixels whatever they were graded.

So this is retention's other half, and its shape is different on purpose:
retention keeps an asset where it is, and this moves a small copy of it somewhere
a delete does not reach.

**Three rules.**

* **Archive before the delete, never instead of it.** The reclaim the user asked
  for still happens; ``os.link`` where the volume allows means the archive costs
  nothing at all until the job directory actually goes, and then it holds the
  only remaining link to the bytes.
* **Only what was judged or what was asked for.** A verdict of either class, or a
  corpus tag. Everything else is ordinary work, and archiving all of it would
  turn ``prune_jobs`` -- whose entire job is reclaiming disk -- into a rename.
* **``job.json`` is written last, and is the completion gate.** ``journal.py``'s
  rule, and here for its reason exactly: an archive interrupted halfway is a
  directory of files nobody can say anything about, and a reader has to be able
  to tell that from a finished one without trusting a directory listing.

``clean_jobs`` is deliberately *not* a caller. It says in as many words that it
keeps nothing -- "a user reclaiming a disk, handing a machine on, or starting a
corpus over is not asking for a reclaim that quietly keeps the largest meshes on
it" -- and an archive behind the user's back would break the one promise it
makes.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from ..provenance import file_fingerprint

log = logging.getLogger(__name__)

#: What is copied out of a job directory, in the order a reader wants them.
#:
#: ``source.glb`` is the point of the exercise -- it is what ``tiercheck`` and the
#: mesh probe read, and the one artifact a row cannot stand in for.
#: ``cutout.png`` is here because it is what the reconstruction was actually run
#: from once a promotion carries an approved matte, and a corpus that recorded
#: the prompt but not the pixels is the mistake this module exists to stop being
#: repeated. Everything under ``files.DERIVED`` is left behind: it is a pure
#: function of a file that is being kept.
ARCHIVED = (
    "input.png",
    "cutout.png",
    "reference.png",
    "control.png",
    "source.glb",
    "model.glb",
)

#: Written last. Its presence is what makes an archive readable; see the module
#: docstring.
MANIFEST = "job.json"


def is_complete(archive_dir: Path) -> bool:
    """Whether this archive finished being written."""
    return (Path(archive_dir) / MANIFEST).is_file()


def _verdicts_by_job(svc: Any) -> dict[str, list[dict[str, Any]]]:
    """Every latest verdict, grouped by job. One query per delete, not per job.

    A failure here degrades to "archive on tags alone" rather than aborting: the
    caller is in the middle of a delete the user asked for, and a database read
    that did not work is no reason to keep two hundred meshes.
    """
    try:
        out: dict[str, list[dict[str, Any]]] = {}
        for verdict in svc.store.latest_verdicts():
            out.setdefault(str(verdict["job_id"]), []).append(dict(verdict))
        return out
    except Exception:
        log.exception("could not read verdicts; archiving on tags alone")
        return {}


def _tagged(job: dict[str, Any]) -> bool:
    """Whether a campaign submitter marked this row as a measurement."""
    return bool((job.get("params") or {}).get("tags"))


def archive_job(
    svc: Any,
    job_id: str,
    *,
    reason: str,
    verdicts: dict[str, list[dict[str, Any]]] | None = None,
) -> Path | None:
    """Copy ``job_id``'s evidence out of the library. -> where, or None.

    Never raises. An archive that cannot be written must not stop the delete the
    user asked for; the failure is logged and the reclaim proceeds, which is the
    call every other non-fatal step in this layer makes.

    ``os.link`` first and ``copy2`` after: on one volume a hard link is free and
    instant, which matters because this runs inside a loop over a whole sweep,
    and it becomes a real copy the moment the original is unlinked. A different
    volume, a filesystem without links, or a file that is already gone all fall
    through to the copy or are skipped.
    """
    try:
        job = svc.store.get(job_id)
        if job is None:
            return None
        job_dir = Path(svc.job_dir(job_id))
        root = Path(svc.config.evidence_dir) / _bucket(reason) / job_id
        root.mkdir(parents=True, exist_ok=True)

        files: dict[str, dict[str, Any]] = {}
        for name in ARCHIVED:
            src = job_dir / name
            if not src.is_file():
                continue
            dest = root / name
            try:
                if dest.exists():
                    dest.unlink()
                os.link(src, dest)
            except OSError:
                try:
                    shutil.copy2(src, dest)
                except OSError:
                    log.exception("could not archive %s of job %s", name, job_id)
                    continue
            files[name] = {
                "bytes": dest.stat().st_size,
                "fingerprint": file_fingerprint(dest),
            }

        doc = {
            "archived_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "reason": reason,
            "job": {
                k: job.get(k)
                for k in ("id", "kind", "stage", "status", "prompt", "created_at")
            },
            "params": job.get("params") or {},
            "verdicts": list((verdicts or {}).get(job_id) or ()),
            "files": files,
        }
        # Last, and that ordering is the whole gate. See MANIFEST.
        (root / MANIFEST).write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return root
    except Exception:
        log.exception("could not archive evidence for job %s", job_id)
        return None


def archive_all(svc: Any, job_ids: list[str], *, reason: str) -> int:
    """Archive every one of these that is evidence. -> how many were.

    The batch entry point, and the only one callers should use: it makes the
    verdict query once for the whole delete rather than once per job, and it is
    where "is this evidence" is decided so a delete path cannot answer that
    question its own way.
    """
    by_job = _verdicts_by_job(svc)
    done = 0
    for job_id in job_ids:
        # Per job, because this is called from inside a delete loop the user
        # asked for: one row that cannot be read is no reason to abandon the
        # reclaim of the other two hundred, and it is certainly no reason to
        # raise out of ``prune_jobs`` half way through a page.
        try:
            job = svc.store.get(job_id)
            if job is None:
                continue
            if job_id not in by_job and not _tagged(job):
                continue
            if archive_job(svc, job_id, reason=reason, verdicts=by_job) is not None:
                done += 1
        except Exception:
            log.exception("could not decide whether to archive job %s", job_id)
    return done


def _bucket(reason: str) -> str:
    """``YYYYMMDD-<reason>`` -- dated first so a listing sorts chronologically,
    which is ``bench/runs``' rule and the reason its names lead with a stamp."""
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in str(reason))[:48]
    return f"{_dt.date.today():%Y%m%d}-{safe.strip('-') or 'delete'}"


def usage(config: Any) -> dict[str, Any]:
    """``{jobs, files, bytes}`` under the archive. Safe on a missing tree.

    What ``doctor`` prints, so a corpus quietly growing to twenty gigabytes is
    something the user finds out about from the app rather than from their disk.
    """
    root = Path(config.evidence_dir)
    jobs = 0
    files = 0
    total = 0
    if not root.is_dir():
        return {"jobs": 0, "files": 0, "bytes": 0}
    for bucket in sorted(root.iterdir()):
        if not bucket.is_dir():
            continue
        for job_dir in sorted(bucket.iterdir()):
            if not job_dir.is_dir():
                continue
            jobs += 1
            for path in job_dir.iterdir():
                try:
                    if path.is_file():
                        files += 1
                        total += path.stat().st_size
                except OSError:
                    continue
    return {"jobs": jobs, "files": files, "bytes": total}
