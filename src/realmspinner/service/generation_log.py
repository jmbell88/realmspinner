"""The generation log: every text/image job beside what was said about it and
what ran, including jobs that no longer exist.

**Why this is a read model of its own.** Hard delete removes the ``jobs`` row
and keeps the ``verdicts`` and ``observations`` rows, so the history of what the
user judged outlives the assets. Before the ``snapshot`` column a verdict on a
deleted job kept only the config vector -- which omits steps, guidance,
scheduler, seed and the prompt on purpose -- and the 2026-10-11 look at the real
store found 13 of 17 verdicts in exactly that state. This module joins the three
tables in a constant number of statements (never one per row; the store is one
sqlite connection behind a lock and the frame thread may be waiting on it) and
hands a pane plain dicts to draw.

Read-only, and nothing here touches ``vector_key`` or the vector: ``snapshot``
and ``note`` are evidence beside the corpus, not part of its key.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .. import models
from ..core.safeio import atomic as safeio_atomic
from ..vectors import settings_snapshot
from .core import RealmspinnerService
from .errors import Invalid

__all__ = ["base_label", "export_jsonl", "history", "lora_label", "settings_line"]

#: The reference images a thumbnail may come from. ``reference.png`` only: it is
#: what trellis saw, and ``input.png`` is an upload that may be enormous.
_THUMB = "reference.png"

_SEP = " · "


def _base_label(key: Any) -> str:
    """A base model's display name; a retired key falls back to the ledger's
    label (stored rows keep naming models that have since been removed), then to
    the raw key."""
    entry = models.BASE_MODELS.get(str(key))
    if entry is not None:
        return entry.label
    retired = models.RETIRED_MODELS.get(str(key))
    return retired.label if retired is not None else str(key)


def _lora_label(key: Any) -> str:
    entry = models.style_loras_snapshot().get(str(key))
    if entry is not None:
        return entry.label
    retired = models.RETIRED_MODELS.get(str(key))
    return retired.label if retired is not None else str(key)


# The labels a filter combo shows for a model key, public so the History pane
# names a retired model the same way its settings line does.
base_label = _base_label
lora_label = _lora_label


def settings_line(snapshot: dict[str, Any] | None) -> str:
    """One short line of what ran, e.g.
    ``SDXL 1.0 · 30 steps · cfg 7 · 3D render 0.9 · seed 1204285292``.

    Each part is present only if the snapshot recorded it -- an old row says
    less, it does not say ``None``.
    """
    if not isinstance(snapshot, dict):
        return ""
    parts: list[str] = []
    if snapshot.get("base_model"):
        parts.append(_base_label(snapshot["base_model"]))
    if isinstance(snapshot.get("steps"), int):
        parts.append(f"{snapshot['steps']} steps")
    if isinstance(snapshot.get("guidance_scale"), (int, float)):
        parts.append(f"cfg {snapshot['guidance_scale']:g}")
    if snapshot.get("style_lora"):
        lora = _lora_label(snapshot["style_lora"])
        weight = snapshot.get("lora_weight")
        parts.append(
            f"{lora} {weight:g}" if isinstance(weight, (int, float)) else lora
        )
    if isinstance(snapshot.get("seed"), int):
        parts.append(f"seed {snapshot['seed']}")
    return _SEP.join(parts)


def _label(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "verdict": row.get("verdict"),
        "grade": row.get("grade"),
        "reasons": list(row.get("reasons") or []),
        "note": row.get("note"),
        "source": row.get("source"),
        "created_at": row.get("created_at"),
    }


def _preferred(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The row to show among several sources' latest: the human's, else the
    newest. The log is for the person, and an AI judge's opinion must not hide
    theirs."""
    if not rows:
        return None
    human = [r for r in rows if r.get("source") == "human"]
    pool = human or rows
    return max(pool, key=lambda r: r["id"])


def history(
    svc: RealmspinnerService,
    *,
    generation_type: str | None = None,
    base_model: str | None = None,
    style_lora: str | None = None,
    judged: bool | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """One dict per job that has a verdict, an observation, or is a live
    text/image job. Newest first.

    Row keys: ``job_id``, ``exists``, ``trashed``, ``created_at``, ``prompt``,
    ``judged``, ``reference_label`` / ``blank_label`` / ``mesh`` (each
    ``{verdict, grade, reasons, note, source, created_at}`` or ``None``),
    ``observation`` (the newest metrics dict or ``None``), ``snapshot`` (the
    newest recorded, else rebuilt from the live job, else ``None``),
    ``settings_line``, ``base_model``, ``style_lora``, ``generation_type``,
    ``thumbnail`` (a path string only when ``reference.png`` is on disk).

    Three statements however many jobs there are.
    """
    jobs = {j["id"]: j for j in svc.store.history_jobs()}
    verdicts: dict[str, list[dict[str, Any]]] = {}
    for row in svc.store.latest_verdicts():
        verdicts.setdefault(row["job_id"], []).append(row)
    observations = {row["job_id"]: row for row in svc.store.latest_observations()}

    out: list[dict[str, Any]] = []
    for job_id in set(jobs) | set(verdicts) | set(observations):
        job = jobs.get(job_id)
        rows = verdicts.get(job_id, [])
        obs = observations.get(job_id)
        if judged is not None and bool(rows) != judged:
            continue

        by_stage = {
            stage: _preferred([r for r in rows if r.get("stage") == stage])
            for stage in ("reference", "blank", "model")
        }
        # Newest snapshot across every row that carries one; the live job only
        # when none does (a job nobody has judged or measured yet).
        carriers = [r for r in (*rows, *([obs] if obs else [])) if r.get("snapshot")]
        snapshot: dict[str, Any] | None = None
        if carriers:
            snapshot = max(carriers, key=lambda r: r.get("created_at") or 0)["snapshot"]
        elif job is not None:
            snapshot = settings_snapshot(job)

        snap = snapshot or {}
        if generation_type is not None and snap.get("generation_type") != generation_type:
            continue
        if base_model is not None and snap.get("base_model") != base_model:
            continue
        if style_lora is not None and snap.get("style_lora") != style_lora:
            continue

        stamps = [r["created_at"] for r in rows] + ([obs["created_at"]] if obs else [])
        created = job["created_at"] if job is not None else (min(stamps) if stamps else 0.0)
        prompt = job.get("prompt") if job is not None else snap.get("prompt_raw")
        thumb = svc.job_dir(job_id) / _THUMB if job is not None else None
        out.append(
            {
                "job_id": job_id,
                "exists": job is not None,
                "trashed": bool(job is not None and job.get("deleted_at") is not None),
                "created_at": created,
                "prompt": prompt or "",
                "judged": bool(rows),
                "reference_label": _label(by_stage["reference"]),
                "blank_label": _label(by_stage["blank"]),
                "mesh": _label(by_stage["model"]),
                "observation": obs.get("metrics") if obs else None,
                "snapshot": snapshot,
                "settings_line": settings_line(snapshot),
                "base_model": snap.get("base_model"),
                "style_lora": snap.get("style_lora"),
                "generation_type": snap.get("generation_type"),
                "thumbnail": str(thumb) if thumb is not None and thumb.is_file() else None,
            }
        )
    out.sort(key=lambda r: (r["created_at"], r["job_id"]), reverse=True)
    start = max(0, int(offset))
    return out[start : start + max(0, int(limit))]


def export_jsonl(svc: RealmspinnerService, dest_dir: Path) -> Path:
    """Write every verdict and observation row to
    ``realmspinner-generation-log-YYYYMMDD-HHMMSS.jsonl`` in ``dest_dir``.

    One JSON object per line, ``{"table": "verdict" | "observation", ...every
    column}`` with ``vector``, ``reasons``, ``metrics`` and ``snapshot`` parsed
    rather than left as strings, and superseded rows kept (it is a log, not the
    newest view). Staged and renamed, so a full disk never leaves half a file
    under the final name. Safe on a worker thread: the store takes its own lock
    and nothing here touches the frame thread's state.
    """
    dest = Path(dest_dir) if dest_dir else None
    if dest is None or not dest.is_dir():
        raise Invalid("choose an existing folder to save the log in", field="dest")
    verdicts, observations = svc.store.log_rows()
    lines = [json.dumps({"table": "verdict", **r}, default=str) for r in verdicts]
    lines += [json.dumps({"table": "observation", **r}, default=str) for r in observations]
    path = dest / f"realmspinner-generation-log-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    try:
        safeio_atomic.write_text(path, "".join(f"{line}\n" for line in lines))
    except OSError as exc:
        raise Invalid(f"could not write to that folder: {exc}", field="dest") from exc
    return path
