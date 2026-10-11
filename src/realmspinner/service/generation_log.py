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

__all__ = [
    "base_label",
    "export_jsonl",
    "history",
    "history_page",
    "lora_label",
    "settings_line",
]

#: The images a thumbnail may come from, in order. ``reference.png`` is what
#: trellis saw but only the mesh stage writes it; the 2026-10-11 audit, finding
#: create-10, found every image-only attempt (reference, tile, sprite, material
#: -- most of what History lists) drawing a flat swatch because ``input.png``,
#: the picture such an attempt actually made, was not a source. The pane decodes
#: whichever it is at 128 px off the frame thread, so an enormous upload costs
#: one background decode, not a frame.
_THUMBS = ("reference.png", "input.png")

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


def _candidates(svc: RealmspinnerService) -> list[dict[str, Any]]:
    """Every attempt the log knows, newest first, with only what sorting needs.

    Three statements however many jobs there are. Nothing here rebuilds a
    snapshot or touches the disk: the 2026-10-11 audit, finding create-20,
    found those per-row costs paid for the whole library before the page was
    sliced off the sorted list.
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
        stamps = [r["created_at"] for r in rows] + ([obs["created_at"]] if obs else [])
        created = job["created_at"] if job is not None else (min(stamps) if stamps else 0.0)
        out.append(
            {"job_id": job_id, "job": job, "rows": rows, "obs": obs, "created": created}
        )
    out.sort(key=lambda c: (c["created"], c["job_id"]), reverse=True)
    return out


def _snapshot(cand: dict[str, Any]) -> dict[str, Any] | None:
    """The newest snapshot across every row that carries one; the live job's
    rebuilt one only when none does (a job nobody has judged or measured yet).
    ``settings_snapshot`` is the per-row cost, so callers ask only for rows they
    must filter on, list options from, or show."""
    rows, obs, job = cand["rows"], cand["obs"], cand["job"]
    carriers = [r for r in (*rows, *([obs] if obs else [])) if r.get("snapshot")]
    if carriers:
        return max(carriers, key=lambda r: r.get("created_at") or 0)["snapshot"]
    if job is not None:
        return settings_snapshot(job)
    return None


def _thumbnail(svc: RealmspinnerService, cand: dict[str, Any]) -> str | None:
    """The first of ``_THUMBS`` on disk for a live job, else ``None``."""
    if cand["job"] is None:
        return None
    directory = svc.job_dir(cand["job_id"])
    for name in _THUMBS:
        path = directory / name
        if path.is_file():
            return str(path)
    return None


def _row(
    svc: RealmspinnerService, cand: dict[str, Any], snapshot: dict[str, Any] | None
) -> dict[str, Any]:
    job, rows, obs = cand["job"], cand["rows"], cand["obs"]
    by_stage = {
        stage: _preferred([r for r in rows if r.get("stage") == stage])
        for stage in ("reference", "blank", "model")
    }
    snap = snapshot or {}
    prompt = job.get("prompt") if job is not None else snap.get("prompt_raw")
    return {
        "job_id": cand["job_id"],
        "exists": job is not None,
        "trashed": bool(job is not None and job.get("deleted_at") is not None),
        "created_at": cand["created"],
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
        "thumbnail": _thumbnail(svc, cand),
    }


def history_page(
    svc: RealmspinnerService,
    *,
    generation_type: str | None = None,
    base_model: str | None = None,
    style_lora: str | None = None,
    judged: bool | None = None,
    limit: int = 200,
    offset: int = 0,
    want_options: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(rows, option_rows)``: one page of :func:`history` and, when
    ``want_options``, one ``{base_model, style_lora, generation_type}`` dict per
    attempt in the *whole* log (the filters ignored), from the same pass.

    The pane's filter combos are built from those, because a filtered page
    cannot name the models it filtered out. The 2026-10-11 audit, finding
    create-20, found that taken as a second ``history(limit=1_000_000)`` call:
    each finished generation then cost two whole-library joins, each rebuilding
    a snapshot per unjudged job and stat-ing an image per job, with the store
    lock held for the reads. Here the snapshot is rebuilt only for rows that
    are filtered on or listed in options, the image is stat-ed only for the rows
    returned, and a read that wants no options stops as soon as the page is full.
    """
    cands = _candidates(svc)
    start = max(0, int(offset))
    stop = start + max(0, int(limit))
    by_snapshot = any(v is not None for v in (generation_type, base_model, style_lora))

    if not want_options and not by_snapshot:
        # The cheap path: slice, then do the per-row work for the page alone.
        if judged is not None:
            cands = [c for c in cands if bool(c["rows"]) == judged]
        page = cands[start:stop]
        return [_row(svc, c, _snapshot(c)) for c in page], []

    page_rows: list[dict[str, Any]] = []
    options: list[dict[str, Any]] = []
    matched = 0
    for cand in cands:
        if not want_options and matched >= stop:
            break
        snapshot = _snapshot(cand)
        snap = snapshot or {}
        if want_options:
            options.append(
                {
                    "base_model": snap.get("base_model"),
                    "style_lora": snap.get("style_lora"),
                    "generation_type": snap.get("generation_type"),
                }
            )
        if judged is not None and bool(cand["rows"]) != judged:
            continue
        if generation_type is not None and snap.get("generation_type") != generation_type:
            continue
        if base_model is not None and snap.get("base_model") != base_model:
            continue
        if style_lora is not None and snap.get("style_lora") != style_lora:
            continue
        if start <= matched < stop:
            page_rows.append(_row(svc, cand, snapshot))
        matched += 1
    return page_rows, options


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
    ``thumbnail`` (a path string when ``reference.png``, else ``input.png``, is
    on disk).

    Three statements however many jobs there are.
    """
    return history_page(
        svc,
        generation_type=generation_type,
        base_model=base_model,
        style_lora=style_lora,
        judged=judged,
        limit=limit,
        offset=offset,
    )[0]


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
