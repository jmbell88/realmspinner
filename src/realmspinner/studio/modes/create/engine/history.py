"""What the History pane decides, headlessly: its filter options, whether a row
can be loaded back into the recipe column, and what that load writes.

The drawing is ``ui/panes/history.py``; the rows are
``service.generation_log.history``. Nothing here draws or touches the store, so
every refusal sentence and every mapping from a recorded snapshot to a form is
assertable without a GL context.

**A row may outlive its job.** A hard delete removes the ``jobs`` row and keeps
the verdicts, so a History row can be a prompt and a settings line with no asset
behind it. "Load settings" therefore works from the *snapshot*, not from a job:
``snapshot_job`` dresses one as the minimal job row ``library.copy_settings``
reads, so a deleted attempt and a live one are restored by the one path.
"""

from __future__ import annotations

from typing import Any

from ..... import models
from .....service import generation_log
from . import assets as create_assets

#: Rows per page: what "More" adds. Small, because a row carries a thumbnail.
PAGE = 50

#: The four filters a pane keeps, with their "no filter" value.
NO_FILTERS: dict[str, Any] = {
    "base_model": None,
    "style_lora": None,
    "generation_type": None,
    "judged": None,
}

#: ``judged`` choices, in the order the combo offers them.
JUDGED_OPTIONS: tuple[tuple[str, str], ...] = (
    ("all", "All"),
    ("judged", "Judged"),
    ("unjudged", "Unjudged"),
)


def judged_value(key: str) -> bool | None:
    """The service's ``judged=`` argument for a combo key."""
    return {"judged": True, "unjudged": False}.get(key)


def judged_key(value: bool | None) -> str:
    return {True: "judged", False: "unjudged"}.get(value, "all")


def service_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """``filters`` as keyword arguments for ``generation_log.history``."""
    return {key: filters.get(key) for key in NO_FILTERS}


def options_from(rows: list[dict[str, Any]]) -> dict[str, list[tuple[str, str]]]:
    """Filter options derived from the rows themselves. **Pure.**

    Built from what the log actually holds rather than from the registry, so a
    model that has since been retired (a stored row keeps naming it) is still an
    option -- the filter that would find the old work must not disappear with
    the model. Each list is sorted by label; a row that recorded none of a field
    contributes nothing.
    """
    seen: dict[str, dict[str, str]] = {"base_model": {}, "style_lora": {}, "generation_type": {}}
    types = dict(create_assets.ASSET_TYPE_OPTIONS)
    for row in rows:
        if row.get("base_model"):
            key = str(row["base_model"])
            seen["base_model"].setdefault(key, generation_log.base_label(key))
        if row.get("style_lora"):
            key = str(row["style_lora"])
            seen["style_lora"].setdefault(key, generation_log.lora_label(key))
        if row.get("generation_type"):
            key = str(row["generation_type"])
            seen["generation_type"].setdefault(key, types.get(key, key))
    return {
        field: sorted(found.items(), key=lambda item: item[1].lower())
        for field, found in seen.items()
    }


def merge_options(
    old: dict[str, list[tuple[str, str]]], new: dict[str, list[tuple[str, str]]]
) -> dict[str, list[tuple[str, str]]]:
    """The union of two option sets, so a filtered page never narrows the combo
    that filtered it."""
    out: dict[str, list[tuple[str, str]]] = {}
    for field in {*old, *new}:
        merged = dict(old.get(field, ()))
        merged.update(dict(new.get(field, ())))
        out[field] = sorted(merged.items(), key=lambda item: item[1].lower())
    return out


def jobs_signature(jobs: list[dict[str, Any]]) -> tuple[int, int, int]:
    """``(rows, terminal, trashed)`` of the jobs cache. **Pure.**

    What the pane compares to notice that a job finished or was deleted without
    reloading on every poll: the cache's generation counter moves on every
    refresh, a live job's progress included, and a reload per tick would be a
    full join per second while a press runs.
    """
    terminal = sum(1 for j in jobs if j.get("status") in ("done", "error", "cancelled"))
    trashed = sum(1 for j in jobs if j.get("deleted_at") is not None)
    return len(jobs), terminal, trashed


# --- Load settings ----------------------------------------------------------------


def retired_reason(snapshot: dict[str, Any] | None) -> str:
    """The refusal sentence when a snapshot names a removed model or style, else ""."""
    snap = snapshot or {}
    for field in ("base_model", "style_lora"):
        entry = models.retired_choice(field, snap.get(field))
        if entry is not None:
            # Said again here rather than only the ledger's own line: the reader
            # pressed Load settings on a row, and "the settings can't be loaded"
            # is what has to come first.
            return f"Can't load these settings. {models.retired_message(entry)}"
    return ""


def no_settings_reason(row: dict[str, Any]) -> str:
    """Why Load settings is greyed for this row, or "". **Pure.**

    Only one case: the job is gone *and* its verdict predates the snapshot
    column, so nothing records what ran. A retired model is not this -- that
    button stays live and is refused with a sentence on the press
    (:func:`load_block`), because a greyed button cannot say which model went.
    """
    if not row.get("exists") and not row.get("snapshot"):
        return "This attempt was deleted before its settings were kept."
    return ""


def load_block(row: dict[str, Any]) -> str:
    """Why pressing Load settings on this row is refused, or "" when it works. **Pure.**

    The greyed case first, then a retired model: that is a refusal even for a
    job that still exists, because loading would put a key into the form that
    every door refuses.
    """
    return no_settings_reason(row) or retired_reason(row.get("snapshot"))


def snapshot_job(row: dict[str, Any]) -> dict[str, Any]:
    """The minimal job row ``library.copy_settings`` needs, built from a snapshot. **Pure.**

    Only the recipe keys the 2D form restores (``form_from_params`` is the
    allowlist), and the seed under ``seed`` so ``recorded_seed`` returns it and
    the loaded form locks it.
    """
    snap = row.get("snapshot") or {}
    params: dict[str, Any] = {}
    for key in ("base_model", "style_lora", "lora_weight", "negative_prompt", "seed"):
        if snap.get(key) is not None and snap.get(key) != "":
            params[key] = snap[key]
    kind = snap.get("generation_type")
    if kind:
        params["asset_type"] = params["generation_type"] = kind
    prompt = snap.get("prompt_raw") or row.get("prompt") or ""
    if prompt:
        params["prompt"] = prompt
    return {
        "id": row.get("job_id"),
        "kind": "text",
        "stage": "reference",
        "prompt": prompt,
        "params": params,
        "status": "done",
    }


#: The snapshot's top-level mesh keys, and the ``form_3d`` key each one fills.
_MESH_TOP = {
    "lowpoly_triangles": "lowpoly_triangles",
    "profile": "profile",
    "bg_removal": "bg_removal",
    "mesh_finishing": "mesh_finishing",
}
#: The snapshot's ``trellis`` block, likewise.
_MESH_TRELLIS = {
    "seed": "mesh_seed",
    "band": "trellis_band",
    "tex_res": "trellis_tex_res",
    "size_m": "size_m",
}


def mesh_updates(snapshot: dict[str, Any] | None, defaults: dict[str, Any]) -> dict[str, Any]:
    """The ``form_3d`` entries a snapshot carries, coerced like their defaults. **Pure.**

    ``defaults`` is ``DEFAULT_FORM_3D``: it is both the allowlist and the type a
    value is coerced to, so a snapshot that crossed a disk with a junk value
    costs that one key. An image-only snapshot carries none and returns ``{}``.
    """
    snap = snapshot or {}
    found: dict[str, Any] = {}
    for src, dst in _MESH_TOP.items():
        if snap.get(src) is not None:
            found[dst] = snap[src]
    trellis = snap.get("trellis")
    if isinstance(trellis, dict):
        for src, dst in _MESH_TRELLIS.items():
            if trellis.get(src) is not None:
                found[dst] = trellis[src]
    out: dict[str, Any] = {}
    for key, value in found.items():
        default = defaults.get(key)
        if key not in defaults:
            continue
        try:
            if isinstance(default, bool):
                out[key] = bool(value)
            elif isinstance(default, int):
                out[key] = int(value)
            elif isinstance(default, float):
                out[key] = float(value)
            elif isinstance(default, str):
                out[key] = str(value)
            elif default is None:
                # ``mesh_seed``: unset by default, an int when set.
                out[key] = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
    return out


# --- row text ---------------------------------------------------------------------


def grade_text(label: dict[str, Any] | None) -> str:
    """``+5`` / ``-3`` / ``0`` for a mesh label, "" when there is none. **Pure.**"""
    if not label or not isinstance(label.get("grade"), int):
        return ""
    grade = int(label["grade"])
    return "0" if grade == 0 else f"{grade:+d}"


def image_mark(label: dict[str, Any] | None) -> str:
    """``Good`` / ``Bad`` for an image label, "" when unlabelled. **Pure.**"""
    if not label:
        return ""
    return {"accept": "Good", "reject": "Bad"}.get(str(label.get("verdict")), "")


def row_tags(row: dict[str, Any]) -> list[str]:
    """Every tag on a row's labels, once each, in first-seen order. **Pure.**"""
    out: list[str] = []
    for key in ("reference_label", "blank_label", "mesh"):
        for tag in (row.get(key) or {}).get("reasons") or ():
            if tag not in out:
                out.append(str(tag))
    return out


def row_notes(row: dict[str, Any]) -> list[str]:
    """The notes on a row's labels, newest first. **Pure.**"""
    labels = [row.get(key) for key in ("reference_label", "blank_label", "mesh")]
    labels = [lb for lb in labels if lb and lb.get("note")]
    labels.sort(key=lambda lb: lb.get("created_at") or 0, reverse=True)
    return [str(lb["note"]) for lb in labels]


def gone_text(row: dict[str, Any]) -> str:
    """``deleted`` / ``in trash`` / "" for a row's asset. **Pure.**"""
    if not row.get("exists"):
        return "deleted"
    return "in trash" if row.get("trashed") else ""
