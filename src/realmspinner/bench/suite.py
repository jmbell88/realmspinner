"""Loading and validating a benchmark suite. Pure; no torch, no GPU, no DB.

A suite is a file, not code -- the same rule ``templates/*.json`` follows.
Adding a suite means adding a file, and a suite file is *never* edited in
place: a run's manifest records its fingerprint, so changing a prompt after
the fact would silently invalidate every comparison made against it. Change
means core-v2.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

SUITE_DIR = Path(__file__).resolve().parent / "suites"
DEFAULT_SUITE = "core-v2"

# The categories a suite may use. Enforced at load: ``--filter`` matches on
# this string, so a typo'd category in a suite file selects zero items and
# reports nothing wrong -- a run that silently does nothing, which is the one
# failure mode worse than a crash.
CATEGORIES = (
    "prop", "weapon", "character", "vehicle", "environment",
    # create-3d-v1's six: what a 3D generator is judged on, not what a game
    # calls the thing. Added, never renamed -- core-v2's ids are frozen.
    "solid_prop", "thin_weapon", "open_structure", "hard_surface", "humanoid", "creature",
)
INPUT_MODES = ("text", "image")


@dataclass(frozen=True, slots=True)
class Item:
    id: str
    category: str
    prompt: str
    guidance: dict[str, Any]
    tags: tuple[str, ...] = ()
    notes: str = ""
    output_family: str = "3d_model"
    settings: dict[str, Any] = field(default_factory=dict)
    # ``image`` items start from a still the maintainer supplies, found under a
    # references root at ``reference_image`` (never inside the repo). ``rubric``
    # names an entry in ``quality.RUBRICS``; None keeps the family's CRITERIA.
    input_mode: str = "text"
    reference_image: str | None = None
    rubric: str | None = None


@dataclass(frozen=True, slots=True)
class Suite:
    key: str
    label: str
    seeds: tuple[int, ...]
    items: tuple[Item, ...]
    path: Path
    notes: str = ""

    def by_category(self) -> dict[str, list[Item]]:
        out: dict[str, list[Item]] = {}
        for item in self.items:
            out.setdefault(item.category, []).append(item)
        return out

    def filter(self, *, categories: tuple[str, ...] = (), ids: tuple[str, ...] = ()) -> list[Item]:
        return [
            i
            for i in self.items
            if (not categories or i.category in categories) and (not ids or i.id in ids)
        ]


def available() -> list[str]:
    return sorted(p.stem for p in SUITE_DIR.glob("*.json"))


def load(key: str = DEFAULT_SUITE) -> Suite:
    """Load and validate a suite by key, or raise ValueError."""
    path = SUITE_DIR / f"{key}.json"
    if not path.exists():
        raise ValueError(f"unknown suite {key!r}; available: {available()}")
    raw = json.loads(path.read_text("utf-8"))
    return parse(raw, path)


def parse(raw: dict[str, Any], path: Path) -> Suite:
    """Validate a suite payload. Every guidance key must be one guidance.py
    actually accepts -- a typo would otherwise fail 160 submits one at a time,
    two hours into a run."""
    from .. import guidance
    from .quality import CRITERIA, RUBRICS

    known = set(guidance.form_fields())
    seeds = tuple(int(s) for s in raw.get("seeds") or ())
    if not seeds:
        raise ValueError(f"{path.name}: a suite needs at least one seed")

    items: list[Item] = []
    seen: set[str] = set()
    for entry in raw.get("items") or ():
        item_id = str(entry.get("id") or "")
        if not item_id:
            raise ValueError(f"{path.name}: an item has no id")
        if item_id in seen:
            raise ValueError(f"{path.name}: duplicate item id {item_id!r}")
        seen.add(item_id)
        if not str(entry.get("prompt") or "").strip():
            raise ValueError(f"{path.name}: item {item_id} has no prompt")
        category = str(entry.get("category") or "")
        if category not in CATEGORIES:
            raise ValueError(
                f"{path.name}: item {item_id} has category {category!r}; "
                f"expected one of {list(CATEGORIES)}"
            )
        fields = dict(entry.get("guidance") or {})
        family = str(entry.get("output_family") or "3d_model")
        if family not in CRITERIA:
            raise ValueError(f"{path.name}: item {item_id} has unknown output family {family!r}")
        unknown = sorted(set(fields) - known)
        if unknown:
            raise ValueError(f"{path.name}: item {item_id} names unknown guidance {unknown}")
        # And the *values*, the same way recipe.parse does. Checking only the
        # keys is what this docstring's promise could not keep: "material":
        # "stonee" names a real field, loads without complaint, and fails at
        # submit -- one item at a time, two hours into a run.
        try:
            guidance.normalize(fields)
        except ValueError as exc:
            raise ValueError(f"{path.name}: item {item_id}: {exc}") from exc
        input_mode = str(entry.get("input") or "text")
        if input_mode not in INPUT_MODES:
            raise ValueError(
                f"{path.name}: item {item_id} has input {input_mode!r}; "
                f"expected one of {list(INPUT_MODES)}"
            )
        reference_image = entry.get("reference_image") or None
        if input_mode == "image":
            reference_image = _reference_path(path, item_id, reference_image)
        elif reference_image is not None:
            raise ValueError(f"{path.name}: text item {item_id} names a reference_image")
        rubric = entry.get("rubric") or None
        if rubric is not None and rubric not in RUBRICS:
            raise ValueError(
                f"{path.name}: item {item_id} has unknown rubric {rubric!r}; "
                f"expected one of {sorted(RUBRICS)}"
            )
        items.append(
            Item(
                id=item_id,
                category=category,
                prompt=str(entry["prompt"]),
                guidance=fields,
                tags=tuple(str(t) for t in entry.get("tags") or ()),
                notes=str(entry.get("notes") or ""),
                output_family=family,
                settings=dict(entry.get("settings") or {}),
                input_mode=input_mode,
                reference_image=reference_image,
                rubric=rubric,
            )
        )
    if not items:
        raise ValueError(f"{path.name}: a suite needs at least one item")
    return Suite(
        key=str(raw.get("key") or path.stem),
        label=str(raw.get("label") or path.stem),
        seeds=seeds,
        items=tuple(items),
        path=path,
        notes=str(raw.get("notes") or ""),
    )


def _reference_path(path: Path, item_id: str, value: Any) -> str:
    """A reference is relative to a root the *runner* chooses, so a suite can
    neither point outside it nor depend on one machine's drive layout."""
    text = str(value or "").replace("\\", "/")
    parts = PurePosixPath(text).parts
    if not text or PurePosixPath(text).is_absolute() or ":" in parts[0] or ".." in parts:
        raise ValueError(
            f"{path.name}: image item {item_id} needs a relative reference_image, got {value!r}"
        )
    return PurePosixPath(*parts).as_posix()
