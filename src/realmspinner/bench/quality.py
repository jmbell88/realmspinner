"""Output-family readiness and blind paired review, independent of aesthetics.

No metric or missing human grade is a substitute for acceptance. Reports give
each family equal weight; unknowns remain unknown rather than passing by default.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import statistics
import threading
from pathlib import Path
from typing import Any

CRITERIA = {
    "image": ("prompt_fidelity", "composition", "visual_defects", "reference_consistency"),
    "3d_model": ("multi_view_shape", "thin_parts", "openings", "pbr_appearance", "triangle_budget"),
    "seamless_material": ("repeated_pattern", "export_seams", "detail_retention"),
    "tileset": ("cell_alignment", "terrain_adjacency", "palette_consistency", "tile_readability"),
    "sprite_sheet": ("identity", "pose_direction", "registration", "palette", "pixel_grid"),
    "authored_character": ("recipe_fidelity", "deformation", "rig_readiness", "sheet_consistency"),
}

# Rubrics are *additional* criteria sets a suite item opts into by name. CRITERIA
# is left alone so every create-v1 report and packet reads exactly as it did.
RUBRICS = {
    "3d_v2": (
        "silhouette",
        "reference_fidelity",
        "unseen_surfaces",
        "texture",
        "finishing_damage",
        "animation_suitability",
    ),
}
# animation_suitability is only a question for things that will be rigged; asking
# it of a crate would make every static prop "unknown" forever.
ANIMATABLE = frozenset({"humanoid", "creature"})
_ANIMATION = "animation_suitability"
ATTEMPT_LIMIT = 3


def criteria_for(
    family: str, rubric: str | None = None, category: str | None = None
) -> tuple[str, ...]:
    if not rubric:
        return CRITERIA[family]
    names = RUBRICS[rubric]
    if category in ANIMATABLE:
        return names
    return tuple(n for n in names if n != _ANIMATION)


def readiness(
    family: str,
    grades: dict[str, Any] | None,
    rubric: str | None = None,
    category: str | None = None,
) -> bool | None:
    values = [(grades or {}).get(name) for name in criteria_for(family, rubric, category)]
    if any(value is False for value in values):
        return False
    return True if all(value is True for value in values) else None


def summarise(
    records: list[dict[str, Any]], grades: dict[str, Any] | None = None
) -> dict[str, Any]:
    families = {}
    for family in CRITERIA:
        rows = [r for r in records if r.get("output_family", "3d_model") == family]
        usable = []
        for row in rows:
            usable.append(
                False
                if row.get("status") != "done"
                else readiness(
                    family,
                    (grades or {}).get(row["key"]),
                    row.get("rubric"),
                    row.get("category"),
                )
            )
        known = [v for v in usable if v is not None]
        families[family] = {
            "outputs": len(rows),
            "failures": sum(r.get("status") != "done" for r in rows),
            "accepted": sum(v is True for v in usable),
            "unknown": sum(v is None for v in usable),
            "usable_output_rate": sum(known) / len(rows)
            if rows and len(known) == len(rows)
            else None,
            "mean_seconds": statistics.mean(
                r["seconds"] for r in rows if r.get("seconds") is not None
            )
            if any(r.get("seconds") is not None for r in rows)
            else None,
        }
    rates = [f["usable_output_rate"] for f in families.values()]
    return {
        "families": families,
        "equal_family_usable_rate": (
            statistics.mean(rates) if all(r is not None for r in rates) else None
        ),
        "aesthetic_preference": "reported separately; excluded from readiness",
    }


def _accepted(record: dict[str, Any], grades: dict[str, Any] | None) -> bool | None:
    if record.get("status") != "done":
        return False
    return readiness(
        record.get("output_family", "3d_model"),
        (grades or {}).get(record["key"]),
        record.get("rubric"),
        record.get("category"),
    )


def _repair_minutes(grades: dict[str, Any] | None, key: str) -> float | None:
    value = ((grades or {}).get(key) or {}).get("repair_minutes")
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        return None
    return float(value)


def _item_attempts(
    attempts: list[dict[str, Any]], grades: dict[str, Any] | None
) -> dict[str, Any]:
    """One item's attempts, in attempt order, reduced to what is known.

    Three-valued throughout, like ``readiness``: a verdict nobody has given is
    None, and a None is never counted as a pass or as a failure.
    """
    attempts = sorted(attempts, key=lambda r: r["attempt"])
    verdicts = [_accepted(r, grades) for r in attempts]
    first = verdicts[0] if attempts[0]["attempt"] == 1 else None
    window = [v for r, v in zip(attempts, verdicts, strict=True) if r["attempt"] <= ATTEMPT_LIMIT]
    if any(v is True for v in window):
        best: bool | None = True
    else:
        best = None if any(v is None for v in window) else False

    seconds: float | None = None
    if True in verdicts:
        # Everything spent up to and including the first accepted attempt; one
        # unrecorded duration along the way makes the total unknown.
        upto = verdicts.index(True) + 1
        spent = [r.get("seconds") for r in attempts[:upto]]
        repair = _repair_minutes(grades, attempts[upto - 1]["key"])
        if all(v is not None for v in spent) and repair is not None:
            seconds = sum(spent) + repair * 60.0
    return {"first": first, "best": best, "time_to_accepted": seconds}


def _rate(values: list[bool | None]) -> float | None:
    if not values or None in values:
        return None
    return sum(v is True for v in values) / len(values)


def _group_metrics(records: list[dict[str, Any]], grades: dict[str, Any] | None) -> dict[str, Any]:
    by_item: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("attempt") is not None:
            by_item.setdefault(record["item"], []).append(record)
    items = [_item_attempts(rows, grades) for rows in by_item.values()]
    times = sorted(i["time_to_accepted"] for i in items if i["time_to_accepted"] is not None)
    seconds = [r["seconds"] for r in records if r.get("seconds") is not None]
    return {
        "items": len(items),
        "attempts": sum(len(rows) for rows in by_item.values()),
        "first_attempt": _rate([i["first"] for i in items]),
        "best_of_3": _rate([i["best"] for i in items]),
        "generation_seconds_total": sum(seconds) if seconds else None,
        "time_to_accepted_seconds": statistics.median(times) if times else None,
        "time_to_accepted_known": len(times),
    }


def attempt_metrics(
    records: list[dict[str, Any]], grades: dict[str, Any] | None = None
) -> dict[str, Any]:
    """First-attempt, best-of-3 and time-to-accepted, overall and split by input
    mode and by category.

    An attempt is one seed slot of an item (``record["attempt"]``, 1-based).
    ``time_to_accepted_seconds`` is the *median* over the items that reached an
    accepted attempt and whose every duration and repair time is known:
    generation seconds through the first accepted attempt plus its repair
    minutes. Items that never reached acceptance, or whose cost is unknown, are
    counted in ``items`` but never in the median.
    """
    out: dict[str, Any] = {"overall": _group_metrics(records, grades)}
    for label, field in (("by_input_mode", "input_mode"), ("by_category", "category")):
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            groups.setdefault(record.get(field) or "text", []).append(record)
        out[label] = {name: _group_metrics(rows, grades) for name, rows in sorted(groups.items())}
    return out


def report(run_dir: Path, grades_path: Path | None = None) -> dict[str, Any]:
    from .runner import latest_items

    if grades_path is None and (run_dir / "human-grades.json").is_file():
        grades_path = run_dir / "human-grades.json"
    grades = json.loads(grades_path.read_text("utf-8")) if grades_path else {}
    records = list(latest_items(run_dir).values())
    doc = summarise(records, grades)
    # Only for runs that recorded attempts: a create-v1 report stays byte-for-byte
    # what it was, and an old run does not grow a block of Nones.
    if any(r.get("attempt") is not None for r in records):
        doc["attempts"] = attempt_metrics(records, grades)
    (run_dir / "quality.json").write_text(json.dumps(doc, indent=2), "utf-8")
    return doc


def blind_review(left: Path, right: Path, out: Path, seed: int = 42) -> Path:
    """Copy paired artifacts to anonymous slots and emit an ungraded rubric.

    The private mapping belongs to the review coordinator. Reviewers receive
    only review.json and assets/. Cases require the same item, seed, and family;
    model/finishing comparisons also require identical reference image bytes.
    """
    from .runner import latest_items

    if out.exists():
        raise ValueError("review output already exists; choose a fresh directory")
    a, b = latest_items(left), latest_items(right)
    pairs = sorted(set(a) & set(b))
    if not pairs:
        raise ValueError("no paired units")
    for key in pairs:
        if (a[key].get("item"), a[key].get("seed"), a[key].get("output_family")) != (
            b[key].get("item"),
            b[key].get("seed"),
            b[key].get("output_family"),
        ):
            raise ValueError(f"{key}: item, seed, or family differs")
        if a[key].get("output_family") == "3d_model":
            hashes = [record.get("reference_sha256") for record in (a[key], b[key])]
            if not hashes[0] or hashes[0] != hashes[1]:
                raise ValueError(f"{key}: mesh comparisons require the same reference image")
        for field in ("conditioning_sha256", "source_sha256"):
            hashes = [record.get(field) for record in (a[key], b[key])]
            if any(hashes) and (not hashes[0] or hashes[0] != hashes[1]):
                raise ValueError(f"{key}: {field} differs")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    out.mkdir(parents=True)
    review, mapping = [], {}
    for index, key in enumerate(pairs, 1):
        family = a[key].get("output_family", "3d_model")
        rubric, category = a[key].get("rubric"), a[key].get("category")
        criteria = criteria_for(family, rubric, category)
        order = [left, right]
        rng.shuffle(order)
        for slot, directory in zip(("A", "B"), order, strict=True):
            label = f"case-{index:03}-{slot}"
            src, dest = directory / "items" / key, out / "assets" / label
            dest.mkdir(parents=True)
            for path in src.iterdir():
                # Provenance exposes the backend; keep it out of blind packets.
                if path.suffix.lower() in (".png", ".glb"):
                    shutil.copyfile(path, dest / path.name)
                elif path.name in ("views", "sheets") and path.is_dir():
                    for image in path.rglob("*.png"):
                        target = dest / path.name / image.relative_to(path)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(image, target)
            mapping[label] = {
                "run": str(directory.resolve()),
                "key": key,
                "criteria": list(criteria),
            }
            review.append(
                {
                    "id": label,
                    "family": family,
                    "prompt": a[key].get("prompt"),
                    "recipe": a[key].get("character_recipe")
                    if family == "authored_character"
                    else None,
                    "assets": str(dest.relative_to(out)),
                    "acceptance": dict.fromkeys(criteria),
                    "aesthetic_preference": None,
                    # Only rubric packets carry these; a create-v1 packet is
                    # unchanged. repair_minutes: hand-fix time to make it usable.
                    **({"rubric": rubric, "repair_minutes": None} if rubric else {}),
                }
            )
    (out / "review.json").write_text(json.dumps(review, indent=2), "utf-8")
    (out / "mapping.private.json").write_text(json.dumps(mapping, indent=2), "utf-8")
    return out


def import_review(review_dir: Path) -> None:
    mapping = json.loads((review_dir / "mapping.private.json").read_text("utf-8"))
    review = json.loads((review_dir / "review.json").read_text("utf-8"))
    runs: dict[str, dict[str, Any]] = {}
    for row in review:
        grades = row["acceptance"]
        target = mapping[row["id"]]
        # The mapping knows which criteria this case was issued with; a mapping
        # from before rubrics has none and falls back to the family's.
        expected = set(target.get("criteria") or CRITERIA[row["family"]])
        if set(grades) != expected or any(
            v is not None and not isinstance(v, bool) for v in grades.values()
        ):
            raise ValueError(f"{row['id']}: acceptance grades must be true, false, or null")
        repair = row.get("repair_minutes")
        if repair is not None and (
            isinstance(repair, bool) or not isinstance(repair, int | float) or repair < 0
        ):
            raise ValueError(f"{row['id']}: repair_minutes must be a number of minutes or null")
        runs.setdefault(target["run"], {})[target["key"]] = {
            **grades,
            "aesthetic_preference": row.get("aesthetic_preference"),
            **({"repair_minutes": repair} if "repair_minutes" in row else {}),
        }
    for directory, grades in runs.items():
        path = Path(directory) / "human-grades.json"
        if path.is_file():
            grades = {**json.loads(path.read_text("utf-8")), **grades}
        path.write_text(json.dumps(grades, indent=2), "utf-8")
        report(Path(directory), path)


# A GLB a service hands back is a few MiB to a few hundred; past this it is not
# a game asset and the copy would only fill the disk.
EXTERNAL_MAX_BYTES = 512 * 1024 * 1024
_GLB_MAGIC = b"glTF"


def _external_glb(path: Path) -> None:
    size = path.stat().st_size
    if size > EXTERNAL_MAX_BYTES:
        raise ValueError(f"{path.name}: {size} bytes is over the {EXTERNAL_MAX_BYTES} byte cap")
    with path.open("rb") as fh:
        if fh.read(4) != _GLB_MAGIC:
            raise ValueError(f"{path.name}: not a binary glTF (missing 'glTF' magic)")


def import_external(
    reference_run: Path, glb_dir: Path, out: Path, label: str
) -> dict[str, Any]:
    """File another service's GLBs as a run that blind_review can pair.

    ``glb_dir`` holds ``<item>/attempt-<n>.glb``, the files a person downloaded
    from Tripo or Meshy by hand; an optional ``timings.json`` maps
    ``"<item>/attempt-<n>.glb"`` to the seconds that attempt took. Each file
    becomes the unit of the same item and attempt in ``reference_run`` and
    shares its ``input.png``, so the pairing rule that mesh comparisons need
    the same reference image holds by construction. Nothing here touches the
    network: this reads local files only.

    Attempts the service never produced are simply absent, and are returned
    under ``missing`` so the caller can say so -- they are not recorded as
    failures, because a person choosing not to download a third try is not a
    generation failure.
    """
    from .runner import append_item, latest_items

    if out.exists():
        raise ValueError("import output already exists; choose a fresh directory")
    if not glb_dir.is_dir():
        raise ValueError(f"{glb_dir}: not a directory")
    label = label.strip()
    if not label:
        raise ValueError("a label is required")
    records = latest_items(reference_run)
    if not records:
        raise ValueError(f"{reference_run}: no recorded units")
    timings: dict[str, Any] = {}
    if (glb_dir / "timings.json").is_file():
        timings = json.loads((glb_dir / "timings.json").read_text("utf-8"))

    imported, missing, used = [], [], set()
    out.mkdir(parents=True)
    for key, record in sorted(records.items()):
        attempt = record.get("attempt")
        if attempt is None or record.get("status") != "done":
            continue
        rel = f"{record['item']}/attempt-{attempt}.glb"
        source = glb_dir / record["item"] / f"attempt-{attempt}.glb"
        reference = reference_run / "items" / key / "input.png"
        if not source.is_file():
            missing.append(rel)
            continue
        if not reference.is_file():
            raise ValueError(f"{reference}: the reference run has no input.png for {key}")
        _external_glb(source)
        item_dir = out / "items" / key
        item_dir.mkdir(parents=True)
        shutil.copyfile(reference, item_dir / "input.png")
        shutil.copyfile(source, item_dir / "model.glb")
        seconds = timings.get(rel)
        used.add(rel)
        append_item(
            out,
            {
                "key": key,
                "item": record["item"],
                "category": record.get("category"),
                "seed": record["seed"],
                "attempt": attempt,
                "status": "done",
                "seconds": float(seconds) if isinstance(seconds, int | float) else None,
                "output_family": record.get("output_family", "3d_model"),
                "prompt": record.get("prompt"),
                "input_mode": record.get("input_mode"),
                "rubric": record.get("rubric"),
                "backend": label,
                "reference_sha256": reference_hash(item_dir / "input.png"),
            },
        )
        imported.append(rel)
    if not imported:
        shutil.rmtree(out, ignore_errors=True)
        raise ValueError(f"{glb_dir}: no <item>/attempt-<n>.glb matched the reference run")
    extra = sorted(
        p.relative_to(glb_dir).as_posix()
        for p in glb_dir.rglob("*.glb")
        if p.relative_to(glb_dir).as_posix() not in used
    )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "imported": True,
                "label": label,
                "reference_run": str(reference_run.resolve()),
            },
            indent=2,
        ),
        "utf-8",
    )
    return {"imported": imported, "missing": missing, "unmatched": extra}


def reference_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MemorySampler:
    """Device-wide used VRAM and host commit peaks; unavailable means unknown.

    These are machine totals, including other processes, not attributed model
    allocations. Sampling every 250 ms can miss shorter peaks.
    """

    def __init__(self):
        self.event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.gpu: float | None = None
        self.host: float | None = None

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.event.set()
        self.thread.join(timeout=2)

    def _sample(self) -> None:
        from .. import memlog, vram

        while True:
            try:
                gpu = vram.live_memory()
                host = memlog.system_memory()
                if gpu is not None:
                    self.gpu = max(self.gpu or 0.0, gpu.total_gib - gpu.free_gib)
                if host is not None:
                    self.host = max(self.host or 0.0, host.commit_total)
            except Exception:
                pass  # Diagnostic failure cannot fail a generation.
            if self.event.wait(0.25):
                return

    def report(self) -> dict[str, Any]:
        return {
            "peak_device_used_gib": self.gpu,
            "peak_host_commit_gib": self.host,
            "scope": "machine totals, includes other processes",
            "sample_seconds": 0.25,
        }
