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


def readiness(family: str, grades: dict[str, Any] | None) -> bool | None:
    values = [(grades or {}).get(name) for name in CRITERIA[family]]
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
                else readiness(family, (grades or {}).get(row["key"]))
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


def report(run_dir: Path, grades_path: Path | None = None) -> dict[str, Any]:
    from .runner import latest_items

    if grades_path is None and (run_dir / "human-grades.json").is_file():
        grades_path = run_dir / "human-grades.json"
    grades = json.loads(grades_path.read_text("utf-8")) if grades_path else {}
    doc = summarise(list(latest_items(run_dir).values()), grades)
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
            mapping[label] = {"run": str(directory.resolve()), "key": key}
            review.append(
                {
                    "id": label,
                    "family": family,
                    "prompt": a[key].get("prompt"),
                    "recipe": a[key].get("character_recipe")
                    if family == "authored_character"
                    else None,
                    "assets": str(dest.relative_to(out)),
                    "acceptance": dict.fromkeys(CRITERIA[family]),
                    "aesthetic_preference": None,
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
        if set(grades) != set(CRITERIA[row["family"]]) or any(
            v is not None and not isinstance(v, bool) for v in grades.values()
        ):
            raise ValueError(f"{row['id']}: acceptance grades must be true, false, or null")
        target = mapping[row["id"]]
        runs.setdefault(target["run"], {})[target["key"]] = {
            **grades,
            "aesthetic_preference": row.get("aesthetic_preference"),
        }
    for directory, grades in runs.items():
        path = Path(directory) / "human-grades.json"
        if path.is_file():
            grades = {**json.loads(path.read_text("utf-8")), **grades}
        path.write_text(json.dumps(grades, indent=2), "utf-8")
        report(Path(directory), path)


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
