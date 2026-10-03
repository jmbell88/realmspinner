"""Offline smoke comparisons using installed models and an isolated asset store.

Run from the checkout: uv run python scripts/benchmark_create_quality.py
This is a one-seed smoke campaign, not a model qualification or human grade.
The full immutable suites use two seeds through `python -m realmspinner.bench`.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
from pathlib import Path

from realmspinner.bench import quality, runner
from realmspinner.config import get_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("build/create-quality"))
    parser.add_argument("--images-only", action="store_true")
    parser.add_argument("--edits-only", action="store_true")
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument(
        "--only-recipe", choices=("sdxl-cfg-raw", "sdxl-cfg-pag-raw", "klein-distilled-raw")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = dataclasses.replace(
        get_config(),
        data_dir=root / "assets",
        db_path=root / "jobs.sqlite",
        bench_dir=root / "bench",
        rank_candidates=False,
        reference_retries=0,
        mesh_retries=0,
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    image_ids = ("image-landscape", "image-interior")
    outputs = {}
    text_recipes = (
        () if args.edits_only else ("sdxl-cfg-raw", "sdxl-cfg-pag-raw", "klein-distilled-raw")
    )
    if args.only_recipe and not args.edits_only:
        parser.error("--only-recipe requires --edits-only")
    if args.edits_only:
        if args.reference_run is None:
            parser.error("--edits-only requires --reference-run")
        outputs["sdxl-cfg-raw"] = args.reference_run.resolve()
    for recipe in text_recipes:
        print(f"Text generation: {recipe}", flush=True)
        outputs[recipe] = runner.run(
            config,
            suite_key="create-v1",
            recipe_key=recipe,
            stage="create",
            started=timestamp,
            ids=image_ids,
            seeds=(42,),
            on_event=lambda s: print(s, flush=True),
        )
        quality.report(outputs[recipe])
    edits = {}
    for recipe in ("sdxl-cfg-raw", "sdxl-cfg-pag-raw", "klein-distilled-raw"):
        if args.only_recipe and recipe != args.only_recipe:
            continue
        if not args.edits_only and recipe != "sdxl-cfg-raw":
            quality.blind_review(
                outputs["sdxl-cfg-raw"], outputs[recipe], root / f"review-{recipe}"
            )
        print(f"Reference editing: {recipe}", flush=True)
        edited = runner.run(
            config,
            suite_key="create-edit-v1",
            recipe_key=recipe,
            stage="create",
            started=timestamp,
            seeds=(42,),
            reference_run=outputs["sdxl-cfg-raw"],
            on_event=lambda s: print(s, flush=True),
        )
        quality.report(edited)
        edits[recipe] = edited
    for recipe in ("sdxl-cfg-pag-raw", "klein-distilled-raw"):
        if "sdxl-cfg-raw" not in edits or recipe not in edits:
            continue
        quality.blind_review(edits["sdxl-cfg-raw"], edits[recipe], root / f"review-edit-{recipe}")
    if args.images_only or args.edits_only:
        return
    all_outputs = runner.run(
        config,
        suite_key="create-v1",
        recipe_key="sdxl-cfg-raw",
        stage="create",
        started=timestamp + "-all",
        seeds=(42,),
        render=True,
        keep_source=True,
        on_event=lambda s: print(s, flush=True),
    )
    quality.report(all_outputs)
    mesh_ids = ("mesh-lantern", "mesh-sword")
    references = runner.run(
        config,
        suite_key="create-v1",
        recipe_key="sdxl-cfg-raw",
        stage="reference",
        started=timestamp + "-reference",
        ids=mesh_ids,
        seeds=(42,),
        on_event=lambda s: print(s, flush=True),
    )
    finishing = []
    for recipe in ("sdxl-preserve-shape", "sdxl-repair"):
        finishing.append(
            runner.run(
                config,
                suite_key="create-v1",
                recipe_key=recipe,
                stage="model",
                started=timestamp,
                ids=mesh_ids,
                seeds=(42,),
                reference_run=references,
                render=True,
                keep_source=True,
                on_event=lambda s: print(s, flush=True),
            )
        )
        quality.report(finishing[-1])
    quality.blind_review(*finishing, root / "review-finishing")
    print(f"Artifacts and ungraded review packets: {root}", flush=True)


if __name__ == "__main__":
    main()
