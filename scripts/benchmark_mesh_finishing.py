"""Compare both finishing methods on identical saved reconstructions, offline."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from realmspinner import provenance, tiercheck
from realmspinner.bench import quality, runner, views
from realmspinner.kernels.rig import blender_spec
from realmspinner.pipelines import blender_run, remesh, retexture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--triangles", type=int, default=5000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; choose a fresh directory")
    if not remesh.TRIANGLES_MIN <= args.triangles <= remesh.TRIANGLES_MAX:
        parser.error("triangle budget outside supported range")
    records = runner.latest_items(args.source_run)
    sources = {
        key: args.source_run / "items" / key / "source.glb"
        for key, row in records.items()
        if row.get("output_family") == "3d_model"
    }
    sources = {key: path for key, path in sources.items() if path.is_file()}
    if not sources:
        parser.error("source run has no retained mesh reconstructions")
    for method in ("preserve_shape", "repair"):
        directory = args.out / method
        directory.mkdir(parents=True)
        manifest = {
            "stage": "finishing",
            "method": method,
            "triangles": args.triangles,
            "source_sha256": {key: quality.reference_hash(path) for key, path in sources.items()},
            "versions": provenance.versions(),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
        for key, source in sources.items():
            print(f"{method}: {key}", flush=True)
            dest = directory / "items" / key
            dest.mkdir(parents=True)
            shutil.copyfile(source, dest / "source.glb")
            reference = source.with_name("input.png")
            shutil.copyfile(reference, dest / "input.png")
            row = {**records[key], "followups": [], "job": None, "error": None}
            row["reference_sha256"] = quality.reference_hash(reference)
            row["source_sha256"] = manifest["source_sha256"][key]
            sampler = quality.MemorySampler()
            sampler.start()
            started = time.monotonic()
            try:
                result = blender_run.run_worker(
                    blender_spec.remesh_spec(
                        source,
                        dest / "model.glb",
                        dest,
                        target_faces=remesh.target_faces(args.triangles),
                        texture_size=retexture.atlas_size(source) or remesh.DEFAULT_TEXTURE_PX,
                        preserve_shape=method == "preserve_shape",
                        close_holes=method == "repair",
                    )
                )
                verdict = tiercheck.compare(
                    tiercheck.survey(source), tiercheck.survey(dest / "model.glb")
                )
                row["status"] = "done"
                row["measurements"] = {
                    "lowpoly": {
                        **result,
                        "requested": args.triangles,
                        "geometry": remesh.compare_geometry(source, dest / "model.glb"),
                        "tiercheck": {"ok": verdict.ok, "failures": list(verdict.failures)},
                    }
                }
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            finally:
                sampler.stop()
            row["seconds"] = round(time.monotonic() - started, 2)
            row["memory"] = sampler.report()
            if row["status"] == "done":
                try:
                    views.render_views(dest / "model.glb", dest / "views")
                    row["views"] = True
                except Exception as exc:
                    row.update(views=False, render_error=str(exc))
            with (directory / "items.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        quality.report(directory)
    quality.blind_review(args.out / "preserve_shape", args.out / "repair", args.out / "review")


if __name__ == "__main__":
    main()
