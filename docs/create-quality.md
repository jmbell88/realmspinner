# Create asset quality

New Create jobs use prompt policy 9. Ordinary images keep the user's composition;
reconstruction references retain isolation and framing guidance. Tiles, materials
and sheets keep their specialized templates. Stored jobs without a policy retain
policy 8. Every reference attempt keeps its own recipe, seed and measurements;
selecting an earlier attempt selects its provenance too.

New mesh jobs default to **Preserve shape**. It welds coincident vertices before
collapse simplification, keeping per-loop UVs and existing material maps. It does
not voxel-reconstruct the surface. **Repair and close holes** keeps the voxel,
remesh, unwrap and bake path. Jobs without a finishing choice retain repair.
Repair bakes metallic and alpha through emission, including linked textures and
mixed materials. Roughness and metallic are exported in glTF's G and B data
channels. Blend and mask behavior survive export. Combining mask and blend into
one atlas is refused rather than silently changing alpha semantics.

The inspector shows actual triangles, shared-frame silhouette changes, filled
opening pixels and material checks. These are advisory measurements, not a
usability score. Unknown measurements remain unknown. Failed or cancelled
finishing keeps the complete reconstruction and its reference for recovery;
`source.glb` remains available. Simplification can still lose fine detail.

## Reproducible quality program

`create-v1` contains two items per output family and fixed seeds 42 and 1337.
`create-edit-v1` exercises reference editing separately. Treat published suites
as immutable: add a new version rather than editing one used for comparison.
Manifests fingerprint suites, recipes, weights, executables and library versions.
Reference reuse hashes the exact image bytes and refuses a changed input on
resume. Native recipes identify TRELLIS.2 and trellis.cpp; release version is
verified against the pinned executable's complete SHA-256. A custom binary's
release is unknown, not guessed from the download catalog.

```powershell
uv run python -m realmspinner.bench run --suite create-v1 --recipe sdxl-cfg-raw --stage create --render --keep-source
uv run python -m realmspinner.bench quality RUN_DIRECTORY
uv run python -m realmspinner.bench blind-review LEFT_RUN RIGHT_RUN REVIEW_DIRECTORY
uv run python -m realmspinner.bench import-review REVIEW_DIRECTORY
```

Use `--reference-run RUN_DIRECTORY` for mesh engine comparisons and image edits.
For finishing alone, freeze the reconstruction as well:

```powershell
uv run python scripts/benchmark_mesh_finishing.py SOURCE_RUN FRESH_OUTPUT_DIRECTORY
```

This runs both methods on the same saved `source.glb`, checks source hashes,
records finishing runtime and memory, and renders views for anonymous review.
The ordinary mesh benchmark includes reconstruction runtime; do not mistake it
for the cost of finishing alone. Offline smoke comparisons of installed SDXL
CFG, SDXL PAG and FLUX.2 Klein are available through
`scripts/benchmark_create_quality.py`. It uses an isolated asset store and a
single seed; the full suites use both seeds.

Give reviewers only `review.json` and `assets/`. Keep `mapping.private.json`
with the coordinator. Fill each acceptance criterion with true, false or null;
import grades only after independent review. Judge tile seams, sprite alignment,
palette and readability on the final reduced exports, not only the original
generated sheet. Automated measurements do not establish those judgments.
The report gives all six families equal weight, separates aesthetic preference,
and leaves usable-output rate unknown until every acceptance criterion is known.
Runtime includes failed attempts. Memory is sampled every 250 ms and records
machine-wide VRAM usage and host commit, including other processes.

## Verification on this checkout

The full repository suite passed: **25,869 passed, 45 skipped**. Real Blender
bake/export regressions cover linked mixed metallic/wood maps, roughness channel
packing, blend and mask transparency, thin torus geometry and intentional holes.
Legacy prompt reproduction, earlier-attempt provenance and cancellation recovery
have dedicated regressions.

The offline RTX 5090 smoke campaign is retained in `build/create-quality/`.
Text generation completed two image subjects each with CFG, PAG and Klein.
The baseline all-output run completed 11 of 12 outputs: the sword reference was
rejected as containing multiple objects. Materials, tilesets, sprite sheets and
authored characters all completed, including sheet and rig follow-ups.
This is completion evidence, not acceptance or a model ranking.

Corrected SDXL editing runs are in `build/create-quality-edits/`; the isolated
Klein retry is in `build/create-quality-klein-edits/` (35.34 and 29.48 seconds).
Earlier Klein attempts hit the host-memory guard during concurrent test workers;
those failures remain in their original run records. Finishing of an identical
saved lantern and its ungraded review packet are in
`build/create-quality-fixed-source/`. Human grades remain unfilled.
On that single 277,052-triangle source, preserve took 5.82 seconds and exported
4,999 triangles; repair took 11.02 seconds and exported 5,000. Worst measured
silhouette coverage loss was 2.65% and 1.69% respectively. Both material checks
passed. These measurements cannot establish which result is more usable.

## Engine experiment gates

Automatic model routing and the pinned v0.6.0 runtime remain unchanged. A v0.8.1
comparison requires that runtime installed separately, using
`REALMSPINNER_TRELLIS_EXE`, and the same frozen references. The executable hash
preserves its identity even when its release cannot be verified locally.
No new binary or weights are fetched by these benchmarks.

[Upstream v0.8.0](https://github.com/pwilkin/trellis.cpp/releases/tag/v0.8.0)
adds Pixal3D through `--model pixal3d` and the HTTP server, with five additional
GGUF weight files. This checkout has no validated Pixal3D adapter or those
weights; its quality and resource requirements remain unmeasured. Treat it as a
separate backend experiment, not a TRELLIS.2 replacement.

[Upstream v0.8.1](https://github.com/pwilkin/trellis.cpp/releases/tag/v0.8.1)
describes optional native quad-retopology source-build dependencies. Prebuilt
archives omit that option, and full Windows image-to-quad validation is still
absent upstream. Native quads and Hunyuan3D integration remain deferred.
Neither upstream release nor Pixal3D was installed or benchmarked in this
offline pass. Change routing only after paired blind acceptance review across
all output families, publishing quality, failure, runtime and memory tradeoffs.
