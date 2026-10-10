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

## The 3D corpus: `create-3d-v1`

`create-3d-v1` is the suite for the question "do usable game assets come out of
this, and how fast?". It has six categories (`solid_prop`, `thin_weapon`,
`open_structure`, `hard_surface`, `humanoid`, `creature`), each with three
image-to-3D items and one text-to-3D item, and seeds 42, 1337 and 7. The seeds
are **attempt slots, in order**: seed 42 is attempt 1, so first-attempt,
best-of-3 and time-to-accepted fall out of the same run. They are also the
slots an external service is filed into (below), whatever seed it used itself.

Image items start from a still you supply. The images are never in the repo:
the runner reads them from `<bench dir>/references/create-3d-v1/<reference_image>`
(for example `humanoid/02.png`), or from the folder given by `--references`.
Use licence-clean stills. A run refuses with **one** message naming every
missing image; `--skip-missing` runs the items whose image exists and records
the rest under `skipped` in the manifest. The images' hashes go into the
manifest, so replacing one refuses a resume.

```powershell
uv run python -m realmspinner.bench run --suite create-3d-v1 --recipe sdxl-cfg-raw --stage model --render --references D:enchefs --skip-missing
```

Items grade on the `3d_v2` rubric: silhouette, reference fidelity, unseen
surfaces, texture, finishing damage and, for `humanoid` and `creature` only,
animation suitability. A packet from this suite also carries `repair_minutes`:
the hand-fix time, in minutes, to make the mesh usable (0 is a real answer;
leave null if unknown). `create-v1` keeps its own criteria and its reports are
unchanged.

`quality` adds an `attempts` block to `quality.json` for runs that recorded
attempts, overall and split by input mode and by category:

- `first_attempt` and `best_of_3`: the share of items accepted on attempt 1, or
  on any of attempts 1-3. Unknown (null) until every item's verdict is known.
- `generation_seconds_total`: every attempt's runtime, failures included.
- `time_to_accepted_seconds`: per item, the generation seconds of every attempt
  through the first accepted one plus its `repair_minutes` x 60. The block
  reports the **median** over items where that is known (and how many were);
  an item that was never accepted, or whose duration or repair time is
  missing, is left out rather than guessed.

### Comparing with Tripo or Meshy

Nothing here calls another service. Generate from the same reference images by
hand, then file the downloads as a run:

```
tripo-downloads/
  humanoid-img-02/attempt-1.glb
  humanoid-img-02/attempt-2.glb
  timings.json            (optional: {"humanoid-img-02/attempt-1.glb": 41.0})
```

```powershell
uv run python -m realmspinner.bench import-external OUR_RUN tripo-downloads tripo-run --label tripo
uv run python -m realmspinner.bench blind-review OUR_RUN tripo-run REVIEW_DIRECTORY
```

Each GLB becomes the unit of the same item and attempt in `OUR_RUN` and shares
its `input.png`, so both sides of a review start from the identical image. The
import checks the glTF magic number and a size cap and records the label as
`backend`; the blind packet leaves it out. Attempts the service has no file
for are reported as `missing` and are not scored as failures, so give each
service the same number of tries. Text-to-3D comparisons are a separate
exercise, because the services' own text pipelines differ from ours.

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
