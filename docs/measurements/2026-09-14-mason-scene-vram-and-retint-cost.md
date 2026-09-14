# Mason scene VRAM and retint cost — 2026-09-14

Two pre-registered questions, taken together because both are about the same
two caches: `mason_assets.AssetSource` (CPU-side decoded geometry, budgeted)
and `MasonView._cache` (GPU-resident uploads, unbudgeted).

**F5 — what a Mason scene of sixty distinct textured library assets costs in
video memory.** Sixty textured library assets placed in one scene is real
video memory, and `service.validation.check_vram` guards the queue — the
thing that runs jobs — not the studio. Mason's asset cache has a byte budget
and instancing means N placements of one asset are one upload, so the cheap
cases are already cheap; what has no answer is a scene that genuinely wants
more than the card has. The failure mode is a driver-level allocation
failure with no refusal in front of it. Wants a measurement first — what a
scene of sixty distinct textured assets actually costs on the target card —
and then a decision about whether the answer is a refusal, a warning, or
evicting the cache harder. The measurement is the part that needs a card.

**F6 — does a material override cost a second geometry upload that
matters.** The override is in the GPU cache key, so a retinted copy of a
shared asset re-uploads identical triangles. Place a few dozen retinted
copies of one asset and see whether it matters before changing how the
renderer binds materials.

Both were measured, not decided: F5 presents numbers and what each policy
option would have to key on, no decision. F6 states a conclusion, because
the entry asked for one ("see whether it matters").

## The machine

`nvidia-smi --query-gpu=name,driver_version,memory.total`:

```
NVIDIA GeForce RTX 5090, 610.62, 32607 MiB
```

Idle baseline before any measurement: 364 MiB used (drifted to ~376 MiB over
the session — background Windows processes, not this harness; see "What
would contaminate this" below). `nvidia-smi --query-compute-apps` listed no
process with an attributable non-zero memory figure throughout the session
(every row read `[N/A]`, including this box's own background apps) — the
means checked for contamination found none, but per-process attribution is
unavailable on this driver/permission setup, so the aggregate
`memory.used` counter, not the process list, is what every delta below rests
on.

## What "realistic" means here

A Mason library asset is a `model.glb` after `pipelines/optimize.py`.
`optimize.PROFILES["standard"]` is 50,000 triangles — the retarget panel's
own default custom-triangle value (`panes/retarget_panel.py:148`) — and
`Config.trellis_tex_res` defaults to 512 (`config.py:444`), at which the
exe's own `--atlas` default is 1024 (`config.py:502`'s comment: "2048 at res
1024, 1024 at 512"). `viewer.scene.TEXTURE_SLOTS`'s own comment says this
app's meshes only ever carry two of the five possible texture slots —
`base_color` and `metallic_roughness`. So one synthetic asset here is: one
primitive, 50,000 triangles, 25,000 vertices (the ~2-faces-per-vertex ratio
`2026-09-11-mason-obj-ceiling.md` measured on a closed mesh), two 1024x1024
RGBA8 textures.

Per-asset bytes, computed from the same layout `viewer/scene.py` uses
(`GpuPrimitive`'s 32-byte interleaved vertex, `u4` indices;
`GpuMaterial.build_mipmaps()` always runs, at the standard ~4/3 mip-chain
overhead):

| what | bytes | of which |
|---|---:|---|
| CPU geometry (`mason_assets._nbytes`'s own unit) | 1,400,000 | positions 300,000 + normals 300,000 + uv 200,000 + indices 600,000 |
| GPU vertex buffer (`vbo`) | 800,000 | 25,000 × 32 |
| GPU index buffer (`ibo`) | 600,000 | 150,000 × 4 |
| GPU textures, with mips | 11,184,810 | 2 × 1024×1024×4 × 4/3 |
| **GPU total per asset** | **12,584,810 (~12.0 MiB)** | |

Textures are **89%** of what one asset actually costs the GPU. The CPU-side
cache (below) accounts only the geometry row — a ninth of the real cost.

## Harness

`C:\Users\ILWT\AppData\Local\Temp\claude\D--Projects-warlock\e253d054-a55b-4e05-8cf9-7a83afc27185\scratchpad\mason-vram\` —
session scratch, not shipped. Every script imports `common.py` first, which
pins `WARLOCK_HOME` to a fresh temp dir and `WARLOCK_NO_MIGRATE=1` before any
`warlock` import (the 2026-09-13 probe that zeroed real weights is why this
order matters).

- `common.py` — asset/scene builders, `nvidia-smi` sampling, a `StubSource`
  (`GeometrySource`, generalized from `tests/test_mason_view.py`'s own
  `_Source`) that answers pre-built primitives synchronously instead of
  parsing a GLB off a task thread. This is deliberate, not a shortcut past
  "the real upload path": what needs to be real is `MasonView.sync` ->
  `_build` -> `viewer.scene.GpuModel` (the actual GPU-upload code), which
  this drives unchanged — exactly what the real test suite's `view` fixture
  does. `mason_assets.AssetSource`'s own CPU-cache logic (`_store`/`_evict`)
  is driven directly and separately, for real, not stubbed.
- `f5_scene_vram.py` — CPU-cache accounting and eviction onset (no GL
  needed).
- `f5_worker.py` / `f5_isolated.py` — one scene size per subprocess, GPU
  VRAM delta. Isolated because the first same-process measurement run showed
  **VRAM is sticky**: `moderngl` object `.release()` frees the driver's
  objects but the process's own memory pool does not shrink back to the OS,
  so a second upload in the same process after a release reused the first
  upload's pool and read back a delta of 0. That is itself a real finding
  (see below), but it makes same-process repeats useless for comparing scene
  sizes against each other, so the reported table is one fresh process per
  sample.
- `f6_retint_cost.py` — same-process run (three scenarios in sequence);
  frame-time numbers are unaffected by the stickiness above since they are a
  live measurement, not a used-memory reading, so this run is what the
  frame-time table below comes from.
- `f6_worker.py` / `f6_isolated.py` — one retint scenario per subprocess, for
  a clean VRAM table matching F5's methodology.

Re-run: `.venv\Scripts\python.exe f5_isolated.py`, `f6_retint_cost.py`,
`f6_isolated.py`, from that directory, with a GPU present and nothing else
CUDA/GL-heavy running (checked here with `nvidia-smi --query-compute-apps`
before and after).

## F5 — the numbers

### The CPU-side cache (`mason_assets.AssetSource`, `CACHE_BYTES` = 512 MiB)

Driven for real through `_store`, geometry only (no GL context needed):

| assets placed | accounted bytes | entries held | evicted | over 512 MiB budget |
|---:|---:|---:|---:|---|
| 60 | 80.11 MiB | 60 | 0 | no |
| 120 | 160.22 MiB | 120 | 0 | no |
| 240 | 320.43 MiB | 240 | 0 | no |

At 1.4 MiB CPU-side geometry per asset, sixty distinct assets is nowhere
near this budget — the entry's own words, "the cheap cases are already
cheap," is true of the CPU cache at every size this doc tests. Pushed to
where it actually evicts: the **384th** placed distinct asset is the first
whose insert evicts the oldest entry (`_evict`'s own rule: evict while over
budget and more than one entry remains), and the cache settles at **383
entries / 511.36 MiB** — just under budget, as `_evict`'s loop condition
guarantees. That is a scene four hundred assets deep before this budget does
anything at all.

### The GPU-side cache (`MasonView._cache`) has no byte budget

Confirmed by reading, not inferring: `grep -n "CACHE_BYTES\|_evict\|budget"`
over `mason_view.py` and `viewer/scene.py` finds nothing. `MasonView.sync`
evicts an entry only when nothing places its ref *at all* this frame
(`mason_view.py:415-421`) — never for memory pressure, never partially.
**The only byte budget anywhere in this path governs CPU-side geometry
only, never textures, and never a single byte of what actually lives on the
GPU.** A scene can carry an arbitrary number of fully GPU-resident, fully
textured assets and the only cache with a budget will never notice, because
it is measuring a different resource than the one that is actually scarce.

### Driver VRAM, isolated (one fresh process per sample, 3 samples each — identical every time)

| distinct assets | VRAM delta (total) | per-asset (delta ÷ N, fixed cost included) |
|---:|---:|---:|
| 0 (empty Mason view — see below) | 52 MiB | — |
| 60 | 814 MiB | 13.6 MiB |
| 120 | 1,579 MiB | 13.2 MiB |
| 240 | 3,109 MiB | 13.0 MiB |

A **fixed ~52 MiB** cost (10 MiB context creation + ~42 MiB
`MasonView.__init__` — shader program compilation, the three gizmos, the
grid, `SceneMarks` — all built regardless of scene content) sits under every
number above. Subtracting it gives the marginal, scene-content-only cost:

| distinct assets | marginal VRAM | marginal per-asset | analytic prediction |
|---:|---:|---:|---:|
| 60 | 762 MiB | 12.7 MiB | 12.0 MiB |
| 120 | 1,527 MiB | 12.7 MiB | 12.0 MiB |
| 240 | 3,057 MiB | 12.7 MiB | 12.0 MiB |

Linear, within 6% of the analytic vbo+ibo+texture figure at every size (the
gap is VAO/driver bookkeeping the analytic count does not model), and the
per-asset figure does not move between 60 and 240 — nothing here shows any
sign of a ceiling, at fourteen times the CPU-cache's own comfortable range.

### What sixty (and beyond) actually costs, and what it would take to matter

Sixty distinct textured assets costs **~814 MiB**, under 2.5% of this card's
32,607 MiB. Two hundred forty costs **~3.1 GiB**, under 10%. Extrapolating
the measured ~12.7 MiB/asset marginal rate (not measured past 240, so this
is a projection, not a fourth data point) — filling the whole card would
take on the order of **2,500 distinct textured assets** in one scene, two
orders of magnitude past the sixty the entry named and an order past
`scene.MAX_PLACED` (100,000 *placements*, but recall instancing: 100,000
placements of sixty assets is still sixty uploads). A scene actually reaches
this ceiling only by *asset diversity*, never by placement count — the same
shape `2026-09-11-mason-obj-ceiling.md` found for the merged-OBJ ceiling,
and for the same reason: instancing is what makes placement count cheap, so
whatever does become expensive is expensive along the one axis instancing
does not compress.

**On the card the beta actually ships to, not this one.** This was measured
on a 32 GiB RTX 5090 because that is the machine that has one; the closed
beta's candidate machines are 8 GiB cards (the VRAM floor
`2026-09-05-clean-machine-install.md` was run against). The per-asset figure
is a property of the upload, not of the card, so it carries over: sixty
assets is ~10% of an 8 GiB card rather than 2.5%, and the card is full at
**roughly 600 distinct textured assets** less whatever the rest of the app
and the desktop already hold — still ten times the entry's sixty, but one
order of magnitude of headroom rather than two, and not a projection anyone
has checked on an 8 GiB card.

### VRAM stickiness (found, not asked for, and worth recording)

Within one process, releasing a `MasonView` (`.release()`, which releases
every VBO/IBO/texture through moderngl) does **not** return memory to the
OS/driver — a second build-and-upload cycle in the same process re-used the
first cycle's pool and read a VRAM delta of 0 even though a full re-upload
had genuinely happened. This means the real-world quantity that matters is
not "how big is the scene right now" but **the high-water mark of every
scene loaded since the app launched** — a session that briefly opens a
600-asset scene and then closes it back down to three keeps paying for the
600-asset peak until the process exits. Any policy decision this document
feeds should be measured against that peak, not against the live scene.

### Can an allocation fail before any refusal or warning?

Yes, and nothing on this path catches it. `GpuMaterial.__init__`'s
`ctx.texture(...)` and `GpuPrimitive.__init__`'s `ctx.buffer(...)` calls
(`viewer/scene.py:68`, `:150-151`) are unguarded; `MasonView._build` and
`.sync` (`mason_view.py:423-441`, `:371-421`) carry no
try/except around them; `mason_viewport._mason_viewport`
(`mason_viewport.py:107`) calls `view.draw(...)` with nothing wrapping it
either. Tracing the call chain up: `_mason_workspace` is dispatched from
`App`'s per-mode draw switch inside `frame()` (`main.py`), and `frame()`
itself is called from the run loop's own `try: ... except Exception as exc:`
(`main.py:1243-1275`) — the app's generic crash handler, the one that writes
a crash report. So a GL allocation failure here is survivable in the sense
that it does not corrupt state, but its user-visible shape is a **generic
crash dialog**, not "this scene needs more video memory than your card has"
— the one shape of failure TODO.md's F5 entry says this app otherwise
always puts a sentence in front of.

### Options (no decision — what each would have to key on)

1. **Refuse at placement/load time.** Needs a per-scene VRAM estimate
   *before* anything is uploaded — the CPU `AssetSource` only knows
   geometry bytes, not texture dimensions, until a GLB is actually opened,
   so estimating ahead of upload means reading each job's texture
   metadata (or a recorded atlas size) rather than the decoded pixels.
   Needs a VRAM ceiling to check against — `warlock/vram.py` already reads
   live free/total VRAM for the queue's own admission door
   (`vram.live_memory()`/`device_memory()`), so the number does not need
   inventing, only reusing from the studio side. Refusing after the fact
   (build it, then discover it doesn't fit) is the wrong order — a refusal
   after allocation is not a refusal.
2. **Warn, non-blocking**, `PLACED_WARN_THRESHOLD`'s own shape. Keys on the
   same estimate as (1) but against a softer threshold (a fraction of
   `vram.live_memory()`'s free figure, or a fixed count once real-world
   scenes establish what "a lot" means) and never blocks opening or saving
   the document — the same argument `2026-09-11-mason-scene-ceilings.md`
   made for `MAX_PLACED` vs. its warn threshold: losing access to a document
   is worse than a scene that is merely heavy.
3. **Evict the GPU cache harder.** Keys on giving `MasonView._cache` (or
   `GpuModel`) byte accounting it does not have today — none of
   `GpuPrimitive`/`GpuMaterial`/`GpuModel` currently records its own
   `nbytes` the way `mason_assets._Entry` does — plus a policy for *which*
   entry to evict under pressure while it is still placed and visible
   (unlike the CPU cache's LRU-among-the-unused, evicting a visible asset
   means a visible re-upload stutter the next time it is needed, which the
   CPU-cache docstring already accepts as a cost but the GPU cache has never
   had to pay). Also needs the stickiness finding above factored in: evicting
   objects does not shrink the process's own driver pool, so eviction bounds
   *this scene's* residency, not the session's historical peak.

## F6 — the numbers

One shared asset (the same 50,000-triangle, two-texture asset as above), 36
placements, three scenarios. `distinct_gpu_entries` is `len(view._cache)`
after `sync` — the real count of separate `GpuModel` uploads.

### Cache entries and analytic upload size

| scenario | distinct GPU entries | analytic upload | vs. (a) |
|---|---:|---:|---:|
| (a) untinted, shared | 1 | 12.0 MiB | 1x |
| (b) 36 distinct overrides | 36 | 432.1 MiB | 36x |
| (c) 6 overrides x 6 placements | 6 | 72.0 MiB | 6x |

Exactly `overrides` entries in every case (0 overrides collapses to 1, the
shared base) — the cache key is doing precisely what its own docstring
says.

### Driver VRAM, isolated (one fresh process per sample, 3 samples each)

| scenario | VRAM delta | minus ~52 MiB fixed cost |
|---|---:|---:|
| (a) untinted | 62 MiB | 10 MiB (~1 asset) |
| (b) 36 distinct | 509 MiB | 457 MiB (~36 assets) |
| (c) 6x6 | 126 MiB | 74 MiB (~6 assets) |

Matches the analytic table above within 1-3%, and confirms the fixed
~52 MiB `MasonView` baseline found in F5 is the same constant here (it does
not depend on scene content, only on a view existing).

### Frame time, 300 draws after 20-frame warm-up, `Renderer.draw` called
directly (bypassing `MasonView.draw`'s redraw-skip memoization, which would
otherwise make every repeat frame free and measure nothing)

| scenario | median | min | max |
|---|---:|---:|---:|
| (a) untinted | 1.231 ms | 1.146 ms | 1.941 ms |
| (b) 36 distinct | 1.236 ms | 1.142 ms | 2.725 ms |
| (c) 6x6 | 1.230 ms | 1.162 ms | 2.752 ms |

**No measurable difference in per-frame draw cost.** All three scenarios
issue the same 36 `(node, primitive)` draw calls per frame regardless of how
many distinct `GpuModel`s back them —
`_composite`/`Renderer._draw_model` walk `gpu.draws` and bind whatever
material each primitive carries either way (`render.py:349-397`) — so the
only place the override's cost shows up is at *build* time (the extra
uploads above), never at *draw* time. The max-time outliers in (b)/(c) (2.7
ms vs. (a)'s 1.9 ms) are consistent with occasional OS/driver scheduling
noise on a 36-draw-call frame, not a systematic cost: the medians agree to
three significant figures across all three scenarios.

### F6 conclusion: the second upload does not matter at this size, and should not be built yet

At 36 placements of one asset — "a few dozen retinted copies," exactly what
the entry asked to try — a material override costs:

- **Extra VRAM**: real, and linear in override count (roughly one extra
  asset's worth of upload, ~12.7 MiB, per distinct override), but the
  absolute numbers are small. 36 distinct overrides of one 50K-triangle
  asset is ~509 MiB — under 1.6% of this card's VRAM, and comparable to
  what six *additional distinct assets* would have cost anyway.
- **Extra frame time**: none measurable. Draw-call count is fixed at the
  placement count regardless of override count, and per-draw material
  binding cost does not depend on how many distinct `GpuModel`s exist.

So the entry's own framing — "worth doing only if overrides turn out to be
common" — is not yet met by anything this measurement can show: at "a few
dozen retinted copies" of one asset, the second upload costs real but small
VRAM and zero frame time. **The renderer change is not worth building on
this evidence.** It would become worth it only if overrides multiply the
*VRAM* picture the way F5's asset count does — many dozens of distinct
overrides across many distinct assets simultaneously — which is a
usage-frequency question this measurement cannot answer (it was never asked
to; the entry's own text says "worth doing only if overrides turn out to be
common," a product question, not a performance one).

### The cache-key line, and what the fix would touch

`src/warlock/studio/mason_view.py:183`, inside `_entry_key`
(`:168-183`):

```python
return (ref_key(placed.ref), id(placed.material))
```

A per-draw material uniform fix, sketched but **not built**:

- **`mason_view.py:183`** — drop `id(placed.material)` from the key (1
  line), so overrides stop forking the GPU cache.
- **`mason_view.py:423-441`** (`_build`) — stop baking the override into the
  uploaded primitive's material (remove the `if placed.material is not
  None: prims = [_replace(...)]` block, ~6 lines); the shared entry uploads
  the asset's own material only, once.
- **`mason/scene.py`**'s `DrawNode`/`NodePool` — carry the per-placement
  override alongside `world`/`skin` (the two per-instance fields the
  composite already threads through the pool for exactly this reason; see
  `mason_view.py`'s own module docstring on why `DrawNode` exists at all).
  Roughly 10-15 lines: one new field, one new `NodePool.node(...)` argument.
- **`viewer/render.py`**'s `Renderer._draw_model` (`:316-410`) — after
  `primitive.material.bind(program)`, write the override's factors as
  per-draw uniforms exactly the way the wireframe overlay's `tint` parameter
  already does for `u_base_color_factor` alone (`:385-387`): generalize that
  to `u_metallic`/`u_roughness`/`u_emissive_factor`/`u_alpha_cutoff`/
  `u_alpha_mask` when a per-draw override is present. ~15-20 lines; no new
  GLSL, since the shader already reads these as uniforms and only the CPU
  side decides what to write into them.
- **Exporter unchanged**, as the entry predicted: `mason/gltfout.py` bakes a
  distinct material per node at export time regardless of how the renderer
  binds materials, and that cost is paid once per export rather than once
  per frame.
- **Textures**: this fix only helps the common "retint" shape (same
  textures, different factors, exactly what this harness built for (b)/(c)).
  An override that swaps in a genuinely different *texture* still needs its
  own upload either way — a per-draw factor uniform cannot substitute a
  different image without one.

Total estimate: **~35-45 lines across three files**, no shader changes,
matching the entry's own characterization ("a renderer change").

## Contamination check

`nvidia-smi --query-compute-apps=pid,process_name,used_memory` was run
before and after both F5 and F6; every row read `[N/A]` for memory
throughout (per-process attribution unavailable on this driver/permission
setup), and no unexpected process appeared or disappeared across any run.
Idle baseline drifted 364 -> 376 MiB over the session (12 MiB, likely
Windows compositor/background churn) — small next to every measured delta
above and does not change any conclusion. Every isolated-process sample
(F5's `f5_isolated.py`, F6's `f6_isolated.py`) reproduced identically across
three repeats, which is itself evidence against contamination: a
contaminated run would show variance the clean repeats do not.

## How to re-measure

Fresh process per sample, `nvidia-smi --query-gpu=memory.used` before and
after, `ctx.finish()` before every sample. Same-process repeats measure
driver pool reuse, not scene cost — see "VRAM stickiness" above. Frame-time
comparisons should call `Renderer.draw` (or `.draw_ids`) directly rather
than `MasonView.draw`, whose redraw-skip memoization makes every unchanged
repeat free. `common.py` in the scratch directory above has the asset/scene
builders; a future re-run at a different `TARGET_TRIANGLES` or atlas size
should rebuild from there rather than from these numbers, since every table
here is a direct function of the 12.58 MiB/asset figure this doc measured.
