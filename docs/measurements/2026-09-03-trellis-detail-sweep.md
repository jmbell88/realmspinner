# Does the shipped default throw away detail the reference shows? — pre-registration, 2026-09-03

**Status: pre-registered 2026-09-03; the machine run follows the same day, the
blind grade is owed.** Everything under "What will be run" and "Decision
rules" is fixed first; numbers go under "Results" afterwards and whichever
rule fires is applied verbatim
([`2026-08-30-art-verdicts-preregistration.md`](2026-08-30-art-verdicts-preregistration.md)).

## Why

Every graded corpus so far has asked whether a mesh is *usable*
([`2026-09-02-trellis-060-props.md`](2026-09-02-trellis-060-props.md): 11 of
22 at the shipped default). None has asked whether it keeps what the
reference shows. Reading the pipeline for that question found that at the
shipped defaults Warlock's own half removes nothing — `mesh_profile=raw`,
`reference_prep=False`, `mesh_retries=0` — and that every loss between
`input.png` and `source.glb` happens inside `trellis-server.exe`, three of
them behind launch flags the app never passed:

- **The exe decimates before Warlock sees the mesh.** A cold res-1024 run
  (`tests/fixtures/trellis_1024_v060.log`) reads `remesh_dc: … F=15210192`
  then `decimate_qem_gpu(target=300000): … F 15210192->297896`. So "Raw (no
  decimation)" was never that; it is the exe's quadric simplify to 300K faces
  at res 1024 (150K at 512). `--decim 0` turns the pass off; a positive grid
  selects the legacy cluster-grid pass. Never passed.
- **The texture is decoded at 512 on a 1024 mesh.** `Config.trellis_tex_res
  = 512` is a pin taken against v0.5.4's auto-tex-res noise; the props
  document's own first decision rule scheduled its re-examination on v0.6.0
  and `TODO.md` P32 carries it as the one surviving human item.
- **The UV atlas is fixed at the exe's default** (2048 at res 1024). `--atlas`
  never passed.
- **Geometry resolution 1536** is priced (`vram.TRELLIS_RES_MULT`) and
  admitted (`validation.ALLOWED_RESOLUTIONS`) but reachable from no form:
  `guidance.PLATFORMS` offers 2d/3d only.

As of 2026-09-03 the first two flags are `Config.trellis_decim` and
`trellis_atlas`, `None` omits the flag, and both are `service.sweeps.SERVER_AXES`
members; `optimize.CUSTOM_MAX` rose from 200k to 2M so a gltfpack budget can
be asked for a million faces once the exe stops throwing them away.

What this does **not** re-ask: the guidance strengths and the token budget
([`2026-09-02-trellis-guidance-sweep.md`](2026-09-02-trellis-guidance-sweep.md),
negative), the band ladder (`config.py`, widening is measured harmful), the
reroll and the hole-closing remesh
([`2026-09-02-hole-audit-vs-grade.md`](2026-09-02-hole-audit-vs-grade.md), 0
of 5 and <0.012). Those are settled instruments for a different question.

## What will be run

**Subjects** — `docs/measurements/corpora/detail-v1.txt`, five props-v1
prompts chosen on two facts the 2026-09-02 runs established: the reference
PNG is byte-identical across submits on this machine (so every delta below is
the reconstruction's, not SDXL's), and each carries fine surface detail with
no open form (so the silhouette audit is a reproducibility check here, not
the question). Treasure chest with iron banding (+4 on v0.6.0), barrel with
iron hoops (+3), weathered skull (+3), knight's helmet with the visor raised
(+3), carved pillar capital with acanthus leaves (1 — the detail-heaviest
subject in the corpus). Seed 42, `text → sdxl_cfg → TRELLIS` at the shipped
defaults except where a rung says otherwise, band auto, gss/gsh/max_tokens
omitted. Submitted by `scripts/campaign_detail.py`, tag `detail-060`,
drained headlessly through the real `studio.runtime.Runtime`.

**Rungs, per subject** — vectors, not OFAT, because two of them are pairs:

| unit | `trellis_decim` | profile | `trellis_tex_res` | `trellis_atlas` | resolution | what it asks |
|---|---|---|---|---|---|---|
| baseline | omit | raw | 512 | omit | 1024 | the shipped default, the control |
| ~~decim0-300k~~ | 0 | custom 300,000 | 512 | omit | 1024 | **retired 2026-09-06**, see the amendment below |
| ~~decim0-1M~~ | 0 | custom 1,000,000 | 512 | omit | 1024 | **retired 2026-09-06**, see the amendment below |
| ~~decim0-raw~~ | 0 | raw | 512 | omit | 1024 | **retired 2026-09-06**, see the amendment below |
| tex1024 | omit | raw | 1024 | omit | 1024 | the P3-owed pin re-examination |
| tex1024-atlas4096 | omit | raw | 1024 | 4096 | 1024 | the texture ceiling |
| res1536 | omit | raw | 512 | omit | 1536 | the geometry ceiling (second pass, exclusive mode) |

Three units per subject in the first pass (15, three server-restart groups,
tag `detail-060` — six and four before the `decim0-*` rungs were retired on
2026-09-06; those three shared one group, differing only in
`profile`/`custom_triangles`, which are not `SERVER_AXES`), one in the second (5, tag `detail-060-1536`, where the
sweep's *base* is resolution 1536 so its single unit is `expand`'s own
`baseline`). The second pass is separate because at 1536 trellis is priced
at 24 GiB beside the 7 GiB image pipe, and on the 32 GB card that is a WDDM
spill into host commit — the 2026-08-03 crash — rather than a clean refusal,
so those five are submitted and drained under `WARLOCK_VRAM_EXCLUSIVE=1`.
Roughly three hours of card time.

**Machine evidence, free, first** — `scripts/hole_audit_vs_grade.py --tag
detail-060 --corpus docs/measurements/corpora/detail-v1.txt`, extended for
this run with the size of `source.glb` and `model.glb` in MiB beside the
audited face count and the whole-job seconds:

1. `mesh_audit.worst` on every unit, as a reproducibility check: on these
   closed forms it must stay within 0.02 of the subject's corpus value on
   the baseline and must not *rise* past 0.07 on any rung (a rung that opens
   the silhouette is a regression whatever it does for detail).
2. Faces, bytes and seconds per rung. (The `decim0-raw` clause here — "its
   GLB is expected to be hundreds of MiB; whether the viewer opens it is
   recorded, not assumed" — was answered before the viewer was ever reached:
   see the amendment.)
3. `tiercheck.compare` for the two gltfpack rungs against their own
   `source.glb`: UVs, both PBR maps and the material must survive, the same
   bar `tests/test_generation_tiers.py` and the tier programme set.
4. The trellis log per unit: the first `decim0-*` unit must show no
   `decimate_qem_gpu` line; the first `tex1024` unit `[6/7] res-1024
   texture`; the first `atlas4096` unit `uv_bake: atlas 4096x4096`. If any of
   those is absent the flag did not reach the exe and the run is void.

**Human evidence** — one blind pass in Review on the −5..+5 scale with tags,
and per subject one pairwise call per rung against the baseline, judged in
the viewer at fit-to-view and at 4× on the busiest surface: **more** of the
reference's detail, **same**, or **less**. The pairwise call is the
instrument; the grade is there so a rung that gains detail and loses
usability is seen as both.

## Decision rules

- **`tex1024`**: clean texture (no per-texel noise, the v0.5.4 defect) on 5 of
  5 **and** "more" on ≥3 of 5 → `Config.trellis_tex_res` becomes 1024, citing
  this document, and P3's surviving item is struck. Noise reproduces on any
  subject → the pin stays and the reproduction is recorded here.
- **`tex1024-atlas4096`**: graded only if `tex1024` is clean; becomes the
  default only if "more" over `tex1024` on ≥3 of 5 **and** whole-job time
  ≤1.5× baseline. Otherwise it stays an axis.
- **`decim0-*`**: `trellis_decim = 0` becomes the default **only paired with
  a gltfpack budget** that passes `tiercheck.compare` on 5 of 5 and is "more"
  on ≥3 of 5; that budget then enters `settings_3d.PROFILES` as a named tier
  and becomes `Config.mesh_profile`, with this corpus standing in for the
  chest/sword/rock trio the tier bar names (the chest is here; the document
  says so where the bar is restated). `decim0-raw` is never a default
  candidate — it is recorded for size and load time, and graded as the
  ceiling only if the viewer opens it. No decim rung wins → the flag stays
  `None`, remains an axis, and the negative is recorded.
- **`res1536`**: "more" on ≥3 of 5 and no new audit failure → a third
  `guidance.PLATFORMS` row (`3d_high`, resolution 1536), which the Mesh
  column's platform combo picks up with no other code, and the manual says it
  forces exclusive mode. Time cap 2× baseline. Otherwise recorded, no row.
- Whatever wins, the rig path's ~300k-face constraint and the remesh panel
  are unaffected: both read `model.glb`, and a budget is what a winning decim
  rung ships with.
- The noise floor for "same" on faces is `trellis_band`'s 0.3 %; for the
  audit it is the 0.02 above. Neither is a floor for the pairwise call,
  which has none — it is one reviewer's eye, and the single-reviewer caveat
  of [`2026-08-09-rebaseline.md`](2026-08-09-rebaseline.md) applies.

## Results — machine evidence

*(appended after the run)*

## Results — grades

*(owed: the blind pass and the pairwise calls)*

## Amendment, 2026-09-06 — the `decim0-*` rule fires negative; `CUSTOM_MAX` back to 250k

The run reached one subject. All three `decim0-*` rungs failed on it, so the
**`decim0-*` decision rule fires as pre-registered**: *"No decim rung wins →
the flag stays `None`, remains an axis, and the negative is recorded."*
`Config.trellis_decim` stays `None`, the three rungs come out of
`scripts/campaign_detail.py`, and this is that record. The first pass is now
three units per subject (baseline, `tex1024`, `tex1024-atlas4096`); nothing
about `tex1024`, `tex1024-atlas4096` or `res1536` changes, and their decision
rules stand untouched.

### What ran (chest, seed 42, `detail-060`, sweep `4b7edc9dc08a`)

| rung | result | wall clock |
| --- | --- | --- |
| `baseline s42` | done, 288,526 tri, 24.5 MB | 8.5 min |
| `decim0-300k s42` | error: `trellis-server returned more than 536870912 bytes; refusing to buffer it` | 28.9 min |
| `decim0-1M s42` | error: `The 3D engine stopped unexpectedly` | 30.4 min |
| `decim0-raw s42` | error: `The 3D engine stopped unexpectedly` | 30.4 min |

The remaining 26 units were cancelled unrun.

### The run is not void: the flag reached the exe

Void-check #4 above requires the first `decim0-*` unit to show no
`decimate_qem_gpu` line. `assets/trellis.log` satisfies it. The baseline at
line 42154 reads `remesh_dc: 28073866 active voxels -> V=17755231
F=35522920` and then `decimate_qem_gpu(target=300000): … F 34951324->296572`,
landing at the 288,526 the row records. The three `decim0` runs (42246,
42334, 42422) repeat that same `remesh_dc` line — byte-identical, as the
seed-stability premise predicts — and have **no** `decimate_qem_gpu` after it.
`--decim 0` arrived and did what it says. These are measurements of the axis.

Two corrections to this document's own arithmetic fall out of that line:

- **The undecimated mesh is 34,951,324 faces after floater removal**, not the
  ~15.2M estimated here from `tests/fixtures/trellis_1024_v060.log`. That
  fixture was a different subject; ~35M is what this corpus produces.
- **`decim0-300k` did not crash the engine — the exe succeeded.** It logged
  `done in 1702.7s` and wrote a textured GLB, having spent its xatlas budget
  (`uv_bake: xatlas charting exceeded 300s; abandoning it`) and fallen back to
  `uv_chart_project` at atlas 2048. The job failed *client-side*, on
  `trellis.MAX_GLB_BYTES` refusing a body over 512 MB. So gltfpack never ran
  and that rung never got to ask its own question — "does gltfpack's simplify
  keep more than the exe's QEM at the same count" is still unanswered, and now
  unaskable by this route.
- `decim0-1M` and `decim0-raw` did kill the server: both logs stop inside
  `uv_bake` and the next line is a fresh `[trellis-server] … listening`.

### Why the axis is retired rather than re-tried

The honest reading is not "the budget was too big" — it is that the pipeline
cannot carry an undecimated res-1024 reconstruction at all. One rung of the
three produced a GLB, at 28.9 minutes and over half a gigabyte, with its UV
charting abandoned on a timeout; the other two killed the process. Nothing in
that is a candidate for a shipped default, and the pre-registration already
said `decim0-raw` never was one. Retrying at a positive `--decim` grid would
be the legacy cluster-grid pass — a different question from the one registered
here, and it needs its own document if anyone wants it.

`MAX_GLB_BYTES` is **not** raised. It refused a body a successful generation
produced, which is worth stating plainly, but with the axis retired no rung
approaches it: at the shipped default `source.glb` is ~24 MB. Raising a
memory-exhaustion guard (MDL-13) to accommodate a configuration that is being
withdrawn would be trading a real protection for nothing.

### `optimize.CUSTOM_MAX`, 2M → 250k

This document raised `CUSTOM_MAX` from 200k to 2M so a gltfpack budget could
ask for a million faces "once the exe stops throwing them away". The exe is
not going to stop, so the premise is gone and the ceiling goes back down.

With decimation on — which is now always — `source.glb` lands at ~300k faces
at res 1024, so any budget above that asks gltfpack to remove nothing. 250k
sits just under the landing point, where a custom budget is always a genuine
reduction of something that exists. `CUSTOM_MIN` is unchanged.

No graded corpus was keyed on the 2M ceiling: it shipped 2026-09-03, and the
only jobs that ever carried a budget above 250k are the two error rows above.
`scripts/campaign_detail.py` was refused by the new ceiling until the rungs
were removed, which is all-or-nothing admission working as designed.

### Owed, unchanged

The `tex1024` / `tex1024-atlas4096` question — the P3 pin re-examination, the
reason this instrument exists — has not been asked yet. Fifteen units.
