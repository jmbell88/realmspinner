# Native kernel batch 10 — seven candidates, measured. No C written.

> **Landed the same day:** C1 (`select.linked` via `csgraph`), C3 (`stamp`
> returns its window) and C2 with the KD-tree accepted — the user's call.
> `FALLOFF_CHUNK_PAIRS` is gone and `MAX_FALLOFF_PAIRS` became
> `MAX_FALLOFF_VERTICES = 300_000`, because the tree's cost follows mesh size,
> not the pair product (fixer's measurement, best of 3: 200k fully selected
> 192 ms, 300k fully selected 362 ms, 1M with 100 selected 320 ms). So a fully
> selected mesh at the cap still exceeds this document's 100 ms gate; the cap
> is set at "well under a second", like Clay's other op ceilings.
> Regression tests, each shown failing against `c5365993`'s source:
> `tests/clay/test_select.py::test_linked_on_a_long_thin_strip_finishes_well_under_a_second`,
> `tests/test_clay_falloff_chunked.py::test_a_large_selection_on_a_large_mesh_still_gets_a_soft_falloff`,
> `tests/inker/test_flourish_stamp_windowed.py::test_stamp_returns_a_window_sized_patch_not_a_full_frame`.
> The tables below are the before record.

2026-09-13, at `c5365993`. Machine: Windows 11, numpy 2.x, Python 3.13,
`vendor/warlockc/warlockc.dll` present (ABI 10) — no site below has a kernel,
so the DLL changes nothing here. Every figure is the **minimum of seven runs in
a fresh process** through `scripts/bench_native.py --sweep`.

Batches 6–9 closed their named list with Python fixes and declared it empty.
This batch surveys the paths they never opened: Clay's pick, falloff and
select-linked, Flourish's particles and smoke, Inker's RotSprite drag and the
Sirens render. Gates were written into the case definitions before the first
number. Each variant asserts that it reproduces the shipped result before it is
timed. `kdtree` is the one exception: it asserts only a 1e-9 match (§2).

## Summary

| # | Candidate | Gate | Shipped | Best variant | Verdict |
|---|---|---|---|---|---|
| C1 | `select.linked`, 200k verts | >50 ms | **11 107 ms** | `csgraph` **4.4 ms**, exact | **fix with SciPy, no kernel** — §1 |
| C2 | `drag._min_distance` @40M pairs | >100 ms | **1 343 ms** | `kdtree` **66 ms**, *not* bit-identical | **decision owed** — §2 |
| C3 | textured particles @400 | >100 ms | **1 930 ms** | `stamp_windowed` **29 ms**, exact | **fix in Python, no kernel** — §3 |
| C4 | smoke layer @80 blobs | >100 ms | **388 ms** | none found | **kernel survivor** — §4 |
| C5 | RotSprite drag @256² | >16 ms | **294 ms** | `bands_merged` 295 ms, rejected | **kernel survivor** — §5 |
| C6 | Sirens render, 3-min busy song | >1 s | **~8.1 s** at 32 ch, ~1.5 s at 5 ch | `decimate_once`, exact, no faster | **re-aim before any kernel** — §6 |
| C7 | BVH pick @200k tris | >2 ms | **1.71 ms** oblique, 0.57 ms grid | — | **under gate, no action** — §7 |

---

## §1 — C1, `select.linked`: label propagation is O(passes × mesh)

| verts | `shipped` | `csgraph` |
|---|---|---|
| 2 000 | 2.16 ms | 0.09 ms |
| 20 000 | 115 ms | 0.48 ms |
| 200 000 | **11 107 ms** | **4.40 ms** |

The fixture is four disconnected strips two faces wide. That is the worst case,
because the number of passes grows with a strip's length, not with the square
root of its size. A compact import converges in far fewer passes. Even so,
11 s on the frame thread for one L key is not a matter of tuning.
`scipy.sparse.csgraph.connected_components` over the edge graph from
`adjacency` returns the same vertex set (compared sorted), and SciPy is already
a Clay dependency through `ops_topo`'s weld. The graph build is untimed here
because `adjacency` is cached per mesh.

**Where the fix goes:** `clay/select.py:linked`. The regression test must fail
against the unfixed code, for example a long-strip mesh under a wall-clock or
pass-count bound.

## §2 — C2, `_min_distance`: a KD-tree wins, but it breaks a documented promise

| pairs | `chunked` | `kdtree` |
|---|---|---|
| 4 M | 134 ms | 45 ms |
| 16 M | 537 ms | 56 ms |
| 40 M | **1 343 ms** | **66 ms** |

The fixture has 200k positions, and the selection is sized so the pair count
hits the target. The shipped chunked search is linear in pairs, so the cap
itself is 13× over the gate. `cKDTree`, build and query both timed, is nearly
flat. However, its distances match the broadcast to 1e-9, not exactly:
`FALLOFF_CHUNK_PAIRS`'s docstring promises bit-identity, and a distance
landing exactly on the radius could flip a vertex in or out of the set.

Two options, and choosing between them is a human decision:
- Accept the tree, rewrite that docstring, and replace `MAX_FALLOFF_PAIRS` with
  a much larger cap.
- Keep exactness with a C running-minimum kernel. It has no temporaries and is
  memory-bound, so it is **estimated** (not measured) at 5–10×, which is
  130–270 ms at the cap and still over the gate.

## §3 — C3, textured particles: one full frame allocated per particle

| particles | `disc` | `textured` | `stamp_windowed` |
|---|---|---|---|
| 50 | 4.6 ms | 237 ms | 6.1 ms |
| 200 | 15.1 ms | 963 ms | 16.0 ms |
| 400 | 17.7 ms | **1 930 ms** | **29.1 ms** |

`stamp` allocates a zero (512, 512, 4) plane for every particle.
`_render_textured` then adds and alpha-composes the whole frame, although only
the particle's window is non-zero. The variant runs the same elementwise
expressions on the window slice only and is `array_equal` to the shipped
output: **66×**, well under the gate. The disc path already stays inside its
window.

**Where the fix goes:** `stamp` returns `(window, patch)` and its callers
compose into the slice. The parity test compares against the full-frame form.

## §4 — C4, smoke: no Python fix found; the one fused-kernel survivor

| blobs | ms |
|---|---|
| 10 | 54 |
| 40 | 202 |
| 80 | **388** |

Profile at 80 blobs (0.39 s): `over_into` 0.18 s tottime, `fbm_plane` 0.13 s
cumulative (three octaves of `value2d` and `hash_lattice`), and the per-blob
loop body 0.06 s. Unlike §3, `over_into` already writes only into its window.
The windows are simply large at the 512² raster, so this is arithmetic, not
waste. No exact numpy fusion exists: each blob's fbm has its own seed, offset
and window.

A single C kernel per blob covering distance, fbm, coverage and over, with the
integer hash exact and the float32 operand order kept, is the survivor.
**Estimate:** 3–6×, since the shape is memory-bound more than dispatch-bound.
That is roughly 65–130 ms, so it may still miss the gate; bench a prototype
before committing the ABI.

## §5 — C5, RotSprite drag: `epx` is the cost, not the rotate

| side | `shipped` | `bands_merged` |
|---|---|---|
| 64 | 13.1 ms | 13.8 ms |
| 128 | 61.9 ms | 66.4 ms |
| 256 | **294 ms** | 295 ms |

At 256² that is 18× the gate on every mouse move. The cost is superlinear:
22.5× for 4× the pixels. Profile at 256² (0.33 s): `epx` 0.20 s cumulative, of
which `_packed` is 0.06 s, and PIL's rotate about 0.07 s. `bands_merged`
rotates pixels and mask per band after the same EPX rounds. It is exact, and
slower, so it is **rejected** and kept in the bench only so it is not proposed
again.

The survivor is a C `epx` round covering the integer compares and copies,
bit-exact by construction. **Estimate:** 5–10× on the EPX share, which leaves
PIL's rotate of the 8× plane as a floor of about 70 ms at 256². Reaching
16 ms means not building the 8× plane at all: inverse-mapped sampling at output
resolution. That requires transcribing Pillow's nearest-affine coordinate rule
exactly. It is the larger prize and the real parity risk, and it gets its own
bench case before any C.

## §6 — C6, Sirens: the decimator was the wrong suspect

| channels | `shipped` (45 s song) | ×4 → 3 min | `decimate_once` |
|---|---|---|---|
| 5 | 366 ms | ~1.5 s | 402 ms |
| 16 | 1 029 ms | ~4.1 s | 1 055 ms |
| 32 | 2 025 ms | **~8.1 s** | 2 031 ms |

The survey assumed that the per-tick streaming decimator was the cost, and that
its BLAS `@` would block exact parity. Both assumptions were wrong. Decimating
the whole mix once at the end is `array_equal` at all three widths, and no
faster. Profile at 16 channels (1.30 s): `voices.pulse` 0.28 s, `triangle`
0.23 s, `_render` 0.13 s, `_sound` 0.11 s, `noise` 0.10 s, and
`Decimator.process` only 0.13 s cumulative.

**Next step, before any kernel:** a `sirens_voices` bench case timing
`pulse`/`triangle`/`noise` at one tick's block size across a song's worth of
calls. It should compare the shipped generators with a generate-per-row (or
per-note) variant that cuts the ~25k per-tick calls. The dispatch count, not
the sample arithmetic, is the likely cost.

## §7 — C7, BVH pick: under the gate

64 picks per timed call. Per pick:

| tris | `grid_down` | `sphere_oblique` |
|---|---|---|
| 2 000 | 0.34 ms | 0.87 ms |
| 20 000 | 0.48 ms | 1.24 ms |
| 200 000 | 0.57 ms | **1.71 ms** |

The first draft of this case used only straight-down rays into a heightfield,
which visit a thin column of tight boxes and flatter the walk by 3×. The
oblique-ray, closed-sphere variant is the honest figure. It is under the gate
and flat (2× for 100× the work), which is the signature of dispatch-bound
cost: a C walk would still win big. But under the gate is under the gate.
Revisit only if imports above about 1M tris become normal.

## What this batch changed in the tree

`scripts/bench_native.py` only: seven new cases, their fixtures, and the
reference variants they time. Nothing under `src/`, no `native/*.c`, no ABI
bump. The bench sits outside `testpaths`.

Ranked follow-up:

1. `select.linked` via `csgraph` — SciPy, exact, 2 500× at the fixture (§1).
2. `stamp` windowed — Python, exact, 66× (§3).
3. `_min_distance`: choose between accepting the KD-tree (~20×, not
   bit-identical) and keeping exactness in C (est. 5–10×) (§2).
4. Kernel survivors, each to be benched as a prototype first: C `epx` round
   (§5), per-blob smoke kernel (§4).
5. A `sirens_voices` case before Sirens gets any kernel (§6).
