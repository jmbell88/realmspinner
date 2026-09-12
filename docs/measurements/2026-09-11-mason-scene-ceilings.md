# Mason's two stored-document ceilings — 2026-09-11

Mason's scene engine fixes two constants that a saved `.wscn` is keyed on:
`terrain.MAX_TERRAIN_SIDE`, which decides whether a document's ground can be
read back at all, and `scene.MAX_PLACED`, which decides whether a document's
tree can be resolved into something drawable. This repo's rule is that a
constant the stored corpus is keyed on gets a dated document here *before* it
is fixed, so this is that document. The Mason programme's plan recorded both
as unmeasured and said so outright: "no performance
numbers exist".

They exist now. Both were measured on the development machine against the
code that landed in Stage C — no extrapolation, no model of what the numbers
ought to be.

## The machine, and what a frame is worth

Windows 11, the checkout's own `uv` environment, `uv run --no-sync`. Every
timing below is the **best of N** rather than a mean: the interesting question
is what the operation costs when the machine is not doing something else, and a
mean over a noisy Windows box measures the scheduler.

The budget both constants are read against is **16.7 ms**, one frame at 60 Hz.
That is the whole frame, not this operation's share of it — which is what makes
"comfortably inside" rather than "under" the criterion below.

## `MAX_TERRAIN_SIDE` = 256

### What was measured

Side is cells per edge; the height array is `(side+1, side+1)` f4. The brush
figures are a radius-16 dab, which is the middle of a sculpt brush's range.

| side | height array | mesh rebuild | vertices | triangles | `raise_lower` | `smooth` | one `TerrainEdit` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.02 MB | 0.25 ms | 4,225 | 8,192 | 0.030 ms | 0.137 ms | 8.5 KB |
| 128 | 0.07 MB | 1.28 ms | 16,641 | 32,768 | 0.030 ms | 0.136 ms | 8.5 KB |
| **256** | **0.26 MB** | **5.65 ms** | **66,049** | **131,072** | **0.029 ms** | **0.135 ms** | **8.5 KB** |
| 512 | 1.05 MB | 28.38 ms | 263,169 | 524,288 | 0.029 ms | 0.135 ms | 8.5 KB |
| 1024 | 4.20 MB | 112.85 ms | 1,050,625 | 2,097,152 | 0.029 ms | 0.135 ms | 8.5 KB |

Rebuild timings past 256 are taken against `_build_mesh`'s own arithmetic
through a stand-in carrying the three fields it reads, because `Terrain` itself
refuses a side over the constant — which is the gate doing its job rather than
a gap in the measurement.

### What the numbers say

**Three of the five columns are not the constraint, and saying so is half the
result.** The height array is 0.26 MB at 256 and 4.2 MB at 1024; neither is a
number worth refusing a document over. A brush dab is flat at about 0.03 ms
across a sixteen-fold range of side, because a brush is local to its radius and
never touches the array it did not return — which is `plotter/tools.py`'s region
rule paying for itself, and it means no brush at any size this project will ever
allow is a frame cost. The undo step is 8.5 KB at every side for the same
reason: `TerrainEdit` records the affected rect's two sub-arrays and not the
height field, so `UNDO_BYTES` stays a meaningful budget instead of being spent
eight dabs at a time.

**The mesh rebuild is the whole constraint**, and it scales with side². A sculpt
drag rebinds `heights` on every frame it is open, and the mesh memo is keyed on
that array's identity, so every frame of a drag pays one full rebuild. At 256
that is 5.65 ms — about a third of a frame, leaving the rest of the frame for
the scene the terrain is under. At 512 it is 28.38 ms, which is not a share of a
frame but longer than a whole one: a drag there would run at roughly 35 Hz
before anything else drew at all.

So 256 is the largest measured side where a sculpt drag holds 60 Hz, and that
is the criterion. 512 fails it by itself.

### What the refusal costs a user

At one vertex per metre — the density at which a heightfield reads as ground
rather than as a mesh — 256 is a **256 × 256 m** plot. Mason's brief is placing
and lighting assets in a scene, not open-world terrain, and 256 m of ground
under a dressed set is not the binding constraint on anything the mode is for.
A user who wants more ground has the honest answer already: `size_x`/`size_z`
are independent of the array, so the same 256 cells can cover a kilometre at
four metres a cell.

### What would move it

The constant is gated by the **full**-rebuild strategy and by nothing else. A
partial rebuild — rewriting only the vertex rows a brush's rect touched, which
the brush already hands back as a rect — would take the per-frame cost off
side² entirely and make the array size the constraint instead, which is to say
make it a non-constraint. That is a real optimisation with a real design
(the memo would have to become per-region), and it is not Stage C's. Whoever
takes it can raise this number and should update this document when they do
rather than only the constant.

## `MAX_PLACED` = 100,000, and `PLACED_WARN_THRESHOLD` = 1,500

### Why this is two numbers

`scene.resolve` walks the whole document once per redrawn frame, and the first
draft of this constant was a single number set at the frame budget — the point
where resolving stops fitting in 60 Hz. That conflates two questions that want
opposite answers.

A ceiling that **refuses** is protecting against a corrupt `.wscn` or an array
op run with a count someone typed a zero too many into, where an unbounded walk
is a hang holding an unsaved document. A scene that merely resolves *slowly* is
not that: a user whose scene grew to 1,500 props should be told it runs at 40 Hz,
not that their own work cannot be opened. Losing access to a document is a much
worse outcome than a sluggish one, and a refusal set at the performance boundary
delivers exactly that.

So `MAX_PLACED` is the absurdity bound and refuses; `PLACED_WARN_THRESHOLD` is
the measured frame budget and refuses nothing. The second exists as a constant
beside the first, rather than as a number typed into a pane in Stage E, so the
warning the user sees and the measurement it rests on cannot drift apart.

### What was measured

`resolve()` wall time, best of 5, warm. Three scene shapes, because composition
cost is per level and prefab expansion re-reads its template on every walk:
**flat** (all nodes at the root), **deep** (eight levels of nested groups), and
**prefab** (instances of a five-leaf template).

| placed items | flat | deep, 8 levels | prefab instances | `Placed` list |
|---:|---:|---:|---:|---:|
| 100 | 0.32 ms | 0.35 ms | 0.42 ms | 0.02 MB |
| 500 | 1.56 ms | 1.68 ms | 2.11 ms | 0.09 MB |
| 1,000 | 3.09 ms | 3.30 ms | 4.26 ms | 0.18 MB |
| **1,500** | **4.51 ms** | 4.97 ms | **6.10 ms** | 0.28 MB |
| 2,000 | 6.46 ms | 6.83 ms | 8.63 ms | 0.35 MB |
| 5,000 | 16.16 ms | 17.77 ms | 21.98 ms | 0.88 MB |
| 10,000 | 32.76 ms | 35.27 ms | 45.92 ms | 1.76 MB |
| 50,000 | 196 ms | 189 ms | 225 ms | 8.82 MB |
| **100,000** | **393 ms** | 401 ms | **480 ms** | 18.40 MB |

Depth barely registers — eight levels of nesting costs about 5% over flat,
because the per-node Python bookkeeping dominates the extra 4×4 multiply.
Prefab expansion costs about 40%, which is the price of reading the template
through on every walk and is the whole reason there is no propagation step.
Memory was never close to binding at any size; wall clock alone decides both
numbers.

### The numbers that decide them

**`PLACED_WARN_THRESHOLD` = 1,500**, because 1,500 items in the worst shape is
6.10 ms — 37% of a 16.7 ms frame. That is the same "about a third of a frame,
with room for the rest of it" criterion `MAX_TERRAIN_SIDE` was set by above
(5.65 ms, 34%), and the two constants being set by one criterion is deliberate:
a scene at both ceilings at once spends about 70% of its frame in this package,
which is the real budget a user can reach.

**`MAX_PLACED` = 100,000**, because 100,000 items costs 480 ms a frame — about
two frames a second. That is not a slow editor, it is not an editor; and at that
size a document is far more likely a corrupt file or a runaway array than
anything a person placed. It is two orders of magnitude past the warn threshold,
which is the point: everything between the two numbers opens, works, and says so.

### A second measurement this replaced, worth recording

The first cut of this table put 500 items at 6.16 ms and would have fixed
`MAX_PLACED` at 500 — which is a number the plan's own headline contradicts,
since "five hundred instances one upload" is the thing instancing was supposed
to buy.

Profiling the resolver found why: 64% of it (11.08 ms of 17.18 ms over 2,000
nodes) was `Node.local()` rebuilding each node's matrix from its TRS, almost all
of that in `m3.quat_to_mat4` — recomputed every frame for nodes that had not
moved, which is the overwhelmingly common case of an orbiting camera over a
still scene. Memoizing `local()` on the identity of its three transform arrays,
the way `clay_view._world` already does and sound for the same reason (every
transform write rebinds those arrays rather than writing through them), made the
cache-hit path **20× faster**: 10.75 ms → 0.52 ms over 2,000 nodes.

The lesson is the one this directory exists for. A constant that stored
documents are keyed on was about to be fixed to an artifact of an inner loop
nobody had profiled, and the number would then have been unmovable without a
format decision. Measure the thing, then fix the constant.

## How to re-measure

Both benchmarks are throwaway and were run out of a scratch directory rather
than committed — they are twenty lines each and the shapes are described above
precisely enough to rebuild. What is worth keeping is the criterion, not the
script: **best of 5, warm, against a 16.7 ms frame, in the worst of the three
scene shapes.** A future change that claims to move either number should report
that same table, not a single figure.
