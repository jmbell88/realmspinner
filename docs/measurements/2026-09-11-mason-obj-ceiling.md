# Mason's merged-OBJ ceiling — 2026-09-11

`mason/objout.MAX_OBJ_VERTS` is the third constant Mason fixes that a stored
document is measured against, and the Mason programme's plan named it
as the one Stage D would have to decide. This repo's rule is
that such a constant gets a dated document here *before* it is fixed, so this
is that document.

It is a different shape of ceiling from the two in
`2026-09-11-mason-scene-ceilings.md`. `MAX_TERRAIN_SIDE` and `MAX_PLACED` are
both about what a *frame* can afford; this one is about what a single,
explicitly-requested export can afford on a task thread, where the budget is
not 16.7 ms and never was. So it needs its own criterion, stated below before
the numbers rather than fitted to them afterwards.

## The criterion

A merged OBJ is text, built in memory and handed to the io layer as bytes —
`plotter/tmx.tmx_export`'s "a mapping of paths" shape, which Mason's three
exporters all keep. So the constraint is **peak allocation**, not wall clock:
the whole file exists at once, and while it is being converted from the
accumulated text to the bytes the caller gets, so does most of a second copy.

The number that budget is measured against already exists in this codebase.
`viewer/gltf.MAX_TOTAL_BYTES` is 768 MiB — "what one *document* may decode to,
across every accessor and texture combined", the largest single allocation this
application deliberately holds open. An export that peaks past it is an export
asking for more memory than the app's own most expensive operation is allowed,
which is the wrong trade for a convenience format.

Secondary, and genuinely secondary: the wait. An export the user asked for by
name is allowed to take seconds. It is not allowed to take minutes, both
because the app has no progress for it and because a file that large is one no
DCC tool opens comfortably at the other end.

**So: the ceiling is the largest vertex count whose peak allocation stays
inside `gltf.MAX_TOTAL_BYTES`, and the wall clock is reported to confirm it
lands in single-digit seconds rather than to set the number.**

## The machine, and the shape measured

Windows 11, the checkout's own `uv` environment, `uv run --no-sync`. Best of
three, warm, as `2026-09-11-mason-scene-ceilings.md` measured its own table.

The shape is one merged OBJ's worth of text: `v`, `vn` and `vt` per vertex and
an `f a/b/c a/b/c a/b/c` per triangle, formatted at `.6f` — which is what
`objout.py` emits, line for line. **Triangles are twice the vertex count**, not
a third of it: a closed triangle mesh has about 2V faces by Euler's formula,
and the face block is the more expensive half to format. The first pass here
used V/3 faces and under-counted the file by a factor of about two; the table
below is the corrected one.

| vertices | triangles | seconds | output |
|---:|---:|---:|---:|
| 30,000 | 60,000 | 0.20 | 5.6 MB |
| 150,000 | 300,000 | 0.97 | 29.9 MB |
| 500,000 | 1,000,000 | 3.25 | 104.4 MB |
| **1,000,000** | **2,000,000** | **6.41** | **210.7 MB** |
| 2,000,000 | 4,000,000 | 13.17 | 441.4 MB |
| 4,000,000 | 8,000,000 | 26.19 | 902.8 MB |

Perfectly linear, at about 6.4 µs and 211 bytes per vertex. There is no knee to
find and no threshold the data itself suggests, which is exactly why the
criterion had to be written down first.

Peak allocation, `tracemalloc` around the same call:

| vertices | output | peak |
|---:|---:|---:|
| 250,000 | 51.2 MB | 179.1 MB |
| **1,000,000** | **210.7 MB** | **737.5 MB** |

**3.5x the output**, which is the number that decides this: the accumulated
text buffer, the string it is joined into, and the bytes it is encoded to are
all alive at the same moment.

## `MAX_OBJ_VERTS` = 1,000,000

At the ceiling the export peaks at **737.5 MB**, just inside
`gltf.MAX_TOTAL_BYTES`'s 768 MiB, and takes **6.4 seconds**. One step up — two
million vertices — peaks at roughly 1.5 GB, which is past that budget by a
factor of two and is the kind of allocation that fails on a machine running the
rest of this app at the same time rather than one that merely gets slow.

Two sanity checks on the figure rather than on the arithmetic:

**It is the same order as the geometry ceiling Clay already keeps.**
`clay/glbimport.MAX_TRIANGLES` is 2,000,000 for a document Clay will edit, and
a million vertices is two million triangles at the same Euler ratio. Mason
refusing an OBJ at about the point Clay refuses an import is a coincidence
worth noting rather than a derivation, but it does mean the two doors of this
app do not disagree by an order of magnitude about how much geometry is "too
much at once".

**It is not a ceiling a real scene reaches by being large.** A merged OBJ is
the one format here that *expands* instancing — five hundred placements of one
barrel are five hundred barrels of text, where the GLB writes one mesh and five
hundred nodes. So this constant is reached by instance count multiplied by
asset density, not by the placed-item count `PLACED_WARN_THRESHOLD` warns at:
1,500 items of a 10k-vertex generated asset is fifteen million vertices, which
this refuses and should. The refusal carries that sentence, because "your scene
is too detailed for OBJ" is actionable and "MemoryError" is not.

**The check runs before anything is formatted.** The vertex count is summed off
the resolved primitives first and the refusal raised there, which is
`gltf._Reader._charge`'s stated rule — the refusal happens *instead of* the
bytes, not alongside them. A ceiling checked after the allocation it is
guarding against is not a ceiling.

## What this does *not* fix

The GLB and the `.wscn` have no equivalent ceiling of their own and do not need
one from this document. A GLB writes shared geometry once, so it is bounded by
the document rather than by the flattening; a `.wscn` stores no geometry at
all. `MAX_PLACED` already refuses a document that resolves to an absurd number
of items, and that is the ceiling both of those formats sit behind.

## How to re-measure

The benchmark is throwaway and was run out of a scratch directory rather than
committed — it is thirty lines and the shape is described above precisely
enough to rebuild. What is worth keeping is the criterion, not the script:
**best of three, warm; peak allocation under `gltf.MAX_TOTAL_BYTES`; two
triangles per vertex; `.6f` floats; `v`/`vn`/`vt`/`f` exactly as `objout.py`
emits them.** A future change that claims to move this number should report
both tables — the timing and the `tracemalloc` peak — not a single figure, and
should say which of the two constraints it believes it moved.
