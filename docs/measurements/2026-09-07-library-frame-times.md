# Library frame times — the uncommitted virtualization/threading/index fix

2026-09-07. Machine: Windows 11, Python 3.13.13, real GL 3.3 standalone
context (`moderngl`, no display), a real (headless) imgui context, sqlite's
bundled amalgamation. Every figure below is a **median of 9–15 runs in one
process** (not a minimum — this is a UI frame-time claim, and a single best
run says nothing about what the user actually sees frame to frame); the raw
run lists are kept in the probe's `-s` output so a re-run can be compared
sample-for-sample, not just median-for-median. `-n 0`, nothing else running.

The change measured here is **uncommitted** in the working tree (`git status`
clean of everything but this session's own edits after that). `HEAD` is
`8f5c63cb`. "Before" was measured in a separate worktree pinned to that
commit (`git worktree add`, never `stash`/`checkout`/`reset` on the main
tree); "after" was measured in the working tree as it stands. Both used the
same probe file (temporary, not committed —
`tests/test_zzz_library_perf_probe.py`, deleted after this document was
written) and, to keep the comparison to the *code* rather than a rebuilt
environment, the worktree ran against the main tree's `.venv`
(`UV_PROJECT_ENVIRONMENT`) with `PYTHONPATH` pointed at the worktree's own
`src/` so the interpreter actually imported the unfixed module tree, not the
editable install's real target. This was checked directly (`warlock.__file__`
and `hasattr(JobsCache, "read")`) before trusting a single number, because an
editable install silently importing the *fixed* code from both directories
was the first thing that went wrong.

## Method

Synthetic cached cards: `svc.store.create(..., status="done")` for N in
`{200, 1000, 5000}`, each given a distinct `created_at` and a real job
directory holding a 2 KB `model.glb`, so `attach_files`' stat/listdir walk
does real filesystem work rather than hitting an early "not found" path. This
is the review's probe shape (card content is irrelevant to a layout/paging
cost; only count and on-disk presence are).

Three things were timed separately, because the change is three parts and
collapsing them into one number would have hidden which one is carrying the
result:

1. **`library_full._grid(ctx, jobs)`** alone, through a real `_frame()` (an
   actual `imgui.new_frame()`/`end`/`render` cycle against the GL-backed
   renderer used in `tests/test_library_browsing.py`), with the job list
   already loaded — this isolates row virtualization (`_row_layout`/
   `_row_clipper`) from the cache refresh.
2. **The cache refresh's frame-thread cost**: `JobsCache.tick()` on the
   before checkout (the whole read ran inline, which is exactly the
   complaint) versus `JobsCache.read()` (now handed to `TaskRunner.submit`,
   timed here inline only to show the work still happens) and
   `JobsCache.adopt()` (what actually runs *on* the frame thread once a
   background read finishes) on the after checkout.
3. **`idx_jobs_created_id` alone**: `store.list(200)` over a 5000-row table
   with the index present, then with both `idx_jobs_created_id` and
   `idx_jobs_created` dropped on the same live connection — isolates the
   index from paging and threading entirely.

## 1. Grid virtualization — `_grid` draw time

| N | before (ms) | after (ms) | speedup |
|---|---|---|---|
| 200 | 16.4 | 5.95 | 2.8× |
| 1,000 | 75.7 | 7.77 | 9.7× |
| 5,000 | **364.9** | 16.90 | 21.6× |

The review's headline number was **372 ms/frame at 5,000 assets, 18.7 ms at
200, 77.4 ms at 1,000**. This reproduces closely — 364.9 / 16.4 / 75.7 ms
here, a few percent off in both directions, consistent with a different
machine rather than a different claim — and it lands almost entirely on
`_grid` alone: the review's number is not "the grid plus a periodic
hitch", it is the grid, every single frame, because the unfixed `_grid` drew
every card regardless of scroll position. This is the part of the fix that
answers the review's own number directly, and it is the largest of the three
by a wide margin: **21.6× at 5,000**, and the win only grows with N (row
clipping is the win; there was never a bound on the unfixed cost).

## 2. JobsCache off the frame thread — where the read's cost is paid

| N | before: `tick()` blocks the frame thread (ms) | after: `read()` (background thread) (ms) | after: `adopt()` on the frame thread (ms) |
|---|---|---|---|
| 200 | 6.96 | ~75 (flat, see below) | 0.11 |
| 1,000 | 33.32 | ~76 | 0.30 |
| 5,000 | **166.37** | ~76 | 1.15 |

This is not a story about the read getting cheaper — it mostly doesn't
(`read()`'s ~75 ms is dominated by `attach_files`' per-row stat/listdir walk
over freshly-created directories, which this probe forces to be cold-ish; see
`attach_files`'s own docstring on why one row costs "upwards of two thousand"
syscalls at 200 rows). It is a story about **which thread pays it**. Before,
`tick()` ran inline on the frame loop and scaled with the loaded window
(6.96 → 166.37 ms as N grew, because the unfixed code re-read and re-stat the
*whole* window every tick). After, the same real work still happens, but on a
`TaskRunner` worker thread; what the frame thread actually executes is
`adopt()`, which only publishes an already-finished result, and that stays
under 1.2 ms even at 5,000. So this part removes a *periodic stall* (every
0.5–3 s, per `REFRESH_SECONDS`/`IDLE_REFRESH_SECONDS`) rather than lowering a
per-frame cost the way part 1 does — a real fix, but for a different symptom
(the intermittent hitch on tick, not the steady per-frame grid cost the
review's 372 ms number was about).

One honest surprise: `read()`'s ~75 ms did **not** scale noticeably with N in
this probe (200, 1000 and 5000 rows all landed in the 71–83 ms band). The
keyset pager reads the newest 200 unconditionally and pages the rest in
200-row chunks; at these N the fixed 200-row `attach_files` cost per page
dominates and the OS's own directory-metadata cache (these are all
freshly-created synthetic directories from the same test process) likely
makes the later pages nearly free. This is a property of the *page size*,
not evidence the paging change stopped mattering — a longer, colder,
truly 5,000-row-scrolled-back history would look different, and this probe
does not speak to that case.

## 3. `idx_jobs_created_id` — isolated

`store.list(200)` over a 5,000-row table, same connection, index dropped
between runs so nothing else changes:

| | median (ms) |
|---|---|
| with `idx_jobs_created_id` | 1.09 |
| without it (and without the older `idx_jobs_created`, so sqlite has no
covering index left for the ORDER BY at all) | 5.85 |

**5.4× on the raw query alone** — real, and `EXPLAIN QUERY PLAN`'s own text
(`USE TEMP B-TREE FOR ORDER BY` gone once the index exists) confirms the
mechanism the migration's comment names. But at 5,000 rows this is 1–5 ms
against a ~75 ms `read()` and a ~17 ms `_grid` — **this is the part that does
not matter measurably at the sizes this document was asked to check.** It
would matter more as the table grows past what a 5,000-row synthetic corpus
tests (sqlite's in-memory sort of 5,000 short rows is already cheap), and it
costs nothing to have, but at 5,000 assets it is not why the Library feels
different after this change. Say so rather than folding it into the other
two numbers to make a rounder story.

## Aggregate, honestly

If you only want one number per size — the steady-state cost of a Library
frame with the window already loaded, which is what part 1 measures and what
the review's number describes — it is the table in §1. The periodic tick
hitch in §2 is a separate, intermittent cost that does not show up in a
"frame time" measured mid-scroll, only in the frame where the 0.5 s (or 3 s
idle) timer fires; before this change that frame could cost the `_grid`
number *plus* the `tick()` number on top (as much as ~530 ms at 5,000,
worse than the review's steady-state 372 ms), and after this change that same
frame costs the `_grid` number plus ~1 ms.

## Caveats for a future reader

- This is a synthetic-card probe, not a full app frame: no thumbnail texture
  upload, no real GLB, no `Runtime`/`Worker` behind it. It isolates the three
  changed code paths on purpose; it is not a substitute for
  `scripts/exercise_mode.py` on Library with a real asset library.
- The "before" numbers were captured by pointing `PYTHONPATH` at a
  `git worktree` while reusing the main tree's `.venv` — verified directly
  (`warlock.__file__`, `hasattr(cache, "read")`) rather than assumed, because
  the editable install's default resolution silently returns the *fixed*
  code and would have made every "before" number in this document the
  after number by accident.
- Part 2's `read()` flatness (§2) is reported as observed and only partially
  explained; a reader who needs the paging change's contribution at a colder,
  larger history should re-run this probe against real accumulated jobs
  rather than trust the flat 75 ms figure to generalize.
- The probe file this document is built from was deleted after this document
  was written (per the house rule against leaving throwaway scripts in the
  tree); its content is reproducible from the method above, and its exact
  git history exists in this session's transcript if it needs to be rebuilt.
