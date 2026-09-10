# Agent bridge round trip — what five commits of per-call work actually cost

2026-09-10. Machine: Windows 11, Python 3.13.13, `uv run python` against the
project's own `.venv`, `-n 0`, nothing else running. Every figure below is a
**median of 51-101 round trips in one process** (not a minimum -- this is a
per-call latency claim about what an agent experiences call to call, and a
single best run says nothing about that; see
`docs/measurements/2026-09-07-library-frame-times.md` for the same reasoning
applied to a UI frame time). The harness is a real `pipe.connect` against a
started `AgentHost`, with a background thread driving `pump()` the way
`main.py:App.frame` does -- the committed benchmark,
`tests/test_agent_perf.py`, is this same harness with the numbers turned
into assertions; this document is the measurement that test's budgets are
set from.

Five commits landed on this bridge just before this measurement
(`git log --oneline 9fd36833..HEAD`):

```
885f212b A misspelled argument is refused with the name it probably meant, not dropped in silence
2511d8b9 A refused call says whether anything moved, and a refused boolean stops eating your selection
56296141 A tool's answer is data as well as prose, so a client stops re-parsing the sentence a model reads
fecd4ae1 An agent that stopped listening can get the answer back, instead of doing the work twice
85d0d443 A call an agent gave up on is cancelled, not run half a minute late
```

Four of them added per-call work that runs **on the frame thread**, where
`AgentHost.pump`'s drain budget is 8 ms: a `blake2b` fingerprint over every
call's canonicalised arguments (`agent_host._fingerprint`), a linear scan of
the per-connection dedup store on every call (`_Calls.pending`), a
`json.loads` of every reply payload that was just `json.dumps`-ed
(`agent_clay._json`), and an unknown-argument check -- a set difference on
every call, plus a `difflib.get_close_matches` pass only when a name is
actually unknown (not exercised by this measurement -- every call below is
well-formed, which is the common case a live agent session spends nearly all
its time in). Each is obviously cheap on its own; nobody had measured them
together. This document answers, with numbers: what does one agent tool call
cost now, and what did these five commits add.

## Method

`HEAD` is `885f212b`. "Before" was measured in a separate `git worktree`
pinned to `9fd36833` (the commit immediately before this tranche begins --
`git worktree add --detach`, never `stash`/`checkout`/`reset` on the main
tree); "after" was measured directly against the main tree at `HEAD`. Both
ran under the main tree's own `.venv` (its interpreter invoked directly,
`D:\Projects\warlock\.venv\Scripts\python.exe`) with `sys.path` pointed at
the worktree's own `src/` for the "before" run, so the interpreter actually
imported the unfixed module tree rather than the editable install's real
target -- the same method
`2026-09-07-library-frame-times.md` uses, for the same reason.

**This was checked directly, not assumed.** The "before" script asserted,
before measuring anything:

* `warlock.__file__` resolves under the worktree path, not the main
  checkout's `src/`.
* `hasattr(agent_host, "_Calls")` is `False` -- the dedup store this
  tranche added does not exist yet.
* `hasattr(agent_host, "STATUS_TOOL")` is `False` -- `warlock_status` does
  not exist yet.
* `hasattr(agent_clay, "difflib")` is `False` -- the did-you-mean check's
  import is not present yet.

All four held. This matters because an editable install's default
resolution can silently import the *fixed* code from both directories and
make every "before" number in a document like this the after number by
accident -- exactly the failure mode `2026-09-07`'s own document calls out
as "the first thing that went wrong" there, and the reason this check is not
optional here either.

**Not apples-to-apples, and said plainly rather than smoothed over.**
`9fd36833` has no `warlock_status`, no dedup store (`_Calls`), no
did-you-mean check, and its `_json` is `ok(text(json.dumps(payload)))` --
one serialisation, no `structuredContent`, no second `json.loads`. So "the
same call" here means **the same request** -- an identical `tools/call`
frame for `clay_scene` (or `tools/list`) sent over the wire -- not the same
code path underneath. The comparison is honest about measuring the sum of
everything that changed in the handler, not just the four additions named
above in isolation; where that sum turns out to be negligible (see below),
isolating the four further was not needed to answer the question this
document asks.

Three shapes were measured, sent as real `tools/call`/`tools/list` frames
over the real pipe, warmed up with 3 discarded calls each so the first
import/allocation on the path is not what gets timed:

1. **`clay_scene` on a 1-object document** -- one `clay_add_primitive`
   (`generator: "box"`) first, then 101 timed `clay_scene` calls.
2. **`clay_scene` on a 50-object document** -- 50 `clay_add_primitive`
   calls first (untimed), then 51 timed `clay_scene` calls, each asserting
   `len(objects) == 50` so the payload actually carries the larger reply
   `_json`'s round trip has to serialise.
3. **`tools/list`** -- 51 timed calls; this is the one call whose cost is
   pure catalogue construction (`agent_clay.tools()` rebuilds all 25 `Tool`
   objects from the live registries every time it is asked, per its own
   docstring), nothing to do with a document.

## The numbers

| shape | before (9fd36833), median ms | after (HEAD), median ms | delta |
|---|---|---|---|
| `clay_scene`, 1 object | 1.71 (min 1.21, max 1.90) | 1.74 (min 1.18, max 2.09) | +0.03 ms -- within run-to-run noise |
| `clay_scene`, 50 objects | 5.63 (min 5.01, max 6.43) | 5.56 (min 5.13, max 6.48) | -0.07 ms -- within run-to-run noise |
| `tools/list` | 0.36 (min 0.34, max 0.55) | 0.51 (min 0.42, max 0.94) | +0.15 ms -- real, see below |

Each pair was measured three times back to back on the same machine, same
process invocation shape; the `clay_scene` numbers moved by a few hundredths
of a millisecond run to run in either direction (once "before" came out
*higher* than "after"), consistent with them being the same cost measured
twice rather than a real difference. `tools/list`'s gap was consistently
positive across all three repeats, ~0.12-0.15 ms every time.

## An independent re-run, on a busier machine

Re-measured after the benchmark was committed, from
`uv run pytest tests/test_agent_perf.py -m perf -n 0 -q -s` (the committed
test prints its own median, min and max for exactly this purpose), on the
same machine but immediately after a full `uv run pytest` run rather than
idle:

| shape | this document (idle) | re-run (busier) |
|---|---|---|
| `clay_scene`, 1 object | 1.74 | 1.63 |
| `clay_scene`, 50 objects | 5.56 | 6.43 |
| `tools/list` | 0.51 | 0.51 |

Recorded because it calibrates what noise looks like here, which a single
column of figures cannot: the 50-object shape came back ~16% higher, landing
almost exactly on the max of this document's own idle run (6.48), while the
other two reproduced closely. That is the spread a wall-clock median over 51
samples of a pipe round trip actually has on this machine under load -- and
it is the reason the budgets in `tests/test_agent_perf.py` are set at
multiples rather than percentages of these numbers, and the reason the
`perf` lane runs serially at all.

## What the `tools/list` delta actually is

Not the four additions above -- none of them run on the `tools/list` path
at all; `_fingerprint`, `_Calls.pending` and the unknown-argument check only
fire inside `tools/call`, and `_json` is not involved in building the
catalogue either. The explanation is smaller than that: `HEAD`'s catalogue
has one more entry. `9fd36833`'s `tools/list` returns `agent_clay.tools()`
directly (25 tools); `HEAD`'s `_serve` hands `dispatch` `[*agent_clay.tools(),
*agent_host._transport_tools()]` (26 tools -- `warlock_status` joins the
list), which is one more `Tool` namedtuple and one more `_tool_json(...)`
dict built on every call, plus the list-concatenation itself. Confirmed
directly: `len(agent_clay.tools())` is 25 on both checkouts;
`len(agent_host._transport_tools())` is 1 on `HEAD` and the function does
not exist on `9fd36833`.

## Conclusion

**The added cost is negligible.** Four commits added per-call work to the
frame thread -- a hash, a linear scan of a 16-entry-capped store, a second
JSON round trip, a set difference -- and together they do not move the
`clay_scene` round trip outside the noise floor of measuring the same call
twice, at either 1 or 50 objects. `tools/list` shows a real but tiny
(~0.1-0.15 ms) increase, and it comes from one more tool in the catalogue,
not from any of the four commits this document set out to weigh. At these
sizes, the round trip is dominated by pipe I/O and JSON framing overhead
(~0.4-1.7 ms baseline just to get a frame there and back) and, for
`clay_scene`, by walking `doc.objects` to build `_scene_row` for each one --
none of that is new here.

So what is the 8 ms frame-thread budget (`AgentHost.pump`'s default)
actually protecting against, if not this? Not today's cost -- today's cost
is roughly 1.7-5.6 ms of which the measured additions are an unmeasurable
sliver. It protects against **scale this measurement does not exercise**:
`_Calls` is capped at `MAX_REMEMBERED_CALLS` (16) precisely so `_Calls.
pending`'s linear scan and the store's memory stay bounded regardless of how
long a connection lives, and `clay_scene` on a document with hundreds or
thousands of objects (this document tested 1 and 50, not 5,000) would make
`_json`'s extra `json.dumps`-then-`json.loads` pass, and the underlying
`_scene_row` walk, cost proportionally more -- a `_grid`-style unbounded
per-frame cost is exactly the failure class
`2026-09-07-library-frame-times.md` measured at 5,000 Library cards. This
document did not test that regime for Clay because nothing in the five
commits under review changes with document size in a way the other four
additions do not already share, and TODO.md is the place a scale probe like
that would belong if it is ever wanted -- not cited from here as a
commitment, per the house rule.

## Caveats for a future reader

- This is a synthetic, minimal-document probe: one or fifty boxes, no
  textures, no real GLB export, no `Runtime`/`Worker` behind it. It isolates
  the bridge's own per-call overhead on purpose; it says nothing about a
  `clay_render` or `clay_export` call's cost, which do far more work per
  call than `clay_scene` or `tools/list`.
- The "before" numbers were captured by pointing `sys.path` at a `git
  worktree` while reusing the main tree's `.venv`, verified directly
  (`warlock.__file__`, and the absence of `_Calls`/`STATUS_TOOL`/`difflib`
  on the old tree) rather than assumed, for the reason given in Method.
- "Before" and "after" are not literally the same code path for `clay_scene`
  -- see the apples-to-apples paragraph in Method. What this document claims
  is that the *sum* of everything that changed in the handler between the
  two commits is within noise at these two document sizes, not that any one
  addition is individually free at every possible scale.
- `tests/test_agent_perf.py`'s own budgets (15 ms / 30 ms / 10 ms for the
  three shapes) are set at roughly 5-20x these measured medians, generously
  above rather than tight against them, per this codebase's usual `perf`
  convention (see `tests/inker/flourish/test_flourish_perf.py`) -- a modest
  CI box should pass comfortably, and a regression that makes one of the
  four additions scale badly (an unbounded dedup store, an
  accidentally-quadratic fingerprint) should fail there rather than first
  being noticed as a sluggish agent session.
