# Setup recipes — what actually produces the pictures

Every path and invocation below was read from the tree on 2026-09-07, not
carried over from memory. Re-verify anything that has since moved — the
constructs (`Runtime`, `create_job`, `require_no_live_writer`) are load-bearing
enough that a rename would be news, but line numbers are not promised.

## The headless drain

For anything that needs the job queue (P16, P19, P21, P32 all go through
ordinary jobs), boot the runtime directly rather than the app:

```python
from warlock.studio.runtime import Runtime
from warlock.service import jobs as svc_jobs

rt = Runtime()
svc = rt.start()                       # opens JobStore, the warlock-loop
                                        # thread and the service — no pygame,
                                        # no moderngl (studio/runtime.py's own
                                        # docstring: three threads, and the
                                        # frame loop is not one of them here)
try:
    job = svc_jobs.create_job(svc, kind="text", output="reference", ...)
    job_id = job["id"]
    # poll
    row = rt.store.get(job_id)
    while row["status"] not in ("finished", "error", "cancelled"):
        ...
        row = rt.store.get(job_id)
finally:
    rt.shutdown()
```

`Runtime.start()` opens the `JobStore`, starts the `warlock-loop` worker
thread and constructs the `WarlockService`, in that order
(`src/warlock/studio/runtime.py:_start`); `Runtime.shutdown()` tears all three
down in reverse, is idempotent, and is called from `start()`'s own exception
handler if anything above it raises. Measured throughput (RTX 5090, recorded
2026-09-03): ~15 s per `output="reference"` job at the 8-step LCM recipe,
dominated by the per-job `text2image_worker` child restart — that restart is
the documented VRAM handoff, not a defect, so do not try to avoid it.

**One process that both submits and drains, never two.** `scripts/campaign_*.py`
are submitters only: they call `service.sweeps.create_sweep` and exit, relying
on the *app's* worker to drain the rows later. `scripts/_campaign.py`'s
`require_no_live_writer` (confirmed at `scripts/_campaign.py`) opens a
`sqlite3.connect(db_path, timeout=2.0)` and issues `BEGIN IMMEDIATE`, which
takes the same reserved lock a running app's `JobStore` holds mid-commit, and
raises `SystemExit` naming the file if it cannot get it. It cannot detect an
*idle* app, only a writing one — the guard is against the dangerous case, not
proof the coast is clear. Two processes writing one rollback-journal sqlite
file is how a submit gets lost; a `Runtime` you started yourself and a
campaign script pointed at the same `WARLOCK_DB` are exactly that hazard.
Boot one `Runtime`, submit through `service.jobs.create_job` or
`service.sweeps.create_sweep` from inside that same process, and poll
`rt.store.get(...)` — never launch a campaign script alongside it against the
same database.

## The campaign submitters

Each is tied to a specific `docs/measurements/` instrument; do not invoke one
without reading the instrument it serves first.

| script | instrument | corpus / subjects |
|---|---|---|
| `scripts/campaign_props.py` | `docs/measurements/2026-08-30-art-verdicts-preregistration.md` (Q1) | `docs/measurements/corpora/props-v1.txt` |
| `scripts/campaign_detail.py` | `docs/measurements/2026-09-03-trellis-detail-sweep.md` | `docs/measurements/corpora/detail-v1.txt` |
| `scripts/campaign_guidance.py` | `docs/measurements/2026-09-02-trellis-guidance-sweep.md` | `--subjects <file>`, one bare prompt per line |
| `scripts/sweep_refill.py` | (companion to any sweep, not its own instrument) | operates on an existing `sweep_id`; re-queues only `cancelled` or shutdown-`error` units, never a genuine refusal — refusals are themselves a measurement and must not be re-rolled |

`campaign_detail.py`'s own docstring records a structural failure worth
knowing before queuing anything against it: the `decim0-*` rungs were retired
2026-09-06 because `--decim 0` produces ~35M faces and a dead
`trellis-server` at both 1M and raw resolution, ~29 minutes a unit either
way — do not re-add that axis without reading
`docs/measurements/2026-09-03-trellis-detail-sweep.md`'s amendment first.
This is also the incident P31 (Group 5, not built) exists to stop recurring
across a whole corpus.

Both `campaign_guidance.py` and `campaign_detail.py` queue one `SweepPlan`
*per subject*, because a plan carries exactly one prompt; every unit is
tagged so `scripts/hole_audit_vs_grade.py --tag <tag>` can tabulate the
campaign afterwards with the unit label beside each row.

## The corpora

`docs/measurements/corpora/*.txt` — one subject per line, format `class |
prompt`; blank and `#` lines ignored. Difficulty classes (`easy`, `medium`,
`hard`, `humanoid` in `props-v1.txt`) are fixed **before** the run, precisely
so a writeup can report per class as well as overall — an aggregate dragged
down by subjects already predicted hard is not a fact about the shipped
default.

**A corpus file is immutable once grades exist against it.** `props-v1.txt`'s
own header says so: *"Editing this file after the grades exist invalidates
that pre-registration -- start a new corpus file instead."* A retry at
different settings or seeds is `props-v2.txt`, never an edit to
`props-v1.txt`. This is a hard rule, not a style preference — the same
reasoning that makes a `YYYY-MM-DD-` writeup document immutable once
published.

**Blindness is structural, not a UI setting.** The corpus's difficulty class
is never written onto the job it produces — `review_mode.py`'s blinding
(below) only randomises presentation order and hides the *arm* on a two-arm
sweep; on a corpus like `props-v1` it cannot hide the subject at all, since
the mesh is the thing being looked at and the prompt names it. The actual
protection, per the 2026-08-30 pre-registration's own words: *"the difficulty
class is never written onto a job, so the reviewer cannot be told which
subjects were predicted hard."* Do not defeat this by labelling a picture
with its class when handing a gallery to the human — that is the one thing
that would turn a blind pass into a leading one.

## The bench

`python -m warlock.bench {suites, recipes, suite, run, score, compare,
calibrate, prune, purge}` (`src/warlock/bench/__main__.py`, confirmed)
is the real end-to-end harness — deliberately not folded into `warlock`'s own
CLI, because everything here is a developer tool and `warlock` is the
user-facing entry point.

- `suites` / `recipes` — list what exists.
- `suite <key>` — describe one suite (defaults to `suite_mod.DEFAULT_SUITE`).
- `run --recipe <r> [--suite <s>] [--stage reference|model] [--seeds ...] [--filter ...] [--ids ...] [--limit N] [--render] [--keep-source] [--resume <dir>] [--dry-run]` —
  `--stage reference` is ~10–45 min for 160 jobs; `--stage model` is 6–8 h of
  GPU. `--render` produces 8 views per finished mesh (+1–1.5 h) and is
  **required** before `score` can do anything.
- `score <run_dir> [--against <other_run_dir>]` — scores a finished run's
  rendered views; refuses with "Nothing was scored. A run needs `--stage
  model` and `--render`" if those were skipped. `--against` prints an A/B
  comparison against a second scored run.
- `compare <a> <b>` — one image against another: same picture, same shape,
  same subject.
- `calibrate [--job ... | --all] [--yaws N] [--elevations ...]` — sweeps
  yaw/elevation over finished meshes to find the matched view.
- `prune --keep N` — deletes all but the newest N runs.
- `purge <run_dir>... | --all` — frees disk by deleting a finished run's
  meshes, references and rendered views while keeping its items, manifest and
  scores.

Suites are never edited in place: changing a prompt is a new suite version,
because old manifests reference the fingerprint of the suite that produced
them, the same immutability rule the corpora observe.

## The in-app verdict loop — Review mode

Read from `src/warlock/studio/review_mode.py` directly (module docstring plus
the `handle_key` function and the `GRADE_KEYS`/`GOOD_TAG_KEYS`/`BAD_TAG_KEYS`
tables), because a paraphrase of this file is exactly the kind of thing that
goes stale first.

**The grade keyboard**, live outside a guided pass, once a unit is on screen:
- **`1`–`5`** file a grade of that magnitude, **+1..+5**.
- **`R`** arms the negative sign; the *next* digit then files **-1..-5**.
  Zero has no sign to arm, so `R` then `0` still files a plain `0`.
- **`0`** is its own key — "no opinion either way" is a real answer, not a
  refusal to give one.
- **`Ctrl`+`1`–`5`** toggles the positional *good* tag
  (`review_mode.GOOD_TAG_KEYS`, derived from `verdicts_mod.GOOD_TAGS`).
- **`Shift`+`1`–`5`** toggles the positional *bad* tag
  (`review_mode.BAD_TAG_KEYS`, from `verdicts_mod.BAD_TAGS`). Tags are staged
  and file *with* the next grade key pressed; anything that moves off the
  current unit drops the staged tags.
- **Left/Right arrows** step to the previous/next unit without recording
  anything.
- **`S`** (no modifier) advances/skips without recording.
- **`Esc`** disarms a pending negative sign (outside a pass).
- **`A`** is deliberately unbound in the ordinary loop — see below.

**The guided pass (`JudgingPass`) is a different, binary loop**, entered
deliberately from a button, and it owns `A`/`R`/`S`/`Esc` while open:
- **`A`** files `verdicts_mod.BINARY_GRADES["accept"]` (**+3**).
- **`R`** here means *reject* outright, filing `BINARY_GRADES["reject"]`
  (**-3**) — **not** "arm the negative sign" the way it does outside a pass,
  because a binary loop has no magnitude to arm.
- **`S`** skips within the pass (`judging_skip`).
- **`Esc`** **ends the pass** rather than disarming anything, because it is
  the key a person in a modal-feeling flow reaches for to leave.
- Digits, tag modifiers and the arrows still fall through to the ordinary
  branches, so a reviewer who wants "-5, holes" mid-pass can still say it and
  the pass still counts it.

**The labelling pass** is a third, separate two-key loop (`A`/`R`, arrows,
`Esc` to close) for the one-bit blank-image question; it owns the keyboard
while open and is unrelated to a mesh verdict.

**Why `A` is unbound outside a pass, stated in the source rather than assumed:**
"a mesh verdict is a grade now: silently mapping it onto +3 would file the
mildest usable grade every time somebody reached for the old key, which is a
wrong number rather than a missing one." Do not tell a human sitting in on a
verdict pass that `A` accepts outside a guided pass — it does not, on
purpose.

**Blinding is a session property**, not a sweep property: `review_mode.py`'s
docstring — *"it renames every unit to a neutral id prefix and presents them
in an order derived from a stable digest of the job id... It is not
persisted, for the reason nothing else here is: a review resumed blinded
without saying so is worse than one that starts unblinded."* Turn it on
per-session when handing a sitting to a human, and say plainly whether it is
on.

## The throwaway home

Copied verbatim from `.claude/skills/exercise-mode/SKILL.md` step 2, because
the rule is identical here — `WARLOCK_DATA_DIR` alone does **not** move the
sqlite store, so all three must be set together or a "throwaway" run seeds
the user's real `~/.warlock` library:

```powershell
$scratch = "<scratchpad>/sitting-<P-number>"
$home_   = "$scratch/home"
New-Item -ItemType Directory -Force $home_ | Out-Null
$env:WARLOCK_HOME     = $home_
$env:WARLOCK_DATA_DIR = "$home_/data"
$env:WARLOCK_DB       = "$home_/warlock.db"
```

**But say plainly when a sitting must run against the real library instead.**
Every entry in `references/entries.md`'s Group 1/1b/2 that reads from
`docs/measurements/corpora/*.txt` or from an existing sweep's rows (P16, P32,
via `campaign_props.py`/`campaign_detail.py`) needs the corpus's jobs to
already exist or to be queued into the library the human will open Review
mode against — a throwaway home would produce a gallery the human's own
Warlock Studio can never open. For those, pin nothing and say so: the run
happens against `~/.warlock` (or whatever `WARLOCK_HOME` the user's install
already uses), on purpose.

## Static geometry to pictures, with no queue and no card

**The gap this section closes, recorded 2026-09-07.** Every recipe above goes
through the job queue, a campaign script, the bench, or a real weight — but
P34 is exactly the "needs nothing but CPU and time" shape this skill
advertises, and none of them fit it: Clay's geometry is built in process, with
no job row at all, and turning it into a picture a human can look at is not
covered anywhere in this file. 0% of the render mechanism below came from a
recipe that already existed here; making it work the first time meant reading
roughly 150 lines against five modules the hard way. Copy the chain below
rather than re-deriving it.

**Build the geometry in process**, the same construction
`panes/clay_tools.py`'s `add_primitive` / `add_assembly` use — copy their
call shape rather than inventing a new one:

- `studio/clay/primitives.py`'s `GENERATORS` dict maps a generator name to
  `(defaults, make)`; call `make(**params)` (params built from `defaults`,
  overridden as needed) to get a `Mesh`.
- `studio/clay/presets.py`'s `build(key)` returns a whole figure (a tuple of
  `Part`) for the eight preset figures (`humanoid`, `quadruped`, `bird`,
  `blob`, and so on) — use this instead of `GENERATORS` for a figure rather
  than a bare primitive.
- `studio/clay/shading.py`'s `auto_smooth(mesh)` applies the same
  flat/smooth angle rule both insertion doors in the running app apply, so a
  primitive built by hand and never passed through this reads wrong beside
  what a user would actually see.

**Export to bytes:** `studio/clay/document.py`'s `to_model(doc)` turns a
`ClayDoc` into a `gltf.Model`; `studio/viewer/glbwrite.py`'s `write_glb(model)`
turns that into the GLB bytes. Assemble a `ClayDoc` from the built parts/mesh
the way the pane's insertion path does, rather than hand-rolling one.

**Render: there is no public entry point for a static, unrigged turntable.**
`pipelines/blender_worker.py` has two public ops and neither fits this job —
`op_views` is shaped for retexture baking (it expects a texture-bake context,
not a plain look-at-the-mesh render) and `op_sheet` is shaped for a rigged
animation sheet (it expects a rig and clips, neither of which a Clay
primitive has). What actually works is calling the module's private helpers
directly, in this order: `_reset_scene`, `_import_glb`, `_scene_bounds`,
`_setup_render`, `_make_lit`, `_setup_camera`, then `_aim_camera` once per
yaw you want a picture from.

**The caveat, stated as a caveat, not a footnote.** Depending on
underscore-prefixed helpers is fragile: they carry no compatibility promise,
and a rename inside `blender_worker.py` for an unrelated reason breaks this
recipe silently, with no test anywhere to catch it before a sitting does. The
durable fix is a public op — `op_turntable` or similar — or a real script
under `scripts/`; until one of those exists, this is what there is. Do not
build that script as part of running a sitting; it is out of scope here and
belongs to whoever picks up the durable fix.

## Contact sheets and zoom levels (P28 in particular)

P28 asks for every direction and every animation at zoom 1 and zoom 4: zoom 1
is the size a player actually sees the sprite at; zoom 4 is where a defect
that is invisible at native size becomes obvious. Lay out the gallery so the
human can move through it fast rather than hunting:

- One contact sheet per (archetype, animation) pair at zoom 1, all eight
  directions across the row, so a cross-direction identity break is visible
  in one glance left to right.
- The same grid again at zoom 4, directly beneath or in a paired file, so the
  human is never more than one scroll away from "why does that look wrong."
- Route the human through Troupe's own heatmap first (`studio/troupe/qa.py`'s
  flagged cells), since P28 itself says the heatmap "is the reading order, not
  the verdict" — click the flagged squares before watching the whole thing
  play, on both zoom levels.
- For P8's per-clip judging and P33's per-figure judging, the same shape
  applies at smaller scale: one sheet per clip/figure at native size, one at
  4x, in the order the entry's own brief lists the things to check, so the
  human is looking at the right region when the question names it.
