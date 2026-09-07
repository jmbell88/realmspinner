# Sweeps — one row per repeated addition

Every path below was read against the tree at HEAD `591836f2` (2026-09-07); an invented
path is worse than an omitted one, so where a symbol had moved from what an earlier
brief claimed, the correction is written into the row rather than silently fixed. Paths
are under `src/warlock/` unless they start with `tests/`, `docs/` or a bare filename
(`TODO.md`). *INVARIANTS lead-in* gives the bold sentence to grep for in
`docs/INVARIANTS.md`, since it is a flat list of paragraphs and line numbers move.

## job-kind

**What it is.** Adding a new value of `job["kind"]` — a new thing the queue can be asked
to do. Do this when a new worker function needs its own row in `jobs`, distinct from an
existing kind's own follow-up (see `followup-kind` below for those instead).

**Sites, in the order a row meets them.** Sites 1-11 are the "sweep of stage-keyed
tables" paragraph's own list, in its own order; sites 12-14 are the rest of `queue.py`'s
"four edits" paragraph, whose fourth edit (progress) is the same site as #3 above and is
not counted twice. INVARIANTS calls it *ten tables* because three of them are pairs that
it names together — `primary_action`/`card_kind`, `_remeshable`/`rerollable`, and
`STAGE_BADGES`/`thumb_glyph` — and each half is a separate edit you can forget on its own,
so this row splits them. Expanded that way the true count is **fifteen sites across
thirteen existing files, plus one new module**.

 1. `src/warlock/service/files.py` — `ready()` / `unready_reason()`. Whether the kind's
    artifact may be served at all, gated on its own sidecar existing.
 2. `src/warlock/service/files.py` — `derived_2d_for()`. Which 2D exports the stage can
    produce; an empty tuple is a legitimate, deliberate answer (a tile sheet returns
    `()` because every entry in that set is a whole-frame transform and a sheet is
    sixty-four frames) — so "I returned `()`" is a valid, checked answer, not a skip.
 3. `src/warlock/progress.py` — a new `PHASES_<KIND>` dict registered in
    `_PHASES_BY_KIND`, read through `phases_for()`. Same site as `queue.py`'s fourth
    edit below; do not create it twice.
 4. `src/warlock/studio/state.py` — `primary_action()`. The card's one button for a job
    of this kind and stage.
 5. `src/warlock/studio/state.py` — `card_kind()`. Which library filter bucket the row
    sorts into.
 6. `src/warlock/studio/panes/library.py` — `_remeshable()`. Whether the overflow menu
    offers Remesh; must never offer an action `service` would refuse.
 7. `src/warlock/studio/palette.py` — `rerollable()`. The command palette's version of
    the same rule.
 8. `src/warlock/studio/create_stages.py` — `IMAGE_STAGES` (does this kind's stage open
    in Create's Reference stage rather than Mesh) and `available("mesh", …)` (if the
    kind cannot promote to a mesh, that function must refuse it **in its own words**,
    not merely by omission).
 9. `src/warlock/studio/widgets.py` — `STAGE_BADGES` and, if the kind ships its own
    exportable-files list, an `ARTIFACTS_<KIND>` constant; `src/warlock/studio/panes/thumbs.py`
    — `thumb_glyph()`. Together, what the card looks like. Miss this and the card draws
    the fallback CIRCLE glyph next to the raw stage string — Muse's own miss.
10. `src/warlock/service/validation.py` — `DERIVED_PARAMS`. Whatever the worker records
    about its own run (a report, a measured seed) that must be stripped on rerun or
    promotion, or a promoted row inherits a stale, worker-written value describing a
    run that never happened for it.
11. `src/warlock/studio/asset_open.py` — `route()`. Where "Show" on a finished row of
    this kind actually opens. This is the table INVARIANTS calls out by name as the
    tenth, forced onto the list by Muse, "the first kind whose surface is a mode rather
    than a Create stage". A kind whose card lives in a workspace rather than a Create
    stage (Muse, Sirens-adjacent, a future case) needs its own branch here or "Show"
    opens a blank Create screen.
12. `src/warlock/_q_generate.py` — `Worker._generate`. The dispatch branch itself; an
    unhandled kind here is a job that sits `queued` forever with nothing running it.
13. `src/warlock/_q_jobs.py` — `Worker._discard_artifacts`. The cancel-cleanup branch,
    and it is the one site with a trap the others do not have: it "must name only this
    job's own files" — a kind that writes into a *predecessor's* directory (see
    `followup-kind`) must delete only its own temp names on cancel, or a cancelled
    follow-up destroys a different, already-successful job's artifacts.
14. `src/warlock/vram.py` — `estimate()` / `estimate_parts()`. Admission pricing; an
    unlisted kind falls through to `return 0.0`, so `check_vram` admits a job that can
    OOM the process at load — "the admission gate becomes a no-op that refuses
    nothing."
15. A new `src/warlock/_q_<kind>.py` module. Ten exist today, confirmed against the
    tree: `_q_mesh.py`, `_q_lora.py`, `_q_generate.py`, `_q_jobs.py`, `_q_troupe.py`,
    `_q_rig.py`, `_q_sprite.py`, `_q_tilesheet.py`, `_q_tileset.py`, `_q_music.py`. Read
    the shape of the nearest existing one before writing an eleventh.

**What a miss looks like, concretely.**

- **#1 files.ready** — the artifact silently never becomes downloadable (`ready()`
  defaults false for a name it does not recognise), or is exposed to `job["files"]`
  before its sidecar exists.
- **#2 derived_2d_for** — the stage's export menu offers a transform that does not
  apply to this kind's artifact, or offers none when one is possible. A silent `()`
  default reads identically to a deliberate one; only the row in this file tells them
  apart.
- **#3 progress** — CON-02. The unregistered kind falls back to `PHASES_IMAGE`,
  whose only real phase is `trellis`; an unknown phase maps onto the *whole* bar, so
  the last real step of the first phase reports 100% and the never-regress creep pins
  it there. Sharpest on a single-generation kind (`tile_sheet`): "there is nothing to
  average the fallback's errors out." This is exactly Troupe's `charsheet` miss.
- **#4/#5 primary_action / card_kind** — a finished card offers no action at all
  (Muse's own miss: "its ladder walking past every arm looking for a `model.glb` it
  will never have"), or files under the wrong filter bucket so the library's own count
  for that bucket is wrong.
- **#6/#7 _remeshable / rerollable** — the context menu or the command palette offers
  Remesh or Reroll on a kind `service` will refuse it for, which is a refusal toast on
  a click that looked legitimate.
- **#8 create_stages** — a blank arrival: the row opens in Create's Mesh stage with no
  mesh, no files, no thumbnail and every download button disabled (this is the exact
  shape `followup-kind` documents below, for a *different* reason — the stage-routing
  default rather than a missing `asset_open.route` branch — so a `create_stages` miss
  and a `route` miss can look identical on screen and have to be told apart by reading).
- **#9 STAGE_BADGES / thumb_glyph** — the fallback CIRCLE glyph with the raw stage
  string ("pixel_sheet", not a picture), Muse's literal defect.
- **#11 asset_open.route** — "Show" opens a blank Create screen for a kind whose real
  home is a workspace.
- **#12 Worker._generate** — the row never runs; it sits `queued` with no error.
- **#13 Worker._discard_artifacts** — either a cancel leaves orphaned temp files
  forever (nothing names them for cleanup), or — the sharper failure — the cleanup is
  scoped to the *source* job's whole directory and a cancelled follow-up deletes a
  different, successful job's `model.glb`.
- **#14 vram.estimate** — the kind is admitted with no VRAM check at all and can OOM
  the process on a card that would have refused it honestly.

**Which sites already have a gate, named.**

- #3 (progress) has a **real but partial** gate:
  `tests/test_progress.py::test_the_multi_pass_kinds_have_their_own_contiguous_tables`
  is parametrised over `["pixel_sheet", "retexture", "tile_sheet", "charsheet"]` and
  checks that the kind's table is not `PHASES_IMAGE` and that its spans start at 0.0 and
  end at 1.0 — but it does **not** check that the spans are gap-free in between (that
  stronger check exists only for one kind, `test_the_sprite_phases_cover_the_whole_bar_contiguously`
  against `PHASES_SPRITE`, and is not parametrised). A new multi-pass kind must be
  **added to that parametrize list by hand** — it is not discovered — so an addition
  that forgets the line is exactly as invisible as the four before it were.
- Every other site (#1, #2, #4–#15) is **prose only**. There is no scan or parametrised
  test that walks `progress`'s sibling tables the way `test_progress.py` walks its one
  table, which is the entire reason the INVARIANTS paragraph exists as prose rather than
  as a passing test — read it as the standing offer this skill's step 5 is for.

**INVARIANTS lead-ins to grep:** `"A new job kind is a sweep of the stage-keyed tables"`
(the ten-table list, the Muse and Troupe miss rates) and `"A new job kind is four edits,
not one"` (the `queue.py` quarter, `sprite_synthesis` as the worked example).

---

## phase

**What it is.** Adding a phase to a kind that **already has** a `PHASES_<KIND>` table in
`src/warlock/progress.py` — the sub-ritual INVARIANTS calls "the edit that looks too
small to check." Do this when a kind's worker function grows a new pass (a new
composite step, a new render sub-stage) rather than when the kind itself is new (that is
`job-kind`, site #3/#14, above).

**The one site, and its own rule.** `_PHASES_BY_KIND[kind]` is a `dict[str, tuple[float,
float]]` — start and end fractions of the progress bar, one pair per named phase. Adding
a phase means:

1. The new phase's span comes **out of** the neighbour it is squeezed between, never
   appended after the last one — appending would leave the bar jumping backwards the
   first time a job actually reaches that phase, since every span before it would then
   overrun 1.0 or leave a gap.
2. The five (or however many) spans must stay **contiguous**: sorted by start, each
   span's end must equal the next span's start, the first must start at 0.0, and the
   last must end at 1.0.
3. Whatever code path emits the new phase's `on_step`/`on_progress` calls must be
   updated in the same change — a phase declared here with nothing ever calling
   `ProgressBus.update(..., phase="the-new-one", ...)` is dead weight, and a phase
   *called* with nothing declared here is the CON-02 fallback all over again.

**Worked example.** `PHASES_CHARSHEET` gained `effects` on 2026-09-05, between `reduce`
and `pack`, and its span came out of `reduce`'s rather than being appended — because
`_maybe_queue_charsheet`'s render composites a themed effect (fire, for the three
`fire` themes) in the gap between the frames reaching their final logical size and the
atlas being packed, and a bar that stalls in an undeclared phase for 256 composites is
exactly the defect this rule exists to prevent. Read `src/warlock/progress.py`'s
`PHASES_CHARSHEET` definition and the paragraph beside it before adding a phase to any
other table — it is the one place in the tree that states the "comes out of a neighbour"
rule in prose.

**What a miss looks like.** Same shape as `job-kind`'s #3/#14: the new phase's calls
land in an *undeclared* name, `ProgressBus.update` falls back to `(0.0, 1.0)` for it,
and the bar either jumps to 100% early (if the undeclared phase is late in the run) or
snaps back toward 0% (if it is early) — the "drag the bar back to zero" case INVARIANTS
names for `_sprite_synthesis`. A phase appended rather than carved out of a neighbour
overruns 1.0 across the remaining spans, which is the same visible symptom from the
other direction.

**Gate.** `tests/test_progress.py::test_the_sprite_phases_cover_the_whole_bar_contiguously`
is the strict, non-parametrised shape (checks true contiguity, not just the two
endpoints) but exists for exactly one table, `PHASES_SPRITE`. Extending a *different*
table's phases has no equivalent strict check today — only the weaker, four-kind
`test_the_multi_pass_kinds_have_their_own_contiguous_tables` from the `job-kind` row
above, which does not verify contiguity between spans at all. Writing (or generalising
the sprite one into) a contiguity check for the table you are extending is exactly the
gate step 5 of the main procedure asks for.

**INVARIANTS lead-in to grep:** the same `"A new job kind is a sweep of the stage-keyed
tables"` paragraph, its closing two sentences ("A phase added to a kind that already
has a table is the same sweep at one table... Adding a job *kind* sweeps ten tables;
adding a *phase* to one is a single edit").

---

## followup-kind

**What it is.** A kind whose rows carry `params["source_job"]` — a job the *worker*
mints as a byproduct of another job, rather than one the user directly requested through
`create_job`. Do this when a new worker action produces an artifact that lives beside an
existing job's own directory rather than in a directory of its own.

**The six kinds, verified against the tree.** INVARIANTS names `rig`, `sheet`,
`pixel_sheet`, `sprite_synthesis`, `charsheet` and `retexture`. `src/warlock/studio/asset_open.py`'s
own `FOLLOWUP_STAGES` table lists five of those six plus a sixth, **`remesh`**, in place
of `charsheet` — and that is not a drift to fix, it is `asset_open.py`'s own documented
design: `charsheet` is "deliberately absent" from `FOLLOWUP_STAGES` because it opens in
**Troupe**, a workspace, not a Create stage, and is routed by its own `if kind ==
"charsheet"` branch in `route()` instead of through the table. So the six kinds with
`params["source_job"]` and the six entries of `FOLLOWUP_STAGES` are two different
"sixes" that overlap in five places — read both lists before assuming either one is the
complete answer for a new kind.

**This is a different sweep from the automatic-queueing guard**, and the two are easy to
conflate because both are called "follow-up" in the tree. This row is about *where a
finished row opens*. The separate guard — "every automatic follow-up's queue attempt is
guarded on the stage that can produce its input" (`_maybe_queue_rig`,
`_maybe_queue_sprite_sheet`, `_maybe_queue_charsheet`, `_maybe_queue_sheet_after_rig`,
all in `src/warlock/_q_jobs.py`, called from `queue.py`'s per-kind `finally` block) — is
about *whether the worker mints the row at all*, and it has its own two rules (guard on
the stage that produces the input; guard against re-triggering on your own kind) that a
`job-kind` addition which is *also* an automatic follow-up must satisfy in addition to
this row. Treat that guard as part of `job-kind` site #11 (`Worker._generate`'s sibling
dispatch) for a kind that mints itself automatically, and use this row only for "where do
its finished rows open and what can they become."

**Sites, in the order a row meets them.**

1. `src/warlock/studio/asset_open.py` — `FOLLOWUP_STAGES`. Maps the kind to the Create
   stage its row should open on (`"rig"`, `"pose"`, `"mesh"`, `"reference"`). Absent
   entirely for a kind whose home is a workspace rather than Create — see `charsheet`.
2. `src/warlock/studio/asset_open.py` — `route()`'s own special-case branches, for a
   kind whose routing needs more than the table above (a `kind ==` check ahead of the
   table lookup, as `charsheet` and `music` both need).
3. `src/warlock/studio/asset_exits.py` — `exits_for()` and its `_is_mesh` predicate
   (`stage == "model" and not params["source_job"]`). The "everywhere this asset can
   go" sibling of `route()`'s "where does this row open" — a charsheet keeps the
   `stage` column's default `'model'`, so without the `source_job` check it would be
   offered a dimmed *Send to Troupe* for a mesh it does not have.
4. `src/warlock/followups.py` — `APPLICABLE_STAGES`. The reader-side guard: a stored
   `params` record about a follow-up a row's stage could never have had is "the
   fingerprint of a missing guard, not evidence," and is filtered out on read rather
   than migrated out of old rows.
5. The mint site itself, in `src/warlock/_q_jobs.py` — a `_maybe_queue_<name>` method,
   called from `queue.py`'s per-kind `finally` block (see `_maybe_queue_rig`,
   `_maybe_queue_sprite_sheet`, `_maybe_queue_charsheet`, `_maybe_queue_sheet_after_rig`
   for the four that exist today) — guarded on the stage that produces its input *and*
   on its own kind, so it cannot re-trigger itself.

**What a miss looks like.**

- **#1/#2 missing** — the "open this row" blank screen: a finished follow-up's row has
  no mesh, no files (`job["files"]` is empty because nothing it writes is in
  `service.files.LISTED`) and no thumbnail; `create_stages.stage_for` routes the
  column-default `stage='model'` to Create's Mesh stage, whose `_STAGE_SECTIONS` holds
  neither the sprite panel nor the sheet panel, and the toast that said "finished" leads
  to eight disabled download buttons while the actual artifact sits one directory over.
- **#3 missing** — a destination offered that the asset cannot reach (a charsheet
  offered a live *Send to Troupe* for a mesh it has none of), or, the opposite failure
  named in the same paragraph, two surfaces (the library's overflow menu and the
  inspector's edit actions) disagreeing about the same asset because each hand-rolled
  its own list instead of both calling `exits_for`.
- **#4 missing** — a stored `params` block from a stage that can no longer produce it
  is read as if it were current evidence, rather than filtered as the fingerprint of a
  bug that has since been fixed elsewhere.
- **#5 missing or unguarded** — either the follow-up never fires (survivable only while
  the miss is silent, which is exactly the trap: `_maybe_queue_rig` had no guard at all
  until it was found), or it fires on the *wrong* stage and searches for an artifact
  that cannot exist yet, writing a permanent false "the generated mesh artifact is
  missing" onto a row that never should have been checked — this happened twice, once
  for a missing stage guard and once for a guard that excluded only its own literal
  kind string rather than every stage the mint should not fire from.

**Which sites already have a gate.** None of the five is a scan or a parametrised test
today; `tests/test_asset_exits.py` (see the "overflow" grep hit in the shell segment)
tests `exits_for`'s *behaviour* per asset shape but is not swept per follow-up kind the
way `test_progress.py` sweeps per job kind. This row is prose-only end to end, which
makes it the sharpest candidate in this file for the gate step to actually add
something.

**INVARIANTS lead-ins to grep:** `"A row with \`params[\"source_job\"]\` is a product of
another asset, and opens where that asset is"` (the six kinds, `FOLLOWUP_STAGES`,
`asset_exits.exits_for`) and, for the separate queueing guard,
`"Every automatic follow-up's queue attempt is guarded on the stage that can produce its
input"`.

---

## mode

**What it is.** Adding a fourteenth top-level mode to the rail. Verified against
`src/warlock/studio/modes.py` at HEAD: there are thirteen today (`home`, `library`,
`create`, `inker`, `clay`, `poser`, `troupe`, `plotter`, `packwright`, `muse`, `sirens`,
`review`, `settings`), and the module's own docstring states the design rule directly:
"the order-and-grouping tuple is not the only thing every reader of this file needs" —
`MODES` is one four-tuple list and every other structure below it is *derived* from that
list or is its own hand-written partition that a test holds against it.

**Sites, in the order a mode is added.**

1. `src/warlock/studio/modes.py` — `MODES` (the four-tuple: key, label, icon, one-line
   purpose). This alone drives `KEYS`, `PURPOSE` and the rail's default drawing order —
   they are `derived`, not separately maintained.
2. `src/warlock/studio/modes.py` — `RAIL_GROUPS` (which of the three sections — Pipeline,
   Workspaces, footer — the new mode sits in) and, if it changes a section's count,
   `RAIL_GROUP_LABELS`.
3. `src/warlock/studio/modes.py` — exactly one of `WORK_MODES` (has a form or viewport
   and takes shortcuts) or neither, plus `WORKSPACE_MODES` (fills the window with its own
   three-column layout) versus a single-pane mode (see #4). These three groupings — plus
   `_SINGLE_PANE_MODES` below — must **partition `KEYS` exactly**, because `_build_ui`'s
   dispatch in `main.py` ends in a bare `else`, and an unlisted mode draws whatever that
   `else` draws rather than failing.
4. `src/warlock/studio/main.py` — `_SINGLE_PANE_MODES` (currently `("home", "settings",
   "library")`). A mode belongs here, in `WORKSPACE_MODES`, or in neither only if it is
   also absent from `WORK_MODES` (a pass-through mode like Home) — check the partition
   rather than picking one home for the new mode by guesswork.
5. `src/warlock/studio/modes.py` — `VIEWPORT_MODES`, only if the new mode should reload
   the selected library asset into the shared 3D viewport (one member, `create`, today —
   most new modes are not this).
6. `src/warlock/studio/modes.py` — `NAV_KEY_MODES`, only if the mode binds arrow keys or
   Space itself and must therefore be excluded from imgui's own keyboard navigation.
7. `src/warlock/studio/main.py` — `DROP_REFUSALS`. A table, not a chain of branches, by
   design (see "what a miss looks like" below) — every mode that is not one of the three
   the app lets a drop *start* something in (`home`, `library`, `create`) needs an entry
   naming, in its own words, what it works on instead.
8. `src/warlock/studio/main.py`, the `_shortcut` dispatcher (see the shell segment's
   INVARIANTS entry at "Every workspace mode's arm in `_shortcut` returns
   unconditionally") — a new workspace mode's arm must end with its own `return`, or a
   later-added mode's branch inherits keys the new mode never claimed.
9. `src/warlock/studio/manual/targets.py` — `HELP_TARGETS`. Not keyed by mode directly —
   it is keyed by the pane or section id a `(?)` button sits beside — so a new mode needs
   at least one entry for wherever its own `(?)` lives, cross-checked by
   `tests/manual/test_coverage.py`, which fails any pane with neither a `(?)` nor a
   stated exemption. This is also the `manual-chapter` sweep's own concern; the two
   sweeps meet here.
10. A manual chapter for the mode (see the `manual-chapter` row below for that sweep in
    full — do not duplicate its checklist here).
11. `README.md` and any other prose that states the mode count in words ("thirteen",
    "eight workspaces") — gated, see below.

**What a miss looks like.**

- **#1 alone with no #2/#3** — a mode key that exists but that `_build_ui` cannot
  dispatch correctly: falls into whichever of `_SINGLE_PANE_MODES` /
  `WORKSPACE_MODES` / neither the bare `else` happens to draw, which is a silent
  wrong-pane rather than a crash.
- **#7 missing** — the chain-of-branches failure this table was built to replace: a
  drop in the new mode "fell through to Create's branches below," either refusing with
  a sentence describing a form that is not on screen, or accepting the drop and
  switching modes out from under the user. This happened to Poser and Troupe on
  2026-09-04 and to Muse, Review and Settings on 2026-09-05 — twice, for the same
  reason, because the table replaced a chain but the chain's failure mode (a new mode
  simply not being in it) is available to a table too.
- **#8 missing** — the sharpest of these: Packwright's per-mode key handling was
  "lost when Troupe's branch was spliced in ahead of it," and because every arm
  shares one function's tail, the symptom landed in a *third*, unrelated mode: Delete
  in the atlas packer sent the selected library asset to the trash with no confirm.
  The per-mode key tests cannot see this class of bug because they call `handle_key`
  directly rather than through the dispatcher.
- **#9/#10 missing** — `tests/manual/test_coverage.py` fails outright; this is one of
  the few sites in this whole file with a real, load-bearing gate already.
- **#11 missing** — a "thirteen" that quietly becomes wrong prose the next mode after
  this one adds, invisible to any reader who does not count the rail by hand.

**Which sites already have a gate, named.** This is the sweep with the best coverage
in this file, and it is worth reading as the positive example:

- `tests/test_modes.py::test_rail_groups_comment_states_the_real_pipeline_and_workspace_counts`
  checks the module's own comment against `len(RAIL_GROUPS[0])` / `[1]`.
- `tests/test_docstring_counts.py` pins the **number word** ("thirteen") in `README.md`
  against `len(modes.KEYS)` (and the workspace count word against `WORK_MODES`), which
  is exactly site #11's gate — the same docstring-count idiom `tests/tour/test_tour_scripts_docstring.py`
  uses for the tour count, described in `gate-patterns.md`.
- `tests/test_studio_smoke.py::test_the_rail_fits_the_resize_floor_at_every_scale` is
  the **exemplary** gate here: it is not a table-membership check but a *geometry* one —
  parametrised over every DPI scale, it asserts the rail's own row-height ladder still
  fits every one of `modes.KEYS` inside the resize floor. Adding Troupe made the rail
  fifteen rows and, at one scale, wanted 23.85 design px against a 24 px floor — "four
  physical pixels of overflow, and four physical pixels of overflow is an unreachable
  mode." A fourteenth mode that this test still passes is a mode every rung of which is
  provably clickable; one that fails it has made an *existing* mode unreachable, which
  no membership check would ever catch.
- `tests/manual/test_coverage.py` gates #9/#10 (a pane with no `(?)` and no exemption
  fails).
- #2–#6's own partition (`WORK_MODES`/`WORKSPACE_MODES`/`_SINGLE_PANE_MODES`/`VIEWPORT_MODES`
  exactly partitioning `KEYS`) has no single named test that asserts the partition
  property directly in one place — `test_mode_keys.py` and `test_studio_smoke.py`
  exercise the dispatcher per mode, which catches a wrongly-categorised mode
  indirectly (it draws the wrong pane) rather than asserting the partition as its own
  fact. Consider whether this sweep's gate step is where that direct assertion
  finally gets written.

**INVARIANTS lead-ins to grep:** the shell segment's `"Every workspace mode's arm in
\`_shortcut\` returns unconditionally"` (site #8) and `"What a drop is told in a mode
that opens no files"` (`DROP_REFUSALS`'s own doc-comment, site #7, in
`src/warlock/studio/main.py` itself rather than `INVARIANTS.md` — read the comment
directly above `DROP_REFUSALS`). `modes.py`'s own module docstring and the comments
beside `MODES`, `RAIL_GROUPS` and each grouping constant are the primary source for
this row and are worth reading in full before adding a mode; they explain, inline, why
each structure is hand-written rather than derived.

---

## manual-chapter

**What it is.** Adding a new numbered chapter under `docs/manual/`. The governing rule,
verified against `src/warlock/studio/manual/loader.py`: **a chapter's number decides
both its order and its part**, because `chapters()` globs and sorts by filename and
`PARTS` maps number ranges onto part headings —

```python
PARTS: tuple[tuple[str, range], ...] = (
    ("Tutorials", range(1, 20)),
    ("Using Warlock Studio", range(20, 39)),
    ("Setup & operations", range(39, 43)),
    ("Architecture", range(43, 46)),
)
```

— so a chapter appended at the end wears the wrong part or grows a second heading with
the same name. Chapters 01–19 are reserved for the tutorial series (today: 01–16 exist,
17–19 a deliberate gap for the series to grow into without a second renumbering wave);
20–45 are the reference chapters, confirmed against `tests/manual/test_docs.py`'s
`EXPECTED_KEYS`, 42 entries from `00-index` through `45-extending`.

**Sites.**

1. The chapter file itself, `docs/manual/NN-slug.md`.
2. `tests/manual/test_docs.py` — `EXPECTED_KEYS`. The literal ordered list; a new
   chapter is a new line, in number order.
3. `docs/manual/00-index.md` — the table of contents entry, in the same order.
4. `src/warlock/studio/manual/targets.py` — `HELP_TARGETS`, for any `(?)` call site the
   new chapter's subject matter should answer — see the `mode` row above where a chapter
   accompanies a new mode.
5. The `(?)` call site itself in the pane the chapter documents, if the subject is a
   pane that has none yet — `render.help_button`/`render.help_button_inline`, gated by
   `tests/manual/test_coverage.py`.
6. `README.md`'s own chapter-count prose, if it states one (parallel to the `mode`
   row's #11).
7. Every cross-link and anchor reference in the ~45 other chapter files that should now
   point at the new one, or that the new chapter's own renumbering has shifted —
   `tests/manual/test_docs.py` checks link and anchor resolution in both directions, so
   a broken cross-link fails there rather than shipping.

**A renumbering is the supported move, not a hazard to avoid** — INVARIANTS states this
directly: "Renumbering is the supported move and it is mechanical rather than risky,"
because the test above holds the chapter list, every cross-link, every anchor, the
index's own links and `HELP_TARGETS` in both directions. Do the renumbering; do not
route around it by appending out of sequence.

**What a miss looks like.** A chapter appended past the highest existing number in its
part reads under the *next* part's heading, or grows a second heading with the same
name as the part it actually belongs to — invisible until a reader opens the table of
contents. A chapter with no `(?)` anywhere and no stated exemption is "the manual
documents this and nothing reaches it," caught by `test_coverage.py`. A dangling
cross-link or a stale mode/chapter count in prose is caught by `test_docs.py` and
`test_docstring_counts.py` respectively, but only if this sweep actually updates the
line count those tests read — the tests fail loudly on a stale count, they do not fail
on a *missing* one that was never wired to the source of truth in the first place.

**Gate.** `tests/manual/test_docs.py` is a real, load-bearing, bidirectional gate —
read it before adding a chapter, not after. `tests/manual/test_coverage.py` is the
pane-side direction the docs test cannot see on its own. `tests/manual/test_shortcuts.py`
is a **one-way** gate specific to chapter 16 (the shortcut sheet) and is named here as
the "what a bad gate looks like" example `gate-patterns.md` discusses: it checks the
popup against the chapter, never the reverse, "because the chapter is documented as the
full list and the popup as a subset" — a deliberate one-way design in this one case,
not a bug, but worth reading as the shape a two-way gate is not.

**INVARIANTS lead-ins to grep:** `"A manual chapter's number decides both its order and
its part, so adding one is a renumbering"` and `"Chapters 01-19 are reserved for the
tutorial series."`

---

## download-kind

**What it is.** Adding a new category of optional, downloadable weight (a new registry
table alongside `models.BASE_MODELS`, `models.STYLE_LORAS`, and so on). This sweep is
**smaller than it looks**, and the reason is worth stating up front: `src/warlock/fetch.py`'s
`Kind` dataclass carries `check_prefix` and `group` as *fields of the one registration*,
and both `doctor`'s row prefix (`CHECK_PREFIXES`, derived) and the Settings pane's
heading (`GROUPS`/`app_settings._GROUPS = fetch.GROUPS`, derived) read them from
`fetch.KINDS` rather than from a second hand-copied table — `app_settings.py`'s own
comment states why: a hand-copied heading list "only draws the kinds it lists, so an
unlisted one vanished from the pane entirely," and it was rebuilt to derive from
`fetch.KINDS` instead. `_UNGROUPED = "Other"` is the belt-and-braces fallback for a row
whose `kind` string is not in `fetch.KINDS` at all (stale data, not a design case a
correct addition should ever hit) — "it should be unreachable... but 'unreachable' is
what the hand-copy assumed too."

**Sites.**

1. `src/warlock/models.py` — define the new registry table itself: a `dict[str, Spec]`
   of key → a spec carrying `.fetch` (a tuple of `models.Fetch(url, sha256, filename)`
   or Hub-shaped equivalents).
2. `src/warlock/fetch.py` — `KINDS`. One new `Kind(key, table, check_prefix, group)`
   tuple entry. This single edit is what supplies both the doctor row prefix and the
   Settings pane heading — there is no second table to remember.
3. Any UI picker that reads the table directly rather than through the generic
   Settings rows (a model combo box, a style-LoRA selector) — this is **kind-specific**,
   not a fixed fourth site every download-kind needs; a kind with no picker (a metric
   model, a matting model) stops at #2.
4. If the new kind is what gates a mode (see `modes.NEEDS_ROWS`, currently `create` and
   `muse` only), a `NEEDS_ROWS` entry in `src/warlock/studio/modes.py`, keyed by the
   literal registry row string (`"engine:trellis_gguf"`), checked by
   `tests/test_mode_gate.py` against `fetch.find` so a renamed row breaks a test rather
   than silently ungating a mode.

**What a miss looks like.** Skipping #2 (registering the table but not adding the
`Kind` entry) leaves the rows genuinely unreachable: `entries()` walks `KINDS` rather
than a second list, so a table with no `Kind` entry offers nothing to download and
`doctor` never checks its presence — silent in exactly the way `_UNGROUPED`'s comment
describes, except that here there is no row at all to fall into "Other," because
nothing ever produced one.

**Gate.** This sweep is close to self-gating by construction: `_GROUPS = fetch.GROUPS`
and `CHECK_PREFIXES` are both derived, so there is no second table for either of them to
drift from once `KINDS` is right. `tests/test_fetch.py` and `tests/test_doctor.py`
exercise the registry generically; neither is parametrised to fail if a *table* exists
in `models.py` with no matching `Kind` — because such a table produces no rows at all
for any test to notice missing. If you add a table without adding its `Kind` entry, no
existing test will tell you; this is the one concrete case in this sweep worth writing
a small new check for (a scan asserting every `models.py` dict of `Fetch`-bearing specs
has a `Kind` in `KINDS` naming it).

**INVARIANTS lead-in to grep:** there is no single bold-lead paragraph naming this
sweep; the governing design is stated in code comments rather than in
`docs/INVARIANTS.md` — read `fetch.py`'s `Kind` docstring and the comment above
`app_settings._GROUPS` directly.

---

## publish-site

**What it is.** A worker stage that renames or copies its finished output onto a
**served name** — a filename the app treats as the completed artifact (`model.glb`,
`rig.glb`, a sheet's sidecar). The rule: "a stage that has published onto a served name
commits the cancel token" (`_Cancel.commit()`), because after publication a cancel
cannot un-publish the file, so recording the row `cancelled` without committing is a
lie about what is actually on disk.

**Sites — eleven known today, eight covered.** `tests/test_job_durability.py`'s
`PUBLISHERS` list, verified against the file:

```python
PUBLISHERS = [
    ("warlock._q_rig", "_rig", "finalize_rig"),
    ("warlock._q_rig", "_sheet", "_publish_text"),
    ("warlock._q_sprite", "_pixel_sheet", "_publish_text"),
    ("warlock._q_sprite", "_sprite_synthesis", "_publish_text"),
    ("warlock._q_sprite", "_retexture", "os.replace"),
    ("warlock._q_tilesheet", "_tile_sheet", "_publish_text"),
    ("warlock._q_tileset", "_tile_set", "_publish_text"),
    ("warlock._q_troupe", "_charsheet", "_publish_text"),
]
```

Three more are named in `TODO.md` P37 as found but not yet added: `_deform_qa`
(`_q_rig.py`), the model-promotion stage in `_q_generate.py`, and `_remesh`
(`_q_mesh.py`) — eight plus three is the eleven this row's title claims, and P37's own
text is explicit that only human judgement, not a rule, can tell a *completion marker*
(must commit) from an *intermediate checkpoint* a cancel may still legitimately unwind:
"a 'last write wins' heuristic promotes the wrong call in some of these functions and
misses the real one in others, which fails open, silently."

**Adding a new publish-site (a new stage, or a new call site inside an existing
stage's worker function) means:**

1. Confirm, by reading the function, whether the rename/copy is a completion marker or
   an intermediate checkpoint — this is the judgement call `PUBLISHERS` is deliberately
   a hand list rather than a scan for.
2. If it is a completion marker: `self._cancel.commit()` must run **after** the rename
   call and **before** the row can be observed as `cancelled` — read
   `test_job_durability.py::test_every_served_publish_commits_the_cancel_token` for the
   exact source-scan shape (it finds the publish call in the function body, then
   searches forward for `_cancel.commit()`).
3. Add the `(module, function, publish_call)` triple to `PUBLISHERS`.

**What a miss looks like.** A cancel landing in the window between the publish and the
commit runs `_set_cancelled` and `_discard_artifacts` over minutes of finished GPU or
Blender work while the row claims the work never happened. For `_retexture` this is
irreversible — it replaces another job's served `model.glb`, so "the skin the user had
is gone." For `_charsheet` it is worse in effect though reversible in principle:
`_discard_artifacts`' sheet branch deletes the **served** pair rather than temps, so a
cancel one line late unlinked a character sheet Troupe was already drawing.

**Gate.** `tests/test_job_durability.py::test_every_served_publish_commits_the_cancel_token`,
parametrised over `PUBLISHERS`. This is the file this skill's whole premise is drawn
from as a cautionary tale: the comment directly above `PUBLISHERS` explains, in its own
words, why the list "stays a list instead of a scan of its own" — the judgement call in
step 1 above cannot be automated — and states that six of the eight rows it now holds
have each shipped this exact bug once already, because the list itself was stale and
"this scan never looked at them." Treat every new call in a worker function that writes
onto a name `service.files` would call served as a candidate for this list, and add it
whether or not you are sure it needs to commit — a checked "no" costs nothing and an
unchecked hole costs a user's asset.

**INVARIANTS lead-in to grep:** `"A stage that has published onto a served name commits
the cancel token, and a stage whose cleanup deletes served names may not mint its own
id."` Cross-reference `TODO.md`'s `## P37.` entry for the three known, un-added sites.
