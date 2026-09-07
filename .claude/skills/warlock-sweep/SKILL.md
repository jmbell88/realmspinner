---
name: warlock-sweep
description: Walk a repeated multi-file addition (a new job kind, a new phase on an existing one, a follow-up kind, a new mode, a new manual chapter, a new downloadable weight kind, or a new stage that publishes onto a served name) across every site it must touch in this codebase, refuse to finish while a site is unvisited, and leave behind a bidirectional test gate so the next one cannot fail silently the way Muse, Troupe, the grid kind and `PUBLISHERS` each did; use it whenever you are adding one of those seven shapes, or auditing an existing one for a hole (`/warlock-sweep job-kind sprite_synthesis`, `/warlock-sweep mode muse`, `/warlock-sweep job-kind --dry-run charsheet`).
argument-hint: <sweep> [<name>] [--dry-run]
arguments: [sweep, name]
---

# Sweep a repeated addition across every site it touches

`docs/INVARIANTS.md` names this codebase's own worst-scoring class of change, in
its own words: **"A new job kind is a sweep of the stage-keyed tables, and the
sweep is the invariant."** Ten tables decide what a row is offered, and "every
one falls through **silently** for a kind it has never heard of — which is a
button that refuses, or no button at all, rather than a failing test." The
paragraph then states the miss rate rather than a hope: the previous grid kind
missed four tables; **Muse swept eight and missed two** — `widgets.STAGE_BADGES`
and `state.primary_action` were both absent when the mode shipped, so every
take drew the fallback CIRCLE glyph with its raw stage string beside it and a
*finished* card offered no action at all; **Troupe swept eight of nine and
missed `progress`** — `charsheet` fell back to `PHASES_IMAGE`, and the
never-regress creep pinned the bar at 100% for the entire pixel-art tail, CON-02
for the fourth time. `_maybe_queue_rig` shipped with no stage guard at all and
wrote a permanent false "the generated mesh artifact is missing" onto every
character anyone started, twice, in two different ways, before it was fixed.
`PUBLISHERS` in `tests/test_job_durability.py` is a hand-written table that
covered eight of the eleven stages that rename onto a served name, and stayed a
list "instead of a scan" on purpose — TODO.md P37 records the three it still
misses, failing open, silently, exactly as the ten stage-keyed tables do. This
skill is the second half of the same paragraph: not just "here is the list",
but "walk it in order, refuse to stop with a hole, and leave a test that would
have caught the miss."

Nothing else in this repository runs the walk. `tests/test_progress.py`'s
phase-table sweep exists only because the fourth failure forced it, and it is
parametrised over the four kinds someone remembered to add — a fifth
multi-pass kind that nobody adds to that list is exactly as invisible as the
first four were. `warlock-audit` finds a hole after the fact, by reading; this
skill exists to walk the list *while adding the thing*, so the hole is never
opened.

**This skill runs the walk it is given. It does not invent a new sweep, and it
does not fix findings from an unrelated audit.** `references/sweeps.md` is the
list of what a sweep is; if the addition you are making is not one of its
rows, this is the wrong skill — read `warlock-audit` or do the change directly.
Your job is to *run this procedure*, not to redesign it. Every long text this
procedure needs is in `references/` next to this file; read a reference the
first time a step names it and paste from it rather than paraphrasing.

## Rules that hold in every step

`references/constraints.md` is pasted, verbatim, into every subagent prompt.
Read it now; it is short. Its load-bearing lines: no subagent spawns a
subagent, no subagent ever runs `git stash`, `checkout`, `reset` or `restore`
(a fixer did on 2026-09-04 and reverted nine other agents' edits plus the
user's uncommitted work), and no subagent ever runs the full test suite — a
subagent runs only the test files its brief names, with `-n 0`.

The orchestrator has one extra rule: the default suite runs **once**, by you,
in step 6, after everyone has returned. Not before, and never while anything
else may be editing.

## 1. Resolve the sweep

`$1` is one of the sweep keys in `references/sweeps.md`: `job-kind`, `phase`,
`followup-kind`, `mode`, `manual-chapter`, `download-kind`, `publish-site`.
Refuse anything else and print the seven keys — do not guess at a near-match,
and do not accept a plural or a hyphenless spelling as close enough.

`$2`, when given, is the name being added or (with `--dry-run`) the existing
name to walk read-only. Some sweeps need it at step 3 (a mode key, a job kind,
a chapter number); others do not (`phase` names the kind and the phase in the
walk itself). If the sweep needs a name and none was given, ask for one rather
than inventing a placeholder.

## 2. Freeze the baseline

Record, and put in the report verbatim:

```
git rev-parse --short HEAD
git status --short
uv run ruff check .
```

Make one scratchpad directory for the run, `<scratchpad>/sweep-<sweep>-<date>/`.
Findings and edits are against the **working tree**; the report says which
files were dirty when the sweep started, and those files are the user's — this
skill does not touch them unless the sweep itself requires it.

## 3. Read the row, then walk the sites in order

Open `references/sweeps.md` and read the one row for `$1` in full. Paste that
row whole into your own working notes (and into every subagent brief that
touches it) rather than summarising it from memory — the row's site list is in
the order a real row meets the tables, per the INVARIANTS paragraph it is
drawn from, and that order is part of what makes the walk checkable.

Read every site **before** editing it. A site already correct for this
addition (an existing default that already covers it, a table that derives
rather than needing a new entry) is still a site you visited, and the coverage
statement says so explicitly rather than omitting it as not applicable.

For a small sweep (`phase`, `download-kind`, most of `followup-kind`), walk it
yourself. For a large one (`job-kind`, `mode`, `manual-chapter`), fan out one
Sonnet subagent per site cluster, with **file ownership** — no two concurrent
agents own one file — pasting `references/constraints.md` verbatim into every
brief, plus the sweep's row from `sweeps.md`, plus the name being added. Use
`model: sonnet`, at most eight agents at once, depth capped at one (no
subagent may launch another). Launch a batch as `Agent` calls in a single
message; when a batch returns, launch the next.

**A `--dry-run` is always walked by one agent, regardless of the sweep's own
size.** `job-kind` is a large sweep by this step's own rule, but a dry run is
read-only and needs none of the fan-out machinery above — there is nothing to
partition file ownership over when nothing is being edited. Fanning it out
would also lose findings, not just add overhead: on 2026-09-07 the real
`charsheet` finding at site 9 (see `sweeps.md`'s "what a miss looks like"
paragraph for that sweep) only surfaced because the one agent doing the walk
still had site 5's `card_kind` result in context while reading site 9 — a
connection eight independent subagent transcripts, each returning in
isolation, cannot make.

## 4. Refuse to report done with an unvisited site

Before any gate or report, check the row's site list against what you
actually touched or verified. **A site you did not reach is not a finished
sweep** — go back and reach it, or state in the coverage section exactly why
it could not be reached (a file that does not exist yet on this branch, a
site that genuinely does not apply to this addition, and why). Do not let a
subagent's silence about a site stand in for having checked it: read what it
returned against the row's list line by line.

## 5. Add the gate

**Stop here for a `--dry-run`.** A dry run truncates the procedure after step
4: no gate is added, step 6 does not run, and there is no hand-off to
`/warlock-land` — the report described in "A dry-run mode" below is the whole
output. Steps 5 through 7 are for an addition actually being made.

Read `references/gate-patterns.md`. A sweep run is not finished by touching
every site; it is finished by leaving something behind that would have caught
the miss if this run had made one — the tour's bidirectional shape where one
exists to extend, a new parametrised case added to an existing sweep test
(`tests/test_progress.py`'s kind list, `tests/manual/test_docs.py`'s
`EXPECTED_KEYS`, `PUBLISHERS`) where one does not, or, for the sweeps that are
already self-gating by construction (`download-kind`'s derived heading), a
one-line note in the report saying why no new test earns its keep here.

**A sweep that adds a table entry without extending the test that walks that
table's kind has left exactly the hole this skill exists to close.** If no
existing test can be extended and a new one is not warranted, say so and say
why in the report — silence on this point is not an acceptable outcome of the
skill.

## 6. Run the gate once, nothing else editing

```
uv run ruff check .
uv run pytest
uv run python scripts/preflight.py --fast
```

In that order. A failure inside a file this run owns is this run's to fix. A
failure outside every file this run touched is a pre-existing finding: record
it in the report and do not fix it here.

## 7. Hand off

This skill does not write `CHANGELOG.md`, `docs/INVARIANTS.md`, a manual
paragraph or a commit — `/warlock-land` does, and its checklist is not
repeated here. When the sweep and its gate are green, say so and name
`/warlock-land` as the next step; do not duplicate its work by hand.

## A dry-run mode

`/warlock-sweep job-kind --dry-run <existing kind>` walks the `job-kind` row's
sites for a kind that **already exists**, read-only, and reports which of the
fifteen sites that kind actually joined, using `git show HEAD:<path>` and
ordinary reads rather than editing anything.

**A dry run stops after step 4 of the main procedure and goes no further.**
It does step 1 (resolve the sweep), step 2 (freeze the baseline — the `git`
and `ruff` lines still get recorded, since they cost nothing and confirm the
tree this walk is read against), step 3 (read the row and walk the sites, by
one agent per the rule above) and step 4 (check every site was reached). It
does **not** reach step 5: no gate is added or extended, because nothing was
changed for a gate to protect. It does not reach step 6: the suite is not run,
`ruff` is not run a second time, and `preflight` is not run at all. It does
not reach step 7: there is nothing finished to hand off, so `/warlock-land` is
never named. A fresh agent that reads steps 5–7 as unconditional and goes on
to run them against a `--dry-run` invocation has changed nothing on disk but
has still broken the "change nothing" rule this section states, by spending
the run on work the invocation never asked for.

**Grepping for the kind's name is not the check, and a walk that does that
will report false holes.** Measured on 2026-09-07: `sprite_synthesis` appears
by name at only eight of the fifteen sites, and most of the rest are correct
anyway because they do not dispatch on kind at all — `files.ready` keys on
the *artifact name* (`model.glb`, `source.glb`) and `create_stages.IMAGE_STAGES`
keys on the *stage*, so a kind whose artifacts and stage are already covered
needs no branch there and its absence is the right answer. `DERIVED_PARAMS`
is the sharper case: `sprite_synthesis` has no entry because its worker
records nothing about its own run that a rerun must strip — which is a real,
checkable claim about that worker, and the only way to know it is to read the
worker.

**But "it keys on the stage, so no branch is needed" is not a verdict either,
and site 9 is where that reasoning fails.** This paragraph said exactly that
about `widgets.STAGE_BADGES` until the 2026-09-07 `--dry-run charsheet` walk
disproved it. `stage_badge()` does key on `job.get("stage")` rather than on
the kind — and that is the defect, not the excuse: every follow-up row's
`stage` is frozen at `db.Store.create`'s `"model"` default, so the badge is
wrong for `rig`, `sheet`, `charsheet`, `retexture`, `pixel_sheet` and
`sprite_synthesis` alike, while `thumbs.thumb_glyph()` — the other half of
the same site — reads `card_kind(job)` and gets it right. A site that
dispatches on a *different* field is only correct if that field carries the
answer; check that it does, rather than stopping at "this site does not read
the kind." Site 9's own row in `sweeps.md` carries the full provenance.

So each site's verdict is **semantic, not textual**: read the site and answer
whether this kind is handled correctly, by an explicit branch *or* by a
default that is right for it, and say which of the two. A site reported
"absent" must say what would break if that were wrong, or it is a grep
result wearing a finding's clothes.

`charsheet` is the second worked example, and the walk was run on
2026-09-07 so its answer is known: the `progress` miss the invariant records
is **closed**. `PHASES_CHARSHEET` is registered in `_PHASES_BY_KIND`, its five
spans are contiguous from 0.0 to 1.0, and
`tests/test_progress.py::test_the_multi_pass_kinds_have_their_own_contiguous_tables`
is parametrised to include `charsheet` and passes. A walk that reports that
site as still broken has read the invariant's history as if it were the
present tense, which is its own kind of false hole — the paragraph records
what Troupe shipped, not what the tree does now. What that walk *did* turn up
is site 9, above, which is live today and which no invariant records at all.

A `--dry-run` on any other sweep key follows the
same shape: read every site for the name given, report present/absent/partial
per site, and change nothing.

## Report back

Name the sweep and the addition. List every site from its row with one of
three states: **touched** (what changed), **already correct** (why nothing
was needed), or **not reached** (why, named honestly). State what gate you
added or extended, and why, or why none was warranted. Paste the gate's last
lines. If this is a `--dry-run`, report the per-site verdicts instead of a
diff. Then name `/warlock-land` as the next step.
