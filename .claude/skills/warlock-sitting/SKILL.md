---
name: warlock-sitting
description: Given a TODO.md entry number for a human judgement sitting, resolve and refuse against TODO.md's real headings, produce every artefact its own Do list calls for into a scratch gallery (renders, sweeps drained, sheets baked — whatever CPU and, where unavoidable, a queued card run can make), hand the human the pictures and the entry's own question list verbatim, stop without offering an opinion, and — once the human answers back — write the dated docs/measurements/ document, strike the TODO.md entry and hand off to /warlock-land for the commit; use it whenever a TODO.md entry needs a human to look at, or listen to, something before it can close (/warlock-sitting P28, /warlock-sitting P8, /warlock-sitting P10).
argument-hint: <P-number>
arguments: [p_number]
---

# Sit in front of one judgement, from setup to writeup

`TODO.md`'s first declared kind of entry is work only a human can do:
art direction, blind grading, listening. At the 2026-09-07 snapshot roughly
fifteen open entries are exactly this shape, and every one of them is blocked
less by the judging than by the hours of setup in front of it — booting a
throwaway home, queuing a sweep, waiting for Blender to render eight
directions of five animations, laying the output out so it can be looked at
in order. Nothing in the repository does that setup on request; a person
either does it by hand each time or the entry stays open by default. This
skill is the setup, run the same way every time, so a human's part is twenty
minutes of looking at pictures that are already made rather than an afternoon
of getting to the point of having something to look at.

Every long text this skill needs is in `references/` next to this file: read
a reference the first time a step below names it and copy from it rather than
paraphrasing. `references/constraints.md` is pasted, verbatim, into every
subagent prompt you launch — copied byte-for-byte from
`.claude/skills/warlock-audit/references/constraints.md` so this skill does
not depend on that directory's continued existence. Your job is to *run this
procedure*, not to redesign it; when the procedure and your own judgement
disagree, the procedure wins unless the user says otherwise.

**It never guesses a verdict.** Not a hedge, not a "this looks like it might
pass" aside, not a description of a picture phrased so as to anchor what the
human is about to say. The entry's own question list is the only thing this
skill hands over about what to look for, and the stop before Part 3 is the
point of the whole exercise: everything up to the pixels is yours; the
verdict itself is never yours.

## Part 1 — Resolve and refuse

1. Get the live list of open entries, not the one in `references/entries.md`
   (that file is a dated snapshot and says so):
   ```
   grep -n '^## P[0-9]' TODO.md
   ```
2. If `$p_number` does not match one of those headings exactly, refuse. Print
   the three buckets from `references/entries.md` (sittings this skill can
   run in full, sittings it can only partly prepare, and entries it refuses
   outright) so the user can see what a correct argument looks like — do not
   guess at a near neighbour ("P8" is not "P08" is not "P4").
3. Read the matched entry **whole**, in `TODO.md` itself, not from
   `references/entries.md`'s summary of it. Treat its *Do* list as the
   specification it is — the entry is often already written in the shape of
   a pre-registration, and inventing a different method than the one it
   states is redoing work the entry already did.
4. If the entry names an existing pre-registration or measurement document
   (check `references/entries.md`'s per-entry notes for known cases — P15 and
   P16 both cite sections of
   `docs/measurements/2026-08-30-art-verdicts-preregistration.md`, P8 and P28
   both cite `docs/measurements/2026-09-02-troupe-qa-thresholds.md`), confirm
   the file exists before doing anything else. **Refuse if it does not** —
   the setup must never run ahead of the commitment the document represents.
5. Classify the entry against `references/entries.md`'s five groups (verify
   the classification against the entry's own current text; the reference
   file's grouping is a starting point, not a substitute for reading):
   - **A picture- or listening-judgement sitting** (Groups 1, 1b, 2, 3 in the
     reference) — continue to Part 2.
   - **An editorial decision with no artefact** (Group 4) — say so, quote the
     decision the entry is actually asking for, and stop. Do not manufacture
     a gallery for a question a picture cannot answer.
   - **Not a sitting at all** (Group 5: deferred-pending-another-decision,
     design-review-gated, a writing task, a code-classification judgement, a
     crash-reproduction hunt) — say which of those five shapes it is and
     stop. These are real work; they are simply not this skill's work.
   - **Group 1b specifically** (an entry whose *Do* list opens with a human
     drawing or posing, not viewing — P8, P33) — say plainly that the
     artwork or the pose is a human prerequisite this skill cannot produce,
     then continue to Part 2 for whatever mechanical preparation *is*
     possible around existing or already-authored material.

## Part 2 — Everything up to the pixels

This part is long-running; run it in the background and keep working on the
report while it goes, or hand it to a fork if the entry's own run is the kind
of thing that would otherwise fill your context with sweep-drain noise.

1. Make one scratch gallery for the run: `<scratchpad>/sitting-<P-number>/`.
   Everything this part produces goes there, laid out per
   `references/setup-recipes.md`'s "Contact sheets and zoom levels" section —
   one file or pane per thing the human will look at once, not a directory
   dump they have to reconstruct an order for.
2. Read `references/setup-recipes.md` for the mechanism the entry actually
   needs, and use the one that matches:
   - **The headless drain** (`Runtime` booted directly, submit through
     `service.jobs.create_job`/`service.sweeps.create_sweep`, poll
     `store.get`) for anything that needs the job queue and where you are
     both submitting and draining yourself. **Never** run this alongside a
     `scripts/campaign_*.py` invocation against the same database — read
     `scripts/_campaign.py`'s `require_no_live_writer` and its exact
     wording before assuming two processes are safe.
   - **A campaign submitter** (`campaign_props.py`, `campaign_detail.py`,
     `campaign_guidance.py`, `sweep_refill.py`) when the entry's corpus
     already has one, and only when nothing else is currently writing that
     database.
   - **The bench** (`python -m warlock.bench ...`) for anything that is a
     reconstruction-benchmark question rather than an ordinary library job.
   - **The throwaway-home block**, copied verbatim from
     `references/setup-recipes.md`, for any run that must not seed the
     user's real library — and the opposite call, stated plainly, for any
     run that must read the user's real corpus instead.
3. For a Group 1/1b entry (nothing but CPU, or CPU once art exists): run the
   render/bake/export/QA-heatmap pipeline the entry's *Do* list names, using
   `warlock` library code directly rather than driving the GUI, unless the
   entry specifically asks for something only the running app can do (a
   Poser pose, an Inker drawing — which you cannot do for the human anyway).
4. For a Group 2/3 entry that needs a card: queue the run, then either wait
   on it in the background or return with "queued, draining" and pick the
   thread back up once notified. Do not report the sitting as ready before
   the queue has actually finished draining — check `store.get`'s terminal
   statuses, not elapsed time.
5. **Never guess a verdict at this stage either.** A render that plainly
   crashed, is blank, or is missing a direction the entry asked for is a
   *setup failure* to report and re-attempt, not a data point to fold into
   the later report as if it were a content judgement.
6. Report, before moving to Part 3: what was produced, into which files;
   what could not be produced and why (a missing weight, a locked database, a
   card that OOM'd); and which of the entry's own *Do* steps are still
   outstanding because they are the human's alone (drawing, posing,
   listening, opening a foreign application, touching a second machine).

## Part 3 — Hand over, then stop

This is the part the whole skill exists to protect.

1. Print the entry's own question list **verbatim** — copy it out of
   `TODO.md` itself, or from `references/entries.md` if that snapshot still
   matches (P28's seven questions, P8's five per-clip briefs, P33's seven
   things to judge separately, whatever the matched entry states). Do not
   summarise it, shorten it, or reorder it into what you think matters most.
2. The hand-over takes one of three shapes, and most Group 1/1b entries are
   the third, not the first two:
   - **A grading pass** through Review mode's keyboard, for a sitting that
     queued real job rows — print the exact keystrokes from
     `references/setup-recipes.md`'s Review-mode section (the digit
     magnitudes, `R` arming the negative sign, `0` as its own key,
     Ctrl/Shift+1–5 for the two tag vocabularies, and the guided pass's
     different `A`/`R`/`Esc` meanings while it is open — get these from the
     reference exactly, they are precise for a reason).
   - **A listening pass**, for ears against a manual chapter followed out
     loud (P14, P23, P24, P35) — print the chapter and page.
   - **Static pictures opened directly, no queue, no job rows, no
     blinding** — most of the CPU-only entries (P30, P34, and any Group 1/1b
     sitting whose output is a contact sheet or a rendered comparison rather
     than a graded library row) are this third shape. There is no verdict
     loop to drive here: no keystrokes to hand over, nothing to blind,
     because nothing was queued for review in the first place. Give the file
     paths to the pictures and quote the entry's own findings verbatim as
     the questions to confirm or correct against them — P34's item 2 ("fish
     and bird silhouettes read weak... wants attachment overlap, tapered
     wedges, a deliberate wing outline and thickness direction") and item 3
     ("the shape chooser undersells the objects... sphere/torus share an
     icon...") are exactly this: claims to check against renders, not a
     seven-question rubric to grade through a keybound loop.
3. Say plainly where the pictures or the queued rows are, and, only where a
   verdict loop actually exists (the first two shapes above), whether the
   pass is blind, per `references/setup-recipes.md`'s note that blindness is
   a session property and must be stated rather than assumed. For the third
   shape, say plainly instead that there is no verdict loop to drive and
   nothing to blind — the human is looking at pictures directly, not grading
   library rows.
4. **Never guess a verdict, never grade on the human's behalf, and never
   describe what you think the pictures show in a way that would anchor the
   judgement.** Not "the walk looks pretty convincing to me" — not even as a
   throwaway aside. The entry's decision rule, once one exists, is applied to
   what the human says, not to what you noticed first.
5. Stop. Do not begin Part 4 in the same turn unless the human's verdict has
   already arrived in this conversation.

## Part 4 — Write it up

Only once the human has actually given verdicts back.

1. Read `references/writeup.md` and follow its section order exactly: why
   now, the retry section only where an antecedent document demands one,
   what is under test, the instrument cited and not restated, what will be
   run, decision rules fixed before results, retention, then results, then
   verdict, then what this changed in the tree.
2. **Apply whichever decision rule fired, verbatim, including the boring
   one.** Quote `2026-08-09-grade-scale.md`'s line if the entry's instrument
   is the mesh grade scale; the same principle holds for every other
   instrument regardless of what document states it.
3. **Never write a `YYYY-MM-DD-` placeholder path.** Name the document by its
   directory until it exists, then by its real name once it does —
   `tests/test_external_doc_links.py` reads every citation across `TODO.md`,
   `README.md`, `CLAUDE.md`, `docs/INVARIANTS.md`, `docs/MODELS.md`,
   `docs/COMPAT.md` and everything under `docs/measurements/*.md` as a claim
   the file exists.
4. Strike the `TODO.md` entry with `~~...~~` and a dated sentence, in the
   voice of the other struck lines already in that file — never a checkbox,
   there isn't one. Move a one-line record to *Closed records*. Entry numbers
   are never reused.
5. Check whether the entry's own text names a manual chapter status line to
   lift (P28 names `docs/manual/11-a-character-sprite-sheet.md`'s line
   directly); lift it if the verdict warrants it.
6. Hand off to `/warlock-land` for the commit. Do not restate its checklist
   here — cite it and call it.

## Coverage

Every report this skill produces — after Part 2, and again after Part 4 —
states plainly: what was produced and where; what the entry's *Do* list still
requires that only a human can supply (art, a pose, a foreign application, a
second machine, ears, a published release); and, if anything failed to
generate, what failed and why, rather than a gallery that quietly reads as
complete when it is not. A sitting that only partially prepared its material
says so in the same breath it hands over what it did manage — silence about
the gap is the one failure mode this whole skill exists to prevent.
