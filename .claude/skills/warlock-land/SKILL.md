---
name: warlock-land
description: Land a behaviour change already sitting in the working tree — separate the session's own edits from whatever else is dirty, prove every new or changed regression test actually fails against the unfixed code, pay what the change owes beyond src/ (the docs/manual/ chapters that describe what moved, docs/INVARIANTS.md, CHANGELOG.md, TODO.md, and the two ungated screenshot trees), run the gate once (ruff, the full suite, preflight --fast) with nothing else editing, and commit only the paths this change touched with the session's attribution trailers; pass --release to instead land the five-file version bump and point at the real release build. Use whenever a working session's behaviour change is finished and needs landing, or when cutting a release (/warlock-land, /warlock-land --release).
argument-hint: [--release]
arguments: [mode]
---

# Land a behaviour change

`/warlock-audit`'s step 8 already runs this ledger, but only for findings that
came out of an audit file — a fixer partitions files, writes the shared docs,
runs the gate once, commits. A change that starts any other way (a bug report,
a user ask, an exploratory session) has nowhere to run the same ledger against,
and it shows: the last thirty commits are the record of what gets missed when
nobody does. The median is five files, but the tail is always the same shape —
`16de9f7b` (25 files: nine `src/` files, ten test files, `CHANGELOG.md`,
`docs/INVARIANTS.md`, three `docs/manual/` chapters and one of their images),
`a56932a7` (18 files: six `src/` files, four test files, `TODO.md`,
`docs/INVARIANTS.md`, two manual chapters, three screenshots), `591836f2` (16
files: seven `src/` files, two test files, `TODO.md`, `docs/INVARIANTS.md`,
the five release files) — `src/` plus a test file named as a claim plus
`CHANGELOG.md` plus `docs/INVARIANTS.md` plus one to three `docs/manual/`
chapters plus a `TODO.md` entry that narrows. `docs/INVARIANTS.md` alone was touched by 16 of
the last 60 commits. And nothing gates the two things a reviewer cannot
eyeball from the diff: `docs/manual/` promises are checked only by bespoke
assertions written one per drift already found
(`tests/manual/test_manual_promises.py`, `test_manual_labels.py`), and the
screenshot refresh (`afe743ee`, 120 files) has no test referencing
`screenshots/` at all. This skill is that ledger, lifted out of the audit's
step 8 so a change landed any other way can still reach it.

**This pass lands one change that already exists in the tree: it separates it
from whatever else is dirty, proves its tests, pays its obligations, runs the
gate, and commits. It does not write the behaviour change, choose the fix, or
decide what the change should have been — that work is already done by the
time this skill starts.** Your job is to *run this procedure*, not to
redesign it.

## 1. Resolve the mode

`$ARGUMENTS` is either empty (the ordinary case) or exactly `--release`.
There is no codebase enum to check this against the way `modes.KEYS` backs
`/exercise-mode` — it is a two-way switch on this skill's own behaviour, not a
value drawn from the tree — so validate it as a literal match and refuse
anything else, printing the two forms rather than guessing at what was meant:

```
Usage: /warlock-land            land the change sitting in the working tree
       /warlock-land --release  land the five-file version bump
```

## 2. Resolve what this pass is landing

Run `git status --short` and `git diff`. This tree carries the user's own
uncommitted work far more often than not, and separating "what this session
changed" from "what was already dirty" is the first step, not an afterthought
— get it wrong here and every later step operates on the wrong file list. If
you cannot identify the session's own changes with confidence (for instance,
edits in a file the session never opened, or a hunk that looks unrelated to
what was asked), ask rather than guess. Never `git add -A` and never
`git add .`, here or anywhere later in this procedure — both would sweep in
whatever else is dirty.

## 3. Prove every regression test

For each test file that is new or has a changed test in the session's own
diff, read `git show HEAD:<path>` (the pre-session version) and check by
reading — not by touching the tree — that the test would actually have failed
against it, and *why*. "Fails" is not enough on its own: the 2026-09-05
lesson, carried in the audit skill, is that a test failing pre-fix with
`AttributeError: no such method` pins nothing about the wiring it claims to
guard, and a finding closed on that evidence has to be re-opened. State the
reason each test fails pre-fix — the assertion it trips, not merely that
collection or execution errors out.

## 4. Work out what else the change owes

Read `references/landing-checklist.md` the first time you reach this step.
It gives five obligations a behaviour change routinely owes beyond `src/` and
its test, each with the test that gates it or the plain statement that
nothing does. Walk all five against the session's diff and decide, for each,
whether it applies here and whether it has been paid. An obligation that does
not apply to this particular change is not a gap — say so and move on; one
that applies and is unpaid is fixed now, before the gate runs, because the
gate cannot see a missing manual sentence or a stale invariant.

Two rules that apply across all five, both load-bearing and both test-gated:

- **Never cite `TODO.md`, a plan file, or an audit file's name from `src/` or
  `scripts/`.** `tests/test_ux_todo_fixes.py` refuses it. Cite the programme
  in prose instead — "the 2026-09-07 landing pass" — never the filename.
- **Never write a placeholder path into a document** — no
  `docs/measurements/YYYY-MM-DD-*.md` stub, no manual link to a chapter that
  does not exist yet. `tests/test_external_doc_links.py` reads a cited path
  as a claim that the file exists, and a placeholder fails it exactly as a
  typo would.

## 5. Run the gate once

In this order, with nothing else editing the tree — CLAUDE.md's own rule is
that `src/` must never move while the suite runs, because several tests read
module source:

```
uv run ruff check .
uv run pytest
uv run python scripts/preflight.py --fast
```

`preflight.py`'s own docstring says exactly what `--fast` skips: nothing but
the suite you already just ran. The four checks it runs in order are version
lockstep across `pyproject.toml`, `src/warlock/__init__.py`, `CHANGELOG.md`
and `INSTALL.md` (read directly, before an install, because a textual drift
between them is the one failure this project has actually shipped); `ruff`;
the non-GPU suite (skipped by `--fast`, since step 5 already ran it); and a
printed reminder that the GPU lane is opt-in and was not run — reported, never
silently absent, because "the suite passed" must not come to mean "the GPU
lane too." A failure here sends you back to step 4 or to the code itself, not
past it.

## 6. Commit only the paths this change touched

Stage exactly the files step 2 identified as the session's own, plus whatever
step 4 added or edited to pay an obligation — named individually, never `-A`,
never `.`. Write a subject in the register of `git log --format=%s -20`: a
sentence about what changed for the user or the project (`Poser can open a
rigged asset's own mesh, not just a bare skeleton`), never a process sentence
like "fix findings" or "land change." Close with the two attribution trailers
this session was given. If `git status` still shows files after this commit,
they are the user's — leave them unstaged and say so in the report; this
skill commits one change, not the whole tree.

## 7. `--release` mode

A release is exactly five files, always the same five, and no others belong
in this commit: `CHANGELOG.md`, `INSTALL.md`, `pyproject.toml`,
`src/warlock/__init__.py`, `uv.lock`. Bump the version in lockstep across all
five (`uv lock` regenerates the lock file's own version-bearing line), write
the `CHANGELOG.md` entry under the new heading in the voice
`references/voice.md` describes, and update `INSTALL.md`'s installer filename
to match. Then run `uv run python scripts/preflight.py` — the full gate, not
`--fast`, since a release is exactly the case that must not skip the suite.
Read `TODO.md`'s P29 entry before telling the user the release is finished: it
names what still has to happen by hand (publishing the actual GitHub release
carrying both the installer and the update manifest) and what would fail if
skipped. Point at, but do not run unless asked: `pwsh scripts\rebuild.ps1` to
build the installer, and `uv run python scripts/make_update_manifest.py` to
produce the manifest that release carries.

## 8. Report

- What landed: the file list from step 2, separated from what was left as the
  user's.
- What the change owed per `landing-checklist.md`'s five obligations, and
  whether each was paid, not applicable, or already covered.
- Each regression test proved, and the specific assertion it trips pre-fix.
- The gate's last lines from step 5 (or step 7's full run, in `--release`
  mode).
- The commit hash and subject.
- **Coverage, stated plainly even when it is clean**: which of the five
  obligations applied and were paid, which did not apply and why, and any
  file `git status` still shows that this pass left alone because it was not
  this session's.
