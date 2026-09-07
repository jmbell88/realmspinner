# What a behaviour change owes — five obligations

Walk all five against the session's diff. For each, decide: applies and paid,
applies and now paid (because this step just paid it), or does not apply —
and say which, in the report. None of these is optional to *consider*; not
all of them apply to every change.

## 1. The `docs/manual/` chapters that describe what moved

If the change moves, renames, or changes the effect of a control a manual
chapter names, that chapter is wrong the moment the commit lands, not at some
later doc pass. `docs/manual/` chapters are numbered 01–19 for the tutorial
series and beyond that by part; a chapter's number decides its order, so
adding one is a renumbering exercise gated by `tests/manual/` in both
directions — do not invent a number by hand.

**What gates this, and what does not.** `tests/manual/test_docs.py` checks the
manual's own structure — links, anchors, headings — and will catch a broken
cross-reference. It does not check that a chapter's *claims* are still true of
the app. That half is `tests/manual/test_manual_promises.py` and
`test_manual_labels.py`, and both are bespoke: each assertion in them exists
because a specific drift was already found and named (four findings from the
2026-09-06 docs audit alone — a zoom ceiling, a purged `examples/` directory,
a skeleton template count, an "only place" claim). There is no general gate
that notices a fifth kind of drift on its own. If this change altered a
number, a label, a keyboard shortcut, or a claim of the form "X is the only
way to Y" that a manual chapter states, either the chapter is fixed in this
same commit or a new bespoke assertion is added guarding it — CLAUDE.md's own
rule is that manual changes ship in the same commit as the behaviour change,
not after.

## 2. `docs/INVARIANTS.md`

**Structural fact, so you look in the right place.** The file is not a
bulleted list. Its `## Hard invariants` heading (line 9 in the current tree)
is followed by a flat run of roughly 287 paragraphs, each one beginning
directly with a **bold lead sentence** and no list marker — a paragraph can
run to a thousand words, folding in every incident, date, and test name that
bears on the rule. "Read the section for a subsystem" therefore means
grepping for words you expect the paragraph to use (a module name, a job
kind, a constant), not navigating to a heading — headings only separate the
top matter, `## Hard invariants` itself, and the closing `## Stack` section.

**What to check.** Does this change make an existing bold-lead sentence false
(a threading rule that no longer holds, a figure that moved, a "the only
place X happens" claim that is now the second place)? If so, that paragraph
is edited in this commit, not left to rot the way the Muse duration claim did
before the 2026-09-06 pass corrected it. Did this change buy a new constraint
with an incident — a race found, a crash reproduced, a silent-wrong-result
caught? If so, a new bold-lead paragraph is owed, in the same voice
(`references/voice.md` gives three real ones and how to write in it). Sixteen
of the last sixty commits touched this file; a change with no `src/` in a
subsystem this file already describes is the unusual case, not the common one,
and unusual is worth a second look before deciding nothing is owed here.

**No test gates this file's content in general** — a small, deliberately
partial set does: `tests/test_root_doc_inventories.py` catches drift in a few
specific counted claims (a stale exact test count, a stale commit-hash
citation, a specific module name), and nothing broader. Getting this
paragraph right is a reading judgement, not a test you can run to check your
work.

## 3. `CHANGELOG.md`, under the current version heading

Every change a user would notice gets an entry under the version heading at
the top of the file — read `references/voice.md` before writing one; the file
states its own voice explicitly and three real entries are quoted there.
`CHANGELOG.md`'s top heading must name the same version as `pyproject.toml`:
`tests/test_changelog.py::test_the_shipped_file_parses_and_leads_with_this_version`
gates that lockstep, and
`test_install_md_names_the_installer_this_version_actually_builds` gates
`INSTALL.md`'s installer filename against the same version. The fourth place a
version is written, `src/warlock/__init__.py`'s `__version__`, is gated
separately by
`tests/test_offline.py::test_the_packages_version_matches_the_distributions`,
which compares it against the installed package metadata rather than against
`CHANGELOG.md` directly — so a version bump that gets `CHANGELOG.md` and
`pyproject.toml` right but forgets `__init__.py` still fails, just in a
different file. Ordinary landings (not `--release`) do not bump the version;
the entry goes under the version heading that is already there.

## 4. `TODO.md`, where an entry has narrowed or closed

If this change built something `TODO.md` named as owed, or answered a
question one of its entries was waiting on, that entry is updated now, in
this commit — not left for a later pass to notice the tree has moved past the
plan. The file's own rule, stated at its head: **a closed entry is struck
through, never ticked** (`~~## P<N>. ...~~ Closed <date>: <what was decided,
one line under Closed records>`), because a plan whose boxes disagree with
the tree is worse than no plan. A plan file that becomes wholly finished is
**deleted outright**, the way every earlier roadmap file in this repository's
history was — git keeps it — not left in the tree half-ticked.
`tests/test_ux_todo_fixes.py` sweeps dead plan filenames out of `src/` and
`scripts/`, and separately refuses any citation of `TODO.md` itself from
those trees: a module that needs to explain something explains it in
`docs/INVARIANTS.md` or a `docs/measurements/` document, never by pointing at
the plan file's own name.

## 5. The screenshots

There are two trees and they are not gated alike. `screenshots/` (the root
directory, `dark-*`/`light-*`/`pixel-*` per mode and component, produced by
`scripts/screenshot_modes.py`) has **no test referencing it at all** — the
"Refresh the mode and component screenshots" commit (`afe743ee`) touched 120
files and nothing in the suite noticed either the before or the after. If
this change altered a mode's visible layout, colours, or default content, the
screenshots are stale until someone runs the refresh script by hand and
nothing will say so before a person looks. `docs/manual/img/` is narrower and
partially checked: `tests/test_external_doc_links.py` and
`tests/manual/test_docs.py` verify that an image a chapter links to actually
exists at that path and that the markdown syntax is well-formed — they do not
check that the pixels still show the current app. A manual screenshot that
was true when the chapter was written and is now showing an old control still
passes both tests. Decide, for this change, whether either tree needs a
refresh; if screenshots/ needs one, say in the report that nothing will catch
it if it is skipped, because that is a true statement about the repository,
not a hedge.
