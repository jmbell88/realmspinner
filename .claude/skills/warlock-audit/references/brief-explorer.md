# Explorer brief

Fill the `{...}` slots, prepend the constraints block, and send as the prompt. One brief per
**segment** of the slice, not one per slice; the segments are listed on the row itself.

---

You are auditing one segment of one slice of Warlock Studio, a fully offline desktop app for
making game assets (one pygame window, one ModernGL context, imgui panels; heavy work in
child processes). You are **read-only**: you report facts about the code, you change nothing.

**Slice:** `{slice key}`
**Segment:** `{segment key}` — one of `{the row's segment keys}`; sibling segments are being
read right now by other explorers, so read yours and trust them with theirs.
**Files you own:** `{the segment's paths, pasted from the row}`
**Working tree at:** `{HEAD hash}` (findings are against the working tree; note any file
you open that `git diff --stat` shows as modified)
**Probes directory:** `{probes path}`
**Theme (only for a theme scope, else "none"):** `{theme item}`

## What to read, in this order

1. `CLAUDE.md`, the whole file. It is short and names the constraints.
2. The slice row below — the whole row, so you can see which files your siblings hold, then
   your segment's line. Open each *INVARIANTS lead-in* your segment names in
   `docs/INVARIANTS.md` and read that paragraph: each is a hard rule plus the incident
   behind it. A violation of one of these is a finding by definition.
3. The defect-class checklist below. Each item is a kind of bug this codebase has grown
   before; walk **every** item against **your segment's** files — the split is by file, so
   no class is somebody else's job. If a theme was given, walk *only* that item, but across
   every file in the row.
4. Your segment's source files, then the tests and manual chapters your segment names (the
   row's, where the segment does not name its own). Read tests to learn what is already
   guarded, and manual chapters to learn what is promised.

```
{the slice's row from subsystems.md, pasted whole}
```

```
{the checklist from defect-classes.md, pasted whole}
```

## How to judge

- **Freshness first.** Before you write a record, confirm the claim is true of the file
  as it is *now*. If a test already refuses the behaviour, name it in `existing_guard`
  and do not report the finding. An audit that lists work already done is itself a
  defect, and it has happened here twice.
- **Severity** uses these words only. *Critical*: crash, data loss, or corrupts a
  document. *High*: a wrong result the user will hit, or a promise in the docs the code
  breaks. *Medium*: a defect with a workaround, or a hard-invariant violation not yet
  reachable by a user. *Low*: hardening, consistency, naming.
- **Evidence** uses these words only. *Reproduced*: you ran something and saw it; the
  script is in the probes directory as `{slice}-NN.py` and runs with
  `uv run --no-sync python <path>` offline, with no weights and no GPU. *Inspected*: the
  claim follows from the code path and you cite every line it depends on. *Evidence gap*:
  proof is missing from the repository (a promise with no test, a figure that is an
  estimate); not a claim that the feature fails.
- **Human-only** is a finding that needs a card, a clean machine, real Aseprite, real
  Tiled, or ears to settle. Record it; the orchestrator moves it to `TODO.md`.
- **Your segment is a reading boundary, not a blinker.** If a file you own leads you into a
  sibling's file and the defect is there, report it with `where` pointing at the real file
  and say in the record that it is outside your segment. The merge deduplicates; a defect
  two explorers see from two sides is the strongest kind of record this audit gets.
- The defect, not the fix. A record whose `claim` is "should use X" is a fix; rewrite it
  as what goes wrong without X.
- Do not report style, formatting, or anything `ruff` would catch.
- Do not report a file's size or a module's shape unless it hides a defect you can name.

## Record shape — one per finding, nothing else in the block

```
- id: {slice}-{segment}-NN          (provisional; the orchestrator renumbers at merge)
  severity: Critical | High | Medium | Low
  evidence: Reproduced | Inspected | Evidence gap
  where: path:line [, path:line ...]      (repo-relative, against the working tree; may
                                           name a file outside your segment)
  claim: one sentence, the defect not the fix
  why_it_matters: one sentence, what a user or the tree loses
  existing_guard: tests/path.py::test_name, or "none"
  suggested_fix: one or two sentences
  regression_test_name: the claim as a test function name, e.g. test_play_refuses_a_stale_buffer
  human_only: yes | no
```

## Return

1. The records, Critical first.
2. A coverage note, about **your segment**: files opened (count and the list), files in your
   segment you did not open and why, checklist items you could not evaluate here, and
   anything that needs hardware. Name any sibling segment's file you leaned on so the merge
   knows the reading was not blind, and say plainly if you ran out of room — an unread file
   in your segment reported as unread is worth more than a silent gap.
3. Nothing else. No summary paragraph, no recommendations beyond `suggested_fix`.
