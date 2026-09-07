# Voice — three registers, three real examples each

Every example below was read out of the tree on 2026-09-07, not invented.
Match the register you are writing in; do not blend them.

## Commit subjects

`git log --format=%s -30` is the source. The pattern: a sentence about what
changed *for the user or the project*, present tense, most often naming the
subject that gained an ability rather than the mechanism that gave it one.
Never "fix findings," never "land change," never a conventional-commits
prefix.

- `Poser can open a rigged asset's own mesh, not just a bare skeleton`
- `Muse names its fields, opens the lyric box, and reaches ten minutes`
- `A cancelled take stops costing the loaded music model`

Notice the shape: subject, verb, what changed — and where the change is a
fix, the sentence names what stops happening ("stops costing," "stops
destroying") rather than what code moved. A subject can be a mode ("Poser"),
a surface ("Muse's brief bar"), or the defect's old behaviour framed as now
absent ("A cancelled take stops costing..."). None of the thirty read for this
skill used an imperative mood ("Add," "Fix," "Update") as the sentence's own
verb.

## `CHANGELOG.md` entries

The file states its own voice at the top, in **A note on how this reads**,
quoted here in full because it is the instruction, not a summary of one:

> These entries are written for whoever maintains this next, which means they
> name the measurement that made a default wrong, the review score that
> condemned a mode, and what a crash actually was — in the belief that a fix
> nobody can audit is a fix nobody should trust.

Three real entries, under their version headings:

- **0.0.39** — `**Poser can open a rigged asset's own mesh, not just a bare
  skeleton.** Until now Poser authored poses against a meshless armature
  preview only — the same generic humanoid, quadruped, bird or blob skeleton,
  with no way to see what a pose actually looked like on a real character.
  The inspector's Pose tab already had a working, skinned, textured pose
  editor for one specific asset; Poser now shares that same machinery on its
  own viewer...`
- **0.0.38** — `**Cancelling a music take no longer throws away the loaded
  model.** The music worker attaches its vitals — including whether a
  checkpoint is loaded — to every answer it sends back, except that the cancel
  reply was built by hand and left them off. The app reads its own "is a model
  loaded" flag from that field, so a missing one read as "nothing loaded"...
  Found by the GPU lane, which asserts exactly that and had been failing.`
- **0.0.38** — `**The Text tool stopped taking the app down.** An ordinary
  paragraph at a large point size measured past the decode ceiling, which
  raised where the caller had no handler at all, so the window closed with the
  native "had to close" dialog. It declines now, like every other thing that
  tool refuses.`

The shape every entry shares: a bold-lead sentence stating what changed, then
what the *old* behaviour actually was (not "there was a bug" — the specific
wrong thing, often with the mechanism), then what fixed it and, where one
exists, the measurement or the failing test that proves it. A short entry
still names the mechanism; length varies with how much the fix required, not
with how much license the voice gives you to be vague.

**Structural fact:** the top heading (`## 0.0.39 — 2026-09-07`) must name the
same version as `pyproject.toml`'s `version = "..."` line —
`tests/test_changelog.py` asserts it — so a version bump that edits one and
forgets the other fails there.

## `docs/INVARIANTS.md` paragraphs

Same shape every time: a **bold lead sentence stating the rule**, then the
incident that bought it, with dates and test names folded into the prose
rather than listed separately. Paragraphs are long — these three are excerpted
fairly, with `...` marking a cut, not a stand-in for content that would change
the sense.

- `**Single sqlite3 connection, serialized by an explicit lock.** JobStore
  (db.py) wraps one sqlite3 connection opened with check_same_thread=False,
  and every method that touches it takes self._lock (an RLock) first.
  asyncio.to_thread does *not* serialize anything... And a params blob that
  will not parse is answered as {} rather than raised (CON-06: one corrupt
  blob raising out of next_queued starved every job behind it forever), which
  makes the damage invisible to every reader — JobStore.unreadable_params
  exists so that service.library.verify is the one place it is visible.`
- `**"Not tried yet" is never spelled the same way as "tried and failed".**
  matte_preview.MatteState.failed_stamp defaulted to None, and
  service.matte.stamp_for also returns None for a job whose input.png is
  missing, so pump's guard against re-asking after a failure —
  if state.failed_stamp == stamp — was accidentally true on the very first
  check... (the 2026-09-05 audit, finding create-03). The latch is now its
  own _tried_and_failed flag... Any "already tried and it was bad" latch keyed
  on a nullable value owes the same separation: a sentinel that merely differs
  by convention from the field's valid values is not a sentinel.`
- `**A pack-install timeout may not kill during the commit phase either.**
  service.packs._run_worker tracks the worker's last-announced phase
  (PHASE_DOWNLOAD/PHASE_COMMIT) and, on a TimeoutExpired reached after
  PHASE_COMMIT was announced, no longer force-kills the child — it waits for
  the child to exit on its own, however long that takes. Before the
  2026-09-07 audit (service-04) the four-hour ceiling killed the child
  regardless of phase... A hung install is visible and recoverable; a
  half-written one is neither.`

Three things every one of these does that a thinner invariant paragraph
skips: it names the specific test or finding id that proves the rule is
enforced (`test_frame_thread_doors.py`, "finding create-03", "service-04");
it states the general rule the specific incident is an instance of, usually
in a second bold sentence near the end ("Any... latch keyed on a nullable
value owes the same separation"); and it gives a date, because the reader is
meant to be able to ask "is this still true, or has something moved since."

**Structural fact:** there is no `- ` list marker on these paragraphs and no
per-subsystem heading to navigate to. The whole `## Hard invariants` section
(roughly 287 of them, in the current tree) is one flat run of paragraphs
under a single heading. "Read the section for a subsystem" means grepping the
file for words the paragraph would use, not opening it and scrolling to a
heading with that subsystem's name.
