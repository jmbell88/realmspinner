# Clay's agent benchmark — the first graded sitting

2026-09-15, on top of `0f2bdafe`. Written up 2026-09-16.

## Why now

[`2026-09-10-clay-agent-benchmark-preregistration.md`](2026-09-10-clay-agent-benchmark-preregistration.md)
fixed a corpus and a bar before anyone had driven Clay's MCP surface for a
score, and said the honest N was **zero** — zero model sessions, zero graded
subjects. Every capability on that surface (element mode, swept and curved
generators, repetition ops, `$ref` names, `clay_program`) was asserted by unit
tests that each prove one tool does one thing. This sitting is the first time a
person looked at what a model built from a brief with all of it together.

There is no retry section: the pre-registration is the first document of its
programme and demands nothing of a rerun.

## What is under test

The Clay tool surface as shipped at `0f2bdafe`, driven by Claude through a
Claude Code MCP client connected to `uv run python scripts/agent_bench.py
--serve`, with the bridge on and the recorder writing. No tool, tool
description or catalogue was altered for the sitting. The catalogue includes
the ten `character_*` tools, which no corpus subject asks for; P43's own
addendum named the context they cost every session.

## The instrument, cited

The mesh grade scale, [`2026-08-09-grade-scale.md`](2026-08-09-grade-scale.md):
integer −5..+5, where *"+5 — game-useable as-is"*, *"−5 — completely
unusable"* and *"0 — no opinion either way, and it is a real answer rather than
a missing one"*. The difficulty classes are the corpus file's own,
[`corpora/clay-agent-v1.txt`](corpora/clay-agent-v1.txt), and are not restated.

## What was run

The five corpus subjects, prompts verbatim. The pre-registration asked for one
subject per session, five sessions. What ran differs, and is stated here rather
than smoothed over:

- **One recorded file, two complete passes.** The transcript holds 108 calls:
  lines 1–63 attempt all five subjects, and lines 64–108 attempt all five
  again. **Only the second pass was graded.** The first is reported below as a
  discarded run, and nothing in it contributes to a grade. The file does not
  say why there were two.
- **Documents were separated inside one session, not by sessions.** Between
  subjects the agent's document was closed (by the operator, as far as the
  file can show), and the agent's next call landed on a session with no
  document, which is where most of the refusals below come from. The model built the telescope and the colonnade in **one**
  document (verified by replay: 20 objects, exactly the two saved files'
  9 + 11), and the operator split it by hand when saving.
- **Blindness to call counts is not claimed.** The operator ran the sessions,
  and nothing on record says the call counts were out of view while grading,
  so this document cannot assert rule 3's condition.

Each graded document's extent in the transcript (line numbers as
`agent_bench.py --show` prints them), established by replaying each slice in a
fresh session and matching the saved file's object names exactly:

| Subject | Class | Lines | Calls | Objects | Saved as |
|---|---|---|---|---|---|
| chair | `easy` | 67–69 | 3 | 10 | `build1.wblk` |
| bracket | `medium` | 71–78 | 8 | 3 | `build2.wblk` |
| telescope | `medium` | 82–89 | 8 | 9 | `build3.wblk` |
| colonnade | `medium` | 90–91 | 2 | 11 | `build4.wblk` |
| serpent | `hard` | 93–108 | 16 | 20 | `build5.wblk` |

The remaining 8 calls of the pass (lines 64–66, 70, 79–81, 92) are the
handovers between documents.

## Decision rules

Fixed on 2026-09-10 and cited, not re-derived: the pre-registration's own
section, rules 1–6. The grade scale's line for why they come first: *"Written
now, this document is a commitment; written after the grades exist, it is a
rationalisation with a table in it."*

## Retention

Rule 6 asks for the transcript and the exported GLB of every graded session.
What exists:

- **The transcript**, [`data/clay-agent-v1/transcript.jsonl`](data/clay-agent-v1/transcript.jsonl),
  tracked.
- **The five graded documents** as the operator saved them,
  `docs/examples/P43/build1.wblk` to `build5.wblk`. These are **local to the
  machine that ran the sitting**, because `docs/examples/` is gitignored.
- **GLBs and renders regenerated from those documents**, in
  `docs/measurements/data/clay-agent-v1/`. The five GLBs
  (`{chair,bracket,telescope,colonnade,serpent}.glb`) came through
  `bd.to_model` and `glbwrite.write_glb`, the encode half of
  `clay_mode.build_asset`. The renders are a lit three-quarter/front/left sheet
  of each, through `ClayView.render_png` on a standalone GL context. Both
  extensions are gitignored under `data/`, as every render there is.

**The GLBs the session itself exported are gone.** The graded pass never called
`clay_export`. The first pass did, twice, and `--serve`'s throwaway home was
deleted when the window closed, taking both Library rows with it. This is the
second time this repository has lost a sitting's assets —
[`2026-08-13-tier-qualification.md`](2026-08-13-tier-qualification.md) was the
first — and it is fixed below.

## Results

### Grades (rule 3)

| Subject | Class | Grade |
|---|---|---|
| chair | `easy` | **+4** |
| bracket | `medium` | **+3** |
| telescope | `medium` | **+2** |
| colonnade | `medium` | **+5** |
| serpent | `hard` | **−3** |

### Buildable (rule 1)

All five are **built**: every graded document serialises and encodes to a GLB
through the export path's own encoder (16 KB to 245 KB). None stopped on a
refusal. Rule 1's export half was met offline, from the saved documents, not by
`clay_export` inside the session, so the graded pass did not exercise the
Library-row import.

### Call counts (rule 2, reported, never graded)

As in the table above: chair 3, bracket 8, telescope 8, colonnade 2, serpent
16, plus 8 handover calls, for 45 in the pass. The chair and the colonnade are
each one building batch followed by a render.

### Refusals (rule 5)

The recorder kept no refusal messages, so every one below was recovered by
replaying the transcript through `agent_clay.call`. Replay cannot recover a
refusal that depended on live state the file does not carry.

**Graded pass, eight refusals:**

| Line | Call | What replay shows | Avoidable? |
|---|---|---|---|
| 65, 66 | `clay_batch` building the whole chair | Succeeds in an empty document. With the first pass's chair still present it refuses *"an object is already named 'seat'."* The model then added `seat` alone and batched the rest. | Yes — a name collision the message names |
| 70, 92 | `clay_scene` | *"This session has no document yet. Call clay_add_primitive, clay_add_figure or clay_add_mesh first."*, verbatim | Consistent with the operator closing the document |
| 80 | `clay_delete` of the bracket's own three uids | **Succeeds on replay.** Undiagnosable. | Unknown |
| 81 | `clay_batch` building the telescope | **Succeeds on replay.** Undiagnosable. | Unknown |
| 104 | `clay_op shade-smooth` | **Succeeds on replay.** The model selected the parts and the retry succeeded. Undiagnosable. | Unknown |
| 105 | `clay_select_by` with no arguments | *"no object with uid None."* | Yes, but the message was a defect (below) |

**First pass, four refusals (not graded):** line 27's `clay_program` refused
with *"steps[1].add: id 'seg1' collides with an object already in the live
document."* and line 36's `clay_batch` with *"an object is already named
'base_step_1'."*. Lines 23 and 35 do not reproduce.

**No refusal was shown to be unavoidable.** That is weaker than "there were
none": three graded refusals cannot be classified at all, and the reason is the
first finding.

### Findings, all built with this document

1. **The recorder threw every refusal message away.**
   `agent_transcript.record` now writes the refused call's own sentence as an
   `error` key, on refused lines only, so every earlier transcript and fixture
   stays valid. `--show` prints it.
   `tests/test_agent_transcript_rules.py::test_record_keeps_the_refusal_message_a_refused_call_answered_with`.
2. **`--serve` deleted the home holding its own exports.**
   `_appharness.KEEP_HOME_ENV` skips the cleanup. `agent_bench.py` sets it
   above its harness import and prints the kept path on the way out.
   `tests/test_exercise_mode.py::test_isolate_home_keeps_the_throwaway_home_when_told_to`
   and
   `tests/scripts/test_agent_bench.py::test_serve_asks_the_harness_to_keep_its_home_before_the_harness_decides`.
3. **A missing `uid` was reported as a missing object.** Line 105's *"no object
   with uid None."* named a uid the model never passed and gave `recovery:
   read_scene`. The shared resolver behind `clay_select_by`, `clay_transform`
   and every other tool that resolves one object by uid now answers *"give a value for 'uid'."* with `recovery:
   fix_arguments`.
   `tests/test_agent_clay.py::test_a_call_with_no_uid_is_told_to_give_one_rather_than_that_uid_none_does_not_exist`.

### P43's five questions — DRAFT, from transcript evidence, for the operator to correct

The operator asked for these to be drafted rather than left blank. None of this
is yet in the operator's own words, and no sentence here moves a grade.

- **Did it build the subject, or something else it found easier?** Four of the
  five built the subject as briefed. The bracket's holes are two cylinders
  subtracted from the plate, and its gusset is a `sweep`. The colonnade's
  column is a 48-point fluted `sweep` joined to a plinth, a capital and an
  abacus in one batch, then arrayed radially into eight; the model first tried that sweep as a scratch
  object (`shaft_test`) in the telescope document and deleted it. **The
  serpent is something easier**: `clay_add_figure(key="quadruped")` and twenty
  preset parts moved around. It used no `sweep`, no `tube` and no taper along a
  path, which is the one thing the `hard` class exists to ask for.
- **Which refusals could it not have avoided?** None demonstrated. Three can't
  be diagnosed from the file (see the table).
- **Where did it stop — surface or patience?** Every document ended on a
  render, not a refusal. The serpent ended after five renders, three
  repositioning batches and a `shade-smooth`. It never reached the part of the
  surface built for its shape, so this sitting does not show that surface
  running out.
- **Is there a shape in the corpus this surface cannot express?** No evidence
  either way for the one that matters. A profile swept straight works (the
  colonnade). A body tapering along a curved path was never attempted.
- **Plan or flailing?** The chair and the colonnade read as a plan executed in
  one batch. The bracket and the telescope are built part by part, with renders
  between. The serpent reads as iterative adjustment of a preset. The file does
  not explain why a full first pass exists.

**Still owed by the operator:** the exact model version (the transcript does not
record the client), why there were two passes, and whether the grades were
given without the call counts in view.

## Verdict

**Rule 4 fired: the bar is met.** Four of five subjects are at grade 0 or
better, and three of those four are `medium`, so the pass is not carried by the
`easy` subject. The surface can model, not merely assemble, on the
pre-registered and deliberately modest definition.

Noted beside the verdict, not instead of it: the one `hard` subject, the class
chosen because it has *"the least evidence behind it of anything on the
surface"*, graded −3 and still has no evidence behind it, because the model
routed around it. The verdict is about the surface. This subject says nothing
yet about `sweep` or `tube` along a path.

## What this changed in the tree

- `TODO.md` P43 closed, one line under *Closed records*. P46 (whether the bridge
  stays lockstep) now reads this document's transcript instead of waiting on
  P43, and says the file carries no timings.
- Findings 1–3 above, built with regression tests.
- Three second-pass documents promoted to tier-one fixtures under
  `tests/fixtures/agent_transcripts/`: `p43-chair`, `p43-bracket` and
  `p43-telescope-colonnade`. They are sliced verbatim, with `clay_render` lines
  dropped and the handover refusals left out, as `tests/test_agent_transcripts.py`'s
  docstring states. The serpent was not promoted.
- The pre-registration is unchanged. Its "what exists today" paragraph is a
  statement of 2026-09-10, superseded by this document rather than edited.
