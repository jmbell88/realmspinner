# Clay's agent benchmark, pre-registered before any subject is graded — 2026-09-10

**Status: pre-registered. No model has driven this surface for a score, and no
grade exists.** Everything under "The corpus" and "Decision rules" below is
fixed now, before the first session. [`2026-08-09-grade-scale.md`](2026-08-09-grade-scale.md)
puts the reason in one line: *"Written now, this document is a commitment;
written after the grades exist, it is a rationalisation with a table in it."*

This is the shape [`2026-08-30-art-verdicts-preregistration.md`](2026-08-30-art-verdicts-preregistration.md)
and [`2026-09-07-mesh-probe-preregistration.md`](2026-09-07-mesh-probe-preregistration.md)
already use, and it is written for the same reason the second one was: *now* is
the last moment at which nobody could have tuned a result to it.

## The question

An external reviewer drove Warlock's MCP bridge, filed five defects and a
five-part roadmap, and its closing complaint was not about any single tool. It
was that an agent could *assemble* — place primitives, boolean them, name them
— and could not *model*. Four tranches of work since have answered that:
element mode and selection over MCP, a transport an agent can recover on,
curved and swept generators, repetition and placement ops, and (2026-09-10)
object names that a batch can address by `$ref`.

None of it has been scored. The capability is asserted by unit tests that each
prove one tool does one thing, and by nothing at all that asks whether the
surface, taken together, can build a recognisable object from a brief. That
gap is what this document fixes the rules for.

## What can honestly be scored by a machine, and what cannot

Splitting this three ways is the whole design, and the split is not a
convenience — each tier can answer a question the others cannot, and none of
them can answer all three.

**Tier one — the suite, unattended, no model.**
`tests/test_agent_transcripts.py` replays a transcript against a real
`ClayDoc` through the real `agent_clay.call` door and asserts the outcome. It
answers exactly one question: *does this sequence of calls still build what it
built before?* It cannot answer whether a model would find that sequence, and
it cannot answer whether the result looks like a chair. It is a regression
gate, and calling it a benchmark score would be the dishonest part.

**Tier two — a real model, over the real pipe.** `scripts/agent_bench.py
--serve` stands the app up with a throwaway home, the bridge on, and
`WARLOCK_AGENT_TRANSCRIPT` pointed at a file, then prints the corpus and the
connect line. It cannot run in CI at all: it needs a window, a GL context and
an MCP-speaking model. It answers *how many calls a real model took, what it
tried, and what it was refused* — and it leaves a recorded transcript, which
is how a tier-two session becomes a tier-one fixture.

**Tier three — a person, looking.** Whether the thing is recognisable, whether
its proportions read, whether an eight-column colonnade looks like
architecture or like eight cylinders. No measurement in this repository can
answer that and none should pretend to. It is `TODO.md` P43.

## The corpus, fixed now

[`corpora/clay-agent-v1.txt`](corpora/clay-agent-v1.txt), five subjects, in
the established `class | prompt` format and parsed by the same
`campaign_props.read_corpus` the three image corpora use. The subjects are the
reviewer's own five, carried over in substance: a chair, a mechanical bracket,
a telescope, a repeated architectural assembly and a curved creature. They
were chosen by somebody who had just driven this surface and found it wanting,
which is a better basis for a first corpus than a list written by the people
who built it.

**The class means difficulty of *construction*, not of recognition** — how
much of Clay's surface the subject cannot be built without. That is the thing
under measurement, and it is why a chair is `easy`: six boxes is genuinely all
it needs. The file states the three bands; they are not restated here, so
there is one definition.

One subject is `hard` and it is deliberately the newest capability: a body
that tapers along a path is what `sweep` and `tube` were added for on
2026-09-10, and it has the least evidence behind it of anything on the
surface.

## What exists today, stated so it cannot be overstated later

- **Two transcripts, both hand-authored, neither recorded.** `chair` and
  `spoked-hub` under `tests/fixtures/agent_transcripts/`. They were written by
  reasoning about the geometry and running the result, not by a model, and
  every file that mentions them says so. `spoked-hub` is not a corpus subject:
  it exists to exercise `clay_batch` with `$ref`, `array-radial` and the
  plural `clay_set_params`, which is a different job from being a benchmark
  case.
- **Of the five corpus subjects, one (`chair`) has a transcript.** The other
  four have none, by anybody, and the honest N for this benchmark today is
  therefore **zero graded subjects and zero model sessions**.
- **No recorder output exists yet.** The recorder landed the same day as this
  document and has never been pointed at a real model.

## Decision rules, fixed before the first session

Applied verbatim afterwards, including the boring outcome.

1. **A subject is *buildable* if a tier-two session produces a document that
   exports**, whatever it looks like. Recorded per subject as built / not
   built, with the refusal that stopped it if it was not.
2. **Call count is reported, never graded.** It is the number a later tranche
   would want to see fall, and it is meaningless across subjects of different
   classes. No threshold is set on it here, because any threshold set before a
   single session would be invented rather than measured.
3. **The mesh grade scale is cited, not restated**: integer −5..+5, as declared
   in [`2026-08-09-grade-scale.md`](2026-08-09-grade-scale.md). Tier three
   grades each subject on it, from renders, blind to the call count — the
   number of calls it took must not be visible while the picture is being
   judged, because it is exactly the kind of thing that talks a grader into a
   verdict.
4. **The bar for "the surface can model, not merely assemble" is three of five
   subjects at grade 0 or better, with at least one of them `medium` or
   `hard`.** Set now, deliberately modest, and deliberately requiring that the
   pass is not carried entirely by the `easy` subject.
5. **A refusal an agent could not have avoided is a defect, not a score.** Any
   session that hits one is written up as a finding whatever the grade, and
   the grade is still recorded rather than discarded — a subject can be both
   buildable and evidence of a bug.
6. **Retention**: the transcript and the exported GLB of every graded session
   are kept until the results document is written.
   [`2026-08-13-tier-qualification.md`](2026-08-13-tier-qualification.md)
   scored zero usable of twenty and could not diagnose why, because its assets
   were gone; that instruction is honoured here.

## What would invalidate this pre-registration

Editing `corpora/clay-agent-v1.txt` once a grade exists. If the corpus turns
out to be wrong — a subject that cannot be expressed at all, a class that was
misjudged — that is a finding to write down and a *new* corpus to start, not
an amendment here. The same rule the image corpora carry at the top of their
own files.
