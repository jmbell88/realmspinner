# The shape of a sitting's writeup

Derived from reading, not inventing: the full text of
`docs/measurements/2026-08-30-art-verdicts-preregistration.md` and the
opening sections of `docs/measurements/2026-09-06-native-batch-6-landed.md`
and `docs/measurements/2026-09-06-native-batch-8-retexture.md`. The shape
below is what those three documents actually do, in the order they do it,
not a template invented to look like them.

## The sections, in order

1. **Title and date.** `# <programme name> — <what the day is about>` then
   the date on its own, e.g. *"2026-08-30, on top of `5676b777`."* Name the
   commit or machine state the run happened against when that matters (the
   native-batch docs do this; the art-verdicts document opens with a status
   line instead, because it is a pre-registration written *before* any run).
2. **Why now.** What made this the day to run it — what shipped recently that
   this measures, what else is pressing on the same subsystem, what the
   document is correcting or preventing. The art-verdicts document's version:
   *"The tree is therefore accumulating unmeasured capability faster than it
   is accumulating evidence. This programme is the correction."*
3. **"This is the retry X demanded" — only where one exists.** When an
   earlier document ended with explicit instructions for whatever ran next
   (the art-verdicts document answers three numbered instructions from
   `2026-08-13-tier-qualification.md` by name), quote them and answer each in
   turn. Do not manufacture this section when there is no prior retry
   condition to satisfy — P28, for instance, has no such antecedent document
   and the section is simply absent there, not filled with something that
   pretends to be one.
4. **What is under test.** The shipped default path and nothing else — list
   every knob at the value it actually holds (mesh profile, texture
   resolution, background removal, LoRA/IP-Adapter/ControlNet state, the
   seed, the prompt template's own boilerplate). The art-verdicts document's
   rule, stated outright: *"Anything this programme could override is a
   departure from the configuration the verdict is about."* If a sitting
   changes even one knob from the shipped default, that is grounds to say the
   result is about that knob, not about "the product."
5. **The instrument — cited, never restated.** A named measurement document
   already carries a scale, a threshold or a rubric (the mesh grade scale in
   `2026-08-09-grade-scale.md`, the QA thresholds in
   `2026-09-02-troupe-qa-thresholds.md`, a corpus's own difficulty classes).
   Link it and quote the one line that matters; do not re-derive the scale in
   the new document. Restating it is how two documents drift apart the first
   time one of them is edited.
6. **What will be run.** The exact corpus or subject list, the exact script
   or in-app path, the sample size and why it is that size (the art-verdicts
   document justifies 22 subjects at one seed each, not five seeds each,
   directly from 2026-08-13's own data: "three of its four subjects scored
   zero accepts across *five* seeds each, so subject dominated seed by a wide
   margin"). Say whether the pass is blind and what blinding does and does
   not buy on this particular corpus — see `setup-recipes.md`'s note that
   blinding cannot hide a subject the prompt itself names.
7. **Decision rules — fixed now, before a single result exists.** Every
   branch, stated as a threshold with a consequence, not as "we'll see how it
   looks." The art-verdicts document's Q1 is the clearest example: three
   bands (≥50%, 25–49%, <25% usable) each tied to a specific claim the README
   may or may not make afterwards. **Whichever rule fires is applied
   verbatim, including the boring one.** Quote
   `2026-08-09-grade-scale.md`'s own line for why this section exists at all:
   *"Written now, this document is a commitment; written after the grades
   exist, it is a rationalisation with a table in it."* A decision rule
   written after seeing the pictures is not a decision rule; it is the
   opposite of what this section is for, and a writeup that reorders these
   two must say so rather than hide it.
8. **Retention.** What survives the sitting and under what tag, and what
   would have to happen for it to be deleted before the writeup exists
   (`sweeps.cleanup_sweep`'s refusal of `RECENT_ID`, a corpus tag surviving a
   "Clean library" press unless someone does it by hand). The art-verdicts
   document's rule, inherited from the run whose evidence was lost the first
   time: *"The library is not cleaned until every writeup exists."*
9. **Results.** Filled in after the sitting, from what the human actually
   said — not from what the skill predicted the human would say. Usable-of-N,
   per class as well as overall where the instrument has classes, exact
   grades or tag counts, verbatim quotes from the human where the entry's own
   question list asked something open-ended.
10. **Verdict.** The decision rule from section 7, applied, named as having
    fired — "the ≥50% rule fired" or "the <25% rule fired" — not restated as
    a fresh judgement. If a result is genuinely ambiguous against the stated
    rules, that is itself worth a sentence, but it is not licence to invent a
    fourth rule after the fact.
11. **What this changed in the tree.** The `TODO.md` strike, the manual line
    lifted or left, the README claim added or withheld, the constant that
    moved from "estimated" to "measured," the follow-on entry this opens or
    closes.

## Two rules to hold to, stated as rules

- **Whichever decision rule fired is applied verbatim, including the boring
  one.** See the quoted line above. If a result technically clears a
  threshold by one unit and the honest reading is "this is really a
  borderline case," record that as a note beside the applied verdict — do not
  quietly relax the rule to make the borderline case feel less final.
- **Never write a `YYYY-MM-DD-` placeholder path.**
  `tests/test_external_doc_links.py` reads every `.md` citation across
  `SOURCES` (confirmed: this includes `README.md`, `CLAUDE.md`,
  `docs/INVARIANTS.md`, `docs/MODELS.md`, `TODO.md`, `docs/COMPAT.md`, and
  every file under `docs/measurements/*.md`) as a claim that the cited file
  exists, and fails the suite if it does not. `TODO.md`'s own P28 entry says
  so explicitly, and names the pattern to use instead: cite the document by
  its directory (`docs/measurements/`) until it exists, then cite it by its
  real, dated name once it is written — never invent the date in advance and
  cite a file that is not there yet.

## Closing the `TODO.md` entry

- **Strike with `~~...~~` and a dated sentence beside it**, in the same voice
  every other closed line in `TODO.md` uses — read a handful of existing
  struck lines in the file before writing one; do not tick a checkbox, there
  are none.
- **Entry numbers are stable and never reused.** A closed `P<N>` stays
  `P<N>`, struck in place; a one-line record of what it decided moves to
  *Closed records* at the bottom of `TODO.md`, which is where the file's own
  header says a closed entry's outcome belongs so nobody re-derives it.
- **A verdict can lift a manual chapter's status line.** P28 is the
  concrete case: a per-archetype "proven" verdict lifts
  `docs/manual/11-a-character-sprite-sheet.md`'s *"Built, awaiting the render
  benchmark"* line (confirmed at line 155) toward *"Proven — per archetype,
  not in one line,"* and a repair verdict for one archetype instead becomes a
  new, ordinary buildable finding rather than closing the entry outright —
  P28 does not close until all four archetypes have an answer. Check every
  entry's own manual cross-reference the same way before assuming there is
  nothing to lift.
- **Hand off to `/warlock-land` for the commit.** Do not restate its
  checklist here or in `SKILL.md` — it already covers separating this
  session's edits from whatever else is dirty, proving the regression tests
  the writeup's own claims imply, paying what the change owes beyond `src/`
  (the manual chapter, `docs/INVARIANTS.md`, `CHANGELOG.md`, `TODO.md`), and
  running the gate once. A sitting's writeup is exactly the kind of
  behaviour-adjacent change that skill exists for; call it rather than
  reimplementing any part of it.
