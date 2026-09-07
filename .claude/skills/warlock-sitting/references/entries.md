# The open judgement sittings — a snapshot, not a ledger

**Snapshot taken 2026-09-07, against `TODO.md` as it stood that day.** Entry
numbers are stable but not contiguous — a closed entry's number is retired,
never reused, and its one-line record moves to *Closed records* at the bottom
of `TODO.md`. **`TODO.md` is the authority, always.** This file is read for the
grouping and the per-entry detail it took an hour to assemble; the moment it
disagrees with `TODO.md` on whether an entry is open, what it asks for, or
what it is called, `TODO.md` wins and this file is stale. `SKILL.md` step 1
re-derives the live heading list from `TODO.md` itself for exactly this
reason — it does not trust the list below to still be current.

At the snapshot date, `TODO.md` carried thirty-two open `## P<N>.` entries:
P1, P4, P6, P7, P8, P10, P11, P13, P14, P15, P16, P17, P19, P20, P21, P22,
P23, P24, P25, P26, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37, P38,
P39. (`grep -n '^## P[0-9]' TODO.md` reproduces this list live.)

## How this is grouped, and why the grouping matters

A "judgement sitting" is specifically an entry whose *Do* list ends at a
human looking at pictures (or listening) and answering a fixed question
list. Not every open entry is that shape, and forcing one into the shape
this skill assumes would be manufacturing a ceremony the entry never asked
for — exactly what the brief that produced this skill warned against. So the
groups below are not the tidy four the skill's brief proposed; two of those
were right, one needed splitting, and two entries needed a fifth and sixth
bucket that the brief's four did not have room for. Each is noted where it
differs.

- **Group 1 — needs nothing but CPU and time.** The skill can produce every
  picture the entry asks for, unattended, and hand the human a finished
  gallery.
- **Group 1b — needs CPU and time, but only *after* a human has authored the
  source material.** The brief's Group 1 included two entries (P8, P33) whose
  *Do* list opens with drawing or posing, not viewing. That is not something a
  script can do instead of a person, so these are a different shape from
  Group 1: the skill's setup phase is bounded to what is mechanical (baking,
  export, the QA heatmap, threshold recalibration), and it must say so rather
  than silently produce a gallery from nothing.
- **Group 2 — needs a card and weights.** The skill can queue the run and
  drain it, but the run is long, sometimes hours.
- **Group 3 — needs hardware or senses this machine has not got: ears, a
  clean machine, a real foreign application, a published release.** The
  skill prepares materials and says plainly what it cannot do itself.
- **Group 4 — editorial decisions with no artefact to produce.** These are
  conversations. The skill refuses rather than manufacturing a sitting.
- **Group 5 — not a sitting at all.** Five entries in the brief's implied
  scope are neither a picture-judgement nor a plain yes/no editorial call:
  a deferred-pending-decision placeholder, a design-review-gated build, a
  writing task, a code-classification judgement, and a crash-reproduction
  hunt. `/warlock-sitting` should refuse these too, but for a different
  reason than Group 4, and say which.

Some entries straddle two groups (P15, P26, P34, P35, P38); they are placed
at the group their *first* blocking need falls into and cross-referenced from
the other.

---

## Group 1 — needs nothing but CPU and time

### P28 — Judge the character render benchmark (four verdicts, not one)

**Needs:** nothing but CPU and Blender-on-CPU time — the entry says so in as
many words: *"No card is needed for any of this. The character route is mesh
generation in-process, Blender on the CPU for the rig and the render, and
numpy for the reduction."*

**Artefacts a sitting requires, per archetype** (`humanoid`, `quadruped`,
`winged`, `amorphous`):
1. one representative species at the ladder's middle (`ogre`, `wolf`,
   `dragon`, `slime` suggested), 64 px, 32 colours, 8 directions, all five
   movements, one seed locked across the whole sitting;
2. every body slider at both bounds, three seeds each — six channels for
   humanoid/quadruped/winged, five for amorphous
   (`family.get_archetype(key).channels`);
3. every direction and every animation, at zoom 1 and zoom 4.

**Pre-registration:** none named for the benchmark itself — the entry's own
*Do* and *What to judge* sections are the specification, in the style of a
pre-registration, and nothing external gates it. It does name a companion
document to update in the same sitting:
`docs/measurements/2026-09-02-troupe-qa-thresholds.md` (confirmed present),
which calibrates `qa.THRESHOLDS` against these sheets and says outright it
was written against synthetic sheets, waiting for rendered motion.

**Questions, verbatim from the entry** (the same seven each time):
- Recognisable from the front, and from the back.
- One identity across the eight directions.
- A readable attack.
- Walk contact.
- Clean loops.
- Effects on their sockets.
- Feet on one line.

**Output document:** cited by directory, not by a placeholder filename — the
entry says so explicitly: *"Named here by its directory rather than by a
`YYYY-MM-DD-` placeholder path: `tests/test_external_doc_links.py` reads a
cited path as a claim that the file exists."* It lifts
`docs/manual/11-a-character-sprite-sheet.md`'s status line — confirmed at
line 155, reading *"Built, awaiting the render benchmark"* — to *"Proven —
per archetype, not in one line"*, or names the repair per archetype.

### P30 — Decide which of Inker's four right-hand panes gives up height

**Needs:** nothing but CPU — this is `scripts/exercise_mode.py inker`'s own
instrument, already built. **Deviation from the brief:** the brief filed this
alongside P28/P34/P33/P8 as "needs nothing but CPU and time"; that is right,
but this entry is really a Group 4 shape (an editorial choice among four
named options) wearing a Group 1 coat, because the choice is measurable —
"whichever is taken, `scripts/exercise_mode.py inker` reports the clipped
count, so the result is measurable rather than a matter of opinion about a
screenshot." So the sitting here is: run the instrument once per candidate
layout change, hand over the clipped counts and screenshots, and let the
human pick one of the four named options (give Preview less height, compact
the doors to two columns, collapse Export behind a header, or decide it is
fine). There is no separate question list in the entry beyond the four
options themselves — quote them verbatim when handing over.

### P34 — Judge Clay's twelve shapes and eight figures (items 2, 3, 7 still open)

**Needs:** nothing but CPU — Clay's geometry is procedural, no GPU or weights
involved, and items 1, 4, 5 and 6 of this entry are already struck (built
2026-09-06). Verified against the current text: what remains open is items
2, 3 and 7. **Two of the struck items carry their own residual rider, not
closed by the strike-through** — a sitting on items 2/3/7 does not touch
either, but a reader trusting only "items 2, 3, 7 still open" would not know
they exist. Item 5 ("Pick the grounding convention"), verbatim after its own
"Answered and built 2026-09-06": *"Still open, and smaller than it was:
whether a 'Place on ground' action over real mesh bounds is wanted for
ordinary objects, which is a different feature from a preset knowing where it
lands."* Item 6 ("Decide whether organic presets insert smooth-shaded"),
verbatim after its own "Answered and built 2026-09-06": *"Still open: whether
Flat/Smooth should be offered as a control at insertion, rather than applied
by the rule and overridden afterwards."* Neither blocks a sitting on 2, 3 or
7 — the point is that a reader trusting the "items 2, 3, 7 still open"
summary would not know these two riders exist; read both struck items' full
text in `TODO.md` before treating either as wholly closed.

- **Item 2 (Group 1 shape — renders, then judge):** fish and bird silhouettes
  read weak; wants attachment overlap, tapered wedges, a deliberate wing
  outline and thickness direction. Produce renders of both figures and hand
  them over.
- **Item 3 (Group 1 shape — renders, then judge):** the shape chooser
  undersells the objects — sphere/torus share an icon, several borrow
  unrelated symbols, the eight figures have no preview. Produce the current
  chooser's icons and a set of rendered thumbnails as the comparison.
- **Item 7 (Group 4 shape — no artefact, a pure decision):** whether a figure
  keeps its identity after placement, which costs a manual change either way.
  This one belongs in Group 4, not here; the skill should say so rather than
  manufacturing a render for a question that has no picture answer.

**Question list:** the entry states items 2 and 3 as findings to confirm or
correct against renders, not as a fixed seven-question rubric like P28 — hand
over the two claims verbatim and ask whether the renders still show them.

---

## Group 1b — needs CPU and time, but only after a human authors the source material

### P8 — Author the 22 keyframes

**Needs:** a human posing the skeleton in Poser — *"Poser → Clips in the left
sidebar. Pick a key, pose the skeleton with the normal gizmos, Update key
from pose."* This is authoring, not viewing, and nothing under `scripts/` or
`studio/` can substitute a judgement about what an attack's hit frame should
look like. The brief's Group 1 placed this here outright ("the skill can
prepare these completely"); it cannot — it can prepare everything **around**
the authoring, not the authoring itself.

**What the skill can still do unattended:** render the sheet through Troupe's
existing preview and QA heatmap (`studio/troupe/qa.py`) for whichever clips
already exist — the 22 shipped, provisional ones, across all four body-plan
libraries (`humanoid`, `quadruped`, `bird`, `blob`) — so the human's twenty
minutes is spent posing against a heatmap that is already flagging silhouette
pops, foot-line jitter, loop seams and cross-direction drift, rather than
spent first discovering where to look.

**Per-clip brief, verbatim from the entry:**
- **Idle** (cyclic): a breath — one or two pixels of vertical bob, shoulders
  and chest, nothing else. The seam must be invisible (`seam` flag).
- **Walk** (cyclic): two contacts, two passing poses, the bob passing through
  zero between them; arms counter-swing the legs. Feet on the ground line at
  the contacts (`foot`), silhouette changing smoothly (`shape`).
- **Run** (cyclic): the same four poses with a flight phase, a forward lean
  and a larger arm swing. Read it in profile first: the knee drive is where
  the current clip crumples.
- **Attack** (one-shot): anticipation (1–2 frames), the hit (the frame with
  the most silhouette change, and the one to draw first), recovery back to
  idle's first pose so the return does not pop.
- **Jump** (one-shot): crouch, launch, apex (held), fall, land in a crouch,
  recover. The apex is the readable frame; the landing is the second.

Two things to flag before the human starts, both from the entry: **easing
does nothing at the current segment lengths** (needs a step of ≥3 frames;
every shipped step is 1 or 2), and **the arms hang slightly forward** on the
shipped keys. Once clips are authored, `qa.THRESHOLDS` is calibrated against
the rendered sheet and recorded in
`docs/measurements/2026-09-02-troupe-qa-thresholds.md` — the same document
P28 updates, so if both entries run in one programme, calibrate once.

### P33 — Judge the 2D walk cycle: an ogre and a humanoid

**Needs:** a human drawing (or opening) one side-view ogre and one side-view
humanoid in Inker — step 1 of the entry's own *Do* list, and explicitly the
question under test is whether *hand-drawn* art cut into rigid pieces reads
as a body walking, not whether a synthetic test fixture does. Using the
existing `tests/inker/walk/_figure.py` test fixture in place of real art
would answer a different, already-answered question (the geometry pins in
`tests/inker/walk/` already prove the *mechanism* is correct) and must not be
offered as a substitute.

**What the skill can still do unattended, once art exists:** step 2's timing
(recording setup time honestly, including artwork repair), step 4's bake and
sheet export, and re-running step 3's playback at native size and 4x are all
mechanical once the art is in the tool. The far-limb shading slider (step 5,
1.0 vs 0.6) is a toggle the skill can render both sides of for the human to
compare directly.

**Seven things to judge separately, verbatim:** readable steps; the knee
bending forward and not backward; gaps at the joints; limb overlap where two
pieces cross; silhouette stability from frame to frame; the loop seam between
frame eight and frame one; and whether the arms read as opposing the legs
rather than merely moving. Plus the explicit fallback instruction: *"If it
reads as a rotating paper puppet, write that down before anything is
expanded."*

### P4 — A textured, rigged humanoid `.glb`

**Needs:** a human to author or commission a character mesh, which is the one
thing on this list that no amount of machine time substitutes for. Placed here
rather than in Group 4 because it is authoring with a mechanical tail, not a
decision: once the file exists, everything after it is a button.

**The constraints the file must satisfy, verbatim from the entry**, because a
commission brief written without them comes back unusable: GLB/glTF only
(`blender_worker._import_glb` is the only importer on this path); T-pose or
A-pose; +Z up, −Y forward; bone names mapping onto the 19-bone template, with
Mixamo or Rigify naming needing a mapping table; **no very short bones** —
Blender silently deletes a bone below a fraction of the mesh's largest
dimension *and takes its children*, so fingers and toes are the usual
casualties; under ~300k faces; male and female variants; and a licence
permitting commercial redistribution of rendered sprites.

**What the skill can do unattended, and it is more than it looks:** the whole
path is already runnable against `tests/fixtures/humanoid/cesium_man.glb`
(CesiumMan, CC-BY 4.0, textured, rigged, +Z up, A-pose-ish, 4,672 polys —
attribution in `tests/fixtures/humanoid/ATTRIBUTION.md`). Putting that through
**Send to Troupe** with `palette=cosmos` is a full dry run of the pipeline the
real art will take, and doing it once already found and fixed three silent rig
defects (`_strip_incoming_rig`, `tests/test_rig_supplied_mesh.py`, and Q5 of
`docs/measurements/2026-08-30-art-verdicts-preregistration.md`). Offer that
rehearsal; do not offer its output as an answer.

**Read the entry's own narrowing before preparing anything.** As of 2026-09-05
this is no longer what stands between Troupe and a verdict, and no longer
chapter 11's tutorial sample — Create's Character type builds a textured,
rigged body from an authored family, so the ramp verdict is **P28's**, not
this one's. The entry says in as many words that **it no longer blocks
anything**. What is still owed is what it always was underneath: a
human-authored character anyone would ship, as the thing a generated species
is judged against and as the mesh a user brings of their own. A sitting that
treats P4 as urgent has misread it; say so rather than manufacturing one.

**A verdict taken on CesiumMan is a claim about CesiumMan.** The entry states
this directly — 3,273 vertices, a small JPEG and no female variant is not a
claim about character art anyone would ship — so the rehearsal above never
closes the entry.

---

---

## Group 2 — needs a card and weights

### P32 — Re-examine the `trellis_tex_res = 512` pin

**Needs:** a card and `trellis-cli.exe` (confirmed present at
`vendor/trellis/trellis-cli.exe`). **Do:** reproduce the auto-tex-res noise
with `trellis-cli.exe --tex-res 1024` on one byte-stable reference from
props-v1 — the rock, jug or loaf, explicitly **not** the pouch or branches.
One reference, one judgement, "well under an hour of card time." No named
pre-registration; the decision rule is inline: clean texture on v0.6.0 lifts
the pin, otherwise the reproduction is recorded and the pin stays.

### P16 — Judge an eight-direction action sprite sheet at 32px

**Needs:** a card and SDXL weights. **Pre-registration exists**: Q4 of
`docs/measurements/2026-08-30-art-verdicts-preregistration.md` (confirmed —
read in full) is this entry's pre-registration, with its own decision rule
already written. Do not re-derive one; cite Q4.

**What will be run:** one character, one reference, then `attack8` at 32px
and `walk8` at 32px; the walk played at 10fps in Inker through all eight
directions.

**Three questions, judged separately, verbatim from Q4:** does one identity
survive all eight bands; does the action read as the action; does the front
row, a literal copy of the back row with a different prompt clause, look
like a different picture. Known and pre-declared, so not findings if
rediscovered: front/back rows are copies, `run8`'s knee-drive reads a little
like a crumple in profile, `cast8`'s release is weaker head-on than in
profile.

### P19 — Measure the generated Flourish texture against the procedural one

**Needs:** "a GPU afternoon." Generate five textures (flame, ember, rune,
skull, shard) with the shipped prompt template; put each on the fireball's
*Sparks* and on a *sprite* layer at 128 px, painterly and pixel. No named
pre-registration document; this entry's own text is the spec. Judge beside
the procedural version; the write-up also records whether the black key or
the matting model made the better cutout.

### P20 — Pick the Flourish prompt's text model, measure it, pin it

**Needs:** weights, but the entry calls it "a CPU afternoon" — small
instruct models (Qwen2.5-0.5B/1.5B-Instruct, SmolLM2-1.7B-Instruct) rather
than a GPU-bound diffusion run. Filed here rather than in Group 3 because it
needs no ears, no foreign app and no clean machine, only time and a CPU.
Twenty fixed sentences, half inside the keyword vocabulary and half outside,
one candidate model at a time in `text-instruct/`. For each: the `[model]`
toast versus the `[keywords]` toast, whether the effect did what the
sentence said, CPU time per prompt.

### P21 — Judge restyled keyframes against the procedural frames

**Needs:** "a GPU afternoon." The fireball's *explosion* and the portal's
*loop*, three and five keyframes, strengths 0.4 and 0.7, "oil painting" and
"ink woodcut." Play beside the procedural layer. Questions, verbatim: does
the in-between read as motion or a fade; does the model keep the silhouette
at 0.4; is five keyframes enough for a twelve-frame phase.

### P38 — cross-reference

Needs a card (see Group 3 below, alongside P23/P24/P35 — Muse also needs
ears, so it is filed there).

---

## Group 3 — needs hardware or senses this machine has not got

### P14 — Listen to Sirens, on a machine with a sound card

**Needs:** ears and a sound card; "every box it was built on is headless."
The skill can prepare nothing here beyond pointing at
`docs/manual/14-making-a-soundtrack.md`, which the entry says to follow "as
written, out loud." Verbatim checklist from the entry: write a bar on the
triangle and press Space (in tune? tempo right?); drag envelope handles and
hear the shape change, then `~~~`/`===` under two notes; type into the other
four columns (volume digit, tempo change, arpeggio/vibrato); drop a `.wav`
and audition it at three pitches; audition an SFX then the song; export and
open `song.wav`, a stem and an `sfx/` file in something that is **not** this
app, confirming the `smpl` loop points and stem sample-alignment; save,
close, reopen.

### P23 — Hear Muse, and give its two VRAM figures real numbers

**Needs:** a card big enough for an 8.3 GB model, and ears. Full checklist
and the two constants to measure (`models.MusicModel.vram_gib` = 10.0,
`host_peak_gib` = 12.0) are in the entry verbatim; the GPU test file is
`tests/test_music_gpu.py -m gpu -n 0`. Output:
`docs/measurements/<date>-ace-step-vram.md`.

### P24 — Judge the loop finder, and hear a stem split

**Needs:** ears and a card. Two independent sittings in one entry: the loop
finder's weights (`W_CONTEXT`, `W_LEVEL`, `W_LENGTH`, `MIN_SPAN` in
`studio/muse/loops.py`) judged by listening to the top candidate loop four or
five times against numbered alternatives; and stem separation, judged by ear
for bleed across percussive/vocal/ambient takes. Three constants to measure:
`SeparationModel.vram_gib`, `host_peak_gib`, `vram.MUSIC_SOURCE_GIB`. Two
possible output documents named in the entry:
`docs/measurements/<date>-loop-weights.md` (optional — "an honest unmeasured
constant beats a measured-sounding one") and
`docs/measurements/<date>-hdemucs-separation.md`.

### P35 — Settle Muse's Steps guidance, and measure it

**Needs:** weights and ears (a listening sitting, like P23/P24, not a picture
sitting). Generate the same prompt/seed at 20, 30, 40, 60, 80, 120 steps and
listen; decide where quality stops improving and where it falls apart; write
one dated document and make the manual (two chapters:
`docs/manual/16-generating-a-soundtrack.md` and `docs/manual/35-muse.md`) and
the tooltip in `studio/panes/muse_recipe.py` quote it.

### P38 — Run a ten-minute Muse take, confirm the ceiling is honest

**Needs:** a card and a stopwatch, and ears to confirm it is music at all.
One 600 s take from an ordinary style-tag brief, watching VRAM through the
whole run (peak, not just the end). Confirms neither OOM nor exceeding
`vram_gib`/`host_peak_gib`. Shares its output document with P23 if that entry
is still open when this runs. This is the item that decides whether
`vram.estimate` needs a duration term at all.

### P1 — Generate an asset from a clean-machine install

**Needs:** a second physical machine — the install half closed 2026-09-05.
What remains (per the entry's numbered *Do* list, items 4–7) is the fetch
pipeline under the bundled interpreter, the three recovery paths (cancel
mid-install, Repair, Restore packs), an upgrade-over-scratch run, and "the
laptop." The skill cannot touch another machine; it can only make sure the
build being carried over is current and point at the exact numbered steps.

### P6 — Open a Warlock-written `.aseprite` in real Aseprite

**Needs:** Aseprite itself, which this repository does not have. Fixture
list is `tests/inker/fixtures/aseprite/FIXTURES.md`; start with the tilemap
fixtures (`tilemap-rgb`, `tilemap-indexed`, `spare-tileset`) because their
chunk field order was written by inverting the reader and has never been
checked against a file Aseprite itself wrote. The skill can generate the
fixture files; opening them is the part it cannot do.

### P7 — Author `.tmx`/`.tsx` fixtures in real Tiled

**Needs:** Tiled itself. Fixture spec is
`tests/plotter/fixtures/tiled/FIXTURES.md`; `basic-ortho` first (8×8
orthogonal at 16px, one external tileset, two layers at 0.5 opacity on the
second, one tile flipped each way), saved as both `.tmx` and an exported
`.tmj`. Also: re-check a grid pack's `.tsx` geometry, since pow2 rounding is
off by default now and nothing about `tsxout`'s margin/spacing/columns
arithmetic has been exercised in real Tiled.

### P15 — Judge a generated terrain set, and open it in real Tiled

**Straddles Group 2 and Group 3.** The first half needs a card and SDXL
weights (Create → Sheet → Terrain set, two surfaces that ought to meet, at
32px, then Plotter's Terrain tool); the second half needs real Tiled, which
this repository does not have — for the same reason as P7 ("our writer and
reader agree on the 47-case ordering by construction, so only Tiled can catch
an error both make"). **Pre-registration exists:** Q3 of
`docs/measurements/2026-08-30-art-verdicts-preregistration.md` (confirmed),
with three named failure modes to watch for (a coverage field that is right
and reads wrong; a boundary too soft at 16px; two materials whose scales
disagree) and its own decision rule.

### P29 — Prove the update path against a real release

**Needs:** a GitHub Release published under the user's own account, which
this skill cannot do on the user's behalf without their explicit action —
publishing a release is exactly the kind of external, identity-bearing act
outside this skill's remit. The skill can build the installer and run
`scripts/make_update_manifest.py`, and stop there.

---

## Group 4 — editorial decisions with no artefact to produce

Refuse and say so; do not manufacture a sitting for these.

### P10 — Decide: what the model picker offers
Three named sub-decisions (hide or measure `juggernaut`/`dreamshaper`;
`sdxl_cfg_pag`'s lost bench; whether `turbo` is also labelled *draft*). No
picture answers any of them.

### P11 — Decide: `plotter-wave-2`
Whether to derive a branch from `refs/tags/archive/plotter-wave-2` and
rebase/cherry-pick, or leave it archived. A git-history and code-review
decision, not a visual one.

### P17 — Decide whether four-direction guides are wanted
Mostly Group 4 — "Do: decide." **But note the branch:** if the decision is
yes, the entry then asks for six guides to be *authored* in
`src/warlock/templates/sprite_guides/`, which is Group 1b's shape (human
art, not a sitting). So this entry is a decision first and an authoring task
second, never a picture-judgement in either branch.

### P25 — Decide: is a non-commercial stem model worth shipping at all
A licensing judgement about the app's purpose. If the decision is "remove,"
the entry lists the exact surface to delete; that is ordinary code work, not
a sitting.

### P26 (item 1 only) — Decide where the three built wheels live
Only item 1 of this six-item entry is a Group 4 decision (host `docopt`,
`mojimoji`, `unidic-lite` as release assets, or drop `cutlet`/`fugashi`).
Item 5 is a different shape — see Group 5.

### P34 (item 7 only) — cross-reference
See Group 1 above: whether a Clay figure keeps its identity after placement.
No render answers it; it is a design decision costing a manual change either
way.

### P36 — Decide whether Clay gets an "adjust last operation" card
One sentence either way; the old bookkeeping's shape is recoverable from git
history if the answer is yes.

---

## Group 5 — not a sitting this skill can run at all

These five are in the neighbourhood of the brief's scope but do not fit
either a picture-judgement sitting or a plain yes/no editorial call. Refuse
them with an explanation rather than forcing a shape onto them.

### P13 — Troupe phases 7 and 8
The second kind of `TODO.md` entry — fully specified, deliberately unstarted,
gated on P11's decision and a working whole-character generation baseline.
Nothing to look at yet; nothing to decide today either, since the decision
("does the programme continue") lives in P11, not here.

### P31 — A sweep that fails should stop repeating the failure
Also the second kind of entry: buildable exactly as written, held back by
the user's own 2026-09-06 call to have the design reviewed before it is
built. That is a code review, not a sitting.

### P22 — Write what the closed beta is told it has not seen
A writing task — one sentence per named gap (Sirens/P14, Muse/P23,
Plotter-Tiled/P7, character sheets/P28, Aseprite/P6, the packs-vs-models
gap), destined for the invite text. No pictures, no listening, nothing to
judge; it is prose to draft.

### P37 — Audit the three other stages that rename onto a served name
Reading `_deform_qa`, the model-promotion stage in `_q_generate.py`, and
`_remesh` in `_q_mesh.py`, then deciding by comment beside each call whether
its rename is a completion marker or a checkpoint. A code-classification
judgement made by reading source, not by looking at a rendered artefact.

### P39 — Reproduce the one faulthandler dump, or let it expire
Run a trellis reconstruction with matting three or four times, closing the
window mid-job on at least one run, watching `crash.log` for a second
`0x80010012`. This is a crash-reproduction hunt with a pass/fail on a log
file, not a judgement over prepared pictures — there is nothing for the
skill to stage in advance beyond the repro sequence itself, and no question
list to hand over.

### P26 (item 5 only) — Collect the `text2image` and `music` packs once
Long-running data collection on a real line (multi-gigabyte downloads,
`music`'s sdist build path), producing figures for a table, not pictures for
a verdict. Closer to P29's shape than to a sitting.

---

## Not P-numbered, and not sittings

`TODO.md`'s "Also owed, smaller" section carries two items with no `P<N>`:
tutorial sample assets (art authoring — a `.ora` sprite, a 16px tileset, a
`crate.glb`) and deleting the archived pre-purge mirror at
`D:/Projects/_archive/warlock-pre-purge.git`. Neither is a judgement sitting;
list them if asked what else is owed, but `/warlock-sitting` has nothing to
prepare for either.
