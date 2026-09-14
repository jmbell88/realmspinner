# TODO.md — everything still owed, and who has to do it

Written 2026-08-21, consolidating the four plan files that had accumulated at
the root and in `docs/`; rewritten 2026-08-25 as a priority list after a
whole-tree review closed everything on it that code could close; purged
2026-09-04 of every entry that had since closed. Git holds every earlier
version and every deleted plan (`git log --all --diff-filter=D`).

**This file has three kinds of entry and no others.**

1. **Work only a human can do**: art direction, authoring keyframes, opening a
   file in real Aseprite or real Tiled, running a card, listening, making a
   decision. None of it is derivable from the tree and none of it can be
   closed by writing code.
2. **Work that is fully specified and deliberately unstarted** — today that is
   Troupe's phases 7 and 8 (P13), here with the argument that makes it
   actionable, not as a title. An entry earns this kind only by an explicit
   decision *not* to build it yet; it is not a parking space for work nobody
   got to.
3. **Open findings** (the section at the end): code work a review or a real
   run turned up and did not fix, numbered `F<N>` so it cannot be confused
   with the `P<N>` entries above. Each is buildable and is struck out the day
   it is built; the section is deleted when it is empty.

**The moment an item could be built, it is built and struck out rather than
tracked.** A plan whose boxes disagree with the tree is worse than no plan, and
that is why every other roadmap file in this repository's history was deleted
rather than ticked. Entry numbers are stable: a closed entry's number is not
reused, and what it decided is one line under *Closed records* at the bottom.

**This file has no `§N` API.** `tests/test_ux_todo_fixes.py` refuses any
citation of this filename from `src/` or `scripts/`. What a module needs to
explain, it explains where it is — or in `docs/INVARIANTS.md`, or in a
`docs/measurements/` document, both of which outlive any plan.

**How to read an entry.** Highest priority first. *Why it is yours* names the
human dependency. *Do* is the concrete steps. *Expected outcome* is what
changes when it is done — the verdict it produces or the thing it unblocks —
so that "done" is recognisable without re-deriving it.

---

## P4. A textured, rigged humanoid `.glb` — one file, three jobs

**Why it is yours:** art. Every Troupe frame to date quantises into the pale
end of whatever ramp it is given because no textured base mesh exists. The
palette ramps are installed (`~/.warlock/palettes/cosmos`, `light_world`) and
proven on 2D; **this file is the only thing between them and a verdict on
Troupe.** The same file is the base mesh the Troupe manual chapter assumes, and
the tutorial sample for chapter 11.

**Constraints a base mesh must satisfy:** GLB/glTF (`blender_worker._import_glb`
is the only importer on this path); T-pose or A-pose; +Z up, −Y forward; if it
ships rigged, its bone names do not need to map onto the 19-bone template —
`_strip_incoming_rig` discards any skin and skeleton a supplied mesh brings in
regardless of naming, and the auto-rig replaces it. Mapping tables now exist
(`src/warlock/clipmaps.py`, `templates/clip_maps/{mixamo,rigify}.json`), built
for "Import clip" — converting an external *animation* onto an already-rigged
Warlock skeleton — not for accepting a supplied mesh's own rig, which this
entry's constraint is about; no very short bones (Blender silently deletes a
bone below a fraction of the mesh's largest dimension *and takes its children*
— fingers and toes are the usual casualties); under ~300k faces; male and
female variants; a licence permitting commercial redistribution of rendered
sprites.

**Do:** author or commission it; put it through **Send to Troupe** (library
menu, inspector, or the picker inside Troupe) with `palette=cosmos`.

**Partially unblocked 2026-08-30.** `tests/fixtures/humanoid/cesium_man.glb`
(CesiumMan, CC-BY 4.0 — `tests/fixtures/humanoid/ATTRIBUTION.md`) is textured,
rigged, +Z up, A-pose-ish and 4,672 polys, so the chain was runnable from that
day. It does not close this entry: a ramp verdict taken on a 3,273-vertex sample
with a small JPEG and no female variant is a claim about CesiumMan, not about
character art anyone would ship. Putting it through the path found and fixed
three silent rig defects (`_strip_incoming_rig`, `tests/test_rig_supplied_mesh.py`,
`docs/measurements/2026-08-30-art-verdicts-preregistration.md` Q5).

**Narrowed 2026-09-05.** This entry is no longer what stands between Troupe and
a verdict, and it is no longer chapter 11's tutorial sample. Create's Character
type builds a textured, rigged body from an authored family — thirty-one species
over four body plans — so there is a base mesh with real colour on it in the
build, and the ramp verdict is P28's. What is still owed here is what it always
was underneath: a *human-authored* character anyone would ship, as the thing a
generated species is judged against and as the mesh a user brings of their own.
It no longer blocks anything.

**Expected outcome:** the first Troupe sheet from art rather than from a
generator, and a verdict on whether the ramp works at sprite scale on it.
Unblocks P11.

**In progress, 2026-09-12 — leaving off here for a future session.** Found a
usable candidate mesh: `docs/examples/Universal Base Characters/` (Quaternius,
CC0 1.0 — `License_Standard.txt`), male and female variants, textured,
pre-rigged, both cleared through the real pipeline. The male
(`Base Characters/Godot - UE/Superhero_Male_FullBody.gltf`, converted to
`.glb` via Blender's own exporter with `--webp off`/PNG textures since
`trellis-cli`'s WebP default and Mason's viewer don't implement
`EXT_texture_webp`) imports clean: 14,318 faces, T-pose, correct +Z-up axis
convention after `_import_glb`, 65 incoming bones stripped with no crash by
`_strip_incoming_rig` (bone-name mapping turned out not to matter — Warlock
discards any incoming rig rather than adopting it, so the finger-bone-length
risk this entry's constraints list warns about never bites). Ran the whole
chain for real: `service.jobs.import_mesh` → `service.troupe.send_to_troupe`
→ rig → charsheet, all `done`, no errors. One deviation: `palette=cosmos`
this entry names is not installed on this machine (only `dawnlight.hex` is,
in `~/.warlock/palettes/`) — this entry's own background text claiming
`cosmos`/`light_world` are installed is stale; rendered with `dawnlight`
instead.

**What stopped this from closing today:** looking at the actual sheet
surfaced two rig-template defects that were never visible on CesiumMan or a
generated species — **F7** (jump's knee bends backward, root-caused to a
sign error in `humanoid.json`'s pose data) and **F8** (walk may play backward
left/right, unconfirmed, needs to be seen in motion rather than as static
frames). Closing this entry on a ramp verdict taken over a visibly broken rig
would be worse than not closing it, so F7 at minimum should be fixed and
re-rendered before asking for the ramp verdict itself. **Next session:** fix
F7, chase F8 (play the yaw-90 walk frames back, or read `rigging.py`'s
yaw/mirroring path against the leading-leg convention), re-render this same
mesh's sheet, *then* hand it over for the actual ramp-at-sprite-scale
judgement this entry is asking for. The mesh, the license, and the working
import/rig/render chain are the hard part and are already proven — what's
left is fixing what looking at it found.

## P6. Open a Warlock-written `.aseprite` in real Aseprite

**Why it is yours:** an app this repository does not have. A green test proves
this reader and this writer agree with *each other*; a round trip through our
own two halves cannot catch an error both halves make together
(`docs/COMPAT.md`, top).

**Do:** `tests/inker/fixtures/aseprite/FIXTURES.md` names the four fixtures
worth authoring first. **Start with the tilemap ones** — `tilemap-rgb`,
`tilemap-indexed`, `spare-tileset`: their chunk field order was written by
inverting the *reader*, field for field, and has never been checked against a
file Aseprite itself wrote. In the same sitting: every RGB and grayscale file
carries a palette chunk derived from the art's own colours, including a
1-entry transparent palette on a blank document — check Aseprite is happy with
that rather than replacing its default with a single swatch.

**Expected outcome:** either the tilemap chunk order is confirmed and
`docs/COMPAT.md`'s Aseprite rows can say "opened in Aseprite 1.3.x", or a
field-order bug is found that no test could have — and it gets a fixture from
the real app.

## P7. Author `.tmx`/`.tsx` fixtures in real Tiled

**Why it is yours:** the same rule as P6. Every map under
`tests/plotter/fixtures/tiled/` was produced by this editor, so every
`round-trips` row in `docs/COMPAT.md`'s Tiled part is a round trip against
ourselves. `TILED_VERSION` already moved to `1.12.2` on 2026-08-29 against
files outside this repository; that moved one attribute and left no golden the
suite can re-run.

**Do:**
1. Author the fixtures in Tiled 1.12.x per `tests/plotter/fixtures/tiled/FIXTURES.md`.
   `basic-ortho` first: 8×8 orthogonal at 16 px, one external tileset, two
   layers with the second at 0.5 opacity, one tile flipped each way. Save it
   twice — `.tmx` and an exported `.tmj` — because the manifest keys on stems
   having both.
2. Re-check a grid pack's `.tsx` geometry in Tiled: pow2 rounding is off by
   default now, and the 2026-08-29 maps came from image tilesets, so nothing
   about `tsxout`'s margin/spacing/columns arithmetic has been exercised there.

**Expected outcome:** the Tiled rows of `docs/COMPAT.md` become claims about
Tiled rather than about ourselves.

## P8. Author the 40 keyframes

**Why it is yours:** animation is art. The shipped 22 (now 40 — see below) are
provisional. Moving authoring from frames to keyframes made a bad clip cheap to
fix, not good; a bad clip reproduces exactly the "stiff posing" flaw being
escaped.

**Wider since 2026-09-05, and easier at the same time.** There are now four
authored clip libraries, not one — `humanoid`, `quadruped`, `bird` and `blob`,
each carrying all five original movements because `charsheet.resolve_layout(None)`
asks for five and `expand_clips` raises on a missing one. The 22 below are the
humanoid's and are the ones to start with, but a four-beat lateral-sequence
walk and a wing beat are their own problems and neither is a humanoid walk with
different bone names. Easier because the thing that was missing is here: every
species is a body you can build in one press with no card, so a clip edit can
be judged on a *rendered* sheet within minutes instead of waiting on P4. That
is also what unblocks the calibration below — `qa.THRESHOLDS` was chosen
against synthetic sheets and explicitly not against rendered motion, and
rendered motion now exists for four body plans. Calibrate it as part of P28
rather than separately; one sitting judging four sheets is where the numbers
come from.

**2026-09-12: the vocabulary opened, and five more provisional clips landed in
every one of the four libraries** — `attack_02`, `cast`, `fall`, `hit` and
`death`, each written with `"provisional": true`
(`docs/measurements/2026-09-12-troupe-open-clip-vocabulary.md`). That is 18
more keyframes per skeleton (5+4+2+3+4, humanoid's count — the other three
species match it clip for clip), on top of the original 22, hence the renamed
heading. They are placeholders in exactly the sense the original 22 are: cheap
to fix, not yet judged as movement. Brief per clip, to start from:

- **Attack 02** (one-shot): a second attack, mirrored or backhanded rather than
  a repeat of Attack — anticipation, the hit, recovery back to idle's first
  pose so the return does not pop.
- **Cast** (one-shot): gather, hands drawn in, then release forward.
- **Fall** (cyclic): the airborne loop Jump's crouch-launch hands off to at its
  apex — no ground contact anywhere in it.
- **Hit** (one-shot): a flinch — the hurt reaction `spritesynth` calls the same
  timing by a different name — that returns to idle's first pose without a pop,
  the same return Attack's recovery makes.
- **Death** (one-shot): a collapse that ends grounded (negative root z) and
  holds its last frame rather than returning anywhere.

None of the five is part of this task's judging brief below (that stays the
original five and the rendered-motion calibration); P28 can judge all ten in
one sitting if there is time.

**Do:** Poser → **Clips** in the left sidebar. Pick a key, pose the skeleton
with the normal gizmos, **Update key from pose**. Onion skin ghosts the keys
either side; **Play** scrubs the real interpolation. **Save clips** writes to
your data folder and never touches what the build ships, so **Revert to
shipped clips** is always available. Manual: *Poser → Editing clips*. Decide
first whether it is you or an animator.

Two things to know before starting. **Easing does nothing at the current
segment lengths**: it needs a step of ≥3 frames and every shipped step is 1 or
2, so `idle`'s `ease` renders identically to `linear` today. **None of the five
new clips reaches 3 either** (the longest segment in any of them is 2, the same
ceiling the original five sit under): `attack_02` and `death` (`ease_in`) and
`hit` (`ease_out`) already show a barely-there kink at their one 2-frame
segment (0.25/0.75 against linear's 0.5), the same as Attack's `ease_in` does
today, while `cast`'s `ease` is exactly 0.5 at both its 2-frame segments and so
renders identically to `linear`, the same as Idle's `ease` does; `fall` is
`linear` throughout. And **the arms hang slightly forward** on the shipped
keys.

**The brief, per clip.** Judge each at 16–32 px through the Troupe preview,
whose heatmap (`troupe/qa.py`) flags silhouette pops, foot-line jitter, a loop
seam and cross-direction drift per cell; click a flagged square to land on the
frame.

- **Idle** (cyclic): a breath — one or two pixels of vertical bob, shoulders
  and chest, nothing else. The seam must be invisible (`seam` flag).
- **Walk** (cyclic): two contacts, two passing poses, the bob passing through
  zero between them; arms counter-swing the legs. Feet on the ground line at
  the contacts (`foot`), silhouette changing smoothly (`shape`).
- **Run** (cyclic): the same four poses with a flight phase, a forward lean and
  a larger arm swing. Read it in profile first: the knee drive is where the
  current clip crumples.
- **Attack** (one-shot): anticipation (1–2 frames), the hit (the frame with the
  most silhouette change, and the one to draw first), recovery back to idle's
  first pose so the return does not pop.
- **Jump** (one-shot): crouch, launch, apex (held), fall, land in a crouch,
  recover. The apex is the readable frame; the landing is the second.

Once the clips are authored, calibrate `qa.THRESHOLDS` against the rendered
sheet and record the values in
`docs/measurements/2026-09-02-troupe-qa-thresholds.md`, which says so.

**Expected outcome:** clips that look like movement at 32 px, verified through
P28's rendered sheets. This is the most important art task in the programme.

## P10. Decide: what the model picker offers

**Why it is yours:** editorial calls that the graded run's numbers now inform.

- `juggernaut` and `dreamshaper` have **no hits anywhere in
  `docs/measurements/`** and sit in the picker as peers of the default at
  6.9 GB each. Hide them behind an Advanced toggle, or measure them.
- `sdxl_cfg_pag` is offered as an equal and **lost its own bench**: the control
  won 55 of 80 paired units and PAG cost +34% sampling time
  (`docs/measurements/2026-08-17-reference-source-bench.md`).
- `turbo` is labelled non-commercial (the disclosure half). Whether it is also
  labelled *draft* is an editorial call.

**Expected outcome:** each is a decision recorded in the model registry's own
comments (or a measurement document), after which the picker stops offering
what has not earned its place.

## P11. Decide: `plotter-wave-2`

**Why it is yours:** design, not implementation.

Poser can now load a rigged asset for preview and posing (`poser_mode.open_asset`,
2026-09-07), which was the blocker the "judge clips as pixels" question named —
that half is built, not decided; whether the pixel verdict actually *moves* out
of Troupe once that preview exists is still open and unrelated to this entry.

- **`plotter-wave-2`.** No branch of that name exists — it was converted to
  `refs/tags/archive/plotter-wave-2` at `d1995fad` (2026-08-14), the same
  commit its last move landed on, so recovering it means deriving a branch
  from the tag first. It holds 52 commits unmerged against master, which has
  moved several hundred since. Gated on P7's fixtures and a whole-tag review.
  Two outcomes: derive a branch and rebase or cherry-pick what still applies,
  or leave it archived.

**Expected outcome:** one recorded decision, into either a branch derived from
the tag or the tag staying archived.

## P13. Troupe phases 7 and 8 — fully specified, deliberately unstarted

The second kind of entry. Phases 0a–0d and 1–6 are implemented and verified;
what they established is in `docs/INVARIANTS.md`, and the measured ULPC facts
are passing oracles in `studio/troupe/ulpc.py`. Phase 6 closed on 2026-09-03
with the three-way re-render merge (`inker/sheetmerge.py`,
`service.troupe.rerender_charsheet`, `_doc_sheet.merge_render`; on conflict the
hand edit stands and the cell is flagged).

**Phase 7 — layered equipment (deferred until whole-character generation
works, which P12 measured it does not at the shipped default).** *Multi-GLB
scene composition*: `op_sheet` takes one `source_glb` and equipment items are
separate assets by construction, so the task is composing N GLBs under a
shared camera, not splitting one (`op_rig` joins every mesh into one object,
which is why splitting is a dead end). *Per-part passes with depth* give
correct per-direction occlusion for free; the depth machinery is proven in
`blender_worker._depth_material`. *Garment fitting*: skin-weight transfer by
proximity (Blender Data Transfer) — hugging garments first; capes and long
skirts are a separate problem. The supplied-base-mesh path (P4) and the
authored-family path Create's Character type shipped on 2026-09-05 are both
untouched by P12's verdict, and are the two to build on.

**Phase 8 — reconsider only against a working system.** *AI restyle*:
`create_pixel_sheet` with `structure_lock` over a rendered sheet; note
`structure_lock` is only a Canny-ControlNet toggle and what keeps silhouettes
exact is `pixelsheet.remask()` stamping the render's own alpha back, and
`check_restylable` refuses `frame_size × columns > 1024`. Opt-in, measured,
never default. *A learned pixel refiner*: once cleanup is routine,
`(render, hand-cleaned)` pairs accumulate for free, perfectly registered, over
a fixed palette. *More animations*: hurt (`hit`), death, cast, fall and
attack_02 were built provisional 2026-09-12 as part of the open-clip-vocabulary
work (`docs/measurements/2026-09-12-troupe-open-clip-vocabulary.md`; the art
pass over them is folded into P8); climb remains additive.
*Natural-language character description*: only over a working catalog, only
local weights through `fetch_worker`.

**Expected outcome:** none until P11 says the programme continues; the value
of this entry is that nobody re-plans it.

## P14. Listen to Sirens, on a machine with a sound card

**Why it is yours:** hardware, and the plainest instance of it in this file.
Sirens landed complete on 2026-08-27 and **nobody has ever heard it**. Every
box it was built on is headless, so the synthesis is proved the only way it
could be — a byte-identical render corpus, a `wavout` reader that is its
writer's inverse, a perf budget — and none of that is a person saying "that is
a pulse wave and it is in tune".

**Do:** open Sirens on a machine with audio and go through
`docs/manual/14-making-a-soundtrack.md` as written, out loud:

- Write a bar on the triangle and press Space. In tune against a reference
  pitch? Is the tempo the BPM the transport claims?
- Drag a decay into a volume envelope and hear the shape change. Drag the
  release marker and hold a long note. Then write `~~~` under one note and
  `===` under another: the first should let go into the tail, the second stop
  dead.
- Type into the other four columns. A volume digit should make one row
  quieter; an `F` and two digits should change the tempo *from that row on*;
  an arpeggio or vibrato should sound like the thing it is named after.
- Drop a `.wav` on the window, point a sample instrument at it, play it from
  the grid at three pitches.
- Audition a sound effect, then play the song again and hear the song.
- Export into an empty folder. Open `song.wav`, a stem and an `sfx/` file in
  something that is **not** this app. Confirm the `smpl` loop points loop, and
  that a stem lines up sample-for-sample with the mix.
- Save, close, reopen.

**Expected outcome:** either the mode is what the manual says it is, or the
first defect only a listener could find — a panning error, a tuning error, a
click at a loop point, a mixer that opens at the wrong rate.

## P15. Judge a generated terrain set, and open it in real Tiled

**Why it is yours:** a card, then eyes, then an application this repository
does not have. The seamless path has been through real weights exactly once
(four isolated 1024px materials, `tests/test_tileset_gpu.py`, 2026-08-29) and
**nothing has generated a terrain set end to end.** This is also where the
*Keep one style across the list* checkbox (P18, built 2026-08-30) gets its
verdict.

**Do:** Create → Sheet → **Terrain set**, two surfaces that ought to meet
(grass into dirt), at 32px. Take the sheet into Plotter, paint with the
**Terrain** tool, look at the joins where the brush turns a corner and where
two strokes meet. Export the map and **open it in Tiled** — our writer and
reader agree on the 47-case ordering by construction, so only Tiled can catch
an error both make.

**Expected outcome:** either the first generated tileset a person would
actually paint a map with, or the first defect only a painted map shows.

## P16. Judge an eight-direction action sprite sheet at 32px

**Why it is yours:** art. The seven pose guides are verified as *guides*, but
**nothing has been through SDXL with them.**

**Do:** one character, one reference, then `attack8` at 32px and `walk8` at
32px. Play the walk at 10fps in Inker through all eight directions. Judge
three things separately: does one identity survive all eight bands; does the
action read as the action; does the front row, a literal copy of the back row
with a different prompt clause, look like a different picture.

Known and recorded, so not the finding: front and back rows are copies,
`run8`'s knee-drive reads a little like a crumple in profile, and `cast8`'s
release is weaker head-on than in profile.

**Expected outcome:** the art verdict on whether the whole action set is worth
having, and whether P17 is a question at all.

## P17. Decide whether four-direction guides are wanted

**Why it is yours:** editorial. `SPRITE_DIRECTION_COUNTS` is `(4, 8)` but only
`*8.json` guides ship, so the Directions control has exactly one option. The
case for four is half the generations and half the wait; the case against is
that a four-direction sheet and the legacy `walk` are near-neighbours, and the
guides are art.

**Do:** decide. If yes, author `idle4`, `walk4`, `run4`, `attack4`, `cast4`,
`hurt4` (and `jump4` if the eight-direction jump survives P16) in
`src/warlock/templates/sprite_guides/`, four views each in the legacy row order
front/left/right/back, with P8's brief. The loader, planner, door and form
already take a four-direction kind.

~~**The one step that is code, whichever way it goes:** while discovery finds
a single count, show the eight-direction count as a label rather than a
one-item combo, and let the combo reappear the day a second count ships.~~
**Built 2026-09-04** (noticed in the 2026-09-06 audit, finding docs-17):
`settings_2d._sprite_layout` already branches `form_ui.readonly` under two
discovered direction counts and `form_ui.segmented_choice` otherwise. (The
2026-09-08 audit, finding docs-10: this cited line numbers, which drift; the
branch is named by its function now.)

**Expected outcome:** either six authored guides and a two-option control, or
the count stated as a fact.

## P19. Measure the generated Flourish texture against the procedural one

**Why it is yours:** a GPU afternoon, and a judgement. The texture door is
built and defaults to nothing: every preset is procedural. Whether a generated
flame or ember *beats* the procedural core at 128 px is a measurement, and the
earlier prompt expander was deleted for shipping without one.

**Do:** generate five textures with the shipped prompt template (flame, ember,
rune, skull, shard); put each on the fireball's *Sparks* and on a *sprite*
layer at 128 px, painterly and pixel. Judge beside the procedural version and
write a document under `docs/measurements/` with the verdict, the prompts, and
whether the black key or the matting model made the better cutout. If a
texture wins, the preset gets it as a file beside its JSON; the door stays
opt-in either way.

**Expected outcome:** one measurement document; presets changed only if it
says so.

## P20. Pick the Flourish prompt's text model, measure it, pin it

**Why it is yours:** a CPU afternoon and a judgement, and the one entry that
touches the offline invariant's stated exception. The door is gated on a
directory (`inker_flourish.TEXT_MODEL_DIR`), not a registry row, because every
`models.py` entry carries a revision pin and the pin comes from this
measurement.

**Do:** candidates Qwen2.5-0.5B-Instruct, 1.5B-Instruct, SmolLM2-1.7B-Instruct,
one at a time in `text-instruct/`. Twenty fixed sentences, half inside the
keyword vocabulary and half outside. For each, the `[model]` toast versus the
`[keywords]` toast, whether the effect did what the sentence said, and CPU time
per prompt. Write a document under `docs/measurements/`. If a model beats the
vocabulary on the outside half without losing the inside half, add a pinned
`TextModel` row to `models.py`; if none does, delete `recipe_worker.py` and the
door.

**Expected outcome:** one measurement document and one of the two edits.

## P21. Judge restyled keyframes against the procedural frames

**Why it is yours:** a GPU afternoon. `Flourish → Restyle keyframes…` is
built and opt-in; a crossfade of two diffusion frames is exactly where the plan
expected it to fail.

**Do:** the fireball's *explosion* and the portal's *loop*, three and five
keyframes, strengths 0.4 and 0.7, "oil painting" and "ink woodcut". Play beside
the procedural layer. Write a document under `docs/measurements/`: does the
in-between read as motion or a fade; does the model keep the silhouette at 0.4;
is five keyframes enough for a twelve-frame phase. If it never reads as motion,
delete `keyframes.py`, the door and its popup.

**Expected outcome:** one measurement document and one of the two edits.

## P22. Write what the closed beta is told it has not seen

**Why it is yours:** it is a claim about the product, made in your name, to
people you invited. Only Troupe carries an **Experimental** chip. Four other
surfaces have evidence gaps and no chip: Sirens has never been heard (P14),
Muse has never been heard (P23), Plotter's Tiled interop has only round-tripped
against itself (P7), no character sheet has been judged by an eye at sprite
scale (P28), and Warlock-written `.aseprite` files have never been opened in Aseprite
(P6). One more belongs here that is not a mode: on a base install, Create and
Muse send you to Settings → **Models**, and the weights are only half of what
they need — the matching **pack** is the other half. The door itself already
says so (built 2026-09-05: `model_gate.mode_gate`/`mode_reason` sends a user
with a missing pack to Packs first and names what is blocked,
`tests/test_pack_gate.py`); what still isn't said anywhere is the invite text
an invitee reads before they download at all.

**Do:** name them, by mode, in whatever the invite is — a note beside the
download. One sentence each: what runs, what has never been checked against the
other implementation, what to report if it breaks. **Not by adding chips**: a
chip is a permanent statement about design; these are temporary statements
about evidence, and the beta is what removes them.

**Expected outcome:** an invited user who hits one of these knows they hit a
known gap, and reports the right thing.

## P23. Hear Muse, and give its two VRAM figures real numbers

**Why it is yours:** a card big enough for an 8.3 GB model, and ears.
Everything provable without a card is proved; what none of it answers is
whether the model produces music, whether a cancel interrupts a real sampling
loop, and what the thing costs.

**Do:**

- `uv sync --extra music`, download the weights from Settings → Models, confirm
  `uv run warlock doctor` flips the ACE-Step row.
- `uv run pytest tests/test_music_gpu.py -m gpu -n 0`. The cancel test is the
  only proof `WARLOCK 1/5` reaches the loop. Its first job is confirming on
  hardware that a take the *model* produced (44.1 kHz 16-bit PCM, `WARLOCK 5/5`)
  opens in Sirens.
- The derived tasks: retake at 0.2 and 0.8, extend a 60 s take, repaint one
  phrase, edit the tags alone. Cancel each mid-run — the `edit` especially,
  whose cancel hook is a second loop and has never been exercised.
- In the app: two takes at 60 s from style tags, and listen. Is it music? Do
  the tags do anything? Does a `[verse]`/`[chorus]` lyric block get sung?
- Cancel mid-generation: child dies, row reads **cancelled**, next take runs
  without a restart. Kill the app mid-generation: no `music_worker` survives.
- **Open in Sirens** on a take; play the sample from the grid at three pitches.

**The measurement.** `models.MusicModel.vram_gib` (10.0) and `host_peak_gib`
(12.0) are documented estimates. Take them from a real run, publish
`docs/measurements/<date>-ace-step-vram.md`, replace the constants with cited
numbers. `cpu_offload` and `overlapped_decode` stay class attributes on
`music_worker._Server` until a measurement says they need to be knobs.

**Expected outcome:** either the mode is what chapters 16 and 36 say it is, or
the first defect only a listener could find — and two constants measured
rather than guessed.

## P24. Judge the loop finder, and hear a stem split

**Why it is yours:** ears, again, and a card.

**The loop finder.** `studio/muse/loops.py`'s `W_CONTEXT` (1.5), `W_LEVEL`
(0.6) and `W_LENGTH` (2.0) were **chosen by ear and ship saying so**. Generate
a dozen takes across styles, run **Find loop points**, listen to the best
candidate looping four or five times, then try the numbered alternatives. Does
the top candidate usually win? Does it favour quiet moments (`W_LEVEL` too
strong) or merely spectrally similar ones (`W_CONTEXT` too weak)? Is `MIN_SPAN`
(0.35) too generous for a two-minute take? Does a 0 ms crossfade click and does
500 ms audibly duck? Either write `docs/measurements/<date>-loop-weights.md` or
leave the constants with their honest "unmeasured" comments — **an honest
unmeasured constant beats a measured-sounding one**.

**Stem separation** has never been run. Download it from Settings → Models and
confirm the red non-commercial marker appears at the moment you agree. Split
three or four takes (percussive, vocal, ambient) and listen to each stem for
**bleed**; chapter 36 promises "a little", which a listener has to confirm or
correct. Cancel a split mid-run: row cancelled, no child, the take reads as
unsplit rather than partly split.

**The measurements.** `SeparationModel.vram_gib` (4.0), `host_peak_gib` (4.0)
and `vram.MUSIC_SOURCE_GIB` (1.0) are guesses. Take all three from real runs —
the third from an `extend` of a 240 s take — and publish
`docs/measurements/<date>-hdemucs-separation.md` with wall clock beside them.
`segment_seconds` (10.0) is the knob if separation is slower than about a
minute for a four-minute take.

**Expected outcome:** three measured constants, a verdict on the loop weights,
and either a confirmation of chapter 36's bleed sentence or a better one.

## P25. Decide: is a non-commercial stem model worth shipping at all

**Why it is yours:** a licensing judgement about what this app is *for*.
Hybrid Demucs ships **labelled** `commercial=False` with a `license_note` and
the red marker; the reasoning is in `docs/MODELS.md`. It is the second
non-commercial entry beside SDXL-Turbo, and unlike Turbo it is optional.

**The question:** for an app whose purpose is making assets people sell, is
"labelled and optional" the right answer, or should the feature not be offered?

**If remove:** the surface is the `SeparationModel` table,
`pipelines/separation_worker.py`, `separate_job`, the `separate` arms in
`_q_music`/`_q_jobs`/`vram`/`validation`/`progress`, the four `files.MEDIA`
keys, the tray button, and chapter 36's Stems section. The `url`/`sha256`
transport in `models.Fetch` **stays** either way.

## P26. Decide whether the dependency packs ship, and wire them up if so

**Why it is yours:** a product decision with a cost attached, and it gates
buildable work that is otherwise ready. The pure and performing halves are
built and proven (`0975721f`, `2556cb6d`, `26b40d8a`, and this session's
commit); what is left is a *choice* about the shipped installer, plus one
packaging question with a real weight in megabytes.

**Where it stands.** `warlock/packs.py` is the registry and planner,
`scripts/make_packs.py` the generator, `pipelines/pack_worker.py` the child
that downloads and installs, `service/packs.py` the parent. Measured against
the real lock: base+studio is 30 distributions; `rig` adds 7, `text2image` 34,
`music` 74, with 27 shared between the two torch packs — 85 wheels, 82 fetched
and 3 built. The rig pack has been collected and installed end to end into a
base-only runtime (0.32 GiB download, 0.63 GiB installed, `import bpy` → 5.2.0
LTS). The second offline exception is recorded in `docs/INVARIANTS.md`.

**The decision was taken on 2026-09-04: they ship.** The installer now stages
`--extra studio` alone and the three heavy extras arrive from Settings, which
takes the base download to roughly a third of 2.91 GB and lets a user who only
draws pixel art never download torch. What that decision costs — a second
install path to support — is real, and P1 is where it is met: every figure in
that entry was measured against the all-extras installer and none of them is
this build's any more.

**Do** — steps 2, 3 and 4 were built the day the decision was taken; what is
left needs a person:

1. **Decide where the three built wheels live.** `docopt`, `mojimoji` and
   `unidic-lite` publish no Windows wheel, so the build compiles them and they
   are marked `bundled` in the manifest: there is no URL to fetch them from, so
   the installer must carry them. `unidic-lite` is ~47 MB of that, all of it in
   the base download, for a pack the user may never install. The alternatives
   are hosting the three built wheels as release assets (a publishing step, and
   the first URL in this project that is ours) or dropping `cutlet`/`fugashi`
   and losing Japanese lyric romanisation. **This is the one open design
   question in the programme.**
2. ~~**Wire `installer/build.ps1`.**~~ Built 2026-09-04. The build syncs the
   full resolution first, collects the packs against it (unpacked sizes exist
   nowhere but an installed tree, and the CUDA 12.8 assertion is what proves
   the collected wheels are the cu128 ones), then syncs the staged runtime
   down to `--extra studio`. `packs.json` is staged beside `pyproject.toml`
   and the bundled wheels into `{app}\packs`, which is
   `service.packs.bundled_dir`. `runtime-manifest.json` deliberately does
   **not** gain them: it is verified against the *checkout* before anything is
   built, and these three files do not exist at that point — they are pinned
   by digest in `packs.json` instead, by the generator that made them, and
   `pack_worker` refuses a bundled wheel that does not match before it goes
   near site-packages.
3. ~~**The Settings pane.**~~ Built 2026-09-04 as its own category beside
   Models: models are weights and packs are the code that reads them. Rows
   carry both volumes' figures; Cancel is offered while it downloads and
   withdrawn once pip starts writing into the running site-packages.
4. ~~**Say what a finished install means.**~~ Answered 2026-09-04, and it is
   both, in that order: the landing re-runs `doctor.run_checks(force=True)`
   the way a finished fetch does, and a module that still will not resolve in
   *this* process (`service.packs.unresolved`, after the import caches are
   invalidated) asks for a restart out loud rather than leaving a mode grey.

5. **Collect the other two packs once, on a real line**, and record the
   figures. Only `rig` has ever been collected; `text2image` and `music` are
   multi-gigabyte and `music` is the only one that exercises the sdist build
   path (three compiles, one of them a C extension). It is also the only one
   that will ever exercise the bundled-wheel branch of `pack_worker.collect`,
   which is today covered by tests alone.

**Expected outcome:** an installer whose base download is around a gigabyte,
three packs a user chooses, and a figure for each of the two that have never
been collected.

## P28. Judge the character render benchmark — four verdicts, not one

**Why it is yours:** art, and it is the entry the whole character programme was
built to reach. Everything up to the pixels is measured, tested and structural:
thirty-one species over four body plans, four clip libraries, union framing
against a table (`docs/measurements/2026-09-05-union-framing.md`), a structural
check that flags every clipped and blank cell. None of that is the question. The
question is whether a 64-pixel sprite of a wolf reads as a wolf, and no test
this repository can write answers it.

**Why it is four verdicts.** The original scope was one fire ogre, and that was
written when there was one body plan. A convincing humanoid walk tells you
nothing about a quadruped: a four-beat lateral-sequence gait has diagonal pairs
moving out of phase, and at 64 px the legs are two pixels wide and overlap for
most of the cycle — it either reads as walking or reads as a smear, and which
one is not derivable from the humanoid's result. A wing beat and a blob's surge
are two more separate questions. So this entry does not close until it has four
answers, and it may well close as *proven, proven, repair, proven*.

**Do**, once per archetype — `humanoid`, `quadruped`, `winged`, `amorphous`:

1. **A representative species at the ladder's middle.** Suggested: `ogre` (it
   carries the fire theme, so it judges the effects pass at the same time),
   `wolf`, `dragon`, `slime`. 64 px, 32 colours, 8 directions, all five
   movements, **seed locked** — the same seed across the whole sitting, so a
   difference between two sheets is the thing you changed.
2. **Each body slider at both bounds, three seeds.** Six channels for humanoid,
   quadruped and winged, five for amorphous (`family.get_archetype(key).channels`
   is the list; every range is −1 to +1 and every default 0). That is the
   generator's actual span, and the failure it is looking for is a bound that
   produces a body the rig no longer fits — an arm inside the ribcage at
   `limb_length = -1`, a neck the clips swing through at `neck_length = +1`.
3. **Look at every direction and every animation, at zoom 1 and at zoom 4.**
   Zoom 1 is the size a player sees; zoom 4 is where you find out *why*. Troupe's
   heatmap is the reading order, not the verdict — click the flagged squares
   first, then watch the whole thing play.
4. **(Optional)** The five provisional clips added 2026-09-12 — `attack_02`,
   `cast`, `fall`, `hit`, `death` — can be judged in the same sitting, at the
   same locked seed, alongside the original five; P8's brief covers what each
   is meant to read as.

**What to judge, and it is the same seven questions each time:**

- **Recognisable from the front, and from the back.** The back is the half a
  generator has no reason to get right and the half a player looks at while
  walking away.
- **One identity across the eight directions.** The same creature turned, not
  eight creatures. Cross-direction size drift is what `qa.py`'s `drift_*`
  scores are pointed at.
- **A readable attack.** One frame that is unmistakably the hit, at 64 px, with
  no colour cue.
- **Walk contact.** Feet on the ground at the contact frames, and a bob that
  passes through zero between them.
- **Clean loops.** Idle, walk and run hand their last frame back to their first
  without a pop. Note that a fire theme's flame is *known* not to loop
  seamlessly on a short idle (the noise field does not come back round); judge
  the body separately from the flame.
- **Effects on their sockets.** For the fire species, the flame at the crown or
  the core, rising in world space in every direction, occluded when the socket
  is behind the body.
- **Feet on one line.** Across every direction of one animation, the ground
  line is the same row of pixels — a sprite that floats in three of eight
  directions is unusable however good it looks in the other five.

**Outcome:** a dated character-benchmark document under `docs/measurements/`
that records, **per archetype**, a verdict and the evidence for it. (Named here
by its directory rather than by a `YYYY-MM-DD-` placeholder path:
`tests/test_external_doc_links.py` reads a cited path as a claim that the file
exists, and a placeholder is a citation of a document nobody can open.) That document is
what lifts chapter 11's *"Built, awaiting the render benchmark"* to **Proven —
per archetype, not in one line** — or names the repair, in which case the
repair is code and comes back here as a finding. In the same sitting, calibrate
`qa.THRESHOLDS` against those sheets and record the numbers in
`docs/measurements/2026-09-02-troupe-qa-thresholds.md`, which was written
against synthetic sheets and says in as many words that it is waiting for
rendered motion. P8's authored keyframes are judged through the same sheets;
if the clips change afterwards, the thresholds are re-taken, not patched.

**No card is needed for any of this.** The character route is mesh generation
in-process, Blender on the CPU for the rig and the render, and numpy for the
reduction — which is what makes "run it again at both slider bounds" a
reasonable instruction rather than an afternoon of GPU time.

## P33. Judge the 2D walk cycle — an ogre and a humanoid

**Why it is yours:** art. The motion is *correct* and that is all a test can
say: `tests/inker/walk/` pins that no limb ever changes length, that the stance
foot is on the ground line on every frame it should be, that the stance foot
travels backwards and never forwards, that the cycle hands its last frame back
to its first, and that one rig renders the same bytes twice. None of that is the
question. The question is whether a drawing cut into fourteen rigid pieces and
turned about its joints reads as a body walking or as a paper puppet rotating,
and no assertion this repository can write answers it.

**Why an ogre and a humanoid, and not one of them.** They fail differently. A
humanoid has thin limbs whose silhouette is mostly outline, so the failure to
look for is joint gaps and the seam where an upper arm's end leaves its lower
arm's start. An ogre is mass: thick limbs overlap through most of the cycle, so
the failure is a limb reading as a flat card sliding over another flat card,
which is exactly what a rigid cut-out is. A verdict taken on one is a claim
about that build, not about the feature.

**Do:**

1. **Draw or open one side-view ogre and one side-view humanoid.** Layers per
   body part if you have them; otherwise one layer and the marquee, which is the
   path most users will take and therefore the one worth walking.
2. **Set both up and record how long it takes**, honestly, including the part
   that is not the tool: repairing artwork that was never drawn to be cut. A
   torso with an arm painted over it has no torso underneath, and the panel
   cannot invent one. If the preparation dominates, that is the finding.
3. **Play each at native size and at 4x**, and judge seven things separately:
   readable steps; the knee bending forward and not backward; gaps at the
   joints; limb overlap where two pieces cross; silhouette stability from frame
   to frame; the loop seam between frame eight and frame one; and whether the
   arms read as opposing the legs rather than merely moving.
4. **Bake both, and export a sheet**, to confirm the timing survives the trip
   and the frames are the ones the preview showed.
5. **Try the far-limb shading slider at 1.0 and at 0.6.** Whether a copied far
   limb needs shading to read as behind the body is the one control decision
   taken here without evidence.

**If it reads as a rotating paper puppet, write that down before anything is
expanded.** That is the milestone, and a negative answer is a real one: it
would mean the next move is deformation -- head and torso counter-motion,
squash on the contact frame -- rather than more directions or more actions, and
it is much cheaper to learn that from two drawings than from a second workflow
built on top of the first.

**Deferred on purpose, and not to be built until this closes:** saved rigs,
regenerating a baked walk from its rig, other directions, other actions,
automatic segmentation, and any AI reconstruction of occluded parts. Each of
them multiplies whatever this verdict says, in whichever direction it says it.

**Expected outcome:** one sentence per figure saying whether the walk is
convincing, a recorded preparation time, and — if the answer is no — the
specific thing that broke the illusion.

## P29. Prove the update path against a real release

**Why it is yours:** it needs a release published under your account and an
older build to offer it to, and neither is something this repository can do to
itself. Everything up to that point is built and tested: `service.updates`,
`pipelines/update_worker.py`, Settings -> Updates, and
`scripts/make_update_manifest.py`. The real check runs today against the public
`jmbell88/warlock-studio` and correctly reports "up to date", because that
repository has published no releases at all -- which is exactly why the
interesting half is unproven.

**Do**, once:

1. Build the installer (`pwsh scripts\rebuild.ps1`), then
   `uv run python scripts/make_update_manifest.py`.
2. Publish a GitHub Release carrying **both** `dist\WarlockSetup-v<version>.exe`
   and `dist\update-manifest.json`.
3. On a machine (or a build) whose `__version__` is lower, open Settings ->
   Updates and go Check -> Download -> Run Installer.

**What would fail:** an asset name that does not match what the manifest pins
(the check refuses, correctly, and says which); a release published without the
manifest (the app says "up to date" and offers nothing, which is the designed
behaviour and needs to be seen once so it is not mistaken for a bug); and the
installer refusing to upgrade an install it is running beside, which is the one
thing no test in this repository can reach.

## Also owed, smaller

- **Tutorial sample assets** (art): a 32×32 `.ora` sprite with a few layers
  and frames for the Inker chapters; a 16 px tileset of sixteen to twenty-four
  tiles with a terrain set for Plotter and Packwright; a low-poly `crate.glb`
  for Clay; ~~the humanoid from P4 for Troupe~~ — struck 2026-09-05: chapter 11
  now opens on Create's Character type, and the thirty-one shipped species *are*
  its samples, in the build, needing no file and no licence. Original and
  project-licensed. If they land: `src/warlock/assets/tutorial/`, added to the
  hatchling force-include, under about a megabyte, used by the last section of
  chapters 05, 07, 09 and 10.
- **Delete the pre-purge mirror** at `D:/Projects/_archive/warlock-pre-purge.git`
  once you have worked in the rewritten repository long enough to be
  satisfied. Keeping it indefinitely means keeping the problem indefinitely.

## P30. Decide which of Inker's four right-hand panes gives up height

**Why it is yours:** art direction. Four panes want the same 750 px and the
question is which one matters least while drawing, which is a judgement about
how the mode is used rather than a fact about the code.

**Where it stands.** Inker's five export doors moved out of the timeline's
second toolbar row into the bridge on 2026-09-05 (`3476f114`), because that row
was measured overflowing at 1280x800 scale 1.0: three of the five collapsed into
a `...` menu, "Skip empty" was clipped mid-word, and the row beneath it was cut
off by the pane's bottom edge. The exports the row existed for were the part of
it a user could not see.

The move fixed that and moved the pressure. `inker-generate` now stacks Drawing
file, Export and the exits, under Preview and Tools in the same column, and at
1280x800 the last three doors, the collapsed **Sheet options** header and the
four exit buttons are below the fold. **Nothing is unreachable** -- the pane is
an ordinary scrolling child -- and the exercise harness reports `clipped: 5`
against a measured baseline of `clipped: 1`, that one being **Revert to
original**, which was already below the fold before this work.

This is a soft overflow where there was a hard one, which is why it shipped. It
is still worse than it should be.

**Do** -- one of these, and it is a choice, not a defect to grind at:

1. **Give Preview less.** It is the largest of the four and it is a playback
    surface; a shorter one may cost nothing while drawing.
2. **Compact the doors to two columns.** Three rows instead of five, about
    76 px back. It costs the labels: "Export per layer..." does not fit ~130 px,
    so they would have to shorten, and the label is the door.
3. **Put Export behind a collapsed header**, as Sheet options already is. Keeps
    the exits visible and adds a click to every export -- which is the thing the
    move just removed, so this is the weakest of the three.
4. **Decide it is fine.** A sidebar that scrolls is a sidebar that scrolls, and
    1280x800 is the floor rather than the common case.

Whichever is taken, `scripts/exercise_mode.py --mode inker --out <dir>` reports
the clipped count, so the result is measurable rather than a matter of opinion
about a screenshot.

**Re-measure before choosing.** The 2026-09-07 layout-share fix changed what an
undragged column starts at: `layout.column` now passes `heights` only the
proportions somebody has actually dragged, so an untouched split gets the even
division instead of the flat 0.55 every key used to borrow. Inker's right column
is exactly such a column, so the `clipped: 5` figure above was measured under
starting heights this build no longer uses. The choice between the four options
is unchanged and still art direction; the number it is being made against is
stale, and re-running the harness is a minute's work.

## P34. Judge Clay's fifteen shapes and eight figures, and settle two defaults

**Why it is yours:** art direction and two design decisions. Every item here was
raised by your own review of the generated geometry on 2026-09-06, and each one
turns on how the shapes should *look* or what a first insert should *do* —
neither is a fact about the code, and the audit that day (findings clay-h1 to
clay-h6, plus clay-08) could measure them but not settle them.

**Where it stands.** The geometry is correct and the suite proves it; what is in
question is whether it reads well. Two of the seven are blocked measurements
rather than opinions — the numbers are taken and written down below, and only
the choice is missing.

**Do** — the four that need eyes on renders:

1. ~~**Figure proportions read as overlapping beads.**~~ **Built 2026-09-06.**
    The cause was mechanical: a capsule whose bone is shorter than twice its
    radius collapses to an exact sphere, which was every torso segment on the
    humanoid and bird, the whole quadruped barrel and most of the serpent.
    `presets._mass` now places body masses as scaled ellipsoids sized by
    anatomy rather than bone length — broad, flattened front-to-back, unequal
    and overlapping — and limbs were thickened to match.
    `test_a_bodys_torso_is_one_form_rather_than_stacked_balls` is the gate.
    **Left alone deliberately:** `insect` (its one collapsed part is not part of
    a chain) and `blob` (stacked lobes are the archetype). Judge those two on
    renders if they still bother you.
2. **Fish and bird silhouettes are weak.** The fish's dorsal fin reads as
    detached, and rectangular fins, wings and beaks hurt recognition. Wants
    attachment overlap, tapered wedges, and a deliberate wing outline and
    thickness direction.
3. **The shape chooser undersells the objects, and by more than it did.** Sphere
    and torus share a circle icon, several others borrow unrelated symbols, and
    the eight figures have labels with no preview. Recognisable silhouettes or
    rendered thumbnails would carry it. **Widened 2026-09-10:** `lathe`, `sweep`
    and `tube` joined the registry and took the spline, layers and waypoints
    glyphs, none of which draws the shape it stands for -- a lathe and a column
    are the pair a reader most needs told apart, and the icon comment in
    `panes/clay_tools.py` already concedes the set is strained. Three shapes
    were judged against renders on 2026-09-06 and these three have never been
    looked at, so this item now wants eyes on the *new* defaults too: the
    goblet, the L-bracket and the S-curve cable are each a default somebody
    picked to demonstrate a generator, not one measured against what a user
    would want first out of the palette.
4. ~~**"Insect / spider (six-legged)" is two animals in one label.** Renaming
    it "Insect" is free;~~ **Built 2026-09-06** (audit finding docs-14): the
    label is `"Insect"` in both `templates/insect.json` and
    `clay/presets.py`. Whether a genuine eight-legged spider template is
    wanted is the actual question, and stays open.

**Do** — the three that are one decision each:

5. ~~**Pick the grounding convention.**~~ **Answered and built 2026-09-06:**
    terrestrial figures sit on the ground, swimmers keep their placement. The six
    that walk (humanoid, biped_tail, quadruped, bird, insect, blob) now come out
    of `presets.build` with their lowest built vertex at exactly Y=0; `serpent`
    and `fish` are named in `presets.SWIMMERS` and are never moved. The shift is
    derived from the built geometry rather than eight constants, so a later edit
    to a radius cannot un-ground a figure —
    `tests/clay/test_presets.py::test_every_terrestrial_figure_preset_meets_the_ground_plane`
    and `::test_swimmers_keep_their_authored_placement` hold both halves.
    **Still open, and smaller than it was:** whether a "Place on ground" action
    over real mesh bounds is wanted for ordinary objects, which is a different
    feature from a preset knowing where it lands.
6. ~~**Decide whether organic presets insert smooth-shaded.**~~ **Answered and
    built 2026-09-06:** they do. `clay_ops._shade_auto`'s angle rule moved into
    `clay/shading.py` and both insertion doors apply it, so spheres,
    icospheres, capsules, toruses and every figure's limbs and heads arrive
    smooth while boxes, pyramids, arches, columns and a figure's hands and jaw
    stay flat. Cylinders and cones stay flat too, by the rule rather than in
    spite of it. `LIMB_RINGS` went 3→4 and the torus default `sides` 12→16
    because both had been stepping by exactly the 30-degree threshold and came
    back a third smooth; silhouettes did not move. A parameter rebuild
    preserves shading (`clay/regen.carry_over`, since 2026-09-10 — it moved
    out of `clay_props._carry_shading` when that rebuild also started
    carrying per-face material).
    **Still open:** whether Flat/Smooth should be offered as a control *at*
    insertion, rather than applied by the rule and overridden afterwards.
7. **Decide whether a figure keeps its identity after placement.**
    `docs/manual/30-clay.md` says a figure "is a starting point that saves you
    the assembly, not a special kind of object — once placed, nothing" marks it,
    and that is a written decision, not an oversight. Reversing it means
    persistent assembly membership, a "Select figure" verb, shared transforms,
    collapsible outliner groups, and the properties panel learning to edit a
    multi-selection. Either answer costs a manual change.

**Expected outcome:** items 5 to 7 answered in a sentence each, which unblocks
the code; items 1 to 4 answered as art direction, against renders rather than
against this file.

## P43. Judge what an agent actually builds in Clay

**Why it is yours:** nobody but a person can say whether the thing is
recognisable. Four tranches of work (2026-09-10) gave an agent element mode and
selection, a recoverable transport, curved and swept generators, repetition and
placement ops, and object names a batch can address — and every one of them is
asserted by unit tests proving one tool does one thing. Nothing has ever asked
whether the surface, taken together, can turn a brief into something that looks
like a chair. That is a judgement against renders, not a measurement.

**Where it stands.** The two tiers a machine *can* run are built and green.
`tests/test_agent_transcripts.py` replays a transcript against a real document
and asserts what it built — a regression gate, not a score, and it says so.
`scripts/agent_bench.py --serve` stands the real app up with a throwaway home,
the bridge switched on and a recorder running, prints the corpus and the
connect line, and leaves a transcript behind; `--show` reads one back. The
corpus and the decision rules are **pre-registered** in
`docs/measurements/2026-09-10-clay-agent-benchmark-preregistration.md`, written
before any session, and the bar is fixed there: three of five subjects at grade
0 or better on the standing −5..+5 mesh scale, at least one of them not the
`easy` one.

The honest N today is **zero** — zero model sessions, zero graded subjects. One
of the five subjects (the chair) has a hand-authored transcript, written by
reasoning about the geometry rather than by a model, and every file that
mentions it says so.

**Addendum (2026-09-13):** the tool catalogue grew again before this sitting's
sessions ran — ten `character_*` tools, one resource family
(`warlock://character/...`) and one prompt (`character_sheets_from_description`)
landed on `feature/game-character-pipeline` for the character pipeline's own
agent surface. This sitting's corpus stays Clay-only; nothing in it asks a
model to touch a character tool, and the pre-registration above is unchanged.
But the fixed context every session pays before its first useful call —
`tools/list` plus the `initialize` instructions — now includes that whole
second surface regardless of whether a session ever calls into it, so whoever
runs these sessions must account for the added catalogue cost when reading how
a model spends its calls, the same way `docs/INVARIANTS.md`'s catalogue-budget
bullet already tracks Clay's own growth.

**Do:**

1. Run `uv run python scripts/agent_bench.py --serve`. It prints a
   `claude mcp add` line carrying the throwaway `WARLOCK_HOME`; paste it into
   your agent client, and connect.
2. Hand the model **one** corpus subject per session, verbatim from
   `docs/measurements/corpora/clay-agent-v1.txt`, and nothing else — no hints,
   no corrections, no "try a lathe for that". What it does unaided is the
   measurement. Five subjects, five sessions.
3. Let it export. Keep the transcript and the GLB; the pre-registration's
   retention rule exists because an earlier sweep scored zero of twenty and
   could not diagnose why, its assets already deleted.
4. Look at the renders, and grade each subject −5..+5 on the scale in
   `docs/measurements/2026-08-09-grade-scale.md`, **without the call count in
   front of you** — the pre-registration requires that, because knowing a
   thing took four calls talks a grader into liking it.

**The questions, to answer in your own words beside the grades:**

- Did it build the subject, or something else it found easier?
- Which refusals did it hit that it could not have avoided? Those are defects,
  and they are worth more than the grades.
- Where did it stop — did it run out of surface, or out of patience?
- Is there a shape in the corpus this surface simply cannot express?
- Does `--show`'s transcript read like a plan, or like flailing?

**Expected outcome:** five grades and the answers above, written into a dated
`docs/measurements/` results document that applies the pre-registered rule
verbatim — including if the answer is that the bar was missed. Any subject
whose session is worth keeping gets its transcript promoted into
`tests/fixtures/agent_transcripts/` with a claim file beside it, which is how a
session becomes a permanent regression test. Then this entry is struck out.

## P35. Settle Muse's Steps guidance, and measure it

**Why it is yours:** the manual and the slider disagree about where Steps stops
paying, and neither number has a measurement behind it, so there is nothing in
the repository that says which is right. The 2026-09-07 audit (finding docs-06)
found the disagreement and deliberately did not pick a winner: choosing one
would have written a figure into the manual on no more authority than the one
already there, and this repository's rule is that a constant the corpus is keyed
on gets a dated document *before* it changes.

**Where it stands.** `docs/manual/16-generating-a-soundtrack.md` and
`docs/manual/36-muse.md` both say "below about 30 the output audibly falls
apart; above about 80 you are paying for time". The slider's own tooltip in
`studio/panes/muse_recipe.py` says "past about 60, not better". Two different
ceilings, and only the manual warns about a floor at all. No
`docs/measurements/` document is keyed on either figure. (The line numbers this
entry used to carry had drifted by up to eighteen lines when the 2026-09-12
audit checked them, finding docs-11; the sentences are named instead, because a
citation that has to be re-verified to be followed is worse than none.)

**Do:**

1. Generate the same prompt and seed at a spread of step counts — 20, 30, 40,
   60, 80, 120 is enough — holding everything else fixed, and listen.
2. Decide where quality stops improving and where it starts falling apart, and
   write both into a dated `docs/measurements/` note with the clips or their
   parameters named, in that directory's format.
3. Make the manual and the tooltip quote that document's numbers, and cite it
   from both.

**Expected outcome:** one figure for the ceiling and one for the floor, recorded
once and referenced twice, so the next person to touch either sentence can see
what it rests on.

## P36. Decide whether Clay gets an "adjust last operation" card

**Why it is yours:** a design decision about whether the mode wants the control
at all. `ClayState.last_op` was written by every op and read by no pane; the
2026-09-07 audit (finding clay-10) found the bookkeeping had no reader anywhere
in the shipped UI and was not on this list either. The audit removed it rather
than leaving dead machinery behind, which is the reversible half of the choice —
the record of *what* an op ran against is a dozen lines to reinstate, and the
git history has them.

**Where it stands.** `LastOp`, `ClayState.last_op`, `clay_ops._remember` and
`_op_context` were removed on 2026-09-07, with explanatory comments left at
`clay_mode.py:711`, `clay_mode.py:1058` and `clay_state.py:44` saying where they
went. `tests/clay/test_clay_last_op_removed.py` pins the removal so nothing
half-reintroduces it.

**Do:** decide whether Clay should offer a card that re-runs the last operation
against the current selection with its parameters adjustable — the thing the
bookkeeping existed to feed. If yes, it is a pane plus the record, and the
record's old shape is in the history. If no, say so here and this entry is
deleted.

**Expected outcome:** one sentence either way.

## P37. Audit the three other stages that rename onto a served name

**Why it is yours:** it needs judgement about each stage's intent, not a rule
that can be written down. Widening the publish/commit scan on 2026-09-07
(finding service-01) turned up three further call sites that rename onto served
names and are not in `PUBLISHERS`: `_deform_qa` (`_q_rig.py`), the
model-promotion stage in `_q_generate.py`, and `_remesh` (`_q_mesh.py`). Only
the comments beside each call distinguish a *completion marker* — which must
commit the cancel token, or a late cancel deletes finished work — from an
intermediate checkpoint a cancel may still legitimately unwind. That is why the
table stayed hand-written rather than derived: a "last write wins" heuristic
promotes the wrong call in some of these functions and misses the real one in
others, which fails open, silently.

**Where it stands.** `PUBLISHERS` in `tests/test_job_durability.py` now covers
nine stages and all nine pass. ~~`_remesh` (`_q_mesh.py`)~~ **Closed 2026-09-08**
(the 2026-09-08 audit, finding docs-05): it already carried the classifying
comment and already committed after its `os.replace`, so only the missing
`PUBLISHERS` row was real, and it is added. `_deform_qa` and the promotion stage
are untouched, and nothing yet says whether they are correct.

**Do:** read each of the two remaining, decide whether its rename is a completion marker
or a checkpoint, and say which in a comment beside it. Add a `PUBLISHERS` row
for every one that is a completion marker, and confirm it commits.

**Expected outcome:** three stages classified, however many rows that adds, and
no remaining rename onto a served name whose status is unstated.

## P38. Run a ten-minute Muse take on the target card, and confirm the ceiling is honest

**Why it is yours:** a card and a stopwatch. `_jobs_music.MAX_DURATION` moved
from four minutes to ten on 2026-09-07 on the strength of an argument, not a
run: `vram.estimate` prices a music job on the registry row
(`models.MusicModel.vram_gib` / `host_peak_gib`, `src/warlock/models.py`
~line 1511, still flagged there as conservative estimates "until the GPU lane
publishes a `docs/measurements/` document for them") plus a flat source term,
and never reads duration at all — so raising the ceiling widened a term
admission cannot see, and nobody has generated anywhere near the new number to
find out what that costs in practice.

**Do:** on the target card, generate one take at 600 s from an ordinary
style-tag brief, watching VRAM through the whole run rather than only at the
end — peak is what `host_peak_gib` claims to bound, and that is a claim about
the sample as a whole, not the loaded checkpoint. Confirm it neither OOMs nor
exceeds `vram_gib` or `host_peak_gib`. Record the result — a clean pass, or the
real figures if it is not one — in a dated `docs/measurements/` document; P23
asks for the same document from a shorter take, and this is the long end of
the same measurement if that entry is still open when this runs.

**This is the item that decides whether `vram.estimate` needs a duration
term.** If a 600 s take fits comfortably under the existing estimates, the
argument that duration does not need pricing holds at ten minutes and not only
at four. If it does not, either the estimates are wrong or the pricing that
ignores duration is, and both are code changes this file cannot make for you.

**Expected outcome:** one dated measurement, and a settled answer to the
question this change made and could not itself answer.

## P40. Settle which way Inker's timeline stack reads on screen

**Why it is yours:** nothing headless can assert imgui's screen Y-order, and the
two accounts inside the code disagree, so the only way to know is to look. The
module docstring of `panes/inker_timeline.py` and chapter 28 both promise "the
background is the bottom row, which is Aseprite's order, Photoshop's order",
while the comment on the draw loop in the same file says the opposite -- that the
panel draws top-down "because Photoshop's does" -- and `row_plan`/`_grid` submit
rows in ascending stack-index order, background first. Under imgui's ordinary
top-down child layout that puts the background at the *top*, which is the reverse
of what both documents claim (the 2026-09-08 audit, finding inker-H1).
`tests/inker/test_timeline_merge.py::test_the_rows_run_bottom_up` pins the data
order and says nothing about the screen.

**Do:** open Inker with three or four layers and look, or run
`/exercise-mode inker` and read the timeline capture. Decide which of the two
intents is real. Then either reverse the `row_plan`/`_grid` walk so the highest
stack index draws first -- matching the manual and Aseprite -- or correct the
module docstring and chapter 28 to describe the ascending order the code draws
and drop the "Aseprite's order, Photoshop's order" claim. Either way check that
`_row_menu`'s **Move up** and **Move down** move a row the direction their labels
say on screen, since they inherit whichever answer is right.

**Expected outcome:** one screenshot, one decision, and the code and both
documents saying the same thing about it -- plus a test pinning whichever of the
two the decision makes true.

## P41. Re-measure the retexture coverage table under the fixed weight bake

**Why it is yours:** it needs a real card, real weights and the retexture corpus,
and the conclusion it feeds is an art-direction call about where coverage work
goes next. `docs/measurements/2026-08-15-retexture-visibility.md` reports the
coverage figures the current constants are pinned on -- the 54.4% facing-only
baseline, the derived "16.2 pp of the shipped coverage was smear", and the 38%
/41% honest-coverage conclusion -- and it was measured five days before
`docs/measurements/2026-08-20-retexture-weight-colorspace.md` found the
sRGB-encoded `weight` bake target that made `MIN_FACING = 0.15` behave as
approximately 0.0196. The 08-20 document corrects the 08-08 table by name and
never mentions 08-15's, although `combine()` thresholds the same facing weights
on top of the newer depth test, so every occlusion-tested arm in 08-15 ran
through the identical uncorrected bake (the 2026-09-08 audit, finding docs-H1).

**Do:** re-run `scripts/retexture_probe.py`'s coverage table for the `a0`, `a1`,
`b`, `c1`, `c2` and `d-ladder-*` arms against the current, colorspace-fixed
weight bake, following 08-15's own "Reproducing" section. Compare each figure
against the one that document reports.

**Expected outcome:** either a dated note confirming 08-15's percentages hold
within noise -- in which case `DEPTH_EPS_LO`/`HI` and the feather constants stay
pinned where they are and the Tier 3 UV-space inpainting question stands as
posed -- or a correction document in the shape 08-20 used for 08-08, and a
re-decision of whether Tier 3 is still the named next coverage step.

## P39. Reproduce the one faulthandler dump, or let it expire

**Why it is yours:** it is one dump with no second occurrence, and nothing in
the log says what the user was doing. `crash.log` carries exactly one
faulthandler traceback, under the `=== session 2026-09-07T17:32:00 pid=18052
warlock=0.0.39 ===` header: `Windows fatal exception: code 0x80010012`
(`RPC_E_SERVER_DIED_DNE`, a COM/RPC teardown code), with both
`pipelines/trellis.py:_pump` and `pipelines/matting.py:_pump` live on their own
threads at the time. Separately, `warlock.log` records one unclean shutdown
(pid 10176, session 2026-09-07T02:19Z, 0.0.38) -- the warning
`_setup_logging`'s session marker exists to raise.

Neither is diagnosable from what was written down. faulthandler dumps some
non-fatal SEH codes and execution continues, so the dump is not by itself proof
of a crash, and the unclean shutdown is from a different session and a different
build. The one thing that would settle it is a second occurrence with a known
sequence in front of it.

**Do:** run a trellis reconstruction with matting in the same session, three or
four times, closing the window while a job is still in flight on at least one of
them -- that is the shape both threads were in. Watch `crash.log` for a second
`0x80010012`. If one appears, note what was on screen and whether the app
survived it; if none does after a handful of runs, strike this item out. Windows
event 2004 is worth reading either way; `_setup_logging`'s own warning points
there.

**Expected outcome:** either a reproduction with a sequence attached -- at which
point it becomes ordinary code work -- or a struck-out line saying it did not
recur across N runs of the shape that produced it. A single dump with no
reproduction is not something this file should carry indefinitely.

## P47. Open a Mason scene in a real engine

**Why it is yours:** an engine this repository does not have. Mason's GLB
export is ordinary glTF and is verified as such, but the `scene.json` manifest
beside it is **ours** -- format `warlock-mason-scene`, version 1 -- invented
with no importer to invent it against, so every claim it makes is checked by
shape rather than by anything reading it. This was the Mason programme's open
question 7 and it is the one that plan closed with an admission rather than an
answer.

The rule is P6's and P7's exactly: a green test proves our writer agrees with
our own reader, and a round trip through two halves we wrote cannot catch an
error both halves make together.

**Do:** build a scene worth testing -- a ground, a few library assets, a group,
a prefab with three instances, a point light, a spot light with a narrowed
cone, a directional light and a camera -- and export it all three ways. Then,
in at least two engines (Godot and Unity are the ones this project's users
name):

1. Import `scene.glb` alone and check the hierarchy, the instance transforms,
   the lights and the camera all arrived. The lights are the sharp ones: they
   go out as `KHR_lights_punctual`, and intensity units are where glTF
   importers most often disagree with each other.
2. Check the handedness and the scale without converting anything. Metres,
   Y-up, -Z forward and right-handed are glTF's own, and the whole point of
   fixing them there was that nothing should need to convert -- a scene that
   arrives mirrored or at a hundredth scale means that claim is wrong
   somewhere.
3. Read `scene.json` beside it and ask whether a person writing an importer
   could actually use it: are the node addresses stable, are the counts the
   ones an engine would want, do the user properties arrive in a usable shape.
4. Import `scene.obj` in the same two engines and confirm the named losses are
   the only losses.

**Expected outcome:** either the manifest is confirmed against something that
reads it and `docs/manual/13-putting-it-in-a-game.md` gains real per-engine
rows for Mason, or the first version of a format we still control is fixed
before anyone has stored scenes against it. If the manifest survives, write
the engine rows into `docs/COMPAT.md` the way the Tiled rows are written, so
the next change to the exporter has something to fail against.

## P50. Open a Warlock Godot scaffold in real Godot 4.x

**Why it is yours:** an engine this repository does not have, P47's exact
argument applied to Troupe's character export instead of Mason's scene one.
`godotscene.py` is stdlib-only and structurally proven (`tests/_tscn.py`
parses the `.tscn` it writes and checks every resource reference resolves),
but every choice in it that Godot itself could plausibly answer differently
across versions is a named, commented constant precisely because nobody has
opened the result in the real editor yet -- the loop-name heuristic, the
`AnimationPlayer` node name the glTF importer creates, the transition
switch/advance-mode integers, and whether a suffixed animation name survives
import or gets stripped back off. See
`docs/measurements/2026-09-12-troupe-open-clip-vocabulary.md` for the naming
rules this scaffold is built from and `docs/INVARIANTS.md`'s Godot-export
entry for what is asserted rather than verified.

**Do:** export a rigged humanoid with **Export for Godot…** (from the
inspector) and import the resulting folder into a Godot 4.3+ project. Then:

1. Confirm on the Godot version actually used that the importer renames
   `idle-loop` to `idle` and sets it looping, the way master's
   `_pre_fix_node` does (`editor/import/3d/resource_importer_scene.cpp`) --
   `godotscene.animation_reference` already predicts that rename and the
   `.tscn` plays the post-import name (`idle`, not `idle-loop`), so this is a
   confirmation, not an open question: does the version in hand agree with
   master, or does an older/newer importer keep the suffix or use a
   different marker set, in which case `_LOOP_NAME_MARKERS` or
   `animation_reference`'s simulation is what to change.
2. Check that `anim_player = NodePath("../Model/AnimationPlayer")` resolves
   under the instanced GLB, and that the relative `ext_resource` path to the
   `.glb` loads without a re-import prompt or a broken-reference warning.
3. Check that the `AnimationTree` actually drives the model's bones, not just
   that it resolves and plays with no error. Since the 4.2 `AnimationMixer`
   refactor, an `AnimationTree` resolves its animation track paths from its
   own `root_node` (default `..`), and in this scaffold the tree is a sibling
   of `Model`, not a child of it -- so `..` from the tree names the scene
   root, not the node the `AnimationPlayer` (and the skinned mesh) actually
   sits under. If the pose does not reach the mesh, the fix is
   `root_node = NodePath("../Model")` on the `AnimationTree` node in
   `godotscene.scene_text`, or parenting the tree under `Model` instead of
   the scene root -- record which one Godot actually wants, and whether
   `anim_player` alone was ever going to be sufficient on its own.
4. `travel()` to each state the exported clips populate and confirm it
   plays: the `locomotion` blend space at positions 0/1/2 (idle/walk/run),
   `attack`, `attack_02`, `cast`, `jump` transitioning to `fall` at its end,
   `hit` returning to locomotion, and `death` as a terminal state nothing
   leaves.
5. Note whether Godot tolerates or ignores a `load_steps` count that turns
   out to be wrong -- this module computes it rather than trusting one, so a
   mismatch would be this module's bug, not the editor's, but the editor's
   reaction to a wrong count either way is worth knowing.
6. Reproduce godotengine/godot#108823 against this export specifically: the
   loop flag is not shown in Godot's own import dialog, and is lost entirely
   once animations are saved to separate files. Confirm whether that loss
   reaches this scaffold's `AnimationPlayer` the same way, since the whole
   point of the `-loop`/`-once` naming is to survive it.
7. Build an `AnimatedSprite2D` (a `SpriteFrames` resource) from an **Export
   frames…** folder and its `manifest.json`, and confirm the compass-named
   subfolders, frame order and `fps`/`loop` fields are enough to wire up
   without guessing.

**Expected outcome:** a dated `docs/measurements/` document recording what
real Godot actually does at each of the six points above, and -- where the
answer is a one-constant fix in `godotscene.py` -- that fix, made and tested
the same day rather than left for a second sitting.

## P51. Judge imported Mixamo clips on a Warlock humanoid

**Why it is yours:** art, and a licence question only a human can clear.
"Import clip" (`clipmaps.py`, `cliptransfer.py`, `service.clip_import`) is
structurally proven -- a synthetic Mixamo-named rig round-trips through it in
`tests/test_clip_import_blender.py` -- but nothing here has judged what a
*real* Mixamo download looks like once it is walking on a Warlock skeleton
at 32px, and two of the pure math module's own tolerances
(`cliptransfer.LOOP_MATCH_DEG` and `KEY_TOLERANCE_DEG`) were picked with no
real clip to check them against.

**You supply the files.** Download a walk, a run and a jump from Mixamo as
FBX **without skin** (Warlock only samples the armature and never reads the
mesh), plus an in-place variant of at least one of them if Mixamo offers it.
**Licence note:** Warlock downloads nothing here -- Adobe's Mixamo terms
govern the motion data you bring in, not this repository's licence, and a
downloaded or derived file must never be committed to this repository.

**Do:** Poser → pick the `humanoid` template → **Import clip…** → point it at
one of the downloaded files → **Save clips**. Then, per clip:

1. Render a Troupe sheet at 32px and 64px and watch it play, and bake
   `animated.glb` and check it in a scratch script.
2. **Foot sliding.** Does a planted foot stay on one ground line across the
   contact frames, or does it drift.
3. **Arm height against the shipped A-pose.** Mixamo's neutral is usually a
   loose T-pose or A-pose of its own; judge whether the rest-alignment math
   (`cliptransfer._rest_alignment`) leaves the arms reading naturally or
   pinned in a way the source clip never intended.
4. **Knee direction.** A mis-signed facing or a mismatched chain shows up
   first as a knee bending backward.
5. **Facing.** Does the character face the camera the same way a shipped
   clip does, front where front should be.
6. **Loop seams**, on the walk and run: does the last sampled frame meet the
   first with no pop, and does auto-loop detection (`loop="auto"`) call it
   correctly.

**Expected outcome:** a dated `docs/measurements/` document recording, per
clip, the six judgements above, and explicitly answering two questions the
code cannot answer of itself: does `LOOP_MATCH_DEG` (3.0 degrees) and
`KEY_TOLERANCE_DEG` (1.5 degrees) hold against a real download, or does one
need retuning; and does resampling to at most 32 frames
(`cliptransfer.MAX_CLIP_FRAMES`) read as stiff at the sizes Troupe actually
renders. Either finding is a one-constant fix in `cliptransfer.py`, recorded
here rather than guessed at.

## P48. Decide whether library mesh rows can be dragged into Mason

**Why it is yours:** it is the library's to decide, not Mason's. `panes/library.can_drag`
lifts only *finished 2D references* today, which is why a mesh reaches a scene
through the exits panel and the Assets pane's own list rather than by being
dragged. Widening that predicate is the obvious fix and it is not Mason's to
make: Create's drop slot reads the same predicate, so widening it changes what
Create accepts as well, and a Mason-specific payload is the alternative that
costs a second mechanism.

This was the Mason programme's open question 4. Stages E, F and G each left it
alone deliberately rather than taking it on the way past, which is the right
call for a predicate two modes read and the reason it has survived to here.

**Do:** decide which of the three it is -- widen `can_drag` and accept what
that means for Create's drop slot, add a Mason-specific drag payload, or leave
it as it is and accept that the Assets pane is how meshes reach a scene. If the
answer is "leave it", say so in `docs/INVARIANTS.md` beside the predicate, so
the next person to notice does not re-derive the question.

**Expected outcome:** one of those three, taken rather than deferred. The
current state is defensible and is not the problem; what is owed is a decision
on record rather than three stages of "not mine".

---

## Open findings

Code work a review or a real run turned up. Each is buildable and is struck out
the day it is built; this section is deleted when it is empty. F1-F4 came out
of the 2026-09-05 clean-machine install
(`docs/measurements/2026-09-05-clean-machine-install.md`) — two installs, four
app sessions, and her `warlock.log` read — and all four were built the same day.
F5 and F6 came out of the Mason programme (open questions 2 and 3); both were
measured on 2026-09-14, F6 closed on the numbers and F5's decision moved to P57.

1. ~~**F1. A failed fetch discarded everything it downloaded.**~~ Built
   2026-09-05. `fetch_one`'s unwind was `except BaseException:
   rmtree(staging)`, and `huggingface_hub` keeps its resume bookkeeping in
   `.cache/` *inside* `local_dir`, so a failure threw away the ability to
   resume along with the bytes — one engine attempt ran eight minutes and
   several GB and was discarded whole. The tree is now kept and keyed by a
   `.warlock-resume.json` marker holding the entire spec; a later fetch resumes
   only on an exact match, a terminal failure (digest mismatch, missing rename
   source, no digest) still drops it, both sweeps spare a marked tree, and both
   transports retry with backoff over the same tree. The destination is
   untouched by any of it — nothing moves until the tree is whole and verified.
   `docs/INVARIANTS.md` carries the reasoning; five tests in
   `tests/test_fetch.py` carry the claims.
2. ~~**F2. A socket error reached a non-developer verbatim.**~~ Built
   2026-09-05. `download.describe_failure` translates by `winerror`/`errno`,
   walking the `__cause__` chain because the transports bury the number, and
   falls back on class name for `hf_xet`'s Rust errors which carry none.
   Offline, reset, timeout, refused and DNS are five remedies where there was
   one stringified exception; a sentence this project wrote itself is passed
   through untouched; anything unrecognised names itself and points at the log.
   The raw exception now goes to `warlock.log` beside the friendly one, because
   this incident was diagnosed from a log and the translation must not have
   made that harder. It lives in `download.py` rather than `fetch_worker.py` so
   it can be imported without setting `HF_HUB_OFFLINE=0`.
   `tests/test_fetch_messages.py`.
3. ~~**F3. The health poll imported torch while pip was writing it.**~~ Built
   2026-09-05. `vram.probe` and `doctor._cuda_check` caught `ImportError`
   only, and a half-written torch raises from the DLL loader — `OSError
   [WinError 126]`, then `PermissionError [WinError 32]` — so the whole health
   task died, five tracebacks in twenty-one seconds. `probe` now falls back to
   NVML on any import failure and the CUDA row reports "installed but will not
   load" with the likely cause, non-fatal, since it clears on the next launch.
   `tests/test_torch_import_failures.py`.
4. ~~**F4. The pack gate at a mode's door was never written.**~~ Built
   2026-09-05. `Pack.modes` was read in one place, a Settings label, so a base
   install sent the user to Models to fetch ~23 GB that could not run without
   `torch`. `model_gate.mode_gate` now answers packs-or-models for a mode and
   **packs come first**, since weights with nothing to read them buy nothing;
   the rail, its tooltip and `set_mode`'s refusal all read that one answer, and
   the library escape applies to both halves so nobody is locked out of
   finished work. `tests/test_pack_gate.py`.
5. ~~**F5. A Mason scene has no VRAM admission control.**~~ Measured
   2026-09-14 (`docs/measurements/2026-09-14-mason-scene-vram-and-retint-cost.md`)
   and no longer buildable as written: what is left is a policy decision, now
   **P57**. Sixty distinct textured assets cost 814 MiB (~12.7 MiB each,
   linear to 240), the GPU-side `MasonView._cache` has no byte budget at all
   (the 512 MiB budget is CPU-side geometry only), VRAM stays at a session's
   high-water mark after release, and an allocation failure reaches only the
   run loop's generic crash handler. The original entry follows. Sixty textured library
   assets placed in one scene is real video memory, and `service.validation.check_vram`
   guards the **queue** — the thing that runs jobs — not the studio. Mason's
   asset cache has a byte budget and instancing means N placements of one asset
   are one upload, so the cheap cases are already cheap; what has no answer is a
   scene that genuinely wants more than the card has. `MAX_PLACED` (100,000) and
   the 1,500-item warning are about what a *frame* can afford and say nothing
   about texture memory. The failure mode is a driver-level allocation failure
   with no refusal in front of it, which is the one shape of failure this app
   otherwise always puts a sentence before. Wants a measurement first — what a
   scene of sixty distinct textured assets actually costs on the target card —
   and then a decision about whether the answer is a refusal, a warning, or
   evicting the cache harder. The measurement is the part that needs a card.
6. ~~**F6. A material override costs a second upload of identical geometry.**~~
   Measured 2026-09-14 and deliberately **not built**
   (`docs/measurements/2026-09-14-mason-scene-vram-and-retint-cost.md`): 36
   retinted copies of one 50K-triangle asset are 36 uploads and ~509 MiB, but
   the median frame is 1.236 ms against 1.231 ms untinted, because the draw
   count is the placement count either way. The per-draw material uniform
   (~35–45 lines over `mason_view.py`, `mason/scene.py` and `viewer/render.py`,
   no shader change) is sketched there for the day overrides prove common.
   The original entry follows. The
   override is in the GPU cache key, so a retinted copy of a shared asset is a
   second upload of the same triangles. That is also true on the way out —
   glTF puts the material on the primitive rather than on the node, so the
   exporter has the same cost and `mason/gltfout.py`'s docstring says so — but
   the exporter's copy is written once and the renderer's is paid every frame.
   The fix is a per-draw material uniform rather than a cache key, which is a
   renderer change and was out of scope for the mode that found it. Worth doing
   only if overrides turn out to be common: place a few dozen retinted copies of
   one asset and see whether it matters before changing how the renderer binds
   materials.
7. ~~**F7. The jump clip's knee bends the wrong way.**~~ Built 2026-09-14 (the
   2026-09-14 audit, finding docs-05): `shin.L`/`shin.R` flipped to `+0.6428` in
   `jump crouch` and `+0.5299` in `jump land`, pinned as data by
   `tests/test_clip_library_poses.py::test_jump_crouch_and_land_bend_the_knee_forward_not_backward`.
   Not re-rendered: no Blender on the fixing machine, so the visual confirmation
   the entry asked for is still owed on the next Troupe run of a humanoid jump.
   `jump launch`/`rise`/`fall`/`apex` still carry opposite-signed thigh and shin
   and were deliberately left alone, since nobody has judged them backward.
   **Corrected the same day, after a render:** the shin flip was half the fix.
   Rendering Quaternius's Superhero Male through `blender_worker` showed the
   crouch floating face-down with its ankles at hip height, because the
   shipped crouch had *every* leg bone's sign inverted and the "thigh and shin
   share a sign" rule this entry reasoned from is true of a walk's swing leg
   and false of any crouch (hip flexion is a negative thigh X, knee flexion a
   positive shin X). `thigh` and `foot` are now negated too (`-0.4384`/`-0.2079`
   crouch, `-0.3746`/`-0.1736` land): forward kinematics puts the ankle under the
   hips at rest height and the foot within 4° of flat, and the re-render
   squats with both feet on the ground. The test was replaced by
   `tests/test_clip_library_poses.py::test_jump_crouch_and_land_flex_the_hip_forward_and_the_knee_back_with_the_foot_flat`.
   The other jump poses are F10. The original entry follows. Found 2026-09-12 running
   a real human-authored mesh (Quaternius's CC0 "Superhero Male") through
   *Send to Troupe* for P4 — the first time this template has been judged on
   art rather than CesiumMan. `src/warlock/templates/clips/humanoid.json`'s
   pose data has a consistent rule across every other flexed pose: `thigh.L`/
   `thigh.R` and `shin.L`/`shin.R` rotate in the **same sign** when a leg
   bends (walk's "passing A" pose: `thigh.R +0.0698, shin.R +0.2924` — hip
   flexes a little, knee bends more, which is a normal swing-through). `jump
   crouch` (`thigh +0.4384, shin -0.6428`) and `jump land`
   (`thigh +0.3746, shin -0.5299`) break that rule — shin rotates opposite in
   sign to thigh, both bent hard. Given every other pose's convention, that
   bends the knee backward rather than forward: confirmed visually on the
   rendered sheet, the landing/crouch legs look reverse-jointed. Reproducible
   on any humanoid character, not specific to the test mesh.

   **Do:** flip the sign of `shin.L`/`shin.R` in `jump crouch` and `jump land`
   (candidates: `+0.6428`/`+0.5299`, matching the walk/run poses' sign
   agreement) and re-render the same character's jump clip to confirm the
   knee now bends forward. Check `jump launch`/`rise`/`fall`/`apex` too —
   they read as closer to straight-legged so the sign disagreement may not
   bite there, but they were not checked with the same rigor this pass gave
   crouch/land.
8. ~~**F8. The walk clip may play backward left/right — unconfirmed.**~~ Built
   2026-09-14, and it was real: not a yaw-mirroring bug (every yaw renders the
   same bone data from an orbiting camera) but the clip itself. "walk contact
   A" plants the left leg behind, and "walk passing A", the key after it,
   lifted the *right* leg and planted the left under the hips, so forward
   kinematics had the planted foot sliding forward 0.2 of a body height and
   the swinging foot travelling back. Every pose was sane; the two passing
   poses were each other's, and "run" had the same swap. The bone data of
   `passing A` and `passing B` is exchanged in both clips (names and key order
   kept, so `pipelines/sheet.py`'s "contact A, passing A, contact B, passing
   B" stays true), pinned by
   `tests/test_clip_library_poses.py::test_the_leg_behind_at_a_contact_is_the_leg_the_next_passing_pose_lifts`,
   with the bird library's walk and run, which already obeyed the rule, as the
   control. The original entry follows. Same run
   as F7: side-view (yaw 90/270) walk frames looked, on a static contact
   sheet, like the gait was reversed. Unlike F7 this is **not yet backed by
   data** — `walk contact A/B` and `walk passing A/B`'s thigh/shin signs are
   internally consistent with each other (unlike jump's), so if there is a
   real bug here it is a different mechanism than F7's — a candidate is a
   yaw-mirroring or leading-leg/screen-direction mismatch for opposite-facing
   views, but this was not traced into code before the session ended. A
   static frame cannot settle "is this backward" on its own; needs the actual
   sheet played back (Troupe's own preview, or a GIF of the yaw-90 walk
   frames in sequence) before concluding anything.
9. ~~**F9. `clay_op` crashes instead of refusing when a value inside `params` has
   the wrong type.**~~ Built 2026-09-14 (the 2026-09-14 audit, finding docs-06):
   `agent_clay._op_params_type_refusal` refuses an undeclared key or a value that
   is not a single number by `field="params"` before `clay_ops.run` sees it; the
   three calls below are
   `tests/test_agent_clay_door_types.py::test_clay_op_refuses_a_wrong_typed_param_value_instead_of_crashing`.
   The original entry follows. Found 2026-09-13 while authoring Clay assistant run B's rows,
   through the real `agent_clay.call` door: `mirror-x` with `{"axis": 0}` answers
   "failed unexpectedly; see the log", and `mirror-copy` with `{"axis": "x"}`
   leaks a raw "could not convert string to float". It reproduced again on
   2026-09-14 in run B's eval as `TypeError: float() argument must be a string
   or a real number, not 'list'`, raised at `clay_ops.run`'s clamp
   (`float(values[param.name])`, `clay_ops.py` ~285, via `agent_clay._h_op`):
   the door checks `params` keys, not value types, so a trained model's
   near-miss reads as a crash rather than a refusal it can learn from.
   `clay_add_primitive` already solved the same class for scalar-only params
   (`agent_clay.py` refuses "`params.base must be a single number`", the fix
   for run A's `pyramid` list-`base` crash), so the fix is that check extended
   to `clay_op`'s declared `Param`s, refusing by field; the regression tests are
   the two calls above plus a list value, each asserting a refusal rather than
   an unexpected failure.
10. **F10. Seven provisional humanoid poses bend the knee backward.** Found
    2026-09-14 by the forward-kinematics pass that settled F7 and F8, which
    measured the signed knee bend of every leg in the humanoid library (hip →
    knee → ankle in the side plane; positive is a real knee). A straight
    planted leg reads a few degrees either side of zero and is fine; these do
    not: `jump rise` (L −40°, R −14°), `jump apex` (L −56°, R −34°), `jump fall`
    (L −22°, R −44°), `fall a`/`fall b` (−90° on the lifted leg), `death fall`
    (−70°), `death crumple` (−100°) and `death down` (−110°), the three deaths
    also with ankles below the ground plane. All are `"provisional": true`
    placeholders in P8's sense, so no new angles were invented here: the
    convention is now written down in `tests/test_clip_library_poses.py`'s
    docstring (negative thigh X flexes the hip forward, positive shin X flexes
    the knee), and **Do:** re-author those poses to it, then add a
    library-wide "no knee bends backward past ~15°" check to that file — it
    fails on the current data today, which is why it was not added alone.
    Needs a rendered sheet per clip to judge, which `render_check.py`-style
    rigging of any shipped species or the Superhero Male now gives in minutes.

**What is left on that machine is not code**: whether the resets stop once a
retry can outlast them (F1 and F2 together should turn "never finishes" into
"finishes eventually"), and the still-unrun `HF_HUB_DISABLE_XET=1` experiment.
Both belonged to P1 step 4, which is now closed (see Closed records) — that
machine work is done; these two items were not re-run and are not tracked
elsewhere, noted here only so they are not lost with the entry that named
them.

---

## Closed records (kept so nobody re-derives them)

- **P49, decide whether `clay_render` gives an agent a lit picture.** Closed
  2026-09-14 as already built: Clay agent round two's T3 (`d1e7f1e0`) gave
  `clay_render` a `shading` enum (`unlit`, `lit`, `wireframe`, `wire_overlay`,
  `xray`, `object_id`) with `unlit` the default, so the trellis path's pinned
  picture never moved and a lit render is opt-in. The description no longer says
  "flat-shaded". `tests/test_clay_view.py::test_render_png_lit_shading_shows_more_than_one_face_grey`
  and `tests/test_agent_clay.py::test_clay_render_shading_defaults_to_unlit_and_threads_through_to_render_png`.

- **P55, finish Clay assistant run B.** Closed 2026-09-14: run B is written up as
  a negative result and run A stays the candidate
  (`docs/measurements/2026-09-14-clay-assistant-run-B.md`). Trained, merged and
  exported at Q8_0, then scored at t0.2 n3: 168.0/232 against run A's 185.0, builds
  125.0 vs 140.7 of 174 with non-overlapping samples, edits and queries flat. The card
  and dataset changed together, so the cause is unseparated; an ablation run (B's
  data with A's card, or the reverse) was left unscheduled on purpose.

- **P32, re-examine the `trellis_tex_res = 512` pin.** Closed 2026-09-12: the
  pin stays
  (`docs/measurements/2026-09-12-trellis-tex-res-pin-reexamined.md`). One
  props-v1 reference (the jug), byte-identical geometry both arms, `--tex-res
  1024` against the shipped `--tex-res 512`: the 1024 texture shows visible
  per-texel noise the 512 one does not, so the noise reproduces on v0.6.0 and
  `config.py`'s pin is confirmed current rather than stale. No `src/` change.
  One side-note recorded there: omitting `--tex-res` decodes at res1024 on
  this binary, not res512 as the CLI's own help text claims for "auto" — not
  a finding about the shipped path, since Warlock pins the value explicitly
  rather than relying on auto, but worth knowing if anyone reaches for an
  unflagged run as a control again.

- **P42, judge the approved cutout against the server's own.** Closed
  2026-09-12: the change stands
  (`docs/measurements/2026-09-08-approved-cutout-as-input.md`'s Results and
  Verdict). Ten-unit run against `detail-v1`: usable 5/5 on both arms, the
  pairwise call better-or-same on 4 of 5 subjects (only the helmet went the
  other way), and the silhouette-regression rule that would have overridden
  everything did not trip. The promote modal's approved cutout stays the
  reconstruction's input, `bg_removal=auto` in that mode included; no other
  door is affected. One gap noted rather than resolved: no per-job
  `trellis.log` exists on this build, so the void check rests on the
  alpha-channel/stored-params stand-in for every host-cut unit rather than
  independent log confirmation — worth a follow-on finding if the log was
  expected to exist.

- **P1, generate an asset from a clean-machine install.** Closed 2026-09-12.
  Every step proved across three dated measurements: install
  (`docs/measurements/2026-09-05-clean-machine-install.md`), the engine as a
  Settings download (`docs/measurements/2026-09-10-engine-as-a-download.md`),
  and fetch/recovery
  (`docs/measurements/2026-09-12-clean-machine-fetch-and-recovery.md`) —
  dinov2, SDXL + Hyper-SD, all three packs, TRELLIS GGUF cancel-and-resume,
  `warlock doctor` exit 0 with `[SETUP]`, Remove stopping the resident server,
  and all four pack recovery paths plus an upgrade-over-itself, all against
  real hardware. The one thing never reached on the 8 GiB clean machine — a
  real generation through TRELLIS — is moot: both other candidate machines
  (laptop, alternative desktop) are also 8 GiB, below `vram.TRELLIS_GIB`
  (16.0) by design, and generation is already proven working on the dev
  machine. Nothing a bigger clean card would show has been left unanswered.
- **P2, purge `examples/` from history.** Done 2026-09-03: `git filter-repo`,
  then the remote deleted and recreated rather than force-pushed, because
  GitHub keeps unreachable objects fetchable by SHA. 963 commits and both tags
  survive; `git log --all -- examples/` is empty; `master`'s tree hash did not
  move. Three traps for the next rewrite: `filter-repo` deletes `origin`; it
  migrates remote-tracking refs into local branches and rewrites those too;
  `gh repo delete` needs the `delete_repo` scope. The mirror is the item under
  *Also owed*.
- **P3, the graded mesh run.** Done 2026-09-02: props-v1 on trellis.cpp
  v0.6.0 is 11 of 22 usable, fantasy-v1 10 of 20
  (`docs/measurements/2026-09-02-trellis-060-props.md`,
  `docs/measurements/2026-09-02-fantasy-v1.md`); the hole audit closed
  (`docs/measurements/2026-09-02-hole-audit-vs-grade.md`). The tex-res pin
  survived as P32 (2026-09-06 audit, finding docs-15: P3 is closed, so the
  follow-on question keeps its own number rather than reusing this one).
- **P5, the end-to-end `charsheet` run.** Struck 2026-09-05, absorbed into P28
  rather than answered. Its premise was "a card": the sheet job had never run on
  hardware and P4's mesh was what it was waiting for. Neither holds now — the
  character route builds mesh, rig and sheet with no GPU at all and no supplied
  file, so the run is a press rather than a scheduled event, and P28 makes it
  four times over with something to judge at the end of each. A one-line "it
  ran" verdict would have been strictly less than that.
- **P9, code signing.** Answered no for the closed beta on 2026-09-03. The
  revisit triggers and the priced option (Azure Trusted Signing) are in
  `docs/INVARIANTS.md`.
- **P27, the release candidate and its figures.** Built and measured
  2026-09-05 into `INSTALL.md`: 810 MB download (846,950,916 bytes), about
  1.4 GB installed base, SHA-256 `254b3af9...`, at v0.0.35. The placeholders it
  existed to replace are gone. Installing that build on a machine that is not
  this one is the surviving half of P1.
- **P10's tile-sheet half.** Shipped 2026-08-29 as Materials and Terrain set,
  with the old path labelled *Grid (legacy)*; the verdict is P15.
- **P11's phase-6 design.** Decided 2026-09-02 and built 2026-09-03: the merge
  happens in Inker, three-way, and on conflict the hand edit stands.
- **P12, humanoid reconstruction from a single image.** Answered no on
  2026-08-30: limbs come back bent and stretched at the shipped default
  (`docs/measurements/2026-08-30-sdxl-cfg-props.md`). Phase 7 stays deferred;
  the supplied-base-mesh path is untouched.
- **P18, `style_lock`.** Built 2026-08-30 as the *Keep one style across the
  list* checkbox on the Materials arm, with its cost beside it.
- **P31, the sweep abort.** Built 2026-09-07 exactly as specified:
  `Worker.on_job_failed` (queue.py, default `None`) fires from `_process`'s
  `error` branch; `service.sweeps.on_job_failed` owns the decision, cancelling
  a failed unit's still-*queued* siblings that share its `server_group`
  (`server_group_of`, factored out of `UnitPlan.server_group`) via
  `JobStore.cancel_sweep_units` (`resolve_candidates`'s shape, conditional on
  `status='queued'`). The reason text names `scripts/sweep_refill.py`, whose
  docstring does re-queue exactly `cancelled` and shutdown-interrupted units.
  `docs/INVARIANTS.md` now says a cancel comes in two kinds.
- **The 3.12 CI leg.** Read 2026-09-03: fourteen failures, six of them rig
  paths that fail rather than skip without `bpy`. The floor was raised to 3.13
  (`bpy` is 3.13-only and the installer packs its own 3.13 runtime).
- **Host commit (D1/D2/D3).** Closed 2026-08-22 by the t2i child process;
  `docs/measurements/2026-08-22-trampoline-child-pids.md` and
  `docs/INVARIANTS.md` hold the figures and the stdin-reader rule.
- **Release audit (2026-08-24).** `REPORT.md` deleted 2026-08-25 once every
  code-closable finding closed: GPL-3.0, sdist allowlist, licence disclosure
  in the picker, notices staged into the installer.
- **GPU lane.** 26 passed, 0 errors on 2026-08-21; the isometric guide and the
  3/4 clause remain unproven on a card.
- **Art direction and palettes.** Ramps installed 2026-08-21
  (`~/.warlock/palettes/cosmos`, `light_world`).

## Not on this list on purpose

Decisions with arguments beside them, not backlog:

- **Scale and crop of a tilemap layer** stay refused, permanently. They
  resample, and a tileset cannot follow a resample.
- **A hexagonal 120° tile rotation** stays refused: not a permutation of the
  pixel grid. `docs/COMPAT.md` carries the argument.
- **Pen/tablet pressure, ICC colour, per-frame palettes** and the rest of the
  Aseprite parity programme's named non-goals, in `docs/INVARIANTS.md`.
  Per-cel opacity and z-index were built on 2026-08-30 and struck from this
  list (`docs/measurements/2026-08-30-cel-z-below-cache.md`).
- **An LLM director for Troupe.** No LLM infrastructure exists, a local HTTP
  endpoint would be the first socket in the app besides the trellis client,
  and it would break `HF_HUB_OFFLINE=1`. The user approves a *picture*, which
  is a better interface than a manifest.
  **This bullet is about Warlock calling out, and it stays refused.** The MCP
  server (`src/warlock/mcp/`, 2026-09-09) is the opposite arrow and is not a
  counter-example: an agent already running on the machine connects *in*, over
  a named pipe, and drives Clay and the character pipeline through the same
  doors a pane does. Warlock runs exactly one pinned model, Familiar, on
  loopback, and still makes no network egress; the network exceptions stay
  three. `HF_HUB_OFFLINE=1` is untouched. What was
  refused was the app acquiring an appetite for a service somewhere else, and
  it still has none. The character pipeline (2026-09-13) does not change this
  either: the agent surface starts no image model and no mesh model of its
  own -- species come only from the procedural family registry, text->SDXL->
  trellis stays unreachable from it (import-pinned), and a human still judges
  the sprite sheets it writes once by eye, the same as one built from a pane.
- **A named animation "set" as a registry.** "Sword-and-shield", "unarmed" and
  the like are not entries anywhere: a set is only ever the list of movements
  it implies, and `character_create`/`character_sheet_create` take that list
  directly. A real weapon-style motion set -- one where "sword-and-shield"
  actually moved differently from "unarmed" rather than sharing the same idle
  and walk -- would be a clip-library-variant registry plus the P8 art pass to
  fill it, not a label over the existing ten clips, and nobody has asked for
  that yet.

## P46. Decide whether the bridge stays lockstep

**Why it is yours:** it is a design call that wants evidence P43 has not
produced yet, and building it first would be guessing at a cost nobody has
measured.

`mcp/bridge.py` relays one frame at a time: read a line from stdin, send it,
block on the reply, write it out. That is what makes the relay dumb enough to
be trustworthy, and it has three consequences an agent can feel. A client
cannot cancel a call -- `notifications/cancelled` is handled nowhere, and the
bridge would not read it during an in-flight call anyway, so
`agent_host.CALL_TIMEOUT`'s thirty seconds is the only escape. There are no
progress notifications, so a long call is indistinguishable from a hung one
until it answers. And requests cannot be pipelined: a client that sends a
second before the first answers has it queued by the OS, which is correct but
serial, so `ping` cannot be answered while `clay_export` encodes a GLB on the
frame thread.

None of that is reachable as a defect today, because the calls an agent makes
are short and `warlock_status` already answers the "is it still running"
question out of band, on the listener thread, without waiting for the frame.
The question is whether that stays true.

**Do:** run P43 first -- it is the sitting that puts a real model through the
surface, and its transcripts are the only place the shape of real agent
traffic is going to show up. Then read them for the three things this is
about: whether any call ran long enough that a human would have wanted to
cancel it, whether any client gave up before `CALL_TIMEOUT` did, and whether
the model ever wanted to ask something while a call was in flight.

**Expected outcome:** either a dated `docs/measurements/` document saying the
lockstep relay is adequate and why, and this struck out -- or one naming the
call shape that broke it, which is then the specification for whatever
replaces it. Do not redesign the transport without that document: the
lockstep is what keeps the untestable half of this feature thin, and trading
that away needs a reason better than symmetry with other MCP servers.

## P44. Re-verify the pixelklein LoRA revision against the hub, and mark it either way

**Why it is yours:** it needs a networked machine, and this one is offline by
construction. Every other `Fetch` revision in `models.py` is stated in its own
comment to have been read off downloaded bytes. `Limbicnation/pixel-art-lora`'s
is not: its comment says outright that the SHA "is unconfirmed against local
bytes -- the SHA comes from the hub's `refs/main`". That caveat exists in the
code and nowhere else. `docs/MODELS.md` lists the same revision alongside every
verified one with nothing to distinguish it, so a reader sizing the risk of a
pasted `hf download` command cannot tell which pins were checked.

The 2026-09-11 audit found this while diffing all 28 `Fetch` records against
`docs/MODELS.md` in both directions (finding service-H1, the only one of the
hundred that a machine could not settle). Everything else in that diff matched
character for character. A revision pin that was never checked against real
bytes is the one place an upstream force-push, or a bad initial read, would
change what a user's download fetches with nothing going red.

**Do:** on a machine with a network, fetch
`Limbicnation/pixel-art-lora` at the pinned revision
`0ac8e5c3400af68228811edc324721e25fc26777`, hash the downloaded file, and
compare it against what the registry expects. Then either (a) confirm the pin,
and rewrite the code comment to say it was verified on this date like its
neighbours, or (b) find it has moved, update the revision and say so in
`docs/MODELS.md`. Whichever happens, remove the asymmetry: the doc should not
present an unverified pin as though it were a verified one.

**Expected outcome:** the comment in `models.py` no longer says "unconfirmed",
and `docs/MODELS.md` either needs no caveat or carries one. Strike this out
with the date it was checked.

## P45. Re-derive the four mesh thresholds nothing measured

**Why it is yours:** it wants a card, a corpus and eyes on the results -- the
three things this machine does not have. Four tuned numbers decide user-visible
behaviour with no `docs/measurements/` document behind any of them, which the
2026-09-11 audit recorded as evidence gaps rather than wrong values (findings
pipelines-08, pipelines-09, pipelines-10):

- `remesh.FACE_PROFILES` (low 2,000 / medium 8,000 / high 30,000 quads) and
  `FACES_MIN`/`FACES_MAX`, live in the remesh panel's combo. The sibling
  `optimize.PROFILES` triangle tiers are deliberately kept *out* of the generate
  form until a qualification run backs them; these went straight in.
- `remesh.VOXEL_FRACTION` (0.005) and `BAKE_MARGIN_PX` (8).
- `meshreport.TRIANGLE_BUDGET` (150,000), `GROUND_TOLERANCE` (0.001) and
  `SIZE_TOLERANCE` (0.01), which together decide a mesh's ready/review verdict --
  next to `HOLE_WARN`, which *is* backed
  (`docs/measurements/2026-08-06-audit-resolution.md`).

Each now carries a comment saying plainly that it is unmeasured and why, so
nothing reads as measured that is not. That is honest, not finished.

**Do:** take the props corpus and, for each ladder, render the same asset at
each rung and look at the results. For the remesh budgets the question is
whether the three named rungs are the right three for a mobile/indie target
(and whether `medium` is the right default); for `VOXEL_FRACTION` whether a
finer or coarser voxel closes the plate-crust gaps `meshaudit` flags without
rounding off a blade; for the `meshreport` thresholds whether the ready/review
line falls where a person would put it on a corpus of real reconstructions.
Write one dated document per ladder in `docs/measurements/` before changing any
value -- these are corpus-keyed the moment a stored report is judged against
them.

**Expected outcome:** three dated documents, and either the constants confirmed
where they are or moved with the run that moved them. Strike this out then.

## P52. Decide whether INSTALL.md keeps exact installer and runtime sizes

**Why it is yours:** it wants a release build on a clean machine and a decision
only the person cutting releases can make. The 2026-09-13 audit (finding
docs-03, an evidence gap) found that `INSTALL.md` states exact byte counts for
the v0.0.46 installer (169,666,529 B) and the installed runtime (539 MB: 446 /
48 / 41) as this build's, but `/warlock-land --release` only renames the
installer file; nothing re-measures those figures, so every later release
inherits numbers that describe an earlier one.

**Do:** decide between the two. Either measure the built installer and the
installed runtime at each release, and add that step to the release walk; or
soften the figures in `INSTALL.md` to approximate ones ("about 170 MB",
"about 540 MB installed") that survive a patch release.

**Expected outcome:** `INSTALL.md` states figures that are either re-measured
per release or honestly approximate. Strike this out then.

## P53. Licence review for redistributing the Clay-assistant fine-tune

**Why it is yours:** it is a legal judgment about a model this project trained
and would ship, not a technical question. T4 (2026-09-13) pinned Familiar's
weights row to a *testing* pin -- Unsloth's Q8_0 requantization of the stock
`google/gemma-4-E2B-it` instruct model, itself Apache-2.0 -- specifically so
Familiar has something real to run before this item is resolved. T10 swaps
that testing pin for the Clay-assistant fine-tune
(`training/clay-assistant/`, trained on top of the same base model), and that
fine-tune is what actually needs the review: whether Warlock may redistribute
a derivative of a Google-published model under Warlock's own download row
(rather than pointing at a third party's Hub repo the way every other entry in
`models.py` does), what licence terms the derivative carries forward, and
whether attribution or a licence file has to ship beside it.

**Do:** read `google/gemma-4-E2B-it`'s actual licence terms (not just its HF
`license` tag, which the T4 pin's testing repo and the base model both state
as Apache-2.0 -- confirm that holds for a fine-tuned derivative too, since a
base-model licence tag is not always the same promise once weights are
retrained on new data) and decide whether Warlock hosting and distributing the
fine-tuned GGUF is clear to do, needs an attribution notice
(`THIRD-PARTY-NOTICES.md`), or is blocked.

**Expected outcome:** a decision recorded in `docs/MODELS.md`'s Familiar
section and `THIRD-PARTY-NOTICES.md`, and T10 either proceeds with the
fine-tune pin or stays on the testing pin with the reason written down. Strike
this out then.

## P54. Familiar tranches T3–T10 — fully specified, deliberately unstarted until T3's card is chosen

**Why it waits:** T3 freezes the prompt card of the model being integrated,
and which card that is, is now a decision rather than a merge. Run B reached
master on 2026-09-14 as a negative result
(`docs/measurements/2026-09-14-clay-assistant-run-B.md`), and run A stays the
candidate. It landed without its figure-part catalogue in `agent_clay.py`, and
the catalogue followed on 2026-09-14 at the user's call, so master's live card
is now run B's (sha `cfa30687`, equal to the dataset manifest's `tools_sha`) —
the card of the run that regressed, not of run A, which was trained on
`70697ece` (`docs/measurements/data/clay-assistant/run-A/card.txt`). Decide
whether T3 freezes run A's recorded card beside a live card that has moved on,
or waits for a model that scores better on master's. The 2026-09-14 ablation
(`docs/measurements/2026-09-14-clay-assistant-ablation.md`) is the evidence for
that choice. Run B's card trained on run A's rows scored 174.0 against A's 185.0.
Run A's card on run B's rows scored 169.0. Neither displaced run A. So no
trained model yet does better on master's card than run A does on its own. T5–T8
build on T3's contract.

**Where it stands (2026-09-13, `feature/familiar`):** T0 (menu-bar status, bottom
pane), T1 (owner-counted agent lanes, in-app session), T2 (Clay scratch preview,
ghost, one-step Apply) and T4 (llama.cpp `b10948` and Unsloth Gemma 4 E2B Q8_0
rows, loopback `llama-server`, GPU lease, idle stop) are landed. The weights row
is a testing pin until the fine-tune replaces it.

**Do, in order, once T3's card is chosen** (`feature/familiar` was merged into
master on 2026-09-14, so this work starts from master): T3 frozen cards (`familiar/cards/`, hash equal to the trained dataset's
`tools_sha`), `contract.py` moved out of `training/clay-assistant/gen/convert.py`,
BM25 Manual retrieval, router and per-tab threads, and supply the card-hash
provider T4's spawn path already takes — the package's import pin must settle
whether `familiar/apply.py` and `scratch_ctx.py`, which reach Clay's GL-side
modules, move out of the pure package; T5 conversation loop, `llama_client`,
and the bottom pane's Familiar states; T6 router and cited Manual answers; T7
character skill with a plan card; T8 navigation and Create-draft doors; T9 GPU
smoke test plus a dated VRAM measurement before `vram.FAMILIAR_GIB` loses its
guess; T10 swap to the fine-tune (after P53). The screenshot and `/exercise-mode`
debt T0–T4 left behind does not wait on the card; it is P56.

**Owed by T5 specifically (2026-09-14 review):** `llama_client` must call
`LlamaServer.touch()` on every request, because the idle sweep reads
`last_used` and nothing but the health poll writes it today -- without the
touch a five-minute conversation has its server evicted mid-reply
(`tests/test_familiar.py::test_touch_resets_the_idle_clock_so_a_live_conversation_is_not_evicted`
is the door's test; the client's own test must show it calling it). It must
also read the key file rather than argv, and surface `ensure_started`'s
"cannot start while a GPU job holds the card" refusal as a pane state. Before
T5 ships a chat loop against the *testing* pin, note that run A's measurement
put base Gemma 4 E2B at 0 % door acceptance on Clay builds (the fine-tune: 74 %),
so a Clay skill on the testing pin fails every build a user asks for -- gate
the Clay skill on the fine-tune pin, or land P53/T10 first. `PARALLEL_SLOTS = 2`
and `CTX_SIZE = 16384` are fixed in `pipelines/llama.py`; the per-tab-threads
idea above has to be reconciled with two slots sharing one context before T5
inherits the numbers.

**Expected outcome:** Familiar answers in the bottom pane, previews Clay builds
as a ghost, and runs the fine-tune. Strike this entry per tranche as each lands.

## P56. Refresh `screenshots/` and exercise a workspace after the Familiar T0–T4 merge

**Why it is yours:** judging whether a screenshot shows the right thing and
whether an exercise pass's presses behaved is a human read of images, and
nothing in the suite checks `screenshots/`. It was split out of P54 by the
2026-09-14 audit (finding docs-08): it had been sitting inside an entry gated
on T3's card while needing nothing from it. `feature/familiar` was to refresh
`screenshots/` and run `/exercise-mode` on a workspace before merging, and it
was merged into master on 2026-09-14 at the user's request without either. T0
moved the status readouts into the menu bar and added the bottom pane, so every
capture of a mode's frame is now stale.

**Do:** on master, `uv run python scripts/screenshot_modes.py` and look at every
capture's menu bar and bottom pane; then `/exercise-mode clay` (the workspace
T2's ghost preview lives in) and read its press-by-press screenshots.

**Expected outcome:** `screenshots/` matches the shipped frame, and an exercise
pass over Clay either comes back clean or names what it found as open findings.

## P57. Decide what Mason does when a scene wants more video memory than the card has

**Why it is yours:** the measurement is done and the three answers trade a
user's access to their own document against a crash, which is a product call.
`docs/measurements/2026-09-14-mason-scene-vram-and-retint-cost.md` (it closed
open finding F5): a distinct textured asset costs ~12.7 MiB of VRAM, linear
from 60 (814 MiB) to 240 (3.1 GiB); the only byte budget on Mason's path is
`mason_assets.AssetSource`'s 512 MiB, which counts CPU-side geometry and never a
texture, while the GPU-side `MasonView._cache` has no budget at all; VRAM stays
at the session's high-water mark after a release; and an allocation failure
reaches only `main.py`'s generic crash handler. Measured on a 32 GiB card —
on the 8 GiB cards the beta targets the same rate fills the card at roughly
600 distinct assets, a projection nobody has run.

**Do:** pick one, or say none is worth it yet:
1. **Refuse** a placement or open that would not fit — needs a per-asset VRAM
   estimate *before* upload (texture dimensions from the job, not decoded
   pixels) checked against `vram.live_memory()`, which the queue's door already
   reads.
2. **Warn**, non-blocking, in `PLACED_WARN_THRESHOLD`'s shape — the same
   estimate against a softer threshold, never refusing to open a document.
3. **Evict the GPU cache harder** — byte accounting on `GpuModel` that it does
   not have today, plus a rule for evicting a still-visible asset (a re-upload
   stutter), and it bounds the live scene, not the session's peak.

Optionally first: run the document's `f5_isolated.py` on an 8 GiB card to turn
the ~600 projection into a number.

**Expected outcome:** a policy chosen and recorded in `docs/INVARIANTS.md`'s
Mason paragraphs, and — if it is 1, 2 or 3 — a fully specified open finding
for it, or a line there saying why a Mason scene deliberately has no VRAM
door.
