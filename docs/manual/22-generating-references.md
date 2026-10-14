# Generating references

A reference is the picture the mesh will be reconstructed from. Everything in this chapter lives on
the Reference stage. Your brief and settings are on the left, the canvas and attempts are in the middle,
and the selected result and History are on the right.

## The prompt

The large text box under **Your brief** (labelled **Describe the asset**, or **Shared style** on a
Tileset) is where you describe the object. Write the subject and nothing else — "a weathered wooden
crate bound with iron", "a compact energy rifle with panel seams". You do
not need to ask for a plain background, a single object, or a studio render on a 3D Model: for
that type (and for Seamless Material, Tileset and Sprite Sheet, each with its own template) the app
wraps whatever you write in a fixed template that already asks for all of that, because those are
the properties that make an image reconstruct cleanly. The **Image** type is the exception: it
sends your text exactly as written, with no template, so ask for the framing and background you
want. Jobs recorded before that change keep the old wrap when you rerun them.

The prompt is capped at 1000 characters and a counter in the bottom-right corner of the box shows
how much you have used once you have typed anything. The counter turns amber inside the last hundred
characters, because past the cap the box simply stops taking text.

When you have generated at least one reference, the **Your brief** card's header carries a
**Recent prompts…** button, at its right edge. It opens your last
twenty prompts, most recent first and deduplicated; with no history yet there is nothing to show
and the button is not drawn. Picking one replaces what is in the box. The history outlives
the session (it is saved with your other Realmspinner preferences and is there again after a restart), and
it is per prompt text only — if you want a whole recipe back, use **Copy settings to form** from a
job's overflow menu instead, which is described in [Rerun and
promotion](33-library-and-jobs.md#rerun-and-promotion).

Under **Advanced → Avoid**, further down the pane, is a second box listing what the image must not
contain. It is pre-filled with the things that most often ruin a reconstruction, and it is yours:
delete the text and the job runs with no negative prompt at all, which is a different request from
the default rather than a way back to it. **Reset image settings…**, in the **New…** menu, brings the
default text back along with the rest of the form. The box is hidden, not greyed, while the model
in use ignores a negative prompt; none of the three models that ship does, so it is always there.

```text
blurry, low quality, multiple objects, cropped, cut off,
text, watermark, signature, busy background, human hands
```

A second subject, or a subject cut off by the frame edge, is the single most common cause of a mesh
that comes out as nonsense — which is why this is a filled-in default rather than an empty field
you have to discover. You can edit it freely, or empty it deliberately. A negative prompt only has an
effect on a model that runs with real classifier-free guidance (guidance above 1.0), which is all
three that ship. See [Models and style LoRAs](#models-and-style-loras).

For the types that use a template, your text is composed into it before the image model sees it; an
Image job's composed prompt is simply your text. The finished job records the result as
`composed_prompt`, which the inspector shows.

The composed prompt has no hard length ceiling. CLIP's text encoders stop at 77 tokens, but the app
splits a longer prompt into several chunks on comma boundaries — never mid-phrase — encodes each
separately and joins them, which the image model's cross-attention accepts without complaint — so a
long prompt costs attention rather than being cut off. The soft limit still applies, though: a
longer conditioning sequence dilutes attention, so your prompt is best kept to a sentence.

## The screen at a glance

The header shows the asset's **stage rail**, **New…** and **Inspector**. Images, materials,
tilesets and sprite sheets use Reference → Export. A 3D model or character also has Mesh, Rig and Pose.
Unavailable stages explain their requirements on hover.

The left column is a stack of cards, each with a title row, 12 px of padding and an 8 px gap to the
next. It begins with **Your brief**: what you are making, a multiline description,
and **Images** (how many pictures one press draws) where the output supports several. Output-specific
artistic controls
follow in cards of their own, then the model and style recipe. **Advanced** (Seed and Avoid) and
**Conditioning** (reference images) are collapsible cards: click the title row to fold one, and the
column remembers whether each was open, across restarts. Character has its own controls and uses no
image model.

The **Generation plan** card is pinned at the foot of the column and is identical on every stage. It
holds the plan lines and, under them, the persistent **Create image**, **Generate reference** or other
output-specific button; the cards above it scroll within their own space, the plan card is as tall as
its own lines need, and the submit button stays visible. The plan states
stages, image count, recipe, estimated work, queue position and validation problems, with repair
buttons where available. A brief with nothing typed yet says *Describe the asset to start* in the
muted colour; the same line turns red once you have edited the prompt or pressed Generate on it. On a
machine with no image model, the plan says *Recipe: none installed*, the button is greyed, and the
problem carries **Download SDXL 1.0 (7.0 GiB)** (the model and its size), which ticks that download in
Settings → Models for you. Ctrl+Enter still submits,
and refuses for the same reasons the button is greyed: while a candidate batch is still undecided it says
"Decide the pending candidates first" and queues nothing -- pick one on the Mesh stage, then generate again.

Every new attempt belongs to the current creation and is saved. The **Attempts** card below the
canvas holds a strip that scrolls horizontally through that creation's results; it does not mix
in unrelated recent jobs.
Clicking an attempt inspects it without changing your draft or chosen reconstruction source.
**Load settings** explicitly loads a result's recipe. **Try again** uses a fresh
seed; **Reproduce recorded attempt** reuses the recorded recipe and seed, with results still subject
to runtime differences (it is greyed, with a reason, for a result that recorded no seed: a rig, a re-texture, a sheet or
a music result). Pin one result, select another and choose **Compare with pinned** for a
side-by-side comparison. **Favourite**, in the inspector's header, records a favourite; it does not
delete alternatives, and **Discard** moves an attempt you do not want to the trash (with an Undo in the
toast).

The right column holds the selected result's **Next step**, details and the **History** pane
(below). **Inspector** hides this column to give the canvas more room. Choosing another creation restores its
draft. **New… → New creation** starts with defaults; **New with previous settings** keeps the recipe
but clears the selected result and source. Both preserve prior attempts.

Tilesets can show the exported sheet or an individual cell, and blob terrain has a small assembled
terrain preview. Sprite sheets show the exported sheet, cells or animation tags, with pause and restart
controls. Playback uses the exported frame durations.

Under those, in amber rather than red, the plan may also draw one or more lines starting **Worth
knowing**. These are *advisories*, and the difference from the red ones is the whole point: an
advisory never disables Generate and never means the press will fail. It is something measured about
this kind of request that you may want to know before spending the time. The plan still says *Ready
to generate* above it. Where there is a safe one-press change, an advisory carries its own ghost
button, exactly as a problem does.

The advisory you are most likely to meet is the **open-form** lint on the *3D Model* type. When the
prompt names something with gaps, slats, spokes or thin members — a birdcage, a cart wheel, a rope
ladder — the plan says so, because open forms are the single largest surviving cause of an unusable
mesh: they reconstruct usable about two times in five, against about one in two across the graded
corpus overall. Two in five is a real chance, which is why this is a note and not a refusal. **Ask
for a closed form** appends a clause asking for filled-in gaps to your prompt; it appends rather than
rewrites, so you can see exactly what changed and delete it.

**What are you making?** has six entries: *Image*, *3D Model*, *Seamless Material*, *Tileset*, *Sprite
Sheet* and *Character*. Three of them are described elsewhere in this chapter — see
[Seamless tiles](#seamless-tiles), [Sheets](#sheets) and [Characters](#characters). *Character* is
the odd one out and worth knowing about early: it is the only type on this screen that generates no
image, uses no model weights and needs no graphics card.

Earlier versions carried a twelve-select creative taxonomy (category, genre, material and the
rest) behind a "More options" reveal. It was retired on 2026-08-17: no taxonomy axis ever measured
a quality win, and your prompt is the brief. Assets generated under it are unaffected — rerolling
or promoting one simply composes without the retired fragments.

**Reset image settings…** and **Reset mesh settings…**, in the **New…** menu (the one for the stage you
are on), reset that stage's form after a confirmation. Image reset
includes the output type, description and all output-specific settings. Mesh reset keeps the chosen
source. Neither reset deletes generated attempts or changes the other stage's form.

## Models and style LoRAs

The **Model** row is one dropdown. Its first entry is *Automatic*, which lets the app resolve a
checkpoint from what you are making and what is installed; every other entry is an installed
checkpoint you are naming yourself.

On *Automatic* the line underneath says what it resolved to — *Quality image · sdxl_cfg*, say —
along with anything that recipe trades. A control whose value is "Automatic" and says nothing else
is a control refusing to tell you what it did.

This used to be three controls: a *Fast*/*Quality* tier, an *Automatic*/*Advanced* switch, and the
checkpoint list. They were three controls for one decision, and the first two only ever chose which
checkpoint. There is no *Fast* tier any more: the four-step distilled recipes it chose between were
retired, and what is left is full-guidance models that hold a shape, which is the property the mesh
stage depends on, so automatic routing chooses among those for anything you mean to reconstruct.

A **Tileset** has no model row at all. Its recipe and its style LoRA are fixed, because a sheet
whose cells have to share a palette and a light direction is not a place to vary the machinery, and
the pane says so where the controls would be.

### The model registry

Naming a checkpoint replaces automatic routing. Three base models ship in the registry:

| Model | What it is | Runs at |
| --- | --- | --- |
| SDXL 1.0 (full CFG, structural control) | **The default.** The checkpoint run the way the checkpoint was trained — the negative prompt and ControlNet both work here. | 1024 px, 30 steps, guidance 7.0 |
| SDXL 1.0 + PAG (full CFG, cleaner structure) | The default's weights and recipe with two training-free sampling upgrades: perturbed-attention guidance and CFG rescale, which counters the washed-out look of high guidance. No new download. | 1024 px, 30 steps, guidance 7.0 |
| FLUX.2 klein-base 4B (full CFG) | A different architecture entirely, and the slowest thing here. | 1024 px, 50 steps, guidance 4.0 |

The sampler settings travel with the checkpoint and are not yours to set. They are part of the
model's identity: a checkpoint run at the wrong step count or scheduler degrades silently.

Three consequences are worth knowing. A **negative prompt** is only encoded when guidance is above
1.0; all three shipped entries run above that, so it is honoured on each, FLUX.2 included — that is
why the `klein-base` variant is the one that ships. **Structure control** (see [Conditioning on an
image](#conditioning-on-an-image)) is a ControlNet, and FLUX.2 cannot take one: it is the one real
refusal among the three, so picking FLUX.2 clears a chosen structure control and the Structure group
is replaced by a note. And **conditioning, seamless tiles and every style LoRA are SDXL-only**: they
are all built against SDXL's internals, so on FLUX.2 the Style LoRA picker is disabled with a note
saying so, and asking for a tile is refused rather than quietly producing one whose edges do not
line up.

Only one base model is resident on the card at a time — a 32 GB card holds the reconstruction engine
plus a single image pipeline, not two — so switching between models between jobs costs a reload of
several seconds. FLUX.2 is the exception to *how* it is held: it is large enough that it is streamed
onto the card one piece at a time rather than kept there whole, which makes each job slower but means
it still does not displace the reconstruction engine. Any model whose weights are not on disk is
still listed, with "— weights missing" appended to its name, so you learn at pick time rather than at
job-failure time. Run `realmspinner doctor` for the exact download command.

**Style LoRAs** are the opposite: they are adapters on whatever pipeline is already resident and
switch for free, with no reload. Four ship:

- **3D render** — a general 3D-render look.
- **3D render (Redmond)** — a second, differently trained take on the same idea.
- **PS1 / low-poly game** — chunky untextured geometry, which is among the easiest things to
  reconstruct cleanly.
- **Pixel art (pixel-art-xl)** — generates on a pixel grid rather than producing a smooth image that
  is later downscaled into one. It defaults to a strength of 1.2 rather than the usual 0.9: below
  that the output keeps SDXL's anti-aliased gradients, and no downscale recovers a clean grid from
  them. It runs on full-CFG SDXL.

An adapter is fitted to one architecture, so the picker offers a model only the styles that fit it.
Choosing a model no style fits disables the picker with a note; choosing one that some styles fit
lists those and says so. A style you picked under a different model stays visible and marked rather
than vanishing, and changing the model clears it with an explanation. That holds when the model is switched to
**Automatic** too: a style LoRA or start image the resolved model cannot take is cleared, with a note.

Choosing one reveals a **Strength** slider, from 0 to 1.5, defaulting to the LoRA's own tuned
weight — 0.9 unless the entry says otherwise. The slider is hidden entirely when no LoRA is chosen.
The SDXL LoRAs are trained against full SDXL at 20 to 25 steps with guidance, and both SDXL
recipes run the same way.

## Seeds and candidates

Generation is deterministic in its seed: the same form and the same seed produce the same image
every time. The **Seed** row, inside the collapsed **Advanced** card, is where you control that.

The **Images** control below the brief picks how many images one submit queues: 1, 2, 4 or 8.
(Mesh has its own, **Attempts**, for how many reconstructions to try.) Each is a real job holding
a place in the serial queue, which is why eight is the ceiling. Because each one gets its own seed,
a fan-out of four is the cheapest way to find out whether an idea works at all.

**Seed** is the number itself. **Reroll** replaces it with a fresh random one. **Lock** decides what
happens when you press Generate:

- Unlocked (the default): every Generate draws a fresh seed, so pressing Generate twice on an
  unchanged form gives you two different images rather than the same one twice.
- Locked: the seed is reused, so an unchanged form reproduces exactly. Lock it when you want to
  change one guidance field and see only that field's effect.

Seeds are whole numbers from 0 to 2147483647. The seed shown when the app opens is rolled fresh at
startup and is deliberately not remembered between sessions — otherwise every launch would open on
the same seed and a first Generate would reproduce last week's image.

Every image appears in the creation's **Attempts** strip under the canvas. Click a thumbnail
to inspect it. When a press of two or more images has finished, the canvas shows them all as a
**contact sheet** (two side by side, three or four in a square, five to eight in two rows of four),
scaled to fit. Click an image to choose it; the arrow keys move the choice, **Enter** is **Make 3D** on
it, and **Esc** (or **Single view**) goes back to one image at a time, which **Compare all** brings
back. The keys act only while the canvas has the focus and no text field is taking keystrokes. The
newest image to finish is chosen for you, unless you have chosen another one in the meantime. **G**
and **B** label the chosen image Good or Bad, and **N** opens its tags and note — see
[Labelling a reference](#labelling-a-reference).

Actions live in the inspector's **Next step**: for a finished 3D reference the one primary button is
**Make 3D** (also over the picture on the canvas, and on the card in the tray), which opens the
Mesh stage with that image chosen and checks the cutout. **More** holds **Load settings**,
**Try again**, **Reproduce recorded attempt**, pin and compare, **Export…** and **Discard**. These
actions preserve other attempts. While a job runs, the strip shows Working now, progress with the time
it is expected to take (*about 15 seconds left*, or the time so far once it has run longer than that),
and Cancel.

On a machine with no image model and nothing selected, the empty canvas is a setup card: **Get the
image model**, **Get the 3D engine** (each with its size, and each ticks its download in Settings) and
*or make a Character (no GPU needed)* — a Character needs the Rigging pack, which the card says.

The mesh has its own separate seed, at the Mesh stage. See
[Mesh parameters](23-generating-meshes.md#mesh-parameters).

### Labelling a reference

The same canvas is where you say which pictures were any good. With an image chosen — on the contact
sheet or on its own — **G** marks it **Good** and **B** marks it **Bad**, and **N** opens a small
popup for **tags** (the same good and bad words the mesh grade uses, such as *clean-shape* or *holes*)
and a **note** in your own words, with a **Save** button; the popup writes nothing until you press it.
Each picture on the sheet shows its label as a green or red edge with the word and its tags, and
the single view has **Good**, **Bad** and **Tags and note…** buttons at its top left. Pressing the
key for the label a picture already has files nothing new.

These are the same labels Review's labelling pass files, as you, so they feed the same
[findings](04-judging-what-you-made.md). The keys are plain letters and are read only while the
canvas has the focus and no text field is taking keystrokes, so typing a *g* in the prompt never
labels anything, and a chord with Ctrl, Alt or Shift is left to the app's own shortcuts.

### History

The **History** pane, under the inspector, is one card that lists everything you have generated or
judged, newest first — **including attempts you have since deleted**, which is the point of it. Each
row is a card of its own. A row shows a thumbnail
when the image still exists, the start of the prompt, the image label (*Image Good* or *Image Bad*)
and the mesh grade, the tags, *note* when you wrote one (hover the row to read it), and a line of what
ran: model, steps, guidance, style and seed. A muted *deleted* or *in trash* says the asset is gone,
and its labels, notes and settings stay. The line above the filters is the corpus summary Review
shows.

Four small menus, laid out two by two, filter the list: model, style, type, and
*All / Judged / Unjudged*. Their choices
come from the rows themselves, so a model that has since been removed from the app is still there to
filter by. **More** at the bottom adds another page. Clicking a row whose asset still exists opens it.

Each row's **Load settings** has a line of its own under the row's text. It copies the recorded
recipe back into the form — model (as an Advanced pick
unless Automatic would have chosen it anyway), style and its strength, the Avoid text, the seed
(locked) and the prompt, plus the mesh settings the attempt carried — and works on a deleted attempt
too, from the settings its verdict stored. It is greyed, with the reason, for a deleted attempt whose
verdict predates those settings being kept, and it is refused, with a toast naming the model, when the
attempt used an image model or style that this version removed: loading it would put a choice in the
form that nothing can run.

**Export…** in the pane's header writes the whole log — every verdict and measurement, with its
settings and your notes — to a JSON Lines file in a folder you pick, and says where it saved it. It is
the same export as **Settings → Advanced → Export generation log**.

### Which candidate to look at first

A fan-out of eight arrives in the order the seeds happened to be drawn in, which is no order at all.
Each finished reference therefore carries an **order hint**, shown as a bare percentage on its library
card and, on its card in Create's tray, only when you hover it ("Order hint N% - a sorting aid, not a
quality score"). It exists to answer one question: which of these is worth opening first. It is called
an *order hint* and nothing more, because that is all it claims — where this candidate sits in its own
strip, not whether it is any good, which is why the tray does not draw it as a number. (Through v0.0.38
the same figure was drawn as "judge: N% likely a keeper", which named the wrong instrument: the trained
probe in [Review](34-review.md) is a different thing and is never consulted here.)

Three things go into it, and each is absent-changes-nothing — a term that could not be measured
leaves the score exactly what it would have been without it:

- **Composition** — the framing report the reference stage takes anyway. It leads, because it is the
  one term that predicts whether the mesh stage can succeed at all: a subject cropped at the edge of
  the frame reconstructs badly however handsome it is. It counts *subjects* — a stray speck of noise
  beside the silhouette is not a second object. (Before v0.0.39 it counted every speck, which floored
  this term to zero on every real image and left the order hint deciding nothing.)
- **Style anchor** — how close the image looks to the reference image attached under *Conditioning*,
  when there is one.
- **Human preference** — how likely a person is to pick this image for this prompt, from PickScore.
  Optional; see [Optional measuring and helper
  models](40-installation.md#optional-measuring-and-helper-models).

**Nothing here rejects anything.** The score sorts, and that is all it does — a low-scoring
candidate is still generated, still kept, and still promotable to a mesh. Turn the whole thing off
with `REALMSPINNER_RANK=off` (see [Configuration](41-configuration.md#environment-variables)) and the
gallery falls back to submission order.

## Conditioning on an image

Beyond the prompt, you can steer the image with a picture. In the **Conditioning** section, either
**Choose an image...** or drop a file onto the window. Everything below the picker stays
hidden until there is an image, because a control with nothing to act on is a control that cannot
do anything.

With a reference loaded, two independent kinds of conditioning become available.

**Appearance** attaches an IP-Adapter — currently one entry, *Appearance reference (IP-Adapter
Plus)*. It conditions on sixteen patch tokens rather than a single pooled embedding, which is the
difference between "the same kind of object" and "this object". Its **Strength** runs from 0 to 1.5
and defaults to 0.6.

**Structure** attaches a ControlNet — currently one entry, *Edge / silhouette lock (Canny)*, which
traces the reference's edges and holds the new image to that shape. It has two controls.
**Strength** (0 to 2.0, default 0.65) is how hard it pulls. **Until** (0 to 1.0, default 0.8) is how
far into the drawing it keeps acting: ending early lets the last steps add detail the reference
never had, while 1.0 holds the shape to the final step and tends to look traced.

Structure needs a checkpoint that can take a ControlNet, which is either SDXL model and not FLUX.2
klein, and the section says so rather than offering a control that cannot work. Which fix it names
depends on which control chose the checkpoint: with a checkpoint named in the **Model** row it says
which of the others could run it, and under Automatic it tells you to pick one of those in the
Model row instead.

Clearing the reference clears both selections with it, since neither can be submitted without an
image.

### Starting from the image

Beside *appearance* and *structure* there is a third way to use the reference: **Start from this image
(img2img)**. The drawing then begins from the reference's pixels rather than from noise, and
**Strength** says how far it may go — at 0.30 the layout and most of the surface survive and the model
repaints the rest; at 0.65 only the gist does. It is the control for "this, but cleaner", "this, but in
the style LoRA", or "this, but with the prompt's changes", and it costs nothing extra on the card: the
start picture is fitted to between 512 and 1024 pixels on its long side (aspect kept) before drawing, so
a very large or very small reference is scaled into that range, and a structure lock combined with it
traces the picture at that same size. It needs an SDXL-family model, like the other two halves, and it
can be combined with either of them. The inspector records *start image* and *start strength* on the
job.

All four strengths above — Appearance, Structure's two, and this one — carry the same findings
machinery as the Model and Style LoRA pickers once enough verdicts exist behind a value: a muted line
saying what the current setting scored, and, when a different value has scored better, a **Use ...**
button that sets it exactly. See [Measuring instead of
guessing](11-tuning-what-you-get.md#measuring-instead-of-guessing).

## Approving a reference

When a reference job finishes it appears in the library and, when selected, fills the viewport at
full size. This is the moment the two-stage pipeline exists for: look at the picture before paying
for the mesh.

What to look for is what the reconstruction engine needs. One subject, complete and not cropped by
the frame, filling a good part of it, against a plain background. The app measures this itself and
records a reference report — if it thinks the image will not reconstruct, it says so in the
inspector's **Reference** section, and promoting it asks you to confirm rather than refusing
outright. The rules are heuristics about composition, and you can see the picture they are arguing
about.

If the image is nearly right, you can fix it by hand: **Open in Inker** on the viewport toolbar
opens the reference as a layered document, and saving writes it back in place. See
[Pipeline bridges](25-inker.md#pipeline-bridges).

When you are happy, press **Make 3D** on the card (or select the reference and step to **Mesh**).
That carries the reference and everything it recorded into the mesh stage, where you can override
the mesh-side settings before committing. The next chapter,
[Generating meshes](23-generating-meshes.md), picks up from there.

## 2D exports

A finished reference is an asset in its own right, not only an input to the mesh stage. Its
inspector's **Export** tab offers, alongside the source image:

| Button | File | What it is |
| --- | --- | --- |
| Icon | `icon.png` | A 512-square transparent cutout, centred with a small margin. |
| Sprite | `sprite.png` | The subject alone at native resolution, trimmed, with a recorded pivot. |
| Pixel art | `pixel_32.png`, `pixel_64.png`, `pixel_128.png` | Nearest-neighbour reductions, optionally palette-limited. |
| Manifest | `manifest.json` | What every artifact above is, measured — see below. |

All of them are cut from the same `input.png` the mesh stage uses, and all are derived the first
time you ask for one, then cached — the same rule the mesh exports follow. Editing the reference in
Inker, or rerolling it, makes every derived file stale, and the next request rebuilds it.

The icon is **fitted** inside its square rather than stretched to it, so a tall sword and a round
shield keep their proportions and an icon set stays readable. The sprite is not resized at all: it
records a bottom-centre pivot in the manifest, because that is where an engine puts a standing
character's feet, and an importer that guesses is wrong for half a set.

### Pixel art

The **Pixel art** section of the inspector's Details tab is where the pixel exports are set up and
previewed. **Size** picks which of the three artifacts you are looking at; **Colours** limits the
palette to 8, 16, 32 or 64, or leaves it off; **Palette** maps the export onto a palette file you
supplied, and **Dither** (offered only with one) mixes two nearby entries where a flat map would
pick one.

A palette is a file you drop into the palette directory (`~/.realmspinner/palettes/` by default — see
[Configuration](41-configuration.md)), in any of the four formats palette sites and editors publish:
Lospec's `.hex`, one `rrggbb` per line, GIMP's `.gpl`, Paint Shop Pro's `.pal` or Paint.NET's
`.txt`. Nothing ships with the app, because a palette is
art direction rather than a default. Colours are matched perceptually (in Oklab) rather than by raw
RGB arithmetic, which is what stops a dark grey being mapped to black and a whole shadow being
eaten. A palette file supersedes the **Colours** cap entirely: the cap is a median cut of the
subject's own colours — the discarded background is not counted, so a subject of four flat colours at
a cap of 4 keeps all four — and a palette is a decision about which colours exist.

Editing a palette in place re-derives every export that used it — freshness is keyed on the file's
*contents*, not its name, because editing one is the normal way to work on it.

The preview is drawn crisp, at a whole multiple of the artifact's own size — a fractional scale
samples some source pixels twice and others once, which reads as banding and is exactly what the
export avoids. A line underneath says what the file on disk actually is, read from the manifest
rather than from the controls, because the two can disagree: switching size shows a file cut under
an earlier palette setting, and a **Rebuild with these colours** button appears when that is the
case.

Two buttons sit under the controls. **Open in Inker** opens the pixels as a new, unlinked drawing:
the artifact is derived, so it is rebuilt whenever the settings above make the copy on disk stale,
and a document that saved back over it would have that edit thrown away — the first `Ctrl+S` is a
Save As. **Export as PNG** writes the size selected here to wherever you choose. Both derive the
artifact first if it does not exist yet, so neither waits on **Preview pixels**. The
[downloads grid](33-library-and-jobs.md) exports the same files; what these add is that they act on
the size you are looking at.

Both settings are app preferences rather than properties of the job, so they persist across
sessions and apply to whichever reference you are looking at. Changing them re-derives; it never
touches `input.png`, so promoting the reference afterwards feeds the mesh engine the same pixels it
always would have.

The reduction is nearest-neighbour, never a smooth resample: a filtered downscale puts a ramp of
in-between colours along every edge, and hard edges are the one property that makes the result read
as pixel art rather than as a small photograph.

There are two reductions, and which one runs is decided by the *image* rather than by a setting.
A pixel-art model draws logical pixels as square blocks — typically eight screen pixels across — and
that lattice has a phase: the first block rarely starts at the very edge of the frame. When one is
detected, the export takes one output pixel per block, sampled at the block's centre, across the
whole frame, and crops to the subject afterwards; the result is exactly the pixels the model
authored. Cropping first — which is what an ordinary export does — would move the origin off the
lattice by however wide the subject happens to be, and shear the art instead of reducing it. An
ordinary render has no lattice to find and takes the plain path unchanged, which is what every asset
already on disk was cut with. The provenance line says which happened. Palette reduction runs on colour only, with
transparency carried around it, so the cutout survives the quantization exactly.

If you are generating for the pixel exports, ask your prompt for flat shading and a bold
silhouette — the things that survive being made small — and avoid the words "pixel art": at 512 or
1024 the image model draws fake chunky pixels that then alias under the real reduction. The
pixel-art LoRA is the exception, because it authors a genuine pixel grid.

### The manifest

`manifest.json` is the sidecar an importer reads. It records, per artifact, the size, the trim box,
the pivot, the palette it was cut to, and which matte produced it — plus the recipe (seed, model,
prompt) behind the image itself. Per artifact, because a file on disk was cut by whatever was
installed when it was made, not by what is installed now.

If the reference was hand-edited in Inker, the manifest says so beside the recipe: the recipe names
a seed and a model, which after a hand edit is no longer the whole story of the pixels.

## Seamless tiles

Setting **the type combo** to *Seamless Material* switches the whole pane to a different kind of output.
A tile is a repeating texture rather than a subject: it is drawn with
wrapping convolutions, so its left edge continues into its right and its top into its bottom.

The rest of the form means the same thing for a tile as for an object: the prompt describes the
surface ("mossy cobblestone"), and the model, LoRA and negative prompt choose the machinery that
draws it.

A tile cannot be made into a mesh, and the app does not offer to: there is no subject to
reconstruct. It also cannot produce the cutout exports, because every one of them lifts a subject
off its background and a tile *is* background — an icon of one would be the whole frame with a matte
guessed over it.

### Whether it actually tiles

When a tile finishes, the app measures its own wrap seam and reports it in the inspector's **Seam**
section. The number is a ratio, and since 2026-08-30 the ratio it quotes is the wrap seam against
the *largest* step the picture already contains — the seam over its worst interior join. That makes
the threshold fixed by construction rather than fitted: at 1.00 the seam is exactly as large as the
hardest edge the texture has anyway, so at or below 1.00 the app says *likely seamless* and above it
says *visible seam*. Both directions are measured, because an image that wraps one way and not the
other is not a tile, and the worse of the two decides.

The verdict used to divide the seam by the picture's *average* grain, with a threshold of 3.5. That
statistic collapses on flat cells parted by thin hard lines — pixel art, grout, riveted panels —
where the average is near zero and the ratio inflates: on a held-out corpus it called 18 of 72
confirmed-seamless tiles seamed, 15 of them under the pixel-art LoRA the tileset track ships with,
where the new statistic called none. Tiles generated before that change keep the number they were
judged by, and the inspector words them in the old vocabulary ("edge/grain") rather than restating
them in the new one.

*Likely*, not *seamless*, and the hedge is measured: the statistic trades a miss for the false
alarms it removed. A picture whose interior already contains a step as hard as its seam ties at 1.00
and passes — 4 of 44 visibly seamed control units did. The wrapped view below is the real check.

A ratio is hard to calibrate against by eye, so the section also offers a wrapped view: the image
rolled by half, which puts what was the wrap edge through the middle of the frame where a
discontinuity is obvious. It is `wrap_preview.png`, a derived export like any other, and the
Export tab offers it too.

## Sheets

Two of the six asset types make several pictures from one prompt rather than one: **Tileset** and
**Sprite Sheet**. Each brings its own layout section above the model, because what the sheet is a
sheet *of* is the choice everything else follows from — including how long the press will take.

On the Materials layout two checkboxes sit under the list. **Keep one style across the list** draws
the first material on its own and uses it as the appearance reference for every one after it (loads
the IP-Adapter). **Erase the seam** adds one masked pass per material: the tile is rolled so its
wrap join runs through the middle, a band around the join is redrawn in place, and it is rolled back
— use it when the wrap preview shows a join.

### Tilesets

A tileset is a grid of related tiles — grass, path, water, cliffs, props — that share a palette, a
light direction and a style. That sharing is what separates a tileset from a folder of unrelated
pictures, and each of the three **Layout** entries buys it a different way.

| Layout | What it draws | Generations |
| --- | --- | --- |
| **Materials** | One seamless tile per surface you name, laid out eight across. | One per cell. |
| **Terrain set** | Two surfaces, composited into the forty-seven cases of a blob autotile. | Two. |
| **Grid (legacy)** | One 1024 px picture painted through a grid guide and cut into 64 cells. | One. |

*Materials* is the default, and it is the one that actually tiles. Type one surface per line, up to
sixteen lines, and **Draws of each** (1 to 4) says how many times each line is drawn — every draw is
its own full generation on its own seed, so lines × draws is the cell count and 64 is the ceiling.
The counter under the box does that arithmetic while you type, and the sheet is refused above the
ceiling naming both numbers rather than quietly trimming. Each cell is drawn on its own with
wrapping convolutions, exactly as a *Seamless Material* is, which is why a tile cut from this sheet
repeats without a join. The old grid did not: its cells came back as one scene cut up.

*Terrain set* asks for two surfaces instead of a list. **Inside** is what a stroke paints — the
islands, coastlines and peninsulas — and **Outside** is what surrounds it; both are generated, so
both have to be described or the sheet is refused. **Shared setting** is optional and is added to
both, so two separate generations come back sharing a world and a palette ("a temperate coastline").
It is not a description of the join: the join is computed from a coverage field, and drawn edge art
would be cut straight across by it. What comes out is a complete forty-seven-case blob set, and it
lands in **Plotter** with the Terrain tool already working — see
[Tools](29-plotter.md#tools).

*Grid (legacy)* is the original single-generation path. It stays for two reasons and the pane says
both: it is the only layout that offers a **View** other than top-down, and it is how a sheet made
under it is rerun. Its cells are drawn from one guide in which every cell is identical, so they tend
to come back as one scene cut up or as one tile repeated — measured on the
maintainer's own corpus.

**Tile size** lives in the **Tileset** section. **View** is drawn in the layout section
itself, and only by the grid — the other two would be a picker with nothing to pick. What each
offers depends on the layout:

| | Tile sizes | Views |
| --- | --- | --- |
| Materials, Terrain set | 16, 32, 64 | Top-down only |
| Grid (legacy) | 16, 32, 48, 64 | Top-down, 3/4, Isometric |

Both restrictions are arithmetic rather than policy, and switching layout migrates a setting that no
longer fits with a sentence saying why. A seamless material is drawn at 1024 px and reduced, so its
tile size has to divide 1024 exactly — which 48 does not. And a seamless tile has to wrap a square:
an isometric tile is a 2:1 diamond, and a 3/4 tile has a visible front face and cannot tile
vertically. If you want either view, you want the grid.

**3/4 needs subjects with height to show.** It is the same square lattice as top-down, and the whole
difference is what the model is asked to draw, so a sheet of flat flagstones comes back looking the
same either way. Ask it for walls, crates, fences, roofs, a well — anything with a front — and the
difference is unmistakable. This was measured rather than assumed, on the
maintainer's own corpus.

One palette is applied across the whole sheet in one pass, never per tile — quantized per tile, the
same moss comes out two different greens in two tiles. *Which* palette is yours to choose; see
**The pixel look** below. In the Materials and Terrain set layouts the words that are actually
generated are the ones you type in the layout section, and the prompt field in the brief
only names the sheet in the library. A **reference image**, if you attach one under
*Conditioning*, shapes the style but is never required. On the Tileset arm Conditioning offers only this reference
image: the appearance adapter, start-image and structure controls are not drawn there, and the header counts only the
reference.

A tileset cannot be made into a mesh, and offers no cutout exports, for the tile's reasons. To
paint with it, take it into **Plotter** or cut it up in **Packwright** — both read the sheet from
the library like any other asset.

### Sprite sheets

The *Sprite Sheet* asset type turns your prompt into a character and then into a sheet of it. The
**Sprite sheet** section asks two things.

**Action** is what the character is doing: *Turnaround (still views)*, or one of the animated
actions this installation has a pose guide for. Seven ship — idle, walk, run, attack, cast, hurt and
jump — but the menu is discovered from the guide files on disk rather than hardcoded, because a
guide is art and they land one at a time: an action with no guide behind it would be an unposed
figure conditioned on nothing, so it is not offered at all.

**Directions** is how many ways the character is drawn facing. **One direction is one generation**,
so this is the control that decides whether a press is one job or eight, and the line underneath
states the whole bill before you press it: cells, generations, drafts and roughly how long. Only the
direction counts with a guide are offered, which today means eight and only eight.

**The size is gated on the action, and this catches people out.** The pixel-art model spends about
eight generation pixels on one art pixel, and one direction is never split across two generations —
so all of an action's frames have to fit inside one 1024 px frame at eight times the cell size. At
32 px every action fits; at 48 px and 64 px only the four-frame ones do, which today means idle and
hurt. Ask for an eight-frame run at 64 px and it is refused naming both numbers — but you will
rarely see that, because the pane shortens the **Cell size** ladder first and says
which action shortened it.

A finished sheet carries **animation tags and per-frame durations**, one tag per direction —
`walk_front`, `walk_front_left` and so on — with the action's own frame time and whether it loops.
So a sheet arrives in Inker with its timeline already tagged and reaches an engine knowing its own
frame rate. A turnaround carries none, deliberately: four still views are not an animation, and
tagging them would put four one-frame loops in the timeline that mean nothing to play.

This asset type is two steps and shows as two rows in the library: the character is drawn first and
kept as its own asset, and candidate sheets are then imagined from it. A small sheet gets two to
pick between; past sixteen cells it gets one, because a second guess at eight bands costs more than
it is worth. The two steps are deliberate — a sheet you dislike still leaves you the drawing it was
made from, to reroll, edit or synthesise again with different settings. An attached reference shapes
the character.

The drafts appear under **Sprite sheet** in the inspector when the character is selected. See
[From a single drawing](#from-a-single-drawing) for what to do with them.

### From a single drawing

A finished **2D reference** can become a sprite sheet: the inspector's **Sprite sheet** header offers to
invent a sheet from the one drawing you have. It is a
different bargain and worth understanding before you press it. There is no object here — only a picture of
one — so the front view is the drawing and the other three are the image model's guess at what the
subject looks like from the side and the back.

That is why it produces **two candidates**, from two different seeds, side by side. You pick.
Nothing is chosen for you, nothing is overwritten, and drafts accumulate until you delete them. The
exception is size: past sixteen cells a sheet draws one candidate instead of two, because at that
point a second guess is another eight generations and costs more than it is worth.

**Type** is which sheet to draw, listed with the grid each one makes. Two are fixed atlases drawn in
a single generation: `turnaround` (a 2x2 grid: front, left, right, back) and `walk` (a 4x4 grid: one
row per direction, four frames of a walk cycle across). The rest are actions — idle, walk, run,
attack, cast, hurt, jump — each drawn one whole direction per generation, so an eight-direction run
is eight of them. The list is discovered from the pose guides on this installation's disk rather
than hardcoded, so an action with no guide behind it is simply absent, and one you add appears.

**Cell size** is the finished pixel size of one cell, 32, 48 or 64 — but only the sizes the chosen
type can actually be drawn at are offered. All three are available for the fixed atlases and for the
four-frame actions; a six- or eight-frame action fits one generation only at 32 px. **Palette** is
how many colours the whole sheet is reduced to. Each candidate has its own seed with a **Reroll**
beside it; the two must differ, or you would be asking for the same picture twice.

Three things make the result hang together rather than being four unrelated drawings:

- **One generation per candidate.** The whole atlas is drawn in a single 1024-pixel pass under a
  single seed, so there is one character wearing one shirt, not four.
- **Pose guides.** Each cell is drawn over a stick figure fed to the edge ControlNet, so where the
  limbs go is imposed rather than requested — and the walk rows get a real contact-passing cycle
  rather than four poses that happen to differ.
- **One palette and one baseline.** The colour reduction runs once across the whole atlas, and
  every cell's subject is moved onto a shared floor line. Feet that move between frames is what
  reads as a broken animation, far more than a slightly wrong arm does.

For a turnaround, your own drawing is pasted back into the front cell when its proportions match
what the model drew — so the one view that is definitely right is definitely right. The panel says
whether it was, and why not when it was not.

Under each candidate is a line per cell that came out doubtful: empty, running off the edge of its
cell, or far off the size of the rest of the sheet. These are notes, never refusals — a warning
costs you a sentence to read, and throwing a candidate away would leave you comparing one draft
against nothing.

**Open in Inker** opens that candidate as an animation, one frame per cell, and this is the point of
the whole feature: what arrives is editable, not final. Any sheet of an action arrives with a tag
per direction — `walk_front`, `walk_front_left` and so on — carrying that action's own frame
duration and whether it loops, so pressing Play loops one direction at a time at the speed it was
meant to run. A turnaround carries none, deliberately: four still views are not an animation, and
tagging them would put four one-frame loops in the timeline that mean nothing to play. The document
is unsaved and belongs to no
file — the first `Ctrl+S` asks where to put it, and the draft on disk is left alone. **Export
sheet** from there writes the atlas back out on the sheet's own fixed grid rather than wrapping it,
so a walk cycle stays four rows of four and each cell's sidecar entry carries the direction and the
frame. See [Inker: animation](26-inker-animation.md).

A finished synthesis also appears in the Library as its own row. Select it and the inspector shows
both candidates; **Open** — the button, the right-click item, or a double-click — takes you back to
this section on the reference it was drawn from, with that draft marked.

A synthesis is a queued job, so it waits its turn behind whatever else is generating. How much it
costs depends on the type: a fixed atlas is one full image generation per candidate, and an action
is one per direction per candidate — sixteen for a pair of eight-direction sheets, which is why a
sheet past sixteen cells draws one candidate rather than two. **Delete draft** removes a pair; a
reference keeps at most 50.

### The pixel look

In the type's own section, below the size controls and the working-resolution choice, both
sheet types offer the same two settings and the sprite sheet offers a third.

**Palette** maps every pixel to the nearest colour of a palette you authored, instead of to the
colours this particular render happened to contain. It is the single highest-leverage art input in
the program: a derived table is the average of whatever came back, which is where "muddy" comes
from, while a designed ramp is a decision. The picker appears once you have palette files
installed — `.hex`, `.gpl`, `.pal` or `.txt` in the palette folder (see
[Configuration](41-configuration.md)) — and also when the form names a palette that is no longer
there, listed and marked rather than silently reverting to *Derived from the render*. Naming the
same palette on each run is how a set of sheets is kept on one set of colours.

**Dither** adds an ordered 4×4 offset before each pixel picks its colour, so a gradient reads as a
texture rather than as a band. It works with or without a named palette: with none, the sheet still
derives its own table and dithers against that. How visible it is depends on how far apart the
colours are — a four-colour ramp dithers unmistakably and a sixty-four-colour one barely at all.

**Outline**, on the sprite sheet only, darkens the edge of each frame: *None*, *Inside* or *Around*.
*Inside* recolours the character's own edge pixels, so it cannot change the silhouette or the pivot;
*Around* grows the silhouette by a pixel. *Inside* is the default because a synthesised cell has no
guaranteed margin — the model filled the atlas as it liked, and a frame already touching its cell
edge has nowhere to grow into, so *Around* would clip it. There is deliberately no outline on a tile
sheet: a tile is opaque edge to edge,
so an outline finds the border of every cell and draws a grid line around all of them rather than
around anything in them.

## Characters

**Character** is the one asset type on this screen that generates no image at all. There is no
model, no LoRA, no seed-per-candidate and no reference: the body is built from a registry of
thirty-one species across four body plans, and it is rigged. The building is CPU work, which is why
the plan block says *no GPU needed* — on a machine that cannot run SDXL at all, this type still
works. (The rig needs Blender; see [What one press makes](#what-one-press-makes).)

Type a brief the way you would for anything else. The Character column reads it and fills itself in:
the species and the look move to what the words said. A hyphenated or
slashed spelling reads the same as a spaced one: `ogre-king` finds the species
exactly as `ogre king` does. Everything the
brief did not mention keeps its default, and anything it said that Realmspinner did not understand is
listed under **Not interpreted** so you can see what was ignored rather than wonder. That line also
names an action, a camera word and a sheet word: Realmspinner understands "a running wolf" or
"isometric view", but a character is a rigged mesh with no pose or camera of its own, so *run* and
*isometric view* are listed there instead of silently disappearing, and so are "sprite sheet" and
"8 directions", which belong to the other asset type. You animate the character afterwards in
[Clay](27-clay.md#bones-and-rigs).

Change a control and it becomes yours. From then on editing the prompt leaves that control alone —
otherwise every keystroke would undo the species you just picked. **Reset to prompt** forgets those
choices and reads the brief again.

### When it does not make that creature

Realmspinner never substitutes. If you ask for a phoenix, the prompt resolver does not quietly hand you a
dragon: the species stays empty, Generate is refused, and the refusal says exactly what happened —
*Realmspinner has no phoenix yet. The closest it makes is a dragon.* Under that sentence are the two
ways forward, and each of them is a press you make:

- **Make it a dragon** applies the offer. This is the only place in the program where a species you
  did not name becomes the one that is built, and it happens because you read the sentence and
  pressed the button. What is offered is always the same body plan — a winged thing is offered a
  winged thing, never a puddle.
- **Sprite sheet (experimental)** switches the asset type and leaves your brief alone. SDXL will
  draw things the registry does not model; see [Sheets](#sheets) for what that costs.

A brief that names no creature at all is refused in the same register, and the Species picker is
there to answer it directly.

### The controls

**Species** is grouped by body plan, because the plan decides the skeleton, the clips and which
appearance sliders this character has. **Look** offers only the palettes that species is painted in;
*The species' own* means its first one. A Look you picked is dropped when an edit to the brief moves
the species to one that is not painted in it, and a Look the species cannot be painted in is refused
on the Look control itself.

Below them is one slider per appearance channel the species' body plan declares — an ogre and a wolf
do not have the same ones — and a **Name**, which is what the library will call it.

A character has no Seed: the build leaves nothing to chance, so the same recipe always builds the
same character.

**Preview character** builds the body on its own and shows it in the viewport. It makes no library
row and never blocks Generate, so it is safe to press while you are still deciding.

### What one press makes

Generate builds the mesh, mints it as a finished asset, and queues its rig. Create moves to the Mesh
stage, because there is no reference image to look at — a character has no drawing behind it. When the
rig lands, the finished character is a rigged mesh like any other: open it in
[Clay](27-clay.md#bones-and-rigs) to pose it and animate it.

Rigging needs Blender. Without it the press is refused before anything is built, because a body with
no skeleton is half an asset and the whole ordering of that door exists to prevent one.
