# Tuning what you get

[Your first asset](02-your-first-asset.md) used defaults for everything. This chapter is about the
controls in Create's recipe column, what each is for, and which of them are worth your attention.

The short version: prompt and seed do most of the work, and the conditioning controls are the ones
that reward learning properly.

## Seeds

A seed makes generation repeatable. Same prompt, same settings, same seed, same picture.

Next to the seed field are **Reroll**, which picks a fresh random one, and **Lock seed**. Unlocked —
the default — every Generate draws a fresh seed, which is what you want while exploring. Locked, the seed is
reused, which is what you want when you are changing *one other thing* and need the comparison to
mean something.

That is the whole discipline: unlock while hunting, lock while comparing.

A result's **Load settings** (and the Library's **Copy settings to form**) loads its recorded seed and locks it, so
the first Generate after you change one thing reuses that seed instead of rolling a new one; a mesh's **Load
settings** does the same with the mesh seed. A result that recorded no seed leaves your seed alone.

The segmented 1 / 2 / 4 / 8 **Images** control under the brief asks for several images at once from
different seeds; each value carries its own tooltip. Generating four and picking
one is usually a better use of the same time than generating one and rerolling it four times,
because you see the spread rather than a sequence.

The mesh stage has its own separate **Seed** (same label, its own field), and its own **Attempts**
control offering up to three
reconstruction attempts to pick the best from. Mesh geometry varies a great deal between seeds — more
than most people expect — so this is often the more valuable of the two.

## Models and LoRAs

One dropdown decides this. Its first entry is *Automatic* — the app resolves a checkpoint and says
which underneath — and every other entry is an installed checkpoint you name yourself. Three ship:
SDXL 1.0 at full CFG, SDXL 1.0 with PAG, and FLUX.2 klein base. See
[Models and style LoRAs](22-generating-references.md#models-and-style-loras).

The default base model is SDXL 1.0 run at **full CFG**, and it is the default because it measured
better than the alternatives as a source for reconstruction, not because it is fastest. The PAG
recipe is the same weights run with two sampler upgrades and costs nothing extra; FLUX.2 klein is a
separate checkpoint. All three run real classifier-free guidance, so the **Avoid** box is drawn and
honoured on each of them.

The one thing a model refuses is a structure control: FLUX.2 klein cannot take a ControlNet. Picking
it clears the structure control, rather than leaving you a Generate button refusing on a hidden field,
and the Structure group is replaced by a note until you pick an SDXL model again. The appearance
reference, style LoRAs, Start from this image and seamless tiles are SDXL-only for the same reason.

Style LoRAs are filtered to those that fit the chosen base's architecture, so you cannot pick an
incompatible pair.

**Leave the style strength alone unless you have a reason.** Selecting a LoRA seeds its own tuned
weight, and those weights are not decorative — they were measured per adapter. One of them runs at
0.0625, because it was trained in a way that makes peft restore a sixteen-fold scale internally; at a
"normal" 0.9 it produces black frames. A flat default across all LoRAs would be wrong for most of
them.

## Style LoRAs trained from the library

Settings' **Train from my library...** button (see [App settings](42-app-settings.md#your-style-loras))
builds a training set out of work you have already judged — favourited jobs, references you
accepted, and the reference images behind meshes that graded usable or better — instead of a folder
you point it at by hand. It is a convenience for gathering images, nothing more.

**Say this plainly: training a style on pictures this same build generated is unmeasured.** Nothing
in this app has run the comparison that would actually tell you whether it helps — generate the same
prompts with the trained style on and with it off, judge the pairs blind, and see whether the
verdicts differ. Until that sitting happens, "trains a working style" and "teaches the model to
repeat its own habits back to itself, indistinguishably to you" are both live possibilities, and nobody
has looked. That measurement is a future sitting, not a condition this feature had to clear before
shipping — the button assembles a training set faithfully; it does not, and cannot yet, tell you the
result is good.

## Conditioning on an image

Two named ways to hand the model a picture, and they do different jobs.

**Appearance** takes the *look* of a reference — colour, material, feel — without copying its shape.

**Structure** takes the *shape*. It is offered on the two SDXL models, because it is a ControlNet and
FLUX.2 klein cannot take one. It has a strength and an **until** value, the latter being how far
through sampling the guidance keeps applying — releasing it early gives the model room to add detail
the control image did not have.

Both need a reference image loaded, and both refuse clearly if you have not.

## The negative prompt

The negative prompt is only encoded when the base model runs guidance above 1.0. All three models
that ship run well above it (7.0 on both SDXL recipes, 4.0 on FLUX.2 klein), so the **Avoid** box is
always drawn and always read.

## Mesh-side settings

The Mesh column shows three things: **Triangle budget**, **Size** and **Rig**. **Size** sets the
real-world scale in metres, where zero means "keep whatever the reference implied". Everything else is
in the **Advanced** card below them. **Mesh resolution** sets the reconstruction's geometry
resolution. **Background** chooses the matting method (Automatic, BiRefNet cutout or a
plain-background threshold). **Centre and enlarge the reference** recentres and rescales the subject
before upload.

**Triangle budget** is the Mesh column's control for how the finished mesh reaches a game-sized
triangle count. It offers Game-ready rungs (2k, 5k, 10k and 20k triangles, with 5k the default), the
gltfpack "Simplify" tiers, Custom... and Raw, each drawn only when this machine can run it. Every pass is
checked per job rather than qualified in advance. [Generating meshes](23-generating-meshes.md)
describes each family and what it needs.

## Re-texturing

A finished mesh can be repainted. Describe the surface you want, and ten sampling passes, one per rendered view,
produce a new texture on the same geometry.

Two settings matter. **Restyle strength** — from 0.30 to 0.85, defaulting to 0.65 — is how far it
departs from what is there. And **Anchor to geometry (depth)**, which is **on by default and should stay on**: it
depth-tests each texel so colour cannot smear through an overhang onto a surface that has no line of
sight to it. Turning it off was measured as smearing colour into essentially every hidden region;
with it on, almost none.

Re-texturing invalidates only the exports that carry the skin. Your rig, poses and sheets survive.

## One thing to watch out for

There is a known defect worth knowing about while you are experimenting with base models: loading an
image model does not return all of its host memory when it unloads. Switch base models repeatedly in
one long session and system commit climbs until the app starts refusing its own work.

It is not subtle when it happens — the refusals say so. The fix is to restart the app, which people
tend to do unprompted anyway. It is on the list.

## Measuring instead of guessing

Everything above changes what comes out, and it is very easy to convince yourself that a setting
helped. If it matters, the app has the machinery to check: launch a sweep varying one axis, judge the
results blind, and read the Axis verdicts.
[Judging what you made](04-judging-what-you-made.md#sweeps-and-what-verdicts-add-up-to) covers it.

Once enough verdicts exist, most controls on this pane and the mesh column — the appearance and
structure strengths, the img2img strength, the model and style pickers, the mesh settings — carry a
small muted line under them saying what the value they are currently set to scored: something like
"usable 6/8 (41%+) · avg +2.6". When a *different* value has scored better, a second line appears
under it naming that value and a **Use ...** button beside it — pressing it sets the control to
exactly that value, in the type the control already holds. It never fires on its own; the app never
picks a setting for you, only offers the click once the evidence says something. Nothing appears
while a value is already the one the evidence favours, or while too few verdicts exist behind any of
them.

## What to read next

[Putting it in a game](12-putting-it-in-a-game.md) — exports, formats and the honest state of
interoperability with other tools.
