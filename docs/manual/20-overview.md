# Overview

## What Realmspinner is

Realmspinner generates game-ready 3D assets on your own machine. You give it a text prompt or an
image; it gives you back a textured GLB — a base colour texture plus a combined
metallic/roughness texture, with surface detail carried on vertex normals rather than a normal map
— ready to import into Godot, Blender, Unity or Unreal.

Two models do the work. An image model (SDXL 1.0 by default) draws the reference picture from
your prompt. A reconstruction engine, Microsoft TRELLIS.2-4B running natively through
`trellis-server.exe`, turns that picture into a mesh. Both run on your GPU.

The app is **fully offline**. There is no provider API, no account, and nothing about your prompts
or your images ever leaves the machine. Three things go online, and only three, each its own
separate process that you start with a click: downloading model weights or the reconstruction
engine (Settings → Models), installing a dependency pack (Settings → Packs), and checking for a new
release (Settings → Updates). Outside those three, the app touches the network nowhere at all — a
missing weight tells you the exact command to fetch it rather than fetching anything itself. That
holds for Familiar too, the built-in assistant described under [The Familiar dock](#the-familiar-dock):
it is a language model that runs on this computer and talks only to the app, over a connection that
never leaves the machine.

It is also a single desktop window. There is no server to start, no browser tab, no `localhost`
address. Everything described in this manual happens in one process.

## The two-stage pipeline

Making a mesh is expensive — roughly two minutes of GPU per attempt — and the single biggest
factor in how good that mesh is turns out to be the picture it was made from. TRELLIS can only be
as good as the image it is handed.

So the pipeline is deliberately split in two, and the split is visible in the app:

1. **The reference stage.** A text job draws an image and stops. This takes a few seconds. The
   image is shown to you full size, and you can generate several candidates at once from different
   seeds before choosing one.
2. **The mesh stage.** Once you approve a reference, you promote it, and only then does the
   reconstruction run.

A text job never falls through to a mesh by accident: the Reference stage always submits with the
output set to `reference`. Going straight from a prompt to a mesh would spend two minutes of GPU on an
image nobody has looked at.

If you already have a picture, you can skip the first stage entirely and upload it — see
[Starting from an upload](23-generating-meshes.md#starting-from-an-upload).

## The modes

A rail down the left edge of the window chooses between thirteen modes, and that rail is the single
thing that decides what the panes show. It is drawn in every mode, so there is no screen you cannot
leave. There is no per-mode keyboard shortcut — the command palette (`Ctrl+K`) is the keyboard
route, see [Keyboard shortcuts](38-shortcuts.md).

The rail shows glyphs by default, in a column just wide enough for them, and expands to show the
labels beside them; **Window → Navigation labels** toggles that, and the choice is remembered.
Every mode carries a short purpose sentence saying what it is for; in icon-only form it names
itself and that sentence in a tooltip, and in the labelled form the sentence is a second, muted
line under the label when the row has room for it. A window too narrow to hold the labelled rail
*and* three usable columns draws the collapsed one until there is room again — what you chose and
what fits are two different facts, so dragging the window wider brings the labels back.

It is drawn in three sections, and the list below is in that order. The first is where an asset
**begins** — what you have, and making another one. The second is the **creative workspaces**. The
third is the **footer**, the last group in the column, carrying no caption: the two destinations
where you are not making something.

- **Home.** What the app opens on: what changed in this build, what the machine is doing, and a
  single list of everything you were recently working on. Returning here is never destructive.
- **Library.** Every asset that has ever been generated, filtered, sorted and searched, with the
  trash and the prune. Covered in [The library and jobs](36-library-and-jobs.md).
- **Create.** One workspace for an asset and its attempts. The header shows the stages relevant to
  the output; 3D assets have Reference, Mesh, Rig, Pose and Export, while image outputs have
  Reference and Export. The brief is on the left, previews and attempts in the centre, and the
  selected result and grouped creation history on the right. **Reference** owns the prompt and every
  control that composes it — the
  negative prompt, the image model and style LoRA, the seed and the candidate count.
  **Mesh** owns no prompt controls at all: a mesh job starts from a finished reference or from an
  uploaded image, and the column holds only the reconstruction decisions. **Rig** fits a skeleton,
  **Pose** edits one, and **Export** is what you can take away. A stage you cannot enter yet is
  drawn dimmed with the reason on hover rather than hidden. Covered in
  [Generating references](22-generating-references.md),
  [Generating meshes](23-generating-meshes.md) and
  [Rigging and posing](25-rigging-and-posing.md).

Then the eight workspaces:

- **Inker.** A layered raster editor, wired into the pipeline in both directions. Covered in
  [Inker](28-inker.md), with the timeline in [Inker: animation](29-inker-animation.md).
- **Clay.** Modelling from primitives: transforms, a material palette, and two ways out —
  export a `.glb` or import the document as an asset. Covered in [Clay](30-clay.md).
- **Mason.** A 3D scene editor: place library assets and primitives into a scene, group and
  duplicate them, light it, sculpt a ground, and export the arrangement as a glTF scene, an
  engine-friendly GLB-plus-manifest, or merged OBJ geometry.
- **Poser.** Authoring reusable poses against a skeleton template, kept in a global pose library
  rather than belonging to any one asset, plus (once a rigged asset is bound) a character-sheet
  section: a prompt becomes a reference, a mesh, a fitted rig and then a rendered, pixelised sprite
  sheet of the clips a character walks and swings through. The sheet keyframes are still
  provisional and the prompt-to-character half does not currently produce usable humanoids
  (measured 2026-08-30), so the route worth using there is a mesh you supply. Covered in
  [Poser](26-poser.md).
- **Plotter.** A tile-map editor: a grid, a layer stack, one or more tilesets, and the objects an
  engine reads as spawn points and trigger volumes — where a sheet of tiles becomes a level. It
  speaks Tiled's formats in both directions. Covered in [Plotter](32-plotter.md).
- **Packwright.** A sprite-atlas packer: many images in, one atlas out, with a sidecar that says
  where everything landed. Covered in [Packwright](33-packwright.md).
- **Muse.** Generated music: a comma-separated style-tag string and an optional lyric block become a
  finished track, one job row per take, auditioned in the mode and openable in Sirens as a sample
  instrument. Covered in [Muse](35-muse.md).
- **Sirens.** A chiptune tracker: NES-era pulse, triangle, noise and sample voices written into a
  pattern grid, stitched into a song by an order list, and saved as a `.rsng`. Instruments carry
  four envelope sequences you drag into shape, a `.wav` dropped on the window becomes a sample, and
  the whole thing exports as a mix, one WAV per channel and one per sound effect. Covered in
  [Sirens](34-sirens.md).

And in the footer:

- **Review.** Judging finished meshes — one at a time or as a parameter sweep — and the "what
  works" findings the verdicts add up to. Covered in [Review](37-review.md).
- **Settings.** The app's own preferences — UI scale, the frame-rate readout, layout resets, and the
  list of models it loaded, from which a missing one can be downloaded. See
  [In-app settings](41-configuration.md#in-app-settings).

This documentation used to be a mode and is not, for the reason nothing here is: it is *about* the
screen you are on rather than a place to go. It opens over the window (`F1`, or any pane's (?)
button) instead of replacing it, so the control you were asking about is still there when you have
the answer.

The search box above the contents list matches chapter titles, headings and body text, and lists the sections that
match. When the optional **Familiar retrieval (EmbeddingGemma 2)** row in Settings → Models is installed, a
**Semantic** checkbox appears under the box, off by default and remembered while the app is open. With it on, what
you typed is ranked by meaning as well as by matching words (the same ranking Familiar answers Manual questions
with) and the best sections are listed in place of the chapters; click one to open it at that heading. The first time
the Manual is searched this way its meaning index is built in the background, a few minutes of processor time that
is done once and kept until the Manual changes. Until it is ready the box says so in one line and keeps showing the
text matches, as it does while a query is being ranked. Without the row there is no checkbox at all.

The **guided tour** is a second overlay, for the same reason: it points at the controls of whatever
mode you are in, so taking that mode away to run it would leave nothing to point at. It never
clicks anything for you. Home offers it on a fresh install and the palette carries it thereafter —
see [New here?](21-home.md#new-here).

Each generation control belongs to exactly one stage. The one setting both Reference and Mesh need
is **platform**, and it is deliberately two separate controls: at the Reference stage it is a hint
that goes into the prompt ("how much fine detail should be drawn"), and at the Mesh stage it is the
geometry resolution sent to the reconstruction engine. One control cannot be owned by two stages,
so there are two.

Switching modes is never destructive. Inker keeps its open documents when you leave it, a queued
job keeps running whichever mode you are in, and the progress card floats over every mode but Home.

## The window

The app opens on Home, every launch, unless you turn on Settings ▸ Startup ▸ **Last workspace**
(Chapter 42): by default no mode is remembered between runs, because none of them is
what you want to be dropped into before you have said what you are doing. **Home**
is the first entry in the rail described above, and returns there at any time.

Once you are in the workspace, the window is five columns, left to right: the rail, the left
sidebar, the canvas, the right sidebar and the Familiar dock — and none of them is a size you drag.
The rail and the closed dock are icon strips, each just wide enough for its icons; the two sidebars
are 15% of the window each; the canvas takes everything left over. Opening the dock grows it to 15%
of the window, three points taken from each sidebar (so they become 12% each) and the rest from the
canvas. Neither sidebar has a drag handle of its own any more; a window too narrow for a column's
comfortable width compresses it instead, following a stated order (the sidebars give way to the open
dock first, then the canvas), so there is one width per mode rather than a per-workspace preference
to lose track of.

- **The left sidebar** is the settings form for the current mode, and nothing else — there is
  nothing left to split against, so it is one scrolling column with no divider. In Create a **stage
  rail** sits above it, naming the five steps an asset goes through and switching the column
  between them.
- **The middle column** is the viewport: the interactive 3D preview, or the reference image at the
  Reference stage, or the canvas in Inker mode. A small toolbar sits over it with the framing,
  wireframe and turntable toggles, and — on a finished reference at the Reference stage — the
  **Open in Inker** button.
- **The right column** is two stacked panels, not one scrolling column. The upper panel is the
  inspector: everything about the selected asset. In Create it carries no tabs — the stage rail is
  what switches it, so it shows the evidence for the stage you are on. Everywhere else it is three
  tabs, **Details**, **Rig & Pose** and **Export**. The lower panel is the asset library — every
  job you have ever run, with its filters. The divider between the two can still be dragged; that
  is a vertical split of the one column's own height, unrelated to the column's width.

Above the columns is the menu bar and to their right is the Familiar dock, and both are described next.

## What is the same in every workspace

Eight workspaces are eight editors, and they are deliberately one program eight times. Six of them
— **Inker**, **Clay**, **Mason**, **Plotter**, **Packwright** and **Sirens** — edit a saved
document, and those six share the file panel and the tab chords below. The other two
do not: **Poser** keeps a pose library and **Muse** produces a job row in the Library, so neither has
a file panel, the four file verbs or document tabs, and `Ctrl+Tab` and `Ctrl+W` have nothing to cycle
or close there. Everything else below holds in all eight.

- **The file panel** of a document workspace, in the right column, carries the same four verbs —
  **New**, **Open**, **Save**, **Save As** — over one sentence saying where the document stands, then
  **Undo** and **Redo** with the step count, which is a button onto the history when the mode has
  one. Under **Take it somewhere** are the ways out of the mode: the library, and whichever
  workspaces read what this one makes. Each panel has exactly one accented button, and it is the
  mode's own commit — export to the library, send to Poser, export the audio.
  **Clay is the exception**: it has no separate file pane. The same verbs, its exports and **Undo** and
  **Redo** are in its **File** menu, and Undo and Redo are also buttons in its header.
- **The same gestures.** The wheel zooms, in 5% steps, in every canvas; `Shift` and the wheel scrolls
  sideways; the middle button pans. In a 3D view, `Alt`+drag orbits and the middle button pans.
  `Ctrl+1`, `Ctrl+3` and `Ctrl+7` look along an axis and `Ctrl+5` toggles perspective, in Clay, in
  Poser and in Mason alike.
- **The same chords.** `Ctrl+S` and `Ctrl+Shift+S` save; `Ctrl+Z`, `Ctrl+Y` and `Ctrl+Shift+Z` walk
  the history; in the six document workspaces `Ctrl+Tab` and `Ctrl+Shift+Tab` cycle the tabs and
  `Ctrl+W` closes one; `Ctrl+Shift+E` is the file export and `Ctrl+E` the library export wherever
  each exists. A chord is printed in a
  control's tooltip and in the menu's right-hand column, never in a button's label.
- **The same words.** **Delete** destroys a thing and wears the trash glyph; **Remove** takes it out
  of this document and leaves it on disk; **Clear** empties a field. **Play** and **Stop** are the
  transport everywhere, over the play and stop glyphs, with `Space` in the tooltip where a mode
  binds it. A label ending in an ellipsis opens a dialog; one without does not.
- **The same refusals.** A gesture the document cannot take right now says so once, as a toast with
  its remedy where one exists, and a tab whose save is still writing says so when you try to close
  it rather than closing anyway. A crash copy that will not reopen warns, with the log behind it, in
  the same sentence in every mode.
- **The same file rules.** A file dropped on a mode that opens no files is refused with a sentence
  naming what that mode works on, rather than switched away from. Opening a file already open
  focuses its tab rather than opening it twice, whatever the spelling of its path.

## The menu bar

One menu bar across the top of the window, drawn in every mode. Its roots are **File**, **Edit**,
**View**, **Workspace**, **Window** and **Help** for every mode — no mode gets a menu root of its
own name. Clay's, Plotter's, Mason's and every other mode's actions are folded into File, Edit and
View alongside the commands every mode already carries, rather than filed under the mode's own
name. Inker is the one exception: between Edit and View it contributes six mode-specific roots —
**Sprite**, **Layer**, **Frame**, **Select**, **Sheet** and **Flourish** — and it also adds rows to
File, Edit and View.

**Nothing in the menu is a second implementation of anything.** Every row is an adapter over the
same command registry the palette searches and the same operation registry the keys dispatch
through, so the menu, `Ctrl+K` and the keyboard cannot disagree about what an action does, whether
it is available, or why it is not. A row you cannot use is greyed with the reason on hover — the
same reason the palette gives — and a row with a keyboard binding prints it on the right.

**Workspace** is the one to know about: it holds all thirteen modes, so it is a third way — beside
the rail and the palette — to change what the window is showing.

## The status group

A right-aligned group at the far end of the same bar carries the app's status readouts: the
workspace you are in, the open document and whether it has unsaved changes, the current tool and
zoom in any workspace that has them (Inker, Plotter and Packwright zoom; Inker and Plotter have
tools), the queue when anything is running or waiting, an **Agent connected** chip while an MCP
client is attached (Settings → Agent), and an amber **N issue(s)** when a startup
check has failed. That figure is a report rather than a control — for the list behind it, go to
**Settings → Health**, which is also where you look when nothing is failing and there is no count
in the group at all. There is no green "all well" state, because a healthy install has nothing to
report. When the menus a workspace needs leave the group no room, items drop lowest-priority-first
— the resource meter, then zoom, then tool, then document, then queue — but the health figure
never drops, whatever the window's width. Every item in the group is a readout, not a button: none
of it is clickable.

Beside the status group, and likewise never dropped, sits **✦ Familiar**. Once Familiar's weights are
installed its menu holds one row, a checked **Show Familiar** toggle, which opens and closes the
Familiar dock on the window's right edge; until then it stays a disabled **Not installed**.

## The Familiar dock

A full-height dock on the window's right edge, from the menu bar to the bottom of the window, outside
every workspace's own columns — the rail's mirror image on the opposite side. Closed, it is a slim strip
the same width as the collapsed rail, with one **✦** button; muted when Familiar's weights are
not yet downloaded, with a tooltip pointing at Settings → Models, or a toggle once they are present:
press **✦** to open the dock and press it again to close it. Open, a centred header reads **✦ Familiar**
beside a **✕** that also collapses it, above
a short conversation: a scrollback of what you and Familiar have said, an input line, and **Send**.
The input line and **Send** stay pinned to the bottom of the dock; only the conversation above them
scrolls.
Opening it grows the dock to 15% of the window's width, three points taken from each of the two
sidebars and the rest from the canvas — there is no grip to drag any more, and nothing to remember:
the width is always that same share. On a narrow window the dock's own floor (260 px) is met by the
two sidebars giving up width first, then the canvas. The per-item readouts that used to sit
in a pane at the foot of the window (workspace, document, tool, zoom, queue, health) live in the menu
bar's own right-aligned group, described above.

**Familiar is a language model that runs on this computer and never calls out.** It is Gemma 4 12B,
started on demand as a local server that listens on this machine only and is told not to go online; its
weights are an ordinary download in Settings → Models, under *Familiar*. It makes no network connection
of its own, so the three things that go online (see [What Realmspinner is](#what-realmspinner-is)) stay
the same three. A message you send is read by this one model and by nothing else, and a picture you
attach never leaves the machine.

**Familiar never shares the graphics card with a job.** A 12-billion-parameter model and a 3D
reconstruction do not both fit a 32 GB card, so the moment a 3D, image or music job is about to start,
Familiar is stopped to make room. Nothing is lost: your conversation is kept, and the next message you
send starts Familiar again, which takes a moment to load. If the card is too full for it to start,
Familiar says so plainly rather than failing quietly.

Familiar reads a sent message before answering it: a short router decision picks what the message is actually asking
for — build something in Clay, edit what's already there, a question about Realmspinner itself, or just
conversation — and answers accordingly, without you having to say which. A question about Realmspinner
(**"how do I export a GLB"**, **"what does the band setting do"**) is answered from the Manual itself, with a
small **[1]**, **[2]**… link under the reply
for each section it actually used; clicking one opens the Manual at that section. If the Manual has nothing on the
question, Familiar says so plainly rather than guessing. Finding those sections is two rankings fused into one:
the keyword ranking the Manual's own search uses and — when the optional **Familiar retrieval
(EmbeddingGemma 2)** row in Settings → Models is installed — a ranking by meaning, so a question that never uses
the Manual's own words still finds the section that answers it. That retrieval model runs on the processor, not
the card, so it is never stopped to make room for a job. Its index of the Manual is built in the background the
first time a question needs it, a few minutes once and then kept until the Manual changes, and the questions asked
meanwhile are answered by the keyword ranking. Without it nothing breaks: the same questions are
answered by the keyword ranking alone, a Library search stays a plain text match, and `realmspinner doctor` lists
the row as `pending_install` (not on disk) rather than as a fault. Familiar can also take you somewhere — say
"open Mason" or "take me to Settings" and
it switches modes, opens the right Settings page, opens the Manual, a tour, the keyboard
shortcuts list, the workspace layout picker or the trash, whichever you asked for — and it can draft
a brief in **Create**: say "make me a reference image of a lantern" and it fills in the asset type
and prompt and takes you to Create's Reference stage, exactly where typing it yourself would have
left you. It never presses **Generate** for you; you check the
brief and do that yourself. Neither door goes anywhere or does anything you could not already reach by hand — a mode
that is not ready yet says why, the same sentence its greyed rail item shows, and a request to draft in Create when
Create itself is not ready says that instead of opening a form that could not generate anything. Ask Familiar for a
character — **"make me a goblin in the swamp"** — and instead of a brief it shows a plan: the species, the theme,
whichever movements, direction count or name you actually asked for, and a time estimate, with **Create**, **Open in
Create** and **Discard** underneath. **Create** queues the character exactly the way pressing Create's own Generate
button would; **Open in Create** drops the same plan into Create's form instead, at the Reference stage, for you to
adjust and generate yourself; **Discard** drops it. Nothing is queued until you press Create — proposing a plan never
mints anything by itself, the same "look before you build" contract a Clay ghost keeps. A word the prompt used that
the plan could not act on (a look the species does not offer, say) is named under the plan rather than silently
dropped. A plan that arrives while a Clay ghost is still waiting is kept, not shown: the ghost's **Apply** and
**Discard** stay the only buttons, and the plan's card appears once you have pressed one of them.

**Clay Build is off in this release.** In **Clay**, with a document open, the expanded pane also offers
**Build**: describe what to add and Familiar proposes it as a translucent ghost over your document, with
**Apply** and **Discard** beside it once it lands — the same ghost a Send message routed to a Clay build
lands as, if the router decides that is what you meant. Building needs Realmspinner's own fine-tune of the model
(`familiar_v1.0`), which has not been published, so with the stock Gemma 4 12B weights a Clay build, however it was
asked for, answers with a plain sentence saying so rather than a ghost. What follows describes how Build behaves
once that model ships. The input stays open while a ghost is showing: a follow-up ("make it taller") refines
the ghost rather than the document, and **Apply** lands the build and every refinement as one undo step.
However a build finishes — a ghost ready to apply, or a refusal — Familiar says so as a line in the conversation and
a toast, so you can tell it is done even with the pane collapsed; **Apply** and **Discard** answer the same way, with
**Discard** skipping the toast. Each document tab keeps its own conversation, the same way it keeps its own undo
stack — closing a tab ends its thread, and every other mode without a document of its own shares one
Realmspinner-wide thread.

A build proposal that gets turned away — an unreadable reply, a tool name outside what the model was trained to use,
or the door that actually tries the proposal against your document refusing it — is not shown to you as a dead end
straight away: Familiar quietly retries, up to twice, showing each attempt's own refusal as its own line in the
conversation (**"Retrying after a refusal: …"**) before either a corrected ghost lands or the refusal is shown exactly
as it always has been once both retries are spent. None of this blocks the app — every retry, like the build itself,
runs off to the side while you keep working.

With Familiar's vision weights installed (Settings → Models → **Familiar vision (mmproj)**, optional — everything
above works without it), you can attach a PNG to a message: type or paste its path into the attach field beside the
input line and it rides along with your next Send or Build. Revising a Clay ghost sends its own current render
automatically, with nothing to attach by hand, so Familiar can see what you are asking it to change rather than only
reading the scene's own numbers. An attach you typed yourself always wins over the automatic ghost render. Familiar
runs entirely on this machine either way — an attached picture never leaves it.

The keyboard shortcut list is `Ctrl+/`, **Help → Keyboard shortcuts**, or **Keyboard shortcuts** in
the command palette, and it is reproduced in [Keyboard shortcuts](38-shortcuts.md).
