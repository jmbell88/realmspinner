# Overview

## What Warlock Studio is

Warlock Studio generates game-ready 3D assets on your own machine. You give it a text prompt or an
image; it gives you back a textured GLB — a base colour texture plus a combined
metallic/roughness texture, with surface detail carried on vertex normals rather than a normal map
— ready to import into Godot, Blender, Unity or Unreal.

Two models do the work. An image model (SDXL 1.0 by default) draws the reference picture from
your prompt. A reconstruction engine, Microsoft TRELLIS.2-4B running natively through
`trellis-server.exe`, turns that picture into a mesh. Both run on your GPU.

The app is **fully offline**. Model weights are downloaded once, by hand, before you start; after
that Warlock Studio never touches the network. There is no provider API, no account, no upload of
your prompts or your images. If a set of weights is missing, the app tells you the exact command
to fetch it rather than fetching anything itself.

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

A rail down the left edge of the window chooses between fourteen modes, and that rail is the single
thing that decides what the panes show. It is drawn in every mode, so there is no screen you cannot
leave. There is no per-mode keyboard shortcut — the command palette (`Ctrl+K`) is the keyboard
route, see [Keyboard shortcuts](39-shortcuts.md).

The rail shows glyphs by default and expands to show the labels beside them; **Window → Navigation
labels** toggles that, and the choice is remembered. Every mode carries a short purpose sentence
saying what it is for; in icon-only form it names itself and that sentence in a tooltip, and in
the labelled form the sentence is a second, muted line under the label when the row has room for
it. A window too narrow to hold the labelled rail *and* three usable columns draws the collapsed one
until there is room again — what you chose and what fits are two different facts, so dragging the
window wider brings the labels back.

It is drawn in three sections, and the list below is in that order. The first is where an asset
**begins** — what you have, and making another one. The second is the **creative workspaces**. The
third is the **footer**, the last group in the column, carrying no caption: the two destinations
where you are not making something.

- **Home.** What the app opens on: what changed in this build, what the machine is doing, and a
  single list of everything you were recently working on. Returning here is never destructive.
- **Library.** Every asset that has ever been generated, filtered, sorted and searched, with the
  trash and the prune. Covered in [The library and jobs](37-library-and-jobs.md).
- **Create.** One mode for the whole asset pipeline, drawn as five **stages** on a rail above the
  settings column. **Reference** owns the prompt and every control that composes it — the
  negative prompt, the image model and style LoRA, the seed and the candidate count.
  **Mesh** owns no prompt controls at all: a mesh job starts from a finished reference or from an
  uploaded image, and the column holds only the reconstruction decisions. **Rig** fits a skeleton,
  **Pose** edits one, and **Export** is what you can take away. A stage you cannot enter yet is
  drawn dimmed with the reason on hover rather than hidden. Covered in
  [Generating references](22-generating-references.md),
  [Generating meshes](23-generating-meshes.md) and
  [Rigging and posing](25-rigging-and-posing.md).

Then the nine workspaces:

- **Inker.** A layered raster editor, wired into the pipeline in both directions. Covered in
  [Inker](28-inker.md), with the timeline in [Inker: animation](29-inker-animation.md).
- **Clay.** Modelling from primitives: transforms, a material palette, and two ways out —
  export a `.glb` or import the document as an asset. Covered in [Clay](30-clay.md).
- **Mason.** A 3D scene editor: place library assets and primitives into a scene, group and
  duplicate them, light it, sculpt a ground, and export the arrangement as a glTF scene, an
  engine-friendly GLB-plus-manifest, or merged OBJ geometry.
- **Poser.** Authoring reusable poses against a skeleton template, kept in a global pose library
  rather than belonging to any one asset. Covered in [Poser](26-poser.md).
- **Troupe.** A character-sprite factory: a prompt becomes a reference, a mesh, a fitted rig and
  then a rendered, pixelised sprite sheet of the clips a character walks and swings through.
  Experimental — the chain runs end to end, but the shipped keyframes are provisional and the
  prompt-to-character half does not currently produce usable humanoids (measured 2026-08-30), so
  the route worth using is a mesh you supply. Covered in [Troupe](34-troupe.md).
- **Plotter.** A tile-map editor: a grid, a layer stack, one or more tilesets, and the objects an
  engine reads as spawn points and trigger volumes — where a sheet of tiles becomes a level. It
  speaks Tiled's formats in both directions. Covered in [Plotter](32-plotter.md).
- **Packwright.** A sprite-atlas packer: many images in, one atlas out, with a sidecar that says
  where everything landed. Covered in [Packwright](33-packwright.md).
- **Muse.** Generated music: a comma-separated style-tag string and an optional lyric block become a
  finished track, one job row per take, auditioned in the mode and openable in Sirens as a sample
  instrument. Covered in [Muse](36-muse.md).
- **Sirens.** A chiptune tracker: NES-era pulse, triangle, noise and sample voices written into a
  pattern grid, stitched into a song by an order list, and saved as a `.wsng`. Instruments carry
  four envelope sequences you drag into shape, a `.wav` dropped on the window becomes a sample, and
  the whole thing exports as a mix, one WAV per channel and one per sound effect. Covered in
  [Sirens](35-sirens.md).

And in the footer:

- **Review.** Judging finished meshes — one at a time or as a parameter sweep — and the "what
  works" findings the verdicts add up to. Covered in [Review](38-review.md).
- **Settings.** The app's own preferences — UI scale, the frame-rate readout, layout resets, and the
  list of models it loaded, from which a missing one can be downloaded. See
  [In-app settings](41-configuration.md#in-app-settings).

This documentation used to be a mode and is not, for the reason nothing here is: it is *about* the
screen you are on rather than a place to go. It opens over the window (`F1`, or any pane's (?)
button) instead of replacing it, so the control you were asking about is still there when you have
the answer.

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

The app opens on Home, every launch: no mode is remembered between runs, because none of them is
what you want to be dropped into before you have said what you are doing. **Home**
is the first entry in the rail described above, and returns there at any time.

Once you are in the workspace, the window is three columns:

- **The left sidebar** is the settings form for the current mode, and nothing else — there is
  nothing left to split against, so it is one scrolling column with no divider. In Create a **stage
  rail** sits above it, naming the five steps an asset goes through and switching the column
  between them. Its width is not draggable; it is one of three named sizes chosen in Settings.
- **The middle column** is the viewport: the interactive 3D preview, or the reference image at the
  Reference stage, or the canvas in Inker mode. A small toolbar sits over it with the framing,
  wireframe and turntable toggles, and — on a finished reference at the Reference stage — the
  **Open in Inker** button.
- **The right column** is two stacked panels, not one scrolling column. The upper panel is the
  inspector: everything about the selected asset. In Create it carries no tabs — the stage rail is
  what switches it, so it shows the evidence for the stage you are on. Everywhere else it is three
  tabs, **Details**, **Rig & Pose** and **Export**. The lower panel is the asset library — every
  job you have ever run, with its filters. The divider between them can be dragged; the sidebar's
  own width is not draggable, only chosen from the three named sizes in Settings.

Above the columns is the menu bar and below them is the bottom pane, and both are described next.

## What is the same in every workspace

Nine workspaces are nine editors, and they are deliberately one program nine times. Whichever
one is open:

- **The file panel** in the right column carries the same four verbs — **New**, **Open**, **Save**,
  **Save As** — over one sentence saying where the document stands, then **Undo** and **Redo** with
  the step count, which is a button onto the history when the mode has one. Under **Take it
  somewhere** are the ways out of the mode: the library, and whichever workspaces read what this one
  makes. Each panel has exactly one accented button, and it is the mode's own commit — export to the
  library, send to Troupe, export the audio.
- **The same gestures.** The wheel zooms, in 5% steps, in every canvas; `Shift` and the wheel scrolls
  sideways; the middle button pans. In a 3D view, `Alt`+drag orbits and the middle button pans.
  `Ctrl+1`, `Ctrl+3` and `Ctrl+7` look along an axis and `Ctrl+5` toggles perspective, in Clay, in
  Poser and in Mason alike.
- **The same chords.** `Ctrl+S` and `Ctrl+Shift+S` save; `Ctrl+Z`, `Ctrl+Y` and `Ctrl+Shift+Z` walk
  the history; `Ctrl+Tab` and `Ctrl+Shift+Tab` cycle the tabs; `Ctrl+W` closes one; `Ctrl+Shift+E`
  is the file export and `Ctrl+E` the library export wherever each exists. A chord is printed in a
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
**View**, **Workspace**, **Window** and **Help**, and between Edit and View sits whatever the
current workspace contributes. For most of them that is a single menu under the mode's own name —
*Clay*, *Plotter*, *Troupe* — holding the actions that belong to that mode alone. Inker, which has
far more of them, contributes several: **Sprite**, **Layer**, **Frame** and **Select**, and it adds
rows to File, Edit and View as well. Either way a mode's actions get their own place rather than
being filed into File or Edit, which would turn the two menus everybody already understands into a
list of everything.

**Nothing in the menu is a second implementation of anything.** Every row is an adapter over the
same command registry the palette searches and the same operation registry the keys dispatch
through, so the menu, `Ctrl+K` and the keyboard cannot disagree about what an action does, whether
it is available, or why it is not. A row you cannot use is greyed with the reason on hover — the
same reason the palette gives — and a row with a keyboard binding prints it on the right.

**Workspace** is the one to know about: it holds all fourteen modes, so it is a third way — beside
the rail and the palette — to change what the window is showing.

## The status group

A right-aligned group at the far end of the same bar carries the app's status readouts: the
workspace you are in, the open document and whether it has unsaved changes, the current tool and
zoom in any workspace that has them (Inker, Plotter and Packwright zoom; Inker and Plotter have
tools), the queue when anything is running or waiting, and an amber **N issue(s)** when a startup
check has failed. That figure is a report rather than a control — for the list behind it, go to
**Settings → Health**, which is also where you look when nothing is failing and there is no count
in the group at all. There is no green "all well" state, because a healthy install has nothing to
report. When the menus a workspace needs leave the group no room, items drop lowest-priority-first
— the resource meter, then zoom, then tool, then document, then queue — but the health figure
never drops, whatever the window's width. Every item in the group is a readout, not a button: none
of it is clickable.

Beside the status group, and likewise never dropped, sits **✦ Familiar**. Once Familiar's weights are
installed its menu holds one row, **Open Familiar**, which expands the bottom pane below; until then
it stays a disabled **Not installed**.

## The bottom pane

One row along the foot of the window, in every mode, where the per-item status line used to live. Its
text depends on whether Familiar's weights are downloaded: **✦ Familiar isn't installed —** beside an
**Install…** button that opens Settings → Models, or once every row is present, a clickable **▸ ✦
Familiar** row that expands into a short conversation: a scrollback of what you and Familiar have
said, an input line, and **Send**. The per-item readouts that used to sit here (workspace, document,
tool, zoom, queue, health) moved to the menu bar's own right-aligned group, described above.

Familiar reads a sent message before answering it: a short router decision picks what the message is
actually asking for — build something in Clay, edit what's already there, a question about Warlock
itself, or just conversation — and answers accordingly, without you having to say which. A question
about Warlock (**"how do I export a GLB"**, **"what does the band setting do"**) is answered from the
Manual itself, with a small **[1]**, **[2]**… link under the reply for each section it actually used;
clicking one opens the Manual at that section. If the Manual has nothing on the question, Familiar says
so plainly rather than guessing. A skill the router recognises but this build does not act on yet
(character, create, navigate) still just answers in chat, honestly, rather than pretending nothing was
asked. In **Clay**, with a document open, the expanded pane also offers **Build**: describe what to
add and Familiar proposes it as a translucent ghost over your document, with **Apply** and **Discard**
beside it once it lands — the same ghost a Send message routed to a Clay build lands as, if the router
decides that is what you meant. Building needs the trained Clay model (`familiar_v1.0`); until that
model replaces the testing pin, a Clay build answers with a plain sentence saying so rather than a
ghost, however it was asked for. Each document tab keeps its own conversation, the same way it keeps
its own undo stack — closing a tab ends its thread, and every other mode without a document of its own
shares one Studio-wide thread.

The keyboard shortcut list is `Ctrl+/`, **Help → Keyboard shortcuts**, or **Keyboard shortcuts** in
the command palette, and it is reproduced in [Keyboard shortcuts](39-shortcuts.md).
