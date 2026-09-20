# Poser

The Poser is where a pose is authored **once, against a skeleton**, rather than against any
particular mesh. Every other way of posing in the app starts from an asset: you rig a mesh, you open
its pose editor, and the pose you save belongs to that mesh. The Poser inverts that. You pick a
skeleton, pose the bare armature, and what you save is offered on every rigged asset that shares it.

That is the whole reason the mode exists. Fitting puts a template's joints in the same relative
place on every mesh it is fitted to, so a rotation authored against the humanoid skeleton means the
same thing on a gnome and on a giant. A pose is a map of joint names to local rotations and nothing
else — see [Posing](25-rigging-and-posing.md#posing) for that contract — which is exactly what makes
it portable.

The Poser needs Blender, for the reason given under [When Blender is missing](#when-blender-is-missing).

## Opening the Poser

Pick **Poser** in the rail. The left panel holds the skeleton picker; choosing a skeleton loads a
**preview armature** into the viewport — the bare bones, with no mesh around them.

The preview is built by the same Blender code path that builds a real rig, and it is exactly one
character-height tall. Both halves matter: what you pose is built the way a rig is built, and it is
scaled the way every bake will scale it, so nothing about the preview is a stand-in for something
you cannot see until later.

It is built once per skeleton and cached, so the first time you open a template takes a moment and
every later time is immediate.

**New pose** clears the editor to the rest skeleton and starts a fresh pose. Over unsaved work it
asks first.

## Posing a skeleton

Posing works exactly the way the pose editor on an asset does:

- **Click a joint** to select it. A rotation gizmo appears on it; drag to rotate. The panel names
  the selected joint, or says "Click a joint to rotate it" when there is none. A joint whose
  rotation has moved off its rest pose is marked with a trailing `*` — hover it for "Changed from
  rest".
- **Three rotation fields**, in degrees, beside the selected joint's name — Euler angles in the
  rig's own XYZ order. Typing a value turns the joint the same way the gizmo does: it is the same
  write underneath, so it undoes the same way too, one field edit to one `Ctrl+Z`.
- **Reset joint** returns the selected joint to its rest rotation. It is disabled until you have
  selected one, and says why on hover.
- **Reset all** returns every joint.
- **Mirror** copies the pose across the body's centre line. It is hidden for a skeleton with no
  mirror pairs — the serpent, whose chain of spine joints has no left and right — where it could
  only do nothing.

Poses are forward-kinematic only: no inverse kinematics, no translation on ordinary joints, no
scaling. The one exception is the root, which is [its own section](#moving-the-root) below.

## Undo and redo

`Ctrl+Z` undoes and `Ctrl+Y` — or `Ctrl+Shift+Z` — redoes. This works in the Poser and in an asset's
pose editor alike, because both are the same editor underneath.

**The unit of undo is the gesture, not the frame.** One whole gizmo drag is one step however many
frames of mouse movement it took, and so is applying a preset, a mirror, a reset and a joint move. A
drag that ends where it started records nothing at all, so an accidental nudge-and-return does not
leave a step you have to undo past.

Undoing back to the point you last saved from leaves the session **clean** rather than still asking
about unsaved changes — the history knows where the last save was, so retreating to it is genuinely
a return to saved state.

The history belongs to the editing session. It is dropped when you leave the mode or load a
different skeleton, and deliberately: a step holds rotations by bone name, so replaying one onto a
different armature would find whichever bones happened to share a name and silently mean something
else.

Your editing session itself survives switching modes, the way a document left open in the
[Inker](28-inker.md) does. Only quitting, switching skeletons, or loading another pose over it asks
about unsaved changes.

## Moving the root

Select the root joint and tick **Move root** to swap its rotation gizmo for translation arrows.
Dragging them offsets the whole pose — a crouch that actually lowers, a leap that leaves the ground.
Three offset fields beside the joint controls do the same thing by number, in the same units the
line below them reads back.

The offset is stored **in character heights**, not in world units, which is what makes it portable
the way the rotations are: a half-height offset lifts a gnome by half a gnome and a giant by half a
giant.

Two limits come with it, both about where the offset shows up:

- Applying a library pose to an asset previews the **rotations only**. The offset is real, but it
  appears in the baked GLB and in sprite sheet rows rather than in the inspector's preview.
- An animated sheet clip interpolates a root offset the same way it interpolates a rotation: frame 0
  sits at the start pose's own offset, and later frames climb toward the end pose's without reaching
  it, so a clip whose endpoint poses carry one plays as a vertical bob rather than being refused —
  see [Sprite sheets](27-sprite-sheets.md).

## Posing a real asset directly

Poser can open a rigged asset's actual mesh here instead of the bare template armature, so you can
use its own view controls and clip editor against the real thing rather than the template preview.
There are three doors in:

- The inspector's Pose panel, on the asset itself — its own **Open in Poser** link (see
  [Posing](25-rigging-and-posing.md#posing)).
- The **"Take it somewhere"** section on any rigged mesh, wherever the library shows one (see
  [The library and jobs](36-library-and-jobs.md)) — including a rig row you have selected directly,
  which offers the same list its mesh does.
- The **Rigged assets** picker at the top of Poser's own sidebar, above the skeleton block. It lists
  every rigged mesh newest first, with the one you currently have open marked, and a click opens it
  — the way in when you are already sitting in Poser and want a different asset, with nothing to find
  in the Library first. Nothing rigged yet points you at Create's Rig stage instead of a button.

While a session is bound to an asset this way:

- The skeleton picker above the library is replaced by a fact — the template's name followed by
  **"(from this asset's rig)"**, for example "Humanoid (from this asset's rig)" — because the
  skeleton is not a free choice here: it is whichever template this mesh was rigged with, and that
  is what decides the library beneath it too.
- A **This asset's poses** section appears above the shared library, listing what you have saved
  onto this asset specifically. It is separate from the shared, skeleton-keyed library below it —
  `Ctrl+S` saves to the asset, `Ctrl+Shift+S` always saves to the shared library, in either kind of
  session.
- **Re-rig...** sits under the skeleton fact. It opens a skeleton picker and, on confirmation, queues
  a fresh rig for this same mesh — the same job the Library's own **Rig** action starts — without
  leaving the session or hunting the source job down in the Library. Over unsaved pose edits it asks
  first, the same as every other destructive action here.

Rigging is queued work, out of process, behind whatever else the queue is already doing, so a re-rig
can take a while; the session stays open and usable meanwhile. Once the new rig lands, the viewport
rebinds to it on its own — unless you kept posing the old rig while it queued, in which case landing
asks before discarding that edit, the same as submitting the re-rig did — and if you picked a
different skeleton than the one you had, the clip editor and the shared library beneath it switch to
match the new one, exactly as they do when you change skeletons in an unbound session.

## Editing the skeleton

Re-rigging picks a different shipped template; editing the skeleton changes the *shape* of the one
you have — moving, adding or removing a pivot, or grafting a whole limb on. It is reached from an
open asset session's own **Skeleton** section, with **Edit skeleton**, and needs a pose with nothing
unsaved on it: save or reset the pose first if the button says so.

While editing, the armature sits at rest and shows the skeleton's structure rather than a pose. Click
a pivot to select it, or its tip for the very last joint of a chain:

- **Add child** — a new pivot continuing the selected one's own direction, half its length, that you
  then drag into place.
- **Split** — cuts the selected bone at its midpoint, inserting a new pivot there; everything that
  used to parent off the far half now parents off the new pivot instead.
- **Delete pivot** — removes just the selected bone, reparenting its children onto its own parent.
  The rest of the skeleton keeps its shape.
- **Delete limb** — removes the selected bone and everything beneath it. Asks first, naming how many
  bones go with it — losing a whole arm by one click on its shoulder is the mistake this catches.
- Renaming — type a new name in the box above the buttons and press Enter (or click away). A name has
  to be unique and cannot be reused from elsewhere in the skeleton.

**Adding limbs.** Pick a preset, a side (Left, Right or Centre) and whether to mirror it onto the
opposite side as well, then **Add limb** to graft it onto the selected pivot. A preset is authored
once and oriented from the bone it lands on, so the same preset reads correctly whichever pivot and
side you choose.

**Mirror edits**, when turned on, replays Add child, Split and Delete onto the bone's `.L`/`.R`
partner as well — for a skeleton whose two sides should stay symmetric. It has no effect on a bone
with no mirror partner.

A skeleton holds at most 64 bones; the count above the buttons turns to a warning colour as you
approach it, and going over is refused rather than silently truncated.

**Apply skeleton** queues a fresh re-rig on the edited shape — the same Blender job an ordinary
re-rig runs, so it re-skins the mesh from scratch and can take a moment; the session stays open and
usable while it works. Bone-heat weighting is attempted first, the same as any other rig, and falls
back to a coarser envelope weighting on a mesh it cannot solve for — the banner that appears once the
new rig lands says so when it happens. **Cancel** leaves editing without applying, asking first if
the draft has unsaved changes.

Applying changes what a pose or a clip can do with this asset, in both directions:

- A pose already saved onto this asset, or the shared library, keeps every bone it named. A bone the
  edit removed is simply skipped when the pose is next applied; a bone the edit added starts at rest,
  since no saved pose has ever said anything about it.
- A clip authored for this skeleton's template may key a bone the edit removed. The
  **Send to Poser…** door (see [Rendering a mesh you already
  have](#rendering-a-mesh-you-already-have) below) warns, at the door, how many bones its
  clips would skip on a custom skeleton like this one, so a thinner walk cycle is not a
  surprise discovered after the render.

Once a skeleton has been edited this way, `rig.json` records it as **custom** rather than as a plain
copy of its starting template — the fact the banner and that door's own warning both read off.
Choosing a different skeleton through **Re-rig...** rebuilds from that template's own stock bones and
discards the edited shape entirely; the confirm says so before it happens.

## Choosing the front

Every directional sprite sheet is a turntable: the renderer stands the camera at yaw 0, calls that
the front, and steps around the model from there. Yaw 0 is a direction in the *mesh's* own space,
though, and nothing guarantees your character is facing along it. The shipped skeleton templates
are; a mesh you brought, or one Trellis reconstructed from a reference image, is facing wherever it
happens to be facing. When it is turned, the sheet still comes out correctly laid out and correctly
tagged — and every cell shows the wrong side of the character.

This is not something the program can work out for you, and that is a measured result rather than a
missing feature. A calibration sweep over 37 finished assets tried to find each mesh's best-matching
view automatically; the answers scattered across a 330-degree range, and two independent metrics
agreed with each other no better than chance. There is no front to read off a mesh. There is only
the one you can see.

So: orbit until the character is facing you the way you want it drawn, and press **Set this view as
the front** in the right-hand panel. The panel reads the angle back to you, **Reset** puts it at 0
again, and **Look at the front** turns the camera back to the recorded angle without changing your
framing — useful for checking a front you set earlier, or one set from the 3D viewport.

Only the **turntable** angle is taken. How far above the horizon a sheet is shot from stays the
Camera preset on the sheet form ([The options](#the-options) below), because that is a choice
about the whole sheet's projection rather than about this model. Tilting the view here changes
nothing.

The front is stored on the asset, not on a sheet, so every sheet you build from it afterwards
inherits it — the character sheets rendered below and the plain pose-by-direction sheets alike — and
the direction previews follow. Sheets you already rendered are not re-rendered; build a new one, or
re-render the sheet you have, to see the change. Nothing about a mesh you never set a front on
changes at all: those sheets render exactly as they did before.

An unrigged prop cannot be opened here, because the Poser needs a skeleton to pose. Props get the
same control on the [3D viewport's own toolbar](24-the-3d-viewport.md#the-toolbar) instead.

## The pose library

**Save** writes over the pose you are editing; **Save as** asks for a name and adds a new one. Both
write into the **global** pose library, per skeleton, rather than into any asset's own saved poses.

The library is listed in the left panel under the skeleton picker, and the pose currently loaded in
the editor is drawn in the accent colour so you can tell which row you are working on. Each row
carries four actions:

| Action | What it does |
| --- | --- |
| Apply | Loads the pose into the editor, over whatever is there. |
| Rename | Renames it in place. |
| Duplicate | Copies it under a new name, so a variant does not cost you the original. |
| Delete | Removes it permanently. |

A filter box appears above the list once it is long enough to need one.

**Delete is permanent.** The pose library has no trash — unlike the asset library, which does — so
that is the one action here that asks first.

Poses saved here are offered on every rigged asset with a matching skeleton: the inspector's
**Pose** panel grows a **Library poses** section listing them. **Apply** there copies the pose into
that asset's own saved list as a snapshot, marked `(library)`, which then behaves exactly like a
pose you saved by hand on that asset.

**The snapshot is the point.** Editing or deleting the library pose afterwards never changes what an
asset already carries, so a bake you liked stays reproducible forever.

## Shipped presets

Below the library is **Shipped presets** — poses that ship with the app for the current skeleton.
They are read-only by design. A preset is a starting point, and the way to keep a version of one is
to apply it, adjust it, and **Save as**, which promotes your edit into the library beside your own
poses.

The section is absent entirely for a skeleton with no presets, rather than drawn empty.

## Editing clips

Below the presets is **Clips** — the keyframe editor for the animations a character sheet plays (see
[Rendering a character sheet](#rendering-a-character-sheet) below). A *clip* is an ordered list of
key poses plus how many frames each step between them holds,
and every skeleton ships ten: the original idle, walk, run, attack and jump, plus five added later
— attack_02 (a second attack), cast, fall, hit and death — built the same way and marked
`provisional` in the library file, meaning they are an early pass still awaiting an art review
rather than something wrong with your own edits. Until this editor existed clips could only be
changed by hand-editing a file inside the app's own installation.

Beside the clip picker, a clip marked provisional carries a muted **provisional** badge — hover it
for "Placeholder keyframes; an animator's pass is still owed", the same words the movement table in
[What a character sheet contains](#what-a-character-sheet-contains) uses for the same fact. It is a
note about the clip, not a fault: nothing about editing or saving it differs from any other clip.

**The armature is the editor.** Picking a key in the list loads it onto the skeleton in the middle
of the screen, and you pose it with exactly the controls on the right that you would use for a
library pose. **Update key from pose** puts it back. There is no second posing surface to learn.
Once the armature has moved off the loaded key, the button grows an accent dot — hover it for
"Pose differs from this key" — so you can tell there is something to store before you move on to
another key and lose it.

Everything the clip adds on top of that is *timing*:

- **Frame time (ms)** — how many milliseconds one *rendered* frame of this clip holds, in steps of
  10 from 10 to 1000, with an "≈ N fps" hint beside it so the number reads as a speed rather than a
  raw duration. This is the clip's own tempo, stored in the library file rather than in a fixed
  table baked into the build — it is what a character sheet plays the clip at unless the sheet itself
  sets a fixed **Frame rate** overriding every movement at once, and it is what `animated.glb`
  bakes the clip's keyframes against.
- **Frames after this key** — how many frames the step out of the selected key holds. Each row in
  the list shows its own, so reordering a key visibly carries its timing with it.
- **Loops** — whether the last key steps back round to the first. A looping clip needs one more
  step than an open one, so switching this resizes the timing list to match; there is no other
  value it could take.
- **Easing** — how the frames are spaced inside each step. Note that it needs at least *three*
  frames in a step to do anything at all: with one or two there is nowhere for it to act, and the
  panel says so rather than leaving you to conclude the setting is ignored.

**Onion skin** ghosts the keys either side of the selected one in the viewport, dimmed, so a
contact pose can be judged against the passing poses it sits between. A looping clip wraps — the
first key's neighbour is the last one, which is exactly the comparison a walk cycle needs.

**Play** scrubs the clip as the renderer will actually build it: the slider runs over the expanded
frames, through the same interpolation the character sheet uses, so what you see is what it will
draw. A scrubbed frame is *between* two keys and has nowhere to store an edit, so while you are
scrubbing the panel says which frame you are on and **Update key from pose** refuses by name.
**Back to key** returns to the selected key's own pose.

### Your clips and the shipped ones

**Save clips** writes your version into your own data folder and leaves the clips the app ships
completely alone. That matters twice: an update cannot overwrite your work, and **Revert to shipped
clips** is simply "delete my copy", so reverting also gets you any improvements a later version
ships.

Your saved copy is the *whole* library, not a set of changes layered on top of the shipped one — so
if you saved your own clips before this update added the five new ones, your copy still holds only
the original five. It does not gain attack_02, cast, fall, hit or death just because the build now
ships them; **Revert to shipped clips** is what gets you the full set of ten, provisional new ones
included, in exchange for whatever you had changed.

A save is refused, by name, if the result is something a character sheet could not be built from —
a clip whose segments do not add up to the frames the sheet's layout expects, a key that no longer
exists, two clips with one name. A refused save leaves your previous clips exactly where they were,
because the alternative is discovering the problem the next time you render a character.

### Importing a clip

**Import clip…**, above the clip picker, brings in an animation someone else authored — a Mixamo
download, a Rigify metarig export — instead of keying one by hand. It needs Blender and a clip
library to import into, and says so when either is missing: "Importing an animation needs Blender,
which is not installed" or "This skeleton has no clip library to import into". While a skeleton edit
is open, the button stays visible but disables the same way, with "Apply or cancel the skeleton edit
first." beneath it, and the Import report stays hidden until the edit is applied or cancelled — every
other control in this section is hidden for the same reason: they all read or write the pose a
skeleton draft holds at rest throughout the edit. Only the humanoid
skeleton has a shipped mapping table today; nothing stops you pressing the button on quadruped,
bird or blob, but there is nowhere for the sampled bones to land, and the import is refused once you
have picked a file rather than before.

Pick an `.fbx`, `.glb` or `.gltf` file with Mixamo or Rigify bone naming. Every animation the file
carries becomes a clip in your *working copy* — not on disk yet. A clip name already in use gets a
`_2` (or `_3`, and so on) appended rather than overwriting the existing one, and the same is true of every key pose
the import needs: an existing pose, imported or original, is never replaced, only added beside. The
first imported clip is selected so you can look it over immediately, and the working copy is marked
unsaved exactly as a hand-keyed edit would be — **Save clips** is still the only thing that writes
anything, so an import you do not like costs nothing but **Revert to shipped clips** or leaving the
screen.

Under the button, the **Import report** says what the conversion actually decided, per clip: which
mapping table matched, how many bones were left at the template's rest pose because the source
skeleton did not name them, which source bones were ignored (fingers and other joints no template
bone corresponds to), whether the result loops and how large the seam residual was, how many frames
and keys it reduced to, and which root-motion mode was used — **in place** (drift removed, the bob
kept) unless you asked for something else. Every import is resampled to at most 32 frames regardless
of the source's own length. The saved clip remembers which file and which mapping table produced
it, so a later look at the library can still say where a clip came from. An action longer than 900
frames is not sampled at all — Blender skips it rather than failing the whole import — so it has
no entry of its own; it is named above them as **Skipped**, with the reason, and a file whose every
action is skipped says so in the toast rather than reading "Imported 0 clip(s)".

**Licensing.** Realmspinner downloads nothing for this — you supply the file. The animation data itself
is governed by wherever you got it: Mixamo's motion library is Adobe's, under Adobe's own terms, and
those terms are what to check before using or redistributing anything you import here, not this
project's licence.

## Rendering a character sheet

A character sheet is a grid of one creature, animated, seen from up to sixteen directions and
reduced to pixels — the clips above, played out and baked into one PNG plus a JSON sidecar that says
which cell is which. This is where a clip stops being something you pose in the editor and becomes
something you watch play.

Two things are worth knowing before you press the button, because they are about which of three
routes actually gets you there. Reconstructing a mesh from one generated drawing is measured not to
work reliably yet: a humanoid built this way, judged on a graded corpus run at the shipped default on
2026-08-30, came back with limbs bent and stretched — the reconstruction is asked for separable limbs
from a single view and does not deliver them. The reference stage itself is fine; it is the mesh that
is lost. And the shipped animation keyframes are **provisional** — authored placeholders rather than
an animator's work, so a walk that reads as a walk is not the same as a walk you would ship.

What follows from that is a recommendation rather than a refusal. Two routes avoid the reconstruction
step entirely, and either is worth reaching for before [Starting a new
character](#starting-a-new-character) below:

- **[Create → Character](22-generating-references.md#characters)** builds the body from an authored
  registry — four body plans, thirty-one species — instead of recovering one from a picture. It needs
  no graphics card, and [chapter 11](11-a-character-sprite-sheet.md) walks it end to end.
- **A mesh you already have** — generated and kept, uploaded, or built in Clay — is the route for a
  character that is yours; see [Rendering a mesh you already have](#rendering-a-mesh-you-already-have)
  below.

Everything downstream of the mesh — the rig, the clips, the render, the reduction — is the same code
on all three routes. [Starting a new character](#starting-a-new-character) is left in place because
it is how you get a reference for a creature the registry does not model, and because the verdict
above is on today's default reconstruction, not on the idea.

The output is an ordinary sprite sheet — the same PNG-and-sidecar pair [Sprite
sheets](27-sprite-sheets.md) describes — plus an `animation` block that carries the frame durations
and one tag per animation and direction. Everything that already reads a sheet reads this one.

### What a character sheet contains

The movement table lists every clip the rig's skeleton actually defines, not a fixed five. On each
of the four shipped skeletons (`humanoid`, `quadruped`, `bird`, `blob`) that is ten:

| Animation | Frames | Loops | Frame time |
| --- | --- | --- | --- |
| idle | 4 | yes | 150 ms |
| walk | 8 | yes | 100 ms |
| run | 8 | yes | 60 ms |
| attack | 6 | no | 80 ms |
| jump | 6 | no | 100 ms |
| attack_02 | 6 | no | 80 ms |
| cast | 6 | no | 90 ms |
| fall | 4 | yes | 100 ms |
| hit | 4 | no | 90 ms |
| death | 6 | no | 120 ms |

The first five are switched on by default; the last five — attack_02, cast, fall, hit and death —
are present on the table but switched off, each carrying a **Provisional** note: placeholder
keyframes an animator's pass has not reached yet, not a fault in your request. Switching one on
folds it into the sheet exactly like any other movement.

That is the default: 32 frames per direction and 256 cells in total for the first five, laid out
eight to a row in the order `animation → direction → frame`. The directions are the eight compass
points of a turn, starting at `front` and going clockwise in 45° steps.

That layout is configurable, and the layout form under **Build a new sheet** — or **Start a new
character**, for a character that does not exist yet — is where you change it. Each movement can be
switched off or given a different frame count, and each can be rendered in 1, 4, 8 or 16 directions.
A sheet warns above 256 cells and refuses above 512.

What does not change is the *shape* of the contract. Eight cells to a row, and the order
`animation → direction → frame`, so an engine that reads the sidecar knows where `walk_left` starts
without guessing. The sidecar carries the layout the sheet was actually built with, which is what
makes a per-character frame count safe to offer.

### Starting a new character

Under **Start a new character**, describe the character, pick a build and a reference pose, and
press **Draw the reference**. This section is always open to you, whether or not an asset is bound to
this Poser session — a fresh character has nothing to bind to yet.

That queues **one image and then stops**. The reference is drawn against a pose guide, which is a
stick figure fed straight to the ControlNet — legs straight, feet on one line, and the arms held out
at whichever of the two poses you picked. The constrained pose is not an aesthetic choice: a
single-view reconstruction has to get limb separation right, and a folded arm is the failure it
cannot recover from.

**A-pose or T-pose.** A-pose is the default and holds the arms 45° down; T-pose holds them straight
out. The difference is not cosmetic and it shows up two steps later, at the rig. The shipped humanoid
rig template is itself an A-pose, so an A-posed mesh is fitted straight to it. A T-posed mesh is not
— fitting that template to one runs the arm chain down through the ribcage — so its joints are
*measured* off the mesh instead, which needs the pose model [Installation](40-installation.md)
covers. T-pose separates the limbs further, which is the one thing a single view has the most
trouble with, so it is the one to reach for if the arms come back fused to the body.

You approve that drawing in [Create](22-generating-references.md), the same way you approve any
reference. Only then does the rest run: the reconstruction, then an automatic rig, then the sheet.
Approving is the gate, and it is deliberate — the reconstruction is the expensive step and it should
not be spent on a drawing you would not have kept.

If the character is a disaster, the drawing is still a row you can reroll or edit. That is the point
of the two-step shape.

### Rendering a mesh you already have

If you already have a character — one you generated, uploaded, or built in Clay — **Send to
Poser...** takes it in directly, without you having to bind it to a Poser session first. It is on the
mesh's right-click menu in the [library](36-library-and-jobs.md), and on the inspector under the
asset.

There is no reference and no gate on this route: the mesh already exists, so the only decision left
is what the sheet should look like. Both doors **ask before they spend anything**: **Send to
Poser…** opens a dialog with the skeleton, the sprite size, the camera, the outline and the colours
on it. That is why the labels carry an ellipsis.

Whatever you choose in that dialog is remembered: it is the same form **Build a new sheet** and
**Start a new character** draw, so a size chosen from the library is the size those sections open on
next time.

If the mesh is not rigged, it is rigged first, with the joints measured off the mesh rather than
fitted to its bounding box — and the dialog's **Skeleton** picker is where you say which rig. It
offers only the skeletons with clips authored for them, because a sheet is animated from a clip
library. Rigging is a real cost: minutes of CPU before a single cell is rendered, which is why the
dialog says so. You get two rows in the queue — the rig, then the sheet — and either can be
cancelled on its own; cancelling the rig simply means no sheet.

Sending the same unrigged mesh through this door a second time while its first rig is still running
is refused rather than queuing a second rig behind it — one rig for a mesh at a time, so wait for the
first (or cancel it) before asking again.

A mesh **already** rigged is animated on the skeleton it already carries, so the dialog does not ask
about a skeleton — and one rigged on a skeleton that has no clips is refused immediately, before
anything is queued: a walk cycle means nothing to a skeleton nobody wrote one for. Four of the eight
templates ship with clips — `humanoid`, `quadruped`, `bird` and `blob` — and each carries all five
movements. `fish`, `insect`, `serpent` and `biped_tail` have none, so a mesh rigged on one of those
means re-rigging it on one of the four (see [Editing the skeleton](#editing-the-skeleton) or
[Re-rig...](#posing-a-real-asset-directly) above) and sending it again.

### The options

| Setting | What it does |
| --- | --- |
| Build | Which guide conditions the reference: male or female. They differ in shoulder width, arm length and stance. Only on **Start a new character**. |
| Reference pose | A-pose or T-pose. A-pose matches the rig template and is the default; T-pose separates the limbs further. Both draw the same figure — only the arms move. Only on **Start a new character**. |
| Camera | One of four **presets**, and the helper under the picker states the chosen one's elevation in degrees. **3/4 top-down** (35°) is the default and the angle most 2D games with depth are drawn at; **Isometric** (30°) matches what tilesets call isometric; **Side** (0°) is straight on; **Top-down** (60°) is as far over as a humanoid still reads — a true overhead figure is a pair of shoulders and a hat brim. The degrees are in the helper rather than left to the name because the number is the thing that transfers: if you are matching these sprites to a Plotter map you already know what elevation that map is drawn at, and "isometric" does not answer that while 30° does. A sheet records which one it was rendered at in its sidecar. This setting is the elevation only — which direction of the model counts as its **front** is a property of the mesh rather than of the sheet, set by orbiting it in [Choosing the front](#choosing-the-front) above or the 3D viewport; the **Send to Poser…** dialog, which always has a real mesh in hand, states that mesh's own recorded front under this same picker. |
| Skeleton | Which rig an **unrigged** mesh is built on, and therefore which clip library its sheet is animated from. Only on the **Send to Poser…** dialog, and only for a mesh that is not rigged yet: a rigged one is animated on the skeleton its own rig records. Only the four templates with clips are offered. |
| **Style** | **Pixel art** or **HD**. Pixel art reduces the render to a logical size, a colour budget and an outline pass — the sheet described everywhere else in this section. **HD** keeps the render as painted, full colour and soft edges, with no colour budget at all — and it disables **Outline**, **Palette**, **Colours** and **Dither** below rather than hiding them, so each says why it is off instead of simply not being there. |
| **Frame rate** | **Authored** — each movement plays at its own clip's recorded frame time — or a fixed rate applied to every movement at once. Choosing a rate also rescales each movement's own default frame count at that rate, so a walk keeps its real length rather than playing faster or slower than it was authored. Authored is the default, and a form that never touches this control sends no rate at all. |
| Sprite size | How many pixels tall one cell is. 16, 24, 32, 48, 64, 96, 128 or 256, or **Custom…** for any whole number from 8 to 256. A custom size that does not divide 512 (the render's own resolution) is still built — it is resized down with nearest-neighbour instead of the usual box reduction, which is a slightly blockier result. At 256, the 8192-pixel atlas ceiling allows exactly 256 cells — the five default movements across eight directions, with nothing to spare — so switching on a sixth movement at that size is refused on the movement table above rather than silently dropping one. |
| Outline | `outer` grows the silhouette by a dark pixel, `inner` recolours the sprite's own edge, `none` leaves it alone. Not offered when **Style** is HD. |
| Palette | A palette file if you have installed one, or a palette derived from the render by median cut. Not offered when **Style** is HD. |
| Colours | The budget for a derived palette. Ignored when a palette file is named, and not offered when **Style** is HD. |
| Dither | Ordered dithering when colours are mapped. Off by default; at sprite sizes it is usually noise. Not offered when **Style** is HD. Only on **Build a new sheet** and **Start a new character** — the **Send to Poser…** dialog does not ask. |

Every one of these is checked when you press the button, not when the sheet is finally rendered —
so an unreadable palette costs you the click rather than an hour. A refusal that names the whole
movement table (a sheet over the cell ceiling, a movement the rig has no clip for) highlights the
table itself rather than one row; a refusal on **Frame rate** highlights that control the way any
other single field does.

Sizes divide evenly out of the 512-pixel render at 16, 32, 64, 128 and 256; the other three go
through a documented resize instead. Neither is wrong, but the exact ones are crisper.

## Watching the sheet

The middle of the window shows one sprite at a whole-number scale with no filtering. A sprite drawn
at 6.3× through a smoothing filter is a blurred sprite, which is the one thing the whole pipeline
exists not to produce.

**It opens paused, on the first frame.** Press play, or `Space`, to run it. A bad frame in a walk
cycle is obvious in half a second of playback and invisible in a contact sheet — but the first thing
anyone does with a new sheet is look at a *frame*: a hand, a silhouette, which way the feet point.
A clip already moving when you arrive is one you have to stop before you can look at anything.

**The heatmap** sits above the sprite: one square per cell of the selected animation, directions
down and frames across. Amber means the frame's silhouette, position or colours jump from the one
before it, a cycle's last frame does not meet its first, or that direction has drifted in size from
the rest; red with a cross means well past that, or an empty cell. Hover for the numbers, click to
land the preview on that cell. It is computed once when you pick a sheet -- "scoring..." while it
runs -- and it ranks cells for you to look at. Nothing downstream reads it; a red square refuses
nothing.

The row above the sprite carries the transport and the two selectors:

- The **stop/play** button pauses and resumes. So does `Space`.
- The **arrows** step one frame, and stepping pauses — stepping means you are looking at something.
  `Left` and `Right` do the same.
- The **animation** and **direction** buttons choose what plays. Changing animation restarts the
  clip; changing direction does not, so you can turn the character mid-stride and see the same
  frame from the other side. `Up` and `Down` turn the character, `PageUp` and `PageDown` change
  animation, and `Home` and `End` jump to the first and last frame of the run.
- The **zoom** field is how many screen pixels one sprite pixel is drawn as.

Two more buttons on that row are about *reading* the sprite rather than playing it:

- **Checkerboard** (`C`) puts a checker behind the sprite, so a transparent pixel looks transparent
  instead of looking like the panel's colour. It is **off** by default: a character sheet is judged
  on its colours first, and a pattern behind every frame is noise until the question you are asking
  is "where is the transparency".
- **Pivot** (`P`) draws the point the sidecar records as this sprite's pivot — where an engine will
  place its feet. It is **on** by default, because it is the one thing on the picture that is not in
  the picture, and a user who does not know the mark exists will never switch it on to find out.
  Nothing is drawn on a sheet that records no pivot, which is any sheet rendered before pivots were.

A looping animation loops. A one-shot — attack, jump — holds its last frame rather than stopping,
because a preview that stops needs a control to start it again.

## The sheet panel

The top of the right column says what you are looking at: the grid, the cell size, how many tagged
runs the sidecar carries, and what the pixel-art pass measured — how many colours came out, which
palette they came from, and how many stray pixels were cleaned up. An **HD** sheet ran no reduction
at all, so this reads **HD -- full colour** instead of a colour count.

A large stray-pixel count is worth noticing. It means the reduction found detail the palette could
not hold, and the usual answer is a bigger sprite or a wider palette rather than a different
character.

### Needs repair, and why it is not the heatmap

A sheet that failed its structural check carries a **Needs repair** pill — in the panel, and again
on the row in the sheet list, so that when you are comparing a 32 px attempt with a 64 px one you can
see which came back broken without selecting each in turn. Under the pill is what was found, as
sentences: how many cells are cut off at the frame edge, how many came back empty, how many the plan
named and nothing rendered, anything the sidecar says that disagrees with itself, and whether the
sheet had to be rendered a second time at a wider margin to fit its poses.

**Two different questions, and blurring them would be the mistake.** The heatmap above the sprite is
*advice about the drawing*: it ranks each cell against its neighbours and flags the worst, and
nothing anywhere refuses a sheet on its account. Needs repair is *facts about the file*: a cell is
cut off, empty, or was never rendered. An amber square and this pill do not mean the same thing.

Needs repair gates nothing either. The sheet plays, exports, and opens in Inker exactly as any other
does — an intentionally edge-to-edge portrait counts as clipped by this measure and is exactly the
sheet its author wanted. What the pill buys you is knowing which runs are worth re-rendering.

The camera is framed to hold every pose in the sheet before anything renders, so a jump apex or an
overhead attack wind is inside the frame rather than cut off at the top; if a silhouette still
touches the edge, the sheet is rendered once more with more room around it, once, and the sidecar
records that it happened.

### Making another one

**Build another sheet** renders the same character again at whatever the layout above currently
says. It re-uses the rig that already exists, so it is minutes of CPU and no GPU at all — which is
what makes "try it at 64 as well" a reasonable thing to do.

**Re-render some runs**, under the same section, rebuilds only the animations and directions you
tick and copies the rest from the sheet you started with, at that sheet's own settings. The result is
a new sheet, and Inker's *Merge re-render* brings it into a document you have been cleaning up:
untouched cells take the render, cells you painted keep your work, and cells where both changed are
flagged for you rather than overwritten.

**Vary in Create** appears on a sheet the character registry built, and loads that character's whole
recipe into Create's Character column as *your own* settings — so a new seed, a wider palette or a
longer horn makes the next one, and editing the prompt afterwards will not undo them. A sheet
rendered from a supplied mesh, or from this section's own form above, has no recipe behind it, so the
button is absent rather than greyed: a disabled control implies a state you could reach.

## Characters that are on fire

Some species carry themes that do more than repaint them. A fire ogre, a fire dragon and a fire
elemental each declare an *effect* as well as a palette, and the sheet draws it: a flame at the
socket the body plan puts it on — the elemental's core, the dragon's and the ogre's crown — in
every cell, animated with the movement it belongs to.

This only happens for a character the app built from a species. A mesh you generated from a
photograph or uploaded has no species behind it, so there is nothing declaring an effect and the
sheet is the sheet it has always been.

Three things are worth knowing about how it is drawn.

The flame is composited **before** the pixel-art pass, not after, so its oranges go through the same
colour cut as the character's skin. That is why a 16-colour sheet of a burning elemental spends
some of those sixteen on fire — and why the flame looks like it belongs to the sprite rather than
sitting on top of it. If you want more of the palette back for the body, ask for more colours.

The flame **rises**, whichever way the character is facing. Fire goes up in the world, not up
relative to the camera, so the eight directions of a turn all show a flame going the same way.

And it is drawn **behind** the body when the socket is on the far side of it. A back-mounted effect
seen from the front is hidden by the character, which is what makes turning around read as turning
around.

One known limitation. **A short loop does not loop seamlessly.** The flame's shape comes from a
scrolling noise field, and that field does not come back round to where it started — so the last
frame of a four-frame idle does not hand cleanly back to the first. At sprite sizes it reads as a
flicker in the tips of the flame rather than a visible jump, and the longer movements (an
eight-frame walk or run) hide it almost entirely. If it bothers you on an idle, a longer idle is
the answer: give the movement more frames in the layout above.

## Taking a sheet somewhere

Four ways out, and the first two are bridges the app already had, because a character sheet is an
ordinary sheet with an animation block on it.

**Open in Inker** opens the sheet sliced on its own grid, one tag per animation and direction, in
the [Inker timeline](29-inker-animation.md). It opens *unlinked*: the first `Ctrl+S`
is a Save As, so cleaning up frames cannot overwrite the render they came from.

**Add to Packwright** parks the sheet as a pending tile-set import: it opens Packwright's tile-size
popup with the cell size already filled in from the render, and nothing is added to the atlas
until you confirm **Import** there.

The last two are the ones that produce *files*, and both write into a folder you choose — or
straight into your configured export folder, if you have one.

**Export package...** copies the PNG and its JSON sidecar together, as a pair on purpose: the PNG is
the atlas and the JSON is what says which cell is `walk` facing south-east, so a folder holding one
without the other holds an asset nothing can interpret. Either both land or neither does.

**Export frames...** cuts the same atlas into one PNG per frame instead, for an engine that wants
`AnimatedSprite2D`-style frame folders rather than an atlas-plus-sidecar pair: one folder per
movement, one subfolder per compass direction inside it, and `000.png`, `001.png` and so on inside
that — beside a `manifest.json` naming the format, the frame size, whether the sheet is pixel art or
HD, and each clip's own loop, frame count, frame time, fps and directions. A re-export replaces the
destination folder whole rather than merging into it — the same all-or-nothing swap **Export
package...** makes, generalised to however many files a frame export writes.

The sheet and its sidecar are on disk beside the mesh either way, in that job's directory, and the
[library](36-library-and-jobs.md)'s export list is where the files themselves are.

## When a sheet goes wrong

**"A character sheet needs a rigged mesh."** The automatic rig failed, or you cancelled it. Rig the
mesh above — joints measured off the mesh's own vertices, which is what the automatic pass tries
first — and then use **Build another sheet**.

**The arms are welded to the chest.** The shipped humanoid template is an A-pose, and fitting it to
a mesh standing in a T-pose puts the arm chain inside the ribcage. The automatic rig asks for
measured joints for exactly this reason; if it still happens, correct the joints here.

**Everything is one pale colour.** The mesh has no texture. The reference chain is what puts colour
on a character; a supplied base mesh with no material has nothing for the palette to quantise, and
the sheet will come back in whatever few greys the render produced.

**The character is lying down.** A rotation stored in the wrong frame. Clips ship authored as deltas
from the skeleton's rest pose, which is the only frame that survives the rig being fitted to a
different mesh — a clip authored against one skeleton's node-local orientations means something else
entirely against another's.

**"That sheet would be …px; the limit is 8192px."** The movement table's own refusal, shown above the
table rather than on one switch. At 256 px sprite size the five default movements across eight
directions already fill the 8192 px atlas exactly, so switching on a sixth movement — or any of the
five Provisional ones — at that size goes over it. Turn one off, or drop **Sprite size**.

## When a pose file goes wrong

A pose file that has gone wrong on disk — truncated, hand-edited into the wrong shape, or simply not
JSON any more — costs itself and nothing else.

It stays in the list, so there is a row to act on. Applying, renaming or duplicating it says the
record could not be read. And **Delete always works**, because a pose you cannot read is exactly the
one you most need to be able to remove. One broken file never takes the library down with it.

## When Blender is missing

The Poser builds its preview armature with Blender, so without it the mode has nothing to show. It
says "Posing needs Blender, which is not installed" and offers nothing, rather than drawing controls
that could not do anything.

Blender ships CPython 3.13 wheels only, so on any other Python version the optional extra installs
nothing at all. See [When rigging is
unavailable](25-rigging-and-posing.md#when-rigging-is-unavailable) for the full list of what that
takes with it, and [Installation](40-installation.md) for how to get it.
