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
  [The library and jobs](37-library-and-jobs.md)) — including a rig row you have selected directly,
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
- A clip authored for this skeleton's template may key a bone the edit removed. [Troupe](34-troupe.md)
  warns, at the door, how many bones its clips would skip on a custom skeleton like this one, so a
  thinner walk cycle is not a surprise discovered after the render.

Once a skeleton has been edited this way, `rig.json` records it as **custom** rather than as a plain
copy of its starting template — the fact the banner and Troupe's warning both read off. Choosing a
different skeleton through **Re-rig...** rebuilds from that template's own stock bones and discards
the edited shape entirely; the confirm says so before it happens.

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
Camera preset on the Troupe form ([Troupe](34-troupe.md#the-options)), because that is a choice
about the whole sheet's projection rather than about this model. Tilting the view here changes
nothing.

The front is stored on the asset, not on a sheet, so every sheet you build from it afterwards
inherits it — the character sheets Troupe renders and the plain pose-by-direction sheets alike — and
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

Below the presets is **Clips** — the keyframe editor for the animations a Troupe character sheet
plays. A *clip* is an ordered list of key poses plus how many frames each step between them holds,
and every skeleton ships ten: the original idle, walk, run, attack and jump, plus five added later
— attack_02 (a second attack), cast, fall, hit and death — built the same way and marked
`provisional` in the library file, meaning they are an early pass still awaiting an art review
rather than something wrong with your own edits. Until this editor existed clips could only be
changed by hand-editing a file inside the app's own installation.

Beside the clip picker, a clip marked provisional carries a muted **provisional** badge — hover it
for "Placeholder keyframes; an animator's pass is still owed", the same words the movement table in
Troupe uses for the same fact. It is a note about the clip, not a fault: nothing about editing or
saving it differs from any other clip.

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
  table baked into the build — it is what a Troupe sheet plays the clip at unless the sheet itself
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
which is not installed" or "This skeleton has no clip library to import into". Only the humanoid
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
it, so a later look at the library can still say where a clip came from.

**Licensing.** Warlock downloads nothing for this — you supply the file. The animation data itself
is governed by wherever you got it: Mixamo's motion library is Adobe's, under Adobe's own terms, and
those terms are what to check before using or redistributing anything you import here, not this
project's licence.

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
