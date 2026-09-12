# Dressing a scene

Every tutorial before this one makes a *thing*: a reference, a mesh, a rig, a sheet, a track. This
one makes a *place* out of the things you have already made. That is the whole difference, and it is
worth saying plainly — Mason generates nothing. It has no model, no prompt and no queue. It
arranges.

This walks one path: a scene, a ground under it, a couple of assets from your library, a wall built
out of primitives, a prefab placed three times, a light, a camera, and an export an engine can open.
Twenty minutes, and not one second of your graphics card spent generating anything.

You need at least one finished mesh in your library. [Your first asset](02-your-first-asset.md) makes
one; so does [Modelling](07-modelling.md), and a Clay blockout works here just as well as a generated
prop. If you have neither, the primitives alone will carry you through — you will just be building
a room out of boxes.

## Opening a scene

Open **Mason** from the rail's workspaces group, just under Clay. With nothing open you get two
buttons in the middle of the window, **New scene** and **Open a file...**, and a list of anything you
had open before. Press **New scene**.

The viewport says "Pick one from the Assets panel." The scene is genuinely empty — no floor, no
light, nothing. Everything you are about to see, you put there.

Press `Ctrl+S` now and save it somewhere you will find again. Scenes are `.wscn` files. Saving early
means the rest of this is one `Ctrl+S` at a time rather than a dialog at the end.

## A ground to stand on

Scroll to the bottom of the **Assets** panel on the left and press **Add ground**.

A flat grid appears, sixty-four metres on a side at one vertex per metre. That is a quarter of what
Mason allows, and it is deliberately generous: you are not going to run out of room, and a ground
four times this size costs frame rate you would rather spend elsewhere.

Now shape it a little. Press `T` for the Sculpt tool, pick the **Raise** brush, set **Radius** to
about 6, and drag across one corner of the ground. The ground comes up under the pointer for as long
as you hold the button, and lets go when you do — one stroke is one undo step, however long you drew
for.

Two things worth knowing the first time:

- **The left button belongs to the brush.** While Sculpt is the active tool you cannot click to
  select, and dragging does not orbit. Use `Alt`+drag to turn the camera instead. That is true in
  every mode of this app's 3D viewport, and here it is the only way.
- **Smooth is the fix for everything.** A raise stroke that came out lumpier than you meant is one
  pass of the **Smooth** brush from acceptable.

Try **Flatten** with **Level from the first click** ticked: start the stroke on the flat part of the
ground and drag up the hill you just made, and the hill comes down to the height where you began.
That is how you cut a ledge without knowing what number the ledge is at.

Press `Q` to get out of the brush and back to Select.

## Putting something in it

The top of the **Assets** panel lists every finished mesh in your library. Click one.

Nothing happens yet — clicking *arms* the asset rather than placing it, and the hint line under the
viewport now says so. Click in the viewport, on the ground, and the asset lands there.

Click a second spot and it lands again. The arming stays until you turn it off, which is what you
want when you are scattering rocks. Press `Esc` when you are done placing.

Two of the same asset in the scene are **one upload to your card**, not two. That is a property of
how Mason caches things and it is the reason a scene with four hundred props in it is affordable at
all. You do not have to do anything to get it.

Now place a primitive: scroll to the **Primitives** rows and put down a **Box**. These are Clay's
shapes, and a scene needs a floor, a step or a wall far more often than it is worth generating one.

## Moving things about

Press `W` for Move and drag the gizmo's arrows. `E` rotates, `R` scales, `Q` goes back to Select.
These are Clay's keys and Clay's gizmo — if you have modelled in this app, your hands already know
them.

`F` frames whatever is selected, which is the key you will press most. `Alt`+drag orbits, middle-drag
pans, the wheel dollies.

Turn on **Snap** in the **Tools** panel with **grid (m)** at 1. Now drag the box: it lands on whole
metres. Turn on **Drop to ground** as well, and it lands *on the ground* rather than floating above
or sinking into it. Between those two, a row of crates lines up without you aiming at anything.

Select two things — click one, `Shift`+click the other — and try **Align** with the axis on `y` and
the mode on **Min**. Both of them come down to the lower one's base. With three selected,
**Distribute** spaces them evenly. Each is one press and one undo step, and each is a minute of
dragging you did not do.

## A wall, and then eight of them

Place a box, scale it into a wall panel, and put it where a wall should start.

With it selected, set **count** to 8 and **offset (m)** to `2, 0, 0` in the **Tools** panel's array
block, then press **Array (linear)**.

Eight panels, evenly spaced, in one press. **Array (radial)** does the same around a circle, which is
how you get a ring of pillars out of one pillar and a number of degrees.

## Grouping, so the scene stays legible

Select the wall panels — the **Outliner** on the right lists every node in the scene, and clicking
rows there is easier than aiming at things in the viewport — and press `G`.

They collapse into one group node. Rename it "West wall" by double-clicking its row. Moving the group
moves all eight; the group itself draws nothing and is just a name and a transform.

This is what keeps a scene of two hundred nodes readable. Twelve rows in the outliner, each of which
opens. `Shift+G` takes a group apart again.

The filter box at the top of the outliner narrows it to matching names, which is how you find one
crate later.

## A prefab, and why it is not a copy

Select something you expect to place repeatedly — a crate, a lamp post, the wall group you just made
— and right-click it. Choose **Make prefab** and give it a name.

A **Prefabs** panel appears in the left column. Click your new row and place it three times, the same
way you placed an asset.

Those three are **instances**, not copies. Select the original, change it — scale it, retint it — and
all three change with it, immediately, with no step to apply and nothing to get out of step. An
instance holds nothing of its own but a position.

When you want *one* of them to differ, select it and choose **Unpack instance**. It becomes a real
copy of the template's contents, editable like anything else, and the link is gone. That is the
trade, and it is the only way to break one.

## Light, and somewhere to stand

Press **Point** in the Assets panel's Lights block and click in the viewport. Do it a couple more
times where a room would be lit.

Select one and look at the **Properties** panel: colour, intensity, and range. A **Spot** adds an
inner and an outer cone angle. A **Directional** light is the sun — its position means nothing, only
which way it faces, so rotate it rather than moving it.

Lights draw as wire marks, not as glowing objects. There is nothing solid there and nothing to click
into by accident.

Then press **Camera** and place one where a player would stand. It is a marker saying "this is a view
into the scene"; it is not the viewport camera, and moving one does not move the other. It exports,
which is the point — an engine opening this scene knows where you meant it to be looked at from.

## Taking it out

`Ctrl+S`. Then look at the **Scene file** panel on the right, under **Take it somewhere**.

**Export GLB** is the one to use. It writes `scene.glb` and a `scene.json` beside it. The GLB is
ordinary glTF — the hierarchy, the instances as real nodes, the lights as `KHR_lights_punctual`, the
cameras as cameras — and any engine that reads glTF reads it. The JSON is ours, and it carries what
the scene *meant*: the node list, the counts, the prefab names, your user properties, and anything
that could not be resolved. `Ctrl+E` does the same thing from the keyboard.

**Export OBJ** is for the importer that will not take glTF. It loses groups, lights and cameras by
design, and says which — geometry and materials are all OBJ has.

**Export to the library** is the interesting one. It puts the scene into your library as an asset
row, with a thumbnail taken from the viewport you are looking at right now. Press it.

## The round trip

Open the **Library** and find the row that just appeared. It is a real mesh: it opens in Create's
Mesh stage, it can go to Clay or Poser or Troupe, and it behaves like anything else you made.

It also carries two doors back. Its exits offer **Open in Mason**, which reopens *the scene*, not the
merged mesh — every node, every group, every instance, exactly as you left it. Mason kept the `.wscn`
beside the asset for precisely this.

And every finished mesh in your library — including this one — offers **Add to Mason as a scene
item**, which drops it into whatever scene you have open. That is how a room becomes a wing of a
building: build the room, export it, start a new scene, and add the room to it three times.

A scene stores a *link* to each asset rather than the asset itself, and resolves it on open. That is
why a `.wscn` is small, and why re-running an asset updates every scene using it. It is also why
deleting an asset from your library leaves a hole: the Scene file panel lists what it could not
resolve, the Properties panel says so on the node, and putting the asset back fixes it. Nothing is
lost from the scene itself.

## What to read next

[Mason](31-mason.md) is the reference chapter: every panel, every brush, every field, the two scene
ceilings and what each export writes.

[Putting it in a game](13-putting-it-in-a-game.md) covers what each engine does with a glTF scene,
and the interop caveats worth knowing before you build a large one.

[Keyboard shortcuts](39-shortcuts.md) has Mason's full table, including the axis views and the
document keys this tutorial skipped past.
