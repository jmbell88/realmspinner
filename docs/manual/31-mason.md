# Mason

Mason is the scene editor. Every other mode in this app makes one *thing* — a reference, a mesh, a
rig, a sheet, a tile map — and Mason is the one that puts those things somewhere. You place library
assets and primitives into a scene, group them, instance them, light them, sculpt a ground under
them, and export the arrangement as something an engine can open.

It is the 3D answer to [Plotter](32-plotter.md), which is the 2D one: a tile map is a place made out
of tiles, and a Mason scene is a place made out of the assets you have already generated. It is
built alongside [Clay](30-clay.md) rather than against it — the viewport, the transform gizmo, the
camera and the undo history are the same ones, and if you have modelled in Clay you already know how
to move around in here.

What Clay does and Mason does not: there are no element modes. Nothing in Mason edits a mesh's
vertices, edges or faces — a scene is an arrangement of finished assets, and the place to change an
asset's shape is Clay or the pipeline that made it. Select, move, rotate and scale act on whole
objects, always.

It is a mode, not a takeover. Switching away leaves every open scene exactly as it was, and several
scenes stay open at once. Scenes are saved as `.wscn` files.

## Starting a scene

With nothing open, the middle column offers **New scene** and **Open a file...** and lists the
scenes you had open recently. The Scene file panel on the right offers the same, with **Save** and
**Save As...** beside them once something is open. `Ctrl+N` and `Ctrl+O` do the same two things from
the keyboard, and `Ctrl+W` closes the scene in front of you.

Choosing **Mason** from the Home screen opens an empty scene when there is nothing open already, and
leaves your scenes alone when there is.

An empty scene says so in the viewport: "Pick one from the Assets panel."

## The Assets panel

The left column's top panel is where everything in a scene comes from. It has four parts, in the
order you usually want them.

**Library assets.** Every finished mesh in your library, as a row. Click one to arm it, then click
in the viewport to put it down. This is the point of the mode: a prop you generated last week is one
click from being in a room.

**Primitives.** The same shapes [Clay](30-clay.md#adding-a-primitive) offers, in the same
categories, armed and placed the same way. A scene often needs a floor, a wall or a step that is not
worth generating, and a box is the right answer to all three.

**Lights.** **Point**, **Spot** and **Directional**, each with a line saying what it is — a point
light is "A bulb: falls off with distance in every direction", a spot is "A cone, narrowed by its
inner and outer angles". A directional light is the sun: its position does not matter, only the way
it faces.

**Camera.** A marker saying where someone stands to look at the scene, described as "A view into the
scene, exported alongside it". It is not the viewport camera and moving one does not move the other.

Lights and cameras draw as wire marks rather than as geometry. They are real nodes and they export,
but there is nothing solid there to bump into.

Below those is the terrain block, covered under [Ground](#ground) below.

### Arming and placing

Clicking a row in this panel *arms* it rather than placing it: the next left-click in the viewport
is where the thing lands. The hint line under the viewport says so while something is armed, and
`Esc` disarms it.

That is one of two jobs `Esc` does, in the order you mean them: it cancels an armed placement first,
and only clears the selection when nothing was armed. Pressing `Esc` after arming a light by mistake
does not also throw away the selection you were about to put it beside.

## Selecting

Left-click selects the thing under the pointer, and clicking nothing clears the selection.
`Shift`+click and `Ctrl`+click both *extend* the selection rather than replacing it — `Ctrl` toggles,
so a prop you clicked by mistake comes back out without starting over.

The same press that selects also arms an orbit: a click selects, a drag from that same press turns
the camera, and the two are told apart by whether the pointer travelled. `Alt`+drag orbits whatever
else is going on, which is why that gesture is reserved.

There is no marquee. Clay has one because picking a hundred vertices out of a mesh needs one;
picking objects out of a scene is what the [outliner](#the-outliner) is for.

## Transforming

`Q`, `W`, `E` and `R` choose Select, Move, Rotate and Scale, and the gizmo in the viewport follows.
Dragging an axis of the gizmo moves, turns or scales every selected node. This is
[Clay's transform gizmo](30-clay.md#transforming) unchanged, and so are the keys.

The **Pivot** choice in the Tools panel decides what a rotation or a scale happens *around* when
more than one thing is selected. The viewport header carries the same field.

### Snapping

**Snap** in the Tools panel rounds a drag to a grid: **grid (m)** is the spacing a move lands on and
**angle (deg)** is the step a rotation turns in. With snapping on and the grid at 1, a row of crates
lines up without you aiming.

**Drop to ground** makes a move end on the ground rather than in the air — on the terrain's surface
where there is one, and on Y=0 where there is not.

### Align, distribute and array

Four buttons that do in one press what a lot of dragging does badly.

Pick an axis and a mode — **Min**, **Centre** or **Max** — then press **Align** to bring at least two
selected nodes into line on that axis, or **Distribute** to space at least three of them evenly
along it. **Drop selection to ground** does the drop for everything selected at once.

**Array (linear)** and **Array (radial)** copy the one selected node repeatedly: `count` copies, each
`offset (m)` further along, or spread around a circle of `degrees`. A colonnade is one press.

Each of these lands as a single undo step.

## The outliner

The right column's scene tree. Every node in the scene, nested, with the eye that hides it and the
icon that says what kind of thing it is. Click a row to select it; drag a row onto another to make
it a child.

The filter box at the top narrows the tree to matching names, which is how you find one crate in a
scene of two hundred. **Solo** shows only what is selected and **Show all** puts everything back —
each greys out with the reason when it cannot act ("Select something to show on its own.", "Nothing
is hidden.").

Right-clicking a row offers **Rename**, **Duplicate**, **Solo**, **Move up**, **Move down**, **Move
to root**, **Group**, **Ungroup**, **Make prefab**, **Unpack instance** and **Delete**. The viewport's
own right-click menu carries the ones that make sense there.

### Grouping

`G` puts everything selected under a new group node, and `Shift+G` takes a group apart again. A group
is a transform and a name and nothing else: moving it moves its children, and it draws nothing
itself. Groups are how a scene of two hundred nodes stays a scene of twelve things.

Nesting is limited to 64 levels deep, which no hand-built scene reaches.

## Properties

The right column's second panel: everything about the one selected node.

**Identity** is its name and three toggles — **Visible**, **Locked** and **Static**. A lock stops
a drag in the viewport, not a click — it is reported by the resolver and never enforced by the
document, so a locked node still selects normally and an undo can always put back what was there
before the lock was set. Static is a hint for the engine you export to, not something the app acts
on.

**Transform** is the node's own position, rotation and scale, typed rather than dragged, with its
world transform shown read-only underneath. The two differ whenever the node is inside a group, and
seeing both is how you tell a group's offset from a child's.

Under those is a block that depends on what the node is. A mesh has its source and a material
override. A light has colour, intensity, range, and — for a spot — its inner and outer cone angles.
A camera has its field of view and its near and far planes. A prefab instance names its template.

At the bottom is a table of **user properties**: your own key-and-value pairs, carried through to the
export's manifest. This is where a scene says "this crate is breakable" to an engine that cares.

A mesh whose source cannot be found says so here — "Missing — the source could not be resolved." See
[Missing sources](#missing-sources).

## Prefabs and instances

A **prefab** is a subtree you have named as a template. **Make prefab**, from either right-click
menu, takes what is selected and stores it under a name; the Prefabs panel then appears in the left
column with a row for it.

Placing a prefab row puts an **instance** in the scene: a node that says "one of those, here". It is
not a copy. Editing the template changes every instance, because an instance holds nothing of its own
but a position — there is no propagation step to wait for and nothing to get out of step.

**Unpack instance** is the one way to make an instance differ from its template: it replaces the
instance with a fresh copy of the template's contents, which you can then edit like anything else.
The relationship is gone at that point, which is the trade.

Instancing is also what makes a large scene affordable. Five hundred instances of one asset are one
upload to your card rather than five hundred, which is the difference between a forest that edits
smoothly and one that does not.

The panel's delete button removes a template, and its tooltip warns what that does to the instances
naming it.

## Ground

Mason scenes have at most one terrain: a height field under the scene rather than a node you place
several of.

**Add ground** in the Assets panel creates it, and **Delete ground** removes it. The ground is a grid
of heights, at one vertex per metre, and its side is capped at **256 cells** — a 256×256 ground
rebuilds in under six milliseconds against a sixteen-millisecond frame, and twice that side does not.
The measurement behind that number is in `docs/measurements/`.

### Sculpting

`T` selects the Sculpt tool, and with a ground in the scene the left button becomes the brush for the
whole stroke — no picking, no gizmo, no selection change. An orbit mid-sculpt has to be `Alt`+drag,
which is exactly why that gesture is reserved.

Five brushes, chosen in the Assets panel:

| Brush | What it does |
| --- | --- |
| Raise | Pulls the ground up under the brush |
| Lower | Pushes it down |
| Smooth | Blends each cell toward its neighbours |
| Flatten | Pulls the ground toward one height |
| Noise | Adds falloff-shaped, seeded roughness |

**Radius** is how wide the brush is. The other fields change with the brush: an amount, a strength, a
level or a seed. Flatten's **Level from the first click** takes the height the ground already is
where the stroke began, which is how you flatten a hillside to the ledge you are standing on rather
than to a number you guessed.

One stroke is one undo step, however many dabs it took.

## The viewport header

The strip over the render, and a subset of [the 3D viewport's](24-the-3d-viewport.md#the-toolbar) own.

**Pivot** repeats the Tools panel's choice where your hand already is. **Solid**, **Material** and
**Wire** choose how the scene draws. **Grid** and **Wire** toggle the ground grid and the wireframe
over the render, and **X-ray** lets you pick something behind a surface.

Under the render is a hint line saying what the current tool does with the mouse, and a corner
readout of how many items the scene resolves to.

## Scene size

Two numbers decide what a scene may be.

Past **1,500 placed items** the readout and the Scene file panel warn: "This scene has more than
1,500 placed items and may cost frame rate to edit." It is a warning and nothing more — the scene
works, it is just no longer free to drag things around in.

At **100,000** the app refuses outright rather than drawing or exporting something truncated. Both
numbers were measured rather than guessed, and the document that fixed them is in
`docs/measurements/`.

## Missing sources

A scene stores a *link* to each library asset, by job id, and resolves it when the scene opens. It
does not embed anything until you export. That is what keeps a `.wscn` small and what lets a scene
pick up a re-run of an asset it names.

It is also what makes a missing source possible: delete an asset from the library and the scenes
using it have a hole. The Scene file panel lists what could not be resolved, the Properties panel
says so on the node, and the exporters name what they dropped rather than quietly writing a smaller
scene. Nothing is lost from the document itself — put the asset back and the link resolves again.

## Saving

`Ctrl+S` saves and `Ctrl+Shift+S` saves under a new name. The Scene file panel shows the path and one
line saying whether it is saved.

Saving happens off the frame thread, so the window keeps drawing; the panel's buttons grey with
"Saving..." while a write is in flight, and the keys that would change the document are ignored until
it lands. A `.wscn` is compressed, and it refuses to open one larger than the ceiling the service
stores scenes at — the app and the service agree on one number rather than each having their own.

If the app is killed with unsaved scenes open, they come back on the next launch as untitled, dirty
documents. They are not adopted onto the path they came from, because the file that path names may
still hold the contents you have not looked at.

## The three ways out

The Scene file panel's **Take it somewhere** section, and the whole reason the mode exists.

**Export GLB** writes `scene.glb` beside a `scene.json` manifest. The GLB is the scene as glTF: the
hierarchy, the instances flattened into real nodes, the lights as `KHR_lights_punctual`, the cameras
as cameras. The manifest is ours — format `warlock-mason-scene`, version 1 — and carries the node
list, the units, the counts (nodes, meshes, lights, cameras, triangles), the prefab names and
anything that could not be resolved. An importer that understands glTF needs only the GLB; one that
wants to know what the scene *meant* reads the manifest beside it. `Ctrl+E` does this from the
keyboard.

**Export OBJ** writes `scene.obj`, `scene.mtl` and a `textures/` directory. OBJ is geometry and
materials and nothing else, so this one loses things by design and says which: groups, lights,
cameras, unresolved meshes. It refuses past a million vertices rather than formatting a file nothing
will open. Use it for the importer that will not take glTF.

**Export to the library** turns the scene into a library asset — a mesh row with its own `model.glb`,
its thumbnail taken from the viewport you are looking at, and the `.wscn` kept beside it. That last
part is what makes it a round trip rather than a one-way flattening.

Everything here is in metres, Y-up, right-handed, with -Z forward and rotations as XYZW quaternions.
Those are glTF's own conventions, which is the point: nothing converts anything on the way out.
See [Putting it in a game](13-putting-it-in-a-game.md) for what each engine does with them.

## The round trip

A scene exported to the library is a real asset row. It appears in the library, it opens in Create's
Mesh stage like any other mesh, and it can be sent on to Clay, Poser or Troupe — because by then it
*is* a mesh.

It also carries two doors back. **Open in Mason** reopens the scene it was, from the `.wscn` stored
beside it; if that sidecar is gone it says so rather than handing you the merged mesh under the same
name. **Add to Mason as a scene item** puts the row into whatever scene you currently have open,
which every finished mesh in the library offers — a room becomes a building's wing without anything
being exported twice.

## Dropping files in

Dropping a `.wscn` onto the Mason window opens it. Dropping a `.glb` imports it into the open scene
as a mesh node.

## Where the files go

A `.wscn` goes wherever you save it; nothing in Mason writes into your library unless you ask it to.
**Export to the library** does, and what it writes lives under the job's own directory like every
other asset — see [The library and jobs](37-library-and-jobs.md).

The scene's own copy of an exported asset is kept beside that asset rather than listed as one of its
downloads: it is what the reopen reads, not a file you were meant to hand to anyone.
