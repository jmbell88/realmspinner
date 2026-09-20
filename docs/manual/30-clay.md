# Clay

Clay is the top-level modelling mode: primitives, transforms and a material palette, in the same
window as everything else. It exists because the pipeline has one weak spot that no prompt fixes —
if you already know the shape you want, describing it to a diffusion model and hoping is a poor way
to get it. Clay lets you say it directly.

It is a mode, not a takeover. Switching away leaves every open document exactly where it was, and a
reconstruction started before you switched keeps running with its progress card floating over the
viewport. Only quitting the app can lose unsaved work, and it asks first.

The layout follows the rest of the app: tools and the selected object's properties on the left, the
viewport in the middle, the outliner and the document panel on the right. Several documents stay
open at once.

## Starting a document

With nothing open, the middle column offers **New model** and **Open a file...**, and lists the
documents you had open recently — clicking one reopens it, and hovering it shows the full path. The
document panel on the right offers the same two buttons, with **Save** and **Save As...** beside
them once a document is open — the same four buttons every workspace has, over the file's path and
one line saying whether it is saved. `Ctrl+N` and `Ctrl+O` do the same two things from the keyboard.

Choosing **Clay** from the Home screen opens an empty document for you when there is nothing open
already. When there is, it leaves your documents exactly as they were — the documents *are* the
work, and entering the mode is not a reason to disturb them.

## Adding a primitive

A document with nothing in it says so in the viewport itself — "Add a shape", with "Pick one from
Tools" underneath and a button that drops a box at the origin — rather than leaving you to notice an
empty grid and go looking for the **add** row on your own.

The **add** row is one icon grid, in three groups. **Primitives**: box, plane, grid, cylinder,
cone, UV sphere, icosphere, capsule and torus. **Structures**: pyramid, arch, column, lathe,
sweep and tube. **Game**: wedge, ramp, rounded box, stairs, wall and doorway. Clicking one places
it at the origin, selects it, and marks it the tool in hand — its icon stays lit until another
button, primitive or figure, is pressed next. Hovering
a button names it. Nothing is lit and no preview block shows below the grid until you have
pressed one -- a fresh document does not arrive with a shape already picked.

Under the grid, a short block names whichever tool is lit and lists the numbers a fresh press of it
starts from — a cylinder's `radius`, `height` and `segments`, say. It is a preview of what the next
click places, not a second place to edit them: a shape's own numbers are edited on the object itself,
in Properties, once it exists.

Shapes arrive with their shading already set, by the same rule the **Shade Auto...** button uses: a
sphere, an icosphere, a capsule and a torus come in smooth, and a box, a pyramid, an arch, a
column, a lathe, a sweep and a tube come in flat (at its default profile, outline or path and side
count — a lathe with a gentler curve and enough segments, or a tube with enough sides, can still
come back with its side band smooth, the caps staying flat for the same reason
the cylinder's and cone's do, below). A **cylinder and a cone come in flat too**, and that is the
rule working rather than missing them — every face on the side band meets a flat cap at a right
angle, and smoothing the band on its own would round the cap's rim, which is the edge the cap is
there to define. Shade Smooth and Shade Flat override any of this whenever you want them to.

The structures are the shapes that are tedious rather than hard — the ones you would otherwise
assemble out of three or four primitives and then have to keep assembled. A **pyramid**'s base sits
square to the axes, which is what separates it from a four-sided cone: a cone stands on a corner,
45 degrees off the box you are putting it on top of, and its `base` is the flat-to-flat width. An
**arch** is a doorway — two legs and a semicircular head, swept through its `depth`, with
`thickness` setting how heavy the wall is; the opening goes right through. A **column** is a lathe
with a fixed shape: `base` and `capital` are the *heights* of the plinth and the block at the top,
and setting both to zero leaves a plain shaft. A **lathe** is the general case of a column — a
`profile` of `[radius, y]` stations, bottom to top, revolved into whatever silhouette they trace,
which is what a bottle, a vase, a goblet, a handle or a turned finial needs and a column's two fixed
numbers cannot reach. The `y` values are read as a shape, not a place — Warlock re-centres them for
you, so a profile running 0 to 1 builds the same silhouette as one running -0.5 to 0.5 and the
object still sits wherever Properties says it does. A station of zero radius at either end comes to
a point rather than a flat cap, which is how a lathe reaches a finial or a chess pawn's rounded top.
There is no profile editor yet: Properties shows a placed lathe's numbers as a read-only line rather
than fields you can drag, the same way any parameter shape nobody has built a widget for yet is
shown. A **sweep** is the other family a lathe cannot reach — a closed 2D `outline` extruded along
`depth` rather than revolved, for anything whose cross-section stays the same, scales or turns
along one axis instead of around it: an L-bracket, a channel, an I-beam, a star, a gear blank, a
picture-frame moulding, a keystone. `taper` narrows or widens the far end about its own centre, and
`twist` turns that end about the extrusion axis; both are plain numbers rather than a second outline
to loft into, on purpose — a frustum, a pedestal and a twisted column are what a loft would be for,
and two sliders already reach all three. As with a lathe's `profile`, there is no outline editor
yet either: Properties shows a placed sweep's corners as a read-only line, and — unlike every other
shape here — a self-crossing outline (a figure-eight) is not caught, so a sweep is the one primitive
where keeping the shape simple is on you rather than on Warlock. A **tube** is a circular
cross-section of one `radius`, swept along a `path` — a cable, a hose, a handle, a pipe run, a bent
exhaust, anything that goes somewhere rather than sitting on one straight or rotational axis, which
is what neither a lathe nor a sweep reaches on its own. The ring stays square to the path the whole
way along rather than tipping into the turn, so a bend does not open a gap on its outside or pinch
its inside. As with a lathe's `profile` and a sweep's `outline`, there is no path editor yet:
Properties shows a placed tube's stations as a read-only line, and — the same admission a sweep's
self-crossing outline already makes — a `radius` wider than the path's own tightest turn passes
through itself uncaught, so a tube is the other primitive where keeping the shape simple is on you.

The game shapes are the pieces a level is blocked out of, and they exist for the same reason the
structures do: each one is a thing you would otherwise build from three boxes and a boolean and then
have to keep in one piece. A **wedge** is a box with one end cut away to a sharp edge — a doorstop,
a buttress, a roof end. A **ramp** is the same triangular prism lying the other way, described the
way you actually think about a ramp: how `wide` it is, how `long` the run is and how `high` it
climbs. A **rounded box** is a box with its four upright edges rounded off by `radius`, which is
what stops a crate or a kiosk reading as a programmer's placeholder; `segments` decides how smooth
the rounding is, and a radius of zero gives you a plain box back. **Stairs** is a single flight,
built as one continuous surface rather than a pile of blocks: `steps` of them, filling
`total_height` and `total_depth`, with `closed_underside` deciding whether it is a solid mass or an
open flight you can see under. A **wall** is a box named for the job, with `length`, `height` and
`thickness` in the order you would say them. A **doorway** is that wall with a rectangular opening
cut through it, the opening's `width`, `height` and `offset` from centre given directly — it is
flat-headed and square, where the **arch** above is the round-headed kind, which is the only
difference between the two.

Three of those are near-duplicates of others and are worth telling apart. **Grid** is a plane cut
into squares; **plane** is the single quad, which is what a decal or a backdrop wants, and a grid is
what you need the moment you want to bend the sheet, because only interior vertices can move.
**Icosphere** and **UV sphere** are both balls: the icosphere's triangles are all much the same size,
which is what makes it the one to sculpt, bevel or bake to, and the UV sphere is laid out in
latitude and longitude, which is what makes it the one to wrap an equirectangular texture round.
**Capsule**'s `height` is its cylindrical middle alone, so the whole shape is that plus a radius at
each end.

### Figures

Below the shape grid is a **figures** list: whole assemblies, one button each, named rather than
drawn as icons because a humanoid and a blob look the same at sixteen pixels. Clicking one places
every part of the figure at once — a humanoid arrives as a head, a torso, arms and legs, each an
ordinary object you can move, scale or delete on its own. It also marks the figure the tool in
hand, exactly as a primitive button does, so the preview under the **add** row names the figure
you just placed until you press something else — as an arrangement rather than as numbers, since
a figure has no single radius or height of its own to show.

They are one undo step, not one per part. `Ctrl+Z` after placing a figure removes the whole thing,
rather than taking sixteen presses through fourteen states in which the figure is half there. The
step is named after the figure in the history panel.

Where a figure lands is decided, not incidental. The six that walk — Humanoid, Biped with tail,
Quadruped, Bird, Insect and Blob — arrive standing exactly on the ground plane. The Serpent
and the Fish are swimmers: they keep the height they were drawn at, above the grid, and are never
dropped to it.

A figure's parts are shaded by that same rule as they are placed: limbs, bodies and heads come in
smooth, and the boxy parts — hands, feet, fins, a jaw — keep their hard edges.

Torsos are shaped as body masses rather than as tubes around a bone: a pelvis is broad and shallow,
a ribcage broader still, and each one overlaps its neighbour so the body reads as one form instead
of a row of balls. That is why some parts arrive with a **scale** that is not 1 — the ellipsoid is
a scaled sphere, and the scale is part of the shape rather than something left over. Change it
freely; it is an ordinary object like any other.

Every part is a normal generated object: it keeps its generator and parameters, so the properties
panel offers a leg's radius exactly as it would for a cylinder you added yourself. A figure is a
starting point that saves you the assembly, not a special kind of object — once placed, nothing
about it is different from the same parts added one at a time. Parts are named so they never
collide, so a second humanoid does not fight the first one for `Head`.

A placed object remembers *how it was made*. Its generator and the parameters it was built with are
kept, so the properties panel offers those parameters — a cylinder's radius, height and segment
count — and changing one rebuilds the mesh. That is a single undo step, so `Ctrl+Z` takes the
object back to the shape it had rather than to some intermediate state.

A rebuild keeps your shading. Change something that leaves the face count alone — a radius, a height
— and whatever shading the object had, hand-picked or automatic, comes through untouched. Change
something that alters the faces themselves, like a segment count, and the shading is worked out
again by the same rule the object arrived with, because they are not the same faces any more.

## The viewport header

The row across the top of the viewport is where everything you change *between* clicks lives: which
element mode you are in, which transform tool you are holding, whether snapping is on, and what the
viewport is drawing. All four used to be blocks down the tool panel, which is on the far side of the
window from the model — a reach away from the thing every one of them is about.

**Mode** and **tool** are the two pill groups. They never fold away, whatever the window is doing;
everything else on the row gives up its label before it gives up its control, and if the window is
narrow enough the rightmost group moves into a `…` menu with the full words back.

**Snap** and **Falloff** are buttons that open the numbers behind them — a grid size and an angle, a
falloff radius. The button stays lit while the setting is on, so a shut popover still says what is
armed. Neither greys out while a document is saving: neither touches the document.

**Solid / Material / Wire** is how the surface itself is drawn. *Material* is the lit render — what
the object will look like. *Solid* is the albedo with no lighting, and it is what you model in: it
shows silhouette and topology without a specular highlight sitting on the vertex you are dragging.
*Wire* replaces the surface with its edges entirely.

**X-ray** makes the surface see-through, so an element behind it can be picked. It is off by default
because it changes what a click *selects* as well as what is drawn.

**Overlays** is what the viewport draws over the model — the grid, a wireframe over whichever shading
is showing, the statistics line, and the god light. **View** is where the camera looks from and how it
projects; both are described below, under [Axis views](#axis-views) and [Snapping](#snapping).

The **Grid** is a fixed size — 100 m by default, in 1 m cells, with a brighter line every ten of them
— rather than one that follows whatever is on screen, and its own field sits under the Grid row in the
Overlays popover, from 1 to 1000 m. Pressing `F` to frame the selection moves the camera only; it no
longer resizes the grid out from under you. The size is remembered across sessions, the way the switch
itself already was.

**God light** replaces the ordinary render with a single light straight down from 100 m overhead onto
a flat ground plane under the grid — a deliberately flat, shadowless look for checking silhouette and
proportions rather than the lit render you model under day to day.

### Statistics

The **Statistics** overlay puts one line in the top-left corner: objects, vertices, edges, faces and
triangles, and how many are selected in whichever element mode you are in. Every one of those numbers
was unavailable anywhere in Clay before — the outliner counted objects and nothing counted the rest —
so "is this mesh 500 triangles or 50,000" was a question the app could not answer about the thing on
screen. It is the question that decides whether a game asset is finished.

The edge figure is the count of *distinct* edges, not of face corners: a cube reads 12, not 24.

### The navigation widget

The six balls in the top-right corner of the viewport are the world axes seen from where you are
standing. The lettered ones are the positive ends — X, Y, Z — and their unlettered opposites sit
behind them, which is how the widget tells you which way round you are looking without a label. Click
any ball to look along that axis; it is the same jump `Ctrl+1`, `Ctrl+3` and `Ctrl+7` make, with the
back views one click rather than a chord.

### The hint line

The line under the viewport says what the mouse and the keyboard do *right now*, and it changes with
the mode, the tool and whether a drag is running. It is there because none of the selection verbs is
a button: `Alt`-click for an edge loop, `L` for everything connected, `Ctrl`+`+` to grow the
selection. Without the line the only way to find out edge mode can do something vertex mode cannot is
to read this chapter.

## Element modes

Every object starts as one thing you can move about. Press `1`, `2` or `3` and it becomes a mesh you
can take apart: vertices, edges, faces. `4` goes back to object mode. The buttons above the add row
say the same thing, and highlight whichever mode the document is in.

| Key | Mode | What clicking selects |
| --- | --- | --- |
| `1` | Verts | One vertex |
| `2` | Edges | One edge |
| `3` | Faces | One face |
| `4` | Object | The whole object |

The mode belongs to the *document*, not to the app, so switching tabs does not reinterpret what you
had selected in the other one.

Clicking replaces the selection, `Shift`+click adds to it and `Ctrl`+click removes from it. Clicking
the object but missing everything on it clears that object; clicking empty space with the **Select**
tool (`Q`) starts a marquee, and a marquee that ends where it started clears everything. A marquee
takes a vertex inside the rectangle, an edge only when *both* ends are inside, and a face only when
*all* its corners are — and it selects through the mesh, back face included, because that is what a
rectangle dragged over a blockout means.

`Alt`+drag always orbits, in every mode. That is the one gesture that is never reinterpreted, since
it is how you look at what you are about to click.

Right-click opens the context menu, listing exactly the operations that apply in the current mode
with the ones that cannot run greyed out. The same list drives the buttons in the tools column, so
neither can offer something the other refuses. Operations that take a number — bevel, inset, weld,
loop cut, smooth — open a small dialog with the fields and an **Apply** button, and remember what you
last used.

`Ctrl+A` selects everything in the current mode's sense of everything, `Ctrl+Shift+I` inverts it, and
`Esc` steps back: first it drops the element selection, then it leaves the element mode, then it
clears the object selection. `Delete` in an element mode deletes *faces*, never the object. In object mode it
removes every selected object as **one** undo step rather than one per object, so a single `Ctrl+Z`
brings the whole selection back. Neither
`Ctrl+A` nor `Ctrl+Shift+I` reaches a **hidden** object, in either sense of everything: hiding
something takes it out of what you are working on, so nothing you select can act on it by accident.

**Element selection is transient.** It is not saved with the document, and an undo that changes
geometry drops it — the indices it named describe a mesh that no longer exists. An undo that only
moves or renames something keeps it, because those cannot invalidate it. A properties-panel or agent
edit that rebuilds a shape from its own parameters is different again: if the face count comes out
the same the selection is carried through unchanged, and if it shrinks the selection is *restricted*
to whatever indices still exist rather than left pointing past the end of a smaller mesh.

### Selecting more than one thing

Clicking picks one element and `Shift`-clicking adds another, and until 2026-09-01 that was the whole
vocabulary — so selecting the ring of edges round a cylinder meant clicking each of them, and
selecting one of two shapes welded into one mesh was not possible at all.

`Alt`-click selects the **edge loop** under the pointer: the run of edges continuing end to end
through it. `Ctrl`+`Alt`-click takes the **ring** instead — the edges parallel to it, each one the far
side of a quad from the last. The two share a button with orbit, and are told apart by whether the
pointer moved: a press and a release in the same place selects, anything further orbits.

A loop stops where a modeller expects it to. It continues through a vertex with **exactly four
edges**, to the edge opposite the one it arrived on, and stops at anything else — a pole, like the tip
of a cone or a cube's corner, has some other number, and a loop that ran through one would wander off
round the mesh. A ring has no such rule and works where a loop does not.

`L` selects everything **joined** to what is selected. `Ctrl`+`=` grows the selection by one ring and
`Ctrl`+`-` shrinks it — shrinking peels the border off, leaving the middle of what you had, which
includes the mesh's own open border. **Select Boundary** in the context menu takes every open edge:
the border of every hole, which is what Fill Hole is about to close.

### The operations

| Mode | Operation | What it does |
| --- | --- | --- |
| Any | Extrude (`E`) | Pulls the selection off the surface and walls in the gap. It moves nothing — drag what it hands back with `W`. |
| Faces | Inset | Shrinks each face in place and rings it with the rim it vacated. |
| Faces | Subdivide | Splits each face into quads without changing the shape. |
| Faces | Flip Normals | Reverses the winding of the selected faces. |
| Edges | Bevel | Replaces each edge with a flat quad, mitring the corners where several meet. |
| Edges | Loop Cut | Rings a strip of quads with a new edge loop. |
| Edges | Bridge Loops | Joins two selected boundary loops with a strip of quads. |
| Edges | Fill Hole | Caps the boundary ring the selected edge belongs to. |
| Edges, Faces | Collapse | Pulls the selection down to a single point. |
| Verts | Weld | Merges vertices closer together than a distance you give. |
| Any | Dissolve | Removes the selection and merges what it separated, rather than leaving a hole. |
| Faces | Merge Faces | The same operation as Dissolve in face mode, under the name most people look for. |
| Any | Smooth | Catmull-Clark subdivision over the whole object. Each level multiplies the face count by four. |

An operation that cannot do what you asked says so in a toast naming the element and what to do
instead, and changes nothing. Bevel refuses a boundary edge; dissolve refuses a selection that rings
a face it does not include; fill hole refuses a pinched boundary. Those are refusals, not failures:
the alternative is geometry that looks right and is not.

**Merge Faces** and **Dissolve** in face mode are one operation with two names, and the duplication
is deliberate: "dissolve" is the modelling word and is what the vertex and edge modes do too, but
somebody who wants two faces to become one searches for "merge", finds nothing, and concludes the
editor cannot do it. Two things about what it merges are worth knowing. A selection that falls into
several disconnected blocks becomes one face *per block*, not one face overall — a face with a hole
in it is not something this editor's meshes can hold, so there would be nothing to make. And a
single face on its own is refused: there is no neighbour to merge it with.

**Extrude** is one operation in all three modes, because it means the same thing in all three. In
edge mode it grows a quad from each selected *boundary* edge — an edge with a face on each side has
no open side to grow into, and it says so. In vertex mode it extrudes the border edges between the
vertices you selected, which is the only reading available: a mesh here stores faces, not loose
wires, so a vertex on its own has nothing to extrude and says that too.

**Bridge Loops** is the one to reach for when two things need joining: select the boundary edges of
both openings and it skins a strip of quads between them, which is what makes two tubes one tube and
what closes the gap left by deleting a band of faces. Both openings must be *boundaries* — bridging
two interior rings means deleting the faces between them first, and doing that for you would remove
geometry you did not ask to lose. It also needs exactly two loops, of the same length, both open or
both closed; anything else it refuses by name, because there is no pairing to guess at. Two very
large rims it refuses as well, past about ten thousand vertices a ring: working out how the rings
line up means comparing every way of turning one against the other, and at that size the window
would sit frozen while it did.

The first operation that changes an object's topology **freezes** it. A box that has been extruded is
no longer describable as "box, size 1", so the properties panel switches from the generator's
parameters to a vertex and face count.

## Transforming

Rotating and scaling turn about the **median of the selection** — the point the gizmo is drawn at.
That was not true until 2026-09-01: the ring was drawn at the median while each object span about its
own origin, so with two objects selected the picture said "these turn about here" and the document did
something else. It also means a single object whose origin is not at the centre of its bounding box
now orbits the ring you can see, which is the same correction rather than a second change.

Four tools, on `Q`, `W`, `E` and `R`:

| Tool | Key | What it does |
| --- | --- | --- |
| Select | `Q` | Click an object in the viewport to select it. |
| Move | `W` | Three arrows; drag one to slide along that axis. |
| Rotate | `E` | Three rings; drag one to turn about that axis. |
| Scale | `R` | Three handles plus a centre handle for uniform scale. |

A drag is one undo step, recorded when you let go — not one step per frame of the drag, which would
bury everything else in the history.

The gizmos work on elements too. In an element mode they sit at the centre of what is selected
*inside* the objects rather than at the object's own centre, and dragging one moves those vertices.
That is one undo step per drag, and a drag that ends where it started records nothing at
all. **Select** (`Q`) shows no gizmo in an element mode, which is what leaves the left button free
for the marquee.

The **Move**, **Rotate** and **Scale** values are also typed directly in the properties panel, which
is the better way to place something exactly. Rotation is shown as a quaternion in `XYZW` order,
which is what every file this app writes uses. Position and scale boxes are labelled X, Y and Z;
rotation is a quaternion labelled X, Y, Z and W. Under them is a read-only **size** row: the object's
world-space width, depth and height in metres, after its transform — the number a scale of 2 on a
generator whose radius is 0.35 does not tell you.

### Moving without a handle

`G` moves the selection and `S` scales it, with no handle grabbed: the drag follows the pointer until
you click to commit or press `Esc` to put it back. Every transform in Clay used to go through
grabbing a coloured arrow, which means finding it, which means never moving an object without first
looking at the gizmo rather than at the model.

`G`, `R` and `S` also switch which transform a *running* drag is doing — "move it; no, turn it" is one
gesture rather than a cancel and a restart. The objects go back to where they started first, so a
rotate that follows a half-finished move is measured from the original position and not from wherever
the abandoned move left them.

Rotate has no letter of its own to start with, because `R` is the Scale tool and `E` is Rotate — both
taken long before this, and moving either would take away a binding you already have. `G` then `R` is
how you start one.

A keyboard drag holds no mouse button, so a **press** is how it ends: left commits, right cancels.
Everything else behaves exactly as it does under a handle drag — the axis lock, the typed value and
`Esc` are the same code and are described next.

### Locking an axis, and typing a number

The keyboard joins a drag already under way. While a gizmo is held:

| Key | What it does |
| --- | --- |
| `X` / `Y` / `Z` | Lock the drag to that axis. The same key again clears the lock. |
| digits, `.`, `-` | Type the value outright — metres for a move, degrees for a rotation, a factor for a scale. |
| `Backspace` | Take back the last character. |
| `Enter` | Commit and end the drag. |
| `Esc` | Cancel it: everything goes back where it was and nothing is recorded. |

A readout beside the cursor says what the drag currently amounts to, so `X` then `2` is "two metres
along X" with no dragging left in it. The two compose, and they are different in kind: a lock says
which *direction*, leaving the mouse in charge of the amount, while a number is the amount. That is
why a number on its own still means something — it sets the distance along whichever way you were
already dragging, and the size of a uniform scale.

### Proportional editing

**Soft falloff**, in the snap section, makes an element drag carry the geometry *around* the
selection with it, fading out over the radius, so the surface bends instead of tearing. The radius
is metres of world space and is measured from the nearest selected vertex, not from the middle of
the selection — dragging one end of a long strip fades out away from that end rather than along the
strip. Setting the radius to zero is the same as switching it off.

Several operations act on the whole selection at once, and all of them are one undo step however
many objects or copies they touch. **Duplicate** (`Ctrl+J`) makes a copy under a new name, counting
up — `Box`, `Box.001`, `Box.002`. **Bake** folds an object's position, rotation and scale into its
geometry and resets the transform to identity, which is what you want before measuring something or
exporting it into a frame that has to match.

**Array Linear...** copies the whole selection several times in a row, each copy a further step
along `x`, `y` and `z` — a negative step runs the array backwards along that axis. `count` is the
total number of instances, so `3` means two copies beside the original; it is capped at 200, a soft
ceiling on outliner rows and document size rather than on memory, since every copy shares its
source's geometry (arraying an array multiplies rather than adds, so two arrays near the ceiling can
still exceed it). **Array Radial...** spins copies of the selection around the *world* origin, not
the object's own centre — put the hub at the origin and one shape beside it, and this makes the rest
of the spokes; reach a hub anywhere else by arraying at the origin first and moving the whole result
together. `angle` is the total sweep in degrees (360 by default) and `axis` picks which world axis it
turns about. At the default 360, `count` copies land evenly spaced all the way around the circle with
no doubled spoke where the last one meets the first; short of a full turn, the copies instead reach
exactly `angle` — a quarter-turn fan of four really does have one at 90 degrees — because a partial
sweep has a real far end that a full turn does not. Unlike a mirror or a bake, a radial array only
ever changes a copy's transform, so every copy stays exactly the primitive its generator describes
and is still editable as one afterwards. Both arrays leave the whole group — originals and copies
together — selected, so arraying an array compounds instead of losing the shapes you started with.

**Place Between...** takes exactly three selected objects and puts the newest of them on the line
between the other two — select two hubs, add a strut between them, and this is what aims it instead
of you working out the angle by hand. The two earlier objects are the anchors, read by their own
**translation** (the point the gizmo sits on, not a bounding-box centre); the object added last is
the one that moves, to their midpoint, turned so its local +Y points from one anchor to the other —
every generator in Clay is built along +Y, which is why one rotation is the right answer whatever
shape you are placing. That replaces the object's own rotation rather than adding to it, so the
result never depends on which way it happened to be facing beforehand. **Fit** also stretches the
object along its own Y so it spans the gap exactly, which is what a strut wants and a hand-placed
prop usually does not; a flat shape with nothing to stretch — a Plane or a Grid lie flat in XZ and
have no Y extent at all — is moved and turned the same as anything else, with the fit itself skipped
rather than refusing the whole placement over an axis that shape has no length along.

**Align...** lines up every selected object's **world box** — its visible edge or middle, not its
pivot — along one axis: `min` and `max` line up the lowest or highest edge, `centre` lines up the
middle of the whole group. Two boxes of different sizes sharing a translation do not share a centre,
which is the whole reason this reads the geometry rather than the transform. **Distribute...** spaces
three or more selected objects evenly along one axis, gap for gap between neighbouring boxes, holding
the two extreme objects fixed — with fewer than three selected there is no gap to distribute against,
so the row is disabled rather than a no-op. **Drop to Ground** rests each selected object's own world
box on `y=0`, one at a time rather than as a group, so an object already on the ground and one
floating three metres up both land correctly in the same press. **Snap to Grid...** rounds every
selected object's translation onto a grid of the given step, each axis independently — unlike **Snap**
below, this acts once on whatever is already selected rather than following a live gizmo drag.

## The outliner

Every object in the document, newest at the top. Click to select, `Ctrl`-click to toggle one and
`Shift`-click to take a range. The filter box above narrows the list by name.

The eye on each row hides an object, and a hidden object does not render, does not export and cannot
be clicked in the viewport. **Solo** above the list hides everything *except* what is selected and
**Show all** brings them back — each is a single undo step, so `Ctrl+Z` is a third way out.

Rows are dragged to reorder them, which matters because display order is the order the objects come
out in an exported GLB. Reordering is switched off while the filter box has something in it: the
rows on screen are then a subset, so there is no honest answer for where a drop between two of them
lands in the real list.

Right-clicking a row selects it and offers **Rename**, **Duplicate**, **Solo** and **Delete**.
Double-clicking a name renames it in place.

The list is a tree. An object with children is indented under its parent and has an expander beside
it; collapsing one hides its subtree, except while the filter box is in use, where a match buried
under a collapsed group still has to be findable. Where a row is dropped decides what the drop
means: onto the middle of a row makes the dragged object that row's child, while between two rows
reorders as before.

The padlock beside the eye locks an object. A locked object cannot be moved, edited or deleted, and
viewport clicks pass through it to whatever is behind — which is the point, when the thing you keep
selecting by accident is the floor. It can still be selected here in the outliner, which is how you
unlock it again. The eye and the padlock are deliberately different: hiding is about what you can
see, locking about what you can change.

## Parents, groups and tags

**Parenting** makes one object follow another. Drag a row onto another in the outliner, or select
the objects and press **Parent to Last**: everything else in the selection becomes a child of the
topmost selected object. Nothing moves when you do it — the child keeps exactly the place it had —
and from then on moving the parent moves the whole subtree. **Clear Parent** frees an object again,
also without moving it.

A child's position, rotation and scale are measured *relative to its parent*, so the Properties
fields are labelled "local" once an object has one. That is why a wheel at local zero sits at the
car's origin rather than the world's. An object can never become its own ancestor; a drop that would
make a loop is refused and says so.

**Group Selected** is parenting with a holder made for you: an empty object appears at the centre of
what you selected, with everything parented to it. The empty has no geometry of its own — it draws
nothing and exports as a bare node — so it is purely a handle for moving a set of things as one.
**Ungroup** dissolves it and leaves the children where they stand.

Deleting a parent never takes its children with it: they attach to the parent's own parent, keeping
their place in the world.

**Tags** are free-form labels in Properties — `blockout`, `greybox`, `export`, whatever the job
needs. An object can carry any number of them, and the outliner's tag box narrows the list to the
ones that match, which is how you work on a subset of a big scene without hiding everything else by
hand.

## Separating and origins

**Separate Loose Parts** splits an object wherever its geometry is not actually connected, **Separate
by Material** splits it one object per palette slot, and **Separate Selection** (`P`, in face mode)
pulls the selected faces out into an object of their own. Each is one undo step, each piece keeps
the original's place, parent and modifiers, and an object with nothing to split says so rather than
making a pointless copy.

An object's **origin** is the point it rotates and scales about, and where its Properties position
is measured. Four operations move it without moving the geometry: **to Bounds** (the centre of the
object's box), **to Base** (the middle of its underside, which is what makes an object sit on a
floor cleanly), **to Selection** (the centre of what is selected in an element mode) and **to World**
(the world origin). Children stay exactly where they are while the pivot moves under them.

One thing to know: a Mirror modifier's plane is the object's own origin, so moving the origin moves
the mirror with it. That is the modifier working as described rather than a surprise, but it is the
one case where moving a pivot changes what you see.

## Measuring

The hint line under the viewport answers the question the selection implies. Two selected vertices
show the distance between them; three show the angle at the middle one; a face selection shows the
total area; and in object mode a selection shows its volume. Everything is measured in world space,
so a scaled parent is accounted for rather than ignored.

## Merging objects

**Merge Objects...** (`Ctrl+M`, object mode, two or more selected) turns several shapes into one.
The survivor is the **topmost selected object in the outliner** — it keeps its name, its transform
and its default material — and everything else is carried into its frame, appended to its geometry
and removed from the document. That is one undo step: a `Ctrl+Z` that put one absorbed object back
while the survivor still carried the merged geometry would show you a shape existing twice.

Only objects you can see are merged. A hidden object stays hidden and untouched even when it is
selected, and **Merge Objects...** greys out unless two visible objects are selected — hiding
something means it is not part of what you are working on, and a merge is the one operation where
absorbing an unseen object would leave no trace of having done so.

The dialog asks for a **weld distance**, in metres of world space. Vertices closer together than
that are merged into one at their centroid, which is what makes two shapes that *touch* come out as
a single continuous surface rather than as two shells sharing a plane. Setting it to zero keeps
every vertex, which is the honest answer for parts that are meant to stay separate inside one
object. The distance means the same thing whatever the survivor is scaled to: 1 mm is 1 mm on the
ruler, not 1 mm in whatever units the survivor's own transform happens to work in.

The weld is applied to the **whole merged result**, not only where the shapes meet. That is what
lets three objects touching at one point come out joined, and it costs nothing for ordinary work —
texture coordinates are stored per face corner, so a weld carries UV seams through untouched, and
nothing else in Clay leaves two vertices sitting at one position. The exception worth knowing:
merge at zero to keep two parts as separate shells, then merge *that* object with a third at a
non-zero distance, and the shells you kept apart are welded together. Merge the third one first, or
keep the parts as separate objects until last.

It is a weld and not a solid union. Geometry inside an overlap is kept rather than cut away — for
that, use **Union Objects...** (`Ctrl+Shift+M`) below — the merge dialog says so too.

A merged object is no longer what a generator would build, so its generator claim is dropped along
with the merge — the properties panel stops offering the size field that would have rebuilt a
pristine box over your work. That drop is part of the same undo step.

## Union objects

**Union Objects...** (`Ctrl+Shift+M`, object mode, two or more visible objects selected) is the
other way of turning
several shapes into one, and it answers a different question. A merge keeps everything: two
interpenetrating cubes come out as one object still carrying both sets of interior walls, which
z-fight, get exported, and mean the result is not a closed solid. A union cuts those walls away, so
what you get back is the single solid the two shapes look like they describe.

Everything about *which* object survives is the same as a merge: the topmost selected object in the
outliner keeps its name, transform and default material, everything else is carried into its frame,
hidden objects are left alone, and the whole thing is one undo step with the generator claim dropped
inside it. There is no weld distance, because a union has nothing to weld — it recomputes the
surface rather than joining two of them.

Three things it costs, which are why the merge is still here:

- **Texture coordinates are lost.** A union recuts every face it touches, so no corner of the result
  corresponds to any corner of the inputs and there is nothing honest to carry across. Unwrap the
  result afterwards.
- **Quads become triangles.** The solver works in triangles and hands back triangles, so a union of
  two clean quad boxes is a triangle soup — worse to subdivide and worse to edit by hand.
- **Every object must be a closed solid.** "Inside" is only meaningful for something that encloses a
  volume, so an object with holes or loose faces is refused by name rather than guessed at. Fill the
  holes first, or merge instead.

Objects that do not touch at all are a perfectly good union: you get one object holding two separate
shells, exactly as a merge at weld distance zero would give you.

**Mirror X / Y / Z** reflects the object across a plane through its own origin, in place — the object
you had is now its own mirror image, and nothing new is added to the document. It is baked into the
mesh rather than expressed as a negative scale, and that is deliberate: glTF readers disagree about
whether a negative scale flips the winding order, so an asset that used one would render correctly
here and inside out in some engines, with nothing in the file to explain it. Mirroring here rebuilds
the geometry and reverses the faces, which is true under every reader.

**Mirror Copy...** is the other half of mirroring, and what the per-object Mirror X/Y/Z cannot do:
it *duplicates* the selection and reflects the copies across a plane you place anywhere in *world*
space, perpendicular to the axis you choose, at the `offset` you give it — which is what mirroring a
limb across a body's centre-line means, with the original left exactly where it was. Like Mirror
X/Y/Z the result is baked into the mesh for the identical reason, so a mirrored copy is no longer
what its generator would build and its size field disappears from Properties. It leaves the copies
selected alongside the originals, the way both Arrays do, so a second Mirror Copy across a different
axis doubles what the first one made: one table leg, mirrored across X and then across Z, is four.

## Modifiers

A modifier changes how an object looks without changing the mesh you edit. Each object carries a
**stack** of them, drawn in the **Modifiers** section of the properties panel, and what the viewport
shows, what gets exported and what the game check measures is the mesh run through that stack from
top to bottom. The mesh underneath, the *base*, is still the one element modes select and edit, so
you can model half a character with a Mirror modifier on and watch the other half follow.

**Add modifier** offers ten:

- **Mirror** adds the mesh's reflection across the X, Y or Z plane through the object's origin.
  Vertices within the weld distance of that plane are merged, so a half model closes along its
  centre line.
- **Array** repeats the mesh a number of times, each copy moved by the offset you give. A weld
  distance above zero joins copies that touch.
- **Radial Array** repeats it around the object's origin over the angle you give. A whole turn
  spaces the copies evenly around the circle; a shorter arc puts the last copy at the arc's end.
- **Solidify** gives a surface thickness, closing its open edges with a rim. The offset decides
  whether the thickness grows inward, outward or both ways.
- **Bevel** bevels every edge sharper than the angle you give, by the width you give.
- **Subdivide** smooths with Catmull-Clark, one to three levels.
- **Weld** merges vertices closer than a distance.
- **Triangulate** turns every face into triangles, for an engine or a tool that wants them.
- **Smooth** relaxes the vertices towards their neighbours without adding any, leaving open edges
  where they are.
- **Boolean** unions, subtracts or intersects another object you choose. The other object can be
  hidden, and usually is: a hidden cutter still cuts, and it is not exported.

Each row has a checkbox that turns it off without losing its settings, arrows that move it up or
down the stack, **Apply** and **Remove**. Order matters: a Mirror and then a Subdivide smooths across
the centre line, while the other way round leaves a crease there. Every change is one undo step,
and typing a number is one step however many keys it took.

A modifier that cannot run — a Boolean with no target, or one whose target was deleted — shows why in
the warning colour under its row and is skipped. The ones after it still run. A Boolean may not name
an object whose own stack cuts with this one, directly or through others, because neither could be
worked out first; Clay refuses that choice instead of accepting it.

**Apply** makes a modifier permanent. The mesh becomes what the stack produced up to and including that
row, and those rows leave the stack. **Apply Modifiers** in the object menu does the whole stack for
every selected object. A modifier that is showing an error cannot be applied, because the result
would not be the mesh you were looking at. Editing the mesh never touches the stack. A generator
object keeps its modifiers when its first edit drops the size fields.

**Merge Objects** and the three Booleans in the object menu work on what you see: each object's stack
is applied first, and the survivor has an empty stack afterwards.

## Axis views

`Ctrl+1`, `Ctrl+3` and `Ctrl+7` snap the camera to the front, right and top views — the numbers
Blender puts them on, so the muscle memory carries over. Hold `Shift` for the opposite view, which
is how three keys cover six. `Ctrl+5` toggles an **orthographic** projection, where parallel edges
stay parallel and there is no perspective foreshortening; it is what you want for lining two things
up. The **View** menu on the viewport header lists all six views and the orthographic toggle by
name, and the navigation widget in the corner does the same with a click.

An axis view changes the *angle* only. It keeps the distance and whatever you were looking at,
because reframing would lose the part of the model you were about to line up. Switching to
orthographic keeps the scale at the point you are looking at, so it reads as a change of projection
rather than a jump cut.

These are Clay's keys, not the app's — which is why switching mode is a click on the rail or a
command in the palette rather than a digit. A global binding is checked above Clay and would take
the key from it permanently.

## Snapping

**Snap** quantises a drag to a grid — a distance for moving, an angle for rotating. Both are set
beside the toggle, and setting either to zero turns that half off rather than snapping everything to
the origin.

Snapping applies to gizmo drags only. A number typed into the properties panel is used exactly as
typed, because you already said what you meant.

**Snap to vertex** is a separate switch beside it, and the two answer different questions: the grid
puts things on round numbers, this puts them exactly *there*. While moving, the drag lands on the
vertex under the cursor. The vertices being moved are never candidates, so a drag cannot snap onto
itself; and locking an axis or typing a value during the drag overrides it, because at that point
you have said where the thing goes.

**Snap to edge** and **Snap to face** are two more switches beside it, and they answer the same
question at coarser grain: the drag lands on the nearest point along the edge under the cursor, or
on the point where the cursor's ray meets the face under it. They are tried finest first — vertex,
then edge, then face — so a corner still wins when the cursor is over one. As with vertex snapping,
whatever is being dragged is never its own target.

## Texture coordinates

Every primitive comes with texture coordinates already on it — a box's six faces, a cylinder's band
with its two caps tucked into the corners, a sphere laid out pole to pole, a torus wrapped both
ways. They are laid out so that no two parts of one shape sit on top of each other in the square,
because a texture cannot be baked onto a layout whose pieces overlap.

**Box Unwrap** recomputes them for whatever is selected, projecting each face along whichever axis
it points along most. It is the same projection the box primitive uses, and it is the right answer
for the blockout geometry Clay makes: instant, predictable, and with no way to fail. It is
deliberately not a conformal unwrap — that is the right tool for an organic surface and the wrong
one for a shape made of flat panels, and it fails in ways a modelling panel has nothing useful to
say about.

Two things follow from it being a *projection*. Faces pointing opposite ways share the same square,
so a texture applied to a box appears on both the front and the back (mirrored, so lettering reads
the right way round on each). And unwrapping does not freeze a generator: coordinates are not
geometry, so an unwrapped box is still a box and editing its size still rebuilds it.

### Seams and unwrapping by them

A **seam** is where you tell Clay to cut the surface open before flattening it, the way a sewing
pattern is cut. Select edges and press **Mark Seam**; **Clear Seam** takes them off again. Seams
belong to the object rather than to its texture coordinates, so they survive being unwrapped again
and are saved with the document, and an edit that changes the mesh drops the ones whose vertices are
gone rather than letting them point somewhere arbitrary.

**Unwrap Seams** flattens the object by cutting along them, which is what gives an organic shape a
layout with far less distortion than a projection can manage. A closed surface with no seam cannot
be flattened at all — there is nowhere for it to open — and Clay says exactly that rather than
producing a tangle. For anything dense, **Smart Unwrap** through Blender is the better tool.

### The UV view

The **UV** panel shows the selected object's texture layout: its islands, which of their edges are
seams, and the element selection highlighted inside them. The wheel zooms, the middle button pans,
and dragging either box-selects islands or moves whichever ones are selected. `E` and `R` rotate and
scale what is selected — the same letters the viewport uses — following the mouse until you click to
keep the result or press `Esc` to drop it. The fields beside the canvas do the same thing to an
exact number. Either way each island turns about its own centre.

Faces that overlap another island are tinted, and so are badly stretched ones — overlap means two
parts of the model would be painted with the same patch of texture, and stretch means a texture will
look squeezed there. **Pack Islands** lays everything out inside the square with a margin you
choose, keeping the relative sizes so nothing changes its texture density. **Texel Density** sets
that density directly: how many pixels of a given texture size each metre of surface gets, which is
what keeps two props next to each other in a game looking equally sharp.

## Checking a mesh

The properties panel has a **mesh check** under the generator's parameters. It measures the selected
object against the defects a mesh can carry without anything noticing: holes, non-manifold edges
(three or more faces on one edge), inconsistently wound faces, duplicate faces and vertices no face
uses. Each finding is a button — clicking it switches to the element mode the defect lives in and
selects exactly the offenders, so "3 non-manifold edges" becomes three edges you are looking at.

It runs when you press the button and not before. Building the tables it needs is proportional to
the size of the mesh, which is fine once and unacceptable sixty times a second, so the answer is
kept and marked *edited since the last check* the moment anything changes the geometry.

None of it is a verdict. An open sheet is a perfectly good mesh and so is a plane with no thickness;
what the check tells you is what is there, and whether that matters depends on where the asset is
going. A game engine will usually want a closed mesh; a decal will not.

## Cleaning up and reducing

Three object-mode operations repair and lighten whole objects. They work on every selected object
at once, and each is one undo step however many objects it touched.

**Clean Up** fixes the defects the mesh check finds and nothing else. In order, it removes faces
with no area, merges vertices closer than the distance you give, removes faces that repeat another
face's corners, and drops vertices no face uses. It then makes every face wind the same way as its
neighbours and turns each closed shell outward. **Fill holes** is off by default, because an open
edge is sometimes the point: a cape, a leaf, a decal. An object with nothing wrong is left exactly
as it was, generator and all, and the toast says so.

**Recalculate Normals** does only the last step: consistent winding, closed shells facing out. It
works on whole objects rather than selected faces, because which way a face should point is a
property of the shell it belongs to, and a handful of selected faces is not a shell.

**Decimate** reduces the triangle count to the fraction you ask for, using the same simplifier the
library's triangle retarget uses. It triangulates the object and keeps each face's material.
**Keep seams** holds open edges and texture seams in place, so a texture still lands where it did.
**Aggressive** lets it change the shape more to reach the count, for when the careful mode stops
short, and the toast tells you when it did.

Decimate runs in the background, so the window stays responsive on a large mesh. If you edit an
object before its result comes back, that object's result is thrown away rather than pasted over
your edit, and the toast names it.

## The game check

The **Game check** section at the bottom of the side panel measures the whole document against a
target: Godot desktop, Godot mobile, Unity, Unreal or WebGL. Pick one and press **Check**.

Each row is one question with a pass, warn or fail mark: triangle and vertex budgets, how many
materials and how much texture memory, whether a textured object has texture coordinates, the mesh
defects Clean Up fixes, faces pointing the wrong way, open edges, negative or uneven scale, whether
the asset is a plausible size, and whether it stands on its origin. A row that has a remedy carries a
**Fix** button. It selects the objects the row names and runs that remedy: Clean Up, Recalculate
Normals, Decimate, Bake Transform, Drop to Ground or Box Unwrap.

The budgets are judgements, not engine limits. An engine will import a mesh well past any of them;
the check says where an asset is heavier than a game usually wants. Only two rows can fail outright:
an empty document, and a textured object with no texture coordinates to put the texture on.

Like the mesh check, it runs when you press the button, and it says *out of date* once the document
changes.

## Retopology, unwrap and bake

Three operations hand the selected objects to Blender and wait for the answer. They need the `rig`
extra installed; without it each one is greyed out and says so. Each runs in the background, so the
window stays responsive, and each throws its result away rather than pasting it over an object you
edited while it was working.

**Retopologise** rebuilds the selected objects as an even quad mesh at roughly the triangle count
you ask for. It is what turns a sculpted or reconstructed blob into something you can edit, and the
answer replaces the mesh you had, so the object stops claiming to be a generated shape. Modifiers
stay on top. Where the quad solver cannot cope — an open or self-intersecting mesh usually — it
falls back to a plain reduction, and the toast says which you got.

**Smart Unwrap** lays out texture coordinates by cutting the mesh where it bends, which is a better
starting point than the Box Unwrap above for anything organic. It changes no geometry, so a
generated shape keeps its size fields.

**Bake Detail** takes the fine detail of the objects you select and paints it onto the simplest one
as textures. The target is the topmost selected object in the outliner, exactly as merging works;
everything else selected is a source. You choose the texture size, how far the bake reaches from
the surface, and which of base colour, roughness and normals to bake. The target needs texture
coordinates first, from either unwrap.

The usual path for a reconstruction is all three in order: retopologise to something editable, unwrap
it, then bake the original's detail onto it.

## Materials

Every object points at a slot in the document's material palette, chosen in the properties panel.
The slot's **base colour**, **metallic** and **roughness** are edited there too, and the change
reaches every object using that slot at once — which is the point of a palette rather than a
material per object.

**Add** appends a new slot and **Remove** drops one — but only a slot no face is using, and the
panel says how many faces are in the way when it will not. Reassigning those faces to some other
slot is the alternative, and it is a silent change to how part of the model looks. A slot is an
index that every face names, so adding always appends rather than inserting; removing one renumbers
the slots above it, and an undo puts the numbering back.

Beside those three are the rest of what a glTF material carries: an **emissive** colour for
something that glows, **double-sided** for a surface an engine should not cull from behind, and an
**alpha mode** — opaque, masked at a cutoff you set, or blended. Each texture slot can be assigned
a PNG from disk or cleared; the file is decoded off the frame thread, so a large map does not stall
the window.

A material can be saved to the **material library**, which lives outside any one document. Saved
materials are listed by name and applied to the selected objects in one step, which is how a set of
props ends up sharing one look without copying numbers between files. Each saved material is its own
small file with its textures beside it, so a half-written or hand-edited one is skipped rather than
taking the rest of the shelf down with it.

**Shade Smooth** and **Shade Flat** set how faces are shaded — the whole object in object mode, the
selected faces in face mode. **Shade Auto** decides per face from the angle between neighbours: a
sphere or a torus comes out smooth, a box stays flat.

## Collision and engine profiles

A game engine does not use the shape you modelled to work out what the player bumps into — that
would be far too expensive — so an asset ships with a simpler **collider** beside it. Clay fits one
from the selected objects: **Box**, **Sphere**, **Capsule**, **Convex Hull** or **Compound** (one
hull per disconnected part). The collider arrives as a child of the object it was fitted to, so
moving the object moves it too, and it is not counted against the triangle budget in the game check,
because it is not part of what gets drawn.

Which engine you are exporting to is a setting, and it decides two things. Colliders are **renamed
on the way out** to whatever that engine recognises — `UCX_Crate_00` for Unreal, a `-colonly`
suffix for Godot — while the names in your document stay as you wrote them. And an OBJ export is
converted to the engine's axis and scale convention. A GLB is deliberately left alone: every engine's
glTF importer does that conversion itself, and doing it twice is the classic way to end up with an
asset lying on its side at a hundred times the size.

The game check reads the same engine choice, so its budgets, its collider rows and the naming all
come from one place rather than three.

One thing about Shade Auto is worth knowing before it surprises you: **a capped cylinder comes out
entirely flat.** Shading here is a per-face flag, so a face is smooth only when it has no sharp edge
at all — and every side quad of a cylinder meets a cap at a right angle. That is also the right
answer for this renderer rather than a gap: smoothing the band while the caps stayed flat would
average the cap normals into the rim and round the very edge the caps are there to define.

Changing a generator's numbers rebuilds the mesh from nothing, and what a rebuild keeps depends on
what changed. A field that only moves a radius, a height or a position leaves the same faces in the
same order, so a hand-picked Shade Smooth and a hand-painted per-face colour both come back exactly
as they were. A field that changes how many faces there are — a segment or ring count — does not:
those are not the same faces any more, so the mesh repaints to the object's own material slot and its
shading is re-derived by the same rule a shape gets when it is first placed, rather than either one
quietly reverting to grey and flat. That is true whichever door asked for the rebuild — typing into
this panel or an agent building in Clay over MCP behave identically here.

Clay paints no textures — but it **carries** them. A material that arrived with an imported asset
keeps its baked maps: they render in the viewport, they are stored in the `.wblk`, and they are
written back into an exported GLB. The properties panel shows which slots a material carries as a
read-only line, because there is nothing here that could replace one and offering a control that
looked like it could would be promising a feature that does not exist.

For an asset modelled here from scratch, the material factors are what the exported asset carries,
and they are enough for a blockout: what the pipeline is for is turning that blockout into something
with a surface.

## Saving

`Ctrl+S` saves the document as a `.wblk` — a zip holding `scene.json` (the objects, their
transforms, their generator parameters and the palette) plus one compressed mesh per object. The
JSON half is sorted and indented so it is readable and diffable, and two saves of an unchanged
document produce byte-identical files.

A saved document keeps every object's identity, so an undo recorded before the save still lands on
the object it was made against after you reopen it. It also keeps the **camera**, so reopening a
document puts you back where you were looking rather than framing it afresh. A file written before
that key existed still opens, and simply gets framed.

Modifier stacks are saved with the objects, settings and all, so a reopened document is still
editable underneath. Every document is now saved in a format that builds from before modifiers
existed cannot read. Those builds refuse it by name rather than opening it without the stacks.

## The two ways out

Clay has two ways to turn a document into an asset, and they do genuinely different things.
Choosing between them is the whole reason both exist. A third door writes a plain file.

**Export to the library** puts the *exact* geometry in the library as an ordinary asset. It is a
finished model row from the moment it lands, so it inherits everything the rest of the app does to a
mesh: rigging, posing, sprite sheets, the triangle retarget, and the STL, OBJ, FBX, collision and
texture exports. Use it when the shape you modelled is the shape you meant.

**Export GLB** and **Export OBJ** write the document straight to a file you choose, for handing to
another tool. OBJ writes a `.mtl` of the same name beside it with each material's colour. Neither
touches the library, and neither counts as saving the document.

**Make 3D** renders the document flat, on a plain background, with no grid and no gizmos, and
hands that picture to the reconstruction stage. What comes back is *not* your geometry — it is a
reconstruction that used your geometry as a suggestion, with surface detail and irregularity nobody
modelled. Use it when the blockout is a silhouette and proportion study rather than a final shape.

A built asset cannot be rerolled or remeshed. There is no generator behind it: a new seed would
change nothing, and there is no reference image to reconstruct from. The way to get a different mesh
is to open the document and change it.

## Importing an asset

**Import Mesh** in the side panel opens a `.glb`, `.obj`, `.stl` or `.ply` file, and dropping one on
the window while Clay is on screen does the same. The library card's overflow menu has **Open in
Clay** for any finished model.

Beside the button are two choices a drop uses too: the file's **units** (metres, centimetres,
millimetres, inches or feet) and which way is **up** (Y or Z). Clay works in metres with Y up, so a
file from a Z-up tool imported as Y-up lies on its back, and one in centimetres arrives a hundred
times too big.

An OBJ keeps its faces as they were written, quads and larger included, and keeps its texture
coordinates. Each `o` or `g` line starts a new object, and `usemtl` picks the material. The colours
come from the `.mtl` file the OBJ names, when it sits in the same folder; without it the materials
arrive grey. An OBJ that Clay exported comes back with the colours it left with. STL and PLY carry
neither materials nor texture coordinates, so each arrives as one grey object with its triangles
joined back into one surface.

**Open in Clay** prefers the document you authored. If the asset was exported from Clay, its
`build.wblk` sidecar is reopened — objects, names, generator parameters and all. If it was not, the
served `model.glb` is imported instead: that is the optimized, grounded mesh, not the raw
reconstruction.

An imported mesh comes in as one object per material, with its vertices merged back together
bitwise. An exporter splits a vertex wherever a normal or a texture coordinate disagrees, and those
split copies are bit-identical in position, so merging them is exact — there is no tolerance to
choose and no chance of welding two features that are a hair apart. If you *want* tolerance welding,
that is what **Weld** is for.

Texture coordinates survive, because Clay stores them per face corner: a seam is two corners at one
vertex, which is exactly what the exporter split produced. Smoothing is a heuristic — a face whose
corner normals agree with its own geometric normal was flat-shaded, and one where they do not was
smooth.

Two things are refused rather than half-done. A **rigged** GLB, because Clay has no skinning and
editing it would drop the rig; open it in Create instead. And a mesh past two million triangles,
because the editor holds every mesh twice per undo step. Past two hundred thousand it asks first,
since every edit rebuilds the whole mesh and you should know that before you press Extrude.

## Where the files go

An exported asset is an ordinary job directory, and the document that produced it is stored beside
the mesh as `build.wblk` (the on-disk name predates the rename). That copy is never served or
downloadable — it exists so that reopening a built asset brings its objects back instead of one
frozen mesh — and it goes away with the job when the job is deleted.

Dropping a `.wblk` on the window while Clay is on screen opens it, and dropping a `.glb` imports it
— see [Importing an asset](#importing-an-asset).

Every binding is listed in [Keyboard shortcuts](38-shortcuts.md).
