# Modelling

Clay is a small low-poly modeller, in the spirit of picoCAD: primitives, element editing and a material
palette, saving to its own `.rblk` format and exporting into the library like anything else.

It needs no GPU and no weights. It is also the answer to a question the generators raise — what to
do when reconstruction gets a shape *nearly* right, or when the thing you want is a crate and asking
a diffusion model for a crate is the long way round.

## Getting a document

`Ctrl+N` starts one, `Ctrl+O` opens a `.rblk`; both, **Open Recent** and the exports are also in the
**File** menu. A finished mesh in the library has **Open in Clay** in
its menu, and a `.glb` dropped onto the window is imported.

Two limits on import, both about memory rather than taste: past about 200,000 triangles Clay asks
first, and past two million it refuses. The editor holds two copies of a mesh per undo step, so a
model that fits comfortably in a viewer does not necessarily fit comfortably in an editor.

A rigged `.glb` is refused outright. Clay has no skinning, and silently dropping the rig on import
would be worse than saying so.

## Primitives

Fifteen, in two groups. The primitives: box, plane, grid, cylinder, cone, UV sphere, icosphere,
torus and capsule. The game shapes, for blocking a level out: wedge, ramp, rounded box, stairs, wall
and doorway. Place one and its parameters — radius, height, segments — stay live in the properties
panel, so a cylinder can become a thinner cylinder without being rebuilt by hand. Anything else is
built from these: a column is a cylinder. Clay no longer builds lathes, sweeps, tubes, arches, columns
or pyramids. The rail down the left edge of the viewport places the shapes: a few as one-click buttons,
and a **+** that lists all fifteen by name.

**Until it freezes.** The first edit that changes topology — an extrude, an inset, a merge of faces —
discards those parameters permanently, and the panel switches to a plain vertex and face count. It
has to: once you have extruded a face, there is no "regenerate this cylinder" that could keep your
extrusion.

Nothing warns you, because it is not a mistake. But it is the reason a cylinder whose height you
wanted to tweak *after* modelling on it cannot be tweaked.

## Selecting and transforming

`4`, `1`, `2`, `3` switch between Object, Vertex, Edge and Face modes. Selection is not undoable, on
the reasoning that clicking a different object should not dirty a document. A selected object is
outlined in orange.

The tools are `Q` select, `G` move, `R` rotate, `S` scale; the last three also start a drag at once
when something is selected, with no handle to grab. During a drag you can press `X`, `Y` or
`Z` to lock to an axis and *type a number* to set the amount exactly — the two compose, so `X` then
`2` moves two metres along X. `Esc` cancels the drag with nothing recorded. `H` hides the selection,
`Shift`+`H` hides everything else and `Alt`+`H` brings it all back.

One undo step per drag, committed on release, not one per mouse-move.

Snapping is a grid for moving and an angle for rotating, and it applies to drags only. Set
either to zero and that half is off. In an element mode each moved vertex lands on a grid point. The
details are in [Clay](30-clay.md), under *Snapping*. The grid is all there is to snap to, with no snap
to another object's vertex and no soft falloff: to make two things touch, type their positions.

Camera: `Alt`-drag always orbits, the middle button (or `Shift` and the middle button) pans, and the
wheel zooms toward the pointer. `F` frames the selection without turning the view. `Ctrl+1`, `Ctrl+3`
and `Ctrl+7` snap to axis views (add `Shift` for the opposite side; the numeric keypad does the same
without `Ctrl`), and `Ctrl+5` toggles orthographic.

## Editing the mesh

The mesh operations live in one registry, which is why the context menu, the menu strip and the
keyboard always offer exactly the same list. Right-click is the easiest way in.

The ones you will use: **Extrude** (`E`), **Inset Faces** (`I`), **Subdivide**, **Merge Faces**,
**Weld**, **Triangulate Faces**, **Flip Normals** and **Delete**. It is a short list on purpose: a
low-poly model is made of a few hundred faces you can see, and a marquee, `L` for everything joined
and these eight reach all of them.

## Merging objects

**Merge Objects** (`Ctrl+M`) welds several objects into one. Vertices within a distance of each
other are joined. Everything else survives — including, if the shapes interpenetrated, the interior
walls now buried inside your model, which will z-fight where they touch the surface and which no light
will ever reach. Clay has no boolean to cut them away; for a low-poly model the buried faces cost
nothing you can see, and where they do, delete them in face mode.

Often you do not want a merge at all. **Group Selected** keeps the parts as separate objects under one
empty, so the set moves as one thing and each part is still a box you can resize.

## Materials and UVs

One material palette per document, with slots referenced per face. A slot is a name, a base colour,
an optional base-colour texture, double-sided and cutout — the picoCAD palette, a flat colour or a
picture. Clicking a swatch in the palette strip under the viewport repaints the object; in Face mode it
paints the selected faces instead, and **Assign Material...** does the same from the menu.

Clay does not paint a texture itself — Inker does, and Clay opens it for you. **Add texture** on a slot
makes a blank 32, 64 or 128 pixel picture in the slot's colour (box-unwrapping any object that had no
UVs), and **Edit texture in Inker** opens it there with the sixteen PICO-8 colours; what you paint
lands back in Clay, one undo step per return. Textures Clay makes render crisp, in the viewport and as
a NEAREST sampler in the GLB, and **Export OBJ** writes them as PNGs beside the file with a `map_Kd`
line. The whole round trip is in [Clay](30-clay.md), under *Texturing*.

Adding a material always appends and never inserts, because inserting would renumber every face
assignment in the document. Removing one is only allowed when nothing uses it, and a document always
keeps at least one material — with a single material left, Remove greys out and says so.

Every primitive already has sensible UVs. **Box Unwrap** re-projects an object planar-per-face by
dominant axis — quick and not conformal, which is the right trade for a blockout and the wrong one
for a hero asset. It is the only unwrap Clay has, and **Pack Islands** is the only other UV operation.

**Shade Smooth** and **Shade Flat** control normals, on the whole object or on the selected faces. A
shape is placed with an angle rule already applied, and be aware that a capped cylinder shades entirely
flat under it, which looks like a bug and is the rule doing what it was told.

## The two ways out

**Export to the library** mints an ordinary finished asset from your exact geometry. Nothing is
reinterpreted. It unlocks everything downstream — rigging, posing, sheets, every export format —
none of which needs to know Clay exists. **Export GLB** and **Export OBJ** write a plain file to
somewhere you choose instead, for another tool, and touch nothing in the library.

An asset built in Clay can never be rerolled or remeshed, because there is no reference image behind
it. The app knows and the buttons are absent rather than broken. Clay does not turn a blockout back into
a generation, either; the reconstruction pipeline starts from a picture, in [Create](22-generating-references.md).

Clay does not reduce triangle counts or rebuild a mesh as quads. A budget on a finished asset is
applied downstream, in the retarget panel, and you can read the triangle count you are working to on the
**Statistics** overlay as you model.

## Round trips

Export to the library, then use **Open in Clay** on the resulting card, and you get your objects back —
names, generator parameters and all. That works because the export keeps a `.rblk` beside the mesh.
Opening a mesh that was *not* authored in Clay instead gives you one frozen object per primitive
of each node, which is the honest answer to "what were the objects in this file" for a file that never had any.

## Try it

1. `Ctrl+N`, and place a box and a sphere so they overlap.
2. Run **Merge Objects** with a small weld distance, orbit into the overlap, and find the interior
   wall.
3. Undo, then select both and use **Group Selected** instead; move the group, and note that each part is
   still its own object.
4. In Face mode, select the top of a box, **Inset**, then **Extrude** upward to make a chimney —
   and watch the properties panel drop the box's parameters the moment the inset lands.
5. **Export to the library**, then **Open in Clay** from the new card, and confirm your objects came
   back.

## What to read next

[Rigging and posing](08-rigging-and-posing.md) — giving a mesh a skeleton, and the two traps that
catch everyone.
