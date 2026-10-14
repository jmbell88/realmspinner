# Putting it in a game

The last tutorial chapter, and the one about leaving. Everything Realmspinner makes is meant to be
imported somewhere else, and this covers what the formats are, what has already been done to make
them import cleanly, and — honestly — which of the interoperability claims have actually been
checked.

## Exports are derived, not stored

Almost nothing in a job's directory is generated up front. Ask for an FBX and it is produced then and
cached; never ask and it never costs you anything. So the list below is what is *available*, not what
is sitting on your disk.

**From a mesh:**

| File | What it is |
| --- | --- |
| `model.glb` | The one to use. Optimised, centred and grounded. |
| `source.glb` | The raw reconstruction, kept as evidence. |
| STL | Geometry only. For printing rather than for engines. |
| OBJ | A zip, since OBJ is never one file. |
| FBX | For pipelines that insist on it. |
| `model_lods.zip` | The mesh at 100%, 50%, 25% and 10% of its triangles, one GLB per level. |
| `collision.glb` | A convex hull. |
| `textures.zip` | The maps on their own. |
| `rig.glb` | Once rigged. |
| A baked GLB per saved pose | Posed geometry, ready to use. |

**From a reference:** a 512 px transparent `icon.png`; a trimmed `sprite.png` with its pivot
recorded; `pixel_32/64/128.png` reduced to a palette; and a `manifest.json` carrying every size,
trim box, pivot and the recipe that produced them. Tiles additionally get an estimated PBR material
set.

**From the workspaces:** sprite sheets as PNG plus a JSON sidecar; Inker's ORA, PNG, GIF and sheets;
Plotter's TMX and TMJ; Packwright's atlases and sidecars.

**In bulk:** zip a named artifact across many jobs at once, or set `REALMSPINNER_EXPORT_DIR` and have
exports mirrored into it — a game project's `assets/` folder, for instance.

## Two things already done for you

**Grounding.** Every mesh has its pivot put on the ground and its X and Z centred, always, whether or
not you asked for a particular size. A model whose origin sits in the middle of its bounding volume
is a manual fix-up on every single import, forever, so the app does it once.

**`model.glb`, not `source.glb`.** Everything downstream uses the derived file. Take that one. The
raw reconstruction is kept so that every derived file can be rebuilt from it and so that you can see
what the engine actually produced — it is not the one to ship.

## 3D engines

glTF is the target format and every major engine reads it.

**Godot** imports `.glb` directly — drop it in the project and it is an importable scene. This is the
smoothest path, and `REALMSPINNER_EXPORT_DIR` pointed at a Godot project makes it smoother still.

**Unity** and **Unreal** both read glTF, Unity via a package and Unreal natively. FBX is there if
your pipeline is built around it.

### Levels of detail

**LOD chain (zip)** gives you `model_LOD0.glb` (the mesh as it is) down to `model_LOD3.glb` (a tenth
of its triangles) and a `lods.json` with each level's triangle count. Unzip it into the project and
import all four GLBs. The levels share a pivot, so swapping between them in place does not shift
the model.

In **Godot**, instance the four scenes as children of one node and give each level's mesh instance a
visibility range: set `visibility_range_end` on LOD0 and `visibility_range_begin` on LOD1 to the same
distance (and so on up the chain), with `visibility_range_fade_mode` on if you want them to dissolve
rather than pop. Godot can also generate its own LODs when it imports a mesh, which is the simpler
route if you only need levels for a model you are not otherwise controlling.

In **Unity**, add a `LODGroup` to an empty object, parent the four models under it, and assign each to
its slot, LOD0 to LOD3, then set the transition heights to suit the model's size on screen.

Be honest about what this has not been checked against: nobody has opened a chain in a real Godot or
Unity editor yet, so the recipes above are how those components are documented to work, not something
observed with these files, and the 50/25/10 percent steps have not been judged by eye for any kind of
model. A model built from thin parts — a blade, a railing — loses them first, and a lower level may
look wrong long before its triangle count says it should.

Rigged exports carry their skeleton and skin weights. A pose bakes to its own GLB rather than
travelling as an animation track — for a still pose that is what you want, and it sidesteps a class
of exporter problem where a posed model arrives at rest with the pose demoted to an animation nobody
plays.

**Export for Godot...**, on a rigged asset with clips authored for it, writes a folder named after
the character with `-godot` added, holding a `.glb` and a `.tscn` sharing that same name — into your export folder, or
one you pick. The GLB is a copy of the animated model with its looping clips renamed for Godot 4's
importer (a `walk` clip becomes `walk-loop`); the scene instances it and wires an `AnimationTree`
state machine over its clips — idle, walk and run blended
by `parameters/locomotion/blend_position` at 0, 1 and 2, attack and attack_02 and cast as their own
states, jump transitioning to fall at its end, hit and death reachable from every state and death
terminal. Call `travel("attack_02")` (or any other state's name) from your own script to play it. In
the imported scene the clips appear under their plain names — `idle`, `walk`, and so on, not
`idle-loop` — because Godot's own importer renames a `-loop`-suffixed animation back to its bare name
the moment it reads the loop flag off it, and the `.tscn` this export writes already plays those
post-import names rather than the ones baked into the GLB.

Be honest about what this has not been checked against: nobody has opened an export in a real Godot
editor yet. The loop-naming behaviour above is read out of Godot's own importer source rather than
observed, and Godot's own import dialog does not show the loop flag at all — it disappears entirely
if you save the imported animations to separate files (godotengine/godot#108823) — so treat a fresh
export as unverified in your specific Godot version until you have opened it once.

## 2D and sprite sheets

A sheet is a PNG plus a JSON sidecar, and the sidecar is deliberately engine-neutral: cell
rectangles, tags, durations and pivots, in plain JSON, for you to read with whatever you already have.

Packwright's sidecar is TexturePacker's format instead, which a great many 2D toolchains already
understand.

**An export never overwrites what is already in your export folder.** If `Knight-godot` is already
there — from an earlier export, from a second character that is also called Knight, or from a folder
of your own — a new **Export for Godot...** takes the next free name (`Knight-godot-2`) and leaves
the existing one exactly as it was. The scene's `.glb` and `.tscn` take the new name with it, so the
folder stays self-consistent.

## Tiled and Aseprite: read this before relying on it

Realmspinner reads and writes Tiled's `.tmx`/`.tmj`/`.tsx` and Aseprite's `.aseprite`, and both are
modelled carefully — the divergences are enumerated individually in the reference chapters rather
than discovered by accident.

There is one caveat, and it is important enough to state plainly rather than bury.

**Every test fixture for both formats was written by Realmspinner itself.** A green test proves that
Realmspinner's reader and Realmspinner's writer agree with each other. It does not prove that either real
application agrees with them.

For Tiled that has now been checked once, by hand, in both directions: on 2026-08-29 a Plotter map
export was opened and worked on in Tiled 1.12.x, and a map Tiled 1.12.2 itself wrote was read back
in Plotter. Both were plain orthogonal maps, so what is confirmed is the map header, the external
tileset reference and the layer data — not flipped tiles, objects or properties. For Aseprite,
nothing has been checked at all.

So: the exports are believed correct, and past plain tile layers that belief has not been checked
against the applications themselves. Try one before building a workflow on it, and expect it to
work — but check.

One specific thing to know if the file is going to Tiled: several constructs are Realmspinner's own
extensions rather than Tiled features — oblique projection, per-layer blend modes, the capsule shape,
per-object opacity, list properties. Round trips fine through Realmspinner; invisible to Tiled.

For Aseprite, the notable losses on write are per-frame palettes and colour profiles. Cel opacity,
a cel's z-index and the colours and notes on layers, cels and tags are written and read back; what is
still dropped from *user data* is a note on a slice, on a tileset or on an individual tile, and
Aseprite's custom properties tree. Each loss is reported with a warning rather than dropped
silently.

## A whole pipeline

To make the shape of it concrete, here is one path end to end. Every step has its own chapter:

1. Prompt a reference in Create, approve it, reconstruct a mesh.
2. Judge it in Review. Remesh if the picture was fine and the mesh was not.
3. Rig it in Create, then open it in Clay and animate it on the timeline.
4. Build the level it lives in with Plotter, or place it in a scene with Mason.
5. Export into your engine.

Not every asset needs all five, and most need two or three. But every one of those steps hands its
output to the next as an ordinary library asset, which is the thing that makes the app one app rather
than seven.

## What to read next

Three tutorials left, starting with the only one that makes a sound:
[Making a soundtrack](13-making-a-soundtrack.md) — Sirens, the tracker.

The reference chapters go deeper on everything touched here, whenever you want them —
[Overview](20-overview.md) is the front door to them, and each workspace has its own.

If something is not behaving, [Troubleshooting](43-troubleshooting.md) is organised by symptom, and
`uv run realmspinner doctor` answers the same questions from a terminal.
