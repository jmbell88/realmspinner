# Putting it in a game

The last tutorial chapter, and the one about leaving. Everything Warlock makes is meant to be
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

**In bulk:** zip a named artifact across many jobs at once, or set `WARLOCK_EXPORT_DIR` and have
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
smoothest path, and `WARLOCK_EXPORT_DIR` pointed at a Godot project makes it smoother still.

**Unity** and **Unreal** both read glTF, Unity via a package and Unreal natively. FBX is there if
your pipeline is built around it.

Rigged exports carry their skeleton and skin weights. A pose bakes to its own GLB rather than
travelling as an animation track — for a still pose that is what you want, and it sidesteps a class
of exporter problem where a posed model arrives at rest with the pose demoted to an animation nobody
plays.

**Export for Godot...**, on a rigged asset with clips authored for it, writes a folder named after
the character, holding a `.glb` and a `.tscn` sharing that same name — into your export folder, or
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

A Poser character sheet has a third way out: **Export frames...** writes a folder named after the
character, one subfolder per movement inside it, one subfolder per compass direction inside that
(`N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`; `S` is the character facing you, `W` its left profile),
and `000.png`, `001.png` and so on inside that — plus a
`manifest.json` (format `warlock-frames`, version 1) stating the frame size, whether the sheet is
pixel art or HD, and each clip's own `loop`, `frames`, `duration_ms`, `fps` and `directions`. It is
for an engine that wants `AnimatedSprite2D`-style frame folders rather than one atlas plus one
sidecar, and a re-export replaces the folder whole.

## Tiled and Aseprite: read this before relying on it

Warlock reads and writes Tiled's `.tmx`/`.tmj`/`.tsx` and Aseprite's `.aseprite`, and both are
modelled carefully — the divergences are enumerated individually in the reference chapters rather
than discovered by accident.

There is one caveat, and it is important enough to state plainly rather than bury.

**Every test fixture for both formats was written by Warlock itself.** A green test proves that
Warlock's reader and Warlock's writer agree with each other. It does not prove that either real
application agrees with them.

For Tiled that has now been checked once, by hand, in both directions: on 2026-08-29 a Plotter map
export was opened and worked on in Tiled 1.12.x, and a map Tiled 1.12.2 itself wrote was read back
in Plotter. Both were plain orthogonal maps, so what is confirmed is the map header, the external
tileset reference and the layer data — not flipped tiles, objects or properties. For Aseprite,
nothing has been checked at all.

So: the exports are believed correct, and past plain tile layers that belief has not been checked
against the applications themselves. Try one before building a workflow on it, and expect it to
work — but check.

One specific thing to know if the file is going to Tiled: several constructs are Warlock's own
extensions rather than Tiled features — oblique projection, per-layer blend modes, the capsule shape,
per-object opacity, list properties. Round trips fine through Warlock; invisible to Tiled.

For Aseprite, the notable losses on write are per-frame palettes and colour profiles. Cel opacity,
a cel's z-index and the colours and notes on layers, cels and tags are written and read back; what is
still dropped from *user data* is a note on a slice, on a tileset or on an individual tile, and
Aseprite's custom properties tree. Each loss is reported with a warning rather than dropped
silently.

## A whole pipeline

To make the shape of it concrete, here is one path end to end. Every step has its own chapter:

1. Prompt a reference in Create, approve it, reconstruct a mesh.
2. Judge it in Review. Remesh if the picture was fine and the mesh was not.
3. Rig it, and pose it in Poser.
4. Render a character sheet, from that same Poser session.
5. Clean the sheet up by hand in Inker.
6. Pack it with other sprites in Packwright.
7. Build the level it lives in with Plotter.
8. Export into your engine.

Not every asset needs all eight, and most need two or three. But every one of those steps hands its
output to the next as an ordinary library asset, which is the thing that makes the app one app rather
than seven.

## What to read next

Three tutorials left, starting with the only one that makes a sound:
[Making a soundtrack](14-making-a-soundtrack.md) — Sirens, the tracker.

The reference chapters go deeper on everything touched here, whenever you want them —
[Overview](20-overview.md) is the front door to them, and each workspace has its own.

If something is not behaving, [Troubleshooting](43-troubleshooting.md) is organised by symptom, and
`uv run warlock doctor` answers the same questions from a terminal.
