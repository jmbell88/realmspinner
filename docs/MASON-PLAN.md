# Mason — a 3D scene editor

A plan file, not a roadmap: it exists to be executed and then deleted, per this repo's
"a finished plan is deleted rather than ticked" rule.

## What this is

Warlock Studio makes game *assets*. A prompt becomes a reference PNG, a PNG becomes a
textured GLB, a GLB becomes a rig, a sheet, a tile map. Every mode produces one thing, and
nothing in the app assembles those things into a **place**.

Plotter is the 2D answer — paint a world out of tiles, export it to Tiled. There is no 3D
answer. A user with thirty generated props and a Clay blockout can look at each of them
alone and never at the place they are for.

**Mason** is that mode: start a scene, place library assets and primitives into it, group
and duplicate them, light it, sculpt a ground, and export the arrangement as a glTF scene,
an engine-friendly GLB-plus-manifest, or merged OBJ geometry.

`TODO.md` carries no entry for this in either direction, so it is net-new design rather
than a struck-out plan.

## Decisions already taken

| | |
|---|---|
| Name | **Mason** — package `studio/mason/`, document suffix `.wscn` |
| Rail position | Workspaces group, immediately after Clay |
| Maturity chip | None — ships as a peer of Clay and Plotter |
| Placeable sources | Library assets (a job row's `model.glb`) and primitives (`clay.primitives.GENERATORS`) |
| Scene features | Lights + camera markers, terrain heightmap, grouping/hierarchy, prefabs/instancing |
| Exports | glTF/GLB scene · GLB + engine manifest JSON · OBJ + MTL |
| Library | Full round trip — export as an asset row, reopen the scene from it |
| Asset references | Link by job id in the document; resolve and embed only at export |

Mason sits after Clay rather than beside Plotter, and the two claims are in tension, so the
comment in `modes.py` settles it: Plotter is the mode Mason is the answer *to*, but Clay is
the mode it is the answer *with*.

## State of the branch

Branch `claude/3d-scene-editor-jhaje8`, commit "Mason takes the fourteenth place on the rail".

**Landed:**
- `icons.BLOCKS = ""`, lucide's `blocks`.
- Mason registered in `modes.py`: the `MODES` 4-tuple after Clay, `RAIL_GROUPS[1]`,
  `WORK_MODES`, `WORKSPACE_MODES`, and the three prose counts `tests/test_modes.py` reads
  as data.
- `tests/test_modes.py` updated to expect fourteen modes and a nine-mode workspace group.

**Verified:**
- `test_the_rail_fits_the_resize_floor_at_every_scale` passes at 1.0 / 1.5 / 1.75 with
  fourteen rows. This was the one thing that could have refused the mode outright.
- Every glyph Mason's panes will want was checked against the vendored
  `resources/fonts/lucide.ttf` with fontTools — the check `tests/test_icons.py` already
  applies to LINK and UNLINK — rather than trusted from a release's `info.json`:
  `` blocks, `` lightbulb, `` video, `` list-tree,
  `` component, `` mountain-snow, `` group, `` ungroup,
  `` shapes.

**Left red, and this is the first job of the next session.** Registering a mode without
sweeping the registries that enumerate modes leaves tests failing. The sweep is listed under
"The registration sweep" below; finish it before anything else, because until it is done a
genuine regression cannot be told from a known gap.

**What is actually known about the failures, and what is not.** A full run on Linux under
`xvfb-run` at this commit reported **487 failed, 17376 passed, 465 skipped** in about three
minutes, with one `INTERNALERROR` from an xdist worker. That number is *not* Mason's bill:
this project is Windows-first with Windows-only CI, so the Linux suite was never green to
begin with. Two failures were checked by hand against the parent commit and fail identically
there — `tests/test_plotter_mode.py::test_one_file_spelled_two_ways_is_one_tab` (it depends
on `os.path.normcase` folding case, which it does on Windows and not on Linux) and
`tests/test_studio_smoke.py::test_the_library_builds_empty_and_populated` (it wants the
`text2image` pack installed).

The full baseline-versus-current diff was **started and not finished** — the second run hung
near the end and was stopped. **Do it before trusting any failure count**, and do not repeat
the mistake that made it necessary: this plan's Stage 0 was called green and pushed on the
strength of five targeted tests, without the suite behind it. The recipe:

    # at HEAD, then at HEAD~1, with the same flags both times
    xvfb-run -a -s "-screen 0 1920x1080x24" uv run pytest -q --tb=no -rf -p no:randomly \
        > cur.log 2>&1
    grep -E "^FAILED" cur.log | sed 's/ - .*//' | sort -u > cur.txt
    comm -13 base.txt cur.txt        # what this change actually broke

## What is reused rather than rebuilt

The single most important fact about the size of this work: **the 3D stack already exists.**

- `studio/viewer/` is a complete moderngl renderer — `render.Renderer.draw`, `programs.py`
  (a three.js-r170-equivalent PBR shader set), `env.py` (procedural IBL probe),
  `camera.Camera` (orbit, `AXIS_VIEWS`, `frame(lo,hi)`, `screen_ray`), `gizmo.py`
  (translate/rotate/scale, constant screen size, analytic hit tests), `picking.py`
  (`ray_object`, `ray_triangles`, `BVH`, `cached_bvh`, a native fast path), `grid.py`,
  `glctx.Viewport` (offscreen, 4x MSAA), `scene.GpuModel`, `capture.py`,
  `sheet.StripRender` (incremental offscreen render, one cell a frame).
- `studio/viewer/gltf.py` already models a **full scene graph**: `Node` with TRS (XYZW
  quaternion) + `children` + a `mesh` *index*, `Model.roots`, `update_world()`.
  `viewer/glbwrite.write_glb` already writes `children` and `scenes[0].nodes`, and two
  nodes sharing a mesh index is real glTF instancing — for free.
- `studio/clay_view.py` and its `_view_*.py` mixins are already a many-object viewport:
  a per-subject GPU cache, one flat draw list a frame, the world matrix as a uniform.
- `clay.primitives.GENERATORS` — fifteen parameterised generators, each with a complete
  defaults dict, plus `CATEGORIES`. The primitive palette is derived, never hand-listed.
- `studio/undo.py`, `studio/journal.py`, `studio/docmodes.py` — the shared document
  machinery every editor mode already runs on.
- `pipelines/postprocess.glb_to_obj_zip` already writes OBJ + MTL + textures through
  trimesh.
- `service._jobs_create.import_mesh` mints a finished model row from authored geometry, and
  `service/files.py` has the sidecar pattern (`CLAY_SOURCE = "build.wblk"`,
  `PLOTTER_SOURCE = "map.wmap"` → `MASON_SOURCE = "scene.wscn"`). **Mason therefore needs no
  new job kind**, which keeps it clear of the two job-kind sweeps in `docs/INVARIANTS.md`
  (the four-edit list and the ten stage-keyed tables) — the most error-prone part of
  extending this app.

## The engine package — `src/warlock/studio/mason/`

Pure the way `studio/plotter/` and `studio/clay/` are: numpy + stdlib + a lazy Pillow, no
imgui, no moderngl, no pygame, no `service`. The outward import set is pinned exactly by
`tests/mason/test_mason_imports.py`, written **first**, because the pin is the contract:
`warlock.studio.undo`, `warlock.studio.viewer` (submodules `gltf`, `math3d`, `picking`
only), `warlock.glbio`, and the guard leaves `zipguard` / `npyguard` / `pixelguard`.

| Module | Holds |
|---|---|
| `refs.py` | `LibraryRef`, `PrimitiveRef`, `ref_key()`, the `GeometrySource` protocol |
| `nodes.py` | the node classes, `new_uid` / `reserve_uid` |
| `document.py` | `MasonDoc` — roots, prefab table, terrain singleton, selection, history |
| `edits.py` | the undo types |
| `scene.py` | **the one resolver** |
| `terrain.py` | the heightfield and the pure sculpt brushes |
| `ops.py` | array-linear/radial, scatter, align, distribute, drop-to-ground, snap |
| `pick.py` | ray-vs-scene over resolved items |
| `serialize.py` | the `.wscn` zip, its version gate, the two-phase snapshot |
| `gltfout.py` | `MasonDoc` → `viewer.gltf.Model`, and the GLB bytes |
| `manifest.py` | the engine-facing sidecar JSON |
| `objout.py` | merged OBJ + MTL |

### Nodes — several classes, not one `kind`

`plotter/_map_model.py` chose four layer classes rather than one class with a mode, and
Mason chooses six node classes for that reason and one more: a light carries colour,
intensity, range and cone angles; a camera carries yfov, znear, zfar; a mesh node carries a
source reference and a material override. One class with a discriminator makes every one of
those fields optional and every consumer an `if kind ==` chain — and the resolver already
wants `isinstance(node, GroupNode)` exactly as `plotter/scene.py` does.

The base carries what every node genuinely has: `uid`, `name`, translation / rotation
(XYZW) / scale as owned f8 arrays, `children`, `visible`, `locked`, `static`, and a
`properties` dict that reaches the manifest. Then `GroupNode`, `MeshNode`, `LightNode`,
`CameraNode`, `PrefabNode`, `TerrainNode`.

**Clay's argument against parenting is answered, not ignored.** `clay/document.py` says
parenting costs "a transform that is not the one you typed, and an outliner that has to
explain itself", and that nothing in *place a few primitives and export them* needs it.
That is right about Clay and wrong about Mason, because Mason's document is an arrangement
*of* assets rather than one asset: the user's mental model is a tree, glTF's node graph is
a tree, and the engine scene the export lands in is a tree. The cost is paid explicitly —
`panes/mason_props.py` shows the **local** TRS as editable fields and the **world** TRS
beside it, read-only, off the resolver. That is the whole answer to "a transform that is
not the one you typed".

**Children, not parent.** A node holds `children`; nothing holds `parent`. The tree is the
storage, sibling order is meaningful (it is the export order and the outliner order), and a
`parent` pointer is a second copy of one fact that can disagree with the first. Acyclicity
comes from three things: by construction (a node object appears in exactly one `children`
list, and `reparent` detaches and re-inserts the same object — there is no "set parent");
by refusal (`reparent` raises when the target is the node or one of its descendants); and
by backstop (the walk carries a `seen` set and a depth ceiling, as `gltf.Model.update_world`
already does, so a hand-edited file costs the corrupt part of itself and not the frame
thread).

**Uids** follow `clay/document.py` verbatim — a process-wide `itertools.count` behind the
package's one lock, `new_uid()` on the frame thread, `reserve_uid()` raising the floor on
load from a task thread. Never reused, because the uid is the address every undo step is
written against.

### How a mesh node names its source

Two frozen dataclasses in `refs.py`. `PrimitiveRef(generator, params)` — a key into
`clay.primitives.GENERATORS` plus normalized, hashable params, exactly as `clay.document.Obj`
carries `generator`/`params`. `LibraryRef(job_id, artifact="model.glb", name, sha256)` —
an opaque job id, never a path, with the job's name at save time for the missing-reference
message and a hash for relink.

Neither touches the filesystem. **The pure package never resolves geometry itself**; it
takes a callback, which is `plotter/tmx.py`'s `tsx_loader`/`image_loader` precedent with one
door instead of two:

    class GeometrySource(Protocol):
        def primitives(self, ref) -> list[gltf.Primitive]: ...
        def box(self, ref) -> tuple[np.ndarray, np.ndarray] | None: ...
        @property
        def rev(self) -> int: ...

The host implementation is `studio/mason_assets.py`, a studio-level module free to import
both halves. A `PrimitiveRef` resolves through `GENERATORS`; a `LibraryRef` resolves through
`svc.config.job_dir(job_id) / "model.glb"` — the resolve `clay_mode.edit_asset_in_clay`
already performs — parsed on a task thread and adopted on the frame thread
(`viewer_embed.parse_model` / `adopt_model`'s shape), with the asset's internal node world
matrices **baked into its primitive positions once** so a ref is always one flat primitive
list. The stated cost: a library asset's own internal hierarchy does not appear in Mason's
outliner, which is right, because a library asset is one *thing* in a scene.

Routing primitives through the same callback rather than importing `clay.primitives` into
`mason/` is deliberate: the library path must be a callback anyway, so one resolution
mechanism beats two, and it keeps `mason/`'s reach to a leaf rather than a chain.

### Prefabs and instancing — two mechanisms, not conflated

**GPU and file instancing is automatic and invisible**: two mesh nodes with the same
`ref_key` share one GPU upload and one glTF mesh index. Nothing in the document says so; it
falls out of keying on the ref.

**Prefabs are authored.** `MasonDoc.prefabs: dict[str, PrefabDef]` holds template subtrees
that are *not* in the scene tree; a `PrefabNode` is an instance carrying its own uid, name,
TRS and flags — **and no per-child overrides**. Deep overrides are what makes prefab systems
hard, and the escape hatch is one edit: `unpack_instance(uid)` replaces the instance with a
deep copy of the template carrying fresh uids. Say that in the docstring as a decision with
an escape hatch, not as an omission.

**A template change reaches every instance on the next frame**, because the walk reads
through `doc.prefabs[name]` every time. There is no propagation step and no "apply to
instances" button — the propagation step is exactly what an editor gets wrong. Recursion is
refused at the door, with the walk's depth ceiling as the backstop.

### Lights and cameras

`LightNode(kind, color, intensity, range, inner_cone_angle, outer_cone_angle)` uses
**KHR_lights_punctual's own field names and units**, and `CameraNode` uses glTF's, for the
reason `clay/document.py` gives for materials being `gltf.Material`: the export is the
definition, and a parallel type buys a conversion function and a place for the two to
drift. Direction is the node's `-Z` per the extension, so there is no separate direction
field to disagree with the rotation. Both draw as `viewer.render.DrawItem` overlays
(`viewer/markers.py` is the precedent) and are pickable and gizmo-draggable like any node.

### Terrain — a document singleton plus one node

`MasonDoc.terrain: Terrain | None` holds the array and a single `TerrainNode` refers to it.
Three reasons: the large array must not be copied into prefabs or instances; the resolver
needs to know which node yields generated geometry; and two terrains is two ground planes,
which nobody asked for and which doubles every sculpt question. The node exists so the
ground has an outliner row, can be hidden, can carry a transform and a material, and
exports as an ordinary mesh node.

    @dataclass
    class Terrain:
        heights: np.ndarray   # (n+1, n+1) f4, vertex heights in metres
        size_x: float
        size_z: float
        material: gltf.Material

Brushes are pure and local — `raise_lower`, `smooth`, `flatten`, `noise`, `set_height` —
each returning `(rect, new_sub_array)`, which is `plotter/tools.py`'s region shape. The edit
records **the affected rect and the before/after sub-arrays only**, so `cost` is honest and
`UNDO_BYTES` stays meaningful. A drag is one undo step through a
`begin_sculpt` / `sculpt` / `end_sculpt` session, which is the gesture-door rule Inker and
Plotter already follow. `terrain_mesh()` is the single conversion out, memoized on
`id(heights)` — sound only because every brush **rebinds** `heights` rather than writing in
place, which must be stated where the array lives.

`MAX_TERRAIN_SIDE` wants a dated note in `docs/measurements/` before it is fixed; see the
open questions.

### The undo edits

All uid-addressed, never by index, and each owns its arrays (copy any view, or `cost` lies).
`NodeAddEdit` / `NodeRemoveEdit` (the subtree travels with the edit), `NodeMoveEdit`
(reparent and reorder in one type), `TransformEdit`, `NodePropsEdit`, `RefEdit`,
`TerrainEdit`, `TerrainConfigEdit`, `PrefabEdit`. Multi-select drags, array ops and
"make prefab" bundle through `history.mark()` / `collapse_since()`.

`dirty` is `history.head != saved_head`, never a flag — an undo is a change, so a
rev-counter would call an undone document unsaved forever.

### `.wscn` — the document links, the export embeds

A zip in `.wblk`'s shape: `scene.json` (version, the nested node tree, the material palette,
the prefab table, the terrain header, the saved camera), `terrain/heights.npy` when a
terrain exists, `textures/<n>.png` for authored material overrides. **No geometry is stored
at all** — primitives regenerate from `(generator, params)` and library assets are links.

`sort_keys=True`, every zip and npz timestamp fixed at `(1980,1,1,0,0,0)`, so two saves of
an unchanged document are byte-identical. A newer version is refused and a declared-but-
missing member is refused. `snapshot()` on the frame thread and `snapshot_bytes()` on a task
thread, which is a finding Clay already closed — do not repeat it. Every read goes through
`zipguard` / `npyguard` / `pixelguard`.

**The dangling-reference story, named.** A `LibraryRef` whose job is gone does **not** refuse
the file. The document opens, the node resolves to a missing-asset proxy drawn in the warn
colour, `doc.missing_refs` lists it, and the document pane offers **Relink…** (one `RefEdit`)
and **Remove**. This looks like a contradiction of `.wblk`'s refuse-rather-than-substitute
rule and is not — say why in the docstring: `.wblk` substituting an empty mesh would show
the user finished work that is not there and let them save over it, whereas a *linked* scene
with a broken link is a known, repairable state the document itself reports, and refusing to
open it would strand every other node in the scene.

Why link at all: the premise of the mode is a scene of library assets. Sixty textured GLBs
embedded is hundreds of megabytes a save, byte-identity stops meaning anything, and
re-exporting an asset from Clay would leave the scene showing a stale copy with nothing in
the file to say why.

## The one resolver — `mason/scene.py`

`plotter/scene.py`'s rule applied to 3D: the viewport, all three exporters and the thumbnail
render must never work out inherited state twice.

    @dataclass(frozen=True)
    class Placed:
        node: Node                       # the leaf that draws, by reference
        path: tuple[int, ...]            # uids root-first; distinguishes an instance's copies
        world: np.ndarray                # 4x4 f8
        visible: bool
        locked: bool
        static: bool
        ref: Ref | None
        material: gltf.Material | None   # nearest-ancestor override, or None for "as authored"
        prefab: str                      # "" when authored, else the template it came from
        dangling: bool

    def walk(doc, visit, *, include_hidden=False, expand_prefabs=True) -> None
    def resolve(doc, *, include_hidden=False, max_items=MAX_PLACED) -> list[Placed]
    def resolved_for(doc, uid) -> Placed | None
    def world_bounds(doc, source, uids=None) -> tuple[np.ndarray, np.ndarray] | None
    def owner_uid(path) -> int

Five combination rules, each stated once and computed once: **transform composes**;
**visible ANDs** (one hidden ancestor hides everything under it and the leaf's own flag
never moves, so unhiding restores exactly what was there); **locked ORs** and is *reported,
never enforced* (plotter's rule — a lock stops the user, not the document, or an undo could
not put back what was there before the lock); **static ORs**, because it reaches the
manifest; and **material override wins nearest-ancestor**, so a group can retint a whole
prop set without touching the assets.

Groups are not in `resolve`'s result: they draw nothing, and a consumer that had to skip
them is a consumer that could forget to.

`walk` is the single traversal. `resolve` is `walk` accumulating world matrices;
`gltfout.scene_model` is `walk` accumulating *structure* and local transforms, because a
glTF export must keep the parents. Two functions, one traversal, one visibility rule, one
prefab expansion — which is what makes "the viewport and the export can never disagree" true
for an exporter that wants the tree rather than the flattening.

## Export

Every exporter returns `dict[str, bytes]` — `plotter/tmx.tmx_export`'s shape, a *mapping of
paths* rather than a file — and the io layer stages the whole set (a dotfile per target,
everything staged before anything is replaced), which is `plotter_io._write`'s rule.

1. **`scene.glb`** — `gltfout.scene_model` then `viewer.glbwrite.write_glb`. Meshes are
   deduplicated by `ref_key`, so one glTF mesh serves many nodes and the instancing is the
   file format's own. Materials dedup by identity, which is the key `GpuMaterial` already
   uses, so the file and the GPU agree. Node names are made unique on the way out, because
   every importer's find-by-name assumes it and the manifest addresses nodes by name.
2. **Engine bundle** — the same `scene.glb` plus `scene.json` from `manifest.py`: format
   version and generator, **units and handedness stated outright** (metres, Y-up,
   right-handed — the one thing an engine import script gets wrong), then one entry per node
   keyed by the same unique name the GLB carries, with local and world TRS, the user
   properties, the source (library job id and name, or generator and params), light and
   camera blocks, and the template an instance came from; plus a `prefabs` section listing
   each template and its instances, which is what lets an import script build engine prefabs
   rather than hundreds of loose meshes. The manifest is the **provenance the GLB cannot
   carry**. Not glTF `extras`: extras survive some importers and are silently dropped by
   others, and the reader of this file wants one file it can parse without a glTF library.
3. **OBJ + MTL** — merged static geometry: walk `resolve`, transform positions by
   `Placed.world` and normals by the inverse-transpose (guarded against a singular 3x3),
   concatenate, emit one `o` group per unique node name so the merged file is still
   navigable, `usemtl` per material, 1-based monotone indices, with a `MAX_OBJ_VERTS`
   refusal carrying a real sentence.
   Lights and cameras are not representable in OBJ and are **reported, not silently
   dropped** — `Model.skipped_textures` is the precedent: a loss that is stated is not the
   same as a loss that is invisible.

   Two implementations are defensible and the choice is open: hand-written and pure inside
   `mason/objout.py` (deterministic bytes, headless test, owns its material names — the
   `plotter/pngio.py` precedent), or handing the scene GLB to
   `pipelines.postprocess.glb_to_obj_zip` from a task thread in the studio layer. The pure
   package may not import `pipelines`, so the second option puts the call in `mason_io.py`.
   Decide at the start of the export stage; the hand-written one is preferred for byte
   determinism, which every other format here has.

### The one shared-file change: cameras and lights in the glTF writer

`viewer/gltf.py` and `viewer/glbwrite.py` carry no camera and no light. Both need
`gltf.Camera` and `gltf.Light` dataclasses, `Node.camera` / `Node.light`, `Model.cameras` /
`Model.lights`, a top-level `cameras` array, and `KHR_lights_punctual` in `extensionsUsed`,
`doc["extensions"]` and `node["extensions"]` — emitted **only when one exists**.

**Extend the shared writer rather than forking one into Mason.** There is one GLB writer in
this project and the loader beside it is the only thing that can test it; the module's own
docstring names the bar ("an authored `Model`, written, read back, and compared array for
array"). A Mason-owned writer would be a second home for the four things that docstring says
are easy to get wrong — the POSITION accessor's `min`/`max`, four-byte view alignment, the
index component width matching the bytes, and the two chunk paddings — and the new code
would have no loader to round-trip against.

The proof that it costs the existing modes nothing is a test: **an existing Clay document
exported before and after the change produces byte-identical GLB.**

## The viewport — `studio/mason_view.py`

A **sibling** to `ClayView`, not a subclass. Element modes, the vertex marquee, proportional
editing and the whole of `_view_drag.py` are Clay's mouse map, not a scene editor's.

Everything genuinely shared already lives in `viewer/`. What is left over is frame plumbing,
so extract `studio/_view_frame.py` first as **pure code motion with Clay's tests green before
Mason exists**: the resize/forget pair (release before forget, or the imgui backend holds a
dead moderngl object under a reissued GL name), the redraw bookkeeping, the camera mouse map
(orbit, middle-button pan, dolly, right-button-is-a-menu-within-four-pixels, modifiers read
from `pygame.key.get_mods()` at the press), the axis-view keys, and `_Composite`.

Six things Mason must do differently:

1. **The GPU cache key is the ref, not the node** — `(ref_key(node.ref), id(material))`.
   Clay keys per object because in Clay every object owns its mesh; Mason keys per asset,
   and that single change is what makes five hundred instances one upload. Each entry pins
   whatever its `id()`s name, because an `id` is sound only while its object is alive.
2. **The composite may not write through a shared node.** `clay_view._composite` does
   `node.world = world` on the cached entry's own `gltf.Node`. That is sound at one entry
   per object and **silently wrong** at one entry per ref: N instances share the node, the
   last write wins, and N−1 draw in the wrong place. Mason emits a per-draw node proxy
   carrying `world` and `skin = None` — the only two attributes the renderer reads off a
   node in the composite path — pooled per frame so a thousand-item scene does not allocate
   a thousand objects a frame. **This is the single most likely quiet bug in the whole
   mode; write the test before the code.**
3. **Frustum culling** — each `Placed` knows its ref's local AABB, so transform the eight
   corners once a frame and test against the six planes. Skipped under a threshold, because
   the test itself costs something. Clay needs none of this; it draws one asset.
4. **A transparency pass** — opaque first, then non-opaque sorted back-to-front. Clay has no
   sort because Clay's materials are opaque; a library asset's need not be.
5. **The redraw key gains the geometry source's revision.** An asset finishing its
   background parse changes the picture with **no document edit**, so without
   `(id(source), source.rev)` in the key the adopting frame is skipped as "nothing moved"
   and the asset appears only when something else forces a redraw.
6. **Picking is per ref, selection is per owner.** `cached_bvh` keyed on the ref's
   primitives, so one BVH serves every instance and the ray is transformed into each
   instance's local space; terrain gets a heightfield ray march instead, which is O(cells
   along the ray). A hit inside an instance maps back through `Placed.path` to the instance,
   not its innards.

The gizmo drags a **set** with a pivot (median / active / world origin), writing one
compound edit.

## The panes

Workspace composition in `studio/mason_viewport.py` (`App._mason_workspace`), a mirror of
`clay_viewport.py`: left column, centre pane, right column, both sidebars from
`skeletons.mason`.

| Slot id | Title | Column | Sizing | `when` |
|---|---|---|---|---|
| `mason-assets` | Assets | left | SHARE | — |
| `mason-tools` | Tools | left | FILL | — |
| `mason-outliner` | Outliner | right | SHARE | — |
| `mason-props` | Properties | right | SHARE | — |
| `mason-prefabs` | Prefabs | right | SHARE | the document has a prefab |
| `mason-bridge` | Document | right | FILL | — |

Plus `panes/mason_header.py` (shading, grid, overlays, camera markers, look-through-camera),
`panes/mason_hud.py` (hint line and stats), `panes/mason_menu.py` (the right-click menu).

- **Assets** — the placeable palette: library mesh rows with thumbnails, the primitive grid
  **derived from `CATEGORIES` / `GENERATORS`** so a sixteenth generator gets a button the day
  it lands, lights, cameras, and the terrain brush. Library rows already drag through
  `panes/library.draggable_source` / `dragged_job`.
- **Tools** — select / move / rotate / scale, snapping (grid, angle, surface, drop-to-ground),
  align and distribute, array-linear and array-radial.
- **Outliner** — the tree: expand, drag-to-reparent, multi-select with a **uid** anchor
  (never an index), per-row visibility and lock, in-place rename, instances badged with
  their template.
- **Properties** — local TRS editable, world TRS read-only, the per-kind block, the material
  override, and the user properties table that reaches the manifest.
- **Document** — save, save as, the three exports, the missing-reference list with Relink,
  scene stats, and Export to the library.

Every pane needs a `(?)` `help_button` and a matching `HELP_TARGETS` row, or
`tests/manual/test_coverage.py` fails; the parity is checked in both directions.

## The registration sweep

This is the part that is currently incomplete, and a missed entry shows up as a dead pane or
a crashed dispatch rather than as an obviously failing test.

**Done:** `modes.py` (`MODES`, `RAIL_GROUPS`, `WORK_MODES`, `WORKSPACE_MODES`, the prose
counts), `icons.py`, `tests/test_modes.py`.

**Still to do:**

- `main.py` — the `_build_ui` arm, `_mason_workspace()`, the `_shortcut` arm that must
  `return` **unconditionally** whether or not it consumed the key, `DROP_REFUSALS` and the
  `_on_drop` route for `.wscn` and `.glb`, the `done.key.startswith("mason-")` claim in
  **both** the done and failed paths, the `_ask_quit` guard chain, and the teardown steps
  (persist, release textures).
- `tests/test_mode_keys.py` — `_WORKSPACE_ARMS`, whose companion test asserts
  `set(_WORKSPACE_ARMS) | {"create"} == set(modes.WORK_MODES)`. This is the test that fails
  first on a new work mode.
- `state.py` — `AppState.mason: Any = None`; `ACTIONS` and `primary_action` / `card_kind`
  only if Mason produces library rows.
- `skeletons.py` — a `mason(ctx)` builder and its `BUILDERS` entry.
- `layout.py` — the per-mode default-pane table.
- `palette.py` `_DOC_MODES`; `docmodes.DOC_MODES`; `recents.KINDS`;
  `journal._PROVIDER_MODULES` plus a `JOURNAL` provider in `mason_mode`.
- `panes/landing.py` — `NEW_ITEMS`, `KIND_OPENERS`, `_KIND_MODES`.
- `panes/overlay.py` — `PLACEHOLDERS["mason"]` and `ACTIONS["mason"]`.
- `shortcuts.py` (a `table("Mason", [...])` block, gated against the shortcuts chapter in
  both directions); `verbs.py`; `filetypes.py`; `dialogs.ARTIFACT_FILTERS` (`.wscn`, `.obj`)
  — each filter its own list object, since a wiring sweep dedupes by identity.
- `asset_exits.py` — a `_mason_add` builder in `_BUILDERS`, so a library mesh reaches a
  scene from the library's overflow menu and the inspector at once; `tests/test_asset_exits.py`
  pins the exact mode set, and the module exists precisely because two lists drifted.
- `asset_open.py` — the route that reopens a Mason-sourced row in Mason.
- `service/files.py` — `MASON_SOURCE`, `save_mason_source`, `mason_source_path`, with the
  size limit and staged write the Clay and Plotter sidecars already use.
- Every sibling headless package's import-pin test enumerates its siblings, so `mason` joins
  the lists in `tests/{inker,clay,plotter,packwright,sirens,muse,troupe}/test_*_imports.py`.
- **The workspace-count prose, verified.** `tests/test_findings_followups.py::test_no_document_miscounts_the_workspaces`
  derives the count from `RAIL_GROUPS` and scans every `*.md`/`*.py` under the root, `docs/`
  and `src/`. With Mason registered it names exactly these ten sites, each of which states
  the old count of the workspace group where the rail now draws nine (two of them say it
  with the word "creative" in the middle, and one is capitalised):

      INSTALL.md:100
      README.md:131  and  README.md:173
      docs/INVARIANTS.md:319
      docs/manual/04-judging-what-you-made.md:149
      docs/manual/20-overview.md:78  and  docs/manual/20-overview.md:159
      src/warlock/studio/layout.py:340  and  src/warlock/studio/layout.py:711
      src/warlock/studio/tour/scripts.py:60

  Note that this scan reads *this file* too, which is why the phrase itself is described
  here rather than quoted.

- The other prose that states a count: `docs/INVARIANTS.md` (`**Thirteen modes and one rail` →
  fourteen, plus the mode enumeration and the `WORK_MODES` / `WORKSPACE_MODES` lists),
  `docs/manual/20-overview.md` (**two** places — the test-gated "chooses between thirteen
  modes" and the *not* test-gated "it holds all thirteen modes"), `README.md` (the
  "thirteen top-level modes" sentence, the numbered mode list, and the chapter count, which
  is asserted), and `CLAUDE.md`'s own thirteen-modes sentence, which is the first thing an
  agent reads.

## The manual

Part I is **not** full: `manual/loader.PARTS` reserves `range(1, 20)` and only 01–16 exist,
so a tutorial chapter (`17-dressing-a-scene.md`) costs **zero renames**, which is exactly
what reserving that block was for.

Part II (`range(20, 39)`) **is** full, 20 through 38, and its order is the rail's order — so
Mason's reference chapter goes at **`31-mason.md`**, immediately after `30-clay.md`, and the
cascade is taken. Appending at 39 would be seven renames instead of fifteen and would file
the mode after the keyboard-shortcuts chapter, which tells every reader it is an
afterthought. The repo has taken this cascade four times already and written down why each
time, in the comment block at `manual/loader.py`; Mason's note joins the end of it.

Renames, **descending so no rename collides**:

    45-extending -> 46      44-pipelines -> 45       43-architecture -> 44
    42-troubleshooting -> 43  41-app-settings -> 42  40-configuration -> 41
    39-installation -> 40   38-shortcuts -> 39       37-review -> 38
    36-library-and-jobs -> 37  35-muse -> 36         34-sirens -> 35
    33-troupe -> 34         32-packwright -> 33      31-plotter -> 32
    (new) -> 31-mason

Then `PARTS` widens to `range(20,40)`, `range(40,44)`, `range(44,47)`.

**Use an exact old-stem → new-stem map applied in descending order, never a regex over
`\d\d-`** — `31-bit`, `32-bit` and `30-degree` appear in unrelated source. It is roughly 258
literal occurrences across `docs/`, `src/`, `tests/`, `README.md` and `TODO.md`.

Moving with it: `loader.PARTS` and its comment; `tests/manual/test_docs.py` `EXPECTED_KEYS`;
`docs/manual/00-index.md`; `studio/manual/targets.py` (the plotter rows change chapter, plus
the new Mason rows); `tests/manual/test_manual_promises.py`, `test_shortcuts.py`,
`test_manual_claims.py`, `test_clay_generators_documented.py`, `test_sections.py`;
`tests/test_fetch.py`. `_COUNT_WORDS` already carries `14: "fourteen"`.

`docs/manual/45-extending.md` has no "adding a mode" section. Mason is the occasion to write
one, because this file is that checklist.

## Landmines

1. **The suite is not green on Linux, and this repo is Windows-first with Windows-only CI.**
   Measure a baseline before attributing any failure to your change, and never call a commit
   green on the strength of the handful of tests you thought to run. Several failures are
   platform behaviour — `os.path.normcase` folds case on Windows and is identity elsewhere,
   which is what `tests/test_plotter_mode.py::test_one_file_spelled_two_ways_is_one_tab`
   depends on.
2. **GL tests skip rather than fail without a display.** Run the suite under `xvfb-run -a`
   or the rail gate passes while testing nothing.
3. **A pipeline masks pytest's exit code.** `pytest ... | tail` reports `tail`'s status.
   Redirect to a file and check `$?`.
4. **`tests/test_findings_followups.py` scans every `*.md` and `*.py` under the repo root,
   `docs/` and `src/`** (excluding `CHANGELOG.md` and `docs/measurements/`) for two things:
   a workspace count that disagrees with `RAIL_GROUPS` — the phrase `N workspaces` or
   `N creative workspaces` — and any line naming the removed styles feature without the word
   "deleted" on that same line. **This file is inside that scan.** Any plan or design
   document added to the repo must keep those rules.
5. **`tests/test_modes.py` reads `modes.py`'s prose comments as data**, so the comment
   counts must move with the tuples.
6. **The default pytest lane's `--dist loadfile` is load-bearing**, not tuning. Do not switch
   to `-n auto`. The xdist workers are not wholly stable here either: two long runs during
   this session ended in an `INTERNALERROR`/hang near the end, so budget for a retry and do
   not read a truncated run as a result.
7. **Never edit `src/` while the suite runs** — several tests read module source.
8. **The dev install is `--extra studio --extra text2image --extra rig --extra music`.** A
   bare `uv sync --extra studio` leaves some tests failing on a missing pack rather than
   skipping.

## Staged sequence

Each stage ends green and, from stage 3, demonstrably usable. Per CLAUDE.md, a manual change
ships in the same commit as the behaviour it describes, so each stage carries its own slice
of the chapter; the manual stage is the renumbering and the cross-document counts.

**Stage A — finish the registration sweep and get back to green.** Everything under
"Still to do" that does not need the engine: the `main.py` arms, `_WORKSPACE_ARMS`,
`AppState.mason`, the skeleton with placeholder panes, the landing and overlay entries, the
prose counts. The mode opens on an empty workspace and does nothing. **This is the first job
of the next session**, because until it is done a genuine regression cannot be told from a
known gap.

**Stage B — the shared viewport leaf.** Extract `studio/_view_frame.py` from `clay_view.py`
as pure code motion, with Clay's suite green before and after. Gate: the existing Clay tests
plus one asserting `ClayView` still exposes every moved name.

**Stage C — the pure engine, headless.** `refs`, `nodes`, `document`, `edits`; then `scene`,
`ops`, `pick`; then `terrain`. No UI. **Write `tests/mason/test_mason_imports.py` first** —
the pin is the contract — then the document tests (uid addressing, reparent refuses a cycle,
add/remove/move round trips, dirty is a comparison), the resolver tests (the five rules, a
hidden ancestor, instance expansion, `owner_uid`), and the terrain tests (a brush outside the
rect changes nothing; the edit's `cost` is what it claims).

**Stage D — serialisation and the three exporters, still headless**, including the
`viewer/gltf.py` and `viewer/glbwrite.py` camera and light extension. Gates: two saves are
byte-identical; the version gate refuses a newer file; a missing member is refused; a missing
library reference opens with a proxy and is listed; the GLB round trip with cameras, three
light kinds and one mesh shared by six nodes; **an existing Clay document's GLB is
byte-identical before and after the writer change**; the manifest and the GLB name the same
nodes; the OBJ's bounds match `world_bounds` and it reports what it could not carry.

**Stage E — the viewport and placing.** `mason_view.py`, `mason_assets.py`, the panes, drag
from the library. The node-proxy test goes in before the instancing code.

**Stage F — hierarchy, prefabs, lights, cameras, terrain in the UI.**

**Stage G — the library round trip.** `import_mesh`, the `.wscn` sidecar, `asset_exits`,
`asset_open`, a scene thumbnail through `viewer/capture.py`.

**Stage H — the manual**: `31-mason.md`, `17-dressing-a-scene.md`, the fifteen renames, the
`PARTS` widening, the overview and README and INVARIANTS counts, the shortcuts section, and
the new "adding a mode" section in the extending chapter.

Stages A, C, H and the pure half of D suit a Sonnet subagent given a written spec. Stages B
(code motion under Clay's tests), D's writer change (a shared, ceiling-guarded file), E's
cache key, and F's prefab rules want review.

## Verification

    uv sync --extra studio --extra text2image --extra rig --extra music
    uv run pytest tests/mason -n 0
    uv run pytest tests/test_mason_mode.py tests/test_modes.py -n 0
    uv run pytest tests/test_studio_smoke.py -k rail -n 0
    uv run pytest tests/manual -n 0
    uv run pytest
    uv run ruff check .
    uv run python scripts/preflight.py --fast
    uv run python scripts/exercise_mode.py mason
    uv run python scripts/screenshot_modes.py

On a Linux checkout, prefix the GL ones with `xvfb-run -a -s "-screen 0 1920x1080x24"`, and
compare against a baseline run rather than against zero.

End to end, by hand: Mason → New scene → place a library asset and a primitive → group them
→ make a prefab → place three instances → add a point light and a camera → sculpt the ground
→ Save → close and reopen → export all three formats → open the exported `scene.glb` in
Create's viewport and confirm the hierarchy, the instances, the light and the camera
survived → Export to the library and confirm the row reopens in Mason.

## Open questions

1. **No performance numbers exist.** The culling threshold, `MAX_PLACED`, the asset cache's
   byte budget and `MAX_TERRAIN_SIDE` are all constants that stored documents will be keyed
   on, and this repo's rule is that such a constant gets a dated document in
   `docs/measurements/` *before* it is fixed. Measure with a real scene.
2. **VRAM.** Sixty textured library assets is real video memory, and `check_vram` guards the
   *queue*, not the studio. A scene large enough to matter is a new failure mode with no
   admission control behind it.
3. **A material override costs a second upload** of identical geometry, because the override
   is in the cache key. If overrides turn out to be common the fix is a per-draw material
   uniform, which is a renderer change and out of scope here.
4. **`panes/library.can_drag` lifts only finished 2D references today.** Dragging mesh rows
   into Mason means either widening that predicate — which changes what Create's drop slot
   sees — or a Mason-specific payload. That is a decision to take with the library's owner.
5. **Units and handedness.** The manifest asserts metres and Y-up, and nothing in the repo
   states a scene unit today. This plan is *making* that decision, so it belongs in
   `docs/INVARIANTS.md` rather than only in an exporter.
6. **Terrain material layers / splat maps are out of scope** (one material). Adding them
   later changes the document format *and* the exporter, so if they are wanted, say so
   before the `.wscn` version is frozen.
7. **The engine manifest's schema is ours to invent**, and no importer exists to test it
   against, so it is verified by shape rather than by a round trip through an engine.
