# Generating meshes

The mesh stage is where the reconstruction engine runs. It costs roughly two minutes of GPU per
attempt, so everything in this chapter is arranged around deciding what to spend that on. All of it
lives at Create's **Mesh** stage. The left column is a stack of cards. It starts with the **Source**
chip and **Attempts** (1, 2 or 3), followed by the reconstruction settings: **Triangle budget**,
**Size** and **Rig**, with everything else in one collapsible **Advanced** card. The **Generation
plan** card is pinned at the foot of the column, as tall as its lines, and holds the persistent
**Make 3D** button under them; the cards above it scroll. **Reset mesh
settings…**, in the **New…** menu, resets mesh controls after confirmation while keeping the chosen
source.

## Starting from a reference

The normal path is promotion: take a finished 2D asset and run the mesh stage from its image.

Press **Make 3D** on a finished reference — over the picture on the canvas, on its card in the
Attempts tray, or at the top of the inspector's **Next step**; it is one button. This explicitly
chooses the source, opens the Mesh stage and starts the cutout check; it does not spend GPU
time until the check passes. Inspecting a different attempt keeps the chosen source and draft.
Library promotion remains available too. A reference drawn on an image model that this version has
since removed can still be made 3D: the mesh stage never runs the image model, so the model the
picture was drawn with stays in its record as history and is not checked again.

With no reference chosen the Mesh canvas is a drop zone: *Drop an image here, or pick a reference*,
with a thumbnail for each finished reference in this creation — click one to make it the source.

The **Source** chip names and previews the image that Make 3D will use. It accepts a library card
drop or an image file. With no explicit source, a selected mesh can use the reference it came from.

The new job is an ordinary image job whose input image is the reference's, recorded as a child of
the reference so the library can show them as one lineage rather than two unrelated rows.

Everything the Mesh stage holds is an **override**. Omitting one means "keep what the reference
recorded", which is not the same as sending the reference's value back — so the selects offer
that unset choice as their first entry and that is where they start. When the chosen reference has a
recorded value for the field, the entry names it — "From reference: High detail" rather than the bare
"keep the reference's" — so leaving a control alone is an informed choice, not a guess. A reference
with nothing recorded for that field (an older job, say) still gets the generic wording.

Two settings are exceptions and are always sent explicitly: the rig checkbox and the
centre-and-enlarge checkbox. Both are the Mesh stage's own decisions, and an omission would let
the promotion inherit whatever the reference happened to record.

Derived values never carry across. Anything the worker recorded about the *source* run's artifacts
— the composed prompt, the mesh report, the applied transform — is stripped, so a new job never
wears a quality verdict about a mesh that does not exist yet. See
[Rerun and promotion](33-library-and-jobs.md#rerun-and-promotion).

## Checking the cutout

**Make 3D** does not submit straight away. It checks the cutout first, and opens **Check the
cutout** when there is anything to see: the subject cut out of the
reference, drawn over a checkerboard so transparency reads as transparency, with the panel naming
which matte produced it — the reference's own alpha, the BiRefNet model, or the corner fill that
answers when BiRefNet's weights are not installed.

The matte is the single decision that most often turns a good reference into a solid slab, and it
used to be made inside the reconstruction engine two minutes after you had committed. The panel is
where you see it first. It also carries the reference's own quality report: the reasons it may not
reconstruct, and the milder warnings — edge contact, a very thin subject — that are worth knowing
before the spend rather than after.

**The picture in the panel is the picture the engine rebuilds from.** Accepting it saves that cutout
and hands it to the reconstruction with the engine told to keep the alpha rather than cut its own.
That is worth stating because it was not always true: the panel used to show you Realmspinner's cutout and
then send the engine the untouched reference, which the engine cut again with a *different* copy of
the background-removal model. If you ask for several candidates, all of them get the same approved
pixels, so a difference between two candidates is the reconstruction seed and nothing else.

If you edit the reference after opening the panel — through **Fix matte**, or by saving in Inker in
another tab — the cutout on screen stops describing the file, and Realmspinner cuts it again rather than
building from pixels that are gone.

Three buttons:

- **Accept** queues the mesh job. When the report refused the reference the button reads
  **Build anyway** instead, and it submits with the refusal overridden — a confirm rather than a
  refusal, because the rules are heuristics about composition and you can see the image they are
  arguing about. What you must not do is spend two minutes of GPU by accident. The override is
  recorded against those exact pixels, so it applies to a rerun of the same job and to nothing else;
  edit the image and the composition rules apply again.
- **Fix matte** does not re-remove the background. It opens the reference in Inker with the cutout
  already folded into its alpha, as one undoable step, and the eraser and the brush then edit the
  matte directly; see [Inker](25-inker.md#fixing-a-matte). Save, then press **Make 3D** again.
- **Cancel** leaves everything as it was.

A matte you edited and saved travels to the engine as the image's own alpha, and the job records
that it was approved — the engine is told to keep the alpha rather than cut its own. A soft edge you
painted stays soft; nothing along this path flattens it to a hard cut.

Settings has a **Don't ask for clean cutouts** switch, **on by default**, that skips this panel for
the case it exists to catch the least: a cutout with no report warnings or refusals, made by the
BiRefNet model itself rather than the corner-fill fallback. Anything the report flags, or anything the
fallback made, still opens the panel whether the switch is on or not — and the skipped submission is
the identical **Accept**, not a different route to the same job. Switch it off to see the panel before
every Make 3D; a choice you have made either way is kept.

## Attempts

The reconstruction engine is deterministic in its seed, and its failure mode is a lottery: the same
reference comes back clean at one seed and with a hole through the shoulder at another. **Attempts**,
near the top of the left column (the Reference stage's equivalent is **Images**, which counts
pictures rather than meshes), is how many meshes one press tries — 1, 2 or 3 — so you can pick the
best. The **Generation plan** card pinned at the foot of the column (the
same card the Reference stage draws, with **Make 3D** inside it) changes with the choice: it says "N candidates ·
about N×2 minutes of GPU", because this is the one control in the pane that multiplies what the button
spends. Under that it shows the queue, then either *Ready to generate.* or, in red, why Make 3D is off
(*Needs attention*) or what the last press was *Refused* for, with **Choose a reference** and **Open
model setup** buttons where those are the fix. **Open model setup** appears when the 3D engine — its
runtime or its weights — has not been downloaded, and it opens Settings with exactly those rows
ticked; **Make 3D** stays off until they are installed, so the missing download is a sentence here
rather than a surprise after the cutout.

Each candidate is an ordinary mesh job: same validation, same VRAM admission, same worker. The first
keeps the mesh seed you pinned, so a pinned seed still reproduces; the rest draw fresh ones.

New candidate batches are saved immediately as attempts in the current creation, and remain
visible in the Library. The horizontally scrolling **Attempts** strip contains every loaded attempt
for this creation. Click one to inspect it; **Load settings** loads its mesh settings and source
explicitly. **Favourite** a result without discarding the rest, or pin one and compare it with another
finished mesh using synchronised views. Rig and Export remain stages on the header's rail.

Older undecided candidate batches retain their Keep/Discard picker so existing work can still be
settled. Keeping or discarding a legacy group never deletes files without the separate confirmation.

The count applies to **Make 3D**, and that includes a mesh made from an upload (see below).

## Starting from an upload

You can skip the reference stage entirely. Press the **Source** chip (it reads **Choose an image...**
while nothing is chosen), or
drop an image file onto the window. Nothing is queued: the image is imported as a finished reference,
the Source chip shows it, and it appears in the Library like any other reference. Then **Make 3D** works
as it does for a generated one: it checks the cutout, and once that passes, makes as many meshes
as **Attempts** says.

Uploads are bounded at the door, and both limits are checked before anything is written:

- **20 MB** on the file itself, checked before the image is decoded.
- **16 megapixels**, read from the image header before any pixel is decoded — a flat 20-megapixel
  PNG is a few hundred kilobytes on disk and hundreds of megabytes decoded.

Anything larger is refused rather than allocated. Whatever format you supply is re-encoded to PNG,
since the reconstruction engine only decodes PNG and JPEG; transparency is preserved when the
source had it, because a pre-matted upload lets the engine's background detection do less work.

A good upload is the same thing a good generated reference is: one complete subject, not cropped,
on a plain background.

## Mesh parameters

The **Mesh** section shows three things: **Triangle budget**, **Size** and **Rig**. The rest of the
reconstruction settings — **Mesh resolution**, **Finishing**, **Background**, **Seed**, **Centre and
enlarge the reference** and the engine's launch flags — are in one **Advanced** card below them, closed
until you open it.

**Mesh resolution** (in **Advanced**) is the geometry resolution handed to the engine, supplied by a
platform preset:

| Platform | Resolution |
| --- | --- |
| 2D | 512 |
| 3D | 1024 |

The question the select is asking is what the asset is *for*: a 2D asset is going to be seen flat
and small, a 3D one in a scene. Higher resolutions cost more VRAM and more time, and on a card that cannot
hold both models at once they may need the exclusive VRAM mode (`REALMSPINNER_VRAM_EXCLUSIVE=1`), which
stops the reconstruction engine while the image model runs.

**Triangle budget** picks how the finished mesh gets to a game-sized triangle count, and it offers up
to four families of entry, each drawn only when this machine can actually run it:

- **Game-ready · 2,000 / 5,000 / 10,000 / 20,000 triangles** brings the surface down to that count inside the
  model job, in Blender, and what it does on the way depends on the **Finishing** choice just
  below it. This is what turns the engine's ~300k-face output into something an engine actually
  budgets for, and **5k is the default** — Crash-Bandicoot-range fidelity on a whole-character
  reconstruction. That figure came from a measurement document that is now lost (the backup it
  lived on is gone), so treat it as the shipped default rather than a number you can check. It
  needs Blender (the `rig` extra); without it, this family does not appear.
  **Finishing** is a combo drawn in **Advanced** whenever a Game-ready rung is chosen, and it is not
  drawn for Simplify, Raw or Custom. **Preserve shape**, the default, welds coincident vertices and
  then collapses triangles, keeping the mesh's own UVs and its existing colour, roughness and
  normal maps: no voxel pass, no new unwrap, no rebake, so it does not close holes. **Repair and
  close holes** is the older path: a voxel pass that seals the gaps a reconstruction leaves, a
  fresh unwrap, and the old colour, roughness and normals baked onto the new geometry, at the cost
  of slightly rounding sharp edges. Repair refuses a mesh that mixes cutout (mask) and blended
  materials, because one atlas cannot carry both; it says so before the bakes start rather than
  after minutes of them, and an opaque mesh skips the opacity bake it would not use. A job saved
  before the choice existed ran Repair, and keeps that reading.
- **Simplify · Draft/Standard/Detailed** are the gltfpack tiers (20k/50k/100k) — a plain triangle
  reduction of the reconstruction's own surface, no rebake. They need `gltfpack`
  (a one-time manual drop into `vendor/gltfpack/`, or `REALMSPINNER_GLTFPACK` pointed at a copy you
  keep elsewhere — see [Installation](40-installation.md#gltfpack)); without it, this family does
  not appear either.
  **Without Blender, Standard is the fallback** — the mesh still gets a second pass, just not the
  in-Blender remesh.
- **Raw** ships the reconstruction as the engine wrote it, ~300k faces. Always offered.
- **Custom** takes a triangle count directly; it appears once `gltfpack` is present, alongside the
  Simplify tiers. Picking Custom starts the count at 50,000, and a count outside 5,000 to 250,000
  is refused in the plan footer, naming the **Triangles** field, before the cutout check.

With neither Blender nor `gltfpack` on this machine, the pane draws no **Triangle budget** row at all, not
a combo with one entry, and the mesh ships as Raw; `realmspinner doctor` says why. Your saved Game-ready choice is not
overwritten: while the tool is missing the pane shows and sends Simplify: Standard (or Raw with neither tool), and the
saved rung comes back once Blender is installed.

A saved Game-ready budget that is not one of the four rungs, say 7,500 from before the ladder
last changed, is kept as it is. It shows as its own **Game-ready · 7,500 triangles** entry until you pick
something else, because opening the pane never changes the budget by itself.

Every gltfpack pass (Simplify, Raw excepted, and Custom) checks its own output before publishing —
did it keep the UVs, both PBR maps and the material assignment, and add no extension the source
lacked — so picking a Simplify tier here does not risk a silently worse mesh: if the check fails,
the job ships the raw reconstruction instead and the mesh's status line names the reason
("the triangle budget was not applied…"). A Game-ready remesh is checked the same way, against the
mesh it replaced, and a failure there keeps the optimized (pre-remesh) mesh instead. What neither
check catches is silhouette damage — no number here scores a mangled blade or a lost finger, so look
at the result. Whatever a job recorded is always recoverable: rebuild at **Raw** from
[Triangle budget](#triangle-budget) in the inspector and the mesh comes back from `source.glb`,
which no tier in this control ever touches.

**Size** is the physical size the finished GLB is scaled to, along its largest dimension. Drag it —
about a centimetre per pixel — or double-click to type a figure; the readout carries the unit, and at
zero it reads *unset*, which means "keep whatever the reference recorded". That in turn falls back
to 1 m. There is no upper stop, because there
is no largest asset: a wall section is legitimately 8 m.

While the size is unset and the source reference's prompt names something the app has a figure for,
a suggestion appears under the control — *barrel — usually 0.9 m*, with a **Use 0.9 m** button. It is
a plain table of typical real-world sizes, not a measurement, and it is never applied for you: unset
is a real answer, and a press is what turns a suggestion into a decision. The match is crude and
takes the last noun it recognises, so "a sword in a barrel" is a barrel; if it names the wrong thing,
ignore it and drag the control.

Scaling is optional, but **grounding is not**. Every finished mesh is centred on X and Z and has its
lowest point put at Y = 0, whether or not a size was asked for. A pivot sitting at the centre of the
reconstruction volume is a manual fixup on every Godot or Unity import, so the app does it for you
on every job.

**Background** (in **Advanced**) chooses how the engine mattes the input image: **Automatic**,
**BiRefNet cutout** or **Plain-background threshold**. Automatic is the default and is right almost
always; the other two exist for images it gets wrong.
A cutout you approved in the panel overrides this and pins `auto`, which is the mode that keeps an
existing alpha — any other setting would re-cut the matte you just approved and make the approval a
lie.

**Seed** here is the reconstruction's own seed, separate from the image seed on the Reference stage, with
its own **Reroll** button and its own **Lock seed** switch. Until you set one it reads *random* — a
new seed every time — rather than a number; **Set a seed** (or the first Make 3D) gives it a value, and
**Random** puts it back. **Lock seed** appears once there is a seed to lock.
Unlocked, every Generate draws a fresh seed for the next one — the engine is deterministic in its seed,
so pressing the button twice on the same reference with the seed left alone would give you the
identical mesh twice. Lock it when you want exactly that.

**Centre and enlarge the reference** (in **Advanced**) recentres the subject and scales it to fill the
frame before the engine sees it. It is off by default: the engine does its own cropping, and whether
doing it twice helps or hurts has not been measured. Treat it as an experiment rather than an improvement.

The **Rig** section holds **Rig when the mesh lands** and a skeleton picker. Without Blender it stays on
screen, greyed, and its tooltip gives the same reason as the Rig segment in the stage rail. Without
Blender, Make 3D also never requests a rig, even if the setting was on when it was saved.
See [Rigging and posing](#rigging-and-posing).

## Advanced

Below Rig is a collapsible **Advanced** card, closed by default; click its title row to open it, and
the column remembers whether you left it open, across restarts. It opens onto the settings listed
under [Mesh parameters](#mesh-parameters) (resolution, finishing, background, seed and centring), then
an **Engine** group holding the seven launch
flags the reconstruction engine (`trellis-server.exe`) itself accepts: **Band**, **Texture resolution**,
**Sparse-structure guidance**, **Structured-latent guidance**, **Token budget**, **Decimation** and
**Atlas resolution**. A findings sweep could already set every one of these — they are exactly
`service.sweeps`'s `SERVER_AXES`, the set that decides how the engine process is launched — but until
now an ordinary Make 3D had no door onto them at all.

Every control here starts **unset**, and unset is a real value: it means "the engine's own default
runs", the same rule every other control on this stage follows. Leave the whole section alone and
nothing about this changes from what shipped before it existed. **Decimation** is the one exception
worth knowing — its unset reading is `-1`, not `0`, because `0` is itself a meaningful setting here
("turn decimation off and ship the full reconstruction"), not an empty box. **Band** takes 1 to 64
voxels and **Texture resolution** 128 to 4096 pixels: a value outside that range is pulled to the
nearest end when you leave the field, because the engine refuses anything else.

Changing any one of these restarts the engine process for the job it applies to — that is what a
launch flag *is* — so the note under the header says so, and this is not a section to open on every
run. It exists for measurement work: sweeping the token budget or the atlas size to find where a
default should move, the way earlier measurement passes on this same axis
already did through Review's sweep form. See
[Configuration](41-configuration.md#environment-variables) for what each flag does and the environment
variable that sets the same thing app-wide, and [Review](34-review.md#what-works) for how a findings
hint next to one of these controls is read.

## Triangle budget

The reconstruction is kept, permanently, as `source.glb`, and nothing ever overwrites it. The
`model.glb` you actually use is derived from it by optimising and then grounding. That is what makes
changing your mind about triangle count cheap: rebuilding at a different budget is a couple of
seconds of mesh processing rather than another two minutes of reconstruction.

The control is in the inspector at the **Mesh** stage, under the collapsed **Triangle budget**
header. It appears only on jobs that have a `source.glb` — older jobs and rig jobs do not.

Five tiers exist in the code: Raw (as reconstructed — the engine has already simplified it to about
300k faces at resolution 1024, 150k at 512, unless `REALMSPINNER_TRELLIS_DECIM=0` is set), Draft (20k),
Standard (50k), Detailed (100k) and
Custom. `gltfpack` — the binary every decimating tier runs through — is a one-time manual drop into
`vendor/gltfpack/`, or `REALMSPINNER_GLTFPACK` pointed at a copy you keep elsewhere — unlike the
reconstruction engine, it is never a Settings → Models download, and not something the
checkout brings with it; see [Installation](40-installation.md#gltfpack). When it is there this panel
offers the whole list, and Custom gains a triangle-count field with its own valid range. When it is
not, `realmspinner doctor` says so and every tier ships the engine's own output instead of failing.
This is the gltfpack half of the same ladder **Triangle budget**, described under
[Mesh parameters](#mesh-parameters), offers on the generate form under "Simplify:" — Standard is
this panel's default when `gltfpack` is present (Raw when `gltfpack` is absent, the only tier it
then offers), and it is also what a new job falls back to when Blender is missing and no
Game-ready remesh can run. This panel is where you change your mind about a tier after the fact, on
a mesh that already exists rather than on a job you are about to wait two minutes for — and where an
old job that predates this default, or ran with gltfpack absent, gets a budget applied for the first
time. It does not offer the Game-ready rungs; those are [Game-ready remesh](#game-ready-remesh)'s
own control, further down this same stage.

Two things the panel will not hide from you. A retarget refuses to run on a job that is still queued
or running, because its write would collide with the worker's. And a retarget makes a rig, its saved
poses and its rendered sheets describe a mesh that no longer exists — those are minutes of your
work, so they are reported rather than deleted, and the warning is shown *before* the button rather
than after. Everything else derived from the mesh (STL, OBJ, FBX, collision, textures) is deleted,
because those describe the old geometry exactly. A retarget rebuilds from the original
reconstruction, so it also undoes a remesh: the warning says so beforehand, and the Remesh panel's
"Last remesh" line is cleared once it lands — but the mesh a retarget replaces is not gone. It is
kept under [Earlier meshes](#earlier-meshes), and a Restore there puts it back. The button also
greys pre-emptively when a rig, a sheet, or a sibling remesh or re-texture is already queued or
running against this same mesh — the same collision the service refuses, now stated before the
press rather than after it.

Press **Rebuild mesh** to apply. The served `model.glb` is never written in place: the new mesh is
staged and swapped, so nothing reading the old one sees a truncated file.

## Game-ready remesh

A triangle budget keeps the reconstruction's surface and thins it. A remesh replaces the surface:
the mesh is rebuilt at a **triangle** budget — quadriflow where the input allows it, a decimate
fallback where it does not, both described below — unwrapped afresh, with the old colour, roughness
and normals baked onto the new geometry. This is the step that turns the engine's ~300k-triangle
output into something an engine budgets for, and it is what every commercial generator calls
"game-ready". Because the high-resolution geometry ends up in a tangent-space normal map, even a
2k-triangle prop keeps most of the detail the budget threw away.

**This is now the default, not an extra step.** Create's **Triangle budget** combo (see
[Mesh parameters](#mesh-parameters)) offers Game-ready 2k/5k/10k/20k, and 5k runs automatically on
every new mesh whenever Blender (the `rig` extra) is installed — chosen on a full reconstruction
to keep silhouette, face markings and other small detail intact (the measurement document behind
that choice is lost; the figure is unmeasured here). It runs inside the model job itself, between
the triangle-budget step and grounding, so a rig queued for the same mesh sees the final, remeshed
geometry rather than the ~300k-triangle reconstruction. The panel below is for **reworking an
existing mesh** — trying a different budget, or remeshing a mesh that predates this default.

The control is in the inspector at the **Mesh** stage, under the collapsed **Game-ready remesh**
header, between the triangle budget and the surface texture. It runs in Blender, so it needs the
`rig` extra; without it the header says so and offers nothing.

**Triangles** is the budget: 2k, 5k, 10k, 20k or Custom. **Bake at** is the texture resolution,
matching the mesh's own atlas by default. **Close holes first** runs a voxel pass before the remesh,
which seals the gaps a reconstruction leaves at the cost of slightly rounding sharp edges, and
defaults **on** — every trellis reconstruction is non-manifold to begin with (hundreds of thousands
of vertices, on a real character), and without the voxel pass the decimate fallback below collapses
the mesh instead of producing something usable. Leave it on unless you have measured a reason not
to.

A remesh from this panel is a queued job, like a re-texture, and its product lands over the source
job's `model.glb`; `source.glb` is never touched. Two things follow. A later **Rebuild mesh** at a
triangle budget rebuilds from the reconstruction, which replaces the remesh — `model.glb` is derived
and `source.glb` is the authority, always — but the remeshed mesh is kept under
[Earlier meshes](#earlier-meshes) rather than lost, and a Restore there brings it back. And a remesh
changes geometry, so every derived export is deleted and a rig, its poses and its sheets are
reported stale before the button, exactly as a retarget reports them.

Coincident vertices are welded before the remesh runs, and that is not a detail. glTF stores one
position per texture coordinate, so *every* GLB splits its vertices along each UV seam — which makes
a mesh non-manifold before anything is actually wrong with it. Until 2026-08-30 that weld was
missing, so every remesh silently took the decimate fallback below without the voxel pass making up
for it.

**Quadriflow almost never runs on a trellis reconstruction, even after the weld and the voxel pass.**
It refuses non-manifold input, and it *returns* a cancelled result rather than raising when it does —
a bug fixed on 2026-09-23, since which a refusal correctly falls through to the fallback instead of
silently shipping the untouched reconstruction under a "quadriflow" label. Measured on a real
character: the voxel pass does make the surface manifold and single-shell, and quadriflow refuses it
anyway ("the mesh needs face normals that point in a consistent direction"). Why it refuses a surface
that measures manifold is unresolved. In practice, expect the decimate fallback to be what runs.

The last remesh's result is printed under the button: triangles, the method (`quadriflow` or
`decimate`), and the bake size. A decimated mesh is not reported as a failure of a quad path it was
never promised — the line reads "N triangles (remeshed and re-baked)" either way, and it also records
whether anything the mesh had before (UVs, both PBR maps, material assignment) was lost, in the same
terms a triangle tier's own `tiercheck.compare` measures.

## Surface texture

A re-texture gives a finished mesh a new skin without touching its geometry. The control is in the
inspector at the **Mesh** stage, under the collapsed **Surface texture** header, below
[Triangle budget](#triangle-budget) and [Game-ready remesh](#game-ready-remesh) — all three are
operations on an asset that already exists, and a prop that never visits the Rig stage still has
them.

Describe the surface you want ("rusted iron", "mossy stone", "painted wood"), set how far the
restyle is taken, pick an atlas size, and press **Re-texture mesh**. What happens then is four
stages: the mesh is rendered flat from ten directions (the six axis views plus four upper
three-quarter diagonals, which see past overhangs and through openings), each render is restyled by
the image model, each restyled view is projected back onto the mesh's own UV layout with a weight
image saying how squarely it faced the camera, and the views are combined into one atlas by a
weighted mean.

Unlike a retarget, this is a queued job rather than a couple of seconds of processing — ten image
generations is a real amount of GPU — so it takes its turn behind whatever else is running, and it
is refused outright on a job that is still queued or running.

**Restyle strength** is how far each view is taken from the mesh's current look. Low keeps the
shapes and recolours them; high reinterprets them. Un-anchored, past about two-thirds the views
start disagreeing with each other about what the object is, which shows up as a muddled atlas
rather than as a bolder one — the top of the range is only worth reaching with the geometry anchor
on, where the depth hint keeps every view telling the same story.

**Anchor to geometry (depth)** renders the mesh's own depth from every view and uses it twice.
Each restyle pass is held to it through a depth ControlNet, so the image model can invent surface
detail without moving edges. And the projection depth-tests every texel — a surface a camera could
not actually see contributes nothing, so an overhang's front-view colours no longer smear onto what
hides behind it. It needs the depth ControlNet downloaded (Settings → Models, under Conditioning);
the button will name the download if it is missing. **Anchor strength** is how firmly each restyle
is held; the default is the model's own.

**Atlas size** is the resolution the new texture is written at. The default, **Match the mesh**,
keeps the mesh's current atlas resolution, held to at most 2048 (a larger atlas is baked at 2048
and the log says so); a fixed size is offered but changes nothing a view covers
and shrinks everything no view does, which keeps its old colour.

Three things the panel tells you rather than letting you find out.

**Your rig is safe.** A rig, its saved poses and its rendered sheets reference geometry, and a
re-texture changes no geometry — so unlike a retarget, none of them is invalidated. The exports that
*do* carry the skin (OBJ, FBX, the texture zip) are deleted and rebuilt on next request. STL and the
collision hull are geometry with no material in them and are left alone.

**The occlusion test is per run, and the panel says which kind you are configuring.** Un-anchored,
a view's contribution is weighted by how squarely each surface faces it, not by whether anything is
in the way — so on a mesh with an overhang, the front view's colours can be smeared onto whatever
hides behind it (on a convex prop this never arises). That is the warning under the form, shown
exactly when it applies; turning the anchor on replaces the facing-only weight with a real
depth test and the warning goes away because the limitation does.

**Coverage is reported.** After a run the panel says what fraction of the atlas any view could speak
about at all. A texel no view could see keeps its old colour rather than being invented — the inside
of a barrel stays the inside of a barrel — so a low coverage figure means most of the mesh kept its
previous skin, which is worth knowing before you conclude the prompt did nothing. An anchored run
also reports how much was actually repainted after the depth test; the two figures do not mean the
same thing, and the line names which kind of run produced it.

`source.glb` is never touched. A re-texture is a derivation, exactly as a retarget is, so the
reconstruction stays the thing both of them rebuild from.

## Fixing a finished asset

A mesh can be the right shape and still be wrong for a game: the wrong size, a pivot that is not on
the floor, or a front that is not the way an engine expects. None of the controls above is for that —
each one rebuilds the mesh — so the **Finished asset** header, in the inspector at the **Mesh** stage
above Triangle budget, changes only those three things and leaves the surface and the triangles as
they are.

**Size (m)** scales the model so its longest side is that many metres, and **Ground** centres it on
the origin and puts its lowest point on the floor. **Apply size and ground** does both; leaving the
size at what it already is just re-grounds. It is allowed on a rigged asset, because it changes the
geometry rather than turning it, and the panel and the result both say that the rig, its poses and
its sheets now describe the old mesh.

**Make current front the model's front** appears only on an asset that already carries a recorded front
(one set in an earlier version of the app; the viewport control that set it is gone, and a new front
can no longer be recorded). A record is not the mesh: an engine imports `model.glb` and sees whichever
way it was built. This turns the mesh so the recorded direction faces +Z, the way glTF and every
engine's "forward" expect, and then clears the angle, since the model now faces front by itself. It is
off for good while the asset has a rig: a rig and its poses were fitted to the mesh as it stood, and
turning only the mesh would leave them describing a different one. To face a model forward now, turn
it in [Clay](27-clay.md) and export it.

Every press keeps the mesh it replaces under [Earlier meshes](#earlier-meshes), so any of it can be
undone, and every export derived from the old mesh is dropped and rebuilt on the next request. A
restore across a front change brings the front back with the mesh. The **Ready for an engine?**
checklist's Pivot and Size lines send you here.

## Earlier meshes

A retarget, a remesh, a re-texture and a resize all write over the same `model.glb`, so each one used to
throw away whatever the last one built — remesh, then re-texture, and the quad mesh was gone the
moment the new skin landed; retarget after either, and both were gone. That is no longer true. Each
one keeps a copy of the mesh it is about to replace before it publishes, and the collapsed
**Earlier meshes** header — in the inspector at the **Mesh** stage, below Surface texture — is where
those copies live.

Each row names what replaced that version (triangle budget, game-ready remesh, surface texture, a
resize or front change, or an earlier restore), when, and a one-line detail — a quad count, a view
count, a file size. Press
**Restore** on a row to put that mesh back. The panel names what goes stale first: restoring across
a triangle budget or a remesh changes geometry, so the rig, its poses and its sheets describe the
old mesh again, the same warning a retarget gives; restoring only across a re-texture changes
nothing but the surface. A restore is itself kept, so restoring is not a one-way trip either — you
can always go back to what you had before pressing it.

Up to four earlier meshes are kept per asset; keeping a fifth drops the oldest, file and row both,
so after four changes in a row the mesh you started from is gone from the list. Restore it before the
fifth change if it matters.
Deleting the asset deletes its earlier meshes with it, the same as every other file in its folder,
and they are never offered as a download — they exist only so a Restore has something to restore.

## Mesh audit and mesh report

The app measures a finished mesh in two deliberately separate ways, and they answer different
questions. Both appear in the inspector's **Details** tab under **Mesh quality**.

The **mesh report** answers *will an importer accept this, and will it sit on the floor*. It is the
topology-and-metadata check: triangle count, material count, whether the surface is **watertight**,
and whether the pivot is at the model's feet. It also carries the pass/fail badge and its reasons.
Only this measurement may use the word watertight, because only this one proves it.

The watertight figure is measured on a **welded** copy of the mesh — vertices at the same position
merged first. The file itself is read unwelded, because the UV and material checks need to see the
split vertices, but a UV atlas splits a vertex at every seam and each of those splits reads as a
boundary edge. Unwelded, the check was mostly counting texture seams and calling almost every mesh
open. When the report says a mesh is not watertight it names the boundary edges and components it
found *after* welding; the raw unwelded counts are still recorded, because how badly a file is split
is its own question for a rig or an exporter.

A mesh that went through a Game-ready budget also gets **finishing lines** at the top of the block,
before the report: `Finished mesh: N triangles; requested M.`, then `Silhouette coverage lost: x%;
added: y%.` and `Source opening pixels filled: z%.`, then any UV or material change the
check found. They compare the finished mesh with the one it replaced in shared-frame silhouettes.
Where a measurement could not be made the line says *unknown* rather than passing it, and a
finishing that failed or was cancelled says so and points at `source.glb`. These are advisory
numbers, not a readiness grade: they cannot tell a lost handle from a lost shadow, and the
Preserve shape default, which does not seal holes, is exactly the case where the opening figure is
worth reading.

The **mesh audit** answers a different question: *can you see through it*. It is a silhouette check
— render the mesh from several angles and measure how much of the subject is holes — reported as
"visible openings" and a percentage. That is what a player actually notices, and it is not the same
property as watertightness at all. A mesh can be watertight and still look wrong, and vice versa.

**Read that percentage in one direction only.** A high reading means a hole, and it means it
reliably. A *low* one means no hole was seen, which is not the same as a good mesh — the most common
way reconstruction fails is a solid, featureless slab, and a slab has no openings at all. Measured
against 84 reviewed meshes, the accepted ones had *more* visible openings than the median discarded
one, so a near-zero reading is close to no information. The app says so where it shows one: nothing
in the interface paints a low figure as a pass, and the inspector adds "a solid, featureless mesh
scores this too" underneath it.

Neither measurement can fail your job. If either cannot be computed, the failure is logged and the
job still completes: the GLB is already on disk, and a missing verdict is better than a lost mesh.

## Rigging and posing

Rigging fits a skeleton to a finished mesh and skins it, so the mesh can be posed. Posing sets each
joint's rotation and saves the result under a name. Both are optional, both need Blender, and both
are covered here.

This section is about rigging and posing **a generated asset** — the mesh in front of you, with its
own saved poses. A rig you build yourself, from shapes, bone by bone, with a timeline to animate it,
is [Clay's](27-clay.md#bones-and-rigs).

### Templates

Realmspinner does not invent a skeleton for your mesh. It fits one of eight shipped **templates**:

| Template | For |
| --- | --- |
| humanoid | Two arms, two legs, upright. |
| quadruped | Four legs, horizontal spine. |
| bird | Wings and a two-legged stance. |
| fish | A spine and fins, no limbs. |
| insect | Six legs. |
| serpent | A long chain of spine joints, no limbs. |
| tailed biped | A humanoid with a tail. |
| blob | No limbs — a body that bulges and sinks instead: slime, ooze, elemental, cloud. |

Each template is a small JSON file listing named joints at normalised positions inside a unit
bounding box. Fitting scales those positions onto the measured bounding box of your mesh — nothing
is learned and nothing is detected. That makes it fast and completely predictable, and it also makes
it **approximate on purpose**: a subject whose proportions differ from the template's will get
joints that are close rather than correct.

Because the fitted positions are approximate, they are written into `rig.json` alongside the rig, so
they can be corrected later without starting over. That is what **Adjust joints** does, described
under [Posing](#posing).

### Rigging a mesh

There are three ways to rig:

- Select a finished mesh in Create and press **Rig** on the stage rail. The stage's column holds
  the skeleton picker and **Rig this mesh**; the inspector beside it shows what the rig produced.
- Press **Rig** on a finished mesh's card in the library.
- Tick **Rig when the mesh lands** at the **Mesh** stage before generating, and pick a skeleton.
  The rig is then queued automatically as soon as the mesh finishes.

All three use the same skeleton — the one the picker is showing — and the same queue slot, so a
second press while one is running is refused rather than queueing a second rig of the same mesh.

### Rigging a mesh by hand

For a mesh none of the eight templates fit — a shape none of them was built for, or one you would
simply rather build yourself — the **Rig** stage, the **Pose** stage and a library card's right-click
menu each offer **Rig by hand in Clay**. It opens the mesh in [Clay](27-clay.md), where you add bones
(**Add Bone**, or mark any shape as a bone), place them where the joints are, and either bind the
shapes under them or have Blender weight the mesh to them: see
[Bones and rigs](27-clay.md#bones-and-rigs) and
[Rigging with Blender weights](27-clay.md#rigging-with-blender-weights). **Export to the library** from
Clay writes the finished rig beside the mesh it makes, as a new library entry.

Nothing about the rig changes because you built it by hand: it is the same `rig.glb` and `rig.json`
every other rig is, with `"skeleton": "custom"`, and Create's Pose stage, the exports and the
animated GLB all work on it.

### The A-pose trap

Here is the failure that catches nearly everyone, and it looks like a bug.

**The shipped humanoid template is an A-pose** — arms angled down. If your mesh is standing in a
T-pose with arms straight out, fitting that template runs the arm chain diagonally down through the
ribcage. Automatic weights then bind the arms to the chest, and every animation you play barely
moves them.

The symptom is "the arms are welded to the body". There is no "Measured joints" control here to
reach for — fitting the arm and leg lines to a mesh's own vertex cloud, and, with the ViTPose
weights installed, to the reference image's own landmarks, is something the character pipeline
does **automatically** for a T-posed reference, before the rig ever reaches this stage. It never
appears as a button you press.

If you rigged the mesh directly in this stage and hit the trap anyway, the fix is **Adjust joints**
(under [Posing](#posing)): move joints by hand. A hand correction always wins over the template's
guess.

### Where the joints come from

By default a skeleton is fitted by proportion: the template's landmarks are scaled onto the mesh's
bounding box, so a shoulder lands where the template says a shoulder is in a body of that height.
That is right for a subject standing in the template's own A-pose (arms angled down) and increasingly
wrong the further from one it is — a T-posed mesh is the case it fits worst; see
[the A-pose trap](#the-a-pose-trap).

For **humanoid** rigs there is a better first guess, and it is on by default when the pose model is
installed. Realmspinner measures the subject's joints off the reference image the mesh was reconstructed
from — the premise being that the mesh *is* that picture in three dimensions — and uses those as
the template. They are still scaled onto the mesh by the same fitter, so nothing else about rigging
changes; what changes is where the fitter starts from.

Two things about it are worth knowing:

- **It is all or nothing.** If any landmark is unconvincing — low confidence, missing, outside the
  subject's silhouette, or a figure whose knees came out above its hips — the whole measurement is
  discarded and the template is used. A skeleton with a measured arm and an assumed shoulder is not
  half-right; it is inconsistent in a way nothing later can detect.
- **Depth is always the template's.** One image fixes left–right and up–down and says nothing about
  front–back.

`rig.json` records which was used, so a rig can be told apart afterwards. Without the pose model the
bbox fit is used and nothing about your rigs changes — see
[Model weights](40-installation.md#model-weights) for the download. Set `REALMSPINNER_POSE_FIT=off` to
force the template everywhere, whatever is installed; it is a kill switch rather than an opt-in,
because the measurement already refuses itself whenever it is unsure.

Joints you have corrected by hand always win over both. See **Adjust joints** under
[Posing](#posing).

### Posing

Once a mesh is rigged, the **Pose** stage offers **Edit pose**. Pressing it swaps the viewport from
the mesh to the rig and enters pose mode.

Leaving the stage while a pose is unsaved asks first, and every way out asks — the rail, a library
card, the palette. A pose lives in the viewer until it is saved.

In pose mode:

- **Click a joint** to select it. A rotation gizmo appears on it; drag the gizmo to rotate.
- **Reset joint** returns the selected joint to its rest rotation; **Reset all** resets every joint.
- **Mirror** copies the pose across the body's centre line. Over unsaved edits it asks first,
  because it rewrites every rotation at once — one press can undo it, but the confirm is there so a
  whole pose is never replaced by a misclick. It is hidden for a skeleton with no mirror pairs —
  the serpent, whose chain of spine joints has no left and right — where it could only do nothing.
- A **preset** picker offers the shipped pose library for this skeleton, when one exists. Because
  fitting puts a template's joints in the same relative place on every mesh, a pose authored against
  one humanoid applies to every other humanoid.

`Ctrl+Z` undoes and `Ctrl+Y` — or `Ctrl+Shift+Z` — redoes. The unit is the **gesture** — one whole
gizmo drag is one step, and so is a preset, a mirror or a reset.

Poses are **forward-kinematic only**. A pose is exactly a map of joint names to local rotations —
there is no inverse kinematics, no translation, and no scaling. That is the whole contract, and it
is what lets a pose be a small file that applies to any mesh sharing the skeleton. A pose with a
moving root, and an animation, are what [Clay's Pose mode and timeline](27-clay.md#pose-mode) are for.

**Save pose...** asks for a name and stores it. Saving while a saved pose is loaded **replaces**
that pose rather than adding a near-duplicate, so refining an "idle" leaves you with one idle rather
than two that differ by a shoulder.

Saved poses are listed below, each with three actions:

- **Apply** loads it into the viewport.
- **Save GLB...** writes a posed GLB — the mesh frozen in that pose — wherever you choose. It is
  baked by Blender the first time you ask and cached afterwards, so the first request takes about a
  second and later ones are immediate. It runs off the job queue deliberately: a one-second job
  should not have to wait behind a two-minute reconstruction.
- **Delete** removes the pose, and its cached GLB with it.

A pose lives only in the viewer until it is saved. Every route out of pose mode — leaving edit mode,
switching to another asset, starting a comparison, closing the window — asks before discarding
unsaved changes. The banner at the top of the panel says "unsaved changes" whenever there are any.

**Adjust joints** is the second mode over the same markers, and it exists because fitting is
approximate. It shows the rest skeleton, lets you drag joints to where they should actually be,
counts how many you have moved, and **Apply joint positions** re-rigs the mesh with the corrected
skeleton. That is a real re-rig — minutes of CPU, queued like the first one — not an edit of the
existing rig. **Revert** undoes your unapplied moves; **Back to posing** returns. Entering joints
mode asks first if you have an unsaved pose, since a leftover rotation would put the markers where
the posed bones are rather than where the rest skeleton is.

Changing the skeleton's actual *shape* — adding or removing bones — is Clay's job: open the
rigged mesh there (**Edit in Clay** opens its rig, with the bones, the weights and any animations),
and see [Bones and rigs](27-clay.md#bones-and-rigs).

### When rigging is unavailable

Rigging, posing, FBX export and Clay's **Weight with Blender...** all need Blender, which the app
installs as an optional extra.

Blender's Python module ships **CPython 3.13 wheels only**. On any other Python version the extra
installs nothing at all. When that happens:

- `realmspinner doctor` reports rigging as unavailable.
- Some controls stay on screen, greyed, with the reason in their tooltip, so you can see the app
  can rig and what it is missing: the **Rig** and **Pose** segments of the stage rail ("Rigging needs
  Blender, which is not installed" and its Posing twin) and the **Rig when the mesh lands** checkbox
  on the Mesh stage, which gives the rail's own sentence — see
  [Mesh parameters](#mesh-parameters).
- Others are replaced by a sentence rather than greyed: the Rig stage's column says "Rigging needs
  Blender, which is not installed", and the **Pose** panel says "Posing needs Blender, which is not
  installed" instead of telling you to press a button that is not on screen. The Rig rows in a
  Library card's right-click menu are simply absent.
- The FBX export button explains itself with "needs Blender".

Everything else in the app works unchanged. Nothing about generating references or meshes depends
on Blender.

## Exports

The **Rig**, **Pose** and **Export** stages each carry the same **Take it somewhere** section Reference and Mesh do.
In Create it sits behind an **Open in...** header at the top of the inspector that starts closed and stays however
you last left it, so open the header to see it. Clay and Mason stay reachable for a rigged mesh however far through
the pipeline you have taken it, rather than only from the Mesh stage it started on. See
[The library and jobs](33-library-and-jobs.md) for what the list offers and how a destination one step away (a mesh
with no rig yet, and so on) is shown rather than hidden.

Standing on the Export stage, above the grid of buttons, is **Ready for an engine?** — a checklist
built from measurements the app already took, so it costs nothing new to look at and nothing here is
re-measured. It reads the mesh report (triangles against the budget, watertightness, UVs, base
colour and metallic/roughness maps, the pivot), a note left behind if a rebuild's normalize step
silently failed, and whether the rig on this asset was fitted before the mesh was last changed (retargeted, remeshed,
resized or restored).
Each line says **OK** or **Attention** in words, not only in colour, with the one fact behind it.
Nothing here refuses an export — like the mesh report itself, this is advice, not a gate — and it
draws nothing at all for a plain reference or a mesh nothing has measured yet, rather than showing an
empty or falsely all-green checklist.

Some lines carry a button and most do not, because a line only gets one where a press already exists
elsewhere in the app. An over-budget triangle count and a normalize failure send you to the **Mesh** stage, where the
remesh and retarget panels live, and a pivot off the floor, a size that landed far from what you
asked for, or no target size at all send you to its **Finished asset** panel. A rig left behind by a
mesh change sends you to the **Rig** stage, to fit it again. The rest — a leaky seal, missing UVs or
texture maps — are named with no button, because nothing in the app fixes them in one press. Size
is the instructive one: the checklist will tell you what a sword is usually sized at but will not
write that number onto the mesh by itself. Resizing changes the geometry and the recorded size in the
same step, from the Finished asset panel, so the record and the file cannot disagree.

The inspector's **Export** tab lists everything you can take away, as a two-column grid of buttons:

| Button | File | Notes |
| --- | --- | --- |
| GLB | `model.glb` | The finished asset: optimised, grounded, textured. |
| Source GLB | `source.glb` | The reconstruction as the engine returned it — already simplified to about 300k faces by the engine unless `REALMSPINNER_TRELLIS_DECIM=0` — before optimisation and grounding. |
| STL | `model.stl` | Geometry only. |
| OBJ (zip) | `model_obj.zip` | OBJ plus its material and texture files. |
| FBX | `model.fbx` | Needs Blender; the button says so when it is missing. |
| LOD chain (zip) | `model_lods.zip` | The mesh at 100%, 50%, 25% and 10% of its triangles as separate GLBs (`model_LOD0.glb` to `model_LOD3.glb`) and a `lods.json` listing each level's triangle count. Needs gltfpack; the button says so when it is missing. |
| Collision | `collision.glb` | A simplified collision shape. |
| Textures | `textures.zip` | The texture images on their own. |
| Rigged GLB | `rig.glb` | Present once the mesh has been rigged. |
| Animated GLB | `animated.glb` | The rig with every authored clip baked on as a named glTF animation — ten per skeleton (idle, walk, run, attack, jump, plus five newer, provisional ones). Needs Blender and a rig on one of the four skeletons that ship with clips. Rebuilt automatically, the next time it's requested, after a clip is edited. |
| Reference image | `input.png` | The picture the mesh was reconstructed from. |

Only `model.glb` and `source.glb` come out of the job itself. Everything else is produced the first
time you ask for it, then cached — a pure function of `model.glb`, except **Animated GLB**, which is
made from `rig.glb`: it is the only export that carries motion, and the only way the clips leave
Realmspinner as something an engine can play rather than as a 2D character sheet. Godot, Unity, Unreal and
three.js all read named glTF animations directly. Retargeting the mesh lists it as stale beside the
rig, for the same reason the rig is listed: both describe geometry that no longer exists — which is why the first STL
of a large mesh takes a moment and the second is instant. Rebuilding the mesh at a new triangle
budget deletes all of them, since they describe the old geometry.

The **LOD chain** is built from `model.glb` by the same gltfpack pass a triangle budget uses, once per
level, each cut from the full mesh rather than from the level above it, and each level goes through
the same check that a budget does: a pass that lost the UVs, a material or a PBR map is refused rather
than shipped. A level that fails that check, or that would not come out smaller than the one above
(a mesh that already has fewer triangles than a level asks for), is left out together with every
coarser level, and `lods.json` says which and why. A mesh that cannot make two levels has no chain,
and the button reports the reason instead of producing a zip of one file. Every level keeps the
pivot LOD0 has, so an engine swapping levels in place does not make the model jump. A re-texture,
a remesh, a retarget or a resize deletes it with the other exports.

A button you cannot press keeps its place and explains itself in a tooltip: "needs Blender" for FBX
without Blender installed, "not available for this asset" for a mesh export on a plain reference
job. A missing button would be a mystery; a disabled one with a reason is information.

This table is the *mesh* half. A finished reference has its own Export tab offering the cutouts,
the pixel-art reductions and the manifest — see
[2D exports](22-generating-references.md#2d-exports).

For bulk export of several assets at once, and for the storage those files occupy, see
[The library and jobs](33-library-and-jobs.md#storage-and-pruning).
