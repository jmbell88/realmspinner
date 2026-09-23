# Generating meshes

The mesh stage is where the reconstruction engine runs. It costs roughly two minutes of GPU per
attempt, so everything in this chapter is arranged around deciding what to spend that on. All of it
lives at Create's **Mesh** stage, whose column holds no prompt controls whatsoever.

## Starting from a reference

The normal path is promotion: take a finished 2D asset and run the mesh stage from its image.

Select a finished reference in the library — its card offers **Make 3D**, and selecting a
promotable reference also makes it the Mesh stage's source automatically. The **Source** section at
the top of that column names what the job will start from, and **Make 3D** at the bottom submits it.

Selecting an already-finished mesh instead — to look at it again, or to send it through Make 3D a
second time — has the Source section describe that mesh and the reference it came from, rather than
asking you to choose one: **Make 3D** rebuilds from that same reference, named on the button.

The new job is an ordinary image job whose input image is the reference's, recorded as a child of
the reference so the library can show them as one lineage rather than two unrelated rows.

Everything the Mesh stage holds is an **override**. Omitting one means "keep what the reference
recorded", which is not the same as sending the reference's value back — so the selects offer
that unset choice as their first entry and that is where they start. When the chosen reference has a
recorded value for the field, the entry names it — "From reference: High detail" rather than the bare
"keep the reference's" — so leaving a control alone is an informed choice, not a guess. A reference
with nothing recorded for that field (an older job, say) still gets the generic wording.

Two settings are exceptions and are always sent explicitly: the rig checkbox and the
normalise-the-reference checkbox. Both are the Mesh stage's own decisions, and an omission would let
the promotion inherit whatever the reference happened to record.

Derived values never carry across. Anything the worker recorded about the *source* run's artifacts
— the composed prompt, the mesh report, the applied transform — is stripped, so a new job never
wears a quality verdict about a mesh that does not exist yet. See
[Rerun and promotion](36-library-and-jobs.md#rerun-and-promotion).

## Checking the cutout

**Make 3D** does not submit straight away. It opens **Check the cutout**: the subject cut out of the
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
- **Fix matte** opens the reference in Inker with the cutout already folded into its alpha, as one
  undoable step. The eraser and the brush then edit the matte directly; see
  [Inker](28-inker.md#fixing-a-matte).
- **Cancel** leaves everything as it was.

A matte you edited and saved travels to the engine as the image's own alpha, and the job records
that it was approved — the engine is told to keep the alpha rather than cut its own. A soft edge you
painted stays soft; nothing along this path flattens it to a hard cut.

Settings has an opt-in **Don't ask for clean cutouts** switch, off by default, that skips this panel
for the case it exists to catch the least: a cutout with no report warnings or refusals, made by the
BiRefNet model itself rather than the corner-fill fallback. Anything the report flags, or anything the
fallback made, still opens the panel whether the switch is on or not — and the skipped submission is
the identical **Accept**, not a different route to the same job.

## Candidates

The reconstruction engine is deterministic in its seed, and its failure mode is a lottery: the same
reference comes back clean at one seed and with a hole through the shoulder at another. **Candidates**,
directly above **Make 3D**, is how many attempts one press buys — 1, 2 or 3. The cost line under it
changes with the choice, because this is the one control in the pane that multiplies what the button
spends.

Each candidate is an ordinary mesh job: same validation, same VRAM admission, same worker. The first
keeps the mesh seed you pinned, so a pinned seed still reproduces; the rest draw fresh ones.

While a group is undecided its members are **hidden from the library** — three near-identical cards
are not a workshop — and the **Candidates** picker at the top of the inspector is where they live
instead. Selecting one shows it in the viewport exactly as selecting any other asset does. Once every
attempt has finished, **Keep this one** settles the group: the one you kept and the ones you did not
all become ordinary assets, and only then are you *asked* whether to delete the ones you did not keep.
Nothing is ever deleted on your behalf, and declining leaves you with ordinary assets rather than
hidden ones. When every attempt in a group fails there is nothing to keep, so the picker offers
**Discard all** in the same place instead. It settles the group exactly as keeping one does — every
attempt becomes an ordinary asset — and only then asks whether to delete them.

Verdicts work on a candidate like any other mesh, so judging the group feeds the same findings pool.
See [Review](37-review.md). The picker itself shows what has been graded so far: a candidate you have
already graded carries its grade — `+4`, `-2` — beside its status, read once for the whole group rather
than asked about candidate by candidate. While any finished attempt in the group is still ungraded, a
line under the picker says so: *"Grade each attempt before you keep one - they feed What works."* That
line is the whole of what grading does here — it never reorders the candidates, never marks one as the
apparent winner, and never stops you from pressing **Keep this one** on an ungraded attempt. It is a
reminder, not a gate: what you decide by pressing Keep is yours to decide, and the sentence only asks
that the mesh you did not choose still teaches the corpus something before it leaves the picker.

The count applies to **Make 3D** only. An upload queues one mesh job, as it always has.

## Starting from an upload

You can skip the reference stage entirely. Press **Open an image...** in the **Source** section, or
drop an image file onto the window, and the app queues a mesh job directly from it.

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

The **Mesh** section holds the reconstruction settings.

**Mesh resolution** is the geometry resolution handed to the engine, supplied by a platform preset:

| Platform | Resolution |
| --- | --- |
| 2D | 512 |
| 3D | 1024 |

The question the select is asking is what the asset is *for*: a 2D asset is going to be seen flat
and small, a 3D one in a scene. Higher resolutions cost more VRAM and more time, and on a card that cannot
hold both models at once they may need the exclusive VRAM mode (`REALMSPINNER_VRAM_EXCLUSIVE=1`), which
stops the reconstruction engine while the image model runs.

**Budget** picks how the finished mesh gets to a game-sized triangle count, and it offers up to four
families of entry, each drawn only when this machine can actually run it:

- **Game-ready — 2k, 5k, 10k or 20k** remeshes the surface itself inside the model job: a voxel
  pass, a fresh unwrap, and the old colour, roughness and normals baked onto the new geometry. This
  is what turns the engine's ~300k-face output into something an engine actually budgets for, and
  **5k is the default** — Crash-Bandicoot-range fidelity on a whole-character reconstruction, per
  `dev/measurements/2026-09-23-default-mesh-budget.md`. It needs Blender (the `rig` extra); without
  it, this family does not appear.
- **Simplify: Draft/Standard/Detailed** are the gltfpack tiers (20k/50k/100k) — a plain triangle
  reduction of the reconstruction's own surface, no rebake. They need `gltfpack`
  (a one-time manual drop into `vendor/gltfpack/`, or a downloaded engine copy — see
  [Installation](40-installation.md#gltfpack)); without it, this family does not appear either.
  **Without Blender, Standard is the fallback** — the mesh still gets a second pass, just not the
  in-Blender remesh.
- **Raw** ships the reconstruction as the engine wrote it, ~300k faces. Always offered.
- **Custom** takes a triangle count directly; it appears once `gltfpack` is present, alongside the
  Simplify tiers.

With neither Blender nor `gltfpack` on this machine, the combo collapses to Raw alone and
`realmspinner doctor` says why.

A saved Game-ready budget that is not one of the four rungs, say 7,500 from before the ladder
last changed, is kept as it is. It shows as its own **Game-ready (7,500)** entry until you pick
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

**Background** chooses how the engine mattes the input image: `auto`, `birefnet` or `threshold`.
`auto` is the default and is right almost always; the other two exist for images `auto` gets wrong.
A cutout you approved in the panel overrides this and pins `auto`, which is the mode that keeps an
existing alpha — any other setting would re-cut the matte you just approved and make the approval a
lie.

**Mesh seed** is the reconstruction's own seed, separate from the image seed, with its own **Reroll**
button and its own **Lock seed** switch. Leave it at zero to let the job pick one. Unlocked, every
accepted **Make 3D** draws a fresh seed for the next one — the engine is deterministic in its seed,
so pressing the button twice on the same reference with the seed left alone would give you the
identical mesh twice. Lock it when you want exactly that.

**Normalise the reference** recentres the subject and scales it to fill the frame before the engine
sees it. It is off by default: the engine does its own cropping, and whether doing it twice helps or
hurts has not been measured. Treat it as an experiment rather than an improvement.

The **Rig** section, present only when Blender is installed, holds **Rig when the mesh lands** and a
skeleton picker. See [Rigging and posing](25-rigging-and-posing.md).

## Engine (advanced)

Below Rig is a collapsed **Engine (advanced)** header, closed by default. It holds the seven launch
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
variable that sets the same thing app-wide, and [Review](37-review.md#what-works) for how a findings
hint next to one of these controls is read.

## Triangle budget

The reconstruction is kept, permanently, as `source.glb`, and nothing ever overwrites it. The
`model.glb` you actually use is derived from it by optimising and then grounding. That is what makes
changing your mind about triangle count cheap: rebuilding at a different budget is a couple of
seconds of mesh processing rather than another two minutes of reconstruction.

The control is in the inspector at the **Rig** stage, under the collapsed **Triangle budget**
header. It appears only on jobs that have a `source.glb` — older jobs and rig jobs do not.

Five tiers exist in the code: Raw (as reconstructed — the engine has already simplified it to about
300k faces at resolution 1024, 150k at 512, unless `REALMSPINNER_TRELLIS_DECIM=0` is set), Draft (20k),
Standard (50k), Detailed (100k) and
Custom. `gltfpack` — the binary every decimating tier runs through — is a one-time manual drop into
`vendor/gltfpack/` like the reconstruction engine, or a downloaded engine copy, not something the
checkout brings with it; see [Installation](40-installation.md#gltfpack). When it is there this panel
offers the whole list, and Custom gains a triangle-count field with its own valid range. When it is
not, `realmspinner doctor` says so and every tier ships the engine's own output instead of failing.
This is the gltfpack half of the same ladder **Budget**, described under
[Mesh parameters](#mesh-parameters), offers on the generate form under "Simplify:" — Standard is
this panel's default, and it is also what a new job falls back to when Blender is missing and no
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

**This is now the default, not an extra step.** Create's **Budget** combo (see
[Mesh parameters](#mesh-parameters)) offers Game-ready 2k/5k/10k/20k, and 5k runs automatically on
every new mesh whenever Blender (the `rig` extra) is installed — measured on a full reconstruction
to keep silhouette, face markings and other small detail intact
(`dev/measurements/2026-09-23-default-mesh-budget.md`). It runs inside the model job itself, between
the triangle-budget step and grounding, so a rig queued for the same mesh sees the final, remeshed
geometry rather than the ~300k-triangle reconstruction. The panel below is for **reworking an
existing mesh** — trying a different budget, or remeshing a mesh that predates this default.

The control is in the inspector at the **Rig** stage, under the collapsed **Game-ready remesh**
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
that measures manifold is unresolved — see `dev/TODO.md`. In practice, expect the decimate fallback
to be what runs.

The last remesh's result is printed under the button: triangles, the method (`quadriflow` or
`decimate`), and the bake size. A decimated mesh is not reported as a failure of a quad path it was
never promised — the line reads "N triangles (remeshed and re-baked)" either way, and it also records
whether anything the mesh had before (UVs, both PBR maps, material assignment) was lost, in the same
terms a triangle tier's own `tiercheck.compare` measures.

## Surface texture

A re-texture gives a finished mesh a new skin without touching its geometry. The control is in the
inspector at the **Rig** stage, under the collapsed **Surface texture** header, directly below
[Triangle budget](#triangle-budget) — both are operations on an asset that already exists.

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
keeps the mesh's current atlas resolution; a fixed size is offered but changes nothing a view covers
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

## Earlier meshes

A retarget, a remesh and a re-texture all write over the same `model.glb`, so each one used to
throw away whatever the last one built — remesh, then re-texture, and the quad mesh was gone the
moment the new skin landed; retarget after either, and both were gone. That is no longer true. Each
one keeps a copy of the mesh it is about to replace before it publishes, and the collapsed
**Earlier meshes** header — in the inspector at the **Rig** stage, below Surface texture — is where
those copies live.

Each row names what replaced that version (triangle budget, game-ready remesh, surface texture, or
an earlier restore), when, and a one-line detail — a quad count, a view count, a file size. Press
**Restore** on a row to put that mesh back. The panel names what goes stale first: restoring across
a triangle budget or a remesh changes geometry, so the rig, its poses and its sheets describe the
old mesh again, the same warning a retarget gives; restoring only across a re-texture changes
nothing but the surface. A restore is itself kept, so restoring is not a one-way trip either — you
can always go back to what you had before pressing it.

Up to four earlier meshes are kept per asset; keeping a fifth drops the oldest, file and row both.
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

## Exports

The **Rig**, **Pose** and **Export** stages each open with the same **Take it somewhere** section
Reference and Mesh do — Clay, Poser and Mason stay reachable for a rigged mesh however far through
the pipeline you have taken it, rather than only from the Mesh stage it started on, and Poser's own
door renders a character sheet directly. See [The library and jobs](36-library-and-jobs.md) for what
the list offers and how a destination one
step away (a mesh with no rig yet, and so on) is shown rather than hidden.

Standing on the Export stage, above the grid of buttons, is **Ready for an engine?** — a checklist
built from measurements the app already took, so it costs nothing new to look at and nothing here is
re-measured. It reads the mesh report (triangles against the budget, watertightness, UVs, base
colour and metallic/roughness maps, the pivot), a note left behind if a rebuild's normalize step
silently failed, and whether the rig on this asset was fitted before the mesh was last retargeted.
Each line says **OK** or **Attention** in words, not only in colour, with the one fact behind it.
Nothing here refuses an export — like the mesh report itself, this is advice, not a gate — and it
draws nothing at all for a plain reference or a mesh nothing has measured yet, rather than showing an
empty or falsely all-green checklist.

Some lines carry a button and most do not, because a line only gets one where a press already exists
elsewhere in the app. An over-budget triangle count, a normalize failure and a rig left behind by a
retarget all send you to the **Rig** stage, where the remesh and retarget panels live. The rest — a
leaky seal, missing UVs or texture maps, a pivot off the floor, a size that landed far from what you
asked for, or no target size at all — are named with no button, because nothing in the app fixes them
in one press. Size is the instructive one: the checklist will tell you what a sword is usually sized
at, but it will not write that number onto a finished mesh. The scale was baked into the geometry
when the mesh was made, and the number you asked for is part of what the app recorded about the run,
so changing it afterwards would only make the record disagree with the file. Ask for the size on the
Mesh stage and rebuild.

The inspector's **Export** tab lists everything you can take away, as a two-column grid of buttons:

| Button | File | Notes |
| --- | --- | --- |
| GLB | `model.glb` | The finished asset: optimised, grounded, textured. |
| Source GLB | `source.glb` | The reconstruction as the engine returned it — already simplified to about 300k faces by the engine unless `REALMSPINNER_TRELLIS_DECIM=0` — before optimisation and grounding. |
| STL | `model.stl` | Geometry only. |
| OBJ (zip) | `model_obj.zip` | OBJ plus its material and texture files. |
| FBX | `model.fbx` | Needs Blender; the button says so when it is missing. |
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

A button you cannot press keeps its place and explains itself in a tooltip: "needs Blender" for FBX
without Blender installed, "not available for this asset" for a mesh export on a plain reference
job. A missing button would be a mystery; a disabled one with a reason is information.

This table is the *mesh* half. A finished reference has its own Export tab offering the cutouts,
the pixel-art reductions and the manifest — see
[2D exports](22-generating-references.md#2d-exports).

For bulk export of several assets at once, and for the storage those files occupy, see
[The library and jobs](36-library-and-jobs.md#storage-and-pruning).
