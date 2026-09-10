# Extending Warlock Studio

Most of the things you might want to add — another image model, another style, another skeleton,
another chapter of this manual — are data rather than code, and the places they are declared are
deliberately the only places they are declared. This chapter is a tour of those extension points and
the rules that keep them from leaking into the rest of the app.

## Adding an image model

An image model is a registry entry, not a directory on disk. `models.py` owns `BASE_MODELS`, and a
`BaseModel` carries both the checkpoint's directory name and the settings it has to be run at:
image size, step count, guidance scale, weight variant, scheduler, and any always-on
step-distillation LoRA fused on at load.

Those settings live with the checkpoint because they are properties of the checkpoint. A four-step
distilled model run at twenty-five steps with classifier-free guidance produces mush, and Hyper-SD
degrades quietly unless its scheduler uses trailing timestep spacing. Neither is a preference, and
neither is something the person writing a prompt should have to know.

Adding one means adding an entry. The fields worth thinking about:

- `dir_name` — resolved under `WARLOCK_T2I_ROOT`, so the model is found by name rather than by path.
- `image_size`, `steps`, `guidance_scale` — the sampler settings the checkpoint was distilled or
  trained for.
- `scheduler` — a key into the scheduler table in `pipelines/text2image.py` (`ddim_trailing` for
  Hyper-SD, `lcm` for a consistency adapter), or left unset to keep whatever the checkpoint's own
  config specifies. A name that table does not know raises — with the weights already in VRAM,
  which is why every shipped entry's name is covered by a test.
- `base_lora` — a step-distillation adapter loaded under a reserved adapter name, so it can never
  collide with a style LoRA key.
- `controlnet` — stated explicitly rather than inferred from the guidance scale, so a future
  checkpoint does not silently become "controllable" by clearing a threshold nobody qualified it
  against.
- `fetch` — a tuple of `Fetch` records saying what to download, where it lands and roughly how big
  it is. This is the field you declare; `download` is a *derived* property that renders those
  records into the one-time `hf download` text the diagnostics show when the weights are absent, so
  passing `download=` to the frozen dataclass is a `TypeError`, and omitting `fetch` leaves the
  model unfetchable from the app.

What you should not do is hardcode any of those numbers in `pipelines/text2image.py`. The pipeline
reads them off the spec, which is what lets a job name a model and get the right sampler settings
without the UI knowing anything about either.

Only one base model is resident at a time. Selecting a different one unloads the previous pipeline
before building the next, because the card holds the reconstruction engine plus one SDXL-class
pipeline and not two. See [VRAM modes](40-configuration.md#vram-modes).

A registry entry is also the right answer when `WARLOCK_T2I_DIR` is not — that variable redirects
where the built-in `turbo` entry loads from and changes nothing about how it is run. See
[Using a different image model](40-configuration.md#using-a-different-image-model).

## Adding a style LoRA

A style LoRA is a `STYLE_LORAS` entry plus a `.safetensors` file under the `loras/` subdirectory of
the model root. The entry carries the filename, a label, a default weight and the trigger words the
adapter was trained with, along with its own download command.

Style LoRAs are the opposite of base models in the way that matters most: they are adapters applied
to whatever pipeline is already resident, switched per job without a reload, so changing style
between jobs is free where changing base model is not.

The trigger words are the detail most likely to be put in the wrong place. They are prepended to the
composed prompt alongside the prompt template, and they are deliberately absent from the guidance
module's prompt fields. A trigger is model-facing scaffolding — it exists because the adapter was
trained to answer to it — never part of the user's brief. Model keys and LoRA keys are validated in
`guidance.py` so that a bad value
produces one kind of error from one place, but they contribute nothing to the composed prompt: its
text is byte-identical with and without any of them.

The entry also declares the architecture family it was fitted to, and that declaration is the whole
of the pairing: an adapter names one architecture's modules, so applying it to another raises at load
time with the checkpoint already in VRAM rather than merely producing a weak result. The picker
offers each model only the styles that fit it, the service refuses a mismatched pair by name, and a
stored job carrying one drops the style and logs it rather than failing. Declared rather than sniffed
from the file's key prefixes, for the reason a base model's family is declared: a load attempt is not
a cheap probe. The default is SDXL, so an SDXL adapter needs no new field.

The `loras/` directory is flat and shared across families, which makes a rename mandatory whenever an
upstream repository ships a generically named file — two entries naming `pytorch_lora_weights.safetensors`
would be one file on disk, silently the wrong weights under one of the keys. The `Fetch` record
carries the rename, and it is applied inside the staging directory before anything is moved into
place, so the generic name never lands in `loras/` at all.

A default weight is a per-entry number and sometimes has to be measured rather than taken from the
model card: the weight is a multiplier on whatever scale the adapter's own metadata declares, and an
adapter trained with `use_rslora` declares a much larger one than an ordinary adapter does.

A missing LoRA file is skipped at load time rather than failing the job, and the diagnostics name it.
See [Models and style LoRAs](22-generating-references.md#models-and-style-loras) and
[Optional image models and style LoRAs](39-installation.md#optional-image-models-and-style-loras).

## Adding a palette

There is nothing to add. A palette is a file in the palette directory (`~/.warlock/palettes/`, or wherever
`WARLOCK_PALETTE_DIR` points), in Lospec's `.hex`, GIMP's `.gpl`, Paint Shop Pro's `.pal` or
Paint.NET's `.txt` format, and the export's palette control lists whatever is there — as does the
Inker's own palette folder browser. No registry entry, no code, no restart. That is deliberate: a
palette is art direction, and the registry pattern the models use exists for things that have to be
downloaded, checked for and reported on.

The one rule worth knowing is that freshness is keyed on a palette's *contents* and not its
filename, so editing one in place re-derives every export that used it, which is what makes working
on a palette feel like working on a file.

## Adding a skeleton

A skeleton is a JSON file in `src/warlock/templates/`, and adding one is the entire procedure — no
bone list in `blender_worker.py`, no branch anywhere that names a template.

Each file declares a key, a label, a root bone, a list of bones — each with a name, a parent and a
head and tail position — and a list of `mirror_pairs`. The positions are normalised landmarks in a
unit bounding box, expressed in
Blender's axes: `+X` is the subject's left, `-Y` is forward, `+Z` is up. The `x` and `y` components
span `-0.5` to `0.5` about the box centre, and `z` spans `0` at the floor to `1` at the top.

`rigging.fit_template` scales those landmarks onto the measured bounding box of the mesh being
rigged. The fit is bbox-proportional and deliberately approximate — a joint lands where the
proportions say it should, not where anatomy says it should. That is why the fitted positions are
written into `rig.json`: a later adjustment pass can correct a joint without re-solving the rig, and
the record of where each joint actually ended up is the input it needs.

For the `humanoid` template there is a second source of landmarks, and it does not replace the
fitter — it replaces the *template*. When a pose model is installed, `pipelines/pose2d.refit` reads
the subject's joints off the reference image and returns them in exactly the template's own format:
the same names, the same parentage, still normalised into a unit box. `fit_template` then scales
those onto the mesh exactly as it scales the shipped ones, so bounding-box scaling stays owned by
one function and nothing downstream learns a second way a joint can be placed. Depth is always the
template's: one view fixes `x` and `z` and says nothing about `y`.

Any doubt refuses the whole measurement rather than part of it — a landmark below the confidence
floor, one the detector never produced, a figure whose knees come out above its hips, a landmark
outside the subject's silhouette. A skeleton half-measured and half-assumed is not partly better; it
is internally inconsistent in a way nothing downstream can detect. `rig.json` records which source
was used in its `fit` field.

Extending this to another template means a detector for that anatomy and a mapping onto its
landmarks: `pose2d.POSE_FIT_TEMPLATES` is the list, and it names `humanoid` alone because COCO-17 is
a human keypoint set. A quadruped needs an AP-10K model and its own mapping. Adding a template
without one is entirely normal — it simply gets the bbox fit, which is what every template got
before this existed.

Mirroring is not inferred from the geometry. `mirror_pairs` is an explicit array of two-element
`[left, right]` name pairs, and it is the only thing that makes the pose editor's Mirror control do
anything: `rigging.mirror_pose` copies each posed bone onto its named partner reflected, and a bone
that appears in no pair is left exactly as it is, on the assumption that it sits on the mirror plane.
The list is carried into `rig.json`, which is where the viewer reads it from, and the Mirror button
is hidden entirely when it is empty — so a template that omits the field loads and rigs perfectly
well and simply cannot be mirrored. The field is optional in the parser and defaults to empty, which
means forgetting it costs you a feature rather than an error. Omit it deliberately, as `serpent.json`
does, or list every symmetric limb, as `humanoid.json` does.

Two conventions are worth honouring for consistency with the templates already there. Forward is
`-Y`, which is what makes column zero of a sprite sheet the front view by default. (A mesh that is
not built to that convention is not stuck with it: its own front is set by orbiting it in
[Poser](26-poser.md#choosing-the-front) or the 3D viewport, and every sheet is measured from there
instead. Honour the convention anyway — it is one less thing for whoever uses your template to have
to notice.) And limbs you intend to pair should be placed mirror-symmetrically about `X`, because the
reflection `mirror_pose` applies assumes that plane.

See [Templates](25-rigging-and-posing.md#templates).

## The derived-params rule

A job's parameters mix two kinds of thing: what you asked for, and what the app worked out. The
second kind is listed in `DERIVED_PARAMS` in `service/validation.py`, and that list is the single
place a rerun or a promotion consults when deciding what to strip.

The rule is short. Anything the worker records about a *finished* job's artifacts — the composed
prompt, the applied transform, the scale factor, the mesh audit, the mesh report, the optimiser
result, the weighting method, the bone count, the sheet id and its cells, the reference report, the
control hint and the recipe — belongs on that list. If it is not there, a rerolled job inherits it,
and you get a fresh mesh wearing a quality verdict about a mesh that no longer exists.

There is a deliberate counterpart. The conditioning selection — the IP-Adapter, the ControlNet and
their strengths — is an *input*, so it survives a reroll. It does not survive a promotion or a
remesh, because those are image jobs that never run the image model at all, and a row claiming an
adapter that cannot have run is a lie about provenance.

Inputs are bounded at the door rather than deep in the pipeline: an upload is size-checked before it
is decoded and pixel-checked from its header before pixels are allocated, prompts are length-capped,
and every service entry point that accepts a seed range-checks it. See
[Rerun and promotion](36-library-and-jobs.md#rerun-and-promotion).

## Pure-module boundaries

Several parts of the app are pure by rule, and the rule is always the same: nothing below the line
imports imgui, moderngl, pygame or the service layer. Three places do this, for three versions of
the same reason.

**The Inker engine.** `studio/inker/` holds blend arithmetic, layers with stable uids, typed undo
edits, selection masks, brush stamps, gradients and OpenRaster I/O, and none of it knows a window
exists. `studio/inker_mode.py` is the only layer that knows about jobs and task threads. That is
what makes every rule about pixels assertable headlessly — and there are a lot of such rules, since
undo is addressed by layer uid rather than index precisely so that an undo issued after a reorder
still lands on the layer the edit was made to.

**Sheet planning.** As described in [Sheet planning](44-pipelines.md#sheet-planning), the grid is
decided in a module with no Blender and no GPU, so the layout can be tested exhaustively and the
preview cannot drift from the render.

**This manual.** `studio/manual/loader.py` finds chapters and reads them; `studio/manual/parser.py`
turns markdown into typed blocks. Neither imports imgui, so every rule about what a chapter may
contain is a headless test. The renderer that draws those blocks in the app is a separate thing
entirely and holds no opinions about syntax.

The pattern generalises. If a rule is worth enforcing, put the thing it governs somewhere a test can
reach without a display.

## Driving Warlock from an AI agent

Warlock speaks the Model Context Protocol, so an agent that already runs on your machine — Claude
Code, Codex, anything with an MCP client — can build in Clay for you. It is off until you switch it
on, in Settings under Advanced. Point the agent at it with a command like
`claude mcp add warlock -- uv run warlock mcp`, and it will find the running app.

The arrow only ever points inwards. Warlock ships no language model, runs no inference and reaches
no endpoint; an agent that is already running connects to it. The transport is a local named pipe
rather than a port, so there is nothing to open in a firewall and nothing off your machine can
reach it. The pipe's key lives in `mcp.token` in your Warlock home and is written when you switch
the setting on, so a program that cannot read your files cannot connect either.

**An agent gets a Clay tab of its own, and can reach no other.** It opens one when it connects, and
every tool it has addresses that tab by name. A document you already have open is not merely
unlikely to be touched; there is no request the agent can make that names it. What the agent does
goes onto that document's ordinary undo stack, one step per action, so taking over means switching
to its tab and pressing Ctrl+Z as often as you want to. It also arrives already knowing Warlock's
units and conventions — metres, which way is up, that a generator stands on the ground rather than
straddling it — rather than working them out by trial, which is why its first attempt at something
now usually stands on the ground instead of floating above it or growing up out of the floor.

The tools are the ones you would reach for yourself, but most of them now do in one call what used
to take several. Placing a primitive or a figure sets its size, its position, its rotation, its
scale, its name and its palette colour all at once, validated before anything appears and landing as
a single undo step — it used to take four round trips to place one sized, positioned, named, coloured
object. **Batch** folds up to thirty-two calls, a whole block-out, into that same one step, so backing
the attempt out is one Ctrl+Z rather than one per primitive. Undo, delete and rename all work by
name, the way you would type them yourself, rather than by whatever the agent last happened to have
selected. The one that makes the rest work is still **render** — it can now look from several angles
in a single call, with an optional ground grid switched on as the only scale cue in what would
otherwise be a flat white square, and a focus that frames the object under discussion while leaving
the rest of the scene drawn around it. An agent that can only read coordinates builds things that are
plausible in numbers and wrong on screen; one that can look at what it made corrects itself the way
you would — and that is now also the argument for reference images: an agent shown the picture you
want matched can put its own render beside it, or blend the two together, and see the difference
instead of only being told about it.

A picture reaches the agent one of two ways. You can point it at a Library row — right-click the
card and choose **Copy job id**, and hand the agent that id — or hand it image data directly,
however your agent client lets you paste or attach one. There is deliberately no third way, where the
agent names a file path on your machine and Warlock opens it: a tool that will open any path it is
given is a tool that reads whatever else is on your disk, and this one does not. What you see of a
reference landing, for now, is a toast the moment the agent adds one — a thumbnail strip in Clay and
a reference plane in the 3D view, so you could see what it is comparing against without having to ask
it, are wanted and not yet built.

Exporting does what pressing the button does: it saves the model, writes a GLB, and mints a Library
entry, so what an agent makes is an ordinary asset with no history of being unusual. Rigging,
posing, sprite sheets and every mesh export work on it exactly as they work on anything else.

### Adding a tool

`studio/agent_clay.py` is the surface and `studio/agent_host.py` is the plumbing. The important
thing about the first is that **most of it is not written down**: the shapes an agent may place come
from `primitives.GENERATORS`, the figures from `presets.ASSEMBLIES`, and the operations from
`clay_ops.OPS` — the same three tables the add panel and the context menu are drawn from. A
thirteenth generator added to Clay appears in the agent's tool list with no edit here at all, and a
test asserts that in both directions, so the two cannot drift apart.

So adding a *shape* or an *operation* is not an edit to the agent surface. Only a genuinely new
verb — something Clay's own registry has no entry for — is, and it goes in beside the others as a
function that takes the context, the session and the arguments, and returns content.

Three rules bind anything you add. It runs on the frame thread, drained under a time budget, because
that is the only thread that may touch a document or the graphics context — the listener never
touches either, and an operation that takes a long time will drop frames rather than corrupt
anything. It must not raise: a refusal is a result an agent can read, and where Warlock knows
which argument was wrong it says so by name, which is a thing the old HTTP interface had nowhere to
put. And it validates before it mutates — every argument it means to act on, checked and refused by
name before the first line that changes the document, not partway through. A tool's own JSON schema
describing an argument as a number, or an array of numbers, is not enforcement: `mcp/protocol.py`
checks only that `arguments` as a whole is a dict before handing it to the handler, so whatever an
agent actually sent — a two-element array where three were meant, a NaN, an infinity — arrives
exactly as typed. `clay_transform` went a release without this third rule and paid for it: an
unchecked `translation` committed a two-element vector to the document, reported success, and only
broke three calls later when `clay_scene` tried to read it back — by then there was nothing to point
at, and the whole document's introspection was bricked until someone thought to undo blind. Reach
for `_validate_vec3` for a TRS-shaped argument, `_validate_unit` for a 0..1 number and
`_validate_number_or_vec` for the `number | array-of-numbers` shape `clay_set_params` uses — all
three live beside `agent_clay.py`'s other validators, in the same "validate everything before the
first mutation" style `_h_add_primitive` and `_h_add_figure` already followed.

A new tool is also two decisions, both of which the test suite makes you take. Name it in
`BATCH_EXCLUDED` if it belongs there — the only two reasons anything is on that list are that its
result is a picture a client has to see as an image, which is not a shape a batch's own result can
carry, or that it is deliberately one-shot, an action nothing should ever want folded silently into
somebody else's block-out. And `tests/test_agent_clay.py` gates the other list: every handler has to
appear in either the tools that need a tab already open or the tools a session can run without one,
and a handler that answers to neither fails the suite instead of quietly falling through — you cannot
add a tool without deciding which kind it is. The reference tools are the one family that pushes no
undo step at all, because adding, listing, fetching or removing a picture never touches the document
in the first place.

## Writing manual chapters

The chapters are markdown files in `docs/manual/`, named `NN-name.md`. They are readable on GitHub
as ordinary markdown and rendered in the app by the parser above; the loader prefers a packaged copy
if one is installed and otherwise reads the repository's copy directly.

A chapter's first line is its H1, and that H1 is its title everywhere — the index, the in-app
navigation, the help button. There is exactly one per file.

The markdown accepted is a strict subset:

| Construct | Accepted |
| --- | --- |
| Headings | `#` through `####` |
| Inline | `**bold**`, `*italic*`, `code`, `[text](target)` |
| Code | fenced blocks, with an optional language |
| Lists | `-` and `1.`, nested by two spaces |
| Tables | pipe rows with a `---` separator row |
| Images | `![alt](path)` on a line of its own, the path relative to the chapter |

Everything else is rejected, including inline images, raw HTML and blockquotes. Two consequences catch
people out. A bare angle bracket followed by a letter reads as HTML, so a placeholder like a job id
in angle brackets has to sit inside a code span. And a table cell cannot contain a pipe character,
because the row is split on pipes before anything else is parsed.

The strictness is a bargain rather than a preference. A construct outside the subset raises an error
at parse time, and `tests/manual/test_docs.py` parses every chapter — so a chapter that drifts
outside the subset fails the test suite rather than rendering wrong for a reader who has no way to
know it was ever meant to look different. The same test resolves every cross-link and every anchor
in every file, and checks that the index links each chapter exactly once. A link to a heading you
renamed is a failing test, not a dead link discovered a year later.

Anchors follow the GitHub convention, which the parser reproduces: lowercase the heading, drop
punctuation, turn runs of whitespace into single hyphens. Link within a chapter with `#anchor`,
across chapters with `file.md#anchor`.

**Your `##` and `###` headings are navigation, not just typography.** The table of contents lists
chapters, and the chapter being read expands to show its own headings as indented rows; clicking one
scrolls to it, and the row you are inside stays lit as you scroll. A search lists the matching
*sections* of every matching chapter rather than only the chapters. Three things follow for an
author:

- A heading is a destination, so write it as a name for the passage under it rather than as a joke
  or a continuation of the sentence before it.
- A long run of prose with no heading is unreachable from the tree. If a passage is worth arriving
  at, give it a heading.
- A search hit is attributed to the **innermost** heading above it, and text before the first
  heading is attributed to no section at all — so the opening paragraphs of a chapter are found by
  the chapter and not by a section row.

Chapter numbers decide order *and* part (`loader.PARTS`), so a new chapter in the middle is a
renumbering of everything after it. That is the supported move rather than a risky one: the tests
below check the chapter list, the index's own sections, every cross-link and every `HELP_TARGETS`
entry, in both directions.

The one piece of wiring outside the files themselves is `HELP_TARGETS` in
`studio/manual/targets.py`. It maps a pane's key to the chapter and heading that documents it, which
is what lets the help affordance in a panel open the manual at the relevant place instead of at the
top. Adding a chapter that documents a pane means adding its entry there; that map is the only
coupling between the UI's structure and the manual's, and keeping it in one table is what stops
chapter names being scattered through pane code.
