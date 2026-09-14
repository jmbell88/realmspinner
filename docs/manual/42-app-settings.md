# App settings

**Settings**, in the rail's footer, holds the handful of preferences that are the app's rather than a
job's. They are stored in `studio_settings.json` beside everything else the app remembers, and none
of them need a variable set before launch.

The pane is one centred column with seven categories across the top — **Appearance**, **Models**,
**Packs**, **Updates**, **Storage**, **Health** and **Advanced**. The column keeps its width however wide the
window is: everything here is a short labelled row, and a form stretched across a 5K display leaves
the label and the control it belongs to at opposite ends of the desk. Which category you last had
open is not remembered between launches — it is where you were, not something you chose.

A **search field** sits above the category list. Typing filters by a row's label, its tooltip, and a
small table of plain-language synonyms — "bigger text" finds *UI scale*, "disk space" finds
**Models**, "music" finds **Packs** — and shows every match under its own category heading, the way
the Manual's own search shows matching sections under their chapter. Clicking a match opens that
category. Clearing the box brings the ordinary category list back; a search that matches nothing says
"Nothing matches" rather than leaving an empty column with no explanation.

## Appearance

*UI scale* is a multiplier on top of whatever your monitor's own DPI scaling already is. It is a
list of named steps rather than a free slider — 50%, 75%, 100%, 125% and 150% — because a slider
was the wrong control for a setting with five sensible values and a font rebuild behind each one.
On a display that is already heavily scaled the list offers fewer steps and says so, because the
combined scale is capped: the control only offers zooms it can actually apply. Picking a step takes
effect at once, and the font atlas is re-baked between frames rather than during one, since a
rebuild invalidates every font handle a half-drawn frame is holding. Nothing needs a restart.

*Theme* switches the whole palette and takes effect at once. There are three. *Dark* is the
default. *Light* keeps the same *roles* as the dark one rather than inverting its numbers: a panel is
still the surface a form sits on, and the elevation steps still read as raised — which on a light
ground means slightly darker rather than lighter. *Pixel* is a second dark palette in the register
the pixel workspaces belong to: warm neutral greys against a near-black canvas surround, with amber
where the other two spend indigo. It keeps dark's direction — a step away from the floor still reads
lighter — and differs from it in temperature, so the Inker, the Plotter and Packwright sit in
something closer to the tools their work comes from. It is not a copy of any one program's chrome:
those are typically grey on grey, and every colour here has to clear a measured contrast bar. The
one thing **no** theme repaints is the 3D
viewport's background, which stays the dark `#0F1014` under all three: that colour is a property
of the renderer rather than of the palette, and making it follow the theme means threading a colour
into the render-skip key so a theme switch triggers a redraw. It is a known gap, deliberately left
open. *Show frame rate* is the same toggle as `F10`.

*Startup* chooses which screen opens when you launch Warlock. **Home** is the default and unchanged
from every earlier build — a fresh install, or a settings file written before this setting existed,
opens on Home exactly as it always has. **Last workspace** reopens whatever mode you were last in,
whatever that was — a workspace, the Library, even Settings itself. If that mode needs model weights
or a dependency pack this machine has not installed, it falls back to Home through the same refusal
a greyed rail item already gives, rather than a second, differently-worded one.

*System resources* puts a live reading of VRAM, RAM and CPU in the menu bar's status group, in
every mode. It is there because the app already forces the question on you: a generation is refused
at the door when there is not enough VRAM free, and re-checked when it is about to run, and until
now nothing on screen said what the number those refusals are about actually was. The VRAM figure
is read from the driver, so it counts every process on the card and not just this one. It updates
once a second, costs about a twentieth of a millisecond to take, and is the first thing dropped
when the workspace's own menus leave the group no room.
A card the driver does not report on simply leaves VRAM out rather than showing a placeholder.

*Reduce motion* turns off every animation in the app at once — the mode transition, hover, the
rail's sliding selection and its expand, a popover's rise. Nothing disappears and nothing behaves differently: things
arrive in place instead of travelling there, and the few effects that are a *fade* rather than a
move (a toast's opacity, the acknowledgement flash when a file is dropped) keep their timing and
lose their ramp.

There is no **Effects** section any more. Four switches used to sit here — soft shadows, translucent
panels, spring motion and continuous corners — while those rendering features were new. Each of them
already falls back on its own when the graphics side cannot provide it (concentric outlines, solid
fills, an eased stop, a circular arc), which is what the switches were really insuring against, so
they were removed rather than left as four settings that only ever meant "pretend the graphics card
failed". *Reduce motion* above is unaffected: it is an accessibility setting, not a rendering tier,
and it still turns off spring motion along with everything else that moves.

## Models

Every model the app knows about — the reconstruction engine, image models, style LoRAs, the
conditioning adapters, and the matting, pose and measurement models — as a table of four columns: **Model**, **Size**,
**Description** and **Actions**. A tick beside the name means the weights are on disk; a hollow mark
with a checkbox means they are not, and **Install** fetches them. It is the same information the
startup diagnostics report, in a place you can look at without opening the log. Tick several rows
and *Download selected* fetches them together; four of the image models share one set of SDXL 1.0
weights, and picking all four downloads them once.

The **Description** column is one sentence saying what the model is *for* — which is the question a
list of thirty names cannot answer, and the reason picking one used to mean reading `docs/MODELS.md`
beside the app. Hovering the name gives the longer version, along with the repository the weights
come from, what the model costs on the card, and, for a style LoRA, the trigger words it was trained
on — the words the app prepends to your prompt whenever that adapter is selected. A model this build
has no download recipe for stays in the table as an inert **Unavailable** rather than disappearing:
a row that says it cannot be fetched is more use than a row that is not there.

A
running download shows its rate and an estimate of the time left, and carries its own **Cancel**
beside the bar. Cancelling installs nothing: the fetch stages beside the destination and only moves
into place once it has finished, and the staging a cancelled fetch leaves is swept the next time
this pane is opened, which says so when it reclaims anything.

Under each image model's size, once the app has measured your card, sits a fit note: nothing at all
when the model runs comfortably, **tight fit** when it will load but cannot stay resident beside the
reconstruction engine — so every 3D job pays a stop and restart for it — and **won't fit** when the
checkpoint alone is larger than the VRAM budget. Both carry the measured figure on hover. A
**Recommended for this GPU** line under the list names the best base model your card can actually
hold. All of it is skipped on a host with no measurable GPU: an unknown budget is not a shortfall.

**Deleting a model.** A downloaded row carries a **Delete** button, and hovering it says how much
removing would actually free. That figure is usually smaller than the download was, and deliberately: the four SDXL 1.0
recipes share one 7 GB checkpoint, so removing *SDXL 1.0 + Hyper-SD* deletes only its own 0.8 GB
adapter and leaves the weights the other three are still standing on. The checkpoint goes when the
last model using it does. A recipe with no files of its own — *SDXL 1.0 (full CFG)* is the plain
case — has nothing to remove and shows no button at all.

Removal asks first, and refuses in three cases rather than doing something surprising. It will not
run while any job is queued or running. It will not touch a directory `WARLOCK_T2I_DIR` points at:
that is a folder you pointed the app at, not one it downloaded. And it never deletes anything
outside the model root. Removing the *default* base model **is** allowed, and the consequence is
that generation refuses every job that does not name another model until you reinstall one or pick a
different base — the refusal says so and links back to this pane.

Downloading does not make the app itself online. The button starts a separate process that fetches
one repository and exits, into a staging folder beside the destination that is only moved into place
if the fetch succeeded — so a download interrupted halfway leaves nothing behind rather than a model
directory that looks finished. Free disk is checked against the whole selection first, and the whole
selection is refused if it will not fit. Everything is still equally installable by hand — see
[Model weights](40-installation.md#model-weights) and
[Adding an image model](46-extending.md#adding-an-image-model).

**The reconstruction engine is two rows, at the top.** *TRELLIS.2 engine* is the program that turns
a picture into a mesh — about 0.7 GB — and *TRELLIS.2 GGUF weights* is the model it loads, about
16 GB. You need both before the Mesh stage can run, and neither is installed for you: until
2026-09-10 the engine was inside the installer, where it was more than half of everything you
downloaded whether or not you ever made a 3D model. It is a row here now, like everything else on
this list.

The engine is the one row that does not come from Hugging Face — it is a single archive published by
trellis.cpp, so the app checks it against a fingerprint rather than a version number, unpacks it, and
puts it under your Warlock home. It needs an NVIDIA card; there is no version that runs on the
processor. **Create stays greyed on the rail until both rows are present**, and clicking it brings
you here with them already ticked.

### Your style LoRAs

Under the model tables is the one place a style comes from somewhere other than a download. Two
buttons, one list.

**Import a LoRA file...** takes any `.safetensors` adapter — one you trained elsewhere, one from a
model site — and asks for the four things the picker and the loader read: a name, the trigger
words it was trained with, its working weight, and which family it fits (SDXL or FLUX.2 klein; an
adapter never loads onto the other). Tick *Licensed for commercial use* only if you know that it
is; the picker shows the answer beside the style. The file is copied under `~/.warlock/models/
loras/` and appears in every LoRA picker immediately.

**Train from a folder...** trains one here, on your card, from your own art. Point it at a folder
of 3 to 100 images in the style — no captions needed — give the style a name and a trigger
phrase, and it queues as a job like any other. It runs on an undistilled SDXL checkpoint (the
Quality recipe's), takes the whole card for the duration (the reconstruction engine and the image
model are unloaded first and come back on the next job), and about 800 steps is half an hour on a
fast card. When it finishes the style is registered exactly as an imported one, at weight 1.0, with
your trigger words. Your images never leave the machine.

**Train from my library...** builds the same training set without a folder, out of work you have
already judged. It gathers three things from your job history — every favourited job, every
reference you labelled *accept* while judging, and the reference image of every mesh graded usable
or better (a mesh that reconstructed well is evidence its picture was a good blank, even if you
never labelled the picture itself) — drops near-duplicates so the same reference reused across two
jobs only trains once, and fills the same form the folder button does. The scan runs in the
background, so the rest of the app stays responsive while it reads through your history; the
summary line under the form says how many images it found and where they came from, for example "24
images (3 near-duplicates dropped; from 9 favourites, 14 accepted references, 4 usable meshes)".
If your library does not yet have enough judged work, the button's scan refuses the same way the
folder door does, naming how many it found.
**Training on pictures this same build generated is unmeasured.** Nothing here has run the paired,
blind comparison — the LoRA on against the LoRA off, judged without knowing which is which — that
would say whether a style trained this way actually helps or just teaches the model to repeat its
own habits back to itself. Treat it as a convenience for gathering images you already vetted by
hand, not as a validated recipe; [Tuning what you get](12-tuning-what-you-get.md#style-loras-trained-from-the-library)
has the longer note.

**Remove** beside an imported or trained style deletes its file and its entry. Built-in styles are
not listed here and cannot be removed.

## Packs

Models are weights; **packs** are the code that reads them. The installed build carries the app, its
renderer and the reconstruction engine — around a gigabyte — and the three heavy dependency sets
arrive here, chosen:

| Pack | What it turns on |
| --- | --- |
| **Image generation** | Create's reference stage, host-side matting, candidate ranking |
| **Rigging** | Poser and Troupe |
| **Music generation** | Muse, and stem separation |

Each row says what the pack is for, which workspaces it unlocks, and what it costs — in two figures,
because they land in two places: the download goes to a wheel cache under `~/.warlock/packs`, and
the unpacked packages go into the app's own runtime, which on a per-user install is often another
drive. Both are checked for free space before anything starts, and the whole pack is refused if
either will not hold it.

**Install** downloads and installs it, and the two phases are visible on the one bar. While it is
downloading, **Cancel** stops it and costs you nothing: every wheel is verified against the digest
this build pins and only then moved into place, so a stopped install leaves a part-file that the
next attempt resumes from. Once the packages start going in, the Cancel button goes away — that
half writes into the runtime the app is running out of, and interrupting it is the one thing that
could leave the installation half-made. It is a separate process throughout; this one stays offline
and never becomes able to download anything.

When it finishes, the app re-runs every startup check and the workspaces the pack unlocks come to
life without a restart. If something still cannot be imported the toast says so and asks for a
restart, rather than leaving you with a mode that is greyed out for no stated reason.

A pack cannot be removed from here. Uninstalling torch out from under a running application is not
the same act as deleting a folder of weights, and the way to get the disk back is to reinstall the
base app.

**A greyed workspace sends you here first when a pack is what it is waiting for.** Create, Poser,
Troupe and Muse each need two separate things — the pack, which is the code, and the model weights,
which are what the code reads — and the pack has to come first, because weights with nothing to load
them do nothing at all. So on a fresh install, clicking one of those modes opens this page and the
tooltip names the pack by its own size; once it is installed, clicking the mode again opens
**Models** instead, with the rows that mode needs already ticked.

On a source checkout there is nothing to install: no build ever generated a pack manifest, and each
row prints the `uv sync --extra ...` line that does the same job. See
[Installation](40-installation.md#if-you-installed-rather-than-cloned).

## Updates

Warlock does not check for a new version unless you ask it to. **Check for Updates** is the only
always-on trigger, and the switch below it — **Check for updates on startup**, off until you turn it
on — is the only way anything happens without a click. Everything else on this page stays dark on a
machine that never presses either.

The check itself happens in a separate process, exactly as a model download and a pack install do:
this one asks GitHub what the latest release is, reads the small manifest that release publishes,
and dies. Nothing in the running app becomes able to reach the network, and nothing about how your
work is generated changes.

If there is a newer version, the page names it and offers three things: **Release notes**, which
opens the release page in your browser; **Download Update**, with the installer's size on the
button; and, once that has finished, **Run Installer** and **Show in Folder**. The download goes
into `updates` under your Warlock home — beside the [pack cache](#packs), and it survives closing
the app, so an installer you downloaded last night is still there this morning. **Cancel** is safe
at any point during it: nothing exists until the whole file has arrived and been checked.

**Warlock never runs the installer for you, and that is deliberate.** What it does instead is prove
the file: the release publishes the installer's SHA-256, and the download is checked against it
before it is given a name you can double-click. A file that does not match is deleted rather than
offered. The page will not say "Update ready" about a file it has not just re-checked, so an
installer left over from a different release cannot be run by mistake. Running it is your click, in
your file manager, with Windows' own prompt in front of it.

If a release was published without that manifest — every release older than this feature was — the
page says you are up to date and offers nothing, because there is nothing it can verify. The same
answer comes back when a background check fails: an automatic check you did not ask for stays quiet,
while one you pressed the button for tells you what went wrong.

## Storage

Four figures and four buttons. The figures are how many job directories exist and what they occupy,
what is sitting in the [trash](37-library-and-jobs.md#the-trash) waiting to be emptied, how much
disk the downloaded model weights are actually using, and what the evidence archive holds. All four
are measured on a background thread and the last answer is drawn until a new one arrives, so none of
them walks the disk while you are looking at something else. The first two are the same measurements
the library reports, not a second opinion about the same directories; the third is a real measurement
of the model store rather than the sizes the Models list declares, which are approximations kept for
the progress bar and the free-disk check.

The fourth is absent until there is something to report. The archive fills up on a *delete* — the
moment you are least expecting anything to be written — so it says so; but a line reading "0 archived
jobs" would be a permanent fixture explaining a thing that has not happened.

**Check library** looks for assets whose files are gone, folders no asset claims, and reviews whose
asset was deleted; it changes nothing, only reports. **Back up the index** copies the database
holding every prompt, seed, name, tag and review — the part that cannot be regenerated from the
files — into a stamped folder under your library; there is no destination dialog, since the
stamped folder beside the library is where a backup goes.

**Prune...** deletes everything but the newest N assets from disk, after a confirm that carries the
count — N is yours to choose and it starts at twenty every time it is asked. Running jobs are never
touched, and neither is anything you accepted or labelled. Anything you *did* grade, and anything a
benchmark run tagged, has its files copied to the evidence archive on the way out — see
[Library and jobs](37-library-and-jobs.md#storage-and-pruning).

**Clean library...** is the other end of the same scale: every asset, trashed or not, including the
accepted ones and the labelled images the quality judge and the tier checks are measured against.
The verdict rows survive; the pixels behind them do not, and nothing is written to the evidence
archive either — starting a corpus over is not a request that quietly keeps the largest meshes on the
disk. Your pose library, Inker autosaves and settings are kept, and it refuses outright while
anything is queued or running.

Both used to sit at the foot of the library, under the list of assets, which is the one place where
"clean library" reads as an action on the assets you can see rather than on all of them.
[Library and jobs](37-library-and-jobs.md#storage-and-pruning) has the longer account of what
each one deletes and why prune removes from disk rather than trashing.

## Health

Three buttons sit at the top of the page, above the list: they act on the whole page rather than on
any one check, so they come first rather than making you scroll past every check to reach them.
**Detail Log** puts the whole list on the clipboard in the form `FAIL name: detail`, which is what to
paste into a bug report. **Health Checks** re-runs every probe including the slow ones the startup
path defers — worth pressing after installing something a row complained about, because the static
checks are otherwise only recomputed at launch. **Troubleshooting** opens
[Troubleshooting](43-troubleshooting.md), which is where a check that keeps failing after its remedy
is covered.

Below them, the summary line and every check `warlock doctor` runs, as a table: a coloured glyph, the
check's name, and the one line of detail saying what it found. Green is passing, amber is a warning,
red is fatal. The line above the table counts the failures, which is the same number Home's health
row shows — clicking that row opens this page.

This is the only place in the app that names a *non-fatal* failure. A fatal one also raises the
error banner across the top of the window, but a style LoRA whose file has been moved, or Blender
missing so rigging is unavailable, is otherwise a count and a tooltip.

**Dismissed** appears only when you have dismissed something from the error banner. The banner's
Dismiss moves a message here rather than deleting it: a worker that died is reported through the
banner and through no check row at all, so this is the only copy.

## Advanced

**Layout.** *Sidebar width* offers narrow, default and wide (260, 300 and 360 px). Three named sizes
rather than a drag: a form has a width that reads well, and what a free drag bought was a way to make
the app look broken — but one number cannot suit a 1600-wide window and a 5120 one. *Reset pane
sizes* puts the split between the inspector and the library — both on the right sidebar — back to its
default, undoing any dragging of that divider. *Reset collapsed sections* re-opens every section that
has been collapsed anywhere in the app.

The sidebars narrow on their own when there is not room for the width you picked — at a high UI scale
in a small window, three columns at their full size want more pixels than the window can be shrunk
to. They give way in that order: both sidebars narrow first, down to a width a form is still usable
at, and only then does the centre pane start to shrink. Nothing is pushed off the edge, so the
inspector stays reachable at any scale the slider offers.

Moving the window to a monitor with different scaling re-reads the new display and rebuilds the
interface at its size, fonts included — you do not have to restart. Your *UI scale* is applied fresh
against the new monitor rather than carried across as a number of pixels, so a zoom that had to be
capped on one display is offered in full again on a display with room for it.

**Workspace layouts.** A **Layout** combo picks the active saved layout, with an explanatory
tooltip and, for one saved on a newer version that cannot be fully read back, a marker beside its
name. **Duplicate** copies the active layout under a new name and switches to the copy;
**Rename...** is disabled for the built-in layouts, which keep their names; **Reset** puts the
active layout back to its built-in arrangement; **Delete this layout** removes it outright and is
likewise disabled for a built-in, which is reset rather than deleted — there is no state a pane
cannot be recovered from. A layout can only reorder and hide panes, never delete one, and a hidden
pane is always listed here with one click to bring it back. This section is on the Settings page
specifically because Settings itself never changes shape with the layout, so it is reachable even
when a saved layout has gone wrong.

**AI agents.** *Allow AI agents to drive the Studio* lets a program that speaks the Model Context
Protocol build in Clay for you, and take a character from a species name to a rigged, animated
sprite sheet on its own. It is off on a fresh install and nothing listens until you switch it
on. Doing so writes a key into `mcp.token` in your Warlock home and opens a local named pipe: there
is no port, no firewall prompt, and nothing off your machine can reach it. Warlock runs exactly one
language model of its own, Familiar, on this computer only, and still connects to nothing — an agent
already running on this computer connects to *it*. While one is attached the menu bar's status group
says so. What it may touch is a two-part
rule: in Clay it works in a tab it opens for itself and cannot address any other document, so
nothing you have open is at risk; against your character Library it may read any row but can only
add to it — a new mesh, rig or sprite sheet, or a copy in your export folder — never rewrite,
re-rig, delete or rerun one, and it can cancel only the jobs it started itself. See
[Extending Warlock Studio](46-extending.md#driving-warlock-from-an-ai-agent) for the tools it is
given and how to point one at the app.

**Configuration.** *Effective configuration* lists every environment variable the app reads and what
this process resolved it to, with the ones actually set by the environment first and named by their
variable. It is the same table `warlock doctor` prints, and *Copy as text* puts it on the clipboard
for a bug report. It is read-only:
every entry is consumed at import time, so an editable version would have to say "restart to apply"
under every field.

**If Warlock crashes**, it says so on the way out rather than simply vanishing: a small dialog
reports that something went wrong, tells you whether there is unsaved work waiting to be offered
back on the next launch, and asks whether to open the folder your log is in. Answering no costs
nothing — the log is written either way, and the recovery offer does not depend on it.

Not everything the app remembers has a control in this pane. `studio_settings.json` also holds
the sidebar's internal split, and the pixel-art export
preferences — the
size and palette set in an asset's [Pixel art](22-generating-references.md#pixel-art) section, which
are the app's preferences rather than any one job's and so apply to whichever asset you look at
next.
