"""Poser mode's controller: authoring reusable poses against a skeleton template,
or against one real rigged asset's own mesh.

The ``studio/modes/clay/mode.py`` pattern -- state and logic here, drawing in ``main.py``
and the two panes, no imgui anywhere under this import -- so everything about
what a session holds is assertable without a GL context.

What it is for. A pose authored here is a *complete* bone map against one of
the shipped skeleton templates, stored globally under ``data_dir/poser/`` and
applied to any rigged asset of the same template from the asset's Pose panel.
By default the preview it is authored on is an armature-only GLB built by the
same Blender code path as a real rig (``op_armature``), over the canonical
unit box -- so the bone frames the editor rotates are the frames every bake
will see, and model units are character heights literally.

**Or the session can bind to one real asset instead** (``open_asset`` /
``close_asset``), loading that job's own ``rig.glb`` and posing the actual
mesh -- the same ``Viewer.enter_pose_mode`` the inspector's Pose panel uses,
just on Poser's own instance. The shared, skeleton-keyed pose library stays
exactly what it always was and stays visible either way (a template pose
applies by bone name, whether or not a mesh happens to be bound); what an
asset session adds is a place to save a pose *onto that asset* rather than
into the library, mirroring what the inspector's Pose tab already offers.

**Poser owns its own Viewer instance.** ``adopt_model`` on the shared viewer
calls ``exit_pose_mode`` unconditionally, so loading the preview into it would
silently discard unsaved inspector pose edits, bypassing ``pose_panel.guard``
-- the exact hazard class the shared viewer's own docs describe. A second
Viewer on the one GL context is already the app's shape (the compare viewport,
ClayView), and with it every cross-mode conflict vanishes structurally: the
inspector session survives on the shared viewer, the Poser session survives
mode trips like an open Inker document, and no guard is needed on *leaving*
the mode -- only on quit and on destructive in-mode actions. The instance
lives on the App/Ctx (``ctx.poser_viewer``), constructed lazily on the frame
thread at first entry and released in teardown; this module only ever reads
it through ``viewer_of``. The same instance is shared by both kinds of
session -- template preview and asset -- because they are mutually exclusive
by construction (``state.job_id`` says which) and never need to coexist.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .... import poselib
from ....kernels.rig import cliplib, poses, skeleton, store, templates
from ... import dialogs, journal

log = logging.getLogger(__name__)

#: How many jobs deep the "Rigged assets" walk looks, and the character-sheet
#: walks below it. P9 (2026-09-18): Troupe's own constants, moved here rather
#: than left behind an ``..troupe`` import -- the mode that owned them no
#: longer exists, and this picker's own docstring already borrowed them by
#: name for the identical reason (a page cap on a corpus that only grows).
SCAN_LIMIT = 400

#: How often the cast-adjacent walks below go back to the store. Troupe's own
#: ``CAST_REFRESH_LIVE``, moved for :data:`SCAN_LIMIT`'s reason -- ``db.list``'s
#: "twice a second" applied to a table shared with a worker that is writing it.
CAST_REFRESH_LIVE = 0.5

#: How often :func:`sheets` and :func:`active_sheet` go back to disk. Troupe's
#: ``SHEETS_REFRESH``, moved whole: the character sheet panes ask three to four
#: times a frame between them, for a directory that changes only when a sheet
#: is built.
SHEETS_REFRESH = 0.5

# Task keys. Prefixed "poser-" because the app claims results by prefix.
LIST_KEY = "poser-list"
SAVE_KEY = "poser-save"
DELETE_KEY = "poser-delete"
DUPLICATE_KEY = "poser-duplicate"
RENAME_KEY = "poser-rename"
PREVIEW_KEY_PREFIX = "poser-preview:"
CLIPS_KEY = "poser-clips"
CLIPS_SAVE_KEY = "poser-clips-save"
#: "Import clip...": sampling an externally authored FBX/GLB/glTF onto the
#: browsed template through ``service.clip_import.analyse`` -- a door, not a
#: job kind (see that module's own docstring), so one key is all a double
#: press needs refusing. Its landing merges into the *working copy* only
#: (:func:`adopt_imported_clips`); nothing reaches disk until Save.
CLIP_IMPORT_KEY = "poser-clip-import"
# The asset session's own keys -- distinct from ``pose_panel``'s ``pose-save:``/
# ``pose-del:`` family on purpose. ``main.py``'s generic task dispatch matches
# any ``"pose-"`` key against the *shared* viewer (``self.viewer.editor.dirty =
# False``); reusing that family from here would clear or misattribute the
# wrong viewer's dirty flag. These start "poser-" instead, so they route to
# this module's own ``on_task_done`` like every other key here does.
ASSET_POSES_KEY_PREFIX = "poser-asset-poses:"
ASSET_SAVE_KEY_PREFIX = "poser-asset-save:"
ASSET_DELETE_KEY_PREFIX = "poser-asset-delete:"
# The 2026-09-07 finding: with an asset bound, the Skeleton combo was replaced
# by a bare fact ("(from this asset's rig)") and there was no way back to a
# different skeleton short of closing the session and finding the source job
# in the Library. This is the key the queued re-rig is submitted under --
# "poser-" for the reason the three above are: main.py's generic "pose-"
# dispatch would clear the *shared* viewer's dirty flag, and the re-rig has to
# land on Poser's own (:func:`on_task_done`). It is a per-job key, matching
# ``stage_rig.rig_key``, so a double press within the same frame is refused by
# ``TaskRunner.submit`` alone -- see :func:`rerig`.
ASSET_RERIG_KEY_PREFIX = "poser-asset-rerig:"

# The automatic viewport binds' own keys (the 2026-09-11 audit, finding
# create-04): ``sync_asset``/``sync_preview`` run from ``_poser_viewport``'s
# draw on *every* frame Poser is open, not only the frame a click asked for
# one -- a queued re-rig landing (:func:`_land_rerig`) or a preview build
# landing (``on_task_done``'s ``PREVIEW_KEY_PREFIX`` branch) both arrive on an
# arbitrary frame with no press behind them. These two keys are the
# ``main.App._sync_viewer`` split (parse on a task thread, adopt on this one)
# applied to that automatic path; the genuinely click-driven load stays
# synchronous where it already was -- see ``open_asset``'s own docstring.
ASSET_LOAD_KEY = "poser-asset-load"
PREVIEW_LOAD_KEY = "poser-preview-load"

# The front-yaw control's own key, one per job -- writing this asset's front
# through ``service.jobs.set_front_yaw`` and reading the normalised value back
# into ``PoserState.asset_front_yaw`` once the write lands (:func:`set_front`,
# :func:`clear_front`). "poser-" for the reason every key above is: main.py's
# generic "poser-" dispatch routes any key with this prefix to this module's
# own :func:`on_task_done`, from wherever it was submitted -- including
# ``panes/overlay.py``'s copy of this control, for the unrigged props Poser
# itself can never open. Sharing the prefix rather than minting a second one
# is also what makes the two copies refuse each other's double-click: a job
# has exactly one front, so it needs exactly one key.
FRONT_KEY_PREFIX = "poser-front:"

# What pose_job_id carries in a *template* authoring session. Can never equal
# a 12-hex job id (a colon fails is_valid_id) -- but an *asset* session
# (``open_asset``) binds the viewer with the real job id instead, deliberately,
# because that save really is addressed to a job.
TOKEN_PREFIX = "poser:"


@dataclass
class PoserState:
    """One authoring session: which template, what the library holds.

    Nothing here is persisted -- a stored template key is the only candidate
    and re-deriving the default is cheaper than migrating a setting.
    """

    template: str = ""
    # The library records for ``template``, and the shipped presets beside
    # them. Both read by one task, adopted wholesale.
    poses: list[dict[str, Any]] = field(default_factory=list)
    presets: list[dict[str, Any]] = field(default_factory=list)
    loading: bool = False
    # Whether the library on screen is behind the store -- the findings_dirty
    # idiom: ``TaskRunner.submit`` refuses a key already in flight and nothing
    # else re-arms it, so a refresh wanted *while the list task runs* must be
    # a flag something pumps, cleared only when a submit is accepted.
    refresh_dirty: bool = False
    # The preview build in flight, and its answer once landed. The path is
    # bound to the viewer by ``sync_preview`` on the frame thread -- the
    # ``viewer.path`` comparison idiom Review uses, so an answer landing after
    # the user switched templates simply never binds.
    building: bool = False
    preview_path: Any = None
    preview_template: str = ""
    error: str = ""

    # -- the clip editor ------------------------------------------------------
    #
    # The Troupe programme's clip-authoring half: a clip was editable only as
    # raw JSON in the package tree. The library here is the *whole* file -- see
    # ``service.clips``, which will not save less than one -- held as the
    # editor's working copy and written back on Save.
    clips: dict[str, Any] = field(default_factory=dict)
    clips_loading: bool = False
    clips_dirty_flag: bool = False
    #: Which clip is open, by name. Names are a clip's identity here for the
    #: same reason they are in the file: that is what a clip's keys reference.
    clip: str = ""
    #: Which key of the open clip the editor is on, by *position*, because a
    #: clip may legitimately use one key twice (a contact pose either side of a
    #: passing one) and a name would not say which.
    key_index: int = 0
    #: Where the scrubber is, in expanded frames of the open clip. -1 is "off
    #: the scrubber, on a key" -- the state the pose gizmos are meaningful in,
    #: since a scrubbed frame is interpolated and editing it would have nowhere
    #: to be stored.
    frame: int = -1
    #: The expanded frames of the open clip, rebuilt whenever the working copy
    #: changes. Held rather than recomputed per draw: the scrubber reads it at
    #: frame rate.
    frames: list[dict[str, Any]] = field(default_factory=list)
    #: Whether the working copy differs from what is on disk. Set by every
    #: mutation, cleared when a save lands -- ``dirty``'s rule everywhere else
    #: in this app: never at submit, because a failed write has to leave the
    #: guard standing.
    clips_unsaved: bool = False
    #: Bumped by every mutation; ``save_clips`` records it so a landing save
    #: can tell whether edits arrived while it was writing.
    clips_touch_serial: int = 0
    clips_save_serial: int = 0
    clips_error: str = ""
    #: Whether the neighbouring keys are ghosted in the viewport. Session
    #: state, not a document fact: it is how the user is *looking* at the
    #: clip, the same argument Inker's tiled view makes about itself.
    onion: bool = False
    #: The per-clip reports from the most recent :func:`adopt_imported_clips`,
    #: for the pane's "Import report" -- ``cliptransfer.transfer``'s own
    #: ``report`` dicts, one per imported action, kept verbatim rather than
    #: re-derived so the pane shows exactly what the sample actually decided.
    #: Not a document fact either: it describes the *import*, not the working
    #: copy, and is not cleared by anything but the next import landing.
    clip_import_reports: list[dict[str, Any]] = field(default_factory=list)
    #: The most recent import's own ``skipped`` list (``service.clip_import.
    #: analyse``'s pass-through of ``op_clip_sample``'s ``payload["skipped"]``)
    #: -- actions Blender refused to sample at all (today, only for exceeding
    #: its frame-count limit), one human-readable line per action. Beside
    #: ``clip_import_reports`` rather than folded into it: a skipped action
    #: was never converted, so it has no ``report`` dict to sit inside. The
    #: 2026-09-18 audit, finding poser-01: a source file whose every action
    #: was too long used to "import" zero clips with nothing on screen saying
    #: why; set on every import landing (``on_task_done``'s ``CLIP_IMPORT_KEY``
    #: branch), including a clean one, so this never shows a stale skip list.
    clip_import_skipped: list[str] = field(default_factory=list)

    # -- the asset session ----------------------------------------------------
    #
    # Empty ``job_id`` is the ordinary, template-browsing session above; a
    # non-empty one means the viewer is bound to this real job's own rig.glb
    # instead of the meshless armature preview. The two are mutually exclusive
    # -- never both at once -- which is what lets everything above (the shared
    # library, the clip editor) keep working unchanged either way.
    job_id: str = ""
    #: The bound asset's own name, for the viewport banner -- read once at
    #: open time rather than looked up by id every draw.
    asset_label: str = ""
    #: The bound asset's ``rig.json``, or None if it had none readable. Held
    #: rather than re-read, the same reason ``pose_panel._enter`` reads it once.
    asset_rig: dict[str, Any] | None = None
    #: This asset's chosen front, in degrees, read once at :func:`open_asset`
    #: and kept current by :func:`on_task_done` when a write through
    #: ``service.jobs.set_front_yaw`` lands. 0.0 doubles as "unset" -- the
    #: service removes ``front_yaw`` from the job's params rather than storing
    #: a zero, so there is no state a stored 0 could mean that "no front
    #: chosen" does not already cover, and every reader here (the readout, the
    #: greyed reasons on Reset and Look at the front) treats the two as one.
    asset_front_yaw: float = 0.0
    #: This asset's own saved poses (``service.rig.list_poses``), distinct from
    #: the shared, skeleton-keyed library above.
    asset_poses: list[dict[str, Any]] = field(default_factory=list)
    #: Job ids with an in-flight ``ASSET_POSES_KEY_PREFIX`` read. A single
    #: bool here, before the 2026-09-08 audit (poser-04), meant opening a
    #: second rigged asset while the first one's fetch was still in flight
    #: silently refused to submit the second read (the shared flag read as
    #: "already loading"), and nothing re-armed it when the first asset's
    #: stale result landed -- the second asset showed "no saved poses" with no
    #: error and no recovery. Scoped per job id, the way the landing key
    #: (``ASSET_POSES_KEY_PREFIX`` + job id) already is.
    asset_poses_loading: set[str] = field(default_factory=set)
    #: Set when binding the viewer to the asset failed (a missing rig.glb, a
    #: GLB with no skin). Cleared only by :func:`retry_asset`, so a broken rig
    #: is not retried every frame.
    asset_error: str = ""
    #: In-flight re-rigs, keyed by the *source* asset's job id and holding the
    #: queued rig job's id -- ``svc_rig.create_rig`` only asks the serial
    #: queue to build a new rig.glb; the write itself lands minutes later, out
    #: of process, on the ``realmspinner-loop`` thread's own schedule -- so this is
    #: what :func:`pump_rerig` watches to notice each one actually finish. An
    #: entry is removed once its job lands (or fails), so a stale id is never
    #: polled forever.
    #:
    #: A single pair of fields here, before the 2026-09-11 audit (poser-05),
    #: meant re-rigging one asset and then, before that job landed, opening a
    #: different rigged asset and re-rigging it too silently overwrote the
    #: first re-rig's tracking with the second's -- exactly the
    #: ``asset_poses_loading`` hole the 2026-09-08 audit's poser-04 fixed by
    #: scoping per job id (see that field's own docstring); the identical fix
    #: was never applied here. The user can also close a session and open a
    #: different asset while the queue is still working on the first one's
    #: re-rig; comparing a landed job's *source* against the live ``job_id``,
    #: not the dict's mere presence, is what stops a re-rig queued for job A
    #: landing on whatever job B happens to be open when it finishes.
    rerig_jobs: dict[str, str] = field(default_factory=dict)
    #: Whether the Re-rig picker is expanded, and which skeleton is chosen in
    #: it. Here rather than in ``ctx.state.preview`` -- the pane-scratch dict
    #: the rest of the app uses for this -- because that dict outlives the
    #: session: leaving an asset with the picker open and opening another one
    #: reopened it, on the new asset, still showing the old asset's skeleton.
    #: :func:`close_asset` clears these with the rest of the session, which is
    #: the whole reason they live on the session's own state.
    rerig_open: bool = False
    rerig_choice: str = ""

    # -- the "Rigged assets" picker --------------------------------------------
    #
    # ``troupe_mode.sendable_meshes``'s two costs, paid the same way here:
    # ``can_open_in_poser`` reads ``files``, which is ``attach_files``' one
    # stat per listed name per row and not a column, and ``poser_library``
    # asks for this list every frame its own header is open. ``riggable_files``
    # is ``attach_files``'s own ``{job: (stamp, names)}`` cache, owned here
    # because the caller is required to own it.
    riggable_cache: list[dict[str, Any]] | None = None
    riggable_next: float = 0.0
    riggable_files: dict[str, Any] = field(default_factory=dict)

    # -- the skeleton editor (P6, 2026-09-13) ----------------------------------
    #
    # ``skeleton_editing`` and ``skeleton_error`` below are asset-session
    # state, exactly like ``job_id``'s own block above: reset in
    # :func:`open_asset`, :func:`close_asset` and :func:`_land_rerig`, never in
    # :func:`_reset_for_template` -- a skeleton switch can happen mid-edit
    # (:func:`set_template` is refused nowhere near this), but only a fresh
    # asset or a landed re-rig legitimately starts a new skeleton-editing
    # session. Mirrors ``viewer.editor.mode == "skeleton"`` rather than
    # replacing it: the pane needs to branch on this with no viewer at hand in
    # some draws (the empty-viewport paths in ``poser_viewport``), and a plain
    # bool answers that with no editor reference required.
    #
    # That reset guarantee stops here: it does not reach all of the rest of
    # this section. ``limb_preset``/``limb_side``/``limb_mirror`` below are
    # cleared by ``open_asset`` and ``close_asset`` only, not by
    # ``_land_rerig`` (a landed re-rig has no reason to blank a form the user
    # may still be filling in). ``skeleton_rename``/``skeleton_rename_for``
    # *are* cleared by all three, alongside ``skeleton_editing`` -- the
    # 2026-09-16 audit, finding poser-03: the in-session re-seed-on-selection-
    # change (see their own docstring below) is not a substitute for that,
    # because every humanoid-template rig shares bone names, so a stale,
    # uncommitted rename left over from a *different* asset session can read
    # as already seeded for the newly opened one's same-named bone and never
    # get re-seeded at all. The 2026-09-15 audit, finding poser-06, first
    # caught this comment overclaiming the reset for the rename buffer;
    # poser-03 is what made the claim true.
    skeleton_editing: bool = False
    #: {"field": str | None, "message": str} from the last refused
    #: :func:`apply_skeleton`, or None. Its own field rather than the app-wide
    #: ``ctx.state.field_errors`` ring: that mechanism addresses a control by
    #: name across the whole app, and the skeleton pane's controls (Add child,
    #: Split, the rename box) are not wired into it -- this is read directly by
    #: ``modes/poser/ui/panes/skeleton.py`` beside the control the field names.
    skeleton_error: dict[str, str] | None = None
    #: The shipped limb presets (``service.rig.limb_presets``), read once and
    #: cached -- job-independent and read-only, like ``rig_templates``' own
    #: catalogue, so re-reading it every frame the section is open would be a
    #: file walk for a menu that never changes underneath a running session.
    limb_presets_cache: list[dict[str, Any]] | None = None
    #: The "Add limb" row's own form fields -- session state, not a document
    #: fact, the same argument ``rerig_choice`` above makes for itself.
    limb_preset: str = ""
    limb_side: str = ""  # "L", "R", or "" for centre
    limb_mirror: bool = False
    #: The rename box's live typing buffer, and which bone it was seeded for
    #: -- ``inker_timeline``'s tag-rename idiom: the widget's value has to be
    #: stored back every frame while it is being typed into, and re-seeded
    #: only when the *selection* changes underneath it, or every keystroke on
    #: a rename would be clobbered by the next draw's "current name" read.
    #: Also cleared on every session boundary (see the section comment above)
    #: -- re-seeding on a name change alone cannot tell a stale cross-session
    #: value from a fresh one when two sessions' skeletons share a bone name.
    skeleton_rename: str = ""
    skeleton_rename_for: str | None = None

    # -- the character sheet (P9, 2026-09-18) ----------------------------------
    #
    # Troupe folded into Poser as a stage: rig, then clips, then a sheet, one
    # workspace. Everything below is Troupe's own ``TroupeState`` -- minus the
    # cross-character cast list, which decision 1 of the folding brief drops
    # outright (binding an asset is now the one way to look at its sheets,
    # through :func:`open_asset` above) -- carried onto this dataclass rather
    # than a nested one. It is *not* journal-tracked: nothing in
    # :data:`JOURNAL`'s ``slots``/``head_of``/``encode`` ever reads a
    # ``sheet_*`` field, exactly as Troupe's own state carried "there is no
    # document here, and that is the mode" -- a sheet is a selection over
    # files a worker already published, not something a crash can cost the
    # user beyond which frame the preview was on.
    #
    # A sheet is always the *bound asset's* own -- ``PoserState.job_id`` above
    # is the one id a sheet session needs, where Troupe kept a second,
    # independent ``job_id`` of its own. :func:`open_asset`/:func:`close_asset`
    # reset the block below with the rest of the session for the identical
    # reason they reset ``asset_poses``/``rerig_open``/etc.
    #: Whether the centre viewport and the right sidebar are showing the
    #: sprite preview instead of the pose editor. Meaningless with no asset
    #: bound; :func:`close_asset` and :func:`open_asset` both clear it.
    sheet_view: bool = False
    #: The selected sheet, by id -- never a record, ``TroupeState``'s own
    #: reason: a record cached across a frame can outlive the file it
    #: describes.
    sheet_id: str = ""
    #: What the preview is playing. Names from the selected sheet's snapshot,
    #: never indices -- see ``TroupeState.animation``.
    sheet_animation: str = "walk"
    sheet_direction: str = "front"
    #: Paused by default -- ``TroupeState.playing``'s own reasoning, carried
    #: whole: stepping already implies looking, which is why ``step`` clears
    #: this, and opening a sheet should mean the same thing.
    sheet_playing: bool = False
    sheet_clock: float = 0.0
    sheet_frame: int = 0
    sheet_speed: float = 1.0
    sheet_zoom: int = 6
    sheet_checker: bool = False
    sheet_show_pivot: bool = True
    #: The new-character and build-a-sheet form, shared the way Troupe's own
    #: ``form`` was shared between its settings pane and the direct "Build
    #: another sheet" door -- one construction of the request, not two.
    sheet_form: dict[str, Any] = field(default_factory=dict)
    #: The throttled directory read behind :func:`sheets`, keyed and timed the
    #: way ``TroupeState.sheets_cache`` was.
    sheets_cache: list[dict[str, Any]] | None = None
    sheets_key: str = ""
    sheets_next: float = 0.0
    #: The throttled sidecar read behind :func:`active_sheet`.
    sheet_cache: dict[str, Any] | None = None
    sheet_cache_key: tuple[str, str] = ("", "")
    sheet_cache_next: float = 0.0
    #: The pixel-art measurement for the selected sheet, keyed on its id --
    #: ``TroupeState.pixel_report_cache``'s own reason: a ``kind``-filtered
    #: page under the store's one lock has no business running every frame a
    #: sheet section is open.
    pixel_report_cache: dict[str, Any] | None = None
    pixel_report_key: str = ""
    pixel_report_next: float = 0.0

    def find_asset_pose(self, pose_id: Any) -> dict[str, Any] | None:
        return next((p for p in self.asset_poses if p.get("id") == pose_id), None)

    def open_clip(self) -> dict[str, Any] | None:
        """The clip record being edited, or None."""
        for record in self.clips.get("clips") or ():
            if record.get("name") == self.clip:
                return record
        return None

    def key_names(self) -> list[str]:
        """Every key pose name in the library, for the add/replace pickers."""
        return [str(p.get("name") or "") for p in self.clips.get("poses") or ()]

    def key_pose(self, name: str) -> dict[str, Any] | None:
        for pose in self.clips.get("poses") or ():
            if pose.get("name") == name:
                return pose
        return None

    def find(self, pose_id: Any) -> dict[str, Any] | None:
        """The library record with this id, or None.

        One method rather than five copies of the same generator expression:
        every caller here is answering "which record is the editor on", and a
        list is the right shape for a library the user reads top to bottom.
        """
        return next((p for p in self.poses if p.get("id") == pose_id), None)


def ensure(ctx: Any) -> PoserState:
    """The mode's state, built on first use -- lazy for the reason Clay's is."""
    state = ctx.state.poser
    if state is None:
        state = PoserState()
        ctx.state.poser = state
    if not state.template:
        entries = templates.catalog()
        default = str(getattr(ctx, "rig_default", "") or "")
        keys = {e["key"] for e in entries}
        state.template = default if default in keys else (entries[0]["key"] if entries else "")
    return state


def viewer_of(ctx: Any) -> Any:
    return getattr(ctx, "poser_viewer", None)


def token(template: str) -> str:
    return f"{TOKEN_PREFIX}{template}"


# --- entering and refreshing -------------------------------------------------


def enter(ctx: Any) -> None:
    """Arriving in the mode: refresh the library and ask for the preview.

    Driven off the mode change (the Review-arrival rule), not off "the list is
    empty" -- which would submit a directory walk every frame on a library
    that genuinely holds nothing. Fires on *every* arrival, including a mode
    trip back into an already-open asset session -- so an asset session asks
    for its own poses to be re-read instead of the meshless preview, which
    would otherwise spend a Blender subprocess building a preview nothing is
    about to show.
    """
    state = ensure(ctx)
    if not state.template:
        return
    refresh(ctx)
    clips_refresh(ctx)
    if state.job_id:
        refresh_asset_poses(ctx)
    else:
        request_preview(ctx)


def _collect(svc: Any, template: str) -> dict[str, Any]:
    """The library rows and the shipped presets, in one task -- both are disk
    reads and the panes need them together."""
    from ....service import poses as svc_poses
    from ....service import rig as svc_rig

    return {
        "template": template,
        "poses": svc_poses.list_library(svc, template)["poses"],
        # Through the service, like the library rows beside it: the service
        # layer is the only business logic, and its refusal wording is the one
        # the generic failure toast shows.
        "presets": svc_rig.template_presets(template)["poses"],
    }


def refresh(ctx: Any) -> None:
    """Ask for the library to be re-read. The read itself is ``pump``'s."""
    ensure(ctx).refresh_dirty = True
    pump(ctx)


def pump(ctx: Any) -> None:
    """Submit the wanted refresh if nothing stands in the way.

    Called every frame from the library pane's draw, and again when the list
    task lands -- the frame pump is what covers a submit the runner refused,
    and the landing pump is what lets a save that arrived mid-list re-read
    without waiting for a frame. The flag clears *only* when the submit is
    accepted, the findings_dirty rule; a refusal leaves it set for the next
    pump rather than dropping the refresh for good.
    """
    state = ensure(ctx)
    if not state.refresh_dirty or state.loading or not state.template:
        return
    state.loading = True
    if ctx.submit(LIST_KEY, _collect, ctx.svc, state.template):
        state.refresh_dirty = False
    else:
        # The runner refuses a key already in flight; leaving the flag set
        # would make the mode permanently inert after a double press.
        state.loading = False


def request_preview(ctx: Any) -> None:
    """Ask for the current template's armature preview to exist.

    A cache hit inside ``template_preview`` is a couple of stats, so this is
    safe to ask on every arrival; a cold build is a Blender subprocess, which
    is exactly why it is a task and the viewport draws a progress row.
    """
    from ....service import poses as svc_poses

    state = ensure(ctx)
    if not state.template or not getattr(ctx, "rigging_available", False):
        return
    key = f"{PREVIEW_KEY_PREFIX}{state.template}"
    if ctx.busy(key):
        return
    state.building = True
    state.error = ""
    if not ctx.submit(key, svc_poses.template_preview, ctx.svc, state.template):
        state.building = False


def _reset_for_template(state: PoserState, template: str) -> None:
    """Clear every field a skeleton switch invalidates -- the poser-01 reset.

    Extracted 2026-09-07 for a third caller: :func:`set_template`'s own
    switch, :func:`open_asset` binding to a rig cut from a different template
    than the one already being browsed, and a re-rig landing
    (:func:`_land_rerig`) that changed the bound asset's own template. All
    three are the same fact -- the clip editor's working copy is keyed by
    template -- and the 2026-09-07 audit (poser-01) found ``set_template``
    left it untouched, so "Save clips" afterwards wrote the *old* template's
    working copy under the *new* template's name, with no prompt at all
    (``clips_pump``'s own ``clips_unsaved`` guard means a bare
    ``clips_refresh`` would not even have re-read it once the switch landed).
    ``state.template`` is part of the reset rather than a separate assignment
    at each call site, so there is exactly one place the two can drift.
    """
    state.template = template
    state.clips = {}
    state.clip = ""
    state.key_index = 0
    state.frame = -1
    state.frames = []
    state.clips_error = ""
    state.clips_unsaved = False


def set_template(ctx: Any, template: str) -> None:
    """Switch skeletons, behind the guard: the editor holds one template's
    pose, and a switch discards it."""
    state = ensure(ctx)
    if template == state.template:
        return

    def proceed() -> None:
        _reset_for_template(state, template)
        state.poses, state.presets = [], []
        state.preview_path, state.preview_template = None, ""
        viewer = viewer_of(ctx)
        if viewer is not None:
            # The old template's armature must not stay poseable under the new
            # template's library; sync_preview binds the new one when it lands.
            viewer.clear()
        refresh(ctx)
        clips_refresh(ctx)
        request_preview(ctx)

    def guarded() -> None:
        guard(ctx, "switch skeletons", proceed)

    # ``guard`` above only reads the *Poser viewer's* editor (the pose gizmo),
    # which is the other half of poser-01: a switch with unsaved clip edits
    # sailed through with no prompt at all, since nothing here ever asked
    # about ``clips_unsaved``. Ask first, in ``revert_clips``'s own words, then
    # fall through to the ordinary pose-gizmo guard.
    if state.clips_unsaved:
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Discard unsaved changes?",
                message="Unsaved clip changes will be lost if you switch skeletons.",
                on_confirm=guarded,
            )
        )
        return
    guarded()


# --- the "Rigged assets" picker -----------------------------------------------
#
# The 2026-09-09 gap: Poser had no way to open an asset from inside the mode
# at all -- the only doors in were the inspector's Pose panel link and, once
# B1 closed it, the library/inspector exits list, both of which mean leaving
# whatever the user was looking at. This is Poser's own picker's data half,
# modelled line for line on Troupe's own ``can_send_to_troupe`` / ``sendable
# _meshes`` -- since folded into this module as :func:`can_render_sheet` (P9,
# 2026-09-18) -- see both docstrings for the two costs paid here too.


def can_open_in_poser(ctx: Any, job: Any) -> bool:
    """Whether this row belongs in the "Rigged assets" picker.

    From the cached row alone -- no filesystem -- :func:`can_render_sheet`'s
    shape and its reason: the pane asks this every frame its own header is
    open. ``rig.glb`` in ``files``, not a rig *row*: a rig job's own row
    carries no files of its own (``asset_open``'s docstring names the trap),
    so it is the *mesh* a rig lands beside that belongs in this picker --
    ``asset_exits._mesh_for`` makes the identical argument for the exits list.
    """
    del ctx
    return bool(
        job
        and job.get("stage") == "model"
        and job.get("status") == "done"
        and not job.get("deleted_at")
        and "rig.glb" in (job.get("files") or [])
    )


def riggable_assets(ctx: Any) -> list[dict[str, Any]]:
    """Every rigged mesh the picker may offer, newest first. Throttled.

    Troupe's own ``sendable_meshes`` pattern (that mode folded into this one in
    P9, 2026-09-18), reused rather than restated: the page cap and the refresh
    cadence are :data:`SCAN_LIMIT`/:data:`CAST_REFRESH_LIVE`, over the same
    store, for the same reason a second set of numbers would just be a second
    answer to a question already answered. ``can_open_in_poser`` reads
    ``files``, which is ``attach_files``' one-stat-per-listed-name-per-row
    doing and not a column -- ``files_cache`` is what ``list_jobs`` offers for
    exactly this, and the throttle is what keeps a 400-row page from being
    thousands of stats a frame on the thread that must not block.

    **This row shape is not ``sendable_meshes``'s, on purpose.** That picker's
    rows are read by ``troupe_send.ask``/``send_to_troupe``, which re-reads
    the row through the service before acting, so a display-only trim (``id``,
    a merged ``prompt``, ``created_at``) costs nothing. This picker's rows are
    handed straight to :func:`open_asset`, which reads the dict it is given
    and never re-reads it -- ``job.get("id")``, ``job.get("name") or
    job.get("prompt")`` for the label, and ``(job.get("params") or
    {}).get("front_yaw")`` for the asset's recorded facing. The 2026-09-09
    review defect this fixes: a first cut copied ``sendable_meshes``' trim
    without checking its consumer, so every session opened from this picker
    silently lost the asset's front and faced yaw 0 regardless of what
    ``poser_mode.set_front`` had recorded for it. ``name`` and ``prompt`` are
    carried raw (not merged, unlike ``sendable_meshes``' display-only
    ``prompt``) so ``open_asset``'s own name-then-prompt preference reads the
    same row the exits list would have handed it; ``params`` is carried whole
    rather than just ``front_yaw`` because it is already a parsed dict on
    every row ``list_jobs`` returns and a second, narrower shape would only
    be one more thing for this and ``open_asset`` to agree about by hand.
    """
    from ....service import jobs as svc_jobs

    state = ensure(ctx)
    now = time.monotonic()
    if state.riggable_cache is None or now >= state.riggable_next:
        out = [
            {
                "id": str(row["id"]),
                "name": row.get("name") or "",
                "prompt": row.get("prompt") or "",
                "created_at": row.get("created_at"),
                "params": row.get("params") or {},
            }
            for row in svc_jobs.list_jobs(
                ctx.svc, limit=SCAN_LIMIT, files_cache=state.riggable_files
            )
            if can_open_in_poser(ctx, row)
        ]
        out.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        state.riggable_cache = out
        state.riggable_next = now + CAST_REFRESH_LIVE
    return state.riggable_cache


def invalidate_riggable(ctx: Any) -> None:
    """Drop the throttled rigged-asset list so the next draw re-reads it.

    ``invalidate_sheets``'s sibling, called from the same kind of place: a rig
    landing while Poser's own picker is open is exactly the event that makes
    the throttled list stale before its interval is up.
    """
    ensure(ctx).riggable_cache = None


# --- the asset session ---------------------------------------------------------


def open_asset(ctx: Any, job: dict[str, Any], *, sheet_id: str | None = None) -> None:
    """Bind the session to one real rigged asset's own mesh, behind the guard.

    Reads the rig's template so the shared library beside it is the one that
    applies -- ``pose_panel.open_in_poser``'s old rule, kept, since a saved
    library pose still applies by bone name whether or not a mesh is bound.
    A missing or unreadable rig.json is not a reason to refuse the trip, the
    same tolerance ``pose_panel._enter`` already applies: only the mirror
    button and the joint editor need what it carries.

    **Binds the viewer synchronously, right here (the 2026-09-11 audit,
    finding create-04).** ``proceed`` runs only from this door's own press or
    the confirm answering it -- exactly the case ``viewer_embed.Viewer.
    load_model``'s own docstring sanctions ("the wait is the point... because
    the user just pressed something"), the same precedent ``pose_panel._enter``
    already stands on. ``sync_asset`` used to be where this load happened
    instead, called every frame from the draw loop regardless of whether a
    click was behind it -- which also made it responsible for the *other*
    case, a queued re-rig landing while the user is doing nothing in
    particular (:func:`_land_rerig`). That automatic case still has no click
    to hide the wait behind, so it goes through :func:`sync_asset`'s own
    parse-on-a-task/adopt-on-this-frame split instead; see its docstring.

    **P9 (2026-09-18):** ``sheet_id``, keyword-only and ``None`` by default,
    is the door Troupe's own ``open_sheet``/``select`` folded into. ``None``
    means an ordinary pose-authoring open and leaves the sheet section alone,
    which is every call site that predates this phase. Any string -- including
    ``""`` for "whichever sheet is newest" -- points the session at that sheet
    and switches the workspace into sheet view once the bind lands, through
    :func:`select_sheet`. Threaded through ``proceed`` rather than applied by
    the caller afterwards, because the caller cannot tell *when* ``proceed``
    actually ran -- a template switch with unsaved clip edits defers it behind
    a confirm the user has not yet answered.
    """
    from ....service import rig as svc_rig

    state = ensure(ctx)
    job_id = str(job.get("id") or "")
    if not job_id:
        return
    rig = None
    with contextlib.suppress(Exception):
        rig = svc_rig.get_rig(ctx.svc, job_id)
    template = str((rig or {}).get("template") or "") or state.template
    switching_template = bool(template) and template != state.template

    def proceed() -> None:
        if switching_template:
            # Opening an asset can also change which skeleton's clip library
            # is open -- ``_reset_for_template``'s reset applies here too.
            _reset_for_template(state, template)
        state.job_id = job_id
        state.asset_label = str(job.get("name") or job.get("prompt") or job_id)
        state.asset_rig = rig
        # From the job dict already in hand, not a fresh read -- ``open_asset``
        # is handed the row the Library or Create already loaded, and
        # everything else here reads it the same way.
        state.asset_front_yaw = float((job.get("params") or {}).get("front_yaw") or 0.0)
        state.asset_error = ""
        state.asset_poses = []
        state.skeleton_editing = False
        state.skeleton_error = None
        # The 2026-09-16 audit, finding poser-03: these two used to be
        # re-seeded only by comparing the selected bone's *name* to
        # ``skeleton_rename_for`` (``poser_skeleton._rename``), and every
        # humanoid-template rig shares bone names -- so a typed-but-uncommitted
        # rename left in the box survived a close/open onto a different asset
        # and showed as though it were that asset's own bone's name.
        state.skeleton_rename = ""
        state.skeleton_rename_for = None
        # The 2026-09-13 audit (poser-01): these five fields are session
        # scratch for the Re-rig picker and the Add-limb form, and their own
        # docstrings say the session clears them -- but only close_asset did.
        # Picking a different asset while the picker was open on the first
        # carried its stale template choice (and any half-filled limb form)
        # onto the new asset, so Confirm re-rigged the wrong mesh under it.
        state.rerig_open = False
        state.rerig_choice = ""
        state.limb_preset = ""
        state.limb_side = ""
        state.limb_mirror = False
        # The character-sheet session, P9's own version of the five fields
        # above: a sheet selected on the asset just left describes nothing
        # about this one, and a stale ``sheet_view`` would show the sprite
        # preview over a session that has not picked a sheet yet.
        state.sheet_view = False
        state.sheet_id = ""
        _release_sheet_caches(ctx, state)
        viewer = viewer_of(ctx)
        if viewer is not None:
            # Whatever the viewer was showing -- another asset, the meshless
            # preview -- is not this one.
            viewer.exit_pose_mode()
            viewer.clear()
            _bind_asset_now(ctx, state, viewer, job_id)
        refresh(ctx)
        clips_refresh(ctx)
        refresh_asset_poses(ctx)
        if sheet_id is not None:
            select_sheet(ctx, sheet_id)

    def guarded() -> None:
        guard(ctx, "open this asset", proceed)

    if switching_template and state.clips_unsaved:
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Discard unsaved changes?",
                message="Unsaved clip changes will be lost if you open this asset.",
                on_confirm=guarded,
            )
        )
        return
    guarded()


def close_asset(ctx: Any) -> None:
    """Leave the asset session, behind the guard, back to browsing templates."""
    state = ensure(ctx)
    if not state.job_id:
        return

    def proceed() -> None:
        state.job_id = ""
        state.asset_label = ""
        state.asset_rig = None
        state.asset_front_yaw = 0.0
        state.asset_poses = []
        state.asset_error = ""
        state.rerig_open = False
        state.rerig_choice = ""
        state.skeleton_editing = False
        state.skeleton_error = None
        # The 2026-09-16 audit, finding poser-03: see open_asset's own comment.
        state.skeleton_rename = ""
        state.skeleton_rename_for = None
        state.limb_preset = ""
        state.limb_side = ""
        state.limb_mirror = False
        state.sheet_view = False
        state.sheet_id = ""
        _release_sheet_caches(ctx, state)
        viewer = viewer_of(ctx)
        if viewer is not None:
            viewer.exit_pose_mode()
            viewer.clear()
        request_preview(ctx)

    guard(ctx, "close this asset", proceed)


def retry_asset(ctx: Any) -> None:
    """Clear a load failure so :func:`sync_asset` tries again next frame."""
    ensure(ctx).asset_error = ""


def refresh_asset_poses(ctx: Any) -> None:
    """Ask for the bound asset's own saved poses to be re-read."""
    from ....service import rig as svc_rig

    state = ensure(ctx)
    if not state.job_id or state.job_id in state.asset_poses_loading:
        return
    state.asset_poses_loading.add(state.job_id)
    key = f"{ASSET_POSES_KEY_PREFIX}{state.job_id}"
    if not ctx.submit(key, svc_rig.list_poses, ctx.svc, state.job_id):
        state.asset_poses_loading.discard(state.job_id)


def save_pose_to_asset(ctx: Any) -> None:
    """Save the pose being edited onto the bound asset, not the shared library.

    The point of opening a real mesh rather than a template: a change here
    should be able to stick to *this* asset, exactly like the inspector's own
    Pose tab already offers via ``service.rig.save_pose``.
    """
    from ....service import rig as svc_rig

    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if not state.job_id or viewer is None or not viewer.pose_mode:
        return
    if viewer.editor.mode == "skeleton":
        # P6 (2026-09-13): the armature is at rest for the whole of a
        # skeleton-editing session (``PoseEditor.enter_skeleton_mode`` resets
        # it on the way in), so "saving" it now would write a rest pose over
        # whatever this asset had -- apply or cancel first.
        ctx.toast("Apply or cancel the skeleton edit before saving a pose.", "info")
        return
    job_id = state.job_id
    existing = viewer.editor.current

    def accept(name: str) -> None:
        # root_translation travels with the pose here too, mirroring _payload's
        # library-save shape exactly -- without it, a crouch or hop authored
        # with Move root and saved onto the asset (rather than the shared
        # library) silently lost its offset with no error (poser-02, the
        # 2026-09-11 audit): the bake reads pose.get("root_translation") and
        # falls through to an offset-less spec when the key is absent.
        payload: dict[str, Any] = {
            "name": name,
            "bones": viewer.get_pose(),
            "root_translation": viewer.editor.root_translation(),
        }
        if existing:
            payload["id"] = existing
        ctx.submit(
            f"{ASSET_SAVE_KEY_PREFIX}{job_id}", svc_rig.save_pose, ctx.svc, job_id, payload
        )

    ctx.prompts.ask(dialogs.Prompt(title="Name this pose", label="Name", on_accept=accept))


def set_front(ctx: Any) -> None:
    """Record the live camera's yaw as this asset's front, through
    ``service.jobs.set_front_yaw``.

    Reads ``camera._goal_theta``, not ``theta`` -- ``CameraState.read_from``'s
    idiom (``studio/modes/clay/state.py:84``), copied rather than restated: the camera is
    damped toward a goal it has not reached yet, so a press mid-glide would
    record the frame the button happened to interrupt, not the direction the
    user actually pointed the camera at.

    **No sign flip and no origin shift.** With ``phi = pi/2 - e``,
    ``Camera.position`` and ``viewer.sheet.camera_position`` are the same
    function of yaw/theta (``viewer/camera.py:57-64``,
    ``viewer/sheet.py:46-62``), and ``viewer.scene.placement`` is a pure
    translation that cannot rotate the frame out from under either -- so the
    viewport's ``theta`` and a rendered sheet's ``yaw`` already agree on what
    the number means, and this only has to carry it across, never correct it.

    A button and not a computed default, because the 2026-08-05 sweep
    (``dev/measurements/2026-08-04-view-calibration.md``) found a mesh's own
    matched view scatters *uniformly* across a 330-degree range over 37 jobs
    -- trellis-server picks its own orientation per subject, and there is
    nothing on disk "the front" could be derived from. Submitted on a per-job
    key (:data:`FRONT_KEY_PREFIX`), so ``TaskRunner.submit`` refuses a second
    press while the first is still in flight -- the same shallow,
    already-accepted double-click guard the rest of this module relies on.
    """
    from ....service import jobs as svc_jobs

    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if not state.job_id or viewer is None:
        return
    job_id = state.job_id
    camera = viewer.camera
    degrees = math.degrees(float(getattr(camera, "_goal_theta", camera.theta))) % 360.0
    key = f"{FRONT_KEY_PREFIX}{job_id}"
    if not ctx.submit(key, svc_jobs.set_front_yaw, ctx.svc, job_id, degrees):
        ctx.toast("Still saving the previous front change.", "info")


def clear_front(ctx: Any) -> None:
    """Put this asset's front back to unset.

    Writing 0 through ``set_front_yaw`` *removes* ``front_yaw`` from the job's
    params rather than storing a zero -- the service's own contract, and the
    reason ``PoserState.asset_front_yaw`` treats 0.0 as "unset" everywhere it
    is read. Refuses with nothing to do when it already is: the pane's own
    Reset button is greyed for the same fact, stated as a reason rather than
    silently doing nothing.
    """
    from ....service import jobs as svc_jobs

    state = ensure(ctx)
    if not state.job_id or not state.asset_front_yaw:
        return
    job_id = state.job_id
    if not ctx.submit(f"{FRONT_KEY_PREFIX}{job_id}", svc_jobs.set_front_yaw, ctx.svc, job_id, 0.0):
        ctx.toast("Still saving the previous front change.", "info")


def look_at_front(ctx: Any) -> None:
    """Turn the camera to the recorded front, keeping everything else.

    Only ``_goal_theta`` moves -- ``phi``, ``distance`` and ``target`` are
    left exactly where they are, ``Camera.look_along``'s own documented rule
    (``viewer/camera.py:99-110``) and its reason: an angle change that also
    reframed would throw away the part of the model the user had lined up on
    the other two axes, which is the one thing they were about to check. This
    is deliberately *not* a call to ``look_along`` itself, and
    ``AXIS_VIEWS["front"]`` is deliberately left untouched by all of this --
    its docstring makes the view *names* the contract, it is shared with Clay
    in four places, and "front" there names the model's own -Z, which the
    front-yaw control has not moved and was never meant to.
    """
    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if not state.job_id or viewer is None or not state.asset_front_yaw:
        return
    viewer.camera._goal_theta = math.radians(state.asset_front_yaw)


def apply_asset_pose(ctx: Any, pose_id: str) -> None:
    """Load one of the asset's own saved poses into the editor, behind the guard."""
    state = ensure(ctx)
    viewer = viewer_of(ctx)
    record = state.find_asset_pose(pose_id)
    if record is None or viewer is None or not viewer.pose_mode:
        return
    if _refuse_while_skeleton_editing(ctx, state):
        return

    def proceed() -> None:
        viewer.reset_all(dirty=False)
        viewer.set_pose(record.get("bones") or {}, pose_id=record["id"], dirty=False)
        # apply_pose's own line, restoring what save_pose_to_asset now saves
        # (poser-02, the 2026-09-11 audit): omitting this made a root offset
        # saved directly onto an asset silently vanish on the very next load.
        viewer.set_root_translation(
            record.get("root_translation") or [0.0, 0.0, 0.0], dirty=False
        )

    guard(ctx, "apply a saved pose", proceed)


def delete_asset_pose(ctx: Any, pose_id: str, name: str) -> None:
    """Delete one of the asset's own saved poses, behind a confirm."""
    from ....service import rig as svc_rig

    state = ensure(ctx)
    if not state.job_id:
        return
    job_id = state.job_id
    dialogs.ask_delete(
        ctx,
        title="Delete this pose?",
        message=(
            f'"{name}" and its saved GLB are deleted. This cannot be undone.\n\n'
            "The asset's mesh and skeleton are untouched."
        ),
        on_confirm=lambda: ctx.submit(
            f"{ASSET_DELETE_KEY_PREFIX}{job_id}:{pose_id}",
            svc_rig.delete_pose,
            ctx.svc,
            job_id,
            pose_id,
        ),
    )


def rerig(ctx: Any, template: str) -> None:
    """Re-rig the bound asset under ``template``, behind the guard.

    The 2026-09-07 finding: once an asset is bound, the Skeleton combo above
    it is replaced by the fact "(from this asset's rig)" -- correct, but a
    dead end, since the only route to a different skeleton was closing the
    session, finding the source job in the Library, and choosing Rig from
    there. This queues the same job the Library's own Rig action does
    (``svc_rig.create_rig``, ``service/rig.py:56``, which already refuses to
    rig a rig job and takes ``state.job_id`` -- the *mesh* job, since a rig's
    artifacts land beside the mesh, never in the rig job's own directory) --
    just reachable from the session that already has the asset open.

    Guarded because a re-rig means a fresh session once it lands
    (:func:`_land_rerig`), and any pose being edited on the old rig is lost
    with it -- exactly the hazard every other destructive door here asks
    about. Submitted under a per-job key (``ASSET_RERIG_KEY_PREFIX``,
    matching ``stage_rig.rig_key``'s shape): ``TaskRunner.submit`` refuses a
    second press while the first is still in flight, which is the whole
    concurrency guard the plan asks for here -- the same shallow, already-
    accepted protection the ordinary "Rig this mesh again" button relies on.

    **A custom skeleton gets its own warning (P6, 2026-09-13), ahead of the
    ordinary pose guard.** The generic guard above only asks about a pose
    being *authored* right now; it says nothing when the bound rig's own
    *shape* -- every bone :func:`apply_skeleton` added, split or renamed --
    was edited away from its template, and a re-rig discards exactly that
    shape by rebuilding from the chosen template's stock bones. A clean editor
    over a custom skeleton would otherwise sail through with no warning at
    all.
    """
    from ....service import rig as svc_rig

    state = ensure(ctx)
    if not state.job_id:
        return
    job_id = state.job_id

    def proceed() -> None:
        key = f"{ASSET_RERIG_KEY_PREFIX}{job_id}"
        if not ctx.submit(key, svc_rig.create_rig, ctx.svc, job_id, template=template):
            ctx.toast("Still re-rigging this asset.", "info")

    if (state.asset_rig or {}).get("skeleton") == "custom":
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Discard this asset's custom skeleton?",
                message=(
                    "This asset's skeleton was edited away from its template. "
                    "Re-rigging discards those edits and any pose being "
                    "authored, and rebuilds from the chosen template's own "
                    "bones."
                ),
                on_confirm=proceed,
            )
        )
        return

    guard(ctx, "re-rig this asset", proceed)


# --- the skeleton editor (P6, 2026-09-13) -------------------------------------
#
# A third mode on the same ``PoseEditor`` a pose session already opened
# (``viewer.editor.mode``: "pose" | "joints" | "skeleton"), reachable only from
# an open asset session -- there is no rig.json to edit against the meshless
# template preview. Applying is a re-rig, reusing :data:`ASSET_RERIG_KEY_PREFIX`
# and the whole :func:`pump_rerig`/:func:`_land_rerig` landing path a template
# re-rig already goes through: the mesh is re-skinned in Blender either way.


def enter_skeleton_edit(ctx: Any) -> None:
    """Switch the bound asset's session into skeleton editing.

    Requires a real asset open with a readable rig -- refused by a toast, not
    by disabling the door silently, so a stale press (the asset closed, the
    rig failed to read) reads as a sentence rather than nothing happening.
    The pane's own "Edit skeleton" button additionally greys itself with a
    reason when there is plainly no asset open; this is the door's own
    refusal for what that greying cannot see ahead of the press.

    Refuses by name, rather than through :func:`guard`'s confirm, when the
    editor already holds unsaved *pose* edits: entering skeleton mode resets
    the armature to rest (``PoseEditor.enter_skeleton_mode``), and the more
    useful answer here is "save or reset the pose first", not one more
    confirm stacked on top of the one every other destructive door in this
    mode already offers.
    """
    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if not state.job_id or viewer is None or not viewer.pose_mode:
        ctx.toast("Open a rigged asset first.", "info")
        return
    if viewer.editor.mode == "skeleton":
        return
    if viewer.editor.has_unsaved_edits():
        ctx.toast("Save or reset the pose before editing the skeleton.", "info")
        return
    rig = state.asset_rig
    if not rig or not rig.get("bones"):
        ctx.toast("This asset has no readable rig to edit.", "warn")
        return
    state.skeleton_error = None
    viewer.enter_skeleton_mode(rig)
    if viewer.editor.mode != "skeleton":
        # ``PoseEditor.enter_skeleton_mode`` no-ops on an editor with nothing
        # bound (``PoseEditor.bound``) -- defensive, since every real door in
        # here binds the editor before this is reachable, but a flag this
        # module then trusts everywhere ("editing" gates saves, scrubbing,
        # the confirm on Cancel) must not go true over a mode that never
        # actually switched.
        ctx.toast("This asset has no readable rig to edit.", "warn")
        return
    state.skeleton_editing = True


def exit_skeleton_edit(ctx: Any) -> None:
    """The frame-thread half of landing or cancelling: back to pose mode,
    with nothing left for the pane to show as "editing"."""
    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if viewer is not None and viewer.editor.mode == "skeleton":
        viewer.exit_skeleton_mode()
    state.skeleton_editing = False
    state.skeleton_error = None


def cancel_skeleton_edit(ctx: Any) -> None:
    """Leave skeleton editing, behind a confirm if the draft is unsaved.

    ``guard``'s own pattern, but over ``draft_dirty`` specifically rather than
    the editor's pose-mode dirty flag -- both live under
    ``has_unsaved_edits()``, so :func:`guard` itself would work here too, but
    its wording ("Unsaved pose changes...") is wrong for a skeleton draft.
    """
    from ... import dialogs

    viewer = viewer_of(ctx)
    if viewer is None or viewer.editor.mode != "skeleton":
        return
    if not viewer.editor.draft_dirty:
        exit_skeleton_edit(ctx)
        return
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Discard unsaved changes?",
            message="Unsaved skeleton changes will be lost if you cancel editing.",
            on_confirm=lambda: exit_skeleton_edit(ctx),
        )
    )


def apply_skeleton(ctx: Any) -> None:
    """Submit the drafted skeleton as a fresh re-rig.

    Through :data:`ASSET_RERIG_KEY_PREFIX`, exactly the key :func:`rerig`
    submits under: a landing runs the identical
    :func:`pump_rerig`/:func:`_land_rerig` path, which is what ends the
    skeleton-editing session once the new rig actually lands (a queued job is
    minutes of Blender, not an inline call -- see :func:`edit_skeleton`'s own
    docstring). A refusal (``service.errors.Invalid``, always field-addressed
    here: ``skeleton.validate_skeleton`` never raises without one) is recorded
    on ``state.skeleton_error`` by :func:`on_task_failed`, so the pane can put
    it under the control it names instead of only the generic red toast.
    """
    from ....service import rig as svc_rig

    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if not state.job_id or viewer is None or viewer.editor.mode != "skeleton":
        return
    job_id = state.job_id
    payload = viewer.skeleton_payload()
    key = f"{ASSET_RERIG_KEY_PREFIX}{job_id}"
    state.skeleton_error = None
    if not ctx.submit(key, svc_rig.edit_skeleton, ctx.svc, job_id, payload):
        ctx.toast("Still re-rigging this asset.", "info")


def _skeleton_call(ctx: Any, fn: Any, *args: Any) -> tuple[bool, Any]:
    """Run one draft mutator, turning a refusal into ``state.skeleton_error``.

    Every ``skel_*`` editor call is local and synchronous -- unlike
    :func:`apply_skeleton`, nothing here touches the queue -- so a refusal is
    a :class:`store.RigError` raised straight out of the call, not a task
    landing minutes later. Cleared on success, the same "only the landing
    clears it" rule the async doors in this module already follow, applied to
    a call that lands immediately. Returns ``(ok, result)`` rather than only
    ``result`` because ``skel_remove_pivot`` succeeds with ``None`` -- a bare
    result cannot tell that apart from a refusal.
    """
    state = ensure(ctx)
    try:
        result = fn(*args)
    except store.RigError as exc:
        state.skeleton_error = {"field": exc.field or "", "message": str(exc)}
        return False, None
    state.skeleton_error = None
    return True, result


def skeleton_select(ctx: Any, name: str | None) -> None:
    """Select a bone (or its tail handle) in the skeleton editor."""
    viewer = viewer_of(ctx)
    if viewer is None or viewer.editor.mode != "skeleton":
        return
    viewer.editor.selected = name


def skeleton_add_child(ctx: Any, parent: str) -> None:
    """``PoseEditor.skel_add_child`` already selects the new bone."""
    viewer = viewer_of(ctx)
    if viewer is None or viewer.editor.mode != "skeleton":
        return
    _skeleton_call(ctx, viewer.skel_add_child, parent)


def skeleton_split(ctx: Any, name: str) -> None:
    viewer = viewer_of(ctx)
    if viewer is None or viewer.editor.mode != "skeleton":
        return
    _skeleton_call(ctx, viewer.skel_split, name)


def skeleton_remove_pivot(ctx: Any, name: str) -> None:
    viewer = viewer_of(ctx)
    if viewer is None or viewer.editor.mode != "skeleton":
        return
    _skeleton_call(ctx, viewer.skel_remove_pivot, name)


def skeleton_remove_subtree(ctx: Any, name: str) -> None:
    """Behind a confirm naming how many bones go with it."""
    from ... import dialogs

    viewer = viewer_of(ctx)
    if viewer is None or viewer.editor.mode != "skeleton":
        return
    count = viewer.subtree_size(name)

    def proceed() -> None:
        _skeleton_call(ctx, viewer.skel_remove_subtree, name)

    if count <= 1:
        proceed()
        return
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Delete this limb?",
            message=f"{name!r} and {count - 1} bone(s) beneath it will be removed.",
            confirm_label="Delete",
            on_confirm=proceed,
        )
    )


def skeleton_rename(ctx: Any, old: str, new: str) -> None:
    if old == new:
        return
    viewer = viewer_of(ctx)
    if viewer is None or viewer.editor.mode != "skeleton":
        return
    _skeleton_call(ctx, viewer.skel_rename, old, new)


def skeleton_attach_limb(ctx: Any, preset_key: str, parent: str, side: str, mirror: bool) -> None:
    viewer = viewer_of(ctx)
    if viewer is None or viewer.editor.mode != "skeleton":
        return
    _skeleton_call(ctx, viewer.skel_attach_limb, preset_key, parent, side or None, mirror)


def limb_preset_rows(ctx: Any) -> list[dict[str, Any]]:
    """The shipped limb presets, cached for the life of the session -- a
    read-only, job-independent catalogue, like ``rig_templates``' own."""
    from ....service import rig as svc_rig

    state = ensure(ctx)
    if state.limb_presets_cache is None:
        state.limb_presets_cache = svc_rig.limb_presets()
    return state.limb_presets_cache


def pump_rerig(ctx: Any) -> None:
    """Notice every queued re-rig reaching a terminal status, every frame.

    Called from ``poser_library.draw`` beside :func:`pump`, its own per-frame
    heartbeat. ``svc_rig.create_rig`` only enqueues the rig job -- the actual
    Blender solve and the ``rig.glb`` write happen minutes later, out of
    process, on the serial queue's own schedule -- so nothing about the
    *submit* landing (:func:`on_task_done`) can tell whether the new rig
    exists yet. This is what does: a couple of dict lookups against
    ``ctx.job``, which is already kept live every frame for every other mode
    (the same cache ``ctx.cache.tick`` refreshes in ``main.py``).

    Iterates every tracked re-rig, not just one (poser-05, the 2026-09-11
    audit): re-rigging asset A and then, before that job lands, re-rigging
    asset B must not drop A's own tracking, so both are polled here and each
    is retired independently.
    """
    state = ensure(ctx)
    if not state.rerig_jobs:
        return
    # A snapshot: landing a re-rig runs guard(), which can call _land_rerig
    # synchronously, and a session could in principle queue another re-rig
    # from inside that callback -- iterating the live dict while it is
    # mutated would skip or repeat an entry.
    for source, rig_job_id in list(state.rerig_jobs.items()):
        job = ctx.job(rig_job_id)
        if job is None:
            # Not yet in the loaded window, or a stale id from a session that
            # has since moved on -- either way there is nothing to act on for
            # this one this frame; it stays tracked and is checked again next.
            continue
        status = job.get("status")
        if status == "done":
            state.rerig_jobs.pop(source, None)
            if state.job_id == source:
                # The 2026-09-08 audit's poser-01: rerig()'s own guard protects
                # only the moment the re-rig is *submitted*, minutes before this
                # fires -- an ordinary thing to do while a Blender job
                # serialises on the queue is to keep posing the old rig in the
                # meantime. Unguarded, _land_rerig's exit_pose_mode()/clear()
                # discarded that edit with no confirm and no toast the instant
                # the job landed. Routed through the same guard() every other
                # destructive door here already uses; when there is nothing
                # unsaved it proceeds immediately, same as before.
                guard(ctx, "land this re-rig", lambda: _land_rerig(ctx))
        elif status in ("error", "cancelled"):
            # The generic job-transition toast (``main.py``'s ``_refresh``)
            # already says why; nothing here is worth watching any further.
            state.rerig_jobs.pop(source, None)


def _land_rerig(ctx: Any) -> None:
    """A queued re-rig has actually finished: rebind onto the new rig.glb.

    **The sharp edge this exists for.** The new rig writes over the *same*
    ``rig.glb`` in the *same* job directory ``state.job_id`` already names --
    a rig belongs to its source mesh, not to the rig job that produced it
    (``_q_rig.py``'s own docstring). ``sync_asset`` short-circuits on
    ``viewer.pose_mode and viewer.pose_job_id == job_id``, both still true
    after a same-job re-rig, so without this a re-rig would be a silent
    no-op: the button would appear to work and the viewport would never
    change. Defeating that short-circuit is ``open_asset``'s own proceed
    path, reused rather than restated -- exit pose mode, clear the viewer, and
    re-read the rig -- with the poser-01 template reset folded in for a rig
    that landed under a different skeleton than the one being browsed.
    """
    from ....service import rig as svc_rig

    state = ensure(ctx)
    job_id = state.job_id
    if not job_id:
        return
    rig = None
    with contextlib.suppress(Exception):
        rig = svc_rig.get_rig(ctx.svc, job_id)
    template = str((rig or {}).get("template") or "") or state.template
    if template != state.template:
        _reset_for_template(state, template)
    state.asset_rig = rig
    state.asset_error = ""
    # A fresh rig ends whatever skeleton-editing session was open on the old
    # one -- the draft it was editing described bones that may no longer
    # exist, and :func:`apply_skeleton` is what queued this landing in the
    # first place, so there is nothing left to be "editing".
    state.skeleton_editing = False
    state.skeleton_error = None
    # The 2026-09-16 audit, finding poser-03: see open_asset's own comment --
    # the new rig may not even have a bone by this name.
    state.skeleton_rename = ""
    state.skeleton_rename_for = None
    if rig is not None and rig.get("skeleton") == "custom":
        # P8's own promise: a re-rig lands with no dialog and no confirm, so
        # the fact that the mesh is now on a hand-edited skeleton (rather than
        # the template it started from) has to surface here, once, or it is
        # never said at all. ``weighting_reason`` is set only on the envelope
        # fallback (``blender_worker._rig_meta``'s own "None on the automatic
        # path" rule), so its mere presence is the signal.
        message = f"Custom skeleton from {template}"
        reason = rig.get("weighting_reason")
        if reason:
            message += f" -- {reason}"
        ctx.toast(message, "info")
    viewer = viewer_of(ctx)
    if viewer is not None:
        # Whatever the viewer is showing is the *old* rig's pose session;
        # sync_asset binds the new one once the viewport next draws.
        viewer.exit_pose_mode()
        viewer.clear()
    refresh(ctx)
    clips_refresh(ctx)
    refresh_asset_poses(ctx)


# --- the preview -------------------------------------------------------------


def preview_bounds(template_key: str) -> tuple[list[float], list[float]]:
    """The glTF-space box to frame the preview with, computed host-side.

    Over every fitted bone's head *and tail* -- leaf tails have no nodes, so a
    joint-node box would clip the skull -- from the same unit-box fit
    ``op_armature`` builds, converted Blender -> glTF per component
    ((x, z, -y), the ``m3.blender_delta_to_gltf`` mapping) and padded ~10%.
    Pure, so the framing is assertable with no GL and no file.
    """
    # Function-level, the module's own rule: nothing under this import may pull
    # in a GL context, and math3d is only wanted by this one function.
    from ....kernels.geom3d import math3d as m3

    template = templates.get_template(template_key)
    fitted = skeleton.fit_template(template, poselib.UNIT_LO, poselib.UNIT_HI)
    # Converted first, then boxed. The hand-coded corner swap this replaces was
    # mathematically the same thing -- min/max commute with a signed axis
    # permutation -- but it restated the mapping, which is exactly what
    # ``blender_delta_to_gltf`` exists to be the only copy of. Its
    # delta-not-absolute caveat does not apply: these are freshly computed fits
    # in the unit box, not offsets against some other frame.
    points = [
        m3.blender_delta_to_gltf(p) for bone in fitted for p in (bone["head"], bone["tail"])
    ]
    glo = [min(float(p[i]) for p in points) for i in range(3)]
    ghi = [max(float(p[i]) for p in points) for i in range(3)]
    pad = 0.10 * max(b - a for a, b in zip(glo, ghi, strict=True))
    return [v - pad for v in glo], [v + pad for v in ghi]


def sync_preview(ctx: Any, viewer: Any) -> bool:
    """Bind the built preview to the Poser viewer if it is not already shown.

    Frame thread only for the upload half; the parse is off it (the 2026-09-11
    audit, finding create-04). Called from ``_poser_viewport``'s draw on
    *every* frame Poser is open with no asset bound, including the frame a
    preview build lands -- ``on_task_done``'s ``PREVIEW_KEY_PREFIX`` branch,
    which writes ``state.preview_path`` off a Blender subprocess with no click
    behind the landing. That is exactly ``main.App._sync_viewer``'s own case
    ("it fires on a timer, on the frame a job finishes"), so this follows the
    same split: ``ctx.submit`` the parse, adopt on the frame it lands, checked
    against ``viewer.pending`` so a build the user has since switched away
    from can never land on top of whatever the viewport now wants (point 4 of
    the fix: the freshness check lives in :func:`_land_preview_load`, not
    here).

    What decides is ``viewer.path`` against the landed answer -- never a
    remembered flag, the Review lesson -- and a preview built for a template
    the user has switched away from never binds. -> whether the viewer is
    showing the preview.
    """
    state = ensure(ctx)
    path = state.preview_path
    if path is None or state.preview_template != state.template:
        return viewer.path is not None
    wanted = Path(path)
    if viewer.path == wanted:
        return True
    if viewer.pending == wanted:
        # Already dispatched; :func:`on_task_done` adopts it when it lands.
        return False
    viewer.pending = wanted
    tag = (state.template, wanted)
    if not ctx.submit(PREVIEW_LOAD_KEY, viewer.parse_model, wanted, tag=tag):
        # The key is refused while another parse is still in flight (a rapid
        # template switch, most likely) -- retried next frame once it frees.
        viewer.pending = None
    return False


def _land_preview_load(ctx: Any, done: Any) -> None:
    """The frame-thread half of :func:`sync_preview`'s automatic bind."""
    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if viewer is None or not isinstance(done.tag, tuple) or len(done.tag) != 2:
        return
    template, wanted = done.tag
    if viewer.pending != wanted or state.template != template or state.preview_template != template:
        # Left this template, switched to another, or the build itself was
        # superseded before the parse landed -- see sync_preview's docstring.
        return
    viewer.pending = None
    try:
        viewer.adopt_model(done.result, wanted)
    except Exception:
        log.exception("could not open the %s pose preview", template)
        state.preview_path = None
        state.error = "Could not open the skeleton preview."
        return
    bind_preview(ctx, viewer, template)


def bind_preview(ctx: Any, viewer: Any, template_key: str) -> None:
    """Enter the authoring session over whatever the viewer just loaded."""
    template = templates.get_template(template_key)
    bones = [b["name"] for b in template.bones]
    viewer.enter_pose_authoring(
        bones, [list(p) for p in template.mirror_pairs], token(template.key)
    )
    viewer.editor.root = template.root
    lo, hi = preview_bounds(template_key)
    viewer.frame_bounds(lo, hi)


def _bind_asset_now(ctx: Any, state: PoserState, viewer: Any, job_id: str) -> None:
    """Load ``job_id``'s rig.glb and enter pose mode, right now, both halves.

    The click-driven half of binding an asset (the 2026-09-11 audit, finding
    create-04) -- called only from :func:`open_asset`'s own proceed, which
    runs from nowhere but that door's press or the confirm answering it.
    Exactly the case ``viewer_embed.Viewer.load_model``'s own docstring
    sanctions ("the wait is the point... because the user just pressed
    something"), the same precedent ``pose_panel._enter`` stands on. A load
    with no click behind it -- a queued re-rig landing while the user is doing
    nothing in particular -- goes through :func:`sync_asset`'s own split
    instead; see its docstring.
    """
    rig_path = ctx.job_dir(job_id) / "rig.glb"
    try:
        viewer.load_model(rig_path)
    except Exception:
        log.exception("could not open the rig for job %s", job_id)
        state.asset_error = "Could not open the rig."
        return
    if not viewer.enter_pose_mode(state.asset_rig, job_id):
        state.asset_error = "That GLB carries no skeleton."
        return
    viewer.frame()


def sync_asset(ctx: Any, viewer: Any) -> bool:
    """Bind the viewer to the session's asset if it is not already shown.

    Frame thread only for the upload half; the parse is off it (the 2026-09-11
    audit, finding create-04). Called from ``_poser_viewport``'s draw on
    *every* frame Poser is open with an asset bound -- not only the frame
    :func:`open_asset` was pressed on (which already bound synchronously,
    right there, and is caught by the short-circuit below before this
    function does anything) but also the frame a queued re-rig lands
    (:func:`_land_rerig`) while the user is doing nothing in particular. That
    automatic arrival is what this dispatches for: ``main.App._sync_viewer``'s
    parse-on-a-task/adopt-on-this-frame split, landed by
    :func:`_land_asset_load` and checked there against ``viewer.pending`` so a
    parse that lands after the session has moved to a different asset (or
    closed this one) can never adopt.

    A failure is remembered in ``state.asset_error`` rather than retried every
    frame -- :func:`retry_asset` is what asks again. -> whether the viewer is
    showing the bound asset.
    """
    state = ensure(ctx)
    job_id = state.job_id
    if not job_id:
        return False
    if viewer.pose_mode and viewer.pose_job_id == job_id:
        return True
    if state.asset_error:
        return False
    rig_path = ctx.job_dir(job_id) / "rig.glb"
    if viewer.pending == rig_path:
        # Already dispatched; :func:`_land_asset_load` adopts it when it lands.
        return False
    viewer.pending = rig_path
    tag = (job_id, rig_path)
    if not ctx.submit(ASSET_LOAD_KEY, viewer.parse_model, rig_path, tag=tag):
        # The key is refused while another parse is still in flight (a rapid
        # asset switch, most likely) -- retried next frame once it frees.
        viewer.pending = None
    return False


def _land_asset_load(ctx: Any, done: Any) -> None:
    """The frame-thread half of :func:`sync_asset`'s automatic bind."""
    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if viewer is None or not isinstance(done.tag, tuple) or len(done.tag) != 2:
        return
    job_id, rig_path = done.tag
    if viewer.pending != rig_path or state.job_id != job_id:
        # Left this asset, bound a different one, or closed the session
        # before the parse landed -- see sync_asset's own docstring.
        return
    viewer.pending = None
    try:
        viewer.adopt_model(done.result, rig_path)
    except Exception:
        log.exception("could not open the rig for job %s", job_id)
        state.asset_error = "Could not open the rig."
        return
    if not viewer.enter_pose_mode(state.asset_rig, job_id):
        state.asset_error = "That GLB carries no skeleton."
        return
    viewer.frame()


def reframe(ctx: Any) -> None:
    """Put the subject back on screen. ``F``, and the button beside it.

    An asset session frames the real mesh (``Viewer.frame``, off
    ``Model.bounds()``); the template session frames ``frame_bounds`` on the
    template's own box instead, because a Poser armature has no mesh and
    framing a zero-size model would put the camera in a point.
    """

    viewer = viewer_of(ctx)
    state = ensure(ctx)
    if viewer is None:
        return
    if state.job_id:
        viewer.frame()
        return
    if not state.template:
        return
    lo, hi = preview_bounds(state.template)
    viewer.frame_bounds(lo, hi)


# --- applying ----------------------------------------------------------------


def apply_pose(ctx: Any, pose_id: str) -> None:
    """Load a library pose into the editor, behind the guard -- it overwrites
    whatever is being authored."""
    state = ensure(ctx)
    record = state.find(pose_id)
    viewer = viewer_of(ctx)
    if record is None or viewer is None or not viewer.pose_mode:
        return
    if _refuse_while_skeleton_editing(ctx, state):
        return

    def proceed() -> None:
        # Reset first, the apply_preset order: set_pose writes only the bones
        # the record lists, so a partial record (pre-completeness saves exist
        # on disk) applied over a posed editor would keep stale rotations --
        # and get_pose() would then save them as authored.
        viewer.reset_all(dirty=False)
        viewer.set_pose(record.get("bones") or {}, pose_id=record["id"], dirty=False)
        viewer.set_root_translation(
            record.get("root_translation") or [0.0, 0.0, 0.0], dirty=False
        )

    guard(ctx, "apply a saved pose", proceed)


def apply_preset(ctx: Any, preset: dict[str, Any]) -> None:
    """Load a shipped preset, behind the guard. Presets are read-only;
    apply-then-Save-as is the promotion path into the library."""
    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return
    if _refuse_while_skeleton_editing(ctx, state):
        return
    guard(ctx, "apply a preset", lambda: viewer.apply_preset(preset))


def new_pose(ctx: Any) -> None:
    """Back to rest with nothing being edited, behind the guard."""
    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return
    if _refuse_while_skeleton_editing(ctx, state):
        return
    guard(ctx, "start a new pose", lambda: viewer.reset_all(dirty=False))


def _refuse_while_skeleton_editing(ctx: Any, state: PoserState) -> bool:
    """Refuse New pose and the three Apply buttons while a skeleton edit is
    open. -> whether the call was refused.

    The 2026-09-14 audit's poser-03: these four doors stayed live during a
    skeleton edit and reposed the mesh the skeleton editor still assumes is
    at rest (``enter_skeleton_edit`` resets the armature to rest on the way
    in, and nothing brings it back until :func:`apply_skeleton` lands or
    :func:`cancel_skeleton_edit` gives up). ``import_clip``'s own refusal
    (P6, 2026-09-13) is the precedent: a disabled button only stops a mouse,
    so the door itself has to say no for whatever else can reach it -- a
    keyboard shortcut, an agent's own call.
    """
    if not state.skeleton_editing:
        return False
    ctx.toast("Apply or cancel the skeleton edit before changing the pose.", "info")
    return True


# --- saving ------------------------------------------------------------------


def _payload(ctx: Any, state: PoserState, viewer: Any, name: str) -> dict[str, Any]:
    """A shared-library save, trimmed to the template's own bones.

    P4 (2026-09-13): an asset session's editor can carry a custom skeleton's
    bones (a grafted limb, a renamed pivot) that the *shared* template does
    not have -- ``poselib.validate_record`` refuses any bone name outside the
    template's own list, by name (``poses.validate_bones``' "unknown bone"),
    so contributing such a pose to the library has to drop them rather than
    fail outright: the whole point of the shared library is a pose every
    asset on this *template* can apply, and a custom bone has no template
    meaning to apply there. Silent dropping would be its own defect (a save
    that quietly threw away part of what was authored), so a toast names the
    count.
    """
    bones = viewer.get_pose()
    known = {b["name"] for b in templates.get_template(state.template).bones}
    extra = sorted(b for b in bones if b not in known)
    if extra:
        bones = {name_: quat for name_, quat in bones.items() if name_ in known}
        ctx.toast(
            f"{len(extra)} custom bone{'s' if len(extra) != 1 else ''} not on "
            f"the {state.template} skeleton were left out of the saved pose.",
            "info",
        )
    return {
        "name": name,
        "template": state.template,
        "bones": bones,
        "root_translation": viewer.editor.root_translation(),
    }


def _mutate(ctx: Any, key: str, fn: Any, *args: Any) -> bool:
    """Submit a library write, saying so when the runner refuses it.

    A refusal means the previous write on this key is still in flight.
    Dropping the click strands nothing -- dirty clears only on landing -- but
    it also answers nothing, and a press that does nothing reads as a broken
    button, so the refusal becomes a sentence rather than silence.
    """
    if ctx.submit(key, fn, *args):
        return True
    ctx.toast("Still working on the previous pose-library change.", "info")
    return False


def active(ctx: Any) -> Any:
    """The pose being edited, or None. The palette's door into this mode.

    Poser has no *document* -- a pose is a library record rather than a file
    the user places -- so what is returned is the viewer, which is the thing
    Save and Save-as act on. That is enough for the palette's dispatch, which
    only ever asks "is there something to save".
    """
    viewer = viewer_of(ctx)
    return viewer if viewer is not None and viewer.pose_mode else None


def document_label(ctx: Any) -> tuple[str, bool] | None:
    """``(name, dirty)`` for the status bar, or None with no pose on screen.

    The name is the library record's, ``save``'s own lookup; a pose not yet
    saved anywhere is ``Untitled`` like every other mode's new document. The
    flag is ``AppState.pose_dirty``, the mirror ``Viewer.on_pose_dirty`` keeps
    for exactly this -- an indicator visible from outside the pose pane.
    """
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return None
    state = ensure(ctx)
    if state.job_id:
        record = state.find_asset_pose(viewer.editor.current)
        name = str((record or {}).get("name") or "") or state.asset_label or "Untitled"
    else:
        record = state.find(viewer.editor.current)
        name = str((record or {}).get("name") or "") or "Untitled"
    return name, bool(getattr(ctx.state, "pose_dirty", False))


def save(ctx: Any, tab: Any = None) -> None:
    """Save over the pose being edited, or fall through to Save-as.

    ``tab`` is accepted and ignored: the palette's dispatch hands every
    document mode its active tab, and Poser's "tab" is the viewer it already
    reads off the ctx. One signature there beats a special case.
    """
    from ....service import poses as svc_poses

    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return
    if viewer.editor.mode == "skeleton":
        ctx.toast("Apply or cancel the skeleton edit before saving a pose.", "info")
        return
    existing = viewer.editor.current
    record = state.find(existing)
    if record is None:
        save_as(ctx)
        return
    # Dirty is cleared when the save *lands* (on_task_done), never at submit:
    # a failed write must leave the guard standing in front of the exits.
    _mutate(
        ctx,
        SAVE_KEY,
        svc_poses.update_library_pose,
        ctx.svc,
        existing,
        _payload(ctx, state, viewer, str(record.get("name") or "")),
    )


def save_as(ctx: Any, tab: Any = None) -> None:
    from ....service import poses as svc_poses

    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return
    if viewer.editor.mode == "skeleton":
        ctx.toast("Apply or cancel the skeleton edit before saving a pose.", "info")
        return

    def accept(name: str) -> None:
        payload = _payload(ctx, state, viewer, name)
        _mutate(ctx, SAVE_KEY, svc_poses.create_library_pose, ctx.svc, payload)

    ctx.prompts.ask(dialogs.Prompt(title="Name this pose", label="Name", on_accept=accept))


def rename(ctx: Any, pose_id: str) -> None:
    from ....service import poses as svc_poses

    state = ensure(ctx)
    record = state.find(pose_id)
    if record is None:
        return

    def accept(name: str) -> None:
        _mutate(ctx, RENAME_KEY, svc_poses.rename_library_pose, ctx.svc, pose_id, name)

    ctx.prompts.ask(
        dialogs.Prompt(
            title="Rename this pose",
            label="Name",
            value=str(record.get("name") or ""),
            on_accept=accept,
        )
    )


def duplicate(ctx: Any, pose_id: str) -> None:
    from ....service import poses as svc_poses

    _mutate(ctx, DUPLICATE_KEY, svc_poses.duplicate_library_pose, ctx.svc, pose_id)


def delete(ctx: Any, pose_id: str) -> None:
    """Behind a confirm: the library has no trash, so this one is genuinely
    irreversible -- the paths that are keep their question."""
    from ....service import poses as svc_poses

    state = ensure(ctx)
    record = state.find(pose_id)
    name = str((record or {}).get("name") or "this pose")

    def proceed() -> None:
        # The id rides in the key so the completion knows which pose died --
        # the result of a delete is only {"ok": True}.
        _mutate(ctx, f"{DELETE_KEY}:{pose_id}", svc_poses.delete_library_pose, ctx.svc, pose_id)

    ctx.confirms.ask(
        dialogs.Confirm(
            title="Delete this pose?",
            message=f'"{name}" will be removed from the library. '
            "Assets it was applied to keep their snapshots.",
            on_confirm=proceed,
        )
    )


# --- the guard ---------------------------------------------------------------


def _dirty_draft_noun(viewer: Any) -> str:
    """Which draft the confirm should name, or "pose" with nothing bound.

    The 2026-09-14 audit's poser-04: ``PoseEditor.has_unsaved_edits()`` folds
    a posed armature (``dirty``/``moved``) and an open skeleton draft
    (``draft_dirty``) into one flag, and :func:`guard` said "Unsaved pose
    changes" for both -- so closing or opening an asset with only a skeleton
    draft dirty warned about discarding a "pose" that was not, in fact,
    posed differently from what is saved. Named after whichever draft is
    actually dirty, both when both are (in practice ``enter_skeleton_mode``
    resets the pose to rest on the way in, so the two are not normally dirty
    together, but the wording should not lie if that ever changes).
    """
    if viewer is None or not viewer.pose_mode:
        return "pose"
    editor = viewer.editor
    pose_dirty = bool(getattr(editor, "dirty", False) or getattr(editor, "moved", False))
    skeleton_dirty = bool(getattr(editor, "draft_dirty", False))
    if pose_dirty and skeleton_dirty:
        return "pose and skeleton"
    if skeleton_dirty:
        return "skeleton"
    return "pose"


def guard(ctx: Any, verb: str, proceed: Any) -> bool:
    """Ask before discarding unsaved Poser edits. -> whether it went ahead now.

    Asks only about the *Poser* viewer's editor, which is what makes it
    mutually exclusive with ``pose_panel.guard`` by construction: that one
    reads the shared viewer, this one reads ``ctx.poser_viewer``, and no edit
    can live in both. Leaving the mode needs no guard at all -- the session
    survives on its own viewer, like an open Inker document -- so this runs
    only on quit and on destructive in-mode actions.
    """
    from ... import docmodes

    viewer = viewer_of(ctx)
    return docmodes.viewer_guard(ctx, viewer, _dirty_draft_noun(viewer), verb, proceed)


# --- keys and task results ---------------------------------------------------


def handle_key(ctx: Any, event: Any) -> bool:
    """Poser's shortcuts. -> whether the key was consumed.

    Esc deselects the joint and Ctrl+Z/Ctrl+Shift+Z/Ctrl+Y walk the pose
    history; nothing else is bound -- the mode is otherwise mouse-shaped. The
    caller returns unconditionally either way, the workspace-mode rule.

    The undo binding is shared with the inspector's asset pose mode through
    ``docmodes.pose_undo_key``, because it is one editor with two doors.

    **P9 (2026-09-18):** with the sheet section on screen (``state.
    sheet_view``), the keyboard belongs to the sprite transport instead --
    Troupe's own ``handle_key``, now :func:`sheet_handle_key`. Checked first
    and unconditionally: the pose viewer's own keys below all require
    ``viewer.pose_mode``, which a sheet-viewing session need not have (an
    asset can be bound and shown as a sheet with no pose edit in progress).
    """
    if ensure(ctx).sheet_view:
        return sheet_handle_key(ctx, event)

    import pygame

    from ... import docmodes

    if event.type != pygame.KEYDOWN:
        return False
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return False
    if docmodes.pose_undo_key(viewer, event):
        return True
    if event.key == pygame.K_ESCAPE and viewer.editor.selected is not None:
        viewer.editor.selected = None
        return True
    # **The view keys Clay already has**, on the same three numbers, with the
    # same Shift-is-the-opposite rule and **the same Ctrl** -- plus F to
    # reframe. Poser had none of them: the only way to look at a pose from
    # the side was to orbit there by hand, and there was no way back to a
    # model that had been orbited off screen at all. Dispatched through
    # ``clay_mode.axis_view_key`` rather than restated, so the two viewports
    # cannot come to disagree about which number is the front. Until
    # 2026-09-05 this copy tested the bare digit: 1, 3, 7 and 5 snapped the
    # camera here and did nothing in Clay without Ctrl.
    from ..clay import mode as clay_mode

    name = pygame.key.name(event.key).lower()
    ctrl = bool(event.mod & pygame.KMOD_CTRL)
    shift = bool(event.mod & pygame.KMOD_SHIFT)
    if ctrl and clay_mode.axis_view_key(viewer.camera, name, shift):
        return True
    if name == "f" and not ctrl:
        reframe(ctx)
        return True
    # Ctrl+S / Ctrl+Shift+S, the one chord a user carries between every editor
    # in this app. Both functions existed and neither had a key: saving a pose
    # was a button in one pane and nothing else. Shift always means the shared
    # library, even in an asset session -- the deliberate "contribute this
    # back" route -- while bare Ctrl+S saves onto the bound asset when one is
    # open, matching the button poser_controls draws as the primary action.
    if ctrl and name == "s":
        if event.mod & pygame.KMOD_SHIFT:
            save_as(ctx)
        elif ensure(ctx).job_id:
            save_pose_to_asset(ctx)
        else:
            save(ctx)
        return True
    return False


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from the app for every ``poser-`` key, and, since P9
    (2026-09-18), every ``troupe-`` one too -- Troupe's own task keys, kept
    exactly as they were (see the character-sheet section's own note), routed
    here now that the mode reading their results is this one.
    """
    state = ensure(ctx)
    key = done.key
    if key.startswith("troupe-atlas:"):
        _adopt_atlas(ctx, done)
        return
    if key.startswith("troupe-qa:"):
        # Adopted only if it is still the sheet on screen, and with no toast:
        # a score is a thing to look at, not news.
        if key == scores_key(state.job_id, state.sheet_id):
            ctx.state.preview["troupe_scores"] = getattr(done, "result", None)
            ctx.state.preview["troupe_scores:key"] = (state.job_id, state.sheet_id)
        return
    if key.startswith("troupe-export:"):
        # **Both names, because the pair is the deliverable.** A toast saying
        # "Exported the sheet" leaves the user to discover for themselves
        # whether the JSON came too, which is the one thing that makes the
        # folder importable. ``None`` is the cancelled picker: no news.
        written = getattr(done, "result", None)
        if isinstance(written, dict):
            png = Path(str(written.get("png") or "")).name
            sidecar = Path(str(written.get("json") or "")).name
            ctx.toast(
                f"Exported {png} and {sidecar} to {written.get('dir') or ''}.",
                "success",
            )
        return
    if key.startswith("troupe-frames:"):
        folder = getattr(done, "result", None)
        if folder:
            ctx.toast(f"Frames exported to {folder}.", "success")
        return
    if key == "troupe-start" or key.startswith(("troupe-sheet:", "troupe-send:")):
        # Every one of these queues or finishes a row the sheet section reads,
        # so the throttled copies are dropped rather than waited out -- the
        # interval is there to stop idle polling, not to delay news the panes
        # already have.
        invalidate_sheets(ctx)
        invalidate_riggable(ctx)
        result = getattr(done, "result", None)
        if key == "troupe-start":
            # The form's own choice: this fires on the way out of the submit,
            # and the row it queued is not readable here yet.
            pose = str((state.sheet_form or {}).get("pose") or "")
            ctx.toast(
                f"Drawing the {POSE_LABELS.get(pose, 'pose')} reference. "
                "Approve it in Create to build the mesh.",
                "success",
            )
            # Only if the user is still standing where they pressed the
            # button -- Troupe's own rule for this branch. This fires when
            # the *submit* returns, which can be seconds later and in another
            # mode entirely, and a mode switch nobody asked for takes the
            # window away from whatever they moved on to.
            if ctx.state.mode == "poser":
                from ...state import set_mode

                set_mode(ctx.state, "create")
        elif key.startswith("troupe-send:"):
            # Two shapes behind one press, and the toast says which happened:
            # an unrigged mesh is minutes of CPU behind a button that is not
            # called "Rig", and a user who is not told that will think it
            # hung. No mode switch -- see ``render_character_sheet``.
            rigged = isinstance(result, dict) and result.get("rigged") is not False
            ctx.toast(
                "Rendering the character sheet. Watch it in Poser."
                if rigged
                else "Rigging the mesh, then rendering the character sheet. Watch it in Poser.",
                "success",
            )
        else:
            ctx.toast("Queued the configured character sheet; give it a few minutes.", "info")
        return
    if key == LIST_KEY:
        state.loading = False
        if isinstance(done.result, dict) and done.result.get("template") == state.template:
            state.poses = list(done.result.get("poses") or ())
            state.presets = list(done.result.get("presets") or ())
        # A save landing while this list was in flight set refresh_dirty and
        # could submit nothing; the landing is the moment the key is free.
        pump(ctx)
        return
    if key == ASSET_LOAD_KEY:
        _land_asset_load(ctx, done)
        return
    if key == PREVIEW_LOAD_KEY:
        _land_preview_load(ctx, done)
        return
    if key.startswith(PREVIEW_KEY_PREFIX):
        template = key[len(PREVIEW_KEY_PREFIX):]
        if template != state.template:
            # The 2026-09-08 audit (poser-03): state.building/state.error are
            # single, un-scoped fields, so a landing for a template the user
            # has since switched away from used to overwrite them regardless
            # -- showing the old template's failure over the new template's
            # still-loading viewport, with Try again silently doing nothing
            # until the real build eventually landed. CLIPS_KEY's landing
            # below already checks this; this is the same check for its
            # sibling.
            return
        state.building = False
        if done.result is not None:
            state.preview_path = Path(done.result)
            state.preview_template = template
        return
    if key == SAVE_KEY:
        viewer = viewer_of(ctx)
        if viewer is not None and viewer.pose_mode and isinstance(done.result, dict):
            # Only now is the pose on disk -- the pose_panel _save rule: a
            # failed write leaves dirty set and the guard standing.
            viewer.editor.dirty = False
            viewer.editor.current = done.result.get("id")
        refresh(ctx)
        return
    if key.startswith(DELETE_KEY):
        deleted = key.partition(":")[2]
        viewer = viewer_of(ctx)
        if viewer is not None and viewer.editor.current == deleted:
            # The record Save would have written to is gone; the edits stay in
            # the editor, and Save now falls through to Save-as.
            viewer.editor.current = None
        refresh(ctx)
        return
    if key in (RENAME_KEY, DUPLICATE_KEY):
        refresh(ctx)
        return
    if key in (CLIPS_KEY, CLIPS_SAVE_KEY):
        # Both land the same way: the door hands back the library as it now is
        # on disk, and that becomes the working copy. Which is also what clears
        # ``clips_unsaved`` -- on the *landing*, never at submit, so a refused
        # write leaves the editor holding the edits that were refused.
        state.clips_loading = False
        if isinstance(done.result, dict) and done.result.get("template") == state.template:
            if (
                done.key == CLIPS_SAVE_KEY
                and state.clips_unsaved
                and state.clips_touch_serial != state.clips_save_serial
            ):
                # Edited while the save was in flight: what landed is behind
                # the working copy, so the working copy stays and stays
                # unsaved. The next Save writes it.
                return
            adopt_clips(ctx, done.result)
        return
    if key == CLIP_IMPORT_KEY:
        if not isinstance(done.result, dict):
            # The user cancelled the file picker -- ``dialogs.open_file``'s
            # own contract: None and nothing else. Nothing to say about it.
            return
        result = done.result
        if result.get("template") != state.template:
            # Left this template before the Blender sample landed --
            # CLIPS_KEY's own landing makes the identical check for the
            # identical reason, a few lines above.
            return
        adopt_imported_clips(ctx, result)
        count = len(result.get("clips") or ())
        file_name = str(result.get("source_name") or "the file")
        # The 2026-09-18 audit, finding poser-01: set on every landing, clean
        # or not, so a later clean import does not leave a stale skip list
        # behind for ``poser_clips._import_report`` to keep showing.
        skipped = list(result.get("skipped") or ())
        state.clip_import_skipped = skipped
        if count == 0 and skipped:
            # Every action was over ``op_clip_sample``'s frame limit --
            # "Imported 0 clip(s)" used to say nothing about why. Name the
            # first reason here; "Import report" (``poser_clips._import_report``)
            # lists every one of them once opened.
            first = skipped[0]
            more = f" and {len(skipped) - 1} more" if len(skipped) > 1 else ""
            ctx.toast(
                f"Nothing imported from {file_name} -- every action was skipped: "
                f"{first}{more}",
                "warn",
            )
        else:
            ctx.toast(
                f"Imported {count} clip(s) from {file_name} — Save clips to keep them",
                "success",
            )
        return
    if key.startswith(ASSET_POSES_KEY_PREFIX):
        job_id = key[len(ASSET_POSES_KEY_PREFIX):]
        state.asset_poses_loading.discard(job_id)
        if job_id == state.job_id and isinstance(done.result, dict):
            state.asset_poses = list(done.result.get("poses") or ())
        return
    if key.startswith(ASSET_SAVE_KEY_PREFIX):
        job_id = key[len(ASSET_SAVE_KEY_PREFIX):]
        viewer = viewer_of(ctx)
        if (
            viewer is not None
            and viewer.pose_mode
            and viewer.pose_job_id == job_id
            and isinstance(done.result, dict)
        ):
            # Only now is the pose actually on disk -- the pose_panel _save
            # rule: a failed write leaves dirty set and the guard standing.
            viewer.editor.dirty = False
            viewer.editor.current = done.result.get("id")
        if job_id == state.job_id:
            refresh_asset_poses(ctx)
        return
    if key.startswith(FRONT_KEY_PREFIX):
        # From the result, never from what was sent -- ``set_front_yaw``
        # normalises into [0, 360), and a failed write must not leave this
        # session believing an angle it never actually wrote. Also lands here
        # for a press from ``panes/overlay.py``'s copy of this control, on
        # whatever asset that toolbar has open; the job-id check below is what
        # keeps that from ever touching a *different* asset's session.
        job_id = key[len(FRONT_KEY_PREFIX):]
        if job_id == state.job_id and isinstance(done.result, dict):
            state.asset_front_yaw = float(done.result.get("front_yaw") or 0.0)
        # Unconditionally, and *outside* the job-id check above. The front is a
        # job param, and every other reader of it -- ``panes/overlay.py``'s own
        # label for this control, the Send to Troupe dialog's helper line --
        # reads the row out of ``ctx.cache`` rather than out of this session.
        # Without this, a press from the viewport toolbar wrote the front and
        # then went on drawing "Set front" until something unrelated happened
        # to dirty the cache, which reads as the button having done nothing.
        ctx.cache.invalidate()
        return
    if key.startswith(ASSET_DELETE_KEY_PREFIX):
        job_id, _, deleted = key[len(ASSET_DELETE_KEY_PREFIX):].partition(":")
        viewer = viewer_of(ctx)
        if viewer is not None and viewer.pose_job_id == job_id and viewer.editor.current == deleted:
            viewer.editor.current = None
        if job_id == state.job_id:
            refresh_asset_poses(ctx)
        return
    if key.startswith(ASSET_RERIG_KEY_PREFIX):
        # Only the queue job's id and which asset it belongs to -- the rig
        # itself is not on disk yet. :func:`pump_rerig` is what notices the
        # actual write, once the queue gets around to it. Keyed by source
        # asset (poser-05, the 2026-09-11 audit), so a second re-rig landing
        # here before the first one's job finishes adds an entry rather than
        # overwriting the first's.
        if isinstance(done.result, dict) and done.result.get("id"):
            source = str(done.result.get("source_job") or "")
            if source:
                state.rerig_jobs[source] = str(done.result["id"])
        return


def on_task_failed(ctx: Any, done: Any) -> None:
    """Flags only: the generic failure path has already toasted the service's
    own message, which for a save names the duplicate or the bad field.

    Since P9 (2026-09-18), also the landing for every ``troupe-`` task key --
    see :func:`on_task_done`'s own note.
    """
    state = ensure(ctx)
    if str(done.key).startswith("troupe-atlas:"):
        # Logged as well as marked: an unreadable atlas leaves nothing in
        # realmspinner.log otherwise, and a blank preview is not a diagnosis.
        ctx.state.preview["troupe_texture:failed"] = getattr(done, "tag", None)
        log.warning("could not read the character sheet: %s", getattr(done, "error", ""))
        return
    if str(done.key).startswith("troupe-qa:"):
        # A sheet that cannot be scored is a log line, not a refusal: the
        # preview still plays it. Latched so it is asked once.
        if done.key == scores_key(state.job_id, state.sheet_id):
            ctx.state.preview["troupe_scores:failed"] = (state.job_id, state.sheet_id)
        log.warning("could not score the character sheet: %s", getattr(done, "error", ""))
        return
    if done.key == "troupe-start" or str(done.key).startswith(("troupe-sheet:", "troupe-send:")):
        invalidate_sheets(ctx)
        invalidate_riggable(ctx)
        ctx.toast(str(getattr(done, "error", "") or "That request was refused."), "error")
        return
    if done.key == LIST_KEY:
        # ``loading`` gates the refresh; leaving it set makes the mode inert.
        state.loading = False
        # A refresh wanted while the failed list was in flight is still wanted.
        pump(ctx)
        return
    if done.key == ASSET_LOAD_KEY:
        viewer = viewer_of(ctx)
        tag_ok = isinstance(done.tag, tuple) and len(done.tag) == 2
        job_id, rig_path = done.tag if tag_ok else (None, None)
        if viewer is not None and viewer.pending == rig_path:
            viewer.pending = None
        if job_id == state.job_id:
            # A stale failure -- the session has since closed this asset or
            # bound a different one -- must not paint today's session with
            # yesterday's error (poser-03's rule, applied to this door too).
            state.asset_error = "Could not open the rig."
        return
    if done.key == PREVIEW_LOAD_KEY:
        viewer = viewer_of(ctx)
        tag_ok = isinstance(done.tag, tuple) and len(done.tag) == 2
        template, wanted = done.tag if tag_ok else (None, None)
        if viewer is not None and viewer.pending == wanted:
            viewer.pending = None
        if template == state.template:
            state.preview_path = None
            state.error = "Could not open the skeleton preview."
        return
    if done.key.startswith(PREVIEW_KEY_PREFIX):
        template = done.key[len(PREVIEW_KEY_PREFIX):]
        if template != state.template:
            # poser-03's guard, mirrored on the failure landing: see the
            # matching comment on the success side in on_task_done.
            return
        state.building = False
        # What the viewport's empty state shows under the placeholder, so a
        # broken Blender is a sentence on screen rather than a toast that
        # scrolled away.
        state.error = str(getattr(done, "message", "") or "Could not build the pose preview.")
        return
    if done.key in (CLIPS_KEY, CLIPS_SAVE_KEY):
        # ``clips_loading`` gates the read, exactly as ``loading`` does above;
        # ``clips_unsaved`` is deliberately *not* cleared, which is the whole
        # point of clearing it on the landing instead: a refused save leaves the
        # editor holding what it refused, with the reason already toasted by the
        # generic failure path.
        state.clips_loading = False
        clips_pump(ctx)
        return
    if done.key.startswith(ASSET_POSES_KEY_PREFIX):
        state.asset_poses_loading.discard(done.key[len(ASSET_POSES_KEY_PREFIX):])
        return
    if done.key.startswith(ASSET_RERIG_KEY_PREFIX):
        # Recorded only while a skeleton draft is what was being submitted
        # (:func:`apply_skeleton`) -- an ordinary template :func:`rerig` can
        # fail too (Blender missing, the job gone), but that failure has
        # nowhere in the skeleton pane to be shown under, and the generic
        # failure toast above has already said it.
        if state.skeleton_editing:
            field = getattr(done.error, "field", None)
            state.skeleton_error = {
                "field": str(field) if isinstance(field, str) else "",
                "message": str(getattr(done, "message", "") or done.error or ""),
            }
        return


# --- the character sheet (P9, 2026-09-18) ------------------------------------
#
# Troupe folded into Poser as a stage: rig, then clips, then a sheet, one
# workspace. Everything below is Troupe's own ``mode.py``, minus the
# cross-character cast list and the in-mode "existing mesh" picker -- both
# superseded by :func:`open_asset`/:func:`riggable_assets` above, per decision
# 1 of the folding brief: a sheet is reachable only once an asset is bound,
# through the same door that binds it to pose. The job kind, ``_q_troupe.py``
# and ``service/troupe.py`` are unchanged; every ``troupe-`` task-key prefix
# below is kept exactly as it was, because renaming a UI-side key with no
# corresponding module rename underneath is pure churn.
#
# **Nothing here is journal-tracked**, Troupe's own rule carried whole: a
# sheet is a selection over files a worker already published, not something a
# crash can cost the user beyond which frame the preview was on. See the
# ``sheet_*`` fields' own note on :class:`PoserState`.


def _release_sheet_caches(ctx: Any, state: PoserState) -> None:
    """Drop every throttled read and cached texture the sheet session holds.

    Called whenever the *bound asset* changes (:func:`open_asset`,
    :func:`close_asset`) as well as whenever the *selected sheet* does
    (:func:`select_sheet`) -- a cache keyed on the wrong asset is as stale as
    one keyed on the wrong sheet.
    """
    state.sheets_cache = None
    state.sheet_cache = None
    state.pixel_report_cache = None
    release_texture(ctx)
    release_scores(ctx)
    release_rerender_selection(ctx)


#: What each reference pose is called in prose -- Troupe's ``POSE_LABELS``,
#: kept whole: the sidebar, the toast and the form all name the same thing,
#: and three spellings of it is three chances to drift.
POSE_LABELS = {"tpose": "T-pose", "apose": "A-pose"}


def _pose_label(row: Any) -> str:
    """What to call the pose a job row was drawn against.

    Falls back to the T-pose, and deliberately not to the door's default: a
    row queued before the pose was a choice was drawn against that guide, and
    ``_q_generate`` redraws it against the same one.
    """
    params = (row or {}).get("params") or {}
    return POSE_LABELS.get(str(params.get("guide_pose") or "tpose"), "reference")


def can_render_sheet(ctx: Any, job: Any) -> bool:
    """Whether "Send to..." belongs on this job's row -- Troupe's own
    ``can_send_to_troupe``, renamed now that the mode it was named for is
    gone. Still the predicate :mod:`.asset_exits`' kept sheet-rendering door
    asks, for any finished mesh, rigged or not.

    From the cached row alone -- no filesystem call, because a toolbar asks
    this every frame. Deliberately does not require a rig: an unrigged mesh is
    exactly what the door is for, and it mints the rig itself.
    """
    del ctx
    return bool(
        job
        and job.get("stage") == "model"
        and job.get("status") == "done"
        and not job.get("deleted_at")
        and "model.glb" in (job.get("files") or [])
    )


def sheets(ctx: Any, job_id: str) -> list[dict[str, Any]]:
    """The character sheets in one job's directory, newest first.

    Only the character sheets: a mesh can also hold ordinary pose sheets, and
    they have no ``animation`` block, no direction runs and nothing this
    section can play. Filtered on the block rather than on the row that made
    it, because the *artifact* is what the preview reads.

    Throttled like every other cast-adjacent read in this module, keyed on
    ``job_id`` as well as timed so switching the bound asset reads
    immediately; :func:`invalidate_sheets` closes the gap a build would
    otherwise leave.
    """
    state = ensure(ctx)
    if not job_id:
        return []
    now = time.monotonic()
    if state.sheets_cache is None or state.sheets_key != job_id or now >= state.sheets_next:
        state.sheets_cache = _read_sheets(ctx, job_id)
        state.sheets_key = job_id
        state.sheets_next = now + SHEETS_REFRESH
    return state.sheets_cache


def invalidate_sheets(ctx: Any) -> None:
    """Drop the throttled sheet reads so the next draw re-reads the directory."""
    state = ensure(ctx)
    state.sheets_cache = None
    state.sheet_cache = None


def _read_sheets(ctx: Any, job_id: str) -> list[dict[str, Any]]:
    """The uncached read :func:`sheets` throttles."""
    from ....kernels.rig import store as rig_store

    out = [
        record
        for record in rig_store.list_sheets(ctx.job_dir(job_id))
        if (record.get("animation") or {}).get("tags")
    ]
    out.sort(key=lambda r: float(r.get("created") or 0.0), reverse=True)
    return out


def open_character_sheet(ctx: Any, job_id: str, sheet_id: str = "") -> bool:
    """Enter Poser pointed at one character sheet. Troupe's own ``open_sheet``,
    adapted to Poser's asset-bound model. **The one door in from elsewhere**
    -- a "Show" toast on a finished charsheet row, or a reopened one
    (:mod:`.asset_open`).

    Returns False, and does not switch modes or bind the asset, when the
    character has no playable sheet at all: the alternative is arriving at
    "No character on screen", which is the blank arrival this door exists to
    stop. Once a sheet is known to exist, the bind itself is best-effort --
    ``pose_panel.open_in_poser``'s own precedent: a template switch with
    unsaved clip edits can still defer behind its own confirm, and this still
    switches to Poser rather than leaving the press looking like it did
    nothing.
    """
    if not job_id or not sheets(ctx, job_id):
        return False
    from ...state import set_mode

    job = None
    cache = getattr(ctx, "cache", None)
    if cache is not None:
        job = cache.get(job_id)
    open_asset(ctx, job or {"id": job_id}, sheet_id=sheet_id)
    set_mode(ctx.state, "poser")
    return True


def select_sheet(ctx: Any, sheet_id: str = "") -> None:
    """Point the sheet section at one of the bound asset's sheets.

    Troupe's own ``select``, minus the job half: which asset is bound is
    :func:`open_asset`'s question now, not this function's. The clock is reset
    here rather than left to run -- carried across a selection it would show
    the new sheet mid-stride at whatever frame the old one happened to be on,
    which reads as a rendering fault rather than as a preview that kept
    playing.
    """
    state = ensure(ctx)
    if not state.job_id:
        return
    # The throttled directory read is dropped rather than waited out, for the
    # reason ``on_task_done`` drops it below: the interval exists to stop idle
    # polling, not to delay news the user has just asked for by name.
    invalidate_sheets(ctx)
    available = sheets(ctx, state.job_id)
    if sheet_id and any(r["id"] == sheet_id for r in available):
        state.sheet_id = sheet_id
    else:
        state.sheet_id = available[0]["id"] if available else ""
    state.sheet_clock = 0.0
    state.sheet_frame = 0
    state.sheet_view = True
    _reconcile_sheet_preview(ctx)
    release_texture(ctx)
    release_scores(ctx)
    release_rerender_selection(ctx)


def active_sheet(ctx: Any) -> dict[str, Any] | None:
    """The selected sheet's sidecar, or None. Throttled like :func:`sheets`.

    Several panes ask for this in their draw, so it was several JSON reads a
    frame of a file that changes only when a sheet is rebuilt.
    """
    from ....kernels.rig import store as rig_store

    state = ensure(ctx)
    if not (state.job_id and state.sheet_id):
        return None
    key = (state.job_id, state.sheet_id)
    now = time.monotonic()
    if (
        state.sheet_cache is None
        or state.sheet_cache_key != key
        or now >= state.sheet_cache_next
    ):
        state.sheet_cache = rig_store.read_sheet(ctx.job_dir(key[0]), key[1])
        state.sheet_cache_key = key
        state.sheet_cache_next = now + SHEETS_REFRESH
    return state.sheet_cache


def preview_layout(ctx: Any) -> dict[str, Any]:
    """The active sheet's immutable layout, with a pre-v2 legacy fallback."""
    from .engine import spec as sheet_spec

    record = active_sheet(ctx) or {}
    snapshot = record.get("troupe")
    if isinstance(snapshot, dict):
        try:
            movements = snapshot.get("movements") or ()
            runs = snapshot.get("runs") or ()
            valid = bool(movements and runs) and all(
                str(m.get("key") or "")
                and int(m.get("frames") or 0) > 0
                and bool(m.get("directions"))
                for m in movements
            ) and all(
                str(run.get("movement") or "")
                and str(run.get("direction") or "")
                and 0 <= int(run.get("start")) <= int(run.get("end"))
                for run in runs
            )
        except (AttributeError, TypeError, ValueError):
            valid = False
        if valid:
            return snapshot
        # Named, not just swallowed. An empty layout draws "That animation and
        # direction are not on this sheet", which reads like a selection
        # problem and sends the user looking at the selectors; the sidecar is
        # the actual answer and nothing else was going to say so.
        #
        # **Once per sheet**, latched the way the scorer latches a failure:
        # this function is called several times a frame by the panes, so a
        # bare warning here would be sixty lines a second in ``realmspinner.log``.
        sheet_id = str(record.get("id") or record.get("sheet_id") or "?")
        if ctx.state.preview.get("troupe_layout:warned") != sheet_id:
            ctx.state.preview["troupe_layout:warned"] = sheet_id
            log.warning(
                "the layout snapshot on sheet %s is not readable; "
                "the preview will be empty",
                sheet_id,
            )
        return {"version": 2, "movements": [], "runs": [], "cell_count": 0}
    table = sheet_spec.load()
    movements = [
        {
            "key": animation.name,
            "label": animation.name.title(),
            "frames": animation.frames,
            "loop": animation.loop,
            "duration_ms": animation.duration_ms,
            "directions": [
                {"key": direction.name, "yaw": direction.yaw}
                for direction in table.directions
            ],
        }
        for animation in table.animations
    ]
    runs = [
        {
            "movement": animation,
            "direction": direction,
            "start": start,
            "end": end,
        }
        for animation, direction, start, end, _loop in table.spans()
    ]
    return {"version": 1, "movements": movements, "runs": runs, "cell_count": 256}


def preview_movement(ctx: Any, name: str | None = None) -> dict[str, Any] | None:
    wanted = name or ensure(ctx).sheet_animation
    return next(
        (m for m in preview_layout(ctx).get("movements") or () if m.get("key") == wanted),
        None,
    )


def _reconcile_sheet_preview(ctx: Any) -> None:
    state = ensure(ctx)
    movements = preview_layout(ctx).get("movements") or ()
    if not movements:
        return
    movement = next(
        (m for m in movements if m.get("key") == state.sheet_animation), movements[0]
    )
    state.sheet_animation = str(movement.get("key") or "")
    directions = movement.get("directions") or ()
    keys = [str(d.get("key") or "") for d in directions]
    if keys and state.sheet_direction not in keys:
        state.sheet_direction = keys[0]
    state.sheet_frame = min(state.sheet_frame, max(int(movement.get("frames") or 1) - 1, 0))


# --- the clock ----------------------------------------------------------


def sheet_advance(ctx: Any, dt: float) -> None:
    """Move the sheet preview on by ``dt`` seconds of wall clock.

    A ``while`` rather than an ``if``: a frame that took longer than one
    sprite frame -- a job finishing, a texture upload, the window being
    dragged -- must skip cells rather than fall behind and never catch up.
    Bounded by the run's own length, so a pathological stall costs at most one
    lap.
    """
    state = ensure(ctx)
    if not state.sheet_playing:
        return
    animation = preview_movement(ctx)
    if animation is None:
        return
    duration = int(animation.get("duration_ms") or 100)
    frames = int(animation.get("frames") or 1)
    interval = max(duration, 1) / 1000.0 / max(state.sheet_speed, 0.01)
    state.sheet_clock += max(dt, 0.0)
    laps = 0
    while state.sheet_clock >= interval and laps <= frames:
        state.sheet_clock -= interval
        state.sheet_frame += 1
        laps += 1
    if animation.get("loop"):
        state.sheet_frame %= frames
    else:
        # A one-shot holds its last frame -- see ``kernels.sheet.
        # interpolate_clip``'s extra landing frame. Held rather than looped
        # *and* rather than stopped: a preview that stops needs a control to
        # start it again, and the point of the mode is that a bad frame is
        # obvious without pressing anything.
        state.sheet_frame = min(state.sheet_frame, frames - 1)


def cell_index(ctx: Any) -> int | None:
    """Which cell of the atlas the preview is showing.

    Through ``spec.cells()`` rather than arithmetic over the animation
    lengths: that table is the studio's copy of the frame table and
    ``tests/modes/poser/test_sheet_geometry_agreement.py`` is the sole owner
    of its agreement with the pipeline's. A second piece of arithmetic here
    would be a third copy nothing owns.
    """
    state = ensure(ctx)
    for run in preview_layout(ctx).get("runs") or ():
        if (
            run.get("movement") == state.sheet_animation
            and run.get("direction") == state.sheet_direction
        ):
            start, end = int(run.get("start") or 0), int(run.get("end") or 0)
            index = start + state.sheet_frame
            return index if start <= index <= end else None
    return None


def set_sheet_animation(ctx: Any, name: str) -> None:
    state = ensure(ctx)
    if name == state.sheet_animation:
        return
    state.sheet_animation = name
    state.sheet_clock = 0.0
    state.sheet_frame = 0
    _reconcile_sheet_preview(ctx)


def set_sheet_direction(ctx: Any, name: str) -> None:
    state = ensure(ctx)
    state.sheet_direction = name
    movement = preview_movement(ctx)
    state.sheet_frame = min(state.sheet_frame, max(int((movement or {}).get("frames") or 1) - 1, 0))


def sheet_step(ctx: Any, delta: int) -> None:
    """Nudge one frame, and stop playing -- stepping implies looking."""
    state = ensure(ctx)
    state.sheet_playing = False
    frames = int((preview_movement(ctx) or {}).get("frames") or 1)
    state.sheet_frame = (state.sheet_frame + delta) % frames
    state.sheet_clock = 0.0


def _cycle(names: list[str], current: str, delta: int) -> str | None:
    """The next name round the ring, or None when there is nothing to cycle.

    None rather than a default on an empty list, because an invalid v2
    snapshot resolves to a layout with no movements (:func:`preview_layout`)
    and a key press must not invent a direction the sheet does not have.
    """
    if not names:
        return None
    try:
        index = names.index(current)
    except ValueError:
        # The selection is off this sheet -- ``_reconcile_sheet_preview``'s
        # case, reached here when a key arrives first. Start from the top.
        return names[0]
    return names[(index + delta) % len(names)]


def cycle_sheet_direction(ctx: Any, delta: int) -> None:
    """Turn the character one direction round the compass.

    Through :func:`set_sheet_direction`, which deliberately does *not* reset
    the clock -- turning mid-stride shows the same frame from the other side.
    """
    state = ensure(ctx)
    names = [
        str(entry.get("key") or "")
        for entry in (preview_movement(ctx) or {}).get("directions") or ()
    ]
    picked = _cycle(names, state.sheet_direction, delta)
    if picked is not None:
        set_sheet_direction(ctx, picked)


def cycle_sheet_animation(ctx: Any, delta: int) -> None:
    """Move to the next animation on the sheet.

    Through :func:`set_sheet_animation`, which *does* reset the clock and
    reconciles the direction -- a movement need not carry the one currently
    selected.
    """
    state = ensure(ctx)
    names = [
        str(movement.get("key") or "")
        for movement in preview_layout(ctx).get("movements") or ()
    ]
    picked = _cycle(names, state.sheet_animation, delta)
    if picked is not None:
        set_sheet_animation(ctx, picked)


def sheet_to_end(ctx: Any, last: bool) -> None:
    """Jump to the first or last frame of the run on screen, and stop."""
    state = ensure(ctx)
    frames = int((preview_movement(ctx) or {}).get("frames") or 1)
    sheet_goto(ctx, state.sheet_direction, frames - 1 if last else 0)


# --- the scores -----------------------------------------------------------
#
# ``poser.engine.qa`` over the selected sheet, run through the task runner
# when the selection changes and read back by the preview's scorecard. Never
# in the frame loop -- a 256-cell atlas is a few hundred milliseconds of
# numpy, which is a stall in the one pane whose whole point is smooth
# playback. The scores rank; nothing reads them to refuse anything.


def cell_geometry(record: Mapping[str, Any] | None) -> tuple[int, int, int] | None:
    """``(columns, frame_w, frame_h)`` of a sidecar, or None if it does not say."""
    if not record:
        return None
    columns = int(record.get("columns") or 8)
    size = int(record.get("frame_size") or 0)
    width = size if size > 0 else int(record.get("frame_w") or 0)
    height = size if size > 0 else int(record.get("frame_h") or width)
    if columns < 1 or width < 1 or height < 1:
        return None
    return columns, width, height


def pivot_of(
    record: Mapping[str, Any] | None, index: int | None
) -> tuple[float, float] | None:
    """One cell's pivot in **cell pixels**, or ``None`` if the sidecar has none.

    **``None`` rather than a centre.** A marker drawn at a guessed origin is a
    lie about where the engine will place the sprite, and the user cannot tell
    it apart from a measured one.
    """
    if not record or index is None:
        return None
    for entry in record.get("cells") or ():
        if not isinstance(entry, Mapping):
            continue
        try:
            if int(entry.get("index")) != int(index):
                continue
        except (TypeError, ValueError):
            continue
        x, y = entry.get("pivot_x"), entry.get("pivot_y")
        if x is None or y is None:
            return None
        try:
            return float(x), float(y)
        except (TypeError, ValueError):
            return None
    return None


def scores_key(job_id: str, sheet_id: str) -> str:
    return f"troupe-qa:{job_id}:{sheet_id}"


def _score_task(path: Path, layout: dict[str, Any], geometry: tuple[int, int, int]) -> Any:
    """The task-thread half: read the PNG, score it. No GL, no state."""
    import numpy as np
    from PIL import Image

    from .engine import qa

    with Image.open(path) as opened:
        opened.load()
        atlas = np.asarray(opened.convert("RGBA"))
    columns, frame_w, frame_h = geometry
    return qa.score_sheet(atlas, layout, columns=columns, frame_w=frame_w, frame_h=frame_h)


def release_scores(ctx: Any) -> None:
    for name in ("troupe_scores", "troupe_scores:key", "troupe_scores:failed"):
        ctx.state.preview.pop(name, None)


#: Where ``poser.ui.panes.sheet``'s "Re-render some runs" checkbox keeps its
#: ticks. Owned here rather than by that pane, because :func:`select_sheet` is
#: what has to clear it (the 2026-09-08 audit, troupe-02): ticks made on one
#: sheet must not reappear pre-checked on the next one whose runs happen to
#: share the same animation/direction vocabulary.
RERENDER_SLOT = "troupe_rerender_runs"


def release_rerender_selection(ctx: Any) -> None:
    """Forget the "Re-render some runs" ticks -- :func:`select_sheet`'s own
    rule, applied to :data:`RERENDER_SLOT`: a set of ticked runs is a fact
    about the sheet on screen, and it must not survive picking a different
    one."""
    ctx.state.preview.pop(RERENDER_SLOT, None)


def scores(ctx: Any) -> Any:
    """The selected sheet's QA score, or None while it is being computed,
    absent or unscorable. Frame thread; cheap."""
    from ....kernels.rig import store as rig_store

    state = ensure(ctx)
    if not (state.job_id and state.sheet_id):
        return None
    key = (state.job_id, state.sheet_id)
    preview = ctx.state.preview
    if preview.get("troupe_scores:key") == key:
        return preview.get("troupe_scores")
    if preview.get("troupe_scores:failed") == key:
        return None
    record = active_sheet(ctx)
    geometry = cell_geometry(record)
    path = rig_store.sheet_png_path(ctx.job_dir(state.job_id), state.sheet_id)
    # is_file(), not exists(): the 2026-09-08 audit's troupe-04 found exists()
    # here, which is also true of a directory and would let a stale or
    # malformed sheet path slip past this refusal only to fail later inside
    # Image.open with no mention of which sheet.
    if geometry is None or not path.is_file():
        preview["troupe_scores:failed"] = key
        return None
    task_key = scores_key(*key)
    if ctx.busy(task_key):
        return None
    ctx.submit(task_key, _score_task, path, dict(preview_layout(ctx)), geometry)
    return None


def scores_failed(ctx: Any) -> bool:
    state = ensure(ctx)
    return ctx.state.preview.get("troupe_scores:failed") == (state.job_id, state.sheet_id)


def sheet_goto(ctx: Any, direction: str, frame: int) -> None:
    """Point the preview at one cell and stop -- a click on the heatmap."""
    state = ensure(ctx)
    set_sheet_direction(ctx, direction)
    frames = int((preview_movement(ctx) or {}).get("frames") or 1)
    state.sheet_frame = max(0, min(int(frame), frames - 1))
    state.sheet_playing = False
    state.sheet_clock = 0.0


# --- the texture ------------------------------------------------------------
#
# One texture for the whole atlas, uploaded once per sheet and drawn as a
# sub-rectangle per frame. The filter is NEAREST, which is the whole point: a
# linear-filtered sprite is the one thing a pixel-art preview must never show.


def release_texture(ctx: Any) -> None:
    """Forget-then-release the cached atlas texture. Also called at teardown."""
    ctx.state.preview.pop("troupe_texture:key", None)
    ctx.state.preview.pop("troupe_texture:failed", None)
    cached = ctx.state.preview.pop("troupe_texture", None)
    if cached is None:
        return
    from ... import imgui_backend

    renderer = imgui_backend.current()
    if renderer is not None:
        renderer.forget_texture(cached)
    cached.release()


def atlas_key(job_id: str, sheet_id: str) -> str:
    return f"troupe-atlas:{job_id}:{sheet_id}"


def _decode_atlas(path: Path) -> tuple[tuple[int, int], bytes]:
    """The task-thread half: the PNG as RGBA bytes. No GL, no state."""
    from PIL import Image

    with Image.open(path) as opened:
        opened.load()
        atlas = opened.convert("RGBA")
        return atlas.size, atlas.tobytes()


def atlas_texture(ctx: Any) -> Any:
    """The selected sheet's atlas, uploaded once. ``None`` when there is none
    -- or not yet: the decode is a task, and the frames it takes show the
    empty state rather than a hitch.

    Keyed on ``(job, sheet)`` rather than on the file's mtime: a published
    sheet is write-once under a fresh id, so a stale texture cannot exist for
    a key that has not changed.
    """
    from ....kernels.rig import store as rig_store

    state = ensure(ctx)
    if ctx.viewer is None or not (state.job_id and state.sheet_id):
        return None
    key = (state.job_id, state.sheet_id)
    preview = ctx.state.preview
    if preview.get("troupe_texture:key") == key:
        return preview.get("troupe_texture")
    # Tried once: a sheet that would not decode is logged, not re-read on
    # every frame (``on_task_failed`` marks it).
    if preview.get("troupe_texture:failed") == key:
        return None
    task = atlas_key(*key)
    if ctx.busy(task):
        return None
    path = rig_store.sheet_png_path(ctx.job_dir(state.job_id), state.sheet_id)
    if not path.is_file():
        return None
    ctx.submit(task, _decode_atlas, path, tag=key)
    return None


def _adopt_atlas(ctx: Any, done: Any) -> None:
    """Upload a decoded atlas, if it is still the sheet on screen."""
    state = ensure(ctx)
    key = getattr(done, "tag", None)
    if key != (state.job_id, state.sheet_id) or ctx.viewer is None:
        return
    size, data = done.result
    release_texture(ctx)
    texture = ctx.viewer.ctx.texture(size, 4, data)
    texture.filter = (ctx.viewer.ctx.NEAREST, ctx.viewer.ctx.NEAREST)
    ctx.state.preview["troupe_texture"] = texture
    ctx.state.preview["troupe_texture:key"] = key


# --- the two doors ----------------------------------------------------------

#: The two Style choices a sheet request may carry. On the form as a string
#: rather than a bare boolean, because a combo needs a value for its *other*
#: state too.
STYLE_PIXEL_ART = "pixel_art"
STYLE_HD = "hd"


def _style_choice(form: Mapping[str, Any]) -> str:
    """Which of the two Style choices a form holds, tolerant of an old one."""
    return str(form.get("style") or STYLE_PIXEL_ART)


def _pixel_style_request(form: Mapping[str, Any]) -> dict[str, Any]:
    """The pixel-art fields a request should carry, gated on Style.

    **Pixel art sends today's request, unchanged.** No ``pixel_art`` key at
    all, so a form that never touches the switch mints the byte-identical row
    it always did.

    **HD sends ``pixel_art: False`` and none of the four fields a pixel-art
    render has.** The door refuses a request that turns ``pixel_art`` off but
    still names a non-empty palette, a true ``dither`` or an outline mode
    other than ``none`` (``service.troupe._check_options``).
    """
    if _style_choice(form) == STYLE_HD:
        return {"pixel_art": False, "reduce_mode": form.get("reduce_mode")}
    return {
        "colors": form.get("colors"),
        "outline": form.get("outline"),
        "reduce_mode": form.get("reduce_mode"),
        "dither": bool(form.get("dither")),
        "palette": form.get("palette") or "",
    }


def _layout_request(form: dict[str, Any]) -> dict[str, Any]:
    """Strip presentation-only flags from the editable sheet form.

    **Version 3 only when the layout actually uses what v3 offers** -- a
    top-level ``fps``, or a movement outside the closed legacy five -- and
    version 2 otherwise, byte-identical to what this function produced before
    the open vocabulary existed.
    """
    from ....kernels import charsheet

    source = form.get("layout") or {}
    movements = [
        {
            "key": row.get("key"),
            "frames": row.get("frames"),
            "directions": row.get("directions"),
        }
        for row in source.get("movements") or ()
        if row.get("enabled", True)
    ]
    legacy_names = {name for name, *_rest in charsheet.ANIMATIONS}
    non_legacy = any(str(m.get("key")) not in legacy_names for m in movements)
    fps = form.get("fps")
    request: dict[str, Any] = {
        "version": 3 if (fps is not None or non_legacy) else 2,
        "columns": int(source.get("columns") or 8),
        "movements": movements,
    }
    if fps is not None:
        request["fps"] = int(fps)
    return request


def camera_elevation(form: Mapping[str, Any]) -> float | None:
    """The elevation a form's camera choice means, or None if it makes none.

    An explicit ``elevation`` on the form still wins, and that is deliberate:
    an angle off the preset ladder has to stay expressible.
    """
    explicit = form.get("elevation")
    if explicit not in (None, ""):
        return float(explicit)
    key = str(form.get("camera") or "")
    if not key:
        return None
    from ....kernels import charsheet

    return next(
        (angle for preset, _label, angle in charsheet.CAMERA_PRESETS if preset == key),
        None,
    )


#: Where :func:`sheet_options` caches the door's answer. On ``state.preview``
#: rather than on ``PoserState``: it is neither a selection nor a clock.
OPTIONS_SLOT = "troupe_options"


def sheet_options(ctx: Any) -> dict[str, Any]:
    """The door's own answer about what a sheet request may ask for.

    Cached on the frame state rather than called per draw: it walks the
    palette directory, and a directory walk sixty times a second is a cost
    with no reader. Keyed on ``stamps.stamp_ns`` of the palette directory, the
    same rule ``panes.inspector.palette_names`` already uses for the
    identical directory.
    """
    from ....service import troupe as svc_troupe
    from ...panes import stamps

    key = stamps.stamp_ns(ctx.svc.config.palette_dir)
    cached = ctx.state.preview.get(OPTIONS_SLOT)
    if cached is not None and cached[0] == key:
        return cached[1]
    value = svc_troupe.troupe_options(ctx.svc)
    # After the read, never beside it: that ordering is what makes the stored
    # stamp's tick provably older than the read it describes.
    if stamps.storable(key):
        ctx.state.preview[OPTIONS_SLOT] = (key, value)
    return value


def sheet_form(ctx: Any) -> dict[str, Any]:
    """The new-character/build-a-sheet request, kept on the mode's own state.

    Public, because Create's Character arm offers "Draw it in Poser" as the
    escape route from a species this program does not model, and that route
    has to put the brief into *this* form -- the one the pane will draw when
    the mode opens. A second construction of the same dict there and here is
    two defaults for one request.
    """
    defaults = sheet_options(ctx).get("defaults") or {}
    state = ensure(ctx)
    if not state.sheet_form:
        state.sheet_form = {
            "prompt": "",
            "variant": str(defaults.get("variant") or "male"),
            "pose": str(defaults.get("pose") or "apose"),
            "logical_size": int(defaults.get("logical_size") or 32),
            "colors": int(defaults.get("colors") or 64),
            "outline": str(defaults.get("outline") or "outer"),
            "reduce_mode": str(defaults.get("reduce_mode") or "box"),
            "camera": str(defaults.get("camera") or ""),
            "template": str(defaults.get("template") or ""),
            "dither": False,
            "palette": "",
            "name": "",
            "layout": _default_sheet_layout(ctx),
            "style": (
                STYLE_PIXEL_ART if defaults.get("pixel_art", True) else STYLE_HD
            ),
            "fps": None,
        }
    elif "layout" not in state.sheet_form:
        # Session-state migration for a form created by a pre-v2 build.
        state.sheet_form["layout"] = _default_sheet_layout(ctx)
    elif state.sheet_form["layout"].get("template") != _bound_sheet_template(ctx):
        # The character bound changed rig -- or the form's layout predates the
        # skeleton it was built for being recorded at all. Rows built for one
        # skeleton describe nothing on another, so a changed rig rebuilds the
        # default rows rather than leaving stale ones.
        state.sheet_form["layout"] = _default_sheet_layout(ctx)
    if "template" not in state.sheet_form:
        state.sheet_form["template"] = str(defaults.get("template") or "")
    if "camera" not in state.sheet_form:
        state.sheet_form["camera"] = str(defaults.get("camera") or "")
    if "style" not in state.sheet_form:
        state.sheet_form["style"] = (
            STYLE_PIXEL_ART if defaults.get("pixel_art", True) else STYLE_HD
        )
    if "fps" not in state.sheet_form:
        state.sheet_form["fps"] = None
    return state.sheet_form


def _bound_sheet_template(ctx: Any) -> str:
    """The skeleton the sheet form's movement rows should be built from.

    Troupe's own ``_bound_template`` read the bound character's rig off a
    ``SCAN_LIMIT``-row SQL walk (``_bound_rig_template``), because Troupe held
    no rig of its own to ask. Poser does: once an asset is bound,
    ``PoserState.template`` already *is* that asset's own rig template --
    ``open_asset`` reads and records it at bind time -- so this is a field
    read rather than a second store scan. A fresh "New character" form with
    nothing bound yet has no rig to defer to, which is the only case the
    door's own default is for.
    """
    state = ensure(ctx)
    if state.job_id and state.template:
        return state.template
    return str((sheet_options(ctx).get("defaults") or {}).get("template") or "")


def _layout_for_sheet_template(ctx: Any, template: str) -> dict[str, Any]:
    """The default layout for *template*'s own clip vocabulary.

    Built from ``clip_vocabulary`` -- the rig's *whole* clip library -- rather
    than the closed legacy five, so a movement the vocabulary opened is a row
    on the form from the first frame. Only the clips the vocabulary itself
    marks ``default`` start ticked, which is what keeps a fresh form's request
    byte-identical to the one it built before this vocabulary opened.
    """
    opts = sheet_options(ctx)
    vocabulary = (opts.get("clip_vocabulary") or {}).get(template) or ()
    return {
        "version": 2,
        "columns": 8,
        "template": template,
        "movements": [
            {
                "key": row.get("name"),
                "enabled": bool(row.get("default", False)),
                "frames": int(row.get("frames") or 1),
                "directions": 8,
            }
            for row in vocabulary
        ],
    }


def _default_sheet_layout(ctx: Any) -> dict[str, Any]:
    """The layout a fresh New Character form opens with.

    Timed to the bound character's own skeleton (:func:`_bound_sheet_template`),
    not always the door's default -- a quadruped, bird or blob bound here used
    to get the humanoid vocabulary's names, frames and provisional flags
    regardless of what its own clip library actually holds.
    """
    return _layout_for_sheet_template(ctx, _bound_sheet_template(ctx))


def cell_count(form: dict[str, Any]) -> int:
    """Troupe's own ``settings.cell_count``, moved here beside the request it
    counts: two panes (the settings form and the "Build another sheet" door)
    read it, and it is not presentation."""
    return sum(
        int(row.get("frames") or 0) * int(row.get("directions") or 0)
        for row in (form.get("layout") or {}).get("movements") or ()
        if row.get("enabled", True)
    )


def start_character(ctx: Any, form: dict[str, Any]) -> bool:
    """Submit the pose reference that starts a character.

    The *first* link only. The gate is the point of the shape: this queues one
    cheap image, the user approves it in Create, and only then is the
    reconstruction spent. Which is also why this hands off to Create rather
    than staying here -- the approval lives where every other reference's
    approval lives, and a second promote button would be a second gate.
    """
    from ....service import jobs as svc_jobs

    key = "troupe-start"
    if ctx.busy(key):
        return False
    ctx.state.clear_field_errors()
    return ctx.submit(
        key,
        svc_jobs.create_job,
        ctx.svc,
        kind="text",
        prompt=str(form.get("prompt") or ""),
        output="reference",
        # ``troupe=`` is the service door's own keyword (``_jobs_create``,
        # out of scope for this phase) -- unchanged, because the field it
        # writes onto the row is read by ``service.troupe`` and
        # ``_q_troupe.py``, neither of which this phase touches.
        troupe={
            "variant": form.get("variant"),
            "pose": form.get("pose"),
            "logical_size": form.get("logical_size"),
            "camera": form.get("camera"),
            "elevation": camera_elevation(form),
            "layout": _layout_request(form),
            **_pixel_style_request(form),
        },
    )


def build_sheet(ctx: Any, job_id: str, form: dict[str, Any]) -> bool:
    """Queue another character sheet for a mesh that is already rigged.

    The direct door -- for a supplied base mesh, or for a second sheet at a
    different size from the same character.
    """
    from ....service import troupe as svc_troupe

    key = f"troupe-sheet:{job_id}"
    if ctx.busy(key):
        return False
    ctx.state.clear_field_errors()
    return ctx.submit(
        key,
        svc_troupe.create_charsheet,
        ctx.svc,
        job_id,
        logical_size=form.get("logical_size"),
        elevation=camera_elevation(form),
        layout=_layout_request(form),
        name=str(form.get("name") or ""),
        **_pixel_style_request(form),
    )


def render_character_sheet(ctx: Any, job: Any, form: dict[str, Any] | None = None) -> bool:
    """Take an existing mesh -- rigged or not -- into a character sheet.
    Troupe's own ``send_to_troupe``, renamed now that it is Poser's "render a
    sheet" door rather than a trip into a second mode. -> whether the request
    was submitted.

    **It does not switch modes.** There is nothing to do next, and yanking
    somebody out of the library mid-review to watch a spinner is the opposite
    of the affordance. The toast says where to watch instead.
    """
    from ....service import troupe as svc_troupe

    job_id = str((job or {}).get("id") or "")
    if not job_id:
        return False
    settings = _layout_request(form or {}) if form else {}
    request = dict(form or {})

    def run() -> Any:
        return svc_troupe.send_to_troupe(
            ctx.svc,
            job_id,
            logical_size=request.get("logical_size"),
            elevation=camera_elevation(request),
            lighting=request.get("lighting"),
            name=str(request.get("name") or ""),
            layout=settings or None,
            template=request.get("template") or None,
            **_pixel_style_request(request),
        )

    return bool(ctx.submit(f"troupe-send:{job_id}", run))


def rerender_runs(ctx: Any, subset: list[dict[str, str]]) -> bool:
    """Re-render some runs of the selected sheet. -> whether it was submitted.

    Takes no options: the door copies them from the row that made the sheet,
    so the cells that come back match the ones they land beside.
    """
    from ....service import troupe as svc_troupe

    state = ensure(ctx)
    if not (state.job_id and state.sheet_id and subset):
        return False
    key = f"troupe-sheet:{state.job_id}"
    if ctx.busy(key):
        return False
    ctx.state.clear_field_errors()
    return bool(
        ctx.submit(
            key,
            svc_troupe.rerender_charsheet,
            ctx.svc,
            state.job_id,
            sheet_id=state.sheet_id,
            subset=list(subset),
        )
    )


def sheet_runs(ctx: Any) -> list[dict[str, str]]:
    """Every ``(animation, direction)`` run on the selected sheet."""
    return [
        {"animation": str(run.get("movement") or ""), "direction": str(run.get("direction") or "")}
        for run in preview_layout(ctx).get("runs") or ()
    ]


def open_in_inker(ctx: Any) -> bool:
    """Hand the selected sheet to Inker as an animated document."""
    from ..inker import mode as inker_mode

    state = ensure(ctx)
    if not (state.job_id and state.sheet_id):
        return False
    inker_mode.open_rendered_sheet(ctx, state.job_id, state.sheet_id)
    return True


def add_to_packwright(ctx: Any) -> bool:
    """The other way out -- one sheet's cells into an atlas beside everything
    else being packed."""
    from ..packwright import mode as packwright_mode

    state = ensure(ctx)
    if not (state.job_id and state.sheet_id):
        return False
    packwright_mode.add_rendered_sheet(ctx, state.job_id, state.sheet_id)
    return True


def export_key(job_id: str, sheet_id: str) -> str:
    return f"troupe-export:{job_id}:{sheet_id}"


def export_package(ctx: Any) -> bool:
    """Copy the selected sheet's PNG and its sidecar out together."""
    from ....service import characters as svc_characters

    state = ensure(ctx)
    if not (state.job_id and state.sheet_id):
        return False
    key = export_key(state.job_id, state.sheet_id)
    if ctx.busy(key):
        return False
    job_id, sheet_id = state.job_id, state.sheet_id
    configured = getattr(ctx, "export_dir", None) or None

    def run() -> Any:
        dest = configured
        if dest is None:
            from ... import dialogs as dialogs_mod

            picked = dialogs_mod.select_folder("Export the character sheet")
            if picked is None:
                return None
            dest = picked
        return svc_characters.export_package(ctx.svc, job_id, sheet_id, dest_dir=dest)

    return bool(ctx.submit(key, run))


def frames_key(job_id: str, sheet_id: str) -> str:
    return f"troupe-frames:{job_id}:{sheet_id}"


def export_frames(ctx: Any) -> bool:
    """Cut the selected sheet's atlas into one PNG per frame, folder per clip
    and compass direction, beside a ``manifest.json`` of the frame rates."""
    from ....service import characters as svc_characters

    state = ensure(ctx)
    if not (state.job_id and state.sheet_id):
        return False
    key = frames_key(state.job_id, state.sheet_id)
    if ctx.busy(key):
        return False
    job_id, sheet_id = state.job_id, state.sheet_id
    configured = getattr(ctx, "export_dir", None) or None

    def run() -> Any:
        dest = configured
        if dest is None:
            from ... import dialogs as dialogs_mod

            picked = dialogs_mod.select_folder("Export the character sheet's frames")
            if picked is None:
                return None
            dest = picked
        return svc_characters.export_frames(ctx.svc, job_id, sheet_id, dest_dir=dest)

    return bool(ctx.submit(key, run))


# --- what the sheet says about itself ----------------------------------------
#
# Two different questions, and the panes must not blur them. ``poser.engine.
# qa`` *ranks*: it scores every cell against its neighbours and flags the
# worst, and nothing anywhere refuses a sheet on its account.
# ``pipelines/sheetcheck.py`` *checks structure*: a cell is clipped at the
# frame edge, empty, or was never rendered -- facts about the file.


def validation_of(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """The sheet's ``validation`` block, or an empty one."""
    block = (record or {}).get("validation")
    return dict(block) if isinstance(block, Mapping) else {}


def needs_repair(record: Mapping[str, Any] | None) -> bool:
    """Whether the sheet failed its structural check."""
    block = validation_of(record)
    return bool(block) and block.get("ok") is False


def repair_notes(record: Mapping[str, Any] | None) -> list[str]:
    """``sheetcheck.describe`` over this sheet's block. The pane's diagnostics."""
    from ....pipelines import sheetcheck

    return list(sheetcheck.describe(validation_of(record) or None))


# --- back to Create ----------------------------------------------------------


def recipe_of(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """The recipe the sheet's ``character`` block carries, or an empty dict."""
    block = (record or {}).get("character")
    if not isinstance(block, Mapping):
        return {}
    recipe = block.get("recipe")
    return dict(recipe) if isinstance(recipe, Mapping) else {}


def vary_in_create(ctx: Any, record: Mapping[str, Any] | None) -> bool:
    """Load this sheet's recipe into Create's form and go to the Reference
    stage. ``settings_character.hand_to_poser``'s opposite number: that route
    carries a brief *out* of Create, and this carries a finished character
    back in so the next one can differ by a seed, a colour count or a horn
    length.

    Returns False when the sheet carries no recipe, so the caller can say so
    rather than switching into a form describing somebody else.
    """
    import json

    from ..create.engine import assets as create_assets
    from ..create.engine import character as character_engine
    from ..create.ui import stages as create_stages

    recipe = recipe_of(record)
    if not recipe:
        return False
    form = ctx.state.form_2d
    form["asset_type"] = "character"
    form["generation_type"] = "character"
    body = recipe.get("appearance")
    animations = recipe.get("animations") or {}
    values = {
        "character_family": str(recipe.get("family") or ""),
        "character_theme": str(recipe.get("theme") or character_engine.THEME_UNSET),
        "character_camera": str(recipe.get("camera") or ""),
        "character_actions": ",".join(
            name for name, _frames in character_engine.MOVEMENTS if name in animations
        ),
        "character_pixel": str(int(recipe.get("logical_size") or 64)),
        "character_colors": str(int(recipe.get("colors") or 32)),
        "character_body": json.dumps(
            {str(k): float(v) for k, v in dict(body).items()}, sort_keys=True
        )
        if isinstance(body, Mapping)
        else "{}",
        "character_name": str(recipe.get("name") or ""),
    }
    form.update(values)
    for key in values:
        character_engine.touched(form, key)
    seed = recipe.get("seed")
    if seed is not None:
        form["seed"] = int(seed)
    create_assets.sync_legacy_fields(form)
    create_stages.go(ctx, "reference")
    return True


def sheet_handle_key(ctx: Any, event: Any) -> bool:
    """The sheet preview's transport, at the keyboard. Troupe's own
    ``handle_key`` -- every key here is one imgui never sees, which is why
    Poser joins ``modes.NAV_KEY_MODES`` once a sheet is on screen.

    **Presses only.** A release is not consumed, because nothing downstream
    acts on a bare KEYUP.
    """
    import pygame

    if event.type != pygame.KEYDOWN:
        return False
    state = ensure(ctx)
    if event.key in (pygame.K_c, pygame.K_p):
        if _typing():
            return False
        if event.key == pygame.K_c:
            state.sheet_checker = not state.sheet_checker
        else:
            state.sheet_show_pivot = not state.sheet_show_pivot
        return True
    if event.key == pygame.K_SPACE:
        state.sheet_playing = not state.sheet_playing
        return True
    if event.key == pygame.K_LEFT:
        sheet_step(ctx, -1)
        return True
    if event.key == pygame.K_RIGHT:
        sheet_step(ctx, 1)
        return True
    if event.key == pygame.K_UP:
        cycle_sheet_direction(ctx, -1)
        return True
    if event.key == pygame.K_DOWN:
        cycle_sheet_direction(ctx, 1)
        return True
    if event.key == pygame.K_PAGEUP:
        cycle_sheet_animation(ctx, -1)
        return True
    if event.key == pygame.K_PAGEDOWN:
        cycle_sheet_animation(ctx, 1)
        return True
    if event.key == pygame.K_HOME:
        sheet_to_end(ctx, last=False)
        return True
    if event.key == pygame.K_END:
        sheet_to_end(ctx, last=True)
        return True
    return False


def _typing() -> bool:
    """Whether a text field has the keyboard right now."""
    try:
        from imgui_bundle import imgui
    except ImportError:  # pragma: no cover -- imgui is a studio dependency
        return False
    if imgui.get_current_context() is None:
        return False
    return bool(imgui.get_io().want_text_input)


# --- crash recovery (UX-05) ---------------------------------------------------
#
# A pose is the smallest thing in the app and the easiest to lose: it lives
# entirely in a ``PoseEditor`` until somebody presses Save, so a crash during an
# authoring session took the whole of it. Both entry points are covered by one
# provider because both are one editor -- Poser's own viewer and the
# inspector's shared one -- which is the same argument
# ``docmodes.pose_undo_key`` makes about the keyboard.
#
# **Payload equality is the head.** A pose has no undo serial to compare
# against... except that since A-2 it does, and it is deliberately not used: the
# editor's history is dropped on every rebind, so its head restarts and an
# unchanged pose would be re-encoded after every model adoption. The payload is
# a few dozen floats and comparing it is cheaper than the encode it prevents.



# --- the clip editor ---------------------------------------------------------
#
# The Troupe programme's clip-authoring half. The clip *format* and its
# expansion shipped with Troupe; what did not was any way to change a clip that
# is not editing JSON in the package tree -- and authoring those keyframes is
# the most important art task in the programme.
#
# **The editor holds a whole library as its working copy**, because
# ``service.clips`` will not save less than one -- a clip library is internally
# consistent by construction, and saving a clip alone would let a key rename
# land while another clip still pointed at the old name. Every mutation below
# therefore edits ``state.clips`` in place and sets ``clips_unsaved``; nothing
# touches disk until :func:`save_clips`.
#
# **A key is edited by posing the preview armature.** That is the whole design:
# Poser already has a skeleton, gizmos and a pose editor, so "author a
# keyframe" is "load the key onto the armature, drag joints, put it back". No
# second posing surface, and the poses a clip is made of are the same kind of
# thing the pose library already holds.


def clips_refresh(ctx: Any) -> None:
    """Ask for the clip library to be re-read."""
    state = ensure(ctx)
    state.clips_dirty_flag = True
    clips_pump(ctx)


def clips_pump(ctx: Any) -> None:
    """Submit the wanted clip read. ``pump``'s rule, on its own key.

    Guarded on ``clips_unsaved`` as well as on the flag: a re-read while the
    user has unsaved keys would silently discard them, and a background refresh
    is not a thing anybody asked for. The editor is re-read on arrival and after
    a save, which is when it is safe.
    """
    state = ensure(ctx)
    if not state.clips_dirty_flag or state.clips_loading or not state.template:
        return
    if state.clips_unsaved:
        return
    state.clips_loading = True
    if ctx.submit(CLIPS_KEY, _collect_clips, ctx.svc, state.template):
        state.clips_dirty_flag = False
    else:
        state.clips_loading = False


def _collect_clips(svc: Any, template: str) -> dict[str, Any]:
    from ....service import clips as svc_clips

    return svc_clips.library(svc, template)


def adopt_clips(ctx: Any, library: dict[str, Any]) -> None:
    """Install a freshly read library as the working copy.

    Keeps the open clip *by name* when the new library still carries one of
    that name -- a save-then-reload must not send the user back to the top of
    the list -- and falls back to the first clip otherwise.
    """
    state = ensure(ctx)
    state.clips = dict(library)
    state.clips_loading = False
    state.clips_unsaved = False
    state.clips_error = ""
    names = [str(c.get("name") or "") for c in state.clips.get("clips") or ()]
    if state.clip not in names:
        state.clip = names[0] if names else ""
        state.key_index = 0
    state.frame = -1
    rebuild_frames(ctx)


def rebuild_frames(ctx: Any) -> None:
    """Re-expand the open clip, for the scrubber.

    Through ``sheet.interpolate_clip`` -- the renderer's own interpolator --
    rather than a preview-shaped reimplementation, so what the scrubber shows is
    what the sheet will draw. A clip that cannot be expanded (segments that do
    not match its keys, mid-edit) leaves the frames empty and says why, rather
    than raising into a draw.
    """
    from ....kernels import sheet as sheetlib

    state = ensure(ctx)
    record = state.open_clip()
    state.frames = []
    if record is None:
        return
    try:
        keys = [state.key_pose(name) for name in record.get("keys") or ()]
        if any(k is None for k in keys):
            raise ValueError("a key pose is missing from the library")
        state.frames = sheetlib.interpolate_clip(
            [dict(k) for k in keys],
            [int(n) for n in record.get("segments") or ()],
            closed=bool(record.get("closed")),
            easing=str(record.get("easing") or "linear"),
            space=str(state.clips.get("space") or "node"),
            clip_id=str(record.get("name") or ""),
        )
        state.clips_error = ""
    except Exception as exc:
        # Not a refusal -- the user is mid-edit and the clip is momentarily
        # inconsistent, which is ordinary. The scrubber empties and the reason
        # is shown; Save is what actually refuses.
        state.clips_error = str(exc)


def select_clip(ctx: Any, name: str) -> None:
    state = ensure(ctx)
    if state.clip == name:
        return

    def proceed() -> None:
        state.clip = str(name)
        state.key_index = 0
        state.frame = -1
        rebuild_frames(ctx)
        apply_key(ctx)

    guard(ctx, "open another clip", proceed)


def _viewer_editor(ctx: Any) -> Any:
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return None
    return viewer.editor



# --- rotation space ---------------------------------------------------------
#
# Two frames, and they are not interchangeable (``blender_worker.POSE_SPACES``).
# ``PoseEditor`` is unconditionally *node-local*: parent-relative absolute
# rotations, which is what ``Model.set_rotation`` writes. A clip library may
# declare ``space="delta"`` instead -- each value a rotation from the bone's own
# rest, composed as ``node = rest * delta``. The shipped humanoid library is
# delta and its arms carry a constant -58 degrees off rest, so pushing its
# values through the editor raw contorts the skeleton, and capturing back out
# raw writes node-local values into a delta-declared file. Everything crossing
# the boundary goes through these two.


def _clip_space(state: Any) -> str:
    return "delta" if str(state.clips.get("space") or "node") == "delta" else "node"


def _convert(editor: Any, bones: dict[str, Any], space: str, how: Any) -> dict[str, list[float]]:
    """One direction of the frame conversion, per bone with a rest rotation.

    ``how`` is ``poses.node_from_delta`` or ``poses.delta_from_node``: the
    algebra is *there*, beside the sentence that justifies it, because the
    Blender worker needs the same rule against Blender's rest quaternions and
    the two used to write it out separately.
    """

    out: dict[str, list[float]] = {}
    for name, quat in bones.items():
        value = [float(v) for v in quat]
        if space == "delta":
            rest = editor.rest.get(name) if editor is not None else None
            if rest is not None:
                value = how(rest, value)
        out[name] = value
    return out


def _to_node(editor: Any, bones: dict[str, Any], space: str) -> dict[str, list[float]]:
    """Library rotations -> the editor's node-local frame."""
    return _convert(editor, bones, space, poses.node_from_delta)


def _from_node(editor: Any, bones: dict[str, Any], space: str) -> dict[str, list[float]]:
    """The editor's node-local frame -> library rotations."""
    return _convert(editor, bones, space, poses.delta_from_node)


def apply_key(ctx: Any) -> None:
    """Put the selected key's rotations onto the preview armature.

    Straight onto the editor, not through the pose *library*: a clip's keys are
    not library poses and giving them ids there would put twenty-two rows a
    user never asked for into the list every asset poses from.
    """
    state = ensure(ctx)
    editor = _viewer_editor(ctx)
    record = state.open_clip()
    if editor is None or record is None:
        return
    keys = list(record.get("keys") or ())
    if not 0 <= state.key_index < len(keys):
        return
    pose = state.key_pose(keys[state.key_index])
    if pose is None:
        return
    state.frame = -1
    # ``apply_preset`` and not ``apply``: a key lists only the bones it moves,
    # so the rest have to go back to rest first, or loading "walk contact"
    # after "attack strike" leaves the sword arm up. It is one undo step, which
    # is right for a discrete "go to this key".
    #
    # The 2026-09-23 audit, finding poser-01: ``apply_preset`` is itself
    # undoable and pushes its own step, but the ``set_root_translation`` call
    # below it was not bracketed at all -- so the root offset landed on the
    # live armature with no history entry of its own. Redoing the
    # ``apply_preset`` step then restored the *pre-offset* snapshot it
    # recorded, putting the bones back but leaving the root offset at zero.
    # ``editor.record()`` is re-entrant (create-02 fixed the same gap at two
    # other call sites the same day), so wrapping both in one outer record()
    # folds the whole "select this key" gesture into the single undo step it
    # already reads as.
    with editor.record():
        editor.apply_preset(
            {"bones": _to_node(editor, dict(pose.get("bones") or {}), _clip_space(state))}
        )
        root = pose.get("root_translation")
        if root:
            editor.set_root_translation([float(v) for v in root])
    # Loading a key is not an unsaved edit: the pose it put on the armature is
    # the clip's own, stored in the working copy. ``apply_preset`` marks the
    # editor dirty because in the pose library that is what it means; here
    # the flag has to mean "the armature differs from the selected key", or
    # every guarded door below would ask after every key click.
    editor.dirty = False
    editor.moved.clear()
    sync_onion(ctx)


def select_key(ctx: Any, index: int) -> None:
    """Behind the guard: ``apply_key`` overwrites the armature, and a pose
    the user has been editing on it is not saved anywhere else."""
    state = ensure(ctx)

    def proceed() -> None:
        state.key_index = max(0, int(index))
        apply_key(ctx)

    guard(ctx, "select another key", proceed)


def scrub(ctx: Any, frame: int) -> None:
    """Put an *interpolated* frame onto the armature.

    ``state.frame`` going non-negative is what tells the pane the gizmos are no
    longer meaningful: a scrubbed frame is between two keys and has nowhere to
    store an edit. Judged as motion here and as pixels in the sprite preview --
    the plan's "judge clips as pixels, not as viewport playback" is about the
    *verdict*, and this is the fast loop that gets you close enough to render.
    """
    state = ensure(ctx)
    editor = _viewer_editor(ctx)
    if editor is None or not state.frames:
        return
    if editor.mode == "skeleton":
        # P6 (2026-09-13): the armature is showing a skeleton draft, not a
        # pose -- there is no key to interpolate towards while the skeleton
        # itself is being edited. ``poser_clips``'s scrubber is hidden for the
        # same fact; this is the defence for a stray call while it was.
        ctx.toast("Apply or cancel the skeleton edit before scrubbing.", "info")
        return
    if state.frame < 0 and editor.has_unsaved_edits():
        # The first tick of a scrub replaces the armature's pose, which is the
        # user's unsaved key edit. A confirm per slider tick is unusable, so
        # this refuses in words instead: capture or reset the edit, then
        # scrub. Once a scrub is under way the armature holds nothing unsaved
        # (every frame lands with ``dirty=False``), so later ticks pass.
        ctx.toast("Capture or reset the pose before scrubbing.", "info")
        return
    index = max(0, min(int(frame), len(state.frames) - 1))
    state.frame = index
    # Scrubbing puts the live skeleton on an in-between pose, so the key ghosts
    # stop meaning anything -- see ``sync_onion``.
    sync_onion(ctx)
    pose = state.frames[index]
    # Rest overlaid with the frame's own bones, through the **undecorated**
    # ``apply``. ``apply_preset`` would be the natural call and is exactly
    # wrong here: it is ``@_undoable``, and this runs once per frame of a
    # slider drag -- a step per frame is a stack full of one gesture, which is
    # the rule ``rotate_selected`` and ``move_root`` already follow. Building
    # the full map first is what makes one non-undoable call equivalent to
    # reset-then-apply.
    full = {name: [float(v) for v in quat] for name, quat in editor.rest.items()}
    full.update(_to_node(editor, dict(pose.get("bones") or {}), _clip_space(state)))
    editor.apply(full, pose_id=None, dirty=False)
    editor.set_root_translation(
        [float(v) for v in (pose.get("root_translation") or (0.0, 0.0, 0.0))],
        dirty=False,
    )


def capture_key(ctx: Any) -> None:
    """Write the armature's current pose back into the selected key.

    Refused while scrubbing, by name: the armature is showing an interpolated
    frame then, and storing it into a key would silently replace an authored
    pose with a computed one. Refused rather than "capture into the nearest
    key", which is the same act with the mistake hidden.
    """
    state = ensure(ctx)
    editor = _viewer_editor(ctx)
    if editor is None:
        return
    if editor.mode == "skeleton":
        ctx.toast("Apply or cancel the skeleton edit before capturing a key.", "info")
        return
    record = state.open_clip()
    if record is None:
        return
    if state.frame >= 0:
        ctx.toast(
            "That is an in-between frame, not a key. Pick a key first.", "warn"
        )
        return
    keys = list(record.get("keys") or ())
    if not 0 <= state.key_index < len(keys):
        return
    pose = state.key_pose(keys[state.key_index])
    if pose is None:
        return
    pose["bones"] = _from_node(editor, dict(editor.pose()), _clip_space(state))
    root = editor.root_translation() if editor.root is not None else None
    if root and any(root):
        pose["root_translation"] = [float(v) for v in root]
    else:
        pose.pop("root_translation", None)
    # The 2026-09-15 audit, finding poser-01: every sibling mutator that folds
    # the armature's live edit into stored data (``apply_key`` above is the
    # clearest example) clears ``dirty``/``moved`` once the edit has somewhere
    # safe to live -- this one wrote the pose into the key and left both set,
    # so ``has_unsaved_edits()`` kept reporting the just-saved pose as an
    # unsaved edit: the pending dot stayed lit and the next guarded action
    # (selecting another key, closing the asset) asked to discard a change
    # that was already captured.
    editor.dirty = False
    editor.moved.clear()
    _touch(ctx)


def set_onion(ctx: Any, on: bool) -> None:
    state = ensure(ctx)
    state.onion = bool(on)
    sync_onion(ctx)


def sync_onion(ctx: Any) -> None:
    """Put the neighbouring keys' poses on the viewer, or clear them.

    The **keys** either side of the selected one, not the frames either side of
    the playhead: a keyframe is judged against the keys it steps between, and
    the in-between frames are what the scrubber is for. A closed clip wraps, so
    the first key's "before" is the last one -- which is exactly the comparison
    a looping walk needs and the one that is easiest to get wrong by hand.

    Cleared while scrubbing: the live skeleton is then an in-between pose, and
    ghosts of the neighbouring *keys* around it would be three poses on screen
    with no stated relationship between them.
    """
    viewer = viewer_of(ctx)
    if viewer is None:
        return
    state = ensure(ctx)
    record = state.open_clip()
    if not state.onion or record is None or state.frame >= 0:
        viewer.onion = []
        return
    keys = list(record.get("keys") or ())
    if not keys:
        viewer.onion = []
        return
    closed = bool(record.get("closed"))
    out: list[dict[str, Any]] = []
    for step in (-1, 1):
        index = state.key_index + step
        if closed:
            index %= len(keys)
        elif not 0 <= index < len(keys):
            out.append({})
            continue
        pose = state.key_pose(keys[index])
        out.append(
            _to_node(_viewer_editor(ctx), dict(pose.get("bones") or {}), _clip_space(state))
            if pose
            else {}
        )
    viewer.onion = out


def _touch(ctx: Any) -> None:
    """Mark the working copy changed and re-expand it. Every mutation ends
    here, so neither half can be forgotten at one call site."""
    state = ensure(ctx)
    state.clips_unsaved = True
    state.clips_touch_serial += 1
    rebuild_frames(ctx)


def set_segment(ctx: Any, index: int, frames: int) -> None:
    from ....service import clips as svc_clips

    state = ensure(ctx)
    record = state.open_clip()
    if record is None:
        return
    segments = list(record.get("segments") or ())
    if not 0 <= index < len(segments):
        return
    value = max(svc_clips.MIN_SEGMENT, min(int(frames), svc_clips.MAX_SEGMENT))
    if segments[index] == value:
        return
    segments[index] = value
    record["segments"] = segments
    _touch(ctx)


def set_easing(ctx: Any, easing: str) -> None:
    state = ensure(ctx)
    record = state.open_clip()
    if record is None or record.get("easing") == easing:
        return
    record["easing"] = str(easing)
    _touch(ctx)


def clip_fps(duration_ms: Any) -> float:
    """Frames per second implied by one rendered frame lasting ``duration_ms``.

    The Timing section's own "≈ N fps" hint, factored out so it is
    assertable with no imgui frame and so the pane and a test read the same
    arithmetic -- ``clips.animation_tracks``' own ``step`` is
    ``ANIMATION_FPS * duration_ms / 1000``, and this is that relationship's
    inverse, in frames of *this* clip per second rather than scene frames per
    clip frame. 0.0 for anything that will not divide, rather than raising: a
    mid-adopt clip (the legacy v2 shape ``cliplib.parse_clip_library`` migrates
    away from before this pane ever sees it) should show no hint at all, not a
    crash from a label.
    """
    try:
        ms = float(duration_ms)
    except (TypeError, ValueError):
        return 0.0
    return 1000.0 / ms if ms > 0 else 0.0


def set_duration(ctx: Any, ms: int) -> None:
    """Set the selected clip's per-rendered-frame duration, in milliseconds.

    Snapped rather than refused -- ``set_segment``'s own precedent, applied to
    this field: a typed 83 lands on 80, the nearest multiple of
    ``cliplib.CLIP_DURATION_STEP_MS`` inside ``cliplib.MIN_CLIP_DURATION_MS``-
    ``cliplib.MAX_CLIP_DURATION_MS``, rather than an error toast over one
    keystroke or a value the write door would refuse outright. This is the
    clip's *only* duration_ms door -- ``clips.animation_tracks``' bake step and
    this pane's own fps hint (:func:`clip_fps`) both read the value this
    writes, so a clip's play speed follows it structurally rather than by a
    second copy either could drift from.

    Marks ``clips_unsaved`` and re-expands the working copy exactly like
    :func:`set_easing`/:func:`set_segment` do -- through :func:`_touch` -- and
    for the same reason neither of them pushes an undo step: a clip's timing
    fields are not the pose gizmo's editor history (``viewer.editor.history``,
    ``dev/INVARIANTS.md``'s "undo is addressed by uid" is about *that* stack),
    they are the clip *library*'s working copy, and the library has exactly one
    undo door -- :func:`revert_clips`, which discards every unsaved field at
    once, this one included.
    """
    state = ensure(ctx)
    record = state.open_clip()
    if record is None:
        return
    step = cliplib.CLIP_DURATION_STEP_MS
    snapped = int(round(int(ms) / step)) * step
    snapped = max(cliplib.MIN_CLIP_DURATION_MS, min(snapped, cliplib.MAX_CLIP_DURATION_MS))
    if record.get("duration_ms") == snapped:
        return
    record["duration_ms"] = snapped
    _touch(ctx)


def set_closed(ctx: Any, closed: bool) -> None:
    """Open or close the loop -- which changes how many segments the clip needs.

    An open clip of N keys has N-1 steps and a closed one has N, so the segment
    list is resized here rather than left for the save to refuse. Growing takes
    the last segment's length (a new step most like its neighbour); shrinking
    drops the one that no longer exists. Doing it silently is right *because*
    the count is derived: there is no other value it could take.
    """
    state = ensure(ctx)
    record = state.open_clip()
    if record is None or bool(record.get("closed")) == bool(closed):
        return
    record["closed"] = bool(closed)
    keys = list(record.get("keys") or ())
    segments = list(record.get("segments") or ())
    wanted = len(keys) if closed else max(0, len(keys) - 1)
    while len(segments) < wanted:
        segments.append(segments[-1] if segments else 1)
    del segments[wanted:]
    record["segments"] = segments
    _touch(ctx)


def move_key(ctx: Any, index: int, delta: int) -> None:
    """Reorder one key, carrying the segment that *follows* it.

    A segment is the step out of a key, so moving a key without its segment
    would reorder the poses and leave the timing where it was -- which reads as
    the reorder having corrupted the clip.
    """
    def proceed() -> None:
        state = ensure(ctx)
        record = state.open_clip()
        if record is None:
            return
        keys = list(record.get("keys") or ())
        segments = list(record.get("segments") or ())
        target = index + int(delta)
        if not (0 <= index < len(keys) and 0 <= target < len(keys)):
            return
        keys[index], keys[target] = keys[target], keys[index]
        if index < len(segments) and target < len(segments):
            segments[index], segments[target] = segments[target], segments[index]
        record["keys"] = keys
        record["segments"] = segments
        state.key_index = target
        _touch(ctx)
        apply_key(ctx)

    guard(ctx, "reorder keys", proceed)


def insert_key(ctx: Any, name: str) -> None:
    """Add an existing key pose to the clip, after the selection.

    From the library's own poses rather than from nothing: a clip references
    keys by name, so "add a key" is either "use one of these" or "author a new
    pose first", and :func:`new_key` is the second.
    """
    guard(ctx, "insert a key", lambda: _insert_key(ctx, name))


def _insert_key(ctx: Any, name: str) -> None:
    state = ensure(ctx)
    record = state.open_clip()
    if record is None or state.key_pose(name) is None:
        return
    keys = list(record.get("keys") or ())
    segments = list(record.get("segments") or ())
    at = min(state.key_index + 1, len(keys))
    keys.insert(at, str(name))
    # One more step exists now, wherever the loop stands: a closed clip gains a
    # segment at the same index, an open one gains the step out of the new key.
    slot = min(at, len(segments))
    segments.insert(slot, segments[slot - 1] if segments else 1)
    record["keys"] = keys
    record["segments"] = segments
    state.key_index = at
    _touch(ctx)
    apply_key(ctx)


def new_key(ctx: Any, name: str) -> None:
    """Author a brand-new key pose from the armature and add it to the clip.

    Refused while scrubbing, by name, for the same reason ``capture_key``
    refuses it: the 2026-09-23 audit (finding poser-03) found the button
    that calls this checked only ``posing``, so scrubbing to an in-between
    frame and then naming a new key silently authored the *interpolated*
    pose as if it were a real one. Checked here too, not only at the button
    -- ``capture_key``'s sibling refusal lives at the call site it protects
    rather than only at the button that reaches it, and this follows suit.
    """
    from ....service import clips as svc_clips

    state = ensure(ctx)
    editor = _viewer_editor(ctx)
    label = str(name or "").strip()
    if editor is None or not label:
        return
    if state.frame >= 0:
        ctx.toast(
            "That is an in-between frame, not a key. Pick a key first.", "warn"
        )
        return
    if state.key_pose(label) is not None:
        ctx.toast(f'a key pose named "{label}" already exists', "warn")
        return
    poses = list(state.clips.get("poses") or ())
    if len(poses) >= svc_clips.MAX_LIBRARY_KEYS:
        ctx.toast(
            f"a clip library holds at most {svc_clips.MAX_LIBRARY_KEYS} key poses",
            "warn",
        )
        return
    record = {
        "name": label,
        "bones": _from_node(editor, dict(editor.pose()), _clip_space(state)),
    }
    root = editor.root_translation() if editor.root is not None else None
    if root and any(root):
        record["root_translation"] = [float(v) for v in root]
    poses.append(record)
    state.clips["poses"] = poses
    # Unguarded on purpose: the armature's edit *is* what was just captured.
    _insert_key(ctx, label)


def remove_key(ctx: Any, index: int) -> None:
    """Drop one key from the clip. The *pose* stays in the library.

    Two different things, deliberately: another clip may use it, and a delete
    that silently reached into the shared pose list would be a delete with a
    blast radius the button does not describe.
    """
    def proceed() -> None:
        from ....service import clips as svc_clips

        state = ensure(ctx)
        record = state.open_clip()
        if record is None:
            return
        keys = list(record.get("keys") or ())
        segments = list(record.get("segments") or ())
        if not 0 <= index < len(keys):
            return
        if len(keys) <= svc_clips.MIN_KEYS:
            ctx.toast(
                f"a clip needs at least {svc_clips.MIN_KEYS} keys; one key is a pose",
                "warn",
            )
            return
        del keys[index]
        if index < len(segments):
            del segments[index]
        elif segments:
            del segments[-1]
        record["keys"] = keys
        record["segments"] = segments
        state.key_index = max(0, min(state.key_index, len(keys) - 1))
        _touch(ctx)
        apply_key(ctx)

    guard(ctx, "remove a key", proceed)


def save_clips(ctx: Any) -> None:
    """Write the working copy. One task, on its own key.

    ``clips_unsaved`` clears when the save *lands*, never at submit: a refused
    write has to leave the editor holding the edits that were refused.
    """
    from ....service import clips as svc_clips

    state = ensure(ctx)
    if not state.template or not state.clips:
        return
    # A deep copy: the task thread reads the payload while the frame thread
    # goes on editing the working lists, and ``adopt_clips`` afterwards
    # replaced them with what hit disk -- discarding every edit made during
    # the write. The serial says whether any were.
    payload = copy.deepcopy(
        {
            "space": state.clips.get("space") or "node",
            "poses": state.clips.get("poses") or [],
            "clips": state.clips.get("clips") or [],
        }
    )
    state.clips_save_serial = state.clips_touch_serial
    if not ctx.submit(CLIPS_SAVE_KEY, svc_clips.save, ctx.svc, state.template, payload):
        ctx.toast("Still saving the previous clip change.", "info")


def revert_clips(ctx: Any) -> None:
    """Throw the working copy away and go back to the shipped clips.

    Behind the guard for ``reset_all``'s reason: it discards authored work and
    this mode has no undo.
    """
    from ....service import clips as svc_clips

    state = ensure(ctx)
    if not state.template:
        return

    def proceed() -> None:
        # The 2026-09-13 finding: this used to submit under the shared
        # CLIPS_SAVE_KEY without recording ``clips_save_serial`` the way
        # :func:`save_clips` does above -- so ``on_task_done``'s own "edited
        # while this was in flight" guard (``clips_touch_serial !=
        # clips_save_serial`` while ``clips_unsaved``) compared the touch
        # serial a Revert was asked at against whatever a previous Save had
        # last recorded, which is behind it whenever there *are* unsaved
        # edits -- the exact case Revert exists for. The landing was silently
        # refused, the shipped clips it had just fetched were thrown away, and
        # ``clips_unsaved`` stayed True forever after a Revert the user had
        # just confirmed. Recording the serial here, save_clips's own line,
        # tells that guard nothing has changed between the ask and the
        # landing, which is true unless another edit races it in.
        state.clips_save_serial = state.clips_touch_serial
        if not ctx.submit(CLIPS_SAVE_KEY, svc_clips.revert, ctx.svc, state.template):
            ctx.toast("Still saving the previous clip change.", "info")

    if state.clips_unsaved or state.clips.get("edited"):
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Revert clips",
                message=(
                    "This throws away every change to this skeleton's clips and "
                    "goes back to the ones the build ships. There is no undo."
                ),
                confirm_label="Revert",
                on_confirm=proceed,
            )
        )
    else:
        proceed()


# --- importing a clip ---------------------------------------------------------
#
# The Troupe programme's other clip-authoring gap: every key so far had to be
# posed by hand on the armature, and Mixamo/Rigify already ship enormous
# libraries of exactly this kind of motion. ``service.clip_import.analyse``
# does the actual sampling (a Blender subprocess) and the pure conversion onto
# a template's own bones; this door only ever merges what it hands back into
# the *working copy* -- never disk, the same rule every other mutation in this
# section already follows -- so an import is undone by :func:`revert_clips`
# exactly like a hand-authored key would be, and only :func:`save_clips` makes
# it permanent.


def _valid_clip_name(name: str) -> bool:
    """Whether *name* would survive ``service.clips``' own Save-time check.

    ``cliplib.reject_direction_named_clip`` is the same function
    ``service.clips._check_shape`` calls -- imported rather than restated, so
    a name this merge accepts and a name Save accepts can never disagree.
    """
    try:
        cliplib.reject_direction_named_clip(name)
    except ValueError:
        return False
    return True


def _dedupe_clip_name(name: str, taken: set[str]) -> str:
    """The first ``"<name>_N"`` (N >= 2) that is neither taken nor a name Save
    would refuse.

    Underscored, unlike :func:`_dedupe_import_pose_name` below -- a clip name
    is what a character sheet and Troupe's tag parser both read, and this is
    simply what an imported action's own name most often already looks like
    (``mixamo.com|Walking`` sampled twice becomes ``Walking``, ``Walking_2``).
    Also re-checked against :func:`_valid_clip_name` on every candidate, not
    only the first: a clash landing on ``_2``/``_3`` is vanishingly unlikely to
    also end in one of Troupe's sixteen facings, but "vanishingly unlikely" is
    exactly the class of bug ``cliplib.parse_clip_library`` exists to catch at
    Save instead of here, and refusing the whole import at that point would be
    a much worse afternoon than looping once more here.
    """
    candidate = name
    n = 2
    while candidate in taken or not _valid_clip_name(candidate):
        candidate = f"{name}_{n}"
        n += 1
    return candidate


def _dedupe_import_pose_name(name: str, taken: set[str]) -> str:
    """The first ``"<name> N"`` (N >= 2) not already in *taken*.

    ``service.clip_import._dedupe_pose_name``'s own scheme, restated rather
    than imported: that one is private to the door that writes straight to
    disk under the clip-store lock, and this module's merge never touches disk
    at all (see the section note above) -- two callers sharing one private
    helper across a service/pane boundary is exactly the "call the service,
    never each other's internals" line ``CLAUDE.md`` draws.
    """
    if name not in taken:
        return name
    n = 2
    while f"{name} {n}" in taken:
        n += 1
    return f"{name} {n}"


def import_clip(ctx: Any) -> bool:
    """Ask for an animation file and sample it onto the browsed template.

    Requires a selected template that already has a clip library open in the
    editor (``state.clips`` -- empty for a template that ships none, and
    :func:`adopt_imported_clips` has no library to merge into then) and
    Blender (``ctx.rigging_available``, the same signal ``request_preview``
    already gates on) -- checked here rather than left for
    ``service.clip_import.analyse``'s own ``doctor.blender_check()`` to refuse,
    so a missing Blender does not cost the user a round trip through the OS
    file picker before saying so. Also refused while ``state.skeleton_editing``
    (P6, 2026-09-13): ``poser_clips._import_button`` draws "Import clip..."
    disabled with that reason, but a disabled button only stops a mouse --
    this is the door a keyboard shortcut or an agent's own call still has to
    go through, and it must say no for the same reason the pane already
    shows.

    The open-file dialog is asked **on the task thread**, inside ``run`` --
    this module's own :func:`export_package` and ``library._export_zip``'s
    arrangement and its reason: a blocking OS picker on the frame thread
    freezes the window behind it. ``None`` from it means the user cancelled,
    and :func:`on_task_done`'s ``CLIP_IMPORT_KEY`` branch does nothing with
    that -- no toast either way, ``dialogs.open_file``'s own contract.
    -> whether the request was taken.
    """
    from ....service import clip_import as svc_clip_import

    state = ensure(ctx)
    if state.skeleton_editing:
        return False
    if not state.template or not state.clips.get("clips"):
        return False
    if not getattr(ctx, "rigging_available", False):
        return False
    if ctx.busy(CLIP_IMPORT_KEY):
        return False
    template = state.template
    svc = ctx.svc

    def run() -> Any:
        path = dialogs.open_file(
            "Import a clip",
            ["Animation files (*.fbx *.glb *.gltf)", "*.fbx *.glb *.gltf"],
        )
        if path is None:
            return None
        result = dict(svc_clip_import.analyse(svc, template, path))
        # The pane's toast names the file; ``analyse`` itself never learns the
        # path came from a file at all.
        result["source_name"] = path.name
        return result

    return bool(ctx.submit(CLIP_IMPORT_KEY, run))


def adopt_imported_clips(ctx: Any, result: dict[str, Any]) -> None:
    """Merge an import's clips into the working copy. Nothing touches disk.

    ``service.clip_import.import_into_library``'s own merge (its docstring:
    name collisions, pose renames, provenance), restated over ``state.clips``
    instead of a locked read-modify-write to the user's file -- because here
    the "file" is the editor's working copy, already sitting in memory and
    already governed by :func:`save_clips`/:func:`revert_clips`. A clip-name
    clash is never a refusal the way that door's ``Conflict`` is: the working
    copy always has room for one more entry, so a clash is renamed
    (:func:`_dedupe_clip_name`) rather than asked about, and a pose-name clash
    is renamed too (:func:`_dedupe_import_pose_name`) -- an existing
    working-copy pose is never overwritten, imported or original.

    Selects the first imported clip and marks the working copy unsaved
    (:func:`_touch`) but does **not** put it on the armature the way
    :func:`select_clip` does: this runs from :func:`on_task_done`, seconds
    after the Blender sample actually finishes, with no click behind the
    landing -- exactly the class of automatic arrival ``adopt_clips`` (its own
    neighbour) already leaves the live pose alone for, rather than clobbering
    whatever the user has been posing on the armature in the meantime.
    """
    state = ensure(ctx)
    if not isinstance(result, dict) or result.get("template") != state.template:
        return
    entries = result.get("clips") or ()
    if not entries:
        return
    poses = list(state.clips.get("poses") or [])
    clip_rows = list(state.clips.get("clips") or [])
    pose_names = {str(p.get("name") or "") for p in poses}
    clip_names = {str(c.get("name") or "") for c in clip_rows}
    source_name = str(result.get("source_name") or "")
    imported_on = datetime.now(UTC).date().isoformat()

    reports: list[dict[str, Any]] = []
    first_name = ""
    for entry in entries:
        clip = dict(entry.get("clip") or {})
        report = dict(entry.get("report") or {})
        reports.append(report)

        rename: dict[str, str] = {}
        for pose_name, pose_body in (entry.get("poses") or {}).items():
            pose_name = str(pose_name)
            new_name = _dedupe_import_pose_name(pose_name, pose_names)
            rename[pose_name] = new_name
            pose_names.add(new_name)
            poses.append({"name": new_name, **dict(pose_body)})
        clip["keys"] = [rename.get(str(k), str(k)) for k in clip.get("keys") or ()]
        if not clip.get("source"):
            # Kept if the analysed clip already carried one (an imported
            # library re-imported, or a test standing in for a richer
            # source); otherwise this is the only record of where it came
            # from, which a later Save writes through untouched
            # (``svc_clips._check_shape`` keeps ``source`` verbatim).
            clip["source"] = {
                "file": source_name,
                "map": str(report.get("map") or ""),
                "imported": imported_on,
            }
        name = _dedupe_clip_name(str(clip.get("name") or ""), clip_names)
        clip["name"] = name
        clip_names.add(name)
        clip_rows.append(clip)
        if not first_name:
            first_name = name

    state.clips["poses"] = poses
    state.clips["clips"] = clip_rows
    state.clip_import_reports = reports
    if first_name:
        state.clip = first_name
        state.key_index = 0
        state.frame = -1
    _touch(ctx)


def _journal_slot_for(ctx: Any, viewer: Any, key: str) -> Any:
    """One journallable pose session, or None.

    A ``SimpleNamespace``-shaped slot rather than a real object: the editor is
    not a document and has no place to keep three bookkeeping fields, so the
    journal's marks live on the *viewer*, which has exactly the lifetime the
    session does. ``key`` distinguishes the two viewers so the inspector's
    session and Poser's cannot share a filename.
    """
    if viewer is None or not getattr(viewer, "pose_mode", False):
        return None
    editor = getattr(viewer, "editor", None)
    if editor is None or not editor.bound or not editor.has_unsaved_edits():
        return None
    return _PoseSlot(viewer=viewer, editor=editor, key=key)


class _PoseSlot:
    """A pose session as the journal sees it. Marks proxy onto the viewer.

    The 2026-09-16 audit, finding shell-02: ``tests/studio/test_journal.py``'s
    six-class pin walks ``dataclasses.fields()`` to catch a slot that drops
    one of the journal's three bookkeeping names (the 2026-08-18
    ``PlotterDoc.journal_at`` incident that pin exists for), and it structurally
    cannot reach this class or ``viewer_embed.Viewer`` -- the object the marks
    actually live on -- because neither one is a dataclass. The storage still
    has to live on the viewer (its lifetime is the session's; a slot is
    rebuilt every frame), so this cannot become a dataclass of its own fields
    the way the other six providers are. Declaring the three names as class
    annotations instead gives a test something to introspect
    (``_PoseSlot.__annotations__``) without changing where the values live.
    """

    journal_name: str
    journal_head: Any
    journal_at: float

    def __init__(self, viewer: Any, editor: Any, key: str) -> None:
        self.viewer = viewer
        self.editor = editor
        self.key = key

    @property
    def journal_name(self) -> str:
        return getattr(self.viewer, "journal_name", "") or ""

    @journal_name.setter
    def journal_name(self, value: str) -> None:
        self.viewer.journal_name = value

    @property
    def journal_head(self) -> Any:
        return getattr(self.viewer, "journal_head", None)

    @journal_head.setter
    def journal_head(self, value: Any) -> None:
        self.viewer.journal_head = value

    @property
    def journal_at(self) -> float:
        return float(getattr(self.viewer, "journal_at", 0.0) or 0.0)

    @journal_at.setter
    def journal_at(self, value: float) -> None:
        self.viewer.journal_at = value


def _pose_payload(slot: Any) -> bytes:
    import json as _json

    editor = slot.editor
    payload: dict[str, Any] = {
        "bones": editor.pose(),
        "root_translation": editor.root_translation(),
        "moved": {name: list(delta) for name, delta in editor.moved.items()},
        "mode": editor.mode,
        # Which job's rig this was bound to, so the inspector's copy can
        # refuse to land on a different asset.
        "job_id": getattr(slot.viewer, "pose_job_id", None) or "",
        "where": slot.key,
    }
    if editor.mode == "skeleton":
        # P7 (2026-09-13): ``bones``/``moved`` above are the *pose*, which is
        # rest for the whole of a skeleton-editing session
        # (``enter_skeleton_mode`` resets it going in) -- the thing actually
        # worth recovering is the draft itself.
        payload["draft"] = [dict(b) for b in editor.draft]
        payload["draft_pairs"] = [list(p) for p in editor.draft_pairs]
        payload["draft_root"] = editor.draft_root
    return _json.dumps(payload, sort_keys=True).encode("utf-8")


def _journal_slots(ctx: Any) -> list[Any]:
    """Both pose sessions -- Poser's own viewer and the inspector's."""
    out = []
    for viewer, key in ((viewer_of(ctx), "poser"), (getattr(ctx, "viewer", None), "asset")):
        slot = _journal_slot_for(ctx, viewer, key)
        if slot is not None:
            out.append(slot)
    return out


def _journal_adopt(ctx: Any, path: Path, meta: dict[str, Any]) -> bool:
    """Put a recovered pose back onto a bound editor, or say why not.

    **A pose is not a document and cannot be opened on its own**: it is a set
    of rotations for a named skeleton, and applying it needs that skeleton
    loaded. So this is the one adopter that can decline for a reason the user
    has to hear -- the job it was authored against may have been deleted, or
    the session may simply not be open -- and it declines by *keeping* the file
    and saying so, because the alternative is applying a stranger's rotations
    to whatever rig happens to be in front of them.
    """
    import json as _json

    try:
        data = _json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.exception("could not read the recovered pose at %s", path)
        return False
    where = str(data.get("where") or "poser")
    viewer = viewer_of(ctx) if where == "poser" else getattr(ctx, "viewer", None)
    editor = getattr(viewer, "editor", None)
    if viewer is None or editor is None or not editor.bound:
        ctx.toast(
            "An unsaved pose was recovered. Open the rig it belongs to and it "
            "will be offered again.",
            "warn",
        )
        return False
    job_id = str(data.get("job_id") or "")
    if job_id and str(getattr(viewer, "pose_job_id", "") or "") != job_id:
        # The asset it was authored against is not the one in front of us. Kept
        # rather than applied: rotations by bone name land on whatever shares a
        # name, silently and wrongly.
        ctx.toast(
            "An unsaved pose was recovered, but it belongs to a different "
            "asset. Open that one and it will be offered again.",
            "warn",
        )
        return False
    mode = str(data.get("mode") or "pose")
    if mode == "skeleton":
        # P7 (2026-09-13): a skeleton draft, not a pose -- re-validated
        # structurally before anything is put back on the editor, since a
        # crash could have caught it between two edits that individually
        # check out but whose combination (through a hand-edited recovery
        # file, or a bone the underlying rig no longer has) does not. Kept,
        # not discarded, on a bad draft: the declined-adopt rule every other
        # refusal here already follows.
        draft = data.get("draft") or []
        try:
            skeleton.check_skeleton_structure(draft)
        except store.RigError:
            ctx.toast(
                "An unsaved skeleton edit was recovered, but it is no longer "
                "usable. Open the rig it belongs to and edit its skeleton "
                "again.",
                "warn",
            )
            return False
        # Through the same public door :func:`poser_mode.enter_skeleton_edit`
        # uses, seeded with the *recovered* draft rather than the rig's own
        # bones -- ``enter_skeleton_mode`` only ever reads ``bones``/
        # ``mirror_pairs``/``root`` off whatever mapping it is handed, so a
        # draft in that same shape re-enters the session exactly where it
        # left off.
        viewer.enter_skeleton_mode(
            {
                "bones": draft,
                "mirror_pairs": data.get("draft_pairs") or [],
                "root": data.get("draft_root"),
            }
        )
        editor.draft_dirty = True
        if where == "poser":
            ensure(ctx).skeleton_editing = True
        ctx.toast("An unsaved skeleton edit was recovered.", "success")
        return True
    editor.apply(data.get("bones") or {}, pose_id=None, dirty=True)
    if data.get("root_translation"):
        editor.set_root_translation(data["root_translation"])
    if mode == "joints":
        # The payload records everything ``_pose_payload`` writes, and joint
        # corrections are half of it: a crash mid-placement recovered as a rest
        # pose under a success toast, the corrections silently gone. Entering
        # joints mode takes fresh homes (the rotations in a joints session are
        # rest anyway -- ``enter_joints_mode`` resets them on the way in, as it
        # did in the crashed session), and each recorded displacement is then
        # replayed through ``move_handle``, the same door a drag uses, so the
        # markers land where the user put them and ``moved`` is re-derived
        # rather than trusted.
        import numpy as np

        from ....kernels.geom3d import math3d as m3

        editor.enter_joints_mode()
        for name, delta in (data.get("moved") or {}).items():
            home = editor.home.get(name)
            if home is None:
                continue
            editor.move_handle(
                name, home + m3.blender_delta_to_gltf(np.asarray(delta, dtype="f8"))
            )
    after = getattr(viewer, "_after_pose_change", None)
    if after is not None:
        after()
    ctx.toast("An unsaved pose was recovered.", "success")
    return True


JOURNAL = journal.register(
    journal.Provider(
        kind="pose",
        ext=".pose.json",
        label="pose",
        slots=_journal_slots,
        uid_of=lambda slot: slot.key,
        title_of=lambda slot: "Pose" if slot.key == "poser" else "Asset pose",
        # Payload equality: see the section note above.
        head_of=_pose_payload,
        encode=_pose_payload,
        adopt=_journal_adopt,
    )
)
