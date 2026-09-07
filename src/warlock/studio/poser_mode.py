"""Poser mode's controller: authoring reusable poses against a skeleton template,
or against one real rigged asset's own mesh.

The ``clay_mode.py`` pattern -- state and logic here, drawing in ``main.py``
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import poselib, rigging
from . import dialogs, journal

log = logging.getLogger(__name__)

# Task keys. Prefixed "poser-" because the app claims results by prefix.
LIST_KEY = "poser-list"
SAVE_KEY = "poser-save"
DELETE_KEY = "poser-delete"
DUPLICATE_KEY = "poser-duplicate"
RENAME_KEY = "poser-rename"
PREVIEW_KEY_PREFIX = "poser-preview:"
CLIPS_KEY = "poser-clips"
CLIPS_SAVE_KEY = "poser-clips-save"
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
    #: This asset's own saved poses (``service.rig.list_poses``), distinct from
    #: the shared, skeleton-keyed library above.
    asset_poses: list[dict[str, Any]] = field(default_factory=list)
    asset_poses_loading: bool = False
    #: Set when binding the viewer to the asset failed (a missing rig.glb, a
    #: GLB with no skin). Cleared only by :func:`retry_asset`, so a broken rig
    #: is not retried every frame.
    asset_error: str = ""
    #: The id of a rig job queued by :func:`rerig`, while it is still
    #: ``queued``/``running``. ``svc_rig.create_rig`` only asks the serial
    #: queue to build a new rig.glb; the write itself lands minutes later, out
    #: of process, on the ``warlock-loop`` thread's own schedule -- so this is
    #: what :func:`pump_rerig` watches to notice the job actually finish.
    #: Empty once it has landed (or failed), so a stale id is never polled
    #: forever.
    rerig_job_id: str = ""
    #: Which asset ``rerig_job_id`` was queued for, captured at submit time
    #: from ``create_rig``'s own ``source_job``. The user can close this
    #: session and open a different asset while the queue is still working;
    #: comparing against this rather than the live ``job_id`` is what stops a
    #: re-rig queued for job A landing on whatever job B happens to be open
    #: when it finishes.
    rerig_source_job: str = ""
    #: Whether the Re-rig picker is expanded, and which skeleton is chosen in
    #: it. Here rather than in ``ctx.state.preview`` -- the pane-scratch dict
    #: the rest of the app uses for this -- because that dict outlives the
    #: session: leaving an asset with the picker open and opening another one
    #: reopened it, on the new asset, still showing the old asset's skeleton.
    #: :func:`close_asset` clears these with the rest of the session, which is
    #: the whole reason they live on the session's own state.
    rerig_open: bool = False
    rerig_choice: str = ""

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
        entries = rigging.catalog()
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
    from ..service import poses as svc_poses
    from ..service import rig as svc_rig

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
    from ..service import poses as svc_poses

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


# --- the asset session ---------------------------------------------------------


def open_asset(ctx: Any, job: dict[str, Any]) -> None:
    """Bind the session to one real rigged asset's own mesh, behind the guard.

    Reads the rig's template so the shared library beside it is the one that
    applies -- ``pose_panel.open_in_poser``'s old rule, kept, since a saved
    library pose still applies by bone name whether or not a mesh is bound.
    A missing or unreadable rig.json is not a reason to refuse the trip, the
    same tolerance ``pose_panel._enter`` already applies: only the mirror
    button and the joint editor need what it carries.
    """
    from ..service import rig as svc_rig

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
        state.asset_error = ""
        state.asset_poses = []
        viewer = viewer_of(ctx)
        if viewer is not None:
            # Whatever the viewer was showing -- another asset, the meshless
            # preview -- is not this one; sync_asset binds the new one when
            # the viewport next draws.
            viewer.exit_pose_mode()
            viewer.clear()
        refresh(ctx)
        clips_refresh(ctx)
        refresh_asset_poses(ctx)

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
        state.asset_poses = []
        state.asset_error = ""
        state.rerig_open = False
        state.rerig_choice = ""
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
    from ..service import rig as svc_rig

    state = ensure(ctx)
    if not state.job_id or state.asset_poses_loading:
        return
    state.asset_poses_loading = True
    key = f"{ASSET_POSES_KEY_PREFIX}{state.job_id}"
    if not ctx.submit(key, svc_rig.list_poses, ctx.svc, state.job_id):
        state.asset_poses_loading = False


def save_pose_to_asset(ctx: Any) -> None:
    """Save the pose being edited onto the bound asset, not the shared library.

    The point of opening a real mesh rather than a template: a change here
    should be able to stick to *this* asset, exactly like the inspector's own
    Pose tab already offers via ``service.rig.save_pose``.
    """
    from ..service import rig as svc_rig

    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if not state.job_id or viewer is None or not viewer.pose_mode:
        return
    job_id = state.job_id
    existing = viewer.editor.current

    def accept(name: str) -> None:
        payload: dict[str, Any] = {"name": name, "bones": viewer.get_pose()}
        if existing:
            payload["id"] = existing
        ctx.submit(
            f"{ASSET_SAVE_KEY_PREFIX}{job_id}", svc_rig.save_pose, ctx.svc, job_id, payload
        )

    ctx.prompts.ask(dialogs.Prompt(title="Name this pose", label="Name", on_accept=accept))


def apply_asset_pose(ctx: Any, pose_id: str) -> None:
    """Load one of the asset's own saved poses into the editor, behind the guard."""
    state = ensure(ctx)
    viewer = viewer_of(ctx)
    record = state.find_asset_pose(pose_id)
    if record is None or viewer is None or not viewer.pose_mode:
        return

    def proceed() -> None:
        viewer.reset_all(dirty=False)
        viewer.set_pose(record.get("bones") or {}, pose_id=record["id"], dirty=False)

    guard(ctx, "apply a saved pose", proceed)


def delete_asset_pose(ctx: Any, pose_id: str, name: str) -> None:
    """Delete one of the asset's own saved poses, behind a confirm."""
    from ..service import rig as svc_rig

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
    """
    from ..service import rig as svc_rig

    state = ensure(ctx)
    if not state.job_id:
        return
    job_id = state.job_id

    def proceed() -> None:
        key = f"{ASSET_RERIG_KEY_PREFIX}{job_id}"
        if not ctx.submit(key, svc_rig.create_rig, ctx.svc, job_id, template=template):
            ctx.toast("Still re-rigging this asset.", "info")

    guard(ctx, "re-rig this asset", proceed)


def pump_rerig(ctx: Any) -> None:
    """Notice a queued re-rig reaching a terminal status, every frame.

    Called from ``poser_library.draw`` beside :func:`pump`, its own per-frame
    heartbeat. ``svc_rig.create_rig`` only enqueues the rig job -- the actual
    Blender solve and the ``rig.glb`` write happen minutes later, out of
    process, on the serial queue's own schedule -- so nothing about the
    *submit* landing (:func:`on_task_done`) can tell whether the new rig
    exists yet. This is what does: a couple of dict lookups against
    ``ctx.job``, which is already kept live every frame for every other mode
    (the same cache ``ctx.cache.tick`` refreshes in ``main.py``).
    """
    state = ensure(ctx)
    if not state.rerig_job_id:
        return
    job = ctx.job(state.rerig_job_id)
    if job is None:
        # Not yet in the loaded window, or a stale id from a session that has
        # since moved on -- either way there is nothing to act on this frame.
        return
    status = job.get("status")
    if status == "done":
        source = state.rerig_source_job
        state.rerig_job_id = ""
        state.rerig_source_job = ""
        if state.job_id == source:
            _land_rerig(ctx)
    elif status in ("error", "cancelled"):
        # The generic job-transition toast (``main.py``'s ``_refresh``)
        # already says why; nothing here is worth watching any further.
        state.rerig_job_id = ""
        state.rerig_source_job = ""


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
    from ..service import rig as svc_rig

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
    from .viewer import math3d as m3

    template = rigging.get_template(template_key)
    fitted = rigging.fit_template(template, poselib.UNIT_LO, poselib.UNIT_HI)
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

    Frame thread only (it loads a model and frames a camera). What decides is
    ``viewer.path`` against the landed answer -- never a remembered flag, the
    Review lesson -- and a preview built for a template the user has switched
    away from never binds. -> whether the viewer is showing the preview.
    """
    state = ensure(ctx)
    path = state.preview_path
    if path is None or state.preview_template != state.template:
        return viewer.path is not None
    if viewer.path == Path(path):
        return True
    try:
        viewer.load_model(Path(path))
    except Exception:
        log.exception("could not open the %s pose preview", state.template)
        state.preview_path = None
        state.error = "Could not open the skeleton preview."
        return False
    bind_preview(ctx, viewer, state.template)
    return True


def bind_preview(ctx: Any, viewer: Any, template_key: str) -> None:
    """Enter the authoring session over whatever the viewer just loaded."""
    template = rigging.get_template(template_key)
    bones = [b["name"] for b in template.bones]
    viewer.enter_pose_authoring(
        bones, [list(p) for p in template.mirror_pairs], token(template.key)
    )
    viewer.editor.root = template.root
    lo, hi = preview_bounds(template_key)
    viewer.frame_bounds(lo, hi)


def sync_asset(ctx: Any, viewer: Any) -> bool:
    """Bind the viewer to the session's asset if it is not already shown.

    Frame thread only, ``sync_preview``'s reason. Unlike the template preview
    there is no background build to track: loading a rig.glb is a synchronous
    parse and GPU upload, the same call ``pose_panel._enter`` makes directly.
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
    try:
        viewer.load_model(rig_path)
    except Exception:
        log.exception("could not open the rig for job %s", job_id)
        state.asset_error = "Could not open the rig."
        return False
    if not viewer.enter_pose_mode(state.asset_rig, job_id):
        state.asset_error = "That GLB carries no skeleton."
        return False
    viewer.frame()
    return True


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
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return
    guard(ctx, "apply a preset", lambda: viewer.apply_preset(preset))


def new_pose(ctx: Any) -> None:
    """Back to rest with nothing being edited, behind the guard."""
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return
    guard(ctx, "start a new pose", lambda: viewer.reset_all(dirty=False))


# --- saving ------------------------------------------------------------------


def _payload(state: PoserState, viewer: Any, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "template": state.template,
        "bones": viewer.get_pose(),
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
    from ..service import poses as svc_poses

    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
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
        _payload(state, viewer, str(record.get("name") or "")),
    )


def save_as(ctx: Any, tab: Any = None) -> None:
    from ..service import poses as svc_poses

    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return

    def accept(name: str) -> None:
        _mutate(
            ctx, SAVE_KEY, svc_poses.create_library_pose, ctx.svc, _payload(state, viewer, name)
        )

    ctx.prompts.ask(dialogs.Prompt(title="Name this pose", label="Name", on_accept=accept))


def rename(ctx: Any, pose_id: str) -> None:
    from ..service import poses as svc_poses

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
    from ..service import poses as svc_poses

    _mutate(ctx, DUPLICATE_KEY, svc_poses.duplicate_library_pose, ctx.svc, pose_id)


def delete(ctx: Any, pose_id: str) -> None:
    """Behind a confirm: the library has no trash, so this one is genuinely
    irreversible -- the paths that are keep their question."""
    from ..service import poses as svc_poses

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


def guard(ctx: Any, verb: str, proceed: Any) -> bool:
    """Ask before discarding unsaved Poser edits. -> whether it went ahead now.

    Asks only about the *Poser* viewer's editor, which is what makes it
    mutually exclusive with ``pose_panel.guard`` by construction: that one
    reads the shared viewer, this one reads ``ctx.poser_viewer``, and no edit
    can live in both. Leaving the mode needs no guard at all -- the session
    survives on its own viewer, like an open Inker document -- so this runs
    only on quit and on destructive in-mode actions.
    """
    from . import docmodes

    return docmodes.viewer_guard(ctx, viewer_of(ctx), "pose", verb, proceed)


# --- keys and task results ---------------------------------------------------


def handle_key(ctx: Any, event: Any) -> bool:
    """Poser's shortcuts. -> whether the key was consumed.

    Esc deselects the joint and Ctrl+Z/Ctrl+Shift+Z/Ctrl+Y walk the pose
    history; nothing else is bound -- the mode is otherwise mouse-shaped. The
    caller returns unconditionally either way, the workspace-mode rule.

    The undo binding is shared with the inspector's asset pose mode through
    ``docmodes.pose_undo_key``, because it is one editor with two doors.
    """
    import pygame

    from . import docmodes

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
    from . import clay_mode

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
    """Called from the app for every ``poser-`` key."""
    state = ensure(ctx)
    key = done.key
    if key == LIST_KEY:
        state.loading = False
        if isinstance(done.result, dict) and done.result.get("template") == state.template:
            state.poses = list(done.result.get("poses") or ())
            state.presets = list(done.result.get("presets") or ())
        # A save landing while this list was in flight set refresh_dirty and
        # could submit nothing; the landing is the moment the key is free.
        pump(ctx)
        return
    if key.startswith(PREVIEW_KEY_PREFIX):
        state.building = False
        template = key[len(PREVIEW_KEY_PREFIX):]
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
    if key.startswith(ASSET_POSES_KEY_PREFIX):
        state.asset_poses_loading = False
        job_id = key[len(ASSET_POSES_KEY_PREFIX):]
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
        # actual write, once the queue gets around to it.
        if isinstance(done.result, dict) and done.result.get("id"):
            state.rerig_job_id = str(done.result["id"])
            state.rerig_source_job = str(done.result.get("source_job") or "")
        return


def on_task_failed(ctx: Any, done: Any) -> None:
    """Flags only: the generic failure path has already toasted the service's
    own message, which for a save names the duplicate or the bad field."""
    state = ensure(ctx)
    if done.key == LIST_KEY:
        # ``loading`` gates the refresh; leaving it set makes the mode inert.
        state.loading = False
        # A refresh wanted while the failed list was in flight is still wanted.
        pump(ctx)
        return
    if done.key.startswith(PREVIEW_KEY_PREFIX):
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
        state.asset_poses_loading = False
        return


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
    from ..service import clips as svc_clips

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
    from ..pipelines import sheet as sheetlib

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

    ``how`` is ``rigging.node_from_delta`` or ``rigging.delta_from_node``: the
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
    return _convert(editor, bones, space, rigging.node_from_delta)


def _from_node(editor: Any, bones: dict[str, Any], space: str) -> dict[str, list[float]]:
    """The editor's node-local frame -> library rotations."""
    return _convert(editor, bones, space, rigging.delta_from_node)


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
    record = state.open_clip()
    if editor is None or record is None:
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
    from ..service import clips as svc_clips

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
    """Author a brand-new key pose from the armature and add it to the clip."""
    from ..service import clips as svc_clips

    state = ensure(ctx)
    editor = _viewer_editor(ctx)
    label = str(name or "").strip()
    if editor is None or not label:
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
        from ..service import clips as svc_clips

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
    from ..service import clips as svc_clips

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
    from ..service import clips as svc_clips

    state = ensure(ctx)
    if not state.template:
        return

    def proceed() -> None:
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
    """A pose session as the journal sees it. Marks proxy onto the viewer."""

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
    return _json.dumps(
        {
            "bones": editor.pose(),
            "root_translation": editor.root_translation(),
            "moved": {name: list(delta) for name, delta in editor.moved.items()},
            "mode": editor.mode,
            # Which job's rig this was bound to, so the inspector's copy can
            # refuse to land on a different asset.
            "job_id": getattr(slot.viewer, "pose_job_id", None) or "",
            "where": slot.key,
        },
        sort_keys=True,
    ).encode("utf-8")


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
    editor.apply(data.get("bones") or {}, pose_id=None, dirty=True)
    if data.get("root_translation"):
        editor.set_root_translation(data["root_translation"])
    if str(data.get("mode") or "pose") == "joints":
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

        from .viewer import math3d as m3

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
