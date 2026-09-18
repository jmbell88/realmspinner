"""The pose editor's state: rotations, joint moves, mirroring.

Two modes over one set of markers. In **pose** mode a marker shows where its
bone ended up and the gizmo rotates the bone; in **joints** mode the marker
*is* the thing being dragged and the gizmo translates it, because a joint's
position is a property of the armature's rest pose that the viewer's copy of
the rig cannot express -- so the marker is the handle and the server re-skins.

The mirror comes from :func:`warlock.poses.mirror_pose`, imported rather than
reimplemented. The browser had its own copy with a comment insisting the two
stay identical, which is exactly the kind of sign convention that is wrong in a
way you cannot see: a mirrored arm rotating the wrong way about one axis still
looks plausible in a still.
"""

from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ... import poselib
from ...core.undo import Edit, UndoStack
from ...kernels.geom3d import math3d as m3
from ...kernels.geom3d.gltf import Model
from ...kernels.rig import poses, skeleton, store

# Re-exported so nothing downstream is tempted to write the sign flip out again.
mirror_quaternion = poses.mirror_quaternion


@dataclass
class PoseSnapshotEdit(Edit):
    """One reversible pose step, as a pair of whole-editor snapshots.

    Whole snapshots rather than a diff, which is the opposite of what the
    raster editor does and is right for the same reason: a ``PatchEdit`` stores
    a rectangle because a canvas is megabytes, and a pose is a few dozen
    quaternions -- under a kilobyte for the canonical armature. A diff would buy
    nothing and would have to describe six different kinds of change (a
    rotation, the root translation, a joint drag, the mode, the identity of the
    pose being edited, the dirty flag) in six ways, any one of which could be
    forgotten when a seventh is added.

    ``cost`` is measured rather than assumed, so the stack's byte budget is
    honest about a two-hundred-bone rig.
    """

    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cost = _snapshot_cost(self.before) + _snapshot_cost(self.after)

    def undo(self, doc: Any) -> None:
        doc.restore(self.before)

    def redo(self, doc: Any) -> None:
        doc.restore(self.after)


def _undoable(method):
    """Mark a discrete operation as one undo step.

    The decorator rather than a ``with`` inside each body, because the thing
    being asserted is uniform -- "this whole call is one step" -- and writing
    it out seven times is seven chances to indent one of them wrong. The
    gesture-level mutators (``rotate_selected``, ``move_root``,
    ``move_handle``) are deliberately *not* decorated: each is called once per
    frame of a drag, and a step per frame is a stack full of one gesture. The
    pane brackets those with ``record()`` across the whole grab instead.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.record():
            return method(self, *args, **kwargs)

    return wrapper


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two snapshots describe the same editor state.

    Written out rather than ``a == b`` because the handle maps hold numpy
    arrays, whose ``==`` is an array and whose truth value raises. Selection is
    deliberately *not* compared: clicking a different marker changes what the
    gizmo points at and nothing about the pose, and a stack that recorded it
    would spend a user's undo presses putting the highlight back.
    """
    for key in (
        "rotations",
        "moved",
        "root_translation",
        "mode",
        "current",
        "dirty",
        "draft",
        "draft_pairs",
        "draft_root",
        "draft_dirty",
    ):
        if a.get(key) != b.get(key):
            return False
    for key in ("handles", "home"):
        left, right = a.get(key) or {}, b.get(key) or {}
        if left.keys() != right.keys():
            return False
        if any(not np.array_equal(left[name], right[name]) for name in left):
            return False
    return True


def _snapshot_cost(snap: dict[str, Any]) -> int:
    """Bytes, roughly: eight per float, which is what the arrays hold."""
    rotations = len(snap.get("rotations") or ()) * 4 * 8
    handles = len(snap.get("handles") or ()) * 3 * 8
    home = len(snap.get("home") or ()) * 3 * 8
    moved = len(snap.get("moved") or ()) * 3 * 8
    draft = len(snap.get("draft") or ()) * 6 * 8
    return rotations + handles + home + moved + draft + 128


class PoseEditor:
    """Rotations and joint moves for one rigged model."""

    def __init__(self) -> None:
        self.model: Model | None = None
        self.bones: list[str] = []
        self.rest: dict[str, np.ndarray] = {}
        self.selected: str | None = None
        self.mode = "pose"  # or "joints"
        self.current: str | None = None  # the saved pose being edited
        self.dirty = False
        self.mirror_pairs: list[list[str]] = []
        self.fitted: list[dict[str, Any]] = []
        # bone -> [dx, dy, dz] in *Blender* space, which is what the API wants.
        self.moved: dict[str, list[float]] = {}
        # Where each marker started this joints-mode session, in model space.
        self.home: dict[str, np.ndarray] = {}
        self.handles: dict[str, np.ndarray] = {}
        # Root-translation authoring (Poser only; pose_panel's entry path never
        # sets these, which is what keeps asset pose mode behaviourally
        # unchanged). ``root`` names the bone whose node translation may move;
        # ``root_translate`` is whether the gizmo currently translates it.
        self.root: str | None = None
        self.root_translate = False
        # Skeleton mode (Poser's structure editor): a DRAFT bone list edited
        # in place, disconnected from the loaded glTF -- a bone this session
        # adds has no node, so it cannot be posed or joint-corrected the way
        # ``fitted``/``moved`` above assume. ``draft`` is
        # ``{name, parent, head, tail}`` dicts, Blender axes, the exact shape
        # ``kernels.rig.skeleton``'s pure editors take and return. ``draft_root`` is the
        # edited skeleton's root name; ``draft_pairs`` its mirror pairs.
        self.draft: list[dict[str, Any]] = []
        self.draft_pairs: list[list[str]] = []
        self.draft_root: str | None = None
        self.draft_dirty = False
        # Whether a skeleton-mode drag also moves the dragged bone's mirror
        # partner, reflected in Blender X. A plain settable flag rather than a
        # mode of its own: the UI toggles it mid-session the way Clay's own
        # mirror toggle works.
        self.mirror_edit = False
        # The fixed anchor a skeleton-mode drag's Blender-space draft point is
        # mapped through to reach glTF world space, and back. Captured once on
        # ``enter_skeleton_mode`` from the *root* bone -- its Blender head and
        # its already-computed glTF handle -- because a new bone has no glTF
        # node of its own to read a world matrix from. Anchoring on a single
        # bone rather than converting through each bone's own parent chain is
        # what keeps the mapping well-defined for a bone the draft has not
        # built a hierarchy for yet.
        self._skel_anchor_blender = np.zeros(3)
        self._skel_anchor_gltf = np.zeros(3)
        # Undo, per pose *session*. Dropped by ``bind`` and ``clear`` rather
        # than carried, because adopting a different model rebuilds ``bones``
        # and ``rest`` -- a surviving step would restore rotations onto a
        # skeleton that never had them, by name, silently.
        self.history = UndoStack()
        # Re-entrancy depth for ``record``; nonzero means a step is open.
        self._depth = 0
        # Bumped by every ``_reset_history``. An open ``record`` that spans a
        # rebind -- a drag live when the model is adopted -- compares this and
        # declines to push, because its 'before' describes a skeleton that is
        # no longer here.
        self._generation = 0

    # -- binding -----------------------------------------------------------

    def bind(self, model: Model, bones: list[str] | None = None) -> None:
        """Attach to a loaded rig. Joint order follows the skin's palette."""
        self.model = model
        if bones is None:
            bones = _skin_bones(model)
        self.bones = [b for b in bones if b in model.by_name]
        self.rest = {b: model.get_rotation(b) for b in self.bones}
        self.selected = None
        self.current = None
        self.dirty = False
        self.moved.clear()
        self.home.clear()
        self.root = None
        self.root_translate = False
        self.handles = {b: model.nodes[model.by_name[b]].world[:3, 3].copy() for b in self.bones}
        self.draft = []
        self.draft_pairs = []
        self.draft_root = None
        self.draft_dirty = False
        self.mirror_edit = False
        self._reset_history()

    def clear(self) -> None:
        self.model = None
        self.bones = []
        self.rest.clear()
        self.handles.clear()
        self.home.clear()
        self.moved.clear()
        self.selected = None
        self.current = None
        self.dirty = False
        self.mode = "pose"
        self.root = None
        self.root_translate = False
        self.draft = []
        self.draft_pairs = []
        self.draft_root = None
        self.draft_dirty = False
        self.mirror_edit = False
        self._reset_history()

    def _reset_history(self) -> None:
        self.history.clear()
        self._depth = 0
        self._generation += 1

    @property
    def bound(self) -> bool:
        return self.model is not None and bool(self.bones)

    def has_unsaved_edits(self) -> bool:
        return bool(self.dirty or self.moved or self.draft_dirty)

    # -- undo ---------------------------------------------------------------
    #
    # The stack lives here rather than in the pane because both entry points
    # into pose editing -- Poser's authoring session and the inspector's asset
    # pose mode -- want the same history over the same object, and a pane that
    # owned it would own it twice. ``core/undo`` is the shared engine Clay
    # already borrows; nothing in it is about pixels.
    #
    # A *snapshot* is the unit, not a diff. See ``PoseSnapshotEdit``.

    def snapshot(self) -> dict[str, Any]:
        """Everything an edit can change, as plain data owned by the caller.

        Copies throughout, for ``undo``'s rule that an edit owns its data: the
        handle arrays are live numpy state, and storing views of them would
        make an undo restore whatever the value happened to become.
        """
        model = self.model
        return {
            "rotations": {} if model is None else {
                bone: [float(v) for v in (m3.quat_identity() if q is None else q)]
                for bone, q in ((b, model.get_rotation(b)) for b in self.bones)
            },
            "root_translation": self.root_translation(),
            "moved": {name: list(delta) for name, delta in self.moved.items()},
            "handles": {name: point.copy() for name, point in self.handles.items()},
            "home": {name: point.copy() for name, point in self.home.items()},
            "mode": self.mode,
            "selected": self.selected,
            "current": self.current,
            "dirty": self.dirty,
            "draft": [dict(b) for b in self.draft],
            "draft_pairs": [list(p) for p in self.draft_pairs],
            "draft_root": self.draft_root,
            "draft_dirty": self.draft_dirty,
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Put the editor back to a snapshot. The other half of an edit.

        ``dirty`` is restored rather than set, which is the whole reason it is
        in the snapshot: undoing back to the state a save was taken at has to
        leave the session clean, or the quit guard asks about work the user
        already stored.
        """
        if self.model is None:
            return
        for bone, quat in (snap.get("rotations") or {}).items():
            if bone in self.model.by_name:
                self.model.set_rotation(bone, quat)
        # After the rotations and before ``update_world``: the root's offset is
        # a node translation, and both feed the same matrix.
        if self.root is not None:
            self.set_root_translation(snap.get("root_translation") or [0.0, 0.0, 0.0])
        self.model.update_world()
        self.mode = snap.get("mode", "pose")
        self.moved = {name: list(delta) for name, delta in (snap.get("moved") or {}).items()}
        self.home = {name: point.copy() for name, point in (snap.get("home") or {}).items()}
        self.draft = [dict(b) for b in (snap.get("draft") or [])]
        self.draft_pairs = [list(p) for p in (snap.get("draft_pairs") or [])]
        self.draft_root = snap.get("draft_root")
        self.draft_dirty = bool(snap.get("draft_dirty"))
        # Handles last and verbatim, because ``_resync_handles`` deliberately
        # refuses to touch them in joints or skeleton mode -- where the marker
        # *is* the edit.
        self._resync_handles()
        self.handles = {
            name: point.copy() for name, point in (snap.get("handles") or {}).items()
        }
        self.selected = snap.get("selected")
        self.current = snap.get("current")
        self.dirty = bool(snap.get("dirty"))

    @contextlib.contextmanager
    def record(self):
        """Bracket one undoable step. Re-entrant; only the outermost pushes.

        Re-entrant because the discrete operations compose -- ``apply_preset``
        is a ``reset_all`` and an ``apply``, and a user pressing a preset made
        one edit, not two. A gizmo drag brackets differently: the pane opens
        this on grab and closes it on release, so the hundreds of intermediate
        ``rotate_selected`` calls collapse into the single step the gesture was.

        Nothing is pushed when nothing changed, so a click that selects a bone
        and a drag that ends where it started both leave the stack alone.
        """
        if self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        before = self.snapshot()
        generation = self._generation
        self._depth = 1
        try:
            yield
        finally:
            self._depth = 0
        if generation != self._generation:
            # The session was rebound or torn down inside the step. Pushing now
            # would file a rotation map keyed on the *previous* skeleton's bone
            # names, which restores silently and wrongly onto whatever shares a
            # name with it.
            return
        after = self.snapshot()
        if not _same(before, after):
            self.history.push(PoseSnapshotEdit(before, after))

    def undo(self) -> bool:
        return self.history.undo(self)

    def redo(self) -> bool:
        return self.history.redo(self)

    # -- rotations ---------------------------------------------------------

    def pose(self) -> dict[str, list[float]]:
        """Every bound bone's current local rotation, XYZW."""
        if self.model is None:
            return {}
        out: dict[str, list[float]] = {}
        for bone in self.bones:
            quat = self.model.get_rotation(bone)
            out[bone] = [float(v) for v in (m3.quat_identity() if quat is None else quat)]
        return out

    def apply(self, bones: dict[str, Any], *, pose_id: str | None = None, dirty: bool = True):
        if self.model is None:
            return
        for name, quat in (bones or {}).items():
            self.model.set_rotation(name, quat)
        self.model.update_world()
        self._resync_handles()
        self.current = pose_id
        self.dirty = dirty

    @_undoable
    def apply_preset(self, preset: dict[str, Any]) -> None:
        """A preset lists only the bones it moves, so the rest is reset first
        -- otherwise applying "idle" after "wave" leaves the arm up."""
        self.reset_all(dirty=True)
        self.apply(preset.get("bones", {}), pose_id=None, dirty=True)

    @_undoable
    def reset_bone(self, bone: str | None = None) -> None:
        bone = bone or self.selected
        if self.model is None or bone is None:
            return
        self.model.set_rotation(bone, self.rest.get(bone, m3.quat_identity()))
        if bone == self.root:
            self._restore_root_rest()
        self.model.update_world()
        self._resync_handles()
        self.dirty = True

    @_undoable
    def reset_all(self, *, dirty: bool = True) -> None:
        if self.model is None:
            return
        for bone, quat in self.rest.items():
            self.model.set_rotation(bone, quat)
        # The root's rest translation comes back with the rotations, which is
        # also what makes a preset zero-translation: apply_preset resets first.
        self._restore_root_rest()
        self.model.update_world()
        self._resync_handles()
        self.current = None
        self.dirty = dirty

    def rotate_selected(self, delta: np.ndarray) -> None:
        """Post-multiply a local-space delta onto the selected bone.

        Post- rather than pre-multiply: the gizmo's rings are drawn in the
        joint's own frame, so a drag round the visible red ring must turn the
        bone about *its* X, not the world's.
        """
        if self.model is None or self.selected is None:
            return
        current = self.model.get_rotation(self.selected)
        if current is None:
            return
        self.model.set_rotation(self.selected, m3.quat_normalize(m3.quat_mul(current, delta)))
        self.model.update_world()
        self._resync_handles()
        self.dirty = True

    @_undoable
    def mirror(self) -> None:
        """Copy every posed bone onto its mirror partner, reflected."""
        if not self.mirror_pairs:
            return
        # ``pose_id=self.current``: apply's clearing of ``current`` is for
        # *loading* a different pose, and a mirror is still the same one --
        # dirty yes, identity no. Without it, Mirror silently turned the next
        # Save into Save-as.
        self.apply(poses.mirror_pose(self.pose(), self.mirror_pairs), pose_id=self.current)
        if self.root is not None:
            # The positional half of the same reflection: a pose that steps
            # left must step right when mirrored.
            self.set_root_translation(
                poselib.mirror_root_translation(self.root_translation())
            )

    def _resync_handles(self) -> None:
        """Markers follow their bones -- except in joints or skeleton mode,
        where a marker *is* the drag (or, in skeleton mode, has no bone to
        follow at all) and snapping it back would undo it as fast as it is
        made."""
        if self.model is None or self.mode in ("joints", "skeleton"):
            return
        self.handles = {
            b: self.model.nodes[self.model.by_name[b]].world[:3, 3].copy() for b in self.bones
        }

    # -- root translation --------------------------------------------------
    #
    # Poser-only: ``root`` is set by the authoring session after bind, never
    # by pose_panel's entry path, so asset pose mode cannot reach any of this.
    # The offset previews live as ``node.translation = rest + delta`` -- the
    # exact arithmetic the bake performs -- with the rest translation
    # remembered on the Model the way rest_rotations already are.

    def _root_index(self) -> int | None:
        if self.model is None or self.root is None:
            return None
        return self.model.by_name.get(self.root)

    def _restore_root_rest(self) -> None:
        index = self._root_index()
        if index is None:
            return
        self.model.nodes[index].translation = self.model.rest_translations[index].copy()

    def move_root(self, point: Any) -> None:
        """Place the root joint during a translate drag.

        ``point`` is in model space -- the caller has already divided the
        placement out, the ``move_handle`` convention. The displacement is
        carried into the root's parent frame (where a glTF node translation
        lives) and soft-clamped to ±MAX_ROOT_TRANSLATION per component: on the
        canonical Poser armature model units are character heights literally,
        and an offset past two of them is a slipped drag, not a pose.
        """
        index = self._root_index()
        if index is None:
            return
        node = self.model.nodes[index]
        # parent_world = world @ local^-1: no parent map needed.
        parent_world = node.world @ np.linalg.inv(node.local())
        target = np.linalg.inv(parent_world) @ np.array(
            [float(point[0]), float(point[1]), float(point[2]), 1.0], dtype="f8"
        )
        rest = self.model.rest_translations[index]
        delta = np.clip(
            target[:3] - rest, -poselib.MAX_ROOT_TRANSLATION, poselib.MAX_ROOT_TRANSLATION
        )
        node.translation = rest + delta
        self.model.update_world()
        self._resync_handles()
        self.dirty = True

    def set_root_translation(self, v: Any, *, dirty: bool = True) -> None:
        """Set the root offset from a stored record: Blender axes,
        character-height units. Used when a saved pose is loaded back."""
        index = self._root_index()
        if index is None:
            return
        delta = np.clip(
            m3.blender_delta_to_gltf(np.asarray(v, dtype="f8")),
            -poselib.MAX_ROOT_TRANSLATION,
            poselib.MAX_ROOT_TRANSLATION,
        )
        node = self.model.nodes[index]
        node.translation = self.model.rest_translations[index] + delta
        self.model.update_world()
        self._resync_handles()
        self.dirty = dirty

    def root_translation(self) -> list[float]:
        """The current root offset -- Blender axes, character-height units.
        Zeros when nothing is bound or the root never moved, so a caller can
        always store what this returns."""
        index = self._root_index()
        if index is None:
            return [0.0, 0.0, 0.0]
        delta = self.model.nodes[index].translation - self.model.rest_translations[index]
        return [float(x) for x in m3.gltf_delta_to_blender(delta)]

    # -- joint placement ---------------------------------------------------

    @_undoable
    def enter_joints_mode(self) -> None:
        # A rotation left over from posing would put the markers where the
        # *posed* bones are, and a joint correction is against the rest skeleton.
        self.reset_all(dirty=False)
        self.mode = "joints"
        self.moved.clear()
        self.selected = None
        self.home = {name: point.copy() for name, point in self.handles.items()}

    @_undoable
    def exit_joints_mode(self) -> None:
        self.mode = "pose"
        self.moved.clear()
        self.selected = None
        self.revert_joints()

    @_undoable
    def revert_joints(self) -> None:
        self.moved.clear()
        for name, point in self.home.items():
            self.handles[name] = point.copy()

    def move_handle(self, key: str, position: np.ndarray) -> None:
        """Place a marker during a joints- or skeleton-mode drag.

        ``key`` is a bone name in joints mode; in skeleton mode it is that same
        bone-name convention plus the ``"@tail"`` suffix a leaf bone's own tail
        handle carries (see :meth:`skeleton_payload`'s neighbours below). The
        dispatch is on ``self.mode`` rather than on the key's shape, because a
        joints-mode bone name never collides with one -- ``BONE_NAME_RE``
        forbids ``@`` -- but asking the mode is the same rule every other
        mode-conditional method here already follows.
        """
        if self.mode == "skeleton":
            self._move_skeleton_handle(key, position)
            return
        bone = key
        if bone not in self.handles:
            return
        self.handles[bone] = np.asarray(position, dtype="f8").copy()
        home = self.home.get(bone)
        if home is None:
            return
        delta = self.handles[bone] - home
        if not np.any(delta):
            self.moved.pop(bone, None)
        else:
            # Recorded in Blender space, because that is the space rig.json's
            # joint positions are already in -- converting the *delta* rather
            # than the absolutes is what keeps this lossless.
            self.moved[bone] = [float(v) for v in m3.gltf_delta_to_blender(delta)]

    def corrected_bones(self) -> list[dict[str, Any]]:
        """The whole skeleton, fitted positions with the dragged joints substituted.

        A bone's tail follows its first child's head where one exists, and
        otherwise moves rigidly with its own head. That is the rule that keeps
        a dragged chain connected -- moving a knee has to shorten the thigh and
        lengthen the shin, not leave a gap where the thigh used to end.
        """

        def shift(point, name):
            d = self.moved.get(name)
            return [point[0] + d[0], point[1] + d[1], point[2] + d[2]] if d else list(point)

        heads = {b["name"]: shift(b["head"], b["name"]) for b in self.fitted}
        first_child: dict[str, str] = {}
        for bone in self.fitted:
            parent = bone.get("parent")
            if parent and parent not in first_child:
                first_child[parent] = bone["name"]
        return [
            {
                "name": bone["name"],
                "head": heads[bone["name"]],
                "tail": heads[first_child[bone["name"]]]
                if bone["name"] in first_child
                else shift(bone["tail"], bone["name"]),
            }
            for bone in self.fitted
        ]

    # -- skeleton editing ----------------------------------------------------
    #
    # A DRAFT bone list -- ``kernels.rig.skeleton``'s pure ``{name, parent, head, tail}``
    # shape -- edited in place. A bone this session adds has no glTF node, so
    # unlike joints mode (which moves a marker that already has one) the mesh
    # cannot be reposed to show it: the mesh stays at rest throughout, and the
    # skeleton is drawn from ``draft`` alone via :func:`bonelines.draft_segments`
    # and the handles below.
    #
    # Every mutator is a thin, undoable wrapper over the corresponding pure
    # function in ``kernels.rig.skeleton`` -- never a second implementation of the edit,
    # only of the bookkeeping (``draft_dirty``, ``selected``, the handle cache)
    # around it. A :class:`RigError` from one of those raises *before* any of
    # that bookkeeping runs, which is what keeps ``@_undoable``'s "no state
    # changed" promise true on a refusal.

    @_undoable
    def enter_skeleton_mode(self, rig: dict[str, Any]) -> None:
        """Seed ``draft`` from ``rig`` and switch to skeleton mode.

        Resets the pose to rest first, exactly like ``enter_joints_mode``: the
        draft's positions are read against the rest skeleton, and a rotation
        left over from posing would draw it in the wrong place.
        """
        if not self.bound:
            return
        self.reset_all(dirty=False)
        self.draft = [dict(b) for b in (rig.get("bones") or [])]
        self.draft_pairs = [list(p) for p in (rig.get("mirror_pairs") or [])]
        self.draft_root = rig.get("root")
        self.draft_dirty = False
        self.mirror_edit = False
        self.selected = None
        self.mode = "skeleton"
        # The anchor is captured from the *current* (rest) handles, before
        # ``_recompute_skeleton_handles`` below replaces them -- a new bone
        # added later has no entry in ``self.handles`` at all.
        anchor_gltf = self.handles.get(self.draft_root)
        self._skel_anchor_gltf = (
            np.zeros(3) if anchor_gltf is None else anchor_gltf.copy()
        )
        anchor_bone = next(
            (b for b in self.draft if b["name"] == self.draft_root), None
        )
        self._skel_anchor_blender = np.asarray(
            anchor_bone["head"] if anchor_bone is not None else [0.0, 0.0, 0.0],
            dtype="f8",
        )
        self._recompute_skeleton_handles()

    @_undoable
    def exit_skeleton_mode(self) -> None:
        """Back to pose mode. The draft is discarded, saved or not: a caller
        that wants to keep it queues the re-rig (via :meth:`skeleton_payload`)
        before calling this."""
        self.mode = "pose"
        self.draft = []
        self.draft_pairs = []
        self.draft_root = None
        self.draft_dirty = False
        self.selected = None
        self._resync_handles()

    def skeleton_payload(self) -> dict[str, Any]:
        """The draft, in ``service.rig.edit_skeleton``'s payload shape."""
        return {
            "bones": [dict(b) for b in self.draft],
            "root": self.draft_root,
            "mirror_pairs": [list(p) for p in self.draft_pairs],
        }

    def selected_bone(self) -> str | None:
        """The selected handle's bone name, with any ``"@tail"`` stripped."""
        if self.selected is None:
            return None
        return self.selected[:-5] if self.selected.endswith("@tail") else self.selected

    def subtree_size(self, name: str) -> int:
        """How many bones :meth:`skel_remove_subtree` would remove -- ``name``
        and everything beneath it. A query, not an edit: the confirmation a UI
        wants before committing to a removal that might take a whole limb."""
        doomed = {name}
        changed = True
        while changed:
            changed = False
            for b in self.draft:
                if b["parent"] in doomed and b["name"] not in doomed:
                    doomed.add(b["name"])
                    changed = True
        return len(doomed)

    def _check_skeleton_cap(self, count: int) -> None:
        if count > skeleton.MAX_SKELETON_BONES:
            raise store.RigError(
                f"a skeleton may hold at most {skeleton.MAX_SKELETON_BONES} bones, "
                f"not {count}",
                field="bones",
            )

    @_undoable
    def skel_add_child(self, parent: str) -> str:
        """A new bone as ``parent``'s child, continuing its direction at half
        its length -- a starting point the user drags into place, not a
        finished joint."""
        self._check_skeleton_cap(len(self.draft) + 1)
        by_name = {b["name"]: b for b in self.draft}
        if parent not in by_name:
            raise store.RigError(f"unknown parent {parent!r}", field="parent")
        parent_bone = by_name[parent]
        head = np.asarray(parent_bone["tail"], dtype="f8")
        direction = head - np.asarray(parent_bone["head"], dtype="f8")
        length = float(np.linalg.norm(direction))
        if length <= 0:
            direction, length = np.array([0.0, 0.0, 1.0]), 1.0
        else:
            direction = direction / length
        tail = head + direction * (0.5 * length)
        new_name = skeleton.unique_name(self.draft, "bone")
        self.draft = skeleton.add_bone(self.draft, parent, new_name, head, tail)
        self.draft_dirty = True
        self.selected = new_name
        self._recompute_skeleton_handles()
        return new_name

    @_undoable
    def skel_split(self, name: str) -> str:
        self._check_skeleton_cap(len(self.draft) + 1)
        new_name = skeleton.unique_name(self.draft, name)
        self.draft = skeleton.split_bone(self.draft, name, new_name)
        self.draft_dirty = True
        self.selected = new_name
        self._recompute_skeleton_handles()
        return new_name

    @_undoable
    def skel_remove_pivot(self, name: str) -> None:
        by_name = {b["name"]: b for b in self.draft}
        if name not in by_name:
            raise store.RigError(f"unknown bone {name!r}", field="name")
        parent_name = by_name[name]["parent"]
        if parent_name is None:
            children = [b["name"] for b in self.draft if b["parent"] == name]
            next_selected = children[0] if len(children) == 1 else None
        else:
            next_selected = parent_name
        self.draft = skeleton.remove_pivot(self.draft, name)
        self.draft_pairs = skeleton.prune_pairs(self.draft, self.draft_pairs)
        if self.draft_root == name and next_selected is not None:
            self.draft_root = next_selected
        self.draft_dirty = True
        self.selected = next_selected
        self._recompute_skeleton_handles()

    @_undoable
    def skel_remove_subtree(self, name: str) -> int:
        by_name = {b["name"]: b for b in self.draft}
        if name not in by_name:
            raise store.RigError(f"unknown bone {name!r}", field="name")
        parent_name = by_name[name]["parent"]
        count = self.subtree_size(name)
        self.draft = skeleton.remove_subtree(self.draft, name)
        self.draft_pairs = skeleton.prune_pairs(self.draft, self.draft_pairs)
        self.draft_dirty = True
        self.selected = parent_name
        self._recompute_skeleton_handles()
        return count

    @_undoable
    def skel_rename(self, old: str, new: str) -> None:
        # Pre-checked here, with ``field="name"``, ahead of
        # ``skeleton.rename_bone``'s own checks (``field="new"``): every other
        # skel_* refusal names the argument the *UI* labels "name", and a
        # rename dialog has exactly one field for the user to blame.
        if not skeleton.BONE_NAME_RE.match(new):
            raise store.RigError(f"bone name {new!r} is not usable", field="name")
        names = {b["name"] for b in self.draft}
        if new != old and new in names:
            raise store.RigError(f"duplicate bone name {new!r}", field="name")
        bones, pairs = skeleton.rename_bone(self.draft, self.draft_pairs, old, new)
        self.draft = bones
        self.draft_pairs = [list(p) for p in pairs]
        if self.draft_root == old:
            self.draft_root = new
        self.draft_dirty = True
        if self.selected == old:
            self.selected = new
        elif self.selected == f"{old}@tail":
            self.selected = f"{new}@tail"
        self._recompute_skeleton_handles()

    @_undoable
    def skel_attach_limb(
        self, preset_key: str, parent: str, side: str | None, mirror: bool
    ) -> list[str]:
        before = len(self.draft)
        bones, pairs = skeleton.attach_limb(
            self.draft, self.draft_pairs, preset_key, parent, side, mirror
        )
        self._check_skeleton_cap(len(bones))
        new_names = [b["name"] for b in bones[before:]]
        self.draft = bones
        self.draft_pairs = [list(p) for p in pairs]
        self.draft_dirty = True
        self.selected = new_names[0] if new_names else None
        self._recompute_skeleton_handles()
        return new_names

    def _first_child(self, parent: str) -> str | None:
        for b in self.draft:
            if b["parent"] == parent:
                return b["name"]
        return None

    def _mirror_partner(self, name: str) -> str | None:
        for a, b in self.draft_pairs:
            if a == name:
                return b
            if b == name:
                return a
        partner = skeleton.mirror_partner_name(name)
        names = {b["name"] for b in self.draft}
        return partner if partner in names else None

    def _set_draft_point(self, name: str, end: str, point: np.ndarray) -> None:
        by_name = {b["name"]: b for b in self.draft}
        bone = by_name.get(name)
        if bone is None:
            return
        bone[end] = [float(v) for v in point]
        if end == "head":
            # The tail-follows-first-child rule ``corrected_bones`` already
            # applies in joints mode, restated here as a write rather than a
            # read: a structural edit has no separate "corrected" view, so the
            # continuity has to land in ``draft`` itself.
            parent_name = bone.get("parent")
            if parent_name is not None and self._first_child(parent_name) == name:
                by_name[parent_name]["tail"] = [float(v) for v in point]

    def _move_skeleton_handle(self, key: str, world_pos: Any) -> None:
        if key is None:
            return
        name, end = (key[:-5], "tail") if key.endswith("@tail") else (key, "head")
        names = {b["name"] for b in self.draft}
        if name not in names:
            return
        point = self._gltf_to_skeleton(world_pos)
        self._set_draft_point(name, end, point)
        if self.mirror_edit:
            partner = self._mirror_partner(name)
            if partner is not None:
                mirrored = point.copy()
                mirrored[0] = -mirrored[0]
                self._set_draft_point(partner, end, mirrored)
        self.draft_dirty = True
        self.selected = key
        self._recompute_skeleton_handles()

    def _gltf_to_skeleton(self, world_pos: Any) -> np.ndarray:
        """glTF world space -> the draft's Blender space, through the anchor
        captured on ``enter_skeleton_mode``. The inverse of
        :meth:`_skeleton_to_gltf`."""
        p = np.asarray(world_pos, dtype="f8")
        return self._skel_anchor_blender + m3.gltf_delta_to_blender(
            p - self._skel_anchor_gltf
        )

    def _skeleton_to_gltf(self, point: Any) -> np.ndarray:
        """A draft point, Blender space -> glTF world space for drawing and
        picking. Anchored on the root the way :func:`ghost_handles` anchors on
        a rest pose rather than accumulating a transform per bone -- a new
        bone has no node to walk a parent chain from."""
        p = np.asarray(point, dtype="f8")
        return self._skel_anchor_gltf + m3.blender_delta_to_gltf(
            p - self._skel_anchor_blender
        )

    def _recompute_skeleton_handles(self) -> None:
        """Every draft bone's head, plus a leaf bone's own tail, in glTF world
        space. A non-leaf bone's tail gets no handle of its own: it is drawn as
        its first child's head, ``draft_segments``' rule, the same one
        ``corrected_bones`` applies in joints mode."""
        if self.mode != "skeleton":
            return
        parents = {b["parent"] for b in self.draft if b.get("parent")}
        handles: dict[str, np.ndarray] = {}
        for b in self.draft:
            handles[b["name"]] = self._skeleton_to_gltf(b["head"])
            if b["name"] not in parents:
                handles[f"{b['name']}@tail"] = self._skeleton_to_gltf(b["tail"])
        self.handles = handles


def _skin_bones(model: Model) -> list[str]:
    """Every joint node of every skin, in palette order.

    Palette order rather than name order: it is the order the rig was built in,
    so a marker list reads root-outwards the way a skeleton does.
    """
    names: list[str] = []
    for skin in model.skins:
        for index in skin.joints:
            name = model.nodes[index].name
            if name and name not in names:
                names.append(name)
    return names


def ghost_handles(
    model: Model, bones: list[str], rotations: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Where *bones* would sit under *rotations*, without moving anything.

    Onion-skinning needs a second skeleton on screen -- the key before this one,
    or the one after -- and the obvious way to get it is to pose the model, read
    the handles and pose it back. That is wrong here for two reasons and both
    bite in the frame loop: it runs every frame while a gizmo is being dragged,
    and ``_resync_handles`` would drag the *live* markers to the ghost's
    positions and back again between the read and the draw.

    So this is a **pure** forward-kinematic walk instead. It is deliberately the
    same walk ``Model.update_world`` makes -- roots down, ``parent @ local()``,
    with the same cycle and range guards, because a hand-supplied GLB reaches
    this path too -- with one substitution: a bone named in ``rotations`` uses
    that rotation in place of its own. Nothing is written back to any node.

    Bones the rotations do not name sit at their **rest** rotation rather than
    at whatever the live pose has them at, which is what makes a ghost the same
    thing ``apply_preset`` would show: a key lists only the bones it moves.
    """
    by_name = model.by_name
    override: dict[int, np.ndarray] = {}
    for name, quat in (rotations or {}).items():
        index = by_name.get(name)
        if index is not None:
            override[index] = np.asarray(quat, dtype="f8")
    wanted = {by_name[b]: b for b in bones if b in by_name}
    out: dict[str, np.ndarray] = {}
    seen: set[int] = set()
    stack = [(r, m3.identity()) for r in reversed(model.roots)]
    while stack:
        index, parent = stack.pop()
        if index in seen or not 0 <= index < len(model.nodes):
            continue
        seen.add(index)
        node = model.nodes[index]
        rotation = override.get(index)
        if rotation is None and index in wanted:
            # Not posed by this ghost, so it is at rest -- see the docstring.
            rotation = model.rest_rotations[index]
        local = (
            node.local()
            if rotation is None
            else m3.compose(node.translation, rotation, node.scale)
        )
        world = parent @ local
        name = wanted.get(index)
        if name is not None:
            out[name] = world[:3, 3].copy()
        for child in reversed(node.children):
            stack.append((child, world))
    return out
