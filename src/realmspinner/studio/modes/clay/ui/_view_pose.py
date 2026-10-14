"""Pose mode in the viewport: the evaluated pose, skinned meshes, bone glyphs and bone picking.

Split out of :mod:`~realmspinner.studio.modes.clay.ui.view` the way the overlay and the picking
are. Four jobs, all of them reading the document's session pose state (``pose_mode``,
``rig_time``, ``live_pose``) and none of them writing the document.

**The pose is one set of arrays per pose change.** ``ClayView._world`` memoises on the *identity*
of the transform arrays up each chain, so the posed local transforms are built once per
``(rev, pose_rev)`` (:meth:`PoseOps._posed_arrays`) and handed out as the same arrays until the pose
moves: a held frame costs dictionary reads, and a moving one rebuilds only the bones the animation
drives. A shape *under* a bone follows it for free, because it composes through the posed parent.

**A weighted mesh is skinned on the CPU** when the pose changes
(:func:`~realmspinner.kernels.mesh.rig.skin_positions`) and written through the drag preview's own
vertex path, so no buffer is rebuilt per frame. Leaving Pose mode (or any change that rebuilds the
object's buffers) evicts those entries, and the next ``sync`` uploads the rest pose again.

**Bones draw as glyphs**, in every mode so a rig is visible while it is built: a small cross at
each pivot and a line to its parent bone (or, for a leaf, a short tick along the bone's own Y).
They are drawn over the model, because a joint hidden inside a mesh is a joint nobody can pick.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

import numpy as np

from .....kernels.mesh import pick as bp
from .....kernels.mesh import rig
from ....viewer.bonelines import ACTIVE, IDLE, BoneLines

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .view import ClayView

#: How close (screen pixels) a click must be to a bone's glyph to pick the bone.
PICK_RADIUS = 14.0
#: A glyph's cross, as a fraction of the rig's largest dimension (never below :data:`MIN_GLYPH`).
GLYPH_FRACTION = 0.03
MIN_GLYPH = 0.02
#: A leaf bone's tick, as a multiple of the glyph.
TICK = 3.0


class PoseOps:
    """``ClayView``'s pose-mode half. See the module docstring."""

    # -- the evaluated pose --------------------------------------------------------------------

    def _posed_arrays(self: ClayView, doc: Any) -> dict[int, tuple[Any, Any, Any]]:
        """``uid -> (translation, rotation, scale)`` for every bone the pose drives, stable
        until the pose or the document changes."""
        key = (doc.rev, doc.pose_rev)
        cached = self._pose_arrays
        if cached is not None and cached[0] == key and cached[1]() is doc:
            return cached[2]
        overrides = doc.pose_overrides()
        by_uid = self._uid_index(doc)
        out: dict[int, tuple[Any, Any, Any]] = {}
        for uid in overrides:
            obj = by_uid.get(uid)
            if obj is None:
                continue
            t, r, s = doc.posed_trs(obj, overrides)
            out[uid] = (
                np.array(t, dtype="f8", copy=True),
                np.array(r, dtype="f8", copy=True),
                np.array(s, dtype="f8", copy=True),
            )
        self._pose_arrays = (key, weakref.ref(doc), out)
        return out

    def _trs(self: ClayView, doc: Any, link: Any) -> tuple[Any, Any, Any]:
        """The transform arrays ``_world`` composes for *link*: posed in Pose mode, else its own."""
        if doc.pose_mode:
            posed = self._posed_arrays(doc).get(link.uid)
            if posed is not None:
                return posed
        return link.translation, link.rotation, link.scale

    # -- weighted meshes -----------------------------------------------------------------------

    def _pose_evict(self: ClayView, doc: Any) -> None:
        """Drop the skinned buffers of anything no longer posed -- *before* ``sync``, so the same
        frame uploads the rest geometry and the object is never missing for one."""
        if doc.pose_mode:
            return
        for uid in list(self._skin_stamps):
            del self._skin_stamps[uid]
            entry = self._cache.pop(uid, None)
            if entry is not None:
                entry.gpu.release()

    def _pose_skin(self: ClayView, doc: Any) -> None:
        """Skin every visible weighted mesh to the pose.

        Run after ``sync`` and before the composite is built. A weighted object whose buffers
        already show this pose is skipped; one whose entry was rebuilt (a palette change) or
        whose mesh was replaced is skinned again. :meth:`_pose_evict` is the way back to rest.
        """
        current: set[int] = set()
        if doc.pose_mode:
            overrides: dict | None = None
            by_uid = None
            stamp = (doc.rev, doc.pose_rev)
            for obj in doc.objects:
                if obj.skin is None or not obj.visible or obj.mesh.joints is None:
                    continue
                entry = self._cache.get(obj.uid)
                if entry is None or entry.mesh is not obj.mesh:
                    continue
                current.add(obj.uid)
                token = (id(entry), id(obj.mesh), stamp)
                if self._skin_stamps.get(obj.uid) == token:
                    continue
                if overrides is None:
                    overrides = doc.pose_overrides()
                    by_uid = self._uid_index(doc)
                palette = doc.skin_palette(obj, overrides, by_uid)
                positions = rig.skin_positions(
                    obj.mesh.positions,
                    obj.mesh.joints,
                    obj.mesh.weights,
                    palette,
                    doc._world_in(obj.uid, by_uid),
                )
                self._preview_positions(doc, obj.uid, positions)
                self._skin_stamps[obj.uid] = token
        # A weighted object hidden or deleted while posed is no longer stamped; ``sync`` has
        # already dropped its entry, so only the stamp needs to go.
        for uid in [u for u in self._skin_stamps if u not in current]:
            del self._skin_stamps[uid]

    # -- bone glyphs ---------------------------------------------------------------------------

    def _bone_draws(self: ClayView, doc: Any) -> list[Any]:
        """The bones as lines: a cross at each pivot, a line to the bone above, a tick on a leaf.

        Selected bones are drawn in the accent over the idle ones. Nothing is drawn for a
        document without bones, and the GL buffers are released the frame the last one goes.
        """
        bones = [obj for obj in doc.objects if obj.bone and obj.visible]
        if not bones:
            self._release_bones()
            return []
        index = self._uid_index(doc)
        world = {obj.uid: self._world(doc, obj, index) for obj in bones}
        pivots = np.array([world[b.uid][:3, 3] for b in bones])
        extent = float(np.ptp(pivots, axis=0).max()) if len(bones) > 1 else 0.0
        glyph = max(MIN_GLYPH, GLYPH_FRACTION * max(extent, 1.0))
        has_child: set[int] = set()
        parent_of: dict[int, int | None] = {}
        for b in bones:
            above = rig.bone_parent(index, b.uid)
            parent_of[b.uid] = above if above in world else None
            if parent_of[b.uid] is not None:
                has_child.add(parent_of[b.uid])
        positions: dict[str, Any] = {}
        pairs = {True: [], False: []}  # selected?, then idle
        for b in bones:
            matrix = world[b.uid]
            origin = matrix[:3, 3]
            name = str(b.uid)
            positions[name] = origin
            group = pairs[b.uid in doc.selection]
            for axis, label in ((0, "x"), (1, "y"), (2, "z")):
                direction = matrix[:3, axis]
                length = float(np.linalg.norm(direction))
                unit = direction / length if length > 1e-12 else np.eye(3)[axis]
                for sign, mark in ((1.0, "+"), (-1.0, "-")):
                    positions[f"{name}{label}{mark}"] = origin + sign * glyph * unit
                group.append((f"{name}{label}-", f"{name}{label}+"))
            above = parent_of[b.uid]
            if above is not None:
                group.append((str(above), name))
            if b.uid not in has_child:
                up = matrix[:3, 1]
                norm = float(np.linalg.norm(up))
                unit = up / norm if norm > 1e-12 else np.array([0.0, 1.0, 0.0])
                tail = origin + unit * glyph * TICK
                positions[f"{name}@tail"] = tail
                group.append((name, f"{name}@tail"))
        if self._bone_lines is None:
            self._bone_lines = (
                BoneLines(self.ctx, self.renderer.programs),
                BoneLines(self.ctx, self.renderer.programs),
            )
        idle_lines, active_lines = self._bone_lines
        alpha = 1.0 if doc.pose_mode else 0.7
        return [
            *idle_lines.draws(positions, pairs[False], None, colour=IDLE, alpha=alpha),
            *active_lines.draws(positions, pairs[True], None, colour=ACTIVE, alpha=alpha),
        ]

    def _release_bones(self: ClayView) -> None:
        if self._bone_lines is not None:
            for lines in self._bone_lines:
                lines.release()
            self._bone_lines = None

    # -- picking a bone ------------------------------------------------------------------------

    def pick_bone(self: ClayView, doc: Any, local: tuple[float, float]) -> int | None:
        """The bone a click in Pose mode means: the nearest glyph within :data:`PICK_RADIUS`
        pixels, else the nearest bone of the shape under the pointer, else ``None``.

        A bone with no mesh of its own has nothing to hit, so its glyph is what is aimed at;
        a click on a shape means "the bone that moves it", which is how a character is posed
        by grabbing its leg.
        """
        index = self._uid_index(doc)
        bones = [obj for obj in doc.objects if obj.bone and obj.visible]
        if bones:
            width, height = int(max(self._rect[2], 1)), int(max(self._rect[3], 1))
            self.camera.aspect = width / max(height, 1)
            points = np.array([self._world(doc, b, index)[:3, 3] for b in bones])
            screen = bp.project(
                points,
                np.eye(4),
                self.camera.projection() @ self.camera.view(),
                self.camera.position,
                width,
                height,
            )
            distance = np.hypot(screen.xy[:, 0] - local[0], screen.xy[:, 1] - local[1])
            distance = np.where(screen.front, distance, np.inf)
            nearest = int(np.argmin(distance))
            if distance[nearest] <= PICK_RADIUS:
                return bones[nearest].uid
        hit = self.pick(doc, local)
        return None if hit is None else rig.nearest_bone(index, hit)
