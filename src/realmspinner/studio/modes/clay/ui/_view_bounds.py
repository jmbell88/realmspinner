"""Where the selection is: centres, world boxes and framing.

Split out of :mod:`~realmspinner.studio.modes.clay.ui.view` as pure code motion. Everything
here is memoized on the identity of what it reads -- the meshes, the selections
and the three transform arrays -- which is sound for the same reason the GPU
cache's key is: each of those is replaced whole rather than mutated, so a memo
misses exactly when the answer can change. These are asked every frame a gizmo
is on screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .view import ClayView

#: How close Clay lets the camera dolly to a framed selection, as a fraction of its
#: bounding radius (the viewer's own floor is 0.5).
CLOSEST_ZOOM = 0.02

#: Metres of room ``F`` leaves round a selected mesh-less bone's pivot.
BONE_FRAME_MARGIN = 0.5


class BoundsOps:
    """``ClayView``'s centres, bounds and framing. See the module docstring."""

    @staticmethod
    def _ancestor_uids(index: dict[int, Any], obj: Any) -> list[int]:
        """``doc.ancestors(obj.uid)`` read through *index* -- the same walk
        (nearest first; a dangling parent or a revisit ends it), no scan."""
        out: list[int] = []
        seen = {obj.uid}
        current = obj.parent
        while current is not None and current not in seen:
            parent = index.get(current)
            if parent is None:
                break
            out.append(current)
            seen.add(current)
            current = parent.parent
        return out

    # -- selection and framing ---------------------------------------------

    def selection_centre(self: ClayView, doc: Any) -> np.ndarray | None:
        """Where the gizmo goes: the object box's centre, or the elements'.

        In an element mode the gizmo belongs to what is selected *inside* the
        objects, not to the objects: putting it at the object's centre would
        leave a user dragging a handle two metres from the four vertices it
        moves, with the rotate ring pivoting about a point they never chose.
        """
        if doc.element_mode != "object":
            return self.element_centre(doc)
        if doc.pose_mode:
            return self._pose_centre(doc)
        lo, hi = self.world_bounds(doc, selected_only=True)
        if lo is not None:
            return (lo + hi) * 0.5
        # The 2026-10-14 audit, finding clay-view-01: a selected bone with no mesh (what Add
        # Bone makes) has no world box, so this answered None -- no gizmo drawn, G/R/S and F
        # silent -- while ``active_gizmo`` still offered a handle that was never placed, which
        # a press near the world origin grabbed and moved the bone with, unseen. Both manuals
        # tell the user to move that joint into place with the gizmo or G. A joint is
        # placed by its pivot, as in Pose mode.
        return self._pose_centre(doc)

    def selected_bone_box(self: ClayView, doc: Any) -> tuple[Any, Any] | None:
        """A small world box round the pivots of the selected, visible, mesh-less bones, or
        ``None`` -- what ``F`` frames for a joint, which has no box of its own (the 2026-10-14
        audit, finding clay-view-01). The padding is a bone-sized margin, so a lone joint is
        framed from a sensible distance rather than from the 0.1 mm the empty box would give."""
        index = self._uid_index(doc)
        pivots = [
            self._world(doc, o, index)[:3, 3]
            for o in doc.objects
            if o.bone and o.visible and o.uid in doc.selection and len(o.mesh.positions) == 0
        ]
        if not pivots:
            return None
        pad = np.full(3, BONE_FRAME_MARGIN)
        return np.min(pivots, axis=0) - pad, np.max(pivots, axis=0) + pad

    def element_centre(self: ClayView, doc: Any) -> np.ndarray | None:
        """The world centroid of every selected element, across every object.

        Memoized on the identities of everything it reads (B25): the selection
        objects, the meshes and the transform arrays are all replaced whole
        rather than mutated, so identity misses exactly when the answer can
        change -- and this is asked every frame a gizmo is on screen.
        """
        from .....kernels.mesh import elements as el
        from .....kernels.mesh.selection import _element_pickable

        # The drag skips a hidden object (``_begin_element_drag``), so this
        # centre must too, or hiding an object that still held an element
        # selection leaves the gizmo, and the pivot a rotate or scale turns
        # about, off the vertices that actually move. The eligibility is in the
        # key as well so hiding or un-hiding misses the memo.
        # The 2026-10-09 audit's clay-22: the key used to resolve each picked
        # object through ``doc.by_uid`` and ``doc.ancestors``, each a linear
        # scan, on every frame a gizmo is up -- picked x objects. One map per
        # call, as ``world_bounds`` builds, makes it one pass.
        index = self._uid_index(doc)
        key = (id(doc), tuple(
            (uid, id(sel), self._pickable(doc, uid, index), *self._obj_key(doc, uid, index))
            for uid, sel in doc.element_sel.items()
        ))
        memo = self._centre_memo
        if memo is not None and memo[0] == key:
            return memo[1]

        total = np.zeros(3)
        count = 0
        # Everything whose id() the key carries, held so those ids cannot be
        # recycled while the memo lives -- the ``_view_cache._Entry`` pin. A
        # freed selection or mesh whose address came back on a new one would
        # otherwise match a stale centre.
        pins: list[Any] = [doc]
        for uid, sel in doc.element_sel.items():
            pins.append(sel)
            obj = index.get(uid)
            if obj is None:
                continue
            pins.append(obj.mesh)
            if not _element_pickable(obj):
                continue
            # Every ancestor's own transform triple too, matching the key
            # above -- an id in the key is only sound while the array it
            # names is alive, and nothing else holds an ancestor's arrays
            # alive on this memo's behalf (the same reason ``view._world``
            # pins its own ancestor chain).
            for ancestor in (obj, *(index[u] for u in self._ancestor_uids(index, obj))):
                pins.extend((ancestor.translation, ancestor.rotation, ancestor.scale))
            verts = el.affected_verts(obj.mesh, sel)
            if not len(verts):
                continue
            matrix = self._world(doc, obj, index)
            local = obj.mesh.positions[verts].astype("f8")
            world = (matrix @ np.hstack([local, np.ones((len(local), 1))]).T).T[:, :3]
            total += world.sum(axis=0)
            count += len(world)
        centre = None if count == 0 else total / count
        self._centre_memo = (key, centre, tuple(pins))
        return centre

    @staticmethod
    def _pickable(doc: Any, uid: int, index: dict[int, Any] | None = None) -> bool:
        """Whether *uid*'s elements are the drag's to move -- ``selection``'s
        one eligibility. A missing uid is not."""
        from .....kernels.mesh.selection import _element_pickable

        if index is not None:
            obj = index.get(uid)
            return obj is not None and _element_pickable(obj)
        try:
            return _element_pickable(doc.by_uid(uid))
        except KeyError:
            return False

    @staticmethod
    def _obj_key(doc: Any, uid: int, index: dict[int, Any] | None = None) -> tuple:
        """The identity triple-plus-mesh that pins one object's geometry --
        **and every ancestor's own transform triple**, ``view._world``'s own
        reason exactly (that method's docstring): an object's ``translation``/
        ``rotation``/``scale`` are local to its parent, so moving a parent
        leaves a *child's* own three arrays untouched. Before this (the
        2026-09-26 audit's clay-view-02), :meth:`element_centre`'s memo keyed
        only on the selected object's own triple, so it kept matching -- and
        kept serving the child's pre-move centroid -- after a parent moved,
        with the gizmo and the drag pivot for a selected child both reading
        it.
        """
        if index is not None:
            obj = index.get(uid)
            if obj is None:
                return ()
            chain = [obj, *(index[u] for u in BoundsOps._ancestor_uids(index, obj))]
            return (
                id(obj.mesh),
                *(id(v) for o in chain for v in (o.translation, o.rotation, o.scale)),
            )
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            return ()
        chain = [obj, *(doc.by_uid(u) for u in doc.ancestors(uid))]
        return (id(obj.mesh), *(id(v) for o in chain for v in (o.translation, o.rotation, o.scale)))

    def world_bounds(self: ClayView, doc: Any, *, selected_only: bool = False) -> tuple[Any, Any]:
        """The world AABB over visible objects, or ``(None, None)`` if there are none.

        Memoized the way ``element_centre`` is (B27): the key names every
        object's visibility, selection membership, mesh identity and transform
        identities, so a miss happens exactly when the box can move.
        """
        # ``obj.parent`` is in the key: ``_world`` composes the ancestor chain
        # into every box, and a reparent that keeps the local arrays in place
        # (``keep_world=False``, and its undo) changes no id below -- the 2026-10-03
        # audit's clay-72 found the gizmo and ``F`` served the box measured under
        # the old parent until some unrelated transform edit invalidated the memo.
        # The parent's own entry carries its transform ids, and a grandparent's
        # reparent shows up as the parent's ``parent`` changing.
        key = (id(doc), doc.pose_mode, doc.pose_rev if doc.pose_mode else 0, tuple(
            (
                obj.uid, obj.visible, obj.uid in doc.selection, obj.parent,
                id(obj.mesh), id(obj.translation), id(obj.rotation), id(obj.scale),
            )
            for obj in doc.objects
        ))
        memo = self._bounds_memo.get(selected_only)
        if memo is not None and memo[0] == key:
            return memo[1]
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        found = False
        # The pin: see ``element_centre``. Every object whose ids the key names
        # is held alive for as long as the memo can match on them.
        pins: list[Any] = [doc]
        index = self._uid_index(doc)
        for obj in doc.objects:
            pins.extend((obj.mesh, obj.translation, obj.rotation, obj.scale))
            if not obj.visible or (selected_only and obj.uid not in doc.selection):
                continue
            box = self._object_world_box(doc, obj, index)
            if box is None:
                continue
            lo = np.minimum(lo, box[0])
            hi = np.maximum(hi, box[1])
            found = True
        out = (lo, hi) if found else (None, None)
        self._bounds_memo[selected_only] = (key, out, tuple(pins))
        return out

    def _object_world_box(
        self: ClayView, doc: Any, obj: Any, index: dict[int, Any] | None = None
    ) -> tuple[Any, Any] | None:
        """``ops.world_box`` over the object's mesh, and deliberately nothing else.

        It used to be a second copy of the same eight corners, which is how the
        properties panel's dimensions row and the camera's framing would have
        come to disagree about the size of one object.

        ``world=self._world(doc, obj)`` (tranche 3: scene structure): left at
        ``world_box``'s own default, this composed only *obj*'s own TRS, which
        for a parented object is local to its parent, not its world placement
        -- so a child's framing and dimensions row would have measured it as
        if it sat at the origin of its parent's frame rather than where it
        actually is. A root's world matrix is exactly its own local TRS (see
        ``document.py``'s module docstring), so an unparented document is
        measured exactly as it always was.

        ``index`` is :meth:`_uid_index`'s map, so a pass over every object
        resolves each ancestor chain without a document scan (clay-21).
        """
        from .....kernels.mesh import ops as bops

        world = self._world(doc, obj, index)
        skinned = self._skinned_positions(doc, obj, index)
        if skinned is not None and len(skinned.positions):
            # The 2026-10-14 audit, finding clay-view-02: in Pose mode a weighted mesh is drawn
            # skinned, so ``F`` and the gizmo frame the box it occupies, not its rest box.
            positions = skinned.positions
            lo, hi = positions.min(axis=0), positions.max(axis=0)
            corners = np.array(
                [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            )
            points = (world @ np.hstack([corners, np.ones((8, 1))]).T).T[:, :3]
            return points.min(axis=0), points.max(axis=0)
        return bops.world_box(obj, obj.mesh, world=world)

    def frame_selection(self: ClayView, doc: Any) -> float:
        """Put the selection -- or the whole document -- on screen.

        **Does not resize the grid.** Every other ``fit_grid`` caller (Mason's
        own ``frame_selection``, ``ClayView.render_png``) sizes the grid to
        whatever it just framed, because their grid follows the model. Clay's
        does not: ``grid_size`` is a user setting Task A gives its own field
        and its own persistence, and ``F`` shrinking a 100 m grid down to a
        1 m prop's footprint the moment it was framed was the bug the field
        exists to fix. ``BoundsOps`` is mixed into ``ClayView`` alone --
        ``MasonView`` keeps its own ``frame_selection`` -- so this is safe to
        change here without a second caller to consider.
        """
        lo, hi = self.world_bounds(doc, selected_only=bool(doc.selection))
        if lo is None:
            bones = self.selected_bone_box(doc)
            lo, hi = bones if bones is not None else self.world_bounds(doc)
        return self._frame_box(lo, hi)

    def frame_whole(self: ClayView, doc: Any) -> float:
        """Put every visible object on screen, whatever is selected.

        :meth:`frame_selection` with the selection ignored: the framing a shape
        placed off-screen asks for (``ClayState.frame_whole``), where the
        selection is the one new part and zooming onto it would push the rest
        of the model out of the picture.
        """
        lo, hi = self.world_bounds(doc)
        return self._frame_box(lo, hi)

    def _frame_box(self: ClayView, lo: Any, hi: Any) -> float:
        """The camera move both framings share. ``(None, None)`` is nothing to
        frame, and the camera stays where it is."""
        if lo is None:
            return 0.0
        # ``keep_angles``: F centres and sizes the selection along the direction
        # the user already chose, instead of throwing a lined-up front view back
        # to the opening three-quarter one. ``min_zoom``: the viewer's floor of
        # half the radius is too far out to place a vertex on a small detail.
        self.radius = self.camera.frame(lo, hi, keep_angles=True, min_zoom=CLOSEST_ZOOM)
        self.camera.set_target((lo + hi) * 0.5)
        return self.radius
