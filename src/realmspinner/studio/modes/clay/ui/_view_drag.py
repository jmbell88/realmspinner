"""The mouse and the keyboard: every gesture the Clay viewport understands.

Split out of :mod:`~realmspinner.studio.modes.clay.ui.view` as pure code motion. The mouse map
and the modifier rule are stated in that module's docstring; what lives here is
their implementation, plus the two halves of a live drag -- the object path,
which moves transforms in place and commits one history step for the whole drag
on release (the 2026-10-10 audit's clay-56: this header said "per object", which
cost 19 Ctrl+Z presses on a figure -- ``_commit_drag`` pushes one step), and the
element path, which previews by writing vertex buffers on the
GPU and never touches the document until the release.

:meth:`DragOps._narrow` is deliberately the **single** narrowing site for axis
locks and typed values, above both paths -- an invariant named in
``dev/INVARIANTS.md`` as ``ClayView._narrow``, which it still is: the class
that carries this mixin is ``ClayView``. The grid snap is the one narrowing that
is *not* there -- ``_apply``, ``_element_world_transform`` and (for a move of
selected elements, per corner) ``_preview_element_drag`` apply it after the
delta arrives, and stand down for a typed value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .....kernels.geom3d import math3d as m3
from .....kernels.mesh import document as bd
from .. import gestures

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .view import ClayView


@dataclass
class _ElementDrag:
    """One object's half of a live element drag.

    Everything is recorded at the press and nothing is re-read during the drag:
    ``before`` is the mesh the gizmo was grabbed over, ``verts`` the vertices it
    moves, ``local`` their object-space positions then, and ``matrix`` /
    ``inverse`` the object's placement. A drag that re-read the object every
    frame would compound its own output -- each frame applying the delta to the
    result of the last -- and drift away from the cursor.
    """

    before: Any
    verts: Any
    local: Any
    matrix: Any
    inverse: Any
    # The positions the last preview frame produced. The commit reads *this*
    # rather than the object's mesh, because the whole point of the preview is
    # that the object's mesh never moved -- reading it back would find the
    # press-time geometry and commit nothing at all.
    preview: Any = None


def _about(centre: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """*matrix* conjugated to act about *centre* rather than about the origin."""
    to = m3.identity()
    to[:3, 3] = -np.asarray(centre, dtype="f8")
    back = m3.identity()
    back[:3, 3] = np.asarray(centre, dtype="f8")
    return back @ matrix @ to


def _world_scale_on_local_axes(rotation: Any, factors: Any) -> np.ndarray | None:
    """Per-world-axis scale *factors* as the object's own per-local-axis factors.

    The 2026-10-09 second audit's clay-58: the gizmo and the HUD say ``X
    (global)``, but the factor was multiplied straight into the object's local
    scale, so a shape turned a quarter about Z stretched along world Y when
    world X was asked for. A scale about a world axis is ``diag(f) @ R @ S``,
    which is still a rotation times a scale only when ``R`` sends every local
    axis onto a world axis -- a multiple of 90 degrees about each -- and then
    the factor of the world axis a local axis lies along is that local axis's
    factor, exactly. Any other turn makes it a shear a TRS cannot hold, so
    this returns ``None`` and the caller refuses. A uniform factor commutes
    with every rotation and is returned as it came.
    """
    f = np.asarray(factors, dtype="f8").reshape(3)
    if float(np.ptp(f)) <= 1e-12:
        return f
    matrix = m3.quat_to_mat4(rotation)[:3, :3]
    out = np.empty(3)
    for local in range(3):
        column = np.abs(matrix[:, local])
        world = int(column.argmax())
        if float(column[world]) < 1.0 - 1e-4:
            return None
        out[local] = f[world]
    return out


def _rotation_hud(quat: Any, entry: Any) -> str:
    """The angle a rotation drag currently amounts to, in degrees.

    Read off the *constrained* quaternion rather than the raw one, so a locked
    axis reads as the angle about that axis and a typed value reads back as
    itself -- which is the whole reason the HUD is worth drawing.
    """
    from .....kernels.mesh import drag as bdrag

    out = bdrag.constrain_rotation(quat, entry)
    length = float(np.linalg.norm(np.asarray(out, dtype="f8")[:3]))
    angle = np.degrees(2.0 * float(np.arctan2(length, float(out[3]))))
    return bdrag.readout("rotate", np.array([angle]), entry)


class DragOps:
    """``ClayView``'s input handling and drags. See the module docstring."""

    # -- input -------------------------------------------------------------

    def handle_event(self: ClayView, doc: Any, event: Any, hovered: bool) -> bool:
        """Feed one pygame event. -> whether the viewport consumed it.

        ``hovered`` is whether the pointer is over the viewport image; a drag
        already in progress ignores it, so crossing onto a panel mid-orbit does
        not drop the drag.
        """
        import pygame

        # clay-15 (the 2026-09-23 audit, second run): this used to mark every
        # event dirty unconditionally, on the reasoning that any event
        # reaching the viewport can move the picture -- but a bare hover (no
        # button down, nothing dragging) reaches here on *every* mouse-move
        # frame and usually changes nothing the picture depends on, so that
        # blanket assignment defeated the redraw skip ``viewer_embed`` was
        # built for (2026-09-02). A press, a release and a wheel dolly always
        # change something, so those still mark dirty here; motion's own
        # dirtiness is decided in :meth:`_motion`, which knows whether a drag
        # is live or the hover state actually moved.
        local = self._local(event)
        # The 2026-10-07 audit's clay-12: a keyboard (G/R/S) or op drag has no
        # button held, so a *press* is how it ends -- but this branch read only
        # presses over the viewport image, and the shell routes a press on a
        # panel here with ``hovered`` false (the viewport takes the pointer
        # while a grab is live). The press on the header's Undo or the outliner
        # then ran against uncommitted displacement and the drag stayed live.
        # Left commits and right cancels wherever the pointer is; every other
        # press is still the hover rule's.
        ends_drag = self._grab in ("keydrag", "opdrag")
        if event.type == pygame.MOUSEBUTTONDOWN and (hovered or ends_drag):
            self._render_dirty = True
            # A click changes what is under the cursor (the selection, often the
            # mesh); the first motion after it must pick, not wait out the step.
            self._hover_element_at = None
            return self._press(doc, event.button, local)
        if event.type == pygame.MOUSEBUTTONUP:
            self._render_dirty = True
            if event.button == 3:
                return self._rmb_release(local)
            return self._release_drag(doc, event.button)
        if event.type == pygame.MOUSEMOTION:
            return self._motion(doc, local)
        if event.type == pygame.MOUSEWHEEL and hovered:
            self._render_dirty = True
            # Toward the pointer, so the point being looked at stays under it.
            self._hover_element_at = None  # the camera moved under a still pointer
            self.camera.dolly(
                event.y, cursor=local, size=(float(self._rect[2]), float(self._rect[3]))
            )
            return True
        return False

    def _press(self: ClayView, doc: Any, button: int, local: tuple[float, float]) -> bool:
        self._last_mouse = local
        # The 2026-10-03 audit's clay-74: one wheel notch delivers
        # ``MOUSEBUTTONDOWN`` (button 4 or 5) as well as ``MOUSEWHEEL``, and the
        # keyboard-drag branch below read every button but the left as a cancel
        # -- so zooming in mid-move, the normal way to place something
        # precisely, threw the whole drag away. The wheel is never a button
        # press for any grab here (the dolly is ``handle_event``'s own
        # ``MOUSEWHEEL`` branch), so it is out before any of them look. It is
        # reported as consumed only while a drag is live, which is what the
        # gizmo branch below already said for its own stray buttons. The middle
        # button deliberately still cancels a keyboard drag: it has no meaning
        # there, and a pan that began under a live drag would strand it.
        if button not in (1, 2, 3):
            return self.dragging
        # A keyboard drag has no button held, so a press is how it *ends*: the
        # left button commits it and the right cancels, which is Blender's
        # arrangement and the one a modeller's hand already knows.
        if self._grab in ("keydrag", "opdrag"):
            if button == 1:
                self._release_drag(doc, button=1)
            else:
                self.cancel_drag(doc)
            return True
        # A gizmo drag owns the mouse until its button comes up. Without this,
        # pressing the middle button mid-drag overwrote ``_grab`` with "pan",
        # so releasing the left button found nothing to commit and the drag was
        # stranded: the object stayed wherever the last motion put it, with no
        # history step and the gizmo still holding a live drag.
        #
        # The 2026-10-07 audit's clay-30: the marquee is the other grab whose
        # button-up *does* something (it applies the sweep), and it had no such
        # guard -- a middle press mid-sweep switched the grab to a pan, so the
        # left release found nothing to commit, the rectangle stayed on screen
        # and the sweep was dropped. Both are owned by the left button. Orbit
        # and pan are left out on purpose: their release commits nothing, and
        # letting a fresh press replace them is what unsticks a grab whose
        # button-up the window never delivered.
        if self._grab in ("gizmo", "marquee") and button != 1:
            return True
        if button == 3:
            self._rmb_at = local
            return True
        if button == 2:
            self._grab = "pan"
            return True
        if button != 1:
            return False

        self._begin_press(local)
        shift, ctrl, alt = self._mods()
        if alt:
            # Alt+drag always orbits, in every mode: it is the one gesture that
            # must never be reinterpreted, because it is how a user looks at
            # what they are about to click.
            self._grab = "orbit"
            return True

        origin, direction = self._ray(local)
        gizmo = self.active_gizmo(doc)
        axis = gizmo.hit(origin, direction) if gizmo is not None else None
        if axis is not None and gizmo.begin(axis, origin, direction):
            if not self._begin_gizmo_drag(doc):
                # Refused -- a toast is already showing why. ``gizmo.begin``
                # has already flipped the gizmo's own internal drag flag, so
                # it is put back rather than left claiming a drag is live
                # that this view never started.
                gizmo.end_drag()
            return True

        if doc.element_mode != "object":
            return self._press_element(doc, local, shift=shift, ctrl=ctrl)

        # The 2026-09-18 audit's clay-02: this used to call ``doc.select``
        # with just the new hit, so Shift and Ctrl were read into ``shift``/
        # ``ctrl`` above and then never consulted -- every click replaced the
        # whole selection and a multi-object op needed the Outliner. Combined
        # with ``doc.selection`` the same way ``_press_element`` combines with
        # an element selection, and the manual (30-clay.md:223) has always
        # promised Shift adds, Ctrl removes.
        #
        # A press on *nothing* no longer clears at the press: the clear used
        # to run here, so the first orbit drag over empty space deselected the
        # object the user was looking at. ``gestures.classify_press`` defers it
        # to the release, and only a release that never left the click slop
        # (``_run_click``) still clears.
        # Pose mode selects bones, not shapes: the nearest glyph, or the bone that moves
        # the shape under the pointer (``PoseOps.pick_bone``).
        hit = self.pick_bone(doc, local) if doc.pose_mode else self.pick(doc, local)
        plan = gestures.classify_press(
            element_mode=False,
            under="object" if hit is not None else "nothing",
            tool=self._tool(),
            shift=shift,
            ctrl=ctrl,
        )
        if plan.now == gestures.PICK:
            how = gestures.pick_how(shift, ctrl)
            if how == "replace":
                doc.select([hit])
            else:
                current = set(doc.selection)
                if how == "add":
                    current.add(hit)
                else:
                    current.discard(hit)
                doc.select(current)
        if plan.click == gestures.CLEAR:
            self._click = ("clear-objects",)
        self._grab = "orbit"
        return True

    def _tool(self: ClayView) -> str:
        return str(getattr(self.state, "tool", "select"))

    def _begin_press(self: ClayView, local: tuple[float, float]) -> None:
        """Forget the last press's click and start measuring this one's travel."""
        self._press_at = local
        self._moved = 0.0
        self._click = None

    def _run_click(self: ClayView, doc: Any, click: tuple[Any, ...]) -> None:
        """Do what a press that never left the click slop was waiting to do.

        Guarded against the document having moved on since the press (a mode
        key, a deleted object): an element click in object mode, or an object
        click in an element mode, is dropped rather than written into a state
        that has no place for it.
        """
        from .....kernels.mesh import elements as el

        kind = click[0]
        in_elements = doc.element_mode != "object"
        if kind == "clear-objects":
            if not in_elements:
                doc.select([])
            return
        if not in_elements:
            return
        # The clears below drop *picks* only: the shape being edited stays
        # selected, so a click on empty space no longer ends the edit.
        if kind == "clear-picks":
            doc.clear_element_sel()
        elif kind == "clear-surface":
            if any(o.uid == click[1] for o in doc.objects):
                doc.set_element_sel(click[1], el.empty())
        elif kind == "pick":
            _kind, uid, one, how = click
            if any(o.uid == uid for o in doc.objects):
                doc.set_element_sel(uid, el.combine(doc.element_sel_of(uid), one, how))

    def _begin_gizmo_drag(self: ClayView, doc: Any) -> bool:
        """Record every selected object's transform at the press. -> whether
        the drag may proceed.

        Not read per frame: that is what ``set_transform``'s ``was`` argument
        takes, and reading it live would compare a value against itself and
        record an empty step.

        The 2026-09-26 audit's clay-view-01: a selected object whose ancestor
        is *also* selected used to be dragged twice over -- ``_apply`` reads
        the parent's world matrix fresh every call, so once the ancestor's
        own turn in :meth:`_drag_gizmo`'s loop had moved it, the descendant's
        ``before_world`` already carried that move, and applying the same
        delta to the descendant on top of it added the delta a second time
        (reproduced: a child's world delta of 10 against the dragged parent's
        5). Only what actually gets a per-object delta applied drops a uid
        with a selected ancestor, because moving that ancestor already carries
        it along.
        """
        # One uid -> object map for the press (the 2026-10-07 audit's clay-21):
        # the ancestor test and the snapshot below each resolved every selected
        # uid with a linear ``index_of`` scan.
        index = self._uid_index(doc)
        # In an element mode the drag moves what is *picked*: the edited shape
        # stays selected with nothing picked, and a drag keyed on ``selection``
        # would have started over no elements at all.
        held = doc.selection if doc.element_mode == "object" else doc.element_sel
        uids = [o.uid for o in doc.objects if o.uid in held]
        if doc.element_mode == "object":
            # The 2026-10-07 audit's clay-06: hiding an object (the outliner's
            # eye, the Properties box) leaves it selected, and this moved it
            # anyway -- unseen, since the gizmo and the bounds skip it. The
            # element path has always dropped a hidden object
            # (``_element_pickable``); a hidden one is not the user's to drag.
            # The ancestor test below reads the *visible* set: a hidden
            # parent is not moved, so it carries nothing and its visible
            # child has to move itself.
            uids = [uid for uid in uids if index[uid].visible]
            # A weighted mesh's own transform is ignored when it is exported, so a drag
            # that moved it would show a placement no file keeps: its bones are what move
            # it. Dropped with a sentence, the way a hidden object is dropped silently
            # above because that one needs no explaining.
            weighted = [uid for uid in uids if index[uid].skin is not None]
            if weighted:
                names = ", ".join(repr(index[uid].name) for uid in weighted)
                self._toast(
                    f"{names} is weighted to a skeleton, so it moves with its bones. "
                    "Move a bone instead."
                )
                uids = [uid for uid in uids if uid not in weighted]
            if not uids:
                return False
            selected = set(uids)
            uids = [
                uid
                for uid in uids
                if selected.isdisjoint(self._ancestor_uids(index, index[uid]))
            ]
        self._drag_uids = uids
        self._drag_start = {
            uid: tuple(np.array(v, copy=True) for v in index[uid].trs())
            for uid in self._drag_uids
        }
        self._drag_quat = m3.quat_identity()
        self._scale_axis_refused = False
        gizmo = self.active_gizmo(doc)
        origin = None if gizmo is None else getattr(gizmo, "origin", None)
        self._drag_origin = np.zeros(3) if origin is None else np.array(origin, dtype="f8")
        self._grab = "gizmo"
        from .....kernels.mesh import drag as bdrag

        self.drag_input = bdrag.DragInput()
        self.drag_hud = ""
        if doc.element_mode != "object":
            self._begin_element_drag(doc)
        return True

    def _toast(self: ClayView, message: str) -> None:
        """A refusal, shown -- or silently dropped when there is nothing to
        show it to.

        ``self.app_ctx`` is optional (a headless view, which is most of this
        package's own test suite -- ``ClayView.state``'s own docstring states
        the same tolerance), so this checks for a ``toast`` method rather
        than assuming one. Routed through :func:`~realmspinner.studio.modes.clay.ops.toast` rather
        than a bare ``ctx.toast(message, "error")`` written out here, since
        that is the one function every other refusal in this mode already
        goes through -- a second copy here is a second place its shape
        (``ctx.toast(text, level)``) could drift from.
        """
        ctx = self.app_ctx
        if ctx is None or not hasattr(ctx, "toast"):
            return
        from .. import ops as clay_ops

        clay_ops.toast(ctx, message)

    def begin_keyboard_drag(self: ClayView, doc: Any, kind: str) -> bool:
        """``G``/``R``/``S``: start a transform with no handle grabbed.

        -> whether one started. The modeller's gesture, and the one Clay had no
        route to at all: every transform went through grabbing a coloured arrow,
        which means finding it, which means the object is never moved without
        first looking at the gizmo rather than at the model.

        Measured on the plane through the pivot perpendicular to the camera,
        which is what "move it where the mouse goes" means when no axis has been
        chosen -- and the same plane all three kinds measure on, so ``X`` mid-
        drag narrows a translation, a rotation and a scale by the one rule in
        :meth:`_narrow` rather than three.

        It reuses ``_begin_gizmo_drag`` outright: the snapshot, the pivot and
        the fresh ``DragInput`` are the same three things a handle drag needs,
        and a second copy of them is a second place for a cancel to restore the
        wrong values from.
        """
        if self._grab is not None or not doc.selection:
            return False
        if doc.element_mode != "object" and not doc.element_sel:
            return False
        centre = self.selection_centre(doc)
        if centre is None:
            return False
        if not self._begin_gizmo_drag(doc):
            # Refused -- a toast is already showing why; ``_grab`` was never
            # set, so there is nothing here to put back.
            return False
        # ``_begin_gizmo_drag`` reads the *gizmo's* origin, which is only placed
        # once a frame has drawn it -- so the pivot is taken from the selection
        # directly, which is the same point the gizmo is placed at.
        self._drag_origin = np.asarray(centre, dtype="f8")
        anchor = self._view_plane_point(self._last_mouse, self._drag_origin)
        if anchor is None:
            self._grab = None
            return False
        self._grab = "keydrag"
        self._key_kind = kind
        self._key_anchor = anchor
        return True

    def begin_extrude_drag(self: ClayView, doc: Any, mark: int, before: Any) -> bool:
        """Start the drag an ``E`` extrude hands straight over to. -> whether one began.

        ``mark`` is the history mark taken **before** the extrude ran and
        ``before`` the element mode and selection it started from: commit folds
        the extrude and the drag into the one step it reads as, and Esc undoes
        the extrude too (the drag never wrote a step of its own to put back, so
        what is left to cancel *is* the extrude). On faces the move is locked to
        the average face normal -- the direction an extrusion is for -- and an
        axis key or a mid-drag G/R/S releases it; edges and vertices have no
        such direction and move freely.
        """
        if not self.begin_keyboard_drag(doc, "move"):
            return False
        self._extrude_gesture = (mark, before)
        normal = self._selection_normal(doc) if doc.element_mode == "face" else None
        if normal is not None:
            self.drag_input.axis = "normal"
            self.drag_input.normal = normal
        return True

    def _selection_normal(self: ClayView, doc: Any) -> np.ndarray | None:
        """The unit average normal of the selected faces, in world space, or ``None``."""
        from .....kernels.mesh import mesh as bm
        from .....kernels.mesh.selection import _element_pickable

        total = np.zeros(3)
        index = self._uid_index(doc)
        for uid, sel in doc.element_sel.items():
            if not len(sel.faces):
                continue
            obj = index.get(uid)
            # The 2026-10-07 audit's clay-11: a hidden object keeps its element
            # selection, and its faces pulled the extrude's normal lock toward
            # geometry the drag does not move (``_begin_element_drag`` drops
            # it). The same eligibility the pick doors and the drag use.
            if obj is None or not _element_pickable(obj):
                continue
            normals = bm.face_normals(obj.mesh)[sel.faces].sum(axis=0)
            # A normal turns by the inverse transpose of the placement, which is
            # not the placement itself under a non-uniform scale.
            matrix = np.asarray(self._world(doc, obj, index), dtype="f8")[:3, :3]
            try:
                total += np.linalg.inv(matrix).T @ normals
            except np.linalg.LinAlgError:
                continue
        length = float(np.linalg.norm(total))
        return None if length < 1e-9 else total / length

    def _fold_extrude(self: ClayView, doc: Any, *, commit: bool) -> None:
        """End an extrude gesture: one step on commit, nothing at all on cancel."""
        gesture, self._extrude_gesture = self._extrude_gesture, None
        if gesture is None:
            return
        from .. import recent_op

        mark, before = gesture
        history = doc.history
        history.collapse_since(mark)
        if commit:
            top = history.top
            if top is not None:
                top.label = "Extrude"
            return
        history.undo(doc, redoable=False)
        # Selection is not undoable, so the undo leaves it on the extrusion's own
        # faces -- indices that mean nothing on the mesh it just put back.
        recent_op.restore(doc, *before)

    def _view_axis(self: ClayView) -> Any:
        """The unit vector from the scene toward the eye, along the **view axis**.

        The camera's own back axis, read off its view matrix, not the line from
        the pivot to the eye. The 2026-10-10 audit's clay-02: G/R/S measured on
        the plane whose normal was ``camera.position - pivot``, so an off-centre
        pivot (3 m aside at distance 5) put the plane 31 degrees off the screen
        and a purely horizontal mouse move also moved the object along the view
        axis -- z moved +0.2485 for 40 px in an orthographic front view, where
        the shift is invisible until the user orbits. With a centred pivot the
        two coincide, which is why nothing noticed.
        """
        back = np.asarray(self.camera.view(), dtype="f8")[2, :3]
        return back / float(np.linalg.norm(back))

    def _view_plane_point(
        self: ClayView, local: tuple[float, float], centre: Any
    ) -> Any:
        """Where the pointer's ray meets the plane through ``centre`` that faces
        the camera -- perpendicular to the view axis (:meth:`_view_axis`), so
        screen-space motion stays in screen space. ``None`` when the ray runs
        parallel to it, which the caller treats as "no movement" rather than as
        an error."""
        from ....viewer import picking

        origin, direction = self._ray(local)
        return picking.ray_plane(
            origin, direction, np.asarray(centre, dtype="f8"), self._view_axis()
        )

    def _drag_keyboard(self: ClayView, doc: Any, local: tuple[float, float]) -> None:
        """One frame of a keyboard drag: the same delta shapes a gizmo makes.

        Deliberately the same three shapes -- a 3-vector for a move, a 4-vector
        quaternion for a rotate, a 3-vector of factors for a scale -- so
        everything downstream (``_narrow``, ``_accumulate``, ``_apply``, the
        element path) is the code that already exists rather than a parallel
        set that has to be kept agreeing with it.
        """
        from .....kernels.mesh import drag as bdrag

        point = self._view_plane_point(local, self._drag_origin)
        if point is None or self._key_anchor is None:
            return
        pivot = np.asarray(self._drag_origin, dtype="f8")
        anchor = np.asarray(self._key_anchor, dtype="f8")
        state = self.state
        if self._key_kind == "rotate":
            # About the view axis, not the eye-to-pivot line (clay-02): the
            # drag is measured on the plane perpendicular to it.
            axis = self._view_axis()
            angle = bdrag.screen_angle(pivot, axis, anchor, point)
            delta = m3.quat_from_axis_angle(axis, angle)
        elif self._key_kind == "scale":
            was = float(np.linalg.norm(anchor - pivot))
            now = float(np.linalg.norm(point - pivot))
            if was < 1e-9:
                return
            factor = now / was
            delta = np.array([factor, factor, factor], dtype="f8")
        else:
            # A translation is reported as the *destination*, not the
            # displacement: that is what the translate gizmo hands back and
            # what ``_apply`` subtracts ``_drag_origin`` from.
            delta = pivot + (point - anchor)
        delta = self._narrow(doc, delta, state, local)
        if doc.element_mode != "object":
            self._queue_element_preview(delta, state)
            return
        self._apply_to_dragged(doc, delta, state)

    def _apply_to_dragged(self: ClayView, doc: Any, delta: Any, state: Any) -> None:
        """One motion frame over every dragged object -- the loop the gizmo and
        the keyboard drag share.

        The map is built once per frame and handed to :meth:`_apply`: the
        2026-10-07 audit's clay-21 measured this loop at 918 ms a frame with
        4,096 objects selected, every iteration resolving the object, its
        parent's world matrix and its local placement by linear scans.
        """
        index = self._uid_index(doc)
        refused = False
        applied: list[Any] = []
        for uid, was in self._drag_start.items():
            obj = index.get(uid)
            if obj is None:
                continue  # deleted mid-drag
            if not self._apply(doc, obj, was, delta, state, index, applied):
                refused = True
        if refused:
            # All or none: a selection half-scaled because one member was
            # turned off-axis is a document the user never asked for. Put
            # every object back where the press found it, and say why once
            # per drag rather than on every mouse-move.
            for uid, was in self._drag_start.items():
                obj = index.get(uid)
                if obj is not None:
                    obj.translation, obj.rotation, obj.scale = (
                        np.array(v, copy=True) for v in was
                    )
            if not self._scale_axis_refused:
                self._scale_axis_refused = True
                self._toast(
                    "Scale along a world axis needs an unrotated shape; use the Size "
                    "card or unlock the axis."
                )
        if applied and not refused and self._snap_live(state):
            # The 2026-10-10 audit's clay-55: ``_narrow`` quotes the pointer's
            # displacement, but the grid snaps the object's *position*, so the
            # line under the viewport said 0.37 m while the object landed on
            # 0.375. The first dragged object's own landed displacement is the
            # quantity actually applied.
            from .....kernels.mesh import drag as bdrag

            self.drag_hud = bdrag.readout(
                self._drag_kind(state), applied[0], self.drag_input
            )
        doc.touch()

    def _press_element(
        self: ClayView, doc: Any, local: tuple[float, float], *, shift: bool, ctrl: bool
    ) -> bool:
        """LMB in an element mode: pick, clear, orbit or start a marquee.

        Read off what is under the cursor -- an element, else a *surface* (the
        user clicked the object and missed everything on it), else nothing --
        and handed to :func:`~..gestures.classify_press`, which owns the rule:
        a plain drag orbits, a click picks or clears, Shift+drag sweeps a box
        that adds and Ctrl+drag one that subtracts. This used to start a
        marquee on every plain press over empty space under the Select tool, so
        a first-time user dragging the void to look around drew a rectangle and
        lost their picks. A plain press on an element still picks at once; a
        Shift or Ctrl press on one waits for the release, so the box can start
        on the model.
        """
        how = gestures.pick_how(shift, ctrl)
        hit = self.pick_face(doc, local)
        picked = self.pick_element(doc, local, hit)
        under = "element" if picked is not None else ("surface" if hit is not None else "nothing")
        plan = gestures.classify_press(
            element_mode=True, under=under, tool=self._tool(), shift=shift, ctrl=ctrl
        )
        one = None
        if picked is not None:
            uid, index = picked
            one = self.element_sel_for(doc, uid, index)
        if plan.now == gestures.PICK:
            from .....kernels.mesh import elements as el

            if how == "replace":
                # A replace-click makes the picked object the one being edited:
                # ``select`` drops every other object's picks (and the object
                # itself, if another was being edited), where this used to
                # clear the picks and leave ``selection`` to be re-derived.
                doc.select([uid])
            doc.set_element_sel(uid, el.combine(doc.element_sel_of(uid), one, how))
        if plan.click == gestures.PICK:
            self._click = ("pick", uid, one, how)
        elif plan.click == gestures.CLEAR:
            self._click = ("clear-surface", hit.uid) if hit is not None else ("clear-picks",)
        if plan.drag == gestures.ORBIT:
            self._grab = "orbit"
        else:
            self._grab = "marquee"
            self._marquee_from = local
            self._marquee_add = "add" if plan.drag == gestures.MARQUEE_ADD else "subtract"
            self.marquee = (local[0], local[1], local[0], local[1])
        return True

    def _release_drag(self: ClayView, doc: Any, button: int = 1) -> bool:
        if self._grab is None:
            return False
        # The press half's guard, mirrored: a grab is owned by the button that
        # began it -- pan by the middle button, everything else by the left --
        # and only that button's release ends it. Without this an MMB release
        # or a wheel tick (buttons 4/5) mid-LMB-drag committed the gizmo drag
        # early, and the real LMB-up then found no grab to commit.
        owner = 2 if self._grab == "pan" else 1
        if button != owner:
            return True
        # Drawn while ``_grab`` still says which transform this is, so the committed
        # position is the one under the pointer and not a frame behind it.
        if self._grab in ("gizmo", "keydrag"):
            self.flush_drag_preview(doc)
        was, self._grab = self._grab, None
        click, self._click = self._click, None
        if was in ("gizmo", "keydrag"):
            self._settle_drag_tail(doc, was)
        elif was == "opdrag":
            self._drop_pending_preview()
            self._op_drag_commit(doc)
        elif was == "marquee":
            if click is not None and gestures.is_click(self._moved):
                # A Shift or Ctrl press that never became a sweep: the click
                # it was waiting to be, with the rectangle dropped unread.
                self.marquee, self._marquee_from = None, None
                self._run_click(doc, click)
            else:
                self._commit_marquee(doc)
        elif was == "orbit" and click is not None and gestures.is_click(self._moved):
            self._run_click(doc, click)
        return True

    def _settle_drag_tail(self: ClayView, doc: Any, was: str) -> None:
        """The commit shared by a real release (:meth:`_release_drag`) and a
        settle with no release at all (:meth:`settle_drag`) -- pulled out
        rather than duplicated so the two can never quietly disagree.

        A keyboard drag holds no handle, so there is no gizmo drag to end --
        but everything after that is identical, which is the whole point of
        routing it through the same delta shapes.

        The callers flush the queued preview before they clear ``_grab``
        (:meth:`flush_drag_preview`), not here: ``_is_scale`` reads the grab, so a
        preview drawn after it was cleared would be the wrong transform.
        """
        gizmo = self.active_gizmo(doc) if was == "gizmo" else None
        if gizmo is not None:
            gizmo.end_drag()
        if doc.element_mode != "object":
            self._commit_element_drag(doc)
        else:
            self._commit_drag(doc)
        self._fold_extrude(doc, commit=True)
        self._clear_drag_input()
        self._end_keyboard_drag()

    def settle_drag(self: ClayView, doc: Any) -> bool:
        """Commit a live transform drag right now, against *doc*. -> whether
        one was live.

        For the one case a release or an Esc cannot cover: nothing the user
        did asked the drag to end, so there is no button-up and no key to
        route through ``_release_drag``/``cancel_drag``. The 2026-09-15
        audit's clay-01: switching the active Clay tab mid-drag left a live
        G/R/S drag's TRS written onto the object with nothing to record it,
        so the eventual mouse-up or Esc committed or cancelled against
        whichever document had *become* active rather than the one the drag
        began on. ``ClayState.activate``'s ``settle_drag`` hook calls this
        against the tab being left, before ``active_uid`` moves.
        """
        if self._grab == "opdrag":
            # Leaving the tab mid op drag commits it, as it does a G/R/S drag.
            self._grab = None
            self._drop_pending_preview()
            self._op_drag_commit(doc)
            return True
        if self._grab == "marquee":
            return self.cancel_marquee()
        if self._grab not in ("gizmo", "keydrag"):
            return False
        self.flush_drag_preview(doc)
        was, self._grab = self._grab, None
        self._settle_drag_tail(doc, was)
        return True

    def cancel_marquee(self: ClayView) -> bool:
        """Drop a live marquee without selecting anything. -> whether one was live.

        The 2026-10-07 audit's clay-85: a marquee is a grab but not a transform
        drag (``dragging``), so Ctrl+Tab and Ctrl+W were not held back by it and
        the eventual button-up swept the old rectangle into whichever document
        had become active. A tab switch or close calls this against the view,
        beside ``settle_drag``/``cancel_drag``. Nothing to commit: the rectangle
        is a gesture, not an edit.
        """
        if self._grab != "marquee":
            return False
        self._grab = None
        self.marquee = None
        self._marquee_from = None
        self._click = None
        self._render_dirty = True
        return True

    # -- the keyboard's half of a drag (Clay18) ------------------------------

    @property
    def dragging(self: ClayView) -> bool:
        """Whether a transform drag is live. What the key handler checks *first*.

        First because the number row is bound to the element modes: a ``1``
        typed into a drag has to be a digit, not a jump into vertex mode
        halfway through moving something.

        Both kinds count. A gizmo drag is a handle held with the mouse down; a
        **keyboard drag** is ``G``/``R``/``S`` with the mouse merely moving, and
        every rule that applies during one applies during the other -- the axis
        lock, the typed value, Esc to cancel. They differ in how they end, and
        nowhere else: a gizmo drag ends when its button comes up, and a keyboard
        drag has no button held, so it ends on a *press*.

        An op drag (``opdrag``) counts too: ``clay_mode.handle_key``'s own
        ``getattr(view, "dragging", False)`` gate is what routes a bare Esc to
        :meth:`cancel_drag` and swallows every other key rather than letting
        it fire a second op mid-drag.
        """
        return self._grab in ("gizmo", "keydrag", "opdrag")

    def _clear_drag_input(self: ClayView) -> None:
        from .....kernels.mesh import drag as bdrag

        self.drag_input = bdrag.DragInput()
        self.drag_hud = ""

    def _end_keyboard_drag(self: ClayView) -> None:
        self._key_kind = ""
        self._key_anchor = None

    def drag_key(self: ClayView, doc: Any, name: str) -> bool:
        """Feed one key name to the live drag. -> whether it was consumed.

        The transform is re-applied immediately rather than waiting for the next
        mouse move, because a typed value with the mouse held still is the whole
        point of typing one. Re-applying is safe because every gizmo's
        ``update`` is a function of the ray it is handed and the press it began
        from: ``RotateGizmo`` returns a zero increment for an unchanged ray,
        and the other two are measured from the press outright.
        """
        if not self.dragging:
            return False
        # ``G``/``R``/``S`` mid-drag switch which transform is running, which is
        # Blender's and is what makes "move it, no -- rotate it" one gesture
        # rather than a cancel and a restart. Only during a keyboard drag: a
        # handle drag is holding a specific arrow, and switching under it would
        # leave the gizmo's own drag state describing a transform nobody is
        # doing any more.
        if self._grab == "opdrag":
            # Digits and the minus sign type the value outright; an axis letter
            # is accepted by ``DragInput`` and means nothing to an op drag.
            if not self.drag_input.key(name):
                return False
            self._op_drag_update(doc)
            return True
        switch = {"g": "move", "r": "rotate", "s": "scale"}.get(name)
        if switch is not None and self._grab == "keydrag":
            self._restart_keyboard_drag(doc, switch)
            return True
        if not self.drag_input.key(name):
            return False
        if self._grab == "keydrag":
            self._drag_keyboard(doc, self._last_mouse)
        else:
            self._drag_gizmo(doc, self._last_mouse)
        return True

    def _restart_keyboard_drag(self: ClayView, doc: Any, kind: str) -> None:
        """Switch a live keyboard drag to another transform.

        The objects are put back first, so the new transform is measured from
        where they *started* rather than from wherever the abandoned one left
        them -- a rotate that began from a half-finished move would carry that
        move into its result and there would be no way to undo one without the
        other.
        """
        if self._key_kind == kind:
            return
        # The abandoned transform's newest position must not be drawn over the
        # zero-delta preview below.
        self._drop_pending_preview()
        for uid, was in self._drag_start.items():
            try:
                obj = doc.by_uid(uid)
            except KeyError:
                continue
            obj.translation, obj.rotation, obj.scale = (
                np.array(v, copy=True) for v in was
            )
        # And the overlays, for ``cancel_drag``'s reason: the objects have been
        # put back, and an overlay still holding the abandoned transform's
        # positions is the new transform measured against a picture of the old
        # one.
        self._restore_overlays(doc, list(self._drag_start))
        # The 2026-09-06 audit found the GPU vertex buffer left one frame
        # stale here: an element drag never touches ``obj.translation`` above,
        # only ``_preview_element_drag``'s per-vertex write, so switching kind
        # mid-drag restored the overlay to the mesh's own positions while the
        # drawn mesh itself still showed the abandoned transform's preview
        # until the next mouse-move or typed digit. A zero-delta preview
        # rewrites the buffer the same frame the overlay is restored.
        for uid, drag in self._element_drags.items():
            drag.preview = None
            self._preview_positions(doc, uid, np.array(drag.before.positions, dtype="f4"))
        self._key_kind = kind
        # The 2026-10-09 second audit's clay-123: a refused typed scale is a
        # fact about the *scale* that was running, and S, a typed -1, then G
        # committed the move with its "scaling refused" toast because nothing
        # here cleared it. Whatever the new transform is, it starts unrefused.
        self._element_scale_refused = False
        self._scale_axis_refused = False
        self._key_anchor = self._view_plane_point(self._last_mouse, self._drag_origin)
        self._clear_drag_input()
        doc.touch()

    def _restore_overlays(self: ClayView, doc: Any, uids: Any) -> None:
        """Put the selection overlays back on the mesh's own positions.

        The preview writes moved vertices straight into each overlay's VBO
        (``_preview_element_drag``), and the overlay is keyed on ``id(mesh)`` --
        which does not change during a drag -- so an abandoned drag left the
        wireframe, the vertex dots and the edge lines at the previewed
        positions while the *mesh* was back where it started. It looked like
        the cancel had half worked, and stayed that way until something else
        rebuilt the overlay.
        """
        overlays = getattr(self, "_overlays", {})
        for uid in uids:
            overlay = overlays.get(uid)
            if overlay is None:
                continue
            try:
                obj = doc.by_uid(uid)
            except KeyError:
                continue
            overlay.write_positions(obj.mesh.positions)

    def cancel_drag(self: ClayView, doc: Any) -> bool:
        """Esc during a drag: put everything back and record nothing.

        Distinct from a release, which commits. Without it Esc cleared the
        pane's own drag bookkeeping and left the view still holding the grab,
        so the objects stayed wherever the last motion put them with no history
        step to take them back.

        """
        if self._grab == "opdrag":
            self._drop_pending_preview()
            self._op_drag_cancel(doc)
            return True
        if not self.dragging:
            return False
        self._drop_pending_preview()
        self._grab = None
        gizmo = self.active_gizmo(doc)
        if gizmo is not None:
            gizmo.end_drag()
        if doc.element_mode != "object":
            drags, self._element_drags = self._element_drags, {}
            for uid in drags:
                entry = self._cache.pop(uid, None)
                if entry is not None:
                    entry.gpu.release()
            self._restore_overlays(doc, drags)
            self._fold_extrude(doc, commit=False)
        else:
            for uid, was in self._drag_start.items():
                try:
                    obj = doc.by_uid(uid)
                except KeyError:
                    continue
                obj.translation, obj.rotation, obj.scale = (np.array(v, copy=True) for v in was)
            self._drag_uids = []
            self._drag_start = {}
        self._clear_drag_input()
        self._end_keyboard_drag()
        doc.touch()
        return True

    def _commit_marquee(self: ClayView, doc: Any) -> None:
        """Apply the swept rectangle. A rectangle with no area changes nothing.

        A zero-area marquee used to be a click on empty space and cleared the
        selection, because a plain press there started a marquee. A marquee is
        now Shift's or Ctrl's alone (``gestures.classify_press``), and a Shift
        press that never left the click slop is handled by the release before
        this is reached; a sweep that is merely thin must not wipe the picks
        it was asked to add to or take from.
        """
        from .....kernels.mesh import elements as el
        from .....kernels.mesh import pick as bp
        from .....kernels.mesh.adjacency import adjacency

        rect, self.marquee, self._marquee_from = self.marquee, None, None
        if rect is None:
            return
        # The 2026-10-07 audit's clay-62: the ``4`` key is not a drag and so is
        # not gated by ``dragging``; pressed mid-sweep it left object mode with
        # a marquee still live, and the release then wrote an *element*
        # selection into a document that has no element mode -- which the next
        # switch back to an element mode would show as a selection nobody made.
        if doc.element_mode == "object":
            return
        if abs(rect[2] - rect[0]) < 2.0 and abs(rect[3] - rect[1]) < 2.0:
            return

        mode = doc.element_mode
        replacing = self._marquee_add == "replace"
        if replacing:
            doc.clear_element_sel()
        received: list[int] = []
        index = self._uid_index(doc)
        for obj in doc.objects:
            if not obj.visible:
                continue
            screen = self.screen_of(doc, obj.uid, index)
            if mode == "vertex":
                swept = el.ElementSel(verts=bp.marquee_verts(screen, rect))
            elif mode == "edge":
                swept = el.ElementSel(
                    edges=bp.marquee_edges(screen, adjacency(obj.mesh).edge_verts, rect)
                )
            else:
                swept = el.ElementSel(
                    faces=bp.marquee_faces(screen, obj.mesh.loops, obj.mesh.starts, rect)
                )
            if el.is_empty(swept):
                continue
            how = "subtract" if self._marquee_add == "subtract" else "add"
            doc.set_element_sel(
                obj.uid, el.combine(doc.element_sel_of(obj.uid), swept, how)
            )
            received.append(obj.uid)
        # A replace-sweep makes the objects that received picks the ones being
        # edited, and leaves the edited set alone when it caught nothing (a
        # sweep over empty space must not end the edit).
        if replacing and received:
            doc.select(received)

    def _commit_drag(self: ClayView, doc: Any) -> None:
        """**One drag, one Ctrl+Z**, recorded against where the drag began.

        This reverses the earlier "one history step per object". That reading
        was defensible while a selection was something the user had assembled
        by hand, object by object; it is not once a multi-object selection
        is the ordinary case, because scaling it then costs one press per
        object to undo a gesture made once -- and the states in between are
        worse than the extra presses: some parts moved and some not, a
        document the user never made. That is exactly the failure
        ``test_one_ctrl_z_puts_all_three_objects_back`` exists to prevent, and
        ``run`` in ``studio/modes/clay/ops.py`` already answers it for ops with ``mark()`` /
        ``collapse_since()``. A drag is the same gesture and gets the same
        shape, labelled after the tool that made it.

        ``collapse_since`` refuses a run of fewer than two, so a one-object
        drag keeps its own ``transform`` step and a drag that moved nothing
        pushes nothing -- no empty compound either way.
        """
        from .....kernels.mesh.elements import OpError

        history = getattr(doc, "history", None)
        head = None if history is None else history.head
        mark = 0 if history is None else history.mark()
        # try/finally: anything that escapes the loop would skip
        # ``collapse_since``, so ``UndoStack._open_gestures`` stayed at 1 and the
        # history's byte and depth eviction was off for the rest of the session.
        try:
            for uid, was in self._drag_start.items():
                try:
                    doc.set_transform(uid, was=was)
                except KeyError:
                    continue  # deleted mid-drag
                except OpError as error:
                    # The drag landed on a scale/rotation the document refuses
                    # (all-zero); put the pre-drag values back rather than leave
                    # a live value no step records, and say why instead of
                    # reverting in silence.
                    obj = doc.by_uid(uid)
                    obj.translation, obj.rotation, obj.scale = (
                        np.array(v, copy=True) for v in was
                    )
                    self._toast(str(error))
        finally:
            if history is not None:
                history.collapse_since(mark)
        if history is not None:
            top = history.top
            # Only when the drag actually pushed: a press-and-release that moved
            # nothing must not relabel whatever step happens to be underneath.
            if top is not None and history.head != head:
                kind = self._key_kind or str(getattr(self.state, "tool", ""))
                if kind == "select":
                    # The Select tool's gizmo is the translate one, so what a
                    # drag of it commits is a move.
                    kind = "move"
                if kind in ("move", "rotate", "scale"):
                    top.label = kind.capitalize()
        self._drag_uids = []
        self._drag_start = {}

    def _motion(self: ClayView, doc: Any, local: tuple[float, float]) -> bool:
        dx = local[0] - self._last_mouse[0]
        dy = local[1] - self._last_mouse[1]
        self._last_mouse = local
        height = int(max(self._rect[3], 1))
        if self._grab is None:
            gizmo = self.active_gizmo(doc)
            prev_gizmo_hover = None if gizmo is None else gizmo.hover
            if gizmo is not None:
                origin, direction = self._ray(local)
                gizmo.hover = gizmo.hit(origin, direction)
            prev_hover_element = self.hover_element
            self._hover_element(doc, local)
            prev_hover_object = self.hover_object
            self._hover_object(doc, local, gizmo_hot=gizmo is not None and gizmo.hover is not None)
            # clay-15 (the 2026-09-23 audit, second run): only a real change
            # to what the cursor is over -- which gizmo arm lit up, which
            # element it now sits over, which object -- earns a redraw; see
            # ``handle_event``'s own comment for why a bare hover must not.
            if (
                (gizmo is not None and gizmo.hover != prev_gizmo_hover)
                or (self.hover_element != prev_hover_element)
                or (self.hover_object != prev_hover_object)
            ):
                self._render_dirty = True
            return False
        self._render_dirty = True
        # A grab in progress is not hovering anything, and the highlight must
        # not trail an orbit or follow a dragged object.
        self.hover_object = None
        self._hover_pick_at = None
        self._hover_element_at = None
        if self._grab in ("orbit", "marquee"):
            # The farthest the pointer has been from the press, so a drag that
            # wanders off and comes back is still a drag (``gestures.is_click``).
            self._moved = max(
                self._moved, math.hypot(local[0] - self._press_at[0], local[1] - self._press_at[1])
            )
        if self._grab == "marquee":
            start = self._marquee_from or local
            self.marquee = (start[0], start[1], local[0], local[1])
            return True
        if self._grab == "orbit":
            self.camera.orbit(dx, dy, height)
        elif self._grab == "pan":
            self.camera.pan(dx, dy, height)
        elif self._grab == "gizmo":
            self._drag_gizmo(doc, local)
        elif self._grab == "keydrag":
            self._drag_keyboard(doc, local)
        elif self._grab == "opdrag":
            # Coalesced like the element preview: the pointer's place is already in
            # ``_last_mouse``, so the one update the frame hook runs reads the
            # newest of however many events arrived.
            self._op_motion_pending = True
        return True

    #: How far the pointer must travel, in pixels, before the object hover casts
    #: another ray. A pick is a BVH walk per visible object; one per pixel of a
    #: slow drift across the viewport buys nothing the eye can see.
    HOVER_PICK_STEP = 3.0

    def _hover_object(
        self: ClayView, doc: Any, local: tuple[float, float], *, gizmo_hot: bool
    ) -> None:
        """Set ``hover_object`` for the pointer at *local* -- object mode only.

        Nothing is hovered over a gizmo handle (the handle is what the click
        would grab), outside the viewport, or in an element mode (which hovers
        elements, not whole objects). Throttled by distance moved so a still or
        slowly drifting pointer costs no rays.
        """
        width, height = float(self._rect[2]), float(self._rect[3])
        inside = 0.0 <= local[0] < width and 0.0 <= local[1] < height
        if doc.element_mode != "object" or gizmo_hot or not inside:
            self.hover_object = None
            self._hover_pick_at = None
            return
        last = self._hover_pick_at
        if (
            last is not None
            and abs(local[0] - last[0]) < self.HOVER_PICK_STEP
            and abs(local[1] - last[1]) < self.HOVER_PICK_STEP
        ):
            return
        self._hover_pick_at = local
        self.hover_object = self.pick(doc, local)

    #: Where, and for which document revision and element mode, the element hover
    #: last ran a full pick -- the object hover's ``_hover_pick_at`` for elements.
    _hover_element_at: tuple[tuple[float, float], int, str] | None = None

    def _hover_element(self: ClayView, doc: Any, local: tuple[float, float]) -> None:
        """Set ``hover_element`` for the pointer at *local*, one pick per
        :attr:`HOVER_PICK_STEP` pixels of travel.

        The 2026-10-10 audit's clay-25: object hover was throttled and this was
        not, so every MOUSEMOTION event ran ``pick_element`` -- a ray cast over
        every visible object plus a projection of every vertex, 13 ms against
        3.3 ms per event over 300 boxes, and a high-rate mouse delivers up to
        ~16 events a frame. The throttle is keyed on the document revision and
        the element mode as well as the position, so a hover left over a mesh
        that has since changed (or a mode just switched) picks again on the very
        next motion rather than after three pixels; a press, an orbit and
        ``clear_hover`` reset it outright, which keeps the first motion after a
        click responsive.
        """
        if doc.element_mode == "object":
            self.hover_element = None
            self._hover_element_at = None
            return
        last = self._hover_element_at
        key = (local, int(getattr(doc, "rev", 0)), doc.element_mode)
        if (
            last is not None
            and last[1:] == key[1:]
            and abs(local[0] - last[0][0]) < self.HOVER_PICK_STEP
            and abs(local[1] - last[0][1]) < self.HOVER_PICK_STEP
        ):
            return
        self._hover_element_at = key
        self.hover_element = self.pick_element(doc, local)

    def clear_hover(self: ClayView, doc: Any) -> bool:
        """Drop every hover highlight; -> whether any was lit.

        The 2026-10-07 audit's clay-83: the shell delivers a mouse move only to
        the viewport it is over, so a pointer that left for a panel never sent
        the motion that clears the highlight and the last element or object
        stayed lit. The viewport pane calls this on a frame where the image is
        not hovered. Left alone mid-grab: that owns the highlight itself.
        """
        if self._grab is not None:
            return False
        # Even with nothing lit: the pointer has left, so wherever it comes back
        # is a first motion and must pick (clay-25's throttle).
        self._hover_element_at = None
        gizmo = self.active_gizmo(doc)
        lit = (
            self.hover_element is not None
            or self.hover_object is not None
            or (gizmo is not None and gizmo.hover is not None)
        )
        if not lit:
            return False
        if gizmo is not None:
            gizmo.hover = None
        self.hover_element = None
        self.hover_object = None
        self._hover_pick_at = None
        self._render_dirty = True
        return True

    def _begin_element_drag(self: ClayView, doc: Any) -> None:
        """Snapshot every selected object's affected vertices at the press."""
        from .....kernels.mesh import elements as el
        from .....kernels.mesh.selection import _element_pickable

        self._element_drags = {}
        self._element_scale_refused = False
        centre = self.element_centre(doc)
        self._element_centre = np.zeros(3) if centre is None else centre
        index = self._uid_index(doc)
        for uid, sel in doc.element_sel.items():
            obj = index.get(uid)
            if obj is None:
                continue
            # Hiding an object leaves its element selection in
            # ``doc.element_sel``, and a gizmo grab would move the vertices of
            # an object the user cannot see. The same eligibility the pick
            # doors use.
            if not _element_pickable(obj):
                continue
            verts = el.affected_verts(obj.mesh, sel)
            if not len(verts):
                continue
            matrix = self._world(doc, obj, index)
            try:
                inverse = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                continue
            self._element_drags[uid] = _ElementDrag(
                before=obj.mesh,
                verts=verts,
                local=obj.mesh.positions[verts].astype("f8"),
                matrix=matrix,
                inverse=inverse,
            )

    def _element_world_transform(self: ClayView, delta: Any, state: Any) -> Any:
        """The gizmo's delta as a world-space affine about the drag's centre.

        One 4x4 for all three tools, so the per-vertex step downstream is a
        single ``inverse @ W @ matrix`` and knows nothing about which gizmo the
        user grabbed.
        """
        from .....kernels.mesh import ops

        centre = self._element_centre
        # The grid stands down for a typed value: re-quantising it onto the
        # grid afterwards would move it off the answer the user just gave.
        snap = bool(getattr(state, "snap", False)) and not self.drag_input.active
        world = m3.identity()
        # Refusal is a property of *this* frame's scale factors (clay-123): a
        # move or rotate frame, and a scale frame that is positive again, must
        # not inherit it.
        self._element_scale_refused = False
        if isinstance(delta, np.ndarray) and delta.shape == (4,):
            if snap:
                delta = ops.snap_rotation(delta, getattr(state, "snap_rotate", 0.0))
            rotation = m3.compose(m3.vec3(), delta, m3.vec3(1.0, 1.0, 1.0))
            world = _about(centre, rotation)
        elif self._is_scale(state):
            factors = np.asarray(delta, dtype="f8").reshape(3)
            # The 2026-10-09 audit's clay-10: a typed -1 on selected elements
            # moved the positions but not the winding, so every face came out
            # inside-out with no word said. The element path stands down for
            # a negative factor -- the preview holds the shape as it began and
            # the commit toasts -- and Mirror is the verb for a flip. Zero is
            # *not* refused (the second audit's clay-81): it flattens the
            # picked elements onto a plane and inverts nothing. (An object's
            # own scale may go negative by design: that is a transform, not a
            # mesh edit.)
            self._element_scale_refused = bool((factors < 0.0).any())
            if self._element_scale_refused:
                factors = np.ones(3)
            scale = np.diag(np.append(factors, 1.0))
            world = _about(centre, scale)
        else:
            # No grid snap on the target here: a move onto the grid is done per
            # vertex in ``_preview_element_drag`` (``_element_snap_step``), where
            # each corner lands on a grid point instead of only the median.
            target = np.asarray(delta, dtype="f8").reshape(3)
            world = m3.identity()
            world[:3, 3] = target - centre
        return world

    def _element_snap_step(self: ClayView, delta: Any, state: Any) -> float:
        """The world-grid step each moved vertex snaps to, or ``0.0`` for none.

        Move only -- a rotation and a scale snap their own delta in
        ``_element_world_transform`` -- and never for a typed value or an axis
        lock (``drag_input.active``): snapping the other two axes of a
        constrained move would shift vertices the user asked to keep still.
        """
        if isinstance(delta, np.ndarray) and delta.shape == (4,):
            return 0.0
        if self._is_scale(state) or not bool(getattr(state, "snap", False)):
            return 0.0
        if self.drag_input.active:
            return 0.0
        return abs(float(getattr(state, "snap_translate", 0.0)))

    #: The newest element-drag ``(delta, state)`` no preview has drawn yet, and
    #: whether an op drag's pointer has moved since its last update. See
    #: :meth:`flush_drag_preview`.
    _pending_preview: tuple[Any, Any] | None = None
    _op_motion_pending: bool = False

    def _queue_element_preview(self: ClayView, delta: Any, state: Any) -> None:
        """Remember *delta* as the element drag's newest position, to be drawn once.

        The 2026-10-10 audit's clay-25 throttled the hover picks and left the drag
        previews: ``_preview_element_drag`` (an affine over every moved vertex, normals
        and a buffer write per object) and ``_op_drag_motion`` (a kernel run) ran once
        per MOUSEMOTION event, ~16 a frame from a high-rate mouse, and all but the last
        were overwritten before they were ever drawn. **Not a distance throttle**: that
        would leave the value the commit lands up to ``HOVER_PICK_STEP`` px behind the
        pointer. The delta is still worked out per event (``RotateGizmo.update`` returns
        an increment that :meth:`_accumulate` must see every one of), only the expensive
        write is deferred to :meth:`flush_drag_preview`, once a frame, and to the commit.
        """
        self._pending_preview = (delta, state)

    def flush_drag_preview(self: ClayView, doc: Any) -> None:
        """Draw the newest queued element preview and run a queued op-drag update.

        Called once a frame from ``ClayView.draw`` before anything is keyed or drawn,
        and by :meth:`_release_drag` and :meth:`settle_drag` ahead of the commit, so
        the value that lands is the one under the pointer and never a frame behind it.
        A no-op when nothing is queued, which is every frame outside a drag.
        """
        pending, self._pending_preview = self._pending_preview, None
        if pending is not None and self._element_drags:
            self._preview_element_drag(doc, *pending)
        if self._op_motion_pending:
            self._op_motion_pending = False
            if self._grab == "opdrag" and self._op_drag is not None:
                self._op_drag_motion(doc)

    def _drop_pending_preview(self: ClayView) -> None:
        """Forget a queued preview without drawing it: the drag was cancelled,
        restarted as another transform, or is committing from the pointer."""
        self._pending_preview = None
        self._op_motion_pending = False

    def _preview_element_drag(self: ClayView, doc: Any, delta: Any, state: Any) -> None:
        """Move the affected vertices on the GPU only, without touching the document.

        Nothing is pushed and no ``Mesh`` reaches the document: the drag writes
        the moved vertices straight into the cached buffers, so ``rebuilds``
        stays flat across a whole drag and there is exactly one history step at
        the end rather than one per mouse-move.
        """
        from .. import element_move

        world = self._element_world_transform(delta, state)
        step = self._element_snap_step(delta, state)
        if step:
            # The 2026-10-10 audit's clay-55: each corner lands on the grid, so
            # the pointer's displacement is not what happened. Quote the median's
            # landed displacement (the point the gizmo is drawn at).
            from .....kernels.mesh import drag as bdrag
            from .. import element_move as _em

            centre = np.asarray(self._element_centre, dtype="f8")
            landed = _em.snap_points(centre + world[:3, 3], step) - centre
            self.drag_hud = bdrag.readout(self._drag_kind(state), landed, self.drag_input)
        for uid, drag in self._element_drags.items():
            if step:
                moved = element_move.moved_local(
                    drag.local, drag.matrix, drag.inverse, world[:3, 3], snap_step=step
                )
            else:
                moved = element_move.apply_affine(
                    drag.inverse @ world @ drag.matrix, drag.local
                )
            positions = np.array(drag.before.positions, dtype="f4")
            positions[drag.verts] = moved
            drag.preview = positions
            self._preview_positions(doc, uid, positions)
        doc.touch()

    def _preview_positions(self: ClayView, doc: Any, uid: int, positions: Any) -> None:
        """Rewrite one object's vertex buffers in place. No VAO is rebuilt.

        Through ``preview_primitives`` rather than ``to_primitives``: the mesh
        the drag began on is fixed for the drag's whole duration, so the
        material grouping, the corner gathers and the index buffer are built
        once and only the positions and normals are recomputed per frame. A
        200k-triangle import went 368 ms a frame to 92 -- see
        ``dev/measurements/2026-08-16-interactive-defects.md``.
        """
        # The overlay first, and whatever the GPU half below does: a
        # ``ValueError`` from ``update_vertices`` returning from here before
        # this write would freeze the selection overlay with the drawn mesh.
        overlay = getattr(self, "_overlays", {}).get(uid)
        if overlay is not None:
            overlay.write_positions(positions)
        entry = self._cache.get(uid)
        if entry is None:
            return
        obj = doc.by_uid(uid)
        drag = self._element_drags.get(uid)
        base = obj.mesh if drag is None else drag.before
        # ``drag.verts`` is exactly the set written into ``positions`` above,
        # which is what makes the incremental normals safe; with no element drag
        # in hand the mover is a gizmo over the whole object and there is no
        # smaller set to name, so nothing is passed and every face is recomputed.
        prims = bd.preview_primitives(
            base, positions, doc.materials, moved=None if drag is None else drag.verts
        )
        for (_node, gpu), primitive in zip(entry.gpu.draws, prims, strict=False):
            try:
                gpu.update_vertices(primitive)
            except ValueError:  # pragma: no cover - topology changed under a drag
                return

    def _commit_element_drag(self: ClayView, doc: Any) -> None:
        """One history step for the whole gesture, against the mesh each drag began on.

        The 2026-09-06 audit found this pushing one ``set_mesh`` per dragged
        object with no fold around the loop -- unlike ``_commit_drag``, its
        sibling on the object-transform path, which was given exactly this
        wrap for the same reason (see that method's docstring), and unlike
        ``run_mesh_op`` in ``studio/modes/clay/ops.py``. A marquee or gizmo grab spanning several
        visible objects otherwise cost one Ctrl+Z per object, with the same
        "some limbs moved and some not" state in between that the other two
        paths were fixed to prevent.
        ``test_a_multiobject_element_drag_is_one_undo_step`` pins the fold;
        ``collapse_since`` refuses a run of fewer than two, so a one-object
        element drag still keeps its own single step.

        A drag that moved nothing pushes nothing -- dirty is a comparison
        against the history head, and a no-op step would make a saved document
        ask to be saved again. It still has to evict the previewed entries: the
        buffers hold whatever the last preview frame wrote, and the mesh
        identity the cache keys on has not changed, so nothing else would ever
        rebuild them.
        """
        from .. import element_move

        drags, self._element_drags = self._element_drags, {}
        if getattr(self, "_element_scale_refused", False):
            self._element_scale_refused = False
            self._toast(
                "Scaling selected elements by a negative factor would turn their faces "
                "inside out, so they were left as they were. Use Mirror to flip them."
            )
        # ``commit_positions`` owns the fold and its ``try/finally`` (an
        # ``OpError`` escaping a ``set_mesh`` must not leave the undo stack's
        # gesture counter open); what stays here is the view's half -- evicting
        # the previewed buffers, which hold whatever the last frame wrote while
        # the mesh identity the cache keys on has not changed.
        changes = {
            uid: (drag.before, drag.before.positions if drag.preview is None else drag.preview)
            for uid, drag in drags.items()
        }
        _pushed, unchanged, refused = element_move.commit_positions(doc, changes)
        for uid in unchanged:
            entry = self._cache.pop(uid, None)
            if entry is not None:
                entry.gpu.release()
        for uid, error in refused:
            entry = self._cache.pop(uid, None)
            if entry is not None:
                entry.gpu.release()
            self._restore_overlays(doc, [uid])
            self._toast(str(error))

    def _drag_gizmo(self: ClayView, doc: Any, local: tuple[float, float]) -> None:
        """Apply a gizmo delta to every selected object, in place.

        In place, and committed as one history step on release: a step per
        mouse-move would fill the undo stack with a hundred entries for one
        drag, and the intermediate positions are not states the user wants to
        step back through.
        """
        gizmo = self.active_gizmo(doc)
        if gizmo is None:
            return
        state = self.state
        origin, direction = self._ray(local)
        delta = gizmo.update(origin, direction)
        if delta is None:
            return
        delta = self._accumulate(delta)
        delta = self._narrow(doc, delta, state, local)
        if doc.element_mode != "object":
            self._queue_element_preview(delta, state)
            return
        self._apply_to_dragged(doc, delta, state)

    def _narrow(
        self: ClayView, doc: Any, delta: Any, state: Any, local: tuple[float, float]
    ) -> Any:
        """Axis lock and typed value, applied to one drag delta.

        One place for both, above both the object path and the element path,
        so neither has to learn what a lock is -- the same argument that keeps
        the constraint out of the three gizmos.

        The anchor for a translation is the same point the consumer measures
        against -- the element centroid in an element mode, the gizmo's own
        origin otherwise -- so the constrained displacement is the one that is
        actually applied rather than one measured from somewhere else.
        """
        from .....kernels.mesh import drag as bdrag
        from .....kernels.mesh import ops

        # The 2026-10-07 audit's clay-31: the readout's unit follows what the
        # drag *is*, and a keyboard drag is what its key said -- the same rule
        # ``_is_scale`` states for the maths. Read off the tool, a G drag under
        # Rotate said "0.4 deg" and under Select (or Scale) had no unit at all,
        # on the line that is the only confirmation of a typed value.
        kind = self._drag_kind(state)
        entry = self.drag_input
        if isinstance(delta, np.ndarray) and delta.shape == (4,):
            constrained = bdrag.constrain_rotation(delta, entry)
            # The 2026-10-10 audit's clay-55: with Snap on the readout quoted the
            # raw angle while ``_apply`` / ``_element_world_transform`` land the
            # object on the snapped one. Shown here, returned unsnapped -- the
            # snap stays where the module docstring says it is.
            shown = delta
            if self._snap_live(state):
                shown = ops.snap_rotation(constrained, getattr(state, "snap_rotate", 0.0))
            self.drag_hud = _rotation_hud(shown, entry)
            return constrained
        if self._is_scale(state):
            out = bdrag.constrain_scale(delta, entry)
            self.drag_hud = bdrag.readout("scale", out, entry)
            return out

        anchor = self._element_centre if doc.element_mode != "object" else self._drag_origin
        target = np.asarray(delta, dtype="f8").reshape(3)
        moved = bdrag.constrain_translation(target - anchor, entry)
        self.drag_hud = bdrag.readout(kind, moved, entry)
        return anchor + moved

    def _drag_kind(self: ClayView, state: Any) -> str:
        """``move`` / ``rotate`` / ``scale``: what the drag in hand *is* (see the
        clay-31 note in :meth:`_narrow`)."""
        tool = getattr(state, "tool", "") if state is not None else ""
        kind = self._key_kind or tool or "move"
        return "move" if kind == "select" else kind

    def _snap_live(self: ClayView, state: Any) -> bool:
        """Whether the grid snap is acting on this drag: on, and standing down
        for a typed value or an axis lock (``drag_input.active``)."""
        return bool(getattr(state, "snap", False)) and not self.drag_input.active

    def _accumulate(self: ClayView, delta: Any) -> Any:
        """Turn one ``update`` return into a quantity measured from the press.

        The two gizmos that need it need it for opposite reasons and both were
        wrong the same way. ``RotateGizmo.update`` documents its return as the
        *increment since the last update* -- composing that against the
        press-time rotation keeps only the last mouse-move, so a 90-degree
        sweep landed at whatever the final frame happened to travel. It is
        summed here instead, which is exact rather than approximate: every
        increment of one drag is about the same local axis, so the products
        commute and the total is the angle the cursor actually swept.

        ``TranslateGizmo.update`` is the mirror image -- it returns the
        gizmo's *new world position*, which is only an object's new position
        for an object whose origin happened to sit under the gizmo. Measured
        against the press-time gizmo origin it becomes a displacement, which
        is what every selected object can be moved by.

        ``ScaleGizmo`` already returns a factor relative to the drag's start
        and passes straight through.
        """
        if isinstance(delta, np.ndarray) and delta.shape == (4,):
            self._drag_quat = m3.quat_normalize(m3.quat_mul(delta, self._drag_quat))
            return self._drag_quat
        return delta

    def _apply(
        self: ClayView,
        doc: Any,
        obj: Any,
        was: Any,
        delta: Any,
        state: Any,
        index: dict[int, Any] | None = None,
        applied: list[Any] | None = None,
    ) -> bool:
        """One object's transform under the live drag, written in **local** space.

        **A rotate and a scale orbit the pivot, not the object's own origin**,
        and that is a fix rather than a feature: the gizmo is drawn at
        ``selection_centre`` -- the median of what is selected -- while this
        wrote only ``obj.rotation``, so dragging the ring with two objects
        selected span each of them in place around a ring drawn somewhere
        neither of them was. The picture said "these turn about here" and the
        document did something else.

        It changes what a *single* off-centre object does too, and that is the
        same correction rather than a side effect: the ring is drawn at the
        bounding box's centre, so an object whose origin is not there has always
        been rotating about a point other than the one on screen. Blender's
        Median Point pivot behaves exactly as this now does.

        ``_drag_origin`` is that pivot -- the gizmo's own origin, captured at
        the press -- which the translation arm has always subtracted and the
        other two never read.

        **Tranche 3: scene structure -- everything below the pivot maths is
        computed in world space and only converted to local at the very end.**
        ``was`` is the object's *local* TRS at the press (``set_transform``'s
        own convention); ``obj.parent``'s world matrix, read fresh each call,
        gives the frame ``was`` is local *to*, so ``before_t``/``before_r``/
        ``before_s`` -- what the pivot maths below actually reads -- are the
        object's **world** placement at the press, not its local one. For a
        root that is the same number either way (a root's local TRS *is* its
        world TRS -- ``document.py``'s module docstring), which is what keeps
        this producing exactly what it always did for every document with no
        parenting. ``doc.local_from_world`` is the one conversion back, at the
        end, through the object's *current* parent -- so a reparent mid-drag
        (not offered today, but nothing here assumes otherwise) would still
        write a sound local value.

        ``index`` is :meth:`_uid_index`'s map for the frame (clay-21): the
        parent's world matrix and the world-to-local conversion then read it
        instead of ``doc.world_matrix`` / ``doc.local_from_world``, each of
        which scans the document. Same arithmetic either way -- ``_world`` is
        ``world_matrix``'s composition and the conversion is
        ``local_from_world``'s (``decompose`` of the inverse parent times the
        target), a root's being ``decompose`` of the target itself.

        ``applied``, when given, receives a move's landed world displacement
        (after the grid snap) so the caller can quote what happened rather than
        what the pointer asked for -- the 2026-10-10 audit's clay-55.
        """
        from .....kernels.mesh import ops

        # See ``_element_world_transform``: the grid stands down for a typed value.
        snap = bool(getattr(state, "snap", False)) and not self.drag_input.active
        pivot = np.asarray(self._drag_origin, dtype="f8")
        parent = obj.parent
        parent_world = m3.identity()
        if parent is not None:
            parent_obj = None if index is None else index.get(parent)
            parent_world = (
                doc.world_matrix(parent)
                if parent_obj is None
                else self._world(doc, parent_obj, index)
            )
        before_world = parent_world @ m3.compose(*was)
        before_t, before_r, before_s = m3.decompose(before_world)

        if isinstance(delta, np.ndarray) and delta.shape == (3,) and self._is_scale(state):
            # The 2026-10-09 second audit's clay-58: the factors are per *world*
            # axis; see :func:`_world_scale_on_local_axes`. -> ``False`` (and
            # nothing written) for a turn that cannot express them.
            local = _world_scale_on_local_axes(before_r, delta)
            if local is None:
                return False
            world_s = before_s * local
            world_t = pivot + (before_t - pivot) * np.asarray(delta, dtype="f8")
            world_r = before_r
        elif isinstance(delta, np.ndarray) and delta.shape == (4,):
            # The *delta* angle is what snaps, exactly as the element path does
            # in ``_element_world_transform`` and for ``ops.snap_rotation``'s
            # own stated reason: "rotate this by fifteen degrees about the ring
            # I grabbed" leaves an already-placed object where it was put,
            # where quantising the absolute orientation visibly swings it the
            # moment the drag starts.
            if snap:
                delta = ops.snap_rotation(delta, state.snap_rotate)
            world_r = m3.quat_normalize(m3.quat_mul(delta, before_r))
            world_t = pivot + m3.quat_rotate(delta, before_t - pivot)
            world_s = before_s
        else:
            moved = np.asarray(delta, dtype="f8").reshape(3) - self._drag_origin
            world_t = before_t + moved
            if snap:
                world_t = ops.snap_translation(world_t, state.snap_translate)
            world_r, world_s = before_r, before_s
            if applied is not None:
                # What actually landed, after the grid -- for the HUD (clay-55).
                applied.append(world_t - before_t)

        world_target = m3.compose(world_t, world_r, world_s)
        if index is None or (parent is not None and parent not in index):
            t, r, s = doc.local_from_world(obj.uid, world_target)
        elif parent is None:
            t, r, s = m3.decompose(world_target)
        else:
            try:
                inverse = np.linalg.inv(parent_world)
            except np.linalg.LinAlgError:
                # The document's own door raises the refusal that names it.
                t, r, s = doc.local_from_world(obj.uid, world_target)
            else:
                t, r, s = m3.decompose(inverse @ world_target)
        obj.translation, obj.rotation, obj.scale = t, r, s
        return True

    def _is_scale(self: ClayView, state: Any) -> bool:
        # The 2026-10-03 audit's clay-19: a keyboard drag is what its *key*
        # said (G move, S scale), whatever tool is active -- reading the tool
        # made S under Select translate by the scale factor and G under Scale
        # scale by the displacement. A gizmo drag is still the tool's.
        if self._grab == "keydrag":
            return self._key_kind == "scale"
        return getattr(state, "tool", "") == "scale"
