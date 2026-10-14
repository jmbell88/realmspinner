"""The Viewer: everything the UI is allowed to know about the 3D pane.

One object, one contract. The panes never touch a framebuffer, a camera or a
GLB -- they call ``adopt_model``, ``enter_pose_mode``, ``get_pose`` and so on,
and get back plain Python. Which is what lets the renderer be tested against a
standalone context with no window, and the panes be tested with no GL at all.

Input arrives as pygame events, but only when the viewport image is hovered:
imgui owns the mouse otherwise, and a drag that started on the image keeps
ownership until the button comes up so a rotate does not stop halfway when the
pointer crosses onto a panel.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..kernels.geom3d import gltf
from ..kernels.geom3d import math3d as m3
from ._view_frame import FrameOps
from ._viewer_pose import PoseOps
from .viewer import bonelines as bonelineslib
from .viewer import capture, glctx, picking
from .viewer import markers as markerslib
from .viewer import scene as scenelib
from .viewer.camera import Camera, screen_ray
from .viewer.gizmo import RotateGizmo, TranslateGizmo
from .viewer.pose import PoseEditor
from .viewer.render import Renderer

log = logging.getLogger(__name__)


class Viewer(PoseOps, FrameOps):
    """The 3D pane, from the UI's point of view.

    The pose half is inherited rather than written here (``_viewer_pose``), the
    ``ClayView`` split: methods only, every attribute still built below.
    """

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.renderer = Renderer(ctx)
        self.viewport = glctx.Viewport(ctx, (16, 16))
        self.compare_viewport: glctx.Viewport | None = None
        self.camera = Camera()
        self.compare_camera = Camera()

        self.model: gltf.Model | None = None
        self.gpu: scenelib.GpuModel | None = None
        self.placement = m3.identity()
        self.radius = 1.0
        self.path: Path | None = None
        # ``(model, world_version, lo, hi)`` for ``stats`` -- see there.
        self._bounds_memo: tuple[Any, int, Any, Any] | None = None

        self.compare_model: gltf.Model | None = None
        self.compare_gpu: scenelib.GpuModel | None = None
        self.compare_placement = m3.identity()

        self.wireframe = False
        self.pose_mode = False
        # Which job's rig the editor is bound to, so a write can never land on
        # whichever asset happens to be selected. See enter_pose_mode.
        self.pose_job_id: str | None = None
        self.editor = PoseEditor()
        self.markers = markerslib.JointMarkers(ctx, self.renderer.programs)
        self.bonelines = bonelineslib.BoneLines(ctx, self.renderer.programs)
        # Which (parent, bone) pairs the skeleton lines connect; derived once
        # per bind from the node graph, cleared on the way out.
        self._bone_pairs: list[tuple[str, str]] = []
        self.rotate_gizmo = RotateGizmo(ctx, self.renderer.programs)
        self.translate_gizmo = TranslateGizmo(ctx, self.renderer.programs)
        self.on_pose_dirty: Any = None

        # Set every frame by the pane so input can be mapped into the image.
        self._rect = (0.0, 0.0, 1.0, 1.0)
        self._grab: str | None = None  # "orbit" | "pan" | "gizmo" | "marker"
        #: Whether the press in hand would deselect if it turns out to be a
        #: click. Set when a left press in pose mode hits no marker; cleared by
        #: anything that starts a real drag. See ``_press``.
        self._deselect_on_click = False
        #: Where the press in hand landed. A click is decided by how far the pointer
        #: has got *from here*, not by the size of one motion event (the 2026-10-14
        #: audit, finding create-viewer-02).
        self._press_at: tuple[float, float] | None = None
        # The open ``editor.record()`` for a live gizmo gesture, or None. One
        # undo step per *gesture*, not per frame: ``rotate_selected`` and
        # ``move_handle`` are called on every motion event, so recording each
        # would fill the stack with a single drag and leave Ctrl+Z walking a
        # rotation back a degree at a time.
        self._pose_step: Any = None
        self._last_mouse = (0.0, 0.0)
        # A reference image is shown as a plain texture rather than as geometry.
        self.reference: Any = None
        # Redraw bookkeeping (B12/B14): the scene is redrawn when anything
        # marked it dirty, when the camera is still gliding, or when the draw's
        # own inputs (size, wireframe, compare, and in pose mode the editor's
        # selection and handles -- ``_pose_signature``) differ from the last
        # frame's.
        self._render_dirty = True
        self._last_render_key: Any = None
        # The GLB a parse has been dispatched for but not yet adopted. Held so
        # the timer that dispatched it does not dispatch it again on every tick
        # until the result lands, and so nothing else decides the viewer is
        # "showing the wrong thing" while a load is in flight.
        self.pending: Path | None = None
        # The create-50 (2026-10-04 audit) half of the same hand-over, for the
        # one load the pose pane owns the landing of: its ``_Entering`` while
        # "Edit pose" / "Apply" has a rig parsing on a task, else None.
        # ``pending`` above is the freshness check; this is what to adopt.
        self.pose_loading: Any = None
        # The 2026-10-11 audit, finding create-03: a library card's thumbnail
        # was captured straight after ``adopt_model``, but adoption only uploads
        # and frames -- the viewport texture still held the previous frame, so
        # the card showed the previous asset. The request (a job id) waits here
        # for ``render`` to draw the adopted model; ``adopt_model`` and
        # ``clear`` drop it, so it can never be served from another model.
        self._thumb_request: str | None = None
        self._thumb_drawn = False

    # -- loading -----------------------------------------------------------

    @staticmethod
    def parse_model(path: Path) -> Any:
        """The blocking half of a load: parse the glTF and decode its textures.

        Touches no GL and no viewer state, so it is safe on a task thread --
        which is the whole point. It is the expensive half by a wide margin (a
        full accessor decode plus a PNG per texture slot), and it used to run
        on the frame thread at the exact moment a job finished, which is when
        the file is largest and coldest.
        """
        return gltf.load(path)

    def adopt_model(self, model: Any, path: Path) -> None:
        """The frame-thread half: upload, and take the parsed model as current.

        Must stay on the frame thread -- ``GpuModel`` creates buffers and
        textures on the one GL context, and so does releasing the old one.
        """
        gpu = scenelib.GpuModel(self.ctx, model)
        self._release_model()
        # Whatever parse was in flight is no longer wanted: this *is* the
        # viewport's content now. Without this, the blocking path -- entering
        # the pose editor, loading a Review unit -- left ``pending`` naming a
        # parse dispatched a moment earlier, which still matched when it landed
        # and was adopted over the top. In the pose case that discarded the
        # editor with unsaved rotations in it, bypassing ``pose_panel.guard``,
        # the confirm every other way out of the editor goes through.
        self.pending = None
        # A thumbnail asked for under the previous model is no longer this
        # model's to take (``request_thumbnail`` is called after adoption).
        self._thumb_request = None
        self._render_dirty = True
        self.model, self.gpu, self.path = model, gpu, Path(path)
        # The 2026-10-10 audit, finding create-33: ``stats`` memoises the
        # bounds with a reference to the model it measured, and only ``clear``
        # dropped it. Review and Create replace a model through here, so
        # the replaced one (its CPU geometry and decoded textures, up to
        # 768 MiB) stayed alive until the next ``stats`` call happened to
        # overwrite the memo -- or for good if none did.
        self._bounds_memo = None
        self.placement = scenelib.placement(model)
        self.clear_reference()
        self.exit_pose_mode()
        self.frame()

    def clear(self) -> None:
        self._render_dirty = True
        self._release_model()
        self.exit_pose_mode()
        self.model = self.gpu = self.path = None
        self._bounds_memo = None
        # See ``adopt_model``: an in-flight parse must not land on an emptied
        # viewport either.
        self.pending = None
        self._thumb_request = None

    def request_thumbnail(self, job_id: str) -> None:
        """Ask for the card picture of the model just adopted, once it is drawn.

        Keyed by job id and single-slot: a second adoption before the draw
        replaces the request (``adopt_model`` clears it first), so the picture
        is never filed under a job whose model is not the one on screen.
        """
        self._thumb_request = job_id
        self._thumb_drawn = False

    def take_thumbnail_request(self) -> str | None:
        """The job id to photograph now, or None. Frame thread, after ``render``.

        Only answers once ``render`` has actually drawn since the request was
        made -- until then ``screenshot`` would read the previous frame.
        """
        if self._thumb_request is None or not self._thumb_drawn:
            return None
        job_id, self._thumb_request = self._thumb_request, None
        return job_id

    def _release_model(self) -> None:
        if self.gpu is not None:
            self.gpu.release()
        self.gpu = None

    @staticmethod
    def parse_reference(path: Path) -> tuple[tuple[int, int], bytes]:
        """The blocking half of showing a picture: decode it. -> (size, RGBA).

        ``parse_model``'s split, for the same reason: ``_sync_viewer`` reaches
        this on a timer, on the frame a job finishes, and a 1024-square PNG
        decode is a visible hitch at exactly the moment the card is meant to
        show the job finishing. No GL, no viewer state.
        """
        from PIL import Image

        with Image.open(path) as im:
            rgba = im.convert("RGBA")
            return (rgba.width, rgba.height), rgba.tobytes()

    def adopt_reference(self, parsed: tuple[tuple[int, int], bytes]) -> None:
        """The frame-thread half: upload the decoded picture."""
        size, data = parsed
        self.clear_reference()
        texture = self.ctx.texture(size, 4, data)
        texture.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.reference = texture

    @property
    def dragging(self) -> bool:
        """Whether a pointer gesture is in progress.

        The public spelling of ``_grab``, because the router that has to know
        is ``App`` and it was reaching into the private attribute of two
        different viewers to ask. What the grab *is* stays private -- the
        router only ever needed to know that there is one.
        """

        return self._grab is not None

    def load_reference(self, path: Path) -> None:
        """Show a 2D reference image instead of a mesh. Both halves, blocking
        -- for the callers where the user just pressed something."""
        self.adopt_reference(self.parse_reference(path))

    def clear_reference(self) -> None:
        if self.reference is not None:
            # Registered via widgets.texture_ref in main._draw_reference, so
            # the backend must drop the name before the driver can reuse it.
            self._forget(self.reference)
            self.reference.release()
            self.reference = None

    @property
    def has_model(self) -> bool:
        return self.gpu is not None

    def stats(self) -> dict[str, Any]:
        """Counted off the loaded scene, not read from mesh_report: the report
        describes model.glb and the viewer may be showing rig.glb or a pose."""
        if self.model is None:
            return {}
        # The 2026-10-03 audit, finding create-28: ``Model.bounds`` walks every
        # mesh instance (about 11 microseconds each) and this runs per frame for
        # the inspector's size line, so a 5,000-instance scene import cost 57 ms
        # a frame to say a number that had not changed. Keyed on the model's own
        # ``world_version`` (bumped by each ``update_world``, i.e. each pose
        # change) rather than on time; placement is applied below, per call,
        # because it is three multiplies.
        version = getattr(self.model, "world_version", None)
        memo = getattr(self, "_bounds_memo", None)
        if memo is not None and version is not None and memo[:2] == (self.model, version):
            lo, hi = memo[2], memo[3]
        else:
            lo, hi = self.model.bounds()
            if version is not None:
                self._bounds_memo = (self.model, version, lo, hi)
        size = (hi - lo) * np.array(
            [self.placement[0, 0], self.placement[1, 1], self.placement[2, 2]]
        )
        return {
            "triangles": self.model.triangle_count,
            # Cached on the immutable model (B18) -- this runs every frame.
            "vertices": self.model.vertex_count,
            "size": tuple(float(v) for v in size),
        }

    # -- camera ------------------------------------------------------------

    def frame(self) -> float:
        self._render_dirty = True
        if self.model is None:
            return 0.0
        lo, hi = self._world_bounds()
        self.radius = self.camera.frame(lo, hi)
        self.renderer.fit_grid(lo, hi)
        return self.radius

    def _world_bounds(self):
        lo, hi = self.model.bounds()
        corners = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        world = (self.placement @ np.hstack([corners, np.ones((8, 1))]).T).T[:, :3]
        return world.min(axis=0), world.max(axis=0)

    def set_wireframe(self, on: bool) -> None:
        self.wireframe = bool(on)
        self._render_dirty = True

    def set_turntable(self, on: bool) -> None:
        # OrbitControls' own autoRotate, not a rotation of the model: rotating
        # the model would move the thing the markers are positioned against.
        self.camera.auto_rotate = bool(on)
        self._render_dirty = True

    # -- compare -----------------------------------------------------------

    # The 2026-09-07 audit, finding create-07: ``compare(path)`` -- load and
    # upload a comparison mesh synchronously on the frame thread -- had no
    # callers left anywhere in the tree. It survived the UX-04 split as "kept
    # for callers that... genuinely want the blocking form", but no such
    # caller ever arrived, and wiring a button to it today would reopen
    # exactly the frame-freeze ``parse_model``/``adopt_compare`` exists to
    # avoid: the library now parses through ``TaskRunner`` and hands the
    # already-parsed model to ``adopt_compare`` below, which is the only
    # entry point that stays.
    def adopt_compare(self, model: Any) -> None:
        """The frame-thread half: upload an already-parsed comparison model.

        Must stay on the frame thread -- ``GpuModel`` creates buffers and
        textures on the one GL context, and ``exit_compare`` releases them.

        The GPU resources are built *before* the old ones are released, so a
        failed upload leaves the previous comparison intact rather than an
        empty pane: adopting is only allowed to succeed or change nothing.
        """
        self._render_dirty = True
        gpu = scenelib.GpuModel(self.ctx, model)
        self.exit_compare()
        self.compare_model, self.compare_gpu = model, gpu
        # Placed the way the left-hand mesh is, or the two would not line up.
        self.compare_placement = scenelib.placement(model)
        if self.compare_viewport is None:
            self.compare_viewport = glctx.Viewport(self.ctx, (16, 16))

    def exit_compare(self) -> None:
        self._render_dirty = True
        if self.compare_gpu is not None:
            self.compare_gpu.release()
        self.compare_gpu = self.compare_model = None

    @property
    def comparing(self) -> bool:
        return self.compare_gpu is not None

    # -- pose ----------------------------------------------------------------
    #
    # In ``_viewer_pose.PoseOps``, which this class inherits: the editor, the
    # dirty listener, the two gizmos and their overlays. State stays here.

    # -- rendering ---------------------------------------------------------

    def render(self, rect: tuple[float, float, float, float], dt: float) -> Any:
        """Draw one frame into the viewport. -> the resolved texture.

        Skipped entirely -- the last resolved texture is returned as-is --
        when nothing that feeds the draw has changed (B12; the compare half
        rides on the same skip, B14). Every mutating entry point sets
        ``_render_dirty``; the camera answers for itself via ``settled``; and pose
        mode adds what its overlays are drawn from to the key
        (:meth:`_pose_signature`). It used to bypass the skip altogether on the
        ground that gizmo hover writes no state a key could see, but ``_motion``
        marks the viewer dirty when the hovered handle changes, so the bypass
        only made an idle pose session re-render the MSAA target and rebuild the
        overlay arrays every frame (the 2026-10-04 audit, finding create-14).
        """
        self._rect = rect
        width, height = int(max(rect[2], 1)), int(max(rect[3], 1))
        key = (width, height, bool(self.wireframe), self.comparing)
        if self.pose_mode:
            key += (self._pose_signature(),)
        if self._frame_unchanged(key):
            return self.viewport.texture
        self._last_render_key = key
        self._render_dirty = False
        # Set only on this path: a skipped frame returns the texture the last
        # draw left, which is the stale one ``request_thumbnail`` waits out.
        # (``getattr``: a test drives ``render`` on a bare namespace.)
        if getattr(self, "_thumb_request", None) is not None:
            self._thumb_drawn = self.gpu is not None
        self._resize(width, height)
        self.camera.update(dt)
        self.renderer.draw(
            self.viewport,
            self.camera,
            self.gpu,
            model_matrix=self.placement,
            # ``wire_overlay``, not ``wireframe``: the manual's Wireframe
            # toggle "draws the triangles over the shaded surface", and
            # ``wireframe`` *replaces* the fill (the 2026-10-03 audit, finding
            # create-19: the chapter described a picture this toggle did not
            # produce, in the one chapter a reader checks a triangle budget
            # against). Clay and Mason already pass the overlay.
            wire_overlay=self.wireframe,
            overlays=self._overlays(height),
        )
        if self.comparing and self.compare_viewport is not None:
            # One camera state, two renders: the point of a comparison is that
            # both meshes are seen from the identical angle.
            self._resize(width, height, self.compare_viewport)
            self.compare_camera.copy_from(self.camera)
            self.renderer.draw(
                self.compare_viewport,
                self.compare_camera,
                self.compare_gpu,
                model_matrix=self.compare_placement,
                wire_overlay=self.wireframe,
            )
        return self.viewport.texture

    def _pose_signature(self) -> tuple[Any, ...]:
        """What pose mode's overlays are drawn from that no mutator here marks.

        Panes write ``editor.selected`` and ``editor.mode`` straight onto the
        editor (the bone list, the joints toggle), and undo/redo restores
        handles through it; none of those pass a ``Viewer`` method that sets
        ``_render_dirty``. Cheap by
        construction -- a handful of scalars, the model's own pose version and
        one flat copy of the handle positions -- because it runs every frame to
        spare the frame a full MSAA draw.

        The rest of what :meth:`_overlays` and :meth:`_active_gizmo` read off
        the editor is here too (F19-4): the handle and bone *names* (a rename
        moves no handle), and the marker radius and placement the picture is
        scaled and placed by.
        """
        editor = self.editor
        handles = editor.handles
        flat = np.asarray(list(handles.values()), dtype="f8").tobytes() if handles else b""
        placement = getattr(self, "placement", None)
        return (
            editor.selected,
            editor.mode,
            getattr(self.model, "world_version", None),
            tuple(handles),
            tuple(editor.bones),
            flat,
            getattr(self, "radius", None),
            None if placement is None else np.asarray(placement, dtype="f8").tobytes(),
        )

    # ``_overlays`` and ``_active_gizmo`` live in ``PoseOps`` with the rest of
    # the pose half: both return nothing outside pose mode, and the gizmo
    # choice is a function of the editor's mode.

    # -- capture -----------------------------------------------------------

    def screenshot(self) -> Any:
        return capture.image(self.viewport)

    # -- input -------------------------------------------------------------

    def handle_event(self, event: Any, hovered: bool) -> bool:
        """Feed one pygame event. -> whether the viewer consumed it.

        ``hovered`` is whether the pointer is over the viewport image; a drag
        already in progress ignores it, so crossing onto a panel mid-rotate
        does not drop the drag.
        """
        import pygame

        local = self._local(event)
        # **Not every event moves the picture, and this used to assume they
        # all did.** A press, a wheel and a drag do; a bare mouse *motion* over
        # the viewport does not, unless it changed which gizmo handle is
        # hovered -- and marking the scene dirty for it re-rendered the whole
        # MSAA target on every mouse move across the pane, whether or not
        # anything in it had changed. ``_motion`` decides that for itself now.
        if event.type == pygame.MOUSEBUTTONDOWN and hovered:
            self._render_dirty = True
            return self._press(event.button, local)
        if event.type == pygame.MOUSEBUTTONUP:
            self._render_dirty = True
            return self._release(event.button)
        if event.type == pygame.MOUSEMOTION:
            return self._motion(local)
        if event.type == pygame.MOUSEWHEEL and hovered:
            self._render_dirty = True
            self.camera.dolly(event.y)
            return True
        return False

    def _press(self, button: int, local: tuple[float, float]) -> bool:
        self._last_mouse = local
        self._press_at = local
        if button not in (1, 2, 3):
            return False
        # The 2026-10-04 audit, finding create-01 (Clay's and Mason's own guard
        # of 2026-10-03, which this viewer never received): a gizmo drag or a
        # marker grab owns the mouse until the left button comes up. A middle
        # press mid-drag replaced "gizmo" with "pan", so the left release found
        # no gizmo to close and the open ``editor.record()`` never exited --
        # ``PoseEditor._depth`` stayed 1 and every later edit nested inside the
        # dead step, pushing nothing: pose undo and redo were dead. ``getattr``
        # because the wiring tests drive this over a stub with no grab at all.
        if getattr(self, "_grab", None) in ("gizmo", "marker") and button != 1:
            return True
        if button == 1 and self.pose_mode and self.editor.bound:
            if self._mods()[2]:
                # The 2026-09-08 audit (finding create-01) found this method
                # never read Alt before testing what was under the cursor, so
                # an Alt+drag that started over a joint marker or a gizmo
                # handle grabbed it instead of orbiting -- quietly moving a
                # pose while the user was only trying to look around it, even
                # though the comment below and shortcuts.py both promise
                # Alt+drag orbits everywhere. Checked first and short-circuited,
                # the same order ``_view_drag.DragOps._press`` uses for Clay.
                self._grab = "orbit"
                return True
            origin, direction = self._ray(local)
            gizmo = self._active_gizmo()
            axis = gizmo.hit(origin, direction) if self.editor.selected else None
            if axis is not None and gizmo.begin(axis, origin, direction):
                self._grab = "gizmo"
                self._pose_step = self.editor.record()
                self._pose_step.__enter__()
                return True
            centres = {
                name: picking.to_world(self.placement, point)
                for name, point in self.editor.handles.items()
            }
            hit = picking.nearest_hit(
                origin, direction, centres, picking.marker_radius(self.radius)
            )
            if hit is not None:
                self.editor.selected = hit
                self._grab = "marker"
                return True
            # **A click on empty space deselects**, which is what every 3D
            # editor does and what this one made you press Esc for -- an
            # undocumented key on a gizmo the user is looking straight at. The
            # orbit below still starts, so a *drag* from empty space turns the
            # model as it always did; only a press that hits nothing and moves
            # nowhere clears the selection, which ``_release`` decides.
            self._deselect_on_click = self.editor.selected is not None
        # **Alt+drag orbits, as it does in Clay, and the middle button pans.**
        # Until 2026-09-05 Alt+drag *panned* here, with a comment citing Maya
        # and Blender, while ``_view_drag`` two viewports over said Alt+drag
        # "must never be reinterpreted" and orbited -- the same hand gesture
        # with opposite results depending on which 3-D view the pointer was
        # over, and the shortcut sheet advertising only Clay's. One rule, the
        # one already written down: Alt+drag orbits in every mode, in every
        # viewport. The right button does nothing here (this viewport has no
        # context menu: the joint menu left with Poser on 2026-10-14, the 2026-10-14
        # audit's finding create-viewer-03) rather than being a second orbit,
        # for the same reason Clay's does nothing on a drag: grabbing the
        # wrong button mid-gesture costs nothing.
        if button == 3:
            return True
        self._grab = "pan" if button == 2 else "orbit"
        return True

    def _release(self, button: int) -> bool:
        # A left press that hit no marker still starts an orbit -- a drag from
        # empty space must turn the model, as it always did -- so the deselect
        # is decided *here*, on the release, and only if the orbit never
        # actually turned anything. ``_motion`` clears the flag on the first
        # movement, which is what tells a click from a drag.
        if self._grab is not None:
            # The 2026-10-04 audit, finding create-02 (``ClayView._release_drag``'s
            # and Mason's own guard of 2026-10-03): a grab belongs to the button
            # that began it -- pan to the middle button, everything else to the
            # left -- and only that button's release ends it. A wheel notch
            # delivers MOUSEBUTTONUP 4/5, which used to end whatever was live,
            # so scrolling to zoom mid-orbit dropped the drag and mid-gizmo
            # committed the pose step early. Checked before the deselect flag
            # is read, so a stray release cannot consume a click's verdict.
            owner = 2 if self._grab == "pan" else 1
            if button != owner:
                return True
        pending, self._deselect_on_click = self._deselect_on_click, False
        if self._grab is None:
            if button == 1 and pending:
                self.editor.selected = None
                self._render_dirty = True
                return True
            return False
        was = self._grab
        self._grab = None
        if button == 1 and pending and was == "orbit":
            self.editor.selected = None
            self._render_dirty = True
            return True
        if was == "gizmo":
            gizmo = self._active_gizmo()
            if gizmo is not None:
                gizmo.end_drag()
            # Closed before the dirty notification, so anything that inspects
            # the history in response sees the finished step rather than an
            # open one. A gesture that ended where it started pushes nothing.
            self._close_pose_step()
            self._notify_pose_dirty()
        del button
        return True

    def _close_pose_step(self) -> None:
        """End the open gesture's undo step, if there is one.

        Idempotent and safe to call from a teardown path as well as from
        ``_release``: a gizmo drag that is still live when pose mode ends (the
        model is adopted, the mode is left) has to be closed rather than
        leaked, and ``record`` itself declines to push across a rebind.
        """
        # ``getattr`` rather than the attribute: ``exit_pose_mode`` is called
        # on partially-built viewers (the wiring tests drive the unbound method
        # over a shell), and a teardown path that raises on a missing field is
        # a teardown path that leaks whatever came after it.
        step = getattr(self, "_pose_step", None)
        self._pose_step = None
        if step is not None:
            step.__exit__(None, None, None)

    def _motion(self, local: tuple[float, float]) -> bool:
        # ``getattr``: the wiring tests drive this over a stub that never saw a press.
        anchor = getattr(self, "_press_at", None) or self._last_mouse
        dx = local[0] - self._last_mouse[0]
        dy = local[1] - self._last_mouse[1]
        self._last_mouse = local
        height = int(max(self._rect[3], 1))
        # The first real movement makes this a drag rather than a click, which
        # is what decides the deselect at the release. A threshold rather than
        # any motion at all: a pen reports sub-pixel jitter on a tap, and a
        # click that happened to wobble must still deselect. Measured from the
        # press: a slow orbit of many sub-2 px events never crossed a per-event
        # threshold, so the release cleared the joint the user had orbited around
        # (the 2026-10-14 audit, finding create-viewer-02).
        away_x = local[0] - anchor[0]
        away_y = local[1] - anchor[1]
        if away_x * away_x + away_y * away_y > 4:
            self._deselect_on_click = False
        if self._grab is None:
            if self.pose_mode and self.editor.selected is not None:
                origin, direction = self._ray(local)
                gizmo = self._active_gizmo()
                if gizmo is not None:
                    was, gizmo.hover = gizmo.hover, gizmo.hit(origin, direction)
                    # Only when it *changed*: the highlight is the whole of
                    # what a hover draws, so an unchanged one is a re-render of
                    # an identical picture.
                    if gizmo.hover != was:
                        self._render_dirty = True
            return False
        self._render_dirty = True
        if self._grab == "orbit":
            self.camera.orbit(dx, dy, height)
        elif self._grab == "pan":
            self.camera.pan(dx, dy, height)
        elif self._grab == "gizmo":
            origin, direction = self._ray(local)
            gizmo = self._active_gizmo()
            if gizmo is None:
                # The 2026-10-03 audit, finding create-11: pose mode ended
                # mid-drag (a model adopted, ``clear``) and ``_grab`` still said
                # "gizmo", so every motion until the button came up raised
                # ``'NoneType' has no attribute 'update'``. ``exit_pose_mode``
                # drops the grab now; this is the same guard ``_release`` keeps.
                self._grab = None
                return True
            # Dispatch on which gizmo is active, not on the mode: only joints
            # mode holds the translate gizmo, but the gizmo is what has the
            # drag state.
            if gizmo is self.translate_gizmo:
                moved = gizmo.update(origin, direction)
                if moved is not None:
                    point = picking.from_world(self.placement, moved)
                    self.editor.move_handle(self.editor.selected, point)
            else:
                delta = gizmo.update(origin, direction)
                if delta is not None:
                    self.editor.rotate_selected(delta)
                    if self.gpu is not None:
                        self.gpu.refresh_palettes()
        return True

    def _ray(self, local: tuple[float, float]):
        return screen_ray(
            self.camera, local[0], local[1], int(self._rect[2]), int(self._rect[3])
        )

    # -- teardown ----------------------------------------------------------

    def release(self) -> None:
        self.clear_reference()
        self._release_model()
        self.exit_compare()
        self.markers.release()
        self.bonelines.release()
        self.rotate_gizmo.release()
        self.translate_gizmo.release()
        self._forget(self.viewport.texture)
        self.viewport.release()
        if self.compare_viewport is not None:
            self._forget(self.compare_viewport.texture)
            self.compare_viewport.release()
        self.renderer.release()


#: The task key every viewport load is submitted under, and the one
#: ``App._adopt_model`` lands. One key, so a selection moving faster than the
#: disk cannot pile up loads. Named here rather than in the frame loop because
#: a mode module that wants a picture shown must not import ``main``.
LOAD_KEY = "viewer-load"


def request_reference(ctx: Any, path: Path) -> bool:
    """Show a picture, decoded off the frame thread. -> was it submitted?

    ``Viewer.load_reference``'s off-thread spelling, and the one a mode should
    reach for: the blocking version froze the frame at exactly the moment the
    user had pressed something. The result lands through ``App._adopt_model``,
    which checks ``pending`` and tells a picture from a mesh by the suffix.
    """

    viewer = getattr(ctx, "viewer", None)
    if viewer is None:
        return False
    previous = viewer.pending
    viewer.pending = path
    if not ctx.submit(LOAD_KEY, viewer.parse_reference, path, tag=path):
        # Another load is in flight; its result is checked against ``pending``
        # before it is adopted, so this one is simply dropped rather than
        # queued -- ``_sync_viewer``'s own rule. The *previous* ``pending`` is
        # put back, not None: that is the in-flight load's own freshness check,
        # and clearing it made its result land as unwanted (the 2026-10-04
        # audit, finding create-13).
        viewer.pending = previous
        return False
    return True
