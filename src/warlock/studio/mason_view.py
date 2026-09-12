"""The Mason viewport: many placements of few assets, drawn in one pass.

A **sibling** to :class:`~.clay_view.ClayView` rather than a subclass. Element
modes, the vertex marquee, proportional editing and the whole of
``_view_drag.py`` are Clay's mouse map -- a modeller's -- and a scene editor
does not want them. What the two genuinely share is frame plumbing, and that
already lives in :mod:`._view_frame` (``FrameOps``): the release-before-forget
pair, the redraw decision, ``_local``, ``_mods``, the four-pixel right-button
rule and :class:`~._view_frame.Composite`. ``ClayView`` is
``(CacheOps, BoundsOps, PickOps, OverlayOps, DragOps, FrameOps)``; this mixes
in ``FrameOps`` and nothing else of Clay's.

The orbit/pan/dolly lines themselves deliberately did **not** move into that
leaf, and this module is where that decision is paid for: they sit interleaved
with Clay's element selection and marquee inside ``_view_drag``'s dispatchers,
and separating them would have been a restructure of those rather than the code
motion the extraction was. ``camera.orbit``/``pan``/``dolly`` are one line each
and are called directly below; what was worth sharing was never those three
calls but the modifier read and the four-pixel rule around them.

Five things Mason does differently from Clay, and each is a measured bug rather
than a preference.

**The GPU cache key is the ref, not the node.** ``(ref_key(node.ref),
id(material))``. Clay keys per object because in Clay every object owns its
mesh; Mason keys per *asset*, and that single change is what makes five hundred
instances one upload. Each entry pins whatever its ``id()``s name -- the
material object and the primitive list -- because an id is only sound while its
object is alive, the trap ``viewer/picking.cached_bvh`` and Clay's own
``_view_cache._Entry`` both already had to write down.

**The composite may not write through a shared node.** ``clay_view._composite``
does ``node.world = world`` on the cached entry's own ``gltf.Node``, which is
sound at one entry per object and silently wrong at one entry per ref: N
instances share the node, the last write wins, and N-1 draw stacked at whichever
was composited last. Nothing raises and the frame renders, which is what makes
it the quietest bug in the mode. :class:`~.mason.scene.DrawNode` carries
``world`` and ``skin`` -- the only two attributes ``Renderer._draw_model`` reads
off a node in the composite path -- and :class:`~.mason.scene.NodePool` pools
them per frame so a thousand-item scene does not allocate a thousand objects a
frame. Both are pure and headless and landed in Stage C with the failing test
written first; this module uses them rather than rediscovering them.

**The redraw key carries the geometry source's revision.** An asset finishing
its background parse changes the picture with *no document edit* -- no node
moved, no undo step was pushed -- so without ``(id(source), source.rev)`` in the
key the adopting frame is skipped as "nothing moved" and the asset appears only
when something else forces a redraw. ``GeometrySource.rev`` exists for exactly
this and its own docstring says so.

**Picking is per ref, selection is per owner.** ``mason.pick.ray_scene`` keys
its BVH on the primitives a ``GeometrySource`` returned rather than on the ref,
because a ``Ref`` compares equal precisely when the geometry behind it changes
-- which kept a stale tree alive across a relink. A hit inside a prefab instance
maps back to the instance through ``Placed.owner``, never through the path:
``owner`` is recorded by the walk because a bare path cannot say where the
prefab boundary falls.

**Frustum culling and a transparency ordering are Mason's and not Clay's.** Clay
draws one asset and needs neither. Culling is skipped under
:data:`CULL_THRESHOLD` because the test itself costs something, and a library
asset's material need not be opaque where every Clay material is.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._view_frame import Composite, FrameOps
from .mason import pick as mpick
from .mason import scene as msc
from .mason.refs import ref_key
from .viewer import capture, glctx
from .viewer import math3d as m3
from .viewer import scene as scenelib
from .viewer.camera import Camera, screen_ray
from .viewer.gizmo import RotateGizmo, ScaleGizmo, TranslateGizmo
from .viewer.render import Renderer

log = logging.getLogger(__name__)


#: Which gizmo each tool drives -- Clay's table, restated here rather than
#: imported because the *tools* are Mason's own (``mason_state.TOOLS``) and a
#: shared lookup would tie two modes' tool sets together for the sake of four
#: entries that happen to agree today.
GIZMO_FOR_TOOL = {"move": "translate", "rotate": "rotate", "scale": "scale"}

#: How opaque the surface is in X-ray, matching Clay's so the same key means
#: the same thing in both 3-D workspaces.
XRAY_ALPHA = 0.33

#: Below this many placed items, the frustum test is skipped and everything is
#: submitted. The test is not free -- eight corners transformed and up to six
#: plane dot products per item -- and under a few hundred placements the GPU
#: discards an off-screen draw more cheaply than Python decides not to make it.
#:
#: **Chosen rather than measured, and that is defensible here only because
#: nothing stored is keyed on it.** This repo's rule is that a constant the
#: stored corpus depends on gets a dated ``docs/measurements/`` document
#: *before* it is fixed, which is why ``scene.MAX_PLACED``,
#: ``scene.PLACED_WARN_THRESHOLD`` and ``terrain.MAX_TERRAIN_SIDE`` each have
#: one. This number decides only whether a visibility test runs on a given
#: frame: every scene draws identically either side of it, no ``.wscn`` records
#: it, and moving it tomorrow invalidates nothing already saved. Set at 256
#: because that is an order of magnitude below ``PLACED_WARN_THRESHOLD``
#: (1,500, the measured point where resolving alone costs a third of a frame),
#: so culling is already running well before a scene is large enough for
#: anything to be tight.
CULL_THRESHOLD = 256

#: A 3x3 whose determinant is under this is treated as having no inverse.
#: ``objout._normal_matrix``'s guard and its reason: ``np.linalg.inv`` does not
#: reliably *raise* on a matrix scaled to zero on one axis, it can hand back
#: ``inf``/``nan`` instead, so the determinant is checked first and the result
#: checked again afterwards.
_SINGULAR_DET_EPS = 1e-12


class _Entry:
    """One *ref*'s GPU state, and the key that says whether it is still valid.

    ``__slots__`` and the three pins are the whole class. ``ref``, ``material``
    and ``prims`` are held because the key names them by ``id()``, and an id is
    an address CPython is free to hand to a different object the moment the old
    one is collected: a freed material's address arriving on a different
    material would otherwise make ``entry.key == key`` true of a stale upload,
    which is the viewport drawing last scene's surface forever with nothing in
    the data to say why.
    """

    __slots__ = ("key", "gpu", "model", "ref", "material", "prims")

    def __init__(
        self, key: Any, gpu: Any, model: Any, ref: Any, material: Any, prims: Any
    ) -> None:
        self.key = key
        self.gpu = gpu
        self.model = model
        self.ref = ref
        self.material = material
        self.prims = prims


def _entry_key(placed: Any) -> tuple[Any, ...]:
    """The cache identity of one placed item: its ref, and its material override.

    **The transform is deliberately absent**, Clay's rule for its own key: a
    world matrix is a uniform written per draw, not a buffer, so moving an
    instance must not rebuild anything. And the *node* is absent, which is the
    whole difference from Clay: two hundred placements of one barrel share this
    key and therefore share one upload.

    The material override is in the key by identity, which costs a second
    upload of identical geometry whenever an instance overrides its material.
    That is a known price rather than an oversight -- the alternative is a
    per-draw material uniform, which is a change to ``Renderer._draw_model``
    and belongs to whoever finds overrides common enough to want it.
    """
    return (ref_key(placed.ref), id(placed.material))


class MasonView(FrameOps):
    """Mason's viewport, from the UI's point of view."""

    def __init__(self, ctx: Any, app_ctx: Any = None) -> None:
        """``ctx`` is the moderngl context; ``app_ctx`` the application's.

        Two arguments for :class:`~.clay_view.ClayView`'s stated reason, which
        is the same one here: everything that draws needs the GL context, and
        the only thing that needs the app is reading which transform tool and
        which pivot are selected -- both *app* settings shared across
        documents, so the view reads them rather than holding copies that could
        drift.
        """
        self.ctx = ctx
        self.app_ctx = app_ctx
        self.renderer = Renderer(ctx)
        self.viewport = glctx.Viewport(ctx, (16, 16))
        self.camera = Camera()

        # Set by the pane each frame, exactly as Clay's are: a view setting
        # with no reader is a switch that does nothing, which is the bug
        # ``ClayView.show_grid`` was added to close.
        self.wireframe = False
        self.flat = False
        self.wire_overlay = False
        self.xray = False
        self.show_grid = True

        self.translate_gizmo = TranslateGizmo(ctx, self.renderer.programs)
        self.rotate_gizmo = RotateGizmo(ctx, self.renderer.programs)
        self.scale_gizmo = ScaleGizmo(ctx, self.renderer.programs)

        self._cache: dict[tuple[Any, ...], _Entry] = {}
        # Counted rather than inferred: "five hundred instances are one upload"
        # is the claim this whole module is built around, and there is no other
        # way to see it from outside.
        self.rebuilds = 0
        # The per-draw proxies, pooled across frames. See the module docstring.
        self._pool = msc.NodePool()

        self._rect = (0.0, 0.0, 1.0, 1.0)
        self._grab: str | None = None  # orbit | pan | gizmo
        self._last_mouse = (0.0, 0.0)
        self._rmb_at: tuple[float, float] | None = None
        self.menu_request: tuple[float, float] | None = None
        # Where an Alt press went down. Alt+drag orbits in every 3-D view in
        # this app and must never be reinterpreted: it is how a user looks at
        # what they are about to click.
        self._alt_at: tuple[float, float] | None = None

        # Redraw bookkeeping, ``FrameOps._frame_unchanged``'s shape.
        self._render_dirty = True
        self._last_render_key: Any = None
        # The document and the source the key above was drawn against, pinned
        # for the reason every identity-keyed cache in this tree states: an id
        # is only sound while its object is alive. Clay's own ``_last_doc`` was
        # found bare by the 2026-09-07 audit (clay-09) -- a closed document
        # collected and a new one minted at the same address, with ``rev``
        # starting at 0 as it does for every fresh document, handed back the
        # closed tab's stale texture. The source is pinned for the same reason
        # and it matters more here, because its ``rev`` is the only thing that
        # ever says an asset arrived.
        self._last_doc: Any = None
        self._last_source: Any = None

        # What ``resolve`` last answered, and the key it was answered for.
        # Memoized because a frame asks for it up to four times (the sync, the
        # composite, the gizmo centre, the HUD) and a resolve at the warn
        # threshold was measured at a third of a frame on its own.
        self._placed: list[Any] = []
        self._placed_key: Any = None
        # How many the frustum test threw away last frame -- read by the HUD,
        # which is the only honest way to see culling working.
        self.culled = 0

        # A live gizmo drag: every selected node's transform at the press, the
        # drag's accumulated rotation, and the pivot it started from. Recorded
        # at the press rather than read per frame, because
        # ``MasonDoc.set_transform``'s ``was`` argument wants the pre-drag
        # values and the node already holds the new ones by the release.
        self._drag_start: dict[int, tuple[Any, Any, Any]] = {}
        self._drag_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._drag_origin = np.zeros(3)
        self._drag_pivot = np.zeros(3)

    # -- the app's settings ------------------------------------------------

    @property
    def state(self) -> Any:
        """Mason's state, or ``None`` when the view is driven headlessly."""
        app_ctx = self.app_ctx
        return None if app_ctx is None else getattr(app_ctx.state, "mason", None)

    @property
    def grabbing(self) -> bool:
        """Whether any pointer gesture is in progress.

        Named ``grabbing`` rather than ``dragging`` deliberately, following the
        2026-09-11 audit's clay-02: Clay had a broad ``dragging`` property on
        the class shadowing a mixin's narrow one, so a bare tool key typed
        while merely orbiting was routed into the drag handler and eaten.
        :attr:`dragging` below keeps the narrow meaning in both modes.
        """
        return self._grab is not None

    @property
    def dragging(self) -> bool:
        """Whether a *transform* drag is live -- not an orbit and not a pan."""
        return self._grab == "gizmo"

    # -- resolving ---------------------------------------------------------

    def resolved(self, doc: Any) -> list[Any]:
        """Every drawable placement in ``doc``, memoized for this frame.

        Keyed on ``(id(doc), doc.rev)`` with the document pinned, which is the
        same soundness argument the redraw key makes: ``rev`` moves on every
        edit, selection change and visibility change, and the pin is what stops
        a collected document's address being reused under a stale answer.

        A document past ``MAX_PLACED`` raises out of ``resolve``; that is the
        resolver refusing rather than silently drawing a truncated scene, and
        it is caught here so the refusal reaches the user as an empty viewport
        and a log line rather than as an exception on the frame thread.
        """
        key = (id(doc), doc.rev)
        if self._placed_key == key and self._last_doc is doc:
            return self._placed
        try:
            placed = msc.resolve(doc)
        except ValueError:
            log.exception("this scene could not be resolved; drawing nothing")
            placed = []
        self._placed = placed
        self._placed_key = key
        self._last_doc = doc
        return placed

    # -- the GPU cache -----------------------------------------------------

    def sync(self, doc: Any, source: Any) -> None:
        """Bring the GPU up to date, uploading one buffer set per *ref*.

        ``live`` is a set of cache keys rather than of uids, which is the whole
        of what makes N instances one upload: every placement of one asset
        produces the same key, so the second and the five-hundredth find the
        entry the first one built.

        A ref that has not resolved yet -- an asset still parsing on a task
        thread -- contributes no entry and draws nothing this frame. That is
        not an error and must not poison the cache: the parse finishing bumps
        ``source.rev``, the redraw key changes, and the next frame builds it.
        """
        live: set[tuple[Any, ...]] = set()
        for placed in self.resolved(doc):
            if not placed.visible or placed.ref is None:
                continue
            key = _entry_key(placed)
            if key in live:
                continue
            entry = self._cache.get(key)
            if entry is not None and entry.key == key:
                live.add(key)
                continue
            prims = source.primitives(placed.ref)
            if not prims:
                continue
            live.add(key)
            if entry is not None:
                entry.gpu.release()
            self._cache[key] = self._build(placed, key, prims)
            self.rebuilds += 1
        for key in [k for k in self._cache if k not in live]:
            # A ref nothing places any more takes its buffers with it, the rule
            # Clay's cache already applies to a hidden object: holding an
            # upload for geometry that does not draw, does not export and is
            # not picked would make it the one of the three that is only half
            # true.
            self._cache.pop(key).gpu.release()

    def _build(self, placed: Any, key: tuple[Any, ...], prims: Any) -> _Entry:
        """One ref's upload.

        A material override is applied by *replacing* each primitive's material
        on a shallow copy rather than by mutating what the source handed back:
        those primitives belong to the asset cache and are shared by every
        other placement of the same ref, so writing through them would hand one
        instance's override to all of them.
        """
        from dataclasses import replace as _replace

        from .viewer import gltf

        if placed.material is not None:
            prims = [_replace(p, material=placed.material) for p in prims]
        node = gltf.Node(name=getattr(placed.node, "name", "") or "node", mesh=0)
        model = gltf.Model([node], [0], [list(prims)], [])
        return _Entry(
            key,
            scenelib.GpuModel(self.ctx, model),
            model,
            placed.ref,
            placed.material,
            prims,
        )

    def clear(self) -> None:
        for entry in self._cache.values():
            entry.gpu.release()
        self._cache.clear()
        self._placed = []
        self._placed_key = None

    # -- drawing -----------------------------------------------------------

    def draw(
        self, doc: Any, source: Any, rect: tuple[float, float, float, float], dt: float
    ) -> Any:
        """Draw one frame into the viewport. -> the resolved texture.

        Skipped -- the last texture returned as-is -- when nothing that feeds
        the draw moved. The key carries the document's own ``rev`` (every edit,
        selection and visibility change), every view setting that changes the
        picture, the tool and pivot that decide which gizmo is on screen and
        where, and **the geometry source's revision**, which is the one of
        those that is Mason's alone: an asset finishing its parse changes the
        picture with no document edit at all, and without ``source.rev`` here
        the adopting frame is skipped as "nothing moved".
        """
        self._rect = rect
        width, height = int(max(rect[2], 1)), int(max(rect[3], 1))
        key = (
            width,
            height,
            bool(self.wireframe),
            bool(self.show_grid),
            bool(self.flat),
            bool(self.wire_overlay),
            bool(self.xray),
            id(doc),
            doc.rev,
            id(source),
            int(getattr(source, "rev", 0)),
            str(getattr(self.state, "tool", "select")),
            str(getattr(self.state, "pivot", "median")),
        )
        if self._frame_unchanged(key):
            return self.viewport.texture
        self._last_render_key = key
        self._last_doc = doc
        self._last_source = source
        self._render_dirty = False
        self._resize(width, height)
        self.camera.update(dt)
        self.sync(doc, source)

        self.renderer.draw(
            self.viewport,
            self.camera,
            self._composite(doc, source),
            wireframe=self.wireframe,
            flat=self.flat,
            show_grid=self.show_grid,
            wire_overlay=self.wire_overlay,
            alpha=XRAY_ALPHA if self.xray else 1.0,
            overlays=self._gizmo_draws(doc, source, height),
        )
        return self.viewport.texture

    def _composite(self, doc: Any, source: Any) -> Any:
        """Every visible placement as one thing the renderer draws in one pass.

        ``Renderer.draw`` clears the target it is given, so a call per item
        would erase the one before it and would mean a grid pass and an overlay
        pass apiece. What it consumes from a ``GpuModel`` is ``draws`` and
        ``palette``, so the composite supplies exactly those.

        **Every placement gets its own proxy.** See the module docstring: the
        cached entry's ``gltf.Node`` is shared by every placement of that ref,
        so writing ``node.world`` on it -- which is precisely what Clay does,
        and correctly -- would leave N-1 instances drawing wherever the last
        write put them. One :class:`~.mason.scene.DrawNode` per placement, out
        of the pool, is the fix: the pool is what makes paying for it
        affordable and the distinctness is what makes it correct, and neither
        one alone is enough.

        Opaque first, then blended back-to-front. Clay needs no such ordering
        because every Clay material is opaque; a library asset's need not be,
        and a blended surface drawn before what is behind it composites against
        a background that has not been painted yet. Ordering the one draw list
        is all this does -- the depth-mask handling for a blended pass stays
        the renderer's, which is why overlapping transparent props remain
        approximate rather than exact.
        """
        placed = self.resolved(doc)
        planes = None
        if len(placed) >= CULL_THRESHOLD:
            planes = _frustum_planes(self.camera.projection() @ self.camera.view())
        self._pool.frame()
        culled = 0
        opaque: list[tuple[Any, Any]] = []
        blended: list[tuple[float, Any, Any]] = []
        eye = np.asarray(self.camera.position, dtype="f8")
        for item in placed:
            if not item.visible or item.ref is None:
                continue
            entry = self._cache.get(_entry_key(item))
            if entry is None:
                continue
            if planes is not None:
                box = source.box(item.ref)
                if box is not None and not _box_visible(planes, box, item.world):
                    culled += 1
                    continue
            proxy = self._pool.node(item.world)
            depth = float(np.linalg.norm(item.world[:3, 3] - eye))
            for _node, primitive in entry.gpu.draws:
                material = getattr(primitive.primitive, "material", None)
                if getattr(material, "alpha_mode", "OPAQUE") == "BLEND":
                    blended.append((depth, proxy, primitive))
                else:
                    opaque.append((proxy, primitive))
        self.culled = culled
        blended.sort(key=lambda row: -row[0])
        draws = opaque + [(proxy, primitive) for _depth, proxy, primitive in blended]
        return Composite(draws) if draws else None

    def _gizmo_draws(self, doc: Any, source: Any, height: int) -> list[Any]:
        gizmo = self.active_gizmo(doc)
        if gizmo is None:
            return []
        centre = self.selection_centre(doc, source)
        if centre is None:
            return []
        gizmo.place(centre, m3.identity(), self.camera, height)
        return gizmo.draws()

    def active_gizmo(self, doc: Any) -> Any:
        """The gizmo for the current tool, or ``None``.

        The tool lives on the mode's state rather than here because it is an
        *app* setting shared across documents. Select draws no gizmo, which is
        what leaves the left button free for picking.
        """
        kind = GIZMO_FOR_TOOL.get(str(getattr(self.state, "tool", "select")), "")
        if not kind or not doc.selection:
            return None
        return {
            "translate": self.translate_gizmo,
            "rotate": self.rotate_gizmo,
            "scale": self.scale_gizmo,
        }[kind]

    # -- bounds and framing ------------------------------------------------

    def world_bounds(self, doc: Any, source: Any, uids: Any = None) -> Any:
        """The engine's own answer, not a second one computed here.

        ``mason.scene.world_bounds`` already filters by ``Placed.owner`` --
        "picking is per ref, selection is per owner" made computable -- so a
        selected prefab instance frames every leaf under it, addressed by the
        one uid the outliner has a row for.
        """
        return msc.world_bounds(doc, source, uids)

    def frame_selection(self, doc: Any, source: Any) -> None:
        """Fit the camera to the selection, or to the whole scene.

        Re-aimed at the measured centre of the box afterwards, for the reason
        ``ClayView.render_png`` already writes down: ``Camera.frame`` aims at
        half the box's *height* above the origin, which centres a **grounded**
        subject because that is what the asset viewer was written for. A Mason
        scene is grounded at y=0 as a whole, but a selected prop three metres
        up is not, and framing it without re-aiming looks at the air above it.
        """
        uids = sorted(doc.selection) or None
        bounds = self.world_bounds(doc, source, uids)
        if bounds is None:
            bounds = self.world_bounds(doc, source, None)
        if bounds is None:
            return
        lo, hi = bounds
        self.camera.frame(lo, hi)
        self.camera.set_target((np.asarray(lo) + np.asarray(hi)) * 0.5)
        # Paired with the frame, as ``BoundsOps.frame_selection`` already pairs
        # them: a ground plane left at the previous subject's span reads as the
        # scene having changed size.
        self.renderer.fit_grid(lo, hi)
        self._render_dirty = True

    def selection_centre(self, doc: Any, source: Any) -> Any:
        """Where the gizmo sits: the pivot the user chose, over the selection.

        Three pivots rather than one, because a scene editor's selection is a
        *set*: the median of the selected placements, the active one alone, or
        the world origin. ``median`` is the default and is the median of the
        placements' world *positions* rather than of their bounding boxes --
        the position is what the user dragged each item to, where a box's
        centre moves the day an asset is re-exported off-centre.
        """
        selection = doc.selection
        if not selection:
            return None
        pivot = str(getattr(self.state, "pivot", "median"))
        if pivot == "origin":
            return np.zeros(3)
        points = [item.world[:3, 3] for item in self.resolved(doc) if item.owner in selection]
        if not points:
            # A group, a light or a camera: nothing drawable under it, so
            # ``resolve`` never reports one. ``resolved_for`` answers for any
            # node, which is what a properties panel and a gizmo both need.
            for uid in sorted(selection):
                found = msc.resolved_for(doc, uid)
                if found is not None:
                    points.append(found.world[:3, 3])
        if not points:
            return None
        if pivot == "active":
            return np.asarray(points[0], dtype="f8")
        return np.asarray(points, dtype="f8").mean(axis=0)

    # -- picking -----------------------------------------------------------

    def _ray(self, local: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        return screen_ray(self.camera, local[0], local[1], int(self._rect[2]), int(self._rect[3]))

    def pick(self, doc: Any, source: Any, local: tuple[float, float]) -> Any:
        """What is under the cursor, as a :class:`~.mason.pick.Hit` or ``None``.

        Terrain is handed in separately because ``ray_scene`` needs the height
        field itself and its placement, while everything else about it -- its
        owner, its lock, its visibility -- the resolver has already worked out
        and put on its :class:`~.mason.scene.Placed`.
        """
        origin, direction = self._ray(local)
        terrain_world = None
        if doc.terrain is not None:
            terrain_node = _terrain_node()
            found = next(
                (p for p in self.resolved(doc) if isinstance(p.node, terrain_node)), None
            )
            terrain_world = None if found is None else found.world
        return mpick.ray_scene(
            self.resolved(doc),
            source,
            origin,
            direction,
            terrain=doc.terrain,
            terrain_world=terrain_world,
        )

    # -- input -------------------------------------------------------------

    def handle_event(self, doc: Any, source: Any, event: Any, hovered: bool) -> bool:
        """Feed one pygame event. -> whether the viewport consumed it.

        ``hovered`` is whether the pointer is over the viewport image; a drag
        already in progress ignores it, so crossing onto a panel mid-orbit does
        not drop the drag. Every event that reaches here can move the picture,
        so the frame is marked dirty rather than each branch deciding -- one
        redraw is cheaper than enumerating which.
        """
        import pygame

        self._render_dirty = True
        local = self._local(event)
        if event.type == pygame.MOUSEBUTTONDOWN and hovered:
            return self._press(doc, source, event.button, local)
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3:
                return self._rmb_release(local)
            return self._release(doc, event.button)
        if event.type == pygame.MOUSEMOTION:
            return self._motion(doc, source, local)
        if event.type == pygame.MOUSEWHEEL and hovered:
            self.camera.dolly(event.y)
            return True
        return False

    def _press(self, doc: Any, source: Any, button: int, local: tuple[float, float]) -> bool:
        self._last_mouse = local
        # A gizmo drag owns the mouse until its button comes up. Without this,
        # a middle press mid-drag overwrote ``_grab`` with "pan" and the left
        # release then found nothing to commit -- the object stranded wherever
        # the last motion put it, with no history step and the gizmo still
        # holding a live drag. Clay's own copy of this guard records it.
        if self._grab == "gizmo" and button != 1:
            return True
        if button == 3:
            self._rmb_at = local
            return True
        if button == 2:
            self._grab = "pan"
            return True
        if button != 1:
            return False

        shift, ctrl, alt = self._mods()
        if alt:
            # Alt+drag always orbits, in every mode: it is the one gesture that
            # must never be reinterpreted, because it is how a user looks at
            # what they are about to click.
            self._grab = "orbit"
            self._alt_at = local
            return True

        origin, direction = self._ray(local)
        gizmo = self.active_gizmo(doc)
        axis = gizmo.hit(origin, direction) if gizmo is not None else None
        if axis is not None and gizmo.begin(axis, origin, direction):
            self._begin_gizmo_drag(doc, source)
            return True

        hit = self.pick(doc, source, local)
        owner = None if hit is None else hit.owner
        if shift or ctrl:
            # Both *extend* rather than replace, which is what a scene editor's
            # selection wants and what this app's outliners already do. Ctrl
            # toggles rather than only adding, so a mis-clicked prop comes back
            # out without starting the selection over.
            selection = set(doc.selection)
            if owner is not None:
                if ctrl:
                    selection.symmetric_difference_update({owner})
                else:
                    selection.add(owner)
            doc.select(selection)
        else:
            doc.select([owner] if owner is not None else [])
        # The press that selected also arms an orbit, exactly as Clay's does: a
        # click selects, a drag from the same press turns the camera, and the
        # two are told apart by whether the pointer travelled.
        self._grab = "orbit"
        return True

    def _motion(self, doc: Any, source: Any, local: tuple[float, float]) -> bool:
        dx = local[0] - self._last_mouse[0]
        dy = local[1] - self._last_mouse[1]
        self._last_mouse = local
        height = int(max(self._rect[3], 1))
        if self._grab is None:
            gizmo = self.active_gizmo(doc)
            if gizmo is not None:
                origin, direction = self._ray(local)
                gizmo.hover = gizmo.hit(origin, direction)
            return False
        if self._grab == "orbit":
            self.camera.orbit(dx, dy, height)
        elif self._grab == "pan":
            self.camera.pan(dx, dy, height)
        elif self._grab == "gizmo":
            self._drag_gizmo(doc, source, local)
        return True

    def _release(self, doc: Any, button: int = 1) -> bool:
        was, self._grab = self._grab, None
        self._alt_at = None
        if was == "gizmo":
            self._end_gizmo_drag(doc)
        return was is not None

    # -- the gizmo drag ----------------------------------------------------

    def _begin_gizmo_drag(self, doc: Any, source: Any) -> None:
        """Record every selected node's transform at the press.

        Recorded rather than read per frame, which is what
        ``MasonDoc.set_transform``'s ``was`` argument exists for: the node
        already holds the *new* values by the time the drag is released, so
        reading "before" off it then would compare a value against itself and
        record nothing -- an undo step that undoes the drag to where the drag
        already is.
        """
        self._grab = "gizmo"
        self._drag_start = {}
        for uid in sorted(doc.selection):
            node = doc.node(uid)
            if node is None or node.locked:
                # The engine's standing rule is that a lock is *reported, never
                # enforced* -- a lock stops the user and not the document. This
                # is the layer that stops the user.
                continue
            self._drag_start[uid] = tuple(np.array(v, copy=True) for v in node.trs())
        centre = self.selection_centre(doc, source)
        self._drag_pivot = np.zeros(3) if centre is None else np.asarray(centre, dtype="f8")
        self._drag_origin = self._drag_pivot.copy()
        self._drag_quat = np.array([0.0, 0.0, 0.0, 1.0])

    def _drag_gizmo(self, doc: Any, source: Any, local: tuple[float, float]) -> None:
        """One frame of a live drag: mutate the nodes, push nothing.

        Nothing is pushed until the release, which is what makes a drag one
        undo step rather than one step per mouse-move.
        """
        origin, direction = self._ray(local)
        tool = str(getattr(self.state, "tool", "select"))
        if tool == "move":
            point = self.translate_gizmo.update(origin, direction)
            if point is None:
                return
            self._apply_drag(doc, delta=np.asarray(point, dtype="f8") - self._drag_origin)
        elif tool == "rotate":
            step = self.rotate_gizmo.update(origin, direction)
            if step is None:
                return
            # ``RotateGizmo.update`` hands back the *increment* since the last
            # call, so it has to be accumulated into a total: the drag is
            # applied against the transforms recorded at the press, and an
            # increment applied to those would be the last mouse-move alone.
            self._drag_quat = m3.quat_mul(np.asarray(step, dtype="f8"), self._drag_quat)
            self._apply_drag(doc, quat=self._drag_quat)
        elif tool == "scale":
            factor = self.scale_gizmo.update(origin, direction)
            if factor is None:
                return
            self._apply_drag(doc, scale=np.asarray(factor, dtype="f8"))

    def _apply_drag(
        self, doc: Any, *, delta: Any = None, quat: Any = None, scale: Any = None
    ) -> None:
        """Write the drag onto every node it holds, in that node's *parent* space.

        A world-space delta is not a local translation for anything under a
        group, and Mason's whole reason for having a hierarchy is that things
        *are* under groups. The parent's world matrix is fetched through
        ``resolved_for`` on the parent uid rather than recovered as ``world @
        inv(node.local())`` -- the inverse-per-node this package already
        refused once, singular the moment anything is scaled to zero, for a
        matrix the resolver is holding in its hand anyway.

        A parent whose own basis has no inverse (scaled flat on an axis) is
        skipped rather than written with a non-finite transform, which is the
        guard ``objout._normal_matrix`` states the reason for.
        """
        for uid, (t0, r0, s0) in self._drag_start.items():
            node = doc.node(uid)
            if node is None:
                continue
            basis = self._parent_basis(doc, uid)
            if basis is None:
                continue
            inverse, parent_world = basis
            translation, rotation, scale_out = t0, r0, s0
            if delta is not None:
                translation = t0 + inverse @ np.asarray(delta, dtype="f8")
            elif quat is not None:
                turn = np.asarray(quat, dtype="f8")
                rotation = m3.quat_mul(turn, r0)
                # The position orbits the pivot as well as the node turning on
                # the spot, which is what makes a multi-node rotate turn the
                # *arrangement* rather than spin each item where it stands.
                world_point = parent_world[:3, :3] @ t0 + parent_world[:3, 3]
                turned = m3.quat_rotate(turn, world_point - self._drag_pivot)
                translation = inverse @ (self._drag_pivot + turned - parent_world[:3, 3])
            elif scale is not None:
                factor = np.asarray(scale, dtype="f8")
                scale_out = s0 * factor
                world_point = parent_world[:3, :3] @ t0 + parent_world[:3, 3]
                offset = (world_point - self._drag_pivot) * factor
                translation = inverse @ (self._drag_pivot + offset - parent_world[:3, 3])
            # **Rebound, never written through.** ``Node.local()`` memoizes its
            # matrix against the *identity* of these three arrays, so
            # ``node.translation[:] = ...`` changes the numbers without
            # changing the objects and the memo keeps handing back the matrix
            # from before the drag -- the node would hold the new transform and
            # the viewport would draw the old one, with nothing in the data to
            # say why. ``nodes.Node.trs``'s own docstring states this rule
            # ("rebind them to change the transform, never write through
            # them"), and the cost of obeying it is a memo miss per mouse-move
            # on the nodes that are actually moving, which is the case the memo
            # was never there to serve.
            node.translation = np.array(translation, dtype="f8")
            node.rotation = np.array(rotation, dtype="f8")
            node.scale = np.array(scale_out, dtype="f8")
        doc.touch()

    def _parent_basis(self, doc: Any, uid: int) -> Any:
        """``(inverse_of_the_parent_basis, parent_world)``, or ``None``.

        Identity at the root, which is the common case and costs nothing.
        """
        parent_uid = doc.parent_uid_of(uid)
        if parent_uid is None:
            return np.eye(3), np.eye(4)
        found = msc.resolved_for(doc, parent_uid)
        if found is None:
            return np.eye(3), np.eye(4)
        world = np.asarray(found.world, dtype="f8")
        basis = world[:3, :3]
        det = float(np.linalg.det(basis))
        if not np.isfinite(det) or abs(det) < _SINGULAR_DET_EPS:
            return None
        try:
            inverse = np.linalg.inv(basis)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(inverse).all():
            return None
        return inverse, world

    def _end_gizmo_drag(self, doc: Any) -> None:
        """Commit the drag as **one** compound step.

        One step rather than one per node, for the reason the 2026-09-07
        audit's clay-02 records about Clay's Delete: a selection spanning
        several objects that pushed a step apiece meant one Ctrl+Z undid only
        the last of them, which reads as an undo that does not work.
        """
        start, self._drag_start = self._drag_start, {}
        if not start:
            return
        mark = doc.mark()
        for uid, was in start.items():
            node = doc.node(uid)
            if node is None:
                continue
            doc.set_transform(
                uid,
                translation=node.translation,
                rotation=node.rotation,
                scale=node.scale,
                was=was,
            )
        doc.collapse_since(mark)

    def cancel_drag(self, doc: Any) -> None:
        """Put every node back where the press found it, and push nothing."""
        start, self._drag_start = self._drag_start, {}
        for uid, (t0, r0, s0) in start.items():
            node = doc.node(uid)
            if node is None:
                continue
            # Rebound for ``_apply_drag``'s reason: a write-through would put
            # the numbers back and leave ``local()``'s memo holding the matrix
            # from the middle of the cancelled drag.
            node.translation = np.array(t0, dtype="f8")
            node.rotation = np.array(r0, dtype="f8")
            node.scale = np.array(s0, dtype="f8")
        self._grab = None
        doc.touch()
        self._render_dirty = True

    # -- capture and teardown ----------------------------------------------

    def screenshot(self) -> Any:
        return capture.image(self.viewport)

    def release(self) -> None:
        self.clear()
        self.translate_gizmo.release()
        self.rotate_gizmo.release()
        self.scale_gizmo.release()
        # Forgotten before the GL object goes, the rule ``_view_frame._resize``
        # states: the imgui backend maps GL names to moderngl objects, and a
        # released texture left registered is a dead object under a name the
        # driver is free to reissue -- which is how an unrelated image starts
        # rendering as this one.
        self._forget(self.viewport.texture)
        self.viewport.release()
        self.renderer.release()


def _terrain_node() -> Any:
    """``mason.nodes.TerrainNode``, imported at the call rather than at the top.

    A narrow rule rather than a lazy-import one: this module reaches into the
    engine for exactly three things -- the resolver, the picker and the proxy
    pool -- and the terrain class is wanted on one line inside one method.
    """
    from .mason.nodes import TerrainNode

    return TerrainNode


def _frustum_planes(view_proj: np.ndarray) -> np.ndarray:
    """The six clip planes of a view-projection matrix, as rows of ``(6, 4)``.

    Gribb-Hartmann: each plane is a sum or difference of two rows of the
    matrix, which needs no inverse and no unprojection. Normalized so a plane
    test is a true signed distance rather than a scaled one -- the box test
    below compares against zero, so the scale would not change the answer, but
    an unnormalized plane makes every intermediate number meaningless to
    anyone reading them in a debugger.
    """
    m = np.asarray(view_proj, dtype="f8")
    planes = np.array(
        [
            m[3] + m[0],
            m[3] - m[0],
            m[3] + m[1],
            m[3] - m[1],
            m[3] + m[2],
            m[3] - m[2],
        ],
        dtype="f8",
    )
    norms = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return planes / norms


def _box_visible(planes: np.ndarray, box: Any, world: np.ndarray) -> bool:
    """Whether a ref's local AABB, placed by ``world``, is inside the frustum.

    The eight corners are transformed once and tested against six planes, which
    is the cheap conservative test: a box wholly outside one plane is outside
    the frustum, and anything else is drawn. It can keep a box that is outside
    the frustum without being outside any single plane -- the classic false
    positive -- and that is the right trade here, because a false positive
    costs one draw call the GPU discards and a false *negative* costs a prop
    that is missing from the picture.
    """
    lo, hi = box
    xs = (float(lo[0]), float(hi[0]))
    ys = (float(lo[1]), float(hi[1]))
    zs = (float(lo[2]), float(hi[2]))
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype="f8")
    placed = np.asarray(world, dtype="f8")
    world_corners = corners @ placed[:3, :3].T + placed[:3, 3]
    return all(
        not np.all(world_corners @ plane[:3] + plane[3] < 0.0) for plane in planes
    )
