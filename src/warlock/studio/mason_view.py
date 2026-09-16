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

**Two things a scene holds that the ref-keyed cache has nowhere to put.** A
light and a camera have no geometry at all, and draw as ``DrawItem`` overlays
through :mod:`.mason_marks` -- without which placing a light put a row in the
outliner and nothing whatever in the viewport. The ground has geometry but no
``Ref``, because it is generated from ``doc.terrain`` rather than resolved from
a :class:`~.mason.refs.GeometrySource`, so :meth:`MasonView.sync_terrain` is its
own one-entry upload path keyed on the identity of the ``heights`` array. Both
gaps were invisible rather than loud: the resolver, the picker and the exporters
all handled these nodes from Stage C, so the ray found ground the eye could not
see.

**A sculpt stroke is a session the view owns.** ``document.begin_sculpt`` /
``sculpt`` / ``end_sculpt`` is one undo step, and the press/motion/release that
drive it are here. It is also the one gesture that changes the picture with no
change the *document* can see -- ``sculpt`` pushes nothing and touches no
revision -- which is why the redraw key carries the height array's identity.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._view_frame import Composite, FrameOps
from .mason import ops as mops
from .mason import pick as mpick
from .mason import scene as msc
from .mason import terrain as mterrain
from .mason.refs import ref_key
from .mason_marks import SceneMarks
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

        # The wire symbols for the node kinds that have no geometry. Without
        # them a placed light or camera is a row in the outliner and nothing at
        # all in the viewport -- see :mod:`.mason_marks`.
        self.marks = SceneMarks(ctx, self.renderer.programs)

        self._cache: dict[tuple[Any, ...], _Entry] = {}
        # The ground's own upload, outside the ref-keyed cache because terrain
        # geometry does not come from a ``GeometrySource`` and so has no ref to
        # be keyed on: ``(heights_array, override, model, GpuModel)``, valid
        # only while the pinned array *is* ``doc.terrain.heights``. Identity
        # and not ``id()``, ``terrain.terrain_mesh``'s own memo rule and for
        # its reason: an id is an address CPython may hand to a different
        # array once the old one is collected, so an id-keyed check can
        # validate against the wrong object. Every brush rebinds the array
        # rather than writing into it, which is what makes the identity check
        # a sound invalidation signal.
        #
        # The 2026-09-15 audit's mason-04: this was annotated as a 3-tuple
        # while :meth:`sync_terrain` has always stored all four of
        # ``(terrain.heights, override, model, gpu)`` -- the annotation had
        # drifted from what is actually assigned a few lines down.
        self._terrain: tuple[Any, Any, Any, Any] | None = None
        # What the terrain GPU state was built from, so a rebuild is counted
        # the way a ref's is.
        self.terrain_rebuilds = 0
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
        # Where a click asked for the armed placement to land, for the pane to
        # drain -- ``menu_request``'s own idiom, and ``MasonState.frame_pending``'s
        # reason: what a placement *means* belongs to ``mason_mode``, which this
        # module deliberately does not import.
        self.place_request: Any = None
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
        # The height array the redraw key's ``id`` named, pinned for the same
        # reason -- see that key's own comment.
        self._last_heights: Any = None

        # What ``resolve`` last answered, and the key it was answered for.
        # Memoized because a frame asks for it up to four times (the sync, the
        # composite, the gizmo centre, the HUD) and a resolve at the warn
        # threshold was measured at a third of a frame on its own.
        self._placed: list[Any] = []
        self._placed_key: Any = None
        # How many the frustum test threw away last frame -- read by the HUD,
        # which is the only honest way to see culling working.
        self.culled = 0

        # The live sculpt stroke's own bookkeeping. A stroke is a *session* and
        # not a stream of edits -- ``document.begin_sculpt``/``sculpt``/
        # ``end_sculpt`` -- so what is held here is only what the session needs
        # between frames: which document it was opened against (so a tab switch
        # mid-drag closes it rather than sculpting the wrong ground), and the
        # flatten level and noise seed the *whole* stroke shares. Re-rolling the
        # seed per frame would lay forty different noise fields down one drag.
        self._sculpt_doc: Any = None
        self._sculpt_level = 0.0
        self._sculpt_seed = 0

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
        """Whether a gesture that is *changing the document* is live -- not an
        orbit and not a pan.

        A sculpt stroke counts, and has to: ``mason_mode._DRAG_BLOCKED_CTRL``
        reads this to refuse Ctrl+Z, Ctrl+S and a tab switch mid-drag, and an
        undo landing between two dabs of one stroke would leave the session
        holding a "before" snapshot of a height field the history has already
        replaced.
        """
        return self._grab in ("gizmo", "sculpt")

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
        # The ground, in the same phase and by the same rule: uploaded if a
        # visible terrain node places it, and released the moment nothing does
        # -- which is what keeps "a ref nothing places any more takes its
        # buffers with it" true of the one drawable that has no ref.
        ground = next(
            (p for p in self.resolved(doc) if p.visible and isinstance(p.node, _terrain_node())),
            None,
        )
        if ground is None:
            self._release_terrain()
        else:
            self.sync_terrain(doc, ground.material)
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

    def sync_terrain(self, doc: Any, override: Any = None) -> Any:
        """The ground's ``GpuModel``, uploaded or re-uploaded as needed. ->
        ``(model, gpu)`` or ``None``.

        **Its own path, outside the ref-keyed cache**, because terrain geometry
        does not come from a :class:`~.mason.refs.GeometrySource`: there is no
        ``Ref`` for the cache to key it on and no asset for it to be shared
        between, since the document has exactly one ground (``mason/terrain.py``
        states why two would be two ground planes nobody asked for). What stands
        in for the key is the identity of ``heights``, which is sound precisely
        because every brush **rebinds** that array -- the same property
        ``terrain_mesh``'s own memo rests on, so the mesh build and the upload
        invalidate together on the same signal rather than on two.

        A sculpt drag therefore pays one mesh rebuild *and* one upload per frame
        it is open, which is measured and is why ``MAX_TERRAIN_SIDE`` is 256:
        at that side a rebuild alone is 5.65 ms, a third of a frame. See
        ``docs/measurements/2026-09-11-mason-scene-ceilings.md``.

        ``override`` is the resolver's nearest-ancestor material, part of what
        the pinned state is validated against for ``_entry_key``'s reason: a
        ground drawn under an override that has since changed would otherwise
        keep the old surface with nothing in the data to say why.
        """
        terrain = doc.terrain
        if terrain is None:
            self._release_terrain()
            return None
        if (
            self._terrain is not None
            and self._terrain[0] is terrain.heights
            and self._terrain[1] is override
        ):
            return self._terrain[2], self._terrain[3]
        self._release_terrain()
        from dataclasses import replace as _replace

        from .viewer import gltf

        primitive = mterrain.terrain_mesh(terrain)
        if override is not None:
            # A copy, never a write through what ``terrain_mesh`` handed back:
            # that primitive is the memo's own and is handed to the *next* caller
            # too, so an override written into it would outlive this draw.
            primitive = _replace(primitive, material=override)
        node = gltf.Node(name="terrain", mesh=0)
        model = gltf.Model([node], [0], [[primitive]], [])
        gpu = scenelib.GpuModel(self.ctx, model)
        self._terrain = (terrain.heights, override, model, gpu)
        self.terrain_rebuilds += 1
        return model, gpu

    def _release_terrain(self) -> None:
        if self._terrain is not None:
            self._terrain[3].release()
            self._terrain = None

    def clear(self) -> None:
        for entry in self._cache.values():
            entry.gpu.release()
        self._cache.clear()
        self._release_terrain()
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
            # **The ground, by the identity of its height array.** A sculpt
            # stroke deliberately does not touch ``doc.rev``: ``document
            # .sculpt`` pushes nothing and calls nothing, because the whole
            # drag is one undo step that only ``end_sculpt`` commits. So the
            # document *does not see* a stroke in progress, and without this
            # entry every frame of a drag would be skipped as "nothing moved"
            # and the ground would jump to its new shape on release -- the same
            # shape of miss ``source.rev`` above closes for an arriving asset.
            #
            # An ``id`` is sound here for the reason ``_last_doc`` is: the array
            # it names is pinned on ``self._last_heights`` below, so the address
            # cannot be reissued to a different array while this key still
            # claims it.
            id(None if doc.terrain is None else doc.terrain.heights),
        )
        if self._frame_unchanged(key):
            return self.viewport.texture
        self._last_render_key = key
        self._last_doc = doc
        self._last_source = source
        self._last_heights = None if doc.terrain is None else doc.terrain.heights
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
            overlays=self._overlays(doc, source, height),
        )
        return self.viewport.texture

    def _overlays(self, doc: Any, source: Any, height: int) -> list[Any]:
        """The marker symbols, then the gizmo over them.

        Order matters and is the renderer's own: it draws depth-tested items
        first and depth-off items over them, so the gizmo (depth off) lands on
        top of a light's symbol (depth tested) whatever order this list is in.
        The order here is the one a reader expects anyway -- the scene's symbols,
        then the handle being dragged over them.
        """
        marks = self.marks.draws(self.resolved(doc), doc.selection)
        return marks + self._gizmo_draws(doc, source, height)

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
        terrain_node = _terrain_node()
        for item in placed:
            if not item.visible:
                continue
            if item.ref is None:
                # The ground is the one drawable with no ref -- see
                # ``sync_terrain``. A light or a camera is the other reason a
                # placement has none, and those draw as overlays instead.
                if isinstance(item.node, terrain_node) and self._terrain is not None:
                    proxy = self._pool.node(item.world)
                    opaque += [(proxy, primitive) for _n, primitive in self._terrain[3].draws]
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
                try:
                    found = msc.resolved_for(doc, uid)
                except ValueError:
                    # mason-03, the 2026-09-13 audit: a document past
                    # ``MAX_PLACED`` makes ``resolved_for`` raise the same
                    # refusal ``resolve`` always has; the frame thread must
                    # not crash over a pivot lookup, so this uid simply
                    # contributes no point, same as "not found".
                    found = None
                if found is not None:
                    points.append(found.world[:3, 3])
        if not points:
            return None
        if pivot == "active":
            # The 2026-09-16 audit's mason-engine-... : this used to be
            # ``points[0]``, the first node in ``resolve()``'s walk order
            # that happened to be selected -- for a fixed selection, the
            # same node however the user built it up, not "the last node
            # clicked" the pivot's own tooltip promises. ``doc.select``
            # records that uid in ``selection_active``; fall back to the
            # median (below) when it is unset or points at a uid no longer
            # in the current selection -- an ambiguous Shift+range or
            # Ctrl+A leaves it stale rather than wrong.
            active_uid = getattr(doc, "selection_active", None)
            if active_uid in selection:
                for item in self.resolved(doc):
                    if item.owner == active_uid:
                        return np.asarray(item.world[:3, 3], dtype="f8")
                try:
                    found = msc.resolved_for(doc, active_uid)
                except ValueError:
                    found = None
                if found is not None:
                    return np.asarray(found.world[:3, 3], dtype="f8")
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

    def _drop_point(self, doc: Any, source: Any, local: tuple[float, float]) -> np.ndarray:
        """Where a click in the viewport means, in world metres.

        What the ray hits, if it hits anything -- so a prop drops onto the
        terrain or onto the roof of another prop, which is the answer a user
        pointing at a surface means. Otherwise the ground plane, and if the ray
        runs parallel to that (a camera looking dead along the horizon), the
        camera's own target, because a click has to mean *somewhere* and the
        thing being looked at is the only defensible somewhere left.

        Snapped by the app's grid when snapping is on, through
        ``ops.snap_translation`` rather than arithmetic here -- the same rule the
        Tools pane follows for align and array: this file decides *whether*, the
        engine decides *where*.
        """
        hit = self.pick(doc, source, local)
        if hit is not None:
            point = np.asarray(hit.point, dtype="f8")
        else:
            origin, direction = self._ray(local)
            if abs(float(direction[1])) < 1e-9:
                point = np.asarray(self.camera.target, dtype="f8")
            else:
                t = -float(origin[1]) / float(direction[1])
                point = (
                    origin + direction * t
                    if t > 0.0
                    else np.asarray(self.camera.target, dtype="f8")
                )
        if getattr(self.state, "snap", False):
            point = mops.snap_translation(point, float(getattr(self.state, "snap_translate", 0.0)))
        return point

    # -- sculpting ---------------------------------------------------------

    def _begin_sculpt(self, doc: Any, local: tuple[float, float]) -> bool:
        """Open a sculpt session from a press on the ground. -> whether it did.

        The session, not the stroke's first dab, is what this opens: a pane that
        pushed a ``TerrainEdit`` per mouse-move would turn one drag into fifty
        presses of Ctrl+Z, which is why ``document.begin_sculpt``/``sculpt``/
        ``end_sculpt`` exists at all. The level Flatten pulls toward and the seed
        Noise uses are read **once, here**, so one drag lays down one field and
        levels to one height however many frames it lasts.
        """
        cell = self._brush_cell(doc, local)
        if cell is None:
            return False
        state = self.state
        self._sculpt_level = float(getattr(state, "brush_level", 0.0))
        if getattr(state, "brush_level_from_pick", False):
            self._sculpt_level = float(self._brush_height(doc, cell))
        self._sculpt_seed = int(getattr(state, "brush_seed", 1))
        doc.begin_sculpt()
        self._sculpt_doc = doc
        self._grab = "sculpt"
        # The first dab, so a click without a drag still does something: every
        # brush is a function of where the cursor is, and a press that waited for
        # motion would read as a dead click.
        self._sculpt_at(doc, cell, self._SCULPT_STEP)
        return True

    def _brush_cell(self, doc: Any, local: tuple[float, float]) -> tuple[float, float] | None:
        """The ground cell under the cursor as fractional ``(col, row)``, or
        ``None`` if the ray misses the ground.

        The brushes work in height-field index space -- they take ``cx``/``cz``
        in cells, because that is the space a falloff radius is honest in -- so
        the conversion from a world ray to a cell is the viewport's, here, and
        not duplicated in a pane.
        """
        terrain = doc.terrain
        if terrain is None:
            return None
        found = next(
            (p for p in self.resolved(doc) if p.visible and isinstance(p.node, _terrain_node())),
            None,
        )
        if found is None:
            return None
        origin, direction = self._ray(local)
        march = mpick.ray_terrain(terrain, found.world, origin, direction)
        if march is None:
            return None
        _t, point = march
        world = np.asarray(found.world, dtype="f8")
        determinant = float(np.linalg.det(world[:3, :3]))
        if abs(determinant) < _SINGULAR_DET_EPS:
            return None
        local_point = np.linalg.inv(world) @ np.array([*point[:3], 1.0], dtype="f8")
        side = terrain.side
        col = (float(local_point[0]) / terrain.size_x + 0.5) * side
        row = (float(local_point[2]) / terrain.size_z + 0.5) * side
        return col, row

    def _brush_height(self, doc: Any, cell: tuple[float, float]) -> float:
        """The ground's height at a cell, clamped to the field -- what Flatten
        levels to when it is told to take its level from the first click."""
        heights = doc.terrain.heights
        col = int(min(max(round(cell[0]), 0), heights.shape[1] - 1))
        row = int(min(max(round(cell[1]), 0), heights.shape[0] - 1))
        return float(heights[row, col])

    #: What one mouse-motion event is worth, as a fraction of a second. The
    #: brush strengths on ``MasonState`` are per second, so something has to
    #: convert; a *motion event* is the unit rather than a frame, because that is
    #: what a stroke is actually delivered in and it is the only one of the two
    #: that does not make the brush stronger on a machine that renders faster.
    #: One sixtieth, so a drag that samples at a typical pointer rate deposits
    #: about the stated amount per second of dragging.
    _SCULPT_STEP = 1.0 / 60.0

    def _sculpt_at(self, doc: Any, cell: tuple[float, float], step: float) -> None:
        """Apply one dab of the current brush. Pushes nothing -- the session
        does, on release.

        ``step`` scales raise/lower and the two strength brushes; see
        :data:`_SCULPT_STEP` for what it is and why it is not a frame time.
        """
        state = self.state
        terrain = doc.terrain
        if terrain is None:
            return
        radius = float(getattr(state, "brush_radius", 6.0))
        brush = str(getattr(state, "brush", "raise"))
        col, row = cell
        result = None
        if brush in ("raise", "lower"):
            amount = float(getattr(state, "brush_amount", 1.0)) * step
            result = mterrain.raise_lower(
                terrain.heights, col, row, radius, -amount if brush == "lower" else amount
            )
        elif brush == "smooth":
            strength = float(getattr(state, "brush_strength", 0.5)) * step
            result = mterrain.smooth(terrain.heights, col, row, radius, strength)
        elif brush == "flatten":
            result = mterrain.flatten(
                terrain.heights,
                col,
                row,
                radius,
                self._sculpt_level,
                float(getattr(state, "brush_strength", 0.5)) * step,
            )
        elif brush == "noise":
            result = mterrain.noise(
                terrain.heights,
                col,
                row,
                radius,
                float(getattr(state, "brush_amount", 1.0)) * step,
                seed=self._sculpt_seed,
            )
        if result is not None:
            doc.sculpt(*result)
            self._render_dirty = True

    def _end_sculpt(self) -> None:
        """Close the session, if one is open. Safe to call from anywhere.

        ``document.end_sculpt`` is idempotent on purpose -- a release can be
        missed to focus loss, Esc or a save beginning mid-drag -- and this
        wrapper is what lets every one of those paths say "close whatever is
        open" without first working out whether anything is.
        """
        doc, self._sculpt_doc = self._sculpt_doc, None
        if doc is not None:
            doc.end_sculpt()
            doc.touch()
            self._render_dirty = True

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

        # The brush owns the left button for the whole stroke, which is what makes
        # Sculpt a tool rather than an armed placement: no gizmo, no pick, no
        # selection change, and an orbit has to be Alt+drag (which the branch
        # above already took).
        sculpting = str(getattr(self.state, "tool", "")) == "sculpt" and doc.terrain is not None
        if sculpting and self._begin_sculpt(doc, local):
            return True

        if getattr(self.state, "place_kind", "") or getattr(self.state, "place_prefab", ""):
            # A *request*, not a placement. The view owns the pointer and the
            # camera; what a placement means -- which document, which node kind,
            # which undo step -- is ``mason_mode``'s, and this module does not
            # import the controller (``clay_view`` does not either). The pane
            # drains this the way it already drains ``menu_request``.
            self.place_request = self._drop_point(doc, source, local)
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
            # ``owner`` -- the node actually under the pointer -- is the
            # "last node clicked" the Active pivot's tooltip promises, even
            # while it is only one of several uids landing in ``selection``
            # here; passed explicitly because ``select`` cannot infer it
            # from an unordered multi-uid set (the 2026-09-16 audit).
            doc.select(selection, active=owner)
        else:
            doc.select([owner] if owner is not None else [], active=owner)
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
        elif self._grab == "sculpt":
            if self._sculpt_doc is not doc:
                # The tab changed under a live stroke. Closed against the
                # document it was opened on rather than carried over: a session
                # holds a snapshot of *that* ground, and sculpting a second
                # document through it would commit one scene's undo step onto
                # another's history.
                self._end_sculpt()
                return True
            cell = self._brush_cell(doc, local)
            if cell is not None:
                self._sculpt_at(doc, cell, self._SCULPT_STEP)
        return True

    def _release(self, doc: Any, button: int = 1) -> bool:
        was, self._grab = self._grab, None
        self._alt_at = None
        if was == "gizmo":
            self._end_gizmo_drag(doc)
        elif was == "sculpt":
            self._end_sculpt()
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
            self._apply_drag(
                doc, source, delta=np.asarray(point, dtype="f8") - self._drag_origin
            )
        elif tool == "rotate":
            step = self.rotate_gizmo.update(origin, direction)
            if step is None:
                return
            # ``RotateGizmo.update`` hands back the *increment* since the last
            # call, so it has to be accumulated into a total: the drag is
            # applied against the transforms recorded at the press, and an
            # increment applied to those would be the last mouse-move alone.
            self._drag_quat = m3.quat_mul(np.asarray(step, dtype="f8"), self._drag_quat)
            self._apply_drag(doc, source, quat=self._drag_quat)
        elif tool == "scale":
            factor = self.scale_gizmo.update(origin, direction)
            if factor is None:
                return
            self._apply_drag(doc, source, scale=np.asarray(factor, dtype="f8"))

    def _apply_drag(
        self,
        doc: Any,
        source: Any = None,
        *,
        delta: Any = None,
        quat: Any = None,
        scale: Any = None,
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

        **Snap is read here, not just at the initial placement click.** The
        2026-09-12 audit (docs-01) found that Chapter 17 promises "turn on
        Snap ... drag the box: it lands on whole metres," but this method
        never once consulted ``state.snap`` -- only ``_drop_point`` (the first
        click) did, so Clay's identical toggle worked and Mason's did not. The
        grid is a *world* quantity (a floor is flat in world space whatever
        group a prop sits under), so the local ``translation``/rotation this
        method already carries in the parent's frame is turned back into a
        world point, snapped there, and carried back through ``inverse`` --
        the same round trip the pivot maths below already does for a rotate
        or scale about a world-space pivot.
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
            snap_on = bool(getattr(self.state, "snap", False))
            if delta is not None:
                translation = t0 + inverse @ np.asarray(delta, dtype="f8")
                if snap_on:
                    step = float(getattr(self.state, "snap_translate", 0.0))
                    world_point = parent_world[:3, :3] @ translation + parent_world[:3, 3]
                    world_point = mops.snap_translation(world_point, step)
                    translation = inverse @ (world_point - parent_world[:3, 3])
            elif quat is not None:
                turn = np.asarray(quat, dtype="f8")
                if snap_on:
                    # The *total* turn since the press is what snaps, the same
                    # register Clay's ``_view_drag`` snaps its own accumulated
                    # delta in rather than the absolute orientation: an object
                    # tilted off the cardinal axes at rest is not yanked
                    # straight the instant the drag starts.
                    turn = mops.snap_rotation(
                        turn, float(getattr(self.state, "snap_rotate", 0.0))
                    )
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
            if delta is not None and source is not None:
                self._drop_dragged_node_to_ground(doc, source, uid, inverse)
        doc.touch()

    def _drop_dragged_node_to_ground(
        self, doc: Any, source: Any, uid: int, inverse: Any
    ) -> None:
        """After a translate drag writes ``uid``'s new position, rest it on the
        terrain if ``state.snap_ground`` is on.

        The 2026-09-12 audit (docs-02) found ``state.snap_ground`` written by
        its own toggle and read nowhere else: Chapter 17 promises a dragged
        prop "lands on the ground rather than floating," and the field that
        promise names was wired to nothing during the one gesture the chapter
        tells the reader to use it for. This is the same ``mops.drop_to_ground``
        the one-shot "Drop selection to ground" button already calls
        (``panes/mason_tools.py``), applied per node, per frame, against the
        position the drag has *just* written -- so the box it measures is
        the box the drag actually produced, not the one before it moved.
        A node with no resolvable geometry (a light, a camera, an unexpanded
        prefab instance) contributes no box, the same as it contributes
        nothing to ``world_bounds`` everywhere else, and is left alone.
        """
        if not bool(getattr(self.state, "snap_ground", False)):
            return
        box = self.world_bounds(doc, source, uids=[uid])
        if box is None:
            return
        ground_delta = mops.drop_to_ground({uid: box}, terrain=doc.terrain).get(uid)
        if ground_delta is None:
            return
        node = doc.node(uid)
        if node is None:
            return
        node.translation = np.array(node.translation, dtype="f8") + inverse @ np.asarray(
            ground_delta, dtype="f8"
        )

    def _parent_basis(self, doc: Any, uid: int) -> Any:
        """``(inverse_of_the_parent_basis, parent_world)``, or ``None``.

        Identity at the root, which is the common case and costs nothing.
        """
        parent_uid = doc.parent_uid_of(uid)
        if parent_uid is None:
            return np.eye(3), np.eye(4)
        try:
            found = msc.resolved_for(doc, parent_uid)
        except ValueError:
            # mason-03, the 2026-09-13 audit: same refusal as the pivot
            # lookup above -- a document past ``MAX_PLACED`` must not crash a
            # drag's basis lookup on the frame thread.
            found = None
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
        # A sculpt stroke is *committed* rather than discarded by a cancel, and
        # the asymmetry is deliberate: a gizmo drag's "before" is three arrays
        # this method still holds, where a stroke's is a whole height field the
        # session snapshotted -- and ``end_sculpt`` is the only thing that turns
        # what is already on the ground into something Ctrl+Z can reach. Dropping
        # the session here would leave the sculpted ground in the document with
        # no undo step for it at all, which is strictly worse than one extra step
        # the user can undo.
        self._end_sculpt()
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
        self.marks.release()
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
