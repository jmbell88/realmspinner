"""The Clay viewport: GL, camera, gizmos and picking over the existing stack.

Modelled on :mod:`~warlock.studio.viewer_embed` and reusing its parts wholesale
-- ``camera``, ``render``, ``glctx``, ``grid``, ``gizmo``, ``programs`` and
``capture`` are all shared unchanged. What is different is the *subject*: the
3D pane shows one loaded GLB, and this shows a live document of many objects,
each of which can change independently while the others do not.

That difference is the whole design of the GPU cache. An entry is keyed on
``(uid, id(obj.mesh), materials)``, and it is sound precisely because ``Mesh``
is a frozen dataclass and every op on it is ``Mesh -> Mesh``: a changed mesh is
a *different object*, so identity misses exactly when it should, and an
unchanged one is the same object however many times the document around it was
edited. An in-place mutation anywhere in ``build.mesh`` would break this and
would show up as the viewport drawing the old shape forever with nothing in the
data to say why -- which is stated in that module's own docstring as the reason
it is immutable.

Two rules travel across from the existing viewer unchanged, and both are the
sort that fail invisibly:

* **imgui draws through moderngl.** ``studio/imgui_backend.py`` reimplements
  imgui's GL3 backend on moderngl because moderngl caches GL state, so a raw
  ``glBindTexture`` behind its back leaves the viewport rendering with whatever
  the panels last bound. Nothing here touches raw GL, and the resolved texture
  reaches imgui through ``widgets.texture_ref`` -- which *registers* it, since
  an id the renderer does not know maps to no moderngl object.
* **The viewport background is deliberately not tone-mapped.** three sets the
  clear colour straight back to sRGB, so it is the literal hex, and the
  renderer owns that; nothing here re-grades it.

**The mouse map is Wings3D's, and RMB pan is gone.** Right-drag used to pan,
and the context menu needs the right button more: a menu that appears under the
cursor with the ops that apply to what is selected is the whole point of an
element-mode editor, and a modifier-plus-right-drag would be a worse pan than
the middle button already is. So button 2 pans, button 3 opens the menu on a
*release within four pixels of the press* -- a right-drag does nothing at all,
which means a user who grabs the wrong button mid-orbit loses nothing.

**Modifiers are read at the press, from ``pygame.key.get_mods()``.** Not from
the event: a ``MOUSEBUTTONDOWN`` carries no modifier state, and tracking KEYDOWN
and KEYUP to shadow it is a second copy of something the platform already knows
and gets wrong the first time the window loses focus with Shift held.

Registration has a matching half that ``Viewport`` makes easy to miss:
``resize`` releases and recreates its texture, so a resize frees a GL name the
imgui backend may still be holding. :meth:`ClayView.draw` forgets the outgoing
texture before that happens.

**``ClayView``'s concerns live in the ``_view_*.py`` mixins it inherits** -- the
GPU cache, the element overlay, the bounds and centres, picking, and the whole
of input and dragging -- while what stays here is the class itself and the
frame: the fields, the draw, the world-matrix memo, the composite the renderer
consumes, the imgui-texture bookkeeping, and the gizmo the app state chooses.
The split is code motion only; ``ClayView`` is one class with one surface, and
every rule stated above (including :meth:`ClayView._narrow` being the single
narrowing site, in ``_view_drag``) still holds of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._view_bounds import BoundsOps
from ._view_cache import CacheOps

# Re-exported rather than left behind: these are the viewport's names for
# them, and ``_Entry`` in particular is what ``__init__`` annotates its
# cache with. ``scenelib`` comes back through here for the same reason --
# ``clay_view.scenelib`` is the module object the GPU upload is patched on.
from ._view_cache import _Entry as _Entry
from ._view_cache import _materials_key as _materials_key
from ._view_cache import _object_key as _object_key
from ._view_cache import scenelib as scenelib
from ._view_drag import DragOps

# ``__init__`` annotates the live-drag map with it.
from ._view_drag import _ElementDrag as _ElementDrag
from ._view_frame import Composite, FrameOps
from ._view_overlay import OverlayOps

# ``__init__`` annotates the overlay cache with it.
from ._view_overlay import _SelOverlay as _SelOverlay

# ``Hit`` is constructed in the mixin, so it is defined there -- but the name
# a caller reaches for is the viewport's, so it is re-exported here.
from ._view_pick import Hit as Hit
from ._view_pick import PickOps
from .viewer import capture, glctx
from .viewer import math3d as m3
from .viewer.camera import Camera, screen_ray
from .viewer.gizmo import RotateGizmo, ScaleGizmo, TranslateGizmo
from .viewer.render import Renderer

log = logging.getLogger(__name__)


# Which gizmo each tool drives. Held as data so the dispatch is one lookup
# rather than a chain that a fifth tool would have to be threaded through.
GIZMO_FOR_TOOL = {"move": "translate", "rotate": "rotate", "scale": "scale"}


#: How opaque the surface is in X-ray. A third: enough to read the silhouette
#: and the shading, and little enough that an edge on the far side is pickable
#: through it -- which is the whole point of the mode.
XRAY_ALPHA = 0.33


@dataclass(frozen=True)
class GizmoDragReadout:
    """What a live G/R/S drag amounts to right now, for the HUD's one line.

    ``kind`` is the verb ("move"/"rotate"/"scale"), ``axis`` the lock
    ("x"/"y"/"z"/""), ``space`` the frame the lock is read in, and ``amount``
    the typed buffer while one is being entered, else the drag's own live
    readout. The one shape :meth:`ClayView.gizmo_drag` hands the pane, so
    ``panes/clay_hud.py`` never has to reach past it at ``_key_kind`` or
    ``drag_input`` directly -- see that property's docstring for why the old
    read of ``_key_kind`` alone went blank for a handle-grabbed drag.
    """

    kind: str
    axis: str
    space: str
    amount: str


class ClayView(CacheOps, BoundsOps, PickOps, OverlayOps, DragOps, FrameOps):
    """The Clay viewport, from the UI's point of view."""

    def __init__(self, ctx: Any, app_ctx: Any = None) -> None:
        """``ctx`` is the moderngl context, as ``Viewer``'s is.

        ``app_ctx`` is separate and optional because the two are genuinely
        different things and conflating them is the bug this signature exists
        to prevent: everything that draws needs the GL context, and the only
        thing that needs the app is reading which transform tool is selected --
        which is an *app* setting shared across documents, so the view reads it
        rather than holding a copy that could drift.
        """
        self.ctx = ctx
        self.app_ctx = app_ctx
        self.renderer = Renderer(ctx)
        self.viewport = glctx.Viewport(ctx, (16, 16))
        self.camera = Camera()
        self.wireframe = False
        # What the surface is drawn as, and what is drawn over it. Set by the
        # pane each frame beside ``wireframe`` and ``show_grid``, which is how
        # every other view setting reaches here.
        #
        # ``flat`` is Blender's *Solid*: the albedo with no lighting, which is
        # the shading a modeller works in because it shows silhouette and
        # topology without a specular highlight sitting on the vertex being
        # dragged. ``xray`` is the see-through pass -- see ``Renderer.draw``.
        self.flat = False
        self.wire_overlay = False
        self.xray = False
        # The grid toggle in the tools pane had no reader at all: the pane wrote
        # ``state.grid`` and the renderer was never told. One field, set by the
        # pane layer beside ``wireframe``, which already worked that way.
        self.show_grid = True
        self.radius = 1.0

        self.translate_gizmo = TranslateGizmo(ctx, self.renderer.programs)
        self.rotate_gizmo = RotateGizmo(ctx, self.renderer.programs)
        self.scale_gizmo = ScaleGizmo(ctx, self.renderer.programs)

        self._cache: dict[int, _Entry] = {}
        # Counted rather than inferred: "only what changed was rebuilt" is a
        # property worth asserting, and there is no other way to see it.
        self.rebuilds = 0

        self._rect = (0.0, 0.0, 1.0, 1.0)
        self._grab: str | None = None  # orbit | pan | gizmo | marquee
        self._last_mouse = (0.0, 0.0)
        self._drag_uids: list[int] = []
        self._drag_start: dict[int, tuple[Any, Any, Any]] = {}
        # The drag's *total* rotation, and the gizmo's origin at the press.
        # ``RotateGizmo.update`` hands back the increment since the last call
        # and ``TranslateGizmo.update`` the gizmo's new world position, so
        # neither is usable against a transform recorded at the press without
        # these two: the first has to be accumulated into a total, the second
        # turned into a displacement.
        self._drag_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._drag_origin = np.zeros(3)
        # A live keyboard drag: which transform it is, and where on the view
        # plane it started. Empty and None between drags -- see
        # ``_view_drag.begin_keyboard_drag``.
        self._key_kind = ""
        self._key_anchor: Any = None
        # Where an Alt press went down, and whether Ctrl was held with it.
        # Alt+drag orbits and Alt+click selects a loop; the two share the button
        # and are told apart on the release -- see ``_view_drag._alt_click``.
        self._alt_at: Any = None
        self._alt_ctrl = False

        # What the cursor is over in an element mode, as ``(uid, index)`` read
        # through the document's own mode. Updated only on motion with no grab
        # -- a hover that recomputed every frame would reproject every vertex
        # of every object while the scene sits perfectly still.
        self.hover_element: tuple[int, int] | None = None
        # ``(key, screen, mesh)`` -- the mesh pinned so the id in the key stays
        # sound; see ``screen_of``.
        self._screens: dict[int, tuple[Any, Any, Any]] = {}

        # Where the right button went down, and where the menu should open.
        # ``menu_request`` is read and cleared by the pane layer, which is the
        # only layer allowed to know imgui exists.
        self._rmb_at: tuple[float, float] | None = None
        self.menu_request: tuple[float, float] | None = None
        # The live marquee rectangle in viewport pixels, drawn by the pane.
        self.marquee: tuple[float, float, float, float] | None = None
        self._marquee_from: tuple[float, float] | None = None
        self._marquee_add = "replace"
        self._element_drags: dict[int, _ElementDrag] = {}
        self._overlays: dict[int, _SelOverlay] = {}
        self._element_centre = np.zeros(3)
        # Redraw bookkeeping (B13), the shape Viewer.render uses (B12).
        self._render_dirty = True
        self._last_render_key: Any = None
        # The document the key above was drawn against, pinned for the reason
        # every other identity-keyed cache in this class already states
        # (``_screens``, ``_world_cache``): an id is only sound while its
        # object is alive. The 2026-09-07 audit's clay-09 found this one bare
        # -- ``id(doc)`` with nothing holding the document itself -- so a
        # closed tab's ``ClayDoc`` could be collected and a *new* document
        # minted at the same address, with ``rev`` starting at 0 the way it
        # does for every fresh document, and this cache handed back the closed
        # tab's stale texture.
        self._last_doc: Any = None
        # Per-object world matrices, keyed on the identity of the three
        # transform arrays -- sound because every transform write *rebinds*
        # them (the documented gizmo rule) rather than mutating in place (B26).
        # The arrays themselves ride in the entry (the ``_view_cache._Entry``
        # pin), because an id is only sound while its object is alive: a freed
        # array's address coming back on a different transform would otherwise
        # match a stale matrix.
        self._world_cache: dict[
            int, tuple[tuple[int, int, int], Any, tuple[Any, Any, Any]]
        ] = {}
        # element_centre / selection_centre / world_bounds memos (B25/B27),
        # each ``(key, answer, pins)`` -- the pins hold what the key's ids name.
        self._centre_memo: tuple[Any, Any, Any] | None = None
        self._bounds_memo: dict[bool, tuple[Any, tuple[Any, Any], Any]] = {}
        # One shared empty element-selection, so an unselected object stops
        # synthesising a fresh empty() per frame (B26).
        self._empty_sel: Any = None

        # What the keyboard has said about the drag under way, and what the HUD
        # should draw for it. Both are per drag: created at the press and
        # dropped at the release, because a lock that outlived one would
        # silently constrain the next.
        from .clay import drag as bdrag

        self.drag_input = bdrag.DragInput()
        self.drag_hud: str = ""
        # The world position a move has snapped onto, or None. Held rather than
        # recomputed by the consumer because it is found from the *cursor*, and
        # the transform is applied a layer down where the cursor is gone.
        self._snap_point: np.ndarray | None = None

    # -- drawing -----------------------------------------------------------

    @property
    def grabbing(self) -> bool:
        """Whether a pointer gesture is in progress. ``Viewer.dragging``'s
        public spelling of ``_grab``, for its reason: the router that has to
        ask is ``App``, and it was reaching into two viewers' privates.

        Named ``grabbing`` rather than ``dragging`` since the 2026-09-11
        audit's clay-02: this class inherits ``DragOps``, whose own
        ``dragging`` means something narrower ("a live transform drag --
        gizmo or keyboard"), and a property defined directly on this class
        always wins over one from a mixin. The old name shadowed it, so
        every ``self.dragging`` in ``_view_drag.py`` and every
        ``getattr(view, "dragging", False)`` in ``clay_mode.py`` -- both of
        which want the narrow, "is a transform running" meaning -- silently
        got this broad, "is any grab (orbit/pan/marquee/gizmo/keydrag) live"
        one instead. Consequence: a bare tool key or mode switch typed while
        the user was merely orbiting the camera was routed into
        ``drag_key`` and eaten, and Ctrl+Z/Y/N/O/W/Tab were refused as
        "blocked by a live drag" for the duration of any camera gesture."""

        return self._grab is not None

    def draw(self, doc: Any, rect: tuple[float, float, float, float], dt: float) -> Any:
        """Draw one frame into the viewport. -> the resolved texture.

        Skipped -- the last resolved texture returned as-is -- when nothing
        that feeds the draw moved (B13): the document's own ``rev`` covers
        every edit, selection and visibility change; ``handle_event`` marks a
        redraw for hover, marquee and drags; the camera answers for itself;
        and the tool decides which gizmo is on screen, so it is in the key.
        ``doc`` itself rides in ``self._last_doc`` alongside the key's
        ``id(doc)`` -- the 2026-09-07 audit's clay-09 -- so a closed
        document cannot be collected and a new one minted at the same
        address while this cache still trusts that id.
        """
        self._rect = rect
        width, height = int(max(rect[2], 1)), int(max(rect[3], 1))
        key = (
            width, height, bool(self.wireframe), bool(self.show_grid),
            # Every view setting that changes the picture has to be in the key,
            # or the frame that turns one on is skipped as "nothing moved" and
            # the switch reads as broken until something else forces a redraw.
            bool(self.flat), bool(self.wire_overlay), bool(self.xray),
            id(doc), doc.rev, getattr(self.state, "tool", "select"),
        )
        if self._frame_unchanged(key):
            return self.viewport.texture
        self._last_render_key = key
        self._last_doc = doc
        self._render_dirty = False
        self._resize(width, height)
        self.camera.update(dt)
        self.sync(doc)

        self.renderer.draw(
            self.viewport,
            self.camera,
            self._composite(doc),
            wireframe=self.wireframe,
            flat=self.flat,
            show_grid=self.show_grid,
            wire_overlay=self.wire_overlay,
            alpha=XRAY_ALPHA if self.xray else 1.0,
            overlays=self._element_overlays(doc) + self._gizmo_draws(doc, height),
        )
        return self.viewport.texture

    def _world(self, obj: Any) -> Any:
        """This object's world matrix, memoized on the transform arrays (B26).

        Sound because every transform write *rebinds* the three arrays -- the
        documented gizmo rule ("rebind rather than write through trs()'s live
        arrays") -- so the identity triple changes exactly when the transform
        does. One memo serves the composite, the overlays, the centres and the
        bounds, which is what "compute world matrices once" means here.
        """
        key = (id(obj.translation), id(obj.rotation), id(obj.scale))
        hit = self._world_cache.get(obj.uid)
        if hit is not None and hit[0] == key:
            return hit[1]
        world = m3.compose(obj.translation, obj.rotation, obj.scale)
        self._world_cache[obj.uid] = (
            key, world, (obj.translation, obj.rotation, obj.scale)
        )
        if len(self._world_cache) > 4096:
            self._world_cache.clear()
        return world

    def _composite(self, doc: Any) -> Any:
        """Every cached object as one thing the renderer can draw in one pass.

        ``Renderer.draw`` clears the target it is given, so a call per object
        would erase the one before it -- and a per-object call would also mean
        a grid pass and an overlay pass apiece. What it actually consumes from
        a ``GpuModel`` is ``draws`` and ``palette``, so the composite supplies
        exactly those over the cached entries, with each node's ``world`` set
        to that object's transform.

        The transform being carried on the node rather than in the cache key is
        what keeps a move from rebuilding a buffer: it is a uniform written per
        frame, which is what ``world`` already is for a glTF node.
        """
        draws = []
        for obj in doc.objects:
            entry = self._cache.get(obj.uid)
            if entry is None:
                continue
            world = self._world(obj)
            for node, primitive in entry.gpu.draws:
                node.world = world
                draws.append((node, primitive))
        return Composite(draws) if draws else None

    def _gizmo_draws(self, doc: Any, height: int) -> list[Any]:
        gizmo = self.active_gizmo(doc)
        if gizmo is None:
            return []
        centre = self.selection_centre(doc)
        if centre is None:
            return []
        gizmo.place(centre, m3.identity(), self.camera, height)
        return gizmo.draws()

    # -- the gizmo, and the app state that chooses it -----------------------

    def active_gizmo(self, doc: Any) -> Any:
        """The gizmo for the current tool, or None.

        The tool lives on the mode's state rather than here, because it is an
        *app* setting shared across documents -- so this reads it rather than
        holding it. Q (select) draws no gizmo in any mode, which in an element
        mode is what frees the left button for the marquee.
        """
        kind = GIZMO_FOR_TOOL.get(getattr(self.state, "tool", "select"), "")
        if not kind or not doc.selection:
            return None
        if doc.element_mode != "object" and not doc.element_sel:
            return None
        return {
            "translate": self.translate_gizmo,
            "rotate": self.rotate_gizmo,
            "scale": self.scale_gizmo,
        }[kind]

    @property
    def state(self) -> Any:
        """Clay's state, or None when the view is driven headlessly."""
        app_ctx = self.app_ctx
        return None if app_ctx is None else getattr(app_ctx.state, "clay", None)

    @property
    def gizmo_drag(self) -> GizmoDragReadout | None:
        """The live G/R/S drag's axis lock, space and amount -- ``None``
        between drags. The one door the pane has onto ``_key_kind`` and
        ``drag_input``, so it stays private and the pane never reaches past it
        (``clay_hud.hint_line`` used to read ``_key_kind`` directly, which is
        set only by :meth:`_view_drag.DragOps.begin_keyboard_drag` -- so the
        line went blank for a drag started by grabbing a handle rather than
        pressing G/R/S. ``_key_kind or state.tool`` is the same fallback
        ``_end_gizmo_drag`` already uses to label the undo step, applied here
        so both agree on what a drag *is*.
        """
        if self._grab not in ("gizmo", "keydrag"):
            return None
        kind = self._key_kind or str(getattr(self.state, "tool", ""))
        entry = self.drag_input
        return GizmoDragReadout(
            kind=kind,
            axis=entry.axis,
            # Clay has one transform frame today: every axis lock is a world
            # axis (``clay.drag.constrain_translation`` et al build it from
            # ``_unit(axis)`` in world space) -- ``panes/clay_header.py``
            # notes the pivot/orientation menu is not built yet. This reports
            # what is actually true rather than a toggle nothing sets.
            space="global",
            amount=self._drag_amount(),
        )

    def _drag_amount(self) -> str:
        """The typed buffer if the user is typing one, else the drag's own
        live readout with the redundant ``[AXIS]`` prefix stripped -- the
        readout already names the axis and ``gizmo_drag`` says it again as
        ``(space)``, so keeping both would repeat the lock in one line."""
        typed = self.drag_input.typed
        if typed:
            return typed
        body = self.drag_hud
        if body.startswith("["):
            _, _, body = body.partition("]")
            body = body.strip()
        return body

    def _ray(self, local: tuple[float, float]):
        return screen_ray(
            self.camera, local[0], local[1], int(self._rect[2]), int(self._rect[3])
        )

    # -- capture and teardown ----------------------------------------------

    def screenshot(self) -> Any:
        return capture.image(self.viewport)

    def render_png(
        self,
        doc: Any,
        *,
        size: int = 1024,
        view: str | None = None,
        frame: bool = True,
        angles: tuple[float, float] | None = None,
        bounds: tuple[Any, Any] | None = None,
        grid: bool = False,
    ) -> bytes:
        """One offscreen square draw of *doc*, flat on white, as PNG bytes.

        Lifted from ``main.py:_render_clay_reference`` (build-to-trellis) and
        generalised for a second caller with a different question: trellis
        always wants the standard three-quarter framing, an MCP client asking
        "what does this look like from the front" wants a named axis. Both are
        answered by the same draw -- frame first, then optionally rotate onto
        an axis without reframing, which is exactly what :meth:`Camera.
        look_along` promises (it "keeps the target and the distance").

        Deliberately **not** ``self.draw``: this always allocates its own
        render target rather than the viewport's live one, because the live
        target is sized to whatever pane is on screen this frame and a second
        caller mid-frame (an agent call queued between two draws) would either
        race the resize or hand back a picture at the wrong resolution. Flat
        shading and a white background always; no gizmos and no overlays
        always, for the reason ``_render_clay_reference`` already stated:
        trellis and an agent are both being shown a *subject*. The grid is the
        one of those four that a caller can now ask for -- see ``grid`` below
        for why that is not a contradiction of the same sentence.

        **Taking a picture may not move the camera the user is looking
        through, and `frame` is why both of those are true at once.** This
        borrows the view's own camera rather than constructing a second one --
        a second camera would be a second place the framing rules in
        ``viewer/camera.py`` have to be kept in sync with -- so every field
        ``frame``/``look_along``/``look_angles`` write is snapshotted and put
        back in the ``finally`` below. Generically, by name, rather than as a
        list of the coupled fields: ``frame`` writes the near and far planes,
        the orbit limits, the target and the spherical triple, *and* the
        damping goals that shadow them, and an explicit list is one field away
        from being wrong. That is also why ``angles`` below is applied through
        :meth:`Camera.look_angles` rather than by writing ``theta``/``phi``
        here directly -- this method already has no business knowing that a
        snap has to move two shadow goal fields as well as the two live ones,
        and restating that coupling at a second call site is exactly the bug
        ``look_along``'s own split into ``look_angles`` exists to prevent. No
        extra restore code is needed for either: the snapshot above and the
        ``vars(self.camera).update(saved)`` below already cover every field
        either method touches, generically.

        ``frame=False`` is the build-to-trellis path and is not a stylistic
        choice. ``_render_clay_reference`` has always drawn through whatever
        camera the user had, so the picture trellis reconstructs from is the
        angle the user was looking at when they pressed the button. Framing it
        here would quietly change the input to every future reconstruction --
        and reconstruction quality in this project is measured against stored
        corpora keyed on their inputs, so a silent change to what the engine
        is handed invalidates comparisons against every measurement already
        taken. An agent asking for a picture has no camera of its own and
        wants the subject to fill the square, so it takes the default.

        ``bounds``, when given, is a ``(lo, hi)`` world AABB framed in place of
        the document's own -- consulted only when ``frame`` is true. This is
        the only way an agent can ask for a picture of *part* of a document:
        the renderer has no per-node alpha, so nothing here can make the rest
        of the scene invisible, and "focus" can only mean "point the camera at
        these objects" rather than hiding the others. The rest still draws.

        ``angles``, when given, is a free ``(theta, phi)`` pair in **radians**
        applied after framing, in place of ``view``. ``view`` is ignored
        whenever ``angles`` is given -- ``agent_clay`` refuses a request that
        supplies both, so this method is never the place that has to decide
        which one wins.

        ``grid``, passed straight through as ``show_grid`` to
        ``Renderer.draw``, is the one exception to "a grid line is a subject
        too" stated above, and the reason is what an agent lacks that a user
        does not: a ruler and a viewport it can walk around in. Handed a
        picture alone, it cannot tell a 10 cm box from a 10 m one -- the grid
        is a **scale cue**, not decoration, and it is the one legitimate
        reason this call ever draws one. When it is true *and* framing found
        bounds to draw (``bounds`` or the document's own), ``renderer.
        fit_grid(lo, hi)`` runs first, the same pairing :meth:`BoundsOps.
        frame_selection` already does before every interactive draw, so the
        ground plane is sized to the subject rather than left at whatever span
        the previous draw set. Sized the way every grid in this app is:
        ``viewer/grid.py``'s ``DIVISIONS = 16`` cells across a span
        ``span_for`` rounds up to a power of ten containing 2.5x the
        footprint, so one cell reads as span/16 metres.

        The default path -- no ``angles``, no ``bounds``, ``grid=False`` -- is
        byte-identical to what this method drew before any of the three
        existed: ``_render_clay_reference`` and every stored-corpus comparison
        keyed on its input depend on that, and it is pinned by
        ``test_render_png_defaults_are_the_picture_the_trellis_path_already_got``.
        """
        self.sync(doc)
        # Shallow-copied, with arrays copied: ``target`` and ``_goal_target``
        # are numpy vectors that ``frame``/``look_angles`` rebind, but a caller
        # that wrote through one in place would otherwise see the restore
        # alias it.
        saved = {
            key: (value.copy() if hasattr(value, "copy") else value)
            for key, value in vars(self.camera).items()
        }
        lo = hi = None
        if frame:
            lo, hi = bounds if bounds is not None else self.world_bounds(doc)
            if lo is not None:
                self.camera.frame(lo, hi)
                # ``frame`` aims at half the box's *height* above the origin,
                # which centres a **grounded** subject -- the asset viewer's
                # models sit with their feet on y=0, and that is what it was
                # written for. A Clay document has no such promise: every
                # generator in ``primitives`` is centred on the origin, so a
                # freshly placed cylinder spans -h/2..+h/2 and the camera ends
                # up looking a full half-height over its top. Re-aiming at the
                # measured centre of the box is what makes an agent's render a
                # picture of the thing rather than of the air above it, and it
                # is done here rather than in ``camera.frame`` because the
                # grounded assumption is right for that method's other callers.
                self.camera.set_target((np.asarray(lo) + np.asarray(hi)) * 0.5)
        if angles is not None:
            # ``view`` is not even inspected in this branch: ``agent_clay``
            # refuses a call that supplies both, so there is no case here
            # where the two could disagree about which one wins.
            self.camera.look_angles(*angles)
        elif view and view != "three_quarter":
            self.camera.look_along(view)
        target = glctx.Viewport(self.ctx, (size, size))
        try:
            if grid and lo is not None:
                self.renderer.fit_grid(lo, hi)
            self.renderer.draw(
                target,
                self.camera,
                self._composite(doc),
                flat=True,
                show_grid=grid,
                background=(1.0, 1.0, 1.0, 1.0),
                overlays=[],
            )
            return capture.png_bytes(target)
        finally:
            target.release()
            vars(self.camera).update(saved)

    def release(self) -> None:
        self.clear()
        self._release_overlays()
        self.translate_gizmo.release()
        self.rotate_gizmo.release()
        self.scale_gizmo.release()
        # Forgotten before the GL object goes, for the reason ``_resize``
        # states -- the driver reissues the name and an unrelated image starts
        # rendering as this one.
        self._forget(self.viewport.texture)
        self.viewport.release()
        self.renderer.release()
