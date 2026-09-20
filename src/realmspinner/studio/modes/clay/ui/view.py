"""The Clay viewport: GL, camera, gizmos and picking over the existing stack.

Modelled on :mod:`~realmspinner.studio.viewer_embed` and reusing its parts wholesale
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

import colorsys
import logging
import weakref
from dataclasses import dataclass
from typing import Any

import moderngl
import numpy as np

from .....kernels.geom3d import math3d as m3
from .....kernels.mesh import mesh as bm
from ...._view_frame import Composite, FrameOps
from ....viewer import capture, glctx
from ....viewer.camera import Camera, screen_ray
from ....viewer.gizmo import RotateGizmo, ScaleGizmo, TranslateGizmo
from ....viewer.render import DrawItem, Renderer
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
from ._view_overlay import OverlayOps

# ``__init__`` annotates the overlay cache with it.
from ._view_overlay import _SelOverlay as _SelOverlay
from ._view_overlay import _toward_eye as _toward_eye

# ``Hit`` is constructed in the mixin, so it is defined there -- but the name
# a caller reaches for is the viewport's, so it is re-exported here.
from ._view_pick import Hit as Hit
from ._view_pick import PickOps

log = logging.getLogger(__name__)


# Which gizmo each tool drives. Held as data so the dispatch is one lookup
# rather than a chain that a fifth tool would have to be threaded through.
GIZMO_FOR_TOOL = {"move": "translate", "rotate": "rotate", "scale": "scale"}


#: How opaque the surface is in X-ray. A third: enough to read the silhouette
#: and the shading, and little enough that an edge on the far side is pickable
#: through it -- which is the whole point of the mode.
XRAY_ALPHA = 0.33


# The Familiar ghost preview's two colours -- reusing the element overlay's
# translucent face-fill recipe (``_view_overlay.FILL_COLOR``) rather than
# inventing a second one, per that module's own docstring on why the fill
# exists at all. Green for what an agent would *add or change*, so it reads
# as "coming" rather than "selected" (which is already red); a dim red tint
# for what it would *remove*, since that is the one direction a fill colour
# already carries the right connotation for.
GHOST_ADD_COLOR = (0.35, 0.9, 0.4, 0.35)
GHOST_REMOVE_COLOR = (0.9, 0.25, 0.25, 0.22)


#: ``ClayView.render_png``'s ``shading`` -> the ``Renderer.draw`` keywords it
#: maps onto. One table rather than a chain of ``if shading == ...:`` because
#: every one of these is a static combination of the interactive viewport's
#: own independent toggles (``flat``/``wireframe``/``wire_overlay``/``xray``,
#: see ``__init__`` above) -- an agent gets six named pictures, the viewport
#: keeps its four orthogonal switches, and this is the one place that maps
#: one onto the other. ``"unlit"`` is first and is the default: it is
#: byte-identical to what ``render_png`` always drew before ``shading``
#: existed (``flat=True`` and nothing else set), which is what
#: ``test_render_png_defaults_are_the_picture_the_trellis_path_already_got``
#: pins -- ``main.py``'s Trellis caller never passes ``shading`` at all, so
#: it is the one path this table must never move. ``"object_id"`` is
#: deliberately absent: it draws through ``Renderer.draw_ids``, a different
#: pass with a different return shape, not a ``Renderer.draw`` keyword
#: combination -- see ``ClayView.render_ids``.
_SHADING_DRAW_KWARGS: dict[str, dict[str, Any]] = {
    "unlit": {"flat": True, "wireframe": False, "wire_overlay": False, "alpha": 1.0},
    "lit": {"flat": False, "wireframe": False, "wire_overlay": False, "alpha": 1.0},
    "wireframe": {"flat": True, "wireframe": True, "wire_overlay": False, "alpha": 1.0},
    "wire_overlay": {"flat": True, "wireframe": False, "wire_overlay": True, "alpha": 1.0},
    "xray": {"flat": True, "wireframe": False, "wire_overlay": False, "alpha": XRAY_ALPHA},
}


#: The golden-ratio conjugate, for ``_id_color``'s hue stepping.
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def _id_color(uid: int) -> tuple[int, int, int]:
    """A deterministic, well-separated 8-bit RGB triple for one uid.

    Golden-ratio hue stepping is the standard answer to "N distinct colours,
    N unknown in advance": stepping the hue by the golden ratio's conjugate
    each time spreads points evenly around the circle with no clustering,
    however many are drawn. Keyed on the uid itself, not a position in some
    ordering, so an object's colour is stable across calls and independent of
    which other objects happen to be visible in any one of them -- which is
    what lets a multi-view ``clay_render`` call share one map across views
    rather than reassigning colours per view.

    Saturation and value are both held comfortably below 1.0 (0.85, 0.95) so
    the brightest channel a colour can ever produce is ``round(0.95 * 255) ==
    242`` -- never 255. ``Renderer.draw_ids`` clears to white, so "never
    white" is what keeps the clear colour unambiguous background rather than
    a collision with whatever uid the golden-ratio sequence happened to land
    on.
    """
    hue = (uid * _GOLDEN_RATIO_CONJUGATE) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (round(r * 255), round(g * 255), round(b * 255))


def _hex_color(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def _count_colors(
    pixels: np.ndarray, colors: dict[int, tuple[int, int, int]]
) -> dict[int, int]:
    """How many pixels of an ``(h, w, 3)`` uint8 image match each of *colors*
    exactly. -> ``{uid: pixel count}``, 0 for a colour the image never drew.

    The 2026-09-20 audit's clay-07: the previous shape, ``_count_color``, ran
    one ``np.all(pixels == color)`` pass over the *whole* image *per object*,
    synchronously on the frame thread that must never block (this module's
    own header) -- ~1.07s for 100 objects at the default 1024 size. Packing
    each pixel's three channels into one integer and asking ``np.unique`` for
    every distinct value and its count is one pass over the pixels regardless
    of how many objects are in *colors*; the per-uid lookup afterwards is a
    dict get, not a second scan. Exact match, not a tolerance:
    ``Renderer.draw_ids`` draws with blending and MSAA both off precisely so
    this never has to guess at a near-miss.
    """
    packed = pixels.astype(np.uint32)
    packed = (packed[..., 0] << 16) | (packed[..., 1] << 8) | packed[..., 2]
    values, counts = np.unique(packed, return_counts=True)
    by_packed = dict(zip(values.tolist(), counts.tolist(), strict=True))
    return {
        uid: by_packed.get((color[0] << 16) | (color[1] << 8) | color[2], 0)
        for uid, color in colors.items()
    }


@dataclass(frozen=True)
class GizmoDragReadout:
    """What a live G/R/S drag amounts to right now, for the HUD's one line.

    ``kind`` is the verb ("move"/"rotate"/"scale"), ``axis`` the lock
    ("x"/"y"/"z"/""), ``space`` the frame the lock is read in, and ``amount``
    the typed buffer while one is being entered, else the drag's own live
    readout. The one shape :meth:`ClayView.gizmo_drag` hands the pane, so
    ``studio/modes/clay/ui/hud.py`` never has to reach past it at ``_key_kind`` or
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
        # The grid's span, in metres, and the god-light toggle -- both pushed
        # by the pane each frame beside ``show_grid`` (``clay_viewport.
        # _clay_viewport``), from ``ClayState.grid_size``/``god_light``. Plain
        # fields rather than reads through ``self.state`` so a headless view
        # (every test in this file, ``render_png``) behaves the same with no
        # app state at all -- ``show_grid`` already works this way.
        self.grid_size = 100.0
        self.god_light = False
        self.radius = 1.0

        self.translate_gizmo = TranslateGizmo(ctx, self.renderer.programs)
        self.rotate_gizmo = RotateGizmo(ctx, self.renderer.programs)
        self.scale_gizmo = ScaleGizmo(ctx, self.renderer.programs)

        self._cache: dict[int, _Entry] = {}
        # Counted rather than inferred: "only what changed was rebuilt" is a
        # property worth asserting, and there is no other way to see it.
        self.rebuilds = 0

        self._rect = (0.0, 0.0, 1.0, 1.0)
        self._grab: str | None = None  # orbit | pan | gizmo | marquee | keydrag | knife
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
        # The knife gesture (tranche 5 integration): armed by
        # ``clay_ops._knife`` firing bare (``DragOps.begin_knife``), live
        # once the press that draws the line lands (``_grab == "knife"``),
        # and cleared on commit or cancel (``DragOps._commit_knife``/
        # ``cancel_drag``). Two screen points only -- the plane is computed
        # once, on release, so the frame loop between press and release does
        # nothing heavier than remembering where the cursor is now. The GL
        # pair below is the line overlay drawn while it is live; see
        # ``_view_overlay.OverlayOps._knife_draws``.
        self._knife_armed = False
        self._knife_from: tuple[float, float] | None = None
        self._knife_to: tuple[float, float] | None = None
        self._knife_vbo: Any = None
        self._knife_vao: Any = None
        self._knife_ibo: Any = None
        self._element_drags: dict[int, _ElementDrag] = {}
        self._overlays: dict[int, _SelOverlay] = {}
        # A collider's own translucent overlay (clay-09, 2026-09-19 audit) --
        # one small per-uid GL cache, the shape ``_overlays`` and
        # ``_ghost_cache`` already use, released in ``release()`` beside them.
        self._collider_overlays: dict[int, _SelOverlay] = {}
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
            tuple[int, int],
            tuple[weakref.ReferenceType[Any], tuple[int, ...], Any, tuple[Any, ...]],
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
        from .....kernels.mesh import drag as bdrag

        self.drag_input = bdrag.DragInput()
        self.drag_hud: str = ""
        # The world position a move has snapped onto, or None. Held rather than
        # recomputed by the consumer because it is found from the *cursor*, and
        # the transform is applied a layer down where the cursor is gone.
        self._snap_point: np.ndarray | None = None

        # The Familiar ghost preview: a scratch document (see
        # ``clay.scratch``) and the diff it produced, or ``None`` between
        # previews. ``_preview_rev`` joins ``draw``'s own skip key -- see
        # ``set_preview``'s own docstring for why a plain field is not enough.
        self._preview: Any = None
        self._preview_scratch: Any = None
        self._preview_rev = 0
        self._ghost_cache: dict[int, _Entry] = {}

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
        every ``self.dragging`` in ``studio/modes/clay/ui/_view_drag.py`` and every
        ``getattr(view, "dragging", False)`` in ``studio/modes/clay/mode.py`` -- both of
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
            float(self.grid_size), bool(self.god_light),
            # A Familiar preview change is not a document edit -- ``doc.rev``
            # does not move for it, on purpose, since nothing has actually
            # happened to the document yet. Without this the frame that
            # brought up (or cleared) a ghost would be skipped as "nothing
            # moved" and the preview would not appear until something else
            # forced a redraw.
            self._preview_rev,
        )
        if self._frame_unchanged(key):
            return self.viewport.texture
        self._last_render_key = key
        self._last_doc = doc
        self._render_dirty = False
        self._resize(width, height)
        self.camera.update(dt)
        self.sync(doc)
        # The grid follows ``grid_size`` every frame rather than the model
        # (Task A) -- set here, after ``camera.update``, so a stale span from
        # a previous document or from ``render_png`` never reaches this draw.
        # Divisions are picked so ``span / divisions`` comes out to 1.0: a 1 m
        # cell whatever the size, which is the promise the size field makes.
        divisions = max(1, int(round(self.grid_size)))
        self.renderer.grid.set_span(self.grid_size, divisions=divisions)
        # ``Camera.frame`` sizes the far plane off the *subject*'s own radius
        # (``radius * 100``), which cuts a 100 m grid clean in half for any
        # prop smaller than a metre. Clamped rather than refactored into
        # ``frame`` itself: every other caller of that method wants the far
        # plane sized to what it framed, and a camera restored from a saved
        # ``.rblk`` never goes through ``frame`` at all on this draw, so the
        # clamp has to live where both paths pass through regardless.
        self.camera.far = max(self.camera.far, self.camera.distance + self.grid_size)
        self.renderer.light_override = self.renderer.env.god_light if self.god_light else None

        self.renderer.draw(
            self.viewport,
            self.camera,
            self._composite(doc),
            wireframe=self.wireframe,
            flat=self.flat,
            show_grid=self.show_grid,
            wire_overlay=self.wire_overlay,
            alpha=XRAY_ALPHA if self.xray else 1.0,
            ground=self.god_light,
            overlays=(
                self._element_overlays(doc)
                + self._collider_draws(doc)
                + self._gizmo_draws(doc, height)
                + self._ghost_draws(doc)
                + self._knife_draws()
            ),
        )
        return self.viewport.texture

    def _world(self, doc: Any, obj: Any) -> Any:
        """This object's world matrix, memoized on the transform arrays (B26,
        extended for tranche 3's parenting).

        Sound because every transform write *rebinds* the three arrays -- the
        documented gizmo rule ("rebind rather than write through trs()'s live
        arrays") -- so the identity triple changes exactly when the transform
        does. One memo serves the composite, the overlays, the centres and the
        bounds, which is what "compute world matrices once" means here.

        **The key carries every ancestor's transform-array identity, not only
        this object's own.** An object's own ``translation``/``rotation``/
        ``scale`` are local to its parent (the module docstring on
        ``document.py``), so moving a *parent* leaves a child's own three
        arrays exactly as they were -- a key built from the child's own arrays
        alone would keep matching after the ancestor moved, and this would
        keep serving the parent's *old* placement composed with the child's
        current local one. Delegated to :meth:`~.document.ClayDoc.world_matrix`
        for the actual composition, which is the one place this package
        answers "what does the parent chain compose to" -- a root's chain is
        empty, so this still costs one ``compose`` for a document with no
        parenting, exactly as it always has.

        **The dict slot is ``(id(doc), obj.uid)``, not ``obj.uid`` alone.**
        The 2026-09-20 audit's clay-20: ``_ghost_draws`` calls this against
        two documents that deliberately share a uid namespace -- the live
        document and its Familiar preview scratch clone -- and a single slot
        per uid meant each document's entry evicted the other's every frame
        both were drawn, defeating the pin that keeps the winning entry's
        transform arrays alive.

        **The entry checks a weak reference to ``doc``, not a strong one.**
        ``_centre_memo``/``_bounds_memo`` (``_view_bounds.py``) pin their
        ``doc`` outright, but those only fire while a gizmo is on screen; this
        memo fires on *every* draw of *every* visible object, so a strong pin
        here would keep a closed tab's whole document graph alive for as long
        as ``_world_cache`` happens to hold that slot (up to the 4096-entry
        cap) -- reopening the exact leak ``tests/modes/clay/
        test_clay_view_cache.py``'s ``test_the_pinned_document_survives_
        every_other_reference_being_dropped`` (2026-09-07 audit's clay-09)
        proves does *not* happen once a tab's only other referrer drops it.
        A dead weakref is simply a miss: ``id(doc)`` is only ever used to pick
        the dict bucket, never trusted on its own, so a collected document's
        address being handed to an unrelated new one is safe too -- the
        dereferenced weakref will not be that new document, however the
        addresses land.
        """
        chain = [obj, *(doc.by_uid(u) for u in doc.ancestors(obj.uid))]
        slot = (id(doc), obj.uid)
        key = tuple(id(v) for o in chain for v in (o.translation, o.rotation, o.scale))
        hit = self._world_cache.get(slot)
        if hit is not None and hit[0]() is doc and hit[1] == key:
            return hit[2]
        world = doc.world_matrix(obj.uid)
        # Every ancestor's arrays are pinned, not only this object's own: an
        # id in the key is only sound while the array it names is alive, and
        # nothing else holds an ancestor's transform alive on this cache's
        # behalf.
        pins = tuple(v for o in chain for v in (o.translation, o.rotation, o.scale))
        self._world_cache[slot] = (weakref.ref(doc), key, world, pins)
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
        removed = self._preview.removed if self._preview is not None else frozenset()
        draws = []
        uids = []
        for obj in doc.objects:
            if obj.uid in removed:
                # Drawn instead as a faint tint by ``_ghost_draws`` -- a
                # Familiar preview that would delete this object should not
                # also show it solid, or the ghost reads as decoration rather
                # than as what would actually happen.
                continue
            if obj.role == "collider":
                # Drawn instead as a translucent fill and wireframe by
                # ``_collider_draws`` (clay-09, 2026-09-19 audit) -- never
                # through this opaque, shaded path, which is what let a
                # collider render as an opaque duplicate of the geometry it
                # previews, occluding or z-fighting it.
                continue
            entry = self._cache.get(obj.uid)
            if entry is None:
                continue
            world = self._world(doc, obj)
            for node, primitive in entry.gpu.draws:
                node.world = world
                draws.append((node, primitive))
                uids.append(obj.uid)
        return Composite(draws, uids) if draws else None

    def _gizmo_draws(self, doc: Any, height: int) -> list[Any]:
        gizmo = self.active_gizmo(doc)
        if gizmo is None:
            return []
        centre = self.selection_centre(doc)
        if centre is None:
            return []
        gizmo.place(centre, m3.identity(), self.camera, height)
        return gizmo.draws()

    # -- the Familiar ghost preview ------------------------------------------

    def set_preview(self, diff: Any, scratch: Any) -> None:
        """Show a Familiar preview: ``diff`` (a ``clay.scratch.PreviewDiff``)
        against ``scratch``, the cloned document it was computed from.

        Bumps ``_preview_rev`` so :meth:`draw`'s own skip key sees it -- a
        preview is not a document edit, so ``doc.rev`` does not move for it,
        and without a key change of some kind the frame that brings the ghost
        up would be skipped as "nothing moved".
        """
        self._preview = diff
        self._preview_scratch = scratch
        self._preview_rev += 1

    def clear_preview(self) -> None:
        """Drop the ghost -- Discard, or Apply once it has landed for real."""
        if self._preview is None:
            return
        self._preview = None
        self._preview_scratch = None
        self._release_ghost()
        self._preview_rev += 1

    def _release_ghost(self) -> None:
        for overlay in self._ghost_cache.values():
            overlay.release()
        self._ghost_cache.clear()

    def _ghost_draws(self, doc: Any) -> list[Any]:
        """The Familiar preview's translucent overlay: green for what an
        agent's scratch run added or changed, a faint red tint for what it
        would remove. Reuses the element overlay's own fill recipe
        (``_view_overlay._SelOverlay``, ``FILL_COLOR``'s translucent
        ``TRIANGLES`` pass) rather than a second one, for that module's own
        reason -- one recipe for "a translucent copy of this face/mesh drawn
        slightly toward the eye."
        """
        preview = self._preview
        if preview is None:
            return []
        scratch = self._preview_scratch
        program = self.renderer.programs.get("solid")
        live: set[int] = set()
        items: list[Any] = []

        def _fill(
            overlay: Any, mesh: Any, world: Any, color: tuple[float, float, float, float]
        ) -> Any:
            tris, _ = bm.triangulate(mesh)
            vao = overlay.indexed(tris, moderngl.TRIANGLES)
            if vao is None:
                return None
            return DrawItem(
                vao=vao,
                color=color,
                model=_toward_eye(self.camera.position) @ world,
                mode=moderngl.TRIANGLES,
                depth=True,
            )

        for uid in preview.added | preview.mesh_changed:
            try:
                obj = scratch.by_uid(uid)
            except KeyError:
                continue
            live.add(uid)
            # Evaluated -- this ghost is standing in for what would actually
            # land on screen, the same reason the real cache (``_view_cache``)
            # draws the evaluated mesh rather than the base.
            mesh = scratch.evaluated(uid)
            key = (uid, id(mesh))
            overlay = self._ghost_cache.get(uid)
            if overlay is None or overlay.key != key:
                if overlay is not None:
                    overlay.release()
                overlay = _SelOverlay(self.ctx, program, key, mesh.positions)
                overlay.pins = mesh
                self._ghost_cache[uid] = overlay
            # ``scratch``, not ``doc``: the added/changed branch's ``obj`` is
            # the scratch clone's own, and its ancestor chain (if any) lives
            # there too -- composing against ``doc`` would walk the wrong
            # document's objects, or a uid this one does not have at all.
            world = self._world(scratch, obj)
            item = _fill(overlay, mesh, world, GHOST_ADD_COLOR)
            if item is not None:
                items.append(item)

        for uid in preview.removed:
            try:
                obj = doc.by_uid(uid)
            except KeyError:
                continue
            live.add(uid)
            mesh = doc.evaluated(uid)
            key = (uid, id(mesh))
            overlay = self._ghost_cache.get(uid)
            if overlay is None or overlay.key != key:
                if overlay is not None:
                    overlay.release()
                overlay = _SelOverlay(self.ctx, program, key, mesh.positions)
                overlay.pins = mesh
                self._ghost_cache[uid] = overlay
            item = _fill(overlay, mesh, self._world(doc, obj), GHOST_REMOVE_COLOR)
            if item is not None:
                items.append(item)

        for uid in [u for u in self._ghost_cache if u not in live]:
            self._ghost_cache.pop(uid).release()
        return items

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
            # ``_unit(axis)`` in world space) -- ``studio/modes/clay/ui/header.py``
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

    def _frame_camera(
        self,
        doc: Any,
        *,
        frame: bool,
        angles: tuple[float, float] | None,
        bounds: tuple[Any, Any] | None,
        view: str | None,
    ) -> tuple[dict[str, Any], Any, Any]:
        """The camera snapshot/frame/rotate steps ``render_png`` and
        ``render_ids`` share. -> ``(saved, lo, hi)``: ``saved`` is what the
        ``finally`` in each restores, ``lo``/``hi`` are the framed bounds (or
        ``None``) for a caller that also wants to fit the grid to them.

        Factored out rather than duplicated because it is exactly the
        coupling ``render_png``'s own docstring warns about: ``frame``,
        ``look_angles`` and ``look_along`` between them write the near/far
        planes, the orbit limits, the target, the spherical triple and their
        damping-goal shadows, and a second call site restating which fields
        move together is one edit away from the two drifting.
        """
        saved = {
            key: (value.copy() if hasattr(value, "copy") else value)
            for key, value in vars(self.camera).items()
        }
        lo = hi = None
        if frame:
            lo, hi = bounds if bounds is not None else self.world_bounds(doc)
            if lo is not None:
                self.camera.frame(lo, hi)
                self.camera.set_target((np.asarray(lo) + np.asarray(hi)) * 0.5)
        if angles is not None:
            self.camera.look_angles(*angles)
        elif view and view != "three_quarter":
            self.camera.look_along(view)
        return saved, lo, hi

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
        god_light: bool = False,
        shading: str = "unlit",
    ) -> bytes:
        """One offscreen square draw of *doc*, on white, as PNG bytes.

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
        race the resize or hand back a picture at the wrong resolution. A
        white background always; no gizmos and no overlays always, for the
        reason ``_render_clay_reference`` already stated: trellis and an
        agent are both being shown a *subject*. The grid is the one exception
        a caller can now ask for -- see ``grid`` below for why that is not a
        contradiction of the same sentence.

        ``shading`` picks which of ``Renderer.draw``'s own combination of
        ``flat``/``wireframe``/``wire_overlay``/``alpha`` this draw uses --
        see ``_SHADING_DRAW_KWARGS`` for the table and why ``"unlit"`` (the
        default) is the one entry that must never move: it is byte-identical
        to what this method always drew before ``shading`` existed, which is
        what keeps ``_render_clay_reference`` -- and every stored-corpus
        comparison keyed on its input -- looking at the same picture it
        always has. ``"object_id"`` is not a legal value here at all; it
        draws through :meth:`render_ids` instead, a different pass with a
        different return shape.

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
        fit_grid(lo, hi)`` runs first, so the ground plane is sized to the
        subject rather than left at whatever span the *interactive* viewport
        currently shows -- Clay's own grid is fixed-size now (Task A,
        ``ClayState.grid_size``) and no longer answers this question the way
        :meth:`BoundsOps.frame_selection` used to. Sized the way every
        model-following grid in this app is: ``viewer/grid.py``'s
        ``DIVISIONS = 16`` cells across a span ``span_for`` rounds up to a
        power of ten containing 2.5x the footprint, so one cell reads as
        span/16 metres -- saved and restored below so it does not leak into
        the live viewport's own, differently-sized grid.

        The default path -- no ``angles``, no ``bounds``, ``grid=False`` -- is
        byte-identical to what this method drew before any of the three
        existed: ``_render_clay_reference`` and every stored-corpus comparison
        keyed on its input depend on that, and it is pinned by
        ``test_render_png_defaults_are_the_picture_the_trellis_path_already_got``.

        ``god_light``, Task C's flat overhead light and ground plane, exists
        here so a test can exercise it through the same headless draw every
        other picture in this method goes through -- **it is not a parameter
        ``agent_clay`` exposes**: an agent asking what an object looks like
        wants the render it will actually be shown under, and a second
        lighting mode on the MCP surface is a second thing a client has to
        know exists before it can ask a useful question.

        ``self.renderer.grid``'s span and divisions are saved and restored in
        the ``finally`` below, the same shape ``self.camera``'s fields are:
        ``fit_grid`` (for ``grid=True``) sizes the grid to *this* subject on
        the app's one shared ``Renderer``, and leaving that in place would
        have an agent's render silently resize the grid the user is looking
        at in the live viewport, the next time it redraws.
        """
        self.sync(doc)
        saved_grid = (self.renderer.grid.span, self.renderer.grid.divisions)
        # ``frame`` aims at half the box's *height* above the origin, which
        # centres a **grounded** subject -- the asset viewer's models sit
        # with their feet on y=0, and that is what it was written for. A
        # Clay document has no such promise: every generator in
        # ``primitives`` is centred on the origin, so a freshly placed
        # cylinder spans -h/2..+h/2 and the camera ends up looking a full
        # half-height over its top. ``_frame_camera`` re-aims at the box's
        # own centre, which is what makes an agent's render a picture of the
        # thing rather than of the air above it.
        saved, lo, hi = self._frame_camera(
            doc, frame=frame, angles=angles, bounds=bounds, view=view
        )
        target = glctx.Viewport(self.ctx, (size, size))
        try:
            if grid and lo is not None:
                self.renderer.fit_grid(lo, hi)
            self.renderer.light_override = self.renderer.env.god_light if god_light else None
            self.renderer.draw(
                target,
                self.camera,
                self._composite(doc),
                show_grid=grid,
                background=(1.0, 1.0, 1.0, 1.0),
                overlays=[],
                ground=god_light,
                **_SHADING_DRAW_KWARGS[shading],
            )
            return capture.png_bytes(target)
        finally:
            target.release()
            vars(self.camera).update(saved)
            self.renderer.grid.set_span(saved_grid[0], divisions=saved_grid[1])
            self.renderer.light_override = None

    def render_ids(
        self,
        doc: Any,
        *,
        size: int = 1024,
        view: str | None = None,
        frame: bool = True,
        angles: tuple[float, float] | None = None,
        bounds: tuple[Any, Any] | None = None,
    ) -> tuple[bytes, list[tuple[int, str, int]]]:
        """The object-id picture ``shading="object_id"`` answers with: every
        visible object flat-coloured by :func:`_id_color`, no lighting, no
        grid, no gizmos, as PNG bytes plus a per-uid pixel count.

        Deliberately **not** ``render_png(shading="object_id")`` even though
        every framing argument is shared: the return shape is different -- a
        picture *and* a table, not a picture alone -- and folding a second
        element onto ``render_png``'s return would have meant every other
        caller of it (the Trellis path chief among them) gaining an optional
        tuple member it never uses. Same reasoning as the ``screenshot``/
        ``draw`` split already in this class.

        No ``grid`` parameter at all: ``agent_clay`` refuses the combination
        before this is ever reached (grid lines drawn through ``draw_ids``
        would be false colour with no uid behind them, corrupting the very
        pixel counts this method exists to produce), so there is no legal
        value for this method to accept and nothing to thread through.

        The colour table is built from ``doc.objects`` filtered on
        ``visible`` -- exactly what :meth:`_composite`'s own cache lookup
        draws, since ``sync`` never uploads a hidden object (see
        ``_view_cache.CacheOps.sync``) -- so every uid this method reports on
        is one the picture could actually have coloured, and a uid entirely
        occluded in this particular view still gets its row, at ``px=0``:
        "hidden from this view", not "does not exist".
        """
        self.sync(doc)
        saved, _lo, _hi = self._frame_camera(
            doc, frame=frame, angles=angles, bounds=bounds, view=view
        )
        target = glctx.Viewport(self.ctx, (size, size), samples=1)
        try:
            colors = {obj.uid: _id_color(obj.uid) for obj in doc.objects if obj.visible}
            self.renderer.draw_ids(target, self.camera, self._composite(doc), id_colors=colors)
            png = capture.png_bytes(target)
            pixels = target.read_rgba()[..., :3]
            counts = _count_colors(pixels, colors)
            rows = [
                (uid, _hex_color(color), counts[uid]) for uid, color in sorted(colors.items())
            ]
            return png, rows
        finally:
            target.release()
            vars(self.camera).update(saved)

    def release(self) -> None:
        self.clear()
        self._release_overlays()
        self._release_collider_overlays()
        self._release_ghost()
        self._release_knife_overlay()
        self.translate_gizmo.release()
        self.rotate_gizmo.release()
        self.scale_gizmo.release()
        # Forgotten before the GL object goes, for the reason ``_resize``
        # states -- the driver reissues the name and an unrelated image starts
        # rendering as this one.
        self._forget(self.viewport.texture)
        self.viewport.release()
        self.renderer.release()
