"""Clay's UV pane: islands, pan/zoom, box-select, move/rotate/scale, Pack.

Tranche 6 ("UV and materials", ``dev/CLAY-PLAN.md``). Drawn with an imgui
draw list the way Inker's and Plotter's canvases are (``add_line``/
``add_rect``/``add_convex_poly_filled`` over an ``invisible_button``'s
region) rather than as a composited texture -- the same reasoning Plotter's
own canvas module gives for its cell loop: a UV layout is a handful of
islands, not a raster, and redrawing it as vector primitives every frame is
cheaper than compositing and uploading a picture of it.

**Pan and zoom reuse :class:`~.shell.paintview.PaintView`**, the one 2-D
view every canvas in this app already shares (Inker, Plotter, Packwright) --
the uv unit square ``[0, 1] x [0, 1]`` is simply treated as a
:data:`REF_PX`-square "texture" for the purpose of that arithmetic, which is
what lets :func:`~.shell.paintview.fit`/``to_screen``/``to_image`` work
unmodified instead of a second, uv-flavoured copy of the same maths.

**Every edit goes through :meth:`~.document.ClayDoc.set_mesh` with
``keep_generator=True``, as one undo step.** A uv change is not geometry --
the object's generator (if it has one) still describes exactly the same
vertices and faces after its islands are moved around the square, which is
why this is the one caller in the pane layer besides the properties panel's
own generator-parameter rebuild that is allowed to pass that flag. Move
(a canvas drag) folds every frame of the gesture into one step through
``controls.fold_undo``, the same "draw, fold, act" pattern the properties
panel's transform and material fields already use; Pack is a single button
press, so it needs no folding at all.

**Rotate and scale are live drags too, armed by E and R** -- Realmspinner's own
Clay-viewport tool letters (``clay_mode.TOOL_KEYS``: Q select, W move, E
rotate, R scale), not Blender's G/R/S, because this app already spends R on
scale for the exact left hand that already knows Q/W/E/R from the 3-D
viewport, and teaching the opposite letter in a sibling pane of the same mode
would be a worse convention than picking one of our own. Pressing E or R
while an island is boxed and the canvas is hovered arms the gesture; moving
the mouse (no button held, the 3-D viewport's own keyboard-drag shape --
``ui._view_drag.DragOps.begin_keyboard_drag``) previews it live, a left click
commits, and Escape or a right-click cancels. Each frame recomputes the
**absolute** angle or factor from the drag's own start -- against a *pivot*
that is only the point the gesture is measured from (the bbox centre of every
selected corner) and never the point an island is actually turned about,
which stays each island's own uv-bbox centre exactly as the one-shot fields
already use (:func:`~.uvtools.transform_islands`'s own per-island pivot) --
and applies it to the **mesh the drag began with**, never to whatever ``doc``
holds this frame: reading the live mesh back would ask each island to turn
about a centre already displaced by every earlier frame's own edit, and an
asymmetric island's axis-aligned bbox centre moves under a *partial* turn in
a way a *full* turn from the original shape would not reproduce -- the pivot
creeping a little further every frame. :func:`begin_live_transform` snapshots
the base mesh and island set once (a ``Mesh`` is immutable, so holding it is
free); :func:`update_live_transform` is the one function every frame of the
drag calls.

Folded the same *shape* ``controls.fold_undo`` gives translate -- a mark
taken before the first frame, collapsed into one step at the end -- but by
hand: a buttonless keyboard-drag has no single imgui item held active for
that helper's activation/deactivation hook to key off of, the way the
canvas's own ``invisible_button`` brackets a mouse drag. A cancel is the same
mark, folded and then reversed and forgotten in one move
(``UndoStack.undo(doc, redoable=False)``) rather than a plain ``doc.undo()``,
which would leave the abandoned gesture sitting on the *redo* stack -- not
"nothing left on the stack" if Ctrl+Shift+Z could still bring it back.

**Left out.** No axis lock: a uv rotation has exactly one axis (the plane's
own normal), and this pane's scale has always been uniform-only, matching
both the typed field and :func:`~.uvtools.transform_islands`'s own
``scale: float`` -- there is no second axis for a lock to mean anything
against on either tool. No typed value mid-drag either: the toolbar's own
rotate/scale fields sit right beside the canvas and already are the exact
path (:func:`apply_rotate`/:func:`apply_scale` with no drag at all), and the
3-D viewport's own typed entry (``kernels.mesh.drag.DragInput``) is read off
*pygame* key events routed through ``ClayView.handle_event``, a different
input path than this imgui-drawn canvas's ``is_key_pressed`` polling --
duplicating that text-buffer handling here would be a second, narrower copy
of an input surface the pane already offers in the adjacent field.

**Selection is not undoable, the document's own rule restated for islands**:
:attr:`UvPaneState.selected_islands` lives on the tab (this pane's own state,
below), never on ``ClayDoc``, and no function here pushes a step for
changing it.

The pure island-selection maths (:func:`islands_in_rect`,
:func:`touched_islands`) take a :class:`~.mesh.Mesh` and plain arrays, not a
document or imgui, so they are unit-tested directly -- see
``tests/modes/clay/test_uv_pane.py``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ......kernels.mesh import elements as el
from ......kernels.mesh import uvtools
from ..... import controls, icons, imgui_backend, theme, widgets
from .....manual import render as manual_render
from .....shell import paintview
from .....tokens import sp
from ... import mode as clay_mode

log = logging.getLogger(__name__)


@dataclass
class UvPaneState:
    """This pane's own view and island selection -- per tab, beside
    :attr:`~.state.ClayTab.view` and for the same reason: a tab remembers
    where it was, so switching away and back to the UV pane must not lose
    your pan, your zoom or which islands you had boxed, any more than
    switching tabs resets the 3-D camera.

    Tranche 6 ("UV and materials"). ``view`` is :class:`~.shell.paintview.
    PaintView`, the same pan/zoom every 2-D canvas in this app already shares
    (Inker, Plotter, Packwright) -- reused rather than reinvented, and it
    is *not* the same thing as :data:`~.state.CameraView`, which is the 3-D
    camera's own yaw/pitch/distance.

    ``selected_islands`` names ``for_uid`` too: an island id only means
    something against the mesh that minted it (``uvtools.islands`` numbers
    islands ``0..k-1`` fresh on every call, off whatever uv the mesh
    currently carries), so a selection made on one object must not go on
    highlighting "island 2" once the properties selection moves to a
    different object with a completely different layout. The pane clears
    both together the moment the selected object changes (see :func:`_body`).

    Lives here rather than in ``state.py``: it is this pane's own drawing
    state (pan, zoom, a live drag's gesture) rather than anything about the
    document, the same reason every other pane's per-tab view state is not on
    the shared tab class. Moved out of ``state.py`` 2026-09-19 -- a second
    ``*State`` class there broke ``test_docmodes.py``'s one-tab-state-class-
    per-mode sweep, which reads a mode's tab-state shape off exactly one
    class. ``ClayTab.uv_view``'s default factory (``state.py``'s
    ``_new_uv_view``) imports this lazily to avoid a cycle: this module
    already imports ``mode``, which imports ``state`` for ``ClayTab`` itself.
    """

    view: Any = field(default_factory=paintview.PaintView)
    for_uid: int = 0
    selected_islands: frozenset[int] = field(default_factory=frozenset)
    # A one-shot rotate/scale field's own pending value -- not persisted
    # across a commit, the same "value='' every frame" sentinel
    # ``clay_props._tags``'s tag-add field already uses for an *action*
    # rather than a property: pressing Apply resets each back to its
    # identity (0 degrees, x1 scale) rather than leaving the box showing a
    # number that was already applied.
    pending_rotate: float = 0.0
    pending_scale: float = 1.0
    # Which gesture is in flight -- "" (none), "box" (a marquee, replacing
    # ``selected_islands``), "move" (dragging the current selection with the
    # mouse held), or "rotate"/"scale" (a buttonless keyboard drag armed by E
    # or R, see the module docstring). "box"/"move" are decided once, on the
    # frame a mouse drag starts (see :func:`_canvas`), by whether the click
    # landed on an island already in ``selected_islands``; "rotate"/"scale"
    # are set by :func:`begin_live_transform` and cleared by
    # :func:`commit_live_transform`/:func:`cancel_live_transform`.
    drag_mode: str = ""
    # For "box"/"move": the uv point the mouse went down at. For
    # "rotate"/"scale": the uv point the keyboard drag was armed at -- the
    # gesture's own zero reading, against which every frame's *absolute*
    # angle or factor is measured (see :func:`drag_angle`/:func:`drag_scale`).
    drag_start: tuple[float, float] = (0.0, 0.0)
    # Where the drag's translate delta was last measured from -- the
    # *previous frame's* pointer, not the drag's start: a move gesture
    # applies one small, exact translate per frame (see ``apply_translate``),
    # because unlike rotate/scale, translate has no pivot to drift as it
    # composes, so accumulating it incrementally is both simpler and exact.
    drag_last: tuple[float, float] = (0.0, 0.0)
    # A live rotate/scale's own three-part snapshot, taken once at
    # :func:`begin_live_transform` and read by every frame afterwards --
    # never the document's current mesh, which is the whole of what keeps a
    # multi-frame rotate or scale exact instead of drifting (module
    # docstring). ``drag_base`` is ``None`` outside such a gesture.
    drag_base: Any = None
    drag_islands: frozenset[int] = field(default_factory=frozenset)
    drag_pivot: tuple[float, float] = (0.0, 0.0)
    # The ``UndoStack`` mark :func:`begin_live_transform` opened, for
    # :func:`commit_live_transform`/:func:`cancel_live_transform` to fold or
    # unwind back to -- meaningless outside a live rotate/scale, the same
    # trust every other ``mark()``/``collapse_since`` pair in this codebase
    # already places in being called in order (``ClayDoc.add_group``'s own
    # local ``mark`` has no guard either).
    drag_mark: int = 0

#: The uv unit square is treated as a texture this many pixels on a side, for
#: :mod:`~.shell.paintview`'s pan/zoom arithmetic alone -- it never appears in
#: anything written to the document. Matches ``uvtools.texel_density``'s own
#: default ``texture_px``, so "100%" in this pane and a 1K-texture density
#: reading are the same familiar scale rather than two arbitrary numbers.
REF_PX = 1024.0

#: :func:`stretch` readings at or past this (in either direction) get a full-
#: strength tint; between 0 and here the tint fades in linearly. One stop
#: either way (``log2`` of a 2x area ratio) is already a visibly soft or
#: aliased texture, which is what earns the full-strength colour.
STRETCH_FULL = 1.0

# --- pure maths: island hit-testing and selection --------------------------


def _face_of_corner(mesh: Any) -> np.ndarray:
    """Corner index -> face index.

    A three-line reimplementation of ``uvtools._face_of_corner`` rather than
    an import of a leading-underscore name out of a sibling module -- that
    underscore means private *to that module*, and this pane is not it.
    """
    counts = np.diff(mesh.starts.astype("i8"))
    return np.repeat(np.arange(len(counts), dtype="i8"), counts)


def _point_in_face(point: tuple[float, float], corners: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test over one face's ``(n, 2)`` uv corners.

    The classic even-odd rule: a horizontal ray from *point* out to
    ``+u`` crosses the polygon's boundary an odd number of times iff the
    point is inside. Good enough for the convex, non-self-intersecting faces
    every generator and unwrap in this package produces (:func:`_faces`'s own
    docstring states the same limit for drawing).
    """
    x, y = point
    inside = False
    n = len(corners)
    for i in range(n):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at:
                inside = not inside
    return inside


def islands_in_rect(
    mesh: Any, ids: np.ndarray, rect: tuple[float, float, float, float]
) -> set[int]:
    """Island ids a box (or a click) at *rect* = ``(u0, v0, u1, v1)`` catches.

    *rect*'s corners need not be ordered -- a marquee dragged up-and-left of
    its start is as valid as one dragged down-and-right, and the caller (a
    live drag) hands over whichever corner the pointer is at this frame
    without knowing which way is "forward".

    **A real box** (``u0 != u1`` or ``v0 != v1``) catches every island with
    at least one uv corner inside it -- not "every corner", so a box only
    partly overlapping an island still catches it, the ordinary box-select
    feel of every 2-D editor and Wings3D's own "going down is a union"
    reasoning (``elements.convert``'s docstring) restated for islands.

    **A degenerate rect** (``u0 == u1 and v0 == v1``) is a click, and a click
    in the *middle* of a face -- nowhere near a corner -- must still hit it,
    which the corner test alone cannot do: it falls back to a proper
    point-in-face test (:func:`_point_in_face`) over every face, run only
    when the cheap corner test caught nothing, so an ordinary drag never
    pays for it.
    """
    if mesh.uv is None or len(ids) == 0:
        return set()
    u0, v0, u1, v1 = rect
    lo_u, hi_u = (u0, u1) if u0 <= u1 else (u1, u0)
    lo_v, hi_v = (v0, v1) if v0 <= v1 else (v1, v0)
    uv = mesh.uv
    inside = (uv[:, 0] >= lo_u) & (uv[:, 0] <= hi_u) & (uv[:, 1] >= lo_v) & (uv[:, 1] <= hi_v)
    if inside.any():
        foc = _face_of_corner(mesh)
        return {int(i) for i in np.unique(ids[foc[inside]])}
    if u0 != u1 or v0 != v1:
        return set()  # a real, empty box: nothing to fall back to
    starts = mesh.starts.astype("i8")
    n_faces = len(starts) - 1
    for face in range(n_faces):
        lo, hi = starts[face], starts[face + 1]
        if hi - lo >= 3 and _point_in_face((u0, v0), uv[lo:hi]):
            return {int(ids[face])}
    return set()


def touched_islands(mesh: Any, ids: np.ndarray, sel: Any) -> set[int]:
    """Island ids the element selection *sel* touches, however it is expressed.

    A **union** test: a vertex or an edge counts a face as touched the moment
    any one of its corners is at that vertex or on that edge --
    ``elements.affected_verts``'s own reasoning for a drag's centroid, not
    ``elements.convert``'s up-conversion rule (which needs *every* corner
    selected and would silently miss an island with only one selected
    vertex on it). ``sel`` may be ``None`` or empty -- the ordinary state of
    an object with nothing selected inside it -- and answers ``set()``.
    """
    if mesh.uv is None or sel is None or el.is_empty(sel):
        return set()
    verts = el.affected_verts(mesh, sel)
    if len(verts) == 0:
        return set()
    touched_corner = np.isin(mesh.loops, verts)
    if not touched_corner.any():
        return set()
    foc = _face_of_corner(mesh)
    faces = np.unique(foc[touched_corner])
    return {int(i) for i in np.unique(ids[faces])}


# --- pure maths: the live rotate/scale gesture ------------------------------


def selection_pivot(
    mesh: Any, ids: np.ndarray, island_ids: Any
) -> tuple[float, float] | None:
    """The ``(u, v)`` point a live rotate/scale is measured from.

    The centre of the combined bbox of every corner belonging to a selected
    island -- **only** used to turn the pointer's motion into an angle or a
    factor. The edit itself still turns each island about its *own* centre
    (:func:`~.uvtools.transform_islands`'s own per-island pivot, unchanged
    from the one-shot fields), the same split the 3-D viewport already makes
    between the selection's centroid (what a keyboard drag measures against)
    and each object's own origin (what it actually turns about).

    ``None`` when the selection touches no uv corner at all -- nothing to
    measure a gesture from -- :func:`islands_in_rect`'s own declining case,
    restated.
    """
    if mesh.uv is None or not island_ids:
        return None
    foc = _face_of_corner(mesh)
    mask = np.isin(ids[foc], list(island_ids))
    if not mask.any():
        return None
    pts = mesh.uv[mask]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return (float((lo[0] + hi[0]) / 2.0), float((lo[1] + hi[1]) / 2.0))


def drag_angle(
    pivot: tuple[float, float], anchor: tuple[float, float], now: tuple[float, float]
) -> float:
    """Degrees swept from *anchor* to *now* about *pivot*, signed the same
    way :func:`~.uvtools.transform_islands`'s own rotation matrix turns a
    positive angle -- u toward v.

    Zero when either arm has collapsed onto the pivot (the pointer has not
    moved away from it yet, at the press or ever): a live drag needs a
    harmless "nothing to report" every frame, not a ``NaN`` out of
    ``atan2(0, 0)``.
    """
    ax, ay = anchor[0] - pivot[0], anchor[1] - pivot[1]
    bx, by = now[0] - pivot[0], now[1] - pivot[1]
    len_a, len_b = math.hypot(ax, ay), math.hypot(bx, by)
    if len_a < 1e-9 or len_b < 1e-9:
        return 0.0
    cos_t = (ax * bx + ay * by) / (len_a * len_b)
    cross = ax * by - ay * bx
    return math.degrees(math.atan2(cross, cos_t))


def drag_scale(
    pivot: tuple[float, float], anchor: tuple[float, float], now: tuple[float, float]
) -> float:
    """The uniform factor a live scale currently amounts to: how much
    farther *now* sits from *pivot* than *anchor* did.

    ``1.0`` (identity) when the anchor sat on the pivot itself -- a
    zero-length reference has no ratio to report, and a live drag needs an
    answer every frame regardless of where the gesture happened to arm.
    """
    len_a = math.hypot(anchor[0] - pivot[0], anchor[1] - pivot[1])
    if len_a < 1e-9:
        return 1.0
    len_b = math.hypot(now[0] - pivot[0], now[1] - pivot[1])
    return len_b / len_a


# --- pure edits: one undo step each -----------------------------------------


def apply_translate(
    doc: Any, uid: int, island_ids: Any, translate: tuple[float, float]
) -> bool:
    """Slide the chosen islands by *translate*, as one step. -> whether it moved."""
    obj = doc.by_uid(uid)
    mesh = uvtools.transform_islands(obj.mesh, list(island_ids), translate=translate)
    return doc.set_mesh(uid, mesh, keep_generator=True)


def apply_rotate(
    doc: Any, uid: int, island_ids: Any, degrees: float, *, base: Any = None
) -> bool:
    """Rotate the chosen islands (each about its own centre) by *degrees*.

    ``base`` is the mesh to rotate *from* -- the object's current mesh by
    default (the one-shot toolbar field's own case), or a live rotate's own
    frozen drag-start snapshot (:func:`update_live_transform`), which is what
    keeps a multi-frame drag exact: rotating from *this frame's* absolute
    angle against the object's *current* mesh would turn each island about a
    centre already displaced by every earlier frame's own edit.
    """
    obj = doc.by_uid(uid)
    source = obj.mesh if base is None else base
    mesh = uvtools.transform_islands(source, list(island_ids), rotate_deg=degrees)
    return doc.set_mesh(uid, mesh, keep_generator=True)


def apply_scale(
    doc: Any, uid: int, island_ids: Any, factor: float, *, base: Any = None
) -> bool:
    """Scale the chosen islands (each about its own centre) by *factor*.

    ``base`` is :func:`apply_rotate`'s own parameter, for the same reason.
    """
    obj = doc.by_uid(uid)
    source = obj.mesh if base is None else base
    mesh = uvtools.transform_islands(source, list(island_ids), scale=factor)
    return doc.set_mesh(uid, mesh, keep_generator=True)


def apply_pack(doc: Any, uid: int, *, margin: float = 0.005, rotate: bool = False) -> bool:
    """Pack every island of *uid*'s mesh into the unit square, as one step."""
    obj = doc.by_uid(uid)
    mesh = uvtools.pack_islands(obj.mesh, margin=margin, rotate=rotate)
    return doc.set_mesh(uid, mesh, keep_generator=True)


# --- the live rotate/scale gesture: begin, one frame, commit, cancel --------


def begin_live_transform(
    doc: Any,
    view_state: UvPaneState,
    kind: str,
    mesh: Any,
    ids: np.ndarray,
    anchor: tuple[float, float],
) -> bool:
    """Arm a live rotate (``kind="rotate"``) or scale (``"scale"``) on
    *view_state*'s current selection. -> whether it armed.

    Snapshots the base mesh and the island set onto *view_state* -- a
    ``Mesh`` is immutable, so holding it costs nothing -- and opens one undo
    gesture by hand (``UndoStack.mark()``): the same mark/collapse shape
    ``controls.fold_undo`` gives the mouse-driven translate drag, taken
    directly here because this drag has no mouse button held for that
    helper's item-activation hook to key off of (module docstring).

    Refuses -- arms nothing, ``view_state`` untouched -- when the selection
    touches no uv corner to measure a pivot from (:func:`selection_pivot`'s
    own declining case): there is nothing a caller could usefully drag.
    """
    pivot = selection_pivot(mesh, ids, view_state.selected_islands)
    if pivot is None:
        return False
    view_state.drag_mode = kind
    view_state.drag_base = mesh
    view_state.drag_islands = view_state.selected_islands
    view_state.drag_pivot = pivot
    view_state.drag_start = anchor
    view_state.drag_mark = doc.history.mark()
    return True


def update_live_transform(
    doc: Any, uid: int, view_state: UvPaneState, now: tuple[float, float]
) -> bool:
    """One frame of an armed live rotate/scale. -> whether the mesh changed.

    Recomputes the **absolute** angle or factor from the drag's own start
    (:func:`drag_angle`/:func:`drag_scale`, against ``view_state.drag_pivot``
    and ``drag_start``) and applies it to ``view_state.drag_base`` -- the
    mesh the gesture began with -- never to whatever ``doc`` holds this
    frame. That is the whole of what keeps this exact instead of drifting:
    see the module docstring for why re-reading the live mesh would not be.
    """
    if view_state.drag_mode == "rotate":
        degrees = drag_angle(view_state.drag_pivot, view_state.drag_start, now)
        return apply_rotate(doc, uid, view_state.drag_islands, degrees, base=view_state.drag_base)
    if view_state.drag_mode == "scale":
        factor = drag_scale(view_state.drag_pivot, view_state.drag_start, now)
        return apply_scale(doc, uid, view_state.drag_islands, factor, base=view_state.drag_base)
    return False


def commit_live_transform(doc: Any, view_state: UvPaneState) -> None:
    """End an armed live rotate/scale, folding however many frames it took
    into the one undo step the whole gesture is.

    ``controls.fold_undo``'s own "draw, fold, act" shape, closed by hand for
    the reason :func:`begin_live_transform` gives. Guarded on the mark
    actually having moved: a press-and-release that never called
    :func:`update_live_transform` (no motion at all) pushed nothing, and
    folding or labelling would either no-op or -- worse -- relabel whatever
    unrelated step already sat on top.
    """
    mark, kind = view_state.drag_mark, view_state.drag_mode
    view_state.drag_mode = ""
    history = doc.history
    if history.head != mark:
        history.collapse_since(mark)
        top = history.top
        if top is not None:
            top.label = kind.capitalize()
    view_state.drag_base = None
    view_state.drag_islands = frozenset()


def cancel_live_transform(doc: Any, view_state: UvPaneState) -> None:
    """Escape or a right-click mid-drag: put the mesh back exactly and leave
    nothing on the stack.

    Every frame the drag ran pushed its own step (nothing has folded them
    yet), so they are folded into one and then reversed and forgotten in a
    single move (``UndoStack.undo(doc, redoable=False)``) rather than a plain
    ``doc.undo()``: that would leave the abandoned gesture sitting on the
    *redo* stack, which is not "nothing left" if Ctrl+Shift+Z could still
    bring it back. Guarded on the mark having moved, the same reason
    :func:`commit_live_transform` is -- an armed-then-untouched cancel must
    not reverse whatever unrelated step already sat on top.
    """
    mark = view_state.drag_mark
    view_state.drag_mode = ""
    history = doc.history
    if history.head != mark:
        history.collapse_since(mark)
        history.undo(doc, redoable=False)
    view_state.drag_base = None
    view_state.drag_islands = frozenset()


# --- selection ---------------------------------------------------------------


def _selected_object(doc: Any) -> Any:
    """The one selected object, or ``None`` -- ``clay_props._selected``'s own
    rule, restated: a pane that edits geometry does not guess which of
    several selected objects it is supposed to act on."""
    if len(doc.selection) != 1:
        return None
    try:
        return doc.by_uid(next(iter(doc.selection)))
    except KeyError:
        return None


def _measurements(mesh: Any) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """``(overlap, stretch, refusal)`` for *mesh* -- ``uvtools.overlap_faces``/
    ``stretch``, computed once here rather than by the caller so both a
    refusal (a mesh past ``uvtools.MAX_OVERLAP_TRIANGLES``) and the ordinary
    answer share one call site. ``refusal`` is ``""`` on success."""
    try:
        overlap = uvtools.overlap_faces(mesh)
    except el.OpError as error:
        # Not a toast: this runs every frame the pane is open, on the frame
        # thread, for a mesh that is simply too dense for the grid check
        # (``uvtools.overlap_faces``'s own ceiling) -- the pane already says
        # so in words next to the canvas (see ``_canvas``); the log line is
        # for whoever is chasing why the tint never lights up on one object.
        log.debug("uv pane: overlap/stretch not shown (%s)", error)
        return None, None, str(error)
    stretch = uvtools.stretch(mesh)
    return overlap, stretch, ""


# --- drawing ------------------------------------------------------------------


def draw(ctx: Any) -> None:
    with widgets.section_blocks():
        _body(ctx)


def _body(ctx: Any) -> None:
    state = clay_mode.ensure(ctx)
    tab = state.active
    widgets.section("UV")
    manual_render.help_button(ctx, "clay-uv")
    if tab is None:
        return
    doc = tab.doc
    obj = _selected_object(doc)
    if obj is None:
        count = len(doc.selection)
        if count > 1:
            widgets.empty_state(
                icons.GRID, f"{count} objects selected", "Select one to see its UVs."
            )
        else:
            widgets.empty_state(icons.GRID, "Nothing selected", "Click an object in the viewport.")
        return
    if obj.mesh.uv is None:
        widgets.empty_state(
            icons.GRID,
            "No UVs",
            f"{obj.name!r} has no texture coordinates -- unwrap it first.",
        )
        return

    view_state: UvPaneState = tab.uv_view
    if view_state.for_uid != obj.uid:
        # A fresh object: island ids from a previous mesh mean nothing here.
        view_state.for_uid = obj.uid
        view_state.selected_islands = frozenset()
        view_state.drag_mode = ""

    _toolbar(ctx, doc, obj, view_state)
    _canvas(ctx, doc, obj, view_state)
    _legend()


def _toolbar(ctx: Any, doc: Any, obj: Any, view_state: UvPaneState) -> None:
    from imgui_bundle import imgui

    selected = view_state.selected_islands
    count = len(selected)
    # A live rotate/scale owns ``obj.mesh`` until it commits or cancels
    # (:func:`update_live_transform` writes it every frame); every other
    # button here reads or replaces that same mesh outright, so all of them
    # wait rather than compounding onto -- or stomping -- an in-progress
    # gesture's own preview.
    live = view_state.drag_mode in ("rotate", "scale")
    widgets.muted(
        "no islands boxed -- drag a box around one or more" if count == 0
        else f"{count} island(s) selected"
    )
    if count and controls.small_button(f"{icons.X} Clear selection##uvclear", enabled=not live):
        view_state.selected_islands = frozenset()
    if count and not live:
        # E and R also drive these live on the canvas -- Realmspinner's own
        # Clay-viewport tool letters (module docstring), not a reinvented
        # pair. Shown only once something is selected, the same gate the
        # fields and Apply buttons below already use.
        # ``muted_wrapped`` rather than ``muted``: both of these are sentences
        # in a pane that shares a narrow column, and ``muted`` cannot wrap one.
        widgets.muted_wrapped(
            "E rotate, R scale -- drag live; click commits, Esc/right-click cancels"
        )
    elif live:
        widgets.muted_wrapped(
            f"live {view_state.drag_mode} -- click commits, Esc/right-click cancels"
        )

    # The typed fields below are the *exact* path and stay independent of
    # any live drag in progress -- a modelling pane wants both (module
    # docstring's "left out" paragraph says why a live drag does not also
    # grow a typed-entry HUD of its own).
    widgets.field_label("rotate (deg)")
    changed, value = controls.input_float("##uv-rotate", view_state.pending_rotate, 1.0)
    if changed:
        view_state.pending_rotate = value
    imgui.same_line()
    if widgets.disabled_button("Apply##uvrotate", bool(selected) and not live):
        apply_rotate(doc, obj.uid, selected, view_state.pending_rotate)
        view_state.pending_rotate = 0.0

    widgets.field_label("scale")
    changed, value = controls.input_float("##uv-scale", view_state.pending_scale, 0.05)
    if changed:
        view_state.pending_scale = value
    imgui.same_line()
    if widgets.disabled_button(
        "Apply##uvscale", bool(selected) and view_state.pending_scale > 0.0 and not live
    ):
        apply_scale(doc, obj.uid, selected, view_state.pending_scale)
        view_state.pending_scale = 1.0

    if controls.small_button(f"{icons.SQUARE} Pack islands##uvpack", enabled=not live):
        apply_pack(doc, obj.uid)


def _canvas(ctx: Any, doc: Any, obj: Any, view_state: UvPaneState) -> None:
    from imgui_bundle import imgui

    avail = imgui.get_content_region_avail()
    # This pane's own dockable slot, not the whole window (``skeletons.clay``
    # gives it a SHARE-sized region beside the outliner and the properties
    # panel) -- ``avail`` is already bounded to it. A fixed reservation below
    # the canvas for the legend's five rows, floored so a very short slot
    # still gets a usable square to drag in; the legend itself simply draws
    # under whatever is left, and a slot too short for both scrolls, the same
    # as every other pane in this dock column.
    region = (max(float(avail.x), 1.0), max(float(avail.y) - sp(130), sp(200)))
    view = view_state.view
    size_px = (REF_PX, REF_PX)
    if not view.fitted:
        paintview.fit(view, size_px, region)

    origin_pt = imgui.get_cursor_screen_pos()
    origin = (origin_pt.x, origin_pt.y)
    imgui.invisible_button("clay-uv-canvas", region)
    # "draw, fold, act" (``controls.fold_undo``'s own docstring): the canvas
    # item is drawn (above), folded (here), and only afterwards does anything
    # below call ``doc.set_mesh``. Harmless on a frame that pushes nothing --
    # a box-select gesture marks and later collapses zero new steps. Covers
    # translate and box-select only -- a live rotate/scale opens and closes
    # its own gesture by hand (:func:`begin_live_transform` and friends,
    # below), since it has no mouse button held for this hook's own
    # activation/deactivation to key off of.
    controls.fold_undo(doc.history)
    hovered = imgui.is_item_hovered()
    mouse = imgui.get_mouse_pos()
    mesh = obj.mesh
    ids = uvtools.islands(mesh)
    uv_here = _to_uv(view, origin, mouse.x, mouse.y)

    if view_state.drag_mode in ("rotate", "scale"):
        _drive_live_transform(doc, obj.uid, view_state, uv_here, hovered)
    else:
        # E/R arm a live rotate/scale -- only while nothing else already
        # owns the mouse over this canvas and there is a selection to turn,
        # and never while a text field (the rotate/scale spinners just
        # above, in ``_toolbar``) is the one taking keystrokes, or typing
        # "-45" into the degrees box would also arm a rotate underneath it.
        if (
            hovered
            and not imgui.is_item_active()
            and view_state.selected_islands
            and not imgui.get_io().want_text_input
        ):
            if imgui.is_key_pressed(imgui.Key.e):
                begin_live_transform(doc, view_state, "rotate", mesh, ids, uv_here)
            elif imgui.is_key_pressed(imgui.Key.r):
                begin_live_transform(doc, view_state, "scale", mesh, ids, uv_here)

        if view_state.drag_mode not in ("rotate", "scale") and imgui.is_item_activated():
            hit = islands_in_rect(mesh, ids, (uv_here[0], uv_here[1], uv_here[0], uv_here[1]))
            view_state.drag_mode = "move" if hit & view_state.selected_islands else "box"
            view_state.drag_start = uv_here
            view_state.drag_last = uv_here

        if imgui.is_item_active() and imgui.is_mouse_dragging(0):
            if view_state.drag_mode == "move" and view_state.selected_islands:
                delta = (uv_here[0] - view_state.drag_last[0], uv_here[1] - view_state.drag_last[1])
                if delta != (0.0, 0.0):
                    apply_translate(doc, obj.uid, view_state.selected_islands, delta)
                view_state.drag_last = uv_here
            else:
                rect = (view_state.drag_start[0], view_state.drag_start[1], uv_here[0], uv_here[1])
                view_state.selected_islands = frozenset(islands_in_rect(mesh, ids, rect))
        elif imgui.is_item_deactivated():
            # Tidy rather than load-bearing -- only ``is_item_active()``
            # gates the block above, so a stale "move"/"box" left over from
            # the last drag is inert -- but leaving it set reads oddly now
            # that this same field also carries "rotate"/"scale", which
            # *are* meaningful between frames.
            view_state.drag_mode = ""

    if hovered:
        _handle_pan_zoom(view, origin, size_px, region, (mouse.x, mouse.y))

    draw_list = imgui.get_window_draw_list()
    draw_list.push_clip_rect(
        (origin[0], origin[1]), (origin[0] + region[0], origin[1] + region[1]), True
    )
    _backdrop(draw_list, view, origin)
    overlap, stretch, refusal = _measurements(mesh)
    covered = touched_islands(mesh, ids, doc.element_sel.get(obj.uid))
    _faces(draw_list, view, origin, mesh, ids, overlap, stretch)
    _edges(draw_list, view, origin, mesh, obj.seams)
    _island_outlines(draw_list, view, origin, mesh, ids, view_state.selected_islands | covered)
    draw_list.pop_clip_rect()
    if refusal:
        widgets.muted(f"overlap/stretch not shown: {refusal}")


def _drive_live_transform(
    doc: Any,
    uid: int,
    view_state: UvPaneState,
    uv_here: tuple[float, float],
    hovered: bool,
) -> None:
    """One frame's input for an armed live rotate/scale: a left click over
    the canvas commits, Escape (from anywhere) or a right-click over the
    canvas cancels, and anything else previews.

    Escape is not gated on ``hovered`` -- a drag already owns the gesture
    once armed, and the 3-D viewport's own Esc-cancels-a-live-drag
    convention (``clay_mode.handle_key``'s ``dragging`` check) does not
    require the pointer to still be over the viewport either. A click does
    stay hover-gated, so a press on the toolbar's own Apply button or the
    Clear-selection button above the canvas cannot also be read as this
    gesture's commit.

    Split out of :func:`_canvas` rather than inlined: the canvas body
    already reads as three unrelated gestures (pan/zoom, box/move, this
    one), and folding a fourth's input handling into that same block would
    make which branch a given frame falls into harder to see.
    """
    from imgui_bundle import imgui

    if imgui.is_key_pressed(imgui.Key.escape) or (hovered and imgui.is_mouse_clicked(1)):
        cancel_live_transform(doc, view_state)
        return
    if hovered and imgui.is_mouse_clicked(0):
        commit_live_transform(doc, view_state)
        return
    update_live_transform(doc, uid, view_state, uv_here)


def _handle_pan_zoom(
    view: Any,
    origin: tuple[float, float],
    size_px: tuple[float, float],
    region: tuple[float, float],
    mouse: tuple[float, float],
) -> None:
    from imgui_bundle import imgui

    if imgui.is_mouse_dragging(2):
        drag = imgui.get_mouse_drag_delta(2)
        imgui.reset_mouse_drag_delta(2)
        paintview.pan_by(view, size_px, region, drag.x, drag.y)
    io = imgui.get_io()
    if io.mouse_wheel or io.mouse_wheel_h:
        # The rule every 2-D canvas in this app shares (``paintview.wheel``'s
        # own docstring): the wheel zooms, Shift+wheel and a tilt wheel scroll
        # sideways. ``io.mouse_wheel`` is the *backend's* notch, scaled up
        # (``imgui_backend.WHEEL_SCALE``); every other wheel-driven canvas
        # divides it back out before handing a notch count to view maths.
        along = paintview.wheel(
            view, origin, mouse,
            io.mouse_wheel / imgui_backend.WHEEL_SCALE,
            io.mouse_wheel_h / imgui_backend.WHEEL_SCALE,
            shift=bool(io.key_shift),
        )
        if along:
            paintview.pan_by(view, size_px, region, paintview.scroll_step(region[0]) * along, 0.0)
    paintview.clamp_pan(view, size_px, region)


def _to_uv(view: Any, origin: tuple[float, float], sx: float, sy: float) -> tuple[float, float]:
    x, y = paintview.to_image(view, origin, sx, sy)
    return (x / REF_PX, y / REF_PX)


def _to_screen(view: Any, origin: tuple[float, float], u: float, v: float):
    return paintview.to_screen(view, origin, u * REF_PX, v * REF_PX)


def _backdrop(draw_list: Any, view: Any, origin: tuple[float, float]) -> None:
    from imgui_bundle import imgui

    p0 = _to_screen(view, origin, 0.0, 0.0)
    p1 = _to_screen(view, origin, 1.0, 1.0)
    draw_list.add_rect_filled(p0, p1, imgui.get_color_u32(theme.rgba(theme.ELEV_1)))
    draw_list.add_rect(p0, p1, imgui.get_color_u32(theme.rgba(theme.EDGE)))


#: A literal RGB for "compressed" (negative :func:`~.uvtools.stretch`) rather
#: than a theme role: this is a heat-map tint over geometry, not a piece of
#: chrome ``test_accessibility`` measures contrast for, and the palette has
#: no cool colour of its own -- ``ACCENT`` is already spent on the selection
#: outline drawn over the same faces, and reusing it here would make a
#: compressed, unselected face and a selected, ordinary one read the same.
_COMPRESSED_RGB = (0.35, 0.55, 0.95)


def _face_fill(
    face: int, overlap: np.ndarray | None, stretch: np.ndarray | None
) -> tuple[float, float, float, float] | None:
    """One face's tint, or ``None`` for "draw the plain neutral fill".

    Overlap wins outright -- it is the one condition that is simply wrong
    (a texture with two faces painting the same texel), where stretch is a
    matter of degree. Kept a pure function so the legend's own swatches and
    a mesh's face colours can never silently disagree about what a colour
    means.
    """
    if overlap is not None and overlap[face]:
        r, g, b = theme.rgba(theme.ERR)[:3]
        return (r, g, b, 0.55)
    if stretch is not None:
        value = float(stretch[face])
        weight = min(abs(value) / STRETCH_FULL, 1.0)
        if weight > 0.02:
            if value > 0:
                r, g, b = theme.rgba(theme.WARN)[:3]
            else:
                r, g, b = _COMPRESSED_RGB
            return (r, g, b, 0.15 + 0.45 * weight)
    return None


def _faces(
    draw_list: Any,
    view: Any,
    origin: tuple[float, float],
    mesh: Any,
    ids: np.ndarray,
    overlap: np.ndarray | None,
    stretch: np.ndarray | None,
) -> None:
    """Every face, filled -- plain where nothing is wrong with it, tinted
    where :func:`_face_fill` has something to say.

    Drawn from each face's own corners in order, as one convex polygon --
    every generator and unwrap in this package produces simple convex faces
    (quads and triangles); a hand-imported mesh with a concave face is drawn
    slightly wrong here rather than triangulated, which this pane accepts as
    a known, cosmetic-only limit rather than paying a triangulation pass on
    every frame for a shape blockout geometry does not produce.
    """
    from imgui_bundle import imgui

    if mesh.uv is None:
        return
    neutral = imgui.get_color_u32(theme.rgba(theme.ELEV_2, 0.18))
    starts = mesh.starts.astype("i8")
    uv = mesh.uv
    n_faces = len(starts) - 1
    for face in range(n_faces):
        lo, hi = starts[face], starts[face + 1]
        if hi - lo < 3:
            continue
        points = [_to_screen(view, origin, float(uv[c][0]), float(uv[c][1])) for c in range(lo, hi)]
        fill = _face_fill(face, overlap, stretch)
        colour = imgui.get_color_u32(fill) if fill is not None else neutral
        draw_list.add_convex_poly_filled(points, colour)


def _edges(draw_list: Any, view: Any, origin: tuple[float, float], mesh: Any, seams: Any) -> None:
    """Island boundaries, with the edges the object's own ``seams`` names
    drawn thicker and in a different colour.

    Boundary detection is :func:`~.uvtools.seams_from_uv` plus the raw uv
    layout's own cuts -- any edge whose loop does not close (a face edge with
    no matching reverse corner elsewhere in the mesh's uv, i.e. every edge
    :func:`~.uvtools.islands` itself would not walk across) -- rather than a
    second, pane-local seam derivation: the spec calls for exactly this
    function, read back against the object's own marked seams to decide
    which of those boundary edges are *also* an authored seam.
    """
    from imgui_bundle import imgui

    if mesh.uv is None:
        return
    boundary = imgui.get_color_u32(theme.rgba(theme.EDGE, 0.8))
    marked = imgui.get_color_u32(theme.rgba(theme.WARN))
    seam_set = {tuple(sorted((int(a), int(b)))) for a, b in (seams or ())}
    derived = uvtools.seams_from_uv(mesh)
    all_cuts = {tuple(sorted((int(a), int(b)))) for a, b in derived}
    # Every face edge is drawn once, from its own two uv corners -- an edge
    # shared by two faces whose uv agrees draws twice, harmlessly (the same
    # line on top of itself), which is cheaper than deriving a dedup set for
    # a pane that redraws every frame.
    starts = mesh.starts.astype("i8")
    loops = mesh.loops
    uv = mesh.uv
    n_faces = len(starts) - 1
    for face in range(n_faces):
        lo, hi = starts[face], starts[face + 1]
        count = hi - lo
        if count < 2:
            continue
        for k in range(count):
            c0, c1 = lo + k, lo + (k + 1) % count
            p0 = _to_screen(view, origin, float(uv[c0][0]), float(uv[c0][1]))
            p1 = _to_screen(view, origin, float(uv[c1][0]), float(uv[c1][1]))
            edge_verts = tuple(sorted((int(loops[c0]), int(loops[c1]))))
            if edge_verts in seam_set:
                draw_list.add_line(p0, p1, marked, 2.5)
            elif edge_verts in all_cuts:
                draw_list.add_line(p0, p1, boundary, 1.0)


def _island_outlines(
    draw_list: Any, view: Any, origin: tuple[float, float], mesh: Any, ids: np.ndarray, wanted: set
) -> None:
    """A thicker accent line around every edge of a highlighted island --
    boxed by the user, or touched by the current element selection."""
    from imgui_bundle import imgui

    if not wanted or mesh.uv is None:
        return
    accent = imgui.get_color_u32(theme.rgba(theme.ACCENT))
    starts = mesh.starts.astype("i8")
    uv = mesh.uv
    n_faces = len(starts) - 1
    for face in range(n_faces):
        if int(ids[face]) not in wanted:
            continue
        lo, hi = starts[face], starts[face + 1]
        count = hi - lo
        if count < 2:
            continue
        for k in range(count):
            c0, c1 = lo + k, lo + (k + 1) % count
            p0 = _to_screen(view, origin, float(uv[c0][0]), float(uv[c0][1]))
            p1 = _to_screen(view, origin, float(uv[c1][0]), float(uv[c1][1]))
            draw_list.add_line(p0, p1, accent, 2.0)


def _legend() -> None:
    from imgui_bundle import imgui

    widgets.field_label("legend")
    draw_list = imgui.get_window_draw_list()
    for label, colour in (
        ("overlapping", (*theme.rgba(theme.ERR)[:3], 1.0)),
        ("stretched", (*theme.rgba(theme.WARN)[:3], 1.0)),
        ("compressed", (*_COMPRESSED_RGB, 1.0)),
        ("selected / touched", (*theme.rgba(theme.ACCENT)[:3], 1.0)),
        ("seam", (*theme.rgba(theme.WARN)[:3], 1.0)),
    ):
        pos = imgui.get_cursor_screen_pos()
        side = sp(10)
        draw_list.add_rect_filled(
            (pos.x, pos.y + sp(3)), (pos.x + side, pos.y + sp(3) + side),
            imgui.get_color_u32(colour),
        )
        imgui.dummy((side, side))
        imgui.same_line()
        widgets.muted(label)
