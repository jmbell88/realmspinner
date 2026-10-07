"""What the viewport says without being asked: the hint line and the axis ball.

Two pieces of chrome that a modeller reads constantly and that Clay had neither
of. Both are **pure** -- numbers and strings, no imgui, no GL -- which is the
whole reason they are here rather than in the pane that draws them: "does edge
mode mention the loop shortcut" and "does the +X ball sit on the right when the
camera is at the front" are questions a headless test can ask, and they are
exactly the questions a screenshot cannot be made to fail on.

Nothing here imports outward *at module scope*. ``studio/modes/clay/ui/
hud.py`` draws it. The one exception is :func:`measure_line`'s own local
import of ``kernels.mesh.measure`` for its arithmetic (the 2026-09-26 audit's
clay-view-06: this docstring used to claim "nothing here imports outward"
outright, which a local import inside a function is still a real outward
edge from -- every other duck-typed ``getattr`` read in this module exists
*because* of that same constraint, so the claim being wrong was worth fixing
rather than restating). No import-pin test covers this module the way
``tests/_pure_packages.py`` covers the kernel packages and each mode's
``engine/`` (CLAUDE.md's own list) -- ``studio/viewport_hints.py`` was never
one of those, so this paragraph is the whole of what holds the line, and it
is a claim about *outward* imports specifically, not a promise this module
never imports anything at all (:mod:`numpy` is a dependency, not an outward
edge in the sense this file's pins mean).
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np

# --- the navigation widget ---------------------------------------------------

#: The six axis ends: the axis, its sign, the ``Camera.AXIS_VIEWS`` name a click
#: on it asks for, and the letter drawn in it.
#:
#: **The view names are the contract and are written out**, exactly as
#: ``AXIS_VIEWS`` writes out its angles: a ball says "put the camera on the +X
#: side" and must not have to know which way ``theta`` runs to say it.
#: ``test_every_ball_puts_the_camera_where_its_axis_points`` checks the six
#: against the camera rather than against this table, so a wrong pairing here is
#: a failure rather than a definition.
#:
#: Only the positive ends carry a letter. Blender's widget does the same and the
#: reason is legible rather than decorative: six labelled balls at 20 px are six
#: things to read, and the negative end of an axis is identified by being
#: opposite the one that is labelled.
AXIS_ENDS: tuple[tuple[int, int, str, str], ...] = (
    (0, 1, "right", "X"),
    (0, -1, "left", ""),
    (1, 1, "top", "Y"),
    (1, -1, "bottom", ""),
    (2, 1, "front", "Z"),
    (2, -1, "back", ""),
)


@dataclass(frozen=True)
class AxisBall:
    """One end of one axis, placed in the widget's own box.

    ``x``/``y`` are pixels from the box's top-left, so the caller adds its
    origin and nothing else. ``depth`` is positive toward the viewer, which is
    what the caller sorts on: the balls behind the centre are drawn first, so
    the near ones overlap them rather than the other way round.
    """

    view: str
    label: str
    x: float
    y: float
    depth: float
    positive: bool


def axis_layout(view_matrix: Any, size: float) -> list[AxisBall]:
    """The six balls, back to front, in a ``size`` x ``size`` box.

    ``view_matrix`` is the camera's own world-to-camera matrix, whose upper-left
    3x3 is the rotation: its rows are the camera's right, up and backward axes
    (``math3d.look_at`` writes ``s``, ``u``, ``-f``). So multiplying a world
    direction by it gives x to the right of the screen, y up it, and z *toward*
    the viewer -- and the only conversion left is that screen y grows downward.

    Sorted rather than left in table order because these overlap: at a
    front-on camera the +Z and -Z balls land on the same pixel, and which of
    them is on top is the whole of what tells the reader which way they are
    looking.
    """

    rotation = np.asarray(view_matrix, dtype="f8")[:3, :3]
    centre = size * 0.5
    # The balls sit inside the box rather than on its edge: a ball is drawn as a
    # disc of its own and one centred on the boundary would be half clipped.
    radius = size * 0.5 * 0.78
    out: list[AxisBall] = []
    for axis, sign, view, label in AXIS_ENDS:
        direction = np.zeros(3, dtype="f8")
        direction[axis] = float(sign)
        camera = rotation @ direction
        out.append(
            AxisBall(
                view=view,
                label=label,
                x=centre + float(camera[0]) * radius,
                y=centre - float(camera[1]) * radius,
                depth=float(camera[2]),
                positive=sign > 0,
            )
        )
    out.sort(key=lambda ball: ball.depth)
    return out


# --- the hint line -----------------------------------------------------------

#: What every mode says about picking, before the tool has its say. The element
#: modes get the selection verbs that only exist there; object mode gets the
#: two that only exist *outside* an element mode.
# The 2026-09-18 audit's clay-05: this named "Tab edit"/"Tab object", a
# binding nothing implements -- ``clay_mode.ELEMENT_KEYS``'s own comment says
# Tab is deliberately left to imgui's keyboard navigation, which would move
# focus out of the viewport as well as changing the mode. The 1/2/3/4 keys
# are what actually switch element mode (``clay_mode.ELEMENT_KEYS``), so the
# line now names those instead.
_PICK = {
    "object": "LMB select . Shift extend . 1/2/3 edit",
    "vertex": "LMB pick . drag marquee . L linked . 4 object",
    "edge": "LMB pick . drag marquee . L linked . 4 object",
    "face": "LMB pick . drag marquee . L linked . 4 object",
}

#: What the tool in hand adds. Keyed on the tool rather than folded into the
#: mode line because the two vary independently: Move in face mode and Move in
#: object mode drag the same way and select differently.
_TOOL = {
    "select": "",
    "move": "G move . drag an arrow",
    "rotate": "R rotate . drag a ring",
    "scale": "S scale . drag a handle",
}

#: Always true, and always last: the two mouse buttons that navigate. They are
#: the keys a newcomer to a 3D viewport asks about first and the ones a manual
#: is least likely to be open at.
_NAVIGATE = "Alt+LMB orbit . MMB / Shift+MMB pan . wheel zoom"


def hint(mode: str, tool: str) -> str:
    """One line of what the mouse and the keyboard do right now.

    Clay's viewport had no such line, and the cost was specific rather than
    general: **every selection verb the mode offers is invisible**. L for
    linked, a marquee, Shift and Ctrl to add and remove -- none of them is a
    button, so a user who has not read chapter 30 has no way to discover what
    the mode can do. (Alt+click loop select and Ctrl+plus grow went with the
    loop and ring queries in the picoCAD cut, and a hint naming either would
    be offer-then-refuse; tests/modes/clay/test_clay_hints.py pins that.)

    Nothing here speaks for a live G/R/S drag: that line is :func:`drag_readout`'s,
    chosen by ``hud.hint_line`` ahead of this one. A ``dragging`` branch used to
    live here with no caller, and its legend promised that a second press of the
    locked axis switches to a local space when ``DragInput`` has none -- the
    second press only clears the lock (the 2026-10-07 audit's clay-63) -- so it
    was deleted rather than kept in step with a line nobody draws.
    """

    parts = [_PICK.get(mode, _PICK["object"])]
    extra = _TOOL.get(tool, "")
    if extra:
        parts.append(extra)
    parts.append(_NAVIGATE)
    return " . ".join(parts)


#: The verb a drag kind reads as. Keyed rather than ``.capitalize()``d inline
#: at every call site, and ``"Drag"`` is what an unrecognised kind falls back
#: to, so a caller that has not yet worked out which of move/rotate/scale is
#: running still gets a word rather than an empty verb.
_VERBS = {"move": "Move", "rotate": "Rotate", "scale": "Scale"}


def drag_readout(kind: str, axis: str, space: str, amount: str) -> str:
    """The live line for a G/R/S drag under way: what it is, the axis lock and
    the frame that lock is read in, and the amount so far.

    Replaces the fixed key legend ``hint()`` used to draw for the whole of a
    drag: "X/Y/Z lock . type a number . Enter/LMB commit . Esc/RMB cancel"
    told a modeller *how* to constrain a drag but never what the constraint
    they had already applied amounted to -- so typing ``X`` then ``2`` gave no
    way to confirm the 2 without looking away from the model at a number
    nothing on screen showed. This is that number, on the line already read
    for exactly this reason.

    ``axis`` empty means the drag is unconstrained, in which case there is no
    lock to name a space for and none is shown -- a bare "(global)" next to no
    axis reads as a setting rather than as the fact that nothing is locked.
    """

    parts = [_VERBS.get(kind, "Drag")]
    if axis:
        parts.append(f"{axis.upper()} ({space})" if space else axis.upper())
    if amount:
        parts.append(amount)
    return " · ".join(parts)


def resolve_hint(*, measure: str, default: str) -> str:
    """Which of the hint line's two non-drag candidates wins: a live
    measurement, then the ordinary mode/tool legend. A live keyboard/gizmo drag
    is decided separately, by ``hud.hint_line``, ahead of both."""
    return measure or default


def keys_named(text: str) -> set[str]:
    """Every key or chord the line mentions, for the parity test.

    Crude on purpose: a token is a key if it is one of the shapes this app's
    shortcut sheet writes -- a capital letter, a chord with a ``+``, or one of
    the named keys. Anything cleverer would be a second parser to keep in
    agreement with the sheet, and the point of this function is to *catch*
    disagreement.

    **A single letter counts only when it is capital**, which is the app's own
    convention throughout (``G``, ``L``, ``Tab``) and not a nicety: "drag a
    ring" reads its article as a binding otherwise, and the parity test then
    fails demanding that Clay implement the ``A`` key.

    **A single digit always counts**, capital or not being meaningless for a
    number: the element modes are bound on ``1``/``2``/``3``/``4``
    (``clay_mode.ELEMENT_KEYS``), and the 2026-09-18 audit's clay-05 found
    this function blind to them -- ``"/" `` group and bare-digit tokens both
    read as English words, so a line naming an unbound digit passed the
    parity test the same way "Tab edit" did.
    """

    words = {
        token.strip(".,")
        for chunk in text.split(" . ")
        for token in chunk.split()
    }
    named = {"Tab", "Enter", "Esc", "LMB", "MMB", "RMB", "wheel"}
    out = set()
    for word in words:
        if word in named or "+" in word or (
            len(word) == 1 and (word.isupper() or word.isdigit())
        ):
            out.add(word)
        elif "/" in word and all(
            len(part) == 1 and (part.isupper() or part.isdigit())
            for part in word.split("/")
            if part
        ):
            out.update(part for part in word.split("/") if part)
    return out


def angle_of(dx: float, dy: float) -> float:
    """The screen angle of a drag, in radians. Here because the hint line and
    the drag machinery both want one and neither should own it."""

    return math.atan2(float(dy), float(dx))


# --- the statistics overlay ---------------------------------------------------


def stats(doc: Any) -> str:
    """What the document holds, and how much of it is selected. One line.

    Blender's statistics overlay, and the reason it earns a place on a viewport
    that already has a hint line: **every number here was otherwise unavailable
    anywhere in Clay**. The outliner counts objects and nothing counted
    vertices, edges, faces or triangles -- so "is this mesh 500 triangles or
    50,000" was a question the app could not answer about the thing on screen,
    which is the question that decides whether a game asset is finished.

    Selected counts are shown only when there *is* a selection, and only for the
    element mode in hand: a face count while vertices are being picked is a
    number about a selection the user does not have.

    Pure, and derived per call rather than cached. It walks the meshes, which
    is O(objects) in numpy shape reads -- the arrays are not touched, only
    their lengths -- so there is nothing to invalidate and nothing to go stale.
    """

    objects = [obj for obj in doc.objects if getattr(obj, "visible", True)]
    verts = edges = faces = tris = 0
    for obj in objects:
        mesh = obj.mesh
        count = _faces_of(mesh)
        loops = getattr(mesh, "loops", None)
        # ``or ()`` is wrong on a numpy array -- truthiness of one with more
        # than one element raises -- so the absence is tested with ``is None``.
        corners = 0 if loops is None else int(len(loops))
        # A polygon of n corners fans into n-2 triangles, so the triangle count
        # is the corner count less twice the face count. The *edge* count is
        # not the corner count: a cube has 24 corners and 12 edges, because
        # every edge is shared by two faces, and reporting 24 would be a number
        # a reader can check against a cube and find wrong.
        mesh_tris = max(0, corners - 2 * count)
        verts += int(len(mesh.positions))
        faces += count
        edges += _unique_edges(mesh)
        tris += mesh_tris
    parts = [
        f"{len(objects)} object{'' if len(objects) == 1 else 's'}",
        f"{verts:,} verts",
        f"{edges:,} edges",
        f"{faces:,} faces",
        f"{tris:,} tris",
    ]
    picked = _selected(doc)
    if picked:
        parts.append(picked)
    return "  ".join(parts)


#: Unique-edge counts, keyed on the mesh object and pinning it -- weakly.
#:
#: Keyed on the ``Mesh`` itself rather than on an id or a revision, which is
#: the rule for any cache about a mesh: a ``Mesh`` is immutable and
#: every op replaces it, so "this count is still about what is on screen" is
#: exactly ``mesh is measured``. An ``id()`` would be recycled by the allocator
#: onto a different mesh and silently report the last edit's edges.
#:
#: A ``WeakKeyDictionary`` rather than the bounded, wholesale-cleared ``dict``
#: this used to be (the 2026-09-26 audit's clay-view-03): capped at 64 entries
#: and dropped *entirely* the moment a 65th mesh arrived, so a scene of more
#: than 64 objects with Stats on cleared the whole cache partway through every
#: single frame's loop over its own objects -- every mesh missed, every
#: frame, 386 ms/call measured at 70 objects, which is worse than never
#: caching at all. Keying per mesh and letting the object's own lifetime
#: govern eviction (a replaced mesh is unreachable and its entry disappears
#: with it) needs no cap and cannot thrash on object count.
_EDGE_CACHE: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()


def _unique_edges(mesh: Any) -> int:
    """How many distinct edges a mesh has. Memoised on the mesh.

    The pair per face corner, sorted within the pair so ``(a, b)`` and
    ``(b, a)`` are one edge, then counted distinct. O(L log L) and run once per
    mesh rather than once per frame -- a statistics overlay that rebuilt an
    adjacency sixty times a second would cost more than everything it reports.
    """
    hit = _EDGE_CACHE.get(mesh)
    if hit is not None:
        return hit
    loops = getattr(mesh, "loops", None)
    starts = getattr(mesh, "starts", None)
    if loops is None or starts is None or len(loops) == 0:
        return 0
    loops = np.asarray(loops)
    starts = np.asarray(starts)
    # The next corner within each face, which is the corner after it except at
    # a face's last corner, where it wraps to that face's first.
    nxt = np.arange(1, len(loops) + 1, dtype="i8")
    nxt[starts[1:] - 1] = starts[:-1]
    pairs = np.stack([loops, loops[nxt]], axis=1)
    pairs = np.sort(pairs, axis=1)
    count = int(len(np.unique(pairs, axis=0)))
    _EDGE_CACHE[mesh] = count
    return count


def _faces_of(mesh: Any) -> int:
    """How many faces a mesh has, off its CSR offsets.

    ``starts`` is ``(F+1,)`` -- one offset per face plus the terminator -- which
    is the shape every op in ``clay/`` reads it as.
    """
    starts = getattr(mesh, "starts", None)
    if starts is None:
        return 0
    return max(0, int(len(starts)) - 1)


# --- the measure readout (tranche 3: scene structure) -----------------------


def _element_objects(doc: Any) -> list[Any]:
    """The objects whose element selections a drag would move:
    ``selection._element_pickable``'s one eligibility.

    This readout and the selected count read ``doc.element_sel`` for every
    object, so without it a hidden object that still held a selection would put
    its distance, area and "N selected" on the HUD while the gizmo and the drag
    ignored it. Local import, for the reason
    :func:`_compute_measure_line`'s is.
    """
    from ..kernels.mesh.selection import _element_pickable

    return [obj for obj in doc.objects if _element_pickable(obj)]


def _selected_vertices(doc: Any) -> list[tuple[Any, int, np.ndarray]]:
    """``(object, vertex index, world position)`` for every selected vertex,
    document order, one object at a time -- capped at four: :func:`measure_line`
    only has an answer for exactly two or exactly three, so a caller past that
    is already "".

    The index rides along because ``ElementSel.verts`` is sorted: the click
    order is gone, so the "middle" of three is the middle *number*, and the
    angle readout has to name which vertex that is (the 2026-10-03 audit's
    clay-103) or it reads as a wrong angle about an apex the user cannot see.

    ``doc.world_matrix`` is duck-typed with ``getattr``: this module imports
    nothing outward at module scope, so a document with no such method
    measures in local space rather than raising.
    """
    world_of = getattr(doc, "world_matrix", None)
    points: list[tuple[Any, int, np.ndarray]] = []
    sels = getattr(doc, "element_sel", {}) or {}
    for obj in _element_objects(doc):
        sel = sels.get(obj.uid)
        verts = None if sel is None else getattr(sel, "verts", None)
        if verts is None or not len(verts):
            continue
        world = None if world_of is None else world_of(obj.uid)
        positions = np.asarray(obj.mesh.positions, dtype="f8")
        for idx in verts:
            homo = np.append(positions[int(idx)], 1.0)
            point = homo[:3] if world is None else (np.asarray(world, dtype="f8") @ homo)[:3]
            points.append((obj, int(idx), point))
            if len(points) > 4:
                return points
    return points


def _selected_vertex_points(doc: Any) -> list[np.ndarray]:
    """:func:`_selected_vertices`' world positions alone."""
    return [point for _obj, _idx, point in _selected_vertices(doc)]


#: *doc* -> ``(rev, line)``, weak so a closed tab's own document takes its
#: entry with it -- ``outliner._HIERARCHY_CACHE``'s own shape, restated for a
#: string rather than a pair of maps.
_MEASURE_CACHE: weakref.WeakKeyDictionary[Any, tuple[int, str]] = weakref.WeakKeyDictionary()


def measure_line(doc: Any) -> str:
    """A live readout for the four selection shapes :mod:`~.kernels.mesh.
    measure` answers: two selected vertices, three, a face selection, or a
    plain object selection -- distance, angle, area and volume in turn.
    ``""`` for anything else (an edge selection, an empty one, more than
    three vertices...), which :func:`~.clay.ui.hud.hint_line` reads as "show
    the ordinary hint instead."

    World-space throughout, through ``doc.world_matrix`` -- see
    :func:`_selected_vertex_points` for why that is a ``getattr`` rather than
    a named import.

    Memoised on ``doc.rev`` (the 2026-09-26 audit's clay-panes-03/clay-
    view-04): ``hud.hint_line`` called this once a frame with no gate at all,
    including every frame a selection sat still doing nothing -- 0.47 s at
    262k faces, 24 ms at 20k, measured, entirely for a volume nothing had
    asked to see recomputed. ``rev`` already covers a changed *selection* and
    not only a changed mesh: ``ClayDoc.select``/``set_element_sel``/
    ``set_element_mode``/``clear_element_sel`` each call ``touch()`` (their
    own docstrings), so there is one key, not two. A live gizmo drag is not a
    counter-case, but not because ``rev`` stands still: ``_drag_gizmo`` and
    ``_drag_keyboard`` call ``doc.touch()`` every frame, so ``rev`` does
    advance during a drag (the 2026-10-03 audit's clay-123). The safety is
    that ``hint_line`` reads ``view.gizmo_drag`` first and calls
    ``drag_readout`` instead of this function for as long as a drag is live
    -- this function is never asked during one, so the memo is never read
    through a moving mesh. ``getattr`` rather than a
    named attribute, matching every other duck-typed read in this module: a
    caller with no ``.rev`` at all (this module's own module-scope imports
    stay inward, so nothing here may assume the real ``ClayDoc``) simply
    measures fresh every time, which is exactly today's behaviour for it.
    """
    rev = getattr(doc, "rev", None)
    if rev is None:
        return _compute_measure_line(doc)
    cached = _MEASURE_CACHE.get(doc)
    if cached is not None and cached[0] == rev:
        return cached[1]
    line = _compute_measure_line(doc)
    _MEASURE_CACHE[doc] = (rev, line)
    return line


def _compute_measure_line(doc: Any) -> str:
    """:func:`measure_line`'s actual arithmetic, unmemoised. Split out so the
    cache wrapper never has to duplicate one of this function's several
    early returns.
    """
    from ..kernels.mesh import measure as bm_measure

    mode = getattr(doc, "element_mode", "object")
    if mode == "vertex":
        picked = _selected_vertices(doc)
        points = [point for _obj, _idx, point in picked]
        if len(points) == 2:
            return f"distance  {bm_measure.distance(points[0], points[1]):.4f} m"
        if len(points) == 3:
            # Named, not implied: the apex is the middle-numbered vertex (the
            # selection is sorted, clay-103), and across objects its owner's
            # name too, since two objects can both have a "vertex 1".
            apex_obj, apex_idx, _pt = picked[1]
            owners = {id(obj) for obj, _i, _p in picked}
            where = f"vertex {apex_idx}"
            if len(owners) > 1:
                where = f"{apex_obj.name} {where}"
            return (
                f"angle  {bm_measure.angle(points[0], points[1], points[2]):.2f}° "
                f"at {where}"
            )
        return ""
    if mode == "face":
        world_of = getattr(doc, "world_matrix", None)
        sels = getattr(doc, "element_sel", {}) or {}
        total = 0.0
        any_sel = False
        for obj in _element_objects(doc):
            sel = sels.get(obj.uid)
            faces = None if sel is None else getattr(sel, "faces", None)
            if faces is None or not len(faces):
                continue
            any_sel = True
            world = None if world_of is None else world_of(obj.uid)
            total += bm_measure.face_area(obj.mesh, faces, world)
        return f"area  {total:.4f} m²" if any_sel else ""
    if mode == "object":
        selection = getattr(doc, "selection", None) or ()
        if not selection:
            return ""
        world_of = getattr(doc, "world_matrix", None)
        total = 0.0
        any_open = False
        for uid in selection:
            try:
                obj = doc.by_uid(uid)
            except (KeyError, AttributeError):
                continue
            mesh = obj.mesh
            world = None if world_of is None else world_of(uid)
            # The 2026-10-03 audit's clay-25 follow-up: ``clay_measure volume``
            # answers ``null, closed: false`` for an open mesh because the
            # divergence sum over an open surface depends on where the object
            # sits, but this line kept printing that number as a volume. One
            # open mesh in the selection makes the sum meaningless, so the
            # line says so instead of adding it in.
            volume = bm_measure.volume_if_closed(mesh, world)
            if volume is None:
                any_open = True
            else:
                total += volume
        if any_open:
            return "volume  -- (open mesh)"
        return f"volume  {total:.4f} m³"
    return ""


def _selected(doc: Any) -> str:
    """The selected count, in the mode it is a count of."""

    mode = getattr(doc, "element_mode", "object")
    if mode == "object":
        count = len(getattr(doc, "selection", ()) or ())
        return f"{count} selected" if count else ""
    total = 0
    sels = getattr(doc, "element_sel", {}) or {}
    for obj in _element_objects(doc):
        sel = sels.get(obj.uid)
        if sel is not None:
            total += sel.count(mode)
    if not total:
        return ""
    word = {"vertex": "vert", "edge": "edge", "face": "face"}.get(mode, mode)
    return f"{total:,} {word}{'' if total == 1 else 's'} selected"
