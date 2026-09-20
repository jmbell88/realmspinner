"""A zoomable, pannable 2D viewport: zoom/pan state, framing, wheel input.

Promoted out of Inker (``modes/inker/state.py:858-1476`` before this move,
dev/RESTRUCTURE.md's P5) because it was never Inker's: Plotter and Packwright
imported ``PaintView`` and its view maths from day one (see
``plotter_state.py``'s and ``packwright_state.py``'s own docstrings, "the
view type is imported, not reimplemented"), which made four sibling-mode
edges in ``tests/test_layering.py`` -- a mode reaching into another mode's
package for something that was never mode-specific in the first place, the
one shape the layer rule bans outright. Nothing about a zoom level, a pan
offset or a wheel notch is about pixels; it is what every 2-D canvas in this
app needs, so it lives at the layer every mode may import: the shell (L4).

**Tool settings belong to the app; the view belongs to the document.** A tab
remembers where it was scrolled to (``PaintView`` is per-document), while the
brush and its options live on the editor's own state -- that split is
Inker's, described in ``modes/inker/state.py``, and unaffected by this move.

**Bounds are parameters, not module constants, on purpose.** :data:`MIN_ZOOM`
and :data:`MAX_ZOOM` are the generic floor and ceiling every function here
defaults to; Inker keeps its own wider pair (``modes.inker.state.INKER_MIN_ZOOM``
/ ``INKER_MAX_ZOOM``) and passes it explicitly at every call site, which is
what let that ceiling change (10x to 64x, see that module's own note) without
touching Plotter or Packwright at all.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

MIN_ZOOM = 0.05
MAX_ZOOM = 32.0
ZOOM_STEP = 1.15

# The wheel's granularity, in percent. Additive and *snapped* rather than the
# multiplicative ``ZOOM_STEP``: a 15% ratio step from 100% lands on 115, 132.25,
# 152.09 -- the status bar reads a different arbitrary number every notch, and
# there is no way back to a round one. See :func:`zoom_step`.
ZOOM_PERCENT_STEP = 5

#: Where the wheel stops being additive and starts walking :data:`ZOOM_LADDER`.
#:
#: 5% notches are right around 1:1, where the difference between 100% and 105%
#: is a real one -- and absurd above it: from 1x to a high ceiling is
#: thousands of notches, which is not a control, it is a workout. Aseprite's
#: wheel goes multiplicative at the top for the same reason.
FINE_ZOOM_MAX = 8.0

#: The keyboard's zoom ladder, as whole scales.
#:
#: **Integer above 1:1, halving below it.** This is the pixel-art rule and the
#: reason the keyboard does not simply reuse the wheel's 5% steps: at 135% a
#: source pixel is 1.35 screen pixels, so the renderer draws some of them one
#: pixel wide and some two, and a checkerboard dither comes out as bands. Every
#: rung here maps one source pixel onto a whole number of screen pixels (or a
#: whole number of source pixels onto one), which is the only family of zooms at
#: which pixel art is being shown rather than resampled.
#:
#: Inker's own zoom combo (``ZOOM_PRESETS``) is a different table for a
#: different question -- see that module's own note on why the two must not be
#: "synced".
ZOOM_LADDER = (
    0.05, 0.1, 0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
    12.0, 16.0, 24.0, 32.0, 48.0, 64.0,
)


def zoom_rung(
    zoom: float, direction: int, ladder: tuple[float, ...] = ZOOM_LADDER
) -> float:
    """The next rung of ``ladder`` in ``direction``. -> the new scale.

    Strictly past the current zoom rather than nearest-then-step, so a view
    sitting between two rungs at 135% zooms *out* to 100% and *in* to 200%
    instead of snapping sideways to 100% on a press labelled "in". At either end
    the ladder holds, which is the same answer the wheel gives at its bounds.

    ``ladder`` is a parameter because the Plotter tileset palette has its own
    (``plotter_state.PALETTE_ZOOM_LADDER``) and the *stepping rule* above is the
    part worth sharing -- particularly "strictly past", which is what makes a
    fit-derived zoom that sits between two rungs step sanely. Copying the scan
    over there would have been two implementations of one sentence.
    """
    if direction > 0:
        return next((rung for rung in ladder if rung > zoom + 1e-6), ladder[-1])
    return next((rung for rung in reversed(ladder) if rung < zoom - 1e-6), ladder[0])


# --- the view ---------------------------------------------------------------


#: The quarter turns the view offers, in degrees. **Quarter turns only, and
#: that is a decision rather than a first instalment.** A free-angle canvas
#: rotation makes every overlay in the pane a rotated quantity: the grid stops
#: being two families of axis-aligned lines, the marquee preview stops being a
#: rect, the transform box's handles stop being squares, and each of those has
#: to be re-derived and re-tested. A quarter turn maps an axis-aligned image
#: rectangle onto an axis-aligned *screen* rectangle, so every one of those
#: stays exactly what it was -- and it delivers what canvas rotation is
#: actually reached for: turning the page to draw a curve, and checking a
#: drawing mirrored. The engine never sees any of it; pixels are untouched.
ROTATIONS = (0, 90, 180, 270)


@dataclass
class PaintView:
    """Where the canvas sits in its pane. Per document, so a tab switch does
    not lose your place."""

    zoom: float = 1.0
    pan: tuple[float, float] = (0.0, 0.0)
    # Whether the view has been framed yet. False asks the canvas to fit on the
    # next frame it draws, which is the only moment it knows how big the pane
    # is -- the state layer never does.
    fitted: bool = False
    # A zoom to snap to on the next frame, for the same reason: "100%, centred"
    # needs the pane's size, and a keypress does not have it.
    pending_zoom: float | None = None
    # A zoom-ladder press waiting for a frame, +1 in or -1 out. Same reason
    # again, and a *direction* rather than a scale because the anchor decides
    # what the step means: over the canvas it holds the pixel under the cursor,
    # off it the middle of the pane, and neither is known to a key handler.
    pending_zoom_rung: int = 0
    # Display only, both of them: see ROTATIONS. ``rotation`` is clockwise on
    # screen in degrees; ``flipped`` mirrors left-to-right *after* it, which is
    # the order a physical sheet of paper does the two in.
    rotation: int = 0
    flipped: bool = False
    # Where the last paint stroke finished, for Shift-click's line. On the
    # *view* rather than on the session because it belongs to the drawing: a
    # tab switch and back should continue the line you were drawing, and a
    # session-wide field would carry one document's last point into another.
    last_paint: tuple[float, float] | None = None
    #: Wheel travel that has not yet added up to a whole notch. A trackpad and
    #: a high-resolution wheel deliver fractions, and ``zoom_step`` added
    #: ``5% x notches`` straight onto the lattice -- so a 0.3 notch took the
    #: view to 101.5% and every later notch carried that fraction forever, at
    #: which point the status bar read 106.5, 111.5, ... and the combo showed a
    #: preset the view was not at. Carried rather than rounded away, so a slow
    #: trackpad scroll still zooms.
    zoom_carry: float = 0.0


@contextmanager
def viewing(tab: Any, index: int):
    """Make ``tab.views[index]`` the view ``tab.view`` answers with, and yield it.

    The split canvas draws two views in one frame, and roughly forty places
    inside the paint and input helpers ask the tab for "the" view rather than
    being handed one. Threading a parameter through all of them would touch
    every gesture in the editor -- every one a place a drawing bug can hide --
    to express something with one true answer at any instant: whose turn it is.

    Restored in a ``finally`` and *to whatever it was* rather than to None, so
    a nested use (a pane's body calling something that scopes a view again)
    unwinds correctly instead of unsetting the outer one.
    """
    before = tab.active_view
    tab.active_view = index
    try:
        yield tab.views[index]
    finally:
        tab.active_view = before


def duplicate_view(tab: Any) -> bool:
    """Open a second pane onto this document. Aseprite's View ▸ Duplicate View.

    The new view is a *copy* of the current one rather than a fresh default:
    the pane appears showing what the user was already looking at, and they
    zoom the one they want changed. A second view arriving fitted-to-window
    would throw away the framing they had and make the command feel like it
    reset something.

    Two panes and no more. Aseprite opens windows and can have many; this is
    one centre pane divided by width, and a third column of a 1600px window is
    not a view of a drawing, it is a stripe.
    """
    if len(tab.views) > 1:
        return False
    tab.views.append(replace(tab.view))
    tab.focus = len(tab.views) - 1
    return True


def close_duplicate_view(tab: Any) -> bool:
    """Go back to one pane, keeping the view that has the user's attention.

    Keeping the *focused* one rather than always the first is the whole
    courtesy of the command: the pane you were working in is the one you meant
    to keep, and closing the split should not also throw away its zoom.
    """
    if len(tab.views) < 2:
        return False
    tab.views = [tab.view]
    tab.focus = 0
    tab.active_view = None
    return True


def clamp_zoom(zoom: float, lo: float = MIN_ZOOM, hi: float = MAX_ZOOM) -> float:
    """Hold a zoom inside its bounds.

    The bounds are arguments rather than the module constants, because the
    three consumers of this view math want different ones -- see
    ``INKER_MIN_ZOOM``. Defaulting to the globals is what keeps Plotter's and
    Packwright's call sites unchanged.
    """

    return max(lo, min(hi, float(zoom)))


def _quarter(view: PaintView) -> int:
    """ROTATIONS' index for the view's rotation, in one spelling.

    A rotation somehow off the quarter lattice reads as 0 -- the answer
    ``basis`` has always given -- rather than raising out of ``index()``.
    Only code can produce one today, which is exactly why the guard lives
    here: ``rotate_view`` restated the lookup without it, so the two answered
    the same bad value differently, one silently and one with a ValueError.
    """
    rotation = int(view.rotation)
    return ROTATIONS.index(rotation % 360) if rotation % 90 == 0 else 0


def basis(view: PaintView) -> tuple[tuple[float, float], tuple[float, float]]:
    """The view's 2x2 orientation, as rows. Orthonormal, determinant +-1.

    Kept separate from the zoom because it is exactly the part that preserves
    *length*: the marching ants measure arc length in canvas space and dash
    along it, so a transform that scaled would have to be threaded through that
    arithmetic, and one that only turns does not.
    """
    quarter = _quarter(view)
    # (x, y) -> (-y, x) is one clockwise quarter turn on a screen whose y grows
    # downward, which is the direction the button's icon points.
    rows = (
        ((1.0, 0.0), (0.0, 1.0)),
        ((0.0, -1.0), (1.0, 0.0)),
        ((-1.0, 0.0), (0.0, -1.0)),
        ((0.0, 1.0), (-1.0, 0.0)),
    )[quarter]
    if view.flipped:
        # After the turn, and on screen x: mirroring in image space instead
        # would put the flip under the rotation and make "flip" mean two
        # different things depending on which way the page was turned.
        rows = ((-rows[0][0], -rows[0][1]), rows[1])
    return rows


def _oriented(view: PaintView, x: float, y: float) -> tuple[float, float]:
    (a, b), (c, d) = basis(view)
    return (a * x + b * y, c * x + d * y)


def view_extent(
    view: PaintView, size: tuple[int, int]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The canvas's oriented box at zoom 1, as ``(low, high)``.

    A quarter turn puts part of the canvas at negative coordinates, so the
    framing functions cannot assume the corner is at the origin any more --
    which is the whole of what rotation costs the layout, and it is contained
    here.
    """
    width, height = float(size[0]), float(size[1])
    corners = [
        _oriented(view, x, y) for x in (0.0, width) for y in (0.0, height)
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return (min(xs), min(ys)), (max(xs), max(ys))


#: What one wheel notch scrolls, as a fraction of the pane along that axis.
#: An eighth means three notches move about a third of what you can see, which
#: is the order imgui's own five-lines-per-notch reaches on a list -- and being
#: a *fraction* it scales down with the narrow panes of a split view instead of
#: throwing them across the page.
SCROLL_FRACTION = 0.125

#: ...with a floor in screen pixels, so a very short pane still moves at all.
#: It only bites under about 200 px.
SCROLL_MIN = 24.0


def scroll_step(span: float) -> float:
    """How far one wheel notch scrolls a pane ``span`` pixels along.

    **Screen pixels, not image pixels.** A scroll moves the *view*, so it has
    to cover the same distance on screen at every zoom; an image-space step
    would crawl at 800% and throw the page across the pane at 5%.
    """
    return max(float(span) * SCROLL_FRACTION, SCROLL_MIN)


def page_box(
    view: PaintView, size: tuple[int, int]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Where the page actually is, in *pane* coordinates, as ``(low, high)``.

    :func:`view_extent` scaled by the zoom and offset by the pan -- the one
    place those three are combined, so everything below is arithmetic on a
    rectangle and none of it has to know the view can be turned or mirrored.
    """
    (lo_x, lo_y), (hi_x, hi_y) = view_extent(view, size)
    zoom, (pan_x, pan_y) = view.zoom, view.pan
    return (
        (lo_x * zoom + pan_x, lo_y * zoom + pan_y),
        (hi_x * zoom + pan_x, hi_y * zoom + pan_y),
    )


def pan_limits(
    view: PaintView, size: tuple[int, int], region: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """How far the pan may travel on each axis, as ``((lo_x, hi_x), (lo_y, hi_y))``.

    Aseprite's rule, which pads the scrollable area by half a viewport rather
    than by nothing. The padding is ``max(pane / 2, pane - span)``, and the
    boundary between its two halves is **half the pane**, not the whole of it:

    * a page longer than **half** the pane may be pushed until one of its edges
      reaches the *middle* of the pane and no further, so at least half a pane
      of drawing is always on screen -- and, the reason the padding exists at
      all, every corner of a page too big to see at once can still be dragged
      into the middle to be worked on;
    * a page shorter than half the pane may go anywhere inside it and never
      partly outside it, because for those the second term is the larger one.

    A page between the two -- longer than half the pane but shorter than it --
    is in the first case, so it *may* hang off an edge. That is deliberate and
    is what lets you push a nearly-pane-sized drawing aside to see what is
    under it.

    ``hi - lo`` is the page's span in the first case and the pane's leftover in
    the second, so it is never negative: the interval cannot invert and
    :func:`clamp_pan` is total at every zoom and every document size.

    Rotation and flip cost this nothing -- :func:`view_extent` is already the
    *oriented* box, so at a quarter turn the horizontal limit is derived from
    the document's height, which is also what the rulers already do.
    """
    low, high = page_box(view, size)
    limits = []
    for axis in (0, 1):
        # Back to a pan-independent box: the limits are about where the pan may
        # put the page, so the pan it currently has must come out first.
        near = low[axis] - view.pan[axis]
        far = high[axis] - view.pan[axis]
        pane = float(region[axis])
        span = far - near
        margin = max(pane * 0.5, pane - span)
        limits.append((pane - margin - far, margin - near))
    return (limits[0], limits[1])


def clamp_pan(
    view: PaintView, size: tuple[int, int], region: tuple[float, float]
) -> None:
    """Pull the pan inside :func:`pan_limits`. Idempotent, and the identity on
    anything :func:`_place` produced.

    Called once a frame by the pane rather than from the three writers, because
    two of them -- ``_place`` and ``_anchor`` -- are shared with Plotter and
    Packwright, and Plotter's canvas has no bound *by design* (an infinite map
    has nothing to bound it against). The guarantee this buys is therefore the
    stronger one anyway: **no frame of Inker's canvas is drawn from an
    unclamped pan**, which also covers the cause no write-site clamp could see
    -- a pane that shrinks under a pan that was legal a frame ago.
    """
    (lo_x, hi_x), (lo_y, hi_y) = pan_limits(view, size, region)
    view.pan = (
        min(max(view.pan[0], lo_x), hi_x),
        min(max(view.pan[1], lo_y), hi_y),
    )


def pan_by(
    view: PaintView,
    size: tuple[int, int],
    region: tuple[float, float],
    dx: float,
    dy: float,
) -> None:
    """Move the view by a screen-space delta, bounded. The wheel and both drags."""
    view.pan = (view.pan[0] + float(dx), view.pan[1] + float(dy))
    clamp_pan(view, size, region)


def scroll_thumb(
    view: PaintView,
    size: tuple[int, int],
    region: tuple[float, float],
    axis: int,
    *,
    min_length: float = 0.0,
) -> tuple[float, float]:
    """One scrollbar's thumb as ``(offset, length)``, both fractions of the track.

    **Derived from :func:`pan_limits` rather than computed beside it**, and
    that is the whole design: a thumb worked out from page-versus-pane overlap
    and a pan clamped by a second rule are two rules that will disagree, and
    the disagreement shows up as a thumb that springs back after a drag. Here
    ``offset`` is 0 exactly at ``hi`` and ``offset + length`` is 1 exactly at
    ``lo``, so the thumb cannot be dragged anywhere the clamp refuses.

    Two consequences worth knowing. At **Fit** the thumb is *half* the track,
    not all of it, because the rule still allows half a pane of travel each way
    -- and it is never longer than two thirds. ``min_length`` keeps a very long
    page's thumb grabbable; it is applied inside the mapping, so an end-to-end
    drag still lands exactly on the limit.
    """
    lo, hi = pan_limits(view, size, region)[axis]
    pane = float(region[axis])
    travel = hi - lo
    total = pane + travel
    if total <= 0.0:
        return (0.0, 1.0)
    length = min(1.0, max(min_length, pane / total))
    offset = 0.0 if travel <= 0.0 else (hi - view.pan[axis]) / total
    return (min(max(offset, 0.0), max(0.0, 1.0 - length)), length)


def scroll_drag(
    view: PaintView,
    size: tuple[int, int],
    region: tuple[float, float],
    axis: int,
    delta: float,
    travel_px: float,
) -> None:
    """Move the view by a thumb drag of ``delta`` px along ``travel_px`` of free
    track. The thumb goes one way and the page the other, which is what a
    scrollbar means."""
    if travel_px <= 0.0:
        return
    lo, hi = pan_limits(view, size, region)[axis]
    shift = -float(delta) / float(travel_px) * (hi - lo)
    pan = list(view.pan)
    pan[axis] += shift
    view.pan = (pan[0], pan[1])
    clamp_pan(view, size, region)


def scroll_thumb_to(
    view: PaintView,
    size: tuple[int, int],
    region: tuple[float, float],
    axis: int,
    centre: float,
) -> None:
    """Put the thumb's *centre* at ``centre`` (a fraction of the track), which
    is what a click on the bare track means -- Plotter's minimap gesture."""
    lo, hi = pan_limits(view, size, region)[axis]
    pane = float(region[axis])
    travel = hi - lo
    total = pane + travel
    if total <= 0.0 or travel <= 0.0:
        return
    length = pane / total
    offset = min(max(float(centre) - length / 2.0, 0.0), max(0.0, 1.0 - length))
    pan = list(view.pan)
    pan[axis] = hi - offset * total
    view.pan = (pan[0], pan[1])
    clamp_pan(view, size, region)


def _place(
    view: PaintView,
    size: tuple[int, int],
    region: tuple[float, float],
    zoom: float,
    *,
    lo: float = MIN_ZOOM,
    hi: float = MAX_ZOOM,
) -> None:
    """Set the zoom and centre the oriented canvas in the region."""
    (lo_x, lo_y), (hi_x, hi_y) = view_extent(view, size)
    view.zoom = clamp_zoom(zoom, lo, hi)
    view.pan = (
        (region[0] - (hi_x - lo_x) * view.zoom) * 0.5 - lo_x * view.zoom,
        (region[1] - (hi_y - lo_y) * view.zoom) * 0.5 - lo_y * view.zoom,
    )
    view.fitted = True


def fit(
    view: PaintView,
    size: tuple[int, int],
    region: tuple[float, float],
    *,
    lo: float = MIN_ZOOM,
    hi: float = MAX_ZOOM,
) -> None:
    """Scale to show the whole document, centred.

    Under a floor (``lo``) a document too large to fit is centred at the floor
    and overflows the pane. That is the stated cost of having a floor at all;
    the alternative is a "fit" that is not one, which is worse in the case the
    floor exists for.
    """
    (lo_x, lo_y), (hi_x, hi_y) = view_extent(view, size)
    zoom = min(region[0] / max(hi_x - lo_x, 1.0), region[1] / max(hi_y - lo_y, 1.0))
    _place(view, size, region, zoom, lo=lo, hi=hi)


def centre(
    view: PaintView,
    size: tuple[int, int],
    region: tuple[float, float],
    zoom: float,
    *,
    lo: float = MIN_ZOOM,
    hi: float = MAX_ZOOM,
) -> None:
    """Set an explicit zoom and re-centre -- what Ctrl+1 (100%) does."""
    _place(view, size, region, zoom, lo=lo, hi=hi)


def rotate_view(view: PaintView, quarter_turns: int = 1) -> None:
    """Turn the page. The zoom is kept and the canvas re-centred next frame.

    Re-centred rather than left where it was, because a quarter turn about the
    view's origin sends the canvas off the pane -- and through ``pending_zoom``
    rather than by clearing ``fitted``, which would also re-scale and throw away
    a zoom the user chose.
    """
    view.rotation = ROTATIONS[(_quarter(view) + int(quarter_turns)) % 4]
    view.pending_zoom = view.zoom


def flip_view(view: PaintView) -> None:
    """Mirror the view left-to-right. The classic check on a drawing, and the
    reason this is a *view* flag rather than an edit: nothing about the document
    changes, so there is nothing to undo and nothing to save."""
    view.flipped = not view.flipped
    view.pending_zoom = view.zoom


def to_image(view: PaintView, origin: tuple[float, float], sx: float, sy: float):
    """Screen -> image coordinates, as floats.

    Floats, not ints: the brush walks sub-pixel positions, and rounding here
    would quantise every stroke to the zoom level it was drawn at.

    The orientation is inverted by **transposing** its matrix, which is exact
    rather than approximate: the basis is orthonormal, so its transpose is its
    inverse whichever of the eight it happens to be.
    """
    u = (sx - origin[0] - view.pan[0]) / view.zoom
    v = (sy - origin[1] - view.pan[1]) / view.zoom
    (a, b), (c, d) = basis(view)
    return (a * u + c * v, b * u + d * v)


def to_screen(view: PaintView, origin: tuple[float, float], x: float, y: float):
    u, v = _oriented(view, x, y)
    return (
        origin[0] + view.pan[0] + u * view.zoom,
        origin[1] + view.pan[1] + v * view.zoom,
    )


def _anchor(
    view: PaintView, origin: tuple[float, float], mouse: tuple[float, float], after: float
) -> None:
    """Move to ``after``, keeping whatever pixel is under the cursor there.

    Factored out rather than written twice: :func:`zoom_about` and
    :func:`zoom_step` differ only in how they pick the new zoom, and a second
    copy of this pan correction is the classic way for one of the two routes
    into the same view to start drifting.
    """
    before = view.zoom
    if after == before:
        return
    local = (mouse[0] - origin[0], mouse[1] - origin[1])
    ratio = after / before
    view.zoom = after
    view.pan = (
        local[0] - (local[0] - view.pan[0]) * ratio,
        local[1] - (local[1] - view.pan[1]) * ratio,
    )


def zoom_about(
    view: PaintView,
    origin: tuple[float, float],
    mouse: tuple[float, float],
    steps: float,
    *,
    lo: float = MIN_ZOOM,
    hi: float = MAX_ZOOM,
) -> None:
    """Zoom keeping whatever pixel is under the cursor under the cursor.

    Multiplicative: a fixed *ratio* per step, which is the right shape for a
    keyboard zoom spanning three orders of magnitude. The wheel uses
    :func:`zoom_step` instead, through :func:`wheel` -- in every canvas, not
    only Inker's, since 2026-09-05.
    """
    _anchor(view, origin, mouse, clamp_zoom(view.zoom * (ZOOM_STEP**steps), lo, hi))


def zoom_step(
    view: PaintView,
    origin: tuple[float, float],
    mouse: tuple[float, float],
    notches: float,
    *,
    lo: float = MIN_ZOOM,
    hi: float = MAX_ZOOM,
) -> None:
    """One wheel notch: +-``ZOOM_PERCENT_STEP`` percent, snapped to the grid.

    ``notches`` may be fractional -- a trackpad and a high-resolution wheel
    both deliver fractions -- and a fraction of a step is *carried*, never
    applied: applying it left the lattice, permanently.

    Snapped *first*, so a zoom arrived at by fitting (an arbitrary 83.4%) joins
    the lattice on the first notch instead of carrying its fraction forever --
    which is what makes the status bar read 85, 90, 95 rather than 88.4, 93.4.
    The rounding is what a user coming from any paint program expects and what
    the multiplicative ratio cannot give: 100% is reachable from either side.
    """
    # Whole notches only; the remainder is kept for the next event. See
    # ``PaintView.zoom_carry``.
    travel = view.zoom_carry + float(notches)
    notches = float(int(travel))
    view.zoom_carry = travel - notches
    if not notches:
        return
    if (view.zoom >= FINE_ZOOM_MAX and notches > 0) or (
        view.zoom > FINE_ZOOM_MAX and notches < 0
    ):
        # One notch, one rung. See :data:`FINE_ZOOM_MAX`: 5% of 1x is a
        # meaningful step and 5% of 64x is a twentieth of a source pixel.
        target = view.zoom
        for _ in range(max(1, min(int(abs(notches)), len(ZOOM_LADDER)))):
            moved = zoom_rung(target, 1 if notches > 0 else -1)
            if moved == target:
                break
            target = moved
    else:
        percent = round(view.zoom * 100 / ZOOM_PERCENT_STEP) * ZOOM_PERCENT_STEP
        percent += ZOOM_PERCENT_STEP * notches
        target = percent / 100.0
    _anchor(view, origin, mouse, clamp_zoom(target, lo, hi))


def wheel(
    view: PaintView,
    origin: tuple[float, float],
    mouse: tuple[float, float],
    notches: float,
    sideways: float = 0.0,
    *,
    shift: bool = False,
    lo: float = MIN_ZOOM,
    hi: float = MAX_ZOOM,
) -> float:
    """The one wheel rule for every 2-D canvas. -> sideways notches to pan by.

    **The wheel zooms; Shift+wheel and a tilt wheel scroll sideways.** Decided
    on 2026-09-05 when the three canvases were found disagreeing: Inker
    scrolled on the wheel and zoomed on Ctrl+wheel (Aseprite's default, since
    2026-08-31), while Plotter and Packwright zoomed on the bare wheel through
    :func:`zoom_about` -- multiplicatively, on the backend-halved count, so
    they never landed on a round percentage. Same hand gesture, two results,
    and no comment on either side saying why. Zoom won because two of three
    did it and both neighbours' manuals promised it; the *lattice* rule won
    because it is the one that makes 100% reachable.

    ``notches`` and ``sideways`` are **physical** notches -- the caller has
    already divided the backend's ``WHEEL_SCALE`` back out. A tilt wheel's
    sign is the opposite of Shift+wheel's: imgui reports a positive
    ``mouse_wheel_h`` as "towards the right", where a positive ``mouse_wheel``
    is "away from the user", which moves the page the other way. Written out
    rather than folded into one term. What is returned is how many pane-widths'
    worth of :func:`scroll_step` the caller pans by, so a pane with bounds
    (Inker) and one without (Packwright) can each apply it their own way.
    """
    along = (notches if shift else 0.0) - sideways
    if not shift and notches:
        zoom_step(view, origin, mouse, notches, lo=lo, hi=hi)
    return along


def zoom_ladder_step(
    view: PaintView,
    origin: tuple[float, float],
    mouse: tuple[float, float],
    direction: int,
    *,
    lo: float = MIN_ZOOM,
    hi: float = MAX_ZOOM,
) -> None:
    """One press of zoom in or out: the next whole scale, cursor held.

    Through the same ``_anchor`` the wheel and ``zoom_about`` use, for the
    reason that helper exists at all -- a third copy of the pan correction is a
    third route into the view that can start to drift from the other two.
    """
    _anchor(view, origin, mouse, clamp_zoom(zoom_rung(view.zoom, direction), lo, hi))


