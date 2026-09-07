"""Nine-slice inference and stretching: pure arithmetic, no editor.

Inker already models a nine-slice centre end to end (:mod:`.slices`,
:mod:`._doc_slices`) -- what this module adds is the two things nobody could
*do* with that data: find the centre from the art instead of dragging four
edges by eye, and turn it into pixels at a size other than the one it was
drawn at.

**Repeat, not resample.** Inker is a pixel-art editor, so the edges and the
middle of a stretched panel are tiled from the source pixels -- the same
strip drawn again, not a bilinear blend of it. A smooth stretch would put a
half-intensity seam exactly where a pixel artist drew a hard one, which is
the one thing a nine-patch must never do to a 1px border. Tiling also keeps
:func:`stretch` a pure gather (fancy indexing, no interpolation kernel), which
is what makes bit-identical corners a two-line assertion rather than a
tolerance.

**Exact equality, not a tolerance, decides "the same column".** A nine-patch's
stretchable middle is conventionally a flat fill -- Android's own guide draws
it that way -- so two columns that are merely *close* are not the same fill
and inferring a join from them would silently stretch through an edge highlight
or a one-shade gradient a pixel artist put there on purpose. A tolerance would
have to be chosen with no way to justify one value over another; "identical" has
no such knob and never guesses wrong in the direction that matters here, which
is claiming a repeat where the art does not have one.
"""

from __future__ import annotations

import numpy as np

from ._doc_slices import fit_center

__all__ = ["fit", "ninepatch", "stretch"]


def _matching_run(panel: np.ndarray, *, axis: int) -> tuple[int, int] | None:
    """The widest run of interior lines identical to their neighbour.

    ``axis=1`` walks columns, ``axis=0`` walks rows -- the same test either
    way, since a column *is* ``panel[:, i]`` and a row is ``panel[i, :]``.
    "Interior" excludes the first and last line on purpose: a run that reached
    the edge would leave no corner at all on that side, and a nine-slice
    without corners is a plain stretch wearing the wrong name. Returns the run
    as ``(start, end)``, exclusive, in the panel's own coordinates -- which is
    already "relative to the bounds origin", :mod:`.slices`' convention for a
    centre, so the caller hands it straight to :func:`~._doc_slices.fit_center`.
    """
    size = panel.shape[1] if axis == 1 else panel.shape[0]
    if size < 4:
        # Two 1px corners plus at least a 2px interior pair to compare -- below
        # that there is no room for a centre that still leaves both corners.
        return None
    best: tuple[int, int] | None = None
    run_start: int | None = None
    for i in range(1, size - 2):
        line = panel[:, i] if axis == 1 else panel[i, :]
        neighbour = panel[:, i + 1] if axis == 1 else panel[i + 1, :]
        if np.array_equal(line, neighbour):
            if run_start is None:
                run_start = i
            end = i + 2  # exclusive: the run covers columns run_start..i+1
            if best is None or (end - run_start) > (best[1] - best[0]):
                best = (run_start, end)
        else:
            run_start = None
    return best


def fit(
    plane: np.ndarray, bounds: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """Infer a slice's stretchable centre from its own pixels, or refuse.

    The standard nine-patch inference: the widest run of interior columns
    that repeat their neighbour, crossed with the same run of interior rows.
    ``None`` on a gradient, a photo, or anything else with no constant run --
    an honest refusal, because a *guessed* rectangle would sit there looking
    authoritative and export a UI panel that tears at its edges the first time
    a game stretches it.

    Clamped through :func:`~._doc_slices.fit_center`, the one function that
    already knows how a centre must sit inside its slice -- so this function
    does not re-derive that rule, and a change to it is felt here too.
    """
    x0, y0, x1, y1 = bounds
    panel = plane[y0:y1, x0:x1]
    columns = _matching_run(panel, axis=1)
    rows = _matching_run(panel, axis=0)
    if columns is None or rows is None:
        return None
    c0, c1 = columns
    r0, r1 = rows
    return fit_center((c0, r0, c1, r1), bounds)


def _axis_index(a: int, b: int, src_len: int, dst_len: int) -> np.ndarray:
    """A gather index for one axis: the two ends 1:1, the run between them
    tiled to fill whatever :func:`stretch` still owes that axis.

    ``[a, b)`` is the constant run inside ``[0, src_len)`` -- a slice's centre,
    on one axis. ``near = a`` and ``far = src_len - b`` are the two corners'
    thickness, copied unscaled; the space left over, ``dst_len - near - far``,
    is filled by repeating the run's own pixels with ``% (b - a)`` -- a tile,
    not a resample, for the module's own reason above. Raises when that space
    is positive but the run has zero width: there is nothing to repeat, and
    silently returning the corners with a gap between them would be a export
    that is short exactly the pixels it claimed to write.
    """
    near = a
    far = src_len - b
    mid_dst = dst_len - near - far
    mid_src = b - a
    if mid_dst > 0 and mid_src <= 0:
        raise ValueError(
            f"the centre run is 0px wide but {mid_dst}px of the target still "
            "needs filling from it -- there is nothing to repeat"
        )
    pieces = [np.arange(near)]
    if mid_dst > 0:
        pieces.append((np.arange(mid_dst) % mid_src) + a)
    pieces.append(np.arange(src_len - far, src_len))
    return np.concatenate(pieces).astype(np.intp)


def stretch(
    plane: np.ndarray,
    bounds: tuple[int, int, int, int],
    center: tuple[int, int, int, int],
    width: int,
    height: int,
) -> np.ndarray:
    """One slice, rebuilt at ``width`` x ``height`` around its centre.

    The corners are copied whole and never touched -- gathered straight out of
    the source, so they are bit-identical to the pixels a plain crop would
    have produced. The two edges repeat along the one axis they border, and
    the middle repeats on both, all through the same index-and-gather
    :func:`_axis_index` builds; there is no interpolation anywhere in this
    function, on purpose (see the module docstring).

    Refuses a target smaller than the four corners alone need -- ``width`` or
    ``height`` beneath what the fixed corners already sum to on that axis --
    because shrinking a nine-slice past its own corners has no honest answer:
    the corners cannot scale (that is the one thing "corners unscaled" rules
    out) and there is no pixel left to drop that is not one of them.
    """
    x0, y0, x1, y1 = bounds
    panel = plane[y0:y1, x0:x1]
    h, w = panel.shape[:2]
    cx0, cy0, cx1, cy1 = center
    width, height = int(width), int(height)
    fixed_w = cx0 + (w - cx1)
    fixed_h = cy0 + (h - cy1)
    if width < fixed_w or height < fixed_h:
        raise ValueError(
            f"target {width}x{height} is smaller than the fixed corners "
            f"({fixed_w}x{fixed_h}) -- a nine-slice cannot shrink past what "
            "its own corners need"
        )
    col_index = _axis_index(cx0, cx1, w, width)
    row_index = _axis_index(cy0, cy1, h, height)
    return panel[row_index][:, col_index]


def ninepatch(
    plane: np.ndarray, bounds: tuple[int, int, int, int], center: tuple[int, int, int, int]
) -> np.ndarray:
    """A slice as Android's own ``.9.png`` interchange format wants it.

    A 1px fully-transparent guide border around the source pixels at their own
    size -- Android stretches a nine-patch itself, at whatever size the widget
    ends up, so unlike :func:`stretch` this never resamples anything -- with
    opaque black marks on the top and left edges over the columns and rows
    ``center`` names as stretchable. That is the whole format this app can
    honestly write: the *padding* guide on the bottom and right edges is a
    second rectangle Android's tool lets an author draw independently of the
    stretch region, and a slice here carries only one rectangle. Marking the
    padding edges from the stretch region would assert a padding rule nobody
    drew, so they are left fully transparent -- "no padding constraint" is
    what Android reads that as, and it is also the honest answer.
    """
    x0, y0, x1, y1 = bounds
    panel = plane[y0:y1, x0:x1]
    h, w = panel.shape[:2]
    cx0, cy0, cx1, cy1 = center
    out = np.zeros((h + 2, w + 2, panel.shape[2]), dtype=panel.dtype)
    out[1:-1, 1:-1] = panel
    black = np.array([0, 0, 0, 255][: panel.shape[2]], dtype=panel.dtype)
    out[0, 1 + cx0 : 1 + cx1] = black
    out[1 + cy0 : 1 + cy1, 0] = black
    return out
