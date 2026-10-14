"""Regression tests for the 2026-10-14 audit's F16 batch, the Inker pane and op
half: inker-15 (the pixel clipboard across tabs), inker-37 (the image brush's
cursor outline) and inker-16 (the cursor's mirrored twin).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.pixel import brush
from realmspinner.studio.modes.inker import ops as inker_ops
from realmspinner.studio.modes.inker import state as inker_state
from realmspinner.studio.modes.inker.ui.panes import canvas as inker_canvas

# --- inker-15 ---------------------------------------------------------------------------


def test_copy_in_one_tab_pastes_in_another():
    """The pixel clipboard was one object per document and Copy never wrote the OS
    clipboard, so Ctrl+V in a second tab said "There is nothing on the clipboard."
    though ``copy_merged``'s hint promises it moves a picture between documents."""
    state = inker_state.InkerState()
    a = inker_state.InkerDoc(doc=inker.Document.blank(8, 8), uid="a", title="A")
    b = inker_state.InkerDoc(doc=inker.Document.blank(8, 8), uid="b", title="B")
    state.add(a)
    state.add(b)
    a.doc.stack.active.pixels[0:4, 0:4] = (200, 20, 20, 255)
    a.doc.invalidate_all()
    a.doc.select_all()
    said: list[str] = []
    state.say = lambda text, **_: said.append(text)
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=state), toast=lambda *args, **_: said.append(args)
    )

    state.activate("a")
    assert inker_ops.run(ctx, inker_ops.get("copy"))
    state.activate("b")
    inker_ops.run(ctx, inker_ops.get("paste"))
    assert b.doc.floating is not None, said


def test_a_second_copy_in_the_other_tab_replaces_the_shared_clipboard():
    state = inker_state.InkerState()
    a = inker_state.InkerDoc(doc=inker.Document.blank(8, 8), uid="a", title="A")
    b = inker_state.InkerDoc(doc=inker.Document.blank(8, 8), uid="b", title="B")
    state.add(a)
    state.add(b)
    state.say = lambda *args, **_: None
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state), toast=lambda *args, **_: None)
    a.doc.stack.active.pixels[0:2, 0:2] = (1, 2, 3, 255)
    a.doc.invalidate_all()
    a.doc.select_all()
    state.activate("a")
    inker_ops.run(ctx, inker_ops.get("copy"))
    b.doc.stack.active.pixels[0:3, 0:3] = (9, 9, 9, 255)
    b.doc.invalidate_all()
    b.doc.select_all()
    state.activate("b")
    inker_ops.run(ctx, inker_ops.get("copy"))
    assert a.doc.clipboard is b.doc.clipboard
    state.activate("a")
    inker_ops.run(ctx, inker_ops.get("paste"))
    assert a.doc.floating is not None
    assert tuple(a.doc.floating.pixels[0, 0][:3]) == (9, 9, 9)


# --- inker-37 ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tip_w", "tip_h", "align"),
    [
        (3, 3, "free"),
        (4, 2, "free"),
        (4, 4, "free"),
        (2, 6, "free"),
        (4, 4, "aligned"),
        (3, 5, "aligned"),
    ],
)
def test_the_image_brush_footprint_is_the_rectangle_the_dab_paints_for_even_and_aligned_tips(
    tip_w, tip_h, align
):
    """The canvas's cursor outline against the pixels ``_image_dab`` really paints."""
    tip = np.zeros((tip_h, tip_w, 4), np.uint8)
    tip[...] = (255, 0, 0, 255)
    stamp = brush.Stamp(tip)
    target = np.zeros((40, 40, 4), np.uint8)
    stroke = brush.StrokeState(
        layer_uid=1,
        size=(40, 40),
        before=target.copy(),
        colour=(0, 0, 0, 255),
        stamp=stamp,
        stamp_align=align,
    )
    stroke.begin((10.5, 10.5), target)
    ys, xs = np.nonzero(target[..., 3])
    painted = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    state = SimpleNamespace(
        tool="stamp", brush_size=9, nib="soft", stamp_align=align, tip_for=lambda tool: stamp
    )
    assert inker_canvas._footprint_box(state, None, (10, 10)) == painted


# --- inker-16 ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [10, 11])
def test_the_cursor_twin_is_the_pixel_the_stroke_paints(width):
    """The twin outline asks the engine where a dab on the pixel is reflected, fed the
    pixel's centre; the stroke itself is fed the same centre by the mouse."""
    state = SimpleNamespace(symmetry="x", symmetry_axis=None, radial_count=6)
    tab = SimpleNamespace(doc=SimpleNamespace(size=(width, 3)))
    for i in range(width):
        twins = inker_canvas._mirror_twins(state, tab, (i, 1))
        doc = inker.Document.from_pixels(np.zeros((3, width, 4), np.uint8))
        doc.begin_stroke((i + 0.5, 1.5), (255, 0, 0, 255), size=1, nib="pixel", symmetry="x")
        doc.end_stroke()
        painted = set(np.nonzero(doc.composite[..., 3])[1].tolist())
        assert {i} | {cell[0] for cell in twins} == painted
