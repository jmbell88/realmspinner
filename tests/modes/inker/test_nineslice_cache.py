"""The nine-slice sidebar costs what an edit costs, not what a frame costs.

``Document.flatten`` copies the whole composite every call -- deliberately, its
callers may write to what they get back -- and the nine-slice panel asked for
one twice per frame: once in ``inker_ops._can_nineslice_fit`` to decide whether
the Auto-fit button is grey, and once in ``modes.inker.ui.panes.tools._nineslice_preview``
to build the swatches. On a 1024 canvas that is two multi-megabyte copies and a
full run scan every frame the sidebar is drawn with a slice selected, for two
answers that only move when the document does. The preview then stretched three
swatches before consulting the texture cache that already held them.

The counted ``Document`` below is the whole method here: every test asserts a
number of flattens or of stretches, because "the cache is used" is not a claim
and "the second frame flattens nothing" is.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from warlock.kernels.pixel import nineslice
from warlock.kernels.pixel.document import Document
from warlock.studio.modes.inker import ops as inker_ops
from warlock.studio.modes.inker.ui.panes import tools as inker_tools

CORNER = (10, 20, 30, 255)
EDGE_H = (40, 50, 60, 255)
EDGE_V = (70, 80, 90, 255)
MIDDLE = (100, 110, 120, 255)


class _CountedDoc(Document):
    """A document that says how many times it was flattened."""

    flattens = 0

    def flatten(self, *, matte: bool = True) -> np.ndarray:
        self.flattens += 1
        return super().flatten(matte=matte)


def _panel_doc(size: int = 12, border: int = 3) -> _CountedDoc:
    """A panel frame with a constant middle, so ``fit`` has an answer.

    ``test_nineslice.py``'s ``_panel`` painted onto a real document, because
    what is under test here is the *document* being flattened once rather than
    the inference itself.
    """
    doc = _CountedDoc.blank(size, size)
    plane = doc.stack[0].pixels
    for y in range(size):
        top, bottom = y < border, y >= size - border
        for x in range(size):
            left, right = x < border, x >= size - border
            if (top or bottom) and (left or right):
                plane[y, x] = CORNER
            elif top or bottom:
                plane[y, x] = EDGE_H
            elif left or right:
                plane[y, x] = EDGE_V
            else:
                plane[y, x] = MIDDLE
    doc.invalidate_all()
    doc.flattens = 0
    return doc


def _gradient_doc(size: int = 12) -> _CountedDoc:
    """No two adjacent columns or rows match -- ``fit``'s refusal case."""
    doc = _CountedDoc.blank(size, size)
    plane = doc.stack[0].pixels
    for y in range(size):
        for x in range(size):
            plane[y, x] = (x * 7 % 256, y * 11 % 256, (x + y) * 5 % 256, 255)
    doc.invalidate_all()
    doc.flattens = 0
    return doc


def _tab(doc: Document, uid: str = "tab-1") -> Any:
    return SimpleNamespace(uid=uid, doc=doc, frame_uid=None, busy=False)


@pytest.fixture(autouse=True)
def _fresh_slots():
    """Forget the module-level slots, so one test cannot answer another's ask."""
    inker_ops._flat_stamp = None
    inker_ops._flat_plane = None
    inker_ops._fit_stamp = None
    inker_ops._fit_center = None
    yield
    inker_ops._flat_stamp = None
    inker_ops._flat_plane = None


# --- the flatten ---------------------------------------------------------------


def test_two_frames_at_one_revision_flatten_the_document_once():
    doc = _panel_doc()
    tab = _tab(doc)
    first = inker_ops.nineslice_flat(tab)
    second = inker_ops.nineslice_flat(tab)
    assert doc.flattens == 1
    assert second is first


def test_an_edit_flattens_again():
    """``doc.rev`` is the stamp, so a stale plane is never served."""
    doc = _panel_doc()
    tab = _tab(doc)
    inker_ops.nineslice_flat(tab)
    doc.stack[0].pixels[0, 0] = (1, 2, 3, 255)
    doc.invalidate_all()
    plane = inker_ops.nineslice_flat(tab)
    assert doc.flattens == 2
    assert tuple(plane[0, 0]) == (1, 2, 3, 255)


def test_the_cached_plane_is_handed_out_read_only():
    """``flatten``'s copy is normally the caller's to scribble on; this one is
    shared between the button and the preview, and that licence does not
    survive being shared."""
    plane = inker_ops.nineslice_flat(_tab(_panel_doc()))
    assert not plane.flags.writeable


def test_a_second_tab_does_not_serve_the_first_tabs_plane():
    left, right = _panel_doc(), _gradient_doc()
    first = inker_ops.nineslice_flat(_tab(left, "tab-1"))
    second = inker_ops.nineslice_flat(_tab(right, "tab-2"))
    assert second is not first
    assert right.flattens == 1


# --- the fit -------------------------------------------------------------------


def test_asking_for_the_same_centre_twice_scans_once():
    doc = _panel_doc()
    tab = _tab(doc)
    bounds = (0, 0, 12, 12)
    assert inker_ops.nineslice_center(tab, bounds) is not None
    assert inker_ops.nineslice_center(tab, bounds) is not None
    assert doc.flattens == 1


def test_a_refusal_is_cached_as_firmly_as_an_answer():
    """A gradient has no centre to infer, and re-deriving that every frame is
    exactly as expensive as re-deriving a hit."""
    doc = _gradient_doc()
    tab = _tab(doc)
    assert inker_ops.nineslice_center(tab, (0, 0, 12, 12)) is None
    assert inker_ops.nineslice_center(tab, (0, 0, 12, 12)) is None
    assert doc.flattens == 1


def test_a_different_slice_is_a_different_question():
    doc = _panel_doc()
    tab = _tab(doc)
    inker_ops.nineslice_center(tab, (0, 0, 12, 12))
    inker_ops.nineslice_center(tab, (0, 0, 6, 6))
    assert doc.flattens == 1  # one plane, two scans
    assert inker_ops._fit_stamp[-1] == (0, 0, 6, 6)


def test_the_greyed_out_check_and_the_click_reach_the_same_answer():
    """Both go through ``nineslice_center``, so a button that says it can run
    cannot then refuse -- and the click adds no flatten to the check."""
    doc = _panel_doc()
    tab = _tab(doc)
    entry = doc.add_slice((0, 0, 12, 12))
    state = SimpleNamespace(slice_uid=entry.uid, transforming=False)
    doc.flattens = 0
    assert inker_ops._can_nineslice_fit(state, tab)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))
    assert inker_ops._run_nineslice_fit(ctx, tab)
    assert doc.flattens == 1
    assert doc.slice_by_uid(entry.uid).center is not None


# --- the preview swatches ------------------------------------------------------
#
# Drawn through a real imgui context, ``test_pattern_fill.py``'s ``ui`` fixture
# in miniature: nothing is rendered, only laid out, and no GL context means the
# swatches take ``_nineslice_preview``'s placeholder path -- which is exactly
# the path that used to stretch three times a frame for pixels it then threw
# away.


@pytest.fixture
def ui():
    from imgui_bundle import imgui

    from warlock.studio import theme

    previous = imgui.get_current_context()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600.0, 950.0)
    io.delta_time = 1.0 / 60.0
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)
    yield imgui
    imgui.destroy_context(ctx)
    if previous is not None:
        imgui.set_current_context(previous)


class _PreviewCtx:
    """Enough of the app context for ``_nineslice_preview``, with no viewer."""

    def __init__(self) -> None:
        self.viewer = None
        self.state = SimpleNamespace(preview={})


def _draw_preview(imgui: Any, ctx: _PreviewCtx, tab: Any, entry: Any) -> None:
    imgui.new_frame()
    imgui.set_next_window_size((520.0, 900.0))
    imgui.begin("##host")
    inker_tools._nineslice_preview(ctx, tab, entry, entry.at(tab.frame_uid))
    imgui.end()
    imgui.end_frame()


@pytest.fixture
def counted_stretch(monkeypatch):
    """Every ``stretch`` the preview runs, by target size."""
    calls: list[tuple[int, int]] = []
    real = nineslice.stretch

    def counted(plane, bounds, center, width, height):
        calls.append((width, height))
        return real(plane, bounds, center, width, height)

    monkeypatch.setattr(inker_tools.nineslice, "stretch", counted)
    return calls


def test_a_second_frame_of_the_preview_stretches_nothing(ui, counted_stretch):
    """The swatches were stretched before the cache was consulted, so three
    whole-slice stretches a frame were computed and then dropped by the stamp
    that had just matched."""
    doc = _panel_doc()
    tab = _tab(doc)
    entry = doc.add_slice((0, 0, 12, 12), center=(3, 3, 9, 9))
    ctx = _PreviewCtx()
    _draw_preview(ui, ctx, tab, entry)
    assert len(counted_stretch) == len(inker_tools._PREVIEW_FACTORS)
    doc.flattens = 0
    counted_stretch.clear()
    _draw_preview(ui, ctx, tab, entry)
    assert counted_stretch == []
    assert doc.flattens == 0


def test_an_edit_rebuilds_the_swatches(ui, counted_stretch):
    doc = _panel_doc()
    tab = _tab(doc)
    entry = doc.add_slice((0, 0, 12, 12), center=(3, 3, 9, 9))
    ctx = _PreviewCtx()
    _draw_preview(ui, ctx, tab, entry)
    counted_stretch.clear()
    doc.stack[0].pixels[5, 5] = (9, 9, 9, 255)
    doc.invalidate_all()
    _draw_preview(ui, ctx, tab, entry)
    assert len(counted_stretch) == len(inker_tools._PREVIEW_FACTORS)


def test_a_size_the_stretch_refuses_is_not_retried_every_frame(ui, monkeypatch):
    """The preview skips a size ``stretch`` will not draw rather than raising
    through a paint frame. That refusal is as worth remembering as a swatch is:
    without it the skipped sizes were the one thing still stretched every frame.

    ``stretch`` is made to refuse here rather than fed a slice that provokes it,
    because at these factors it cannot be provoked -- a centre is clamped inside
    its slice, so the 1x preview is never smaller than the corners. The branch
    is the defensive one the docstring promises, and this is what keeps it from
    becoming a per-frame cost the day some factor below 1.0 is added.
    """
    doc = _panel_doc()
    tab = _tab(doc)
    entry = doc.add_slice((0, 0, 12, 12), center=(3, 3, 9, 9))
    ctx = _PreviewCtx()
    calls: list[tuple[int, int]] = []

    def refuses(plane, bounds, center, width, height):
        calls.append((width, height))
        raise ValueError("target smaller than the fixed corners")

    monkeypatch.setattr(inker_tools.nineslice, "stretch", refuses)
    _draw_preview(ui, ctx, tab, entry)
    assert len(calls) == len(inker_tools._PREVIEW_FACTORS)
    calls.clear()
    _draw_preview(ui, ctx, tab, entry)
    assert calls == []
    assert set(ctx.state.preview.values()) >= {inker_tools._REFUSED}


def test_the_swatch_cache_lives_under_the_tabs_own_prefix(ui):
    """``inker_textures.release_doc`` frees a closed tab by prefix, and the two
    string markers must be swept with the textures rather than left behind."""
    doc = _panel_doc()
    tab = _tab(doc)
    entry = doc.add_slice((0, 0, 12, 12), center=(3, 3, 9, 9))
    ctx = _PreviewCtx()
    _draw_preview(ui, ctx, tab, entry)
    assert ctx.state.preview
    assert all(k.startswith(f"inker_tex:{tab.uid}:") for k in ctx.state.preview)
