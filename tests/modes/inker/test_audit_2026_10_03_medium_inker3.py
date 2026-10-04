"""The 2026-10-03 audit's Medium findings inker-41 .. inker-50 (Inker paint,
filters, sheet, playback, palette IO, canvas).

Each test's name is the claim, and each fails against the code as it stood
before the fix.
"""

from __future__ import annotations

import time
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.pixel import brush, filters
from realmspinner.kernels.pixel.sheetin import document_from_sheet
from realmspinner.studio import state as state_mod
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker import ops as inker_ops
from realmspinner.studio.modes.inker import palette_io as inker_palette_io
from realmspinner.studio.modes.inker import playback as inker_playback
from realmspinner.studio.modes.inker import sheet as inker_sheet
from realmspinner.studio.modes.inker import state as inker_state
from realmspinner.studio.modes.inker.ui import ants

RED = (255, 0, 0, 255)


# --- fixtures ----------------------------------------------------------------


def _ctx_for(state, settings=None):
    app = SimpleNamespace(inker=state, toasts=[])
    app.toast = MethodType(state_mod.AppState.toast, app)
    app.toast_once = MethodType(state_mod.AppState.toast_once, app)
    stored: dict = {}
    if settings is None:
        settings = SimpleNamespace(
            get=lambda key: stored.get(key, {}), set=lambda key, value: stored.update({key: value})
        )
    return SimpleNamespace(state=app, toast=app.toast, settings=settings)


def _session(doc=None, *, frames=1, layers=1):
    if doc is None:
        doc = inker.Document.blank(8, 8)
        for i in range(1, layers):
            doc.add_layer(f"L{i}")
        if frames > 1:
            doc.ensure_animation()
            for _ in range(frames - 1):
                doc.add_frame()
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")
    state = inker_state.InkerState()
    state.add(tab)
    return _ctx_for(state), state, tab


def _op(name):
    return next(op for op in inker_ops.OPS if op.name == name)


def _toasts(ctx):
    return [toast.text for toast in getattr(ctx.state, "toasts", [])]


# --- inker-41 ----------------------------------------------------------------


def test_nothing_commits_the_float_while_a_transform_is_open():
    """Next/Previous frame (the op and the raw ``step_frame`` the timeline pane
    calls), and the three landings, each used to commit the free transform's
    floating buffer and leave ``state.transforming`` true with nothing floating."""

    ctx, state, tab = _session(frames=3)
    tab.doc.set_current_frame(1)
    state.transforming = True

    for name in ("next_frame", "prev_frame"):
        assert inker_ops.run(ctx, _op(name)) is False
    assert inker_ops.reason_for(_op("next_frame"), state, tab)
    inker_playback.step_frame(ctx, 1, tab)
    inker_playback.step_frame(ctx, -1, tab)
    assert tab.doc.anim.current == 1, "the playhead must not move under an open transform"

    # The landings refuse the way they refuse a busy tab: a toast, no write.
    toasts_before = len(_toasts(ctx))
    inker_mode._done_tileset_import(
        ctx,
        state,
        SimpleNamespace(key=f"inker-tileset-import:{tab.uid}", result={"tileset": object()}),
    )
    assert len(_toasts(ctx)) == toasts_before + 1
    assert tab.doc.tilesets == [] or len(tab.doc.tilesets) == 0

    assert inker_palette_io.index_to(ctx, tab, [(0, 0, 0, 255), (255, 255, 255, 255)]) is False
    assert tab.doc.palette in (None, [], ())

    assert inker_sheet.land_merge(
        ctx,
        state,
        SimpleNamespace(key=f"inker-merge:{tab.uid}", result={"cells": [], "sheet": "S"}),
    ) is False


# --- inker-42 ----------------------------------------------------------------


def _sheet_session():
    cell = 8
    atlas = np.zeros((cell, cell * 2, 4), dtype=np.uint8)
    atlas[..., 3] = 255
    cells = [{"x": i * cell, "y": 0, "w": cell, "h": cell} for i in range(2)]
    anim = {"tags": [{"name": "walk_front", "start": 0, "end": 1, "loop": True}], "frames": []}
    doc = document_from_sheet(atlas, cells, anim, source={"job": "J", "sheet": "S"})
    doc.history.clear()
    return _session(doc)


def _indexed_animated():
    doc = inker.Document.blank(8, 8)
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    doc.convert_to_indexed([(0, 0, 0, 0), RED, (0, 255, 0, 255)], "nearest", transparent=0)
    doc.ensure_animation()
    doc.add_frame(link=True)
    doc.set_current_frame(0)
    return _session(doc)


#: Rows that push a history step (or move the playhead) and that were gated on
#: their own predicate alone.
_HISTORY_ROWS = {
    "reselect": lambda: _reselect_session(),
    "select_used_colours": lambda: _indexed_animated(),
    "select_unused_colours": lambda: _indexed_animated(),
    "frame_palette": lambda: _indexed_animated(),
    "clear_frame_palette": lambda: _own_palette_session(),
    "sheet_keep_edit": lambda: _sheet_session(),
    "next_frame": lambda: _session(frames=3),
    "prev_frame": lambda: _session(frames=3),
}


def _reselect_session():
    ctx, state, tab = _session()
    tab.doc.select_all()
    tab.doc.deselect()
    return ctx, state, tab


def _own_palette_session():
    ctx, state, tab = _indexed_animated()
    tab.doc.set_frame_palette(list(tab.doc.palette))
    return ctx, state, tab


def _busy_variants(state, tab):
    """Each way a tab is not ready, as a setter and an undo."""

    def saving(on):
        tab.saving = on

    def playing(on):
        tab.playing = on

    def transforming(on):
        state.transforming = on

    return {"saving": saving, "playing": playing, "transforming": transforming}


@pytest.mark.parametrize("name", sorted(_HISTORY_ROWS))
def test_every_history_pushing_op_is_greyed_while_the_tab_is_busy(name):
    op = _op(name)
    for variant in ("saving", "playing", "transforming"):
        ctx, state, tab = _HISTORY_ROWS[name]()
        assert op.enabled(state, tab), f"{name} must be live on a ready tab (fixture)"
        _busy_variants(state, tab)[variant](True)
        assert not op.enabled(state, tab), f"{name} stayed enabled while {variant}"
        assert inker_ops.reason_for(op, state, tab) == inker_ops.BUSY


def test_play_is_greyed_while_saving_or_transforming_but_stays_live_to_stop():
    op = _op("play")
    ctx, state, tab = _session(frames=3)
    assert op.enabled(state, tab)
    tab.saving = True
    assert not op.enabled(state, tab)
    assert inker_ops.reason_for(op, state, tab)
    tab.saving = False
    state.transforming = True
    assert not op.enabled(state, tab)
    assert inker_ops.reason_for(op, state, tab)
    state.transforming = False
    # Playing: the way out must stay available.
    tab.playing = True
    assert op.enabled(state, tab)


# --- inker-43 ----------------------------------------------------------------


def test_indexing_to_a_palette_over_256_colours_is_refused_by_name():
    ctx, state, tab = _session()
    inker_mode.ensure(ctx)
    colours = [(i % 256, (i * 7) % 256, (i * 13) % 256, 255) for i in range(300)]

    assert inker_palette_io.index_to(ctx, tab, colours) is False

    said = " ".join(_toasts(ctx))
    assert "256" in said and "Cannot index" in said


# --- inker-44 ----------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        [float("inf"), 0, 0, 255],
        [float("nan"), 0, 0, 255],
        [300, 0, 0, 255],
        [-5, 0, 0, 255],
    ],
)
def test_a_non_finite_or_out_of_range_swatch_in_settings_does_not_stop_inker_opening(bad):
    good = [10, 20, 30, 255]
    stored = {"inker": {"swatches": [bad, good]}}
    settings = SimpleNamespace(
        get=lambda key: stored.get(key, {}), set=lambda key, value: stored.update({key: value})
    )
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=None, toast=lambda *a, **k: None), settings=settings
    )

    state = inker_mode.ensure(ctx)

    assert state.swatches == [(10, 20, 30, 255)]
    for swatch in state.swatches:
        assert all(0 <= c <= 255 for c in swatch)


def test_a_preset_option_of_the_wrong_type_is_dropped_on_restore():
    state = inker_state.InkerState()
    inker_mode._restore_presets(
        state,
        {
            "mine": {
                "tool": "brush",
                "options": {
                    "brush_size": "huge",
                    "hardness": float("nan"),
                    "shape_filled": "yes",
                    "nib": 7,
                    "opacity": 0.5,
                },
            }
        },
    )
    options = state.presets["mine"]["options"]
    assert options == {"opacity": 0.5}


# --- inker-45 ----------------------------------------------------------------


def _big_square(side):
    top = [(x, 0) for x in range(side)]
    right = [(side, y) for y in range(side)]
    bottom = [(x, side) for x in range(side, 0, -1)]
    left = [(0, y) for y in range(side, 0, -1)]
    return top + right + bottom + left


def test_dash_segments_cost_follows_the_visible_span_not_the_loop_perimeter(monkeypatch):
    verts, cum = ants.prepare([_big_square(10_000)])[0]  # a 40,000-px perimeter
    window = (0.0, 0.0, 400.0, 300.0)
    seen: list[int] = []
    real = ants._merge_runs

    def spy(head, tail, lit, *args, **kwargs):
        seen.append(len(head))
        return real(head, tail, lit, *args, **kwargs)

    monkeypatch.setattr(ants, "_merge_runs", spy)
    starts, ends, lit = ants.dash_segments(
        verts, cum, 64.0, (-300.0, 10.0), 1.0, window=window
    )
    # The unbounded form hands over ~ perimeter * zoom / dash = 430k pieces.
    assert seen and seen[0] < 5_000, seen
    assert len(starts) == len(ends) == len(lit)

    began = time.perf_counter()
    ants.dash_segments(verts, cum, 64.0, (-300.0, 10.0), 1.0, window=window)
    assert time.perf_counter() - began < 0.5


def _clipped_lengths(starts, ends, lit, window):
    """Lit and dark length of axis-aligned runs inside ``window``."""
    left, top, right, bottom = window
    totals = {True: 0.0, False: 0.0}
    for (ax, ay), (bx, by), on in zip(starts.tolist(), ends.tolist(), lit.tolist(), strict=True):
        x0, x1 = sorted((ax, bx))
        y0, y1 = sorted((ay, by))
        dx = max(0.0, min(x1, right) - max(x0, left))
        dy = max(0.0, min(y1, bottom) - max(y0, top))
        if x1 - x0 > 0 and not (top <= y0 <= bottom):
            continue
        if y1 - y0 > 0 and not (left <= x0 <= right):
            continue
        totals[bool(on)] += dx if x1 - x0 > 0 else dy
    return totals


def test_a_windowed_dash_walk_draws_the_same_ants_inside_the_window():
    verts, cum = ants.prepare([_big_square(200)])[0]
    for zoom, offset, phase, window in (
        (4.0, (-120.0, -80.0), 2.5, (0.0, 0.0, 300.0, 200.0)),
        (1.0, (10.0, 10.0), 0.0, (50.0, 20.0, 150.0, 90.0)),
        (8.0, (-500.0, -500.0), 7.3, (0.0, 0.0, 640.0, 480.0)),
    ):
        full = ants.dash_segments(verts, cum, zoom, offset, phase)
        cut = ants.dash_segments(verts, cum, zoom, offset, phase, window=window)
        want = _clipped_lengths(*full, window)
        got = _clipped_lengths(*cut, window)
        assert got[True] == pytest.approx(want[True], abs=1e-6)
        assert got[False] == pytest.approx(want[False], abs=1e-6)


def test_a_window_holding_the_whole_loop_changes_nothing():
    verts, cum = ants.prepare([_big_square(30)])[0]
    plain = ants.dash_segments(verts, cum, 3.0, (5.0, 5.0), 1.5)
    boxed = ants.dash_segments(
        verts, cum, 3.0, (5.0, 5.0), 1.5, window=(-1e6, -1e6, 1e6, 1e6)
    )
    for a, b in zip(plain, boxed, strict=True):
        assert np.allclose(a, b)


# --- inker-48 ----------------------------------------------------------------


def _locked_blur(pixels):
    stroke = brush.StrokeState(
        layer_uid=1,
        size=(pixels.shape[1], pixels.shape[0]),
        before=pixels.copy(),
        colour=RED,
        diameter=40,
        hardness=1.0,
        mode="blur",
        strength=1.0,
        alpha_lock=True,
    )
    stroke.begin((16, 16), pixels)


def test_a_locked_blur_keeps_an_opaque_edge_pixels_colour_instead_of_darkening_it():
    pixels = np.zeros((32, 32, 4), dtype=np.uint8)
    pixels[8:24, 8:24] = (200, 120, 60, 255)

    _locked_blur(pixels)

    inside = pixels[8:24, 8:24]
    assert (inside[..., 3] == 255).all(), "the lock still holds the alpha"
    # A uniform colour blurred with itself is that colour, edge pixels included.
    # 8-bit rounding allows a couple of units; the bug darkened by tens.
    assert np.abs(inside[..., :3].astype(int) - (200, 120, 60)).max() <= 4


def test_a_locked_blur_leaves_a_transparent_pixels_rgb_byte_identical():
    pixels = np.zeros((32, 32, 4), dtype=np.uint8)
    pixels[8:24, 8:24] = (200, 120, 60, 255)
    pixels[0:8, :, :3] = (7, 8, 9)  # hidden colour under alpha 0
    before = pixels.copy()

    _locked_blur(pixels)

    assert np.array_equal(pixels[0:8], before[0:8])


# --- inker-49 ----------------------------------------------------------------


def _locked_empty_doc():
    doc = inker.Document.blank(8, 8)
    layer = doc.stack.active
    layer.pixels[..., :3] = (5, 6, 7)  # hidden colour under alpha 0
    layer.pixels[..., 3] = 0
    doc.set_layer_props(alpha_lock=True)
    doc.history.clear()
    return doc


def _do_write_colour(doc):
    doc.write_colour((0, 0, 8, 8), RED, np.ones((8, 8), dtype=np.float32))


def _do_gradient(doc):
    doc.gradient((0.0, 0.0), (7.0, 0.0), RED, (0, 0, 255, 255))


def _do_shape(doc):
    doc.shape("rect", (0, 0), (7, 7), RED, 2, filled=True)


def _do_filter(doc):
    doc.begin_filter()
    doc.preview_filter("brightness / contrast", brightness=0.5)
    doc.commit_filter()


@pytest.mark.parametrize(
    "act", [_do_write_colour, _do_gradient, _do_shape, _do_filter], ids=lambda f: f.__name__
)
def test_fill_gradient_shape_and_filter_under_alpha_lock_leave_transparent_rgb_alone_and_push_no_step(  # noqa: E501
    act,
):
    doc = _locked_empty_doc()
    before = doc.stack.active.pixels.copy()
    head = doc.history.head

    act(doc)

    assert np.array_equal(doc.stack.active.pixels, before)
    assert doc.history.head == head, "nothing visible changed, so nothing should be undoable"


# --- inker-50 ----------------------------------------------------------------


def test_shrink_matte_inside_a_selection_does_not_erode_along_the_selections_own_edge():
    # A selection's crop lying wholly inside solid opaque pixels: no real
    # silhouette edge is in the crop, so nothing may erode.
    crop = np.zeros((6, 6, 4), dtype=np.uint8)
    crop[..., :3] = (10, 20, 30)
    crop[..., 3] = 255

    out = filters.matte_grow(crop, grow=-1.0)

    assert np.array_equal(out, crop)


def test_shrink_matte_still_erodes_a_real_silhouette_edge_inside_the_crop():
    crop = np.zeros((8, 8, 4), dtype=np.uint8)
    crop[2:6, 2:6] = (10, 20, 30, 255)

    out = filters.matte_grow(crop, grow=-1.0)

    assert out[2, 2, 3] == 0 and out[3, 3, 3] == 255


# --- inker-46 / inker-47 -----------------------------------------------------


def _fixture_layers():
    """Three layers, the middle one active and hidden, a selection, a reselect
    memory and a repeatable export: every layer-menu predicate is *true*."""
    ctx, state, tab = _session(layers=3)
    tab.doc.set_active_layer(1)
    tab.doc.set_layer_props(1, visible=False)
    tab.doc.select_all()
    tab.doc.deselect()
    tab.doc.select_all()
    tab.export_kind = "png"
    tab.doc.history.clear()
    return ctx, state, tab


def _fixture_linked():
    ctx, state, tab = _session(layers=2)
    tab.job_id = "job-1"
    tab.has_original = True
    return ctx, state, tab


def _fixture_animated():
    ctx, state, tab = _session(frames=3)
    tab.export_kind = "sheet"
    return ctx, state, tab


def _fixture_background():
    ctx, state, tab = _session(layers=2)
    tab.doc.set_active_layer(0)
    tab.doc.to_background()
    return ctx, state, tab


def _fixture_sheet():
    return _sheet_session()


_FIXTURES = (
    _fixture_layers,
    _fixture_linked,
    _fixture_animated,
    _fixture_background,
    _fixture_sheet,
    _indexed_animated,
    _own_palette_session,
)


def test_a_greyed_op_names_busy_when_only_busy_greys_it():
    """For every registered op and every way a tab can be not ready: if the op
    is live on a ready tab and greyed on a busy one, the busy tab is the whole
    reason, so the sentence must say BUSY and not a document fact that is false."""

    flipped = set()
    for make in _FIXTURES:
        for variant in ("saving", "playing", "transforming"):
            for op in inker_ops.OPS:
                ctx, state, tab = make()
                try:
                    live = op.enabled(state, tab)
                except Exception:  # noqa: BLE001 - an op needing richer state is not this test's
                    continue
                if not live:
                    continue
                _busy_variants(state, tab)[variant](True)
                if op.enabled(state, tab):
                    continue
                flipped.add(op.name)
                said = inker_ops.reason_for(op, state, tab)
                assert said == inker_ops.BUSY, (
                    f"{op.name} is greyed only because the tab is {variant}, but says: {said!r}"
                )
    # The sweep must actually have reached the rows the audit names.
    for name in (
        "delete_layer",
        "layer_up",
        "layer_down",
        "flatten",
        "merge_down",
        "show_layer",
        "from_background",
        "save_as_reference",
        "revert",
        "animate",
        "export_sheet",
        "export_gif",
        "repeat_export",
        "delete_frame",
    ):
        assert name in flipped, f"{name} was never exercised by the sweep"


def test_sheet_ops_greyed_by_a_busy_tab_say_why():
    for name in (
        "sheet_propagate",
        "sheet_replace",
        "sheet_shift",
        "sheet_mirror",
        "sheet_mirror_run",
        "sheet_merge",
        "sheet_keep_edit",
        "sheet_conflict_next",
    ):
        op = _op(name)
        for variant in ("saving", "playing", "transforming"):
            ctx, state, tab = _sheet_session()
            _busy_variants(state, tab)[variant](True)
            assert not op.enabled(state, tab), f"{name} stayed enabled while {variant}"
            assert inker_ops.reason_for(op, state, tab) == inker_ops.BUSY, (name, variant)
